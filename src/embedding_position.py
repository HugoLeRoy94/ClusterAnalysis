from src.embedding_base import EmbeddingBase
from src.markov_analysis import Markov
from typing import List, Optional, Sequence
from scipy.spatial.distance import cdist
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from src.trajectory_utils import canonicalize_trajectory


class EmbeddingPosition(EmbeddingBase):
    """
    Like Embedding, but also stores a *translated* coordinate block
    (e.g. absolute positions) alongside the absolute features used to build Y.

    Only DataFrame-based initialization is supported here because we need
    both `columns` and `columns_translated` extracted with the same T_min.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str],
        columns_translated: Sequence[str],
        ID_NAME: str = "ID",
        n_trajectories: Optional[int] = None,
        n_windows: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        label_col: str = "move_type",
    ) -> None:
        self.columns_translated = tuple(columns_translated)
        self.ID_NAME = ID_NAME
        self.n_windows = n_windows

        if n_trajectories:
            ids = data[ID_NAME].drop_duplicates().to_numpy()
            rng = rng or np.random.default_rng(42)
            ids = rng.choice(ids, size=int(n_trajectories), replace=False)
            data = data[data[ID_NAME].isin(ids)]

        # group without reordering IDs
        groups = [g.sort_index() for _, g in data.groupby(ID_NAME, sort=False)]

        # common T across abs and translated
        T = min(min(len(g[columns]), len(g[columns_translated])) for g in groups)
        T = int(T)

        Y_abs  = np.stack([g.loc[:, columns].to_numpy(float)[:T]            for g in groups], axis=0)
        self.Y_tr   = np.stack([g.loc[:, columns_translated].to_numpy(float)[:T] for g in groups], axis=0)
        Y_lab  = (np.stack([g[label_col].to_numpy(object)[:T] for g in groups], axis=0)
                if label_col in data.columns else None)

        # hand off to base with arrays only (no re-parse, labels preserved)
        super().__init__(
            data=None,
            columns=tuple(columns),
            ID_NAME=ID_NAME,
            Y=Y_abs,
            Y_labels=Y_lab,
            n_windows=n_windows,
            rng=rng,
        )

        # optional: total feature dimension if you concatenate later
        self.D_total = self.D + self.Y_tr.shape[2]


    def make_embedding(self, K: int):
        """
        Build windows with absolute coords + canonicalized translated coords.
        Also build per-coordinate labels:
        embedding_matrix:         (N, L, K*(d_abs+d_rel))
        embedding_labels (object):(N, L, K*(d_abs+d_rel))  # label of each coord comes from its timepoint
        """
        self.K = K

        if self.K > self.T:
            raise ValueError("K must be in [1, T]")        
        

        E = np.empty((self.N, self.T-self.K+1, self.D_total*self.K), dtype=float)
        LBL = None
        have_labels = getattr(self, "Y_labels", None) is not None
        if have_labels:
            if self.Y_labels.shape[1] != self.T:
                print("reshape the Y_label data")
                self.Y_labels = self.Y_labels[:, :self.T]
            LBL = np.empty((self.N, self.T-self.K+1, self.K), dtype=object)

        r = 0
        for n in range(self.N):
            for t0 in range(self.T-self.K+1):
                # abs part
                w_abs = self.Y[n, t0:t0+self.K].reshape(-1)  # (K*d_abs,)
                # rel part (canonicalized per window)
                w_rel = canonicalize_trajectory(self.Y_tr[n, t0:t0+self.K]).reshape(-1)  # (K*d_rel,)
                w = np.concatenate([w_abs, w_rel], axis=0)  # (K*(d_abs+d_rel),)
                E[n, t0] = w

                if have_labels:
                    # per-timepoint labels -> per-coordinate labels
                    LBL[n, t0] = self.Y_labels[n, t0:t0 + self.K]

                r += 1

        self.embedding_matrix = E
        self.flatten_embedding_matrix = E.reshape(-1, self.D_total*self.K)

        if have_labels:
            self.embedding_labels = LBL
            self.flatten_embedding_labels = LBL.reshape(-1, self.K)
        else:
            self.embedding_labels = None
            self.flatten_embedding_labels = None



    
    def classify_trajectory(self, trajectory_abs: Optional[np.ndarray] = None, trajectory_trans: Optional[np.ndarray] = None) -> np.ndarray:
        """Classify each point of a single trajectory into a cluster.

        Parameters
        ----------
        trajectory_abs : np.ndarray, optional
            A single trajectory of shape (T, d_abs) for absolute features.
            Required if the model was trained with absolute features.
        trajectory_trans : np.ndarray, optional
            A single trajectory of shape (T, d_trans) for translated features.
            Required if the model was trained with translated features.

        Returns
        -------
        np.ndarray
            An array of cluster labels for each point in the trajectory.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Clustering must be performed first.")

        d_abs_model = len(self.columns)
        d_trans_model = len(self.columns_translated)

        if d_abs_model > 0 and trajectory_abs is None:
            raise ValueError("Model was trained with absolute features, but 'trajectory_abs' was not provided.")
        if d_trans_model > 0 and trajectory_trans is None:
            raise ValueError("Model was trained with translated features, but 'trajectory_trans' was not provided.")
        if d_abs_model == 0 and trajectory_abs is not None:
            raise ValueError("'trajectory_abs' was provided, but model was not trained with absolute features.")
        if d_trans_model == 0 and trajectory_trans is not None:
            raise ValueError("'trajectory_trans' was provided, but model was not trained with translated features.")

        # Determine trajectory length and check consistency
        T = -1
        if trajectory_abs is not None:
            T = trajectory_abs.shape[0]
            if trajectory_abs.shape[1] != d_abs_model:
                raise ValueError(f"trajectory_abs has wrong dimension {trajectory_abs.shape[1]}, expected {d_abs_model}")
        if trajectory_trans is not None:
            if T != -1 and trajectory_trans.shape[0] != T:
                raise ValueError("Absolute and translated trajectories must have the same length.")
            T = trajectory_trans.shape[0]
            if trajectory_trans.shape[1] != d_trans_model:
                raise ValueError(f"trajectory_trans has wrong dimension {trajectory_trans.shape[1]}, expected {d_trans_model}")

        if T == -1:
            if d_abs_model > 0 or d_trans_model > 0:
                raise ValueError("At least one trajectory must be provided.")
            else: # No features were used in the model, so no classification is possible.
                return np.array([])


        L = T - self.K + 1
        if L < 1:
            raise ValueError("Trajectory is too short for the given embedding window K.")

        total_embedded_dim = self.K * (d_abs_model + d_trans_model)
        embedded_trajectory = np.empty((L, total_embedded_dim), dtype=float)

        for t in range(L):
            windows = []
            if trajectory_abs is not None:
                win_abs = trajectory_abs[t:t + self.K]
                windows.append(win_abs)
            
            if trajectory_trans is not None:
                win_rel = canonicalize_trajectory(trajectory_trans[t:t + self.K])
                windows.append(win_rel)
            
            if windows:
                combined = np.concatenate(windows, axis=1)
                embedded_trajectory[t] = combined.reshape(-1)

        distances = cdist(embedded_trajectory, self.cluster_centers_)
        labels = np.argmin(distances, axis=1)
        
        return labels

    def analyze_markov_process(self) -> Markov:
        """Create and return a MarkovAnalysis object."""
        if self.labels is None:
            raise RuntimeError("Need labels; call make_cluster() first.")
        return Markov(self)
