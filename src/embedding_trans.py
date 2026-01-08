from src.embedding_base import EmbeddingBase
from src.markov_analysis import Markov
from typing import List, Optional, Sequence
from scipy.spatial.distance import cdist
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from src.trajectory_utils import canonicalize_trajectory

class EmbeddingTrans(EmbeddingBase):
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
        
        # Use Base class logic to filter IDs first
        if n_trajectories:
            ids = data[ID_NAME].drop_duplicates().to_numpy()
            rng = rng or np.random.default_rng(42)
            ids = rng.choice(ids, size=int(n_trajectories), replace=False)
            data = data[data[ID_NAME].isin(ids)]

        groups = [g for _, g in data.groupby(ID_NAME, sort=False)]

        # --- CHANGE: List Comprehension instead of np.stack ---
        # We do not cut by [:T] anymore. We keep full length of each group.
        Y_abs = [g.loc[:, columns].to_numpy(float) for g in groups]
        self.Y_tr = [g.loc[:, columns_translated].to_numpy(float) for g in groups]
        
        Y_lab = None
        if label_col in data.columns:
            Y_lab = [g[label_col].to_numpy(object) for g in groups]

        # Initialize base with LISTS
        super().__init__(
            data=None, # bypass data parsing in base
            columns=tuple(columns),
            ID_NAME=ID_NAME,
            Y=Y_abs,
            Y_labels=Y_lab,
            n_windows=n_windows,
            rng=rng,
        )

        # Helper dimension check (using first valid trajectory)
        if self.N > 0:
            self.D_total = self.D + self.Y_tr[0].shape[1]
        else:
            self.D_total = 0

    def make_embedding(self, K: int):
        """
        Builds hybrid absolute+relative embedding for variable length tracks.
        """
        self.K = int(K)
        
        batch_matrices = []
        batch_labels = []
        
        # We iterate over the list of trajectories (N)
        for n in range(self.N):
            # Get data for this specific trajectory
            y_abs_track = self.Y[n]      # shape (T_n, D_abs)
            y_tr_track = self.Y_tr[n]    # shape (T_n, D_rel)
            
            T_n = len(y_abs_track)
            
            if T_n < self.K:
                continue
                
            # Number of windows for this track
            L_n = T_n - self.K + 1
            
            # Pre-allocate window matrix for this specific track
            # Shape: (L_n, K * (D_abs + D_rel))
            E_track = np.empty((L_n, self.D_total * self.K), dtype=float)

            
            # Labels for this track
            if self.Y_labels is not None:
                # Shape: (L_n, K) -> assumes you want label per coordinate
                # Or (L_n,) if you want one label per window. 
                # Adapting to your original shape: (N, L, K)
                LBL_track = np.empty((L_n, self.K), dtype=object)
                y_lbl_track = self.Y_labels[n]
            
            # Loop over time for this track
            for t_idx, t0 in enumerate(range(L_n)):
                # 1. Abs part
                w_abs = y_abs_track[t0 : t0 + self.K].reshape(-1)
                
                # 2. Rel part (canonicalized)
                # Assumes canonicalize_trajectory takes (K, D) -> returns (K*D,)
                window_tr = y_tr_track[t0 : t0 + self.K]
                w_rel = canonicalize_trajectory(window_tr).reshape(-1)
                
                # 3. Concatenate
                E_track[t_idx] = np.concatenate([w_abs, w_rel], axis=0)

                if self.Y_labels is not None:
                    LBL_track[t_idx] = y_lbl_track[t0 : t0 + self.K]

            batch_matrices.append(E_track)
            if self.Y_labels is not None:
                batch_labels.append(LBL_track)

        # --- Final Assembly ---
        
        # self.embedding_matrix is now a LIST of 2D arrays (Ragged structure)
        self.embedding_matrix = batch_matrices
        
        # self.flatten_embedding_matrix is the stacked result (Ready for Clustering)
        if batch_matrices:
            self.flatten_embedding_matrix = np.concatenate(batch_matrices, axis=0)
        else:
            self.flatten_embedding_matrix = np.empty((0, self.D_total*self.K))

        if self.Y_labels is not None and batch_labels:
            self.embedding_labels = batch_labels
            self.flatten_embedding_labels = np.concatenate(batch_labels, axis=0).reshape(-1, self.K)
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
