from src.embedding_base import EmbeddingBase
from src.markov_analysis import Markov
from typing import List, Optional, Sequence
from scipy.spatial.distance import cdist
import numpy as np
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
        columns: Sequence[str],              # absolute features for Y (e.g. speed, curvature)
        columns_translated: Sequence[str],   # shifted features (e.g. x, y, z)
        ID_NAME: str = "ID",
        n_trajectories: Optional[int] = None,
        n_windows: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if data is None:
            raise ValueError("EmbeddingPosition requires `data` (DataFrame).")
        for c in list(columns) + list(columns_translated):
            if c not in data.columns:
                raise ValueError(f"Column '{c}' not found in DataFrame.")
        if ID_NAME not in data.columns:
            raise ValueError(f"`ID_NAME='{ID_NAME}'` not found in DataFrame.")

        self.columns_translated = tuple(columns_translated)
        self.ID_NAME = ID_NAME

        # Select IDs (optionally subsample)
        ids = data[ID_NAME].unique()
        if n_trajectories is not None:
            if rng is None:
                rng = np.random.default_rng(42)
            if n_trajectories > len(ids):
                raise ValueError(f"n_trajectories={n_trajectories} > available IDs={len(ids)}.")
            ids = rng.choice(ids, size=int(n_trajectories), replace=False)

        subset = data[data[ID_NAME].isin(ids)]


        # Extract both sets of trajectories with same T_min
        trajs_abs, trajs_trans, T_min = [], [], np.inf
        for _, traj_df in subset.groupby(ID_NAME, sort=False):
            traj_abs = traj_df.sort_index()[columns].values.astype(float)
            traj_trans = traj_df.sort_index()[columns_translated].values.astype(float)
            T_min = min(T_min, traj_abs.shape[0], traj_trans.shape[0])
            trajs_abs.append(traj_abs)
            trajs_trans.append(traj_trans)

        T_min = int(T_min)
        Y_abs = np.stack([traj[:T_min] for traj in trajs_abs])
        Y_trans = np.stack([traj[:T_min] for traj in trajs_trans])
        # Build aligned blocks with common T_min
        #trajs_abs, trajs_trans = [], []
        #T_min = np.inf
        #for _, traj_df in subset.groupby(ID_NAME, sort=False):
        #    arr_abs = traj_df[list(columns)].to_numpy(dtype=float)
        #    arr_trans = traj_df[list(columns_translated)].to_numpy(dtype=float)
        #    if arr_abs.size != 0:
        #        trajs_abs.append(arr_abs)
        #    if arr_trans.size != 0:
        #        trajs_trans.append(arr_trans)
        #    T_min = min(T_min, arr_abs.shape[0], arr_trans.shape[0])
                        
        #T_min = int(T_min)
        #Y_abs = np.stack([a[:T_min] for a in trajs_abs], axis=0)          # (N, T, d_abs)
        #Y_trans = np.stack([a[:T_min] for a in trajs_trans], axis=0)      # (N, T, d_trans)

        # Store translated block and initialize base with Y_abs
        self.Y_translated = Y_trans                                       # kept aligned with self.Y
        super().__init__(
            data=None,            # we pass Y directly to avoid re-parsing data
            columns=tuple(columns),
            ID_NAME=ID_NAME,
            n_trajectories=None,  # already applied
            Y=Y_abs,
            n_windows=n_windows,
            rng=rng,
        )
        
        # Optionally, record the total feature dimension if you plan to concatenate later
        # self.D is from Y_abs; add translated block dimensionality if useful downstream
        self.D_total = self.D + Y_trans.shape[2]
    def make_embedding(self, K: int) -> (np.ndarray, np.ndarray):
        if K < 3 or K > self.T:
            # minimum value of K for the SVD in canonicalize trajectory, where V the rotation matrix
            # will have the dimension min(K,d).
            raise ValueError("K must be in the range [3, T]")

        self.K = K
        self.N = self.Y.shape[0]
        self.L = self.T - K + 1
        total_D = self.K * self.D_total

        self.embedding_matrix = np.empty((self.N, self.L, total_D), dtype=float)
        self.flatten_embedding_matrix = np.empty((self.N * self.L, total_D), dtype=float)

        flatten_out_row = 0
        
        for n in range(self.N):
            out_row = 0
            for t in range(self.L):
                # Absolute window (no shift)
                #win_abs = self.Y[n, t:t + K].reshape(-1)
                # Translated window (relative shift)
                #win_rel = self.canonicalize_trajectory(self.Y_translated[n, t:t + K] ) # shape : (K,d)
                #win_rel = win_rel.reshape(-1) # shape K*d
                #full_window = np.concatenate([win_abs, win_rel])

                # win_abs: (K, d_abs), win_rel: (K, d_rel)                
                win_abs = self.Y[n, t:t + K].reshape(-1)                     # shape (K, d_abs)
                win_rel = canonicalize_trajectory(self.Y_translated[n, t:t + K]).reshape(-1)  # shape (K, d_rel)
                # Concatenate per-time-step: result is (K, d_abs + d_rel)

                # Flatten to shape (K*(d_abs + d_rel),)
                full_window = np.concatenate([win_abs, win_rel])

                self.embedding_matrix[n, out_row] = full_window
                self.flatten_embedding_matrix[flatten_out_row] = full_window

                out_row += 1
                flatten_out_row += 1

        return self.embedding_matrix, self.flatten_embedding_matrix
    
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
