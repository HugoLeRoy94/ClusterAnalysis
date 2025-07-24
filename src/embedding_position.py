from src.embedding_base import EmbeddingBase
from src.markov_analysis import Markov
from typing import List, Optional
import numpy as np
import pandas as pd
from src.trajectory_utils import canonicalize_trajectory


class EmbeddingPosition(EmbeddingBase):
    def __init__(
        self,
        data: pd.DataFrame,
        columns: List[str],              # absolute features (e.g. speed, curvature)
        columns_translated: List[str],   # shifted features (e.g. x, y, z)
        Y: np.ndarray | None = None,
        ID_NAME: str = "ID",
        n_trajectories: Optional[int] = None,
        n_windows: Optional[int] = None,
    ) -> None:
        self.columns_translated = columns_translated
        self.ID_NAME = ID_NAME

        if Y is not None:
            raise NotImplementedError("Only raw DataFrame input is supported in this subclass.")

        # Get subset of trajectories
        if n_trajectories is None:
                wanted_ids = data[ID_NAME].unique()
        else:
            rng = np.random.default_rng(seed=42)  # or pass the seed as an argument
            wanted_ids = rng.choice(data[ID_NAME].unique(), size=int(n_trajectories), replace=False)
        subset = data[data[ID_NAME].isin(wanted_ids)]

        # Extract both sets of trajectories with same T_min
        trajs_abs, trajs_trans, T_min = [], [], np.inf
        for _, traj_df in subset.groupby(ID_NAME, sort=False):
            traj_abs = traj_df.sort_index()[columns].values.astype(float)
            traj_trans = traj_df.sort_index()[columns_translated].values.astype(float)
            T_min = min(T_min, traj_abs.shape[0], traj_trans.shape[0])
            trajs_abs.append(traj_abs)
            trajs_trans.append(traj_trans)

        Y_abs = np.stack([traj[:T_min] for traj in trajs_abs])
        Y_trans = np.stack([traj[:T_min] for traj in trajs_trans])

        self.Y_translated = Y_trans
        super().__init__(data, columns=columns, Y=Y_abs, ID_NAME=ID_NAME, n_trajectories=n_trajectories,n_windows=n_windows)
        self.D += len(columns_translated)

    def make_embedding(self, K: int) -> (np.ndarray, np.ndarray):
        if K < 3 or K > self.T:
            # minimum value of K for the SVD in canonicalize trajectory, where V the rotation matrix
            # will have the dimension min(K,d).
            raise ValueError("K must be in the range [3, T]")

        self.K = K
        self.N = self.Y.shape[0]
        self.L = self.T - K + 1
        total_D = self.K * self.D

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
                win_abs = self.Y[n, t:t + K]                     # shape (K, d_abs)
                win_rel = canonicalize_trajectory(self.Y_translated[n, t:t + K])  # shape (K, d_rel)

                # Concatenate per-time-step: result is (K, d_abs + d_rel)
                combined = np.concatenate([win_abs, win_rel], axis=1)

                # Flatten to shape (K*(d_abs + d_rel),)
                full_window = combined.reshape(-1)

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

        from scipy.spatial.distance import cdist
        distances = cdist(embedded_trajectory, self.cluster_centers_)
        labels = np.argmin(distances, axis=1)
        
        return labels

    def analyze_markov_process(self) -> Markov:
        """Create and return a MarkovAnalysis object."""
        if self.labels is None:
            raise RuntimeError("Need labels; call make_cluster() first.")
        return Markov(self)
