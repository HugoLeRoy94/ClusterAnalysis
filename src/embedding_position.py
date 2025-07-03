from src.embedding import Embedding
from typing import List
import numpy as np
import pandas as pd


class EmbeddingPosition(Embedding):
    def __init__(
        self,
        data: pd.DataFrame,
        columns: List[str],              # absolute features (e.g. speed, curvature)
        columns_translated: List[str],   # shifted features (e.g. x, y, z)
        Y: np.ndarray | None = None,
        Nsamples: int | str = "all",
        ID_NAME: str = "ID",
    ) -> None:
        self.columns_translated = columns_translated
        self.D_trans = len(columns_translated)
        self.ID_NAME = ID_NAME

        if Y is not None:
            raise NotImplementedError("Only raw DataFrame input is supported in this subclass.")

        # Get subset of trajectories
        if Nsamples == "all":
            wanted_ids = data[ID_NAME].unique()
        else:
            wanted_ids = data[ID_NAME].unique()[: int(Nsamples)]
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
        super().__init__(data, columns=columns, Y=Y_abs, Nsamples=Nsamples, ID_NAME=ID_NAME)


    def make_embedding(self, K: int) -> (np.ndarray, np.ndarray):
        if K < 3 or K > self.T:
            # minimum value of K for the SVD in canonicalize trajectory, where V the rotation matrix
            # will have the dimension min(K,d). 
            raise ValueError("K must be in the range [3, T]")

        self.K = K
        self.N = self.Y.shape[0]
        self.L = self.T - K + 1
        total_D = self.K * (self.D + self.D_trans)

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
                win_rel = self.canonicalize_trajectory(self.Y_translated[n, t:t + K])  # shape (K, d_rel)

                # Concatenate per-time-step: result is (K, d_abs + d_rel)
                combined = np.concatenate([win_abs, win_rel], axis=1)

                # Flatten to shape (K*(d_abs + d_rel),)
                full_window = combined.reshape(-1)

                self.embedding_matrix[n, out_row] = full_window
                self.flatten_embedding_matrix[flatten_out_row] = full_window

                out_row += 1
                flatten_out_row += 1

        return self.embedding_matrix, self.flatten_embedding_matrix

    @staticmethod
    def canonicalize_trajectory(coords, *, return_rotation=False, tol=1e-12):
        """
        Rotate `coords` (K×3) into a unique canonical frame.
        Any rigid-body rotation + translation of the same trajectory
        maps to the *identical* output.

        Algorithm
        ---------
        1.  centre at the centroid
        2.  PCA → eigenvectors V (columns)
        3.  for each axis j:                       # sign disambiguation
            m3 = Σ (x_j')³   (third central moment)
            if |m3| < tol use   Σ x_j'² x_{j+1}'
            flip V[:,j] if m3 < 0
        4.  make the basis right-handed (det = +1)
        5.  rotated = centred @ V

        Returns
        -------
        canon : (K,3)  canonical coordinates (centroid at the origin)
        R      : (3,3) rotation matrix  (only if `return_rotation=True`)
        """

        X   = np.asarray(coords, dtype=float)
        C   = X - X.mean(axis=0)               # 1
        _,  _, Vt = np.linalg.svd(C, full_matrices=False)
        V   = Vt.T                             # 2

        Y   = C @ V                            # projections for moments
        for j in range(3):                     # 3
            m3 = (Y[:, j] ** 3).sum()
            if abs(m3) < tol:                  # nearly symmetric
                k = (j + 1) % 3
                m3 = (Y[:, j]**2 * Y[:, k]).sum()
            if m3 < 0:
                V[:, j] *= -1
                Y[:, j] *= -1

        if np.linalg.det(V) < 0:               # 4
            V[:, 2] *= -1
            Y[:, 2] *= -1

        canon = Y                               # 5
        return (canon, V) if return_rotation else canon
