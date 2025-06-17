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
        if K < 1 or K > self.T:
            raise ValueError("K must be in the range [1, T]")

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
                win_abs = self.Y[n, t:t + K].reshape(-1)

                # Translated window (relative shift)
                win_rel = self.Y_translated[n, t:t + K] - self.Y_translated[n, t]
                win_rel = win_rel.reshape(-1)

                full_window = np.concatenate([win_abs, win_rel])

                self.embedding_matrix[n, out_row] = full_window
                self.flatten_embedding_matrix[flatten_out_row] = full_window

                out_row += 1
                flatten_out_row += 1

        return self.embedding_matrix, self.flatten_embedding_matrix