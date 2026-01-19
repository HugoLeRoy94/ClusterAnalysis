from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Sequence, List, Union
from numpy.typing import NDArray

from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from scipy.spatial.distance import squareform
from src.distance import compute_condensed_distance_matrix
from scipy.spatial.distance import cdist
from numpy.lib.stride_tricks import sliding_window_view

import umap


class EmbeddingBase:
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        columns: Optional[Sequence[str]] = None,
        ID_NAME: str = "ID",
        label_col: Optional[str] = "move_type",
        n_trajectories: Optional[int] = None,
        Y: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        Y_labels: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        n_windows: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> None:
        # Core placeholders
        self.K = None
        self.embedding_matrix = None         # Now a List[np.ndarray]
        self.flatten_embedding_matrix = None # np.ndarray (stacked)
        self.embedding_labels = None         # Now a List[np.ndarray]
        self.flatten_embedding_labels = None # np.ndarray (stacked)
        self.indices = None

        self.columns = tuple(columns) if columns is not None else None
        self.ID_NAME = ID_NAME
        self.n_windows = n_windows

        # --- Direct array path ---
        if Y is not None:
            # Check if Y is a list (ragged) or a dense array
            if isinstance(Y, list):
                self.Y = [np.asarray(y, dtype=float) for y in Y]
                self.N = len(self.Y)
                # T is now a list of lengths
                self.T = [arr.shape[0] for arr in self.Y]
                self.D = self.Y[0].shape[1] if self.N > 0 else 0
            else:
                # Fallback for dense input, convert to list for consistency
                arr = np.asarray(Y, dtype=float)
                self.N, t_dim, self.D = arr.shape
                self.Y = [arr[i] for i in range(self.N)]
                self.T = [t_dim] * self.N

            if self.columns is None:
                self.columns = tuple(f"x{j}" for j in range(self.D))
            
            # Handle Labels similar to Y
            self.Y_labels = None
            if Y_labels is not None:
                if isinstance(Y_labels, list):
                    self.Y_labels = [np.asarray(l, dtype=object) for l in Y_labels]
                else:
                    arr = np.asarray(Y_labels, dtype=object)
                    self.Y_labels = [arr[i] for i in range(arr.shape[0])]
            return

        # --- DataFrame path ---
        if data is None or self.columns is None:
            raise ValueError("Provide either Y or DataFrame data+columns.")

        df = data
        if n_trajectories:
            ids = df[ID_NAME].drop_duplicates().to_numpy()
            rng = rng or np.random.default_rng(42)
            ids = rng.choice(ids, size=int(n_trajectories), replace=False)
            df = df[df[ID_NAME].isin(ids)]

        # Group WITHOUT enforcing min(T)
        groups = [g for _, g in df.groupby(ID_NAME, sort=False)]
        
        # Store as LIST of arrays
        self.Y = [g.loc[:, self.columns].to_numpy(float) for g in groups]
        self.N = len(self.Y)
        self.T = [arr.shape[0] for arr in self.Y] # List of lengths
        self.D = self.Y[0].shape[1] if self.N > 0 else 0

        if label_col and (label_col in df.columns):
            self.Y_labels = [g[label_col].to_numpy(object) for g in groups]
        else:
            self.Y_labels = None

    def make_embedding(self, K: int = 1, embedding_matrix: Optional[np.ndarray] = None):
        """
        Builds embedding for variable length trajectories.
        """
        if embedding_matrix is not None:
            # Legacy handling for pre-computed dense matrix
            self.embedding_matrix = [embedding_matrix] # Wrap in list
            self.flatten_embedding_matrix = embedding_matrix.reshape(-1, embedding_matrix.shape[-1])
            return self.embedding_matrix, self.flatten_embedding_matrix

        if self.Y is None:
            raise RuntimeError("Y is not set.")
        
        self.K = int(K)
        
        batch_matrices = []
        batch_labels = []

        # Iterate over each trajectory individually
        for n in range(self.N):
            y_curr = self.Y[n]
            t_curr = len(y_curr)
            
            # Skip trajectories shorter than window size
            if t_curr < self.K:
                # Optional: append empty array or handle strictly
                continue

            # Vectorized windowing for this trajectory
            # shape: (L, K, D)
            win = sliding_window_view(y_curr, self.K, axis=0) 
            
            # Flatten the K*D dimension: (L, K*D)
            L_curr = win.shape[0]
            flat_win = win.reshape(L_curr, self.K * self.D)
            
            batch_matrices.append(flat_win)

            # Handle Labels
            if self.Y_labels is not None:
                lbl_curr = self.Y_labels[n]
                # Label is typically the last point in the window
                batch_labels.append(lbl_curr[self.K - 1:])

        # Store structured results (List of Arrays)
        self.embedding_matrix = batch_matrices
        
        # Store flattened results (Single Stacked Array)
        if batch_matrices:
            self.flatten_embedding_matrix = np.concatenate(batch_matrices, axis=0)
        else:
            self.flatten_embedding_matrix = np.empty((0, self.K * self.D))

        if self.Y_labels is not None and batch_labels:
            self.embedding_labels = batch_labels
            self.flatten_embedding_labels = np.concatenate(batch_labels, axis=0)
        else:
            self.embedding_labels = None
            self.flatten_embedding_labels = None

        return self.embedding_matrix, self.flatten_embedding_matrix

    def compute_averages_embedding_chunk(self) -> (np.ndarray, np.ndarray):
        if self.Y.shape[2] == 2:
            self.embedded_av_velocity = np.mean(self.flatten_embedding_matrix[:, 0::2], axis=1)
            self.embedded_av_ang_velocity = np.mean(self.flatten_embedding_matrix[:, 1::2], axis=1)
            return self.embedded_av_velocity, self.embedded_av_ang_velocity
        elif self.Y.shape[2] == 3:
            self.embedded_av_velocity = np.mean(self.flatten_embedding_matrix[:, 0::3], axis=1)
            self.embedded_av_ang_velocity = np.mean(self.flatten_embedding_matrix[:, 1::3], axis=1)
            self.embedded_av_torsion_velocity = np.mean(self.flatten_embedding_matrix[:, 2::3], axis=1)
            return self.embedded_av_velocity, self.embedded_av_ang_velocity, self.embedded_av_torsion_velocity

    def set_n_windows(self, random_state: int = 0) -> None:
        if self.flatten_embedding_matrix is None:
            raise RuntimeError("Call make_embedding() first.")
        if self.n_windows is None:
            return  # no window subsampling
        if self.n_windows > self.flatten_embedding_matrix.shape[0]:
            raise ValueError("n_windows too large")
        rng = np.random.default_rng(random_state)
        self.indices = rng.choice(self.flatten_embedding_matrix.shape[0], self.n_windows, replace=False)

    def fit_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, with_cluster_centers=False) -> np.ndarray:
        """Fit a UMAP model and return the embedding."""
        if self.flatten_embedding_matrix is None:
            raise RuntimeError("Call make_embedding() first.")

        if self.n_windows is not None and self.indices is None:
            self.set_n_windows()

        if self.indices is not None:
            data_to_embed = self.flatten_embedding_matrix[self.indices]
        else:
            data_to_embed = self.flatten_embedding_matrix

        if with_cluster_centers:
            if self.cluster_centers_ is not None:
                data_to_embed = np.append(data_to_embed, self.cluster_centers_, axis=0)

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="euclidean")
        reduced_all = reducer.fit_transform(data_to_embed)
        reduced_points = reduced_all[:self.n_windows]
        reduced_centers = reduced_all[self.n_windows:]
        return reduced_points, reduced_centers

    def make_cluster(
    self,
    n_clusters: int = 10,
    *,
    use: str | np.ndarray = "embedding",  # "embedding" | "umap" | custom 2D array
    random_state: int = 42,
    n_init: int | str = "auto",
    max_iter: int = 300,
    sample_weight: Optional[np.ndarray] = None,
    ):
        """
        K-Means on window vectors.
        use="embedding" -> self.flatten_embedding_matrix
        use="umap"      -> self.umap_embedding
        use=array       -> your own (M x d) ndarray
        Stores: self.labels (M,), self.labels_matrix (N,L), self.cluster_centers_, self.n_clusters
        Returns (labels, centers).
        """


        # pick data matrix X (M x d)
        if isinstance(use, str):
            if use == "umap":
                X = getattr(self, "umap_embedding", None)
            else:
                if self.flatten_embedding_matrix is None and self.embedding_matrix is not None:
                    self.flatten_embedding_matrix = self.embedding_matrix.reshape(-1, self.embedding_matrix.shape[2])
                X = self.flatten_embedding_matrix
        else:
            X = np.asarray(use)
        if X is None:
            raise RuntimeError("No data to cluster. Build embedding or pass a custom array.")

        km = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=n_init, max_iter=max_iter)
        km.fit(X, sample_weight=sample_weight)

        self.n_clusters = int(n_clusters)
        self.labels = km.labels_.astype(int)
        self.cluster_centers_ = km.cluster_centers_

        # reshape to (N, L) if embedding dimensions are known
        if getattr(self, "N", None) and getattr(self, "L", None):
            self.labels_matrix = self.labels.reshape(self.N, self.L)
        else:
            self.labels_matrix = None

        return self.labels, self.cluster_centers_


    def pick_random_trajectory_in_cluster(self, cluster_id: int) -> NDArray[np.float64]:
        if self.labels is None:
            raise RuntimeError("Need the labels first.")
        words = np.argwhere(self.labels==cluster_id)[:,0]
        index = np.random.randint(0, words.shape[0])
        return self.flatten_embedding_matrix[words[index]]

    def classify_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
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

        d = len(self.columns)

        if d > 0 and trajectory is None:
            raise ValueError("Model was trained with absolute features, but 'trajectory_abs' was not provided.")        
        if d == 0 and trajectory is not None:
            raise ValueError("'trajectory_abs' was provided, but model was not trained with absolute features.")

        # Determine trajectory length and check consistency
        T = -1
        if trajectory is not None:
            T = trajectory.shape[0]
            if trajectory.shape[1] != d:
                raise ValueError(f"trajectory_abs has wrong dimension {trajectory.shape[1]}, expected {d}")

        if T == -1:
            if d > 0:
                raise ValueError("At least one trajectory must be provided.")
            else: # No features were used in the model, so no classification is possible.
                return np.array([])


        L = T - self.K + 1
        if L < 1:
            raise ValueError("Trajectory is too short for the given embedding window K.")

        embedded_dim = self.K * d
        embedded_trajectory = np.empty((L, embedded_dim), dtype=float)

        for t in range(L):
            windows = []
            win = trajectory[t:t + self.K]
            embedded_trajectory[t] = win.reshape(-1)

        distances = cdist(embedded_trajectory, self.cluster_centers_)
        labels = np.argmin(distances, axis=1)
        
        return labels

    def cluster_label_profile(self):
        """
        Returns:
        profiles: (C, L) array of per-cluster mean label proportions
        label_names: list of L label strings in column order
        counts: (C,) number of samples per cluster
        global_profile: (L,) overall mean label proportions
        """
        # M samples (windows), P coords per window
        LBL = self.flatten_embedding_labels
        if LBL is None:
            raise RuntimeError("flatten_embedding_labels is None")
        M, P = LBL.shape
        # normalize missing as a single class, then factorize
        flat = LBL.ravel().astype(object)
        flat[pd.isna(flat)] = "__MISSING__"
        label_names, inv = np.unique(flat, return_inverse=True)
        Z = inv.reshape(M, P)               # int codes in [0..L-1]
        L = label_names.size

        # Row-wise normalized histograms H (M, L)
        H = np.zeros((M, L), dtype=float)
        np.add.at(H, (np.repeat(np.arange(M), P), Z.ravel()), 1.0)
        H /= float(P)                       # proportion per sample

        # Aggregate by cluster
        C = int(self.n_clusters)
        cl = self.labels.astype(int)
        profiles = np.zeros((C, L), dtype=float)
        counts = np.zeros(C, dtype=float)
        for c in range(C):
            mask = (cl == c)
            if mask.any():
                profiles[c] = H[mask].mean(axis=0)
                counts[c] = mask.sum()

        global_profile = H.mean(axis=0)     # sanity check: weighted avg over clusters

        return profiles, list(label_names), counts, global_profile