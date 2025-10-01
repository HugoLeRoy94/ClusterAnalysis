from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
#from sklearn_extra.cluster import KMedoids
#from pyclustering.cluster.kmedoids import kmedoids
#from pyclustering.utils.metric import distance_metric, type_metric
from scipy.spatial.distance import squareform
from src.distance import compute_condensed_distance_matrix
from scipy.spatial.distance import cdist


import umap

__all__ = ["EmbeddingBase"]

class EmbeddingBase:
    """
    EmbeddingBase
    -------------
    Initialize from either:
      (A) a tidy pandas DataFrame with one row per timepoint and a column holding
          the trajectory identifier (e.g. "ID"), plus position columns (e.g. ["x","y","z"]);
      (B) a prebuilt NumPy array Y with shape (N, T, d), where N = #trajectories,
          T = #timepoints per trajectory (assumed equal across trajectories),
          and d = dimensionality (2 or 3 typically).

    Use case (A) when your raw data is tabular and you want this class to stack
    trajectories into a uniform Y array. Use case (B) when you already have Y.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        # Parameters for DataFrame-based init (case A)
        columns: Optional[Sequence[str]] = None,   # e.g. ("x","y","z")
        ID_NAME: str = "ID",
        n_trajectories: Optional[int] = None,      # sample at most this many IDs
        # Parameters for prebuilt-array init (case B)
        Y: Optional[np.ndarray] = None,
        # Misc metadata (optional)
        n_windows: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:

        # ---- choose init path ----
        if (Y is None) == (data is None):
            raise ValueError("Pass exactly one of `data` or `Y` (not both, not neither).")

        self.n_windows = n_windows

        if Y is not None:
            # ---------- Case B: direct array ----------
            Y = np.asarray(Y, dtype=float)
            if Y.ndim != 3:
                raise ValueError(f"`Y` must have shape (N, T, d); got {Y.shape}.")
            self.Y = Y
            self.T = int(Y.shape[1])
            self.Nsample = int(Y.shape[0])
            self.D = int(Y.shape[2])
            # DF-related attributes not applicable
            self.columns = tuple(f"x{j}" for j in range(self.D))
            self.ID_NAME = ID_NAME  # kept for API symmetry, but unused in this mode

        else:
            # ---------- Case A: DataFrame -> Y ----------
            if columns is None or len(columns) == 0:
                raise ValueError("`columns` must be provided (e.g. ('x','y','z')) when initializing from DataFrame.")
            if ID_NAME not in data.columns:
                raise ValueError(f"`ID_NAME='{ID_NAME}'` not found in DataFrame columns.")
            for c in columns:
                if c not in data.columns:
                    raise ValueError(f"Column '{c}' not found in DataFrame.")

            self.columns = tuple(columns)
            self.D = len(self.columns)
            self.ID_NAME = ID_NAME

            ids = data[ID_NAME].unique()
            if n_trajectories is not None:
                if rng is None:
                    rng = np.random.default_rng(42)
                if n_trajectories > len(ids):
                    raise ValueError(f"n_trajectories={n_trajectories} > available IDs={len(ids)}.")
                ids = rng.choice(ids, size=int(n_trajectories), replace=False)

            subset = data[data[ID_NAME].isin(ids)]
            # Group per trajectory, keep original order (no sort by key)
            trajs = []
            T_min = np.inf
            for _, traj_df in subset.groupby(ID_NAME, sort=False):
                # If you have a time column, you could sort by it here before taking values.
                arr = traj_df[self.columns].to_numpy(dtype=float)
                if arr.shape[0] == 0:
                    continue
                trajs.append(arr)
                T_min = min(T_min, arr.shape[0])

            if not trajs:
                raise ValueError("No trajectories found after filtering.")
            self.T = int(T_min)
            self.Y = np.stack([a[:self.T] for a in trajs], axis=0)  # (N, T, d)
            self.Nsample = self.Y.shape[0]

        # ---- placeholders filled later by downstream steps ----
        self.K: Optional[int] = None
        self.n_clusters: Optional[int] = None
        self.embedding_matrix: Optional[np.ndarray] = None       # (N, T-K+1, K*d)
        self.flatten_embedding_matrix: Optional[np.ndarray] = None  # (N*(T-K+1), K*d)
        self.labels: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None

    def make_embedding(self, K: int) -> (np.ndarray, np.ndarray):
        """Construct K‑delay vectors and concatenate over trajectories.

        After the call, ``self.embedding_matrix`` has shape *(N,(T-K+1), K·d)* and is returned.
        ``self.flatten_embedding_matrix`` has shape *(N*T-K+1), K·d)* and is returned.
        """
        if K < 1 or K > self.T:
            raise ValueError("K must be in the range [1, T]")
        self.K = K
        self.N = self.Y.shape[0]
        self.L = self.T - K + 1  # number of windows per trajectory
        self.embedding_matrix = np.empty((self.N, self.L, self.K * self.D), dtype=float)
        self.flatten_embedding_matrix = np.empty((self.N * self.L, self.K * self.D), dtype=float)
        flatten_out_row = 0
        for n in range(self.N):
            out_row = 0
            for t in range(self.L):
                window = self.Y[n, t : t + self.K].reshape(-1)
                self.embedding_matrix[n, out_row] = window
                self.flatten_embedding_matrix[flatten_out_row] = window
                out_row += 1
                flatten_out_row += 1
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

    def make_cluster(self, n_clusters: int, random_state: int = 0, clustering_method: str = 'kmeans', batchsize: Optional[int] = None, tol: float = 0.001, degree: int = 5) -> np.ndarray:
        """Run clustering on the embedding matrix and store the labels.
        Returns the 1‑D label array of length *self.embedding_matrix.shape[0]*.
        """
        if self.embedding_matrix is None:
            raise RuntimeError("Call make_embedding() first.")
        if n_clusters > self.flatten_embedding_matrix.shape[0]:
            raise ValueError("n_clusters must be lower than the number of samples")
        self.n_clusters = n_clusters

        if self.n_windows is not None and self.indices is None:
            self.set_n_windows( random_state=random_state)

        if self.indices is not None:
            subset = self.flatten_embedding_matrix[self.indices]
        else:
            subset = self.flatten_embedding_matrix

        if clustering_method == 'kmeans':
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
            labels_subsample = km.fit_predict(subset)
            self.cluster_centers_ = km.cluster_centers_
            # Predict labels for the entire dataset
            distances = cdist(self.flatten_embedding_matrix, self.cluster_centers_)
            self.labels = np.argmin(distances, axis=1)

        elif clustering_method == 'spectral':
            sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=random_state)
            self.labels = sc.fit_predict(subset)
            self.cluster_centers_ = np.array([subset[self.labels == i].mean(axis=0) for i in range(np.max(self.labels) + 1)])
        elif clustering_method == 'minibatch_kmeans':
            if batchsize is None:
                batchsize = n_clusters * 5
            mbk = MiniBatchKMeans(batch_size=batchsize, n_clusters=n_clusters, random_state=random_state)
            self.labels = mbk.fit_predict(subset)
            self.cluster_centers_ = mbk.cluster_centers_
        elif clustering_method == 'kmedoids':
            metric = distance_metric(type_metric.CHEBYSHEV)
            initial_medoid_indices = np.random.choice(np.arange(len(subset)), n_clusters, replace=False)
            kmedoids_instance = kmedoids(subset, initial_medoid_indices, metric=metric, tolerance=tol)
            kmedoids_instance.process()
            clusters = kmedoids_instance.get_clusters()
            medoids = kmedoids_instance.get_medoids()
            
            labels = np.arange(len(subset))
            for kc, cluster in enumerate(clusters):
                labels[cluster] = kc
            self.labels = labels
            self.cluster_centers_ = np.array(medoids)
        elif clustering_method == 'polynomial_distances':
            if self.distance_matrix is None:
                subset_reshape = np.ascontiguousarray(subset.reshape(subset.shape[0],self.K,self.D),dtype=np.float32)
                distance_matrix = squareform(compute_condensed_distance_matrix(subset_reshape,degree))
            model = KMedoids(n_clusters=n_clusters,
                     metric='precomputed',
                     init='k-medoids++',
                     random_state=random_state)
            model.fit(distance_matrix)
            self.labels = model.labels_
            self.cluster_centers_ = self.flatten_embedding_matrix[self.indices[model.medoid_indices_]]
        return self.labels

    def classify_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Classify each point of a single trajectory into a cluster.

        Parameters
        ----------
        trajectory : np.ndarray
            A single trajectory of shape (T, d).

        Returns
        -------
        np.ndarray
            An array of cluster labels for each point in the trajectory.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Clustering must be performed first.")
        
        # Perform time-delay embedding on the trajectory
        T, d = trajectory.shape
        if d != self.D:
            raise ValueError(f"Trajectory has wrong dimension {d}, expected {self.D}")
        
        L = T - self.K + 1
        if L < 1:
            raise ValueError("Trajectory is too short for the given embedding window K.")

        embedded_trajectory = np.empty((L, self.K * self.D), dtype=float)
        for t in range(L):
            embedded_trajectory[t] = trajectory[t:t + self.K].reshape(-1)

        # For each embedded vector, find the closest cluster center
        from scipy.spatial.distance import cdist
        distances = cdist(embedded_trajectory, self.cluster_centers_)
        labels = np.argmin(distances, axis=1)
        
        return labels

    def pick_random_trajectory_in_cluster(self, cluster_id: int) -> NDArray[np.float_]:
        if self.labels is None:
            raise RuntimeError("Need the labels first.")
        words = np.argwhere(self.labels==cluster_id)[:,0]
        index = np.random.randint(0, words.shape[0])
        return self.flatten_embedding_matrix[words[index]]