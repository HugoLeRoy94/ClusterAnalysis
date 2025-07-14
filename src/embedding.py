

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from scipy.spatial.distance import squareform
from src.distance import compute_condensed_distance_matrix
from src.trajectory_utils import (
    count_transitions,
    stationary_distribution,
    entropy_rate,
    time_reversed_transition_matrix,
    metastability,
    reconstruct_Y_from_embedding,
    embed_move_type,
)


import umap

__all__ = ["Embedding"]


class Embedding:
    """Time‑delay embedding + k‑means + Markov analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Long‑table format. Each row is a single time‑step.  One column contains the
        trajectory identifier (``ID_NAME``).  The columns listed in *columns* are
        the dynamical coordinates to embed.
    columns : List[str]
        Names of the coordinate columns to use for the embedding.
    Nsamples : int | "all", default "all"
        How many trajectories (unique IDs) to load.  Useful for fast prototyping.
    ID_NAME : str, default "ID"
        Column that carries the trajectory identifier.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        columns: List[str],
        Y: np.ndarray | None = None,
        Nsamples: int | str = "all",
        ID_NAME: str = "ID",
        n_subsample: Optional[int] = None,
    ) -> None:
        self.columns = columns
        self.D = len(columns)
        self.ID_NAME = ID_NAME
        self.n_subsample = n_subsample
        # Grab at most *Nsamples* trajectories
        # Re‑index time per trajectory so that T is consistent across worms        
        if Y is None:
            if Nsamples == "all":
                wanted_ids = data[ID_NAME].unique()
            else:
                wanted_ids = data[ID_NAME].unique()[: int(Nsamples)]
            subset = data[data[ID_NAME].isin(wanted_ids)]
            trajs = []
            T_min = np.inf
            for _, traj_df in subset.groupby(ID_NAME, sort=False):
                traj_arr = traj_df.sort_index()[columns].values.astype(float)
                trajs.append(traj_arr)
                T_min = min(T_min, traj_arr.shape[0])
            self.T = int(T_min)
            self.Y = np.stack([traj[: self.T] for traj in trajs])  # shape (N, T, d)
        else:
            self.Y = Y
            self.T = self.Y.shape[1]
        self.Nsample = self.Y.shape[0]# number of individual trajectories
        # Place‑holders that will be filled later
        self.K: Optional[int] = None
        self.n_clusters: Optional[int] = None
        self.tau: Optional[int] = None
        self.embedding_matrix: Optional[np.ndarray] = None  # flattened embedding, shape (N,(T-K+1), K·d)
        self.flatten_embedding_matrix: Optional[np.ndarray] = None  # flattened embedding, shape (N*(T-K+1), K·d)
        self.labels: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.pi: Optional[np.ndarray] = None
        self.state: Optional[int]=None
        self.indices: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None


    # ---------------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------------
    def make_embedding(self, K: int) -> (np.ndarray,np.ndarray):
        """Construct K‑delay vectors and concatenate over trajectories.

        After the call, ``self.embedding_matrix`` has shape *(N,(T-K+1), K·d)* and is returned.
        ``self.flatten_embedding_matrix`` has shape *(N*T-K+1), K·d)* and is returned.
        """
        if K < 1 or K > self.T:
            raise ValueError("K must be in the range [1, T]")
        self.K = K
        self.N = self.Y.shape[0]
        self.L = self.T - K + 1  # number of windows per trajectory
        self.embedding_matrix = np.empty((self.N , self.L, self.K * self.D), dtype=float)
        self.flatten_embedding_matrix = np.empty((self.N * self.L, self.K * self.D), dtype=float)
        flatten_out_row = 0
        for n in range(self.N):
            out_row = 0
            for t in range(self.L):
                window = self.Y[n, t : t + self.K].reshape(-1)
                self.embedding_matrix[n,out_row] = window
                self.flatten_embedding_matrix[flatten_out_row] = window
                out_row += 1
                flatten_out_row+=1
        return self.embedding_matrix,self.flatten_embedding_matrix

    def compute_averages_embedding_chunk(self) -> (np.ndarray,np.ndarray):
        if self.Y.shape[2] == 2:
            self.embedded_av_velocity = np.mean(self.flatten_embedding_matrix[:,0::2],axis=1)
            self.embedded_av_ang_velocity = np.mean(self.flatten_embedding_matrix[:,1::2],axis=1)
            return self.embedded_av_velocity,self.embedded_av_ang_velocity
        elif self.Y.shape[2] == 3:
            self.embedded_av_velocity = np.mean(self.flatten_embedding_matrix[:,0::3],axis=1)
            self.embedded_av_ang_velocity = np.mean(self.flatten_embedding_matrix[:,1::3],axis=1)
            self.embedded_av_torsion_velocity = np.mean(self.flatten_embedding_matrix[:,2::3],axis=1)
            return self.embedded_av_velocity,self.embedded_av_ang_velocity,self.embedded_av_torsion_velocity

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def set_subsample(self, n_subsample: int, random_state: int = 0) -> None:
        """Generate and store indices for a random subsample of the data."""
        if self.flatten_embedding_matrix is None:
            raise RuntimeError("Call make_embedding() first.")
        
        if n_subsample > self.flatten_embedding_matrix.shape[0]:
            raise ValueError("n_subsample cannot be greater than the total number of samples.")

        rng = np.random.default_rng(random_state)
        self.indices = rng.choice(self.flatten_embedding_matrix.shape[0], n_subsample, replace=False)

    def fit_umap(self, n_neighbors: int = 15, min_dist: float = 0.1,with_cluster_centers = False) -> np.ndarray:
        """Fit a UMAP model and return the embedding.
        """
        if self.flatten_embedding_matrix is None:
            raise RuntimeError("Call make_embedding() first.")

        if self.n_subsample is not None and self.indices is None:
            self.set_subsample(self.n_subsample, random_state=random_state)

        if self.indices is not None:
            data_to_embed = self.flatten_embedding_matrix[self.indices]
        else:
            data_to_embed = self.flatten_embedding_matrix
        
        if with_cluster_centers:
            if self.cluster_centers_ is not None:
                data_to_embed = np.append(data_to_embed,self.cluster_centers_,axis=0)

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,n_components=2,metric="euclidean")
        reduced_all = reducer.fit_transform(data_to_embed)
        reduced_points = reduced_all[:self.n_subsample]
        reduced_centers = reduced_all[self.n_subsample:]
        return reduced_points,reduced_centers

    def make_cluster(self, n_clusters: int, random_state: int = 0, clustering_method: str = 'kmeans', batchsize: Optional[int] = None, tol: float = 0.001,degree : int =5) -> np.ndarray:
        """Run clustering on the embedding matrix and store the labels.
        Returns the 1‑D label array of length *self.embedding_matrix.shape[0]*.
        """
        if self.embedding_matrix is None:
            raise RuntimeError("Call make_embedding() first.")
        if n_clusters > self.flatten_embedding_matrix.shape[0]:
            raise ValueError("n_clusters must be lower than the number of samples")
        self.n_clusters = n_clusters

        if self.n_subsample is not None and self.indices is None:
            self.set_subsample(self.n_subsample, random_state=random_state)

        if self.indices is not None:
            subset = self.flatten_embedding_matrix[self.indices]
        else:
            subset = self.flatten_embedding_matrix

        if clustering_method == 'kmeans':
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
            labels_subsample = km.fit_predict(subset)
            self.cluster_centers_ = km.cluster_centers_
            # Predict labels for the entire dataset
            from scipy.spatial.distance import cdist
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

    def make_transition_matrix(
        self,
        tau: int = 1,
    ) -> np.ndarray:
        """Compute the row‑stochastic transition matrix P(τ)."""
        if self.labels is None:
            raise RuntimeError("Need labels; call cluster() or pass them explicitly.")        
        self.tau = tau
        C = count_transitions(self.labels, self.n_clusters, tau = self.tau,nsample = self.Nsample,TmKp1 = self.T - self.K +1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.P = C / C.sum(axis=1, keepdims=True)
        self.P[np.isnan(self.P)] = 0.0  # rows with zero counts
        self.pi = stationary_distribution(self.P)
        return self.P

    def reversibilized_matrix(self):
        if self.pi is None or self.P is None:
            raise RuntimeError("need a stationary distribution and a stochastic matrix first")
        rev_P = time_reversed_transition_matrix(self.P,self.pi)
        self.Pr = 0.5*(self.P+rev_P)
        return self.Pr

    def __repr__(self) -> str:  # pragma: no cover – just convenience
        out = [f"Embedding(K={self.K}, Ntraj={self.Y.shape[0]}, dims={self.D})"]
        if self.embedding_matrix is not None:
            out.append(f"embedding_matrix  : {self.embedding_matrix.shape}")
        if self.labels is not None:
            out.append(f"clusters      : {len(self.cluster_centers_)}")
        if self.P is not None:
            out.append("transition P  : available")
        if self.pi is not None:
            out.append("stationary π  : available")
        return "\n".join(out)

    def implied_timescales(self,tau: float) -> np.ndarray:
        """Return relaxation times τᵢ = −lag / ln λᵢ for eigenvalues λᵢ < 1."""
        if self.P is None:
            raise RuntimeError("Need the transition matrix first.")
        #lag = 1  # because transition_matrix() uses this default; could store the lag
        evals = np.linalg.eigvals(self.P)
        evals = np.real(evals)
        evals = evals[np.argsort(-evals)]  # descending order; λ₀ = 1 comes first
        return -tau / np.log(np.clip(evals[1:], 1e-15, 1 - 1e-15))
        #return tau/(1-evals[1:])
    # ------------------------------------------------------------------
    # Simulate trajectories
    # ------------------------------------------------------------------        
    def initialize_state(self):
        self.state = np.random.randint(0,set(self.labels).__len__())
    def pick_random_trajectory_in_cluster(self,cluster_id:int) -> NDArray[np.float_]:
        if self.labels is None:
            raise RuntimeError("Need the labels first.")
        if self.state is None:
            raise RuntimeError("Need to initialize the state first.")
        words = np.argwhere(self.labels==cluster_id)[:,0]
        index = np.random.randint(0,words.shape[0])
        return self.flatten_embedding_matrix[words[index]]
    def make_transition(self)-> int:
        """ given a current state : a cluster id, returns the id of the next cluster, selected according to the transition matrix."""
        if self.state is None:
            raise RuntimeError("Need to initialize the state first.")
        if self.P is None:
            raise RuntimeError("Need to make the transition matrix first.")
        cum_prob_array = np.cumsum(self.P[self.state])
        rd = np.random.randint(0,1000)/1000.
        self.state= np.searchsorted(cum_prob_array,rd , side='right')
        return self.state

