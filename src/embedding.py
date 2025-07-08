from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

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
    ) -> None:
        self.columns = columns
        self.D = len(columns)
        self.ID_NAME = ID_NAME
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
    def make_cluster(self, n_clusters: int, random_state: int = 0, n_subsample: Optional[int] = None, clustering_method: str = 'kmeans', batchsize: Optional[int] = None, tol: float = 0.001) -> np.ndarray:
        """Run clustering on the embedding matrix and store the labels.
        Returns the 1‑D label array of length *self.embedding_matrix.shape[0]*.
        """
        if self.embedding_matrix is None:
            raise RuntimeError("Call make_embedding() first.")
        if n_clusters > self.flatten_embedding_matrix.shape[0]:
            raise ValueError("n_clusters must be lower than the number of samples")
        self.n_clusters = n_clusters

        X = self.flatten_embedding_matrix
        subset = X if n_subsample is None else X[:n_subsample]

        if clustering_method == 'kmeans':
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
            self.labels = km.fit_predict(subset)
            self.cluster_centers_ = km.cluster_centers_
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

        return self.labels
    def make_transition_matrix(
        self,
        tau: int = 1,
    ) -> np.ndarray:
        """Compute the row‑stochastic transition matrix P(τ)."""
        if self.labels is None:
            raise RuntimeError("Need labels; call cluster() or pass them explicitly.")        
        self.tau = tau
        C = self.count_transitions(self.labels, self.n_clusters, tau = self.tau,nsample = self.Nsample,TmKp1 = self.T - self.K +1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.P = C / C.sum(axis=1, keepdims=True)
        self.P[np.isnan(self.P)] = 0.0  # rows with zero counts
        self.pi = self.stationary_distribution(self.P)
        return self.P

    def reversibilized_matrix(self):
        if self.pi is None or self.P is None:
            raise RuntimeError("need a stationary distribution and a stochastic matrix first")
        rev_P = self.time_reversed_transition_matrix(self.P,self.pi)
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
        #return -lag / np.log(np.clip(evals[1:], 1e-15, 1 - 1e-15))
        return tau/(1-evals[1:])
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
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    """
       ▄▄▄  ■  ▗▞▀▜▌   ■  ▄ ▗▞▀▘    ▗▞▀▀▘█  ▐▌▄▄▄▄  ▗▞▀▘   ■  ▄  ▄▄▄  ▄▄▄▄   ▄▄▄ 
    ▀▄ ▗▄▟▙▄▖▝▚▄▟▌▗▄▟▙▄▖▄ ▝▚▄▖    ▐▌   ▀▄▄▞▘█   █ ▝▚▄▖▗▄▟▙▄▖▄ █   █ █   █ ▀▄▄  
     ▄▄▄▀ ▐▌         ▐▌  █         ▐▛▀▘      █   █       ▐▌  █ ▀▄▄▄▀ █   █ ▄▄▄▀ 
          ▐▌         ▐▌  █         ▐▌                    ▐▌  █                  
          ▐▌         ▐▌                                  ▐▌                     
    """
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def count_transitions(labels: np.ndarray, n_clusters: int, tau: int,nsample: int, TmKp1: int) -> np.ndarray:
        """
        Return the raw transition count matrix C (without normalisation).
        Notice that we do not concatenate the pieces of trajectories one after the other
        to avoid unrealistic transition
        
        nsample: the number of trajectories
        TmKp1 : T - K +1
        """
        C = np.zeros((n_clusters, n_clusters), dtype=float)
        for n in range(nsample):            
            for start in range(TmKp1 - tau):
                i, j = labels[n*TmKp1+start], labels[n*TmKp1+start + tau]
                C[i, j] += 1.0
        return C

    @staticmethod
    def stationary_distribution(P: NDArray[np.float_], tol: float = 1e-12, maxiter: int = 10000) -> NDArray[np.float_]:
        """
        Compute the stationary distribution π such that πᵀ P = πᵀ.

        Uses power iteration on Pᵀ.

        Returns
        -------
        pi : 1D ndarray
            Stationary distribution.
        """
        n = P.shape[0]
        pi = np.ones(n) / n
        i = 0
        while True:
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi, 1) < tol:
                break
            pi = pi_new
            i+=1
            if i >= maxiter:
                val,vec = np.linalg.eig(P.T)
                vec = vec[:,np.argsort(np.real(val))]            
                return np.real(vec[:,-1]/np.sum(vec[:,-1]))

        return pi
    @staticmethod
    def entropy_rate(P: np.ndarray, pi: Optional[np.ndarray] = None) -> float:
        """Shannon entropy rate *h = -∑_i π_i ∑_j P_ij log P_ij* in *bits* per step.

        Parameters
        ----------
        P : ndarray, shape (n, n)
            Row‑stochastic transition matrix.
        pi : Optional ndarray, shape (n,)
            Stationary distribution.  If *None* it is computed internally.
        base : float, default 2.0
            Logarithm base.  ``base=2`` → bits; ``np.e`` → nats.
        """
        if pi is None:
            pi = Embedding.stationary_distribution(P)
        with np.errstate(divide="ignore", invalid="ignore"):
            logP = np.log(P) #/ np.log(base)
            logP[np.isneginf(logP)] = 0.0  # define 0·log0 = 0
            #h= -np.sum(np.sum(P * logP,axis=1)*pi)
            h = -(pi[:, None] * P * logP).sum()
        return float(h)
    @staticmethod
    def time_reversed_transition_matrix(P: np.ndarray, pi: np.ndarray, eps=1e-15) -> np.ndarray:
        """
        Compute the time-reversed transition matrix from a row-stochastic matrix P and stationary distribution pi.

        Parameters
        ----------
        P : (N, N) ndarray
            Row-stochastic transition matrix P_{ij} = P(i → j)
        pi : (N,) ndarray
            Stationary distribution pi[i] > 0 and sum(pi) == 1
        eps : float
            Small number to avoid division by zero

        Returns
        -------
        P_rev : (N, N) ndarray
            Time-reversed transition matrix: P_{ij}(-tau)
        """
        pi = np.asarray(pi, dtype=float)
        P  = np.asarray(P, dtype=float)

        if P.shape[0] != P.shape[1] or P.shape[0] != pi.shape[0]:
            raise ValueError("Shape mismatch: P must be (N, N) and pi must be (N,)")
        P_rev = np.zeros(P.shape,dtype=float)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if pi[i]!=0:
                    P_rev[i,j] = pi[j]*P[j,i]/pi[i]

        return P_rev
    @staticmethod
    def metastability(P: NDArray[np.float_], pi: NDArray[np.float_], S: NDArray[np.int_]) -> float:
        """
        Metastability of a subset S:
        Probability of remaining in S after one step, conditioned on being in S.

        h(S) = ∑_{i∈S, j∈S} π_i P_ij / ∑_{i∈S} π_i
        """
        pi_S = pi[S]
        P_SS = P[np.ix_(S, S)]
        numer = np.sum(pi_S[:, None] * P_SS)
        denom = np.sum(pi_S)
        return numer / denom if denom > 0 else 0.0

    @staticmethod
    def reconstruct_Y_from_embedding(embedding_matrix: np.ndarray, K: int, d: int) -> np.ndarray:
        """
        Reconstruct the original Y array of shape (N, T, d) from the embedding_matrix.

        Parameters
        ----------
        embedding_matrix : ndarray of shape (N, T-K+1, K*d)
            The delay embedding for N trajectories.
        K : int
            The delay length.
        d : int
            Number of dynamical coordinates.

        Returns
        -------
        Y : ndarray of shape (N, T, d)
            Reconstructed original trajectories (approximately).
        """
        N, L, _ = embedding_matrix.shape
        T = L + K - 1
        Y = np.zeros((N, T, d), dtype=embedding_matrix.dtype)
        counts = np.zeros((N, T, d), dtype=int)

        for t in range(L):
            window = embedding_matrix[:, t].reshape(N, K, d)
            for k in range(K):
                Y[:, t + k] += window[:, k]
                counts[:, t + k] += 1

        # Average overlapping entries
        counts[counts == 0] = 1  # avoid divide-by-zero
        Y /= counts
        return Y