from typing import Optional
import numpy as np
from numpy.typing import NDArray
from src.trajectory_utils import (
    stationary_distribution,
    time_reversed_transition_matrix,
    count_transitions,
    metastability,
    entropy_rate,
)
from src.embedding_base import EmbeddingBase

def _sort_idx(vals: np.ndarray) -> np.ndarray:
    # sort by increasing Re(λ), break ties by |λ|
    return np.lexsort((np.abs(vals), np.real(vals)))

class StochasticMatrix:
    def __init__(self, P: NDArray[np.float64]) -> None:
        self.P = P
        self.pi: Optional[np.ndarray] = stationary_distribution(P)
        self.Pr: Optional[np.ndarray] = None
        self.slow_mode: Optional[np.ndarray] = None
        self.tr_slow_mode: Optional[np.ndarray] = None

    def reversibilized_matrix(self) -> NDArray[np.float64]:
        if self.pi is None or self.P is None:
            raise RuntimeError("Need stationary distribution and stochastic matrix.")
        rev_P = time_reversed_transition_matrix(self.P, self.pi)
        self.Pr = 0.5 * (self.P + rev_P)
        return self.Pr

    def implied_timescales(self, tau: float) -> NDArray[np.float64]:
        evals = np.linalg.eigvals(self.P)
        evals = np.real(evals)
        evals = evals[np.argsort(-evals)]
        return -tau / np.log(np.clip(evals[1:], 1e-15, 1 - 1e-15))

    def compute_spectrum(self) -> None:
        # right: P r = λ r
        vals_r, vecs_r = np.linalg.eig(self.P)
        # left:  l^T P = λ l^T  <=>  P^T l = λ l
        vals_l, vecs_l = np.linalg.eig(self.P.T)
        # sort both with the same rule
        idx_r = _sort_idx(vals_r)
        idx_l = _sort_idx(vals_l)
        self.eigvals = np.real(vals_l[idx_r])          # single eigenvalue array
        self.right_eigvecs = np.real(vecs_r[:, idx_r]) # columns = right eigenvectors
        self.left_eigvecs  = np.real(vecs_l[:, idx_l]) # columns = left  eigenvectors
        # convenience: slow modes exclude λ≈1 (last after this sorting)
        self.slow_modes = self.right_eigvecs[:, :-1]
        self.slow_mode  = self.slow_modes[:, -1] if self.slow_modes.size else None

    def compute_tr_spectrum(self) -> None:
        if self.Pr is None:
            raise RuntimeError("Run reversibilized_matrix first.")

        vals_r, vecs_r = np.linalg.eig(self.Pr)
        vals_l, vecs_l = np.linalg.eig(self.Pr.T)

        idx_r = _sort_idx(vals_r)
        idx_l = _sort_idx(vals_l)

        self.tr_eigvals = np.real(vals_l[idx_r])
        self.tr_right_eigvecs = np.real(vecs_r[:, idx_r])
        self.tr_left_eigvecs  = np.real(vecs_l[:, idx_l])

        self.tr_slow_modes = self.tr_right_eigvecs[:, :-1]
        self.tr_slow_mode  = self.tr_slow_modes[:, -1] if self.tr_slow_modes.size else None


    def compute_metastability(self,time_reversed = True) -> None:
        slow_mode= self.slow_modes[-1]
        if time_reversed:
            slow_mode = self.tr_slow_modes[-1]
        if slow_mode is None:
            raise RuntimeError("Compute spectrum first.")
        self.thresholds = np.linspace(slow_mode.min(), slow_mode.max(), 100)
        meta_in, meta_out = [], []
        for t in self.thresholds:
            A = np.where(slow_mode >= t)[0]
            B = np.where(slow_mode < t)[0]
            meta_in.append(metastability(self.P, self.pi, A))
            meta_out.append(metastability(self.P, self.pi, B))
        self.meta_in = np.array(meta_in)
        self.meta_out = np.array(meta_out)
    
    def compute_entropy_rate(self) -> float:
        return entropy_rate(self.P, self.pi)



class Markov(StochasticMatrix):
    def __init__(self, embedding: EmbeddingBase, tau: int = 1) -> None:
        if embedding.labels is None:
            raise RuntimeError("Embedding must have cluster labels.")
        
        self.labels = embedding.labels
        self.n_clusters = embedding.n_clusters
        self.Nsample = embedding.Nsample
        self.T = embedding.T
        self.K = embedding.K
            
        self.tau = tau
        self.state: Optional[int] = None

        self.make_transition_matrix()

    def make_transition_matrix(self) -> NDArray[np.float64]:
        C = count_transitions(
            self.labels,
            self.n_clusters,
            tau=self.tau,
            nsample=self.Nsample,
            TmKp1=self.T - self.K + 1,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            P = C / C.sum(axis=1, keepdims=True)
        P[np.isnan(P)] = 0.0
        super().__init__(P)
    def initialize_state(self):
        self.state = np.random.randint(0, self.n_clusters)
    
    def make_transition(self) -> int:
        """ given a current state : a cluster id, returns the id of the next cluster, selected according to the transition matrix."""
        if self.state is None:
            raise RuntimeError("Need to initialize the state first.")
        if self.P is None:
            raise RuntimeError("Need to make the transition matrix first.")
        cum_prob_array = np.cumsum(self.P[self.state])
        rd = np.random.randint(0, 1000) / 1000.0
        self.state = np.searchsorted(cum_prob_array, rd, side="right")
        return self.state
    def build_trajectory(self,T_tot:int)->np.ndarray:
        res = list()
        N_mkv_steps = T_tot//self.K
        self.initialize_state()
        res.append(self.state)
        for step in range(N_mkv_steps):
            res.append(self.make_transition())
        return np.array(res)