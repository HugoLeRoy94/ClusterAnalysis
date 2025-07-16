from typing import Optional
import numpy as np
from numpy.typing import NDArray
from src.trajectory_utils import (
    stationary_distribution,
    time_reversed_transition_matrix,
    count_transitions,
    metastability,
)
from src.embedding_base import EmbeddingBase

class StochasticMatrix:
    def __init__(self, P: NDArray[np.float_]) -> None:
        self.P = P
        self.pi: Optional[np.ndarray] = stationary_distribution(P)
        self.Pr: Optional[np.ndarray] = None
        self.slow_mode: Optional[np.ndarray] = None

    def reversibilized_matrix(self) -> NDArray[np.float_]:
        if self.pi is None or self.P is None:
            raise RuntimeError("Need stationary distribution and stochastic matrix.")
        rev_P = time_reversed_transition_matrix(self.P, self.pi)
        self.Pr = 0.5 * (self.P + rev_P)
        return self.Pr

    def implied_timescales(self, tau: float) -> NDArray[np.float_]:
        evals = np.linalg.eigvals(self.P)
        evals = np.real(evals)
        evals = evals[np.argsort(-evals)]
        return -tau / np.log(np.clip(evals[1:], 1e-15, 1 - 1e-15))

    def compute_spectrum(self) -> None:
        self.val, self.vec = np.linalg.eig(self.P.T)
        idx = np.argsort(self.val)
        self.val = self.val[idx]
        self.vec = self.vec[:, idx]
        self.slow_mode = np.real(self.vec[:, -2])

    def compute_tr_spectrum(self) -> None:
        if self.Pr is None:
            raise RuntimeError("Run reversibilized_matrix first.")
        self.tr_val, self.tr_vec = np.linalg.eig(self.Pr.T)
        idx = np.argsort(self.tr_val)
        self.tr_val = self.tr_val[idx]
        self.tr_vec = self.tr_vec[:, idx]
        self.tr_slow_mode = np.real(self.tr_vec[:, -2])

    def compute_metastability(self) -> None:
        if self.slow_mode is None:
            raise RuntimeError("Compute spectrum first.")
        self.thresholds = np.linspace(self.slow_mode.min(), self.slow_mode.max(), 100)
        meta_in, meta_out = [], []
        for t in self.thresholds:
            A = np.where(self.slow_mode >= t)[0]
            B = np.where(self.slow_mode < t)[0]
            meta_in.append(metastability(self.P, self.pi, A))
            meta_out.append(metastability(self.P, self.pi, B))
        self.meta_in = np.array(meta_in)
        self.meta_out = np.array(meta_out)



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

        self.make_transition_matrix()

    def make_transition_matrix(self) -> NDArray[np.float_]:
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