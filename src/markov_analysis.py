from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from src.trajectory_utils import (
    count_transitions,
    stationary_distribution,
    time_reversed_transition_matrix,
    metastability,
)
from src.embedding_base import EmbeddingBase

__all__ = ["MarkovAnalysis"]


class MarkovAnalysis:
    """Analyzes the Markov process resulting from a clustering."""

    def __init__(
        self,
        embedding_base_instance: EmbeddingBase,
    ) -> None:
        if embedding_base_instance.labels is None:
            raise RuntimeError("EmbeddingBase instance must have labels (i.e., clustering must be performed).")
        self.labels = embedding_base_instance.labels
        self.n_clusters = embedding_base_instance.n_clusters
        self.nsample = embedding_base_instance.Nsample
        self.T = embedding_base_instance.T
        self.K = embedding_base_instance.K
        self.P: Optional[np.ndarray] = None
        self.pi: Optional[np.ndarray] = None
        self.Pr: Optional[np.ndarray] = None
        self.slow_mode : Optional[np.ndarray] = None
        self.make_transition_matrix()

    def make_transition_matrix(
        self,
        tau: int = 1,
    ) -> np.ndarray:
        """Compute the row‑stochastic transition matrix P(τ)."""
        self.tau = tau
        C = count_transitions(self.labels, self.n_clusters, tau=self.tau, nsample=self.nsample, TmKp1=self.T - self.K + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.P = C / C.sum(axis=1, keepdims=True)
        self.P[np.isnan(self.P)] = 0.0  # rows with zero counts
        self.pi = stationary_distribution(self.P)
        return self.P

    def reversibilized_matrix(self):
        if self.pi is None or self.P is None:
            raise RuntimeError("need a stationary distribution and a stochastic matrix first")
        rev_P = time_reversed_transition_matrix(self.P, self.pi)
        self.Pr = 0.5 * (self.P + rev_P)
        return self.Pr

    def implied_timescales(self, tau: float) -> np.ndarray:
        """Return relaxation times τᵢ = −lag / ln λᵢ for eigenvalues λᵢ < 1."""
        if self.P is None:
            raise RuntimeError("Need the transition matrix first.")
        evals = np.linalg.eigvals(self.P)
        evals = np.real(evals)
        evals = evals[np.argsort(-evals)]  # descending order; λ₀ = 1 comes first
        return -tau / np.log(np.clip(evals[1:], 1e-15, 1 - 1e-15))
    def compute_spectrum(self):
        self.val,self.vec = np.linalg.eig(self.P.T)
        self.vec = self.vec[:,np.argsort(self.val)]
        self.val = self.val[np.argsort(self.val)]
        self.slow_mode = np.real(self.vec[:,-2])

    def compute_tr_spectrum(self):
        self.tr_val,self.tr_vec = np.linalg.eig(self.Pr.T)
        self.tr_vec = self.tr_vec[:,np.argsort(self.tr_val)]
        self.tr_val = self.tr_val[np.argsort(self.tr_val)]
        self.tr_slow_mode = np.real(self.tr_vec[:,-2])

    def compute_metastability(self):
        self.thresholds = np.linspace(min(self.slow_mode),max(self.slow_mode),100)
        meta_in,meta_out = list(),list()
        for threshold in thresholds:
            meta_in.append(metastability(self.P,self.pi,np.argwhere(self.slow_mode>=threshold)[:,0]))
            meta_out.append(metastability(self.P,self.pi,np.argwhere(self.slow_mode<threshold)[:,0]))
        self.meta_in = np.array(meta_in)
        self.meta_out = np.array(meta_out)

