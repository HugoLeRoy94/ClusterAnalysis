import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Dict, Any
from deeptime.markov.msm import MarkovStateModel

import sys
sys.path.append('../src')
from src.trajectory_utils import (stationary_distribution)

@dataclass
class CrispPCCAPlusResult:
    chi_soft: NDArray[np.float64]            # (N, n_c_soft)
    chi_crisp: NDArray[np.float64]           # (N, n_c_active)
    labels: NDArray[np.int_]                # (N,) remapped to [0..n_c_active-1]
    pi: NDArray[np.float64]                  # (N,)
    Pc_crisp: NDArray[np.float64]            # (n_c_active, n_c_active)
    active_macro_ids: NDArray[np.int_]      # indices (in original 0..n_c-1) that survived
    meta: Dict[str, Any]                    # small extras (e.g. counts, dropped ids)
    Pc_soft_timescale: Optional[NDArray[np.float64]] = None  # (n_c_active,n_c_active), optional

class crisp_PCCAp:
    def __init__(self, P: NDArray[np.float64], n_clusters: int,
                 compute_now: bool = True,
                 also_soft_timescale_preserving: bool = False,
                 prune_zero_weight: bool = True,
                 zero_tol: float = 0.0) -> None:
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be square.")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
            print(P)
            raise ValueError("P must be row-stochastic.")
        self.P = P
        self.N = P.shape[0]
        self.n_clusters = int(n_clusters)
        self.also_soft_timescale_preserving = also_soft_timescale_preserving
        self.prune_zero_weight = prune_zero_weight
        self.zero_tol = float(zero_tol)

        self._msm: Optional[MarkovStateModel] = None
        self.re: Optional[CrispPCCAPlusResult] = None
        if compute_now:
            self.initialize_PCCAp()


    def initialize_PCCAp(self) -> CrispPCCAPlusResult:
        # MSM + PCCA+
        self._msm = MarkovStateModel(transition_matrix=self.P,reversible=True)
        pcca = self._msm.pcca(self.n_clusters)
        chi_soft = getattr(pcca, "memberships", None)
        if chi_soft is None:
            raise RuntimeError("deeptime PCCA+ had no 'chi'/'membership'.")
        chi_soft = np.asarray(chi_soft, float)  # (N, n_c_soft) usually n_c_soft == n_clusters

        # Crispify (argmax)
        labels = np.argmax(chi_soft, axis=1)
        chi_crisp = np.zeros_like(chi_soft)
        chi_crisp[np.arange(self.N), labels] = 1.0

        # Stationary weights
        pi = stationary_distribution(self.P)
        D2 = np.diag(pi)
        mu_c = chi_crisp.T @ pi                       # (n_c_soft,)

        # Optionally prune zero-weight macrostates
        if self.prune_zero_weight:
            active = mu_c > self.zero_tol             # keep strictly positive (or > tol)
            active_ids = np.where(active)[0]
            dropped_ids = np.where(~active)[0]

            # Remap labels to compact ids; states mapped to dropped ids keep their label but will be removed from chi_crisp
            remap = -np.ones(len(active), dtype=int)
            remap[active_ids] = np.arange(active_ids.size)
            labels = remap[labels]

            #  Keep only active columns in χ (drop empty macrostates)
            chi_crisp = chi_crisp[:, active]
            mu_c = mu_c[active]
            #chi_soft_active = chi_soft[:, active]     # keep same macro set for soft operator
        else:
            active_ids = np.arange(chi_crisp.shape[1])
            dropped_ids = np.array([], dtype=int)
            #chi_soft_active = chi_soft

        # Build coarse stochastic Pc (crisp)
        Dc2_inv = np.zeros((mu_c.size, mu_c.size))
        nz = mu_c > 0.0
        Dc2_inv[nz, nz] = 1.0 / mu_c[nz]
        Pc_crisp = Dc2_inv @ (chi_crisp.T @ (D2 @ (self.P @ chi_crisp)))

        # Optional: timescale-preserving soft operator, restricted to active set
        Pc_soft_timescale = None
        if self.also_soft_timescale_preserving:
            G = chi_soft_active.T @ (D2 @ chi_soft_active)
            G_inv = np.linalg.pinv(G)  # robust if near-singular
            Pc_soft_timescale = G_inv @ (chi_soft_active.T @ (D2 @ (self.P @ chi_soft_active)))

        res = CrispPCCAPlusResult(
            chi_soft=chi_soft,                  # original soft memberships (pre-prune view)
            chi_crisp=chi_crisp,               # pruned columns only
            labels=labels,                     # remapped to [0..n_active-1], -1 if a state was assigned to a dropped macro (rare if prune)
            pi=pi,
            Pc_crisp=Pc_crisp,
            active_macro_ids=active_ids,
            meta={
                "dropped_macro_ids": dropped_ids,
                "n_active": int(len(active_ids)),
                "n_requested": int(self.n_clusters),
                "zero_tol": self.zero_tol
            },
            Pc_soft_timescale=Pc_soft_timescale
        )
        self.res = res
        return res

    def result(self) -> CrispPCCAPlusResult:
        return self.res if self.res is not None else self.initialize_PCCAp()
    
    def coarse_entropy_rate(self) -> float:
        """
        Entropy rate of the coarse-grained Markov chain (crisp macrostates).
        H = -sum_i pi_c(i) * sum_j P_c(i,j) log P_c(i,j)
        
        Parameters
        ----------
        base : float
            Log base. Use np.e for nats, 2 for bits.

        Returns
        -------
        float
            Entropy rate in 'base' units per step.
        """
        if self.res is None:
            self.initialize_PCCAp()

        Pc = self.res.Pc_crisp                     # (nc, nc) row-stochastic
        # coarse stationary distribution: pi_c = chi_crisp^T pi  (already restricted to active macrostates)
        pi_c = self.res.chi_crisp.T @ self.res.pi
        # numerical guard (should already sum to 1 after pruning)
        s = pi_c.sum()
        if s <= 0:
            return 0.0
        pi_c = pi_c / s

        # handle 0*log(0) = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            logPc = np.log(Pc, dtype=float)
            logPc[~np.isfinite(logPc)] = 0.0  # sets log(0) -> 0 so 0*log(0) contributes 0
            return -np.sum(pi_c[:, None] * Pc * logPc)

