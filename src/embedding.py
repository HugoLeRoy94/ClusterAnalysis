

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.embedding_base import EmbeddingBase
from src.markov_analysis import Markov,StochasticMatrix

__all__ = ["Embedding"]


class Embedding(EmbeddingBase):
    """
    Time-delay embedding + k-means + Markov analysis.

    Initialize either from a DataFrame (`data` + `columns` + `ID_NAME`),
    or directly from a prebuilt array `Y` with shape (N, T, d).
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        columns: Optional[Sequence[str]] = None,
        ID_NAME: str = "ID",
        n_trajectories: Optional[int] = None,
        Y: Optional[np.ndarray] = None,
        n_windows: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            data=data,
            columns=columns,
            ID_NAME=ID_NAME,
            n_trajectories=n_trajectories,
            Y=Y,
            n_windows=n_windows,
            rng=rng,
        )
        self.P: Optional[np.ndarray] = None
        self.pi: Optional[np.ndarray] = None

    def analyze_markov_process(self) -> Markov:
        """Create and return a MarkovAnalysis object."""
        if self.labels is None:
            raise RuntimeError("Need labels; call make_cluster() first.")
        return Markov(self)

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

    


