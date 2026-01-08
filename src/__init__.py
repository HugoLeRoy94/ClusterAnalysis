# src/__init__.py

"""
Copepod trajectory analysis toolkit:
- Preprocessing and segmentation
- Delay embedding and clustering
- Markov model construction and analysis
"""

# --- Main Imports ---
from .preprocessing import (
    compute_speed_turning_angles,
    compute_phases,
    split_trajectories,
    filter_trajectories,
)

from .embedding_base import EmbeddingBase
from .embedding import Embedding
from .embedding_trans import EmbeddingTrans
# Assuming EmbeddingPosition is now in embedding.py or similar, import it if needed
# from .embedding import EmbeddingPosition 

# --- Markov Imports (Ensure these exist in markov_analysis.py) ---
from .markov_analysis import (
    count_transitions,
    stationary_distribution,
    entropy_rate,
    time_reversed_transition_matrix,
    metastability,
)

# --- Legacy Sub-package Import ---
# This makes 'src.legacy' available
from . import legacy

__all__ = [
    # preprocessing
    "compute_speed_turning_angles",
    "compute_phases",
    "split_trajectories",
    "filter_trajectories",
    # embedding
    "Embedding",
    "EmbeddingBase",
    "EmbeddingTrans",
    # markov
    "count_transitions",
    "stationary_distribution",
    "entropy_rate",
    "time_reversed_transition_matrix",
    "metastability",
    # legacy
    "legacy"
]