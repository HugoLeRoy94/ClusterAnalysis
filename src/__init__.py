# src/copepod/__init__.py

"""
Copepod trajectory analysis toolkit:
- Preprocessing and segmentation
- Delay embedding and clustering
- Markov model construction and analysis
"""

from .preprocessing import (
    compute_speed_turning_angles,
    compute_phases,
    split_trajectories,
    filter_trajectories,
)

from .embedding import Embedding

__all__ = [
    # preprocessing
    "compute_speed_turning_angles",
    "compute_phases",
    "split_trajectories",
    "filter_trajectories",
    # embedding
    "Embedding",
    # markov
    "count_transitions",
    "stationary_distribution",
    "entropy_rate",
    "time_reversed_transition_matrix",
    "metastability",
    "reversibilize_transition_matrix",
    "implied_timescales",
]
