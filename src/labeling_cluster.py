import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict


def embed_move_type(data: pd.DataFrame, K: int, Nsamples=None, ID_NAME='label') -> np.ndarray:
    """Extract K-step sliding windows of `move_type` labels."""
    if Nsamples is None:
        wanted_ids = data[ID_NAME].unique()
    else:
        wanted_ids = data[ID_NAME].unique()[:int(Nsamples)]
    subset = data[data[ID_NAME].isin(wanted_ids)]

    labels = []
    T_min = np.inf
    for _, label_df in subset.groupby(ID_NAME, sort=False):
        traj_arr = label_df.sort_index()['move_type'].values
        labels.append(traj_arr)
        T_min = min(T_min, traj_arr.shape[0])

    if K < 1 or K > T_min:
        raise ValueError("K must be in the range [1, T]")

    Y = np.stack([label[:T_min] for label in labels])  # shape (N, T)
    N, T = Y.shape
    embedding_matrix = np.empty((N * (T - K + 1), K), dtype='U20')

    row = 0
    for n in range(N):
        for t in range(T - K + 1):
            embedding_matrix[row] = Y[n, t:t + K]
            row += 1

    return embedding_matrix


def average_out(embedded_label: np.ndarray, K: int) -> List[Counter]:
    """Convert K-label windows to normalized label frequency counters."""
    counters = []
    for window in embedded_label:
        count = Counter(window)
        for key in count:
            count[key] /= K
        counters.append(count)
    return counters


def label_clusters(
    n_clusters: int,
    labels: np.ndarray,
    embedded_counter: List[Counter]
) -> List[Dict[str, float]]:
    """Compute cluster-wise label frequencies from sliding label counters."""
    cluster_sum = [Counter() for _ in range(n_clusters)]
    cluster_nwin = [0] * n_clusters

    for i, c_id in enumerate(labels):
        cluster_sum[c_id] += embedded_counter[i]
        cluster_nwin[c_id] += 1

    cluster_freq = []
    for c_id in range(n_clusters):
        total = cluster_nwin[c_id]
        if total == 0:
            cluster_freq.append({'straight': 0.0, 'cast_and_surge': 0.0})
            continue

        freq_straight = cluster_sum[c_id]['straight'] / total
        freq_cast = cluster_sum[c_id]['cast_and_surge'] / total
        norm = freq_straight + freq_cast

        cluster_freq.append({
            'straight': freq_straight / norm if norm else 0.0,
            'cast_and_surge': freq_cast / norm if norm else 0.0
        })

    return cluster_freq

def compute_stereotypy_of_states(trajectory : pd.DataFrame, 
                                labels : np.ndarray,
                                K : int,
                                ID_NAME: str='label',
                                ):
    embedded_stereo = embed_move_type(trajectory,K=K,ID_NAME=ID_NAME)
    embedded_counter = average_out(embedded_stereo,K)
    label_clusters(n_clusters = Counter(labels).keys().__len__(), labels = labels,embedded_counter = embedded_counter)
    return label_clusters
