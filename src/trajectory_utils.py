import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Optional
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R


def count_transitions(labels: np.ndarray, n_clusters: int, tau: int, nsample: int, TmKp1: int) -> np.ndarray:
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
def embed_move_type(data : pd.DataFrame, K:int, Nsamples: Optional[int] = None,ID_NAME:str='label') -> np.ndarray:
    """
    data 
    """
    if Nsamples == "all":
        wanted_ids = data[ID_NAME].unique()
    else:
        wanted_ids = data[ID_NAME].unique()[: int(Nsamples)]
    subset = data[data[ID_NAME].isin(wanted_ids)]
    labels = []
    T_min = np.inf
    for _, label_df in subset.groupby(ID_NAME, sort=False):
        traj_arr = traj_df.sort_index()['move_type'].values
        trajs.append(traj_arr)
        T_min = min(T_min, traj_arr.shape[0])
    Y = np.stack([traj[: T_min] for traj in trajs])  # shape (N, T)

    if K < 1 or K > T_min:
        raise ValueError("K must be in the range [1, T]")
    L = T - K + 1  # number of windows per trajectory
    embedding_matrix = np.empty((self.N * self.L, self.K), dtype=str)
    flatten_out_row = 0
    for n in range(self.N):
        for t in range(self.L):
            window = Y[n, t : t + K]
            embedding_matrix[flatten_out_row] = window
            flatten_out_row+=1
    return embedding_matrix

def canonicalize_trajectory(coords, *, return_rotation=False, tol=1e-12):
    """
    Rotate `coords` (K×3) into a unique canonical frame.
    Any rigid-body rotation + translation of the same trajectory
    maps to the *identical* output.

    Algorithm
    ---------
    1.  centre at the centroid
    2.  PCA → eigenvectors V (columns)
    3.  for each axis j:                       # sign disambiguation
        m3 = Σ (x_j')³   (third central moment)
        if |m3| < tol use   Σ x_j'² x_{j+1}'
        flip V[:,j] if m3 < 0
    4.  make the basis right-handed (det = +1)
    5.  rotated = centred @ V

    Returns
    -------
    canon : (K,3)  canonical coordinates (centroid at the origin)
    R      : (3,3) rotation matrix  (only if `return_rotation=True`)
    """

    X   = np.asarray(coords, dtype=float)
    C   = X - X.mean(axis=0)               # 1
    _,  _, Vt = np.linalg.svd(C, full_matrices=False)
    V   = Vt.T                             # 2

    Y   = C @ V                            # projections for moments
    for j in range(3):                     # 3
        m3 = (Y[:, j] ** 3).sum()
        if abs(m3) < tol:                  # nearly symmetric
            k = (j + 1) % 3
            m3 = (Y[:, j]**2 * Y[:, k]).sum()
        if m3 < 0:
            V[:, j] *= -1
            Y[:, j] *= -1

    if np.linalg.det(V) < 0:               # 4
        V[:, 2] *= -1
        Y[:, 2] *= -1

    canon = Y                               # 5
    return (canon, V) if return_rotation else canon