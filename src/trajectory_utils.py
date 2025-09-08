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
        pi = stationary_distribution(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.log(P) #/ np.log(base)
        logP[np.isneginf(logP)] = 0.0  # define 0·log0 = 0
        #h= -np.sum(np.sum(P * logP,axis=1)*pi)
        h = -(pi[:, None] * P * logP).sum()
    return float(h)

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
    # Step 6: chiral disambiguation via lexicographic minimization
    mirrors = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
        [-1, -1, -1]
    ])  # shape (8, 3)

    mirrored = np.einsum('ij,kj->kij', canon, mirrors)  # shape (8, K, 3)

    flat = mirrored.reshape(8, -1)  # shape (8, 3K)
    best_index = np.lexsort(flat.T)[0]
    canon = mirrored[best_index]


    return (canon, V) if return_rotation else canon


import numpy as np

def _norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else v*0.0

def _rot_between(a, b, eps=1e-12):
    """
    Minimal rotation that maps unit vector a -> unit vector b.
    Returns a 3x3 rotation matrix.
    """
    a = _norm(a)
    b = _norm(b)
    c = np.dot(a, b)
    if c > 1 - eps:            # nearly identical
        return np.eye(3)
    if c < -1 + eps:           # opposite; pick any axis ⟂ a
        # Choose axis as any unit vector orthogonal to 'a'
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = _norm(np.cross(a, tmp))
        return _rot_axis_angle(axis, np.pi)
    axis = _norm(np.cross(a, b))
    angle = np.arccos(np.clip(c, -1.0, 1.0))
    return _rot_axis_angle(axis, angle)

def _rot_axis_angle(axis, angle):
    """
    Rodrigues' rotation formula. Axis must be unit-length (or zero for identity).
    """
    axis = _norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ],
    ])

def _project_to_plane(v, n):
    n = _norm(n)
    return v - np.dot(v, n) * n

def _signed_angle_in_plane(a, b, plane_normal, eps=1e-12):
    """
    Signed angle from a to b within the plane orthogonal to plane_normal.
    """
    a_p = _project_to_plane(a, plane_normal)
    b_p = _project_to_plane(b, plane_normal)
    na = np.linalg.norm(a_p); nb = np.linalg.norm(b_p)
    if na < eps or nb < eps:
        return 0.0
    a_u = a_p / na
    b_u = b_p / nb
    cosang = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
    ang = np.arccos(cosang)
    # Sign via right-hand rule around plane_normal
    sign = np.sign(np.dot(plane_normal, np.cross(a_u, b_u)))
    return ang * sign

def _discrete_tangent(piece):
    # piece: (K,3)
    # use last two for end, first two for start
    t_start = _norm(piece[1] - piece[0])
    t_end   = _norm(piece[-1] - piece[-2])
    return t_start, t_end

def _discrete_curvature(piece):
    # discrete 2nd-difference at start and end
    K_start = piece[2] - 2*piece[1] + piece[0]
    K_end   = piece[-1] - 2*piece[-2] + piece[-3]
    return K_start, K_end

def reconstruct_trajectory_from_canonical_pieces(
    pieces, K, d=3, enforce_curvature=True, eps=1e-10
):
    """
    Stitch canonicalized windows into a single 3D trajectory.

    Parameters
    ----------
    pieces : array, shape (M, K*d)
        Consecutive canonicalized segments (flattened row-major).
        Each segment is centered/rotated canonically.
    K : int
        Window length used to build each piece.
    d : int
        Dimensionality (must be 3 here).
    enforce_curvature : bool
        If True, add a twist R_psi to match curvature direction (C2).
        If False, only enforce tangent continuity (C1).
    eps : float
        Numerical tolerance.

    Returns
    -------
    X : array, shape (K + (M-1)*(K-1), 3)
        Reconstructed global trajectory.
    """
    assert d == 3, "This reconstruction targets 3D."
    M = pieces.shape[0]
    assert pieces.shape[1] == K*d

    # reshape all pieces to (K,3)
    segs = pieces.reshape(M, K, d).copy()

    # Initialize global coordinates with the first piece as-is
    X = [segs[0][i].copy() for i in range(K)]
    # Keep track of the current last-segment (already placed)
    prev = np.stack(X[-K:], axis=0)  # last placed segment (K,3)

    # Precompute end tangent/curvature of the placed segment
    t_prev_start, t_prev_end = _discrete_tangent(prev)
    k_prev_start, k_prev_end = _discrete_curvature(prev)

    for m in range(1, M):
        seg = segs[m]  # (K,3), canonical

        # Compute start tangent/curvature of the new piece (in its own frame)
        t_next_start, _ = _discrete_tangent(seg)
        k_next_start, _ = _discrete_curvature(seg)

        # (1) Align tangents: R_theta maps t_next_start -> t_prev_end
        R_theta = _rot_between(t_next_start, t_prev_end)

        seg_rot = seg @ R_theta.T
        k_next_start_rot = k_next_start @ R_theta.T

        # (2) Optional twist around the (now aligned) tangent to match curvature
        if enforce_curvature:
            # Project curvature vectors to plane ⟂ t_prev_end
            # If curvature is degenerate, skip twist.
            a = _project_to_plane(k_prev_end, t_prev_end)
            b = _project_to_plane(k_next_start_rot, t_prev_end)
            if np.linalg.norm(a) > eps and np.linalg.norm(b) > eps:
                psi = _signed_angle_in_plane(b, a, t_prev_end)
                R_psi = _rot_axis_angle(t_prev_end, psi)
                seg_rot = seg_rot @ R_psi.T
            # else: psi = 0 (no reliable curvature direction)

        # (3) Translate so seg_rot[0] lands on prev[-1]
        delta = prev[-1] - seg_rot[0]
        seg_place = seg_rot + delta

        # Append without duplicating the first point
        X.extend(seg_place[1:])

        # Update 'prev' and its end derivatives for next loop
        prev = seg_place
        _, t_prev_end = _discrete_tangent(prev)
        _, k_prev_end = _discrete_curvature(prev)

    return np.array(X)
