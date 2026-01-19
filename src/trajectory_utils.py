import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Optional
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R


def count_transitions(labels: List[NDArray[np.int_]], n_clusters: int, tau: int) -> np.ndarray:
    """
    Return the raw transition count matrix C (without normalisation).
    Notice that we do not concatenate the pieces of trajectories one after the other
    to avoid unrealistic transition
    
    nsample: the number of trajectories
    TmKp1 : T - K +1
    """
    C = np.zeros((n_clusters, n_clusters), dtype=float)
    for trajectory in labels:
        for t in range(trajectory.__len__() - tau):
            i,j = trajectory[t],trajectory[t+tau]
            C[i,j] += 1
    return C

def stationary_distribution(P: NDArray[np.float64], tol: float = 1e-12, maxiter: int = 10000) -> NDArray[np.float64]:
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
def metastability(P: NDArray[np.float64], pi: NDArray[np.float64], S: NDArray[np.int_]) -> float:
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

import itertools

def canonicalize_trajectory(coords, *, return_rotation=False, tol=1e-12):
    """
    Rotate `coords` (K×D) into a unique canonical frame.
    Works for 2D (D=2) and 3D (D=3).
    
    Any rigid-body rotation + translation of the same trajectory
    maps to the *identical* output.
    """
    X = np.asarray(coords, dtype=float)
    K, D = X.shape                     # Detect dimensionality (2 or 3)
    
    # 1. Centre at centroid
    C = X - X.mean(axis=0)
    
    # 2. PCA -> eigenvectors V
    # Vt is (D, D), V is (D, D)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    V = Vt.T

    Y = C @ V
    
    # 3. Sign disambiguation (loop over D dimensions)
    for j in range(D):
        m3 = (Y[:, j] ** 3).sum()
        if abs(m3) < tol:
            k = (j + 1) % D            # Wrap around (0->1, 1->2 or 1->0)
            m3 = (Y[:, j]**2 * Y[:, k]).sum()
        if m3 < 0:
            V[:, j] *= -1
            Y[:, j] *= -1

    # 4. Make basis right-handed (det = +1)
    if np.linalg.det(V) < 0:
        V[:, -1] *= -1                 # Flip last axis (generic for 2D/3D)
        Y[:, -1] *= -1

    canon = Y
    
    # Step 6: Chiral disambiguation via lexicographic minimization
    # Generate mirrors dynamically for D dimensions (4 for 2D, 8 for 3D)
    mirrors = np.array(list(itertools.product([1, -1], repeat=D)))  # shape (2^D, D)

    # (2^D, D) x (K, D) -> (2^D, K, D) via broadcasting/einsum
    # Einstein sum: 'ij,kj->kij' (mirrors[i, j] * canon[k, j])
    mirrored = np.einsum('ij,kj->kij', canon, mirrors)

    flat = mirrored.reshape(len(mirrors), -1)  # shape (2^D, K*D)
    best_index = np.lexsort(flat.T)[0]
    canon = mirrored[best_index]

    return (canon, V) if return_rotation else canon


#def _norm(v, eps=1e-12):
#    n = np.linalg.norm(v)
#    return v / n if n > eps else v*0.0
#
#def _rot_between(a, b, eps=1e-12):
#    """
#    Minimal rotation that maps unit vector a -> unit vector b.
#    Returns a 3x3 rotation matrix.
#    """
#    a = _norm(a)
#    b = _norm(b)
#    c = np.dot(a, b)
#    if c > 1 - eps:            # nearly identical
#        return np.eye(3)
#    if c < -1 + eps:           # opposite; pick any axis ⟂ a
#        # Choose axis as any unit vector orthogonal to 'a'
#        tmp = np.array([1.0, 0.0, 0.0])
#        if abs(np.dot(a, tmp)) > 0.9:
#            tmp = np.array([0.0, 1.0, 0.0])
#        axis = _norm(np.cross(a, tmp))
#        return _rot_axis_angle(axis, np.pi)
#    axis = _norm(np.cross(a, b))
#    angle = np.arccos(np.clip(c, -1.0, 1.0))
#    return _rot_axis_angle(axis, angle)
#
#def _rot_axis_angle(axis, angle):
#    """
#    Rodrigues' rotation formula. Axis must be unit-length (or zero for identity).
#    """
#    axis = _norm(axis)
#    x, y, z = axis
#    c = np.cos(angle)
#    s = np.sin(angle)
#    C = 1 - c
#    return np.array([
#        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
#        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
#        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ],
#    ])
#
#def _project_to_plane(v, n):
#    n = _norm(n)
#    return v - np.dot(v, n) * n
#
#def _signed_angle_in_plane(a, b, plane_normal, eps=1e-12):
#    """
#    Signed angle from a to b within the plane orthogonal to plane_normal.
#    """
#    a_p = _project_to_plane(a, plane_normal)
#    b_p = _project_to_plane(b, plane_normal)
#    na = np.linalg.norm(a_p); nb = np.linalg.norm(b_p)
#    if na < eps or nb < eps:
#        return 0.0
#    a_u = a_p / na
#    b_u = b_p / nb
#    cosang = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
#    ang = np.arccos(cosang)
#    # Sign via right-hand rule around plane_normal
#    sign = np.sign(np.dot(plane_normal, np.cross(a_u, b_u)))
#    return ang * sign
#
#def _discrete_tangent(piece):
#    # piece: (K,3)
#    # use last two for end, first two for start
#    t_start = _norm(piece[1] - piece[0])
#    t_end   = _norm(piece[-1] - piece[-2])
#    return t_start, t_end
#
#def _discrete_curvature(piece):
#    # discrete 2nd-difference at start and end
#    K_start = piece[2] - 2*piece[1] + piece[0]
#    K_end   = piece[-1] - 2*piece[-2] + piece[-3]
#    return K_start, K_end
#
#def reconstruct_trajectory_from_canonical_pieces(
#    pieces, K, d=3, enforce_curvature=True, eps=1e-10
#):
#    """
#    Stitch canonicalized windows into a single 3D trajectory.
#
#    Parameters
#    ----------
#    pieces : array, shape (M, K*d)
#        Consecutive canonicalized segments (flattened row-major).
#        Each segment is centered/rotated canonically.
#    K : int
#        Window length used to build each piece.
#    d : int
#        Dimensionality (must be 3 here).
#    enforce_curvature : bool
#        If True, add a twist R_psi to match curvature direction (C2).
#        If False, only enforce tangent continuity (C1).
#    eps : float
#        Numerical tolerance.
#
#    Returns
#    -------
#    X : array, shape (K + (M-1)*(K-1), 3)
#        Reconstructed global trajectory.
#    """
#    assert d == 3, "This reconstruction targets 3D."
#    M = pieces.shape[0]
#    assert pieces.shape[1] == K*d
#
#    # reshape all pieces to (K,3)
#    segs = pieces.reshape(M, K, d).copy()
#
#    # Initialize global coordinates with the first piece as-is
#    X = [segs[0][i].copy() for i in range(K)]
#    # Keep track of the current last-segment (already placed)
#    prev = np.stack(X[-K:], axis=0)  # last placed segment (K,3)
#
#    # Precompute end tangent/curvature of the placed segment
#    t_prev_start, t_prev_end = _discrete_tangent(prev)
#    k_prev_start, k_prev_end = _discrete_curvature(prev)
#
#    for m in range(1, M):
#        seg = segs[m]  # (K,3), canonical
#
#        # Compute start tangent/curvature of the new piece (in its own frame)
#        t_next_start, _ = _discrete_tangent(seg)
#        k_next_start, _ = _discrete_curvature(seg)
#
#        # (1) Align tangents: R_theta maps t_next_start -> t_prev_end
#        R_theta = _rot_between(t_next_start, t_prev_end)
#
#        seg_rot = seg @ R_theta.T
#        k_next_start_rot = k_next_start @ R_theta.T
#
#        # (2) Optional twist around the (now aligned) tangent to match curvature
#        if enforce_curvature:
#            # Project curvature vectors to plane ⟂ t_prev_end
#            # If curvature is degenerate, skip twist.
#            a = _project_to_plane(k_prev_end, t_prev_end)
#            b = _project_to_plane(k_next_start_rot, t_prev_end)
#            if np.linalg.norm(a) > eps and np.linalg.norm(b) > eps:
#                psi = _signed_angle_in_plane(b, a, t_prev_end)
#                R_psi = _rot_axis_angle(t_prev_end, psi)
#                seg_rot = seg_rot @ R_psi.T
#            # else: psi = 0 (no reliable curvature direction)
#
#        # (3) Translate so seg_rot[0] lands on prev[-1]
#        delta = prev[-1] - seg_rot[0]
#        seg_place = seg_rot + delta
#
#        # Append without duplicating the first point
#        X.extend(seg_place[1:])
#
#        # Update 'prev' and its end derivatives for next loop
#        prev = seg_place
#        _, t_prev_end = _discrete_tangent(prev)
#        _, k_prev_end = _discrete_curvature(prev)
#
#    return np.array(X)

import numpy as np

def _norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else v*0.0

def _rot_axis_angle(axis, angle):
    axis = _norm(axis)
    x, y, z = axis
    c = np.cos(angle); s = np.sin(angle); C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ],
    ])

def _rot_between(a, b, eps=1e-12):
    a = _norm(a); b = _norm(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1 - eps:  return np.eye(3)
    if c < -1 + eps:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, tmp)) > 0.9: tmp = np.array([0.0, 1.0, 0.0])
        axis = _norm(np.cross(a, tmp))
        return _rot_axis_angle(axis, np.pi)
    axis = _norm(np.cross(a, b))
    ang  = np.arccos(c)
    return _rot_axis_angle(axis, ang)

def _project_to_plane(v, n):
    n = _norm(n); return v - np.dot(v, n)*n

def _discrete_tangent(piece):
    t0 = _norm(piece[1] - piece[0])
    t1 = _norm(piece[-1] - piece[-2])
    return t0, t1

def _discrete_curvature_vecs(piece):
    # second differences as curvature proxies along the piece
    # returns array of shape (K-2, 3)
    return piece[2:] - 2*piece[1:-1] + piece[:-2]

def _frenet_frames(piece, eps=1e-12):
    """Return arrays T,N,B for interior points (indices 1..K-2)."""
    K = piece.shape[0]
    T = np.zeros((K-2, 3)); N = np.zeros((K-2, 3)); B = np.zeros((K-2, 3))
    for i in range(1, K-1):
        t = _norm(piece[i+1] - piece[i-1])
        k = piece[i+1] - 2*piece[i] + piece[i-1]           # curvature proxy
        n = _project_to_plane(k, t)
        n = _norm(n)
        b = _norm(np.cross(t, n))
        T[i-1], N[i-1], B[i-1] = t, n, b
    return T, N, B

def reconstruct_trajectory_from_canonical_pieces(
    pieces, K, d=3, enforce_curvature=True, enforce_torsion=True, Ltwist=3, eps=1e-10
):
    """
    Stitch canonicalized windows into a single 3D trajectory with C1 + (approx) C2/C3 frame continuity.
    - enforce_curvature: roll to align normal at the seam using a curvature proxy
    - enforce_torsion:   estimate roll ψ by averaging normal/binormal mismatch over Ltwist points
    """
    assert d == 3
    M = pieces.shape[0]
    assert pieces.shape[1] == K*d
    segs = pieces.reshape(M, K, d).copy()

    # init with first piece
    X = [segs[0][i].copy() for i in range(K)]
    prev = np.stack(X[-K:], axis=0)

    # frames on prev
    T_prev, N_prev, B_prev = _frenet_frames(prev)
    t_prev_end  = T_prev[-1] if len(T_prev) else _norm(prev[-1]-prev[-2])
    n_prev_end  = N_prev[-1] if len(N_prev) else _norm(_project_to_plane(prev[-1]-2*prev[-2]+prev[-3], t_prev_end))
    b_prev_end  = _norm(np.cross(t_prev_end, n_prev_end))

    for m in range(1, M):
        seg = segs[m]

        # frames at start of seg (its own local frame)
        T_next, N_next, B_next = _frenet_frames(seg)
        t_next_start = T_next[0] if len(T_next) else _norm(seg[1]-seg[0])
        n_next_start = N_next[0] if len(N_next) else _norm(_project_to_plane(seg[2]-2*seg[1]+seg[0], t_next_start))

        # (1) align tangents
        R_theta = _rot_between(t_next_start, t_prev_end)
        seg_rot = seg @ R_theta.T
        # rotate frames for the first Ltwist interior points
        Tn = (T_next @ R_theta.T) if len(T_next) else np.empty((0,3))
        Nn = (N_next @ R_theta.T) if len(N_next) else np.empty((0,3))
        Bn = (B_next @ R_theta.T) if len(B_next) else np.empty((0,3))

        # (2) roll about t to align normals/binormals near the seam
        psi = 0.0
        if enforce_torsion and len(Nn) and len(N_prev):
            L = int(max(1, min(Ltwist, len(Nn), len(N_prev))))
            # use last L of prev and first L of next
            nA = N_prev[-L:]; nB = Nn[:L]
            # project normals to plane ⟂ t_prev_end
            a = np.array([_norm(_project_to_plane(v, t_prev_end)) for v in nA])
            b = np.array([_norm(_project_to_plane(v, t_prev_end)) for v in nB])
            # average signed angle from b -> a around t_prev_end
            cross_sum = sum(np.dot(t_prev_end, np.cross(b[i], a[i])) for i in range(L))
            dot_sum   = float(np.clip(np.sum(np.einsum('ij,ij->i', b, a)), -L, L))
            psi = np.arctan2(cross_sum, dot_sum)  # best roll

        elif enforce_curvature:
            # fallback: single-vector curvature alignment
            k_prev_end = prev[-1] - 2*prev[-2] + prev[-3]
            k_next0    = seg_rot[2] - 2*seg_rot[1] + seg_rot[0]
            a = _project_to_plane(k_prev_end, t_prev_end)
            b = _project_to_plane(k_next0,    t_prev_end)
            if np.linalg.norm(a) > eps and np.linalg.norm(b) > eps:
                a = _norm(a); b = _norm(b)
                cross = np.dot(t_prev_end, np.cross(b, a))
                dot   = float(np.clip(np.dot(b, a), -1.0, 1.0))
                psi = np.arctan2(cross, dot)

        if (enforce_torsion or enforce_curvature) and abs(psi) > 0:
            R_psi = _rot_axis_angle(t_prev_end, psi)
            seg_rot = seg_rot @ R_psi.T

            # optional: ensure binormal handedness continuity at seam
            # (flip by π if binormals disagree in sign)
            # recompute first binormal of rotated seg
            if len(Bn):
                b0 = _norm(np.cross(t_prev_end, (_project_to_plane(Nn[0] @ R_psi.T, t_prev_end))))
                if np.dot(b0, b_prev_end) < 0:
                    seg_rot = seg_rot @ _rot_axis_angle(t_prev_end, np.pi).T

        # (3) translate to connect endpoints
        delta = prev[-1] - seg_rot[0]
        seg_place = seg_rot + delta

        # append
        X.extend(seg_place[1:])
        prev = seg_place

        # update end frames
        T_prev, N_prev, B_prev = _frenet_frames(prev)
        t_prev_end  = T_prev[-1] if len(T_prev) else _norm(prev[-1]-prev[-2])
        n_prev_end  = N_prev[-1] if len(N_prev) else _norm(_project_to_plane(prev[-1]-2*prev[-2]+prev[-3], t_prev_end))
        b_prev_end  = _norm(np.cross(t_prev_end, n_prev_end))

    return np.array(X)


