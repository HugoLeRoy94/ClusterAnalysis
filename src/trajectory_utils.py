
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform
from typing import List, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from numba import jit,prange
from scipy.spatial.distance import squareform
from src.distance import compute_condensed_distance_matrix

def subsample_dataframe(df: pd.DataFrame, n_subsample: int, id_column: str = 'ID', random_state: int = 42) -> pd.DataFrame:
    """
    Subsamples a DataFrame to contain trajectories of `n_subsample` unique IDs.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. Must contain an `id_column`.
    n_subsample : int
        The number of unique IDs to subsample.
    id_column : str, optional
        The name of the column containing the trajectory IDs, by default 'ID'.
    random_state : int, optional
        The random seed for reproducibility, by default 42.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only the trajectories of the subsampled IDs.
    """
    np.random.seed(random_state)
    unique_ids = df[id_column].unique()
    subsampled_ids = np.random.choice(unique_ids, size=n_subsample, replace=False)
    return df[df[id_column].isin(subsampled_ids)]

def calculate_rolling_mean(df: pd.DataFrame, columns: List[str], window_size: int, id_column: str = 'ID') -> pd.DataFrame:
    """
    Calculates the rolling mean for specified columns, grouped by trajectory ID.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    columns : List[str]
        A list of column names on which to calculate the rolling mean.
    window_size : int
        The size of the rolling window.
    id_column : str, optional
        The name of the column containing the trajectory IDs, by default 'ID'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with rolling mean columns added. The new columns will be named as '{original_column}_rolling_mean'.
    """
    df_copy = df.copy()
    for col in columns:
        df_copy[f'{col}_rolling_mean'] = df_copy.groupby(id_column)[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    return df_copy

def calculate_center_of_mass(df: pd.DataFrame, position_columns: List[str], id_column: str = 'ID') -> pd.DataFrame:
    """
    Calculates the center of mass for each trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    position_columns : List[str]
        A list of column names representing the positions (e.g., ['x', 'y', 'z']).
    id_column : str, optional
        The name of the column containing the trajectory IDs, by default 'ID'.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a trajectory ID and contains the coordinates of its center of mass.
    """
    return df.groupby(id_column)[position_columns].mean()

def align_trajectories_to_z_axis(df: pd.DataFrame, id_column: str = 'ID', position_columns: List[str] = ['x', 'y', 'z']) -> pd.DataFrame:
    """
    Aligns each trajectory to the z-axis based on its principal component.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    id_column : str, optional
        The name of the column containing the trajectory IDs, by default 'ID'.
    position_columns : List[str], optional
        A list of column names representing the positions, by default ['x', 'y', 'z'].

    Returns
    -------
    pd.DataFrame
        A new DataFrame with aligned trajectory coordinates.
    """
    from sklearn.decomposition import PCA

    aligned_dfs = []
    for name, group in df.groupby(id_column):
        pca = PCA(n_components=3)
        pca.fit(group[position_columns])
        
        # The first principal component
        v1 = pca.components_[0]
        
        # Align v1 with the z-axis (0, 0, 1)
        z_axis = np.array([0, 0, 1])
        rotation = R.align_vectors([z_axis], [v1])[0]
        
        # Apply the rotation to the coordinates
        aligned_coords = rotation.apply(group[position_columns])
        
        aligned_df = group.copy()
        aligned_df[position_columns] = aligned_coords
        aligned_dfs.append(aligned_df)
        
    return pd.concat(aligned_dfs)

def compute_frenet_serret_frames(df: pd.DataFrame, id_column: str = 'ID', position_columns: List[str] = ['x', 'y', 'z'], window_size: int = 5, poly_order: int = 2) -> pd.DataFrame:
    """
    Computes the Frenet-Serret frames (tangent, normal, binormal) for each trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    id_column : str, optional
        The name of the column containing the trajectory IDs, by default 'ID'.
    position_columns : List[str], optional
        A list of column names representing the positions, by default ['x', 'y', 'z'].
    window_size : int, optional
        The window size for the Savitzky-Golay filter, by default 5.
    poly_order : int, optional
        The polynomial order for the Savitzky-Golay filter, by default 2.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with columns for tangent, normal, and binormal vectors, as well as curvature and torsion.
    """
    results = []
    for _, group in df.groupby(id_column):
        coords = group[position_columns].values
        
        # Ensure window_size is odd and less than the number of points
        if len(coords) <= window_size:
            continue
        
        # Savitzky-Golay filter to smooth and get derivatives
        r_prime = savgol_filter(coords, window_size, poly_order, deriv=1, axis=0)
        r_double_prime = savgol_filter(coords, window_size, poly_order, deriv=2, axis=0)
        
        # Tangent
        T = r_prime / np.linalg.norm(r_prime, axis=1)[:, np.newaxis]
        
        # Normal
        N = r_double_prime - np.sum(r_double_prime * T, axis=1)[:, np.newaxis] * T
        N /= np.linalg.norm(N, axis=1)[:, np.newaxis]
        
        # Binormal
        B = np.cross(T, N)
        
        # Curvature and Torsion
        # ... (implementation for curvature and torsion)
        
        group_results = group.copy()
        group_results[['Tx', 'Ty', 'Tz']] = T
        group_results[['Nx', 'Ny', 'Nz']] = N
        group_results[['Bx', 'By', 'Bz']] = B
        results.append(group_results)
        
    return pd.concat(results)

def compute_spline_and_derivatives(df: pd.DataFrame, id_column: str = 'ID', position_columns: List[str] = ['x', 'y', 'z'], time_column: str = 'time') -> pd.DataFrame:
    """
    Computes spline-based derivatives for trajectory data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    id_column : str, optional
        The name of the column containing the trajectory IDs, by default 'ID'.
    position_columns : List[str], optional
        A list of column names representing the positions, by default ['x', 'y', 'z'].
    time_column : str, optional
        The name of the column containing time information, by default 'time'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with columns for first and second derivatives.
    """
    results = []
    for _, group in df.groupby(id_column):
        t = group[time_column].values
        coords = group[position_columns].values
        
        cs = CubicSpline(t, coords)
        
        group_results = group.copy()
        group_results[[f'{col}_prime' for col in position_columns]] = cs(t, 1)
        group_results[[f'{col}_double_prime' for col in position_columns]] = cs(t, 2)
        results.append(group_results)
        
    return pd.concat(results)

def compute_curvature_and_torsion(df: pd.DataFrame, id_column: str = 'ID', position_columns: List[str] = ['x', 'y', 'z']) -> pd.DataFrame:
    """
    Computes curvature and torsion for each trajectory. Assumes derivatives are pre-computed.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. Must contain columns for first, second, and third derivatives
        (e.g., 'x_prime', 'x_double_prime', 'x_triple_prime').
    id_column : str, optional
        The name of the column containing the trajectory IDs, by default 'ID'.
    position_columns : List[str], optional
        A list of column names representing the positions, by default ['x', 'y', 'z'].

    Returns
    -------
    pd.DataFrame
        A new DataFrame with 'curvature' and 'torsion' columns.
    """
    results = []
    for _, group in df.groupby(id_column):
        r_prime_cols = [f'{col}_prime' for col in position_columns]
        r_double_prime_cols = [f'{col}_double_prime' for col in position_columns]
        r_triple_prime_cols = [f'{col}_triple_prime' for col in position_columns] # Assuming third derivative is available

        r_prime = group[r_prime_cols].values
        r_double_prime = group[r_double_prime_cols].values
        r_triple_prime = group[r_triple_prime_cols].values

        # Curvature
        cross_product = np.cross(r_prime, r_double_prime)
        curvature = np.linalg.norm(cross_product, axis=1) / (np.linalg.norm(r_prime, axis=1)**3)

        # Torsion
        triple_product = np.sum(cross_product * r_triple_prime, axis=1)
        torsion = triple_product / (np.linalg.norm(cross_product, axis=1)**2)

        group_results = group.copy()
        group_results['curvature'] = curvature
        group_results['torsion'] = torsion
        results.append(group_results)

    return pd.concat(results)

def compute_distance_matrix(points: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Computes the distance matrix for a set of points.

    Parameters
    ----------
    points : np.ndarray
        An array of points, where each row is a point.
    metric : str, optional
        The distance metric to use, by default 'euclidean'.

    Returns
    -------
    np.ndarray
        The square distance matrix.
    """
def count_transitions(labels: np.ndarray, n_clusters: int, tau: int,nsample: int, TmKp1: int) -> np.ndarray:
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
