import numpy as np
from numba import njit, prange
from scipy.spatial.distance import squareform

@njit
def gram_matrix_f32(degree):
    G = np.empty((degree + 1, degree + 1), dtype=np.float32)
    for i in range(degree + 1):
        for j in range(degree + 1):
            G[i, j] = 1.0 / (i + j + 1)
    return G

@njit
def build_vandermonde_f32(t, degree):
    K = len(t)
    V = np.empty((K, degree + 1), dtype=np.float32)
    for i in range(K):
        x = t[i]
        V[i, 0] = 1.0
        for j in range(1, degree + 1):
            V[i, j] = V[i, j - 1] * x
    return V

@njit
def fit_poly_f32(y, V):
    lhs = V.T @ V
    rhs = V.T @ y
    return np.linalg.solve(lhs, rhs)

@njit
def index_condensed(i, j, N):
    # i < j required
    return i * N - (i * (i + 1)) // 2 + (j - i - 1)

@njit(parallel=True)
def compute_condensed_distance_matrix(series, degree=5):
    """
    Compute upper-triangular (condensed) canonical distance matrix.

    Parameters:
        series: array of shape (N, K, d)
        degree: polynomial degree

    Returns:
        D: condensed distance matrix, shape (N * (N - 1) // 2,)
    """
    N, K,d = series.shape
    t = np.ascontiguousarray(np.linspace(0.0, 1.0, K).astype(np.float32))
    V = np.ascontiguousarray(build_vandermonde_f32(t, degree))
    G = np.ascontiguousarray(gram_matrix_f32(degree))
    D = np.ascontiguousarray(np.empty(N * (N - 1) // 2, dtype=np.float32))

    # Precompute all coefficients
    coeffs = np.empty((N, d, degree + 1), dtype=np.float32)
    for i in prange(N):
        for j in range(d):
            coeffs[i, j] = fit_poly_f32(series[i, :, j], V)

    # Fill condensed matrix
    for i in prange(N):
        for j in range(i + 1, N):
            dist2 = 0.0
            for k in range(d):
                diff = coeffs[i, k] - coeffs[j, k]
                dist2 += diff @ G @ diff
            D[index_condensed(i, j, N)] = np.sqrt(dist2)
    return D
    
@njit(parallel=True)
def compute_polynomial_coefficients(series, degree):
    """
    Compute polynomial coefficients for each trajectory and dimension.

    Parameters:
        series: array of shape (N, K, d)
        degree: polynomial degree

    Returns:
        coeffs: array of shape (N, d, degree + 1)
        G: Gram matrix of shape (degree + 1, degree + 1)
    """
    N, K, d = series.shape
    t = np.linspace(0.0, 1.0, K).astype(np.float32)
    V = np.ascontiguousarray(build_vandermonde_f32(t, degree))
    G = gram_matrix_f32(degree)
    
    coeffs = np.empty((N, d, degree + 1), dtype=np.float32)
    for i in prange(N):
        for j in range(d):
            y = np.ascontiguousarray(series[i, :, j].astype(np.float32))
            coeffs[i, j] = fit_poly_f32(y, V)

    return coeffs, G

@njit
def index_condensed(i, j, N):
    return i * N - (i * (i + 1)) // 2 + (j - i - 1)

@njit(parallel=True)
def compute_condensed_from_coeffs(coeffs, G):
    """
    Compute canonical distance matrix from polynomial coefficients.

    Parameters:
        coeffs: array of shape (N, d, degree + 1)
        G: Gram matrix of shape (degree + 1, degree + 1)

    Returns:
        D: condensed distance matrix, shape (N * (N - 1) // 2,)
    """
    N, d, deg1 = coeffs.shape
    D = np.empty(N * (N - 1) // 2, dtype=np.float32)

    for i in prange(N):
        for j in range(i + 1, N):
            dist2 = 0.0
            for k in range(d):
                diff = coeffs[i, k] - coeffs[j, k]
                dist2 += diff @ G @ diff
            D[index_condensed(i, j, N)] = np.sqrt(dist2)
    return D
