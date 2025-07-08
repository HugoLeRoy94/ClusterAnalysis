import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List, Optional, Tuple
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append('../')


def count_transitions(labels: np.ndarray, n_clusters: int, tau: int, nsample: int, TmKp1: int) -> np.ndarray:
    """
    Computes the transition count matrix between clusters.

    This function calculates the number of times a transition occurs from one cluster to another
    over a given time lag (tau). It iterates through multiple trajectories to build the count matrix,
    ensuring that transitions are not counted across separate trajectories.

    Args:
        labels (np.ndarray): A 1D array of cluster labels for all points from all trajectories, concatenated.
        n_clusters (int): The total number of clusters.
        tau (int): The time lag for which transitions are counted.
        nsample (int): The number of trajectories.
        TmKp1 (int): The length of each trajectory (T - K + 1, where T is the total number of time points
                     and K is the window size used in a previous step, if any).

    Returns:
        np.ndarray: A 2D array (n_clusters x n_clusters) representing the raw transition counts,
                    where C[i, j] is the number of transitions from cluster i to cluster j.
    """
    C = np.zeros((n_clusters, n_clusters), dtype=float)
    for n in range(nsample):
        for start in range(TmKp1 - tau):
            i, j = labels[n * TmKp1 + start], labels[n * TmKp1 + start + tau]
            C[i, j] += 1.0
    return C

def stationary_distribution(P: NDArray[np.float_], tol: float = 1e-12, maxiter: int = 10000) -> NDArray[np.float_]:
    """
    Computes the stationary distribution of a Markov chain.

    This function finds the stationary distribution vector π for a given transition matrix P,
    such that πᵀ * P = πᵀ. It uses the power iteration method on the transpose of P.
    If the power iteration does not converge within the specified number of iterations,
    it falls back to computing the eigenvector corresponding to the eigenvalue 1 of P.T.

    Args:
        P (NDArray[np.float_]): The transition probability matrix (row-stochastic).
        tol (float): The tolerance for checking convergence of the power iteration.
        maxiter (int): The maximum number of iterations for the power iteration method.

    Returns:
        NDArray[np.float_]: A 1D array representing the stationary distribution.
    """
    n = P.shape[0]
    pi = np.ones(n) / n
    i = 0
    while True:
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, 1) < tol:
            break
        pi = pi_new
        i += 1
        if i >= maxiter:
            val, vec = np.linalg.eig(P.T)
            vec = vec[:, np.argsort(val)]
            return np.real(vec[:, -1] / np.sum(vec[:, -1]))

    return pi

def entropy_rate(P: np.ndarray, pi: Optional[np.ndarray] = None) -> float:
    """
    Calculates the Shannon entropy rate of a Markov chain.

    The entropy rate is a measure of the uncertainty or randomness of a stochastic process.
    It is computed as h = -∑_i π_i ∑_j P_ij log(P_ij), where π is the stationary distribution
    and P is the transition matrix. The result is given in nats.

    Args:
        P (np.ndarray): The row-stochastic transition matrix.
        pi (Optional[np.ndarray]): The stationary distribution. If None, it is computed internally.

    Returns:
        float: The Shannon entropy rate in nats per step.
    """
    if pi is None:
        pi = stationary_distribution(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.log(P)
        logP[np.isneginf(logP)] = 0.0  # Define 0 * log(0) = 0
        h = -(pi[:, None] * P * logP).sum()
    return float(h)

def generate_random_points_in_sphere(N: int, d: int) -> np.ndarray:
    """
    Generates N uniformly distributed random points within a d-dimensional unit sphere.

    This method ensures a uniform distribution by first generating points on the surface
    of a hypersphere and then scaling them by a random radius drawn from a distribution
    that compensates for the change in volume with radius in higher dimensions.

    Args:
        N (int): The number of points to generate.
        d (int): The dimension of the space.

    Returns:
        np.ndarray: An array of shape (N, d) containing the generated points.
    """
    points = np.random.randn(N, d)
    norm = np.linalg.norm(points, axis=1)[:, np.newaxis]
    points = points / (norm + 1e-10)
    radii = np.random.rand(N, 1)**(1 / d)
    points_in_sphere = points * radii
    return points_in_sphere

def generate_clustered_points(Nsample: int, trajectory_length: int, d: int, N_centers: int = 10, Radius: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates trajectories of points clustered around several centers.

    This function creates a dataset of trajectories where each trajectory's points are
    generated from a distribution around a specific cluster center. The centers themselves
    are randomly distributed within a sphere. To ensure connectivity between clusters,
    one trajectory is generated by hopping between different cluster centers.

    Args:
        Nsample (int): The number of trajectories to generate.
        trajectory_length (int): The number of points in each trajectory.
        d (int): The dimension of the space.
        N_centers (int): The number of cluster centers.
        Radius (int): The radius of the sphere containing the cluster centers.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - points (np.ndarray): The generated points, with shape (Nsample, trajectory_length, d).
            - centers (np.ndarray): The coordinates of the cluster centers.
    """
    points = np.zeros((Nsample, trajectory_length, d), dtype=float)
    centers = generate_random_points_in_sphere(N_centers, d) * Radius

    for n in range(Nsample - 1):
        center_id = np.random.randint(0, N_centers)
        points[n] = generate_random_points_in_sphere(trajectory_length, d) + centers[center_id]

    center_id = np.random.randint(0, N_centers, trajectory_length)
    points[-1] = centers[center_id] + generate_random_points_in_sphere(trajectory_length, d)
    return points, centers

def compute_entropy(points: np.ndarray, n_clusterss: List[int], Nsample: int, trajectory_length: int, d: int) -> List[float]:
    """
    Computes the entropy rate for different numbers of clusters.

    This function takes a set of trajectories, performs K-Means clustering for each number
    of clusters specified in `n_clusterss`, and then computes the entropy rate for the
    resulting Markov chain. This is useful for analyzing how the entropy of the system
    changes with the granularity of the state space partitioning.

    Args:
        points (np.ndarray): The trajectory data.
        n_clusterss (List[int]): A list of numbers of clusters to use for K-Means.
        Nsample (int): The number of trajectories.
        trajectory_length (int): The length of each trajectory.
        d (int): The dimension of the space.

    Returns:
        List[float]: A list of entropy rates corresponding to each number of clusters.
    """
    h = []
    for n_clusters in n_clusterss:
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        labels = km.fit_predict(points.reshape(-1, d))
        h.append(compute_entropy_from_labels(labels, n_clusters, Nsample, trajectory_length, d))
    return h

def compute_entropy_from_labels(labels: np.ndarray, n_clusters: int, Nsample: int, trajectory_length: int, d: int) -> float:
    """
    Computes the entropy rate from pre-computed cluster labels.

    This function calculates the entropy rate of a Markov chain defined by a sequence of
    cluster labels. It first computes the transition matrix and then the entropy rate.

    Args:
        labels (np.ndarray): The array of cluster labels.
        n_clusters (int): The number of clusters.
        Nsample (int): The number of trajectories.
        trajectory_length (int): The length of each trajectory.
        d (int): The dimension of the space.

    Returns:
        float: The computed entropy rate.
    """
    C = count_transitions(labels, n_clusters, nsample=Nsample, tau=1, TmKp1=trajectory_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = C / C.sum(axis=1, keepdims=True)
    P[np.isnan(P)] = 0.0  # Handle rows with zero counts
    pi = stationary_distribution(P)
    return entropy_rate(P, pi)

def compute_entropy_several_times(points: np.ndarray, n_clusters: int, Nsample: int, trajectory_length: int, d: int, N_times: int) -> List[float]:
    """
    Computes the entropy rate multiple times to assess variability.

    This function runs the K-Means clustering and entropy rate calculation multiple times
    to understand the variability of the results due to the random initialization of K-Means.

    Args:
        points (np.ndarray): The trajectory data.
        n_clusters (int): The number of clusters to use.
        Nsample (int): The number of trajectories.
        trajectory_length (int): The length of each trajectory.
        d (int): The dimension of the space.
        N_times (int): The number of times to repeat the calculation.

    Returns:
        List[float]: A list of entropy rates from the repeated computations.
    """
    h = []
    for n in range(N_times):
        km = KMeans(n_clusters=n_clusters, n_init="auto")
        labels = km.fit_predict(points.reshape(-1, d))
        h.append(compute_entropy_from_labels(labels, n_clusters, Nsample, trajectory_length, d))
    return h