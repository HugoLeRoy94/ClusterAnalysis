import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List, Optional, Tuple
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append('../')


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
            vec = vec[:,np.argsort(val)]
            #print("return the last eigen_vector")
            #print(vec[:,-1])
            #print(np.sum(vec[:,-1]))
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

def generate_random_points_in_sphere(N, d):
    """
    Generate N random points in a d-dimensional sphere of radius 1.

    Args:
        N (int): Number of points to generate.
        d (int): Dimension of the space.

    Returns:
        np.ndarray: An array of shape (N, d) containing the points.
    """
    # Generate points from a Gaussian distribution.
    points = np.random.randn(N, d)

    # Normalize each vector to project it onto the surface of a d-sphere.
    # A small epsilon is added to the norm to avoid division by zero, though it's highly unlikely.
    norm = np.linalg.norm(points, axis=1)[:, np.newaxis]
    points = points / (norm + 1e-10)

    # Generate random radii to ensure uniform distribution within the sphere.
    # This is done by taking the d-th root of a uniform random number.
    radii = np.random.rand(N, 1)**(1/d)

    # Scale the points by these radii.
    points_in_sphere = points * radii

    return points_in_sphere

def generate_clustered_points(Nsample,trajectory_length,d,N_centers=10,Radius=4):
    """
    generate points uniformly distributed in N_centers, all the centers are also uniformely distributed in a sphere of radius Radius. The points are generated as series of trajectory_length, which is then used to compute transitions from one point to another.
    In general each trajectory is within one cluster, but to avoid a pathological behavior where there is no transition between clusters at all, we add a trajectory where the transition is only between clusters
    """

    points = np.zeros((Nsample,trajectory_length,d),dtype=float)

    centers = generate_random_points_in_sphere(N_centers, d) * Radius

    for n in range(Nsample-1):
        center_id = np.random.randint(0,N_centers)
        points[n] = generate_random_points_in_sphere(trajectory_length,d) + centers[center_id]

    center_id = np.random.randint(0,N_centers,trajectory_length)
    points[-1] = centers[center_id] + generate_random_points_in_sphere(trajectory_length,d)
    return points,centers
def compute_entropy(points,n_clusterss,Nsample,trajectory_length,d):
    h = list()
    #n_clusterss = [2,3,4,5,10,15,20,30,50,100,200,300,400,500]
    for n_clusters in n_clusterss:
        #print("number of clusters used for kmeans"+str(n_clusters))
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        labels = km.fit_predict(points.reshape(-1,d))
        C = count_transitions(labels, n_clusters,nsample=Nsample, tau = 1,TmKp1 = trajectory_length)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = C / C.sum(axis=1, keepdims=True)
        P[np.isnan(P)] = 0.0  # rows with zero counts
        pi = stationary_distribution(P)
        h.append(entropy_rate(P,pi))
    return h
