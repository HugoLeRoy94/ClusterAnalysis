# src/copepod/preprocessing.py

"""
Preprocessing utilities for 2D/3D trajectory data:
- Compute intrinsic kinematics (speed, curvature, torsion)
- Phase classification
- Trajectory filtering and segmentation
- Reconstruction from (|v|, θ, φ)
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing import Union
from scipy.signal import savgol_filter


def compute_speed_turning_angles(X: np.ndarray, dt: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """
    Compute speed, turning angle, and a torsion-like quantity from a 2D or 3D trajectory.

    Parameters
    ----------
    X : (N, D) array_like
        Cartesian positions in 2D or 3D space.
    dt : float
        Time step between consecutive samples.
    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    features : (N-3, 3) ndarray (3D) or (N-2, 3) ndarray (2D)
        Columns are:
        - |v|     : average speed between X[i] and X[i+2]
        - θ       : turning angle between T_i and T_{i+1}
        - torsion : signed angle change (2D) or discrete torsion rate (3D)
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    if D not in (2, 3):
        raise ValueError("Only 2D or 3D input is supported.")

    # Velocities and tangent vectors
    dX = np.diff(X, axis=0)
    v = dX / dt
    v_norm = np.linalg.norm(v, axis=1)
    T = v / np.maximum(v_norm[:, None], eps)

    dot = np.einsum("ij,ij->i", T[:-1], T[1:])
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    if D == 2:
        # Signed 2D turning angle
        cross = T[:-1, 0] * T[1:, 1] - T[:-1, 1] * T[1:, 0]
        signed_theta = np.sign(cross) * theta
        speed = 0.5 * (v_norm[:-1] + v_norm[1:])
        torsion = signed_theta  # convention
        return np.column_stack([speed, theta, torsion])
    
    else:
        # 3D discrete torsion
        n = np.cross(T[:-1], T[1:])
        n_norm = np.linalg.norm(n, axis=1)
        n_unit = n / np.maximum(n_norm[:, None], eps)

        dot_n = np.einsum("ij,ij->i", n_unit[:-1], n_unit[1:])
        dot_n = np.clip(dot_n, -1.0, 1.0)
        angle = np.arccos(dot_n)
        sign = np.sign(np.einsum("ij,ij->i", np.cross(n_unit[:-1], n_unit[1:]), T[1:-1]))
        torsion_rate = sign * angle / dt
        
        speed = 0.5 * (v_norm[:-1] + v_norm[1:])[:-1]
        theta = theta[:-1]
        return np.column_stack([speed, theta, torsion_rate])

def suppress_torsion_when_low_curvature(df, curvature_col="curvature_angle", torsion_col="torsion_angle", threshold=1e-1):
    df = df.copy()
    mask = df[curvature_col].abs() < threshold
    df.loc[mask, torsion_col] = 0.0
    return df

def compute_phases(
    df: pd.DataFrame,
    column_names=("x", "y", "z"),
    dt: float = 1.0,
    groupby: str = "ID",
    groupsort: str = "frame",
) -> pd.DataFrame:
    """
    Compute speed and angular speed from positions per trajectory.

    Adds columns: ['speed', 'curvature_angle', 'torsion_angle'] if 3D.

    Parameters
    ----------
    df : pd.DataFrame
        Input trajectory data.
    column_names : tuple of str
        Position column names.
    dt : float
        Time step between frames.
    """
    phase_data = []
    for _, group in df.groupby(groupby):
        group = group.sort_values(groupsort)
        X = group[list(column_names)].to_numpy()
        phases = compute_speed_turning_angles(X, dt=dt)
        temp = group.iloc[1:-2].copy()
        temp["speed"] = phases[:, 0]
        temp["curvature_angle"] = phases[:, 1]
        if phases.shape[1] == 3:
            temp["torsion_angle"] = phases[:, 2]
            temp["abs_torsion_angle"] = abs(phases[:,2])
            #temp = suppress_torsion_when_low_curvature(
            #    temp, curvature_col="curvature_angle",
            #    torsion_col="torsion_angle", threshold=1e-1
            #)
        phase_data.append(temp)

    return pd.concat(phase_data, ignore_index=True)


def reconstruct_trajectory_from_angles(
    features: np.ndarray,
    dt: float = 1.0,
    initial_pos: Union[None, np.ndarray] = None,
    initial_heading: Union[None, float, np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Reconstruct a trajectory from (|v|, θ, φ).

    Parameters
    ----------
    features : (M, 3) ndarray
        Columns are: speed, turning angle, torsion.
    Returns
    -------
    traj : (M+1, D) ndarray
    """
    f = np.asarray(features, float)
    M = f.shape[0]
    
    D = f.shape[1]
    if D==3:
        v, theta, phi = f.T
    elif D==2:
        v, theta = f.T
    if D == 2:
        pos = np.zeros(2) if initial_pos is None else np.asarray(initial_pos, float)
        θ = 0.0 if initial_heading is None else (
            float(initial_heading) if np.isscalar(initial_heading)
            else np.arctan2(initial_heading[1], initial_heading[0])
        )
        traj = np.empty((M + 1, 2))
        traj[0] = pos
        for i in range(M):
            pos += v[i] * dt * np.array([np.cos(θ), np.sin(θ)])
            θ += theta[i]
            traj[i + 1] = pos
        return traj
    else:
        P = np.zeros(3) if initial_pos is None else np.asarray(initial_pos, float)
        T = np.array([1.0, 0.0, 0.0]) if initial_heading is None else np.asarray(initial_heading, float)
        T /= max(np.linalg.norm(T), eps)

        if abs(T[0]) < 0.9:
            arb = np.array([1.0, 0.0, 0.0])
        else:
            arb = np.array([0.0, 1.0, 0.0])
        N = arb - np.dot(arb, T) * T
        N /= max(np.linalg.norm(N), eps)
        B = np.cross(T, N)

        traj = np.empty((M + 1, 3))
        traj[0] = P
        for i in range(M):
            ds = v[i] * dt
            if abs(phi[i]) > eps:
                T = R.from_rotvec(phi[i] * B).apply(T)
                N = R.from_rotvec(phi[i] * B).apply(N)
            if abs(theta[i]) > eps:
                T = R.from_rotvec(theta[i] * N).apply(T)
                B = R.from_rotvec(theta[i] * N).apply(B)
            P += T * ds
            traj[i + 1] = P
            T /= max(np.linalg.norm(T), eps)
            N -= np.dot(N, T) * T
            N /= max(np.linalg.norm(N), eps)
            B = np.cross(T, N)
        return traj


def split_trajectories(
    df: pd.DataFrame,
    chunk_size: int = 100,
    groupby: str = "ID",
    sort_values: str = "frame"
) -> pd.DataFrame:
    """
    Split trajectories into fixed-length chunks.

    Parameters
    ----------
    df : pd.DataFrame
    chunk_size : int
        Minimum length to retain
    Returns
    -------
    pd.DataFrame with re-indexed IDs
    """
    chunks = []
    new_id = 1
    for _, group in df.groupby(groupby):
        group = group.sort_values(sort_values)
        for i in range(len(group) // chunk_size):
            chunk = group.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
            chunk[groupby] = new_id
            chunks.append(chunk)
            new_id += 1
    return pd.concat(chunks, ignore_index=True)


def filter_trajectories(df: pd.DataFrame, min_length: int = 100, groupby="ID") -> pd.DataFrame:
    """
    Remove short trajectories and reindex IDs.
    """
    lengths = df.groupby(groupby).size()
    valid_ids = lengths[lengths > min_length].index
    df_filtered = df[df[groupby].isin(valid_ids)].copy()
    new_ids = {old: new for new, old in enumerate(valid_ids, start=0)}
    df_filtered[groupby] = df_filtered[groupby].map(new_ids)
    return df_filtered

def smooth_trajectory_savgol(
    df: pd.DataFrame,
    columns=("x", "y", "z"),
    window: int = 7,
    polyorder: int = 3,
    groupby: str = "ID",
) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to smooth trajectory.

    Notes
    -----
    - `window` must be odd and >= polyorder + 2
    """
    df_smoothed = df.copy()
    for col in columns:
        df_smoothed[col] = (
            df.groupby(groupby)[col]
            .transform(lambda x: savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp"))
        )
    return df_smoothed
