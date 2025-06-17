# src/copepod/viz.py

"""
Visualization utilities for embedding, clustering, and Markov analysis.
Uses matplotlib and seaborn.
"""

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm  # For colormap
from matplotlib.colors import Normalize, LogNorm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Sequence

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"


def plot_embedding_2d(embedding: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "", ax=None):
    """
    Scatter plot of a 2D embedding.

    Parameters
    ----------
    embedding : (N, 2) ndarray
        2D embedding coordinates.
    labels : (N,) optional
        Cluster labels or other categorical annotations.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if labels is not None:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def plot_transition_matrix(P: np.ndarray, title: str = "Transition matrix", ax=None):
    """
    Heatmap of a row-stochastic transition matrix.

    Parameters
    ----------
    P : (n, n) ndarray
        Transition matrix.
    """
    if ax is None:
        fig, ax = plt.subplots()
    sns.heatmap(P, ax=ax, cmap="viridis", square=True, cbar=True)
    ax.set_title(title)
    ax.set_xlabel("To")
    ax.set_ylabel("From")


def plot_distribution(pi: np.ndarray, title: str = "distribution", ax=None):
    """
    Bar plot of the stationary distribution π.

    Parameters
    ----------
    pi : (n,) ndarray
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(np.arange(len(pi)), pi)#, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")


def plot_implied_timescales(tscales: np.ndarray, title: str = "Implied timescales", ax=None):
    """
    Plot implied timescales (in log scale if needed).

    Parameters
    ----------
    tscales : (n−1,) ndarray
        Relaxation timescales from Markov model.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(tscales) + 1), tscales, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("τ")
    ax.grid(True)

def plot_trajectories(
    df: pd.DataFrame, 
    xcol="x", ycol="y", zcol=None, 
    groupby="ID", ax=None, sample: Optional[int] = None, 
    t_min = 0,t_max: int = -1,
    fig_kwargs=None, ax_kwargs=None
):
    """
    Plot 2D or 3D trajectories from a DataFrame.

    Parameters
    ----------
    df : DataFrame with trajectory data
    xcol, ycol : str
        Columns for x and y coordinates.
    zcol : str or None
        Column for z coordinate (for 3D plots).
    groupby : str
        Column name grouping trajectories.
    sample : int or None
        Max number of trajectories to plot.
    fig_kwargs : dict
        Keyword arguments passed to plt.subplots().
    ax_kwargs : dict
        Keyword arguments applied to the axis object.
    """
    fig_kwargs = fig_kwargs or {}
    ax_kwargs = ax_kwargs or {}

    if ax is None:
        if zcol is None:
            fig, ax = plt.subplots(**fig_kwargs)
        else:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), **fig_kwargs)

    groups = df.groupby(groupby)
    if sample:
        groups = list(groups)[:sample]

    for _, g in groups:
        if zcol is None:
            ax.plot(g[xcol][t_min:t_max], g[ycol][t_min:t_max], lw=1, alpha=0.8,**ax_kwargs)
        else:
            ax.plot(g[xcol][t_min:t_max], g[ycol][t_min:t_max], g[zcol][t_min:t_max], lw=1, alpha=0.8,**ax_kwargs)
            ax.set_zlabel(zcol)

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Trajectories")


    return ax

    

def plot_colored_trajectory(
    ax,
    track,
    x_col="POSITION_X",
    y_col="POSITION_Y",
    z_col=None,
    color_col="angular_speed",
    sort_col="FRAME",
    cmap="viridis",
    vmin=None,
    vmax=None,
    logscale=False,
    add_colorbar=False,
    linewidth=1,
):
    """
    Plot a 2D or 3D trajectory with color varying along the path.

    Parameters
    ----------
    ax : matplotlib Axes or Axes3D
        Target axis to plot on.
    track : pd.DataFrame
        DataFrame with trajectory data.
    x_col, y_col, z_col : str
        Coordinate columns. If z_col is None, plots in 2D.
    color_col : str
        Column for coloring the trajectory.
    sort_col : str
        Column to sort the trajectory (e.g. time or frame).
    cmap : str
        Colormap to use.
    vmin, vmax : float or None
        Bounds for colormap normalization.
    add_colorbar : bool
        Whether to add colorbar.
    linewidth : float
        Line width for trajectory.

    Returns
    -------
    LineCollection or Line3DCollection
    """
    track = track.sort_values(sort_col)
    x = track[x_col].values
    y = track[y_col].values
    c = track[color_col].values

    if z_col is None:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)        
        norm_cls = LogNorm if logscale else Normalize
        norm = norm_cls(vmin if vmin is not None else np.min(c),
                    vmax if vmax is not None else np.max(c))

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
        lc.set_array(c[:-1])
        ax.add_collection(lc)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
    else:
        z = track[z_col].values
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm_cls = LogNorm if logscale else Normalize
        norm = norm_cls(vmin if vmin is not None else np.min(c),
                    vmax if vmax is not None else np.max(c))

        lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
        lc.set_array(c[:-1])
        ax.add_collection(lc)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_zlim(np.min(z), np.max(z))

    if add_colorbar:
        plt.colorbar(lc, ax=ax, label=color_col)

    return lc