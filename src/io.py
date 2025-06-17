# src/copepod/io.py

"""
I/O utilities for Copepod analysis pipeline.
Supports CSV, Parquet, and NumPy array formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def load_dataframe(path: PathLike) -> pd.DataFrame:
    """Load a DataFrame from CSV or Parquet."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def save_dataframe(df: pd.DataFrame, path: PathLike) -> None:
    """Save a DataFrame to CSV or Parquet."""
    path = Path(path)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_array(path: PathLike) -> np.ndarray:
    """Load a NumPy array from .npy or .npz."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        return np.load(path)["arr_0"]
    else:
        raise ValueError(f"Unsupported array format: {path.suffix}")


def save_array(arr: np.ndarray, path: PathLike) -> None:
    """Save a NumPy array to .npy."""
    path = Path(path)
    if path.suffix != ".npy":
        raise ValueError("Only .npy format is supported for saving arrays.")
    np.save(path, arr)
