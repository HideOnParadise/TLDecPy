"""Loader for Refglow/GLOCANIN thermoluminescence reference datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REFGLOW_META: Final[dict[str, str]] = {
    "x001": "Synthetic, 1 Peak (FO)",
    "x002": "Synthetic, 4 Peaks (FO)",
    "x003": "TLD-100, 5 Peaks",
    "x004": "TLD-100, 5 Peaks",
    "x005": "TLD-100, 5 Peaks",
    "x006": "TLD-100, 5 Peaks",
    "x007": "TLD-100, 5 Peaks",
    "x008": "TLD-100, 5 Peaks",
    "x009": "TLD-700, 9 Peaks (Complex)",
    "x010": "TLD-100, Low Dose",
}


def get_data_dir() -> Path:
    """Return package directory that stores bundled Refglow CSV files."""
    return Path(__file__).resolve().parent / "refglow_files"


def list_refglow() -> list[str]:
    """
    List available Refglow dataset identifiers.

    Returns
    -------
    list[str]
        Available IDs from ``x001`` to ``x010``.
    """
    return list(REFGLOW_META.keys())


def resolve_refglow_path(key: str) -> Path:
    """Resolve the CSV path for a Refglow key."""
    clean_key = str(key).strip().lower().replace("refglow", "x")
    if clean_key not in REFGLOW_META:
        raise ValueError(f"Unknown Refglow key: {key}")

    filename = f"Refglow{clean_key[1:]}.csv"
    csv_path = get_data_dir() / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset {key} not found at {csv_path}. "
            "Ensure CSV files exist under tldecpy/data/refglow_files/."
        )
    return csv_path


def load_refglow(key: str) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Load one Refglow curve as temperature and intensity arrays.

    Parameters
    ----------
    key : str
        Dataset identifier in the form ``xNNN`` or ``RefglowNNN``
        with ``NNN`` in ``001..010``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Temperature :math:`T` (K) and measured TL intensity :math:`I(T)`.

    Raises
    ------
    ValueError
        If the key is not recognized or the file does not contain valid numeric pairs.
    FileNotFoundError
        If the requested dataset file cannot be located.

    References
    ----------
    .. [1] Bos, A. J. J., et al. (1993, 1994), Refglow/GLOCANIN benchmark data.
    """
    clean_key = str(key).strip().lower().replace("refglow", "x")
    csv_path = resolve_refglow_path(clean_key)
    filename = csv_path.name

    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError(f"Dataset {filename} must contain at least two columns.")

    t_values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    i_values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t_values) & np.isfinite(i_values)

    if int(valid.sum()) < 2:
        raise ValueError(f"Dataset {filename} does not contain enough valid numeric samples.")

    return t_values[valid], i_values[valid]
