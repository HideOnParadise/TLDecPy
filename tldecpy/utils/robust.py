"""
Robust statistical utilities for outlier detection and cleaning.
"""

import numpy as np


def mad(data: np.ndarray, scale: float = 1.4826) -> float:
    """
    Calculate Median Absolute Deviation (MAD).
    Scale factor 1.4826 makes it consistent with sigma for normal distribution.
    """
    median = np.median(data)
    deviation = np.abs(data - median)
    return scale * np.median(deviation)


def robust_zscore(data: np.ndarray) -> np.ndarray:
    """
    Calculate Modified Z-score using MAD.
    Robust against outliers compared to standard std-dev Z-score.
    """
    median = np.median(data)
    mad_val = mad(data)

    if mad_val == 0:
        return np.zeros_like(data)

    return 0.6745 * (data - median) / mad_val


def clean_outliers(y: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Replace outliers with local median (simple winsorization-like).
    Uses gradient (derivative) to detect spikes, not just amplitude.
    """
    # Analyze first derivative to find sudden spikes
    dy = np.gradient(y)
    z_scores = robust_zscore(dy)

    # Mask outliers
    mask = np.abs(z_scores) > threshold

    if not np.any(mask):
        return y

    y_clean = y.copy()
    # Simple fix: replace outlier with median of neighbors (window 3)
    # For a vectorized approach, we interpolate, but iterative replacement is safer for spikes
    idxs = np.where(mask)[0]
    for i in idxs:
        # Bounds check
        start = max(0, i - 2)
        end = min(len(y), i + 3)
        neighbors = y[start:end]
        # Exclude the point itself from median calc roughly
        y_clean[i] = np.median(neighbors)

    return y_clean
