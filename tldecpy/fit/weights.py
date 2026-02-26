"""
Weighting utilities for heteroscedastic data.
"""

import numpy as np


def calculate_weights(y_obs: np.ndarray, mode: str = "none") -> np.ndarray:
    """
    Calculate weights vector W such that residual = W * (y - y_hat).

    For Poisson statistics (counts), variance ~ y.
    Standardizing residuals requires dividing by std_dev = sqrt(y).
    So weight = 1/sqrt(y).
    """
    if mode == "none":
        return np.ones_like(y_obs)

    if mode == "poisson":
        # Avoid division by zero
        # Assume min count is 1 for weighting purposes
        y_safe = np.maximum(y_obs, 1.0)
        return 1.0 / np.sqrt(y_safe)

    return np.ones_like(y_obs)
