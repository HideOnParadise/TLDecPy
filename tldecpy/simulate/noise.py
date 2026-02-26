"""
Noise generation utilities for simulation.
"""

from __future__ import annotations

import numpy as np


def add_noise(
    y: np.ndarray,
    mode: str = "gaussian",
    sigma: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Inject noise into the signal.

    Args:
        y: Clean signal.
        mode: 'gaussian' (constant sigma) or 'poisson' (heteroscedastic).
        sigma: Std dev for gaussian noise.
    """
    rng = np.random.default_rng(seed)

    if mode == "gaussian":
        noise = rng.normal(0, sigma, size=y.shape)
        return y + noise

    if mode == "poisson":
        # Poisson is integer based, but we approximate for floats
        # Variance = y. Std = sqrt(y).
        # We use gaussian approximation for large N or poisson for small
        y_noisy = rng.poisson(np.maximum(y, 0))
        return y_noisy.astype(float)

    return y
