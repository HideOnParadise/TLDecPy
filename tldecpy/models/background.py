"""
Background models for Glow Curve Deconvolution.
"""

import numpy as np


def linear_bg(T: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Linear background: I = a + b*T
    """
    return a + b * T


def exponential_bg(T, a, b, c):
    """
    Exponential background: I_bg = a * exp(b/T) + c

    Args:
        T: Temperature array (K)
        a: Amplitude
        b: Exponential rate (K)
        c: Offset

    Returns:
        Background intensity array
    """
    T = np.asarray(T)
    safe_T = np.maximum(T, 1.0)

    # Exponential decay with T
    exponent = b / safe_T
    exponent = np.clip(exponent, -50, 50)

    return a * np.exp(exponent) + c


BG_REGISTRY = {"linear": linear_bg, "exponential": exponential_bg, "exp": exponential_bg}
