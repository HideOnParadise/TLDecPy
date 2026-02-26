"""
Numerical stability helpers: clamps, safe division, and scaling.
"""

import numpy as np


def safe_exp(x: np.ndarray, min_val: float = -700, max_val: float = 700) -> np.ndarray:
    """
    Exponential with hard clamping to avoid Overflow/Underflow.
    """
    return np.exp(np.clip(x, min_val, max_val))


def safe_div(n: np.ndarray, d: float, epsilon: float = 1e-10) -> np.ndarray:
    """
    Division protecting against zero denominator.
    """
    if abs(d) < epsilon:
        d = epsilon * np.sign(d) if d != 0 else epsilon
    return n / d


def clamp_R_near_one(R: float, epsilon: float = 1e-4) -> float:
    """
    OTOR diverges at R=1. This forces R away from exactly 1.
    """
    if abs(R - 1.0) < epsilon:
        if R >= 1.0:
            return 1.0 + epsilon
        else:
            return 1.0 - epsilon
    return R
