"""
Wrappers for Savitzky-Golay filtering with validation.
"""

import numpy as np
from scipy.signal import savgol_filter


def safe_savgol(
    y: np.ndarray, window_length: int = 11, polyorder: int = 3, deriv: int = 0
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter with parameter safety checks.
    """
    n = len(y)

    # Window length must be odd and less than signal size
    if window_length >= n:
        window_length = n - 1 if n % 2 == 0 else n

    if window_length % 2 == 0:
        window_length += 1

    # Window must be >= polyorder + 2 (scipy requires window > polyorder)
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    # Fallback if signal is too short for settings
    if n < 5:
        return y

    return savgol_filter(y, window_length, polyorder, deriv=deriv)
