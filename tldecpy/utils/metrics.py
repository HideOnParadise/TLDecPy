"""
Statistical metrics for model selection.
"""

import numpy as np


def calculate_aic_bic(n: int, k: int, ssr: float):
    """
    Calculate AIC and BIC.
    n: number of data points
    k: number of parameters
    ssr: sum of squared residuals
    """
    if ssr <= 0 or n <= k:
        return np.inf, np.inf

    # Log-likelihood approx for least squares: -n/2 * ln(SSR/n)
    # AIC = 2k - 2ln(L) = 2k + n * ln(SSR/n) + const
    aic = n * np.log(ssr / n) + 2 * k

    # BIC = k * ln(n) - 2ln(L)
    bic = n * np.log(ssr / n) + k * np.log(n)

    return aic, bic
