"""
Robust loss functions and utilities.
Implements Tukey's Biweight for scipy.optimize.least_squares.
"""

import numpy as np


def tukey_loss(z: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Tukey's Biweight loss function implementation for scipy.
    z = r^2 (squared residuals).
    Scipy expects returns: (rho, rho_prime, rho_double_prime)

    Rho(r) formula:
    rho(r) = (c^2 / 6) * (1 - (1 - (r/c)^2)^3)  if |r| <= c
           = c^2 / 6                            if |r| > c
    """
    # r = sqrt(z)
    r = np.sqrt(z)

    rho = np.empty_like(z)
    rho_prime = np.empty_like(z)
    rho_dprime = np.empty_like(z)  # Approximation for 2nd derivative usually sufficient

    # Inliers mask
    mask = r <= c

    # Calculate rho (Cost)
    # Constant part for outliers
    const_val = (c**2) / 6.0
    rho[~mask] = const_val

    # Variable part for inliers
    u = r[mask] / c
    term = 1.0 - u**2
    rho[mask] = const_val * (1.0 - term**3)

    # Calculate rho' (First derivative w.r.t z = r^2)
    # d(rho)/dz = d(rho)/dr * dr/dz
    # dr/dz = 1 / (2r)
    # d(rho)/dr = r * (1 - (r/c)^2)^2  (Standard Tukey influence function is psi(r) = r*w(r))
    # Influence psi(r) = r * (1 - (r/c)^2)^2
    # rho'(z) = psi(r) / (2r) = 0.5 * (1 - z/c^2)^2

    rho_prime[~mask] = 0.0
    z_in = z[mask]
    rho_prime[mask] = 0.5 * (1.0 - z_in / (c**2)) ** 2

    # Calculate rho'' (Second derivative w.r.t z)
    # rho'(z) = 0.5 * (1 - z/c^2)^2
    # rho''(z) = 0.5 * 2 * (1 - z/c^2) * (-1/c^2) = - (1 - z/c^2) / c^2
    rho_dprime[~mask] = 0.0
    rho_dprime[mask] = -1.0 / (c**2) * (1.0 - z_in / (c**2))

    # Pack for scipy
    return np.vstack((rho, rho_prime, rho_dprime))


def get_loss_func(robust_opts):
    """Factory to return scipy-compatible loss argument."""
    if robust_opts.loss == "linear":
        return "linear"

    if robust_opts.loss == "tukey":
        # Create a closure with specific 'c'
        c_val = robust_opts.loss_param if robust_opts.loss_param else 4.685
        return lambda z: tukey_loss(z, c=c_val)

    # Standard Scipy robust losses
    return robust_opts.loss
