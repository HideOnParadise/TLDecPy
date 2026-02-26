"""Special functions and fast approximations used by TL models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expi, lambertw, wrightomega

FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=float)


def expi_integral(T: ArrayLike, E: float, k: float) -> FloatArray:
    """
    Temperature integral using Ei:
    ``T * exp(-E/kT) + (E/k) * Ei(-E/kT)``.
    """
    temp = np.maximum(_as_float_array(T), np.finfo(float).tiny)
    x = E / (k * temp)
    with np.errstate(all="ignore"):
        ei_val = expi(-x)
    return temp * np.exp(-x) + (E / k) * ei_val


def exp_int_approx(z: ArrayLike) -> FloatArray:
    """
    Rational approximation ``R(z) = exp(z) * E2(z)``.

    Coefficients follow the Abramowitz and Stegun rational form for the
    exponential integral, reused in Benavente et al. (2019), Eq. 28
    (same coefficients as their Eq. 5).
    """
    x = np.clip(_as_float_array(z), 0.0, 1e6)

    a0, a1 = 0.250621, 2.334733
    b0, b1 = 1.681534, 3.330657

    num = x * x + a1 * x + a0
    den = x * x + b1 * x + b0
    return 1.0 - (num / den)


def e2_rational(z: ArrayLike) -> FloatArray:
    """
    Fast approximation of ``E2(z)`` with ``E2(z) ≈ exp(-z) * R(z)``.

    This uses the same rational form cited by Benavente et al. (2019), Eq. 27-28,
    whose primary coefficients come from Abramowitz and Stegun.
    """
    x = np.clip(_as_float_array(z), 0.0, 1e6)
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-x) * exp_int_approx(x)


def lambertw_safe(z: ArrayLike, k: int) -> FloatArray:
    """
    Real-valued wrapper around ``scipy.special.lambertw``.

    Parameters
    ----------
    z : ArrayLike
        Input argument.
    k : int
        Lambert W branch index.

    Returns
    -------
    numpy.ndarray
        Real part of ``W_k(z)`` as a float array.
    """
    return np.real(lambertw(_as_float_array(z), k=k)).astype(float)


def wrightomega_safe(z: ArrayLike) -> FloatArray:
    """
    Real-valued wrapper around ``scipy.special.wrightomega``.

    Parameters
    ----------
    z : ArrayLike
        Input argument.

    Returns
    -------
    numpy.ndarray
        Real part of ``omega(z)`` as a float array.
    """
    return np.real(wrightomega(_as_float_array(z))).astype(float)
