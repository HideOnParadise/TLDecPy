"""First-order kinetic models for thermoluminescence glow curves."""

from __future__ import annotations

from typing import Callable
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tldecpy.utils.aaa_fo import Q_aaa
from tldecpy.utils.constants import KB_EV

FloatArray = NDArray[np.float64]
QFn = Callable[[FloatArray], FloatArray]


def alpha_poly(x: ArrayLike) -> FloatArray:
    r"""
    Evaluate Bos rational approximation of :math:`x\,Q(x)`.

    Parameters
    ----------
    x : ArrayLike
        Dimensionless argument :math:`x = E / (kT)`.

    Returns
    -------
    numpy.ndarray
        Rational approximation values of :math:`x\,Q(x)`.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    num = np.polyval([1.0, 8.5733287401, 18.059016973, 8.6347608925, 0.267773734], x_arr)
    den = np.polyval([1.0, 9.5733223454, 25.6329561486, 21.0996530827, 3.9584969228], x_arr)
    return np.asarray(num / den, dtype=np.float64)


def _fo_core(
    T: ArrayLike,
    Im: float,
    E: float,
    Tm: float,
    Q_fn: QFn,
) -> FloatArray:
    r"""
    Compute first-order TL intensity using a configurable :math:`Q(x)` backend.

    Parameters
    ----------
    T : ArrayLike
        Temperature values in kelvin.
    Im : float
        Peak intensity at :math:`T_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature in kelvin.
    Q_fn : Callable[[numpy.ndarray], numpy.ndarray]
        Auxiliary backend for :math:`Q(x)=e^xE_2(x)`.

    Returns
    -------
    numpy.ndarray
        Simulated glow-curve intensity values.
    """
    t_arr = np.maximum(np.asarray(T, dtype=np.float64), 1e-5)
    x_m = E / (KB_EV * Tm)
    x = np.clip(E / (KB_EV * t_arr), 1e-3, 1e5)

    Q_m = float(Q_fn(np.asarray([x_m], dtype=np.float64))[0])
    Q_x = np.asarray(Q_fn(x), dtype=np.float64)

    a_term = x_m - x
    b_term = x_m * Q_m
    exp_a = np.exp(np.clip(a_term, -100.0, 100.0))

    c_term = (x_m**2 / x) * Q_x * exp_a

    exponent = a_term + b_term - c_term
    return np.asarray(Im * np.exp(np.clip(exponent, -100.0, 100.0)), dtype=np.float64)


def fo_rq(T: ArrayLike, Im: float, E: float, Tm: float) -> FloatArray:
    r"""
    Evaluate first-order model ``fo_rq`` using Bos polynomial-rational :math:`xQ(x)`.

    Parameters
    ----------
    T : ArrayLike
        Temperature values in kelvin.
    Im : float
        Peak intensity at :math:`T_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.

    References
    ----------
    .. [1] Bos, A. J. J., et al. (1993). Intercomparison synthetic glow curves (GLOCANIN).
    .. [2] Kitis, G., Gomes-Ros, J. M., and Tuyn, J. W. N. (1998).
           Thermoluminescence glow-curve deconvolution functions for first,
           second and general orders of kinetics.
    """
    return _fo_core(
        T=T,
        Im=Im,
        E=E,
        Tm=Tm,
        Q_fn=lambda x: np.asarray(1.0 - alpha_poly(x), dtype=np.float64),
    )


def fo_rb(T: ArrayLike, Im: float, E: float, Tm: float) -> FloatArray:
    r"""
    Evaluate first-order model ``fo_rb`` using AAA approximation for :math:`Q(x)`.

    Parameters
    ----------
    T : ArrayLike
        Temperature values in kelvin.
    Im : float
        Peak intensity at :math:`T_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.
    """
    try:
        return _fo_core(T=T, Im=Im, E=E, Tm=Tm, Q_fn=Q_aaa)
    except (FileNotFoundError, ModuleNotFoundError):
        warnings.warn(
            "AAA constants not found for fo_rb; falling back to (1 - alpha_poly) backend.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fo_core(
            T=T,
            Im=Im,
            E=E,
            Tm=Tm,
            Q_fn=lambda x: np.asarray(1.0 - alpha_poly(x), dtype=np.float64),
        )


def fo_ka(T: ArrayLike, Im: float, E: float, Tm: float) -> FloatArray:
    r"""
    Evaluate first-order variant ``fo_ka`` (approximate closed form).

    Parameters
    ----------
    T : ArrayLike
        Temperature values in kelvin.
    Im : float
        Peak intensity at :math:`T_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.
    """
    t_arr = np.maximum(np.asarray(T, dtype=np.float64), 1e-5)

    delta = (2.0 * KB_EV * t_arr) / E
    delta_m = (2.0 * KB_EV * Tm) / E
    arg = (E / (KB_EV * t_arr)) * ((t_arr - Tm) / Tm)
    exp_arg = np.exp(np.clip(arg, -50.0, 50.0))
    term = (t_arr**2 / Tm**2) * exp_arg * (1.0 - delta)
    exponent = 1.0 + arg - term - delta_m
    return np.asarray(Im * np.exp(np.clip(exponent, -100.0, 100.0)), dtype=np.float64)


def fo_wp(T: ArrayLike, Im: float, E: float, Tm: float) -> FloatArray:
    r"""
    Evaluate first-order variant ``fo_wp`` (Weibull approximation).

    Parameters
    ----------
    T : ArrayLike
        Temperature values in kelvin.
    Im : float
        Peak intensity at :math:`T_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.
    """
    t_arr = np.asarray(T, dtype=np.float64)
    omega_approx = (2.52 * KB_EV * Tm**2) / (E + 2.0 * KB_EV * Tm)
    b_weibull = 6.44 * omega_approx
    x_val = (t_arr - Tm) / b_weibull + 0.996

    mask = x_val > 0.0
    y = np.zeros_like(t_arr, dtype=np.float64)
    x_safe = x_val[mask]
    term1 = x_safe**15
    term2 = np.exp(np.clip(-(x_safe**16), -100.0, 100.0))
    y[mask] = 2.713 * Im * term1 * term2
    return y
