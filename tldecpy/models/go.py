"""General-order kinetic models for thermoluminescence glow curves."""

from __future__ import annotations

import numpy as np

from tldecpy.utils.constants import KB_EV


def go_kg(T: np.ndarray, Im: float, E: float, Tm: float, b: float) -> np.ndarray:
    r"""
    Evaluate the Kitis-general expression for general-order TL kinetics.

    Parameters
    ----------
    T : numpy.ndarray
        Temperature grid :math:`T` in kelvin.
    Im : float
        Peak intensity :math:`I_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature :math:`T_m` in kelvin.
    b : float
        Kinetic order parameter.

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.

    References
    ----------
    .. [1] Kitis, G., Gomes-Ros, J. M., and Tuyn, J. W. N. (1998).
           Thermoluminescence glow-curve deconvolution functions for first,
           second and general orders of kinetics.
    """
    if abs(b - 1.0) < 1e-3:
        b = 1.001

    temperature = np.maximum(np.asarray(T, dtype=float), 1e-5)
    k = KB_EV

    delta = (2.0 * k * temperature) / E
    delta_m = (2.0 * k * Tm) / E

    arg = (E / (k * temperature)) * ((temperature - Tm) / Tm)
    exp_arg = np.exp(np.clip(arg, -50.0, 50.0))

    prefactor = Im * (b ** (b / (b - 1.0))) * exp_arg
    bracket = (
        (b - 1.0) * (1.0 - delta) * (temperature**2 / Tm**2) * exp_arg
        + 1.0
        + (b - 1.0) * delta_m
    )
    bracket = np.maximum(bracket, 1e-9)
    return prefactor * (bracket ** (-b / (b - 1.0)))


def go_rq(T: np.ndarray, Im: float, E: float, Tm: float, b: float) -> np.ndarray:
    r"""
    Evaluate the rational-quadratic approximation for general-order TL kinetics.

    Parameters
    ----------
    T : numpy.ndarray
        Temperature grid :math:`T` in kelvin.
    Im : float
        Peak intensity :math:`I_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature :math:`T_m` in kelvin.
    b : float
        Kinetic order parameter.

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.

    References
    ----------
    .. [1] Gomez-Ros, J. M., and Kitis, G. (2002).
           Computerised glow-curve deconvolution using general and mixed-order kinetics.
    """
    if abs(b - 1.0) < 1e-3:
        b = 1.001

    temperature = np.asarray(T, dtype=float)
    k = KB_EV

    def F_func(x: np.ndarray) -> np.ndarray:
        num = np.polyval([1.0, 2.334733, 0.250621], x)
        den = np.polyval([1.0, 3.330657, 1.681534], x)
        return 1.0 - (num / den)

    x_t = E / (k * temperature)
    x_tm = E / (k * Tm)
    x_tm_arr = np.asarray([x_tm], dtype=float)

    exp_diff = np.exp(np.clip(x_tm - x_t, -50.0, 50.0))
    term_inside = (temperature / Tm) * exp_diff * F_func(x_t) - F_func(x_tm_arr)[0]
    bracket = 1.0 + ((b - 1.0) / b) * x_tm * term_inside
    bracket = np.maximum(bracket, 1e-9)

    return Im * exp_diff * (bracket ** (-b / (b - 1.0)))


def go_ge(T: np.ndarray, Im: float, E: float, Tm: float, b: float) -> np.ndarray:
    r"""
    Evaluate the exponentialized approximation for general-order TL kinetics.

    Parameters
    ----------
    T : numpy.ndarray
        Temperature grid :math:`T` in kelvin.
    Im : float
        Peak intensity :math:`I_m`.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature :math:`T_m` in kelvin.
    b : float
        Kinetic order parameter.

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.

    References
    ----------
    .. [1] Gomez-Ros, J. M., and Kitis, G. (2002).
           Computerised glow-curve deconvolution using general and mixed-order kinetics.
    """
    if abs(b - 1.0) < 1e-3:
        b = 1.001

    temperature = np.asarray(T, dtype=float)
    k = KB_EV

    arg = (E / (k * Tm**2)) * (temperature - Tm)
    exp_arg = np.exp(np.clip(arg, -50.0, 50.0))

    numerator = Im * exp_arg
    denominator_term = (1.0 / b) + ((b - 1.0) / b) * exp_arg
    denominator_term = np.maximum(denominator_term, 1e-9)
    return numerator * (denominator_term ** (-b / (b - 1.0)))
