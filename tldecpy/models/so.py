"""Second-order kinetic models for thermoluminescence glow curves."""

from __future__ import annotations

import numpy as np

from tldecpy.utils.constants import KB_EV


def so_ks(T: np.ndarray, Im: float, E: float, Tm: float) -> np.ndarray:
    r"""
    Evaluate the Kitis-standard second-order expression.

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
    temperature = np.maximum(np.asarray(T, dtype=float), 1e-5)
    k = KB_EV

    delta = (2.0 * k * temperature) / E
    delta_m = (2.0 * k * Tm) / E

    arg = (E / (k * temperature)) * ((temperature - Tm) / Tm)
    exp_arg = np.exp(np.clip(arg, -50.0, 50.0))

    numerator = 4.0 * Im * exp_arg
    denominator_term = (temperature**2 / Tm**2) * (1.0 - delta) * exp_arg + 1.0 + delta_m

    return numerator * (denominator_term**-2)


def so_la(T: np.ndarray, Im: float, E: float, Tm: float) -> np.ndarray:
    r"""
    Evaluate the logistic-asymmetric second-order approximation.

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

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.

    References
    ----------
    .. [1] Pagonis, V., and Kitis, G. (2001). Fit of second-order
           thermoluminescence glow peaks using the logistic distribution function.
    """
    temperature = np.asarray(T, dtype=float)
    k = KB_EV

    numerator = 1.189 * (Tm**4) * (k**2)
    denominator = E**2 + 4.0 * E * Tm * k
    a2 = np.sqrt(numerator / denominator)

    x_value = (temperature - Tm) / a2 + 0.38542
    exp_neg_x = np.exp(np.clip(-x_value, -50.0, 50.0))
    term = (1.0 + exp_neg_x) ** (-2.4702)
    return 5.2973 * Im * term * exp_neg_x

