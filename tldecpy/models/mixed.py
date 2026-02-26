"""
Mixed-Order Kinetics (MOK) Models.
Parameters: Im, E, Tm, alpha (0 < alpha < 1).
"""

from __future__ import annotations

import numpy as np
from tldecpy.utils.constants import KB_EV
from tldecpy.utils.special import exp_int_approx


def _check_alpha(alpha):
    """Clamp mixed-order parameter ``alpha`` to the open interval ``(0, 1)``."""
    return np.clip(alpha, 1e-4, 1.0 - 1e-4)


def mo_kitis(T: np.ndarray, Im: float, E: float, Tm: float, alpha: float) -> np.ndarray:
    """
    Mixed-order expression after Kitis and Gomez-Ros (2000).
    """
    alpha = _check_alpha(alpha)
    k = KB_EV
    T = np.maximum(T, 1e-5)

    Am = 2.6 - 1.9203 * alpha + 0.324 * (alpha**3.338)
    Rm = (Am + alpha) / (Am - alpha)

    delta = (2 * k * T) / E
    delta_m = (2 * k * Tm) / E

    arg_exp = (E / (k * T)) * ((T - Tm) / Tm)
    exp_theta = np.exp(np.clip(arg_exp, -50, 50))

    term1 = (np.exp((1.0 - delta_m) / Rm) - alpha) ** 2
    term3_arg = (T**2 / (Tm**2 * Rm)) * exp_theta * (1.0 - delta)
    term3 = np.exp(np.clip(term3_arg, -50, 50))

    numerator = Im * term1 * exp_theta * term3
    denom_pre = np.exp((1.0 - delta_m) / Rm)
    denom_bracket = term3 - alpha

    denominator = denom_pre * (denom_bracket**2)
    return numerator / np.maximum(denominator, 1e-9)


def mo_quad(T: np.ndarray, Im: float, E: float, Tm: float, alpha: float) -> np.ndarray:
    """
    Quadratic mixed-order approximation after Gomez-Ros and Kitis (2002).
    Corrected with auto-normalization to ensure peak height = Im.
    """
    alpha = _check_alpha(alpha)
    k = KB_EV
    T = np.maximum(T, 1e-5)

    Rm = (1.0 - alpha) * (1.0 + 0.2922 * alpha - 0.2783 * alpha**2)
    z_Tm = E / (k * Tm)
    F_Tm = exp_int_approx(z_Tm)

    # Calculate shape function S(T)
    def calculate_shape(temp_arr):
        z = E / (k * temp_arr)
        F_val = exp_int_approx(z)

        # exp( E/kTm - E/kT )
        exp_diff = np.exp(np.clip(z_Tm - z, -50, 50))

        bracket_arg = Rm * z_Tm * ((temp_arr / Tm) * exp_diff * F_val - F_Tm)
        exp_bracket = np.exp(np.clip(bracket_arg, -50, 50))

        C = (1.0 - Rm) / (1.0 + Rm)

        num = 4.0 * (Rm**2) * exp_diff * exp_bracket
        den = (1.0 + Rm) * ((exp_bracket - C) ** 2)
        return num / np.maximum(den, 1e-9)

    S_T = calculate_shape(T)

    # Normalize by value at Tm
    S_Tm = calculate_shape(np.array([Tm]))[0]

    return Im * (S_T / S_Tm)


def mo_vej(T: np.ndarray, Im: float, E: float, Tm: float, alpha: float) -> np.ndarray:
    r"""
    Mixed-order expression by Vejnovic et al. (2008).

    Notes
    -----
    This implementation follows the four-parameter fitting form shown as Eq. (20)
    in Vejnovic et al., Radiation Measurements 43 (2008) 1325-1330, using:

    - :math:`l = 2/(2-\alpha)` (Eq. 21)
    - :math:`\sigma = T - T_m`
    - :math:`\Delta = 2kT/E`
    """
    alpha = _check_alpha(alpha)
    k = KB_EV
    T = np.maximum(T, 1e-5)

    l_val = 2.0 / (2.0 - alpha)
    delta_T = (2 * k * T) / E

    arg_rise = (E / (k * Tm)) * ((T - Tm) / Tm)
    exp_rise = np.exp(np.clip(arg_rise, -50, 50))

    factor_l = (2.0 / l_val) - 1.0
    bracket_arg = (T**2 / Tm**2) * factor_l * exp_rise * (1.0 - delta_T)
    exp_bracket = np.exp(np.clip(bracket_arg, -50, 50))

    prefactor = alpha * Im * ((2.0 - l_val) ** 2) / (l_val - 1.0)

    numerator = prefactor * exp_bracket * exp_rise
    denominator = (exp_bracket - alpha) ** 2

    return numerator / np.maximum(denominator, 1e-9)
