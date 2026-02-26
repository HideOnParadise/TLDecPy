"""
OTOR Model using Wright Omega function (otor_wo).

Uses Wright Omega forms consistent with OTOR derivations in:
- Singh and Gartia (2013, 2014, 2015)
- Kitis and Vlachos (2013)
- Sadek et al. (2015)
"""

import numpy as np
from tldecpy.utils.constants import KB_EV
from tldecpy.utils.special import wrightomega_safe, expi_integral
from tldecpy.utils.stability import clamp_R_near_one, safe_exp


def otor_wo(
    T: np.ndarray, Im: float, E: float, Tm: float, R: float, beta: float = 1.0
) -> np.ndarray:
    """
    Evaluate OTOR intensity using Wright Omega representation.

    Parameters
    ----------
    T : numpy.ndarray
        Temperature grid in kelvin.
    Im : float
        Peak intensity.
    E : float
        Activation energy in eV.
    Tm : float
        Peak temperature in kelvin.
    R : float
        Retrapping ratio :math:`R=A_n/A_m`.
    beta : float, default=1.0
        Heating rate (kept for API compatibility).

    Returns
    -------
    numpy.ndarray
        Intensity values :math:`I(T)`.
    """
    k = KB_EV
    R = clamp_R_near_one(R)

    # 1. Integral F(T)
    F_T = expi_integral(T, E, k)
    F_Tm = expi_integral(np.array([Tm]), E, k)[0]

    # 2. Correction factor (same empirical branch logic used by the LW variant)
    # Wright Omega form uses the same denominator correction
    # (1 - 1.05 * R^1.26) typically for R < 1.
    # Note: Wright Omega handles the transition continuously, but the empirical
    # correction factor for peak position derived by Kitis depends on R.
    # We use the same logic as otor_lw for the 'Q' factor.

    if R < 1.0:
        corr = 1.0 - 1.05 * (R**1.26)
    else:
        corr = 2.963 - 3.24 * (R**-0.74)

    if abs(corr) < 1e-3:
        corr = 1e-3 * np.sign(corr)

    exp_term = float(safe_exp(np.asarray([E / (k * Tm)], dtype=float))[0])
    Q = (E * exp_term) / (k * Tm**2 * corr)

    # 3. Argument Z for Wright Omega.
    # Wright Omega solves w + ln(w) = z and corresponds to W(exp(z)).

    if R < 1:
        # Term: ln( (1-R)/R )
        log_term = np.log((1.0 - R) / R)
        Z0 = (R / (1.0 - R)) - log_term
    else:
        # For R > 1, use the same real-valued branch convention as the LW variant.
        abs_frac = R / (R - 1.0)  # Negative number
        log_term = np.log((R - 1.0) / R)
        Z0 = abs_frac + log_term

    Z_T = Z0 + Q * F_T
    Z_Tm = Z0 + Q * F_Tm

    # 4. Evaluate Wright Omega.
    w_T = wrightomega_safe(Z_T)
    w_Tm = wrightomega_safe(Z_Tm)

    # 5. Intensity
    term_exp = safe_exp((-E / (k * T)) + (E / (k * Tm)))

    num = w_Tm + w_Tm**2
    den = w_T + w_T**2

    # Guard
    den = np.where(den == 0, 1e-9, den)

    I_calc = Im * term_exp * (num / den)

    return np.maximum(I_calc, 0.0)
