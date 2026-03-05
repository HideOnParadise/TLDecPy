"""
OTOR Model using Lambert W function (otor_lw).

Uses semi-analytical OTOR expressions from the primary literature.

Numerical-stability strategy:
- For R < 1: compute W(exp(Z)) in a stable way (no exp overflow) using:
    * W(exp(Z)) = lambertw(exp(Z)) for moderate Z
    * asymptotic approximation W(exp(Z)) ~ Z - ln(Z) for large Z
- Compute Q in log-space to avoid overflow from exp(E/kTm)
- Clip R for R < 1 away from the empirical singularity near ~0.96 (use <= 0.95)
- Keep the R > 1 branch behavior intact, with an extra safety clip so the lambertw(-1, .)
  argument stays in the real-domain interval [-1/e, 0).

References:
- Kitis and Vlachos (2013), Radiation Measurements 48, 47-54.
- Sadek et al. (2015), Applied Radiation and Isotopes 95, 214-221.
"""

from __future__ import annotations

import numpy as np

from tldecpy.utils.constants import KB_EV
from tldecpy.utils.special import lambertw_safe, expi_integral
from tldecpy.utils.stability import clamp_R_near_one, safe_exp


def _wexpz_principal_stable(Z: np.ndarray) -> np.ndarray:
    """
    Stable evaluation of W0(exp(Z)) for real Z.

    For Z <= threshold: compute W(exp(Z)) safely (exp(Z) won't overflow).
    For Z  > threshold: use asymptotic approximation W(exp(Z)) ~ Z - ln(Z).

    This keeps evaluation stable for very large positive arguments.
    """
    Z = np.asarray(Z, dtype=np.float64)
    out = np.empty_like(Z)

    threshold = 500.0  # exp(500) is huge but still finite in float64
    mask = Z <= threshold

    if np.any(mask):
        out[mask] = lambertw_safe(np.exp(Z[mask]), k=0)

    if np.any(~mask):
        z = np.maximum(Z[~mask], 1e-300)
        out[~mask] = z - np.log(z)

    return out


def otor_lw(
    T: np.ndarray, Im: float, E: float, Tm: float, R: float, beta: float = 1.0
) -> np.ndarray:
    r"""
    OTOR model using the Lambert W semi-analytical solution.

    Implements the One Trap One Recombination center (OTOR) kinetics using
    the analytical Lambert W expressions derived by Kitis & Vlachos (2013)
    and refined by Sadek et al. (2015).  The peak maximum is forced to occur
    at :math:`T_m` via empirical correction factors applied to the frequency
    factor :math:`s`.

    Two branches are handled automatically:

    - :math:`R < 1` branch — principal Lambert W function :math:`W_0`.
    - :math:`R > 1` branch — branch :math:`W_{-1}`.

    Parameters
    ----------
    T : numpy.ndarray
        Temperature array in kelvin (1-D, float64).  Values below 1e-12 K
        are clamped to prevent division by zero.
    Im : float
        Peak maximum intensity :math:`I_m` in detector counts.
        Must be > 0.
    E : float
        Trap activation energy :math:`E` in eV.  Range: (0.1, 5.0) eV.
    Tm : float
        Temperature at peak maximum :math:`T_m` in kelvin.  Range: > 0 K.
    R : float
        Retrapping ratio :math:`R = A_n / A_m`, where :math:`A_n` is the
        retrapping coefficient and :math:`A_m` the recombination coefficient
        (both in cm³/s).

        - :math:`R \to 0` — negligible retrapping (approaches FO kinetics).
        - :math:`R = 0.5` — equal retrapping and recombination.
        - :math:`R \to 1` — dominant retrapping (approaches SO kinetics).

        .. warning::
           A numerical singularity exists near :math:`R \approx 0.96` on the
           :math:`R < 1` branch.  The implementation clamps :math:`R \leq 0.95`
           internally.  **Always set the upper bound to at most 0.90** in
           :class:`~tldecpy.schemas.PeakSpec` bounds.
    beta : float, default=1.0
        Heating rate :math:`\beta` in K/s.  Kept for API compatibility;
        the current semi-analytical form absorbs :math:`\beta` into the
        empirical correction of :math:`s` and does not use it directly.

    Returns
    -------
    numpy.ndarray
        TL intensity array :math:`I(T)` in the same units as ``Im``.
        All values are non-negative (clipped to 0).

    Notes
    -----
    The intensity is computed as:

    .. math::

        I(T) = I_m \,
        \exp\!\left[-\frac{E}{kT}\cdot\frac{T_m-T}{T_m}\right]
        \frac{W(T_m) + W(T_m)^2}{W(T) + W(T)^2}

    where :math:`W(T)` is the Lambert W function evaluated at a
    temperature-dependent argument derived from the OTOR rate equations.

    Examples
    --------
    >>> import numpy as np
    >>> import tldecpy as tl
    >>> T = np.linspace(350.0, 600.0, 300)
    >>> I = tl.get_model("otor_lw")(T, Im=1000.0, E=1.3, Tm=460.0, R=0.15)
    >>> print(f"Peak at T={T[I.argmax()]:.1f} K, I_max={I.max():.1f}")

    References
    ----------
    .. [1] Kitis, G., and Vlachos, N. D. (2013). *General semi-analytical
           expressions for TL, OSL and other luminescence stimulation modes
           derived from the OTOR model using the Lambert-W function.*
           Radiat. Meas. 48, 47.
    .. [2] Sadek, A. M., et al. (2015). *Characterization of the one trap
           one recombination (OTOR) level model using the Lambert-W function.*
           Appl. Radiat. Isot. 95, 214.
    """
    k = KB_EV

    T = np.asarray(T, dtype=np.float64)
    T = np.maximum(T, 1e-12)

    # ---- Integral function F(T, E) ----
    # F(T,E) = T*exp(-E/kT) + (E/k)*Ei(-E/kT)
    F_T = expi_integral(T, E, k)
    F_Tm = expi_integral(np.array([Tm], dtype=np.float64), E, k)[0]

    # ---- Branch handling ----
    # We decide branch based on the incoming R, but we still keep it away from exactly 1.
    R = clamp_R_near_one(float(R))

    if R < 1.0:
        # === R < 1 branch (Eq 10.1 style) ===
        # Keep away from the empirical singularity near ~0.96
        R = float(np.clip(R, 1e-12, 0.95))

        # Empirical correction factor C_R = 1 - 1.05 * R^1.26
        C_R = 1.0 - 1.05 * (R**1.26)
        C_R = max(C_R, 1e-12)

        # term_R = R/(1-R) - ln((1-R)/R)
        term_R = (R / (1.0 - R)) - np.log((1.0 - R) / R)

        # Q = E * exp(E/kTm) / (k * Tm^2 * C_R)  (computed in log-space)
        # log_Q = ln(E) + E/(kTm) - ln(k*Tm^2*C_R)
        # Use exp(clipped) to avoid overflow
        log_Q = np.log(max(E, 1e-300)) + (E / (k * Tm)) - np.log(max(k * (Tm**2) * C_R, 1e-300))
        Q = np.exp(np.clip(log_Q, -700.0, 700.0))

        Z_Tm = term_R + Q * F_Tm
        Z_T = term_R + Q * F_T

        # W(exp(Z)) on principal branch, computed stably
        W_Tm = _wexpz_principal_stable(np.array([Z_Tm], dtype=np.float64))[0]
        W_T = _wexpz_principal_stable(Z_T)

    else:
        # === R > 1 branch (Eq 10.2 style) ===
        # Keep away from exactly 1
        if R <= 1.0:
            R = 1.0 + 1e-4

        # Correction factor: 2.963 - 3.24 * R^(-0.74)
        corr = 2.963 - 3.24 * (R**-0.74)
        # Keep the original sign behavior, but avoid a near-zero denominator
        if abs(corr) < 1e-12:
            corr = 1e-12 * (1.0 if corr >= 0 else -1.0)

        # Keep the established Q form here (to avoid changing behavior near the R>1 correction sign),
        # but still avoid exp overflow via safe_exp.
        exp_term = float(safe_exp(np.asarray([E / (k * Tm)], dtype=float))[0])
        Q = (E * exp_term) / (k * (Tm**2) * corr)

        # Z0 = |R/(1-R)| + log(|(1-R)/R|)
        abs_frac = R / (R - 1.0)
        Z0 = abs_frac + np.log((R - 1.0) / R)

        Z_T = Z0 + Q * F_T
        Z_Tm = Z0 + Q * F_Tm

        # Argument for branch -1: -exp(-Z)
        arg_T = -safe_exp(-Z_T)
        arg_Tm = float(-safe_exp(np.asarray([-Z_Tm], dtype=float))[0])

        # Real-valued W_{-1}(x) exists for x in [-1/e, 0).
        # Clip to stay in-domain.
        eps = 1e-15
        lower = -1.0 / np.e + eps
        upper = -1e-300
        arg_T = np.clip(arg_T, lower, upper)
        arg_Tm = float(np.clip(arg_Tm, lower, upper))

        W_T = lambertw_safe(arg_T, k=-1)
        W_Tm = lambertw_safe(arg_Tm, k=-1)

    # ---- Intensity ----
    # exp_arg = (-E/(kT)) * ((Tm - T)/Tm) = -E/(kT) + E/(kTm)
    exp_arg = (-E / (k * T)) * ((Tm - T) / Tm)
    term_exp = np.exp(np.clip(exp_arg, -700.0, 700.0))

    num = W_Tm + W_Tm**2
    den = W_T + W_T**2
    den = np.maximum(den, 1e-12)

    I_calc = Im * term_exp * (num / den)
    return np.maximum(I_calc, 0.0)
