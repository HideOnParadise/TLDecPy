"""Continuous trap-distribution TL models.

This module implements the continuous-distribution fitting form summarized by
Benavente et al. (2019), while keeping explicit links to earlier primary
sources cited there (for example Chen and McKeever, 1997; Kitis and
Gomez-Ros, 2000; Gomez-Ros et al., 2006a,c).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson

from tldecpy.utils.constants import EPSILON, KB_EV
from tldecpy.utils.special import e2_rational

FloatArray = NDArray[np.float64]
ContinuousModel = Literal[
    "continuous_gaussian",
    "continuous_exponential",
    "cont_gauss",
    "cont_exp",
    "cont_lorentz",
]


def _resolve_continuous_aliases(
    Tn: float | None,
    In: float | None,
    E0: float | None,
    Im: float | None,
    E: float | None,
    Tm: float | None,
) -> tuple[float, float, float]:
    """
    Resolve canonical continuous parameters from either canonical or legacy names.

    Canonical: ``(Tn, In, E0)``.
    Legacy aliases: ``(Tm, Im, E)``.
    """
    resolved_Tn = Tn if Tn is not None else Tm
    resolved_In = In if In is not None else Im
    resolved_E0 = E0 if E0 is not None else E

    if resolved_Tn is None or resolved_In is None or resolved_E0 is None:
        raise ValueError(
            "Continuous models require (Tn, In, E0, sigma) "
            "or legacy aliases (Tm, Im, E) plus sigma."
        )
    return float(resolved_Tn), float(resolved_In), float(resolved_E0)


def gaussian_trap_density(E: ArrayLike, E0: float, sigma: float) -> FloatArray:
    """
    Gaussian trap distribution ``n(E)``.

    This distribution form is used in Benavente et al. (2019), Eq. 13, and is
    attributed there to earlier continuous-distribution work (Gomez-Ros et al., 2006a).
    """
    energy = np.asarray(E, dtype=float)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    exponent = -0.5 * ((energy - E0) / sigma) ** 2
    return prefactor * np.exp(exponent)


def exponential_trap_density(E: ArrayLike, E0: float, sigma: float) -> FloatArray:
    """
    Exponential trap distribution ``n(E)``.

    This distribution form is used in Benavente et al. (2019), Eq. 14, and is
    attributed there to earlier continuous-distribution work (Gomez-Ros et al., 2006a).
    """
    energy = np.asarray(E, dtype=float)
    density = np.zeros_like(energy, dtype=float)
    mask = energy >= E0
    density[mask] = (1.0 / sigma) * np.exp(-(energy[mask] - E0) / sigma)
    return density


def _is_cont_gaussian(model: ContinuousModel) -> bool:
    """Return ``True`` when the model key maps to the Gaussian distribution variant."""
    return model in {"continuous_gaussian", "cont_gauss"}


def _is_cont_exponential(model: ContinuousModel) -> bool:
    """Return ``True`` when the model key maps to the Exponential distribution variant."""
    return model in {"continuous_exponential", "cont_exp", "cont_lorentz"}


def _build_energy_grid(
    E0: float,
    sigma: float,
    model: ContinuousModel,
    n_energy: int,
    gaussian_sigma_span: float,
    exponential_sigma_span: float,
) -> FloatArray:
    """
    Build a numerically stable integration grid over activation energy.

    The grid bounds depend on the selected trap distribution:
    Gaussian spans symmetric windows around ``E0`` while Exponential starts at ``E0``.
    """
    points = max(3, int(n_energy))
    if points % 2 == 0:
        points += 1

    if _is_cont_gaussian(model):
        e_min = max(EPSILON, E0 - gaussian_sigma_span * sigma)
        e_max = E0 + gaussian_sigma_span * sigma
    else:
        e_min = max(EPSILON, E0)
        e_max = E0 + exponential_sigma_span * sigma

    if e_max <= e_min:
        e_max = e_min + max(1e-6, 0.1 * max(E0, 1.0))
    return np.linspace(e_min, e_max, points, dtype=float)


def _log_survival_factor(log_s_over_beta: float, F_term: FloatArray) -> FloatArray:
    """
    Logarithm of the survival term ``exp(-(s/beta) * F_term)``.

    Returned value is ``-(s/beta) * F_term`` with stable clipping.
    """
    safe_F = np.maximum(F_term, np.finfo(float).tiny)
    log_arg = log_s_over_beta + np.log(safe_F)
    arg = np.exp(np.clip(log_arg, -745.0, 700.0))
    return -np.minimum(arg, 1e6)


def continuous_glow_peak(
    T: ArrayLike,
    Tn: float,
    In: float,
    E0: float,
    sigma: float,
    model: ContinuousModel = "continuous_gaussian",
    n_energy: int = 801,
    gaussian_sigma_span: float = 8.0,
    exponential_sigma_span: float = 20.0,
    k: float = KB_EV,
) -> FloatArray:
    """
    TL intensity for a continuous trap distribution using the Eq. 17 ratio form
    from Benavente et al. (2019).

    Parameters are the geometric set ``(Tn, In, E0, sigma)`` from
    Benavente et al. (2019), Eqs. 15-16. The underlying first-order kernel and
    continuous-energy integral were introduced in earlier TL literature and are
    cited in that paper.
    """
    if Tn <= 0.0:
        raise ValueError("Tn must be > 0.")
    if In < 0.0:
        raise ValueError("In must be >= 0.")
    if E0 <= 0.0:
        raise ValueError("E0 must be > 0.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")

    temperature = np.asarray(T, dtype=float)
    if temperature.ndim != 1:
        raise ValueError("T must be a 1D array.")
    temp = np.maximum(temperature, EPSILON)

    energy = _build_energy_grid(
        E0=E0,
        sigma=sigma,
        model=model,
        n_energy=n_energy,
        gaussian_sigma_span=gaussian_sigma_span,
        exponential_sigma_span=exponential_sigma_span,
    )

    if _is_cont_gaussian(model):
        f_E = gaussian_trap_density(energy, E0=E0, sigma=sigma)
    elif _is_cont_exponential(model):
        f_E = exponential_trap_density(energy, E0=E0, sigma=sigma)
    else:
        raise ValueError(f"Unsupported continuous model '{model}'.")

    # Eq. 15: s / beta = (E0 / (k*Tn^2)) * exp(E0 / (k*Tn))
    s_over_beta_prefactor = E0 / (k * Tn * Tn)
    log_s_over_beta = np.log(s_over_beta_prefactor) + (E0 / (k * Tn))

    x_T = energy[:, np.newaxis] / (k * temp[np.newaxis, :])
    x_Tn = energy / (k * Tn)

    # Inner temperature integral from Eq. 17 via E2 approximation
    # (Benavente Annex Eq. 27-28, coefficients from Abramowitz and Stegun):
    # F(T, E) = T * E2(E / kT)
    F_T = temp[np.newaxis, :] * e2_rational(x_T)
    F_Tn = Tn * e2_rational(x_Tn)

    f_safe = np.maximum(f_E, np.finfo(float).tiny)
    log_f = np.log(f_safe)
    log_survival_T = _log_survival_factor(log_s_over_beta, F_T)
    log_survival_Tn = _log_survival_factor(log_s_over_beta, F_Tn)

    # Evaluate energy integrals in log-space to prevent underflow.
    log_num_integrand = log_f[:, np.newaxis] - x_T + log_survival_T
    max_log_num = np.max(log_num_integrand, axis=0)
    scaled_num = np.exp(log_num_integrand - max_log_num[np.newaxis, :])
    integral_num_scaled = simpson(scaled_num, x=energy, axis=0)
    integral_num_scaled = np.maximum(integral_num_scaled, np.finfo(float).tiny)
    log_numerator = max_log_num + np.log(integral_num_scaled)

    log_den_integrand = log_f - x_Tn + log_survival_Tn
    max_log_den = float(np.max(log_den_integrand))
    scaled_den = np.exp(log_den_integrand - max_log_den)
    integral_den_scaled = float(simpson(scaled_den, x=energy))
    if (not np.isfinite(integral_den_scaled)) or integral_den_scaled <= 0.0:
        return np.zeros_like(temp, dtype=float)

    log_denominator = max_log_den + np.log(max(integral_den_scaled, np.finfo(float).tiny))
    log_ratio = np.clip(log_numerator - log_denominator, -745.0, 700.0)
    intensity = In * np.exp(log_ratio)
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(intensity, 0.0)


def continuous_gaussian(
    T: ArrayLike,
    Tn: float | None = None,
    In: float | None = None,
    E0: float | None = None,
    sigma: float = 0.05,
    *,
    Im: float | None = None,
    E: float | None = None,
    Tm: float | None = None,
    n_energy: int = 801,
    k: float = KB_EV,
) -> FloatArray:
    r"""
    Evaluate TL intensity for a Gaussian continuous trap-energy distribution.

    Parameters
    ----------
    T : ArrayLike
        Temperature grid :math:`T` in kelvin.
    Tn : float | None, optional
        Characteristic temperature :math:`T_N` (K).
    In : float | None, optional
        Characteristic intensity :math:`I_N`.
    E0 : float | None, optional
        Characteristic activation energy :math:`E_0` (eV).
    sigma : float, default=0.05
        Standard deviation of the Gaussian energy distribution (eV).
    Im, E, Tm : float | None, optional
        Legacy aliases mapped internally to ``In``, ``E0`` and ``Tn``.
    n_energy : int, default=801
        Number of quadrature nodes for the energy integral.
    k : float, default=KB_EV
        Boltzmann constant in eV/K.

    Returns
    -------
    numpy.ndarray
        Simulated TL intensity evaluated at each value of ``T``.
    """
    resolved_Tn, resolved_In, resolved_E0 = _resolve_continuous_aliases(
        Tn=Tn,
        In=In,
        E0=E0,
        Im=Im,
        E=E,
        Tm=Tm,
    )
    return continuous_glow_peak(
        T=T,
        Tn=resolved_Tn,
        In=resolved_In,
        E0=resolved_E0,
        sigma=sigma,
        model="continuous_gaussian",
        n_energy=n_energy,
        k=k,
    )


def continuous_exponential(
    T: ArrayLike,
    Tn: float | None = None,
    In: float | None = None,
    E0: float | None = None,
    sigma: float = 0.05,
    *,
    Im: float | None = None,
    E: float | None = None,
    Tm: float | None = None,
    n_energy: int = 801,
    k: float = KB_EV,
) -> FloatArray:
    r"""
    Evaluate TL intensity for an Exponential continuous trap-energy distribution.

    Parameters
    ----------
    T : ArrayLike
        Temperature grid :math:`T` in kelvin.
    Tn : float | None, optional
        Characteristic temperature :math:`T_N` (K).
    In : float | None, optional
        Characteristic intensity :math:`I_N`.
    E0 : float | None, optional
        Characteristic activation energy :math:`E_0` (eV).
    sigma : float, default=0.05
        Characteristic width of the exponential tail (eV).
    Im, E, Tm : float | None, optional
        Legacy aliases mapped internally to ``In``, ``E0`` and ``Tn``.
    n_energy : int, default=801
        Number of quadrature nodes for the energy integral.
    k : float, default=KB_EV
        Boltzmann constant in eV/K.

    Returns
    -------
    numpy.ndarray
        Simulated TL intensity evaluated at each value of ``T``.
    """
    resolved_Tn, resolved_In, resolved_E0 = _resolve_continuous_aliases(
        Tn=Tn,
        In=In,
        E0=E0,
        Im=Im,
        E=E,
        Tm=Tm,
    )
    return continuous_glow_peak(
        T=T,
        Tn=resolved_Tn,
        In=resolved_In,
        E0=resolved_E0,
        sigma=sigma,
        model="continuous_exponential",
        n_energy=n_energy,
        k=k,
    )


__all__ = [
    "continuous_exponential",
    "continuous_gaussian",
    "continuous_glow_peak",
    "exponential_trap_density",
    "gaussian_trap_density",
]
