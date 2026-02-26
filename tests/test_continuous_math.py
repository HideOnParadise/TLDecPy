"""Mathematical validation tests for continuous trap-distribution models."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import expn

from tldecpy.models.continuous import continuous_glow_peak
from tldecpy.utils.constants import KB_EV
from tldecpy.utils.special import exp_int_approx


def _gaussian_density(E: np.ndarray, E0: float, sigma: float) -> np.ndarray:
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return prefactor * np.exp(-0.5 * ((E - E0) / sigma) ** 2)


def _exponential_density(E: np.ndarray, E0: float, sigma: float) -> np.ndarray:
    out = np.zeros_like(E, dtype=float)
    mask = E >= E0
    out[mask] = (1.0 / sigma) * np.exp(-(E[mask] - E0) / sigma)
    return out


def _energy_grid(
    *,
    model: str,
    E0: float,
    sigma: float,
    n_energy: int = 2001,
    gaussian_sigma_span: float = 10.0,
    exponential_sigma_span: float = 30.0,
) -> np.ndarray:
    n = max(3, int(n_energy))
    if n % 2 == 0:
        n += 1

    if model == "cont_gauss":
        lo = max(1e-8, E0 - gaussian_sigma_span * sigma)
        hi = E0 + gaussian_sigma_span * sigma
    else:
        lo = max(1e-8, E0)
        hi = E0 + exponential_sigma_span * sigma

    return np.linspace(lo, hi, n, dtype=float)


def _continuous_reference_exact_e2(
    T: np.ndarray,
    *,
    Tn: float,
    In: float,
    E0: float,
    sigma: float,
    model: str,
    k: float = KB_EV,
) -> np.ndarray:
    """
    Independent reference for Eq. 17 using exact E2 (scipy.special.expn).

    This intentionally avoids the rational E2 approximation used in production.
    """
    temp = np.asarray(T, dtype=float)
    energy = _energy_grid(model=model, E0=E0, sigma=sigma)

    if model == "cont_gauss":
        f_E = _gaussian_density(energy, E0=E0, sigma=sigma)
    else:
        f_E = _exponential_density(energy, E0=E0, sigma=sigma)

    x_T = energy[:, None] / (k * temp[None, :])
    x_Tn = energy / (k * Tn)

    F_T = temp[None, :] * expn(2, x_T)
    F_Tn = Tn * expn(2, x_Tn)

    log_s_over_beta = np.log(E0 / (k * Tn * Tn)) + (E0 / (k * Tn))
    s_over_beta = float(np.exp(np.clip(log_s_over_beta, -300.0, 300.0)))

    expo_num = np.minimum(s_over_beta * F_T, 700.0)
    expo_den = np.minimum(s_over_beta * F_Tn, 700.0)

    num = f_E[:, None] * np.exp(-x_T) * np.exp(-expo_num)
    den = f_E * np.exp(-x_Tn) * np.exp(-expo_den)

    numerator = simpson(num, x=energy, axis=0)
    denominator = float(simpson(den, x=energy))
    denominator = max(denominator, np.finfo(float).tiny)
    y = In * (numerator / denominator)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(y, 0.0)


@pytest.mark.parametrize(
    "model,params",
    [
        ("cont_gauss", {"Tn": 500.0, "In": 12000.0, "E0": 1.0, "sigma": 0.05}),
        ("cont_exp", {"Tn": 480.0, "In": 9000.0, "E0": 0.9, "sigma": 0.04}),
    ],
)
def test_continuous_glow_peak_matches_exact_e2_reference(
    model: str,
    params: dict[str, float],
) -> None:
    """
    Production curve should match an independent exact-E2 reference closely.
    """
    T = np.linspace(350.0, 650.0, 301)
    y_fast = continuous_glow_peak(T=T, model=model, n_energy=1201, **params)
    y_ref = _continuous_reference_exact_e2(T=T, model=model, **params)

    norm = np.maximum(np.max(y_ref), 1.0)
    rel_l2 = float(np.linalg.norm(y_fast - y_ref) / np.linalg.norm(y_ref))
    rel_max = float(np.max(np.abs(y_fast - y_ref)) / norm)

    assert rel_l2 < 0.02, f"{model} rel_l2={rel_l2:.4g}"
    assert rel_max < 0.03, f"{model} rel_max={rel_max:.4g}"


def test_e2_rational_scaled_accuracy_vs_scipy_expn() -> None:
    """
    Validate Eq. 28 scaled approximation R(x) = exp(x) * E2(x).
    """
    # The production fit is the Abramowitz-Stegun rational form (restated in Benavente),
    # used in the TL domain x = E/(kT), where
    # practical values are typically >= 1. Near x -> 0 the approximation is
    # not constrained to match the exact limit R(0) = 1.
    x = np.logspace(0.0, 2.1, 800, dtype=float)  # ~[1, 126]
    approx = exp_int_approx(x)
    exact = np.exp(x) * expn(2, x)

    abs_err = np.abs(approx - exact)
    max_abs = float(np.max(abs_err))
    p95_abs = float(np.percentile(abs_err, 95))

    assert max_abs < 0.02, f"max_abs={max_abs:.4g}"
    assert p95_abs < 0.01, f"p95_abs={p95_abs:.4g}"


@pytest.mark.parametrize("model", ["cont_gauss", "cont_exp"])
def test_continuous_glow_peak_respects_characteristic_intensity(model: str) -> None:
    """
    By construction of Eq. 17 reparametrization, I(Tn) should be close to In.
    """
    params = {"Tn": 500.0, "In": 7000.0, "E0": 1.0, "sigma": 0.05}
    y = continuous_glow_peak(np.array([params["Tn"]]), model=model, n_energy=1201, **params)
    assert np.isfinite(y[0])
    assert y[0] > 0.0
    assert np.isclose(y[0], params["In"], rtol=5e-3), f"{model} I(Tn)={y[0]:.6g}"
