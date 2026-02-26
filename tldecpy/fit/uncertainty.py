"""Uncertainty utilities for deconvolution outputs."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


def compute_parameter_covariance(
    jac: np.ndarray | None,
    residuals: np.ndarray,
    dof: int | None = None,
) -> np.ndarray | None:
    """
    Estimate parameter covariance from the final least-squares Jacobian.

    Parameters
    ----------
    jac:
        Jacobian matrix of shape (n_samples, n_params).
    residuals:
        Residual vector of shape (n_samples,).
    dof:
        Degrees of freedom used to estimate residual variance.
        If omitted, uses ``max(n_samples - n_params, 1)``.
    """
    if jac is None or jac.size == 0:
        return None

    jac_arr = np.asarray(jac, dtype=float)
    if jac_arr.ndim != 2:
        return None

    n_samples, n_params = jac_arr.shape
    if n_samples <= 0 or n_params <= 0:
        return None

    resid = np.asarray(residuals, dtype=float).reshape(-1)
    if resid.size != n_samples:
        return None

    if dof is None:
        dof_eff = max(n_samples - n_params, 1)
    else:
        dof_eff = max(int(dof), 1)

    residual_variance = float(np.dot(resid, resid) / dof_eff)

    try:
        jtj = jac_arr.T @ jac_arr
        cond = np.linalg.cond(jtj)
        if not np.isfinite(cond) or cond > 1e14:
            return None

        cov = np.linalg.pinv(jtj) * residual_variance
        cov = 0.5 * (cov + cov.T)
        if not np.all(np.isfinite(cov)):
            return None
        return cov
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None


def compute_sensitivity_matrix(
    theta: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    step_scale: float = 1e-6,
    min_step: float = 1e-8,
) -> np.ndarray:
    """Compute model sensitivities dy/dtheta using central differences."""
    theta0 = np.asarray(theta, dtype=float).reshape(-1)
    if theta0.size == 0:
        return np.zeros((0, 0), dtype=float)

    y0 = np.asarray(model(theta0), dtype=float).reshape(-1)
    n_points = y0.size
    n_params = theta0.size
    sens = np.zeros((n_points, n_params), dtype=float)

    for j in range(n_params):
        step = max(abs(theta0[j]) * step_scale, min_step)
        theta_plus = theta0.copy()
        theta_minus = theta0.copy()
        theta_plus[j] += step
        theta_minus[j] -= step
        y_plus = np.asarray(model(theta_plus), dtype=float).reshape(-1)
        y_minus = np.asarray(model(theta_minus), dtype=float).reshape(-1)
        if y_plus.size != n_points or y_minus.size != n_points:
            raise ValueError("Model output size changed during sensitivity evaluation.")
        sens[:, j] = (y_plus - y_minus) / (2.0 * step)

    return sens


def compute_absolute_uncertainty_from_covariance(
    sensitivity: np.ndarray,
    covariance: np.ndarray | None,
) -> np.ndarray:
    """Compute absolute uncertainty curve from sensitivity and covariance."""
    sens = np.asarray(sensitivity, dtype=float)
    if sens.ndim != 2:
        return np.full(0, np.nan, dtype=float)

    n_points, n_params = sens.shape
    if covariance is None:
        return np.full(n_points, np.nan, dtype=float)

    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (n_params, n_params):
        return np.full(n_points, np.nan, dtype=float)

    uc_abs_sq = np.einsum("ij,jk,ik->i", sens, cov, sens, optimize=True)
    uc_abs_sq = np.where(np.isfinite(uc_abs_sq) & (uc_abs_sq >= 0.0), uc_abs_sq, np.nan)
    return np.sqrt(uc_abs_sq)


def _relative_percent_curve(y_fit: np.ndarray, abs_curve: np.ndarray) -> np.ndarray:
    """Convert absolute uncertainty values into channel-wise relative percentages."""
    y_arr = np.asarray(y_fit, dtype=float).reshape(-1)
    abs_arr = np.asarray(abs_curve, dtype=float).reshape(-1)
    if y_arr.size != abs_arr.size:
        return np.full(y_arr.size, np.nan, dtype=float)

    if y_arr.size == 0:
        return np.full(0, np.nan, dtype=float)

    scale = max(float(np.nanmax(np.abs(y_arr))), 1.0)
    denom = np.maximum(np.abs(y_arr), 1e-12 * scale)
    uc_rel = 100.0 * abs_arr / denom
    return np.where(np.isfinite(uc_rel), uc_rel, np.nan)


def absolute_to_relative_percent(y_fit: np.ndarray, abs_curve: np.ndarray) -> np.ndarray:
    """Public wrapper to convert absolute uncertainty to relative percent."""
    return _relative_percent_curve(y_fit, abs_curve)


def compute_uc_curve(
    x: np.ndarray,
    y_fit: np.ndarray,
    theta: np.ndarray,
    cov_theta: np.ndarray | None,
    model: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Compute channel-wise relative combined standard uncertainty in percent.

    Returns ``u_c(T_i)`` as percentages for each channel.
    """
    _ = np.asarray(x, dtype=float)  # kept for API consistency
    y_arr = np.asarray(y_fit, dtype=float).reshape(-1)
    if y_arr.size == 0:
        return np.full(0, np.nan, dtype=float)

    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if theta_arr.size == 0 or cov_theta is None:
        return np.full(y_arr.size, np.nan, dtype=float)

    try:
        sensitivity = compute_sensitivity_matrix(theta_arr, model=model)
    except (ValueError, FloatingPointError):
        return np.full(y_arr.size, np.nan, dtype=float)

    uc_abs = compute_absolute_uncertainty_from_covariance(sensitivity, cov_theta)
    if uc_abs.size != y_arr.size:
        return np.full(y_arr.size, np.nan, dtype=float)
    return _relative_percent_curve(y_arr, uc_abs)


def combine_uncertainty_sources(
    y_fit: np.ndarray,
    source_abs_curves: Mapping[str, np.ndarray],
    correlations: Mapping[str, float] | None = None,
) -> np.ndarray:
    """
    Combine source uncertainty curves including optional source correlations.

    Parameters
    ----------
    y_fit:
        Fitted signal used for relative normalization.
    source_abs_curves:
        Absolute uncertainty curves per source.
    correlations:
        Correlation map with keys ``"sourceA:sourceB"`` and values in [-1, 1].
    """
    y_arr = np.asarray(y_fit, dtype=float).reshape(-1)
    if y_arr.size == 0:
        return np.full(0, np.nan, dtype=float)

    arrays: dict[str, np.ndarray] = {}
    for name, values in source_abs_curves.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != y_arr.size:
            continue
        arrays[name] = np.where(np.isfinite(arr), np.abs(arr), 0.0)

    if not arrays:
        return np.full(y_arr.size, np.nan, dtype=float)

    names = list(arrays.keys())
    u_sq = np.zeros_like(y_arr, dtype=float)
    for name in names:
        u_sq += arrays[name] ** 2

    corr = correlations or {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            rho = corr.get(f"{left}:{right}", corr.get(f"{right}:{left}", 0.0))
            rho = float(np.clip(rho, -0.99, 0.99))
            if rho == 0.0:
                continue
            u_sq += 2.0 * rho * arrays[left] * arrays[right]

    u_sq = np.where(np.isfinite(u_sq), np.maximum(u_sq, 0.0), np.nan)
    return _relative_percent_curve(y_arr, np.sqrt(u_sq))


def compute_global_contribution_percent(y_fit: np.ndarray, abs_curve: np.ndarray) -> float:
    """Compute area-normalized global contribution from an absolute uncertainty curve."""
    y_arr = np.asarray(y_fit, dtype=float).reshape(-1)
    u_arr = np.asarray(abs_curve, dtype=float).reshape(-1)
    if y_arr.size == 0 or y_arr.size != u_arr.size:
        return float("nan")

    valid = np.isfinite(y_arr) & np.isfinite(u_arr)
    if not np.any(valid):
        return float("nan")

    area = float(np.sum(np.abs(y_arr[valid])))
    if area <= 0.0:
        return float("nan")

    return float(100.0 * np.sum(np.abs(u_arr[valid])) / area)


def compute_uc_global(x: np.ndarray, y_fit: np.ndarray, uc_curve: np.ndarray) -> float:
    """
    Compute global uncertainty criterion as an area-normalized percentage.

    The scalar is analogous to FOM but replacing residual magnitudes with
    uncertainty magnitudes.
    """
    _ = np.asarray(x, dtype=float)  # reserved for future weighted integrations
    y_arr = np.asarray(y_fit, dtype=float).reshape(-1)
    uc_arr = np.asarray(uc_curve, dtype=float).reshape(-1)
    if y_arr.size == 0 or uc_arr.size != y_arr.size:
        return float("nan")

    valid = np.isfinite(uc_arr) & np.isfinite(y_arr)
    if not np.any(valid):
        return float("nan")

    scale = max(float(np.nanmax(np.abs(y_arr[valid]))), 1.0)
    uc_abs = (uc_arr[valid] / 100.0) * np.maximum(np.abs(y_arr[valid]), 1e-12 * scale)
    area = float(np.sum(np.abs(y_arr[valid])))
    if area <= 0.0:
        return float("nan")

    return float(100.0 * np.sum(uc_abs) / area)


def monte_carlo_uncertainty_curve(
    theta: np.ndarray,
    cov_theta: np.ndarray | None,
    model: Callable[[np.ndarray], np.ndarray],
    *,
    n_samples: int = 200,
    seed: int | None = 42,
    bounds_low: np.ndarray | None = None,
    bounds_high: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate absolute uncertainty curve via parameter Monte Carlo sampling.

    Samples are drawn from ``N(theta, cov_theta)``.
    """
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if cov_theta is None or theta_arr.size == 0 or n_samples < 2:
        y0 = np.asarray(model(theta_arr), dtype=float).reshape(-1)
        return np.full(y0.size, np.nan, dtype=float)

    cov = np.asarray(cov_theta, dtype=float)
    if cov.shape != (theta_arr.size, theta_arr.size):
        y0 = np.asarray(model(theta_arr), dtype=float).reshape(-1)
        return np.full(y0.size, np.nan, dtype=float)

    try:
        eval0 = np.asarray(model(theta_arr), dtype=float).reshape(-1)
    except Exception:
        return np.full(0, np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    n_points = eval0.size
    samples = np.empty((n_samples, n_points), dtype=float)

    cov_reg = cov.copy()
    jitter = 1e-14 * max(float(np.trace(cov_reg)), 1.0)
    cov_reg.flat[:: cov_reg.shape[0] + 1] += jitter

    b_low = None if bounds_low is None else np.asarray(bounds_low, dtype=float).reshape(-1)
    b_high = None if bounds_high is None else np.asarray(bounds_high, dtype=float).reshape(-1)

    for i in range(n_samples):
        try:
            th = rng.multivariate_normal(theta_arr, cov_reg, check_valid="ignore")
        except Exception:
            th = theta_arr.copy()

        if (
            b_low is not None
            and b_high is not None
            and b_low.size == th.size
            and b_high.size == th.size
        ):
            th = np.clip(th, b_low, b_high)

        try:
            y_i = np.asarray(model(th), dtype=float).reshape(-1)
        except Exception:
            y_i = np.full(n_points, np.nan, dtype=float)
        if y_i.size != n_points:
            y_i = np.full(n_points, np.nan, dtype=float)
        samples[i, :] = y_i

    if not np.any(np.isfinite(samples)):
        return np.full(n_points, np.nan, dtype=float)

    return np.nanstd(samples, axis=0, ddof=1)


def bootstrap_uncertainty_curve(
    fitter: Any,
    *,
    y_hat: np.ndarray,
    residuals: np.ndarray,
    y_obs: np.ndarray,
    raw_params: np.ndarray,
    n_iter: int = 100,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Estimate absolute uncertainty curve via residual bootstrap re-fitting.

    This function expects a fitter with ``solve_core``, ``_reconstruct``, and ``y``.
    """
    y_hat_arr = np.asarray(y_hat, dtype=float).reshape(-1)
    if n_iter < 2 or y_hat_arr.size == 0:
        return np.full(y_hat_arr.size, np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    resid = np.asarray(residuals, dtype=float).reshape(-1)
    if resid.size != y_hat_arr.size:
        return np.full(y_hat_arr.size, np.nan, dtype=float)

    curves: list[np.ndarray] = []
    original_y = fitter.y
    try:
        for _ in range(n_iter):
            sampled_residuals = rng.choice(resid, size=resid.size, replace=True)
            fitter.y = y_hat_arr + sampled_residuals
            try:
                result = fitter.solve_core(init_params=np.asarray(raw_params, dtype=float))
            except Exception:
                continue
            if not getattr(result, "success", False):
                continue
            try:
                curve = np.asarray(fitter._reconstruct(result.x)[0], dtype=float).reshape(-1)
            except Exception:
                continue
            if curve.size == y_hat_arr.size:
                curves.append(curve)
    finally:
        fitter.y = y_obs if y_obs is not None else original_y

    if len(curves) < 2:
        return np.full(y_hat_arr.size, np.nan, dtype=float)

    stack = np.vstack(curves)
    return np.nanstd(stack, axis=0, ddof=1)


def build_uncertainty_report(
    metrics: Mapping[str, float | None],
    budget: Mapping[str, float],
    options: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable technical report payload."""
    rows = []
    for source in sorted(budget.keys()):
        value = budget[source]
        rows.append(
            {
                "source": source,
                "contribution_pct": float(value) if np.isfinite(value) else None,
            }
        )

    summary = {
        "uc_global": metrics.get("uc_global"),
        "uc_max": metrics.get("uc_max"),
        "uc_p95": metrics.get("uc_p95"),
        "FOM": metrics.get("FOM"),
        "R2": metrics.get("R2"),
    }
    return {
        "summary": summary,
        "table": rows,
        "quality": dict(quality or {}),
        "validation": dict(validation or {}),
        "options": dict(options),
    }
