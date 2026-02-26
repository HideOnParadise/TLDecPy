"""Confidence interval utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def bootstrap_ci(
    fitter,
    *,
    y_hat: np.ndarray,
    residuals: np.ndarray,
    y_obs: np.ndarray,
    raw_params: np.ndarray,
    n_iter: int = 100,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]] | None:
    """Estimate percentile bootstrap confidence intervals for fitted parameters."""
    if n_iter < 2:
        return None

    boot_samples: Dict[str, list[float]] = {name: [] for name in fitter.param_names_flat}
    original_y = fitter.y

    try:
        for _ in range(n_iter):
            sampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
            fitter.y = y_hat + sampled_residuals

            try:
                result = fitter.solve_core(init_params=raw_params)
            except Exception:
                continue

            if not result.success:
                continue

            for i, name in enumerate(fitter.param_names_flat):
                boot_samples[name].append(float(result.x[i]))
    finally:
        fitter.y = y_obs if y_obs is not None else original_y

    lower_p = alpha * 50.0
    upper_p = 100.0 - lower_p
    ci_results: Dict[str, Tuple[float, float]] = {}

    for name, samples in boot_samples.items():
        if len(samples) <= n_iter * 0.5:
            ci_results[name] = (np.nan, np.nan)
            continue

        arr = np.asarray(samples, dtype=float)
        ci_results[name] = (
            float(np.percentile(arr, lower_p)),
            float(np.percentile(arr, upper_p)),
        )

    return ci_results
