"""Compatibility auto-fit utilities.

The original iterative residual algorithm was removed from active use.
This module keeps a backward-compatible entry point that performs a
single automatic detection + multi-peak fit pass.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from tldecpy.fit.bounds import make_bounds_from_init
from tldecpy.fit.detection import detect_peaks_cwt
from tldecpy.fit.init import (
    PeakSeed,
    _select_model,
    estimate_energy_chen,
    pick_peaks,
    preprocess,
)
from tldecpy.fit.multi import fit_multi
from tldecpy.schemas import BackgroundSpec, FitOptions, MultiFitResult, PeakSpec, RobustOptions


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite 1D float array for fitting utilities."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size < 10:
        raise ValueError(f"{name} must contain at least 10 points.")
    return arr


def _seed_from_index(x: np.ndarray, y: np.ndarray, idx: int) -> PeakSeed:
    """Build a fallback peak seed centered at a sample index."""
    i = int(np.clip(idx, 0, x.size - 1))
    if x.size < 3:
        mean_step = float(np.mean(np.diff(x)))
        fwhm = max(mean_step, 1.0)
        return PeakSeed(index=i, Tm=float(x[i]), Im=float(y[i]), fwhm=fwhm, symmetry=0.5)

    local = max(1, min(i, x.size - 2))
    span = abs(float(x[min(local + 1, x.size - 1)] - x[max(local - 1, 0)]))
    fwhm = max(span * 3.0, np.finfo(float).eps)
    return PeakSeed(index=i, Tm=float(x[i]), Im=float(y[i]), fwhm=fwhm, symmetry=0.5)


def _seed_to_spec(seed: PeakSeed, idx: int, allow_set: set[str], sample_spacing: float) -> PeakSpec:
    """Map a detected seed to a concrete ``PeakSpec`` with model-aware defaults."""
    model, shape_hint = _select_model(seed.symmetry, allow_set)
    if shape_hint == "continuous":
        init_params: dict[str, float] = {"Tn": seed.Tm, "In": seed.Im, "E0": 1.0, "sigma": 0.05}
    else:
        safe_fwhm = max(float(seed.fwhm), np.finfo(float).eps)
        # Chen-type estimates become unstable for very narrow/noisy widths.
        if safe_fwhm < 6.0 * sample_spacing:
            energy_est = 1.1
        else:
            energy_est = float(
                np.clip(estimate_energy_chen(seed.Tm, safe_fwhm, shape_hint), 0.2, 2.0)
            )
        init_params = {"Tm": seed.Tm, "Im": seed.Im, "E": energy_est}
        if shape_hint == "go":
            init_params["b"] = 1.5
        if shape_hint == "otor":
            init_params["R"] = 1e-3
        if shape_hint == "mix":
            init_params["alpha"] = 0.5

    bounds = make_bounds_from_init(init_params, model)
    payload = {"name": f"P{idx + 1}", "model": model, "init": init_params, "bounds": bounds}
    return PeakSpec.model_validate(payload)


def _infer_background(x: np.ndarray, y_clean: np.ndarray, bg_mode: str) -> BackgroundSpec | None:
    """Infer a background specification from cleaned signal tails."""
    if bg_mode == "none":
        return None
    if bg_mode in {"linear", "exponential"}:
        return BackgroundSpec.model_validate({"type": bg_mode})
    # auto
    noise_floor = float(np.percentile(y_clean, 5))
    if y_clean[0] > noise_floor * 2.0 or y_clean[-1] > noise_floor * 2.0:
        return BackgroundSpec.model_validate(
            {"type": "exponential", "init": {"a": 0.0, "b": 1.0, "c": 100.0}}
        )
    return None


def _select_distinct_seeds(seeds: list[PeakSeed], max_count: int, min_sep: float) -> list[PeakSeed]:
    """Keep up to ``max_count`` seeds while enforcing a minimum thermal separation."""
    chosen: list[PeakSeed] = []
    for seed in sorted(seeds, key=lambda candidate: candidate.Im, reverse=True):
        if len(chosen) >= max_count:
            break
        if all(abs(seed.Tm - kept.Tm) >= min_sep for kept in chosen):
            chosen.append(seed)
    return sorted(chosen, key=lambda seed: seed.index)


def iterative_deconvolution(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_peaks: int = 6,
    allow_models: tuple[str, ...] = ("fo", "go", "otor_lw"),
    bg_mode: Literal["linear", "exponential", "none", "auto"] = "auto",
    sensitivity: float = 1.0,
    min_snr: float = 1.0,
    widths: np.ndarray | None = None,
    residual_sigma_threshold: float = 3.0,
    beta: float = 1.0,
    robust: RobustOptions | None = None,
    options: FitOptions | None = None,
    strategy: Literal["local", "global_hybrid", "global_hybrid_pso"] = "local",
) -> MultiFitResult:
    r"""
    Run automatic TL deconvolution with heuristic seeding and one-shot fitting.

    Parameters
    ----------
    x : numpy.ndarray
        Temperature grid :math:`T` in kelvin.
    y : numpy.ndarray
        Measured glow-curve intensity :math:`I(T)`.
    max_peaks : int, default=6
        Maximum number of components used in the automatic model.
    allow_models : tuple[str, ...], default=("fo", "go", "otor_lw")
        Allowed model families/aliases for automatic seed-to-model mapping.
    bg_mode : {"linear", "exponential", "none", "auto"}, default="auto"
        Background inference mode.
    sensitivity : float, default=1.0
        Sensitivity used by fallback local-peak detector.
    min_snr : float, default=1.0
        Minimum SNR used by CWT-based detector.
    widths : numpy.ndarray | None, optional
        Optional CWT width grid in sample units.
    residual_sigma_threshold : float, default=3.0
        Kept for backward compatibility with historical iterative API.
    beta : float, default=1.0
        Heating rate :math:`\beta` in K/s.
    robust : RobustOptions | None, optional
        Robust loss/weighting settings.
    options : FitOptions | None, optional
        Local optimizer and uncertainty options.
    strategy : {"local", "global_hybrid", "global_hybrid_pso"}, default="local"
        Optimization strategy used by the underlying fitter.

    Returns
    -------
    MultiFitResult
        Full multi-peak fit result including diagnostics and uncertainty payloads.
    """
    _ = residual_sigma_threshold
    x_arr = _as_1d(x, "x")
    y_arr = _as_1d(y, "y")
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length.")
    if max_peaks < 1:
        raise ValueError("max_peaks must be >= 1.")

    y_clean = preprocess(x_arr, y_arr)
    allow_set = {model.lower() for model in allow_models}
    sample_spacing = max(float(np.median(np.diff(x_arr))), np.finfo(float).eps)

    seeds = detect_peaks_cwt(x_arr, y_clean, min_snr=min_snr, widths=widths)
    if not seeds:
        seeds = pick_peaks(x_arr, y_clean, sensitivity=sensitivity)
    if not seeds:
        seeds = [_seed_from_index(x_arr, y_clean, int(np.argmax(y_clean)))]

    x_span = float(np.max(x_arr) - np.min(x_arr))
    min_sep = max(float(np.median(np.diff(x_arr))) * 5.0, x_span / 25.0)
    selected = _select_distinct_seeds(seeds, int(max_peaks), min_sep=min_sep)
    specs = [_seed_to_spec(seed, i, allow_set, sample_spacing) for i, seed in enumerate(selected)]
    background = _infer_background(x_arr, y_clean, bg_mode)

    return fit_multi(
        x_arr,
        y_arr,
        peaks=specs,
        bg=background,
        beta=beta,
        robust=robust,
        options=options,
        strategy=strategy,
    )
