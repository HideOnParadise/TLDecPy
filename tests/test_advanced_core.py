"""Regression tests for advanced-core v1.2.0 algorithms."""

from __future__ import annotations

import numpy as np

from tldecpy.fit.automator import iterative_deconvolution
from tldecpy.fit.detection import detect_peaks_cwt
from tldecpy.fit.multi import fit_multi
from tldecpy.models.fo import fo_rq
from tldecpy.schemas import PeakSpec


def _synthetic_two_peak_curve() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    t = np.linspace(300.0, 600.0, 301)
    y = (
        fo_rq(t, Im=6000.0, E=1.00, Tm=395.0)
        + fo_rq(t, Im=5000.0, E=1.15, Tm=505.0)
        + 250.0
        + 0.35 * (t - t.min())
    )
    y_noisy = np.maximum(y + rng.normal(0.0, 40.0, size=t.shape), 0.0)
    return t, y_noisy


def test_detect_peaks_cwt_finds_main_peaks() -> None:
    t, y = _synthetic_two_peak_curve()
    seeds = detect_peaks_cwt(t, y, min_snr=1.0)
    assert len(seeds) >= 2
    centers = np.array([seed.Tm for seed in seeds], dtype=float)
    assert float(np.min(np.abs(centers - 395.0))) < 18.0
    assert float(np.min(np.abs(centers - 505.0))) < 18.0


def test_fit_multi_global_hybrid_converges() -> None:
    t, y = _synthetic_two_peak_curve()
    specs = [
        PeakSpec(
            name="P1",
            model="fo_rq",
            init={"Tm": 360.0, "Im": 4000.0, "E": 0.8},
            bounds={"Tm": (320.0, 460.0), "Im": (0.0, 20000.0), "E": (0.5, 2.0)},
        ),
        PeakSpec(
            name="P2",
            model="fo_rq",
            init={"Tm": 545.0, "Im": 4500.0, "E": 0.9},
            bounds={"Tm": (430.0, 580.0), "Im": (0.0, 20000.0), "E": (0.5, 2.0)},
        ),
    ]

    result = fit_multi(t, y, peaks=specs, bg=None, strategy="global_hybrid")
    assert result.converged, result.message
    assert result.metrics.R2 > 0.97


def test_iterative_deconvolution_adds_components() -> None:
    t, y = _synthetic_two_peak_curve()
    result = iterative_deconvolution(
        t,
        y,
        max_peaks=4,
        allow_models=("fo",),
        bg_mode="none",
        strategy="local",
        residual_sigma_threshold=3.0,
    )
    assert result.converged, result.message
    assert len(result.peaks) >= 2
    assert result.metrics.R2 > 0.97
