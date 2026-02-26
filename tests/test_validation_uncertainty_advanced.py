"""Advanced uncertainty-validation regression tests (v1.3.0)."""

from __future__ import annotations

import numpy as np

from tldecpy.fit.multi import fit_multi
from tldecpy.models.fo import fo_rq
from tldecpy.schemas import FitOptions, PeakSpec, UncertaintyOptions


def _fit_single_peak(
    *,
    n_points: int,
    noise_std: float,
    seed: int,
    uncertainty: UncertaintyOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    t = np.linspace(320.0, 560.0, n_points)
    y_clean = fo_rq(t, Im=3000.0, E=1.10, Tm=430.0)
    rng = np.random.default_rng(seed)
    y_obs = np.maximum(y_clean + rng.normal(0.0, noise_std, size=t.shape), 0.0)
    peaks = [
        PeakSpec(
            name="P1",
            model="fo_rq",
            init={"Im": 2600.0, "E": 1.0, "Tm": 426.0},
            bounds={"Im": (0.0, 6000.0), "E": (0.5, 2.0), "Tm": (380.0, 470.0)},
        )
    ]
    result = fit_multi(
        t,
        y_obs,
        peaks=peaks,
        options=FitOptions(uncertainty=uncertainty or UncertaintyOptions()),
    )
    assert result.converged, result.message
    return t, y_obs, result


def test_monte_carlo_uncertainty_validation_present() -> None:
    _, _, result = _fit_single_peak(
        n_points=220,
        noise_std=8.0,
        seed=11,
        uncertainty=UncertaintyOptions(
            validation_mode="monte_carlo",
            n_validation_samples=80,
            validation_seed=7,
            export_report=True,
        ),
    )
    assert result.uncertainty_validation is not None
    assert result.uncertainty_validation["mode"] == "monte_carlo"
    assert np.isfinite(float(result.uncertainty_validation["rel_l2"]))
    assert np.isfinite(float(result.uncertainty_validation["rel_max"]))
    assert float(result.uncertainty_validation["rel_l2"]) < 0.8


def test_channel_size_effect_uc_decreases_with_more_channels() -> None:
    _, _, coarse = _fit_single_peak(
        n_points=120,
        noise_std=10.0,
        seed=21,
        uncertainty=UncertaintyOptions(),
    )
    _, _, fine = _fit_single_peak(
        n_points=600,
        noise_std=10.0,
        seed=21,
        uncertainty=UncertaintyOptions(),
    )

    assert coarse.metrics.uc_global is not None
    assert fine.metrics.uc_global is not None
    assert fine.metrics.uc_global < coarse.metrics.uc_global


def test_scatter_effect_increases_fom_and_uc() -> None:
    _, _, low_scatter = _fit_single_peak(
        n_points=260,
        noise_std=4.0,
        seed=33,
        uncertainty=UncertaintyOptions(),
    )
    _, _, high_scatter = _fit_single_peak(
        n_points=260,
        noise_std=20.0,
        seed=33,
        uncertainty=UncertaintyOptions(),
    )

    assert high_scatter.metrics.FOM > low_scatter.metrics.FOM
    assert high_scatter.metrics.uc_global is not None
    assert low_scatter.metrics.uc_global is not None
    assert high_scatter.metrics.uc_global > low_scatter.metrics.uc_global


def test_overlap_effect_increases_uncertainty() -> None:
    t = np.linspace(320.0, 560.0, 260)
    rng = np.random.default_rng(101)

    y_sep = (
        fo_rq(t, Im=2600.0, E=1.05, Tm=390.0)
        + fo_rq(t, Im=2400.0, E=1.15, Tm=500.0)
        + rng.normal(0.0, 8.0, size=t.shape)
    )
    y_ov = (
        fo_rq(t, Im=2600.0, E=1.05, Tm=420.0)
        + fo_rq(t, Im=2400.0, E=1.15, Tm=445.0)
        + rng.normal(0.0, 8.0, size=t.shape)
    )

    peaks_sep = [
        PeakSpec(name="P1", model="fo_rq", init={"Im": 2200.0, "E": 1.0, "Tm": 386.0}),
        PeakSpec(name="P2", model="fo_rq", init={"Im": 2200.0, "E": 1.1, "Tm": 504.0}),
    ]
    peaks_ov = [
        PeakSpec(name="P1", model="fo_rq", init={"Im": 2200.0, "E": 1.0, "Tm": 418.0}),
        PeakSpec(name="P2", model="fo_rq", init={"Im": 2200.0, "E": 1.1, "Tm": 447.0}),
    ]
    options = FitOptions(uncertainty=UncertaintyOptions())
    res_sep = fit_multi(t, np.maximum(y_sep, 0.0), peaks=peaks_sep, options=options)
    res_ov = fit_multi(t, np.maximum(y_ov, 0.0), peaks=peaks_ov, options=options)

    assert res_sep.converged and res_ov.converged
    c_sep = float(res_sep.metrics.contrib_E or 0.0) + float(res_sep.metrics.contrib_Tm or 0.0)
    c_sep += float(res_sep.metrics.contrib_Im or 0.0)
    c_ov = float(res_ov.metrics.contrib_E or 0.0) + float(res_ov.metrics.contrib_Tm or 0.0)
    c_ov += float(res_ov.metrics.contrib_Im or 0.0)
    assert c_ov > c_sep
