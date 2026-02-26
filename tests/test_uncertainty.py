from __future__ import annotations

import numpy as np

from tldecpy.fit.multi import fit_multi
from tldecpy.fit.uncertainty import (
    compute_parameter_covariance,
    compute_uc_curve,
    compute_uc_global,
)
from tldecpy.models.fo import fo_rq
from tldecpy.schemas import FitOptions, PeakSpec, UncertaintyOptions


def test_compute_parameter_covariance_smoke() -> None:
    jac = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    residuals = np.array([0.2, -0.1, 0.05], dtype=float)
    cov = compute_parameter_covariance(jac=jac, residuals=residuals, dof=1)
    assert cov is not None
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)


def test_compute_uc_curve_and_global_linear_model() -> None:
    x = np.linspace(0.0, 10.0, 51)

    def model(theta: np.ndarray) -> np.ndarray:
        return theta[0] + theta[1] * x

    theta = np.array([2.0, 1.5], dtype=float)
    y_fit = model(theta)
    cov = np.array([[0.04, 0.0], [0.0, 0.01]], dtype=float)

    uc_curve = compute_uc_curve(
        x=x,
        y_fit=y_fit,
        theta=theta,
        cov_theta=cov,
        model=model,
    )
    uc_global = compute_uc_global(x=x, y_fit=y_fit, uc_curve=uc_curve)

    assert uc_curve.shape == x.shape
    assert np.all(np.isfinite(uc_curve))
    assert float(np.max(uc_curve)) > 0.0
    assert np.isfinite(uc_global)
    assert uc_global > 0.0


def test_fit_multi_reports_uncertainty_metrics() -> None:
    rng = np.random.default_rng(1234)
    t = np.linspace(320.0, 560.0, 180)
    y_clean = fo_rq(t, Im=3000.0, E=1.10, Tm=430.0)
    y_obs = np.maximum(y_clean + rng.normal(0.0, 8.0, size=t.shape), 0.0)

    peaks = [
        PeakSpec(
            name="P1",
            model="fo_rq",
            init={"Im": 2500.0, "E": 1.0, "Tm": 425.0},
            bounds={"Im": (0.0, 6000.0), "E": (0.5, 2.0), "Tm": (380.0, 470.0)},
        )
    ]
    result = fit_multi(t, y_obs, peaks=peaks, bg=None)

    assert result.converged, result.message
    assert result.uc_curve is not None
    assert result.uc_curve.shape == t.shape
    assert np.any(np.isfinite(result.uc_curve))
    assert np.isfinite(result.metrics.uc_global)
    assert np.isfinite(result.metrics.uc_p95)
    assert np.isfinite(result.metrics.uc_max)


def test_fit_multi_uncertainty_budget_and_report() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(320.0, 560.0, 200)
    y_clean = fo_rq(t, Im=2500.0, E=1.05, Tm=420.0)
    y_obs = np.maximum(y_clean + rng.normal(0.0, 10.0, size=t.shape), 0.0)

    peaks = [
        PeakSpec(
            name="P1",
            model="fo_rq",
            init={"Im": 2400.0, "E": 1.0, "Tm": 418.0},
            bounds={"Im": (0.0, 6000.0), "E": (0.5, 2.0), "Tm": (370.0, 470.0)},
        )
    ]
    options = FitOptions(
        uncertainty=UncertaintyOptions(
            noise_pct=0.5,
            calibration_pct=0.7,
            heating_rate_pct=0.3,
            reader_drift_pct=0.2,
            uc_metric_min_rel_intensity=0.08,
            uc_metric_noise_sigma_factor=4.0,
            correlations={"noise:calibration": 0.2},
            export_report=True,
        )
    )
    result = fit_multi(t, y_obs, peaks=peaks, bg=None, options=options)

    assert result.converged, result.message
    assert result.metrics.contrib_E is not None
    assert result.metrics.contrib_Tm is not None
    assert result.metrics.contrib_Im is not None
    assert result.metrics.contrib_bg is not None
    assert result.metrics.contrib_E >= 0.0
    assert result.metrics.contrib_Tm >= 0.0
    assert result.metrics.contrib_Im >= 0.0
    assert result.metrics.contrib_bg >= 0.0

    assert "contrib_E" in result.uncertainty_budget
    assert "contrib_Tm" in result.uncertainty_budget
    assert "contrib_Im" in result.uncertainty_budget
    assert "contrib_bg" in result.uncertainty_budget
    assert "contrib_noise" in result.uncertainty_budget
    assert "contrib_calibration" in result.uncertainty_budget
    assert "contrib_heating_rate" in result.uncertainty_budget
    assert "contrib_reader_drift" in result.uncertainty_budget

    assert result.uncertainty_report is not None
    assert "summary" in result.uncertainty_report
    assert "table" in result.uncertainty_report
    assert isinstance(result.uncertainty_report["table"], list)
    assert result.uncertainty_report["options"]["uc_metric_min_rel_intensity"] == 0.08
    assert result.uncertainty_report["options"]["uc_metric_noise_sigma_factor"] == 4.0
