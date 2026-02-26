"""Compatibility tests for local optimizer and robust loss combinations."""

from __future__ import annotations

import numpy as np
import pytest

from tldecpy.fit.multi import fit_multi
from tldecpy.models.fo import fo_rq
from tldecpy.schemas import FitOptions, PeakSpec, RobustOptions


@pytest.mark.parametrize("optimizer", ["trf", "dogbox"])
@pytest.mark.parametrize("loss", ["linear", "soft_l1", "huber", "cauchy", "arctan", "tukey"])
def test_local_optimizer_loss_combinations_converge(optimizer: str, loss: str) -> None:
    """Ensure supported optimizer/loss combinations converge on a simple FO peak."""
    t = np.linspace(320.0, 540.0, 181)
    y_true = fo_rq(t, Im=9000.0, E=1.24, Tm=492.0)

    spec = PeakSpec(
        name="P1",
        model="fo_rq",
        init={"Tm": 480.0, "Im": 7000.0, "E": 1.0},
        bounds={"Tm": (440.0, 520.0), "Im": (0.0, 20000.0), "E": (0.5, 2.0)},
    )
    robust = RobustOptions(loss=loss, f_scale=5.0)
    options = FitOptions(local_optimizer=optimizer, max_nfev=600)

    result = fit_multi(t, y_true, peaks=[spec], robust=robust, options=options)
    assert result.converged, result.message
    assert result.metrics.R2 > 0.99


def test_lm_with_linear_loss_converges() -> None:
    """LM should converge when used with linear loss."""
    t = np.linspace(320.0, 540.0, 181)
    y_true = fo_rq(t, Im=6000.0, E=1.15, Tm=480.0)
    spec = PeakSpec(
        name="P1",
        model="fo_rq",
        init={"Tm": 470.0, "Im": 5000.0, "E": 1.0},
        bounds={"Tm": (430.0, 520.0), "Im": (0.0, 20000.0), "E": (0.5, 2.0)},
    )
    options = FitOptions(local_optimizer="lm", max_nfev=800)
    result = fit_multi(
        t,
        y_true,
        peaks=[spec],
        robust=RobustOptions(loss="linear"),
        options=options,
    )
    assert result.converged, result.message
    assert result.metrics.R2 > 0.99


def test_lm_rejects_non_linear_loss() -> None:
    """LM must reject robust nonlinear losses."""
    t = np.linspace(320.0, 540.0, 181)
    y_true = fo_rq(t, Im=6000.0, E=1.15, Tm=480.0)
    spec = PeakSpec(name="P1", model="fo_rq", init={"Tm": 470.0, "Im": 5000.0, "E": 1.0})
    options = FitOptions(local_optimizer="lm", max_nfev=200)
    with pytest.raises(ValueError, match="Incompatible optimizer/loss combination"):
        fit_multi(
            t,
            y_true,
            peaks=[spec],
            robust=RobustOptions(loss="huber"),
            options=options,
        )


def test_global_hybrid_pso_converges() -> None:
    """PSO-seeded hybrid strategy should converge on a simple two-peak case."""
    t = np.linspace(300.0, 600.0, 301)
    y = fo_rq(t, Im=5200.0, E=1.03, Tm=392.0) + fo_rq(t, Im=4300.0, E=1.18, Tm=508.0)

    specs = [
        PeakSpec(
            name="P1",
            model="fo_rq",
            init={"Tm": 360.0, "Im": 3000.0, "E": 0.8},
            bounds={"Tm": (320.0, 460.0), "Im": (0.0, 20000.0), "E": (0.5, 2.0)},
        ),
        PeakSpec(
            name="P2",
            model="fo_rq",
            init={"Tm": 545.0, "Im": 3500.0, "E": 0.9},
            bounds={"Tm": (430.0, 580.0), "Im": (0.0, 20000.0), "E": (0.5, 2.0)},
        ),
    ]

    result = fit_multi(
        t,
        y,
        peaks=specs,
        robust=RobustOptions(loss="linear"),
        options=FitOptions(local_optimizer="trf", max_nfev=1000),
        strategy="global_hybrid_pso",
    )
    assert result.converged, result.message
    assert result.metrics.R2 > 0.98
