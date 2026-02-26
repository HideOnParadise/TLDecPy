"""Hybrid synthetic validation for CGCD-style decomposition."""

from __future__ import annotations

import numpy as np

from tldecpy.fit.multi import fit_multi
from tldecpy.models.background import linear_bg
from tldecpy.models.registry import get_model
from tldecpy.schemas import BackgroundSpec, FitOptions, PeakSpec, RobustOptions


def _rel_err(value: float, target: float) -> float:
    return abs(value - target) / max(abs(target), 1e-12)


def test_hybrid_cgcd_synthetic_roundtrip() -> None:
    """
    Validate the full hybrid stack on data generated from the same model family.

    If this test passes, the continuous/discrete models and the optimizer can
    recover known parameters in a controlled setting.
    """
    T = np.linspace(298.15, 620.0, 316)

    p1_true = {"Tn": 385.0, "In": 7000.0, "E0": 0.812, "sigma": 0.0346}
    p2_true = {"Tn": 422.5, "In": 36000.0, "E0": 0.860, "sigma": 0.0400}
    p3_true = {"Tm": 515.0, "Im": 25000.0, "E": 1.120}
    p4_true = {"Tm": 569.0, "Im": 22000.0, "E": 1.068}
    bg_true = {"a": -2200.0, "b": 8.4}

    y = (
        get_model("cont_gauss")(T, **p1_true)
        + get_model("cont_gauss")(T, **p2_true)
        + get_model("fo_rq")(T, **p3_true)
        + get_model("fo_rq")(T, **p4_true)
        + linear_bg(T, **bg_true)
    )

    peaks = [
        PeakSpec(
            name="P1",
            model="cont_gauss",
            init={"Tn": 378.0, "In": 6500.0, "E0": 0.80, "sigma": 0.038},
            bounds={
                "Tn": (360.0, 402.0),
                "In": (100.0, 20000.0),
                "E0": (0.70, 0.95),
                "sigma": (0.02, 0.06),
            },
        ),
        PeakSpec(
            name="P2",
            model="cont_gauss",
            init={"Tn": 425.0, "In": 34000.0, "E0": 0.88, "sigma": 0.045},
            bounds={
                "Tn": (405.0, 445.0),
                "In": (5000.0, 60000.0),
                "E0": (0.74, 1.00),
                "sigma": (0.025, 0.065),
            },
        ),
        PeakSpec(
            name="P3",
            model="fo_rq",
            init={"Tm": 512.0, "Im": 24000.0, "E": 1.10},
            bounds={"Tm": (500.0, 530.0), "Im": (1000.0, 50000.0), "E": (0.95, 1.35)},
        ),
        PeakSpec(
            name="P4",
            model="fo_rq",
            init={"Tm": 566.0, "Im": 21000.0, "E": 1.10},
            bounds={"Tm": (555.0, 585.0), "Im": (1000.0, 50000.0), "E": (0.88, 1.30)},
        ),
    ]

    bg = BackgroundSpec(
        type="linear",
        init={"a": -2000.0, "b": 8.0},
        bounds={"a": (-5000.0, 2000.0), "b": (0.0, 15.0)},
    )

    result = fit_multi(
        T,
        y,
        peaks=peaks,
        bg=bg,
        robust=RobustOptions(loss="linear"),
        options=FitOptions(max_nfev=15000, ftol=1e-10, xtol=1e-10, gtol=1e-10),
    )

    assert result.converged, result.message
    assert result.metrics.R2 > 0.9999
    assert result.metrics.FOM < 0.1

    got = {peak.name: peak.params for peak in result.peaks}

    assert _rel_err(float(got["P1"]["E0"]), p1_true["E0"]) < 0.02
    assert _rel_err(float(got["P1"]["sigma"]), p1_true["sigma"]) < 0.03
    assert _rel_err(float(got["P1"]["Tn"]), p1_true["Tn"]) < 0.02

    assert _rel_err(float(got["P2"]["E0"]), p2_true["E0"]) < 0.02
    assert _rel_err(float(got["P2"]["sigma"]), p2_true["sigma"]) < 0.03
    assert _rel_err(float(got["P2"]["Tn"]), p2_true["Tn"]) < 0.02

    assert _rel_err(float(got["P3"]["E"]), p3_true["E"]) < 0.03
    assert _rel_err(float(got["P3"]["Tm"]), p3_true["Tm"]) < 0.02
    assert _rel_err(float(got["P4"]["E"]), p4_true["E"]) < 0.03
    assert _rel_err(float(got["P4"]["Tm"]), p4_true["Tm"]) < 0.02

    assert result.background is not None
    bg_fit = result.background.params
    assert _rel_err(float(bg_fit["a"]), bg_true["a"]) < 0.10
    assert _rel_err(float(bg_fit["b"]), bg_true["b"]) < 0.05
