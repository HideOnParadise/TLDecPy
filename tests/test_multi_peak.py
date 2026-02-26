import numpy as np
from tldecpy.fit.multi import fit_multi
from tldecpy.fit.init import autoinit_multi
from tldecpy.schemas import PeakSpec, BackgroundSpec
from tldecpy.models.fo import fo_rq
from tldecpy.models.go import go_kg
from tldecpy.models.background import exponential_bg


def test_multi_peak_exactness():
    """Test recovery of 2 mixed peaks + exponential background."""
    T = np.linspace(300, 600, 200)

    # Synthetic Data Generation
    # Peak 1: First Order
    p1_true = {"Im": 1000, "E": 1.2, "Tm": 400}
    y1 = fo_rq(T, **p1_true)

    # Peak 2: General Order
    p2_true = {"Im": 800, "E": 1.5, "Tm": 500, "b": 1.5}
    y2 = go_kg(T, **p2_true)

    # Background: Exp
    bg_true = {"a": 10, "b": 50, "c": 100}
    y_bg = exponential_bg(T, **bg_true)

    y_total = y1 + y2 + y_bg

    # Add noise
    np.random.seed(42)
    y_obs = y_total + np.random.normal(0, 10, size=T.shape)

    # Specs
    peaks = [
        PeakSpec(model="fo_rq", init={"Tm": 390, "E": 1.0}, name="P1"),
        PeakSpec(model="go_kg", init={"Tm": 510, "E": 1.0, "b": 1.2}, name="P2"),
    ]
    bg = BackgroundSpec(type="exponential", init={"a": 0, "b": 10, "c": 50})

    res = fit_multi(T, y_obs, peaks, bg)

    assert res.converged
    # R2 should be healthy now that fo_rq and bg are stable
    assert res.metrics.R2 > 0.95

    p1_fit = res.peaks[0].params
    assert abs(p1_fit["E"] - 1.2) < 0.1
    assert abs(p1_fit["Tm"] - 400) < 5.0


def test_overlap_resolution():
    """Test two peaks very close to each other."""
    T = np.linspace(300, 500, 100)
    # Peaks at 400K and 420K
    y = fo_rq(T, 1000, 1.2, 400) + fo_rq(T, 1000, 1.2, 420)

    # Init close to truth but slightly off
    peaks = [
        PeakSpec(model="fo_rq", init={"Tm": 395}, bounds={"Tm": (380, 410)}),
        PeakSpec(model="fo_rq", init={"Tm": 425}, bounds={"Tm": (410, 440)}),
    ]

    res = fit_multi(T, y, peaks)

    assert res.converged
    tm1 = res.peaks[0].params["Tm"]
    tm2 = res.peaks[1].params["Tm"]
    # Allow 5K tolerance for overlap resolution
    assert abs(tm1 - 400) < 5
    assert abs(tm2 - 420) < 5


def test_autoinit_multi_detects_distinct_peaks():
    """Test that automatic initialization returns reasonable peak specs."""
    T = np.linspace(300, 600, 200)
    y = fo_rq(T, 1000, 1.2, 400) + fo_rq(T, 500, 1.2, 500)

    specs, bg = autoinit_multi(T, y, max_peaks=4, allow_models=("fo",), bg_mode="none")

    assert len(specs) >= 2
    assert all(spec.model.startswith("fo_") for spec in specs)
    assert bg is None

    tms = sorted(spec.init["Tm"] for spec in specs)
    assert any(abs(tm - 400.0) < 8.0 for tm in tms)
    assert any(abs(tm - 500.0) < 8.0 for tm in tms)
