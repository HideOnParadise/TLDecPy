import numpy as np
from tldecpy.fit.multi import fit_multi
from tldecpy.schemas import PeakSpec, RobustOptions
from tldecpy.models.fo import fo_rq


def test_outlier_rejection():
    """Compare Linear vs Soft_L1 loss in presence of spikes."""
    T = np.linspace(300, 500, 100)
    # True peak
    y_true = fo_rq(T, 1000, 1.2, 400)
    y_obs = y_true.copy()

    # Add 3 massive outliers
    y_obs[45] = 5000
    y_obs[50] = 0
    y_obs[55] = 5000

    spec = PeakSpec(model="fo_rq", init={"Tm": 395, "Im": 900, "E": 1.0})

    # 1. Linear Fit (Should be distorted)
    res_lin = fit_multi(T, y_obs, [spec], beta=1.0)

    # 2. Robust Fit (Should ignore outliers)
    opt = RobustOptions(loss="soft_l1", f_scale=10.0)
    res_rob = fit_multi(T, y_obs, [spec], beta=1.0, robust=opt)

    err_lin = abs(res_lin.peaks[0].params["Im"] - 1000)
    err_rob = abs(res_rob.peaks[0].params["Im"] - 1000)

    assert err_rob < err_lin, "Robust fit should be closer to truth than linear fit with outliers"
    assert err_rob < 50, "Robust fit failed to recover Im"


def test_weighted_fit():
    """Test Poisson weighting on heteroscedastic noise."""
    T = np.linspace(300, 500, 200)
    y_true = fo_rq(T, 10000, 1.2, 400)  # High count

    # Poisson noise: variance = y
    np.random.seed(42)
    y_obs = np.random.poisson(y_true).astype(float)

    spec = PeakSpec(model="fo_rq", init={"Tm": 395, "Im": 9000, "E": 1.0})

    # Weighted fit
    opt = RobustOptions(loss="linear", weights="poisson")
    res = fit_multi(T, y_obs, [spec], robust=opt)

    assert res.converged
    assert abs(res.peaks[0].params["E"] - 1.2) < 0.05


def test_hit_bounds_diagnostic():
    """Ensure hit_bounds detects stuck parameters."""
    T = np.linspace(300, 400, 50)
    y = fo_rq(T, 100, 1.0, 350)

    # Force E to hit upper bound
    spec = PeakSpec(model="fo_rq", init={"E": 1.45}, bounds={"E": (1.4, 1.5)})

    res = fit_multi(T, y, [spec])

    # Check diagnostics
    # Param name format depends on implementation, checking logic
    hits = [k for k, v in res.hit_bounds.items() if v]
    assert len(hits) > 0, "Should detect hit bounds"
    assert any("E" in k for k in hits)
