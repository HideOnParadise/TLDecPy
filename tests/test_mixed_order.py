import numpy as np
from tldecpy.models.mixed import mo_kitis, mo_quad, mo_vej
from tldecpy.fit.multi import fit_multi
from tldecpy.schemas import PeakSpec


def test_mixed_models_parity():
    """
    Test that all 3 mixed models produce similar shapes for valid params.
    """
    T = np.linspace(300, 500, 100)
    params = {"Im": 1000, "E": 1.2, "Tm": 400, "alpha": 0.5}

    y1 = mo_kitis(T, **params)
    y2 = mo_quad(T, **params)
    y3 = mo_vej(T, **params)

    # Calculate Peak Areas
    a1, a2, a3 = np.sum(y1), np.sum(y2), np.sum(y3)

    # They are different approximations, so exact match isn't expected,
    # but they should be within ~10% area of each other
    assert abs(a1 - a2) / a1 < 0.15
    assert abs(a1 - a3) / a1 < 0.15

    # Peak position should be conserved at Tm
    assert np.argmax(y1) == np.argmax(y2)


def test_mixed_fit_recovery():
    """Fit a synthetic mixed curve and recover alpha."""
    T = np.linspace(300, 600, 200)

    # True: Alpha=0.3 (closer to 1st order)
    true_params = {"Im": 5000, "E": 1.0, "Tm": 450, "alpha": 0.3}
    y_obs = mo_kitis(T, **true_params)

    # Add noise
    np.random.seed(42)
    y_obs += np.random.normal(0, 20, size=T.shape)

    spec = PeakSpec(
        model="mo_kitis",
        init={"Tm": 445, "E": 0.9, "alpha": 0.5},  # Bad init
        bounds={"alpha": (0.01, 0.99)},
    )

    res = fit_multi(T, y_obs, [spec])

    assert res.converged
    assert res.metrics.R2 > 0.99

    p = res.peaks[0].params
    assert abs(p["E"] - 1.0) < 0.05
    assert abs(p["alpha"] - 0.3) < 0.05  # Alpha is sensitive but recoverable


def test_extreme_alpha_stability():
    """Test numerical stability near alpha=0 and alpha=1."""
    T = np.linspace(300, 500, 50)

    # Alpha very small
    y_small = mo_quad(T, 100, 1.0, 400, alpha=0.001)
    assert not np.any(np.isnan(y_small))
    assert np.max(y_small) > 0

    # Alpha very large
    y_large = mo_quad(T, 100, 1.0, 400, alpha=0.999)
    assert not np.any(np.isnan(y_large))
    assert np.max(y_large) > 0
