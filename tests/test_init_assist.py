import numpy as np
from tldecpy.fit.init import preprocess, pick_peaks, autoinit_multi
from tldecpy.models.fo import fo_rq
from tldecpy.models.go import go_kg


def test_preprocess_outliers():
    """Test outlier removal."""
    x = np.linspace(0, 10, 100)
    y = np.exp(-((x - 5) ** 2))
    y_noise = y.copy()
    y_noise[50] = 10.0  # Huge outlier

    y_clean = preprocess(x, y_noise, remove_outliers=True)
    assert y_clean[50] < 2.0  # Should be cleaned
    assert y_clean[50] > 0.5  # Should preserve signal roughly


def test_pick_peaks_shape():
    """Test FWHM and symmetry calculation on synthetic FO peak."""
    T = np.linspace(300, 500, 200)
    # FO peak
    y = fo_rq(T, 1000, 1.2, 400)

    seeds = pick_peaks(T, y, sensitivity=1.0)
    assert len(seeds) >= 1
    seed = min(seeds, key=lambda candidate: abs(candidate.Tm - 400.0))

    assert abs(seed.Tm - 400) < 3
    assert 0.35 < seed.symmetry < 0.55
    # Check width non-zero
    assert seed.fwhm > 10


def test_autoinit_generation():
    """Test that autoinit produces valid specs for fitting."""
    T = np.linspace(300, 600, 200)
    # Create mix: FO + GO
    y = fo_rq(T, 1000, 1.2, 380) + go_kg(T, 800, 1.5, 500, 1.5)

    specs, bg = autoinit_multi(T, y, max_peaks=3, allow_models=("fo", "go"))

    assert len(specs) >= 2
    p1 = min(specs, key=lambda spec: abs(spec.init["Tm"] - 380.0))
    assert abs(p1.init["Tm"] - 380) < 10
    assert 0.1 < p1.init["E"] < 5.0

    p2 = min(specs, key=lambda spec: abs(spec.init["Tm"] - 500.0))
    assert abs(p2.init["Tm"] - 500) < 10

    assert bg is None or bg.type in {"linear", "exponential", "none"}
