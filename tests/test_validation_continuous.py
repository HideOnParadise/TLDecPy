"""End-to-end validation tests for continuous trap-distribution models."""

from __future__ import annotations

import numpy as np
import pytest

from tldecpy.fit.init import autoinit_multi
from tldecpy.fit.multi import fit_multi
from tldecpy.simulate.continuous import simulate_continuous_curve

TRUE_PARAMS = {
    "Tn": 500.0,
    "In": 8000.0,
    "E0": 1.0,
    "sigma": 0.05,
}


@pytest.mark.parametrize("model_key", ["cont_gauss", "cont_exp"])
def test_validation_continuous_end_to_end_recovery(model_key: str) -> None:
    """
    Validate that a synthetic continuous curve can recover original parameters.

    Workflow:
    1. Simulate using known geometric parameters.
    2. Auto-initialize with the continuous model enabled.
    3. Fit with ``fit_multi`` and verify 1-2% recovery.
    """
    T = np.linspace(350.0, 650.0, 301)
    sim = simulate_continuous_curve(T, model=model_key, **TRUE_PARAMS)

    peaks, bg = autoinit_multi(
        sim.T,
        sim.I,
        max_peaks=1,
        allow_models=(model_key,),
        bg_mode="none",
        sensitivity=1.2,
    )

    assert bg is None
    assert len(peaks) == 1
    assert peaks[0].model == model_key

    result = fit_multi(sim.T, sim.I, peaks=peaks, bg=None)

    assert result.converged, result.message
    assert result.metrics.R2 > 0.995

    fitted = result.peaks[0].params
    for name, true_value in TRUE_PARAMS.items():
        fitted_value = float(fitted[name])
        rel_err = abs(fitted_value - true_value) / true_value
        assert rel_err <= 0.02, f"{model_key} {name} relative error {rel_err:.2%}"
