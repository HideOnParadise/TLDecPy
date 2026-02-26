import numpy as np
from tldecpy.fit.solvers import fit_single_peak
from tldecpy.models.otor_lw import otor_lw
from tldecpy.models.otor_wo import otor_wo


def test_otor_parity():
    """
    Verify that Lambert W and Wright Omega implementations yield
    nearly identical curves for valid parameters.
    """
    T = np.linspace(300, 600, 200)
    params = {"Im": 1000, "E": 1.2, "Tm": 450, "R": 0.1, "beta": 1.0}

    y_lw = otor_lw(T, **params)
    y_wo = otor_wo(T, **params)

    # Calculate relative RMS difference
    rms_diff = np.sqrt(np.mean((y_lw - y_wo) ** 2))
    max_val = np.max(y_lw)

    rel_error = rms_diff / max_val

    # Expect < 1% difference (numerical precision differences in special funcs)
    assert rel_error < 0.01, f"OTOR implementations diverge: {rel_error:.2%}"


def test_otor_fit_recovery():
    """Test fitting synthetic OTOR data."""
    T = np.linspace(300, 600, 100)
    # R=0.5 (Intermediate order)
    true_params = {"Im": 500.0, "E": 1.0, "Tm": 420.0, "R": 0.5}

    y_true = otor_lw(T, **true_params, beta=1.0)

    # Fit
    res = fit_single_peak(
        T, y_true, model="otor_lw", beta=1.0, init={"Im": 500, "E": 0.9, "Tm": 410, "R": 0.1}
    )

    assert res.converged
    assert res.metrics["R2"] > 0.99
    assert abs(res.params["E"] - 1.0) < 0.1  # 10% tol
    assert abs(res.params["R"] - 0.5) < 0.15  # R is harder to fit
