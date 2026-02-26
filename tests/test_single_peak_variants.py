import pytest
import numpy as np
from tldecpy.fit.solvers import fit_single_peak
from tldecpy.models.registry import ALL_VARIANTS, get_model, get_order_from_key

VARIANTS = sorted(ALL_VARIANTS.keys())


def _case_setup(
    order: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    true_params = {"Im": 1000.0, "E": 1.2, "Tm": 420.0}
    init_guess = {"Im": 900.0, "E": 1.0, "Tm": 415.0}
    bounds = {
        "Im": (100.0, 2500.0),
        "E": (0.2, 3.0),
        "Tm": (340.0, 500.0),
    }

    if order == "go":
        true_params["b"] = 1.5
        init_guess["b"] = 1.3
        bounds["b"] = (1.001, 2.0)
    elif order == "otor":
        true_params["R"] = 0.5
        init_guess["R"] = 0.2
        bounds["R"] = (1e-6, 0.95)
    elif order == "mix":
        true_params["alpha"] = 0.35
        init_guess["alpha"] = 0.55
        bounds["alpha"] = (0.01, 0.99)

    return true_params, init_guess, bounds


def _relative_tolerance(param_name: str) -> float:
    if param_name == "b":
        return 0.12
    if param_name == "R":
        return 0.35
    if param_name == "alpha":
        return 0.25
    return 0.10


@pytest.mark.parametrize("model_key", VARIANTS)
def test_synthetic_recovery(model_key: str):
    """
    Generate synthetic data for each model family and check parameter recovery
    with model-specific tolerances.
    """
    T = np.linspace(300, 550, 100)
    order = get_order_from_key(model_key)
    true_params, init_guess, bounds = _case_setup(order)

    model_func = get_model(model_key)
    y_true = model_func(T, **true_params)

    np.random.seed(42)
    noise = np.random.normal(0, 20.0, size=T.shape)
    y_obs = np.maximum(y_true + noise, 0)

    res = fit_single_peak(T, y_obs, model=model_key, init=init_guess, bounds=bounds)

    assert res.converged, f"Model {model_key} failed to converge: {res.message}"
    assert res.metrics["R2"] > 0.95, f"Model {model_key} bad R2: {res.metrics['R2']}"

    for p_name, p_val in true_params.items():
        est_val = res.params[p_name]
        rel_err = abs(est_val - p_val) / max(abs(p_val), 1e-12)
        tol = _relative_tolerance(p_name)
        assert rel_err < tol, f"Param {p_name} in {model_key} off by {rel_err:.2%} (tol={tol:.0%})"


def test_backward_compatibility():
    """Ensure model='fo' defaults to 'fo_rq' behavior."""
    T = np.linspace(300, 500, 50)
    # Generate fo_rq data
    y = get_model("fo_rq")(T, 100, 1.0, 400)

    # Fit with generic 'fo'
    res = fit_single_peak(T, y, model="fo")

    assert res.converged
    assert res.model_type == "fo"
    # The fit should be nearly perfect since the default is fo_rq
    assert res.metrics["R2"] > 0.99


def test_invalid_model_key():
    """Ensure clear error for bad keys."""
    with pytest.raises(ValueError, match="not found"):
        fit_single_peak(np.array([300]), np.array([10]), model="invalid_v9")
