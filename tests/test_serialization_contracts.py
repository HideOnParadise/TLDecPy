import numpy as np
import json
from tldecpy.schemas import MultiFitResult, Metrics, PeakResult


def test_multifitresult_json_serialization():
    """Test that results can be serialized to JSON (contract)."""

    # Mock data
    metrics = Metrics(FOM=1.5, R2=0.99, RCS=1.2, SSR=100.0, AIC=50.0, BIC=55.0)
    peak = PeakResult(
        name="P1", model="fo_rq", params={"E": 1.0}, y_hat=np.array([0.1, 0.5, 0.1]), area=100.0
    )

    res = MultiFitResult(
        peaks=[peak],
        y_obs=np.array([0.1, 0.5, 0.1]),
        y_hat_total=np.array([0.1, 0.5, 0.1]),
        residuals=np.zeros(3),
        metrics=metrics,
        converged=True,
        message="OK",
        n_iter=10,
        hit_bounds={"E": False},
    )

    # Serialize
    json_str = res.model_dump_json()

    # Validate JSON structure
    data = json.loads(json_str)
    assert data["metrics"]["FOM"] == 1.5
    assert data["peaks"][0]["model"] == "fo_rq"
    assert isinstance(data["y_hat_total"], list)  # Numpy converted to list

    # Round-trip (Deserialization)
    res_back = MultiFitResult.model_validate_json(json_str)
    assert res_back.metrics.R2 == 0.99
    assert isinstance(res_back.y_hat_total, np.ndarray)  # Back to numpy if handled or list
    # Note: standard pydantic loads lists. If numpy needed, validator required.
    # For SDKs, list is preferred transport format.
