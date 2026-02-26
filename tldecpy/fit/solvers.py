"""Public solver helpers built on top of :func:`tldecpy.fit.multi.fit_multi`."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from tldecpy.fit.multi import fit_multi
from tldecpy.models.registry import get_order_from_key
from tldecpy.schemas import FitResult, PeakSpec, RobustOptions


def fit_single_peak(
    x: np.ndarray,
    y: np.ndarray,
    model: str = "fo",
    init: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    beta: float = 1.0,
    robust: Optional[RobustOptions] = None,
) -> FitResult:
    r"""
    Fit a single TL component using the multi-component engine.

    Parameters
    ----------
    x : numpy.ndarray
        Temperature grid :math:`T` in kelvin.
    y : numpy.ndarray
        Measured intensity :math:`I(T)`.
    model : str, default="fo"
        Model key/family accepted by :mod:`tldecpy.models.registry`.
    init : dict[str, float] | None, optional
        Initial parameter guesses (for example ``Tm``, ``Im``, ``E``).
    bounds : dict[str, tuple[float, float]] | None, optional
        Optional parameter bounds as ``(min, max)``.
    beta : float, default=1.0
        Heating rate :math:`\beta` in K/s.
    robust : RobustOptions | None, optional
        Robust fitting configuration.

    Returns
    -------
    FitResult
        Backward-compatible single-peak result schema.
    """
    order_type = get_order_from_key(model)
    init_data = dict(init) if init else {}
    bounds_data = dict(bounds) if bounds else None
    fixed_data: Optional[Dict[str, float]] = None

    if order_type == "continuous":
        # Accept legacy single-peak aliases for compatibility.
        if "In" not in init_data and "Im" in init_data:
            init_data["In"] = init_data["Im"]
        if "E0" not in init_data and "E" in init_data:
            init_data["E0"] = init_data["E"]
        if "Tn" not in init_data and "Tm" in init_data:
            init_data["Tn"] = init_data["Tm"]

        if bounds_data is not None:
            if "In" not in bounds_data and "Im" in bounds_data:
                bounds_data["In"] = bounds_data["Im"]
            if "E0" not in bounds_data and "E" in bounds_data:
                bounds_data["E0"] = bounds_data["E"]
            if "Tn" not in bounds_data and "Tm" in bounds_data:
                bounds_data["Tn"] = bounds_data["Tm"]

        # Legacy continuous calls (Im/E/Tm) do not specify sigma.
        # Keep the historical synthetic-variant behavior with sigma=0.05.
        has_sigma_bound = bounds_data is not None and "sigma" in bounds_data
        if "sigma" not in init_data and not has_sigma_bound:
            fixed_data = {"sigma": 0.05}

    spec = PeakSpec(
        model=model,
        init=init_data,
        bounds=bounds_data,
        fixed=fixed_data,
        name="Single Peak",
    )

    result_multi = fit_multi(x, y, peaks=[spec], bg=None, beta=beta, robust=robust)
    peak = result_multi.peaks[0]
    params = dict(peak.params)
    if order_type == "continuous":
        # Legacy aliases expected by existing single-peak tests/consumers.
        params.setdefault("Im", params.get("In", np.nan))
        params.setdefault("E", params.get("E0", np.nan))
        params.setdefault("Tm", params.get("Tn", np.nan))

    return FitResult(
        params=params,
        cov=None,
        metrics=result_multi.metrics.model_dump(),
        y_hat=result_multi.y_hat_total,
        residuals=result_multi.residuals,
        converged=result_multi.converged,
        message=result_multi.message,
        model_type=model,
    )
