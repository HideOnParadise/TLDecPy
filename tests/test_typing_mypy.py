"""
Type contract test for PEP 561 support.
This module is statically checked by mypy in CI.
"""

import numpy as np
from tldecpy import MultiFitResult, Metrics, PeakSpec, RobustOptions, fit_multi


def run_typed_workflow(T: np.ndarray, intensity: np.ndarray) -> Metrics:
    """
    Typed workflow used by mypy to verify exported package annotations.
    """

    specs: list[PeakSpec] = [PeakSpec(model="fo_rq", init={"Tm": 400.0, "E": 1.0, "Im": 1000.0})]

    opts: RobustOptions = RobustOptions(loss="soft_l1")

    result: MultiFitResult = fit_multi(T, intensity, specs, robust=opts)

    _fom: float = result.metrics.FOM
    _ = _fom
    return result.metrics
