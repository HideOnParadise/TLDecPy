"""
Diagnostics for fit stability.
"""

import numpy as np


def check_conditioning(jac: np.ndarray) -> float:
    """
    Estimate condition number of the Jacobian.
    Cond = sigma_max / sigma_min.
    """
    if jac is None or jac.size == 0:
        return 0.0

    try:
        # Singular Value Decomposition
        s = np.linalg.svd(jac, compute_uv=False)
        if s[-1] == 0:
            return np.inf
        return s[0] / s[-1]
    except Exception:
        return 0.0


def check_hit_bounds(params: np.ndarray, bounds: tuple, param_names: list) -> dict:
    """
    Identify parameters that are dangerously close to bounds.
    """
    lower, upper = bounds
    hit_dict = {}

    tol = 1e-5

    for i, p_val in enumerate(params):
        lb = lower[i]
        ub = upper[i]

        is_hit = False
        # Check lower
        if np.isfinite(lb) and abs(p_val - lb) < tol * max(1.0, abs(lb)):
            is_hit = True
        # Check upper
        if np.isfinite(ub) and abs(p_val - ub) < tol * max(1.0, abs(ub)):
            is_hit = True

        name = param_names[i] if i < len(param_names) else f"p{i}"
        hit_dict[name] = is_hit

    return hit_dict
