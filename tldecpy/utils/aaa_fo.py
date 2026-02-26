"""AAA rational backend for FO auxiliary function alpha(z)."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class AAAQConstants(NamedTuple):
    zmin: float
    zmax: float
    support_points: FloatArray
    support_values: FloatArray
    weights: FloatArray


@lru_cache(maxsize=1)
def load_aaa_Q_constants() -> AAAQConstants:
    """
    Load offline AAA constants for ``Q(z) = exp(z) * E2(z)``.

    The resource is expected at ``tldecpy/data/aaa_Q_z4p5_130.npz``.
    """
    resource = files("tldecpy.data").joinpath("aaa_Q_z4p5_130.npz")
    try:
        with resource.open("rb") as fh:
            with np.load(fh) as data:
                zmin = float(data["zmin"])
                zmax = float(data["zmax"])
                support_points = np.asarray(data["support_points"], dtype=np.float64)
                support_values = np.asarray(data["support_values"], dtype=np.float64)
                weights = np.asarray(data["weights"], dtype=np.float64)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing AAA resource 'tldecpy/data/aaa_Q_z4p5_130.npz'. "
            "Ensure package data includes '*.npz' when building/installing tldecpy."
        ) from exc

    if support_points.ndim != 1 or support_values.ndim != 1 or weights.ndim != 1:
        raise ValueError("AAA constants must be 1-D arrays.")
    if not (len(support_points) == len(support_values) == len(weights)):
        raise ValueError("AAA constants lengths are inconsistent.")
    if len(support_points) == 0:
        raise ValueError("AAA constants are empty.")

    return AAAQConstants(
        zmin=zmin,
        zmax=zmax,
        support_points=support_points,
        support_values=support_values,
        weights=weights,
    )


def Q_aaa(z: ArrayLike) -> FloatArray:
    """
    Evaluate the AAA rational approximation of ``Q(z)``.

    Uses barycentric evaluation:
    ``r(z) = sum(w_j f_j / (z - z_j)) / sum(w_j / (z - z_j))``.
    """
    constants = load_aaa_Q_constants()
    z_arr = np.asarray(z, dtype=np.float64)
    z_eval = np.clip(z_arr, constants.zmin, constants.zmax)

    z_flat = z_eval.reshape(-1)
    numerator = np.zeros_like(z_flat)
    denominator = np.zeros_like(z_flat)
    exact_idx = np.full(z_flat.shape, -1, dtype=np.int64)

    for j in range(constants.support_points.size):
        diff = z_flat - constants.support_points[j]
        is_exact = diff == 0.0
        if np.any(is_exact):
            exact_idx[is_exact] = j

        inv_diff = np.divide(1.0, diff, out=np.zeros_like(diff), where=~is_exact)
        term = constants.weights[j] * inv_diff
        numerator += term * constants.support_values[j]
        denominator += term

    out = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0.0,
    )

    matched = exact_idx >= 0
    if np.any(matched):
        out[matched] = constants.support_values[exact_idx[matched]]

    return out.reshape(z_arr.shape)


def alpha_aaa(z: ArrayLike) -> FloatArray:
    """Evaluate ``alpha(z) = 1 - Q(z)`` with the AAA backend."""
    return 1.0 - Q_aaa(z)
