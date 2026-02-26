"""Simulation helpers for continuous trap-distribution TL models."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tldecpy.models.continuous import continuous_glow_peak
from tldecpy.schemas import SimulationResult
from tldecpy.simulate.noise import add_noise

FloatArray = NDArray[np.float64]
ContinuousModel = Literal[
    "continuous_gaussian",
    "continuous_exponential",
    "cont_gauss",
    "cont_exp",
    "cont_lorentz",
]


def _as_1d_temperature(T: ArrayLike) -> FloatArray:
    """Normalize and validate the temperature axis."""
    temperature = np.asarray(T, dtype=float)
    if temperature.ndim != 1:
        raise ValueError("T must be a 1D array.")
    return temperature


def simulate_continuous_curve(
    T: ArrayLike,
    *,
    model: ContinuousModel = "cont_gauss",
    Tn: float,
    In: float,
    E0: float,
    sigma: float,
    n_energy: int = 801,
    noise_config: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """
    Generate a synthetic TL curve for a continuous trap distribution.

    Parameters follow the geometric re-parametrization ``(Tn, In, E0, sigma)``
    introduced for fitting in Benavente et al. (2019), Eqs. 15-16.
    """
    temperature = _as_1d_temperature(T)

    intensity_clean = continuous_glow_peak(
        temperature,
        Tn=Tn,
        In=In,
        E0=E0,
        sigma=sigma,
        model=model,
        n_energy=n_energy,
    )

    intensity = np.asarray(intensity_clean, dtype=float)
    states: dict[str, FloatArray] | None = None

    if noise_config:
        mode_str = str(noise_config.get("mode", "gaussian"))
        mode: Literal["gaussian", "poisson"] = "poisson" if mode_str == "poisson" else "gaussian"
        sigma_noise = float(noise_config.get("sigma", 0.0))
        seed_value = noise_config.get("seed")
        seed = int(seed_value) if seed_value is not None else None
        intensity = np.asarray(
            add_noise(intensity_clean, mode=mode, sigma=sigma_noise, seed=seed),
            dtype=float,
        )
        states = {"I_clean": intensity_clean}

    return SimulationResult(T=temperature, I=intensity, states=states)


__all__ = ["simulate_continuous_curve"]
