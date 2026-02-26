"""Core ODE solver wrapper for thermoluminescence simulations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import warnings
from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_ivp

from tldecpy.schemas import SimulationResult
from tldecpy.simulate.noise import add_noise


def simulate(
    rhs_func: Callable[..., Sequence[float] | np.ndarray],
    params: tuple[float, ...],
    *,
    T0: float,
    T_end: float,
    beta: float,
    y0: list[float],
    state_keys: list[str],
    method: str = "LSODA",
    points: int = 1000,
    noise_config: dict[str, Any] | None = None,
) -> SimulationResult:
    r"""
    Integrate TL kinetic ODEs under linear heating and return typed simulation payload.

    Parameters
    ----------
    rhs_func : Callable
        ODE right-hand side callable with signature compatible with
        ``rhs_func(t, y, *params, beta=beta, T0=T0)``.
    params : tuple[float, ...]
        Model-specific kinetic parameters.
    T0 : float
        Initial temperature, in kelvin.
    T_end : float
        Final temperature, in kelvin.
    beta : float
        Heating rate :math:`\beta` in K/s.
    y0 : list[float]
        Initial state vector.
    state_keys : list[str]
        Names assigned to state trajectories in output.
    method : str, default="LSODA"
        Integration method accepted by ``scipy.integrate.solve_ivp``.
    points : int, default=1000
        Number of temperature samples in output grid.
    noise_config : dict[str, float] | None, optional
        Optional additive-noise configuration forwarded to ``add_noise``.

    Returns
    -------
    SimulationResult
        Typed result with temperature, intensity and state trajectories.
    """
    total_time = (T_end - T0) / beta
    t_span = (0.0, total_time)
    t_eval = np.linspace(0.0, total_time, points)

    solution = solve_ivp(
        lambda t, y: rhs_func(t, y, *params, beta=beta, T0=T0),
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        warnings.warn(
            f"Simulation did not converge: {solution.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    t_out = T0 + beta * solution.t

    derivs: list[np.ndarray] = []
    for idx in range(len(solution.t)):
        deriv = rhs_func(solution.t[idx], solution.y[:, idx], *params, beta=beta, T0=T0)
        derivs.append(np.asarray(deriv, dtype=float))
    deriv_array = np.asarray(derivs, dtype=float).T

    intensity = -deriv_array[0]
    if noise_config:
        mode_str = str(noise_config.get("mode", "gaussian"))
        mode: Literal["gaussian", "poisson"] = "poisson" if mode_str == "poisson" else "gaussian"
        sigma = float(noise_config.get("sigma", 0.0))
        seed_value = noise_config.get("seed")
        seed = int(seed_value) if seed_value is not None else None
        intensity = add_noise(intensity, mode=mode, sigma=sigma, seed=seed)

    states = {key: solution.y[i] for i, key in enumerate(state_keys)}
    return SimulationResult(T=t_out, I=intensity, states=states, time=solution.t)
