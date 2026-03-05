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
    Integrate TL kinetic ODEs under linear heating and return a typed result.

    The time variable is related to temperature by the linear heating law
    :math:`T(t) = T_0 + \beta t`.  The ODE is integrated in time and the
    output is re-expressed on a temperature grid.

    TL intensity is computed as minus the time-derivative of the first state
    variable:  :math:`I(T) = -\dot{y}_0(t)`.

    Parameters
    ----------
    rhs_func : Callable
        ODE right-hand side callable.  Must accept positional arguments
        ``(t, y, *params)`` followed by keyword arguments ``beta`` and ``T0``:
        ``rhs_func(t, y, *params, beta=beta, T0=T0) -> array-like``.
        The first element of the return value is treated as the trap
        population :math:`n(t)`; TL intensity is its negative derivative.
    params : tuple[float, ...]
        Model-specific kinetic parameters passed as positional args to
        ``rhs_func`` after ``t`` and ``y``.  Order must match the function
        signature.
    T0 : float
        Initial temperature in kelvin.  Must be < ``T_end``.
    T_end : float
        Final temperature in kelvin.  Must be > ``T0``.
    beta : float
        Linear heating rate :math:`\beta` in K/s.  Must be > 0.
    y0 : list[float]
        Initial state vector passed to ``scipy.integrate.solve_ivp``.
        Length must match the number of ODE states in ``rhs_func``.
    state_keys : list[str]
        Human-readable names for each state variable.  Used as keys in
        ``SimulationResult.states``.  Length must match ``len(y0)``.
    method : str, default="LSODA"
        Integration algorithm accepted by ``scipy.integrate.solve_ivp``
        (e.g. ``"LSODA"``, ``"RK45"``, ``"Radau"``).  ``"LSODA"`` is
        recommended for stiff TL kinetics.
    points : int, default=1000
        Number of evenly-spaced output temperature samples.
    noise_config : dict | None, optional
        If provided, additive noise is applied to the intensity array.
        Recognised keys:

        - ``"mode"`` — ``"gaussian"`` (default) or ``"poisson"``
        - ``"sigma"`` — float, standard deviation for Gaussian noise
        - ``"seed"`` — int or ``None``, RNG seed for reproducibility

    Returns
    -------
    SimulationResult
        Typed result with fields:

        - ``T`` — temperature grid in kelvin (float64, 1-D, length ``points``)
        - ``I`` — simulated TL intensity (float64, 1-D)
        - ``states`` — dict mapping each ``state_key`` to its trajectory
        - ``time`` — integration time vector in seconds

    Notes
    -----
    A ``RuntimeWarning`` is emitted if the ODE integrator does not converge.
    The result is still returned with whatever data was produced up to the
    failure point.

    Examples
    --------
    >>> import numpy as np
    >>> import tldecpy as tl
    >>> from tldecpy.simulate.fo import ode_fo, infer_n0_s
    >>> T0, T_end, beta = 300.0, 600.0, 1.0
    >>> E, Tm = 1.2, 450.0
    >>> n0, s = infer_n0_s(Im=1.0, Tm=Tm, E=E, beta=beta)
    >>> result = tl.simulate(
    ...     ode_fo, params=(s, E),
    ...     T0=T0, T_end=T_end, beta=beta,
    ...     y0=[n0], state_keys=["n"],
    ... )
    >>> print(result.T.shape, result.I.max())
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
