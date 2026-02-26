import numpy as np
from tldecpy.simulate.core import simulate
from tldecpy.simulate.fo import ode_fo
from tldecpy.models.fo import fo_rq


def test_fo_simulation_parity():
    """
    Compare numerical ODE solution vs analytical fo_rq model.
    """
    # Params
    E = 1.0
    Tm = 400.0
    beta = 1.0
    T0, T_end = 300, 500
    k = 8.617e-5

    # 1. Analytical (Target)
    T_anal = np.linspace(T0, T_end, 200)
    Im_target = 1000.0
    I_anal = fo_rq(T_anal, Im_target, E, Tm)

    # 2. Infer Kinetic Params from Shape Params
    # s = beta*E / (k*Tm^2) * exp(E/kTm)
    s_val = (beta * E) / (k * Tm**2) * np.exp(E / (k * Tm))

    # 3. Numerical Simulation
    # Initial population n0 is unknown relative to Im.
    # We simulate with n0=1, then scale I_max to match Im_target.
    res = simulate(
        rhs_func=ode_fo,
        params=(s_val, E),
        T0=T0,
        T_end=T_end,
        beta=beta,
        y0=[1.0],  # Normalized n0
        state_keys=["n"],
        points=200,
    )

    # Normalize Simulation to match Im
    I_sim_norm = res.I * (Im_target / np.max(res.I))

    # Interpolate simulation to analytical grid for comparison
    I_sim_interp = np.interp(T_anal, res.T, I_sim_norm)

    # Check Agreement
    # RMS error should be small (< 1% of max)
    rms = np.sqrt(np.mean((I_anal - I_sim_interp) ** 2))
    rel_error = rms / Im_target

    assert rel_error < 0.01, f"ODE simulation deviates from analytical fo_rq: {rel_error:.2%}"

    # Check Peak Position
    idx_max = np.argmax(I_sim_norm)
    Tm_sim = res.T[idx_max]
    assert abs(Tm_sim - Tm) < 2.0, f"Simulated Tm {Tm_sim} != Target {Tm}"
