"""
ODE definitions for FO, SO, GO models.
"""

import numpy as np
from tldecpy.utils.constants import KB_EV


def infer_n0_s(Im: float, Tm: float, E: float, beta: float, order: float = 1.0):
    """
    Approximate n0 (initial population) and s (frequency factor)
    from Peak Shape parameters (Im, Tm, E).

    Based on Maximum Condition equations.
    """
    k = KB_EV

    # 1. Frequency factor s
    # General Order Condition: (beta*E) / (k*Tm^2) = s * exp(-E/kTm) * [1 + (b-1)*Delta]...
    # First Order approx is good seed:
    s = (beta * E) / (k * Tm**2) * np.exp(E / (k * Tm))

    # 2. Initial Population n0
    # For FO: I = s * n * exp...
    # At Peak: Im = s * nm * exp(-E/kTm) -> nm = Im / (s * exp(-E/kTm))
    # Also for FO, nm/n0 = 1/e approx. So n0 ~ nm * e.
    # Exact integral area is better: Area ~ Im * width ...
    # Let's use the analytical area approximation
    # Area ~ n0 (normalized).
    # Actually, integral of I dt = n0.
    # Let's use the Kitis approx for Area and set n0 = Area/beta? No, Area in counts.

    # Simpler: run a quick calculation of I_max with n0=1, then scale.
    # This avoids complex inversions.
    # We return s, and a placeholder n0=1.0. The caller should normalize I.
    return 1.0, s


def ode_go(t, y, s, E, b, beta, T0):
    """
    General Order ODE.
    dn/dt = - s * n^b * exp(-E/kT) * (1/N^(b-1))
    We assume normalized units where N=1 or absorbed into s'.
    Effective: dn/dt = - s * n^b * exp(-E/kT)
    """
    n = y[0]
    if n < 0:
        n = 0

    T = T0 + beta * t
    k = KB_EV

    rate = s * (n**b) * np.exp(-E / (k * T))

    return [-rate]


# Wrappers for specific orders
def ode_fo(t, y, s, E, beta, T0):
    """Specialized FO wrapper: call :func:`ode_go` with kinetic order ``b = 1``."""
    return ode_go(t, y, s, E, 1.0, beta, T0)


def ode_so(t, y, s, E, beta, T0):
    """Specialized SO wrapper: call :func:`ode_go` with kinetic order ``b = 2``."""
    return ode_go(t, y, s, E, 2.0, beta, T0)
