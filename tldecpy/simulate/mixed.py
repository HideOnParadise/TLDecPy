"""
ODE definitions for Mixed-Order Kinetics.
"""

import numpy as np
from tldecpy.utils.constants import KB_EV


def ode_mixed(t, y, s, E, alpha, c, beta, T0):
    """
    Mixed Order ODE.
    dn/dt = - s' * n * (n + c) * exp(-E/kT)

    Relation: alpha = n0 / (n0 + c)  =>  c = n0 * (1-alpha)/alpha
    """
    n = y[0]
    if n < 0:
        n = 0

    T = T0 + beta * t
    k = KB_EV

    # s' is the pre-exponential factor for MO.
    # Note: s' has units cm^3/s usually.
    # We treat 's' passed here as the effective frequency factor.

    dndt = -s * n * (n + c) * np.exp(-E / (k * T))
    return [dndt]
