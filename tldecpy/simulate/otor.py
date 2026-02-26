"""
ODE definitions for OTOR model.
"""

import numpy as np
from tldecpy.utils.constants import KB_EV


def ode_otor(t, y, s, E, R, N, beta, T0):
    """
    One Trap One Recombination (OTOR) Rate Equations.
    Robust form valid for all R > 0.

    Reference: Sadek et al. (2015), Eq 1.

    dn/dt = - [ n * s * exp(-E/kT) ] * P_recombination
    P_recomb = n / (n + R*(N-n))
    """
    n = y[0]

    # Physical constraint: population cannot be negative
    if n < 0:
        n = 0
    # Constraint: n cannot exceed total traps N
    if n > N:
        n = N

    T = T0 + beta * t
    k = KB_EV

    # Rate of escape from trap (detrapping)
    detrapping_rate = n * s * np.exp(-E / (k * T))

    # Probability of recombination vs retrapping
    # R = An / Am
    # Prob = (Am * n) / (Am * n + An * (N-n))
    # Divide by Am -> n / (n + R*(N-n))

    denominator = n + R * (N - n)

    # Avoid division by zero if n=0 and R=0 (unlikely but safe)
    if denominator <= 1e-20:
        recomb_prob = 0.0
    else:
        recomb_prob = n / denominator

    dndt = -detrapping_rate * recomb_prob

    return [dndt]
