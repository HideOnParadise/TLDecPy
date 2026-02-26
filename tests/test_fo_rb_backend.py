from __future__ import annotations

import numpy as np

from tldecpy.models.fo import fo_rb, fo_rq


def test_fo_rb_matches_fo_rq_on_typical_grid() -> None:
    """
    Compare FO rational-quartic and FO rational-barycentric backends.

    A denominator floor avoids tail-dominated relative errors where intensity -> 0.
    """
    T = np.linspace(295.0, 700.0, 3000, dtype=np.float64)
    y_ref = fo_rq(T, Im=2500.0, E=1.2, Tm=450.0)
    y_aaa = fo_rb(T, Im=2500.0, E=1.2, Tm=450.0)

    rel_diff = np.abs(y_ref - y_aaa) / np.maximum(np.abs(y_ref), 1.0)
    max_rel_diff = float(np.max(rel_diff))
    assert max_rel_diff < 0.10
