from __future__ import annotations

from importlib.resources import files

import numpy as np
from scipy.special import expn

from tldecpy.utils.aaa_fo import Q_aaa, alpha_aaa, load_aaa_Q_constants


def _q_truth(z: np.ndarray) -> np.ndarray:
    return np.exp(z) * expn(2, z)


def test_aaa_Q_matches_truth_on_domain() -> None:
    constants = load_aaa_Q_constants()
    z = np.geomspace(constants.zmin, constants.zmax, 20_000, dtype=np.float64)

    q_aaa = Q_aaa(z)
    q_truth = _q_truth(z)

    rel = np.abs(q_aaa - q_truth) / np.abs(q_truth)
    rel_max = float(np.max(rel))
    rel_p95 = float(np.percentile(rel, 95))
    rel_p99 = float(np.percentile(rel, 99))

    assert rel_max < 1e-12, (
        f"AAA Q(z) max relative error too high: rel_max={rel_max:.3e}, "
        f"p95={rel_p95:.3e}, p99={rel_p99:.3e}"
    )
    assert rel_p95 < 5e-13, (
        f"AAA Q(z) 95th percentile relative error too high: "
        f"p95={rel_p95:.3e}, rel_max={rel_max:.3e}"
    )


def test_aaa_alpha_matches_truth_on_domain() -> None:
    constants = load_aaa_Q_constants()
    z = np.geomspace(constants.zmin, constants.zmax, 20_000, dtype=np.float64)

    alpha_vals = alpha_aaa(z)
    alpha_truth = 1.0 - _q_truth(z)

    rel = np.abs(alpha_vals - alpha_truth) / np.abs(alpha_truth)
    max_rel_diff = float(np.max(rel))

    assert max_rel_diff < 1e-12, f"alpha_aaa max relative error too high: {max_rel_diff:.3e}"
    assert np.all(alpha_vals > 0.0), "alpha_aaa must remain positive on AAA domain."
    assert np.all(alpha_vals <= 1.0 + 1e-12), "alpha_aaa must stay <= 1 on AAA domain."


def test_aaa_resource_loads() -> None:
    resource = files("tldecpy.data").joinpath("aaa_Q_z4p5_130.npz")
    assert resource.is_file(), (
        "AAA NPZ resource not found at 'tldecpy/data/aaa_Q_z4p5_130.npz'. "
        "Check package data configuration for '*.npz'."
    )

    with resource.open("rb") as fh:
        with np.load(fh) as data:
            keys = set(data.files)
    expected = {"zmin", "zmax", "support_points", "support_values", "weights"}
    assert expected <= keys, (
        f"AAA NPZ missing keys: expected {sorted(expected)}, got {sorted(keys)}"
    )


def test_out_of_domain_behavior() -> None:
    constants = load_aaa_Q_constants()
    z_low = constants.zmin * 0.9
    z_high = constants.zmax * 1.1

    alpha_oob = alpha_aaa(np.asarray([z_low, z_high], dtype=np.float64))
    alpha_clipped = alpha_aaa(np.asarray([constants.zmin, constants.zmax], dtype=np.float64))

    np.testing.assert_array_equal(
        alpha_oob,
        alpha_clipped,
        err_msg=(
            "Out-of-domain alpha_aaa behavior changed. Current policy is clipping to "
            "[zmin, zmax] before barycentric evaluation."
        ),
    )
