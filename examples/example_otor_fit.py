#!/usr/bin/env python3
"""End-to-end OTOR deconvolution example for TLDecPy."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

try:
    import tldecpy as tl
except ModuleNotFoundError:
    # Allow direct execution from a source checkout without installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tldecpy as tl


def main() -> None:
    """Generate synthetic OTOR data, fit it, and print recovered parameters."""
    rng = np.random.default_rng(42)

    # Temperature grid (K) and heating rate (K/s)
    temperature = np.linspace(300.0, 620.0, 450)
    beta = 1.0

    # Synthetic OTOR curve using the Lambert W implementation.
    model = tl.get_model("otor_lw")
    true_params = {"Im": 1400.0, "E": 1.35, "Tm": 470.0, "R": 0.20}
    intensity_clean = model(temperature, **true_params, beta=beta)

    noise_sigma = 0.02 * float(np.max(intensity_clean))
    intensity_obs = np.clip(
        intensity_clean + rng.normal(0.0, noise_sigma, size=temperature.shape),
        0.0,
        None,
    )

    peak = tl.PeakSpec(
        name="OTOR_P1",
        model="otor_lw",
        init={"Im": 1100.0, "E": 1.10, "Tm": 455.0, "R": 0.10},
        bounds={
            "Im": (0.0, 3000.0),
            "E": (0.3, 2.5),
            "Tm": (380.0, 560.0),
            "R": (1e-6, 0.95),
        },
    )

    result = tl.fit_multi(
        temperature,
        intensity_obs,
        peaks=[peak],
        bg=None,
        beta=beta,
        robust=tl.RobustOptions(loss="soft_l1", f_scale=2.0, weights="poisson"),
        options=tl.FitOptions(local_optimizer="trf"),
    )

    fitted = result.peaks[0].params
    print(f"Converged: {result.converged}")
    print(f"R2: {result.metrics.R2:.5f}")
    print(f"FOM: {result.metrics.FOM:.3f}%")
    print("Recovered kinetic parameters (true -> fitted):")
    for key in ("Im", "E", "Tm", "R"):
        print(f"  {key}: {true_params[key]:.6g} -> {fitted[key]:.6g}")


if __name__ == "__main__":
    main()
