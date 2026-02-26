#!/usr/bin/env python3
"""Manual multi-peak deconvolution example using Refglow x002."""

from __future__ import annotations

from pathlib import Path
import sys

try:
    import tldecpy as tl
except ModuleNotFoundError:
    # Allow direct execution from a source checkout without installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tldecpy as tl


def main() -> None:
    """
    Fit Refglow x002 using manually specified peak models and parameter bounds.

    This example does not use autoinit; all components are defined explicitly.
    """
    temperature, intensity = tl.load_refglow("x002")

    # Manual peak setup (four components) with user-defined initial values and bounds.
    peaks = [
        tl.PeakSpec(
            name="P1",
            model="fo_rq",
            init={"Tm": 417.0, "Im": 12000.0, "E": 1.35},
            bounds={"Tm": (390.0, 435.0), "Im": (0.0, 40000.0), "E": (0.8, 2.2)},
        ),
        tl.PeakSpec(
            name="P2",
            model="fo_rq",
            init={"Tm": 456.0, "Im": 18000.0, "E": 1.48},
            bounds={"Tm": (435.0, 472.0), "Im": (0.0, 50000.0), "E": (0.8, 2.4)},
        ),
        tl.PeakSpec(
            name="P3",
            model="fo_rq",
            init={"Tm": 484.0, "Im": 28000.0, "E": 1.60},
            bounds={"Tm": (468.0, 500.0), "Im": (0.0, 70000.0), "E": (0.9, 2.6)},
        ),
        tl.PeakSpec(
            name="P4",
            model="fo_rq",
            init={"Tm": 512.0, "Im": 45000.0, "E": 2.00},
            bounds={"Tm": (498.0, 530.0), "Im": (0.0, 120000.0), "E": (1.0, 3.0)},
        ),
    ]

    result = tl.fit_multi(
        temperature,
        intensity,
        peaks=peaks,
        bg=None,
        beta=8.4,  # Refglow x002 heating rate used in benchmark definitions.
        robust=tl.RobustOptions(loss="linear"),
        options=tl.FitOptions(local_optimizer="trf"),
    )

    print(f"Converged: {result.converged}")
    print(f"R2: {result.metrics.R2:.6f}")
    print(f"FOM: {result.metrics.FOM:.4f}%")
    print("Fitted peak parameters:")
    for peak in result.peaks:
        params = peak.params
        print(
            f"  {peak.name} ({peak.model}) -> "
            f"Tm={params.get('Tm', float('nan')):.3f} K, "
            f"E={params.get('E', float('nan')):.6g} eV, "
            f"Im={params.get('Im', float('nan')):.6g}"
        )


if __name__ == "__main__":
    main()
