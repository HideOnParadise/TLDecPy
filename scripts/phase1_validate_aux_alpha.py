#!/usr/bin/env python3
r"""Phase-1 numerical validation for the TLDecPy auxiliary function backend.

This script compares the thermoluminescence auxiliary function

.. math::
   Q(z) = 1 - z e^z E_1(z), \quad z = E/(kT)

using three approaches:

1. Reference (SciPy): direct evaluation via ``scipy.special.expn(1, z)``.
2. TLDecPy AAA (7-term barycentric): ``tldecpy.utils.aaa_fo.Q_aaa``.
3. Traditional Bos et al. (1993a) rational form: ``1.0 - alpha_poly(z)``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import expn

# Allow direct execution from repository root:
#   python scripts/phase1_validate_aux_alpha.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldecpy.models.fo import alpha_poly  # noqa: E402
from tldecpy.utils.aaa_fo import Q_aaa, load_aaa_Q_constants  # noqa: E402

FloatArray = NDArray[np.float64]

# Radiation Measurements plotting standard for this validation protocol.
# This is the definitive visual standard to reuse in phases 2-5.
RM_PAPER_RCPARAMS: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 6.8,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "axes.grid": False,
}


@dataclass(frozen=True)
class ValidationConfig:
    """Runtime configuration for the Phase-1 numerical validation."""

    z_min: float
    z_max: float
    num_points: int
    dpi: int
    figure_format: str
    output_dir: Path
    basename: str


def Q_reference(z: FloatArray) -> FloatArray:
    r"""Evaluate the reference auxiliary function :math:`Q(z)`.

    Parameters
    ----------
    z : numpy.ndarray
        Dimensionless argument :math:`z = E/(kT)`.

    Returns
    -------
    numpy.ndarray
        Reference values from
        :math:`Q(z)=1-z e^z E_1(z)` with :math:`E_1(z)=\mathrm{expn}(1,z)`.
    """
    return np.asarray(1.0 - z * np.exp(z) * expn(1, z), dtype=np.float64)


def Q_aaa_eval(z: FloatArray) -> FloatArray:
    r"""Evaluate AAA barycentric approximation of :math:`Q(z)`.

    Parameters
    ----------
    z : numpy.ndarray
        Dimensionless argument :math:`z = E/(kT)`.

    Returns
    -------
    numpy.ndarray
        AAA values loaded from ``aaa_Q_z4p5_130.npz``.
    """
    return np.asarray(Q_aaa(z), dtype=np.float64)


def Q_bos_rational(z: FloatArray) -> FloatArray:
    r"""Evaluate the traditional Bos (1993a) rational approximation.

    Parameters
    ----------
    z : numpy.ndarray
        Dimensionless argument :math:`z = E/(kT)`.

    Returns
    -------
    numpy.ndarray
        Traditional rational approximation for :math:`Q(z)`.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    # alpha_poly approximates (1 - Q(z)), therefore Q(z) = 1 - alpha_poly(z).
    return np.asarray(1.0 - alpha_poly(z_arr), dtype=np.float64)


def relative_abs_error(approx: FloatArray, reference: FloatArray) -> FloatArray:
    r"""Compute absolute relative error :math:`\varepsilon`.

    Parameters
    ----------
    approx : numpy.ndarray
        Approximated values.
    reference : numpy.ndarray
        Reference values.

    Returns
    -------
    numpy.ndarray
        Element-wise error:
        :math:`\varepsilon = |(\mathrm{approx} - \mathrm{ref}) / \mathrm{ref}|`.
    """
    denom = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return np.asarray(np.abs((approx - reference) / denom), dtype=np.float64)


def configure_matplotlib(dpi: int) -> None:
    """Apply journal-style Matplotlib settings for scientific figures.

    Parameters
    ----------
    dpi : int
        Output resolution in dots per inch.
    """
    style = dict(RM_PAPER_RCPARAMS)
    style.update({"savefig.dpi": dpi, "figure.dpi": dpi, "font.size": 10})
    plt.rcParams.update(style)


def build_config(args: argparse.Namespace) -> ValidationConfig:
    """Build immutable runtime configuration from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    ValidationConfig
        Normalized validation configuration.
    """
    return ValidationConfig(
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        num_points=int(args.num_points),
        dpi=int(args.dpi),
        figure_format=str(args.figure_format).lower(),
        output_dir=Path(args.output_dir),
        basename=str(args.basename),
    )


def compute_dataset(config: ValidationConfig) -> dict[str, FloatArray]:
    """Compute reference and approximation arrays over the requested domain.

    Parameters
    ----------
    config : ValidationConfig
        Validation runtime configuration.

    Returns
    -------
    dict[str, numpy.ndarray]
        Computed arrays and masks for plotting and CSV export.
    """
    z = np.linspace(config.z_min, config.z_max, config.num_points, dtype=np.float64)
    constants = load_aaa_Q_constants()
    aaa_support = (z >= constants.zmin) & (z <= constants.zmax)

    Q_ref = Q_reference(z)
    Q_aaa_clipped = Q_aaa_eval(z)
    Q_bos = Q_bos_rational(z)

    err_aaa_clipped = relative_abs_error(Q_aaa_clipped, Q_ref)
    err_aaa_support = np.full_like(err_aaa_clipped, np.nan)
    err_aaa_support[aaa_support] = err_aaa_clipped[aaa_support]
    err_bos = relative_abs_error(Q_bos, Q_ref)

    Q_aaa_support = np.full_like(Q_aaa_clipped, np.nan)
    Q_aaa_support[aaa_support] = Q_aaa_clipped[aaa_support]

    return {
        "z": z,
        "Q_ref": Q_ref,
        "Q_aaa_clipped": Q_aaa_clipped,
        "Q_aaa_support": Q_aaa_support,
        "Q_bos": Q_bos,
        "err_aaa_clipped": err_aaa_clipped,
        "err_aaa_support": err_aaa_support,
        "err_bos": err_bos,
        "aaa_support_mask": aaa_support.astype(np.float64),
    }


def export_csv(path: Path, dataset: dict[str, FloatArray]) -> None:
    """Write full comparison dataset to CSV.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV file.
    dataset : dict[str, numpy.ndarray]
        Output of :func:`compute_dataset`.
    """
    header = [
        "z",
        "Q_ref",
        "Q_aaa_clipped",
        "Q_aaa_support",
        "Q_bos_rational",
        "rel_err_aaa_clipped",
        "rel_err_aaa_support",
        "rel_err_bos_rational",
        "aaa_support_mask",
    ]

    rows = zip(
        dataset["z"],
        dataset["Q_ref"],
        dataset["Q_aaa_clipped"],
        dataset["Q_aaa_support"],
        dataset["Q_bos"],
        dataset["err_aaa_clipped"],
        dataset["err_aaa_support"],
        dataset["err_bos"],
        dataset["aaa_support_mask"],
    )

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def export_summary(path: Path, dataset: dict[str, FloatArray]) -> None:
    """Write compact numerical metrics for manuscript reporting.

    Parameters
    ----------
    path : pathlib.Path
        Destination summary file.
    dataset : dict[str, numpy.ndarray]
        Output of :func:`compute_dataset`.
    """
    mask = dataset["aaa_support_mask"] > 0.5
    err_aaa_in = dataset["err_aaa_clipped"][mask]
    err_bos_all = dataset["err_bos"]

    lines = [
        "TLDecPy Phase-1 numerical validation summary",
        f"AAA support points used: {int(np.sum(mask))}",
        f"AAA max rel. err (in-support): {float(np.max(err_aaa_in)):.3e}",
        f"AAA p95 rel. err (in-support): {float(np.percentile(err_aaa_in, 95)):.3e}",
        f"AAA p99 rel. err (in-support): {float(np.percentile(err_aaa_in, 99)):.3e}",
        f"Bos max rel. err (full range): {float(np.max(err_bos_all)):.3e}",
        f"Bos p95 rel. err (full range): {float(np.percentile(err_bos_all, 95)):.3e}",
        f"Bos p99 rel. err (full range): {float(np.percentile(err_bos_all, 99)):.3e}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_figure(path: Path, config: ValidationConfig, dataset: dict[str, FloatArray]) -> None:
    """Create log-scale relative error figure in journal-friendly format.

    Parameters
    ----------
    path : pathlib.Path
        Destination figure file.
    config : ValidationConfig
        Runtime plotting configuration.
    dataset : dict[str, numpy.ndarray]
        Output of :func:`compute_dataset`.
    """
    configure_matplotlib(config.dpi)

    z = dataset["z"]
    err_aaa_support = dataset["err_aaa_support"]
    err_bos = dataset["err_bos"]

    aaa_plot = np.where(np.isfinite(err_aaa_support), np.maximum(err_aaa_support, 1e-16), np.nan)
    bos_plot = np.maximum(err_bos, 1e-16)

    constants = load_aaa_Q_constants()

    fig, ax = plt.subplots()
    fig.set_size_inches(3.5, 3.0)
    ax.axvspan(constants.zmin, constants.zmax, color="0.5", alpha=0.1, zorder=0)
    ax.plot(
        z,
        aaa_plot,
        color="#1f77b4",
        label="Barycentric Rational (AAA)",
    )
    ax.plot(
        z,
        bos_plot,
        color="#d62728",
        label="Polynomial Rational (Bos et al., 1993a)",
    )
    ax.axvline(constants.zmin, color="0.55", linestyle="--", linewidth=0.9)
    ax.axvline(constants.zmax, color="0.55", linestyle="--", linewidth=0.9)

    ax.set_yscale("log")
    ax.set_xlim(config.z_min, config.z_max)
    y_min = max(float(np.nanmin(np.concatenate([aaa_plot[np.isfinite(aaa_plot)], bos_plot]))) * 0.5, 1e-17)
    y_max = max(float(np.nanmax(bos_plot) * 4.0), 1.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$z = E/kT$")
    ax.set_ylabel(r"Relative Error $|(Q_{approx} - Q_{ref}) / Q_{ref}|$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.89),
        handletextpad=0.5,
        handlelength=1.2,
        borderpad=0.25,
        labelspacing=0.2,
        borderaxespad=0.3,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.7")

    plt.tight_layout()
    fig.savefig(path, format=config.figure_format, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for this script.
    """
    parser = argparse.ArgumentParser(
        description="Phase-1 validation of auxiliary function Q(z) for TLDecPy.",
    )
    parser.add_argument("--z-min", type=float, default=4.0, help="Lower bound of z range.")
    parser.add_argument("--z-max", type=float, default=150.0, help="Upper bound of z range.")
    parser.add_argument("--num-points", type=int, default=20_000, help="Number of z samples.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    parser.add_argument(
        "--figure-format",
        type=str,
        default="pdf",
        choices=("pdf", "tiff", "png"),
        help="Output figure format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/phase1_validation",
        help="Directory where CSV/figure/summary will be stored.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="aux_Q_comparison",
        help="Base filename for generated artifacts.",
    )
    return parser


def main() -> None:
    """Run the end-to-end numerical validation workflow."""
    args = build_parser().parse_args()
    config = build_config(args)

    if config.z_max <= config.z_min:
        raise ValueError("z-max must be greater than z-min.")
    if config.num_points < 100:
        raise ValueError("num-points must be >= 100 for stable visual comparison.")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = compute_dataset(config)

    csv_path = config.output_dir / f"{config.basename}.csv"
    fig_path = config.output_dir / f"{config.basename}.{config.figure_format}"
    summary_path = config.output_dir / f"{config.basename}_summary.txt"

    export_csv(csv_path, dataset)
    export_figure(fig_path, config, dataset)
    export_summary(summary_path, dataset)

    support_mask = dataset["aaa_support_mask"] > 0.5
    aaa_max_support = float(np.max(dataset["err_aaa_clipped"][support_mask]))
    bos_max = float(np.max(dataset["err_bos"]))

    print("Phase-1 validation finished.")
    print(f"CSV: {csv_path}")
    print(f"Figure: {fig_path}")
    print(f"Summary: {summary_path}")
    print(f"AAA max relative error (support only): {aaa_max_support:.3e}")
    print(f"Bos max relative error (full range): {bos_max:.3e}")


if __name__ == "__main__":
    main()
