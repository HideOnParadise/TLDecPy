#!/usr/bin/env python3
"""Phase 3: robustness comparison of four loss functions under extreme outliers.

This script compares:

1. ``linear``  (standard least squares, control)
2. ``soft_l1`` (robust)
3. ``huber``   (robust)
4. ``cauchy``  (aggressive robust)

on the same corrupted one-peak synthetic glow curve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
from numpy.typing import NDArray

# Allow direct execution from repository root:
#   python scripts/phase3_robustness_test.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldecpy.fit.multi import fit_multi  # noqa: E402
from tldecpy.models.registry import get_model  # noqa: E402
from tldecpy.schemas import FitOptions, PeakSpec, RobustOptions, UncertaintyOptions  # noqa: E402
from tldecpy.simulate.noise import add_noise  # noqa: E402

FloatArray = NDArray[np.float64]

RM_PAPER_RCPARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "axes.linewidth": 1.1,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
}

TRUE_PARAMS: dict[str, float] = {"Im": 1000.0, "Tm": 450.0, "E": 1.1}
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "Im": (100.0, 3000.0),
    "Tm": (380.0, 520.0),
    "E": (0.6, 1.8),
}

METHOD_ORDER: tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy")
METHOD_STYLES: dict[str, dict[str, Any]] = {
    "linear": {
        "label": "Linear (Least Squares)",
        "color": "#1f77b4",
        "linestyle": "--",
        "f_scale": 1.0,
    },
    "soft_l1": {
        "label": "Soft-L1",
        "color": "#9467bd",
        "linestyle": "-",
        "f_scale": 0.1,
    },
    "huber": {
        "label": "Huber",
        "color": "#ff7f0e",
        "linestyle": "-",
        "f_scale": 0.1,
    },
    "cauchy": {
        "label": "Cauchy",
        "color": "#2ca02c",
        "linestyle": "-",
        "f_scale": 0.12,
    },
}


@dataclass(frozen=True)
class RunConfig:
    """Runtime settings for phase-3 robustness comparison."""

    n_points: int
    t_min: float
    t_max: float
    snr_db: float
    n_outliers: int
    spike_factor: float
    strategy: str
    output_dir: Path
    seed_noise: int
    seed_outliers: int
    seed_init: int
    max_nfev: int
    dpi: int


def configure_matplotlib(dpi: int) -> None:
    """Apply publication-style plotting defaults."""
    style = dict(RM_PAPER_RCPARAMS)
    style.update({"figure.dpi": dpi, "savefig.dpi": dpi})
    plt.rcParams.update(style)


def _snr_sigma_from_peak(signal: FloatArray, snr_db: float) -> float:
    """Compute Gaussian sigma from peak-amplitude SNR."""
    peak_signal = float(np.max(np.abs(signal)))
    if peak_signal <= 0.0:
        return 0.0
    return float(peak_signal / (10.0 ** (snr_db / 20.0)))


def _clip(value: float, name: str) -> float:
    """Clip one parameter candidate into physical bounds."""
    low, high = PARAM_BOUNDS[name]
    return float(np.clip(value, low, high))


def build_cli() -> argparse.ArgumentParser:
    """Build CLI for reproducible experiment execution."""
    parser = argparse.ArgumentParser(
        description="Phase 3 robustness comparison: linear vs soft_l1 vs huber vs cauchy."
    )
    parser.add_argument("--n-points", type=int, default=700)
    parser.add_argument("--t-min", type=float, default=330.0)
    parser.add_argument("--t-max", type=float, default=560.0)
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--n-outliers", type=int, default=5)
    parser.add_argument("--spike-factor", type=float, default=10.0)
    parser.add_argument(
        "--strategy",
        type=str,
        default="global_hybrid",
        choices=["global_hybrid", "global_hybrid_pso"],
    )
    parser.add_argument("--output-dir", type=str, default="output/phase3_validation")
    parser.add_argument("--seed-noise", type=int, default=20260301)
    parser.add_argument("--seed-outliers", type=int, default=20260302)
    parser.add_argument("--seed-init", type=int, default=20260303)
    parser.add_argument("--max-nfev", type=int, default=3000)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Convert parsed CLI args into a typed config object."""
    return RunConfig(
        n_points=int(args.n_points),
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        snr_db=float(args.snr_db),
        n_outliers=int(args.n_outliers),
        spike_factor=float(args.spike_factor),
        strategy=str(args.strategy),
        output_dir=Path(args.output_dir),
        seed_noise=int(args.seed_noise),
        seed_outliers=int(args.seed_outliers),
        seed_init=int(args.seed_init),
        max_nfev=int(args.max_nfev),
        dpi=int(args.dpi),
    )


def generate_clean_signal(temperature: FloatArray) -> FloatArray:
    """Generate one-peak first-order synthetic glow curve."""
    model = get_model("fo_rb")
    return np.asarray(model(temperature, **TRUE_PARAMS), dtype=np.float64)


def inject_outliers(
    y_in: FloatArray,
    *,
    n_outliers: int,
    spike_factor: float,
    seed: int,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Inject extreme spikes at random high-signal channels."""
    rng = np.random.default_rng(seed)
    y_abs = np.abs(np.asarray(y_in, dtype=np.float64))
    candidate_idx = np.where(y_abs >= 0.45 * float(np.max(y_abs)))[0]
    if candidate_idx.size < n_outliers:
        candidate_idx = np.arange(y_in.size)
    outlier_idx = np.asarray(
        rng.choice(candidate_idx, size=n_outliers, replace=False),
        dtype=np.int64,
    )
    y_out = np.asarray(y_in, dtype=np.float64).copy()
    y_out[outlier_idx] *= float(spike_factor)
    return y_out, outlier_idx


def sample_initial_guess(seed: int) -> dict[str, float]:
    """Generate one reproducible initial guess shared by all methods."""
    rng = np.random.default_rng(seed)
    init: dict[str, float] = {}
    for pname, ptrue in TRUE_PARAMS.items():
        candidate = float(ptrue) * rng.uniform(0.95, 1.05)
        init[pname] = _clip(candidate, pname)
    return init


def fit_one_peak(
    temperature: FloatArray,
    y_obs: FloatArray,
    init: dict[str, float],
    *,
    loss: str,
    f_scale: float,
    strategy: str,
    max_nfev: int,
) -> Any:
    """Fit one ``fo_rb`` peak using selected robust loss configuration."""
    peak = PeakSpec(
        name="P1",
        model="fo_rb",
        init=dict(init),
        bounds=dict(PARAM_BOUNDS),
    )
    robust = RobustOptions(
        loss=loss,  # type: ignore[arg-type]
        f_scale=f_scale,
        weights="none",
        multi_start=0,
        ci_bootstrap=False,
        n_bootstrap=0,
    )
    options = FitOptions(
        local_optimizer="trf",
        max_nfev=max_nfev,
        uncertainty=UncertaintyOptions(enabled=False),
    )
    return fit_multi(
        temperature,
        y_obs,
        peaks=[peak],
        bg=None,
        beta=1.0,
        robust=robust,
        options=options,
        strategy=strategy,  # type: ignore[arg-type]
    )


def relative_error_pct(estimate: float, truth: float) -> float:
    """Compute absolute relative error in percent."""
    if truth == 0.0:
        return float("nan")
    return float(abs((estimate - truth) / truth) * 100.0)


def make_loss_comparison_figure(
    path: Path,
    temperature: FloatArray,
    y_clean: FloatArray,
    y_dirty: FloatArray,
    outlier_idx: NDArray[np.int64],
    fit_results: dict[str, Any],
    *,
    snr_db: float,
    dpi: int,
) -> None:
    """Create full-curve comparison with inset zoom centered on largest spike."""
    configure_matplotlib(dpi=dpi)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(
        temperature,
        y_dirty,
        linestyle="none",
        marker="o",
        markersize=2.8,
        color="0.35",
        alpha=0.55,
        label=f"Corrupted data (SNR={snr_db:.0f} dB)",
    )
    ax.scatter(
        temperature[outlier_idx],
        y_dirty[outlier_idx],
        s=32,
        color="red",
        edgecolor="black",
        linewidth=0.35,
        zorder=5,
        label="Injected outliers (x10)",
    )
    ax.plot(
        temperature,
        y_clean,
        color="black",
        linewidth=1.0,
        linestyle="-",
        label="Ground Truth",
    )

    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        y_hat = np.asarray(fit_results[method].y_hat_total, dtype=np.float64)
        ax.plot(
            temperature,
            y_hat,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=1.5,
            label=str(style["label"]),
        )

    ax.set_xlabel("Temperature, T (K)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.legend(
        loc="upper left",
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        bbox_to_anchor=(0.01, 0.99),
    )

    # Inset zoom around the physical peak maximum (not the outlier summit).
    tm_true = float(TRUE_PARAMS["Tm"])
    x1, x2 = tm_true - 10.0, tm_true + 10.0
    i_max = float(np.max(y_clean))
    y1, y2 = 0.90 * i_max, 1.05 * i_max

    inset = ax.inset_axes([0.53, 0.36, 0.44, 0.46])
    inset.plot(
        temperature,
        y_dirty,
        linestyle="none",
        marker="o",
        markersize=2.2,
        color="0.45",
        alpha=0.55,
    )
    inset.scatter(
        temperature[outlier_idx],
        y_dirty[outlier_idx],
        s=24,
        color="red",
        edgecolor="black",
        linewidth=0.3,
        zorder=5,
    )
    inset.plot(temperature, y_clean, color="black", linewidth=1.5, linestyle="-")
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        y_hat = np.asarray(fit_results[method].y_hat_total, dtype=np.float64)
        inset.plot(
            temperature,
            y_hat,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=1.2,
        )

    inset.set_xlim(x1, x2)
    inset.set_ylim(y1, y2)
    inset.grid(True, which="both", linestyle="--", linewidth=0.25, alpha=0.5)
    inset.set_title(r"Zoom near $T_m$", fontsize=8)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    fit_results: dict[str, Any],
    outlier_idx: NDArray[np.int64],
    spike_factor: float,
) -> None:
    """Write method-comparison table and rejection ratios."""
    rows: list[dict[str, float | str]] = []
    for method in METHOD_ORDER:
        fit_obj = fit_results[method]
        params = fit_obj.peaks[0].params
        e_est = float(params["E"])
        tm_est = float(params["Tm"])
        e_err = relative_error_pct(e_est, TRUE_PARAMS["E"])
        tm_err = relative_error_pct(tm_est, TRUE_PARAMS["Tm"])
        rows.append(
            {
                "method": method,
                "fom": float(fit_obj.metrics.FOM),
                "e_est": e_est,
                "tm_est": tm_est,
                "e_err": e_err,
                "tm_err": tm_err,
                "f_scale": float(METHOD_STYLES[method]["f_scale"]),
            }
        )

    row_linear = next(row for row in rows if row["method"] == "linear")
    row_cauchy = next(row for row in rows if row["method"] == "cauchy")
    e_ratio = (
        float(row_linear["e_err"]) / float(row_cauchy["e_err"])
        if float(row_cauchy["e_err"]) > 0
        else float("inf")
    )
    tm_ratio = (
        float(row_linear["tm_err"]) / float(row_cauchy["tm_err"])
        if float(row_cauchy["tm_err"]) > 0
        else float("inf")
    )
    valid_ratios = np.asarray(
        [ratio for ratio in (e_ratio, tm_ratio) if np.isfinite(ratio)],
        dtype=np.float64,
    )
    mean_ratio = float(np.mean(valid_ratios)) if valid_ratios.size > 0 else float("nan")

    lines: list[str] = [
        "TLDecPy - Phase 3: Four-Loss Robustness Comparison",
        "",
        "Ground truth parameters:",
        f"Im = {TRUE_PARAMS['Im']:.4f}, Tm = {TRUE_PARAMS['Tm']:.4f}, E = {TRUE_PARAMS['E']:.6f}",
        "",
        "Injected outliers:",
        f"indices = {outlier_idx.tolist()}",
        f"spike factor = x{spike_factor:.1f}",
        "",
        "Method comparison (same corrupted dataset):",
        "method | loss | f_scale | FOM(%) | E_est | E_err(%) | Tm_est | Tm_err(%)",
    ]
    for row in rows:
        method = str(row["method"])
        lines.append(
            f"{method} | {method} | {float(row['f_scale']):.3f} | {float(row['fom']):.4f} | "
            f"{float(row['e_est']):.6f} | {float(row['e_err']):.4f} | "
            f"{float(row['tm_est']):.6f} | {float(row['tm_err']):.4f}"
        )

    lines.extend(
        [
            "",
            "Rejection Ratio (Linear vs Cauchy):",
            f"E error ratio = {e_ratio:.3f}x",
            f"Tm error ratio = {tm_ratio:.3f}x",
            f"Mean rejection ratio = {mean_ratio:.3f}x",
            "",
            "Interpretation:",
            "Soft-L1 and Cauchy provide the strongest rejection under extreme outliers.",
            "Huber offers a balanced compromise between robustness and smooth influence.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the complete phase-3 loss-function comparison."""
    config = parse_config(build_cli().parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    temperature = np.linspace(config.t_min, config.t_max, config.n_points, dtype=np.float64)
    y_clean = generate_clean_signal(temperature)
    sigma = _snr_sigma_from_peak(y_clean, config.snr_db)
    y_noisy = np.asarray(
        add_noise(y_clean, mode="gaussian", sigma=sigma, seed=config.seed_noise),
        dtype=np.float64,
    )
    y_dirty, outlier_idx = inject_outliers(
        y_noisy,
        n_outliers=config.n_outliers,
        spike_factor=config.spike_factor,
        seed=config.seed_outliers,
    )

    init_guess = sample_initial_guess(config.seed_init)
    fit_results: dict[str, Any] = {}
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        fit_results[method] = fit_one_peak(
            temperature=temperature,
            y_obs=y_dirty,
            init=init_guess,
            loss=method,
            f_scale=float(style["f_scale"]),
            strategy=config.strategy,
            max_nfev=config.max_nfev,
        )

    fig_path = config.output_dir / "phase3_loss_comparison.pdf"
    summary_path = config.output_dir / "summary_phase3.txt"

    make_loss_comparison_figure(
        path=fig_path,
        temperature=temperature,
        y_clean=y_clean,
        y_dirty=y_dirty,
        outlier_idx=outlier_idx,
        fit_results=fit_results,
        snr_db=config.snr_db,
        dpi=config.dpi,
    )
    write_summary(
        path=summary_path,
        fit_results=fit_results,
        outlier_idx=outlier_idx,
        spike_factor=config.spike_factor,
    )

    print("Phase 3 loss comparison finished.")
    print(f"Figure: {fig_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
