#!/usr/bin/env python3
"""Phase 2 benchmark: parameter recovery and global solver stress test.

This script executes a reproducible benchmark for TLDecPy by comparing:

- ``global_hybrid_pso`` (PSO-seeded local fit)
- ``global_hybrid`` (DE-seeded local fit)

on a synthetic multimodel glow curve composed of:

1. ``fo_rb`` peak
2. ``fo_rb`` peak
3. ``go_kg`` (general-order) peak

Outputs are written to ``output/phase2_validation`` by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import permutations
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import NDArray

# Allow direct execution from repository root:
#   python scripts/phase2_benchmark_solvers.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldecpy.fit import solvers as fit_solvers  # noqa: E402
from tldecpy.models.registry import get_model  # noqa: E402
from tldecpy.schemas import FitOptions, PeakSpec, RobustOptions, UncertaintyOptions  # noqa: E402
from tldecpy.simulate.noise import add_noise  # noqa: E402

FloatArray = NDArray[np.float64]

# Reused visual standard (Radiation Measurements style).
# Keep this visual and naming standard for phases 2-5.
RM_PAPER_RCPARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
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

SOLVER_SPECS: tuple[tuple[str, str], ...] = (
    ("PSO", "global_hybrid_pso"),
    ("DE", "global_hybrid"),
)
LOCAL_OPTIMIZER = "trf"

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "fo_rb": "First-Order (Rational-Barycentric)",
    "go_kg": "General-Order (Kitis)",
    "so_ks": "Second-Order (Kinetic-Standard)",
    "otor_lw": "OTOR (Lambert W)",
}

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "Im": (100.0, 4000.0),
    "E": (0.5, 2.0),
    "Tm": (340.0, 540.0),
    "b": (1.01, 2.0),
}

TRUE_COMPONENTS: tuple[dict[str, Any], ...] = (
    {"name": "P1", "model": "fo_rb", "params": {"Im": 1000.0, "E": 0.9, "Tm": 400.0}},
    {"name": "P2", "model": "fo_rb", "params": {"Im": 1500.0, "E": 1.1, "Tm": 440.0}},
    {"name": "P3", "model": "go_kg", "params": {"Im": 1200.0, "E": 1.3, "Tm": 480.0, "b": 1.5}},
)


@dataclass(frozen=True)
class RunConfig:
    """Configuration container for phase-2 benchmark execution."""

    n_runs: int
    snr_db: float
    n_points: int
    t_min: float
    t_max: float
    beta: float
    output_dir: Path
    seed_data: int
    seed_init: int
    max_nfev: int
    dpi: int


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for synthetic TL dataset and clean components."""

    temperature: FloatArray
    y_clean: FloatArray
    components: dict[str, FloatArray]
    noise_sigma: float


def configure_matplotlib(dpi: int) -> None:
    """Apply plotting style for publication-ready figures."""
    style = dict(RM_PAPER_RCPARAMS)
    style.update({"figure.dpi": dpi, "savefig.dpi": dpi})
    plt.rcParams.update(style)


def build_cli() -> argparse.ArgumentParser:
    """Construct CLI parser."""
    parser = argparse.ArgumentParser(
        description="Phase 2 benchmark: PSO vs DE for TLDecPy multimodel recovery.",
    )
    parser.add_argument("--n-runs", type=int, default=100, help="Independent runs per solver.")
    parser.add_argument("--snr-db", type=float, default=30.0, help="Gaussian noise SNR in dB.")
    parser.add_argument("--n-points", type=int, default=700, help="Temperature grid size.")
    parser.add_argument("--t-min", type=float, default=330.0, help="Minimum temperature (K).")
    parser.add_argument("--t-max", type=float, default=560.0, help="Maximum temperature (K).")
    parser.add_argument("--beta", type=float, default=1.0, help="Heating rate beta (K/s).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/phase2_validation",
        help="Output directory for benchmark artifacts.",
    )
    parser.add_argument("--seed-data", type=int, default=20260215, help="Seed for synthetic noise.")
    parser.add_argument("--seed-init", type=int, default=20260216, help="Seed for initial guesses.")
    parser.add_argument(
        "--max-nfev", type=int, default=3000, help="Maximum local function evaluations."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Parse validated runtime configuration from argparse namespace."""
    return RunConfig(
        n_runs=int(args.n_runs),
        snr_db=float(args.snr_db),
        n_points=int(args.n_points),
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        beta=float(args.beta),
        output_dir=Path(args.output_dir),
        seed_data=int(args.seed_data),
        seed_init=int(args.seed_init),
        max_nfev=int(args.max_nfev),
        dpi=int(args.dpi),
    )


def _snr_noise_sigma(signal: FloatArray, snr_db: float) -> float:
    """Compute Gaussian sigma from peak-amplitude SNR definition."""
    peak_signal = float(np.max(np.abs(signal)))
    if peak_signal <= 0.0:
        return 0.0
    return float(peak_signal / (10.0 ** (snr_db / 20.0)))


def generate_synthetic_dataset(config: RunConfig) -> SyntheticDataset:
    """Generate multimodel synthetic TL curve and noise scale for Monte Carlo runs."""
    temperature = np.linspace(config.t_min, config.t_max, config.n_points, dtype=np.float64)
    components: dict[str, FloatArray] = {}
    y_clean = np.zeros_like(temperature)
    for component in TRUE_COMPONENTS:
        peak_name = str(component["name"])
        model_key = str(component["model"])
        model = get_model(model_key)
        y_peak = np.asarray(model(temperature, **dict(component["params"])), dtype=np.float64)
        components[peak_name] = y_peak
        y_clean = y_clean + y_peak
    sigma = _snr_noise_sigma(y_clean, config.snr_db)

    return SyntheticDataset(
        temperature=temperature,
        y_clean=y_clean,
        components=components,
        noise_sigma=sigma,
    )


def generate_noisy_observation(
    y_clean: FloatArray,
    sigma: float,
    seed: int,
) -> tuple[FloatArray, dict[str, float]]:
    """Generate one noisy realization from the clean synthetic curve."""
    y_obs = np.asarray(
        add_noise(y_clean, mode="gaussian", sigma=sigma, seed=seed), dtype=np.float64
    )
    noise = y_obs - y_clean
    stats = {
        "noise_mean": float(np.mean(noise)),
        "noise_std": float(np.std(noise, ddof=1 if noise.size > 1 else 0)),
        "noise_rms": float(np.sqrt(np.mean(np.square(noise)))),
    }
    return y_obs, stats


def _clip_to_bounds(value: float, pname: str) -> float:
    """Clip a candidate parameter value to configured physical bounds."""
    bound = PARAM_BOUNDS.get(pname)
    if bound is None:
        return float(value)
    return float(np.clip(value, bound[0], bound[1]))


def sample_initial_guess(rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Sample one blind-search initialization set (uniform ±25% for all parameters)."""
    init_by_peak: dict[str, dict[str, float]] = {}
    for component in TRUE_COMPONENTS:
        peak_name = str(component["name"])
        params_true = dict(component["params"])
        peak_init: dict[str, float] = {}
        for pname, pvalue in params_true.items():
            value_true = float(pvalue)
            guess = value_true * (1.0 + rng.uniform(-0.25, 0.25))
            peak_init[pname] = _clip_to_bounds(guess, pname)
        init_by_peak[peak_name] = peak_init
    return init_by_peak


def build_peak_specs(init_by_peak: dict[str, dict[str, float]]) -> list[PeakSpec]:
    """Build list of PeakSpec objects for the requested initialization."""
    specs: list[PeakSpec] = []
    for component in TRUE_COMPONENTS:
        name = str(component["name"])
        model = str(component["model"])
        init = dict(init_by_peak[name])
        bounds: dict[str, tuple[float, float]] = {}
        for pname in init:
            if pname in PARAM_BOUNDS:
                bounds[pname] = PARAM_BOUNDS[pname]
        specs.append(
            PeakSpec(
                name=name,
                model=model,
                init=init,
                bounds=bounds,
            )
        )
    return specs


def _matched_error_template_keys() -> list[str]:
    """Return CSV keys for matched-parameter relative errors."""
    keys: list[str] = []
    for component in TRUE_COMPONENTS:
        name = str(component["name"])
        for pname in dict(component["params"]):
            keys.append(f"matched_{name}_{pname}_rel_err")
    return keys


def _extract_fitted_peak_params(peak_results: list[Any]) -> list[dict[str, Any]]:
    """Normalize fit output peaks into a list of dicts with finite parameters."""
    extracted: list[dict[str, Any]] = []
    for peak in peak_results:
        params: dict[str, float] = {}
        for pname, pval in peak.params.items():
            value = float(pval)
            if np.isfinite(value):
                params[str(pname)] = value
        extracted.append(
            {
                "name": str(peak.name),
                "model": str(peak.model),
                "params": params,
            }
        )
    return extracted


def _match_indices_by_tm(
    truth_components: tuple[dict[str, Any], ...],
    fitted_components: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Match truth and fitted peaks via one-to-one minimum |Tm difference| assignment."""
    n_truth = len(truth_components)
    n_fit = len(fitted_components)
    if n_truth == 0 or n_fit == 0:
        return []

    fit_indices = list(range(n_fit))
    if n_fit < n_truth:
        return []

    best_perm: tuple[int, ...] | None = None
    best_score = float("inf")
    for perm in permutations(fit_indices, n_truth):
        score = 0.0
        for idx_true, idx_fit in enumerate(perm):
            tm_true = float(dict(truth_components[idx_true]["params"]).get("Tm", np.nan))
            tm_fit = float(fitted_components[idx_fit]["params"].get("Tm", np.nan))
            if not np.isfinite(tm_true) or not np.isfinite(tm_fit):
                score = float("inf")
                break
            score += abs(tm_true - tm_fit)
        if score < best_score:
            best_score = score
            best_perm = perm

    if best_perm is None:
        return []
    return [(idx_true, idx_fit) for idx_true, idx_fit in enumerate(best_perm)]


def calculate_matched_error(peak_results: list[Any]) -> tuple[dict[str, float], float, float]:
    """
    Compute matched relative errors after one-to-one Tm-based peak association.

    Notes
    -----
    This metric avoids false penalties caused by peak-index swapping in global solvers.
    """
    fitted_components = _extract_fitted_peak_params(peak_results)
    assignments = _match_indices_by_tm(TRUE_COMPONENTS, fitted_components)

    rel_errors: dict[str, float] = {key: float("nan") for key in _matched_error_template_keys()}
    for idx_true, idx_fit in assignments:
        true_component = TRUE_COMPONENTS[idx_true]
        fit_component = fitted_components[idx_fit]
        true_name = str(true_component["name"])
        true_params = dict(true_component["params"])
        fit_params = fit_component["params"]
        for pname, target in true_params.items():
            key = f"matched_{true_name}_{pname}_rel_err"
            estimate = float(fit_params.get(pname, np.nan))
            target_value = float(target)
            if np.isfinite(estimate) and target_value != 0.0:
                rel_errors[key] = float(abs((estimate - target_value) / target_value))

    finite_values = np.asarray(
        [val for val in rel_errors.values() if np.isfinite(val)],
        dtype=np.float64,
    )
    if finite_values.size == 0:
        return rel_errors, float("nan"), float("nan")
    return rel_errors, float(np.mean(finite_values)), float(np.max(finite_values))


def run_one_fit(
    temperature: FloatArray,
    y_obs: FloatArray,
    strategy: str,
    init_by_peak: dict[str, dict[str, float]],
    beta: float,
    max_nfev: int,
) -> tuple[dict[str, Any], Any | None]:
    """Run one fit with a specific global-seeding strategy."""
    robust = RobustOptions(
        loss="linear",
        weights="none",
        multi_start=0,
        ci_bootstrap=False,
        n_bootstrap=0,
    )
    options = FitOptions(
        local_optimizer=LOCAL_OPTIMIZER,
        max_nfev=max_nfev,
        uncertainty=UncertaintyOptions(enabled=False),
    )
    specs = build_peak_specs(init_by_peak)

    t_start = perf_counter()
    try:
        result = fit_solvers.fit_multi(
            temperature,
            y_obs,
            peaks=specs,
            bg=None,
            beta=beta,
            robust=robust,
            options=options,
            strategy=strategy,  # type: ignore[arg-type]
        )
        elapsed = perf_counter() - t_start

        rel_errors, rel_mean, rel_max = calculate_matched_error(result.peaks)
        record: dict[str, Any] = {
            "converged": bool(result.converged),
            "time_s": float(elapsed),
            "FOM": float(result.metrics.FOM),
            "R2": float(result.metrics.R2),
            "mean_matched_rel_error": rel_mean,
            "max_matched_rel_error": rel_max,
            "message": str(result.message),
        }
        record.update(rel_errors)
        return record, result
    except Exception as exc:  # pragma: no cover - benchmark script guard
        elapsed = perf_counter() - t_start
        record = {
            "converged": False,
            "time_s": float(elapsed),
            "FOM": float("nan"),
            "R2": float("nan"),
            "mean_matched_rel_error": float("nan"),
            "max_matched_rel_error": float("nan"),
            "message": f"{type(exc).__name__}: {exc}",
        }
        for key in _matched_error_template_keys():
            record[key] = float("nan")
        return record, None


def summarize_solver(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean/std summary metrics for one solver."""
    fom = np.asarray([r["FOM"] for r in records if np.isfinite(r["FOM"])], dtype=np.float64)
    time_s = np.asarray(
        [r["time_s"] for r in records if np.isfinite(r["time_s"])], dtype=np.float64
    )
    param_err = np.asarray(
        [r["mean_matched_rel_error"] for r in records if np.isfinite(r["mean_matched_rel_error"])],
        dtype=np.float64,
    )
    success = float(sum(1 for r in records if bool(r["converged"])))
    total = float(len(records))

    def _mean_std(values: FloatArray) -> tuple[float, float]:
        if values.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values, ddof=1 if values.size > 1 else 0))

    fom_mean, fom_std = _mean_std(fom)
    t_mean, t_std = _mean_std(time_s)
    p_mean, p_std = _mean_std(param_err)

    return {
        "n_total": total,
        "n_success": success,
        "success_rate_pct": 100.0 * success / total if total > 0 else 0.0,
        "fom_mean": fom_mean,
        "fom_std": fom_std,
        "time_mean": t_mean,
        "time_std": t_std,
        "matched_rel_mean": p_mean,
        "matched_rel_std": p_std,
    }


def save_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Save full run-by-run benchmark table as CSV."""
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in records for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def save_summary_text(
    path: Path, summary: dict[str, dict[str, float]], best_meta: dict[str, Any]
) -> None:
    """Persist human-readable summary report."""
    lines: list[str] = [
        "TLDecPy - Phase 2: Parameter Recovery and Global Benchmark",
        "",
    ]
    for solver_name in ("PSO", "DE"):
        stats = summary[solver_name]
        lines.extend(
            [
                f"[{solver_name}]",
                f"Runs: {int(stats['n_total'])}",
                f"Success: {int(stats['n_success'])} ({stats['success_rate_pct']:.1f}%)",
                f"FOM (%): {stats['fom_mean']:.4f} ± {stats['fom_std']:.4f}",
                f"Runtime (s): {stats['time_mean']:.4f} ± {stats['time_std']:.4f}",
                f"Mean matched relative error: {stats['matched_rel_mean']:.6f} ± {stats['matched_rel_std']:.6f}",
                "",
            ]
        )

    if best_meta:
        lines.extend(
            [
                "Best fit (minimum FOM across all runs):",
                f"Solver: {best_meta['solver']}",
                f"Run ID: {int(best_meta['run_id'])}",
                f"FOM (%): {best_meta['fom']:.4f}",
                f"Runtime (s): {best_meta['time_s']:.4f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _solver_boxplot_data(records: list[dict[str, Any]], key: str, solver: str) -> FloatArray:
    """Extract finite metric arrays for a solver from run records."""
    values = [float(r[key]) for r in records if r["solver"] == solver and np.isfinite(r[key])]
    return np.asarray(values, dtype=np.float64)


def model_display_name(model_key: str) -> str:
    """Return publication-friendly model label."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def create_benchmark_boxplots(
    path: Path,
    records: list[dict[str, Any]],
    dpi: int,
    n_runs: int,
) -> None:
    """Create 1x2 boxplot figure: FOM distribution and runtime distribution."""
    configure_matplotlib(dpi=dpi)

    fom_pso = _solver_boxplot_data(records, "FOM", "PSO")
    fom_de = _solver_boxplot_data(records, "FOM", "DE")
    t_pso = _solver_boxplot_data(records, "time_s", "PSO")
    t_de = _solver_boxplot_data(records, "time_s", "DE")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    color_pso = "#1f77b4"
    color_de = "#ff7f0e"

    tick_labels = [f"PSO (N={n_runs})", f"DE (N={n_runs})"]

    def _draw_panel(ax: plt.Axes, values_a: FloatArray, values_b: FloatArray, ylabel: str) -> None:
        box = ax.boxplot(
            [values_a, values_b],
            tick_labels=tick_labels,
            notch=True,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], (color_pso, color_de)):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.9)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.3, alpha=0.5)

    _draw_panel(axes[0], fom_pso, fom_de, "FOM (%)")
    _draw_panel(axes[1], t_pso, t_de, "Runtime (s)")

    axes[0].text(0.03, 0.95, "A", transform=axes[0].transAxes, fontweight="bold", va="top")
    axes[1].text(0.03, 0.95, "B", transform=axes[1].transAxes, fontweight="bold", va="top")

    legend_handles = [
        Patch(facecolor=color_pso, edgecolor="black", alpha=0.65, label="PSO"),
        Patch(facecolor=color_de, edgecolor="black", alpha=0.65, label="DE"),
    ]
    fig.legend(
        handles=legend_handles,
        labels=["PSO", "DE"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=8,
        frameon=True,
        fancybox=False,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_best_fit_figure(
    path: Path,
    temperature: FloatArray,
    y_obs: FloatArray,
    best_result: Any,
    best_meta: dict[str, Any],
    snr_db: float,
    dpi: int,
) -> None:
    """Create best-fit visualization with residual panel."""
    configure_matplotlib(dpi=dpi)

    fig = plt.figure(figsize=(7.0, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.06)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

    ax_main.plot(
        temperature,
        y_obs,
        linestyle="none",
        marker="o",
        markersize=2.6,
        color="0.3",
        alpha=0.55,
        label=f"Synthetic data (SNR={snr_db:.0f} dB)",
    )

    peak_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for idx, peak in enumerate(best_result.peaks):
        peak_label = f"Peak {idx + 1}: {model_display_name(str(peak.model))}"
        ax_main.plot(
            temperature,
            np.asarray(peak.y_hat, dtype=np.float64),
            linestyle="--",
            linewidth=1.2,
            color=peak_colors[idx % len(peak_colors)],
            label=peak_label,
        )

    ax_main.plot(
        temperature,
        np.asarray(best_result.y_hat_total, dtype=np.float64),
        color="black",
        linewidth=2.0,
        label="Total fit",
    )
    ax_main.set_ylabel("Intensity (a.u.)")
    ax_main.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax_main.legend(
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
    )

    annotation = (
        f"Best solver: {best_meta['solver']}\n"
        f"Run: {int(best_meta['run_id'])}\n"
        f"FOM: {best_meta['fom']:.4f}%"
    )
    ax_main.text(
        0.015,
        0.97,
        annotation,
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 1.0, "pad": 2.0},
    )

    residuals = np.asarray(best_result.residuals, dtype=np.float64)
    ax_res.axhline(0.0, color="black", linewidth=0.9)
    ax_res.plot(
        temperature,
        residuals,
        linestyle="none",
        marker="o",
        markersize=2.2,
        color="#7f3b08",
        alpha=0.75,
    )
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel("Temperature, T (K)")
    ax_res.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)

    plt.setp(ax_main.get_xticklabels(), visible=False)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _sorted_truth_components(
    true_components: dict[str, FloatArray],
) -> list[tuple[str, float, FloatArray]]:
    """Return ground-truth components sorted by Tm."""
    sorted_list: list[tuple[str, float, FloatArray]] = []
    for component in TRUE_COMPONENTS:
        name = str(component["name"])
        tm = float(dict(component["params"]).get("Tm", np.nan))
        y_peak = np.asarray(true_components[name], dtype=np.float64)
        sorted_list.append((name, tm, y_peak))
    sorted_list.sort(key=lambda item: item[1])
    return sorted_list


def _sorted_result_components(result: Any) -> list[tuple[str, float, FloatArray]]:
    """Return fitted component curves sorted by fitted Tm."""
    sorted_list: list[tuple[str, float, FloatArray]] = []
    for peak in result.peaks:
        tm = float(peak.params.get("Tm", np.nan))
        y_peak = np.asarray(peak.y_hat, dtype=np.float64)
        sorted_list.append((str(peak.name), tm, y_peak))
    sorted_list.sort(key=lambda item: item[1] if np.isfinite(item[1]) else float("inf"))
    return sorted_list


def create_solver_comparison_figure(
    path: Path,
    temperature: FloatArray,
    y_obs: FloatArray,
    y_truth: FloatArray,
    true_components: dict[str, FloatArray],
    best_pso_result: Any,
    best_de_result: Any,
    snr_db: float,
    dpi: int,
) -> None:
    """Create three-panel comparison for data, components and total-fit equifinality."""
    configure_matplotlib(dpi=dpi)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
    ax_data, ax_components, ax_total = axes

    color_pso = "#1f77b4"
    color_de = "#ff7f0e"

    # Panel A: noisy observation + total ground truth.
    ax_data.plot(
        temperature,
        y_obs,
        linestyle="none",
        marker="o",
        markersize=2.5,
        color="0.4",
        alpha=0.45,
        label=f"Synthetic data (SNR={snr_db:.0f} dB)",
    )
    ax_data.plot(
        temperature,
        y_truth,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="Ground Truth (Total)",
    )
    ax_data.set_ylabel("Data")
    ax_data.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax_data.legend(loc="upper right", fontsize=7.5, frameon=True, fancybox=False, framealpha=1.0)
    ax_data.text(0.02, 0.95, "A", transform=ax_data.transAxes, fontweight="bold", va="top")

    # Panel B: individual components, ordered by Tm (critical for DE visualization).
    truth_sorted = _sorted_truth_components(true_components)
    pso_sorted = _sorted_result_components(best_pso_result)
    de_sorted = _sorted_result_components(best_de_result)
    n_components = min(len(truth_sorted), len(pso_sorted), len(de_sorted))

    for idx in range(n_components):
        peak_idx = idx + 1
        truth_curve = truth_sorted[idx][2]
        pso_curve = pso_sorted[idx][2]
        de_curve = de_sorted[idx][2]
        ax_components.plot(
            temperature,
            truth_curve,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"Peak {peak_idx} Truth" if idx == 0 else "_nolegend_",
        )
        ax_components.plot(
            temperature,
            pso_curve,
            color=color_pso,
            linestyle="-",
            linewidth=1.0,
            label=f"Peak {peak_idx} PSO" if idx == 0 else "_nolegend_",
        )
        ax_components.plot(
            temperature,
            de_curve,
            color=color_de,
            linestyle="-",
            linewidth=1.0,
            label=f"Peak {peak_idx} DE" if idx == 0 else "_nolegend_",
        )

    ax_components.set_ylabel("Peaks")
    ax_components.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax_components.legend(
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
    )
    ax_components.text(
        0.02,
        0.95,
        "B",
        transform=ax_components.transAxes,
        fontweight="bold",
        va="top",
    )

    # Panel C: total-fit comparison (equifinality check).
    ax_total.plot(
        temperature,
        y_truth,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="Ground Truth (Total)",
    )
    ax_total.plot(
        temperature,
        np.asarray(best_pso_result.y_hat_total, dtype=np.float64),
        color=color_pso,
        linestyle="-",
        linewidth=1.2,
        label="Best PSO Fit",
    )
    ax_total.plot(
        temperature,
        np.asarray(best_de_result.y_hat_total, dtype=np.float64),
        color=color_de,
        linestyle="-",
        linewidth=1.2,
        label="Best DE Fit",
    )
    ax_total.set_ylabel("Total Fit")
    ax_total.set_xlabel("Temperature, T (K)")
    ax_total.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax_total.legend(loc="upper right", fontsize=7.5, frameon=True, fancybox=False, framealpha=1.0)
    ax_total.text(0.02, 0.95, "C", transform=ax_total.transAxes, fontweight="bold", va="top")

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Execute the complete phase-2 benchmark workflow."""
    args = build_cli().parse_args()
    config = parse_config(args)

    if config.n_runs <= 0:
        raise ValueError("n-runs must be > 0.")
    if config.t_max <= config.t_min:
        raise ValueError("t-max must be greater than t-min.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_dataset(config)
    print(f"[phase2] Local optimizer for hybrid refinement: {LOCAL_OPTIMIZER}")
    print(
        "[phase2] Global strategies: "
        + ", ".join(f"{solver}={strategy}" for solver, strategy in SOLVER_SPECS)
    )

    noise_rng = np.random.default_rng(config.seed_data)
    run_rng = np.random.default_rng(config.seed_init)
    records: list[dict[str, Any]] = []
    best_result = None
    best_y_obs: FloatArray | None = None
    best_meta: dict[str, Any] = {}
    best_by_solver: dict[str, dict[str, Any]] = {}

    for run_id in range(1, config.n_runs + 1):
        # Monte Carlo: regenerate observation noise for every run.
        noise_seed = int(noise_rng.integers(0, np.iinfo(np.uint32).max))
        y_obs_run, noise_stats = generate_noisy_observation(
            y_clean=dataset.y_clean,
            sigma=dataset.noise_sigma,
            seed=noise_seed,
        )
        init_guess = sample_initial_guess(run_rng)
        for solver_name, strategy in SOLVER_SPECS:
            record, result = run_one_fit(
                temperature=dataset.temperature,
                y_obs=y_obs_run,
                strategy=strategy,
                init_by_peak=init_guess,
                beta=config.beta,
                max_nfev=config.max_nfev,
            )

            row: dict[str, Any] = {
                "run_id": run_id,
                "solver": solver_name,
                "strategy": strategy,
                "snr_db": config.snr_db,
                "noise_sigma": dataset.noise_sigma,
                "noise_seed": noise_seed,
                "noise_mean": noise_stats["noise_mean"],
                "noise_std": noise_stats["noise_std"],
                "noise_rms": noise_stats["noise_rms"],
            }
            row.update(record)
            records.append(row)

            if result is not None and np.isfinite(record["FOM"]):
                current_solver_best = best_by_solver.get(solver_name)
                if current_solver_best is None or float(record["FOM"]) < float(
                    current_solver_best["fom"]
                ):
                    best_by_solver[solver_name] = {
                        "solver": solver_name,
                        "run_id": run_id,
                        "fom": float(record["FOM"]),
                        "time_s": float(record["time_s"]),
                        "result": result,
                        "y_obs": np.asarray(y_obs_run, dtype=np.float64).copy(),
                    }

                if best_result is None or float(record["FOM"]) < float(best_meta["fom"]):
                    best_result = result
                    best_y_obs = np.asarray(y_obs_run, dtype=np.float64).copy()
                    best_meta = {
                        "solver": solver_name,
                        "run_id": run_id,
                        "fom": float(record["FOM"]),
                        "time_s": float(record["time_s"]),
                    }

    records_csv = config.output_dir / "benchmark_runs.csv"
    summary_txt = config.output_dir / "summary_phase2.txt"
    fig_box_pdf = config.output_dir / "phase2_benchmark_boxplots.pdf"
    fig_best_pdf = config.output_dir / "phase2_best_fit_with_residuals.pdf"
    fig_compare_pdf = config.output_dir / "phase2_solver_comparison.pdf"

    save_records_csv(records_csv, records)

    solver_summary = {
        "PSO": summarize_solver([r for r in records if r["solver"] == "PSO"]),
        "DE": summarize_solver([r for r in records if r["solver"] == "DE"]),
    }
    save_summary_text(summary_txt, solver_summary, best_meta)

    create_benchmark_boxplots(fig_box_pdf, records, dpi=config.dpi, n_runs=config.n_runs)

    if best_result is not None and best_y_obs is not None:
        create_best_fit_figure(
            path=fig_best_pdf,
            temperature=dataset.temperature,
            y_obs=best_y_obs,
            best_result=best_result,
            best_meta=best_meta,
            snr_db=config.snr_db,
            dpi=config.dpi,
        )

    if (
        "PSO" in best_by_solver
        and "DE" in best_by_solver
        and best_y_obs is not None
        and best_result is not None
    ):
        create_solver_comparison_figure(
            path=fig_compare_pdf,
            temperature=dataset.temperature,
            y_obs=best_y_obs,
            y_truth=dataset.y_clean,
            true_components=dataset.components,
            best_pso_result=best_by_solver["PSO"]["result"],
            best_de_result=best_by_solver["DE"]["result"],
            snr_db=config.snr_db,
            dpi=config.dpi,
        )

    print("Phase 2 benchmark finished.")
    print(f"Records CSV: {records_csv}")
    print(f"Summary: {summary_txt}")
    print(f"Figure (boxplots): {fig_box_pdf}")
    print(f"Figure (best fit): {fig_best_pdf}")
    print(f"Figure (solver comparison): {fig_compare_pdf}")
    if best_meta:
        print(
            f"Best run -> solver={best_meta['solver']}, run={best_meta['run_id']}, "
            f"FOM={best_meta['fom']:.4f}%"
        )


if __name__ == "__main__":
    main()
