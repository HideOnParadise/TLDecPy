#!/usr/bin/env python3
"""Phase 4 Refglow benchmark (hybrid model plan: x001/x002 + x005).

Scope requested for this run
----------------------------
- x001, x002: fit only with fo_ka (Kinetic-Asymptotic approximation).
- x005: fit only with fo_wp (Weibull-Peak approximation) using guided seeds.
- Outputs:
  - phase4_results_x001.csv, phase4_results_x002.csv, phase4_results_x005.csv
  - fit + residual PDFs for the 3 curves
  - aggregated phase4_model_results.csv, phase4_peak_params_long.csv
  - summary_phase4.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldecpy.data.refglow import load_refglow, resolve_refglow_path  # noqa: E402
from tldecpy.fit.multi import fit_multi  # noqa: E402
from tldecpy.schemas import (  # noqa: E402
    BackgroundSpec,
    FitOptions,
    MultiFitResult,
    PeakSpec,
    RobustOptions,
    UncertaintyOptions,
)
from tldecpy.utils.provenance import file_sha256_hex  # noqa: E402

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

# GLOCANIN truth entries used in this benchmark slice.
GLOCANIN_TRUTH: dict[str, dict[str, Any]] = {
    "x001": {
        "beta": 1.0,
        "peaks": [
            {"id": 1, "T_m": 490.5, "I_m": 11000.0, "E": 1.1824, "s": 8.0573e10},
        ],
    },
    "x002": {
        "beta": 8.4,
        "peaks": [
            {"id": 2, "T_m": 417.2, "I_m": 400.0, "E": 1.3834, "s": 3.9810e16},
            {"id": 3, "T_m": 456.6, "I_m": 540.0, "E": 1.4883, "s": 1.6401e16},
            {"id": 4, "T_m": 484.1, "I_m": 820.0, "E": 1.5832, "s": 2.0054e16},
            {"id": 5, "T_m": 511.7, "I_m": 1620.0, "E": 2.0038, "s": 4.0695e19},
        ],
    },
    "x005": {
        "beta": 1.0,
        "peaks": [
            {"id": 1, "T_m": 358.0, "I_m": 20.0, "E": 1.18, "s": 1.9e16},
            {"id": 2, "T_m": 415.2, "I_m": 169.0, "E": 1.48, "s": 4.3e17},
            {"id": 3, "T_m": 453.8, "I_m": 154.0, "E": 1.54, "s": 5.5e16},
            {"id": 4, "T_m": 482.0, "I_m": 237.0, "E": 1.54, "s": 4.3e15},
            {"id": 5, "T_m": 509.3, "I_m": 484.0, "E": 2.15, "s": 8.7e20},
        ],
    },
}

# Guided fo_wp seeds (TLDec Web) for x005 to lock onto the known optimum basin.
X005_FO_WP_GUIDED_SEEDS: tuple[dict[str, float], ...] = (
    {"id": 1.0, "T_m": 358.2, "I_m": 19.9, "E": 1.25},
    {"id": 2.0, "T_m": 415.3, "I_m": 169.0, "E": 1.49},
    {"id": 3.0, "T_m": 453.6, "I_m": 154.4, "E": 1.57},
    {"id": 4.0, "T_m": 482.2, "I_m": 237.3, "E": 1.50},
    {"id": 5.0, "T_m": 509.4, "I_m": 484.2, "E": 2.25},
)

CURVE_PLAN: dict[str, dict[str, Any]] = {
    "x001": {"model": "fo_ka", "preprocess_mode": "none", "bg": None},
    "x002": {"model": "fo_ka", "preprocess_mode": "none", "bg": None},
    # Linear background is enabled for x005 to match experimental baseline drift
    # and recover the validated <1% FOM regime.
    "x005": {"model": "fo_wp", "preprocess_mode": "none", "bg": "linear"},
}

MODEL_DISPLAY: dict[str, str] = {
    "fo_ka": "First-order peak (Kinetic-Asymptotic approximation)",
    "fo_wp": "First-order peak (Weibull-Peak approximation)",
}

E_BOUNDS = (0.5, 5.0)
IM_SCALE_LO = 0.05
IM_SCALE_HI = 20.0
TM_HALF_WIDTH_FO_KA_K = 15.0
TM_HALF_WIDTH_FO_WP_K = 8.0


@dataclass(frozen=True)
class RunConfig:
    """Runtime settings."""

    output_dir: Path
    strategy: str
    max_nfev: int
    dpi: int


@dataclass(frozen=True)
class FitArtifacts:
    """Fit output and diagnostics."""

    result: MultiFitResult
    runtime_s: float
    fom_proc: float
    fom_raw: float
    uc_curve: FloatArray
    warning_count: int


def build_cli() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Phase 4 Refglow benchmark (x001/x002/x005).")
    parser.add_argument("--output-dir", type=str, default="output/phase4_validation")
    parser.add_argument(
        "--strategy",
        type=str,
        default="global_hybrid_pso",
        choices=["local", "global_hybrid", "global_hybrid_pso"],
    )
    parser.add_argument("--max-nfev", type=int, default=4000)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate and normalize runtime settings."""
    output_dir = Path(str(args.output_dir))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return RunConfig(
        output_dir=output_dir,
        strategy=str(args.strategy),
        max_nfev=max(int(args.max_nfev), 400),
        dpi=int(args.dpi),
    )


def configure_matplotlib(dpi: int) -> None:
    """Apply project paper style."""
    style = dict(RM_PAPER_RCPARAMS)
    style.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )
    plt.rcParams.update(style)


def _safe_float(value: Any) -> float | None:
    """Return finite float when available."""
    if value is None:
        return None
    try:
        f_val = float(value)
    except (TypeError, ValueError):
        return None
    return f_val if np.isfinite(f_val) else None


def parse_peak_id(peak_name: str, fallback: int) -> int:
    """Extract numeric peak id from names like P1, P5."""
    match = re.search(r"(\d+)", peak_name)
    return int(match.group(1)) if match else fallback


def format_refglow_label(curve_id: str) -> str:
    """Format curve title as REFGLOW.00N."""
    return f"REFGLOW.{int(curve_id[1:]):03d}"


def calculate_fom(y_obs: FloatArray, y_fit: FloatArray) -> float:
    """Compute FOM (%)."""
    y_o = np.asarray(y_obs, dtype=np.float64)
    y_h = np.asarray(y_fit, dtype=np.float64)
    if y_o.size == 0 or y_h.size != y_o.size:
        return float("inf")
    area = float(np.sum(y_o))
    if area <= 0.0 or not np.isfinite(area):
        return float("inf")
    fom = 100.0 * float(np.sum(np.abs(y_o - y_h))) / area
    return float(fom) if np.isfinite(fom) else float("inf")


def count_warnings(result: MultiFitResult) -> int:
    """Count warning-like diagnostics for summary logging."""
    count = 0
    if not bool(result.converged):
        count += 1
    count += int(sum(1 for hit in result.hit_bounds.values() if bool(hit)))
    msg = str(result.message).lower()
    for token in ("warn", "fail", "error", "bound"):
        if token in msg:
            count += 1
    return count


def build_fit_options(max_nfev: int) -> FitOptions:
    """Build deterministic fit options with uncertainty enabled."""
    uncertainty = UncertaintyOptions(
        enabled=True,
        include_parameter_covariance=True,
        noise_from_residuals=True,
        noise_pct=0.0,
        calibration_pct=0.0,
        heating_rate_pct=0.0,
        reader_drift_pct=0.0,
        export_report=True,
        validation_mode="none",
    )
    return FitOptions(local_optimizer="trf", max_nfev=max_nfev, uncertainty=uncertainty)


def build_peak_specs(curve_id: str, model_key: str, temperature: FloatArray) -> list[PeakSpec]:
    """Build peak specs from truth / guided seeds without using truth s as seed."""
    t_min = float(np.min(temperature))
    t_max = float(np.max(temperature))

    if curve_id == "x005" and model_key == "fo_wp":
        peak_table: list[dict[str, float]] = [dict(seed) for seed in X005_FO_WP_GUIDED_SEEDS]
        tm_half_width = TM_HALF_WIDTH_FO_WP_K
    else:
        peak_table = [dict(p) for p in GLOCANIN_TRUTH[curve_id]["peaks"]]
        tm_half_width = TM_HALF_WIDTH_FO_KA_K

    specs: list[PeakSpec] = []
    for peak in peak_table:
        peak_id = int(peak["id"])
        tm = float(peak["T_m"])
        im = float(peak["I_m"])
        e_val = float(peak["E"])

        tm_lo = max(t_min + 1e-6, tm - tm_half_width)
        tm_hi = min(t_max - 1e-6, tm + tm_half_width)
        if tm_hi <= tm_lo:
            tm_lo = max(t_min + 1e-6, tm - 3.0)
            tm_hi = min(t_max - 1e-6, tm + 3.0)
        if tm_hi <= tm_lo:
            tm_lo = t_min + 1e-3
            tm_hi = t_max - 1e-3

        specs.append(
            PeakSpec(
                name=f"P{peak_id}",
                model=model_key,
                init={"Tm": tm, "Im": im, "E": e_val},
                bounds={
                    "Tm": (tm_lo, tm_hi),
                    "Im": (max(1e-9, IM_SCALE_LO * im), max(10.0, IM_SCALE_HI * im)),
                    "E": E_BOUNDS,
                },
            )
        )
    return specs


def run_curve_fit(
    curve_id: str,
    model_key: str,
    temperature: FloatArray,
    intensity_raw: FloatArray,
    beta: float,
    bg_mode: str | None,
    *,
    strategy: str,
    max_nfev: int,
) -> FitArtifacts:
    """Run one fit and return metrics and diagnostics."""
    robust = RobustOptions(
        loss="linear",
        f_scale=1.0,
        weights="none",
        multi_start=0,
        ci_bootstrap=False,
        n_bootstrap=0,
    )
    options = build_fit_options(max_nfev=max_nfev)
    specs = build_peak_specs(curve_id=curve_id, model_key=model_key, temperature=temperature)
    bg_spec = None if bg_mode is None else BackgroundSpec(type=bg_mode)

    start = perf_counter()
    result = fit_multi(
        temperature,
        intensity_raw,
        peaks=specs,
        bg=bg_spec,
        beta=beta,
        robust=robust,
        options=options,
        strategy=strategy,  # type: ignore[arg-type]
    )
    runtime_s = float(perf_counter() - start)

    y_hat = np.asarray(result.y_hat_total, dtype=np.float64)
    fom_proc = calculate_fom(intensity_raw, y_hat)
    fom_raw = calculate_fom(intensity_raw, y_hat)

    if result.uc_curve is None:
        uc_curve = np.full_like(intensity_raw, np.nan, dtype=np.float64)
    else:
        uc_curve = np.asarray(result.uc_curve, dtype=np.float64)
        if uc_curve.size != intensity_raw.size:
            uc_curve = np.full_like(intensity_raw, np.nan, dtype=np.float64)

    return FitArtifacts(
        result=result,
        runtime_s=runtime_s,
        fom_proc=fom_proc,
        fom_raw=fom_raw,
        uc_curve=uc_curve,
        warning_count=count_warnings(result),
    )


def uc_at_temperature(temperature: FloatArray, uc_curve: FloatArray, t_value: float) -> float | None:
    """Interpolate u_c(T) at Tm."""
    if uc_curve.size != temperature.size:
        return None
    if not np.any(np.isfinite(uc_curve)):
        return None
    t_min = float(np.min(temperature))
    t_max = float(np.max(temperature))
    if not (t_min <= t_value <= t_max):
        return None
    uc_val = float(np.interp(t_value, temperature, uc_curve))
    return uc_val if np.isfinite(uc_val) else None


def plot_curve_result(
    output_path: Path,
    curve_id: str,
    model_display: str,
    beta: float,
    temperature: FloatArray,
    y_raw: FloatArray,
    fit_pack: FitArtifacts,
    dpi: int,
) -> None:
    """Create two-panel fit and residual figure."""
    configure_matplotlib(dpi=dpi)

    y_hat = np.asarray(fit_pack.result.y_hat_total, dtype=np.float64)
    residual = y_raw - y_hat
    curve_label = format_refglow_label(curve_id)

    fig = plt.figure(figsize=(6.8, 4.8), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

    ax_main.plot(
        temperature,
        y_raw,
        linestyle="none",
        marker="o",
        markersize=2.1,
        color="0.55",
        alpha=0.65,
        label="Data",
    )
    ax_main.plot(temperature, y_hat, color="black", linewidth=2.4, label="Total fit")

    cmap = plt.get_cmap("tab10")
    for idx, peak in enumerate(fit_pack.result.peaks, start=1):
        color = cmap((idx - 1) % 10)
        peak_y = np.asarray(peak.y_hat, dtype=np.float64)
        peak_id = parse_peak_id(peak.name, fallback=idx)
        ax_main.plot(
            temperature,
            peak_y,
            linestyle="--",
            linewidth=1.4,
            color=color,
            label=f"Peak {peak_id}",
        )

    uc_global = _safe_float(fit_pack.result.metrics.uc_global)
    uc_p95 = _safe_float(fit_pack.result.metrics.uc_p95)
    uc_global_math = "n/a" if uc_global is None else f"{uc_global:.3f}\\%"
    uc_p95_math = "n/a" if uc_p95 is None else f"{uc_p95:.3f}\\%"
    info = [
        curve_label,
        model_display,
        rf"$\beta={beta:.1f}\ \mathrm{{K\,s^{{-1}}}}$",
        rf"$\mathrm{{FOM}}={fit_pack.fom_proc:.4f}\%$",
        rf"$u_{{c,\mathrm{{global}}}}={uc_global_math}$",
        rf"$u_{{c,\mathrm{{p95}}}}={uc_p95_math}$",
    ]
    ax_main.text(
        0.02,
        0.98,
        "\n".join(info),
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.2},
    )

    ax_main.set_ylabel("Intensity (a.u.)")
    ax_main.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.3)
    ax_main.legend(loc="upper right", fontsize=7.2)

    ax_res.plot(
        temperature,
        residual,
        linestyle="none",
        marker="o",
        markersize=1.9,
        color="#8c4b12",
        alpha=0.75,
    )
    ax_res.axhline(0.0, color="black", linewidth=0.9)
    res_q99 = float(np.nanpercentile(np.abs(residual), 99))
    if not np.isfinite(res_q99) or res_q99 <= 0.0:
        res_q99 = float(np.nanmax(np.abs(residual))) if residual.size > 0 else 1.0
    if not np.isfinite(res_q99) or res_q99 <= 0.0:
        res_q99 = 1.0
    ax_res.set_ylim(-1.1 * res_q99, 1.1 * res_q99)
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel("Temperature, T (K)")
    ax_res.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.3)

    plt.setp(ax_main.get_xticklabels(), visible=False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    """Run benchmark for x001/x002/x005."""
    config = parse_config(build_cli().parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for curve_id, plan in CURVE_PLAN.items():
        model_key = str(plan["model"])
        model_display = MODEL_DISPLAY[model_key]
        preprocess_mode = str(plan["preprocess_mode"])
        bg_mode = plan["bg"]
        beta = float(GLOCANIN_TRUTH[curve_id]["beta"])

        data_file = resolve_refglow_path(curve_id)
        data_file_sha256 = file_sha256_hex(data_file)
        temperature, intensity = load_refglow(curve_id)
        t_arr = np.asarray(temperature, dtype=np.float64)
        i_raw = np.asarray(intensity, dtype=np.float64)

        fit_pack = run_curve_fit(
            curve_id=curve_id,
            model_key=model_key,
            temperature=t_arr,
            intensity_raw=i_raw,
            beta=beta,
            bg_mode=bg_mode,
            strategy=config.strategy,
            max_nfev=config.max_nfev,
        )

        uc_global = _safe_float(fit_pack.result.metrics.uc_global)
        uc_p95 = _safe_float(fit_pack.result.metrics.uc_p95)
        uc_max = _safe_float(fit_pack.result.metrics.uc_max)

        wide_rows.append(
            {
                "curve": curve_id,
                "model": model_key,
                "model_display": model_display,
                "data_file": str(data_file),
                "data_file_sha256": data_file_sha256,
                "data_hash_algorithm": "sha256",
                "beta": beta,
                "preprocess_mode": preprocess_mode,
                "bg_mode": "none" if bg_mode is None else str(bg_mode),
                "FOM_proc": fit_pack.fom_proc,
                "FOM_raw": fit_pack.fom_raw,
                "uc_global": uc_global,
                "uc_p95": uc_p95,
                "uc_max": uc_max,
                "runtime_s": fit_pack.runtime_s,
                "converged": bool(fit_pack.result.converged),
                "warning_count": int(fit_pack.warning_count),
                "notes": str(fit_pack.result.message),
            }
        )

        truth_by_id = {int(p["id"]): p for p in GLOCANIN_TRUTH[curve_id]["peaks"]}
        per_curve_rows: list[dict[str, Any]] = []

        for idx, peak in enumerate(fit_pack.result.peaks, start=1):
            peak_id = parse_peak_id(peak.name, fallback=idx)
            params = dict(peak.params)
            tm_fit = _safe_float(params.get("Tm"))
            uc_tm = None if tm_fit is None else uc_at_temperature(t_arr, fit_pack.uc_curve, tm_fit)
            truth = truth_by_id.get(peak_id)

            row = {
                "curve": curve_id,
                "Model": model_key,
                "Model_Display": model_display,
                "Peak_ID": peak_id,
                "Peak_Name": peak.name,
                "E": _safe_float(params.get("E")),
                "Tm": tm_fit,
                "Im": _safe_float(params.get("Im")),
                "s_fit": _safe_float(params.get("s")),
                "u_c_at_Tm": uc_tm,
                "FOM": fit_pack.fom_proc,
                "Time": fit_pack.runtime_s,
                "Converged": bool(fit_pack.result.converged),
                "truth_Tm": None if truth is None else float(truth["T_m"]),
                "truth_Im": None if truth is None else float(truth["I_m"]),
                "truth_E": None if truth is None else float(truth["E"]),
                "truth_s": None if truth is None else float(truth["s"]),
            }
            per_curve_rows.append(row)

            long_rows.append(
                {
                    "curve": curve_id,
                    "model": model_key,
                    "model_display": model_display,
                    "beta": beta,
                    "peak_id": peak_id,
                    "peak_name": peak.name,
                    "Tm_fit": tm_fit,
                    "Im_fit": _safe_float(params.get("Im")),
                    "E_fit": _safe_float(params.get("E")),
                    "s_fit": _safe_float(params.get("s")),
                    "u_c_at_Tm": uc_tm,
                    "unc_Tm": _safe_float(peak.uncertainties.get("Tm")),
                    "unc_Im": _safe_float(peak.uncertainties.get("Im")),
                    "unc_E": _safe_float(peak.uncertainties.get("E")),
                    "truth_Tm": None if truth is None else float(truth["T_m"]),
                    "truth_Im": None if truth is None else float(truth["I_m"]),
                    "truth_E": None if truth is None else float(truth["E"]),
                    "truth_s": None if truth is None else float(truth["s"]),
                    "FOM_proc": fit_pack.fom_proc,
                    "FOM_raw": fit_pack.fom_raw,
                    "runtime_s": fit_pack.runtime_s,
                    "converged": bool(fit_pack.result.converged),
                }
            )

        curve_csv = config.output_dir / f"phase4_results_{curve_id}.csv"
        pd.DataFrame(per_curve_rows).sort_values("Peak_ID").to_csv(curve_csv, index=False)

        fig_pdf = config.output_dir / f"phase4_{curve_id}_{model_key}_fit_residual.pdf"
        plot_curve_result(
            output_path=fig_pdf,
            curve_id=curve_id,
            model_display=model_display,
            beta=beta,
            temperature=t_arr,
            y_raw=i_raw,
            fit_pack=fit_pack,
            dpi=config.dpi,
        )

    wide_df = pd.DataFrame(wide_rows).sort_values("curve").reset_index(drop=True)
    long_df = pd.DataFrame(long_rows).sort_values(["curve", "peak_id"]).reset_index(drop=True)
    model_csv = config.output_dir / "phase4_model_results.csv"
    peaks_csv = config.output_dir / "phase4_peak_params_long.csv"
    summary_txt = config.output_dir / "summary_phase4.txt"

    wide_df.to_csv(model_csv, index=False)
    long_df.to_csv(peaks_csv, index=False)

    x005_row = wide_df.loc[wide_df["curve"] == "x005"].iloc[0]
    x005_fom = float(x005_row["FOM_proc"])
    x005_status = "PASS (<1%)" if np.isfinite(x005_fom) and x005_fom < 1.0 else "CHECK (>=1%)"

    lines = [
        "PHASE4_REFGLOW_BENCHMARK",
        "",
        "Model plan:",
        "- x001 -> fo_ka (synthetic)",
        "- x002 -> fo_ka (synthetic)",
        "- x005 -> fo_wp (experimental TLD-100, guided seeds)",
        "",
        f"Strategy: {config.strategy}",
        f"Max evaluations: {config.max_nfev}",
        "",
        "Per-curve results:",
    ]
    for _, row in wide_df.iterrows():
        lines.append(
            f"- {row['curve']}: model={row['model']}, beta={float(row['beta']):.1f}, "
            f"FOM={float(row['FOM_proc']):.6f}%, "
            f"sha256={row.get('data_file_sha256', 'n/a')}"
        )
    lines.extend(
        [
            "",
            f"x005 experimental validation: FOM={x005_fom:.6f}% -> {x005_status}",
            "",
            "Outputs:",
            f"- {model_csv}",
            f"- {peaks_csv}",
            f"- {config.output_dir / 'phase4_results_x001.csv'}",
            f"- {config.output_dir / 'phase4_results_x002.csv'}",
            f"- {config.output_dir / 'phase4_results_x005.csv'}",
            f"- {config.output_dir / 'phase4_x001_fo_ka_fit_residual.pdf'}",
            f"- {config.output_dir / 'phase4_x002_fo_ka_fit_residual.pdf'}",
            f"- {config.output_dir / 'phase4_x005_fo_wp_fit_residual.pdf'}",
        ]
    )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Phase 4 benchmark completed.")
    for _, row in wide_df.iterrows():
        print(
            f"{row['curve']}: model={row['model']} | FOM={float(row['FOM_proc']):.6f}% | "
            f"beta={float(row['beta']):.1f} K/s"
        )
    print(f"x005 target check: {x005_status}")
    print(f"Output dir: {config.output_dir}")


if __name__ == "__main__":
    main()
