#!/usr/bin/env python3
"""Phase 7 uncertainty validation using TLD-100 experimental data.

Final validation setup:
- Raw data in a tight ROI centered on the dosimetric complex (P2/P3/P4): 165-260 °C.
- Three first-order peaks using the fo_ka (Kinetic-Asymptotic) approximation.
- Analytic uncertainty from Jacobian/covariance propagation (Hessian-equivalent).
- Monte Carlo Poisson re-fitting for uncertainty cross-check on Peak 3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
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

from tldecpy.fit.multi import fit_multi  # noqa: E402
from tldecpy.schemas import (  # noqa: E402
    FitOptions,
    MultiFitResult,
    PeakSpec,
    RobustOptions,
    UncertaintyOptions,
)
from tldecpy.utils.provenance import file_sha256_hex  # noqa: E402

try:  # noqa: E402
    from phase4_refglow_benchmark import RM_PAPER_RCPARAMS  # type: ignore
except ModuleNotFoundError:
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

FloatArray = NDArray[np.float64]

BENCH_NAME = "PHASE7_UNCERTAINTY_VALIDATION"
MODEL_KEY = "fo_ka"
MODEL_DISPLAY = "First-order peak (Kinetic-Asymptotic approximation)"
UNCERTAINTY_METHOD_LABEL = "Analytic (Hessian-equivalent local covariance)"
E_BOUNDS = (0.5, 5.0)
TM_HALF_WIDTH_K = 5.0

PEAK_LAYOUT = (
    {
        "name": "P1",
        "label": "P1 (Ref P2, 188°C)",
        "tm_k": 457.6,
        "im": 39800.0,
        "e": 1.66,
    },
    {
        "name": "P2",
        "label": "P2 (Ref P3, 218°C)",
        "tm_k": 484.1,
        "im": 53600.0,
        "e": 1.53,
    },
    {
        "name": "P3",
        "label": "Peak 3 (Dosimetric, 511 K)",
        "tm_k": 511.2,
        "im": 103700.0,
        "e": 2.13,
    },
)


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration."""

    data_path: Path
    output_dir: Path
    strategy: str
    beta: float
    max_nfev: int
    n_mc: int
    seed: int
    progress_every: int
    dpi: int
    roi_c_min: float
    roi_c_max: float


@dataclass(frozen=True)
class PreparedData:
    """Prepared curve and preprocessing metadata."""

    t_c: FloatArray
    t_k: FloatArray
    y_raw: FloatArray
    t_roi_c: FloatArray
    t_roi_k: FloatArray
    y_roi_raw: FloatArray
    y_roi_fit: FloatArray
    baseline_roi: float


@dataclass(frozen=True)
class FitPack:
    """Fit output wrapper."""

    result: MultiFitResult
    runtime_s: float
    fom: float
    uc_curve: FloatArray


def build_cli() -> argparse.ArgumentParser:
    """Build command-line interface."""
    parser = argparse.ArgumentParser(description=f"{BENCH_NAME} runner.")
    parser.add_argument("--data-path", type=str, default="scripts/TLD100Exp.csv")
    parser.add_argument("--output-dir", type=str, default="output/phase7_validation")
    parser.add_argument(
        "--strategy",
        type=str,
        default="local",
        choices=["local", "global_hybrid", "global_hybrid_pso"],
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=4000)
    parser.add_argument("--n-mc", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--roi-c-min", type=float, default=165.0)
    parser.add_argument("--roi-c-max", type=float, default=260.0)
    return parser


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Normalize CLI arguments."""
    data_path = Path(str(args.data_path))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    output_dir = Path(str(args.output_dir))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return RunConfig(
        data_path=data_path,
        output_dir=output_dir,
        strategy=str(args.strategy),
        beta=float(args.beta),
        max_nfev=max(int(args.max_nfev), 300),
        n_mc=max(int(args.n_mc), 10),
        seed=int(args.seed),
        progress_every=max(int(args.progress_every), 1),
        dpi=int(args.dpi),
        roi_c_min=float(args.roi_c_min),
        roi_c_max=float(args.roi_c_max),
    )


def configure_matplotlib(dpi: int) -> None:
    """Apply paper-style plotting settings."""
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
    """Convert value to finite float."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detect temperature/intensity columns."""
    normalized = {col: str(col).strip().lower() for col in df.columns}

    t_col = None
    for col, low in normalized.items():
        if low in {"t", "temperature", "temperatura", "temp"}:
            t_col = col
            break
    if t_col is None:
        raise ValueError("Temperature column not found (expected T).")

    i_col = None
    for col, low in normalized.items():
        if low in {"i", "l1", "intensity", "intensidad"}:
            i_col = col
            break
    if i_col is None:
        for col in df.columns:
            if col != t_col:
                i_col = col
                break
    if i_col is None:
        raise ValueError("Intensity column not found (expected I or L1).")

    return t_col, i_col


def load_curve(data_path: Path) -> tuple[FloatArray, FloatArray]:
    """Load experimental TLD100Exp curve."""
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    df = pd.read_csv(data_path)
    t_col, i_col = detect_columns(df)
    t_c = np.asarray(df[t_col], dtype=np.float64)
    y = np.asarray(df[i_col], dtype=np.float64)
    return t_c, y


def calculate_fom(y_obs: FloatArray, y_fit: FloatArray) -> float:
    """Compute FOM (%)."""
    y_o = np.asarray(y_obs, dtype=np.float64)
    y_h = np.asarray(y_fit, dtype=np.float64)
    if y_o.size == 0 or y_h.size != y_o.size:
        return float("inf")
    area = float(np.sum(y_o))
    if not np.isfinite(area) or area <= 0.0:
        return float("inf")
    fom = 100.0 * float(np.sum(np.abs(y_o - y_h))) / area
    return float(fom) if np.isfinite(fom) else float("inf")


def prepare_data(config: RunConfig) -> PreparedData:
    """Convert to kelvin and apply raw-only tight ROI extraction."""
    t_c, y_raw = load_curve(config.data_path)
    t_k = t_c + 273.15

    roi_mask = (t_c >= config.roi_c_min) & (t_c <= config.roi_c_max)
    if int(np.sum(roi_mask)) < 20:
        raise RuntimeError("ROI has too few points for stable 3-peak fitting.")

    t_roi_c = t_c[roi_mask]
    t_roi_k = t_k[roi_mask]
    y_roi_raw = y_raw[roi_mask]
    y_roi_fit = np.asarray(y_roi_raw, dtype=np.float64).copy()
    baseline = 0.0

    return PreparedData(
        t_c=t_c,
        t_k=t_k,
        y_raw=y_raw,
        t_roi_c=t_roi_c,
        t_roi_k=t_roi_k,
        y_roi_raw=y_roi_raw,
        y_roi_fit=y_roi_fit,
        baseline_roi=baseline,
    )


def build_peak_specs_from_web_seeds(t_roi_k: FloatArray) -> list[PeakSpec]:
    """Build 3-peak specs from externally optimized TLDec Web parameters."""
    t_min = float(np.min(t_roi_k))
    t_max = float(np.max(t_roi_k))

    specs: list[PeakSpec] = []
    for peak in PEAK_LAYOUT:
        tm_seed = float(peak["tm_k"])
        tm_lo = max(t_min + 1e-6, tm_seed - TM_HALF_WIDTH_K)
        tm_hi = min(t_max - 1e-6, tm_seed + TM_HALF_WIDTH_K)
        if tm_hi <= tm_lo:
            raise RuntimeError(f"Invalid strict Tm bounds for {peak['name']} in ROI.")

        im_seed = float(peak["im"])
        e_seed = float(peak["e"])
        specs.append(
            PeakSpec(
                name=str(peak["name"]),
                model=MODEL_KEY,
                init={"Tm": tm_seed, "Im": im_seed, "E": e_seed},
                bounds={
                    "Tm": (tm_lo, tm_hi),
                    "Im": (max(1e-9, 0.2 * im_seed), max(3.0 * im_seed, im_seed + 1000.0)),
                    "E": E_BOUNDS,
                },
            )
        )
    return specs


def build_peak_specs_from_fit(result: MultiFitResult) -> list[PeakSpec]:
    """Build Monte Carlo re-fit specs keeping expanded E bounds (0.5, 5.0)."""
    by_name: dict[str, PeakSpec] = {}
    for peak in result.peaks:
        params = dict(peak.params)
        tm = _safe_float(params.get("Tm"))
        im = _safe_float(params.get("Im"))
        e_val = _safe_float(params.get("E"))
        if tm is None or im is None or e_val is None:
            continue

        by_name[str(peak.name)] = PeakSpec(
            name=str(peak.name),
            model=str(peak.model),
            init={"Tm": tm, "Im": im, "E": e_val},
            bounds={
                "Tm": (tm - TM_HALF_WIDTH_K, tm + TM_HALF_WIDTH_K),
                "Im": (max(1e-9, 0.35 * im), max(2.5 * im, im + 2000.0)),
                "E": E_BOUNDS,
            },
        )

    specs: list[PeakSpec] = []
    for peak in PEAK_LAYOUT:
        name = str(peak["name"])
        spec = by_name.get(name)
        if spec is None:
            raise RuntimeError(f"Missing {name} in reference fit for MC setup.")
        specs.append(spec)
    return specs


def build_robust_options() -> RobustOptions:
    """Build deterministic OLS-like robust options."""
    return RobustOptions(
        loss="linear",
        f_scale=1.0,
        weights="none",
        multi_start=0,
        ci_bootstrap=False,
        n_bootstrap=0,
    )


def build_fit_options(max_nfev: int, uncertainty_enabled: bool) -> FitOptions:
    """Build options with analytic uncertainty enabled.

    Note
    ----
    TLDecPy currently exposes Hessian-equivalent local covariance propagation
    through ``include_parameter_covariance=True`` (no explicit ``method`` field).
    """
    uncertainty = UncertaintyOptions(
        enabled=uncertainty_enabled,
        include_parameter_covariance=True,
        noise_from_residuals=True,
        noise_pct=0.0,
        calibration_pct=0.0,
        heating_rate_pct=0.0,
        reader_drift_pct=0.0,
        export_report=True,
        validation_mode="none",
        n_validation_samples=0,
    )
    return FitOptions(local_optimizer="trf", max_nfev=max_nfev, uncertainty=uncertainty)


def result_is_valid(result: MultiFitResult, y_obs: FloatArray) -> bool:
    """Check result consistency."""
    y_hat = np.asarray(result.y_hat_total, dtype=np.float64)
    if y_hat.size != y_obs.size or not np.all(np.isfinite(y_hat)):
        return False
    if not np.isfinite(calculate_fom(y_obs, y_hat)):
        return False
    if len(result.peaks) != 3:
        return False
    return True


def run_fit(
    t_roi_k: FloatArray,
    y_roi_fit: FloatArray,
    specs: list[PeakSpec],
    config: RunConfig,
    *,
    uncertainty_enabled: bool,
) -> FitPack:
    """Run one fit with configured options."""
    robust = build_robust_options()
    options = build_fit_options(config.max_nfev, uncertainty_enabled=uncertainty_enabled)

    start = perf_counter()
    result = fit_multi(
        t_roi_k,
        y_roi_fit,
        peaks=specs,
        bg=None,
        beta=config.beta,
        robust=robust,
        options=options,
        strategy=config.strategy,  # type: ignore[arg-type]
    )
    runtime_s = float(perf_counter() - start)

    if not bool(result.converged) or not result_is_valid(result, y_roi_fit):
        raise RuntimeError(f"Fit failed: {result.message}")

    y_hat = np.asarray(result.y_hat_total, dtype=np.float64)
    fom = calculate_fom(y_roi_fit, y_hat)
    uc_curve = (
        np.asarray(result.uc_curve, dtype=np.float64)
        if result.uc_curve is not None and np.asarray(result.uc_curve).size == y_roi_fit.size
        else np.full_like(y_roi_fit, np.nan, dtype=np.float64)
    )
    return FitPack(result=result, runtime_s=runtime_s, fom=fom, uc_curve=uc_curve)


def uc_at_temperature(temperature_k: FloatArray, uc_curve: FloatArray, t_value: float) -> float | None:
    """Interpolate u_c(T) at given temperature."""
    t_arr = np.asarray(temperature_k, dtype=np.float64)
    u_arr = np.asarray(uc_curve, dtype=np.float64)
    if t_arr.size == 0 or u_arr.size != t_arr.size:
        return None
    if not np.any(np.isfinite(u_arr)):
        return None
    if not (float(np.min(t_arr)) <= t_value <= float(np.max(t_arr))):
        return None
    value = float(np.interp(t_value, t_arr, u_arr))
    return value if np.isfinite(value) else None


def extract_peak_rows(result: MultiFitResult, uc_curve: FloatArray, t_roi_k: FloatArray) -> list[dict[str, Any]]:
    """Extract rows for P1/P2/P3 including uncertainties and derived s."""
    by_name: dict[str, dict[str, Any]] = {}
    for peak in result.peaks:
        params = dict(peak.params)
        tm = _safe_float(params.get("Tm"))
        im = _safe_float(params.get("Im"))
        e_val = _safe_float(params.get("E"))
        s_eval = _safe_float(params.get("s"))
        sigma_e = _safe_float(peak.uncertainties.get("E"))
        sigma_s = _safe_float(peak.uncertainties.get("s"))
        if tm is None or im is None or e_val is None:
            continue

        by_name[str(peak.name)] = {
            "name": str(peak.name),
            "model": str(peak.model),
            "Tm_K": tm,
            "Tm_C": tm - 273.15,
            "Im": im,
            "E_eV": e_val,
            "sigma_E_anal_eV": sigma_e,
            "s_eval": s_eval,
            "sigma_s_eval": sigma_s,
            "u_c_at_Tm_percent": uc_at_temperature(t_roi_k, uc_curve, tm),
            "area": float(peak.area),
        }

    rows: list[dict[str, Any]] = []
    for peak in PEAK_LAYOUT:
        name = str(peak["name"])
        row = by_name.get(name)
        if row is None:
            continue
        row["peak_role"] = name
        row["peak_label"] = str(peak["label"])
        rows.append(row)

    if len(rows) != 3:
        fallback = sorted(by_name.values(), key=lambda row: float(row["Tm_K"]))
        rows = []
        for idx, row in enumerate(fallback):
            role = f"P{idx + 1}"
            row_copy = dict(row)
            row_copy["peak_role"] = role
            row_copy["peak_label"] = str(PEAK_LAYOUT[idx]["label"])
            rows.append(row_copy)

    rows = sorted(rows, key=lambda row: float(row["Tm_K"]))
    return rows


def run_monte_carlo(
    t_roi_k: FloatArray,
    y_fit_reference: FloatArray,
    config: RunConfig,
    ref_specs: list[PeakSpec],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run Poisson MC refits and collect Peak 3 energy samples."""
    robust = build_robust_options()
    options = build_fit_options(config.max_nfev, uncertainty_enabled=False)
    lam = np.clip(np.asarray(y_fit_reference, dtype=np.float64), 0.0, None)

    rows: list[dict[str, Any]] = []
    for run_id in range(1, config.n_mc + 1):
        y_mc = rng.poisson(lam).astype(np.float64)
        rec: dict[str, Any] = {
            "run_id": run_id,
            "success": False,
            "E_P3_eV": np.nan,
            "Tm_P3_K": np.nan,
            "FOM": np.nan,
            "message": "",
        }

        try:
            result = fit_multi(
                t_roi_k,
                y_mc,
                peaks=ref_specs,
                bg=None,
                beta=config.beta,
                robust=robust,
                options=options,
                strategy=config.strategy,  # type: ignore[arg-type]
            )
            if bool(result.converged) and result_is_valid(result, y_mc):
                by_name = {str(peak.name): peak for peak in result.peaks}
                p3 = by_name.get("P3")
                if p3 is None:
                    p3 = sorted(result.peaks, key=lambda p: float(p.params.get("Tm", np.inf)))[-1]

                e_val = _safe_float(p3.params.get("E"))
                tm_val = _safe_float(p3.params.get("Tm"))
                if e_val is not None and tm_val is not None:
                    rec["success"] = True
                    rec["E_P3_eV"] = e_val
                    rec["Tm_P3_K"] = tm_val
                    rec["FOM"] = calculate_fom(y_mc, np.asarray(result.y_hat_total, dtype=np.float64))
                    rec["message"] = str(result.message)
                else:
                    rec["message"] = "missing_P3_params"
            else:
                rec["message"] = str(result.message)
        except Exception as exc:  # noqa: BLE001
            rec["message"] = f"exception:{exc}"

        rows.append(rec)
        if run_id % config.progress_every == 0 or run_id == config.n_mc:
            n_ok = int(sum(bool(r["success"]) for r in rows))
            print(f"[phase7] MC {run_id}/{config.n_mc} | success={n_ok}", flush=True)

    return pd.DataFrame(rows)


def normal_pdf(x: FloatArray, mu: float, sigma: float) -> FloatArray:
    """Evaluate Gaussian PDF."""
    z = (x - mu) / sigma
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def plot_phase7(
    output_pdf: Path,
    output_png: Path,
    prep: PreparedData,
    fit_pack: FitPack,
    peak_rows: list[dict[str, Any]],
    mc_df: pd.DataFrame,
    sigma_anal: float | None,
    sigma_mc: float | None,
    ratio: float | None,
    dpi: int,
) -> None:
    """Create phase7 figure (fit panel + uncertainty panel)."""
    configure_matplotlib(dpi=dpi)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 3.5), constrained_layout=True)

    t_k = prep.t_roi_k
    y_obs = prep.y_roi_fit
    y_hat = np.asarray(fit_pack.result.y_hat_total, dtype=np.float64)

    ax_left.plot(
        t_k,
        y_obs,
        linestyle="none",
        marker="o",
        markersize=2.2,
        color="0.45",
        alpha=0.75,
        label="ROI raw data",
    )
    ax_left.plot(t_k, y_hat, color="black", linewidth=2.0, label="3-peak fit total")

    peak_by_name = {str(p.name): p for p in fit_pack.result.peaks}
    colors = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c"}
    for peak_cfg in PEAK_LAYOUT:
        name = str(peak_cfg["name"])
        peak = peak_by_name.get(name)
        if peak is None:
            continue
        component = np.asarray(peak.y_hat, dtype=np.float64)
        tm = _safe_float(peak.params.get("Tm"))
        ax_left.plot(
            t_k,
            component,
            linestyle="--",
            linewidth=1.45,
            color=colors[name],
            label=str(peak_cfg["label"]),
        )
        if tm is not None:
            ax_left.axvline(tm, color=colors[name], linestyle=":", linewidth=1.0, alpha=0.9)

    uc_global = _safe_float(fit_pack.result.metrics.uc_global)
    ax_left.text(
        0.02,
        0.02,
        "\n".join(
            [
                MODEL_DISPLAY,
                rf"$\mathrm{{FOM}}={fit_pack.fom:.3f}\%$",
                (
                    rf"$u_{{c,\mathrm{{global}}}}={uc_global:.3f}\%$"
                    if uc_global is not None
                    else r"$u_{c,\mathrm{global}}=\mathrm{n/a}$"
                ),
            ]
        ),
        transform=ax_left.transAxes,
        fontsize=7.3,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.0},
    )
    ax_left.set_xlabel(r"Temperature, $T$ (K)")
    ax_left.set_ylabel("Intensity (a.u.)")
    ax_left.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
    ax_left.legend(loc="upper left", fontsize=7.0)

    mc_success = mc_df[mc_df["success"] == True]  # noqa: E712
    e_vals = np.asarray(mc_success["E_P3_eV"], dtype=np.float64)
    e_vals = e_vals[np.isfinite(e_vals)]

    if e_vals.size > 0:
        _, bins, _ = ax_right.hist(
            e_vals,
            bins=28,
            color="#4c78a8",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            label=f"MC samples for Peak 3 (n={e_vals.size})",
        )

        if sigma_anal is not None and sigma_anal > 0.0:
            e_center = _safe_float(next((row["E_eV"] for row in peak_rows if row["peak_role"] == "P3"), np.nan))
            mu = float(np.mean(e_vals)) if e_center is None else float(e_center)
            x_grid = np.linspace(float(np.min(bins)), float(np.max(bins)), 400, dtype=np.float64)
            bw = float(np.mean(np.diff(bins))) if bins.size > 1 else 1.0
            y_gauss = normal_pdf(x_grid, mu, sigma_anal) * float(e_vals.size) * bw
            ax_right.plot(
                x_grid,
                y_gauss,
                color="#d62728",
                linewidth=1.8,
                label=rf"Analytic Gaussian ($\sigma_{{anal}}={sigma_anal:.4f}\,\mathrm{{eV}}$)",
            )

        ax_right.axvline(
            float(np.mean(e_vals)),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=rf"$\mu_{{MC}}={float(np.mean(e_vals)):.4f}\,\mathrm{{eV}}$",
        )
    else:
        ax_right.text(0.5, 0.5, "No valid MC samples", transform=ax_right.transAxes, ha="center", va="center")

    stats_lines = [
        rf"$\sigma_{{MC}}={'n/a' if sigma_mc is None else f'{sigma_mc:.4f}'}\,\mathrm{{eV}}$",
        rf"$\sigma_{{anal}}={'n/a' if sigma_anal is None else f'{sigma_anal:.4f}'}\,\mathrm{{eV}}$",
    ]
    if ratio is not None:
        stats_lines.append(rf"$\sigma_{{MC}}/\sigma_{{anal}}={ratio:.3f}$")
    ax_right.text(
        0.98,
        0.98,
        "\n".join(stats_lines),
        transform=ax_right.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.0},
    )
    ax_right.set_xlabel(r"Peak 3 (Dosimetric, 511 K) energy, $E$ (eV)")
    ax_right.set_ylabel("Count")
    ax_right.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
    ax_right.legend(loc="upper left", fontsize=6.8)

    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    """Run phase7 final validation."""
    config = parse_config(build_cli().parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data_file_sha256 = file_sha256_hex(config.data_path)

    prep = prepare_data(config)
    specs = build_peak_specs_from_web_seeds(prep.t_roi_k)
    analytic_fit = run_fit(
        prep.t_roi_k,
        prep.y_roi_fit,
        specs,
        config,
        uncertainty_enabled=True,
    )

    peak_rows = extract_peak_rows(analytic_fit.result, analytic_fit.uc_curve, prep.t_roi_k)
    if len(peak_rows) != 3:
        raise RuntimeError("Expected three fitted peaks.")

    p3_row = next((row for row in peak_rows if row["peak_role"] == "P3"), peak_rows[-1])
    p3_e = _safe_float(p3_row.get("E_eV"))
    sigma_anal = _safe_float(p3_row.get("sigma_E_anal_eV"))

    hit_bounds_map = dict(analytic_fit.result.hit_bounds)
    p3_e_hit_bounds_flag = bool(
        hit_bounds_map.get("P3_E", False)
        or hit_bounds_map.get("P3_E0", False)
        or any(
            bool(value)
            for key, value in hit_bounds_map.items()
            if str(key).startswith("P3_") and str(key).endswith("_E")
        )
    )
    p3_e_hits_upper_bound = bool(p3_e is not None and p3_e >= (E_BOUNDS[1] - 1e-6))
    p3_e_near_legacy_2p4_cap = bool(p3_e is not None and abs(p3_e - 2.4) <= 1e-3)

    ref_specs = build_peak_specs_from_fit(analytic_fit.result)
    rng = np.random.default_rng(config.seed)
    mc_df = run_monte_carlo(
        prep.t_roi_k,
        np.asarray(analytic_fit.result.y_hat_total, dtype=np.float64),
        config,
        ref_specs,
        rng,
    )

    mc_success = mc_df[mc_df["success"] == True]  # noqa: E712
    e_mc = np.asarray(mc_success["E_P3_eV"], dtype=np.float64)
    e_mc = e_mc[np.isfinite(e_mc)]
    sigma_mc = float(np.std(e_mc, ddof=1)) if e_mc.size >= 2 else None

    ratio = None
    if sigma_mc is not None and sigma_anal is not None and sigma_anal > 0.0:
        ratio = float(sigma_mc / sigma_anal)

    peak_df = pd.DataFrame(peak_rows)
    peak_df["model_key"] = MODEL_KEY
    peak_df["model_display"] = MODEL_DISPLAY
    peak_df["FOM_percent"] = analytic_fit.fom

    uc_curve_df = pd.DataFrame(
        {
            "temperature_C": prep.t_roi_c,
            "temperature_K": prep.t_roi_k,
            "intensity_roi_raw": prep.y_roi_raw,
            "intensity_roi_fit_input": prep.y_roi_fit,
            "intensity_fit_total": np.asarray(analytic_fit.result.y_hat_total, dtype=np.float64),
            "u_c_percent": analytic_fit.uc_curve,
        }
    )

    budget = dict(analytic_fit.result.uncertainty_budget)
    budget_df = pd.DataFrame(
        [{"source": str(key), "value_percent": _safe_float(value)} for key, value in budget.items()]
    )

    model_results_df = pd.DataFrame(
        [
            {
                "bench": BENCH_NAME,
                "curve": "TLD100Exp",
                "data_file": str(config.data_path),
                "data_file_sha256": data_file_sha256,
                "data_hash_algorithm": "sha256",
                "model_key": MODEL_KEY,
                "model_display": MODEL_DISPLAY,
                "strategy": config.strategy,
                "beta_K_per_s": config.beta,
                "roi_C_min": config.roi_c_min,
                "roi_C_max": config.roi_c_max,
                "roi_K_min": float(np.min(prep.t_roi_k)),
                "roi_K_max": float(np.max(prep.t_roi_k)),
                "preprocessing_mode": "raw_only",
                "baseline_roi_first5_mean": prep.baseline_roi,
                "E_bounds_min": E_BOUNDS[0],
                "E_bounds_max": E_BOUNDS[1],
                "FOM_percent": analytic_fit.fom,
                "runtime_s_analytic": analytic_fit.runtime_s,
                "mc_samples_requested": config.n_mc,
                "mc_samples_success": int(e_mc.size),
                "mc_samples_failed": int(config.n_mc - e_mc.size),
                "E_P3_eV": p3_e,
                "sigma_E_P3_anal_eV": sigma_anal,
                "sigma_E_P3_mc_eV": sigma_mc,
                "sigma_ratio_mc_over_anal": ratio,
                "P3_E_hit_bounds_flag": p3_e_hit_bounds_flag,
                "P3_E_hits_upper_bound": p3_e_hits_upper_bound,
                "P3_E_near_legacy_2p4_cap": p3_e_near_legacy_2p4_cap,
                "uc_global_percent": _safe_float(analytic_fit.result.metrics.uc_global),
                "uc_p95_percent": _safe_float(analytic_fit.result.metrics.uc_p95),
                "uc_max_percent": _safe_float(analytic_fit.result.metrics.uc_max),
                "message": str(analytic_fit.result.message),
            }
        ]
    )

    p3_budget_df = pd.DataFrame(
        [
            {
                "peak": "P3",
                "method": UNCERTAINTY_METHOD_LABEL,
                "E_center_eV": _safe_float(p3_row.get("E_eV")),
                "sigma_E_eV": sigma_anal,
                "n_samples": np.nan,
            },
            {
                "peak": "P3",
                "method": "Monte Carlo (Poisson re-fit)",
                "E_center_eV": float(np.mean(e_mc)) if e_mc.size > 0 else np.nan,
                "sigma_E_eV": sigma_mc,
                "n_samples": int(e_mc.size),
            },
            {
                "peak": "P3",
                "method": "Ratio MC/Analytic",
                "E_center_eV": np.nan,
                "sigma_E_eV": ratio,
                "n_samples": np.nan,
            },
        ]
    )

    fig_pdf = config.output_dir / "phase7_uncertainty.pdf"
    fig_png = config.output_dir / "phase7_uncertainty.png"
    model_csv = config.output_dir / "phase7_model_results.csv"
    peaks_csv = config.output_dir / "phase7_peak_params_long.csv"
    uc_csv = config.output_dir / "phase7_uc_curve.csv"
    mc_csv = config.output_dir / "phase7_mc_E_samples.csv"
    budget_csv = config.output_dir / "phase7_uncertainty_budget.csv"
    p3_budget_csv = config.output_dir / "phase7_p3_uncertainty_budget.csv"
    summary_txt = config.output_dir / "summary_phase7.txt"
    legacy_p5_budget_csv = config.output_dir / "phase7_p5_uncertainty_budget.csv"

    plot_phase7(
        output_pdf=fig_pdf,
        output_png=fig_png,
        prep=prep,
        fit_pack=analytic_fit,
        peak_rows=peak_rows,
        mc_df=mc_df,
        sigma_anal=sigma_anal,
        sigma_mc=sigma_mc,
        ratio=ratio,
        dpi=config.dpi,
    )

    model_results_df.to_csv(model_csv, index=False)
    peak_df.to_csv(peaks_csv, index=False)
    uc_curve_df.to_csv(uc_csv, index=False)
    mc_df.to_csv(mc_csv, index=False)
    budget_df.to_csv(budget_csv, index=False)
    p3_budget_df.to_csv(p3_budget_csv, index=False)
    if legacy_p5_budget_csv.exists():
        legacy_p5_budget_csv.unlink()

    if ratio is None:
        conclusion = "Inconclusive: MC/analytic ratio could not be computed."
    elif 0.8 <= ratio <= 1.25:
        conclusion = "Reasonable agreement between analytic and Monte Carlo methods."
    else:
        conclusion = "Significant difference between analytic and Monte Carlo methods."

    summary_lines = [
        BENCH_NAME,
        "",
        f"Input file: {config.data_path}",
        f"Input file SHA-256: {data_file_sha256}",
        "Temperature conversion: T_K = T_C + 273.15",
        f"ROI: {config.roi_c_min:.1f} to {config.roi_c_max:.1f} °C "
        f"({float(np.min(prep.t_roi_k)):.2f} to {float(np.max(prep.t_roi_k)):.2f} K)",
        "Preprocessing: disabled (raw intensity used directly)",
        "Baseline subtraction: disabled",
        f"Energy bounds applied (all peaks): E in [{E_BOUNDS[0]}, {E_BOUNDS[1]}] eV",
        "Frequency-factor bound note: model fo_ka does not optimize s directly; s is derived post-fit.",
        "",
        f"Model: {MODEL_DISPLAY} ({MODEL_KEY})",
        f"Strategy: {config.strategy}",
        f"Beta: {config.beta:.4f} K/s",
        f"FOM: {analytic_fit.fom:.6f}%",
        f"u_c_global: {_safe_float(analytic_fit.result.metrics.uc_global)}%",
        f"u_c_p95: {_safe_float(analytic_fit.result.metrics.uc_p95)}%",
        "",
        "Peak 3 uncertainty comparison (Energy E):",
        f"E_P3 fitted: {'n/a' if p3_e is None else f'{p3_e:.6f} eV'}",
        f"P3_E hit-bounds flag from solver: {p3_e_hit_bounds_flag}",
        f"E_P3 hits upper bound (5.0 eV): {p3_e_hits_upper_bound}",
        f"E_P3 near previous cap (2.4 eV): {p3_e_near_legacy_2p4_cap}",
        f"Analytic method: {UNCERTAINTY_METHOD_LABEL}",
        f"sigma_anal: {'n/a' if sigma_anal is None else f'{sigma_anal:.6f} eV'}",
        f"sigma_MC: {'n/a' if sigma_mc is None else f'{sigma_mc:.6f} eV'}",
        f"ratio sigma_MC/sigma_anal: {'n/a' if ratio is None else f'{ratio:.6f}'}",
        f"MC successes: {int(e_mc.size)}/{config.n_mc}",
        "",
        "Uncertainty budget contributions (%):",
    ]
    for _, row in budget_df.iterrows():
        summary_lines.append(f"- {row['source']}: {row['value_percent']}")
    summary_lines.extend(
        [
            "",
            f"Conclusion: {conclusion}",
            "",
            "Outputs:",
            f"- Figure PDF: {fig_pdf}",
            f"- Figure PNG: {fig_png}",
            f"- Model results: {model_csv}",
            f"- Peak params: {peaks_csv}",
            f"- u_c(T): {uc_csv}",
            f"- MC samples: {mc_csv}",
            f"- Uncertainty budget: {budget_csv}",
            f"- P3 uncertainty budget: {p3_budget_csv}",
        ]
    )
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"{BENCH_NAME} completed.")
    print(
        " | ".join(
            [
                f"FOM={analytic_fit.fom:.5f}%",
                f"sigma_anal={'n/a' if sigma_anal is None else f'{sigma_anal:.5f}'}",
                f"sigma_MC={'n/a' if sigma_mc is None else f'{sigma_mc:.5f}'}",
                f"ratio={'n/a' if ratio is None else f'{ratio:.4f}'}",
            ]
        )
    )
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    main()
