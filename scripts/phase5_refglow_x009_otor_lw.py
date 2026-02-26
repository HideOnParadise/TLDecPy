#!/usr/bin/env python3
"""Bench5: RefGlow x009 deconvolution with OTOR_LW (Lambert-W), seed-comparable.

This benchmark runs only curve x009 using the OTOR_LW model with 9 peaks.
It mimics a trial-and-error workflow by running randomized multistarts
around reference initial seeds (inisPAR-equivalent), then selecting the
best fit by minimum FOM.
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
    FitOptions,
    MultiFitResult,
    PeakSpec,
    RobustOptions,
    UncertaintyOptions,
)
from tldecpy.utils.provenance import file_sha256_hex  # noqa: E402

from phase4_refglow_benchmark import RM_PAPER_RCPARAMS  # noqa: E402

FloatArray = NDArray[np.float64]

CURVE_ID = "x009"
CURVE_DISPLAY = "REFGLOW.009"
MODEL_KEY = "otor_lw"
MODEL_INTERNAL = "OTOR_LW"
MODEL_DISPLAY = "OTOR (Lambert-W)"
BETA = 1.0


REFERENCE_SEEDS: tuple[dict[str, float], ...] = (
    {"peak_index": 1.0, "Im": 9855.768, "E": 1.225681, "Tm": 387.5201, "R": 2.142983e-02},
    {"peak_index": 2.0, "Im": 21554.421, "E": 1.370533, "Tm": 428.8617, "R": 2.011305e-01},
    {"peak_index": 3.0, "Im": 24719.536, "E": 2.285317, "Tm": 462.6737, "R": 8.251181e-01},
    {"peak_index": 4.0, "Im": 54945.499, "E": 2.314386, "Tm": 488.3172, "R": 5.790214e-02},
    {"peak_index": 5.0, "Im": 5261.516, "E": 1.106069, "Tm": 494.5081, "R": 8.828895e-03},
    {"peak_index": 6.0, "Im": 3758.059, "E": 1.436794, "Tm": 523.9569, "R": 7.012545e-02},
    {"peak_index": 7.0, "Im": 7353.395, "E": 1.678277, "Tm": 555.0372, "R": 3.662934e-04},
    {"peak_index": 8.0, "Im": 2109.270, "E": 2.619323, "Tm": 583.3247, "R": 2.225439e-07},
    {"peak_index": 9.0, "Im": 2298.912, "E": 1.879263, "Tm": 603.2252, "R": 4.821910e-01},
)


@dataclass(frozen=True)
class RunConfig:
    """Typed runtime configuration."""

    output_dir: Path
    nstart: int
    kkf: float
    tm_jitter_k: float
    seed: int
    strategy: str
    max_nfev: int
    dpi: int
    progress_every: int
    enable_smoothing: bool
    enable_baseline: bool


@dataclass(frozen=True)
class BestResult:
    """Container for best selected fit."""

    start_id: int
    fom: float
    runtime_s: float
    warning_count: int
    hit_bounds_count: int
    mean_rel_change_pct: float
    tm_order_inversion: bool
    result: MultiFitResult
    seeds: list[dict[str, float]]


def build_cli() -> argparse.ArgumentParser:
    """Construct CLI parser."""
    parser = argparse.ArgumentParser(
        description="Bench5 x009 with OTOR_LW and reference-seed multistart trial-and-error."
    )
    parser.add_argument("--output-dir", type=str, default="output/phase5_validation")
    parser.add_argument("--nstart", type=int, default=60)
    parser.add_argument("--kkf", type=float, default=0.03)
    parser.add_argument("--tm-jitter-k", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument(
        "--strategy",
        type=str,
        default="global_hybrid_pso",
        choices=["local", "global_hybrid", "global_hybrid_pso"],
    )
    parser.add_argument("--max-nfev", type=int, default=3000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--enable-smoothing",
        action="store_true",
        help="Optional fallback smoothing (disabled by default for seed comparability).",
    )
    parser.add_argument(
        "--enable-baseline",
        action="store_true",
        help="Optional fallback baseline subtraction (disabled by default for seed comparability).",
    )
    return parser


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Normalize and validate runtime configuration."""
    output_dir = Path(str(args.output_dir))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    nstart = max(int(args.nstart), 1)
    kkf = float(np.clip(float(args.kkf), 1e-6, 0.5))
    tm_jitter_k = max(float(args.tm_jitter_k), 0.0)

    return RunConfig(
        output_dir=output_dir,
        nstart=nstart,
        kkf=kkf,
        tm_jitter_k=tm_jitter_k,
        seed=int(args.seed),
        strategy=str(args.strategy),
        max_nfev=int(args.max_nfev),
        dpi=int(args.dpi),
        progress_every=max(int(args.progress_every), 1),
        enable_smoothing=bool(args.enable_smoothing),
        enable_baseline=bool(args.enable_baseline),
    )


def configure_matplotlib(dpi: int) -> None:
    """Apply bench plotting style."""
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
    """Return finite float or None."""
    if value is None:
        return None
    try:
        f_val = float(value)
    except (TypeError, ValueError):
        return None
    return f_val if np.isfinite(f_val) else None


def parse_peak_id(peak_name: str, fallback: int) -> int:
    """Extract integer peak index from peak label."""
    match = re.search(r"(\d+)", peak_name)
    return int(match.group(1)) if match else fallback


def calculate_fom(y_obs: FloatArray, y_fit: FloatArray) -> float:
    """Compute Figure of Merit (%) using project convention."""
    y_obs_arr = np.asarray(y_obs, dtype=np.float64)
    y_fit_arr = np.asarray(y_fit, dtype=np.float64)
    if y_obs_arr.size == 0 or y_obs_arr.size != y_fit_arr.size:
        return float("inf")
    area = float(np.sum(y_obs_arr))
    if area <= 0.0 or not np.isfinite(area):
        return float("inf")
    fom = float(100.0 * np.sum(np.abs(y_obs_arr - y_fit_arr)) / area)
    return fom if np.isfinite(fom) else float("inf")


def preprocess_signal(
    y_raw: FloatArray,
    *,
    enable_smoothing: bool,
    enable_baseline: bool,
) -> tuple[FloatArray, str]:
    """Return fit signal following bench comparability policy."""
    # Comparability default: raw curve for both fit and FOM.
    y = np.asarray(y_raw, dtype=np.float64).copy()
    mode = "raw"

    if enable_smoothing:
        from tldecpy.utils.sg import safe_savgol

        y = np.asarray(safe_savgol(y, window_length=11, polyorder=3), dtype=np.float64)
        mode = "smooth"

    if enable_baseline:
        baseline_level = float(np.percentile(y, 5))
        y = np.clip(y - baseline_level, 0.0, None)
        mode = "smooth+baseline" if enable_smoothing else "baseline"

    return y, mode


def build_tm_bounds_by_peak(
    base_seeds: tuple[dict[str, float], ...],
    t_min: float,
    t_max: float,
    *,
    delta_k: float = 3.0,
) -> dict[int, tuple[float, float]]:
    """Build non-overlapping Tm bounds from reference seeds ordered by Tm."""
    sorted_seeds = sorted(base_seeds, key=lambda row: float(row["Tm"]))
    bounds_by_peak: dict[int, tuple[float, float]] = {}

    for idx, seed in enumerate(sorted_seeds):
        tm_seed = float(seed["Tm"])
        peak_idx = int(seed["peak_index"])

        if idx == 0:
            lo_mid = t_min
        else:
            lo_mid = 0.5 * (float(sorted_seeds[idx - 1]["Tm"]) + tm_seed)

        if idx == len(sorted_seeds) - 1:
            hi_mid = t_max
        else:
            hi_mid = 0.5 * (tm_seed + float(sorted_seeds[idx + 1]["Tm"]))

        tm_lo = max(t_min + 1e-6, lo_mid + delta_k)
        tm_hi = min(t_max - 1e-6, hi_mid - delta_k)
        if tm_hi <= tm_lo:
            tm_lo = max(t_min + 1e-6, tm_seed - 5.0)
            tm_hi = min(t_max - 1e-6, tm_seed + 5.0)
            if tm_hi <= tm_lo:
                tm_lo = max(t_min + 1e-6, tm_seed - 1.0)
                tm_hi = min(t_max - 1e-6, tm_seed + 1.0)

        bounds_by_peak[peak_idx] = (tm_lo, tm_hi)

    return bounds_by_peak


def build_bounds(
    seed: dict[str, float],
    tm_bounds_by_peak: dict[int, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Build physically constrained relative bounds anchored to each reference seed."""
    peak_idx = int(seed["peak_index"])
    im0 = float(seed["Im"])
    e0 = float(seed["E"])
    r0 = max(float(seed["R"]), 1e-12)

    tm_lo, tm_hi = tm_bounds_by_peak[peak_idx]

    im_lo = max(1e-12, 0.2 * im0)
    im_hi = max(im_lo + 1e-12, 5.0 * im0)

    e_lo = max(1e-6, 0.7 * e0)
    e_hi = min(6.0, 1.3 * e0)
    if e_hi <= e_lo:
        e_hi = min(6.0, e_lo + 1e-3)

    r_lo = max(1e-12, r0 / 10.0)
    r_hi = min(1.0 - 1e-9, r0 * 10.0)
    if r_hi <= r_lo:
        r_hi = min(1.0 - 1e-9, r_lo * 1.01 + 1e-12)

    return {
        "Im": (im_lo, im_hi),
        "E": (e_lo, e_hi),
        "Tm": (tm_lo, tm_hi),
        "R": (r_lo, r_hi),
    }


def jitter_seed(
    rng: np.random.Generator,
    seed: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    kkf: float,
    tm_jitter_k: float,
) -> dict[str, float]:
    """Create one randomized inisPAR-like seed around reference values."""
    im = float(seed["Im"]) * float(rng.uniform(1.0 - kkf, 1.0 + kkf))
    e_val = float(seed["E"]) * float(rng.uniform(1.0 - kkf, 1.0 + kkf))
    tm = float(seed["Tm"]) + float(rng.uniform(-tm_jitter_k, tm_jitter_k))

    r0 = max(float(seed["R"]), 1e-300)
    log10_factor = float(np.log10(1.0 + kkf))
    log_r = float(np.log10(r0)) + float(rng.uniform(-log10_factor, log10_factor))
    r_val = float(10.0**log_r)

    return {
        "peak_index": float(seed["peak_index"]),
        "Im": float(np.clip(im, *bounds["Im"])),
        "E": float(np.clip(e_val, *bounds["E"])),
        "Tm": float(np.clip(tm, *bounds["Tm"])),
        "R": float(np.clip(r_val, *bounds["R"])),
    }


def build_peak_specs(
    start_seeds: list[dict[str, float]],
    bounds_by_peak: dict[int, dict[str, tuple[float, float]]],
) -> list[PeakSpec]:
    """Build OTOR_LW PeakSpec list for one multistart trial."""
    specs: list[PeakSpec] = []
    for idx, start_seed in enumerate(start_seeds, start=1):
        peak_idx = int(start_seed.get("peak_index", idx))
        bounds = bounds_by_peak[peak_idx]
        init = {
            "Im": float(start_seed["Im"]),
            "E": float(start_seed["E"]),
            "Tm": float(start_seed["Tm"]),
            "R": float(start_seed["R"]),
        }
        specs.append(
            PeakSpec(
                name=f"P{peak_idx}",
                model=MODEL_KEY,
                init=init,
                bounds=bounds,
            )
        )
    return specs


def build_fit_options(max_nfev: int, uncertainty_enabled: bool) -> FitOptions:
    """Create fit options with toggleable uncertainty analysis."""
    uncertainty = UncertaintyOptions(
        enabled=uncertainty_enabled,
        include_parameter_covariance=uncertainty_enabled,
        noise_from_residuals=True,
        noise_pct=0.0,
        calibration_pct=0.0,
        heating_rate_pct=0.0,
        reader_drift_pct=0.0,
        export_report=uncertainty_enabled,
        validation_mode="none",
    )
    return FitOptions(local_optimizer="trf", max_nfev=max_nfev, uncertainty=uncertainty)


def build_robust_options() -> RobustOptions:
    """Create robust options for OLS-equivalent objective."""
    return RobustOptions(
        loss="linear",
        f_scale=1.0,
        weights="none",
        multi_start=0,
        ci_bootstrap=False,
        n_bootstrap=0,
    )


def result_is_valid(result: MultiFitResult, y_obs: FloatArray) -> bool:
    """Check numerical validity of a fit result."""
    y_hat = np.asarray(result.y_hat_total, dtype=np.float64)
    if y_hat.size != y_obs.size or not np.all(np.isfinite(y_hat)):
        return False
    if not np.isfinite(calculate_fom(y_obs, y_hat)):
        return False
    for peak in result.peaks:
        params = dict(peak.params)
        im = _safe_float(params.get("Im"))
        e_val = _safe_float(params.get("E"))
        tm = _safe_float(params.get("Tm"))
        r_val = _safe_float(params.get("R"))
        if im is None or e_val is None or tm is None or r_val is None:
            return False
        if im <= 0.0 or e_val <= 0.0 or not (0.0 < r_val < 1.0):
            return False
    return True


def count_warning_tokens(result: MultiFitResult) -> int:
    """Count warning-like signals for tie-breaking diagnostics."""
    count = 0
    if not bool(result.converged):
        count += 1
    msg = str(result.message).lower()
    for token in ("warn", "fail", "error", "bound"):
        if token in msg:
            count += 1
    return count


def count_hit_bounds(result: MultiFitResult) -> int:
    """Count parameters that ended on optimization bounds."""
    return int(sum(1 for hit in result.hit_bounds.values() if bool(hit)))


def has_tm_order_inversion(result: MultiFitResult) -> bool:
    """Return True when fitted Tm order violates peak index order."""
    indexed_tm: list[tuple[int, float]] = []
    for idx, peak in enumerate(result.peaks, start=1):
        peak_idx = parse_peak_id(peak.name, fallback=idx)
        tm_val = _safe_float(dict(peak.params).get("Tm"))
        if tm_val is None:
            return True
        indexed_tm.append((peak_idx, tm_val))

    indexed_tm.sort(key=lambda row: row[0])
    tm_vals = [row[1] for row in indexed_tm]
    return bool(any(tm_vals[i] > tm_vals[i + 1] for i in range(len(tm_vals) - 1)))


def mean_relative_change_vs_seed(
    result: MultiFitResult,
    seeds: tuple[dict[str, float], ...],
) -> float:
    """Compute average absolute relative change (%) vs reference seeds."""
    seed_by_peak = {int(seed["peak_index"]): seed for seed in seeds}
    rel_changes: list[float] = []

    for idx, peak in enumerate(result.peaks, start=1):
        peak_idx = parse_peak_id(peak.name, fallback=idx)
        seed = seed_by_peak.get(peak_idx)
        if seed is None:
            continue
        params = dict(peak.params)
        for key in ("Im", "E", "Tm", "R"):
            fit_val = _safe_float(params.get(key))
            seed_val = _safe_float(seed.get(key))
            if fit_val is None or seed_val is None:
                continue
            if seed_val == 0.0:
                rel_changes.append(abs(fit_val - seed_val))
            else:
                rel_changes.append(100.0 * abs(fit_val - seed_val) / abs(seed_val))

    if not rel_changes:
        return float("inf")
    return float(np.mean(rel_changes))


def uc_at_temperature(
    temperature: FloatArray,
    uc_curve: FloatArray | None,
    tm_value: float,
) -> float | None:
    """Interpolate Sadek coefficient u_c(T) at one temperature."""
    if uc_curve is None:
        return None
    uc_arr = np.asarray(uc_curve, dtype=np.float64)
    t_arr = np.asarray(temperature, dtype=np.float64)
    if uc_arr.size != t_arr.size:
        return None
    if not np.any(np.isfinite(uc_arr)):
        return None
    t_min = float(np.min(t_arr))
    t_max = float(np.max(t_arr))
    if not (t_min <= tm_value <= t_max):
        return None
    value = float(np.interp(tm_value, t_arr, uc_arr))
    return value if np.isfinite(value) else None


def plot_fit_with_residual(
    output_path: Path,
    temperature: FloatArray,
    y_obs: FloatArray,
    best: BestResult,
    beta: float,
    dpi: int,
) -> None:
    """Create basic two-panel figure: fit/components and residuals."""
    configure_matplotlib(dpi=dpi)

    y_hat = np.asarray(best.result.y_hat_total, dtype=np.float64)
    residual = y_obs - y_hat
    uc_global = _safe_float(best.result.metrics.uc_global)
    uc_p95 = _safe_float(best.result.metrics.uc_p95)

    fig, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.06},
        constrained_layout=True,
    )

    ax_fit.plot(
        temperature,
        y_obs,
        linestyle="none",
        marker="o",
        markersize=2.2,
        color="0.55",
        alpha=0.7,
        label="Data (raw)",
    )
    ax_fit.plot(temperature, y_hat, color="black", linewidth=2.0, label="Total fit")

    cmap = plt.get_cmap("tab10")
    uc_tm_lines: list[str] = []
    for idx, peak in enumerate(best.result.peaks, start=1):
        y_peak = np.asarray(peak.y_hat, dtype=np.float64)
        ax_fit.plot(
            temperature,
            y_peak,
            linestyle="--",
            linewidth=1.2,
            color=cmap((idx - 1) % 10),
            alpha=0.95,
            label=f"Peak {idx}",
        )
        tm_fit = _safe_float(peak.params.get("Tm"))
        uc_tm = None if tm_fit is None else uc_at_temperature(temperature, best.result.uc_curve, tm_fit)
        if uc_tm is not None:
            uc_tm_lines.append(f"P{idx}:{uc_tm:.2f}%")
        else:
            uc_tm_lines.append(f"P{idx}:n/a")

    info_lines = [
        CURVE_DISPLAY,
        MODEL_DISPLAY,
        f"beta={beta:.3g} K/s",
        f"FOM={best.fom:.4f}%",
        f"u_c_global={'n/a' if uc_global is None else f'{uc_global:.3f}%'}",
        f"u_c_p95={'n/a' if uc_p95 is None else f'{uc_p95:.3f}%'}",
        "u_c(Tm): " + ", ".join(uc_tm_lines),
    ]
    ax_fit.text(
        0.015,
        0.98,
        "\n".join(info_lines),
        transform=ax_fit.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.4},
    )

    ax_fit.set_ylabel("Intensity (a.u.)")
    ax_fit.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
    ax_fit.legend(loc="upper right", fontsize=7.2, ncol=2, frameon=True)

    ax_res.plot(
        temperature,
        residual,
        linestyle="none",
        marker="o",
        markersize=2.0,
        color="#8c4b12",
        alpha=0.75,
    )
    ax_res.axhline(0.0, color="black", linewidth=1.0)
    q99 = float(np.nanpercentile(np.abs(residual), 99))
    if not np.isfinite(q99) or q99 <= 0.0:
        q99 = float(np.nanmax(np.abs(residual))) if residual.size > 0 else 1.0
    if not np.isfinite(q99) or q99 <= 0.0:
        q99 = 1.0
    ax_res.set_ylim(-1.1 * q99, 1.1 * q99)
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel("Temperature, T (K)")
    ax_res.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_seed_vs_fit_table(best: BestResult) -> pd.DataFrame:
    """Create reference-seed vs fitted-parameter deltas by peak."""
    rows: list[dict[str, Any]] = []

    fitted_by_peak: dict[int, dict[str, float]] = {}
    for idx, peak in enumerate(best.result.peaks, start=1):
        peak_idx = parse_peak_id(peak.name, fallback=idx)
        params = dict(peak.params)
        fitted_by_peak[peak_idx] = {
            "Im": float(params["Im"]),
            "E": float(params["E"]),
            "Tm": float(params["Tm"]),
            "R": float(params["R"]),
        }

    for seed in REFERENCE_SEEDS:
        peak_idx = int(seed["peak_index"])
        fit_vals = fitted_by_peak.get(peak_idx)
        if fit_vals is None:
            continue

        row: dict[str, Any] = {"peak_index": peak_idx}
        for key in ("Im", "E", "Tm", "R"):
            seed_val = float(seed[key])
            fit_val = float(fit_vals[key])
            delta_abs = fit_val - seed_val
            delta_rel = float("nan") if seed_val == 0.0 else 100.0 * delta_abs / seed_val
            row[f"{key}_seed"] = seed_val
            row[f"{key}_fit"] = fit_val
            row[f"{key}_delta_abs"] = delta_abs
            row[f"{key}_delta_rel_pct"] = delta_rel
        rows.append(row)

    return pd.DataFrame(rows).sort_values("peak_index").reset_index(drop=True)


def run_trial(
    temperature: FloatArray,
    y_fit: FloatArray,
    peak_specs: list[PeakSpec],
    config: RunConfig,
    *,
    uncertainty_enabled: bool,
    strategy_override: str | None = None,
) -> tuple[MultiFitResult | None, float]:
    """Run one fit attempt and return result plus runtime."""
    robust = build_robust_options()
    options = build_fit_options(max_nfev=config.max_nfev, uncertainty_enabled=uncertainty_enabled)
    strategy = config.strategy if strategy_override is None else strategy_override

    start_time = perf_counter()
    try:
        result = fit_multi(
            temperature,
            y_fit,
            peaks=peak_specs,
            bg=None,
            beta=BETA,
            robust=robust,
            options=options,
            strategy=strategy,  # type: ignore[arg-type]
        )
    except Exception:
        return None, float(perf_counter() - start_time)
    return result, float(perf_counter() - start_time)


def main() -> None:
    """Execute bench5 x009 OTOR_LW deconvolution workflow."""
    config = parse_config(build_cli().parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    data_file = resolve_refglow_path(CURVE_ID)
    data_file_sha256 = file_sha256_hex(data_file)
    temperature, intensity_raw = load_refglow(CURVE_ID)
    t_arr = np.asarray(temperature, dtype=np.float64)
    y_raw = np.asarray(intensity_raw, dtype=np.float64)
    y_fit, preprocess_mode = preprocess_signal(
        y_raw,
        enable_smoothing=config.enable_smoothing,
        enable_baseline=config.enable_baseline,
    )

    t_min = float(np.min(t_arr))
    t_max = float(np.max(t_arr))

    tm_bounds_by_peak = build_tm_bounds_by_peak(
        REFERENCE_SEEDS, t_min=t_min, t_max=t_max, delta_k=3.0
    )
    bounds_by_peak = {
        int(seed["peak_index"]): build_bounds(seed=seed, tm_bounds_by_peak=tm_bounds_by_peak)
        for seed in REFERENCE_SEEDS
    }

    reference_seeds = [
        {
            "peak_index": float(seed["peak_index"]),
            "Im": float(seed["Im"]),
            "E": float(seed["E"]),
            "Tm": float(seed["Tm"]),
            "R": float(seed["R"]),
        }
        for seed in REFERENCE_SEEDS
    ]
    reference_specs = build_peak_specs(start_seeds=reference_seeds, bounds_by_peak=bounds_by_peak)
    reference_result, reference_runtime_s = run_trial(
        temperature=t_arr,
        y_fit=y_fit,
        peak_specs=reference_specs,
        config=config,
        uncertainty_enabled=False,
        strategy_override="local",
    )
    reference_valid = (
        reference_result is not None
        and bool(reference_result.converged)
        and result_is_valid(reference_result, y_fit)
    )
    reference_fom_proc = (
        calculate_fom(y_fit, np.asarray(reference_result.y_hat_total, dtype=np.float64))
        if reference_valid and reference_result is not None
        else float("inf")
    )
    reference_fom_raw = (
        calculate_fom(y_raw, np.asarray(reference_result.y_hat_total, dtype=np.float64))
        if reference_valid and reference_result is not None
        else float("inf")
    )
    reference_warning_count = (
        count_warning_tokens(reference_result) if reference_result is not None else np.nan
    )
    reference_hit_bounds_count = (
        count_hit_bounds(reference_result) if reference_result is not None else np.nan
    )
    reference_tm_order_inversion = (
        has_tm_order_inversion(reference_result) if reference_result is not None else True
    )
    reference_mean_rel_change = (
        mean_relative_change_vs_seed(reference_result, REFERENCE_SEEDS)
        if reference_result is not None
        else float("inf")
    )

    start_rows: list[dict[str, Any]] = []
    fom_tie_tol = 5e-4
    best_candidate: BestResult | None = None

    for start_id in range(1, config.nstart + 1):
        start_seeds: list[dict[str, float]] = []
        for seed in REFERENCE_SEEDS:
            bounds = bounds_by_peak[int(seed["peak_index"])]
            start_seeds.append(
                jitter_seed(
                    rng=rng,
                    seed=seed,
                    bounds=bounds,
                    kkf=config.kkf,
                    tm_jitter_k=config.tm_jitter_k,
                )
            )

        specs = build_peak_specs(start_seeds=start_seeds, bounds_by_peak=bounds_by_peak)
        trial_result, runtime_s = run_trial(
            temperature=t_arr,
            y_fit=y_fit,
            peak_specs=specs,
            config=config,
            uncertainty_enabled=False,
        )

        if trial_result is None:
            start_rows.append(
                {
                    "start_id": start_id,
                    "valid": False,
                    "converged": False,
                    "fom": np.nan,
                    "runtime_s": runtime_s,
                    "n_iter": 0,
                    "warning_count": np.nan,
                    "hit_bounds_count": np.nan,
                    "mean_rel_change_pct": np.nan,
                    "tm_order_inversion": True,
                    "message": "exception",
                }
            )
        else:
            is_valid = bool(trial_result.converged) and result_is_valid(trial_result, y_fit)
            fom = calculate_fom(y_fit, np.asarray(trial_result.y_hat_total, dtype=np.float64))
            warning_count = count_warning_tokens(trial_result)
            hit_bounds_count = count_hit_bounds(trial_result)
            tm_order_inversion = has_tm_order_inversion(trial_result)
            mean_rel_change = mean_relative_change_vs_seed(trial_result, REFERENCE_SEEDS)

            start_rows.append(
                {
                    "start_id": start_id,
                    "valid": is_valid,
                    "converged": bool(trial_result.converged),
                    "fom": fom if is_valid else np.nan,
                    "runtime_s": runtime_s,
                    "n_iter": int(trial_result.n_iter),
                    "warning_count": warning_count,
                    "hit_bounds_count": hit_bounds_count,
                    "mean_rel_change_pct": mean_rel_change,
                    "tm_order_inversion": bool(tm_order_inversion),
                    "message": str(trial_result.message),
                }
            )

            if is_valid:
                cand_secondary = (
                    int(tm_order_inversion),
                    hit_bounds_count,
                    warning_count,
                    mean_rel_change,
                    fom,
                )
                if best_candidate is None:
                    best_candidate = BestResult(
                        start_id=start_id,
                        fom=fom,
                        runtime_s=runtime_s,
                        warning_count=warning_count,
                        hit_bounds_count=hit_bounds_count,
                        mean_rel_change_pct=mean_rel_change,
                        tm_order_inversion=bool(tm_order_inversion),
                        result=trial_result,
                        seeds=start_seeds,
                    )
                else:
                    best_secondary = (
                        int(best_candidate.tm_order_inversion),
                        best_candidate.hit_bounds_count,
                        best_candidate.warning_count,
                        best_candidate.mean_rel_change_pct,
                        best_candidate.fom,
                    )
                    better_fom = fom < (best_candidate.fom - fom_tie_tol)
                    same_fom_bucket = abs(fom - best_candidate.fom) <= fom_tie_tol
                    if better_fom or (same_fom_bucket and cand_secondary < best_secondary):
                        best_candidate = BestResult(
                            start_id=start_id,
                            fom=fom,
                            runtime_s=runtime_s,
                            warning_count=warning_count,
                            hit_bounds_count=hit_bounds_count,
                            mean_rel_change_pct=mean_rel_change,
                            tm_order_inversion=bool(tm_order_inversion),
                            result=trial_result,
                            seeds=start_seeds,
                        )

        if start_id % config.progress_every == 0 or start_id == config.nstart:
            valid_count = int(sum(bool(row["valid"]) for row in start_rows))
            best_fom_txt = "n/a" if best_candidate is None else f"{best_candidate.fom:.6f}%"
            best_warn_txt = "n/a" if best_candidate is None else str(best_candidate.warning_count)
            best_hit_txt = "n/a" if best_candidate is None else str(best_candidate.hit_bounds_count)
            print(
                f"[bench5] start {start_id}/{config.nstart} | "
                f"valid={valid_count} | best_fom={best_fom_txt} | "
                f"best_warn={best_warn_txt} | best_hit_bounds={best_hit_txt}",
                flush=True,
            )

    starts_df = pd.DataFrame(start_rows)
    if best_candidate is None:
        raise RuntimeError("All multistart attempts failed or produced invalid fits.")

    # Re-fit best start with uncertainty enabled to extract u_c(T) metrics.
    best_specs = build_peak_specs(start_seeds=best_candidate.seeds, bounds_by_peak=bounds_by_peak)
    final_result, final_runtime_s = run_trial(
        temperature=t_arr,
        y_fit=y_fit,
        peak_specs=best_specs,
        config=config,
        uncertainty_enabled=True,
    )
    if final_result is None or not result_is_valid(final_result, y_fit):
        raise RuntimeError("Final uncertainty-enabled fit failed for selected best start.")

    final_fom = calculate_fom(y_fit, np.asarray(final_result.y_hat_total, dtype=np.float64))
    best = BestResult(
        start_id=best_candidate.start_id,
        fom=final_fom,
        runtime_s=final_runtime_s,
        warning_count=count_warning_tokens(final_result),
        hit_bounds_count=count_hit_bounds(final_result),
        mean_rel_change_pct=mean_relative_change_vs_seed(final_result, REFERENCE_SEEDS),
        tm_order_inversion=has_tm_order_inversion(final_result),
        result=final_result,
        seeds=best_candidate.seeds,
    )

    n_success = int(starts_df["valid"].fillna(False).sum()) if not starts_df.empty else 0
    n_failed = int(config.nstart - n_success)
    total_runtime_s = float(starts_df["runtime_s"].fillna(0.0).sum() + final_runtime_s)

    uc_global = _safe_float(best.result.metrics.uc_global)
    uc_p95 = _safe_float(best.result.metrics.uc_p95)
    uc_max = _safe_float(best.result.metrics.uc_max)

    model_results_df = pd.DataFrame(
        [
            {
                "bench": "bench5",
                "curve": CURVE_ID,
                "curve_label": CURVE_DISPLAY,
                "data_file": str(data_file),
                "data_file_sha256": data_file_sha256,
                "data_hash_algorithm": "sha256",
                "model": MODEL_INTERNAL,
                "model_key": MODEL_KEY,
                "model_display": MODEL_DISPLAY,
                "beta": BETA,
                "n_peaks": len(REFERENCE_SEEDS),
                "nstart": config.nstart,
                "kkf": config.kkf,
                "tm_jitter_k": config.tm_jitter_k,
                "strategy": config.strategy,
                "preprocess_mode": preprocess_mode,
                "smoothing_applied": config.enable_smoothing,
                "baseline_applied": config.enable_baseline,
                "best_start_id": best.start_id,
                "reference_strategy": "local",
                "reference_runtime_s": reference_runtime_s,
                "reference_valid": bool(reference_valid),
                "reference_FOM_proc": reference_fom_proc if np.isfinite(reference_fom_proc) else np.nan,
                "reference_FOM_raw": reference_fom_raw if np.isfinite(reference_fom_raw) else np.nan,
                "reference_warning_count": reference_warning_count,
                "reference_hit_bounds_count": reference_hit_bounds_count,
                "reference_tm_order_inversion": bool(reference_tm_order_inversion),
                "reference_mean_rel_change_pct": (
                    reference_mean_rel_change if np.isfinite(reference_mean_rel_change) else np.nan
                ),
                "FOM_proc": best.fom,
                "FOM_raw": calculate_fom(y_raw, np.asarray(best.result.y_hat_total, dtype=np.float64)),
                "R2": float(best.result.metrics.R2),
                "SSR": float(best.result.metrics.SSR),
                "uc_global": uc_global,
                "uc_p95": uc_p95,
                "uc_max": uc_max,
                "best_warning_count": best.warning_count,
                "best_hit_bounds_count": best.hit_bounds_count,
                "best_tm_order_inversion": bool(best.tm_order_inversion),
                "best_mean_rel_change_pct": best.mean_rel_change_pct,
                "n_success": n_success,
                "n_failed": n_failed,
                "runtime_s_best": best.runtime_s,
                "runtime_s_total": total_runtime_s,
                "converged": bool(best.result.converged),
                "warnings": best.warning_count,
                "message": str(best.result.message),
            }
        ]
    )

    peak_rows: list[dict[str, Any]] = []
    uc_curve = None if best.result.uc_curve is None else np.asarray(best.result.uc_curve, dtype=np.float64)
    for idx, peak in enumerate(best.result.peaks, start=1):
        params = dict(peak.params)
        peak_idx = parse_peak_id(peak.name, fallback=idx)
        tm_fit = _safe_float(params.get("Tm"))
        uc_tm = None if tm_fit is None else uc_at_temperature(t_arr, uc_curve, tm_fit)

        seed_row = REFERENCE_SEEDS[peak_idx - 1]
        peak_rows.append(
            {
                "curve": CURVE_ID,
                "model": MODEL_INTERNAL,
                "model_key": MODEL_KEY,
                "peak_index": peak_idx,
                "peak_name": peak.name,
                "Im": _safe_float(params.get("Im")),
                "E": _safe_float(params.get("E")),
                "Tm": tm_fit,
                "R": _safe_float(params.get("R")),
                "s_fit": _safe_float(params.get("s")),
                "unc_Im": _safe_float(peak.uncertainties.get("Im")),
                "unc_E": _safe_float(peak.uncertainties.get("E")),
                "unc_Tm": _safe_float(peak.uncertainties.get("Tm")),
                "unc_R": _safe_float(peak.uncertainties.get("R")),
                "u_c_at_Tm": uc_tm,
                "area": float(peak.area),
                "seed_Im": float(seed_row["Im"]),
                "seed_E": float(seed_row["E"]),
                "seed_Tm": float(seed_row["Tm"]),
                "seed_R": float(seed_row["R"]),
                "FOM_proc": best.fom,
                "uc_global": uc_global,
                "uc_p95": uc_p95,
                "converged": bool(best.result.converged),
            }
        )
    peak_df = pd.DataFrame(peak_rows).sort_values("peak_index").reset_index(drop=True)

    seed_vs_fit_df = build_seed_vs_fit_table(best)

    if uc_curve is not None and uc_curve.size == t_arr.size:
        uc_curve_df = pd.DataFrame(
            {"temperature_K": t_arr, "u_c_percent": uc_curve, "intensity_fit": best.result.y_hat_total}
        )
    else:
        uc_curve_df = pd.DataFrame({"temperature_K": t_arr, "u_c_percent": np.nan, "intensity_fit": best.result.y_hat_total})

    figure_path = config.output_dir / "bench5_x009_otor_lw_fit_residual.pdf"
    plot_fit_with_residual(
        output_path=figure_path,
        temperature=t_arr,
        y_obs=y_raw,
        best=best,
        beta=BETA,
        dpi=config.dpi,
    )

    starts_path = config.output_dir / "bench5_x009_otor_lw_starts.csv"
    model_path = config.output_dir / "bench5_model_results.csv"
    peaks_path = config.output_dir / "bench5_peak_params_long.csv"
    seed_cmp_path = config.output_dir / "bench5_x009_seed_vs_fit.csv"
    uc_curve_path = config.output_dir / "bench5_x009_uc_curve.csv"
    summary_path = config.output_dir / "summary_bench5.txt"

    starts_df.to_csv(starts_path, index=False)
    model_results_df.to_csv(model_path, index=False)
    peak_df.to_csv(peaks_path, index=False)
    seed_vs_fit_df.to_csv(seed_cmp_path, index=False)
    uc_curve_df.to_csv(uc_curve_path, index=False)

    summary_lines = [
        "TLDecPy BENCH5 - RefGlow x009 OTOR_LW",
        "",
        f"Curve: {CURVE_ID} ({CURVE_DISPLAY})",
        f"Data file: {data_file}",
        f"Data file SHA-256: {data_file_sha256}",
        f"Model: {MODEL_INTERNAL} ({MODEL_DISPLAY})",
        f"Beta: {BETA:.3g} K/s",
        f"Preprocess mode: {preprocess_mode}",
        f"nstart={config.nstart}, kkf={config.kkf:.4f}, tm_jitter_k={config.tm_jitter_k:.3f}",
        f"Strategy: {config.strategy}",
        "",
        f"Successful starts: {n_success}",
        f"Failed starts: {n_failed}",
        f"Reference fit (exact seeds, local) valid: {bool(reference_valid)}",
        (
            "Reference FOM (proc/raw): "
            f"{reference_fom_proc:.6f}% / {reference_fom_raw:.6f}%"
            if np.isfinite(reference_fom_proc) and np.isfinite(reference_fom_raw)
            else "Reference FOM (proc/raw): n/a"
        ),
        (
            "Reference diagnostics: "
            f"warnings={reference_warning_count}, "
            f"hit_bounds={reference_hit_bounds_count}, "
            f"Tm inversion={bool(reference_tm_order_inversion)}, "
            f"mean|Δ|%={reference_mean_rel_change:.4f}"
            if np.isfinite(reference_mean_rel_change)
            else "Reference diagnostics: n/a"
        ),
        "",
        f"Best start id: {best.start_id}",
        f"Best FOM (proc/raw): {best.fom:.6f}% / {calculate_fom(y_raw, np.asarray(best.result.y_hat_total, dtype=np.float64)):.6f}%",
        f"u_c_global: {'n/a' if uc_global is None else f'{uc_global:.6f}%'}",
        f"u_c_p95: {'n/a' if uc_p95 is None else f'{uc_p95:.6f}%'}",
        (
            "Best diagnostics: "
            f"warnings={best.warning_count}, "
            f"hit_bounds={best.hit_bounds_count}, "
            f"Tm inversion={best.tm_order_inversion}, "
            f"mean|Δ|%={best.mean_rel_change_pct:.4f}"
        ),
        "",
        f"Figure: {figure_path}",
        f"Model table: {model_path}",
        f"Peak table: {peaks_path}",
        f"Seed comparison: {seed_cmp_path}",
        f"Starts table: {starts_path}",
        f"u_c(T) curve: {uc_curve_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("BENCH5 completed.")
    print(
        f"{CURVE_ID} | {MODEL_INTERNAL} | beta={BETA:.3g} K/s | "
        f"best FOM={best.fom:.6f}% | "
        f"u_c_global={'n/a' if uc_global is None else f'{uc_global:.6f}%'} | "
        f"hit_bounds={best.hit_bounds_count} | tm_inversion={best.tm_order_inversion}"
    )
    print(f"Output directory: {config.output_dir}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
