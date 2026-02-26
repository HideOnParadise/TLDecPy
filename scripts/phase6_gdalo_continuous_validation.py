#!/usr/bin/env python3
"""GDALO_CONTINUOUS_VALIDATION benchmark (Exp + Gauss + Local + Local).

This benchmark reproduces a mixed deconvolution workflow inspired by the
GdAlO3 paper structure (Eq. 16-style composition):

1) Continuous trap distribution (Exponential)
2) Continuous trap distribution (Gaussian)
3) Localized first-order peak (closed-form)
4) Localized first-order peak (closed-form)
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
    BackgroundSpec,
    FitOptions,
    MultiFitResult,
    PeakSpec,
    RobustOptions,
    UncertaintyOptions,
)
from tldecpy.utils.provenance import file_sha256_hex  # noqa: E402
from tldecpy.utils.sg import safe_savgol  # noqa: E402

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

BENCH_NAME = "GDALO_CONTINUOUS_VALIDATION"
BENCH_PREFIX = "bench6"
MODEL_DISPLAY = "Mixed continuous + localized first-order deconvolution"

FORMAL_EXP = "Continuous trap distribution (Exponential)"
FORMAL_GAUSS = "Continuous trap distribution (Gaussian)"
FORMAL_LOCAL = "Localized first-order peak (closed-form)"


COMPONENT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": "C1_EXP",
        "kind": "cont_exp",
        "model": "cont_exp",
        "label": FORMAL_EXP,
        "tm_seed_k": 373.15,
        "e_seed": 0.812,
        "sigma_seed": 0.0346,
        "tm_window_k": 30.0,
        "target_e": 0.812,
        "target_sigma": 0.0346,
    },
    {
        "id": "C2_GAUSS",
        "kind": "cont_gauss",
        "model": "cont_gauss",
        "label": FORMAL_GAUSS,
        "tm_seed_k": 413.15,
        "e_seed": 0.860,
        "sigma_seed": 0.0400,
        "tm_window_k": 30.0,
        "target_e": 0.860,
        "target_sigma": 0.0400,
    },
    {
        "id": "C3_LOC",
        "kind": "local_fo",
        "model": "fo_ka",
        "label": FORMAL_LOCAL,
        "tm_seed_k": 513.15,
        "e_seed": 1.120,
        "sigma_seed": np.nan,
        "tm_window_k": 45.0,
        "target_e": 1.120,
        "target_sigma": np.nan,
    },
    {
        "id": "C4_LOC",
        "kind": "local_fo",
        "model": "fo_ka",
        "label": FORMAL_LOCAL,
        "tm_seed_k": 563.15,
        "e_seed": 1.068,
        "sigma_seed": np.nan,
        "tm_window_k": 45.0,
        "target_e": 1.068,
        "target_sigma": np.nan,
    },
)


@dataclass(frozen=True)
class RunConfig:
    """Typed runtime configuration."""

    data_path: Path
    output_dir: Path
    strategy: str
    max_nfev: int
    nstart: int
    tm_jitter_k: float
    retry_local_tm_jitter_k: float
    seed: int
    dpi: int
    progress_every: int
    enable_smoothing: bool
    bg_mode: str
    sg_window: int
    sg_poly: int


@dataclass(frozen=True)
class DataPack:
    """Loaded signal plus unit metadata."""

    temperature_input: FloatArray
    temperature_k: FloatArray
    intensity_raw: FloatArray
    temperature_unit_input: str
    converted_c_to_k: bool


@dataclass(frozen=True)
class PreprocessPack:
    """Preprocessed signal and audit metadata."""

    y_fit: FloatArray
    mode: str
    baseline_used: bool
    smoothing_used: bool


@dataclass(frozen=True)
class Candidate:
    """One optimization candidate used in start ranking."""

    result: MultiFitResult
    runtime_s: float
    fom_fit: float
    fom_raw: float
    warning_count: int
    hit_bounds_count: int
    mean_rel_change_pct: float
    degenerate: bool
    bounds_expanded: bool
    start_note: str
    init_seed_rows: list[dict[str, Any]]


def build_cli() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=f"{BENCH_NAME} benchmark runner.")
    parser.add_argument("--data-path", type=str, default="scripts/gdalo.csv")
    parser.add_argument("--output-dir", type=str, default="output/bench6_gdalo_continuous_validation")
    parser.add_argument(
        "--strategy",
        type=str,
        default="local",
        choices=["local", "global_hybrid", "global_hybrid_pso"],
    )
    parser.add_argument("--max-nfev", type=int, default=3000)
    parser.add_argument("--nstart", type=int, default=8)
    parser.add_argument("--tm-jitter-k", type=float, default=4.0)
    parser.add_argument("--retry-local-tm-jitter-k", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--enable-smoothing", action="store_true")
    parser.add_argument(
        "--bg-mode",
        type=str,
        default="exponential",
        choices=["exponential", "linear", "none"],
        help="Background model extracted by TLDecPy during fit.",
    )
    parser.add_argument("--sg-window", type=int, default=11)
    parser.add_argument("--sg-poly", type=int, default=3)
    return parser


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Normalize parsed CLI configuration."""
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
        max_nfev=max(int(args.max_nfev), 200),
        nstart=max(int(args.nstart), 1),
        tm_jitter_k=max(float(args.tm_jitter_k), 0.0),
        retry_local_tm_jitter_k=max(float(args.retry_local_tm_jitter_k), 0.0),
        seed=int(args.seed),
        dpi=int(args.dpi),
        progress_every=max(int(args.progress_every), 1),
        enable_smoothing=bool(args.enable_smoothing),
        bg_mode=str(args.bg_mode),
        sg_window=int(args.sg_window),
        sg_poly=int(args.sg_poly),
    )


def configure_matplotlib(dpi: int) -> None:
    """Apply benchmark plotting style."""
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
    """Return finite float value or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def calculate_fom(y_obs: FloatArray, y_fit: FloatArray) -> float:
    """Compute FOM (%) with area-normalized absolute residuals."""
    obs = np.asarray(y_obs, dtype=np.float64)
    fit = np.asarray(y_fit, dtype=np.float64)
    if obs.size == 0 or fit.size != obs.size:
        return float("inf")
    area = float(np.sum(obs))
    if area <= 0.0 or not np.isfinite(area):
        return float("inf")
    fom = float(100.0 * np.sum(np.abs(obs - fit)) / area)
    return fom if np.isfinite(fom) else float("inf")


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detect temperature and intensity columns."""
    normalized = {col: str(col).strip().lower() for col in df.columns}

    temp_col = None
    for col, low in normalized.items():
        if low in {"temperatura", "temperature", "temp", "t"} or "temperatura" in low:
            temp_col = col
            break
    if temp_col is None:
        raise ValueError("Could not detect temperature column (expected 'Temperatura').")

    intensity_col = None
    for col, low in normalized.items():
        if low == "l1":
            intensity_col = col
            break
    if intensity_col is None:
        # fallback: first non-temperature numeric-like column
        for col in df.columns:
            if col != temp_col:
                intensity_col = col
                break
    if intensity_col is None:
        raise ValueError("Could not detect intensity column (expected 'L1').")

    return temp_col, intensity_col


def load_gdalo_dataset(data_path: Path) -> DataPack:
    """Load gdalo.csv assuming temperature is already in kelvin."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    temp_col, intensity_col = detect_columns(frame)

    temp_input = np.asarray(frame[temp_col], dtype=np.float64)
    intensity = np.asarray(frame[intensity_col], dtype=np.float64)

    return DataPack(
        temperature_input=temp_input,
        temperature_k=temp_input.copy(),
        intensity_raw=intensity,
        temperature_unit_input="K",
        converted_c_to_k=False,
    )


def preprocess_signal(y_raw: FloatArray, config: RunConfig) -> PreprocessPack:
    """Apply optional light smoothing before fitting."""
    y = np.asarray(y_raw, dtype=np.float64).copy()
    mode_tokens: list[str] = []

    if config.enable_smoothing:
        y = np.asarray(
            safe_savgol(y, window_length=config.sg_window, polyorder=config.sg_poly),
            dtype=np.float64,
        )
        mode_tokens.append("smooth")

    mode = "raw" if not mode_tokens else "+".join(mode_tokens)
    return PreprocessPack(
        y_fit=y,
        mode=mode,
        baseline_used=False,
        smoothing_used=config.enable_smoothing,
    )


def build_background_spec(bg_mode: str) -> BackgroundSpec | None:
    """Return TLDecPy background spec for fit-time background extraction."""
    if bg_mode == "none":
        return None
    return BackgroundSpec.model_validate({"type": bg_mode})


def nearest_intensity_at_tm(temperature_k: FloatArray, intensity: FloatArray, tm_k: float) -> float:
    """Return nearest-neighbor intensity at target temperature."""
    idx = int(np.argmin(np.abs(temperature_k - float(tm_k))))
    value = float(intensity[idx])
    return max(value, 1e-12)


def build_component_defs(
    temperature_k: FloatArray,
    intensity_fit: FloatArray,
) -> list[dict[str, Any]]:
    """Build component definition list with seeds and bounds."""
    t_min = float(np.min(temperature_k))
    t_max = float(np.max(temperature_k))
    i_max = max(float(np.max(intensity_fit)), 1.0)

    defs: list[dict[str, Any]] = []
    for tpl in COMPONENT_TEMPLATES:
        tm_seed = float(tpl["tm_seed_k"])
        im_seed = nearest_intensity_at_tm(temperature_k, intensity_fit, tm_seed)
        tm_window = float(tpl["tm_window_k"])
        tm_lo = max(t_min + 1e-6, tm_seed - tm_window)
        tm_hi = min(t_max - 1e-6, tm_seed + tm_window)
        if tm_hi <= tm_lo:
            tm_lo = max(t_min + 1e-6, tm_seed - 10.0)
            tm_hi = min(t_max - 1e-6, tm_seed + 10.0)

        if str(tpl["kind"]).startswith("cont_"):
            e_bounds = (0.4, 2.5)
        else:
            e_bounds = (0.6, 3.0)

        bounds: dict[str, tuple[float, float]] = {
            "Im": (1e-12, max(i_max * 5.0, im_seed * 5.0)),
            "Tm": (tm_lo, tm_hi),
            "E": e_bounds,
        }
        if str(tpl["kind"]).startswith("cont_"):
            bounds["sigma"] = (0.005, 0.15)

        defs.append(
            {
                "id": str(tpl["id"]),
                "kind": str(tpl["kind"]),
                "model": str(tpl["model"]),
                "label": str(tpl["label"]),
                "tm_seed_k": tm_seed,
                "im_seed": im_seed,
                "e_seed": float(tpl["e_seed"]),
                "sigma_seed": float(tpl["sigma_seed"])
                if np.isfinite(float(tpl["sigma_seed"]))
                else np.nan,
                "target_e": float(tpl["target_e"]),
                "target_sigma": float(tpl["target_sigma"])
                if np.isfinite(float(tpl["target_sigma"]))
                else np.nan,
                "bounds": bounds,
            }
        )
    return defs


def jitter_inits(
    rng: np.random.Generator,
    component_defs: list[dict[str, Any]],
    tm_jitter_k: float,
    local_tm_extra: float = 0.0,
) -> list[dict[str, Any]]:
    """Create jittered initial values for one multistart run."""
    init_rows: list[dict[str, Any]] = []
    for comp in component_defs:
        bounds = dict(comp["bounds"])

        im = float(comp["im_seed"]) * float(rng.uniform(0.9, 1.1))
        tm = float(comp["tm_seed_k"]) + float(rng.uniform(-tm_jitter_k, tm_jitter_k))
        if comp["kind"] == "local_fo" and local_tm_extra > 0.0:
            tm += float(rng.uniform(-local_tm_extra, local_tm_extra))

        row = {
            "id": str(comp["id"]),
            "kind": str(comp["kind"]),
            "model": str(comp["model"]),
            "label": str(comp["label"]),
            "Im": float(np.clip(im, *bounds["Im"])),
            "Tm": float(np.clip(tm, *bounds["Tm"])),
            "E": float(comp["e_seed"]),
            "sigma": float(comp["sigma_seed"]),
        }
        init_rows.append(row)
    return init_rows


def build_specs_from_inits(
    init_rows: list[dict[str, Any]],
    component_defs: list[dict[str, Any]],
) -> list[PeakSpec]:
    """Build PeakSpec objects from initialized parameter rows."""
    defs_by_id = {str(comp["id"]): comp for comp in component_defs}
    specs: list[PeakSpec] = []

    for init in init_rows:
        comp = defs_by_id[str(init["id"])]
        bounds = dict(comp["bounds"])
        kind = str(init["kind"])

        if kind.startswith("cont_"):
            spec = PeakSpec(
                name=str(init["id"]),
                model=str(init["model"]),
                init={
                    "Tm": float(init["Tm"]),
                    "Im": float(init["Im"]),
                    "E": float(init["E"]),
                    "sigma": float(init["sigma"]),
                },
                bounds={
                    "Tm": bounds["Tm"],
                    "Im": bounds["Im"],
                    "E": bounds["E"],
                    "sigma": bounds["sigma"],
                },
            )
        else:
            spec = PeakSpec(
                name=str(init["id"]),
                model=str(init["model"]),
                init={
                    "Tm": float(init["Tm"]),
                    "Im": float(init["Im"]),
                    "E": float(init["E"]),
                },
                bounds={
                    "Tm": bounds["Tm"],
                    "Im": bounds["Im"],
                    "E": bounds["E"],
                },
            )
        specs.append(spec)

    return specs


def build_fit_options(max_nfev: int, uncertainty_enabled: bool) -> FitOptions:
    """Create fit options with optional uncertainty report."""
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
    """Create standard linear-loss robust options."""
    return RobustOptions(
        loss="linear",
        f_scale=1.0,
        weights="none",
        multi_start=0,
        ci_bootstrap=False,
        n_bootstrap=0,
    )


def extract_param(params: dict[str, float], key: str) -> float | None:
    """Extract canonical/alias parameter value."""
    if key == "Im":
        return _safe_float(params.get("Im")) if "Im" in params else _safe_float(params.get("In"))
    if key == "Tm":
        return _safe_float(params.get("Tm")) if "Tm" in params else _safe_float(params.get("Tn"))
    if key == "E":
        return _safe_float(params.get("E")) if "E" in params else _safe_float(params.get("E0"))
    return _safe_float(params.get(key))


def result_is_valid(result: MultiFitResult, y_obs: FloatArray) -> bool:
    """Validate numerical sanity of one fit result."""
    y_hat = np.asarray(result.y_hat_total, dtype=np.float64)
    if y_hat.size != y_obs.size or not np.all(np.isfinite(y_hat)):
        return False

    fom = calculate_fom(y_obs, y_hat)
    if not np.isfinite(fom):
        return False

    for peak in result.peaks:
        params = dict(peak.params)
        im = extract_param(params, "Im")
        tm = extract_param(params, "Tm")
        e_val = extract_param(params, "E")
        if im is None or tm is None or e_val is None:
            return False
        if im <= 0.0 or e_val <= 0.0:
            return False

    return True


def count_warnings(result: MultiFitResult) -> int:
    """Count warning-like conditions."""
    count = 0
    if not bool(result.converged):
        count += 1
    msg = str(result.message).lower()
    for token in ("warn", "fail", "error"):
        if token in msg:
            count += 1
    return count


def count_hit_bounds(result: MultiFitResult) -> int:
    """Count bound-hit flags in one result."""
    return int(sum(1 for value in result.hit_bounds.values() if bool(value)))


def mean_relative_change_pct(
    result: MultiFitResult,
    component_defs: list[dict[str, Any]],
) -> float:
    """Average absolute relative deviation vs benchmark seeds."""
    defs_by_id = {str(comp["id"]): comp for comp in component_defs}
    rel_errors: list[float] = []

    for peak in result.peaks:
        comp = defs_by_id.get(str(peak.name))
        if comp is None:
            continue
        params = dict(peak.params)
        im = extract_param(params, "Im")
        tm = extract_param(params, "Tm")
        e_val = extract_param(params, "E")
        if im is None or tm is None or e_val is None:
            continue

        for value, seed in (
            (im, float(comp["im_seed"])),
            (tm, float(comp["tm_seed_k"])),
            (e_val, float(comp["e_seed"])),
        ):
            if seed == 0.0:
                rel_errors.append(abs(value))
            else:
                rel_errors.append(100.0 * abs(value - seed) / abs(seed))

        sigma_seed = float(comp["sigma_seed"])
        sigma_fit = extract_param(params, "sigma")
        if np.isfinite(sigma_seed) and sigma_fit is not None and sigma_seed > 0.0:
            rel_errors.append(100.0 * abs(sigma_fit - sigma_seed) / sigma_seed)

    if not rel_errors:
        return float("inf")
    return float(np.mean(rel_errors))


def is_degenerate_solution(result: MultiFitResult) -> tuple[bool, str]:
    """Detect duplicated/degenerate components."""
    tm_rows: list[tuple[str, float]] = []
    local_rows: list[tuple[str, float]] = []

    for peak in result.peaks:
        params = dict(peak.params)
        tm = extract_param(params, "Tm")
        if tm is None:
            return True, "missing_Tm"
        tm_rows.append((str(peak.name), tm))
        if str(peak.name) in {"C3_LOC", "C4_LOC"}:
            local_rows.append((str(peak.name), tm))

    tm_sorted = sorted(tm_rows, key=lambda row: row[1])
    for i in range(len(tm_sorted) - 1):
        if abs(tm_sorted[i + 1][1] - tm_sorted[i][1]) < 4.0:
            return True, "global_tm_overlap"

    if len(local_rows) == 2:
        if abs(local_rows[0][1] - local_rows[1][1]) < 8.0:
            return True, "local_tm_overlap"

    return False, "ok"


def uc_at_tm(temperature_k: FloatArray, uc_curve: FloatArray | None, tm_value: float) -> float | None:
    """Interpolate u_c(T) at fitted Tm."""
    if uc_curve is None:
        return None
    t = np.asarray(temperature_k, dtype=np.float64)
    uc = np.asarray(uc_curve, dtype=np.float64)
    if uc.size != t.size or not np.any(np.isfinite(uc)):
        return None
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if tm_value < t_min or tm_value > t_max:
        return None
    value = float(np.interp(tm_value, t, uc))
    return value if np.isfinite(value) else None


def run_fit_once(
    temperature_k: FloatArray,
    y_fit: FloatArray,
    specs: list[PeakSpec],
    config: RunConfig,
    bg_spec: BackgroundSpec | None,
    *,
    uncertainty_enabled: bool,
) -> tuple[MultiFitResult | None, float]:
    """Run one fit attempt and return result plus runtime."""
    robust = build_robust_options()
    options = build_fit_options(max_nfev=config.max_nfev, uncertainty_enabled=uncertainty_enabled)

    start = perf_counter()
    try:
        result = fit_multi(
            temperature_k,
            y_fit,
            peaks=specs,
            bg=bg_spec,
            robust=robust,
            options=options,
            strategy=config.strategy,  # type: ignore[arg-type]
        )
    except Exception:
        return None, float(perf_counter() - start)
    return result, float(perf_counter() - start)


def build_init_from_result(
    result: MultiFitResult,
    component_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert fitted result into init rows for a second-stage refit."""
    defs_by_id = {str(comp["id"]): comp for comp in component_defs}
    init_rows: list[dict[str, Any]] = []
    for peak in result.peaks:
        comp = defs_by_id[str(peak.name)]
        params = dict(peak.params)
        im = extract_param(params, "Im")
        tm = extract_param(params, "Tm")
        e_val = extract_param(params, "E")
        sigma = extract_param(params, "sigma")
        init_rows.append(
            {
                "id": str(comp["id"]),
                "kind": str(comp["kind"]),
                "model": str(comp["model"]),
                "label": str(comp["label"]),
                "Im": float(im if im is not None else comp["im_seed"]),
                "Tm": float(tm if tm is not None else comp["tm_seed_k"]),
                "E": float(e_val if e_val is not None else comp["e_seed"]),
                "sigma": float(
                    sigma if sigma is not None and np.isfinite(sigma) else comp["sigma_seed"]
                ),
            }
        )
    return init_rows


def expand_component_bounds(
    component_defs: list[dict[str, Any]],
    temperature_k: FloatArray,
) -> list[dict[str, Any]]:
    """Expand bounds for automatic retry when solutions hit limits."""
    t_min = float(np.min(temperature_k))
    t_max = float(np.max(temperature_k))
    expanded: list[dict[str, Any]] = []

    for comp in component_defs:
        bounds = dict(comp["bounds"])
        tm_lo, tm_hi = bounds["Tm"]
        im_lo, im_hi = bounds["Im"]
        e_lo, e_hi = bounds["E"]

        bounds["Tm"] = (max(t_min + 1e-6, tm_lo - 10.0), min(t_max - 1e-6, tm_hi + 10.0))
        bounds["Im"] = (max(1e-12, 0.5 * im_lo), max(im_hi * 1.8, im_lo + 1e-9))
        bounds["E"] = (max(0.2, e_lo - 0.2), min(4.0, e_hi + 0.3))
        if bounds["E"][1] <= bounds["E"][0]:
            bounds["E"] = (bounds["E"][0], bounds["E"][0] + 0.05)

        if "sigma" in bounds:
            s_lo, s_hi = bounds["sigma"]
            new_lo = max(0.003, s_lo * 0.7)
            new_hi = min(0.25, s_hi * 1.5)
            if new_hi <= new_lo:
                new_hi = min(0.25, new_lo + 0.01)
            bounds["sigma"] = (new_lo, new_hi)

        expanded_comp = dict(comp)
        expanded_comp["bounds"] = bounds
        expanded.append(expanded_comp)

    return expanded


def evaluate_candidate(
    result: MultiFitResult,
    runtime_s: float,
    y_fit: FloatArray,
    y_raw: FloatArray,
    component_defs: list[dict[str, Any]],
    note: str,
    init_seed_rows: list[dict[str, Any]],
    bounds_expanded: bool,
) -> Candidate:
    """Build candidate metadata object."""
    y_hat = np.asarray(result.y_hat_total, dtype=np.float64)
    fom_fit = calculate_fom(y_fit, y_hat)
    fom_raw = calculate_fom(y_raw, y_hat)
    warning_count = count_warnings(result)
    hit_bounds_count = count_hit_bounds(result)
    mean_rel = mean_relative_change_pct(result, component_defs)
    degenerate, reason = is_degenerate_solution(result)
    note_combined = note if reason == "ok" else f"{note}; {reason}"
    return Candidate(
        result=result,
        runtime_s=runtime_s,
        fom_fit=fom_fit,
        fom_raw=fom_raw,
        warning_count=warning_count,
        hit_bounds_count=hit_bounds_count,
        mean_rel_change_pct=mean_rel,
        degenerate=degenerate,
        bounds_expanded=bounds_expanded,
        start_note=note_combined,
        init_seed_rows=init_seed_rows,
    )


def candidate_rank(candidate: Candidate) -> tuple[float, int, int, int, float]:
    """Ranking tuple for best-candidate selection."""
    return (
        candidate.fom_fit,
        int(candidate.degenerate),
        candidate.hit_bounds_count,
        candidate.warning_count,
        candidate.mean_rel_change_pct,
    )


def select_best_candidate(candidates: list[Candidate]) -> Candidate:
    """Return best valid candidate."""
    valid = [cand for cand in candidates if np.isfinite(cand.fom_fit)]
    if not valid:
        raise RuntimeError("No valid candidate fits found.")
    valid.sort(key=candidate_rank)
    return valid[0]


def plot_fit(
    output_pdf: Path,
    output_png: Path,
    data: DataPack,
    pre: PreprocessPack,
    best: Candidate,
    component_defs: list[dict[str, Any]],
    dpi: int,
) -> None:
    """Create figure with fit + components and residual panel."""
    configure_matplotlib(dpi=dpi)

    x_plot = data.temperature_input
    x_label = "Temperature (°C)" if data.converted_c_to_k else "Temperature, T (K)"

    y_raw = np.asarray(data.intensity_raw, dtype=np.float64)
    y_fit = np.asarray(pre.y_fit, dtype=np.float64)
    y_hat = np.asarray(best.result.y_hat_total, dtype=np.float64)
    residual = y_fit - y_hat

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.08},
        constrained_layout=True,
    )

    ax_top.plot(
        x_plot,
        y_raw,
        linestyle="none",
        marker="o",
        markersize=2.0,
        color="0.55",
        alpha=0.7,
        label="Experimental data",
    )
    if pre.mode != "raw":
        ax_top.plot(
            x_plot,
            y_fit,
            linestyle="none",
            marker=".",
            markersize=1.8,
            color="#1f77b4",
            alpha=0.8,
            label="Preprocessed signal",
        )

    ax_top.plot(x_plot, y_hat, color="black", linewidth=2.1, label="Total fit")
    if best.result.background is not None:
        ax_top.plot(
            x_plot,
            np.asarray(best.result.background.y_hat, dtype=np.float64),
            color="0.4",
            linewidth=1.2,
            linestyle=":",
            label="Exponential background",
        )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    defs_by_id = {str(comp["id"]): comp for comp in component_defs}
    for idx, peak in enumerate(best.result.peaks):
        comp = defs_by_id.get(str(peak.name), {})
        label = str(comp.get("label", "Component"))
        if str(peak.name) == "C3_LOC":
            label = f"{FORMAL_LOCAL} A"
        if str(peak.name) == "C4_LOC":
            label = f"{FORMAL_LOCAL} B"
        ax_top.plot(
            x_plot,
            np.asarray(peak.y_hat, dtype=np.float64),
            linestyle="--",
            linewidth=1.4,
            color=colors[idx % len(colors)],
            alpha=0.95,
            label=label,
        )

    uc_global = _safe_float(best.result.metrics.uc_global)
    uc_p95 = _safe_float(best.result.metrics.uc_p95)
    info_lines = [
        BENCH_NAME,
        MODEL_DISPLAY,
        f"FOM_fit={best.fom_fit:.4f}% | FOM_raw={best.fom_raw:.4f}%",
        f"u_c_global={'n/a' if uc_global is None else f'{uc_global:.3f}%'} | "
        f"u_c_p95={'n/a' if uc_p95 is None else f'{uc_p95:.3f}%'}",
        f"start_note={best.start_note}",
    ]
    ax_top.text(
        0.015,
        0.98,
        "\n".join(info_lines),
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.0},
    )
    ax_top.set_ylabel("Intensity (a.u.)")
    ax_top.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
    ax_top.legend(loc="upper right", fontsize=7, frameon=True)

    ax_bot.plot(
        x_plot,
        residual,
        linestyle="none",
        marker="o",
        markersize=1.9,
        color="#8c4b12",
        alpha=0.75,
    )
    ax_bot.axhline(0.0, color="black", linewidth=0.9)
    q99 = float(np.nanpercentile(np.abs(residual), 99))
    if not np.isfinite(q99) or q99 <= 0.0:
        q99 = float(np.nanmax(np.abs(residual))) if residual.size > 0 else 1.0
    if not np.isfinite(q99) or q99 <= 0.0:
        q99 = 1.0
    ax_bot.set_ylim(-1.1 * q99, 1.1 * q99)
    ax_bot.set_ylabel("Residual")
    ax_bot.set_xlabel(x_label)
    ax_bot.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)

    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def public_component_label(component_id: str, formal_label: str) -> str:
    """Return public-facing label without internal ids."""
    if component_id == "C3_LOC":
        return f"{formal_label} A"
    if component_id == "C4_LOC":
        return f"{formal_label} B"
    return formal_label


def main() -> None:
    """Execute GDALO_CONTINUOUS_VALIDATION benchmark."""
    config = parse_config(build_cli().parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    data = load_gdalo_dataset(config.data_path)
    data_file_sha256 = file_sha256_hex(config.data_path)
    pre = preprocess_signal(data.intensity_raw, config)
    bg_spec = build_background_spec(config.bg_mode)
    component_defs = build_component_defs(data.temperature_k, pre.y_fit)

    all_candidates: list[Candidate] = []
    start_rows: list[dict[str, Any]] = []

    for start_id in range(1, config.nstart + 1):
        init_rows = jitter_inits(
            rng=rng,
            component_defs=component_defs,
            tm_jitter_k=config.tm_jitter_k,
            local_tm_extra=0.0,
        )
        specs = build_specs_from_inits(init_rows=init_rows, component_defs=component_defs)
        result, runtime_s = run_fit_once(
            data.temperature_k,
            pre.y_fit,
            specs,
            config,
            bg_spec,
            uncertainty_enabled=False,
        )

        if result is None or not bool(result.converged) or not result_is_valid(result, pre.y_fit):
            start_rows.append(
                {
                    "start_id": start_id,
                    "valid": False,
                    "fom_fit": np.nan,
                    "fom_raw": np.nan,
                    "runtime_s": runtime_s,
                    "warning_count": np.nan,
                    "hit_bounds_count": np.nan,
                    "mean_rel_change_pct": np.nan,
                    "degenerate": True,
                    "bounds_expanded": False,
                    "note": "fit_failed",
                }
            )
            continue

        candidate = evaluate_candidate(
            result=result,
            runtime_s=runtime_s,
            y_fit=pre.y_fit,
            y_raw=data.intensity_raw,
            component_defs=component_defs,
            note="initial",
            init_seed_rows=init_rows,
            bounds_expanded=False,
        )

        # Automatic bound expansion retry when hitting bounds.
        if candidate.hit_bounds_count > 0:
            expanded_defs = expand_component_bounds(component_defs, data.temperature_k)
            expanded_inits = build_init_from_result(candidate.result, component_defs)
            expanded_specs = build_specs_from_inits(
                init_rows=expanded_inits, component_defs=expanded_defs
            )
            exp_result, exp_runtime = run_fit_once(
                data.temperature_k,
                pre.y_fit,
                expanded_specs,
                config,
                bg_spec,
                uncertainty_enabled=False,
            )
            if (
                exp_result is not None
                and bool(exp_result.converged)
                and result_is_valid(exp_result, pre.y_fit)
            ):
                expanded_candidate = evaluate_candidate(
                    result=exp_result,
                    runtime_s=candidate.runtime_s + exp_runtime,
                    y_fit=pre.y_fit,
                    y_raw=data.intensity_raw,
                    component_defs=expanded_defs,
                    note="bounds_expanded",
                    init_seed_rows=expanded_inits,
                    bounds_expanded=True,
                )
                if candidate_rank(expanded_candidate) < candidate_rank(candidate):
                    candidate = expanded_candidate

        # Degeneracy retry with larger local Tm jitter.
        if candidate.degenerate:
            retry_rows = jitter_inits(
                rng=rng,
                component_defs=component_defs,
                tm_jitter_k=config.tm_jitter_k,
                local_tm_extra=config.retry_local_tm_jitter_k,
            )
            retry_specs = build_specs_from_inits(init_rows=retry_rows, component_defs=component_defs)
            retry_result, retry_runtime = run_fit_once(
                data.temperature_k,
                pre.y_fit,
                retry_specs,
                config,
                bg_spec,
                uncertainty_enabled=False,
            )
            if (
                retry_result is not None
                and bool(retry_result.converged)
                and result_is_valid(retry_result, pre.y_fit)
            ):
                retry_candidate = evaluate_candidate(
                    result=retry_result,
                    runtime_s=candidate.runtime_s + retry_runtime,
                    y_fit=pre.y_fit,
                    y_raw=data.intensity_raw,
                    component_defs=component_defs,
                    note="degeneracy_retry",
                    init_seed_rows=retry_rows,
                    bounds_expanded=candidate.bounds_expanded,
                )
                if candidate_rank(retry_candidate) < candidate_rank(candidate):
                    candidate = retry_candidate

        all_candidates.append(candidate)
        start_rows.append(
            {
                "start_id": start_id,
                "valid": True,
                "fom_fit": candidate.fom_fit,
                "fom_raw": candidate.fom_raw,
                "runtime_s": candidate.runtime_s,
                "warning_count": candidate.warning_count,
                "hit_bounds_count": candidate.hit_bounds_count,
                "mean_rel_change_pct": candidate.mean_rel_change_pct,
                "degenerate": candidate.degenerate,
                "bounds_expanded": candidate.bounds_expanded,
                "note": candidate.start_note,
            }
        )

        if start_id % config.progress_every == 0 or start_id == config.nstart:
            current_best = select_best_candidate(all_candidates) if all_candidates else None
            best_txt = "n/a" if current_best is None else f"{current_best.fom_fit:.4f}%"
            print(
                f"[{BENCH_PREFIX}] start {start_id}/{config.nstart} | "
                f"valid={len(all_candidates)} | best_fom={best_txt}",
                flush=True,
            )

    if not all_candidates:
        raise RuntimeError("No valid candidate from multistart.")

    best_screen = select_best_candidate(all_candidates)

    # Final uncertainty-enabled refit from best screened initialization.
    final_specs = build_specs_from_inits(
        init_rows=best_screen.init_seed_rows,
        component_defs=component_defs,
    )
    final_result, final_runtime = run_fit_once(
        data.temperature_k,
        pre.y_fit,
        final_specs,
        config,
        bg_spec,
        uncertainty_enabled=True,
    )
    if final_result is None or not bool(final_result.converged) or not result_is_valid(
        final_result, pre.y_fit
    ):
        # fallback: keep screened result if uncertainty rerun fails
        final_candidate = best_screen
        final_refit_ok = False
    else:
        final_candidate = evaluate_candidate(
            result=final_result,
            runtime_s=best_screen.runtime_s + final_runtime,
            y_fit=pre.y_fit,
            y_raw=data.intensity_raw,
            component_defs=component_defs,
            note=f"{best_screen.start_note}; uncertainty_refit",
            init_seed_rows=best_screen.init_seed_rows,
            bounds_expanded=best_screen.bounds_expanded,
        )
        final_refit_ok = True

    # Build output tables.
    starts_df = pd.DataFrame(start_rows).sort_values("start_id").reset_index(drop=True)
    best_result = final_candidate.result
    uc_global = _safe_float(best_result.metrics.uc_global)
    uc_p95 = _safe_float(best_result.metrics.uc_p95)
    uc_max = _safe_float(best_result.metrics.uc_max)
    bg_type = best_result.background.type if best_result.background is not None else "none"
    bg_params = best_result.background.params if best_result.background is not None else {}

    model_results_df = pd.DataFrame(
        [
            {
                "bench": BENCH_NAME,
                "curve": "gdalo_L1",
                "model_display": MODEL_DISPLAY,
                "data_file": str(config.data_path),
                "data_file_sha256": data_file_sha256,
                "data_hash_algorithm": "sha256",
                "temperature_input_unit": data.temperature_unit_input,
                "converted_c_to_k": bool(data.converted_c_to_k),
                "preprocess_mode": pre.mode,
                "smoothing_applied": pre.smoothing_used,
                "baseline_applied": pre.baseline_used,
                "bg_mode": config.bg_mode,
                "bg_type_fitted": bg_type,
                "bg_a": _safe_float(bg_params.get("a")),
                "bg_b": _safe_float(bg_params.get("b")),
                "bg_c": _safe_float(bg_params.get("c")),
                "strategy": config.strategy,
                "nstart": config.nstart,
                "tm_jitter_k": config.tm_jitter_k,
                "retry_local_tm_jitter_k": config.retry_local_tm_jitter_k,
                "max_nfev": config.max_nfev,
                "best_start_id": int(starts_df.loc[starts_df["fom_fit"].idxmin(), "start_id"])
                if starts_df["fom_fit"].notna().any()
                else np.nan,
                "FOM_fit": final_candidate.fom_fit,
                "FOM_raw": final_candidate.fom_raw,
                "R2": float(best_result.metrics.R2),
                "SSR": float(best_result.metrics.SSR),
                "uc_global": uc_global,
                "uc_p95": uc_p95,
                "uc_max": uc_max,
                "warning_count": final_candidate.warning_count,
                "hit_bounds_count": final_candidate.hit_bounds_count,
                "degenerate": final_candidate.degenerate,
                "bounds_expanded": final_candidate.bounds_expanded,
                "final_uncertainty_refit_ok": bool(final_refit_ok),
                "runtime_s": final_candidate.runtime_s,
                "message": str(best_result.message),
            }
        ]
    )

    defs_by_id = {str(comp["id"]): comp for comp in component_defs}
    peak_rows: list[dict[str, Any]] = []
    for peak in best_result.peaks:
        comp = defs_by_id[str(peak.name)]
        params = dict(peak.params)
        tm_k = extract_param(params, "Tm")
        im_val = extract_param(params, "Im")
        e_val = extract_param(params, "E")
        sigma_val = extract_param(params, "sigma")
        uc_tm = None if tm_k is None else uc_at_tm(data.temperature_k, best_result.uc_curve, tm_k)

        tm_plot = tm_k - 273.15 if (tm_k is not None and data.converted_c_to_k) else tm_k
        peak_rows.append(
            {
                "curve": "gdalo_L1",
                "_component_id": str(comp["id"]),
                "component_label_formal": public_component_label(
                    str(comp["id"]), str(comp["label"])
                ),
                "component_family": (
                    "Continuous trap distribution"
                    if str(comp["kind"]).startswith("cont_")
                    else "Localized first-order peak"
                ),
                "Imax_or_Im": im_val,
                "Tm_K": tm_k,
                "Tm_plot_units": tm_plot,
                "E_or_E0_eV": e_val,
                "sigma_eV": sigma_val,
                "u_c_at_Tm_percent": uc_tm,
                "area": float(peak.area),
                "seed_Im": float(comp["im_seed"]),
                "seed_Tm_K": float(comp["tm_seed_k"]),
                "seed_E_or_E0": float(comp["e_seed"]),
                "seed_sigma": float(comp["sigma_seed"])
                if np.isfinite(float(comp["sigma_seed"]))
                else np.nan,
                "FOM_fit": final_candidate.fom_fit,
            }
        )
    peak_df = pd.DataFrame(peak_rows).sort_values("Tm_K").reset_index(drop=True)
    peak_df["report_order"] = np.arange(1, len(peak_df) + 1)

    comparison_rows: list[dict[str, Any]] = []
    for _, row in peak_df.iterrows():
        target_comp = defs_by_id[str(row["_component_id"])]
        fit_e = _safe_float(row["E_or_E0_eV"])
        target_e = float(target_comp["target_e"])
        e_rel = (
            np.nan
            if (fit_e is None or target_e == 0.0)
            else 100.0 * (float(fit_e) - target_e) / target_e
        )
        fit_sigma = _safe_float(row["sigma_eV"])
        target_sigma = (
            float(target_comp["target_sigma"])
            if np.isfinite(float(target_comp["target_sigma"]))
            else np.nan
        )
        sigma_rel = (
            np.nan
            if (
                fit_sigma is None
                or not np.isfinite(target_sigma)
                or target_sigma == 0.0
            )
            else 100.0 * (float(fit_sigma) - target_sigma) / target_sigma
        )
        comparison_rows.append(
            {
                "curve": "gdalo_L1",
                "component_label_formal": str(row["component_label_formal"]),
                "target_E_or_E0_eV": target_e,
                "fit_E_or_E0_eV": fit_e,
                "rel_error_E_percent": e_rel,
                "target_sigma_eV": target_sigma,
                "fit_sigma_eV": fit_sigma,
                "rel_error_sigma_percent": sigma_rel,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    peak_public_df = peak_df.drop(columns=["_component_id"])

    uc_curve_df = pd.DataFrame(
        {
            "temperature_input_units": data.temperature_input,
            "temperature_K": data.temperature_k,
            "u_c_percent": (
                np.asarray(best_result.uc_curve, dtype=np.float64)
                if best_result.uc_curve is not None
                else np.full_like(data.temperature_k, np.nan)
            ),
            "intensity_fit_total": np.asarray(best_result.y_hat_total, dtype=np.float64),
        }
    )

    # Save outputs.
    fig_pdf = config.output_dir / f"{BENCH_PREFIX}_gdalo_fit_residual.pdf"
    fig_png = config.output_dir / f"{BENCH_PREFIX}_gdalo_fit_residual.png"
    plot_fit(
        output_pdf=fig_pdf,
        output_png=fig_png,
        data=data,
        pre=pre,
        best=final_candidate,
        component_defs=component_defs,
        dpi=config.dpi,
    )

    starts_path = config.output_dir / f"{BENCH_PREFIX}_gdalo_starts.csv"
    model_path = config.output_dir / f"{BENCH_PREFIX}_model_results.csv"
    peaks_path = config.output_dir / f"{BENCH_PREFIX}_peak_params_long.csv"
    compare_path = config.output_dir / f"{BENCH_PREFIX}_gdalo_target_comparison.csv"
    uc_curve_path = config.output_dir / f"{BENCH_PREFIX}_gdalo_uc_curve.csv"
    summary_path = config.output_dir / f"summary_{BENCH_PREFIX}.txt"

    starts_df.to_csv(starts_path, index=False)
    model_results_df.to_csv(model_path, index=False)
    peak_public_df.to_csv(peaks_path, index=False)
    comparison_df.to_csv(compare_path, index=False)
    uc_curve_df.to_csv(uc_curve_path, index=False)

    tm_vals = np.asarray(peak_df["Tm_K"], dtype=np.float64)
    im_vals = np.asarray(peak_df["Imax_or_Im"], dtype=np.float64)
    t_lo = float(np.min(data.temperature_k))
    t_hi = float(np.max(data.temperature_k))
    sanity_tm_range_ok = bool(np.all((tm_vals >= t_lo) & (tm_vals <= t_hi)))
    sanity_im_positive = bool(np.all(im_vals > 0.0))

    summary_lines = [
        f"{BENCH_NAME}",
        "",
        f"Data file: {config.data_path}",
        f"Data file SHA-256: {data_file_sha256}",
        f"Temperature input unit: {data.temperature_unit_input}",
        f"Converted C->K: {bool(data.converted_c_to_k)}",
        f"Preprocessing: {pre.mode}",
        f"Smoothing applied: {pre.smoothing_used}",
        f"Baseline applied: {pre.baseline_used}",
        f"Background extraction mode: {config.bg_mode}",
        f"Background fitted type: {bg_type}",
        f"Background params: a={_safe_float(bg_params.get('a'))}, "
        f"b={_safe_float(bg_params.get('b'))}, c={_safe_float(bg_params.get('c'))}",
        "",
        f"Strategy: {config.strategy}",
        f"nstart: {config.nstart}",
        f"tm_jitter_k: {config.tm_jitter_k}",
        f"retry_local_tm_jitter_k: {config.retry_local_tm_jitter_k}",
        f"max_nfev: {config.max_nfev}",
        "",
        f"Best FOM_fit: {final_candidate.fom_fit:.6f}%",
        f"Best FOM_raw: {final_candidate.fom_raw:.6f}%",
        f"u_c_global: {'n/a' if uc_global is None else f'{uc_global:.6f}%'}",
        f"u_c_p95: {'n/a' if uc_p95 is None else f'{uc_p95:.6f}%'}",
        f"Hit bounds count: {final_candidate.hit_bounds_count}",
        f"Degenerate: {final_candidate.degenerate}",
        f"Bounds expanded: {final_candidate.bounds_expanded}",
        f"Final uncertainty refit ok: {final_refit_ok}",
        f"Sanity check (Tm in data range): {sanity_tm_range_ok}",
        f"Sanity check (Imax/Im > 0): {sanity_im_positive}",
        "",
        "Outputs:",
        f"- Figure PDF: {fig_pdf}",
        f"- Figure PNG: {fig_png}",
        f"- model_results: {model_path}",
        f"- peak_params_long: {peaks_path}",
        f"- target_comparison: {compare_path}",
        f"- u_c(T) curve: {uc_curve_path}",
        f"- starts log: {starts_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"{BENCH_NAME} completed.")
    print(
        f"FOM_fit={final_candidate.fom_fit:.6f}% | "
        f"FOM_raw={final_candidate.fom_raw:.6f}% | "
        f"u_c_global={'n/a' if uc_global is None else f'{uc_global:.6f}%'}"
    )
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    main()
