#!/usr/bin/env python3
"""Phase 1b benchmark: runtime comparison for auxiliary function evaluation.

Compares vectorized wall-time for:
1) SciPy exact reference (via expn)
2) TLDecPy AAA (Q_aaa)
3) Optional Numba-compiled AAA kernel (if numba available)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import expn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldecpy.utils.aaa_fo import Q_aaa, load_aaa_Q_constants  # noqa: E402

FloatArray = NDArray[np.float64]

try:
    import numba as nb  # type: ignore

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    nb = None
    HAS_NUMBA = False


RM_STYLE: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.constrained_layout.use": True,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclass(frozen=True)
class RunConfig:
    """Runtime settings for the speed benchmark."""

    output_dir: Path
    repeats: int
    warmup: int
    z_min: float
    z_max: float
    n_min_pow: int
    n_max_pow: int
    n_sizes: int
    dpi: int


def apply_style(dpi: int) -> None:
    """Apply RM style."""
    style = dict(RM_STYLE)
    style["figure.dpi"] = dpi
    style["savefig.dpi"] = dpi
    plt.rcParams.update(style)


def Q_exact_scipy(z: FloatArray) -> FloatArray:
    """Exact Q(z) reference via SciPy E1."""
    z_arr = np.asarray(z, dtype=np.float64)
    return np.asarray(1.0 - z_arr * np.exp(z_arr) * expn(1, z_arr), dtype=np.float64)


def Q_aaa_eval(z: FloatArray) -> FloatArray:
    """TLDecPy AAA Q(z) backend."""
    return np.asarray(Q_aaa(z), dtype=np.float64)


def build_numba_kernel() -> Callable[[FloatArray], FloatArray] | None:
    """Build optional Numba AAA evaluator."""
    if not HAS_NUMBA:
        return None

    constants = load_aaa_Q_constants()
    zmin = float(constants.zmin)
    zmax = float(constants.zmax)
    support = np.asarray(constants.support_points, dtype=np.float64)
    values = np.asarray(constants.support_values, dtype=np.float64)
    weights = np.asarray(constants.weights, dtype=np.float64)

    @nb.njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _q_aaa_numba(
        z: FloatArray,
        zmin_i: float,
        zmax_i: float,
        support_i: FloatArray,
        values_i: FloatArray,
        weights_i: FloatArray,
    ) -> FloatArray:
        n = z.size
        m = support_i.size
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            zi = z[i]
            if zi < zmin_i:
                zi = zmin_i
            elif zi > zmax_i:
                zi = zmax_i

            exact = -1
            num = 0.0
            den = 0.0
            for j in range(m):
                diff = zi - support_i[j]
                if diff == 0.0:
                    exact = j
                    break
                term = weights_i[j] / diff
                num += term * values_i[j]
                den += term

            if exact >= 0:
                out[i] = values_i[exact]
            else:
                out[i] = num / den
        return out

    def wrapped(z: FloatArray) -> FloatArray:
        z_arr = np.asarray(z, dtype=np.float64)
        return _q_aaa_numba(z_arr, zmin, zmax, support, values, weights)

    # Compile once (warm-up for JIT)
    _ = wrapped(np.linspace(10.0, 100.0, 128, dtype=np.float64))
    return wrapped


def build_cli() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Phase 1b speed benchmark for auxiliary backend.")
    parser.add_argument("--output-dir", type=str, default="output/phase1_validation")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--z-min", type=float, default=10.0)
    parser.add_argument("--z-max", type=float, default=100.0)
    parser.add_argument("--n-min-pow", type=int, default=3)
    parser.add_argument("--n-max-pow", type=int, default=7)
    parser.add_argument("--n-sizes", type=int, default=9)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Parse and normalize config."""
    out = Path(str(args.output_dir))
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return RunConfig(
        output_dir=out,
        repeats=max(int(args.repeats), 1),
        warmup=max(int(args.warmup), 0),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        n_min_pow=int(args.n_min_pow),
        n_max_pow=int(args.n_max_pow),
        n_sizes=max(int(args.n_sizes), 2),
        dpi=int(args.dpi),
    )


def benchmark_callable(
    func: Callable[[FloatArray], FloatArray],
    z: FloatArray,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    """Return (mean_time, std_time) in seconds."""
    sink = 0.0
    for _ in range(warmup):
        out = func(z)
        sink += float(out[0]) if out.size > 0 else 0.0

    times: list[float] = []
    for _ in range(repeats):
        t0 = perf_counter()
        out = func(z)
        t1 = perf_counter()
        sink += float(out[-1]) if out.size > 0 else 0.0
        times.append(t1 - t0)
    if sink == -1.23456789:  # pragma: no cover
        print("")
    arr = np.asarray(times, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if arr.size > 1 else 0))


def plot_results(path: Path, df: pd.DataFrame, dpi: int) -> None:
    """Create log-log speed benchmark figure."""
    apply_style(dpi=dpi)

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax.plot(
        df["N"],
        df["time_scipy_mean_s"],
        marker="o",
        color="#4E79A7",
        label="SciPy Exact",
    )
    ax.plot(
        df["N"],
        df["time_aaa_mean_s"],
        marker="o",
        color="#F28E2B",
        label="TLDecPy AAA",
    )
    if "time_numba_mean_s" in df.columns and np.any(np.isfinite(df["time_numba_mean_s"])):
        ax.plot(
            df["N"],
            df["time_numba_mean_s"],
            marker="o",
            color="#59A14F",
            label="Fast AAA (Numba)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Array size, $N$")
    ax.set_ylabel(r"Wall time (s)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    """Run end-to-end phase1 speed benchmark."""
    cfg = parse_config(build_cli().parse_args())
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    n_values = np.unique(
        np.logspace(cfg.n_min_pow, cfg.n_max_pow, cfg.n_sizes, dtype=np.int64)
    )
    numba_func = build_numba_kernel()

    rows: list[dict[str, Any]] = []
    for n in n_values:
        z = np.linspace(cfg.z_min, cfg.z_max, int(n), dtype=np.float64)
        scipy_mean, scipy_std = benchmark_callable(
            Q_exact_scipy, z, warmup=cfg.warmup, repeats=cfg.repeats
        )
        aaa_mean, aaa_std = benchmark_callable(Q_aaa_eval, z, warmup=cfg.warmup, repeats=cfg.repeats)

        row: dict[str, Any] = {
            "N": int(n),
            "time_scipy_mean_s": scipy_mean,
            "time_scipy_std_s": scipy_std,
            "time_aaa_mean_s": aaa_mean,
            "time_aaa_std_s": aaa_std,
            "speedup_scipy_over_aaa": (scipy_mean / aaa_mean) if aaa_mean > 0 else np.nan,
        }
        if numba_func is not None:
            numba_mean, numba_std = benchmark_callable(
                numba_func, z, warmup=cfg.warmup, repeats=cfg.repeats
            )
            row["time_numba_mean_s"] = numba_mean
            row["time_numba_std_s"] = numba_std
            row["speedup_scipy_over_numba"] = (
                scipy_mean / numba_mean if numba_mean > 0 else np.nan
            )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
    csv_path = cfg.output_dir / "phase1_speed_benchmark.csv"
    fig_path = cfg.output_dir / "phase1_speed_benchmark.pdf"
    summary_path = cfg.output_dir / "phase1_speed_benchmark_summary.txt"

    df.to_csv(csv_path, index=False)
    plot_results(fig_path, df, dpi=cfg.dpi)

    n_max = int(df["N"].max())
    row_max = df.loc[df["N"] == n_max].iloc[0]
    speedup = float(row_max["speedup_scipy_over_aaa"])
    lines = [
        "PHASE1_SPEED_BENCHMARK",
        "",
        f"Domain: z in [{cfg.z_min}, {cfg.z_max}]",
        f"Sizes: {list(map(int, df['N'].tolist()))}",
        f"Repeats: {cfg.repeats}",
        f"Warmup: {cfg.warmup}",
        f"Numba available: {HAS_NUMBA}",
        "",
        f"Speedup at N={n_max}: SciPy/AAA = {speedup:.4f}x",
    ]
    if "speedup_scipy_over_numba" in row_max and pd.notna(row_max["speedup_scipy_over_numba"]):
        lines.append(
            f"Speedup at N={n_max}: SciPy/FastAAA = {float(row_max['speedup_scipy_over_numba']):.4f}x"
        )
    lines.extend(
        [
            "",
            f"CSV: {csv_path}",
            f"Figure: {fig_path}",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Phase1 speed benchmark finished.")
    print(f"CSV: {csv_path}")
    print(f"Figure: {fig_path}")
    print(f"Summary: {summary_path}")
    print(f"Speedup SciPy/AAA @ N={n_max}: {speedup:.4f}x")


if __name__ == "__main__":
    main()
