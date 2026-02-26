#!/usr/bin/env python3
"""Paper orchestration for Radiation Measurements figures.

Generates final figures in ``output_finals_finals/``:

- Figure_1.pdf
- Figure_2a.pdf
- Figure_2b.pdf
- Figure_3.pdf
- Figure_4.pdf
- Figure_5.pdf
- Figure_6.pdf
- Figure_7.pdf
- Figure_8.pdf
- Figure_9.pdf

and copies tabular/text outputs from all phases into
``output_finals_finals/results/``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tldecpy.models.registry import get_model_info  # noqa: E402


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


def apply_rm_style(dpi: int) -> None:
    """Apply strict MathText + RM style."""
    style = dict(RM_STYLE)
    style["figure.dpi"] = dpi
    style["savefig.dpi"] = dpi
    mpl.rcParams.update(style)
    plt.rcParams.update(style)


def formal_label(model_key: str, fallback: str) -> str:
    """Resolve formal model label from registry."""
    try:
        label = str(get_model_info(model_key).label)
    except Exception:
        return fallback
    if model_key == "otor_lw":
        return label.replace("LambertW", "Lambert W")
    return label


@contextmanager
def temporary_argv(args: list[str]):
    """Temporarily replace ``sys.argv``."""
    old = list(sys.argv)
    try:
        sys.argv = args
        yield
    finally:
        sys.argv = old


def run_module_main(
    module_name: str,
    cli_args: list[str],
    patch: Callable[[Any], None] | None = None,
) -> Any:
    """Import/reload module and run its ``main()`` with patched behavior."""
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if patch is not None:
        patch(module)
    with temporary_argv([f"{module_name}.py", *cli_args]):
        try:
            module.main()
        except SystemExit as exc:
            if int(exc.code) != 0:
                raise RuntimeError(f"{module_name} exited with code {exc.code}") from exc
    return module


def copy_csv_txt(src_dir: Path, dst_dir: Path, prefix: str) -> None:
    """Copy csv/txt files with a phase prefix."""
    for path in sorted(src_dir.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".txt"}:
            continue
        shutil.copy2(path, dst_dir / f"{prefix}_{path.name}")


def patch_phase_style(module: Any, dpi: int) -> None:
    """Inject RM style into phase module."""
    apply_rm_style(dpi=dpi)
    if hasattr(module, "RM_PAPER_RCPARAMS"):
        style_dict = getattr(module, "RM_PAPER_RCPARAMS")
        if isinstance(style_dict, dict):
            style_dict.update(RM_STYLE)


def patch_phase2(module: Any, dpi: int) -> None:
    """Patch phase2 boxplot styling to make outliers explicit in Figure 2a."""
    patch_phase_style(module, dpi=dpi)

    def create_benchmark_boxplots(
        path: Path,
        records: list[dict[str, Any]],
        dpi: int,
        n_runs: int,
    ) -> None:
        module.configure_matplotlib(dpi=dpi)

        fom_pso = module._solver_boxplot_data(records, "FOM", "PSO")
        fom_de = module._solver_boxplot_data(records, "FOM", "DE")
        t_pso = module._solver_boxplot_data(records, "time_s", "PSO")
        t_de = module._solver_boxplot_data(records, "time_s", "DE")

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), constrained_layout=True)
        color_pso = "#4E79A7"
        color_de = "#F28E2B"
        tick_labels = [f"PSO (N={n_runs})", f"DE (N={n_runs})"]
        flierprops = dict(
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.7,
            alpha=1.0,
        )

        def _draw(ax: plt.Axes, a: np.ndarray, b: np.ndarray, ylabel: str) -> None:
            box = ax.boxplot(
                [a, b],
                tick_labels=tick_labels,
                notch=True,
                patch_artist=True,
                widths=0.58,
                showfliers=True,
                flierprops=flierprops,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
            )
            for patch, color in zip(box["boxes"], (color_pso, color_de)):
                patch.set_facecolor(color)
                patch.set_alpha(0.68)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.9)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.3, alpha=0.5)

        _draw(axes[0], fom_pso, fom_de, r"$FOM$ (\%)")
        _draw(axes[1], t_pso, t_de, r"Runtime (s)")
        axes[0].text(0.03, 0.95, "A", transform=axes[0].transAxes, fontweight="bold", va="top")
        axes[1].text(0.03, 0.95, "B", transform=axes[1].transAxes, fontweight="bold", va="top")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    module.create_benchmark_boxplots = create_benchmark_boxplots


def patch_phase3(module: Any) -> None:
    """Patch phase3 plot to add 15% bottom breathing space."""
    patch_phase_style(module, dpi=300)

    def make_loss_comparison_figure(
        path: Path,
        temperature: np.ndarray,
        y_clean: np.ndarray,
        y_dirty: np.ndarray,
        outlier_idx: np.ndarray,
        fit_results: dict[str, Any],
        *,
        snr_db: float,
        dpi: int,
    ) -> None:
        module.configure_matplotlib(dpi=dpi)

        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        ax.plot(
            temperature,
            y_dirty,
            linestyle="none",
            marker="o",
            markersize=2.8,
            color="0.35",
            alpha=0.55,
            label=rf"Corrupted data (SNR={snr_db:.0f} dB)",
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
        ax.plot(temperature, y_clean, color="black", linewidth=1.0, linestyle="-", label="Ground Truth")

        for method in module.METHOD_ORDER:
            style = module.METHOD_STYLES[method]
            y_hat = np.asarray(fit_results[method].y_hat_total, dtype=np.float64)
            ax.plot(
                temperature,
                y_hat,
                color=str(style["color"]),
                linestyle=str(style["linestyle"]),
                linewidth=1.5,
                label=str(style["label"]),
            )

        ax.set_xlabel(r"Temperature, $T$ (K)")
        ax.set_ylabel(r"Intensity, $I$ (a.u.)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
        ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)

        _, y1 = ax.get_ylim()
        ax.set_ylim(0.0, y1)

        tm_true = float(module.TRUE_PARAMS["Tm"])
        x1, x2 = tm_true - 10.0, tm_true + 10.0
        i_max = float(np.max(y_clean))
        yy1, yy2 = 0.90 * i_max, 1.05 * i_max

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
        for method in module.METHOD_ORDER:
            style = module.METHOD_STYLES[method]
            y_hat = np.asarray(fit_results[method].y_hat_total, dtype=np.float64)
            inset.plot(
                temperature,
                y_hat,
                color=str(style["color"]),
                linestyle=str(style["linestyle"]),
                linewidth=1.2,
            )
        inset.set_xlim(x1, x2)
        inset.set_ylim(yy1, yy2)
        inset.grid(True, which="both", linestyle="--", linewidth=0.25, alpha=0.5)
        inset.set_title(r"Zoom near $T_m$", fontsize=8)
        module.mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    module.make_loss_comparison_figure = make_loss_comparison_figure


def patch_phase4(module: Any, dpi: int) -> None:
    """Patch phase4 labels to strict MathText style."""
    patch_phase_style(module, dpi=dpi)

    def plot_curve_result(
        output_path: Path,
        curve_id: str,
        model_display: str,
        beta: float,
        temperature: np.ndarray,
        y_raw: np.ndarray,
        fit_pack: Any,
        dpi: int,
    ) -> None:
        module.configure_matplotlib(dpi=dpi)
        y_hat = np.asarray(fit_pack.result.y_hat_total, dtype=np.float64)
        residual = y_raw - y_hat
        curve_label = module.format_refglow_label(curve_id)

        fig = plt.figure(figsize=(6.8, 4.8), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.08)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

        ax_main.plot(
            temperature,
            y_raw,
            linestyle="none",
            marker="o",
            markersize=2.0,
            color="0.55",
            alpha=0.65,
            label="Data",
        )
        ax_main.plot(temperature, y_hat, color="black", linewidth=2.2, label="Total fit")

        cmap = plt.get_cmap("tab10")
        for idx, peak in enumerate(fit_pack.result.peaks, start=1):
            peak_y = np.asarray(peak.y_hat, dtype=np.float64)
            peak_id = module.parse_peak_id(peak.name, fallback=idx)
            ax_main.plot(
                temperature,
                peak_y,
                linestyle="--",
                linewidth=1.3,
                color=cmap((idx - 1) % 10),
                label=rf"Peak {peak_id}",
            )

        uc_global = module._safe_float(fit_pack.result.metrics.uc_global)
        uc_p95 = module._safe_float(fit_pack.result.metrics.uc_p95)
        uc_global_txt = "n/a" if uc_global is None else f"{uc_global:.3f}\\%"
        uc_p95_txt = "n/a" if uc_p95 is None else f"{uc_p95:.3f}\\%"
        box = [
            curve_label,
            formal_label("fo_ka" if "fo_ka" in model_display.lower() else "fo_wp", model_display),
            rf"$\beta={beta:.2f}\ \mathrm{{K\,s^{{-1}}}}$",
            rf"$FOM={fit_pack.fom_proc:.4f}\%$",
            rf"$u_{{c,\mathrm{{global}}}}={uc_global_txt}$",
            rf"$u_{{c,\mathrm{{p95}}}}={uc_p95_txt}$",
        ]
        ax_main.text(
            0.02,
            0.98,
            "\n".join(box),
            transform=ax_main.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.2},
        )

        ax_main.set_ylabel(r"Intensity, $I$ (a.u.)")
        ax_main.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
        ax_main.legend(loc="best", fontsize=7.2, framealpha=0.9)

        ax_res.plot(
            temperature,
            residual,
            linestyle="none",
            marker="o",
            markersize=1.8,
            color="#8c4b12",
            alpha=0.75,
        )
        ax_res.axhline(0.0, color="black", linewidth=0.9)
        q99 = float(np.nanpercentile(np.abs(residual), 99))
        if not np.isfinite(q99) or q99 <= 0.0:
            q99 = 1.0
        ax_res.set_ylim(-1.1 * q99, 1.1 * q99)
        ax_res.set_ylabel(r"Residual, $I_{\mathrm{data}}-I_{\mathrm{fit}}$")
        ax_res.set_xlabel(r"Temperature, $T$ (K)")
        ax_res.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    module.plot_curve_result = plot_curve_result


def rerun_phase4_x005_with_constant_bg_subtraction(
    module: Any,
    out_dir: Path,
    dpi: int,
    strategy: str,
    max_nfev: int,
) -> Path:
    """Run x005 fo_wp using I-1.0 baseline correction before fitting."""
    t, i_raw = module.load_refglow("x005")
    t_arr = np.asarray(t, dtype=np.float64)
    i_arr = np.asarray(i_raw, dtype=np.float64)
    i_fit = np.clip(i_arr - 1.0, 0.0, None)

    fit_pack = module.run_curve_fit(
        curve_id="x005",
        model_key="fo_wp",
        temperature=t_arr,
        intensity_raw=i_fit,
        beta=float(module.GLOCANIN_TRUTH["x005"]["beta"]),
        bg_mode=None,
        strategy=strategy,
        max_nfev=max_nfev,
    )

    # Overwrite x005 csv with bg-subtracted run for paper consistency.
    truth_by_id = {int(p["id"]): p for p in module.GLOCANIN_TRUTH["x005"]["peaks"]}
    rows: list[dict[str, Any]] = []
    for idx, peak in enumerate(fit_pack.result.peaks, start=1):
        peak_id = module.parse_peak_id(peak.name, fallback=idx)
        params = dict(peak.params)
        tm_fit = module._safe_float(params.get("Tm"))
        uc_tm = (
            None
            if tm_fit is None
            else module.uc_at_temperature(t_arr, fit_pack.uc_curve, tm_fit)
        )
        truth = truth_by_id.get(peak_id)
        rows.append(
            {
                "curve": "x005",
                "Model": "fo_wp",
                "Model_Display": formal_label("fo_wp", "First-order kinetics (FO)"),
                "Peak_ID": peak_id,
                "Peak_Name": peak.name,
                "E": module._safe_float(params.get("E")),
                "Tm": tm_fit,
                "Im": module._safe_float(params.get("Im")),
                "s_fit": module._safe_float(params.get("s")),
                "u_c_at_Tm": uc_tm,
                "FOM": fit_pack.fom_proc,
                "Time": fit_pack.runtime_s,
                "Converged": bool(fit_pack.result.converged),
                "truth_Tm": None if truth is None else float(truth["T_m"]),
                "truth_Im": None if truth is None else float(truth["I_m"]),
                "truth_E": None if truth is None else float(truth["E"]),
            }
        )
    pd.DataFrame(rows).sort_values("Peak_ID").to_csv(out_dir / "phase4_results_x005.csv", index=False)

    fig_path = out_dir / "phase4_x005_fo_wp_fit_residual.pdf"
    module.plot_curve_result(
        output_path=fig_path,
        curve_id="x005",
        model_display=formal_label("fo_wp", "First-order kinetics (FO)"),
        beta=float(module.GLOCANIN_TRUTH["x005"]["beta"]),
        temperature=t_arr,
        y_raw=i_fit,
        fit_pack=fit_pack,
        dpi=dpi,
    )
    return fig_path


def patch_phase5(module: Any, dpi: int) -> None:
    """Patch phase5 figure: minimal legend + u_c(Tm) labels under peaks."""
    patch_phase_style(module, dpi=dpi)

    def plot_fit_with_residual(
        output_path: Path,
        temperature: np.ndarray,
        y_obs: np.ndarray,
        best: Any,
        beta: float,
        dpi: int,
    ) -> None:
        module.configure_matplotlib(dpi=dpi)

        y_hat = np.asarray(best.result.y_hat_total, dtype=np.float64)
        residual = y_obs - y_hat
        uc_global = module._safe_float(best.result.metrics.uc_global)
        uc_p95 = module._safe_float(best.result.metrics.uc_p95)
        y_span = float(np.nanmax(y_obs) - np.nanmin(y_obs))
        offset = max(0.04 * y_span, 0.01 * float(np.nanmax(y_obs)))

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
            markersize=2.0,
            color="0.55",
            alpha=0.7,
            label="Data",
        )
        ax_fit.plot(temperature, y_hat, color="black", linewidth=2.1, label="Total Fit")
        cmap = plt.get_cmap("tab10")
        # Hidden style-handle for requested model legend entry.
        otor_handle = Line2D([0], [0], color="black", linewidth=1.1, linestyle="--")

        for idx, peak in enumerate(best.result.peaks, start=1):
            y_peak = np.asarray(peak.y_hat, dtype=np.float64)
            ax_fit.plot(
                temperature,
                y_peak,
                linestyle="--",
                linewidth=0.95,
                color=cmap((idx - 1) % 10),
                alpha=0.65,
            )
            tm_fit = module._safe_float(peak.params.get("Tm"))
            if tm_fit is None:
                continue
            i_tm = float(np.interp(tm_fit, temperature, y_peak))
            uc_tm = module.uc_at_temperature(temperature, best.result.uc_curve, tm_fit)
            uc_text = "n/a" if uc_tm is None else f"{uc_tm:.1f}%"
            ax_fit.text(
                tm_fit,
                i_tm - offset,
                uc_text,
                fontsize=6.4,
                ha="center",
                va="top",
                color="0.15",
            )

        uc_global_txt = "n/a" if uc_global is None else f"{uc_global:.3f}\\%"
        uc_p95_txt = "n/a" if uc_p95 is None else f"{uc_p95:.3f}\\%"
        info = [
            module.CURVE_DISPLAY,
            "OTOR (Lambert W)",
            rf"$\beta={beta:.2f}\ \mathrm{{K\,s^{{-1}}}}$",
            rf"$FOM={best.fom:.4f}\%$",
            rf"$u_{{c,\mathrm{{global}}}}={uc_global_txt}$",
            rf"$u_{{c,\mathrm{{p95}}}}={uc_p95_txt}$",
        ]
        ax_fit.text(
            0.015,
            0.98,
            "\n".join(info),
            transform=ax_fit.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.2},
        )

        ax_fit.set_ylabel(r"Intensity, $I$ (a.u.)")
        ax_fit.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
        handles, labels = ax_fit.get_legend_handles_labels()
        # Keep only requested legend entries + OTOR display name.
        keep_h, keep_l = [], []
        for handle, label_text in zip(handles, labels):
            if label_text in {"Data", "Total Fit"}:
                keep_h.append(handle)
                keep_l.append(label_text)
        handles, labels = keep_h, keep_l
        handles.append(otor_handle)
        labels.append("OTOR (Lambert W)")
        ax_fit.legend(handles, labels, loc="best", fontsize=7.3, framealpha=0.9)

        ax_res.plot(
            temperature,
            residual,
            linestyle="none",
            marker="o",
            markersize=1.8,
            color="#8c4b12",
            alpha=0.75,
        )
        ax_res.axhline(0.0, color="black", linewidth=1.0)
        q99 = float(np.nanpercentile(np.abs(residual), 99))
        if not np.isfinite(q99) or q99 <= 0.0:
            q99 = 1.0
        ax_res.set_ylim(-1.1 * q99, 1.1 * q99)
        ax_res.set_ylabel(r"Residual, $I_{\mathrm{data}}-I_{\mathrm{fit}}$")
        ax_res.set_xlabel(r"Temperature, $T$ (K)")
        ax_res.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    module.plot_fit_with_residual = plot_fit_with_residual


def patch_phase6(module: Any, dpi: int) -> None:
    """Patch phase6 figure text cleanup for paper panel."""
    patch_phase_style(module, dpi=dpi)

    def plot_fit(
        output_pdf: Path,
        output_png: Path,
        data: Any,
        pre: Any,
        best: Any,
        component_defs: list[dict[str, Any]],
        dpi: int,
    ) -> None:
        module.configure_matplotlib(dpi=dpi)

        x_plot = np.asarray(data.temperature_input, dtype=np.float64)
        y_raw = np.asarray(data.intensity_raw, dtype=np.float64)
        y_fit = np.asarray(pre.y_fit, dtype=np.float64)
        y_hat = np.asarray(best.result.y_hat_total, dtype=np.float64)
        residual = y_fit - y_hat
        uc_global = module._safe_float(best.result.metrics.uc_global)

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
            label="Data",
        )
        ax_top.plot(x_plot, y_hat, color="black", linewidth=2.1, label="Total Fit")

        colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
        defs_by_id = {str(comp["id"]): comp for comp in component_defs}
        for idx, peak in enumerate(best.result.peaks):
            comp = defs_by_id.get(str(peak.name), {})
            label = str(comp.get("label", "Component"))
            label = label.replace("Continuous trap distribution", "Continuous")
            label = label.replace("Localized first-order peak (closed-form)", "Localized FO")
            ax_top.plot(
                x_plot,
                np.asarray(peak.y_hat, dtype=np.float64),
                linestyle="--",
                linewidth=1.3,
                color=colors[idx % len(colors)],
                label=label,
            )

        uc_txt = "n/a" if uc_global is None else f"{uc_global:.3f}\\%"
        stats = [
            r"GdAlO$_3$ ($\beta=10$ K/s, 13.2 Gy)",
            rf"$FOM={best.fom_fit:.4f}\%$",
            rf"$u_c={uc_txt}$",
        ]
        ax_top.text(
            0.015,
            0.98,
            "\n".join(stats),
            transform=ax_top.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75", "pad": 2.0},
        )
        ax_top.set_ylabel(r"Intensity, $I$ (a.u.)")
        ax_top.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)
        ax_top.legend(loc="best", fontsize=7.1, framealpha=0.9)

        ax_bot.plot(
            x_plot,
            residual,
            linestyle="none",
            marker="o",
            markersize=1.8,
            color="#8c4b12",
            alpha=0.75,
        )
        ax_bot.axhline(0.0, color="black", linewidth=0.9)
        q99 = float(np.nanpercentile(np.abs(residual), 99))
        if not np.isfinite(q99) or q99 <= 0.0:
            q99 = 1.0
        ax_bot.set_ylim(-1.1 * q99, 1.1 * q99)
        ax_bot.set_ylabel(r"Residual, $I_{\mathrm{data}}-I_{\mathrm{fit}}$")
        ax_bot.set_xlabel(r"Temperature, $T$ (K)")
        ax_bot.grid(True, which="major", linestyle="--", linewidth=0.3, alpha=0.35)

        fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    module.plot_fit = plot_fit


def build_parser() -> argparse.ArgumentParser:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Generate final paper figures for Radiation Measurements."
    )
    parser.add_argument("--output-dir", type=str, default="output_finals_finals")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--phase2-n-runs", type=int, default=100)
    parser.add_argument("--phase7-n-mc", type=int, default=100000)
    parser.add_argument("--phase7-roi-min-c", type=float, default=165.0)
    parser.add_argument("--phase7-roi-max-c", type=float, default=255.0)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only remap figure names from existing run outputs.",
    )
    return parser


def main() -> None:
    """Run sequential paper pipeline."""
    args = build_parser().parse_args()
    apply_rm_style(dpi=int(args.dpi))

    output_dir = Path(str(args.output_dir))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    p1_dir = runs_dir / "phase1"
    p2_dir = runs_dir / "phase2"
    p1b_dir = runs_dir / "phase1b"
    p3_dir = runs_dir / "phase3"
    p4_dir = runs_dir / "phase4"
    p5_dir = runs_dir / "phase5"
    p7_dir = runs_dir / "phase7"
    p6_dir = runs_dir / "phase6"
    for d in (p1_dir, p1b_dir, p2_dir, p3_dir, p4_dir, p5_dir, p7_dir, p6_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        run_module_main(
            "phase1_validate_aux_alpha",
            [
                "--output-dir",
                str(p1_dir),
                "--figure-format",
                "pdf",
                "--basename",
                "aux_alpha_comparison",
                "--dpi",
                str(args.dpi),
            ],
            patch=lambda m: patch_phase_style(m, int(args.dpi)),
        )

        run_module_main(
            "phase1_time_benchmark",
            [
                "--output-dir",
                str(p1b_dir),
                "--dpi",
                str(args.dpi),
            ],
            patch=lambda m: patch_phase_style(m, int(args.dpi)),
        )

        run_module_main(
            "phase2_benchmark_solvers",
            [
                "--output-dir",
                str(p2_dir),
                "--n-runs",
                str(int(args.phase2_n_runs)),
                "--dpi",
                str(args.dpi),
            ],
            patch=lambda m: patch_phase2(m, int(args.dpi)),
        )

        run_module_main(
            "phase3_robustness_test",
            [
                "--output-dir",
                str(p3_dir),
                "--dpi",
                str(args.dpi),
            ],
            patch=patch_phase3,
        )

        phase4_mod = run_module_main(
            "phase4_refglow_benchmark",
            [
                "--output-dir",
                str(p4_dir),
                "--strategy",
                "global_hybrid_pso",
                "--dpi",
                str(args.dpi),
            ],
            patch=lambda m: patch_phase4(m, int(args.dpi)),
        )
        # Figure 6 correction: x005 fit on (I - 1.0) baseline-adjusted signal.
        rerun_phase4_x005_with_constant_bg_subtraction(
            module=phase4_mod,
            out_dir=p4_dir,
            dpi=int(args.dpi),
            strategy="global_hybrid_pso",
            max_nfev=4000,
        )

        run_module_main(
            "phase5_refglow_x009_otor_lw",
            [
                "--output-dir",
                str(p5_dir),
                "--strategy",
                "global_hybrid_pso",
                "--nstart",
                "60",
                "--kkf",
                "0.03",
                "--dpi",
                str(args.dpi),
            ],
            patch=lambda m: patch_phase5(m, int(args.dpi)),
        )

        run_module_main(
            "phase7_uncertainty_validation",
            [
                "--output-dir",
                str(p7_dir),
                "--data-path",
                str(PROJECT_ROOT / "scripts" / "TLD100Exp.csv"),
                "--roi-c-min",
                str(float(args.phase7_roi_min_c)),
                "--roi-c-max",
                str(float(args.phase7_roi_max_c)),
                "--n-mc",
                str(int(args.phase7_n_mc)),
                "--progress-every",
                "500",
                "--dpi",
                str(args.dpi),
            ],
            patch=lambda m: patch_phase_style(m, int(args.dpi)),
        )

        run_module_main(
            "phase6_gdalo_continuous_validation",
            [
                "--output-dir",
                str(p6_dir),
                "--data-path",
                str(PROJECT_ROOT / "scripts" / "gdalo.csv"),
                "--bg-mode",
                "exponential",
                "--dpi",
                str(args.dpi),
            ],
            patch=lambda m: patch_phase6(m, int(args.dpi)),
        )

    figure_map: dict[str, Path] = {
        "Figure_1.pdf": p1_dir / "aux_alpha_comparison.pdf",
        "Figure_1b.pdf": p1b_dir / "phase1_speed_benchmark.pdf",
        "Figure_2a.pdf": p2_dir / "phase2_benchmark_boxplots.pdf",
        "Figure_2b.pdf": p2_dir / "phase2_best_fit_with_residuals.pdf",
        "Figure_3.pdf": p3_dir / "phase3_loss_comparison.pdf",
        "Figure_4.pdf": p4_dir / "phase4_x001_fo_ka_fit_residual.pdf",
        "Figure_5.pdf": p4_dir / "phase4_x002_fo_ka_fit_residual.pdf",
        "Figure_6.pdf": p4_dir / "phase4_x005_fo_wp_fit_residual.pdf",
        "Figure_7.pdf": p5_dir / "bench5_x009_otor_lw_fit_residual.pdf",
        "Figure_8.pdf": p7_dir / "phase7_uncertainty.pdf",
        "Figure_9.pdf": p6_dir / "bench6_gdalo_fit_residual.pdf",
    }

    for name, src in figure_map.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing figure source: {src}")
        shutil.copy2(src, output_dir / name)

    copy_csv_txt(p1_dir, results_dir, "phase1")
    copy_csv_txt(p1b_dir, results_dir, "phase1b")
    copy_csv_txt(p2_dir, results_dir, "phase2")
    copy_csv_txt(p3_dir, results_dir, "phase3")
    copy_csv_txt(p4_dir, results_dir, "phase4")
    copy_csv_txt(p5_dir, results_dir, "phase5")
    copy_csv_txt(p7_dir, results_dir, "phase7")
    copy_csv_txt(p6_dir, results_dir, "phase6")

    summary = output_dir / "summary_make_paper_figures.txt"
    lines = [
        "MAKE_PAPER_FIGURES",
        "",
        f"Output root: {output_dir}",
        "Generated figures:",
    ]
    for key in (
        "Figure_1.pdf",
        "Figure_1b.pdf",
        "Figure_2a.pdf",
        "Figure_2b.pdf",
        "Figure_3.pdf",
        "Figure_4.pdf",
        "Figure_5.pdf",
        "Figure_6.pdf",
        "Figure_7.pdf",
        "Figure_8.pdf",
        "Figure_9.pdf",
    ):
        lines.append(f"- {output_dir / key}")
    lines.extend(
        [
            "",
            f"Phase7 ROI (°C): {float(args.phase7_roi_min_c):.1f} to {float(args.phase7_roi_max_c):.1f}",
            f"Phase7 Monte Carlo n_mc: {int(args.phase7_n_mc)}",
            f"Phase2 n_runs: {int(args.phase2_n_runs)}",
            "",
            f"Results directory: {results_dir}",
        ]
    )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("make_paper_figures completed.")
    for name in (
        "Figure_1.pdf",
        "Figure_1b.pdf",
        "Figure_2a.pdf",
        "Figure_2b.pdf",
        "Figure_3.pdf",
        "Figure_4.pdf",
        "Figure_5.pdf",
        "Figure_6.pdf",
        "Figure_7.pdf",
        "Figure_8.pdf",
        "Figure_9.pdf",
    ):
        print(f"- {output_dir / name}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
