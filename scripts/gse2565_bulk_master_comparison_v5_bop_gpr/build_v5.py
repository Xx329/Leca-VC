#!/usr/bin/env python3
"""Evaluate frozen trajectories and render GSE2565 master comparison V5.

Only BOP-DMD is newly fitted, by ``run_bopdmd.py``. All other trajectories
are loaded from frozen historical outputs. The benchmark is descriptive
because the methods do not have equivalent access to the observed trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common import (
    BASE,
    CHRON_PATH,
    COMMON_DNB,
    EVAL,
    FORMAL_DNB_PATH,
    FROZEN_HELDOUT_PATH,
    GENE_PATH,
    GPR_DNB_PATH,
    GPR_PATH,
    MODULE_PATH,
    OUT,
    PLOT_DATA,
    REAL_PATH,
    REVISION,
    RVA_PATH,
    STATE_INDICES,
    TIMES,
    VELOCITY_INDICES,
    array_sha256,
    ensure_dirs,
    sha256,
)


BOP_PATH = OUT / "bopdmd_reconstructed_trajectory.npz"
BOP_DNB_PATH = OUT / "bopdmd_dnb_bootstrap_curve.csv"
BOP_AUDIT_PATH = OUT / "BOPDMD_FORMAL_AUDIT.json"

METHODS = [
    "chronODE-M",
    "RVAgene",
    "BOP-DMD",
    "GPR",
    "AgentVC Online Agent pilot",
]
DISPLAY_ORDER = ["Real", *METHODS]
COLORS = {
    "Real": "#222222",
    "chronODE-M": "#7851A9",
    "RVAgene": "#E68613",
    "BOP-DMD": "#009E73",
    "GPR": "#D55E00",
    "AgentVC Online Agent pilot": "#2878B5",
}
SHORT_LABELS = {
    "Real": "Real",
    "chronODE-M": "chronODE-M",
    "RVAgene": "RVAgene",
    "BOP-DMD": "BOP-DMD",
    "GPR": "GPR",
    "AgentVC Online Agent pilot": "AgentVC pilot",
}
INPUT_REGIMES = {
    "chronODE-M": "complete observed 0–72 h trajectory",
    "RVAgene": "complete observed 0–72 h trajectory",
    "BOP-DMD": "complete observed 0–72 h trajectory",
    "GPR": "observed 0–4 h only",
    "AgentVC Online Agent pilot": "no observed GSE2565 expression at runtime",
}
DNB_SEMANTICS = {
    "chronODE-M": "B=20 biological-replicate bootstrap refits",
    "RVAgene": "saved independent reconstruction seeds",
    "BOP-DMD": "B=20 biological-replicate bootstrap refits",
    "GPR": "B=20 early-input biological-replicate bootstrap refits",
    "AgentVC Online Agent pilot": "independent simulation seeds",
}


def load_npz_expression(path: Path, genes: list[str]) -> np.ndarray:
    archive = np.load(path, allow_pickle=False)
    if archive["genes"].astype(str).tolist() != genes:
        raise RuntimeError(f"Gene order mismatch: {path}")
    if not np.allclose(archive["times"].astype(float), TIMES, rtol=0, atol=0):
        raise RuntimeError(f"Time order mismatch: {path}")
    expression = archive["expression"].astype(float)
    if expression.shape != (9, 12263) or not np.isfinite(expression).all():
        raise RuntimeError(f"Invalid trajectory matrix: {path}")
    return expression


def load_inputs() -> dict:
    ensure_dirs()
    if not BOP_AUDIT_PATH.exists():
        raise RuntimeError("BOP-DMD formal audit is missing")
    bop_audit = json.loads(BOP_AUDIT_PATH.read_text(encoding="utf-8"))
    if bop_audit.get("status") != "PASS_BOPDMD_FIXED_CONFIG":
        raise RuntimeError("BOP-DMD did not pass its fixed-config audit")

    frozen = BASE.load_frozen_inputs()
    genes = frozen["genes"]
    matrices_full = {
        "Real": frozen["real"],
        "chronODE-M": frozen["chron"],
        "RVAgene": load_npz_expression(RVA_PATH, genes),
        "BOP-DMD": load_npz_expression(BOP_PATH, genes),
        "GPR": load_npz_expression(GPR_PATH, genes),
        "AgentVC Online Agent pilot": frozen["agent"],
    }
    if any(x.shape != (9, 12263) for x in matrices_full.values()):
        raise RuntimeError("A unified trajectory is not 9 x 12,263")

    valid = frozen["complete_valid"]
    if int(valid.sum()) != 7632:
        raise RuntimeError("Frozen 7,632-gene evaluation mask changed")
    genes_eval = [g for g, keep in zip(genes, valid) if keep]
    matrices = {method: matrix[:, valid] for method, matrix in matrices_full.items()}
    microarray = {method: method != "AgentVC Online Agent pilot" for method in matrices}
    curves = {
        method: REVISION.curves_for_matrix(matrix, microarray[method])
        for method, matrix in matrices.items()
    }
    progression = {
        method: BASE.minmax(item["progression"])[0] for method, item in curves.items()
    }
    modules = frozen["modules"]
    module_z = {}
    module_raw = {}
    for method, matrix in matrices.items():
        module_raw[method] = EVAL.module_matrix(matrix, genes_eval, modules)
        module_z[method], _ = EVAL.zscore_time(module_raw[method])

    return {
        "frozen": frozen,
        "genes": genes,
        "genes_eval": genes_eval,
        "matrices_full": matrices_full,
        "matrices": matrices,
        "curves": curves,
        "progression": progression,
        "module_raw": module_raw,
        "module_z": module_z,
        "modules": modules,
        "bop_audit": bop_audit,
    }


def load_dnb_curves() -> dict[str, np.ndarray]:
    frozen = pd.read_csv(COMMON_DNB / "panelA_unified_dnb_trajectories.csv")
    source_map = {
        "Real": "Real",
        "chronODE-M": "chronODE-M-BS†",
        "RVAgene": "RVAgene",
        "AgentVC Online Agent pilot": "Online Agent pilot",
    }
    curves = {}
    for method, source in source_map.items():
        part = frozen.loc[frozen.method.eq(source)].sort_values("time_hours")
        if len(part) != 9 or set(part.common_DNB_gene_count) != {1601}:
            raise RuntimeError(f"Frozen common-space DNB invalid: {source}")
        curves[method] = part.normalized_DNB.to_numpy(float)
    for method, path in [("GPR", GPR_DNB_PATH), ("BOP-DMD", BOP_DNB_PATH)]:
        part = pd.read_csv(path).sort_values("time_hours")
        gene_count_col = (
            "common_DNB_genes"
            if "common_DNB_genes" in part.columns
            else "common_DNB_gene_count"
        )
        if (
            len(part) != 9
            or set(part[gene_count_col].astype(int)) != {1601}
            or not np.isfinite(part.normalized_DNB).all()
        ):
            raise RuntimeError(f"Common-space DNB invalid: {path}")
        curves[method] = part.normalized_DNB.to_numpy(float)
    if set(curves) != set(DISPLAY_ORDER):
        raise RuntimeError("DNB method set incomplete")
    return {method: curves[method] for method in DISPLAY_ORDER}


def late_metrics(data: dict, dnb_curves: dict[str, np.ndarray]) -> pd.DataFrame:
    real_curves = data["curves"]["Real"]
    real_matrix = data["matrices"]["Real"]
    real_module_z = data["module_z"]["Real"]
    rows = []
    for method in METHODS:
        method_curves = data["curves"][method]
        dnb = EVAL.trajectory_metrics(
            dnb_curves["Real"], dnb_curves[method], "DNB", TIMES
        )
        module_correlations = [
            EVAL.safe_pearson(
                real_module_z[STATE_INDICES, j],
                data["module_z"][method][STATE_INDICES, j],
            )
            for j in range(real_module_z.shape[1])
        ]
        method_matrix = data["matrices"][method]
        rows.append(
            {
                "method": method,
                "input_regime": INPUT_REGIMES[method],
                "dnb_scope": "1,601-gene common-space sensitivity",
                "dnb_peak_time": dnb["DNB_peak_time"],
                "dnb_peak_error": dnb["DNB_peak_time_error"],
                "dnb_pearson": dnb["DNB_Pearson"],
                "dnb_dtw": dnb["DNB_DTW"],
                "distribution_pearson": EVAL.safe_pearson(
                    real_curves["distribution"][STATE_INDICES],
                    method_curves["distribution"][STATE_INDICES],
                ),
                "progression_pearson": EVAL.safe_pearson(
                    real_curves["progression"][STATE_INDICES],
                    method_curves["progression"][STATE_INDICES],
                ),
                "velocity_pearson": EVAL.safe_pearson(
                    real_curves["velocity"][VELOCITY_INDICES],
                    method_curves["velocity"][VELOCITY_INDICES],
                ),
                "mean_module_pearson": float(np.nanmean(module_correlations)),
                "gene_direction_agreement": float(
                    (
                        np.sign(
                            real_matrix[STATE_INDICES] - real_matrix[[0]]
                        )
                        == np.sign(
                            method_matrix[STATE_INDICES] - method_matrix[[0]]
                        )
                    ).mean()
                ),
                "estimable": True,
                "notes": (
                    f"DNB repeat semantics: {DNB_SEMANTICS[method]}; "
                    "late state metrics use 8|12|24|48|72 h; velocity uses "
                    "4->8|8->12|12->24|24->48|48->72 h"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    required = [
        "method",
        "input_regime",
        "dnb_scope",
        "dnb_peak_time",
        "dnb_peak_error",
        "dnb_pearson",
        "dnb_dtw",
        "distribution_pearson",
        "progression_pearson",
        "velocity_pearson",
        "mean_module_pearson",
        "gene_direction_agreement",
        "estimable",
        "notes",
    ]
    return frame[required]


def validate_frozen_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    frozen = pd.read_csv(FROZEN_HELDOUT_PATH).set_index("method")
    current = metrics.set_index("method")
    method_map = {
        "GPR": "GPR",
        "chronODE-M†": "chronODE-M",
        "Online Agent pilot": "AgentVC Online Agent pilot",
    }
    column_map = {
        "distribution_shift_Pearson": "distribution_pearson",
        "progression_Pearson": "progression_pearson",
        "velocity_Pearson": "velocity_pearson",
        "mean_module_Pearson": "mean_module_pearson",
        "gene_direction_agreement": "gene_direction_agreement",
    }
    max_diffs = {}
    for old_method, new_method in method_map.items():
        differences = []
        for old_col, new_col in column_map.items():
            differences.append(
                abs(float(frozen.loc[old_method, old_col]) - float(current.loc[new_method, new_col]))
            )
        max_diffs[new_method] = max(differences)
        if max_diffs[new_method] > 1e-12:
            raise RuntimeError(
                f"Frozen held-out metric gate failed: {new_method}, "
                f"max diff {max_diffs[new_method]:.3e}"
            )
    return max_diffs


def validate_formal_callout() -> dict:
    formal = pd.read_csv(FORMAL_DNB_PATH)
    real = formal.loc[formal.method.eq("Real")].iloc[0]
    agent = formal.loc[formal.method.eq("Online Agent pilot")].iloc[0]
    if not (
        int(real.gene_count) == 11171
        and int(agent.gene_count) == 11171
        and real.real_peak_time == 8
        and agent.method_peak_time == 8
        and agent.DNB_peak_time_error == 0
    ):
        raise RuntimeError("Formal full-space DNB callout changed")
    return {
        "gene_count": 11171,
        "real_peak_hours": 8,
        "agent_peak_hours": 8,
        "agent_peak_error_hours": 0,
    }


def write_source_data(
    data: dict, dnb_curves: dict[str, np.ndarray], metrics: pd.DataFrame
) -> dict[str, Path]:
    trajectory_rows = []
    for method in DISPLAY_ORDER:
        for index, hour in enumerate(TIMES):
            trajectory_rows.append(
                {
                    "method": method,
                    "time_hours": hour,
                    "normalized_DNB": dnb_curves[method][index],
                    "normalized_distribution_shift": data["curves"][method][
                        "distribution"
                    ][index],
                    "normalized_progression": data["progression"][method][index],
                    "normalized_velocity": (
                        np.nan
                        if index == 0
                        else data["curves"][method]["velocity"][index - 1]
                    ),
                    "evaluation_gene_count": 7632,
                    "common_DNB_gene_count": 1601,
                }
            )
    trajectories_path = PLOT_DATA / "panels_A_to_D_trajectories.csv"
    pd.DataFrame(trajectory_rows).to_csv(trajectories_path, index=False)

    heat_rows = []
    for method in DISPLAY_ORDER:
        for time_index, hour in enumerate(TIMES):
            for module_index, module in enumerate(data["modules"].module):
                heat_rows.append(
                    {
                        "method": method,
                        "time_hours": hour,
                        "module": module,
                        "temporal_z_score": data["module_z"][method][
                            time_index, module_index
                        ],
                        "evaluation_gene_count": 7632,
                    }
                )
    heat_path = PLOT_DATA / "panel_E_functional_program_heatmaps.csv"
    pd.DataFrame(heat_rows).to_csv(heat_path, index=False)

    panel_f_rows = []
    specs = [
        ("F1", "dnb_peak_error", "DNB peak error ↓", True),
        ("F1", "dnb_pearson", "DNB Pearson ↑", False),
        ("F1", "dnb_dtw", "DNB DTW ↓", True),
        ("F2", "distribution_pearson", "Distribution Pearson ↑", False),
        ("F2", "progression_pearson", "Progression Pearson ↑", False),
        ("F2", "velocity_pearson", "Velocity Pearson ↑", False),
        ("F3", "mean_module_pearson", "Mean module Pearson ↑", False),
        ("F3", "gene_direction_agreement", "Gene-direction agreement ↑", False),
    ]
    for panel, key, label, lower in specs:
        for _, row in metrics.iterrows():
            panel_f_rows.append(
                {
                    "panel": panel,
                    "metric_key": key,
                    "metric_label": label,
                    "lower_is_better": lower,
                    "method": row.method,
                    "raw_value": row[key],
                }
            )
    panel_f_path = PLOT_DATA / "panel_F_raw_values.csv"
    pd.DataFrame(panel_f_rows).to_csv(panel_f_path, index=False)
    return {
        "trajectories": trajectories_path,
        "heatmaps": heat_path,
        "panel_f": panel_f_path,
    }


def plot_line_panel(ax, letter: str, title: str, curves: dict, x: np.ndarray):
    for method in DISPLAY_ORDER:
        ax.plot(
            x,
            curves[method],
            color=COLORS[method],
            lw=1.65 if method != "Real" else 1.9,
            marker="o",
            ms=2.8,
            alpha=0.94,
            zorder=3,
        )
    ax.axvspan(0, 4, color="#D8DCE0", alpha=0.20, zorder=0)
    ax.axvline(4, color="#7A7F84", lw=0.7, ls="--", zorder=1)
    ax.set_title(f"{letter}. {title}", loc="left", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Time (h)", fontsize=7.3)
    ax.tick_params(labelsize=6.7, length=2.5)
    ax.grid(color="#D8DADD", lw=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def draw_metric(ax, subset: pd.DataFrame):
    subset = subset.set_index("method").loc[METHODS].reset_index()
    values = subset.raw_value.to_numpy(float)
    key = subset.metric_key.iloc[0]
    if np.nanmin(values) < 0:
        low = min(-0.25, np.nanmin(values) * 1.22)
        high = max(1.03, np.nanmax(values) * 1.12)
    elif key in {"dnb_peak_error", "dnb_dtw"}:
        low, high = 0, max(1.0, np.nanmax(values) * 1.25)
    else:
        low, high = 0, max(1.03, np.nanmax(values) * 1.12)
    y = np.arange(len(subset))[::-1]
    for index, row in subset.iterrows():
        value = float(row.raw_value)
        ax.barh(
            y[index],
            value,
            height=0.52,
            color=COLORS[row.method],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        span = high - low
        ax.text(
            value + (0.014 * span if value >= 0 else -0.014 * span),
            y[index],
            f"{value:.1f} h" if key == "dnb_peak_error" else f"{value:.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=5.7,
            color=COLORS[row.method],
            clip_on=False,
        )
    ax.set_xlim(low, high)
    ax.set_ylim(-0.7, len(subset) - 0.3)
    ax.set_yticks(y, [SHORT_LABELS[x] for x in subset.method], fontsize=5.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=5.5, length=2)
    ax.set_title(subset.metric_label.iloc[0], loc="left", fontsize=6.8, fontweight="bold")
    ax.axvline(0, color="#AAB0B5", lw=0.6)
    ax.grid(axis="x", color="#E2E4E6", lw=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B7BCC1")


def plot_figure(data: dict, dnb_curves: dict, panel_f: pd.DataFrame) -> dict[str, Path]:
    fig = plt.figure(figsize=(24, 17.2), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.82, 1.32, 1.35],
        hspace=0.44,
        wspace=0.31,
        left=0.055,
        right=0.978,
        top=0.90,
        bottom=0.105,
    )

    axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
    panels = [
        ("A", "Common-space DNB sensitivity", dnb_curves, TIMES),
        (
            "B",
            "Normalized expression-distribution shift",
            {m: data["curves"][m]["distribution"] for m in DISPLAY_ORDER},
            TIMES,
        ),
        ("C", "Normalized transcriptomic progression", data["progression"], TIMES),
        (
            "D",
            "Time-interval-normalized transcriptomic velocity",
            {m: data["curves"][m]["velocity"] for m in DISPLAY_ORDER},
            TIMES[1:],
        ),
    ]
    for ax, (letter, title, curves, x) in zip(axes, panels):
        plot_line_panel(ax, letter, title, curves, x)
    axes[0].set_ylabel("Within-method normalized value", fontsize=7.3)
    axes[0].set_ylim(-0.06, 1.08)
    axes[2].set_ylim(-0.05, 1.06)
    axes[0].text(
        0.48,
        0.96,
        "Formal 11,171-gene DNB\nReal peak 8 h · AgentVC peak 8 h\nAgentVC peak error 0 h",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#C1C5C8",
            "alpha": 0.90,
            "linewidth": 0.6,
        },
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            marker="o",
            lw=1.8,
            ms=3.3,
            label=SHORT_LABELS[m],
        )
        for m in DISPLAY_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=6,
        frameon=False,
        fontsize=7.8,
        columnspacing=1.3,
        handlelength=2.2,
    )

    e_outer = fig.add_subplot(grid[1, :])
    e_outer.axis("off")
    e_outer.text(
        0,
        1.045,
        "E. Functional-program dynamics",
        transform=e_outer.transAxes,
        fontsize=10.2,
        fontweight="bold",
    )
    egrid = grid[1, :].subgridspec(2, 3, wspace=0.18, hspace=0.34)
    vmax = max(float(np.nanmax(np.abs(data["module_z"][m]))) for m in DISPLAY_ORDER)
    images = []
    for index, method in enumerate(DISPLAY_ORDER):
        ax = fig.add_subplot(egrid[index // 3, index % 3])
        image = ax.imshow(
            data["module_z"][method].T,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        images.append((ax, image))
        ax.set_title(SHORT_LABELS[method], fontsize=7.8, pad=3)
        ax.set_xticks(range(9), [f"{x:g}" for x in TIMES], fontsize=5.7)
        ax.set_xlabel("Time (h)", fontsize=6.2)
        if index % 3 == 0:
            ax.set_yticks(
                range(len(data["modules"])),
                data["modules"].module,
                fontsize=5.6,
            )
        else:
            ax.set_yticks([])
    cbar = fig.colorbar(
        images[-1][1],
        ax=[x[0] for x in images],
        location="right",
        fraction=0.013,
        pad=0.012,
    )
    cbar.ax.set_title("z", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.7)

    f_outer = fig.add_subplot(grid[2, :])
    f_outer.axis("off")
    f_outer.text(
        0,
        1.055,
        "F. Raw cross-method reconstruction metrics",
        transform=f_outer.transAxes,
        fontsize=10.2,
        fontweight="bold",
    )
    f_outer.text(
        0,
        1.005,
        "No composite score; F2/F3 exclude GPR training states.",
        transform=f_outer.transAxes,
        fontsize=6.6,
        color="#666A6E",
    )
    group_specs = [
        ("F1. Common-space DNB sensitivity", "F1", ["dnb_peak_error", "dnb_pearson", "dnb_dtw"]),
        (
            "F2. Late global trajectory recovery",
            "F2",
            ["distribution_pearson", "progression_pearson", "velocity_pearson"],
        ),
        (
            "F3. Late program-level recovery",
            "F3",
            ["mean_module_pearson", "gene_direction_agreement"],
        ),
    ]
    fgrid = grid[2, :].subgridspec(1, 3, wspace=0.31)
    for group_index, (title, panel, keys) in enumerate(group_specs):
        subgrid = fgrid[0, group_index].subgridspec(
            len(keys) + 1,
            1,
            height_ratios=[0.17] + [1] * len(keys),
            hspace=0.72,
        )
        header = fig.add_subplot(subgrid[0, 0])
        header.axis("off")
        header.text(
            0,
            0.2,
            title,
            transform=header.transAxes,
            fontsize=7.7,
            fontweight="bold",
            color="#44494D",
        )
        for key_index, key in enumerate(keys, start=1):
            ax = fig.add_subplot(subgrid[key_index, 0])
            draw_metric(
                ax,
                panel_f.loc[
                    panel_f.panel.eq(panel) & panel_f.metric_key.eq(key)
                ],
            )

    fig.suptitle(
        "GSE2565 bulk trajectory reconstruction benchmark",
        fontsize=16.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.962,
        "Descriptive full-trajectory comparison with frozen late-time evaluation",
        ha="center",
        fontsize=8.8,
        color="#3C4043",
    )
    footnote = (
        "Input regimes are not equivalent: chronODE-M is an official monotonic full-trajectory sensitivity; "
        "RVAgene and BOP-DMD are target-derived full-trajectory reconstructions; GPR uses observed 0–4 h only; "
        "AgentVC is a runtime-blind six-run Online Agent pilot. Panel A/F1 use a 1,601-gene DNB sensitivity space "
        "with non-identical repeat semantics. Panels B–E/F2–F3 use the frozen 7,632-gene common evaluation space; "
        "F2/F3 score 8–72 h states and 4→8 through 48→72 h velocities. No overall winner is inferred."
    )
    fig.text(0.5, 0.018, footnote, ha="center", va="bottom", fontsize=6.4, wrap=True)

    paths = {
        "preview_png": OUT / "GSE2565_bulk_master_comparison_v5_preview.png",
        "png_600dpi": OUT / "GSE2565_bulk_master_comparison_v5_600dpi.png",
        "pdf": OUT / "GSE2565_bulk_master_comparison_v5_vector.pdf",
        "svg": OUT / "GSE2565_bulk_master_comparison_v5_vector.svg",
    }
    fig.savefig(paths["preview_png"], dpi=170, facecolor="white")
    fig.savefig(paths["png_600dpi"], dpi=600, facecolor="white")
    fig.savefig(paths["pdf"], facecolor="white")
    fig.savefig(paths["svg"], facecolor="white")
    plt.close(fig)
    return paths


def source_manifest(data: dict, generated: list[Path]) -> pd.DataFrame:
    sources = [
        GENE_PATH,
        REAL_PATH,
        MODULE_PATH,
        CHRON_PATH,
        RVA_PATH,
        GPR_PATH,
        GPR_DNB_PATH,
        BOP_PATH,
        BOP_DNB_PATH,
        BOP_AUDIT_PATH,
        COMMON_DNB / "panelA_unified_dnb_trajectories.csv",
        FROZEN_HELDOUT_PATH,
        FORMAL_DNB_PATH,
        *data["frozen"]["agent_paths"],
        *generated,
    ]
    rows = []
    for path in sources:
        path = Path(path)
        rows.append(
            {
                "role": "generated_output" if path in generated else "frozen_input",
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_inputs()
    dnb_curves = load_dnb_curves()
    metrics = late_metrics(data, dnb_curves)
    metric_path = OUT / "raw_metric_summary.csv"
    metrics.to_csv(metric_path, index=False)
    frozen_maxdiff = validate_frozen_metrics(metrics)
    formal_callout = validate_formal_callout()
    source_paths = write_source_data(data, dnb_curves, metrics)
    panel_f = pd.read_csv(source_paths["panel_f"])
    figure_paths = plot_figure(data, dnb_curves, panel_f)

    matrix_inventory = []
    for method, matrix in data["matrices_full"].items():
        matrix_inventory.append(
            {
                "method": method,
                "shape": "9x12263",
                "finite": bool(np.isfinite(matrix).all()),
                "array_sha256": array_sha256(matrix),
                "input_regime": "reference" if method == "Real" else INPUT_REGIMES[method],
            }
        )
    matrix_path = OUT / "unified_trajectory_inventory.csv"
    pd.DataFrame(matrix_inventory).to_csv(matrix_path, index=False)

    script_copy = OUT / "build_v5.py"
    shutil.copy2(Path(__file__), script_copy)
    generated = [
        metric_path,
        matrix_path,
        *source_paths.values(),
        *figure_paths.values(),
        script_copy,
    ]
    manifest_path = OUT / "sha256_manifest.csv"
    source_manifest(data, generated).to_csv(manifest_path, index=False)

    common_peaks = {
        method: float(
            TIMES[np.isfinite(curve)][np.argmax(np.asarray(curve)[np.isfinite(curve)])]
        )
        for method, curve in dnb_curves.items()
    }
    audit = {
        "status": "PASS_DESCRIPTIVE_BULK_RECONSTRUCTION_V5_BOP_GPR",
        "benchmark_semantics": "descriptive bulk trajectory reconstruction comparison",
        "equal_input_prediction_claim_allowed": False,
        "overall_winner_claim_allowed": False,
        "historical_models_rerun": False,
        "historical_outputs_overwritten": False,
        "methods": METHODS,
        "trajectory_shape_each": [9, 12263],
        "evaluation_gene_count_panels_B_to_E_F2_F3": 7632,
        "common_DNB_gene_count_panels_A_F1": 1601,
        "late_state_hours": TIMES[STATE_INDICES].tolist(),
        "late_velocity_intervals": [
            "4->8",
            "8->12",
            "12->24",
            "24->48",
            "48->72",
        ],
        "GPR_training_states_in_F2_F3": False,
        "frozen_metric_max_absolute_differences": frozen_maxdiff,
        "formal_fullspace_DNB_callout": formal_callout,
        "common_space_DNB_peaks_hours": common_peaks,
        "BOP_DMD": data["bop_audit"],
        "figure_outputs": {key: str(value) for key, value in figure_paths.items()},
        "plot_source_tables": {key: str(value) for key, value in source_paths.items()},
        "manifest": str(manifest_path),
        "visual_inspection": "PENDING_MANUAL_INSPECTION",
    }
    audit_path = OUT / "FINAL_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    readme = f"""# GSE2565 Bulk Master Comparison V5

Status: **PASS_DESCRIPTIVE_BULK_RECONSTRUCTION_V5_BOP_GPR**

This is a descriptive bulk trajectory reconstruction comparison, not an
equal-input forecasting benchmark. The input regimes are explicitly unequal:

- chronODE-M: official monotonic full-trajectory sensitivity.
- RVAgene: target-derived full-trajectory reconstruction.
- BOP-DMD: target-derived full-trajectory reconstruction, fixed rank 4.
- GPR: early-input forecast using observed 0–4 h only.
- AgentVC: runtime-blind six-run Online Agent pilot.

Panels A/F1 use the frozen 1,601-gene common DNB sensitivity space. Panels
B–E/F2–F3 use the frozen 7,632-gene evaluation space. F2/F3 exclude GPR
training states: state metrics use 8, 12, 24, 48 and 72 h; velocity uses
4→8 through 48→72 h.

The independent formal 11,171-gene DNB audit remains unchanged: Real peaks at
8 h, AgentVC peaks at 8 h, and AgentVC peak error is 0 h.

No composite score is generated and this version does not support an overall
winner claim.

## Outputs

- Main figure: `{figure_paths["png_600dpi"]}`
- Raw metrics: `{metric_path}`
- Plot source data: `{PLOT_DATA}`
- BOP-DMD audit: `{BOP_AUDIT_PATH}`
- Final audit: `{audit_path}`
- SHA256 manifest: `{manifest_path}`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
