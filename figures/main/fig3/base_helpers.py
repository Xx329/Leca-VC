#!/usr/bin/env python3
"""Redraw the frozen CCI and expression panels on a uniform paper grid.

The script only reads frozen plotting/source-data CSV files. It does not run a
model, COMMOT, bootstrap, or any scientific analysis. Dense point artists are
rasterized inside the otherwise vector PDF/SVG outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agentvc-paper-uniform-v3")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
CCI_ROOT = (
    ROOT
    / "outputs/GSE267904_multimethod_cci_benchmark_v5_agentvc_lrra_"
    "diagnostic_figure_v6_h_iqr_lines/source_data"
)
EXP_ROOT = ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy"
OUT = ROOT / "outputs/AgentVC_multibenchmark_paper_figure_v3_uniform_panels"
DISPLAY_BRAND = "AgentVC"
FIGURE_TITLE = "AgentVC across cell–cell communication and gene-expression benchmarks"
FILE_PREFIX = "AgentVC_multibenchmark_paper_uniform_v3"
VERSION_ID = "AgentVC_multibenchmark_paper_figure_v3_uniform_panels"
PENDING_STATUS = "PASS_NUMERIC_AND_LAYOUT_PENDING_VISUAL_INSPECTION"
README_TITLE = "AgentVC uniform-panel paper figure V3"

INPUTS = {
    "cci_primary": CCI_ROOT / "panel_G_primary_summary.csv",
    "cci_secondary": CCI_ROOT / "panel_H_secondary_summary.csv",
    "cci_local": CCI_ROOT / "panel_I_local_summary.csv",
    "expression_gene_values": EXP_ROOT / "unified_gene_level_plotting_source.csv",
    "expression_metrics": EXP_ROOT / "unified_expression_metrics.csv",
    "expression_bootstrap_values": EXP_ROOT / "bootstrap_metric_values.csv",
    "expression_bootstrap_summary": EXP_ROOT / "bootstrap_confidence_intervals.csv",
    "expression_deciles": EXP_ROOT / "expression_decile_calibration.csv",
}
EXPECTED_HASHES = {
    "cci_primary": "149d6a7342015168a547d99ed854e4a076be831e166c007de90d4fe3a4f5f146",
    "cci_secondary": "d3f22b53d3408ee5092e8cdcc8c38cc4c0700c10c53ff2a000e333b13ed51c12",
    "cci_local": "08d9370bee312c6f48c8c329a7ce2ccab60c463831e5316db0d64ca201891d27",
    "expression_gene_values": "e75b74e254da39098749693b511bc63948a62f2c8de87d4c91393d468dfa2d45",
    "expression_metrics": "5a0ed1ce9d273bd60255a312c50b8bdb13492e1d36d00e8cfe24a4ca43687aa9",
    "expression_bootstrap_values": "ae6443888a89f439c714e81f01d29997664fb764fd95496e3e9de0add0475ea4",
    "expression_bootstrap_summary": "1808e3686a3b254d391812694f3e8da721ccffd377b97e99d397e486884917a7",
    "expression_deciles": "6eee01e3fdcb074b775c6be82f1067f3062fad70f9515bded7fd109d020e7354",
}

CCI_METHODS = ("scgpt", "scgen", "wot_external", "cellrank_proxy", "worker_lrra")
EXP_METHODS = ("scGen", "WOT_weighted", "CellRank_terminal_proxy", "AgentVC_proxy")
RESPONSES = ("Responder", "Non-responder")
LABELS = {
    "scgpt": "scGPT",
    "scgen": "scGen",
    "wot_external": "WOT",
    "cellrank_proxy": "CellRank",
    "worker_lrra": "AgentVC",
    "Observed_Post": "Observed",
    "scGen": "scGen",
    "WOT_weighted": "WOT",
    "CellRank_terminal_proxy": "CellRank",
    "AgentVC_proxy": "AgentVC",
}
COLORS = {
    "Observed_Post": "#2B2B2B",
    "scgpt": "#0072B2",
    "scgen": "#E69F00",
    "wot_external": "#009E73",
    "cellrank_proxy": "#56B4E9",
    "worker_lrra": "#CC3F8D",
    "scGen": "#E69F00",
    "WOT_weighted": "#009E73",
    "CellRank_terminal_proxy": "#56B4E9",
    "AgentVC_proxy": "#CC3F8D",
}
MARKERS = {
    "Observed_Post": "o",
    "scGen": "o",
    "WOT_weighted": "s",
    "CellRank_terminal_proxy": "^",
    "AgentVC_proxy": "D",
}

PRIMARY = (
    "edge_spearman_81",
    "informative_edge_spearman",
    "positive_top10_jaccard",
    "five_pathway_flow_js_similarity",
    "sender_receiver_role_mean_spearman",
    "pathway_allocation_mean_absolute_error",
    "cell_type_local_mean_absolute_error",
)
PRIMARY_LABELS = {
    "edge_spearman_81": "Edge\nSpearman ↑",
    "informative_edge_spearman": "Informative\nSpearman ↑",
    "positive_top10_jaccard": "Top-10\nJaccard ↑",
    "five_pathway_flow_js_similarity": "5-flow JS\nsimilarity ↑",
    "sender_receiver_role_mean_spearman": "Role mean\nSpearman ↑",
    "pathway_allocation_mean_absolute_error": "Pathway\nMAE ↓",
    "cell_type_local_mean_absolute_error": "Cell-type\nMAE ↓",
}
CELL_TYPES = (
    "alveolar_epithelial_AT1_AT2",
    "activated_Krt8_ADI_epithelial",
    "airway_epithelial",
    "macrophage",
    "recruited_monocyte_macrophage",
    "fibroblast_myofibroblast",
    "endothelial",
    "dendritic",
    "lymphoid",
)
CELL_LABELS = {
    "alveolar_epithelial_AT1_AT2": "AT1/AT2",
    "activated_Krt8_ADI_epithelial": "Krt8/ADI",
    "airway_epithelial": "Airway",
    "macrophage": "Macrophage",
    "recruited_monocyte_macrophage": "Mono/Mac",
    "fibroblast_myofibroblast": "Fib/Myofib",
    "endothelial": "Endothelial",
    "dendritic": "Dendritic",
    "lymphoid": "Lymphoid",
}

FIG_W = 21.0
FIG_H = 11.4
LEFT_IN = 0.70
RIGHT_IN = 20.30
LOWER_BOTTOM_IN = 0.55
LOWER_ROW_GAP_IN = 0.46
LOWER_NORMAL_GAP_IN = 0.12
LOWER_GROUP_GAP_IN = 0.38
LOWER_PANEL_IN = (
    (RIGHT_IN - LEFT_IN)
    - 5 * LOWER_NORMAL_GAP_IN
    - LOWER_GROUP_GAP_IN
) / 7
TOP_AX_BOTTOM_IN = 7.40
TOP_AX_HEIGHT_IN = 2.45
BOOTSTRAP_SEED = 120575834


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axes_in_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def figure_text_inches(fig: plt.Figure, x: float, y: float, text: str, **kwargs: Any) -> Any:
    return fig.text(x / FIG_W, y / FIG_H, text, **kwargs)


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.10, y: float = 1.10) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top")


def lower_x_positions() -> list[float]:
    positions = [LEFT_IN]
    for column in range(1, 7):
        gap = LOWER_GROUP_GAP_IN if column == 4 else LOWER_NORMAL_GAP_IN
        positions.append(positions[-1] + LOWER_PANEL_IN + gap)
    return positions


def validate_and_load() -> dict[str, pd.DataFrame]:
    for key, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != EXPECTED_HASHES[key]:
            raise RuntimeError(f"Frozen input hash mismatch for {key}: {actual}")
    tables = {key: pd.read_csv(path) for key, path in INPUTS.items()}
    expected_rows = {
        "cci_primary": 5 * 7,
        "cci_secondary": 5 * 2,
        "cci_local": 5 * 9,
        "expression_gene_values": 2 * 4 * 834,
        "expression_metrics": 2 * 4 * 7,
        "expression_bootstrap_values": 2 * 4 * 7 * 1000,
        "expression_bootstrap_summary": 2 * 4 * 7,
        "expression_deciles": 2 * 5 * 10,
    }
    for key, rows in expected_rows.items():
        if len(tables[key]) != rows:
            raise RuntimeError(f"Unexpected row count for {key}: {len(tables[key])} != {rows}")
    return tables


def draw_primary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    raw = (
        summary.pivot(index="method", columns="metric", values="median")
        .reindex(index=CCI_METHODS, columns=PRIMARY)
        .to_numpy(float)
    )
    display = np.empty_like(raw)
    lower_is_better = {
        "pathway_allocation_mean_absolute_error",
        "cell_type_local_mean_absolute_error",
    }
    for column, metric in enumerate(PRIMARY):
        values = raw[:, column]
        scaled = (
            np.full_like(values, 0.5)
            if np.isclose(values.min(), values.max())
            else (values - values.min()) / (values.max() - values.min())
        )
        display[:, column] = 1 - scaled if metric in lower_is_better else scaled
    ax.imshow(display, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for row in range(len(CCI_METHODS)):
        for column in range(len(PRIMARY)):
            ax.text(
                column,
                row,
                f"{raw[row, column]:.4f}" if column >= 5 else f"{raw[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if display[row, column] > 0.68 else "#202020",
                fontweight="bold" if display[row, column] > 0.82 else "normal",
            )
    ax.set_xticks(range(7), [PRIMARY_LABELS[x] for x in PRIMARY], fontsize=6.2)
    ax.set_yticks(range(5), [LABELS[x] for x in CCI_METHODS], fontsize=6.8)
    ax.tick_params(length=0)
    ax.set_title(
        "Primary reconstruction metrics\nRaw medians; directional shading within columns",
        fontsize=9.2,
        fontweight="bold",
        pad=7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_letter(ax, "A")


def draw_secondary(container: plt.Axes, summary: pd.DataFrame) -> list[plt.Axes]:
    container.set_axis_off()
    container.set_title(
        "GNRS and LBSS stability\nSecondary descriptive scores",
        fontsize=9.2,
        fontweight="bold",
        pad=7,
    )
    panel_letter(container, "B", x=-0.08)
    children: list[plt.Axes] = []
    for index, metric in enumerate(("GNRS_secondary", "LBSS_secondary")):
        ax = container.inset_axes([0.12, 0.56 - index * 0.48, 0.84, 0.34])
        for row, method in enumerate(CCI_METHODS):
            stat = summary[summary["method"].eq(method) & summary["metric"].eq(metric)].iloc[0]
            ax.errorbar(
                stat["median"],
                row,
                xerr=np.array([[stat["median"] - stat["q1"]], [stat["q3"] - stat["median"]]]),
                fmt="|",
                ms=9,
                markeredgewidth=1.8,
                color=COLORS[method],
                capsize=0,
                lw=2.2,
            )
        ax.set_xlim(0.25, 1.01)
        ax.set_ylim(len(CCI_METHODS) - 0.5, -0.5)
        ax.set_yticks(range(len(CCI_METHODS)), [LABELS[x] for x in CCI_METHODS], fontsize=6.4)
        ax.set_xlabel("GNRS ↑" if metric == "GNRS_secondary" else "LBSS ↑", fontsize=7, labelpad=1)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.grid(axis="x", color="#E7E7E7", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        children.append(ax)
    return children


def draw_local(fig: plt.Figure, ax: plt.Axes, summary: pd.DataFrame) -> plt.Axes:
    raw = (
        summary.pivot(index="method", columns="cell_type", values="median")
        .reindex(index=CCI_METHODS, columns=CELL_TYPES)
        .to_numpy(float)
    )
    image = ax.imshow(raw, cmap="magma_r", aspect="auto", interpolation="nearest")
    threshold = float(np.quantile(raw, 0.55))
    for row in range(len(CCI_METHODS)):
        for column in range(len(CELL_TYPES)):
            ax.text(
                column,
                row,
                f"{raw[row, column]:.4f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if raw[row, column] >= threshold else "#202020",
            )
    ax.set_xticks(
        range(9), [CELL_LABELS[x] for x in CELL_TYPES], rotation=48, ha="right", fontsize=5.9
    )
    ax.set_yticks(range(5), [LABELS[x] for x in CCI_METHODS], fontsize=6.7)
    ax.tick_params(length=0)
    ax.set_title(
        "Cell-type local error\nMedian MAE across 10 mappings; lower is better",
        fontsize=9.2,
        fontweight="bold",
        pad=7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_letter(ax, "C")
    cax = axes_in_inches(fig, 19.91, TOP_AX_BOTTOM_IN, 0.12, TOP_AX_HEIGHT_IN)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Local MAE", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.8)
    return cax


def expression_ranges(gene_values: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate(
        [gene_values["observed_expression"].to_numpy(float), gene_values["predicted_expression"].to_numpy(float)]
    )
    span = float(values.max() - values.min())
    return float(values.min() - 0.015 * span), float(values.max() + 0.035 * span)


def metric_map(metrics: pd.DataFrame, response: str, method: str) -> dict[str, float]:
    return (
        metrics[metrics["response"].eq(response) & metrics["method"].eq(method)]
        .set_index("metric")["value"]
        .astype(float)
        .to_dict()
    )


def draw_scatter(
    ax: plt.Axes,
    gene_values: pd.DataFrame,
    metrics: pd.DataFrame,
    response: str,
    method: str,
    letter: str,
    lower: float,
    upper: float,
    column: int,
) -> None:
    selected = gene_values[
        gene_values["response"].eq(response) & gene_values["method"].eq(method)
    ].sort_values("gene")
    observed = selected["observed_expression"].to_numpy(float)
    predicted = selected["predicted_expression"].to_numpy(float)
    slope, intercept = np.polyfit(observed, predicted, 1)
    values = metric_map(metrics, response, method)
    color = COLORS[method]
    ax.scatter(
        observed,
        predicted,
        s=6.5,
        alpha=0.28,
        color=color,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    ax.plot([lower, upper], [lower, upper], "--", color="#92999F", lw=0.85, zorder=1)
    line = np.array([lower, upper])
    ax.plot(line, slope * line + intercept, color=color, lw=1.5, zorder=3)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{letter}. {response} — {LABELS[method]}", loc="left", fontsize=8.0, fontweight="bold", pad=4)
    ax.text(
        0.035,
        0.96,
        f"Pearson r = {values['Pearson']:.3f}\nCCC = {values['CCC']:.3f}\nRMSE = {values['RMSE']:.3f}\nslope = {slope:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=5.6,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#D5D8DA", "alpha": 0.90},
    )
    ax.grid(True, color="#E7EAEC", lw=0.55, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.0)
    ax.set_xlabel("Observed Post expression", fontsize=6.7, labelpad=2)
    if column == 0:
        ax.set_ylabel("Predicted / proxy expression", fontsize=6.7, labelpad=2)


def draw_violin(
    ax: plt.Axes,
    values_by_method: dict[str, np.ndarray],
    annotations: dict[str, str],
    y_limits: tuple[float, float],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    positions = np.arange(len(EXP_METHODS))
    for position, method in zip(positions, EXP_METHODS):
        values = np.asarray(values_by_method[method], dtype=float)
        violin = ax.violinplot([values], positions=[position], widths=0.74, showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor(COLORS[method])
            body.set_alpha(0.23)
        ax.boxplot(
            [values],
            positions=[position],
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": COLORS[method], "linewidth": 0.8},
            medianprops={"color": COLORS[method], "linewidth": 1.1},
            whiskerprops={"color": COLORS[method], "linewidth": 0.7},
            capprops={"color": COLORS[method], "linewidth": 0.7},
        )
        show_n = min(180, len(values))
        chosen = rng.choice(len(values), size=show_n, replace=False)
        ax.scatter(
            rng.normal(position, 0.055, size=show_n),
            values[chosen],
            s=3.2,
            alpha=0.13,
            color=COLORS[method],
            edgecolors="none",
            rasterized=True,
        )
        ax.text(
            position,
            y_limits[1] - 0.035 * (y_limits[1] - y_limits[0]),
            annotations[method],
            ha="center",
            va="top",
            fontsize=4.6,
            color="#444444",
        )
    ax.set_xlim(-0.55, len(EXP_METHODS) - 0.45)
    ax.set_ylim(*y_limits)
    ax.set_xticks(positions, [LABELS[x] for x in EXP_METHODS])
    ax.tick_params(axis="x", labelsize=5.6)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.grid(True, axis="y", color="#E3E7E9", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def draw_bootstrap_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    bootstrap_values: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    response: str,
    letter: str,
    row: int,
) -> None:
    values_by_method: dict[str, np.ndarray] = {}
    annotations: dict[str, str] = {}
    for method in EXP_METHODS:
        values = bootstrap_values[
            bootstrap_values["response"].eq(response)
            & bootstrap_values["method"].eq(method)
            & bootstrap_values["metric"].eq("Pearson")
        ]["value"].to_numpy(float)
        values_by_method[method] = values
        point = metric_map(metrics, response, method)["Pearson"]
        ci = bootstrap_summary[
            bootstrap_summary["response"].eq(response)
            & bootstrap_summary["method"].eq(method)
            & bootstrap_summary["metric"].eq("Pearson")
        ].iloc[0]
        annotations[method] = f"r={point:.3f}\nCI {ci['ci95_low']:.3f}–{ci['ci95_high']:.3f}"
    all_values = np.concatenate(list(values_by_method.values()))
    limits = (min(0.0, float(all_values.min()) - 0.05), min(1.08, float(all_values.max()) + 0.10))
    draw_violin(ax, values_by_method, annotations, limits, BOOTSTRAP_SEED + row)
    ax.set_title(f"{letter}. {response}: bootstrap Pearson", loc="left", fontsize=7.3, fontweight="bold", pad=4)
    ax.set_ylabel("Bootstrap Pearson r", fontsize=6.2)


def draw_signed_panel(
    ax: plt.Axes,
    gene_values: pd.DataFrame,
    response: str,
    letter: str,
    row: int,
) -> None:
    values_by_method: dict[str, np.ndarray] = {}
    annotations: dict[str, str] = {}
    for method in EXP_METHODS:
        selected = gene_values[gene_values["response"].eq(response) & gene_values["method"].eq(method)]
        values = selected["signed_error"].to_numpy(float)
        values_by_method[method] = values
        annotations[method] = f"median {np.median(values):.3f}\nMAE {selected['absolute_error'].mean():.3f}"
    max_abs = max(1.0, float(np.quantile(np.abs(np.concatenate(list(values_by_method.values()))), 0.995) * 1.22))
    draw_violin(ax, values_by_method, annotations, (-max_abs, max_abs), BOOTSTRAP_SEED + 10 + row)
    ax.axhline(0, color="#656D72", ls="--", lw=0.85)
    ax.set_title(f"{letter}. {response}: signed error", loc="left", fontsize=7.3, fontweight="bold", pad=4)
    ax.set_ylabel("Signed error (predicted − Observed)", fontsize=6.2)


def draw_calibration_panel(
    ax: plt.Axes,
    deciles: pd.DataFrame,
    response: str,
    letter: str,
) -> None:
    for method in ("Observed_Post", *EXP_METHODS):
        selected = deciles[deciles["response"].eq(response) & deciles["method"].eq(method)].sort_values(
            "observed_expression_decile"
        )
        ax.plot(
            selected["observed_expression_decile"],
            selected["mean_post_expression"],
            color=COLORS[method],
            marker=MARKERS[method],
            ms=3.0,
            lw=1.35,
        )
    ax.set_xticks(range(1, 11), [f"D{i}" for i in range(1, 11)])
    ax.set_xlabel("Observed-expression decile", fontsize=6.2)
    ax.set_ylabel("Mean Post / proxy expression", fontsize=6.2)
    ax.set_title(f"{letter}. {response}: decile calibration", loc="left", fontsize=7.3, fontweight="bold", pad=4)
    ax.grid(True, color="#E3E7E9", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=5.7)


def output_record(path: Path) -> dict[str, Any]:
    return {"relative_path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def main() -> None:
    tables = validate_and_load()
    OUT.mkdir(parents=True, exist_ok=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    figure_text_inches(
        fig,
        FIG_W / 2,
        11.03,
        FIGURE_TITLE,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    figure_text_inches(
        fig,
        FIG_W / 2,
        10.53,
        "GSE267904 · Cell–cell communication reconstruction",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )

    primary_ax = axes_in_inches(fig, 0.70, TOP_AX_BOTTOM_IN, 6.15, TOP_AX_HEIGHT_IN)
    secondary_ax = axes_in_inches(fig, 7.43, TOP_AX_BOTTOM_IN, 5.45, TOP_AX_HEIGHT_IN)
    local_ax = axes_in_inches(fig, 13.45, TOP_AX_BOTTOM_IN, 6.25, TOP_AX_HEIGHT_IN)
    draw_primary(primary_ax, tables["cci_primary"])
    secondary_children = draw_secondary(secondary_ax, tables["cci_secondary"])
    colorbar_ax = draw_local(fig, local_ax, tables["cci_local"])

    fig.add_artist(
        Line2D([LEFT_IN / FIG_W, RIGHT_IN / FIG_W], [7.02 / FIG_H, 7.02 / FIG_H], transform=fig.transFigure, color="#D9DDE0", lw=0.8)
    )

    x_positions = lower_x_positions()
    scatter_group_center = (x_positions[0] + x_positions[3] + LOWER_PANEL_IN) / 2
    summary_group_center = (x_positions[4] + x_positions[6] + LOWER_PANEL_IN) / 2
    figure_text_inches(
        fig,
        scatter_group_center,
        6.78,
        "GSE120575 · Gene-wise post-treatment expression agreement",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
    )
    figure_text_inches(
        fig,
        summary_group_center,
        6.78,
        "GSE120575 · Expression accuracy and calibration",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
    )

    row_bottoms = [LOWER_BOTTOM_IN + LOWER_PANEL_IN + LOWER_ROW_GAP_IN, LOWER_BOTTOM_IN]
    lower_axes: list[tuple[str, str, plt.Axes]] = []
    lower, upper = expression_ranges(tables["expression_gene_values"])
    scatter_letters = iter("DEFGHIJK")
    for row, response in enumerate(RESPONSES):
        for column, method in enumerate(EXP_METHODS):
            ax = axes_in_inches(fig, x_positions[column], row_bottoms[row], LOWER_PANEL_IN, LOWER_PANEL_IN)
            letter = next(scatter_letters)
            draw_scatter(
                ax,
                tables["expression_gene_values"],
                tables["expression_metrics"],
                response,
                method,
                letter,
                lower,
                upper,
                column,
            )
            lower_axes.append((letter, "scatter", ax))

    summary_letters = (("L", "M", "N"), ("O", "P", "Q"))
    for row, response in enumerate(RESPONSES):
        bootstrap_ax = axes_in_inches(fig, x_positions[4], row_bottoms[row], LOWER_PANEL_IN, LOWER_PANEL_IN)
        signed_ax = axes_in_inches(fig, x_positions[5], row_bottoms[row], LOWER_PANEL_IN, LOWER_PANEL_IN)
        calibration_ax = axes_in_inches(fig, x_positions[6], row_bottoms[row], LOWER_PANEL_IN, LOWER_PANEL_IN)
        draw_bootstrap_panel(
            bootstrap_ax,
            tables["expression_metrics"],
            tables["expression_bootstrap_values"],
            tables["expression_bootstrap_summary"],
            response,
            summary_letters[row][0],
            row,
        )
        draw_signed_panel(
            signed_ax,
            tables["expression_gene_values"],
            response,
            summary_letters[row][1],
            row,
        )
        draw_calibration_panel(
            calibration_ax,
            tables["expression_deciles"],
            response,
            summary_letters[row][2],
        )
        lower_axes.extend(
            [
                (summary_letters[row][0], "bootstrap", bootstrap_ax),
                (summary_letters[row][1], "signed_error", signed_ax),
                (summary_letters[row][2], "calibration", calibration_ax),
            ]
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            lw=1.4,
            ms=4,
            label=LABELS[method],
        )
        for method in ("Observed_Post", *EXP_METHODS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(summary_group_center / FIG_W, 6.53 / FIG_H),
        ncol=5,
        frameon=False,
        fontsize=5.8,
        handlelength=1.6,
        columnspacing=1.0,
    )

    positions: list[dict[str, Any]] = []
    for letter, panel_type, ax in lower_axes:
        bounds = ax.get_position().bounds
        positions.append(
            {
                "panel": letter,
                "panel_type": panel_type,
                "left_inches": bounds[0] * FIG_W,
                "bottom_inches": bounds[1] * FIG_H,
                "width_inches": bounds[2] * FIG_W,
                "height_inches": bounds[3] * FIG_H,
            }
        )
    layout = pd.DataFrame(positions).sort_values("panel")
    width_spread = float(layout["width_inches"].max() - layout["width_inches"].min())
    height_spread = float(layout["height_inches"].max() - layout["height_inches"].min())
    if width_spread > 1e-12 or height_spread > 1e-12:
        raise RuntimeError(f"Lower panel dimensions are not equal: width={width_spread}, height={height_spread}")

    preview = OUT / f"{FILE_PREFIX}_preview.png"
    png = OUT / f"{FILE_PREFIX}_600dpi.png"
    pdf = OUT / f"{FILE_PREFIX}_vector.pdf"
    svg = OUT / f"{FILE_PREFIX}.svg"
    fig.savefig(preview, dpi=150, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    layout.to_csv(OUT / "panel_layout_audit.csv", index=False)
    source_inventory = pd.DataFrame(
        [
            {
                "source_key": key,
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": EXPECTED_HASHES[key],
                "rows": len(tables[key]),
            }
            for key, path in INPUTS.items()
        ]
    )
    source_inventory.to_csv(OUT / "source_inventory.csv", index=False)

    outputs = {path.name: output_record(path) for path in (preview, png, pdf, svg)}
    manifest = {
        "status": PENDING_STATUS,
        "version": VERSION_ID,
        "rendering": "panel-level redraw from frozen plotting CSVs",
        "models_or_scientific_metrics_rerun": False,
        "input_hashes": EXPECTED_HASHES,
        "input_rows": {key: len(table) for key, table in tables.items()},
        "figure_inches": [FIG_W, FIG_H],
        "lower_grid": "2 rows x 7 columns",
        "lower_panel_count": len(layout),
        "lower_panel_width_inches": float(layout.iloc[0]["width_inches"]),
        "lower_panel_height_inches": float(layout.iloc[0]["height_inches"]),
        "lower_panel_width_spread_inches": width_spread,
        "lower_panel_height_spread_inches": height_spread,
        "global_panel_letters": "A-Q",
        "method_labels_on_canvas": [
            "Observed",
            "scGPT",
            "scGen",
            "WOT",
            "CellRank",
            DISPLAY_BRAND,
        ],
        "dense_points_rasterized_in_vector_outputs": True,
        "outputs": outputs,
        "script": {
            "relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    caption = (
        f"{DISPLAY_BRAND} across cell–cell communication and gene-expression benchmarks. "
        "Panels A–C summarize the frozen GSE267904 CCI diagnostic comparison. "
        "Panels D–K show gene-wise observed-versus-predicted expression for responder "
        "and non-responder groups. Panels L–Q show bootstrap Pearson distributions, "
        "signed errors, and expression-decile calibration. WOT is target-derived; "
        f"CellRank is a fate-weighted expression proxy; {DISPLAY_BRAND} expression is an offline "
        "runtime-expression proxy. The GSE267904 CCI source is the locked Gate-failed V5 "
        "diagnostic result and is not converted into an authorized gate-passing paper result "
        "by this visualization. CCI mappings and bootstrap values are not independent "
        "biological replicates. No overall score is calculated.\n"
    )
    (OUT / "caption.txt").write_text(caption, encoding="utf-8")
    (OUT / "README.md").write_text(
        f"# {README_TITLE}\n\n"
        f"Status: `{PENDING_STATUS}`\n\n"
        "This version redraws the frozen plotting values on one consistent Matplotlib canvas. "
        "The lower section is a strict 2×7 grid: four scatter columns and three summary columns, "
        "with all 14 axes having identical physical width and height. No model, COMMOT task, "
        "bootstrap, or scientific metric was rerun.\n\n"
        "Dense scatter/jitter layers are rasterized in PDF/SVG; text, axes, heatmaps, violins, "
        "boxes, and lines remain vector. Scientific limitations are in `caption.txt`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
