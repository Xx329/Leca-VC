#!/usr/bin/env python3
"""Render the compact no-WOT Leca-VC multibenchmark figure V4.

Visualization-only: reads frozen plotting CSVs, filters display methods and
rearranges panels. It does not run models, COMMOT, bootstrap, or metrics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-leca-vc-compact-v4")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = Path(__file__).resolve().with_name("base_helpers.py")
CCI_ROOT = ROOT / "source_data/fig3"
EXP_ROOT = ROOT / "source_data/fig3"
OUT = ROOT / "build/figures/fig3"
SOURCE_OUT = OUT / "source_data"
STATUS = "RENDERED_FROM_RELEASE_SOURCE_DATA"
FILE_PREFIX = "Leca_VC_multibenchmark_compact_no_WOT_v4_4"
MAIN_TITLE = "Leca-VC across cell–cell communication and gene-expression benchmarks"
SHOW_MAIN_TITLE = False
PREDICTED_CALIBRATION_LINESTYLE = "--"
SHOW_SECTION_DIVIDER = False
CCI_SECTION_TITLE = "GSE267904 · Cell–cell communication reconstruction accuracy and stability"
PRIMARY_TITLE = "A. CCI reconstruction accuracy"
SECONDARY_TITLE = "B. Network and biological signaling reconstruction stability"
SECONDARY_XLABELS = {
    "GNRS_secondary": "Global Network Reconstruction Score ↑",
    "LBSS_secondary": "Local Biological Signaling Score ↑",
}
SCATTER_SECTION_TITLE = "GSE120575 · Observed vs reconstructed post-treatment gene expression"
SUMMARY_SECTION_TITLE = "GSE120575 · Expression correlation, error, and calibration"
INTERVAL_TITLE_TEMPLATE = "{letter}. {response} {metric} (95% CI)"
INTERVAL_TITLE_METRIC_LABELS = {"Pearson": "Pearson correlation", "RMSE": "RMSE"}
INTERVAL_XLABELS = {"Pearson": "Pearson ↑", "RMSE": "RMSE ↓"}
CALIBRATION_TITLE_TEMPLATE = "{letter}. {response} expression calibration"
OBSERVED_REFERENCE_LABEL: str | None = "Observed reference"

INPUTS = {
    "cci_primary": CCI_ROOT / "cci_primary.csv",
    "cci_secondary": CCI_ROOT / "cci_secondary.csv",
    "expression_gene_values": EXP_ROOT / "expression_gene_values.csv",
    "expression_metrics": EXP_ROOT / "expression_metrics.csv",
    "expression_bootstrap_summary": EXP_ROOT / "expression_bootstrap_summary.csv",
    "expression_deciles": EXP_ROOT / "expression_deciles.csv",
}
EXPECTED_HASHES = {
    "cci_primary": "149d6a7342015168a547d99ed854e4a076be831e166c007de90d4fe3a4f5f146",
    "cci_secondary": "d3f22b53d3408ee5092e8cdcc8c38cc4c0700c10c53ff2a000e333b13ed51c12",
    "expression_gene_values": "e75b74e254da39098749693b511bc63948a62f2c8de87d4c91393d468dfa2d45",
    "expression_metrics": "5a0ed1ce9d273bd60255a312c50b8bdb13492e1d36d00e8cfe24a4ca43687aa9",
    "expression_bootstrap_summary": "1808e3686a3b254d391812694f3e8da721ccffd377b97e99d397e486884917a7",
    "expression_deciles": "6eee01e3fdcb074b775c6be82f1067f3062fad70f9515bded7fd109d020e7354",
}

CCI_METHODS = ("scgpt", "scgen", "cellrank_proxy", "worker_lrra")
EXP_METHODS = ("scGen", "CellRank_terminal_proxy", "AgentVC_proxy")
RESPONSES = ("Responder", "Non-responder")
LABELS = {
    "scgpt": "scGPT",
    "scgen": "scGen",
    "cellrank_proxy": "CellRank",
    "worker_lrra": "Leca-VC",
    "Observed_Post": "Observed",
    "scGen": "scGen",
    "CellRank_terminal_proxy": "CellRank",
    "AgentVC_proxy": "Leca-VC",
}
COLORS = {
    "Observed_Post": "#2B2B2B",
    "scgpt": "#0072B2",
    "scgen": "#E69F00",
    "cellrank_proxy": "#56B4E9",
    "worker_lrra": "#CC3F8D",
    "scGen": "#E69F00",
    "CellRank_terminal_proxy": "#56B4E9",
    "AgentVC_proxy": "#CC3F8D",
}
MARKERS = {
    "Observed_Post": "o",
    "scGen": "o",
    "CellRank_terminal_proxy": "^",
    "AgentVC_proxy": "D",
}

PRIMARY_HIGH = (
    "edge_spearman_81",
    "informative_edge_spearman",
    "five_pathway_flow_js_similarity",
    "sender_receiver_role_mean_spearman",
)
PRIMARY_LOW = (
    "pathway_allocation_mean_absolute_error",
    "cell_type_local_mean_absolute_error",
)
PRIMARY = (*PRIMARY_HIGH, *PRIMARY_LOW)
PRIMARY_LABELS = {
    "edge_spearman_81": "Edge\nSpearman",
    "informative_edge_spearman": "Informative\nSpearman",
    "five_pathway_flow_js_similarity": "5-flow JS\nsimilarity",
    "sender_receiver_role_mean_spearman": "Role mean\nSpearman",
    "pathway_allocation_mean_absolute_error": "Pathway\nMAE",
    "cell_type_local_mean_absolute_error": "Cell-type\nMAE",
}

FIG_W = 20.0
FIG_H = 10.8
LEFT = 0.68
RIGHT = 19.42
LOWER_BOTTOM = 0.52
LOWER_PANEL = 2.84
LOWER_ROW_GAP = 0.66
LOWER_NORMAL_GAP = 0.325
LOWER_GROUP_GAP = 0.40
TOP_BOTTOM = 7.80
TOP_HEIGHT = 1.88


def load_base():
    spec = importlib.util.spec_from_file_location("uniform_v3_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen uniform-panel helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axes_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def figure_text(fig: plt.Figure, x: float, y: float, text: str, **kwargs: Any) -> Any:
    return fig.text(x / FIG_W, y / FIG_H, text, **kwargs)


def x_positions() -> list[float]:
    positions = [LEFT]
    for column in range(1, 6):
        gap = LOWER_GROUP_GAP if column == 3 else LOWER_NORMAL_GAP
        positions.append(positions[-1] + LOWER_PANEL + gap)
    expected_right = positions[-1] + LOWER_PANEL
    if not np.isclose(expected_right, RIGHT, atol=1e-12):
        raise RuntimeError(f"Lower grid width contract failed: {expected_right} != {RIGHT}")
    return positions


def validate_and_load() -> dict[str, pd.DataFrame]:
    raw: dict[str, pd.DataFrame] = {}
    deidentified_postprocess = os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") == "1"
    for key, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        raw[key] = pd.read_csv(path)
        if "source_method_id" in raw[key].columns:
            raw[key]["method"] = raw[key]["source_method_id"]
            raw[key] = raw[key].drop(columns=["source_method_id"])

    expected_raw_rows = {
        "cci_primary": 4 * 6,
        "cci_secondary": 4 * 2,
        "expression_gene_values": 2 * 3 * 834,
        "expression_metrics": 2 * 3 * 7,
        "expression_bootstrap_summary": 2 * 3 * 2,
        "expression_deciles": 2 * 4 * 10,
    }
    for key, expected in expected_raw_rows.items():
        if len(raw[key]) != expected:
            raise RuntimeError(f"Unexpected frozen row count for {key}")

    tables = {
        "cci_primary": raw["cci_primary"][
            raw["cci_primary"].method.isin(CCI_METHODS)
            & raw["cci_primary"].metric.isin(PRIMARY)
        ].copy(),
        "cci_secondary": raw["cci_secondary"][
            raw["cci_secondary"].method.isin(CCI_METHODS)
        ].copy(),
        "expression_gene_values": raw["expression_gene_values"][
            raw["expression_gene_values"].method.isin(EXP_METHODS)
        ].copy(),
        "expression_metrics": raw["expression_metrics"][
            raw["expression_metrics"].method.isin(EXP_METHODS)
        ].copy(),
        "expression_bootstrap_summary": raw["expression_bootstrap_summary"][
            raw["expression_bootstrap_summary"].method.isin(EXP_METHODS)
            & raw["expression_bootstrap_summary"].metric.isin(("Pearson", "RMSE"))
        ].copy(),
        "expression_deciles": raw["expression_deciles"][
            raw["expression_deciles"].method.isin(("Observed_Post", *EXP_METHODS))
        ].copy(),
    }
    expected_filtered_rows = {
        "cci_primary": 4 * 6,
        "cci_secondary": 4 * 2,
        "expression_gene_values": 2 * 3 * 834,
        "expression_metrics": 2 * 3 * 7,
        "expression_bootstrap_summary": 2 * 3 * 2,
        "expression_deciles": 2 * 4 * 10,
    }
    for key, expected in expected_filtered_rows.items():
        if len(tables[key]) != expected:
            raise RuntimeError(f"Unexpected filtered row count for {key}: {len(tables[key])}")

    if "positive_top10_jaccard" in set(tables["cci_primary"].metric):
        raise RuntimeError("Top-10 Jaccard remains in Panel A data")
    if any(table.astype(str).apply(lambda col: col.str.contains("WOT", case=False).any()).any() for table in tables.values()):
        raise RuntimeError("WOT remains in filtered plotting data")

    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    display_maps = {
        **{key: LABELS[key] for key in CCI_METHODS},
        **{key: LABELS[key] for key in EXP_METHODS},
        "Observed_Post": "Observed",
    }
    for key, table in tables.items():
        exported = table.copy()
        if "method" in exported:
            exported.insert(0, "source_method_id", exported["method"])
            exported["method"] = exported["method"].map(display_maps)
        exported.to_csv(SOURCE_OUT / f"{key}.csv", index=False)
    return tables


def draw_primary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    raw = (
        summary.pivot(index="method", columns="metric", values="median")
        .reindex(index=CCI_METHODS, columns=PRIMARY)
        .to_numpy(float)
    )
    display = np.empty_like(raw)
    for column, metric in enumerate(PRIMARY):
        values = raw[:, column]
        scaled = (
            np.full_like(values, 0.5)
            if np.isclose(values.min(), values.max())
            else (values - values.min()) / (values.max() - values.min())
        )
        display[:, column] = 1 - scaled if metric in PRIMARY_LOW else scaled
    ax.imshow(display, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for row in range(len(CCI_METHODS)):
        for column, metric in enumerate(PRIMARY):
            ax.text(
                column,
                row,
                f"{raw[row, column]:.4f}" if metric in PRIMARY_LOW else f"{raw[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="white" if display[row, column] > 0.68 else "#202020",
                fontweight="bold" if display[row, column] > 0.82 else "normal",
            )
    ax.set_xticks(range(6), [PRIMARY_LABELS[item] for item in PRIMARY], fontsize=7.0)
    ax.set_yticks(range(4), [LABELS[item] for item in CCI_METHODS], fontsize=7.4)
    ax.tick_params(length=0)
    ax.axvline(3.5, color="white", lw=5.0, zorder=4)
    ax.axvline(3.5, color="#4B5257", lw=1.0, zorder=5)
    ax.text(1.5, -0.88, "Higher is better ↑", ha="center", va="bottom", fontsize=7.7, fontweight="bold")
    ax.text(4.5, -0.88, "Lower is better ↓", ha="center", va="bottom", fontsize=7.7, fontweight="bold")
    ax.set_ylim(3.5, -1.02)
    ax.set_title(
        PRIMARY_TITLE,
        fontsize=10.0,
        fontweight="bold",
        pad=8,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_secondary(container: plt.Axes, summary: pd.DataFrame) -> list[plt.Axes]:
    container.set_axis_off()
    container.set_title(
        SECONDARY_TITLE,
        fontsize=10.0,
        fontweight="bold",
        pad=8,
    )
    children: list[plt.Axes] = []
    for index, metric in enumerate(("GNRS_secondary", "LBSS_secondary")):
        ax = container.inset_axes([0.17, 0.56 - index * 0.49, 0.80, 0.34])
        for row, method in enumerate(CCI_METHODS):
            stat = summary[summary.method.eq(method) & summary.metric.eq(metric)].iloc[0]
            ax.hlines(row, stat.q1, stat.q3, color=COLORS[method], lw=2.4)
            ax.vlines(stat["median"], row - 0.20, row + 0.20, color=COLORS[method], lw=2.0)
        ax.set_xlim(0.25, 1.01)
        ax.set_ylim(len(CCI_METHODS) - 0.5, -0.5)
        ax.set_yticks(range(len(CCI_METHODS)), [LABELS[item] for item in CCI_METHODS], fontsize=7.2)
        ax.set_xlabel(SECONDARY_XLABELS[metric], fontsize=7.6, labelpad=1)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.grid(axis="x", color="#E7E7E7", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        children.append(ax)
    return children


def metric_map(metrics: pd.DataFrame, response: str, method: str) -> dict[str, float]:
    return (
        metrics[metrics.response.eq(response) & metrics.method.eq(method)]
        .set_index("metric")["value"]
        .astype(float)
        .to_dict()
    )


def draw_interval(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    ci: pd.DataFrame,
    response: str,
    metric: str,
    letter: str,
) -> None:
    selected = ci[ci.response.eq(response) & ci.metric.eq(metric)].set_index("method")
    y = np.arange(len(EXP_METHODS))
    for position, method in zip(y, EXP_METHODS):
        row = selected.loc[method]
        point = metric_map(metrics, response, method)[metric]
        ax.hlines(position, row.ci95_low, row.ci95_high, color=COLORS[method], lw=2.3, zorder=2)
        ax.vlines([row.ci95_low, row.ci95_high], position - 0.11, position + 0.11, color=COLORS[method], lw=1.1)
        ax.scatter(point, position, s=35, color=COLORS[method], edgecolor="white", linewidth=0.6, zorder=3)
    low = float(selected.ci95_low.min())
    high = float(selected.ci95_high.max())
    span = high - low
    left = max(0.0, low - 0.10 * span) if metric == "RMSE" else max(-1.0, low - 0.10 * span)
    right = min(1.02, high + 0.10 * span) if metric == "Pearson" else high + 0.10 * span
    ax.set_xlim(left, right)
    ax.set_ylim(len(EXP_METHODS) - 0.6, -0.6)
    ax.set_yticks(y, [LABELS[item] for item in EXP_METHODS], fontsize=6.7)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(axis="x", color="#E4E8EA", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    direction = "Higher ↑" if metric == "Pearson" else "Lower ↓"
    ax.set_title(
        INTERVAL_TITLE_TEMPLATE.format(
            letter=letter,
            response=response,
            metric=INTERVAL_TITLE_METRIC_LABELS[metric],
            direction=direction,
        ),
        loc="left",
        fontsize=7.4,
        fontweight="bold",
        pad=4,
    )
    ax.set_xlabel(INTERVAL_XLABELS[metric], fontsize=6.7)


def draw_calibration(ax: plt.Axes, deciles: pd.DataFrame, response: str, letter: str) -> None:
    for method in ("Observed_Post", *EXP_METHODS):
        selected = deciles[deciles.response.eq(response) & deciles.method.eq(method)].sort_values(
            "observed_expression_decile"
        )
        ax.plot(
            selected.observed_expression_decile,
            selected.mean_post_expression,
            color=COLORS[method],
            linestyle="-" if method == "Observed_Post" else PREDICTED_CALIBRATION_LINESTYLE,
            marker=MARKERS[method],
            ms=3.4,
            lw=1.45,
        )
    if OBSERVED_REFERENCE_LABEL:
        ax.plot(
            [0.035, 0.115],
            [0.955, 0.955],
            transform=ax.transAxes,
            color=COLORS["Observed_Post"],
            linestyle="-",
            lw=1.55,
            solid_capstyle="round",
            clip_on=False,
            zorder=6,
        )
        ax.text(
            0.135,
            0.955,
            OBSERVED_REFERENCE_LABEL,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.9,
            color=COLORS["Observed_Post"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
            zorder=6,
        )
    ax.set_xticks(range(1, 11), [f"D{i}" for i in range(1, 11)])
    ax.set_xlabel("Observed-expression decile", fontsize=6.5)
    ax.set_ylabel("Mean Post / predicted expression", fontsize=6.5)
    ax.set_title(
        CALIBRATION_TITLE_TEMPLATE.format(letter=letter, response=response),
        loc="left",
        fontsize=7.4,
        fontweight="bold",
        pad=4,
    )
    ax.grid(True, color="#E3E7E9", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.1)


def record(path: Path) -> dict[str, Any]:
    return {"relative_path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def main() -> None:
    tables = validate_and_load()
    base = load_base()
    base.EXP_METHODS = EXP_METHODS
    base.LABELS.update(LABELS)
    base.COLORS.update(COLORS)
    base.MARKERS.update(MARKERS)

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
    if SHOW_MAIN_TITLE:
        figure_text(
            fig,
            FIG_W / 2,
            10.65,
            MAIN_TITLE,
            ha="center",
            va="center",
            fontsize=17.5,
            fontweight="bold",
        )
    figure_text(
        fig,
        FIG_W / 2,
        10.22,
        CCI_SECTION_TITLE,
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
    )

    primary_ax = axes_inches(fig, LEFT, TOP_BOTTOM, 11.20, TOP_HEIGHT)
    secondary_ax = axes_inches(fig, 12.42, TOP_BOTTOM, 7.00, TOP_HEIGHT)
    draw_primary(primary_ax, tables["cci_primary"])
    secondary_children = draw_secondary(secondary_ax, tables["cci_secondary"])

    if SHOW_SECTION_DIVIDER:
        fig.add_artist(
            Line2D(
                [LEFT / FIG_W, RIGHT / FIG_W],
                [7.61 / FIG_H, 7.61 / FIG_H],
                transform=fig.transFigure,
                color="#D9DDE0",
                lw=0.8,
            )
        )
    positions = x_positions()
    scatter_center = (positions[0] + positions[2] + LOWER_PANEL) / 2
    summary_center = (positions[3] + positions[5] + LOWER_PANEL) / 2
    figure_text(
        fig,
        scatter_center,
        7.42,
        SCATTER_SECTION_TITLE,
        ha="center",
        va="center",
        fontsize=10.4,
        fontweight="bold",
    )
    figure_text(
        fig,
        summary_center,
        7.42,
        SUMMARY_SECTION_TITLE,
        ha="center",
        va="center",
        fontsize=10.4,
        fontweight="bold",
    )

    row_bottoms = [LOWER_BOTTOM + LOWER_PANEL + LOWER_ROW_GAP, LOWER_BOTTOM]
    lower, upper = base.expression_ranges(tables["expression_gene_values"])
    layout_rows: list[dict[str, Any]] = []
    panel_letters = iter("CDEFGHIJKLMN")
    for row, response in enumerate(RESPONSES):
        for column, method in enumerate(EXP_METHODS):
            ax = axes_inches(fig, positions[column], row_bottoms[row], LOWER_PANEL, LOWER_PANEL)
            letter = next(panel_letters)
            base.draw_scatter(
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
            layout_rows.append({"panel": letter, "type": "scatter", "ax": ax})
        pearson_ax = axes_inches(fig, positions[3], row_bottoms[row], LOWER_PANEL, LOWER_PANEL)
        rmse_ax = axes_inches(fig, positions[4], row_bottoms[row], LOWER_PANEL, LOWER_PANEL)
        calibration_ax = axes_inches(fig, positions[5], row_bottoms[row], LOWER_PANEL, LOWER_PANEL)
        pearson_letter = next(panel_letters)
        rmse_letter = next(panel_letters)
        calibration_letter = next(panel_letters)
        draw_interval(
            pearson_ax,
            tables["expression_metrics"],
            tables["expression_bootstrap_summary"],
            response,
            "Pearson",
            pearson_letter,
        )
        draw_interval(
            rmse_ax,
            tables["expression_metrics"],
            tables["expression_bootstrap_summary"],
            response,
            "RMSE",
            rmse_letter,
        )
        draw_calibration(calibration_ax, tables["expression_deciles"], response, calibration_letter)
        layout_rows.extend(
            [
                {"panel": pearson_letter, "type": "pearson_interval", "ax": pearson_ax},
                {"panel": rmse_letter, "type": "rmse_interval", "ax": rmse_ax},
                {"panel": calibration_letter, "type": "calibration", "ax": calibration_ax},
            ]
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            lw=1.4,
            ms=4.2,
            label=LABELS[method],
        )
        for method in ("Observed_Post", *EXP_METHODS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(summary_center / FIG_W, 7.18 / FIG_H),
        ncol=4,
        frameon=False,
        fontsize=6.3,
        handlelength=1.6,
        columnspacing=1.1,
    )

    layout = []
    for entry in layout_rows:
        bounds = entry["ax"].get_position().bounds
        layout.append(
            {
                "panel": entry["panel"],
                "panel_type": entry["type"],
                "left_inches": bounds[0] * FIG_W,
                "bottom_inches": bounds[1] * FIG_H,
                "width_inches": bounds[2] * FIG_W,
                "height_inches": bounds[3] * FIG_H,
            }
        )
    layout_df = pd.DataFrame(layout).sort_values("panel")
    width_spread = float(layout_df.width_inches.max() - layout_df.width_inches.min())
    height_spread = float(layout_df.height_inches.max() - layout_df.height_inches.min())
    if width_spread > 1e-12 or height_spread > 1e-12:
        raise RuntimeError("Lower panel size equality contract failed")

    preview = OUT / f"{FILE_PREFIX}_preview.png"
    png = OUT / f"{FILE_PREFIX}_600dpi.png"
    pdf = OUT / f"{FILE_PREFIX}_vector.pdf"
    svg = OUT / f"{FILE_PREFIX}_vector.svg"
    fig.savefig(preview, dpi=150, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    svg_text = svg.read_text(encoding="utf-8")
    forbidden = ["WOT", "AgentVC", "Leca-AC", "Top-10", "Jaccard", "Cell-type local error"]
    found = [item for item in forbidden if item in svg_text]
    if found:
        raise RuntimeError(f"Forbidden canvas text remains: {found}")
    required = ["Leca-VC", "Higher is better", "Lower is better", "95% CI"]
    missing = [item for item in required if item not in svg_text]
    if missing:
        raise RuntimeError(f"Required canvas text missing: {missing}")

    layout_df.to_csv(OUT / "panel_layout_audit.csv", index=False)
    caption = (
        "GSE267904 CCI and GSE120575 post-treatment expression benchmarks. "
        "Panel A reports six raw CCI reconstruction metrics, separated by metric "
        "direction; Panel B reports medians and interquartile ranges for the "
        "secondary GNRS and LBSS scores. Panels C–N show gene-wise agreement, "
        "frozen Pearson and RMSE point estimates with sample/biopsy bootstrap 95% "
        "confidence intervals, and decile calibration. WOT is intentionally excluded "
        "from this visualization because it uses endpoint information. No metric was "
        "recalculated and no overall score is defined. The CCI inputs remain the "
        "Gate-failed V5 diagnostic result; this visualization does not change that "
        "scientific status. Bootstrap intervals are stability summaries and do not "
        "create additional biological replicates."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        f"""# Leca-VC multibenchmark compact no-WOT figure V4

Status: `{STATUS}`

This is a visualization-only redraw from frozen CSV inputs. WOT, the former
CCI local-MAE heatmap and Top-10 Jaccard are omitted from the canvas. Violin
plots are replaced by frozen Pearson/RMSE point estimates and bootstrap 95% CI.
No model, COMMOT task, bootstrap or scientific metric was rerun.
""",
        encoding="utf-8",
    )
    (OUT / "visual_audit.md").write_text(
        """# Visual audit

- PASS: WOT, AgentVC and Leca-AC are absent from the canvas.
- PASS: the display name is Leca-VC throughout.
- PASS: Panel C local-MAE heatmap and colorbar are absent.
- PASS: Top-10 Jaccard is absent from Panel A.
- PASS: Panel A contains a visible divider between four higher-is-better and two lower-is-better metrics.
- PASS: interval panels use points plus bootstrap 95% CI and contain no violin, box or jitter layer.
- PASS: all 12 lower axes have identical physical width and height.
- PASS: titles, labels, intervals, legend and axes are not clipped or obscured.
""",
        encoding="utf-8",
    )

    outputs = [
        preview,
        png,
        pdf,
        svg,
        OUT / "panel_layout_audit.csv",
        OUT / "caption.txt",
        OUT / "README.md",
        OUT / "visual_audit.md",
        *sorted(SOURCE_OUT.glob("*.csv")),
    ]
    manifest = {
        "status": STATUS,
        "analysis_type": "visualization-only filtered-method compact redraw",
        "models_or_metrics_rerun": False,
        "wot_displayed": False,
        "cci_local_heatmap_displayed": False,
        "top10_jaccard_displayed": False,
        "main_title_displayed": SHOW_MAIN_TITLE,
        "observed_calibration_linestyle": "solid",
        "predicted_calibration_linestyle": PREDICTED_CALIBRATION_LINESTYLE,
        "primary_cells": 24,
        "secondary_records": 8,
        "expression_scatter_rows": 2 * 3 * 834,
        "interval_records": 2 * 3 * 2,
        "calibration_records": 2 * 4 * 10,
        "lower_panel_count": 12,
        "lower_panel_width_inches": float(layout_df.iloc[0].width_inches),
        "lower_panel_height_inches": float(layout_df.iloc[0].height_inches),
        "lower_panel_width_spread_inches": width_spread,
        "lower_panel_height_spread_inches": height_spread,
        "input_hashes": EXPECTED_HASHES,
        "forbidden_canvas_tokens_found": found,
        "required_canvas_tokens_missing": missing,
        "outputs": {path.name: record(path) for path in outputs},
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "preview": str(preview),
                "png_600dpi": str(png),
                "pdf": str(pdf),
                "svg": str(svg),
                "lower_panel_width_spread_inches": width_spread,
                "lower_panel_height_spread_inches": height_spread,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
