#!/usr/bin/env python3
"""Render the frozen GSE267904 AgentVC-HW V3 3x3 comparison figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    ALL_SOURCES,
    COLORS,
    METHOD_LABELS,
    OUT,
    PREDICTED_METHODS,
    SAMPLE_SEEDS,
    TYPES,
    TYPE_SHORT,
    sha256_file,
    write_json,
)


METRICS = OUT / "05_metrics"
FIGURE = OUT / "06_figure"
SOURCE = FIGURE / "source_data"
PRIMARY = [
    "edge_spearman_81",
    "informative_edge_spearman",
    "positive_top10_jaccard",
    "five_pathway_flow_js_similarity",
    "sender_receiver_role_mean_spearman",
    "pathway_allocation_mean_absolute_error",
    "cell_type_local_mean_absolute_error",
]
PRIMARY_LABEL = {
    "edge_spearman_81": "Edge\nSpearman ↑",
    "informative_edge_spearman": "Informative\nSpearman ↑",
    "positive_top10_jaccard": "Top-10\nJaccard ↑",
    "five_pathway_flow_js_similarity": "5-flow JS\nsimilarity ↑",
    "sender_receiver_role_mean_spearman": "Role mean\nSpearman ↑",
    "pathway_allocation_mean_absolute_error": "Pathway\nMAE ↓",
    "cell_type_local_mean_absolute_error": "Cell-type\nMAE ↓",
}
SHORT = {
    "scgpt": "scGPT",
    "scgen": "scGen",
    "wot_external": "WOT*",
    "cellrank_proxy": "CellRank†",
    "worker": "AgentVC-old",
    "worker_hw": "AgentVC-HW",
}
HEATMAP_TITLE = {
    "observed": "Observed d21\nreference",
    "scgpt": "Frozen scGPT +\nexternal head",
    "scgen": "scGen 2.1.0",
    "wot_external": "external-WOT\ntransport projection*",
    "cellrank_proxy": "CellRank fate-weighted\nexpression proxy†",
    "worker_hw": "AgentVC heterogeneous-worker\nexpression proxy",
}


def letter(ax: plt.Axes, value: str) -> None:
    ax.text(
        -0.12,
        1.10,
        value,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def load_networks(edge_values: pd.DataFrame) -> pd.DataFrame:
    affine = edge_values[edge_values["geometry_mode"].eq("affine_pixel")].copy()
    consistency = (
        affine.groupby(["sampling_seed", "sender", "receiver"])[
            "observed_normalized_weight"
        ]
        .agg(lambda x: float(x.max() - x.min()))
        .max()
    )
    if consistency > 1e-12:
        raise RuntimeError("Observed network differs among method copies")
    observed = (
        affine.drop_duplicates(["sampling_seed", "sender", "receiver"])[
            ["sampling_seed", "sender", "receiver", "observed_normalized_weight"]
        ]
        .rename(columns={"observed_normalized_weight": "normalized_weight"})
        .assign(source="observed")
    )
    predicted = affine[affine["method"].isin(ALL_SOURCES)][
        [
            "sampling_seed",
            "method",
            "sender",
            "receiver",
            "method_normalized_weight",
        ]
    ].rename(
        columns={"method": "source", "method_normalized_weight": "normalized_weight"}
    )
    values = pd.concat([observed, predicted], ignore_index=True)
    counts = values.groupby(["source", "sampling_seed"]).size()
    expected = len(ALL_SOURCES) * len(SAMPLE_SEEDS)
    if len(counts) != expected or not (counts == 81).all():
        raise RuntimeError("A-F network source is not six complete 10x81 sets")
    sums = values.groupby(["source", "sampling_seed"])["normalized_weight"].sum()
    if not np.allclose(sums, 1.0, atol=1e-9):
        raise RuntimeError("At least one A-F network is not normalized")
    summary = (
        values.groupby(["source", "sender", "receiver"])["normalized_weight"]
        .agg(n="count", median="median", minimum="min", maximum="max")
        .reset_index()
    )
    if len(summary) != len(ALL_SOURCES) * 81 or not (summary["n"] == 10).all():
        raise RuntimeError("A-F summary is incomplete")
    values.to_csv(SOURCE / "panels_A-F_network_seed_values.csv", index=False)
    summary.to_csv(SOURCE / "panels_A-F_network_heatmaps.csv", index=False)
    return summary


def matrix(networks: pd.DataFrame, source: str) -> np.ndarray:
    table = networks[networks["source"].eq(source)].pivot(
        index="sender", columns="receiver", values="median"
    )
    return table.reindex(index=TYPES, columns=TYPES).to_numpy(float)


def summarize(
    values: pd.DataFrame, group: list[str], value: str = "value"
) -> pd.DataFrame:
    return (
        values.groupby(group)[value]
        .agg(
            n="count",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )


def load_metrics(
    seed_metrics: pd.DataFrame, local: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = seed_metrics[
        seed_metrics["geometry_mode"].eq("affine_pixel")
        & seed_metrics["method"].isin(PREDICTED_METHODS)
        & seed_metrics["metric"].isin(PRIMARY)
    ][["sampling_seed", "method", "metric", "value"]].copy()
    secondary = seed_metrics[
        seed_metrics["geometry_mode"].eq("affine_pixel")
        & seed_metrics["method"].isin(PREDICTED_METHODS)
        & seed_metrics["metric"].isin(["GNRS_secondary", "LBSS_secondary"])
    ][["sampling_seed", "method", "metric", "value"]].copy()
    local_values = local[
        local["geometry_mode"].eq("affine_pixel")
        & local["method"].isin(PREDICTED_METHODS)
        & local["cell_type"].isin(TYPES)
    ][
        [
            "sampling_seed",
            "method",
            "cell_type",
            "local_mean_absolute_error",
        ]
    ].copy()
    expected = {
        "primary": (primary, 6 * 7 * 10, ["method", "metric"]),
        "secondary": (secondary, 6 * 2 * 10, ["method", "metric"]),
        "local": (local_values, 6 * 9 * 10, ["method", "cell_type"]),
    }
    for name, (table, rows, grouping) in expected.items():
        counts = table.groupby(grouping).size()
        if len(table) != rows or not (counts == 10).all():
            raise RuntimeError(f"{name} source is incomplete")
    primary_summary = summarize(primary, ["method", "metric"])
    secondary_summary = summarize(secondary, ["method", "metric"])
    local_summary = summarize(
        local_values,
        ["method", "cell_type"],
        "local_mean_absolute_error",
    )
    primary.to_csv(SOURCE / "panel_G_primary_seed_values.csv", index=False)
    primary_summary.to_csv(SOURCE / "panel_G_primary_summary.csv", index=False)
    secondary.to_csv(SOURCE / "panel_H_secondary_seed_values.csv", index=False)
    secondary_summary.to_csv(SOURCE / "panel_H_secondary_summary.csv", index=False)
    local_values.to_csv(SOURCE / "panel_I_local_seed_values.csv", index=False)
    local_summary.to_csv(SOURCE / "panel_I_local_summary.csv", index=False)
    return primary_summary, secondary, secondary_summary, local_summary


def heatmap(
    ax: plt.Axes, values: np.ndarray, source: str, panel: str, vmax: float
):
    image = ax.imshow(
        values,
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks(
        range(9), [TYPE_SHORT[x] for x in TYPES], rotation=52, ha="right", fontsize=6
    )
    ax.set_yticks(range(9), [TYPE_SHORT[x] for x in TYPES], fontsize=6)
    ax.set_xlabel("Receiver cell type", fontsize=7)
    ax.set_ylabel("Sender cell type", fontsize=7)
    ax.set_title(
        HEATMAP_TITLE[source],
        fontsize=9.5,
        fontweight="bold",
        color=COLORS[source],
        pad=7,
    )
    for spine in ax.spines.values():
        spine.set_color(COLORS[source])
        spine.set_linewidth(1.2)
    letter(ax, panel)
    return image


def primary_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    raw = (
        summary.pivot(index="method", columns="metric", values="median")
        .reindex(index=PREDICTED_METHODS, columns=PRIMARY)
        .to_numpy(float)
    )
    shown = np.empty_like(raw)
    lower = {
        "pathway_allocation_mean_absolute_error",
        "cell_type_local_mean_absolute_error",
    }
    for col, metric in enumerate(PRIMARY):
        x = raw[:, col]
        scaled = (
            np.full_like(x, 0.5)
            if np.isclose(x.min(), x.max())
            else (x - x.min()) / (x.max() - x.min())
        )
        shown[:, col] = 1 - scaled if metric in lower else scaled
    ax.imshow(shown, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            ax.text(
                col,
                row,
                f"{raw[row, col]:.4f}" if col >= 5 else f"{raw[row, col]:.3f}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if shown[row, col] > 0.68 else "#202020",
                fontweight="bold" if shown[row, col] > 0.82 else "normal",
            )
    ax.set_xticks(range(7), [PRIMARY_LABEL[x] for x in PRIMARY], fontsize=5.7)
    ax.set_yticks(range(6), [SHORT[x] for x in PREDICTED_METHODS], fontsize=6.4)
    ax.tick_params(length=0)
    ax.set_title(
        "Primary reconstruction metrics\nRaw medians; directional shading within columns",
        fontsize=8.7,
        fontweight="bold",
    )
    ax.spines[:].set_visible(False)
    letter(ax, "G")


def secondary_panel(
    container: plt.Axes, values: pd.DataFrame, summary: pd.DataFrame
) -> None:
    container.set_axis_off()
    container.set_title(
        "GNRS and LBSS stability\nSecondary descriptive scores",
        fontsize=8.7,
        fontweight="bold",
    )
    letter(container, "H")
    for panel_index, metric in enumerate(["GNRS_secondary", "LBSS_secondary"]):
        ax = container.inset_axes([0.12, 0.54 - panel_index * 0.47, 0.84, 0.36])
        for row, method in enumerate(PREDICTED_METHODS):
            selected = values[
                values["method"].eq(method) & values["metric"].eq(metric)
            ].sort_values("sampling_seed")
            ax.scatter(
                selected["value"],
                row + np.linspace(-0.11, 0.11, len(selected)),
                s=12,
                color=COLORS[method],
                alpha=0.55,
                edgecolors="none",
            )
            stat = summary[
                summary["method"].eq(method) & summary["metric"].eq(metric)
            ].iloc[0]
            ax.errorbar(
                stat["median"],
                row,
                xerr=np.array(
                    [[stat["median"] - stat["q1"]], [stat["q3"] - stat["median"]]]
                ),
                fmt="D",
                ms=3.8,
                color=COLORS[method],
                capsize=2,
                lw=1.6,
                zorder=4,
            )
        old = summary[
            summary["method"].eq("worker") & summary["metric"].eq(metric)
        ]["median"].iloc[0]
        new = summary[
            summary["method"].eq("worker_hw") & summary["metric"].eq(metric)
        ]["median"].iloc[0]
        ax.annotate(
            f"old→HW  Δ={new-old:+.3f}",
            xy=(new, 5),
            xytext=(max(0.26, min(old, new) - 0.08), 4.45),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555555"},
            fontsize=5.7,
            color="#444444",
        )
        ax.set_xlim(0.25, 1.01)
        ax.set_ylim(5.5, -0.5)
        ax.set_yticks(range(6), [SHORT[x] for x in PREDICTED_METHODS], fontsize=5.8)
        ax.set_xlabel(
            "GNRS ↑" if metric == "GNRS_secondary" else "LBSS ↑",
            fontsize=6.5,
            labelpad=1,
        )
        ax.tick_params(axis="x", labelsize=5.8)
        ax.grid(axis="x", color="#E7E7E7", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)


def local_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    raw = (
        summary.pivot(index="method", columns="cell_type", values="median")
        .reindex(index=PREDICTED_METHODS, columns=TYPES)
        .to_numpy(float)
    )
    image = ax.imshow(raw, cmap="magma_r", aspect="auto")
    threshold = float(np.quantile(raw, 0.55))
    for row in range(6):
        for col in range(9):
            ax.text(
                col,
                row,
                f"{raw[row, col]:.4f}",
                ha="center",
                va="center",
                fontsize=4.9,
                color="white" if raw[row, col] >= threshold else "#202020",
            )
    ax.set_xticks(
        range(9), [TYPE_SHORT[x] for x in TYPES], rotation=49, ha="right", fontsize=5.5
    )
    ax.set_yticks(range(6), [SHORT[x] for x in PREDICTED_METHODS], fontsize=6.2)
    ax.tick_params(length=0)
    ax.set_title(
        "Cell-type local error\nMedian MAE across 10 mappings; lower is better",
        fontsize=8.7,
        fontweight="bold",
    )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Local MAE", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5)
    ax.spines[:].set_visible(False)
    letter(ax, "I")


def main() -> int:
    FIGURE.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    inputs = [
        METRICS / "edge_level_values.csv",
        METRICS / "seed_level_metrics.csv",
        METRICS / "cell_type_local_errors.csv",
        OUT / "07_audits/final_metric_status.json",
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    edge_values = pd.read_csv(inputs[0])
    seed_metrics = pd.read_csv(inputs[1])
    local_values = pd.read_csv(inputs[2])
    status = json.loads(inputs[3].read_text(encoding="utf-8"))
    networks = load_networks(edge_values)
    primary, secondary_values, secondary, local = load_metrics(
        seed_metrics, local_values
    )
    matrices = {source: matrix(networks, source) for source in ALL_SOURCES}
    vmax = float(max(value.max() for value in matrices.values()))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(19.5, 14.4), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        left=0.055,
        right=0.965,
        bottom=0.11,
        top=0.91,
        hspace=0.49,
        wspace=0.31,
        height_ratios=[1, 1, 1.08],
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(3) for col in range(3)]
    image = None
    for ax, source, panel in zip(axes[:6], ALL_SOURCES, "ABCDEF"):
        image = heatmap(ax, matrices[source], source, panel, vmax)
    color_ax = fig.add_axes([0.972, 0.487, 0.009, 0.37])
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label("Median normalized CCI edge weight", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    primary_panel(axes[6], primary)
    secondary_panel(axes[7], secondary_values, secondary)
    local_panel(axes[8], local)

    fig.suptitle(
        "GSE267904 heterogeneous-worker cell–cell communication benchmark",
        fontsize=17,
        fontweight="bold",
        y=0.972,
    )
    outcome = (
        "AgentVC-HW ranks first for frozen GNRS and overall local-MAE targets."
        if status["status"] == "PASS_AGENTVC_HW_TOP1_GNRS_AND_LOCAL"
        else "AgentVC-HW target status is reported without post-evaluation tuning."
    )
    fig.text(
        0.055,
        0.041,
        "* external-WOT uses an external d21 population; † CellRank is a fate-weighted "
        "terminal-state expression proxy. AgentVC-HW deterministically preserves d7 worker "
        "heterogeneity while conserving each frozen parent target. GNRS/LBSS are secondary "
        "descriptive scores; no overall score or biological confidence interval is implied. "
        + outcome,
        fontsize=7.35,
        color="#3E3E3E",
        ha="left",
        wrap=True,
    )
    fig.text(
        0.055,
        0.019,
        "Post-hoc target-blind architectural repair; late evaluation was performed once. "
        "Capacity-matched worker-level comparison; pseudo-spot confirmation remains unavailable.",
        fontsize=7.35,
        color="#6A3D00",
        ha="left",
        fontweight="bold",
    )
    outputs = {
        "preview_png": FIGURE / "GSE267904_AgentVC_HW_CCI_main_v3_preview.png",
        "png_600dpi": FIGURE / "GSE267904_AgentVC_HW_CCI_main_v3_600dpi.png",
        "pdf": FIGURE / "GSE267904_AgentVC_HW_CCI_main_v3.pdf",
        "svg": FIGURE / "GSE267904_AgentVC_HW_CCI_main_v3.svg",
    }
    fig.savefig(outputs["preview_png"], dpi=150, bbox_inches="tight")
    fig.savefig(outputs["png_600dpi"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure X. GSE267904 heterogeneous-worker CCI benchmark. A-F show median "
        "normalized 9x9 networks using one shared scale. G reports raw primary metric "
        "medians and direction-aligned display shading; old AgentVC is retained only as "
        "an architectural ablation in G-I. H reports ten frozen mappings for descriptive "
        "GNRS and LBSS with medians and IQR. I reports all nine cell-type local errors. "
        "AgentVC-HW is a post-hoc target-blind proxy, not a native worker transcriptome. "
        "No pseudo-spot confirmation or biological inferential claim is made."
    )
    (FIGURE / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    source_files = sorted(SOURCE.glob("*.csv"))
    manifest = {
        "status": "PASS_AGENTVC_HW_V3_SINGLE_3X3_FIGURE",
        "metric_status": status["status"],
        "heatmap_sources": ALL_SOURCES,
        "summary_methods": PREDICTED_METHODS,
        "shared_heatmap_vmin": 0.0,
        "shared_heatmap_vmax": vmax,
        "sampling_seeds": SAMPLE_SEEDS,
        "edges_per_network": 81,
        "GNRS_LBSS_secondary_descriptive_only": True,
        "overall_score": None,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "source_data_sha256": {
            str(path): sha256_file(path) for path in source_files
        },
        "output_sha256": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
    }
    write_json(FIGURE / "figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
