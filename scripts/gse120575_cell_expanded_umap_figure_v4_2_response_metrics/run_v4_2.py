#!/usr/bin/env python3
"""Visual-only V4.2 using the frozen response-level CCC/RMSE values."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR", "/tmp/numba-gse120575-expression-accuracy-v4-2"
)
os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/matplotlib-gse120575-expression-accuracy-v4-2"
)

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy"
V3_OUT = BASE / "cell_expanded_umap_figure_v3"
V41_OUT = BASE / "cell_expanded_umap_figure_v4_1_layout_refined"
OUT = BASE / "cell_expanded_umap_figure_v4_2_response_metrics"

V3_H5AD = V3_OUT / "unified_cell_expanded_expression_834genes.h5ad"
V3_SCRIPT = ROOT / "scripts/gse120575_cell_expanded_umap_figure_v3/run_v3.py"
V41_SCRIPT = (
    ROOT
    / "scripts/gse120575_cell_expanded_umap_figure_v4_1_layout_refined"
    / "run_v4_1.py"
)
V41_AUDIT = V41_OUT / "figure_audit.json"
FROZEN_METRICS = BASE / "unified_expression_metrics.csv"
EXPRESSION_V3_AUDIT = BASE / "V3_AUDIT.json"

EXPECTED_HASHES = {
    "v3_h5ad": "d3aa2ede49a58b9b7328b93ca6aa8938dfd7efdcf344cd858fc7faa2c0f90934",
    "v3_script": "ef81489f7ed7e29fce33e6ed3ae9af6cb4b60d06a8ad2e31607537c0e37ac279",
    "v41_script": "2e33497b495fc09986ac9b59fb9bbbb652de2eeac8b5dd350bd4811b4753695a",
    "v41_audit": "0c660588bf694ede7d03b7cec3d59288b60938c68b69f52f816f579aaff7302d",
    "frozen_metrics": "5a0ed1ce9d273bd60255a312c50b8bdb13492e1d36d00e8cfe24a4ca43687aa9",
    "expression_v3_audit": "c41bb0bbc21ff19571ecc5533a4453330d8da3a88c5b54c78717465aa9e61568",
}

METHODS = (
    "scGen",
    "WOT_weighted",
    "CellRank_terminal_proxy",
    "AgentVC_proxy",
)
METHOD_LABELS = {
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "CellRank_terminal_proxy": "CellRank\u2020",
    "AgentVC_proxy": "AgentVC",
}
METHOD_COLORS = {
    "scGen": "#E69F00",
    "WOT_weighted": "#009E73",
    "CellRank_terminal_proxy": "#56B4E9",
    "AgentVC_proxy": "#CC79A7",
}
RESPONSES = ("Responder", "Non-responder")
RESPONSE_MARKERS = {"Responder": "o", "Non-responder": "^"}

FIGURE_STEM = (
    "GSE120575_cell_expanded_response_expression_accuracy_UMAP_v4_2"
)
PREVIEW = OUT / f"{FIGURE_STEM}_preview.png"
PNG_600 = OUT / f"{FIGURE_STEM}_600dpi.png"
PDF = OUT / f"{FIGURE_STEM}_vector.pdf"
SVG = OUT / f"{FIGURE_STEM}.svg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_sources() -> dict[str, str]:
    paths = {
        "v3_h5ad": V3_H5AD,
        "v3_script": V3_SCRIPT,
        "v41_script": V41_SCRIPT,
        "v41_audit": V41_AUDIT,
        "frozen_metrics": FROZEN_METRICS,
        "expression_v3_audit": EXPRESSION_V3_AUDIT,
    }
    hashes = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
        if hashes[name] != EXPECTED_HASHES[name]:
            raise RuntimeError(f"Frozen source changed: {name}")
    if (
        json.loads(V41_AUDIT.read_text())["status"]
        != "PASS_V4_1_PANEL_F_LAYOUT_REFINEMENT_ONLY"
    ):
        raise RuntimeError("V4.1 layout is not final PASS")
    if (
        json.loads(EXPRESSION_V3_AUDIT.read_text())["status"]
        != "PASS_GSE120575_EXPRESSION_V3_CELLRANK_PROXY"
    ):
        raise RuntimeError("Frozen response-level benchmark is not final PASS")
    return hashes


def load_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    adata = ad.read_h5ad(V3_H5AD, backed="r")
    if adata.shape != (16178, 834):
        raise RuntimeError("Unexpected frozen H5AD shape")
    metadata = adata.obs.copy().reset_index(drop=True)
    coordinates = np.asarray(adata.obsm["X_umap"], dtype=float).copy()
    adata.file.close()

    frozen = pd.read_csv(FROZEN_METRICS)
    selected = frozen[
        frozen["metric"].isin(["CCC", "RMSE"])
        & frozen["method"].isin(METHODS)
        & frozen["response"].isin(RESPONSES)
    ].copy()
    if len(selected) != 16:
        raise RuntimeError("Frozen CCC/RMSE table is not 16 long rows")
    values = selected.pivot(
        index=[
            "response",
            "method",
            "method_label",
            "n_virtual_source_samples",
            "n_observed_post_biopsies",
            "gene_count",
            "expression_scale",
        ],
        columns="metric",
        values="value",
    ).reset_index()
    values.columns.name = None
    values = values.rename(columns={"CCC": "lins_ccc", "RMSE": "rmse_log1p_tpm"})
    if len(values) != 8:
        raise RuntimeError("Expected eight response-method rows")
    if not np.isfinite(
        values[["lins_ccc", "rmse_log1p_tpm"]].to_numpy(float)
    ).all():
        raise RuntimeError("Response-level metrics are not finite")
    return metadata, coordinates, values


def plot_response_metric_axis(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    metric: str,
    xlabel: str,
    xlim: tuple[float, float],
    perfect_value: float,
    show_ylabels: bool,
) -> None:
    y_positions = {
        method: len(METHODS) - 1 - index
        for index, method in enumerate(METHODS)
    }
    response_offsets = {"Responder": 0.10, "Non-responder": -0.10}
    text_offsets = {"Responder": 10, "Non-responder": -13}
    vertical_alignment = {"Responder": "bottom", "Non-responder": "top"}
    for method in METHODS:
        method_rows = metrics[metrics["method"].eq(method)]
        if len(method_rows) != 2:
            raise RuntimeError(f"Expected two response rows for {method}")
        for response in RESPONSES:
            row = method_rows[method_rows["response"].eq(response)]
            if len(row) != 1:
                raise RuntimeError(f"Missing {method}/{response}")
            value = float(row.iloc[0][metric])
            y_value = y_positions[method] + response_offsets[response]
            label_offset = text_offsets[response]
            label_alignment = vertical_alignment[response]
            if method == "AgentVC_proxy" and response == "Non-responder":
                label_offset = 9
                label_alignment = "bottom"
            ax.scatter(
                value,
                y_value,
                s=42,
                marker=RESPONSE_MARKERS[response],
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.65,
                zorder=3,
            )
            ax.annotate(
                f"{value:.3f}",
                xy=(value, y_value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va=label_alignment,
                fontsize=6.1,
                color="#3F4447",
            )
    ax.axvline(
        perfect_value,
        color="#73797D",
        lw=0.9,
        ls=(0, (3, 2)),
        zorder=1,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.56, 3.58)
    ax.set_xlabel(xlabel, fontsize=6.8)
    ax.set_yticks([y_positions[method] for method in METHODS])
    if show_ylabels:
        ax.set_yticklabels(
            [METHOD_LABELS[method] for method in METHODS],
            fontsize=6.8,
        )
    else:
        ax.set_yticklabels([])
    ax.grid(axis="x", color="#E4E7E9", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#777D80")
    ax.tick_params(axis="x", labelsize=6.2)
    ax.tick_params(axis="y", length=0)


def draw_figure(
    v3,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    metrics: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    limits = v3.normalized_limits(coordinates)
    fig = plt.figure(figsize=(15.4, 9.55), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.985,
        bottom=0.225,
        top=0.855,
        wspace=0.15,
        hspace=0.22,
    )
    umap_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    specifications = (
        ("A", "Observed Post reference", None),
        ("B", "scGen", "scGen"),
        ("C", "WOT* (target-derived)", "WOT_weighted"),
        ("D", "CellRank\u2020 proxy", "CellRank_terminal_proxy"),
        ("E", "AgentVC proxy", "AgentVC_proxy"),
    )
    for ax, (letter, title, method) in zip(umap_axes, specifications):
        v3.plot_umap_panel(
            ax,
            metadata,
            coordinates,
            limits,
            letter,
            title,
            method,
        )

    metric_grid = grid[1, 2].subgridspec(
        2,
        2,
        height_ratios=(0.18, 0.82),
        width_ratios=(1.08, 1.0),
        wspace=0.28,
        hspace=0.01,
    )
    header_ax = fig.add_subplot(metric_grid[0, :])
    header_ax.set_axis_off()
    header_ax.text(
        0.0,
        0.83,
        "F. Post-treatment expression fidelity",
        transform=header_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
    )
    response_handles = [
        Line2D(
            [0],
            [0],
            marker=RESPONSE_MARKERS[response],
            linestyle="none",
            markerfacecolor="#73787B",
            markeredgecolor="white",
            markersize=5.5,
            label=response,
        )
        for response in RESPONSES
    ]
    header_ax.legend(
        handles=response_handles,
        frameon=False,
        fontsize=6.2,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0,
    )
    ccc_ax = fig.add_subplot(metric_grid[1, 0])
    rmse_ax = fig.add_subplot(metric_grid[1, 1])
    plot_response_metric_axis(
        ccc_ax, metrics, "lins_ccc", "CCC \u2191", (0.0, 1.02), 1.0, True
    )
    plot_response_metric_axis(
        rmse_ax,
        metrics,
        "rmse_log1p_tpm",
        "RMSE \u2193",
        (-0.025, 0.9),
        0.0,
        False,
    )

    population_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=v3.POPULATION_COLORS[population],
            markeredgecolor="none",
            markersize=6.0,
            label=f"{index}  {population}",
        )
        for index, population in enumerate(v3.FINE_POPULATIONS, start=1)
    ]
    fig.legend(
        handles=population_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=4,
        frameon=False,
        fontsize=7.1,
        columnspacing=1.25,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Cell-expanded gene-expression accuracy across immune populations",
        y=0.975,
        fontsize=18.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.936,
        (
            "GSE120575 \u00b7 7,858 Observed Post cells \u00b7 "
            "1,664 deterministic proxy cells per method \u00b7 834 genes"
        ),
        ha="center",
        fontsize=9.6,
        color="#4D5357",
    )
    fig.text(
        0.5,
        0.030,
        (
            "Post-hoc cell-expanded sample-level expression proxies; not native "
            "single-cell predictions. Population labels are inherited from real "
            "source Pre cells.\n"
            "Panel F reuses the frozen response-level sample-expression CCC and "
            "RMSE values from the 834-gene benchmark. "
            "WOT* is target-derived."
        ),
        ha="center",
        va="bottom",
        fontsize=7.15,
        color="#3F4447",
    )
    fig.savefig(PREVIEW, dpi=150, facecolor="white")
    fig.savefig(PNG_600, dpi=600, facecolor="white")
    fig.savefig(PDF, facecolor="white")
    fig.savefig(SVG, facecolor="white")
    plt.close(fig)


def refresh_manifest() -> None:
    files = sorted(
        path
        for path in OUT.iterdir()
        if path.is_file() and path.name != "output_manifest.json"
    )
    write_json(
        OUT / "output_manifest.json",
        {
            "version": "GSE120575_RESPONSE_METRICS_UMAP_V4_2",
            "status": "visual-only Panel F response-metric substitution",
            "hash_algorithm": "SHA256",
            "files": [
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
            "frozen_metric_source": str(FROZEN_METRICS.relative_to(ROOT)),
            "frozen_umap_source": str(V3_H5AD.relative_to(ROOT)),
        },
    )


def build(rebuild_current: bool = False) -> None:
    if OUT.exists() and any(OUT.iterdir()):
        if not rebuild_current:
            raise FileExistsError(f"Refusing to overwrite V4.2: {OUT}")
        audit_path = OUT / "figure_audit.json"
        build_path = OUT / "build_state.json"
        audit_status = (
            json.loads(audit_path.read_text()).get("status")
            if audit_path.exists()
            else None
        )
        build_status = (
            json.loads(build_path.read_text()).get("status")
            if build_path.exists()
            else None
        )
        if (
            audit_status
            != "PASS_NUMERIC_IDENTITY_PENDING_VISUAL_INSPECTION"
            and build_status != "BUILDING"
        ):
            raise RuntimeError("Current V4.2 is not safe to rebuild")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "build_state.json",
        {"version": "GSE120575_RESPONSE_METRICS_UMAP_V4_2", "status": "BUILDING"},
    )
    hashes = validate_sources()
    metadata, coordinates, metrics = load_inputs()
    v3 = load_module(V3_SCRIPT, "gse120575_v3_frozen_for_v4_2")
    draw_figure(v3, metadata, coordinates, metrics)
    metrics.to_csv(
        OUT / "response_level_expression_accuracy_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    shutil.copy2(
        __file__, OUT / "run_GSE120575_response_metrics_UMAP_v4_2.py"
    )
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# GSE120575 response-level expression metrics UMAP V4.2",
                "",
                "This revision changes only the data displayed in Panel F.",
                "Panels A-E, the Panel F layout, title, colors and legend are unchanged.",
                "",
                "Panel F now displays the exact frozen response-level CCC/RMSE",
                "values used by the earlier 2x4 gene-expression scatter figure.",
                "Each method has one Responder and one Non-responder value.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "caption.txt").write_text(
        "Panels A-E reuse the frozen V3 UMAP coordinates. Panel F displays "
        "the exact frozen response-level sample-expression CCC and RMSE values "
        "from the earlier 834-gene benchmark: circles indicate Responder and "
        "triangles Non-responder. No expression, coordinate or metric was "
        "recomputed. WOT* is target-derived.\n",
        encoding="utf-8",
    )
    write_json(
        OUT / "visual_inspection.json",
        {
            "status": "PENDING_MANUAL_VISUAL_INSPECTION",
            "figures": [PREVIEW.name, PNG_600.name, PDF.name, SVG.name],
        },
    )
    write_json(
        OUT / "figure_audit.json",
        {
            "status": "PASS_NUMERIC_IDENTITY_PENDING_VISUAL_INSPECTION",
            "final_expected_status": "PASS_V4_2_FROZEN_RESPONSE_METRICS",
            "source_hashes": hashes,
            "umap_coordinate_change": 0.0,
            "panels_a_to_e_changed": False,
            "panel_f_layout_changed_from_v4_1": False,
            "panel_f_values": "exact frozen unified_expression_metrics.csv",
            "panel_f_rows": 8,
            "panel_f_responses_per_method": 2,
            "historical_v1_v2_v3_v4_v4_1_modified": False,
        },
    )
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_RESPONSE_METRICS_UMAP_V4_2",
            "status": "COMPLETE_PENDING_VISUAL_INSPECTION",
        },
    )
    refresh_manifest()
    print("GSE120575 RESPONSE METRICS UMAP V4.2")
    print("- A-E coordinate change: 0")
    print("- Panel F rows: 8")
    print("- Frozen CCC/RMSE identity: PASS")
    print("- Visual status: PENDING")


def mark_visual_pass() -> None:
    audit_path = OUT / "figure_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS_NUMERIC_IDENTITY_PENDING_VISUAL_INSPECTION":
        raise RuntimeError("V4.2 is not awaiting visual inspection")
    for path in (PREVIEW, PNG_600, PDF, SVG):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing figure: {path}")
    visual = {
        "status": "PASS_RESPONSE_METRICS_READABLE_NO_OVERLAP",
        "panel_f_title_readable": True,
        "response_legend_does_not_cover_points": True,
        "all_16_numeric_labels_readable": True,
        "method_labels_readable": True,
        "population_legend_readable": True,
        "outliers_hidden": False,
    }
    write_json(OUT / "visual_inspection.json", visual)
    audit["status"] = "PASS_V4_2_FROZEN_RESPONSE_METRICS"
    audit["visual_inspection"] = visual["status"]
    write_json(audit_path, audit)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_RESPONSE_METRICS_UMAP_V4_2",
            "status": "COMPLETE_FINAL_PASS",
        },
    )
    refresh_manifest()
    print("VISUAL INSPECTION FINALIZED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-visual-pass", action="store_true")
    parser.add_argument("--rebuild-current", action="store_true")
    arguments = parser.parse_args()
    if arguments.mark_visual_pass:
        mark_visual_pass()
    else:
        build(rebuild_current=arguments.rebuild_current)


if __name__ == "__main__":
    main()
