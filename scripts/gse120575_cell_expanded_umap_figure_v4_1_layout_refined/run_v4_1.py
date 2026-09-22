#!/usr/bin/env python3
"""Visual-only layout refinement of GSE120575 UMAP figure V4.

V4.1 reuses the frozen V3 UMAP coordinates and frozen V4 CCC/RMSE values.
It only changes Panel F spacing, legend placement, and title wording.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR", "/tmp/numba-gse120575-expression-accuracy-v4-1"
)
os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/matplotlib-gse120575-expression-accuracy-v4-1"
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
V4_OUT = BASE / "cell_expanded_umap_figure_v4_expression_accuracy"
OUT = BASE / "cell_expanded_umap_figure_v4_1_layout_refined"

V3_H5AD = V3_OUT / "unified_cell_expanded_expression_834genes.h5ad"
V3_SCRIPT = ROOT / "scripts/gse120575_cell_expanded_umap_figure_v3/run_v3.py"
V4_SCRIPT = (
    ROOT
    / "scripts/gse120575_cell_expanded_umap_figure_v4_expression_accuracy"
    / "run_v4.py"
)
V4_METRICS = V4_OUT / "direct_local_expression_accuracy_metrics.csv"
V4_SUMMARY = V4_OUT / "direct_local_expression_accuracy_summary.csv"
V4_AUDIT = V4_OUT / "analysis_audit.json"

EXPECTED_HASHES = {
    "v3_h5ad": "d3aa2ede49a58b9b7328b93ca6aa8938dfd7efdcf344cd858fc7faa2c0f90934",
    "v3_script": "ef81489f7ed7e29fce33e6ed3ae9af6cb4b60d06a8ad2e31607537c0e37ac279",
    "v4_script": "12ef34b128fee4655725ab415884c61093f8b89b811e3bd855c58f7c6db731b0",
    "v4_metrics": "db9f50a9ebaccef8f2f6c830d5ee14e7b80a6412eff7630a65653d907af7a61a",
    "v4_summary": "676d2727253ba4e1b31386337c9d0faa5794268fad979527105543303dd535e6",
    "v4_audit": "0da4f89403ff45265f923852ffdd72eb9a12a04e7a0ade9c060122420ecddd41",
}

METHODS = (
    "scGen",
    "WOT_weighted",
    "CellRank_terminal_proxy",
    "AgentVC_proxy",
)
RESPONSES = ("Responder", "Non-responder")
RESPONSE_MARKERS = {"Responder": "o", "Non-responder": "^"}

FIGURE_STEM = (
    "GSE120575_cell_expanded_direct_expression_accuracy_UMAP_v4_1"
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
        raise RuntimeError(f"Could not import frozen module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_sources() -> dict[str, str]:
    paths = {
        "v3_h5ad": V3_H5AD,
        "v3_script": V3_SCRIPT,
        "v4_script": V4_SCRIPT,
        "v4_metrics": V4_METRICS,
        "v4_summary": V4_SUMMARY,
        "v4_audit": V4_AUDIT,
    }
    hashes = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
        if hashes[name] != EXPECTED_HASHES[name]:
            raise RuntimeError(f"Frozen source changed: {name}")
    audit = json.loads(V4_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_POST_HOC_DIRECT_EXPRESSION_ACCURACY_UMAP":
        raise RuntimeError("V4 direct-expression analysis is not final PASS")
    return hashes


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    adata = ad.read_h5ad(V3_H5AD, backed="r")
    if adata.shape != (16178, 834):
        raise RuntimeError(f"Unexpected V3 H5AD shape: {adata.shape}")
    metadata = adata.obs.copy().reset_index(drop=True)
    coordinates = np.asarray(adata.obsm["X_umap"], dtype=float).copy()
    adata.file.close()
    metrics = pd.read_csv(V4_METRICS)
    if len(metrics) != 48:
        raise RuntimeError("Frozen V4 metrics are not 48 rows")
    if not metrics.groupby("method", observed=True).size().eq(12).all():
        raise RuntimeError("Frozen V4 metrics are not 12 rows per method")
    if not np.isfinite(
        metrics[["lins_ccc", "rmse_log1p_tpm"]].to_numpy(float)
    ).all():
        raise RuntimeError("Frozen V4 metrics are non-finite")
    return metadata, coordinates, metrics


def draw_figure(
    v3,
    v4,
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

    # A dedicated header row separates the title/legend from the scGen points.
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
    v4.plot_accuracy_axis(
        ccc_ax,
        metrics,
        "lins_ccc",
        "CCC \u2191",
        (0.0, 1.02),
        1.0,
        True,
    )
    v4.plot_accuracy_axis(
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
            "Panel F directly compares 834-gene response \u00d7 broad-cell-type "
            "centroids with Observed Post after equal biopsy weighting; "
            "CCC and RMSE are computed directly in log1p(TPM). "
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
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4_1",
            "status": "visual-only Panel F layout refinement",
            "hash_algorithm": "SHA256",
            "files": [
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
            "frozen_metric_source": str(V4_METRICS.relative_to(ROOT)),
            "frozen_umap_source": str(V3_H5AD.relative_to(ROOT)),
        },
    )


def build(rebuild_current: bool = False) -> None:
    if OUT.exists() and any(OUT.iterdir()):
        if not rebuild_current:
            raise FileExistsError(f"Refusing to overwrite V4.1: {OUT}")
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
            raise RuntimeError("Current V4.1 is not safe to rebuild")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4_1",
            "status": "BUILDING",
        },
    )
    hashes = validate_sources()
    metadata, coordinates, metrics = load_data()
    v3 = load_module(V3_SCRIPT, "gse120575_v3_frozen_for_v4_1")
    v4 = load_module(V4_SCRIPT, "gse120575_v4_frozen_for_v4_1")
    draw_figure(v3, v4, metadata, coordinates, metrics)

    shutil.copy2(
        V4_METRICS, OUT / "direct_local_expression_accuracy_metrics.csv"
    )
    shutil.copy2(
        V4_SUMMARY, OUT / "direct_local_expression_accuracy_summary.csv"
    )
    shutil.copy2(
        __file__, OUT / "run_GSE120575_expression_accuracy_UMAP_v4_1.py"
    )
    caption = (
        "Expression agreement with observed Post states. Panels A-E reuse the "
        "frozen V3 UMAP coordinates exactly. Panel F displays the unchanged V4 "
        "Lin's CCC and log1p(TPM) RMSE values after equal biopsy weighting. "
        "The V4.1 revision only separates the Panel F header, response legend "
        "and metric axes to prevent overlap. Expanded rows are post-hoc "
        "sample-level proxies rather than native single-cell predictions; "
        "population labels are source-inherited. WOT* is target-derived."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# GSE120575 expression-accuracy UMAP V4.1",
                "",
                "This is a visual-only refinement of the frozen V4 figure.",
                "",
                "- Panel F title: `Post-treatment expression fidelity`.",
                "- The response legend is isolated in a dedicated header row.",
                "- CCC and RMSE axes have wider separation.",
                "- The overlapping in-panel group-count text was removed.",
                "- V3 UMAP coordinates and V4 metrics are unchanged.",
                "",
                "No metric, expression value, coordinate or conclusion was changed.",
            ]
        )
        + "\n",
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
            "final_expected_status": (
                "PASS_V4_1_PANEL_F_LAYOUT_REFINEMENT_ONLY"
            ),
            "source_hashes": hashes,
            "umap_coordinate_change": 0.0,
            "metric_value_change": 0.0,
            "panel_title": "Post-treatment expression fidelity",
            "response_legend_location": "dedicated Panel F header row",
            "overlapping_group_count_text_removed": True,
            "historical_v1_v2_v3_v4_modified": False,
        },
    )
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4_1",
            "status": "COMPLETE_PENDING_VISUAL_INSPECTION",
        },
    )
    refresh_manifest()
    print("GSE120575 EXPRESSION ACCURACY UMAP V4.1")
    print("- UMAP coordinate change: 0")
    print("- CCC/RMSE value change: 0")
    print("- Panel F title: Post-treatment expression fidelity")
    print("- Numeric identity: PASS")
    print("- Visual status: PENDING")


def mark_visual_pass() -> None:
    audit_path = OUT / "figure_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status")
        != "PASS_NUMERIC_IDENTITY_PENDING_VISUAL_INSPECTION"
    ):
        raise RuntimeError("V4.1 is not awaiting visual inspection")
    for path in (PREVIEW, PNG_600, PDF, SVG):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing figure: {path}")
    visual = {
        "status": "PASS_PANEL_F_NO_OVERLAP",
        "panel_f_title_readable": True,
        "response_legend_does_not_cover_points": True,
        "ccc_rmse_median_labels_readable": True,
        "method_labels_readable": True,
        "population_legend_readable": True,
        "figure_footnote_readable": True,
        "outliers_hidden": False,
    }
    write_json(OUT / "visual_inspection.json", visual)
    audit["status"] = "PASS_V4_1_PANEL_F_LAYOUT_REFINEMENT_ONLY"
    audit["visual_inspection"] = visual["status"]
    write_json(audit_path, audit)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4_1",
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
