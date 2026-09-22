#!/usr/bin/env python3
"""Render a clean, publication-style GSE120575 heterogeneity figure.

This script reads the frozen V1 plotting tables only. It does not rerun any
model, bootstrap, PCA, cell expansion, or scientific metric.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse120575-heterogeneity-clean-v2")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("GSE120575_HETEROGENEITY_OUT", str(ROOT / "outputs/GSE120575_cell_expanded_heterogeneity_audit_v1"))).resolve()
OUT = Path(os.environ.get("GSE120575_HETEROGENEITY_FIGURE_OUT", str(ROOT / "outputs/GSE120575_cell_expanded_heterogeneity_audit_v2_clean_figure"))).resolve()
STEM = "GSE120575_within_cell_type_expression_heterogeneity_recovery"

METHODS = ("scGen", "CellRank_terminal_proxy", "AgentVC_proxy")
METHOD_LABELS = {
    "scGen": "scGen",
    "CellRank_terminal_proxy": "CellRank",
    "AgentVC_proxy": "Leca-VC",
}
METHOD_COLORS = {
    "scGen": "#E69F00",
    "CellRank_terminal_proxy": "#56B4E9",
    "AgentVC_proxy": "#CC79A7",
}
RESPONSES = ("Responder", "Non-responder")
CELL_TYPES = ("B cell", "Monocyte/Macrophage", "NK cell", "T cell")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_frozen_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = {
        "dispersion_stratum_summary": SOURCE / "dispersion_stratum_summary.csv",
        "energy_distance_by_stratum": SOURCE / "energy_distance_by_stratum.csv",
        "overall_metric_summary": SOURCE / "overall_metric_summary.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: sha256(path) for name, path in paths.items()}
    dispersion = pd.read_csv(paths["dispersion_stratum_summary"])
    energy = pd.read_csv(paths["energy_distance_by_stratum"])
    summary = pd.read_csv(paths["overall_metric_summary"])
    if len(dispersion) != 24 or len(energy) != 24 or len(summary) != 6:
        raise RuntimeError(
            f"Unexpected frozen-table dimensions: {len(dispersion)}, {len(energy)}, {len(summary)}"
        )
    return dispersion, energy, summary, hashes


def render(
    stratum_summary: pd.DataFrame,
    energy: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, object]:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gse120575-heterogeneity-clean-v2",
        }
    )
    fig = plt.figure(figsize=(13.6, 5.55), constrained_layout=False, facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.14, 1.50],
        left=0.072,
        right=0.987,
        bottom=0.145,
        top=0.815,
        wspace=0.245,
    )
    ax_heat = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(1, 2, wspace=0.34)
    ax_disp = fig.add_subplot(right[0, 0])
    ax_energy = fig.add_subplot(right[0, 1])

    row_pairs = [(response, cell_type) for response in RESPONSES for cell_type in CELL_TYPES]
    heat = np.empty((len(row_pairs), len(METHODS)), dtype=float)
    for row_index, (response, cell_type) in enumerate(row_pairs):
        for column_index, method in enumerate(METHODS):
            local = stratum_summary[
                stratum_summary["response"].eq(response)
                & stratum_summary["broad_cell_type"].eq(cell_type)
                & stratum_summary["method"].eq(method)
            ]
            if len(local) != 1:
                raise RuntimeError("Heatmap source row is not unique")
            heat[row_index, column_index] = float(local.iloc[0]["median_log2_sd_ratio"])

    color_limit = max(0.5, math.ceil(float(np.max(np.abs(heat))) / 0.25) * 0.25)
    image = ax_heat.imshow(
        heat,
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="auto",
    )
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            color = "white" if abs(heat[row, column]) > 0.58 * color_limit else "#202020"
            ax_heat.text(
                column,
                row,
                f"{heat[row, column]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    ax_heat.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    row_labels = [
        f"{'R' if response == 'Responder' else 'NR'}  ·  "
        f"{cell_type.replace('Monocyte/Macrophage', 'Mono/Mac')}"
        for response, cell_type in row_pairs
    ]
    ax_heat.set_yticks(range(len(row_labels)), row_labels)
    ax_heat.axhline(3.5, color="white", linewidth=2.5)
    ax_heat.set_title("A   Within-cell-type dispersion ratio", loc="left", fontweight="bold", pad=10)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.048, pad=0.035)
    colorbar.set_label(r"Median log$_2$(predicted SD / observed SD)")
    colorbar.ax.axhline(colorbar.norm(0), color="black", linewidth=0.7)

    metric_axes = (
        (
            ax_disp,
            "median_absolute_log2_sd_ratio",
            "B   Dispersion error",
            r"Median |log$_2$ SD ratio|",
            stratum_summary,
            "median_absolute_log2_sd_ratio",
        ),
        (
            ax_energy,
            "normalized_energy_distance",
            "Distribution distance",
            "Normalized energy distance",
            energy,
            "normalized_energy_distance",
        ),
    )
    jitter = np.linspace(-0.09, 0.09, len(row_pairs))
    for axis, metric, title, ylabel, source, source_column in metric_axes:
        for method_index, method in enumerate(METHODS):
            local_points = []
            for response, cell_type in row_pairs:
                local = source[
                    source["method"].eq(method)
                    & source["response"].eq(response)
                    & source["broad_cell_type"].eq(cell_type)
                ]
                if len(local) != 1:
                    raise RuntimeError("Summary-plot source row is not unique")
                local_points.append(float(local.iloc[0][source_column]))
            axis.scatter(
                method_index + jitter,
                local_points,
                s=25,
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.68,
                zorder=2,
            )
            local_summary = summary[
                summary["method"].eq(method) & summary["metric"].eq(metric)
            ]
            if len(local_summary) != 1:
                raise RuntimeError("Overall summary row is not unique")
            record = local_summary.iloc[0]
            point = float(record["bootstrap_median"])
            lower = float(record["ci_2_5"])
            upper = float(record["ci_97_5"])
            axis.errorbar(
                method_index,
                point,
                yerr=np.asarray([[point - lower], [upper - point]]),
                fmt="D",
                markersize=7.5,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="#222222",
                markeredgewidth=0.7,
                ecolor="#222222",
                elinewidth=1.4,
                capsize=3.2,
                zorder=4,
            )
        axis.set_xticks(
            range(len(METHODS)),
            [METHOD_LABELS[method] for method in METHODS],
            rotation=22,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold", pad=10, loc="left" if axis is ax_disp else "center")
        axis.set_xlim(-0.35, len(METHODS) - 0.65)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#dddddd", linewidth=0.75, alpha=0.85)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "GSE120575 Within-Cell-Type Expression Heterogeneity Recovery",
        fontsize=15.5,
        fontweight="bold",
        y=0.955,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    preview = OUT / f"{STEM}_preview.png"
    png = OUT / f"{STEM}_600dpi.png"
    pdf = OUT / f"{STEM}_vector.pdf"
    svg = OUT / f"{STEM}.svg"
    pdf_metadata = {
        "Creator": "GSE120575 heterogeneity clean figure V2",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(preview, dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(
        svg,
        facecolor="white",
        bbox_inches="tight",
        metadata={
            "Creator": "GSE120575 heterogeneity clean figure V2",
            "Date": "2026-09-07",
        },
    )
    plt.close(fig)
    return {
        "preview": preview,
        "png_600dpi": png,
        "vector_pdf": pdf,
        "svg": svg,
        "heatmap_color_limit": color_limit,
        "figure_size_inches": [13.6, 5.55],
    }


def main() -> None:
    dispersion, energy, summary, source_hashes = load_frozen_tables()
    outputs = render(dispersion, energy, summary)

    caption = (
        "GSE120575 within-cell-type expression heterogeneity recovery. "
        "(A) Median gene-wise log2 ratio of predicted to observed Post within-biopsy standard deviation for "
        "scGen, CellRank and Leca-VC, stratified by response and broad cell type. Zero denotes matched dispersion; "
        "negative and positive values denote under- and over-dispersion, respectively. R and NR denote responder "
        "and non-responder. (B) Across-stratum summaries of the median absolute log2 dispersion ratio and normalized "
        "energy distance in the frozen 50-dimensional Observed-reference PCA space. Small points denote the eight "
        "response-by-cell-type strata; diamonds and whiskers denote bootstrap medians and 95% biological-unit "
        "cluster-bootstrap confidence intervals (1,000 replicates). B cell, Monocyte/Macrophage, NK cell and T cell "
        "passed the prespecified coverage gate. All proxy cells were deterministically expanded from frozen "
        "sample-level expression predictions, and their labels were inherited from source Pre cells. This analysis "
        "therefore evaluates cell-expanded proxy distribution fidelity, not native single-cell transcriptome generation."
    )
    (OUT / "FIGURE_CAPTION.txt").write_text(caption + "\n", encoding="utf-8")

    audit = {
        "status": "PASS_VISUALIZATION_ONLY_CLEAN_FIGURE_V2",
        "scientific_metrics_recomputed": False,
        "models_or_bootstrap_rerun": False,
        "source_directory": str(SOURCE),
        "source_table_sha256": source_hashes,
        "visual_changes": [
            "shortened title",
            "removed across-stratum header",
            "removed lower-is-better annotations",
            "removed heatmap x-axis descriptor",
            "moved explanatory notes and symbol definitions to the caption",
            "reduced bottom whitespace and enlarged plotting area",
        ],
        "unchanged": [
            "heatmap values and color scale",
            "stratum points",
            "bootstrap medians",
            "95% cluster-bootstrap confidence intervals",
            "method colors",
        ],
        "figure": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in outputs.items()
        },
    }
    write_json(OUT / "figure_audit.json", audit)

    output_files = [outputs[key] for key in ("preview", "png_600dpi", "vector_pdf", "svg")]
    manifest = {
        "status": audit["status"],
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        ],
    }
    write_json(OUT / "output_manifest.json", manifest)
    write_json(
        OUT / "COMPLETE.json",
        {
            "status": audit["status"],
            "main_figure": outputs["png_600dpi"].name,
            "evidence": "figure_audit.json",
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
