#!/usr/bin/env python3
"""Render two standalone supplementary heatmap candidates from frozen CSVs.

This script is visualization-only.  It never refits a model or recomputes a
benchmark metric.  Historical source identifiers are retained in source-data
tables, while the publication display name is standardized to ``Leca-AC``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "Supplementary_standalone_heatmaps_final"
SOURCE_OUT = OUT / "source_data"

CCI_SOURCE = (
    ROOT
    / "outputs/GSE267904_multimethod_cci_benchmark_v2/07_figure/source_data/"
    "panels_A-F_network_heatmaps.csv"
)
CCI_AUDIT = (
    ROOT
    / "outputs/GSE267904_multimethod_cci_benchmark_v2/08_audits/"
    "final_validation.json"
)
BULK_SOURCE = (
    ROOT
    / "outputs/GSE2565_bulk_master_comparison_v5_3_formal_dnb_display/"
    "plot_input_tables/panel_E_functional_program_heatmaps.csv"
)
BULK_AUDIT = (
    ROOT
    / "outputs/GSE2565_bulk_master_comparison_v5_3_formal_dnb_display/"
    "FINAL_AUDIT.json"
)

DISPLAY_NAME = "Leca-AC"

CCI_SOURCES = [
    "observed",
    "scgpt",
    "scgen",
    "wot_external",
    "cellrank_proxy",
    "worker",
]
CCI_TITLES = {
    "observed": "Observed",
    "scgpt": "scGPT",
    "scgen": "scGen",
    "wot_external": "WOT",
    "cellrank_proxy": "CellRank",
    "worker": DISPLAY_NAME,
}
CCI_COLORS = {
    "observed": "#333333",
    "scgpt": "#0072B2",
    "scgen": "#E69F00",
    "wot_external": "#009E73",
    "cellrank_proxy": "#56B4E9",
    "worker": "#CC79A7",
}
CELL_TYPES = [
    "alveolar_epithelial_AT1_AT2",
    "activated_Krt8_ADI_epithelial",
    "airway_epithelial",
    "macrophage",
    "recruited_monocyte_macrophage",
    "fibroblast_myofibroblast",
    "endothelial",
    "dendritic",
    "lymphoid",
]
CELL_SHORT = {
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

BULK_METHODS = [
    "Real",
    "chronODE-M",
    "RVAgene",
    "BOP-DMD",
    "GPR",
    "AgentVC Online Agent pilot",
]
BULK_TITLES = {
    "Real": "Observed",
    "chronODE-M": "chronODE-M",
    "RVAgene": "RVAgene",
    "BOP-DMD": "BOP-DMD",
    "GPR": "GPR",
    "AgentVC Online Agent pilot": DISPLAY_NAME,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [CCI_SOURCE, CCI_AUDIT, BULK_SOURCE, BULK_AUDIT]:
        if not path.is_file():
            raise FileNotFoundError(path)

    cci_audit = json.loads(CCI_AUDIT.read_text(encoding="utf-8"))
    if cci_audit.get("status") != "PASS_EXPANDED_MULTIMETHOD_CCI_WORKER_LEVEL":
        raise RuntimeError("The frozen GSE267904 V2 benchmark is not PASS")
    bulk_audit = json.loads(BULK_AUDIT.read_text(encoding="utf-8"))
    if bulk_audit.get("status") != "PASS_V5_3_FORMAL_DNB_DISPLAY_CORRECTION":
        raise RuntimeError("The frozen GSE2565 V5.3 source is not PASS")

    cci = pd.read_csv(CCI_SOURCE)
    if cci.shape != (486, 7) or set(cci["source"]) != set(CCI_SOURCES):
        raise RuntimeError("Unexpected CCI heatmap source contract")
    counts = cci.groupby("source").size()
    if not (counts == 81).all() or not (cci["n"] == 10).all():
        raise RuntimeError("CCI heatmap summaries are incomplete")
    if not np.isfinite(cci[["median", "minimum", "maximum"]]).all().all():
        raise RuntimeError("CCI heatmap source contains non-finite values")

    bulk = pd.read_csv(BULK_SOURCE)
    if bulk.shape != (540, 5) or set(bulk["method"]) != set(BULK_METHODS):
        raise RuntimeError("Unexpected bulk heatmap source contract")
    if not (bulk.groupby("method").size() == 90).all():
        raise RuntimeError("Bulk functional-program heatmaps are incomplete")
    if not np.isfinite(bulk["temporal_z_score"]).all():
        raise RuntimeError("Bulk heatmap source contains non-finite values")
    return cci, bulk


def cci_matrix(frame: pd.DataFrame, source: str) -> np.ndarray:
    table = frame.loc[frame["source"].eq(source)].pivot(
        index="sender", columns="receiver", values="median"
    )
    matrix = table.reindex(index=CELL_TYPES, columns=CELL_TYPES).to_numpy(float)
    if matrix.shape != (9, 9) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Incomplete CCI matrix: {source}")
    return matrix


def plot_cci(frame: pd.DataFrame) -> dict[str, Path]:
    matrices = {source: cci_matrix(frame, source) for source in CCI_SOURCES}
    vmax = max(float(x.max()) for x in matrices.values())
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.925, bottom=0.10, top=0.89, hspace=0.44, wspace=0.30)
    image = None
    for index, (ax, source) in enumerate(zip(axes.ravel(), CCI_SOURCES)):
        image = ax.imshow(
            matrices[source], cmap="YlGnBu", vmin=0, vmax=vmax,
            interpolation="nearest", aspect="equal"
        )
        ax.set_xticks(
            np.arange(9), [CELL_SHORT[x] for x in CELL_TYPES],
            rotation=48, ha="right", fontsize=7.4,
        )
        ax.set_yticks(np.arange(9), [CELL_SHORT[x] for x in CELL_TYPES], fontsize=7.4)
        ax.set_xlabel("Receiver cell type", fontsize=8.2)
        ax.set_ylabel("Sender cell type", fontsize=8.2)
        ax.set_title(CCI_TITLES[source], fontsize=12, fontweight="bold", color=CCI_COLORS[source], pad=7)
        ax.text(-0.12, 1.08, chr(65 + index), transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        for spine in ax.spines.values():
            spine.set_color(CCI_COLORS[source])
            spine.set_linewidth(1.35)
    cax = fig.add_axes([0.945, 0.19, 0.013, 0.61])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Median normalized CCI edge weight", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        "GSE267904 cell–cell communication network reconstruction",
        fontsize=18, fontweight="bold", y=0.965,
    )
    stem = OUT / "GSE267904_CCI_network_heatmaps_Leca_AC"
    outputs = {
        "preview_png": stem.with_name(stem.name + "_preview.png"),
        "png_600dpi": stem.with_name(stem.name + "_600dpi.png"),
        "pdf": stem.with_name(stem.name + "_vector.pdf"),
        "svg": stem.with_name(stem.name + "_vector.svg"),
    }
    fig.savefig(outputs["preview_png"], dpi=180, facecolor="white")
    fig.savefig(outputs["png_600dpi"], dpi=600, facecolor="white")
    fig.savefig(outputs["pdf"], facecolor="white")
    fig.savefig(outputs["svg"], facecolor="white")
    plt.close(fig)
    return outputs


def bulk_matrix(frame: pd.DataFrame, method: str, modules: list[str], times: list[float]) -> np.ndarray:
    table = frame.loc[frame["method"].eq(method)].pivot(
        index="module", columns="time_hours", values="temporal_z_score"
    )
    matrix = table.reindex(index=modules, columns=times).to_numpy(float)
    if matrix.shape != (10, 9) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Incomplete bulk heatmap: {method}")
    return matrix


def pretty_module(label: str) -> str:
    replacements = {
        "oxidative_stress": "Oxidative stress",
        "inflammation": "Inflammation",
        "NFkB_TNF": "NFκB/TNF",
        "apoptosis_cell_death": "Apoptosis/cell death",
        "hypoxia": "Hypoxia",
        "alveolar_epithelial_injury_barrier": "Alveolar injury/barrier",
        "vascular_permeability_edema_proxy": "Vascular permeability/edema",
        "immune_recruitment": "Immune recruitment",
        "tissue_repair": "Tissue repair",
        "ECM_remodeling": "ECM remodeling",
    }
    return replacements.get(label, label.replace("_", " "))


def plot_bulk(frame: pd.DataFrame) -> dict[str, Path]:
    modules = frame.loc[frame["method"].eq("Real"), "module"].drop_duplicates().tolist()
    times = sorted(frame["time_hours"].unique().astype(float).tolist())
    matrices = {method: bulk_matrix(frame, method, modules, times) for method in BULK_METHODS}
    vmax = max(float(np.abs(x).max()) for x in matrices.values())
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.7), facecolor="white")
    fig.subplots_adjust(left=0.11, right=0.93, bottom=0.10, top=0.88, hspace=0.34, wspace=0.20)
    image = None
    for index, (ax, method) in enumerate(zip(axes.ravel(), BULK_METHODS)):
        image = ax.imshow(
            matrices[method], aspect="auto", cmap="coolwarm",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.set_xticks(range(len(times)), [f"{x:g}" for x in times], fontsize=7.5)
        ax.set_xlabel("Time (h)", fontsize=8.2)
        if index % 3 == 0:
            ax.set_yticks(range(len(modules)), [pretty_module(x) for x in modules], fontsize=7.1)
        else:
            ax.set_yticks(range(len(modules)), [])
        ax.set_title(BULK_TITLES[method], fontsize=11.5, fontweight="bold", pad=6)
        ax.text(-0.12, 1.08, chr(65 + index), transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    cax = fig.add_axes([0.947, 0.18, 0.014, 0.63])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Temporal z-score", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        "GSE2565 functional-program response profiles",
        fontsize=18, fontweight="bold", y=0.96,
    )
    stem = OUT / "GSE2565_functional_program_heatmaps_Leca_AC"
    outputs = {
        "preview_png": stem.with_name(stem.name + "_preview.png"),
        "png_600dpi": stem.with_name(stem.name + "_600dpi.png"),
        "pdf": stem.with_name(stem.name + "_vector.pdf"),
        "svg": stem.with_name(stem.name + "_vector.svg"),
    }
    fig.savefig(outputs["preview_png"], dpi=180, facecolor="white")
    fig.savefig(outputs["png_600dpi"], dpi=600, facecolor="white")
    fig.savefig(outputs["pdf"], facecolor="white")
    fig.savefig(outputs["svg"], facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })
    cci, bulk = validate_inputs()

    cci_display = cci.copy()
    cci_display["display_method"] = cci_display["source"].map(CCI_TITLES)
    cci_display.to_csv(SOURCE_OUT / "GSE267904_CCI_network_heatmaps_source_data.csv", index=False)
    bulk_display = bulk.copy()
    bulk_display["display_method"] = bulk_display["method"].map(BULK_TITLES)
    bulk_display.to_csv(SOURCE_OUT / "GSE2565_functional_program_heatmaps_source_data.csv", index=False)

    cci_outputs = plot_cci(cci)
    bulk_outputs = plot_bulk(bulk)
    all_outputs = {"cci_" + k: v for k, v in cci_outputs.items()} | {"bulk_" + k: v for k, v in bulk_outputs.items()}
    manifest = {
        "status": "PASS_SUPPLEMENTARY_STANDALONE_HEATMAPS_FINAL",
        "analysis_type": "visualization-only replot of frozen heatmap source data",
        "display_name": DISPLAY_NAME,
        "model_reruns": 0,
        "llm_calls": 0,
        "physicell_reruns": 0,
        "cci_contract": {"rows": len(cci), "sources": 6, "edges_per_source": 81, "mappings_per_edge": 10},
        "bulk_contract": {"rows": len(bulk), "methods": 6, "modules": 10, "times": 9},
        "input_sha256": {
            str(CCI_SOURCE.relative_to(ROOT)): sha256(CCI_SOURCE),
            str(CCI_AUDIT.relative_to(ROOT)): sha256(CCI_AUDIT),
            str(BULK_SOURCE.relative_to(ROOT)): sha256(BULK_SOURCE),
            str(BULK_AUDIT.relative_to(ROOT)): sha256(BULK_AUDIT),
        },
        "output_sha256": {key: sha256(path) for key, path in all_outputs.items()},
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "caption_CCI.txt").write_text(
        "GSE267904 cell–cell communication network reconstruction. Six 9×9 heatmaps show median normalized CCI edge weights across ten frozen mappings. All methods use the frozen 58-pair COMMOT protocol. This worker-level comparison lacks passed Visium pseudo-spot confirmation; mappings are not biological replicates.\n",
        encoding="utf-8",
    )
    (OUT / "caption_bulk.txt").write_text(
        "GSE2565 functional-program response profiles. Temporal z-scores for ten frozen functional modules are shown for Observed and five reconstruction methods across nine time points. Method input regimes are not equivalent, and the heatmaps are descriptive rather than an overall performance score.\n",
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "# Standalone supplementary heatmap candidates\n\n"
        "Two heatmap-only figures were redrawn from frozen panel-level CSVs. No model or metric was rerun. Publication display name: `Leca-AC`. Historical source identifiers remain in source-data CSVs for traceability.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
