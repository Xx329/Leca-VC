#!/usr/bin/env python3
"""Create a landscape GSE120575 expression/heterogeneity composite.

The frozen UMAP/expression figure is placed on the left.  The frozen
heterogeneity source tables are re-rendered in a compact layout on the right:
dispersion heatmap on the left and the two summary metrics stacked vertically.
Only layout and panel letters are changed; scientific values are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
UMAP_DIR = ROOT / "source_data/fig4/components"
UMAP_PDF = UMAP_DIR / "GSE120575_cell_expanded_expression_accuracy_UMAP_deidentified_v1_vector.pdf"
UMAP_PNG = UMAP_DIR / "GSE120575_cell_expanded_expression_accuracy_UMAP_deidentified_v1_600dpi.png"
HET_SOURCE = ROOT / "source_data/fig4"
OUT = ROOT / "build/figures/fig4"

METHODS = ("scGen", "CellRank_terminal_proxy", "AgentVC_proxy")
METHOD_LABELS = {"scGen": "scGen", "CellRank_terminal_proxy": "CellRank", "AgentVC_proxy": "Leca-VC"}
METHOD_COLORS = {"scGen": "#E69F00", "CellRank_terminal_proxy": "#56B4E9", "AgentVC_proxy": "#CC79A7"}
RESPONSES = ("Responder", "Non-responder")
CELL_TYPES = ("B cell", "Monocyte/Macrophage", "NK cell", "T cell")

UMAP_WIDTH_PT = 1108.8
UMAP_HEIGHT_PT = 687.6
COMPACT_WIDTH_IN = 8.6
COMPACT_HEIGHT_IN = 7.0
GAP_PT = 20.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_tables():
    paths = {
        "dispersion": HET_SOURCE / "dispersion_stratum_summary.csv",
        "energy": HET_SOURCE / "energy_distance_by_stratum.csv",
        "summary": HET_SOURCE / "overall_metric_summary.csv",
    }
    missing = [str(path) for path in (UMAP_PDF, UMAP_PNG, *paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")
    dispersion = pd.read_csv(paths["dispersion"])
    energy = pd.read_csv(paths["energy"])
    summary = pd.read_csv(paths["summary"])
    if (len(dispersion), len(energy), len(summary)) != (24, 24, 6):
        raise RuntimeError("Unexpected frozen heterogeneity table dimensions")
    return dispersion, energy, summary, paths


def render_compact_heterogeneity(dispersion, energy, summary, component_dir: Path):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gse120575-horizontal-compact-heterogeneity-v1",
        }
    )
    fig = plt.figure(figsize=(COMPACT_WIDTH_IN, COMPACT_HEIGHT_IN), facecolor="white")
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.18, 0.92],
        left=0.135, right=0.975, bottom=0.12, top=0.86, wspace=0.42,
    )
    ax_heat = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, hspace=0.50)
    ax_disp = fig.add_subplot(right[0, 0])
    ax_energy = fig.add_subplot(right[1, 0])

    row_pairs = [(response, cell_type) for response in RESPONSES for cell_type in CELL_TYPES]
    heat = np.empty((len(row_pairs), len(METHODS)), dtype=float)
    for row_index, (response, cell_type) in enumerate(row_pairs):
        for column_index, method in enumerate(METHODS):
            local = dispersion[
                dispersion["response"].eq(response)
                & dispersion["broad_cell_type"].eq(cell_type)
                & dispersion["method"].eq(method)
            ]
            if len(local) != 1:
                raise RuntimeError("Non-unique frozen heatmap row")
            heat[row_index, column_index] = float(local.iloc[0]["median_log2_sd_ratio"])

    limit = max(0.5, math.ceil(float(np.max(np.abs(heat))) / 0.25) * 0.25)
    image = ax_heat.imshow(heat, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            color = "white" if abs(heat[row, column]) > 0.58 * limit else "#202020"
            ax_heat.text(column, row, f"{heat[row, column]:.2f}", ha="center", va="center", color=color, fontsize=8.2)
    ax_heat.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    row_labels = [
        f"{'R' if response == 'Responder' else 'NR'} · {cell_type.replace('Monocyte/Macrophage', 'Mono/Mac')}"
        for response, cell_type in row_pairs
    ]
    ax_heat.set_yticks(range(len(row_labels)), row_labels)
    ax_heat.axhline(3.5, color="white", linewidth=2.3)
    ax_heat.set_title("G   Dispersion ratio", loc="left", fontweight="bold", pad=8)
    colorbar = fig.colorbar(image, ax=ax_heat, orientation="horizontal", fraction=0.046, pad=0.075)
    colorbar.set_label(r"Median log$_2$(predicted SD / observed SD)", fontsize=8.0)
    colorbar.ax.tick_params(labelsize=7.4)

    specifications = (
        (ax_disp, dispersion, "median_absolute_log2_sd_ratio", "median_absolute_log2_sd_ratio", "H   Dispersion error", r"Median |log$_2$ SD ratio|"),
        (ax_energy, energy, "normalized_energy_distance", "normalized_energy_distance", "I   Distribution distance", "Normalized energy distance"),
    )
    jitter = np.linspace(-0.08, 0.08, len(row_pairs))
    for axis, source, source_column, metric, title, ylabel in specifications:
        for method_index, method in enumerate(METHODS):
            points = []
            for response, cell_type in row_pairs:
                local = source[
                    source["method"].eq(method)
                    & source["response"].eq(response)
                    & source["broad_cell_type"].eq(cell_type)
                ]
                if len(local) != 1:
                    raise RuntimeError("Non-unique frozen metric row")
                points.append(float(local.iloc[0][source_column]))
            axis.scatter(
                method_index + jitter, points, s=20,
                facecolor=METHOD_COLORS[method], edgecolor="white", linewidth=0.45, alpha=0.70, zorder=2,
            )
            record = summary[summary["method"].eq(method) & summary["metric"].eq(metric)]
            if len(record) != 1:
                raise RuntimeError("Non-unique frozen bootstrap summary row")
            record = record.iloc[0]
            point = float(record["bootstrap_median"])
            lower = float(record["ci_2_5"])
            upper = float(record["ci_97_5"])
            axis.errorbar(
                method_index, point, yerr=np.asarray([[point - lower], [upper - point]]),
                fmt="D", markersize=6.5, markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="#222222", markeredgewidth=0.65,
                ecolor="#222222", elinewidth=1.15, capsize=2.8, zorder=4,
            )
        axis.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS], rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold", pad=7)
        axis.set_xlim(-0.34, len(METHODS) - 0.66)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.65, alpha=0.9)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Within-Cell-Type Expression Heterogeneity Recovery", fontsize=13.2, fontweight="bold", y=0.955)
    component_dir.mkdir(parents=True, exist_ok=True)
    pdf = component_dir / "compact_heterogeneity_vector.pdf"
    png = component_dir / "compact_heterogeneity_600dpi.png"
    fig.savefig(pdf, facecolor="white", metadata={"Creator": "Leca-VC horizontal composite", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=600, facecolor="white")
    plt.close(fig)
    return pdf, png, limit


def render_vector_composite(right_pdf: Path, output_pdf: Path):
    right_width_pt = COMPACT_WIDTH_IN * 72.0
    right_height_pt = COMPACT_HEIGHT_IN * 72.0
    right_scale = UMAP_HEIGHT_PT / right_height_pt
    scaled_right_width = right_width_pt * right_scale
    page_width = UMAP_WIDTH_PT + GAP_PT + scaled_right_width
    page_height = UMAP_HEIGHT_PT
    with tempfile.TemporaryDirectory(prefix="gse120575_horizontal_") as temp_name:
        temp = Path(temp_name)
        left_eps = temp / "left.eps"
        right_eps = temp / "right.eps"
        ps = temp / "composition.ps"
        for source, target in ((UMAP_PDF, left_eps), (right_pdf, right_eps)):
            subprocess.run(
                ["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=eps2write", "-dEPSCrop", f"-sOutputFile={target}", str(source)],
                check=True,
            )
        content = f"""%!PS-Adobe-3.0
%%BoundingBox: 0 0 {int(page_width + 0.999)} {int(page_height + 0.999)}
<< /PageSize [{page_width:.6f} {page_height:.6f}] >> setpagedevice
/BeginEPSF {{ /b4_Inc_state save def /dict_count countdictstack def /op_count count 1 sub def userdict begin /showpage {{}} def 0 setgray 0 setlinecap 1 setlinewidth 0 setlinejoin 10 setmiterlimit [] 0 setdash newpath }} bind def
/EndEPSF {{ count op_count sub {{pop}} repeat countdictstack dict_count sub {{end}} repeat b4_Inc_state restore }} bind def
gsave 0 0 translate BeginEPSF ({left_eps}) run EndEPSF grestore
gsave {UMAP_WIDTH_PT + GAP_PT:.6f} 0 translate {right_scale:.9f} {right_scale:.9f} scale BeginEPSF ({right_eps}) run EndEPSF grestore
showpage
%%EOF
"""
        ps.write_text(content)
        subprocess.run(
            ["gs", "-q", "-dBATCH", "-dNOPAUSE", "-dNOSAFER", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.7", "-dPDFSETTINGS=/prepress", f"-sOutputFile={output_pdf}", str(ps)],
            check=True,
        )
    return page_width, page_height


def render_raster_composite(right_png: Path, output: Path, preview: Path):
    with Image.open(UMAP_PNG).convert("RGB") as left_source, Image.open(right_png).convert("RGB") as right_source:
        target_height = left_source.height
        target_right_width = round(right_source.width * target_height / right_source.height)
        right = right_source.resize((target_right_width, target_height), Image.Resampling.LANCZOS)
        gap = round(GAP_PT / UMAP_WIDTH_PT * left_source.width)
        canvas = Image.new("RGB", (left_source.width + gap + target_right_width, target_height), "white")
        canvas.paste(left_source, (0, 0))
        canvas.paste(right, (left_source.width + gap, 0))
        canvas.save(output, dpi=(600, 600), optimize=True)
        preview_width = 3200
        preview_height = round(canvas.height * preview_width / canvas.width)
        preview_image = canvas.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        preview_image.save(preview, optimize=True)
        return canvas.size, preview_image.size


def main() -> int:
    dispersion, energy, summary, table_paths = load_tables()
    component_dir = OUT / "components"
    figure_dir = OUT / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    compact_pdf, compact_png, color_limit = render_compact_heterogeneity(dispersion, energy, summary, component_dir)
    vector_pdf = figure_dir / "GSE120575_expression_accuracy_and_heterogeneity_horizontal_vector.pdf"
    png_600 = figure_dir / "GSE120575_expression_accuracy_and_heterogeneity_horizontal_600dpi.png"
    preview = figure_dir / "GSE120575_expression_accuracy_and_heterogeneity_horizontal_preview.png"
    page_size = render_vector_composite(compact_pdf, vector_pdf)
    png_size, preview_size = render_raster_composite(compact_png, png_600, preview)

    (figure_dir / "caption.txt").write_text(
        (Path(__file__).resolve().parent / "caption.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source_hashes = {str(UMAP_PDF.relative_to(ROOT)): sha256(UMAP_PDF), str(UMAP_PNG.relative_to(ROOT)): sha256(UMAP_PNG)}
    source_hashes.update({str(path.relative_to(ROOT)): sha256(path) for path in table_paths.values()})
    audit = {
        "status": "PASS_GSE120575_HORIZONTAL_COMBINED_FIGURE_V1",
        "visualization_only": True,
        "layout": "landscape: frozen UMAP/expression block left; compact heterogeneity block right",
        "continuous_panel_letters": "A-I",
        "scientific_values_changed": False,
        "source_sha256": source_hashes,
        "heatmap_color_limit": color_limit,
        "vector_page_points": list(page_size),
        "png_600_dimensions": list(png_size),
        "preview_dimensions": list(preview_size),
    }
    write_json(OUT / "analysis_audit.json", audit)
    write_json(OUT / "COMPLETE.json", {"status": audit["status"], "figure_generated": True})
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
