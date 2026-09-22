#!/usr/bin/env python3
"""Mechanically assemble the frozen GSE267904 metric and spatial figures.

The two source figures are not modified.  The spatial source is the historical
panel-normalized rendering requested by the user, so the combined PDF is
raster-backed even though a vector source exists for the metric panel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
METRIC_SOURCE = (
    ROOT
    / "outputs/GSE267904_agent_granularity_heterogeneity_v1_5_paper_title/figure"
    / "GSE267904_LecaVC_agent_granularity_heterogeneity_paper_title_600dpi.png"
)
SPATIAL_SOURCE = (
    ROOT
    / "outputs/GSE267904_spatial_agent_granularity/figures"
    / "module_expression_K20_K50_K100_old_style_horizontal"
    / "GSE267904_module_expression_K20_K50_K100_old_style_panel_normalized.png"
)
OUT_DIR = ROOT / "outputs/GSE267904_agent_granularity_spatial_combined_v1"
FIG_DIR = OUT_DIR / "figure"
STEM = "GSE267904_agent_granularity_metrics_and_spatial_maps_landscape"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_rgba(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def main() -> None:
    for source in (METRIC_SOURCE, SPATIAL_SOURCE):
        if not source.is_file():
            raise FileNotFoundError(source)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(METRIC_SOURCE) as metric_raw, Image.open(SPATIAL_SOURCE) as spatial_raw:
        metric = flatten_rgba(metric_raw)
        spatial = flatten_rgba(spatial_raw)

    # Preserve the full metric panel at native 600-DPI height and resize the
    # spatial panel proportionally to the same height.
    target_height = metric.height
    spatial_width = round(spatial.width * target_height / spatial.height)
    spatial_equal_height = spatial.resize(
        (spatial_width, target_height), Image.Resampling.LANCZOS
    )

    outer_margin = 100
    gap = 140
    canvas = Image.new(
        "RGB",
        (
            outer_margin * 2 + metric.width + gap + spatial_equal_height.width,
            outer_margin * 2 + target_height,
        ),
        "white",
    )
    metric_xy = (outer_margin, outer_margin)
    spatial_xy = (outer_margin + metric.width + gap, outer_margin)
    canvas.paste(metric, metric_xy)
    canvas.paste(spatial_equal_height, spatial_xy)

    png_path = FIG_DIR / f"{STEM}_600dpi.png"
    pdf_path = FIG_DIR / f"{STEM}_600dpi_raster.pdf"
    preview_path = FIG_DIR / f"{STEM}_preview.png"
    canvas.save(png_path, dpi=(600, 600), optimize=True)
    canvas.save(pdf_path, "PDF", resolution=600.0)

    preview_width = 3000
    preview_height = round(canvas.height * preview_width / canvas.width)
    preview = canvas.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
    preview.save(preview_path, optimize=True)

    audit = {
        "status": "PASS_MECHANICAL_HORIZONTAL_ASSEMBLY",
        "scientific_values_recomputed": False,
        "source_figures_modified": False,
        "layout": "metric figure left; spatial panel-normalized figure right; equal height; aspect ratios preserved",
        "metric_source": {
            "path": str(METRIC_SOURCE),
            "sha256": sha256(METRIC_SOURCE),
            "native_size_px": list(metric.size),
            "placed_xy_px": list(metric_xy),
            "placed_size_px": list(metric.size),
        },
        "spatial_source": {
            "path": str(SPATIAL_SOURCE),
            "sha256": sha256(SPATIAL_SOURCE),
            "native_size_px": list(spatial.size),
            "placed_xy_px": list(spatial_xy),
            "placed_size_px": list(spatial_equal_height.size),
            "rendering_semantics": "historical per-panel 1st-99th percentile min-max normalization",
        },
        "canvas_size_px": list(canvas.size),
        "dpi": 600,
        "outputs": {
            "png_600dpi": str(png_path),
            "pdf_600dpi_raster": str(pdf_path),
            "preview_png": str(preview_path),
        },
    }
    audit_path = OUT_DIR / "assembly_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# GSE267904 metric and spatial-map horizontal assembly\n\n"
        "This is a mechanical, equal-height, aspect-ratio-preserving assembly of two frozen figures. "
        "No scientific value was recomputed and neither source was modified. The spatial source is the "
        "historical panel-normalized rendering, so the combined PDF is explicitly raster-backed.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
