#!/usr/bin/env python3
"""Render the current V4.4 GSE120575 UMAP template from de-identified outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse120575-deid-v1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-gse120575-deid-v1")

import matplotlib.figure
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V42_SCRIPT = (
    ROOT
    / "scripts/gse120575_cell_expanded_umap_figure_v4_2_response_metrics/run_v4_2.py"
)
V3_SCRIPT = ROOT / "scripts/gse120575_cell_expanded_umap_figure_v3/run_v3.py"
BASE = Path(os.environ["GSE120575_V3_OUT"]).resolve()
V3_OUT = BASE / "cell_expanded_umap_figure_v3"
OUT = Path(os.environ["GSE120575_UMAP_FIGURE_OUT"]).resolve()
STEM = "GSE120575_cell_expanded_expression_accuracy_UMAP_deidentified_v1"
STATUS = "PASS_GSE120575_DEIDENTIFIED_CURRENT_UMAP_TEMPLATE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    h5ad = V3_OUT / "unified_cell_expanded_expression_834genes.h5ad"
    metrics_path = BASE / "unified_expression_metrics.csv"
    required = (h5ad, metrics_path, V3_SCRIPT, V42_SCRIPT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing de-identified UMAP inputs: {missing}")

    v42 = load(V42_SCRIPT, "gse120575_deid_v42_template")
    v42.V3_H5AD = h5ad
    v42.FROZEN_METRICS = metrics_path
    metadata, coordinates, metrics = v42.load_inputs()
    if coordinates.shape != (16178, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("new frozen-reference UMAP coordinates failed shape/finite gate")
    if len(metrics) != 8:
        raise RuntimeError("response-level CCC/RMSE table must contain eight rows")

    v3 = load(V3_SCRIPT, "gse120575_deid_v3_plot_helpers")
    OUT.mkdir(parents=True, exist_ok=True)
    v42.OUT = OUT
    v42.PREVIEW = OUT / f"{STEM}_preview.png"
    v42.PNG_600 = OUT / f"{STEM}_600dpi.png"
    v42.PDF = OUT / f"{STEM}_vector.pdf"
    v42.SVG = OUT / f"{STEM}.svg"
    v42.METHOD_LABELS = {
        "scGen": "scGen",
        "WOT_weighted": "WOT",
        "CellRank_terminal_proxy": "CellRank",
        "AgentVC_proxy": "Leca-VC",
    }

    title_map = {
        "WOT* (target-derived)": "WOT",
        "CellRank\u2020 proxy": "CellRank",
        "AgentVC proxy": "Leca-VC",
    }
    original_panel = v3.plot_umap_panel

    def renamed_panel(ax, panel_metadata, panel_coordinates, limits, letter, title, method):
        return original_panel(
            ax,
            panel_metadata,
            panel_coordinates,
            limits,
            letter,
            title_map.get(title, title),
            method,
        )

    v3.plot_umap_panel = renamed_panel
    original_text = matplotlib.figure.Figure.text

    def renamed_text(figure, x, y, text, *args, **kwargs):
        if isinstance(text, str):
            text = text.replace("WOT* is target-derived.", "WOT is target-derived.")
            text = text.replace("AgentVC", "Leca-VC")
            text = text.replace("frozen response-level", "recomputed response-level")
        return original_text(figure, x, y, text, *args, **kwargs)

    matplotlib.figure.Figure.text = renamed_text
    try:
        v42.draw_figure(v3, metadata, coordinates, metrics)
    finally:
        matplotlib.figure.Figure.text = original_text

    metrics_out = OUT / "response_level_expression_accuracy_metrics.csv"
    metrics.to_csv(metrics_out, index=False, float_format="%.17g")
    svg_text = v42.SVG.read_text(encoding="utf-8")
    if "Leca-VC" not in svg_text or "AgentVC" in svg_text:
        raise RuntimeError("V4.4 display-name gate failed")
    artifacts = (v42.PREVIEW, v42.PNG_600, v42.PDF, v42.SVG, metrics_out)
    if any(not path.is_file() or path.stat().st_size == 0 for path in artifacts):
        raise RuntimeError("one or more current-template UMAP artifacts are missing")
    audit = {
        "status": STATUS,
        "template": "GSE120575 cell-expanded UMAP V4.4",
        "umap_recomputed_from_new_leca_vc_proxy": True,
        "response_metrics_recomputed_from_new_leca_vc_proxy": True,
        "observed_and_external_baselines_frozen": True,
        "input_hashes": {
            str(h5ad.relative_to(ROOT)): sha256(h5ad),
            str(metrics_path.relative_to(ROOT)): sha256(metrics_path),
        },
        "visible_method_names": ["scGen", "WOT", "CellRank", "Leca-VC"],
        "output_hashes": {path.name: sha256(path) for path in artifacts},
    }
    (OUT / "figure_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "caption.txt").write_text(
        "Cell-expanded gene-expression accuracy across immune populations after "
        "the locked de-identified Leca-VC rerun. Observed and external-baseline "
        "inputs are frozen; only the Leca-VC proxy, its frozen-reference UMAP "
        "transform and its response-level CCC/RMSE values were regenerated. "
        "Expanded rows are deterministic sample-level expression proxies, not "
        "native single-cell predictions; WOT is target-derived.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": STATUS, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
