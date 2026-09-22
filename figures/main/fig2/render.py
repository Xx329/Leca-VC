#!/usr/bin/env python3
"""Render manuscript Figure 2 from its frozen release source tables."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "source_data/fig2"
OUT = ROOT / "build/figures/fig2"


def load_base():
    path = HERE / "base_plot.py"
    spec = importlib.util.spec_from_file_location("fig2_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    profiles = pd.read_csv(SOURCE / "panels_A_to_D_time_resolved_profiles.csv")
    metrics = pd.read_csv(SOURCE / "panels_E_to_G_raw_metrics.csv")
    profiles["method"] = profiles["source_method_id"]
    metrics["method"] = metrics["source_method_id"]
    if profiles.shape[0] != 45 or metrics.shape[0] != 32:
        raise RuntimeError("Frozen Figure 2 source dimensions changed")

    base = load_base()
    base.OUT = OUT
    base.SOURCE_OUT = OUT / "source_data"
    base.FILE_STEM = "GSE2565_bulk_time_resolved_transcriptomic_response_benchmark_deidentified_v1"
    base.STATUS = "RENDERED_FROM_RELEASE_SOURCE_DATA"
    base.OVERALL_TITLE = "GSE2565 bulk time-resolved transcriptomic response benchmark"
    base.OVERALL_SUBTITLE = ""
    base.DISPLAY_NAMES["AgentVC Online Agent pilot"] = "Leca-VC"
    base.PANEL_SPECS = [
        ("A", "DNB dynamics and peak timing", "normalized_DNB"),
        ("B", "Expression distribution shift over time", "normalized_distribution_shift"),
        ("C", "Transcriptomic deviation from baseline over time", "normalized_progression"),
        ("D", "Transcriptomic change rate over time", "normalized_velocity"),
    ]
    base.GROUP_SPECS = [
        ("E. DNB timing and temporal-profile accuracy", "F1", ["dnb_peak_error", "dnb_pearson", "dnb_dtw"]),
        ("F. Late-time global transcriptomic agreement", "F2", ["distribution_pearson", "progression_pearson", "velocity_pearson"]),
        ("G. Late-time functional and gene-direction agreement", "F3", ["mean_module_pearson", "gene_direction_agreement"]),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    base.SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(base.SOURCE_OUT / "panels_A_to_D_time_resolved_profiles.csv", index=False)
    metrics.to_csv(base.SOURCE_OUT / "panels_E_to_G_raw_metrics.csv", index=False)
    paths = base.render(profiles, metrics)
    result = {name: str(path.relative_to(ROOT)) for name, path in paths.items()}
    (OUT / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
