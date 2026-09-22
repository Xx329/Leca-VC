#!/usr/bin/env python3
"""Render the locked de-identified GSE267904/GSE120575 paper composite.

The geometry and visual grammar come from the approved compact V4.4 template.
Only source tables produced by the locked contamination-control rerun are read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts/leca_vc_multibenchmark_compact_no_wot_v4/plot_v4.py"
CCI_ROOT = Path(
    os.environ.get(
        "GSE267904_DEID_CCI_SOURCE",
        ROOT / "outputs/GSE267904_deidentified_rerun_v1/worker_cci_lrra/06_figure/source_data",
    )
).resolve()
EXP_ROOT = Path(
    os.environ.get(
        "GSE120575_DEID_EXPRESSION_ROOT",
        ROOT / "outputs/GSE120575_deidentified_rerun_v1/postprocess/expression_benchmark_v3_cellrank",
    )
).resolve()
OUT = Path(
    os.environ.get(
        "LECAVC_DEID_PAPER_FIGURE_OUT",
        ROOT / "outputs/LecaVC_paper_figures_deidentified_v1/multibenchmark",
    )
).resolve()
PREFIX = "Leca_VC_multibenchmark_deidentified_v1"
STATUS = "PASS_LOCKED_DEIDENTIFIED_MULTIBENCHMARK_FIGURE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location("deid_multibenchmark_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import approved figure template: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_pass(path: Path, prefix: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required completion audit missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not str(value.get("status", "")).startswith(prefix):
        raise RuntimeError(f"upstream audit is not a PASS: {path}")


def main() -> None:
    require_pass(
        ROOT / "outputs/GSE120575_deidentified_rerun_v1/audit/execution_audit.json",
        "PASS_",
    )
    require_pass(
        ROOT / "outputs/GSE267904_deidentified_rerun_v1/audit/execution_audit.json",
        "PASS_",
    )
    marker = (
        ROOT
        / "outputs/GSE267904_deidentified_rerun_v1/worker_cci_lrra/"
        "PASS_AGENTVC_LRRA_TOP1_LOCAL_NO_REGRESSION.marker"
    )
    if not marker.is_file():
        raise RuntimeError("formal composite forbidden: the frozen CCI LRRA gate did not pass")

    base = load_base()
    base.OUT = OUT
    base.SOURCE_OUT = OUT / "source_data"
    base.STATUS = STATUS
    base.FILE_PREFIX = PREFIX
    base.SHOW_MAIN_TITLE = False
    base.SHOW_SECTION_DIVIDER = False
    base.PREDICTED_CALIBRATION_LINESTYLE = "--"
    base.CCI_SECTION_TITLE = (
        "GSE267904 · Cell–cell communication reconstruction accuracy and stability"
    )
    base.PRIMARY_TITLE = "A. CCI reconstruction accuracy"
    base.SECONDARY_TITLE = (
        "B. Network and biological signaling reconstruction stability"
    )
    base.SECONDARY_XLABELS = {
        "GNRS_secondary": "Global Network Reconstruction Score ↑",
        "LBSS_secondary": "Local Biological Signaling Score ↑",
    }
    base.SCATTER_SECTION_TITLE = (
        "GSE120575 · Observed vs reconstructed post-treatment gene expression"
    )
    base.SUMMARY_SECTION_TITLE = (
        "GSE120575 · Expression correlation, error, and calibration"
    )
    base.INTERVAL_TITLE_TEMPLATE = "{letter}. {response} {metric} (95% CI)"
    base.INTERVAL_TITLE_METRIC_LABELS = {
        "Pearson": "Pearson correlation",
        "RMSE": "RMSE",
    }
    base.INTERVAL_XLABELS = {"Pearson": "Pearson ↑", "RMSE": "RMSE ↓"}
    base.CALIBRATION_TITLE_TEMPLATE = "{letter}. {response} expression calibration"
    base.OBSERVED_REFERENCE_LABEL = "Observed reference"
    base.INPUTS = {
        "cci_primary": CCI_ROOT / "panel_G_primary_summary.csv",
        "cci_secondary": CCI_ROOT / "panel_H_secondary_summary.csv",
        "expression_gene_values": EXP_ROOT / "unified_gene_level_plotting_source.csv",
        "expression_metrics": EXP_ROOT / "unified_expression_metrics.csv",
        "expression_bootstrap_summary": EXP_ROOT / "bootstrap_confidence_intervals.csv",
        "expression_deciles": EXP_ROOT / "expression_decile_calibration.csv",
    }
    missing = [str(path) for path in base.INPUTS.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"de-identified figure inputs missing: {missing}")
    base.EXPECTED_HASHES = {key: sha256(path) for key, path in base.INPUTS.items()}

    os.environ["LECAVC_DEIDENTIFIED_POSTPROCESS"] = "1"
    base.main()

    svg = OUT / f"{PREFIX}_vector.svg"
    svg_text = svg.read_text(encoding="utf-8")
    required = [
        base.CCI_SECTION_TITLE,
        base.PRIMARY_TITLE,
        base.SECONDARY_TITLE,
        base.SCATTER_SECTION_TITLE,
        base.SUMMARY_SECTION_TITLE,
        "E. Responder — Leca-VC",
        "K. Non-responder — Leca-VC",
        "Observed reference",
    ]
    absent = [text for text in required if text not in svg_text]
    if absent:
        raise RuntimeError(f"approved canvas text missing: {absent}")

    caption = (
        "GSE267904 CCI and GSE120575 post-treatment expression benchmarks after "
        "the locked de-identified DeepSeek rerun. Panel A reports six raw CCI "
        "reconstruction metrics separated by metric direction, and Panel B reports "
        "median and interquartile range for the Global Network Reconstruction and "
        "Local Biological Signaling scores. Panels C–N show gene-level agreement, "
        "Pearson and RMSE estimates with sample/biopsy bootstrap 95% confidence "
        "intervals, and decile calibration. Observed and external-baseline inputs "
        "are frozen; only Leca-VC was regenerated. WOT is excluded because it uses "
        "endpoint information. Bootstrap intervals summarize stability and do not "
        "create additional biological replicates."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    audit = {
        "status": STATUS,
        "experiment_role": "locked post-hoc contamination-control rerun",
        "template": str(BASE_SCRIPT.relative_to(ROOT)),
        "template_geometry_preserved": True,
        "labeled_method": "Leca-VC",
        "wot_displayed": False,
        "source_hashes": base.EXPECTED_HASHES,
        "upstream_cci_gate_marker": str(marker.relative_to(ROOT)),
        "metrics_recomputed_from_new_leca_vc": True,
        "external_baselines_reused": True,
    }
    (OUT / "deidentified_figure_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = OUT / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(audit)
    manifest["outputs"]["deidentified_figure_audit.json"] = base.record(
        OUT / "deidentified_figure_audit.json"
    )
    manifest["outputs"]["caption.txt"] = base.record(OUT / "caption.txt")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": STATUS, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
