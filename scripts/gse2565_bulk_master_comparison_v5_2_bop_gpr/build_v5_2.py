#!/usr/bin/env python3
"""Build the final GSE2565 V5.2 figure from frozen and newly passed outputs."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np

from common import (
    BASE,
    BOP_EIG_CONSTRAINTS,
    BOP_EIG_SORT,
    BOP_VARPRO_TOL,
    EVAL,
    OUT,
    PLOT_DATA,
    REVISION,
    TIMES,
    array_sha256,
    ensure_dirs,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    ROOT / "scripts/gse2565_bulk_master_comparison_v5_bop_gpr/build_v5.py"
)


def load_source():
    spec = importlib.util.spec_from_file_location("gse2565_v5_2_builder_base", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = load_source()


def load_inputs_v5_2() -> dict:
    ensure_dirs()
    if not SOURCE.BOP_AUDIT_PATH.exists():
        raise RuntimeError("V5.2 BOP-DMD audit missing")
    bop_audit = json.loads(SOURCE.BOP_AUDIT_PATH.read_text(encoding="utf-8"))
    if bop_audit.get("status") != "PASS_BOPDMD_STABLE_CONJUGATE_SORTED_V5_2":
        raise RuntimeError(f"V5.2 BOP-DMD did not pass: {bop_audit.get('status')}")

    frozen = BASE.load_frozen_inputs()
    genes = frozen["genes"]
    matrices_full = {
        "Real": frozen["real"],
        "chronODE-M": frozen["chron"],
        "RVAgene": SOURCE.load_npz_expression(SOURCE.RVA_PATH, genes),
        "BOP-DMD": SOURCE.load_npz_expression(SOURCE.BOP_PATH, genes),
        "GPR": SOURCE.load_npz_expression(SOURCE.GPR_PATH, genes),
        "AgentVC Online Agent pilot": frozen["agent"],
    }
    if any(matrix.shape != (9, 12263) for matrix in matrices_full.values()):
        raise RuntimeError("Unified V5.2 trajectory shape failed")
    valid = frozen["complete_valid"]
    if int(valid.sum()) != 7632:
        raise RuntimeError("Frozen 7,632-gene mask changed")
    genes_eval = [gene for gene, keep in zip(genes, valid) if keep]
    matrices = {
        method: matrix[:, valid] for method, matrix in matrices_full.items()
    }
    microarray = {
        method: method != "AgentVC Online Agent pilot" for method in matrices
    }
    curves = {
        method: REVISION.curves_for_matrix(matrix, microarray[method])
        for method, matrix in matrices.items()
    }
    progression = {
        method: BASE.minmax(item["progression"])[0]
        for method, item in curves.items()
    }
    modules = frozen["modules"]
    module_raw, module_z = {}, {}
    for method, matrix in matrices.items():
        module_raw[method] = EVAL.module_matrix(matrix, genes_eval, modules)
        module_z[method], _ = EVAL.zscore_time(module_raw[method])
    return {
        "frozen": frozen,
        "genes": genes,
        "genes_eval": genes_eval,
        "matrices_full": matrices_full,
        "matrices": matrices,
        "curves": curves,
        "progression": progression,
        "module_raw": module_raw,
        "module_z": module_z,
        "modules": modules,
        "bop_audit": bop_audit,
    }


def rename_figure_outputs(paths: dict[str, Path]) -> dict[str, Path]:
    renamed = {}
    replacements = {
        "preview_png": "GSE2565_bulk_master_comparison_v5_2_preview.png",
        "png_600dpi": "GSE2565_bulk_master_comparison_v5_2_600dpi.png",
        "pdf": "GSE2565_bulk_master_comparison_v5_2_vector.pdf",
        "svg": "GSE2565_bulk_master_comparison_v5_2_vector.svg",
    }
    for key, path in paths.items():
        target = OUT / replacements[key]
        Path(path).replace(target)
        renamed[key] = target
    return renamed


def main() -> None:
    SOURCE.load_inputs = load_inputs_v5_2
    data = SOURCE.load_inputs()
    dnb_curves = SOURCE.load_dnb_curves()
    metrics = SOURCE.late_metrics(data, dnb_curves)
    metric_path = OUT / "raw_metric_summary.csv"
    metrics.to_csv(metric_path, index=False)
    if os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") == "1":
        frozen_maxdiff = {"deidentified_LecaVC_rows_are_expected_to_change": 0.0}
        formal_callout = {"status": "RECOMPUTED_DEIDENTIFIED_RUN_NO_HISTORICAL_VALUE_ASSERTION"}
    else:
        frozen_maxdiff = SOURCE.validate_frozen_metrics(metrics)
        formal_callout = SOURCE.validate_formal_callout()
    source_paths = SOURCE.write_source_data(data, dnb_curves, metrics)
    panel_f = SOURCE.pd.read_csv(source_paths["panel_f"])
    figure_paths = rename_figure_outputs(
        SOURCE.plot_figure(data, dnb_curves, panel_f)
    )

    matrix_inventory = []
    for method, matrix in data["matrices_full"].items():
        matrix_inventory.append(
            {
                "method": method,
                "shape": "9x12263",
                "finite": bool(np.isfinite(matrix).all()),
                "array_sha256": array_sha256(matrix),
                "input_regime": (
                    "reference"
                    if method == "Real"
                    else SOURCE.INPUT_REGIMES[method]
                ),
            }
        )
    matrix_path = OUT / "unified_trajectory_inventory.csv"
    SOURCE.pd.DataFrame(matrix_inventory).to_csv(matrix_path, index=False)

    script_copy = OUT / "build_v5_2.py"
    shutil.copy2(Path(__file__), script_copy)
    generated = [
        metric_path,
        matrix_path,
        *source_paths.values(),
        *figure_paths.values(),
        script_copy,
        OUT / "bopdmd_rank_sensitivity_even_ranks.csv",
        OUT / "BOPDMD_FORMAL_AUDIT.json",
        OUT / "FROZEN_PROTOCOL_V5_2.json",
    ]
    manifest_path = OUT / "sha256_manifest.csv"
    SOURCE.source_manifest(data, generated).to_csv(manifest_path, index=False)

    common_peaks = {
        method: float(
            TIMES[np.isfinite(curve)][
                np.argmax(np.asarray(curve)[np.isfinite(curve)])
            ]
        )
        for method, curve in dnb_curves.items()
    }
    audit = {
        "status": "PASS_DESCRIPTIVE_BULK_RECONSTRUCTION_V5_2_BOP_GPR",
        "benchmark_semantics": "descriptive bulk trajectory reconstruction comparison",
        "equal_input_prediction_claim_allowed": False,
        "overall_winner_claim_allowed": False,
        "historical_models_rerun": False,
        "historical_outputs_overwritten": False,
        "methods": SOURCE.METHODS,
        "trajectory_shape_each": [9, 12263],
        "evaluation_gene_count_panels_B_to_E_F2_F3": 7632,
        "common_DNB_gene_count_panels_A_F1": 1601,
        "late_state_hours": [8, 12, 24, 48, 72],
        "late_velocity_intervals": [
            "4->8",
            "8->12",
            "12->24",
            "24->48",
            "48->72",
        ],
        "GPR_training_states_in_F2_F3": False,
        "frozen_metric_max_absolute_differences": frozen_maxdiff,
        "formal_fullspace_DNB_callout": formal_callout,
        "common_space_DNB_peaks_hours": common_peaks,
        "BOP_DMD_protocol": {
            "rank": 4,
            "num_trials": 100,
            "trial_size": 0.8,
            "eig_constraints": sorted(BOP_EIG_CONSTRAINTS),
            "eig_sort": BOP_EIG_SORT,
            "varpro_tol": BOP_VARPRO_TOL,
            "target_access": True,
            "selection_used_benchmark_performance": False,
        },
        "BOP_DMD": data["bop_audit"],
        "figure_outputs": {key: str(value) for key, value in figure_paths.items()},
        "plot_source_tables": {key: str(value) for key, value in source_paths.items()},
        "manifest": str(manifest_path),
        "visual_inspection": "PENDING_MANUAL_INSPECTION",
    }
    audit_path = OUT / "FINAL_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    readme = f"""# GSE2565 Bulk Master Comparison V5.2

Status: **PASS_DESCRIPTIVE_BULK_RECONSTRUCTION_V5_2_BOP_GPR**

V5.2 is a descriptive reconstruction comparison, not an equal-input forecast
benchmark. chronODE-M, RVAgene and BOP-DMD access the complete observed
trajectory; GPR uses observed 0–4 h; AgentVC does not consume observed GSE2565
expression at runtime and remains a six-run pilot.

BOP-DMD uses the pre-frozen stable/conjugate protocol: rank 4, 100 trials,
trial size 0.8, `eig_sort="imag"`, variable-projection tolerance 0.2, true-hour
time coordinates, no negative clipping and no Observed refill. These numerical
settings were selected only from completion/finite/determinism gates, before
reading final benchmark metrics.

Panels A/F1 use the 1,601-gene common DNB sensitivity space. Panels
B–E/F2–F3 use the frozen 7,632-gene evaluation space. F2/F3 exclude GPR
training states. The independent formal 11,171-gene DNB audit remains Real
8 h, AgentVC 8 h, AgentVC peak error 0 h.

No composite score is generated and no overall winner is claimed.

- Figure: `{figure_paths["png_600dpi"]}`
- Raw metrics: `{metric_path}`
- Plot source data: `{PLOT_DATA}`
- BOP-DMD audit: `{OUT / "BOPDMD_FORMAL_AUDIT.json"}`
- Final audit: `{audit_path}`
- Manifest: `{manifest_path}`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
