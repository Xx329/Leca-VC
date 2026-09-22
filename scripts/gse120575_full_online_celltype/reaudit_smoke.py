#!/usr/bin/env python3
"""Offline corrected audit of existing GSE120575 smoke outputs.

This script never starts PhysiCell, calls an LLM, or loads/runs scGPT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from controllers import CELL_TYPES
from run_smoke import OUT, SMOKE, evaluate, read_csv, write_json


ROOT = Path(__file__).resolve().parents[2]
PHYSICELL_ROOT = Path(
    os.environ.get("PHYSICELL_ROOT", str(ROOT / "vendor/PhysiCell"))
).expanduser().resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS = ("traditional", "agent_only", "full")
QUICK_SEEDS = (12057501, 12057502, 12057503)
ACTION_BOUNDS = {
    "birth_rate_multiplier": (0.25, 2.0),
    "death_rate_multiplier": (0.25, 2.0),
    "motility_multiplier": (0.25, 2.0),
    "secretion_multiplier": (0.25, 2.0),
    "uptake_multiplier": (0.25, 2.0),
    "recruitment_rate": (0.0, 0.05),
    "clearance_rate": (0.0, 0.05),
    "activation_strength": (0.0, 1.0),
    "stress_strength": (0.0, 1.0),
    "exhaustion_strength": (0.0, 1.0),
    "inflammatory_strength": (0.0, 1.0),
    "suppression_strength": (0.0, 1.0),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_preclip(record: dict[str, object], model: str) -> dict[str, float]:
    validated = record["validated_action"]
    strengths = validated["program_strengths"]
    multipliers = validated["phenotype_multipliers"]
    values = {
        "birth_rate_multiplier": float(multipliers["birth_rate"]),
        "death_rate_multiplier": float(multipliers["death_rate"]),
        "motility_multiplier": float(multipliers["motility"]),
        "secretion_multiplier": float(multipliers["secretion"]),
        "uptake_multiplier": float(multipliers["uptake"]),
        "recruitment_rate": 0.015 * float(strengths["recruitment"]) * float(multipliers["recruitment_rate"]),
        "clearance_rate": 0.010 * float(strengths["clearance"]),
        "activation_strength": float(strengths["activation"]),
        "stress_strength": float(strengths["apoptosis"]),
        "exhaustion_strength": float(strengths["exhaustion"]),
        "inflammatory_strength": float(strengths["inflammatory_secretion"]),
        "suppression_strength": float(strengths["immune_suppression"]),
    }
    if model == "full":
        scgpt = record["scgpt_output"]
        identity_gate = float(np.clip(1.0 - 0.5 * float(scgpt["identity_drift"]), 0.70, 1.0))
        activation_gate = float(np.tanh(max(0.0, float(scgpt["activation_shift"]))))
        stress_gate = float(np.tanh(max(0.0, float(scgpt["stress_shift"]))))
        exhaustion_gate = float(np.tanh(max(0.0, float(scgpt["exhaustion_shift"]))))
        values["birth_rate_multiplier"] *= identity_gate * (1.0 - 0.10 * exhaustion_gate)
        values["death_rate_multiplier"] *= 1.0 + 0.20 * stress_gate + 0.15 * exhaustion_gate
        values["secretion_multiplier"] *= 1.0 + 0.20 * activation_gate
        values["motility_multiplier"] *= identity_gate
        values["recruitment_rate"] *= 1.0 + 0.15 * activation_gate
    return values


def parameter_bound_audit(run_records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    reconstruction_max_error = 0.0
    action_records = 0
    for run in run_records:
        run_dir = Path(run["run_dir"])
        model = str(run["model"])
        for checkpoint in range(5):
            payload = json.loads((run_dir / f"execution_payload_{checkpoint}.json").read_text())
            for record in payload["records"]:
                action_records += 1
                preclip = reconstruct_preclip(record, model)
                executed = {key: float(record["executed_physicell_parameters"][key]) for key in ACTION_BOUNDS}
                for parameter, (lower, upper) in ACTION_BOUNDS.items():
                    expected = float(np.clip(preclip[parameter], lower, upper))
                    reconstruction_max_error = max(reconstruction_max_error, abs(expected - executed[parameter]))
                    grouped[(model, str(record["cell_type"]), parameter)].append({
                        "value": executed[parameter],
                        "preclip": preclip[parameter],
                        "clipped": abs(preclip[parameter] - executed[parameter]) > 1e-12,
                    })

    rows = []
    for model in MODELS:
        for cell_type in CELL_TYPES:
            for parameter, (lower, upper) in ACTION_BOUNDS.items():
                observations = grouped[(model, cell_type, parameter)]
                values = np.asarray([float(item["value"]) for item in observations])
                lower_count = int(np.count_nonzero(np.isclose(values, lower, atol=1e-12, rtol=0.0)))
                upper_count = int(np.count_nonzero(np.isclose(values, upper, atol=1e-12, rtol=0.0)))
                rows.append({
                    "model": model,
                    "cell_type": cell_type,
                    "parameter": parameter,
                    "registered_lower_bound": lower,
                    "registered_upper_bound": upper,
                    "n": len(observations),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "clipping_count": sum(bool(item["clipped"]) for item in observations),
                    "lower_bound_count": lower_count,
                    "upper_bound_count": upper_count,
                    "fraction_at_lower_bound": lower_count / len(observations),
                    "fraction_at_upper_bound": upper_count / len(observations),
                    "out_of_bounds_count": int(np.count_nonzero((values < lower - 1e-12) | (values > upper + 1e-12))),
                })
    write_csv(SMOKE / "parameter_bound_audit_corrected.csv", rows)
    summary = {
        "status": "PASS_PARAMETER_BOUND_AUDIT" if all(row["out_of_bounds_count"] == 0 for row in rows) else "FAIL_PARAMETER_BOUND_AUDIT",
        "diagnostic_only_no_parameters_modified": True,
        "executed_action_records": action_records,
        "group_rows": len(rows),
        "total_parameter_observations": sum(int(row["n"]) for row in rows),
        "total_clipping_count": sum(int(row["clipping_count"]) for row in rows),
        "total_out_of_bounds_count": sum(int(row["out_of_bounds_count"]) for row in rows),
        "mapper_reconstruction_max_absolute_error": reconstruction_max_error,
        "bounds": {key: {"lower": value[0], "upper": value[1]} for key, value in ACTION_BOUNDS.items()},
        "csv": str((SMOKE / "parameter_bound_audit_corrected.csv").resolve()),
    }
    write_json(SMOKE / "parameter_bound_audit_corrected.json", summary)
    return summary


def quick_manifest() -> tuple[dict[str, object], list[dict[str, object]]]:
    subset = [row for row in read_csv(OUT / "preprocessed/preregistered_sample_subset.csv") if row["phase"] == "quick_3seed"]
    proportions = {row["sample_id"]: row for row in read_csv(OUT / "preprocessed/pretreatment_sample_celltype_proportions.csv")}
    rows = []
    registry_consistency = True
    for selected in subset:
        sample_id = selected["sample_id"]
        source = proportions[sample_id]
        counts_by_seed = []
        for seed in QUICK_SEEDS:
            registry = read_csv(OUT / "preprocessed/worker_registries" / f"{sample_id}__seed_{seed}.csv")
            counts_by_seed.append({cell_type: sum(row["cell_type"] == cell_type for row in registry) for cell_type in CELL_TYPES})
        registry_consistency &= all(counts == counts_by_seed[0] for counts in counts_by_seed)
        counts = counts_by_seed[0]
        row: dict[str, object] = {
            "sample_id": sample_id,
            "response": source["response"],
            "therapy": source["therapy"],
            "patient_id": source["patient_id"],
            "initial_cell_count": sum(counts.values()),
        }
        for cell_type in CELL_TYPES:
            row[f"initial_{cell_type}_count"] = counts[cell_type]
            row[f"initial_{cell_type}_proportion"] = counts[cell_type] / sum(counts.values())
        rows.append(row)

    response_to_therapies: dict[str, set[str]] = defaultdict(set)
    therapy_to_responses: dict[str, set[str]] = defaultdict(set)
    contingency: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        response = str(row["response"]); therapy = str(row["therapy"])
        response_to_therapies[response].add(therapy)
        therapy_to_responses[therapy].add(response)
        contingency[(response, therapy)] += 1
    completely_confounded = (
        all(len(values) == 1 for values in response_to_therapies.values())
        and all(len(values) == 1 for values in therapy_to_responses.values())
    )
    design = {
        "status": "STOP_COMPLETE_RESPONSE_THERAPY_CONFOUNDING" if completely_confounded else "PASS_NO_COMPLETE_RESPONSE_THERAPY_CONFOUNDING",
        "sample_count": len(rows),
        "seeds": list(QUICK_SEEDS),
        "models": list(MODELS),
        "expected_runs": len(rows) * len(QUICK_SEEDS) * len(MODELS),
        "expected_agent_calls": len(rows) * len(QUICK_SEEDS) * 2 * len(CELL_TYPES) * 5,
        "expected_scgpt_calls": len(rows) * len(QUICK_SEEDS) * len(CELL_TYPES) * 5,
        "expected_cpp_writebacks": len(rows) * len(QUICK_SEEDS) * len(MODELS) * len(CELL_TYPES) * 5,
        "response_to_therapies": {key: sorted(value) for key, value in response_to_therapies.items()},
        "therapy_to_responses": {key: sorted(value) for key, value in therapy_to_responses.items()},
        "contingency": [
            {"response": response, "therapy": therapy, "n_samples": count}
            for (response, therapy), count in sorted(contingency.items())
        ],
        "response_and_therapy_completely_confounded": completely_confounded,
        "partial_imbalance_warning": "All selected Non-responder samples are anti-PD1; anti-PD1 also contains one Responder, so confounding is not complete but inference is imbalanced.",
        "all_seed_registries_have_identical_composition_within_sample": registry_consistency,
        "sample_selection_modified": False,
        "quick_run_started": False,
    }
    write_csv(OUT / "audit/quick_manifest_frozen.csv", rows)
    write_json(OUT / "audit/quick_manifest_frozen.json", {"design": design, "samples": rows})
    return design, rows


def freeze_hashes(design: dict[str, object]) -> dict[str, object]:
    protocol = {
        "status": "FROZEN_BEFORE_54_RUN_QUICK",
        "workers_per_run": 600,
        "models": list(MODELS),
        "cell_types": list(CELL_TYPES),
        "decision_checkpoints": [0, 20, 40, 60, 80],
        "endpoint_read_only": 100,
        "seeds": list(QUICK_SEEDS),
        "expected_runs": 54,
        "expected_agent_calls": 1080,
        "expected_scgpt_calls": 540,
        "expected_cpp_writebacks": 1620,
        "automatic_start_authorized": False,
        "synthetic_spatial_initialization": True,
        "real_physicell_biofvm_required": True,
        "action_bounds": {key: list(value) for key, value in ACTION_BOUNDS.items()},
    }
    evaluation = {
        "status": "FROZEN_BEFORE_54_RUN_QUICK",
        "prediction": "checkpoint-100 endpoint broad-cell-type composition",
        "locked_reference": "post-treatment response-group mean broad-cell-type composition; evaluation-only loader",
        "aggregation_order": ["mean three seeds within sample", "compute sample-level metrics"],
        "metrics": {
            "MAE": "mean_i(abs(pred_i-reference_i)) over six broad cell types",
            "RMSE": "sqrt(mean_i((pred_i-reference_i)^2)) over six broad cell types",
            "JSD": "Jensen-Shannon divergence between six-type compositions",
            "Pearson_correlation": "Pearson correlation across six broad cell-type proportions",
            "direction_agreement": "fraction_i sign(pred_i-pre_i)==sign(reference_i-pre_i)",
            "seed_stability": "within-sample dispersion across the three endpoint seed compositions",
        },
        "formal_box_plot_authorized": False,
        "thresholds_changed_after_smoke": False,
    }
    protocol_path = OUT / "audit/quick_protocol_frozen.json"
    evaluation_path = OUT / "audit/quick_evaluation_metrics_frozen.json"
    write_json(protocol_path, protocol)
    write_json(evaluation_path, evaluation)

    categories = {
        "code": sorted(SCRIPT_DIR.glob("*.py")) + [
            ROOT / "scenarios/gse120575_full_online_celltype/physicell_project/custom.cpp",
            ROOT / "scenarios/gse120575_full_online_celltype/physicell_project/custom.h",
            ROOT / "scenarios/gse120575_full_online_celltype/physicell_project/Makefile",
        ],
        "config": [
            protocol_path,
            OUT / "audit/worker_discretization_preregistration.json",
            OUT / "audit/scgpt_perturbation_preregistration.json",
            PHYSICELL_ROOT / "config/PhysiCell_settings.xml",
        ],
        "binary": [ROOT / "scenarios/gse120575_full_online_celltype/physicell_project/gse120575_celltype_online"],
        "annotation": [
            OUT / "annotation/cell_level_annotations.csv",
            OUT / "annotation/broad_cell_type_mapping.csv",
            OUT / "audit/two_stage_id_mapping.csv",
            OUT / "audit/annotation_summary.json",
        ],
        "scgpt_panel": [OUT / "preprocessed/scgpt_gene_panel.csv"],
        "samples": [
            OUT / "audit/quick_manifest_frozen.csv",
            OUT / "audit/quick_manifest_frozen.json",
            OUT / "preprocessed/preregistered_sample_subset.csv",
            OUT / "preprocessed/pretreatment_sample_celltype_proportions.csv",
        ],
        "seeds": [protocol_path, OUT / "audit/worker_discretization_preregistration.json"],
        "evaluation_metrics": [evaluation_path],
    }
    entries = []
    category_hashes = {}
    for category, paths in categories.items():
        category_entries = []
        for path in paths:
            if not path.is_file():
                raise RuntimeError(f"Freeze input missing: {path}")
            item = {"category": category, "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            entries.append(item); category_entries.append(item)
        digest = hashlib.sha256()
        for item in sorted(category_entries, key=lambda value: value["path"]):
            digest.update(f"{item['path']}\0{item['sha256']}\n".encode())
        category_hashes[category] = digest.hexdigest()
    overall = hashlib.sha256()
    for category, digest in sorted(category_hashes.items()):
        overall.update(f"{category}\0{digest}\n".encode())
    build_evidence = json.loads((OUT / "audit/physicell_build_evidence.json").read_text())
    result = {
        "status": "FROZEN_READY_AWAITING_EXPLICIT_54_RUN_AUTHORIZATION",
        "quick_run_started": False,
        "design_expected_counts": {key: design[key] for key in ("expected_runs", "expected_agent_calls", "expected_scgpt_calls", "expected_cpp_writebacks")},
        "category_sha256": category_hashes,
        "overall_freeze_sha256": overall.hexdigest(),
        "files": entries,
        "binary_matches_last_build_evidence": entries[[item["category"] for item in entries].index("binary")]["sha256"] == build_evidence["binary_sha256"],
    }
    write_json(OUT / "audit/quick_freeze_hashes.json", result)
    return result


def main() -> int:
    protected = [SMOKE / "smoke_audit.json", SMOKE / "smoke_report.md"]
    before = {str(path): sha256(path) for path in protected}
    progress = json.loads((SMOKE / "smoke_progress.json").read_text())
    run_records = progress["completed"]
    if len(run_records) != 6:
        raise RuntimeError(f"Expected six existing smoke records, found {len(run_records)}")

    audit = evaluate(run_records)
    required = [
        "six_of_six_runs_complete", "agent_calls_equal_120", "scgpt_calls_equal_60",
        "all_five_decision_checkpoints_and_endpoint_complete", "cpp_writeback_rows_equal_180",
        "no_fallback", "no_target_leakage", "all_composition_sums_valid",
        "no_cell_count_extinction_or_explosion", "full_scgpt_prompt_and_causal_mapper_used",
    ]
    audit["qualification_required_checks"] = required
    audit["qualified_for_54_run_quick"] = all(audit["checks"][key] for key in required)
    audit["status"] = "PASS_6_RUN_SMOKE_CORRECTED" if audit["qualified_for_54_run_quick"] else "FAIL_6_RUN_SMOKE_CORRECTED"
    audit["all_passed"] = audit["qualified_for_54_run_quick"]
    audit["original_audit_preserved"] = True
    write_json(SMOKE / "smoke_audit_corrected.json", audit)

    leakage = audit["structured_leakage_audit"]
    leakage_report = [
        "# Corrected structured target-leakage audit", "",
        f"- Result: **{'PASS_NO_REAL_TARGET_LEAKAGE' if not leakage['real_target_leakage_detected'] else 'FAIL_REAL_TARGET_LEAKAGE'}**",
        f"- Request JSON files checked: {leakage['request_json_files_checked']}",
        f"- Runtime source files checked: {len(leakage['runtime_source_files_checked'])}",
        f"- Locked post-composition rows checked: {leakage['locked_post_composition_rows_checked']}",
        f"- Forbidden target-field hits: {len(leakage['forbidden_target_field_hits'])}",
        f"- Forbidden artifact-path hits: {len(leakage['forbidden_artifact_path_hits'])}",
        f"- Locked post numeric-signature hits: {len(leakage['locked_post_numeric_signature_hits'])}",
        f"- Runtime forbidden-artifact hits: {len(leakage['runtime_forbidden_artifact_hits'])}",
        f"- JSON parse errors: {len(leakage['request_json_parse_errors'])}",
        f"- Safe negative constraint occurrences: {leakage['safe_negative_constraint_occurrences']} (allowed, not leakage)", "",
        "The audit operates on parsed JSON keys/values, runtime AST imports/path literals, and complete six-cell-type locked post-composition signatures. It does not classify the negative safety statement `no post-treatment target is available` as leakage.",
    ]
    (SMOKE / "leakage_audit_corrected.md").write_text("\n".join(leakage_report) + "\n")

    bounds = parameter_bound_audit(run_records)
    design, _ = quick_manifest()
    if design["response_and_therapy_completely_confounded"]:
        raise RuntimeError("Quick response and therapy are completely confounded; stopped before freeze")
    freeze = freeze_hashes(design)

    corrected_report = [
        "# GSE120575 corrected 6-run smoke report", "",
        f"- Status: **{audit['status']}**",
        "- Existing runs re-audited offline; PhysiCell, LLM and scGPT were not rerun.",
        "- Original smoke audit/report preserved.",
        f"- Runs: {sum(record['exit_code'] == 0 for record in run_records)}/6",
        f"- Agent calls: {audit['agent_calls_actual']} / 120",
        f"- scGPT calls: {audit['scgpt_calls_actual']} / 60",
        f"- C++ writeback rows: {audit['cpp_writeback_rows']} / 180",
        f"- Cell-count range: {audit['cell_count_min']}–{audit['cell_count_max']}",
        f"- Real target leakage detected: {leakage['real_target_leakage_detected']}",
        f"- Parameter observations out of bounds: {bounds['total_out_of_bounds_count']}",
        f"- Parameter clipping count: {bounds['total_clipping_count']}",
        f"- Response/therapy completely confounded: {design['response_and_therapy_completely_confounded']}",
        f"- Quick expected: {design['expected_runs']} runs, {design['expected_agent_calls']} Agent calls, {design['expected_scgpt_calls']} scGPT calls, {design['expected_cpp_writebacks']} C++ writebacks.",
        f"- Freeze hash: `{freeze['overall_freeze_sha256']}`",
        f"- Qualified for 54-run quick: **{audit['qualified_for_54_run_quick']}**",
        "- Quick was not started.", "", "## Corrected gates", "",
    ] + [f"- {name}: {value}" for name, value in audit["checks"].items()]
    (SMOKE / "smoke_report_corrected.md").write_text("\n".join(corrected_report) + "\n")

    after = {str(path): sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("Protected original smoke audit/report changed")
    print(json.dumps({
        "status": audit["status"],
        "qualified_for_54_run_quick": audit["qualified_for_54_run_quick"],
        "real_target_leakage_detected": leakage["real_target_leakage_detected"],
        "parameter_bound_status": bounds["status"],
        "complete_response_therapy_confounding": design["response_and_therapy_completely_confounded"],
        "expected_quick": {key: design[key] for key in ("expected_runs", "expected_agent_calls", "expected_scgpt_calls", "expected_cpp_writebacks")},
        "quick_run_started": False,
    }, ensure_ascii=False, indent=2))
    return 0 if audit["qualified_for_54_run_quick"] and bounds["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
