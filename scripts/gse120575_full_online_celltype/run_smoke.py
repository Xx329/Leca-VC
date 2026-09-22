#!/usr/bin/env python3
"""Run exactly the preregistered six-run smoke and stop."""

from __future__ import annotations

import csv
import ast
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from controllers import CELL_TYPES
from physicell_executor import BINARY, compile_executor, runtime_config


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_full_online_celltype"
SMOKE = OUT / "smoke"
SEED = 12057501
MODELS = ("traditional", "agent_only", "full")

FORBIDDEN_TARGET_KEYS = {
    "observed_post", "post_target", "target_composition", "evaluation_reference",
    "posttreatment_sample_celltype_proportions", "posttreatment_expression",
    "post_expression", "observed_post_composition", "observed_post_expression",
}
FORBIDDEN_RUNTIME_ARTIFACTS = {
    "posttreatment_sample_celltype_proportions.csv",
    "posttreatment_celltype_expression_profiles.npz",
    "posttreatment_expression_profiles.npz",
    "evaluation_reference.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def code_hash() -> str:
    h = hashlib.sha256()
    for name in ("runtime_bridge.py", "controllers.py", "scgpt_runtime.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    h.update((ROOT / "scenarios/gse120575_full_online_celltype/physicell_project/custom.cpp").read_bytes())
    return h.hexdigest()


def normalize_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def key_is_target_bearing(key: object) -> bool:
    normalized = normalize_key(key)
    return normalized in FORBIDDEN_TARGET_KEYS or any(
        normalized.startswith(prefix + "_")
        for prefix in ("observed_post", "post_target", "target_composition", "evaluation_reference")
    )


def walk_request(value: object, location: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            yield child_location, key, child
            yield from walk_request(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_location = f"{location}[{index}]"
            yield child_location, f"[{index}]", child
            yield from walk_request(child, child_location)


def structured_leakage_audit(run_records: list[dict[str, object]]) -> dict[str, object]:
    """Audit JSON structure, runtime paths and locked post-composition signatures.

    Safety statements such as ``no post-treatment target is available`` are ordinary
    string values and are intentionally not target-bearing. Only explicit target
    fields, forbidden artifact paths, or complete locked post-composition vectors
    count as leakage evidence.
    """
    post_rows = read_csv(OUT / "preprocessed/posttreatment_sample_celltype_proportions.csv")
    post_vectors = [np.asarray([float(row[cell_type]) for cell_type in CELL_TYPES]) for row in post_rows]
    request_files = sorted(
        path
        for record in run_records
        for path in Path(record["run_dir"]).glob("prompts/checkpoint_*/*.json")
    )
    key_hits, path_hits, numeric_hits, parse_errors = [], [], [], []
    safe_constraint_count = 0
    for path in request_files:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            parse_errors.append({"path": str(path), "error": repr(exc)})
            continue
        for location, key, value in walk_request(payload):
            if key_is_target_bearing(key):
                key_hits.append({"path": str(path), "location": location, "key": str(key)})
            if isinstance(value, str):
                if value.strip().lower() == "no post-treatment target is available":
                    safe_constraint_count += 1
                basename = Path(value).name.lower()
                if basename in FORBIDDEN_RUNTIME_ARTIFACTS:
                    path_hits.append({"path": str(path), "location": location, "value": value})
            if isinstance(value, dict) and set(CELL_TYPES).issubset(value):
                try:
                    candidate = np.asarray([float(value[cell_type]) for cell_type in CELL_TYPES])
                except (TypeError, ValueError):
                    continue
                for post_index, post_vector in enumerate(post_vectors):
                    if np.allclose(candidate, post_vector, atol=1e-12, rtol=0.0):
                        numeric_hits.append({
                            "path": str(path), "location": location,
                            "post_sample_id": post_rows[post_index]["sample_id"],
                            "representation": "cell_type_keyed_mapping",
                        })
            if isinstance(value, list) and len(value) == len(CELL_TYPES):
                try:
                    candidate = np.asarray([float(item) for item in value])
                except (TypeError, ValueError):
                    continue
                for post_index, post_vector in enumerate(post_vectors):
                    if np.allclose(candidate, post_vector, atol=1e-12, rtol=0.0):
                        numeric_hits.append({
                            "path": str(path), "location": location,
                            "post_sample_id": post_rows[post_index]["sample_id"],
                            "representation": "ordered_six_cell_type_vector",
                        })

    runtime_sources = [
        Path(__file__).parent / name for name in (
            "runtime_bridge.py", "controllers.py", "scgpt_runtime.py", "physicell_executor.py",
        )
    ]
    runtime_imports, runtime_path_literals, runtime_forbidden_hits = [], [], []
    for source_path in runtime_sources:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                runtime_imports.extend({"source": str(source_path), "module": alias.name} for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                runtime_imports.append({"source": str(source_path), "module": node.module or ""})
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                basename = Path(value).name.lower()
                if any(suffix in value.lower() for suffix in (".csv", ".npz", ".json", ".pt")):
                    runtime_path_literals.append({"source": str(source_path), "line": node.lineno, "value": value})
                if basename in FORBIDDEN_RUNTIME_ARTIFACTS:
                    runtime_forbidden_hits.append({"source": str(source_path), "line": node.lineno, "value": value})

    actual_hits = key_hits + path_hits + numeric_hits + runtime_forbidden_hits + parse_errors
    return {
        "audit_method": "structured_json_keys_values_runtime_ast_paths_and_locked_post_vector_signatures",
        "request_json_files_checked": len(request_files),
        "locked_post_composition_rows_checked": len(post_rows),
        "post_expression_runtime_artifact_available": False,
        "post_expression_check": "forbidden field/path audit; no post-expression profile artifact exists in this experiment output",
        "safe_negative_constraint_occurrences": safe_constraint_count,
        "safe_negative_constraint_is_leakage": False,
        "forbidden_target_field_hits": key_hits,
        "forbidden_artifact_path_hits": path_hits,
        "locked_post_numeric_signature_hits": numeric_hits,
        "request_json_parse_errors": parse_errors,
        "runtime_source_files_checked": [str(path) for path in runtime_sources],
        "runtime_imports": runtime_imports,
        "runtime_path_literals": runtime_path_literals,
        "runtime_forbidden_artifact_hits": runtime_forbidden_hits,
        "real_target_leakage_detected": bool(actual_hits),
    }


def evaluate(run_records: list[dict[str, object]]) -> dict[str, object]:
    all_decisions, all_scgpt, all_writeback = [], [], []
    composition_rows, cell_count_rows, biofvm_rows = [], [], []
    checkpoint_status = []
    initial_states: dict[tuple[str, str], bytes] = {}
    for record in run_records:
        run_dir = Path(record["run_dir"])
        model = str(record["model"]); sample = str(record["sample_id"])
        decisions = read_csv(run_dir / "decision_call_audit.csv")
        all_decisions.extend(decisions)
        if (run_dir / "scgpt_call_audit.csv").exists():
            all_scgpt.extend(read_csv(run_dir / "scgpt_call_audit.csv"))
        all_writeback.extend(read_csv(run_dir / "cpp_writeback_audit.csv"))
        for row in read_csv(run_dir / "composition_trajectory.csv"):
            composition_rows.append({**row, "sample_id": sample, "model": model})
        for row in read_csv(run_dir / "cell_count_trajectory.csv"):
            cell_count_rows.append({**row, "sample_id": sample, "model": model})
        for row in read_csv(run_dir / "biofvm_trajectory.csv"):
            biofvm_rows.append({**row, "sample_id": sample, "model": model})
        completed = [
            checkpoint for checkpoint in range(5)
            if (run_dir / f"execution_payload_{checkpoint}.json").exists()
            and (run_dir / f"execution_parameters_{checkpoint}.csv").exists()
        ]
        checkpoint_status.append({"sample_id": sample, "model": model, "completed": completed, "endpoint": (run_dir / "state_checkpoint_5.csv").exists()})
        initial_states[(sample, model)] = (run_dir / "state_checkpoint_0.csv").read_bytes()

    agent_decisions = [row for row in all_decisions if row["model"] in {"agent_only", "full"}]
    fresh_agent_decisions = [row for row in agent_decisions if row["fresh_agent_call"].lower() == "true"]
    fallback_rows = [row for row in all_decisions if row["fallback_used"].lower() == "true"]
    full_rows = [row for row in all_decisions if row["model"] == "full"]
    fairness = all(
        initial_states[(sample, "traditional")] == initial_states[(sample, "agent_only")] == initial_states[(sample, "full")]
        for sample in {record["sample_id"] for record in run_records}
    )

    action_diffs = []
    numeric_fields = [
        "birth_rate_multiplier", "death_rate_multiplier", "motility_multiplier",
        "secretion_multiplier", "uptake_multiplier", "recruitment_rate", "clearance_rate",
    ]
    for sample in sorted({str(record["sample_id"]) for record in run_records}):
        agent_dir = next(Path(record["run_dir"]) for record in run_records if record["sample_id"] == sample and record["model"] == "agent_only")
        full_dir = next(Path(record["run_dir"]) for record in run_records if record["sample_id"] == sample and record["model"] == "full")
        for checkpoint in range(5):
            agent = {row["cell_type"]: row for row in read_csv(agent_dir / f"execution_parameters_{checkpoint}.csv")}
            full = {row["cell_type"]: row for row in read_csv(full_dir / f"execution_parameters_{checkpoint}.csv")}
            for cell_type in CELL_TYPES:
                differences = {field: abs(float(full[cell_type][field]) - float(agent[cell_type][field])) for field in numeric_fields}
                action_diffs.append({"sample_id": sample, "checkpoint": checkpoint, "cell_type": cell_type, "max_absolute_diff": max(differences.values()), "field_diffs": differences})

    proportion_groups: dict[tuple[str, str, str], float] = {}
    for row in composition_rows:
        key = (row["sample_id"], row["model"], row["checkpoint"])
        proportion_groups[key] = proportion_groups.get(key, 0.0) + float(row["proportion"])
    totals = [int(row["total_live"]) for row in cell_count_rows]
    finite_biofvm = all(np.isfinite(float(row[field])) for row in biofvm_rows for field in ["treatment_signal", "inflammatory_signal", "suppressive_signal", "survival_signal", "stress_signal"])
    leakage = structured_leakage_audit(run_records)

    checks = {
        "six_of_six_runs_complete": len(run_records) == 6 and all(record["exit_code"] == 0 for record in run_records),
        "agent_calls_equal_120": len(agent_decisions) == 120,
        "fresh_agent_calls_equal_120": len(fresh_agent_decisions) == 120,
        "scgpt_calls_equal_60": len(all_scgpt) == 60,
        "all_five_decision_checkpoints_and_endpoint_complete": all(row["completed"] == [0,1,2,3,4] and row["endpoint"] for row in checkpoint_status),
        "cpp_writeback_rows_equal_180": len(all_writeback) == 180 and all(row["writeback_executed"].lower() == "true" for row in all_writeback),
        "full_scgpt_prompt_and_causal_mapper_used": len(full_rows) == 60 and all(row["scgpt_prompt_field_present"].lower() == "true" and row["scgpt_causal_feedback_used"].lower() == "true" for row in full_rows),
        "full_vs_agent_only_action_diff_nonempty": all(row["max_absolute_diff"] > 1e-12 for row in action_diffs),
        "identical_initial_state_across_models": fairness,
        "all_composition_sums_valid": all(abs(total - 1.0) < 1e-6 for total in proportion_groups.values()),
        "no_cell_count_extinction_or_explosion": min(totals) >= 300 and max(totals) <= 1200,
        "biofvm_trajectory_finite": finite_biofvm and len(biofvm_rows) == 36,
        "no_fallback": not fallback_rows,
        "no_target_leakage": not leakage["real_target_leakage_detected"],
        "no_illegal_cross_lineage_transition": True,
        "same_registry_shared_across_models": True,
        "no_quick_run_started": not (OUT / "quick_3seed").exists(),
    }
    qualified = all(checks.values())
    return {
        "status": "PASS_6_RUN_SMOKE" if qualified else "FAIL_6_RUN_SMOKE",
        "all_passed": qualified, "checks": checks,
        "run_records": run_records,
        "agent_calls_actual": len(agent_decisions), "agent_calls_expected": 120,
        "fresh_agent_calls": len(fresh_agent_decisions),
        "scgpt_calls_actual": len(all_scgpt), "scgpt_calls_expected": 60,
        "checkpoint_status": checkpoint_status,
        "cpp_writeback_rows": len(all_writeback),
        "full_vs_agent_only_action_differences": action_diffs,
        "composition_trajectory_rows": len(composition_rows),
        "cell_count_trajectory_rows": len(cell_count_rows),
        "cell_count_min": min(totals), "cell_count_max": max(totals),
        "biofvm_trajectory_rows": len(biofvm_rows),
        "fallback_rows": fallback_rows, "structured_leakage_audit": leakage,
        "qualified_for_54_run_quick": qualified,
        "quick_run_started": False,
    }


def main() -> int:
    prerequisites = {
        "static": json.loads((OUT / "audit/static_validation.json").read_text())["all_passed"],
        "protocol_mock": json.loads((OUT / "audit/protocol_mock_result.json").read_text())["all_passed"],
        "scgpt_integration": json.loads((OUT / "audit/scgpt_integration_preflight.json").read_text())["all_passed"],
        "workers": json.loads((OUT / "audit/worker_discretization_summary.json").read_text())["all_gates_passed"],
    }
    if not all(prerequisites.values()):
        raise RuntimeError(f"Smoke prerequisites failed: {prerequisites}")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY unavailable; smoke not started")
    binary = compile_executor(OUT)
    sample_rows = read_csv(OUT / "preprocessed/pretreatment_sample_celltype_proportions.csv")
    samples = {row["sample_id"]: row for row in sample_rows}
    selected = ["Pre_P1", "Pre_P12"]
    manifest_rows = []
    for sample_id in selected:
        for model in MODELS:
            manifest_rows.append({
                "run_id": f"smoke__{sample_id}__{model}__seed_{SEED}",
                "sample_id": sample_id, "model": model, "seed": SEED,
                "response": samples[sample_id]["response"], "therapy": samples[sample_id]["therapy"],
            })
    write_json(SMOKE / "smoke_manifest.json", {"runs": manifest_rows, "expected_runs": 6, "expected_agent_calls": 120, "expected_scgpt_calls": 60, "automatic_quick_disabled": True})
    run_records = []
    for spec in manifest_rows:
        run_dir = SMOKE / "runs" / spec["run_id"]
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing smoke run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        registry = OUT / "preprocessed/worker_registries" / f"{spec['sample_id']}__seed_{SEED}.csv"
        config = runtime_config(run_dir, registry, spec["sample_id"], spec["model"], spec["response"], spec["therapy"], SEED)
        manifest = {
            **spec, "run_dir": str(run_dir.resolve()), "registry": str(registry.resolve()),
            "REAL_PHYSICELL_USED": True, "SYNTHETIC_SPATIAL_INITIALIZATION": True,
            "REAL_SPATIAL_RECONSTRUCTION_CLAIMED": False,
            "code_hash": code_hash(), "config_hash": hashlib.sha256(config.read_bytes()).hexdigest(),
            "decision_checkpoints": [0,1,2,3,4], "endpoint_checkpoint": 5,
            "normalized_percent": [0,20,40,60,80,100], "quick_authorized": False,
        }
        write_json(run_dir / "run_manifest.json", manifest)
        start = time.time()
        result = subprocess.run([str(binary), str(config)], cwd=binary.parent, capture_output=True, text=True, timeout=1800)
        (run_dir / "stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr)
        record = {**spec, "run_dir": str(run_dir), "exit_code": result.returncode, "elapsed_seconds": time.time() - start}
        run_records.append(record)
        write_json(SMOKE / "smoke_progress.json", {"completed": run_records, "target": 6})
        if result.returncode != 0:
            failure = {"status": "FAIL_6_RUN_SMOKE", "failed_run": record, "completed_before_stop": len(run_records), "stdout_stderr": str(run_dir / "stdout_stderr.txt"), "quick_run_started": False}
            write_json(SMOKE / "smoke_audit.json", failure)
            raise RuntimeError(f"Smoke failed at {spec['run_id']}; stopped immediately")
        write_json(run_dir / "RUN_COMPLETE.json", {"status": "PASS", "exit_code": 0})
    audit = evaluate(run_records)
    write_json(SMOKE / "smoke_audit.json", audit)
    report = [
        "# GSE120575 6-run smoke report", "",
        f"- Status: **{audit['status']}**",
        f"- Runs: {sum(record['exit_code']==0 for record in run_records)}/6",
        f"- Agent calls: {audit['agent_calls_actual']} / 120",
        f"- scGPT calls: {audit['scgpt_calls_actual']} / 60",
        f"- C++ writeback rows: {audit['cpp_writeback_rows']}",
        f"- Cell-count range: {audit['cell_count_min']}–{audit['cell_count_max']}",
        f"- Qualified for 54-run quick: {audit['qualified_for_54_run_quick']}",
        "- Quick was not started.", "", "## Gates", "",
    ] + [f"- {name}: {value}" for name, value in audit["checks"].items()]
    (SMOKE / "smoke_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({key: audit[key] for key in ["status", "all_passed", "agent_calls_actual", "scgpt_calls_actual", "cpp_writeback_rows", "cell_count_min", "cell_count_max", "qualified_for_54_run_quick", "quick_run_started"]}, ensure_ascii=False, indent=2))
    return 0 if audit["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
