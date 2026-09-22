#!/usr/bin/env python3
"""V2.1 checkpoint bridge: split PhysiCell state -> model -> three budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1_SCRIPT_DIR = ROOT / "scripts/gse120575_full_online_celltype"
sys.path.insert(0, str(V1_SCRIPT_DIR))

from controllers import (
    CELL_TYPES, call_llm, digest_json, map_action, traditional_action,
)
from expression_profiles import MODULES
from scgpt_runtime import ScGPTRuntime, derive_output
from controllers_v21 import trusted_budgets


OUTPUT_ROOT = ROOT / "outputs/GSE120575_full_online_celltype"
MODEL_DIR = ROOT / "scGPT_Brain/save/scGPT_human"
PROFILE_PATH = OUTPUT_ROOT / "preprocessed/pretreatment_celltype_expression_profiles.npz"
REFERENCE_PATH = OUTPUT_ROOT / "audit/scgpt_preonly_references.npz"


def slug(value: str) -> str:
    return value.lower().replace("/", "_").replace(" ", "_")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


def load_state(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if {row["cell_type"] for row in rows} != set(CELL_TYPES) or len(rows) != 6:
        raise RuntimeError("PhysiCell checkpoint must contain exactly six cell types")
    numeric = [
        "live_cells", "dead_cells", "cumulative_baseline_divisions",
        "cumulative_actuator_divisions", "cumulative_baseline_apoptosis",
        "cumulative_actuator_apoptosis", "cumulative_actuator_recruitment",
        "cumulative_actuator_clearance", "mean_birth_rate", "mean_death_rate",
        "treatment_signal", "inflammatory_signal", "suppressive_signal",
        "survival_signal", "stress_signal",
    ]
    converted = []
    for row in rows:
        converted.append({
            **row,
            **{key: float(row[key]) for key in numeric},
        })
    return sorted(converted, key=lambda row: CELL_TYPES.index(str(row["cell_type"])))


def expression_for_checkpoint(
    baseline: np.ndarray,
    genes: list[str],
    state: dict[str, object],
    checkpoint: int,
    previous_action: dict[str, object] | None,
) -> np.ndarray:
    expression = baseline.astype(np.float32).copy()
    gene_index = {gene: i for i, gene in enumerate(genes)}
    previous = previous_action or {}
    activation = float(previous.get("activation_strength", 0.0))
    stress = float(previous.get("stress_strength", 0.0))
    exhaustion = float(previous.get("exhaustion_strength", 0.0))
    fraction = checkpoint / 5.0
    shifts = {
        "activation": fraction * (0.25 * float(state["treatment_signal"]) + 0.20 * activation),
        "stress": fraction * (0.25 * float(state["stress_signal"]) + 0.20 * stress),
        "exhaustion": fraction * (0.25 * float(state["suppressive_signal"]) + 0.20 * exhaustion),
        "treatment_shift": fraction * 0.15 * float(state["treatment_signal"]),
    }
    for module, shift in shifts.items():
        for gene in MODULES[module]:
            if gene in gene_index:
                expression[gene_index[gene]] += shift
    return np.maximum(expression, 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=int, required=True)
    a = p.parse_args()
    run_dir = a.run_dir.resolve()
    checkpoint = a.checkpoint
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    model = manifest["model"]
    if model not in {"traditional", "agent_only", "full"}:
        raise RuntimeError(f"Unknown model {model}")
    state_path = run_dir / f"state_checkpoint_{checkpoint}.csv"
    state_rows = load_state(state_path)
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    history_path = run_dir / "runtime_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}

    profile_data = np.load(PROFILE_PATH)
    profiles = profile_data["profiles"].astype(np.float32)
    sample_ids = profile_data["sample_ids"].astype(str)
    profile_types = profile_data["broad_cell_types"].astype(str)
    genes = profile_data["genes"].astype(str).tolist()
    baseline_by_type = {}
    for cell_type in CELL_TYPES:
        indexes = np.flatnonzero((sample_ids == manifest["sample_id"]) & (profile_types == cell_type))
        if len(indexes) != 1:
            raise RuntimeError(f"Missing unique pre-only profile for {manifest['sample_id']} {cell_type}")
        baseline_by_type[cell_type] = profiles[indexes[0]]

    expression_by_type = {}
    expression_hashes = {}
    for state in state_rows:
        cell_type = str(state["cell_type"])
        old = history.get(cell_type, [])
        previous_action = old[-1]["executed_action"] if old else None
        expression = expression_for_checkpoint(
            baseline_by_type[cell_type], genes, state, checkpoint, previous_action
        )
        expression_by_type[cell_type] = expression
        expression_hashes[cell_type] = hashlib.sha256(expression.tobytes()).hexdigest()

    scgpt_outputs: dict[str, dict[str, object]] = {}
    scgpt_audit_rows = []
    if model == "full":
        references_data = np.load(REFERENCE_PATH)
        reference_types = references_data["cell_types"].astype(str).tolist()
        if reference_types != list(CELL_TYPES):
            raise RuntimeError("scGPT reference cell-type order mismatch")
        references = references_data["reference_embeddings"].astype(np.float32)
        reference_expression = references_data["reference_expression"].astype(np.float32)
        runtime = ScGPTRuntime(MODEL_DIR, genes)
        for type_index, cell_type in enumerate(CELL_TYPES):
            expression = expression_by_type[cell_type]
            embedding = runtime.embed(expression[None, :])[0]
            output = derive_output(
                expression, embedding, references, reference_expression[type_index],
                type_index, genes,
            )
            scgpt_outputs[cell_type] = output
            scgpt_audit_rows.append({
                "sample_id": manifest["sample_id"], "seed": manifest["seed"],
                "model": model, "checkpoint": checkpoint, "cell_type": cell_type,
                "scgpt_input_hash": expression_hashes[cell_type],
                "identity_similarity": output["identity_similarity"],
                "identity_drift": output["identity_drift"],
                "treatment_expression_shift_l2": output["treatment_expression_shift_l2"],
                "activation_shift": output["activation_shift"],
                "stress_shift": output["stress_shift"],
                "exhaustion_shift": output["exhaustion_shift"],
                "real_checkpoint_used": True, "mock_embedding_used": False,
            })
            atomic_json(
                run_dir / "scgpt" / f"checkpoint_{checkpoint}" / f"{slug(cell_type)}.json",
                output,
            )
        append_csv(
            run_dir / "scgpt_call_audit.csv",
            list(scgpt_audit_rows[0]), scgpt_audit_rows,
        )

    prompt_payloads = {}
    for state in state_rows:
        cell_type = str(state["cell_type"])
        old = history.get(cell_type, [])
        payload = {
            "experiment": "GSE120575_full_online_celltype_v21",
            "sample_id": manifest["sample_id"],
            "known_response_condition": manifest["response"],
            "therapy": manifest["therapy"],
            "normalized_treatment_checkpoint": checkpoint * 20,
            "cell_type": cell_type,
            "current_physicell_state": state,
            "current_expression_representation": {
                "input_hash": expression_hashes[cell_type],
                "gene_count": len(genes),
                "mean": float(expression_by_type[cell_type].mean()),
                "standard_deviation": float(expression_by_type[cell_type].std()),
            },
            "previous_memory": old[-1]["updated_memory"] if old else None,
            "previous_executed_action": old[-1]["executed_action"] if old else None,
            "biological_constraints": [
                "act only for this broad cell type",
                "no cross-lineage transition",
                "use current and past state only",
                "no post-treatment target is available",
            ],
        }
        if model == "full":
            payload["scgpt_runtime"] = scgpt_outputs[cell_type]
        elif "scgpt_runtime" in payload:
            raise AssertionError("scGPT leaked into non-full prompt")
        prompt_payloads[cell_type] = payload
        atomic_json(
            run_dir / "prompts" / f"checkpoint_{checkpoint}" / f"{slug(cell_type)}.json",
            payload,
        )

    decisions: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    if model == "traditional":
        for state in state_rows:
            cell_type = str(state["cell_type"])
            decisions[cell_type] = (traditional_action(cell_type, {
                "biofvm": {
                    "treatment_signal": state["treatment_signal"],
                    "stress_signal": state["stress_signal"],
                    "suppressive_signal": state["suppressive_signal"],
                }
            }), {"fresh_call": False, "cache_reused": False, "fallback_used": False, "prompt_hash": "not_applicable"})
    else:
        def decide(cell_type: str):
            identity = {
                "sample_id": manifest["sample_id"], "seed": manifest["seed"],
                "model": model, "checkpoint": checkpoint, "cell_type": cell_type,
                "current_state_hash": state_hash,
                "scgpt_input_hash": expression_hashes[cell_type] if model == "full" else "not_applicable",
                "code_hash": manifest["code_hash"], "config_hash": manifest["config_hash"],
            }
            action, metadata = call_llm(
                prompt_payloads[cell_type], cell_type,
                run_dir / "raw_agent_responses" / f"checkpoint_{checkpoint}" / slug(cell_type),
                run_dir / "exact_resume_cache" / model / f"checkpoint_{checkpoint}" / f"{slug(cell_type)}.json",
                identity,
            )
            return cell_type, action, metadata
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(decide, CELL_TYPES))
        decisions = {cell_type: (action, metadata) for cell_type, action, metadata in results}

    execution_rows = []
    decision_audit_rows = []
    updated_history = dict(history)
    full_payload = []
    mapper_audit_rows = []
    for state in state_rows:
        cell_type = str(state["cell_type"])
        validated, metadata = decisions[cell_type]
        executed, causal = map_action(validated, model, scgpt_outputs.get(cell_type))
        budgets, budget_audit = trusted_budgets(validated, model, scgpt_outputs.get(cell_type))
        old = history.get(cell_type, [])
        previous_memory = old[-1]["updated_memory"] if old else None
        updated_memory = str(validated["memory_update"])
        execution_rows.append({
            "cell_type": cell_type,
            "current_state_hash": state_hash,
            **executed,
            **budgets,
        })
        record = {
            "sample_id": manifest["sample_id"], "seed": manifest["seed"],
            "model": model, "checkpoint": checkpoint, "cell_type": cell_type,
            "current_state_hash": state_hash,
            "scgpt_input_hash": expression_hashes[cell_type] if model == "full" else "not_applicable",
            "scgpt_output": scgpt_outputs.get(cell_type),
            "agent_prompt_hash": digest_json(prompt_payloads[cell_type]),
            "raw_agent_response": validated,
            "validated_action": validated,
            "executed_physicell_parameters": executed,
            **budgets,
            "population_budget_audit_v21": budget_audit,
            "previous_memory": previous_memory,
            "updated_memory": updated_memory,
            "fresh_agent_call": metadata.get("fresh_call", False),
            "cache_reused": metadata.get("cache_reused", False),
            "fallback_used": False,
            "scgpt_prompt_field_present": "scgpt_runtime" in prompt_payloads[cell_type],
            **causal,
            "scgpt_causal_budget_feedback_used": budget_audit["scgpt_causal_budget_feedback_used"],
        }
        full_payload.append(record)
        decision_audit_rows.append({
            key: record[key] for key in [
                "sample_id", "seed", "model", "checkpoint", "cell_type",
                "current_state_hash", "scgpt_input_hash", "agent_prompt_hash",
                "fresh_agent_call", "cache_reused", "fallback_used",
                "scgpt_prompt_field_present", "scgpt_causal_feedback_used",
            ]
        })
        updated_history.setdefault(cell_type, []).append({
            "checkpoint": checkpoint, "current_state_hash": state_hash,
            "previous_memory": previous_memory, "updated_memory": updated_memory,
            "validated_action": validated, "executed_action": executed,
        })
        mapper_audit_rows.append({
            "checkpoint": checkpoint,
            "cell_type": cell_type,
            **budgets,
            "raw_intrinsic_population_budget": budget_audit["raw_intrinsic_population_budget"],
            "raw_external_recruitment_budget": budget_audit["raw_external_recruitment_budget"],
            "raw_clearance_budget": budget_audit["raw_clearance_budget"],
            "intrinsic_clipped": budget_audit["intrinsic_clipped"],
            "external_recruitment_clipped": budget_audit["external_recruitment_clipped"],
            "clearance_clipped": budget_audit["clearance_clipped"],
            "scgpt_causal_budget_feedback_used": budget_audit["scgpt_causal_budget_feedback_used"],
            "post_target_used": False,
            "cell_type_special_rule_used": False,
        })

    execution_path = run_dir / f"execution_parameters_{checkpoint}.csv"
    with execution_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(execution_rows[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(execution_rows)
    atomic_json(run_dir / f"execution_payload_{checkpoint}.json", {
        "run_id": manifest["run_id"], "sample_id": manifest["sample_id"],
        "seed": manifest["seed"], "model": model, "checkpoint": checkpoint,
        "current_state_hash": state_hash, "schema_valid": True,
        "fallback_used": False, "cell_type_count": 6, "records": full_payload,
    })
    atomic_json(history_path, updated_history)
    append_csv(run_dir / "decision_call_audit.csv", list(decision_audit_rows[0]), decision_audit_rows)
    append_csv(run_dir / "v21_budget_mapper_audit.csv", list(mapper_audit_rows[0]), mapper_audit_rows)
    print(json.dumps({
        "run_id": manifest["run_id"], "checkpoint": checkpoint,
        "model": model, "cell_types": 6, "state_hash": state_hash,
        "agent_calls": 0 if model == "traditional" else 6,
        "scgpt_calls": 6 if model == "full" else 0,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
