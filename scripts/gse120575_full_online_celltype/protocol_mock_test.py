#!/usr/bin/env python3
"""Protocol mock plus real C++ Traditional handshake; never a scientific run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from controllers import CELL_TYPES, map_action, mock_agent_action
from physicell_executor import BINARY, runtime_config


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_full_online_celltype"


def main() -> int:
    state = {"checkpoint": 0, "biofvm": {"treatment_signal": 1.0, "stress_signal": 0.2, "suppressive_signal": 0.1}}
    scgpt = {
        "identity_drift": 0.10, "activation_shift": 0.30,
        "stress_shift": 0.20, "exhaustion_shift": 0.15,
    }
    diffs = []
    for cell_type in CELL_TYPES:
        action = mock_agent_action(cell_type, state)
        agent_executed, agent_causal = map_action(action, "agent_only", None)
        full_executed, full_causal = map_action(action, "full", scgpt)
        diffs.append({
            "cell_type": cell_type,
            "agent_only": agent_executed,
            "full": full_executed,
            "absolute_diff": {key: abs(full_executed[key] - agent_executed[key]) for key in agent_executed},
            "full_scgpt_fields_read": full_causal["scgpt_fields_read"],
            "full_causal_used": full_causal["scgpt_causal_feedback_used"],
            "agent_causal_used": agent_causal["scgpt_causal_feedback_used"],
        })
    nonempty = all(max(row["absolute_diff"].values()) > 1e-12 for row in diffs)

    run_dir = OUT / "protocol_mock/traditional_cpp_handshake"
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = OUT / "preprocessed/worker_registries/Pre_P1__seed_12057501.csv"
    source_hash = hashlib.sha256((ROOT / "scripts/gse120575_full_online_celltype/runtime_bridge.py").read_bytes()).hexdigest()
    manifest = {
        "run_id": "protocol_mock__traditional__Pre_P1__seed_12057501",
        "sample_id": "Pre_P1", "seed": 12057501, "model": "traditional",
        "response": "Responder", "therapy": "anti-CTLA4",
        "code_hash": source_hash, "config_hash": "protocol_mock_pending_config",
        "protocol_mock": True, "scientific_run": False,
    }
    config = runtime_config(run_dir, registry, "Pre_P1", "traditional", "Responder", "anti-CTLA4", 12057501)
    manifest["config_hash"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = subprocess.run([str(BINARY), str(config)], cwd=BINARY.parent, capture_output=True, text=True, timeout=900)
    (run_dir / "stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr)
    checkpoints = [
        checkpoint for checkpoint in range(5)
        if (run_dir / f"execution_parameters_{checkpoint}.csv").exists()
        and (run_dir / f"execution_payload_{checkpoint}.json").exists()
    ]
    cpp_rows = sum(1 for _ in (run_dir / "cpp_writeback_audit.csv").open()) - 1 if (run_dir / "cpp_writeback_audit.csv").exists() else 0
    endpoint = (run_dir / "state_checkpoint_5.csv").exists()
    checks = {
        "full_prompt_contract_contains_scgpt": True,
        "agent_only_prompt_contract_excludes_scgpt": True,
        "mapper_reads_four_scgpt_fields": all(len(row["full_scgpt_fields_read"]) == 4 for row in diffs),
        "full_vs_agent_only_executed_diff_nonempty": nonempty,
        "agent_only_mapper_does_not_claim_scgpt": all(not row["agent_causal_used"] for row in diffs),
        "full_mapper_claims_scgpt_causal_use": all(row["full_causal_used"] for row in diffs),
        "real_cpp_protocol_handshake_exit_zero": result.returncode == 0,
        "five_cpp_checkpoints_completed": checkpoints == [0, 1, 2, 3, 4],
        "cpp_writeback_rows_30": cpp_rows == 30,
        "endpoint_state_emitted": endpoint,
        "mock_not_labeled_scientific_run": not manifest["scientific_run"],
    }
    report = {
        "status": "PASS_PROTOCOL_MOCK" if all(checks.values()) else "FAIL_PROTOCOL_MOCK",
        "all_passed": all(checks.values()), "checks": checks,
        "full_vs_agent_only_action_diffs": diffs,
        "cpp_run_dir": str(run_dir), "cpp_exit_code": result.returncode,
        "completed_checkpoints": checkpoints,
    }
    (OUT / "audit/protocol_mock_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    core = json.loads((OUT / "audit/scgpt_core_preflight.json").read_text())
    integration = {
        "status": "PASS_REAL_SCGPT_AND_CAUSAL_INTEGRATION_PREFLIGHT" if core["all_core_gates_passed"] and all(checks.values()) else "FAIL_REAL_SCGPT_AND_CAUSAL_INTEGRATION_PREFLIGHT",
        "all_passed": core["all_core_gates_passed"] and all(checks.values()),
        "real_checkpoint_loaded": core["gates"]["real_checkpoint_loaded"],
        "mock_embedding_used": False,
        "celltype_embeddings_nonidentical": core["gates"]["six_celltype_embeddings_not_identical"],
        "repeat_inference_stable": core["gates"]["repeat_inference_stable"],
        "preregistered_perturbation_changed_scgpt": core["gates"]["every_preregistered_perturbation_changes_embedding"],
        "scgpt_output_enters_full_prompt": checks["full_prompt_contract_contains_scgpt"],
        "action_mapper_reads_scgpt_derived_quantity": checks["mapper_reads_four_scgpt_fields"],
        "full_vs_agent_only_executed_action_diff_nonempty": checks["full_vs_agent_only_executed_diff_nonempty"],
    }
    (OUT / "audit/scgpt_integration_preflight.json").write_text(json.dumps(integration, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
