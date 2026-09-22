#!/usr/bin/env python3
"""Static gates for the GSE120575 online cell-type benchmark."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_full_online_celltype"
HERE = Path(__file__).resolve().parent
CPP = ROOT / "scenarios/gse120575_full_online_celltype/physicell_project/custom.cpp"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    mapping = load(OUT / "audit/two_stage_mapping_summary.json")
    profiles = load(OUT / "preprocessed/expression_profile_manifest.json")
    scgpt = load(OUT / "audit/scgpt_core_preflight.json")
    workers = load(OUT / "audit/worker_discretization_summary.json")
    build = load(OUT / "audit/physicell_build_evidence.json")
    bridge = (HERE / "runtime_bridge.py").read_text()
    controllers = (HERE / "controllers.py").read_text()
    cpp = CPP.read_text()
    runtime_text = bridge + controllers + cpp
    checks = {
        "stage0_mapping_passed": mapping["status"] == "PASS_WITH_TWO_STAGE_DETERMINISTIC_ID_MAPPING",
        "pre_profiles_are_sample_by_type": profiles["profile_count"] == 114 and profiles["pre_sample_count"] == 19,
        "pre_profiles_exclude_post": profiles["pre_only"] and not profiles["post_cells_used"],
        "scgpt_panel_frozen_1200": profiles["panel_size"] == 1200,
        "real_scgpt_core_preflight_passed": scgpt["all_core_gates_passed"] and scgpt["gates"]["no_mock_embedding_used"],
        "worker_discretization_passed": workers["all_gates_passed"] and workers["worker_count"] == 600,
        "real_physicell_compiled": build["REAL_PHYSICELL_USED"] and Path(build["binary"]).exists(),
        "synthetic_not_reconstructed_spatial": build["SYNTHETIC_SPATIAL_INITIALIZATION"] and not build["REAL_SPATIAL_RECONSTRUCTION_CLAIMED"],
        "five_online_decision_checkpoints": "online_checkpoint(0)" in cpp and "double minutes[4]={2,4,6,8}" in cpp,
        "endpoint_read_without_decision": "emit_state(5)" in cpp and "online_checkpoint(5)" not in cpp,
        "cpp_birth_writeback": "transition_rate(0,0)=BASE_BIRTH[index]*q.birth" in cpp,
        "cpp_death_writeback": "death.rates[0]=BASE_DEATH[index]*q.death" in cpp,
        "cpp_recruitment_writeback": "create_cell(*cell_definitions_by_name[name])" in cpp,
        "cpp_clearance_writeback": "start_death(0)" in cpp,
        "cpp_biofvm_writeback": all(token in cpp for token in ["secretion_rates[inflammatory]", "uptake_rates[treatment]", "migration_speed"]),
        "no_cross_lineage_transformations": "transformation_rates" not in cpp,
        "no_global_tissue_coordinator": "Tissue Coordinator" not in runtime_text and "tissue_coordinator" not in runtime_text,
        "agent_only_prompt_excludes_scgpt": "if model == \"full\":" in bridge and "payload[\"scgpt_runtime\"]" in bridge,
        "full_prompt_includes_scgpt": "payload[\"scgpt_runtime\"] = scgpt_outputs[cell_type]" in bridge,
        "mapper_reads_scgpt_fields": all(token in controllers for token in ["identity_drift", "activation_shift", "stress_shift", "exhaustion_shift"]),
        "scgpt_changes_executed_parameters": "executed[\"birth_rate_multiplier\"] *= identity_gate" in controllers and "executed[\"death_rate_multiplier\"] *=" in controllers,
        "strict_no_fallback_llm": "raise RuntimeError" in controllers and "no fallback" in controllers,
        "exact_resume_scope_complete": all(token in bridge for token in ["sample_id", "seed", "model", "checkpoint", "cell_type", "current_state_hash", "scgpt_input_hash", "code_hash", "config_hash"]),
        "runtime_has_no_post_endpoint_loader": "posttreatment_sample_celltype_proportions" not in bridge + controllers,
        "runtime_audits_required_fields": all(token in bridge for token in ["previous_memory", "updated_memory", "raw_agent_response", "validated_action", "executed_physicell_parameters"]),
        "quick_not_implemented_as_automatic_followon": "run_quick_3seed" not in runtime_text,
    }
    result = {"status": "PASS_STATIC_VALIDATION" if all(checks.values()) else "FAIL_STATIC_VALIDATION", "all_passed": all(checks.values()), "checks": checks}
    (OUT / "audit/static_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
