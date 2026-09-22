#!/usr/bin/env python3
"""Static safety gates for independent Population Actuator V2.1."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/gse120575_full_online_celltype_v21"
P = ROOT / "scenarios/gse120575_full_online_celltype_v21/physicell_project"
O = ROOT / "outputs/GSE120575_full_online_celltype_v21"
controller = (S / "controllers_v21.py").read_text()
bridge = (S / "online_inference_v21.py").read_text()
cpp = (P / "custom.cpp").read_text()

checks = {
    "independent_v21_paths": all("full_online_celltype_v21" in str(x) for x in (S, P, O)),
    "three_budget_dimensions": all(x in controller for x in (
        "intrinsic_population_budget", "external_recruitment_budget", "clearance_budget"
    )),
    "trusted_formula_not_direct_llm_budget": '"direct_llm_budget_used": False' in controller,
    "same_mapper_all_models": "trusted_budgets(validated, model" in bridge,
    "external_uses_total_live": "q.external_recruitment_budget*current_total" in cpp,
    "intrinsic_uses_current_type": "q.intrinsic_budget*current" in cpp,
    "clearance_uses_current_type": "q.clearance_budget*current" in cpp,
    "zero_intrinsic_cannot_divide": "if(current==0&&divide>0)" in cpp,
    "no_minimum_one_recruitment": "std::max(1" not in cpp,
    "three_residual_carries": all(x in cpp for x in (
        "intrinsic_residual", "recruitment_residual", "clearance_residual"
    )),
    "split_event_sources": all(x in cpp for x in (
        "baseline_divisions", "actuator_divisions", "baseline_apoptosis",
        "actuator_apoptosis", "actuator_recruitment", "actuator_clearance"
    )),
    "no_mixed_v2_event_audit": "v2_interval_event_audit.csv" not in cpp,
    "explicit_conservation_error": "conservation_error" in cpp,
    "baseline_rates_not_actuator_rates": "BASE_BIRTH[index];" in cpp and "BASE_DEATH[index];" in cpp,
    "interval_audits": "v21_budget_audit.csv" in cpp and "v21_interval_event_audit.csv" in cpp,
    "normalized_time_label": "normalized_internal_minutes" in cpp,
    "runtime_no_post_artifact": "posttreatment" not in controller + bridge and "target_composition" not in controller + bridge,
    "no_celltype_or_sample_special_rule": "Plasma cell" not in controller and "Pre_P1" not in controller,
}
status = "PASS_V21_STATIC_VALIDATION" if all(checks.values()) else "FAIL_V21_STATIC_VALIDATION"
files = sorted(list(S.glob("*.py")) + [P / "custom.cpp", P / "custom.h", P / "Makefile", P / "zero_basis_actuator_test.cpp"])
result = {
    "status": status,
    "checks": checks,
    "files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
    "V1_modified_by_validator": False,
    "V2_modified_by_validator": False,
}
(O / "audit").mkdir(parents=True, exist_ok=True)
(O / "audit/static_validation_v21.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if status.startswith("PASS") else 2)
