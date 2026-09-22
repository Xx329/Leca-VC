#!/usr/bin/env python3
"""Local, network-free qualification tests for the de-identification firewall."""

from __future__ import annotations

import json
from pathlib import Path

from prompt_firewall import (
    PromptFirewallError,
    assert_clean,
    find_forbidden,
    sanitize_gse120575_payload,
    sanitize_gse2565_request,
    sanitize_gse267904_payload,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/LecaVC_prompt_contamination_audit_v1/static_qualification"


def must_reject(text: str) -> None:
    try:
        assert_clean(text, label="negative control")
    except PromptFirewallError:
        return
    raise AssertionError(f"firewall accepted forbidden text: {text}")


def main() -> None:
    for text in (
        "GSE120575", "GSM8281387", "Pre_P24", "Non-responder", "anti-PD1",
        "phosgene", "bleomycin", "melanoma", "pulmonary fibrosis", "d21",
        "fibrosis", "Sade-Feldman", "held-out target",
    ):
        must_reject(text)

    p120 = sanitize_gse120575_payload(
        {
            "experiment": "GSE120575 historical envelope",
            "sample_id": "Pre_P24",
            "known_response_condition": "Responder",
            "therapy": "anti-PD1",
            "normalized_treatment_checkpoint": 40,
            "cell_type": "T cell",
            "current_physicell_state": {"live_cells": 100, "treatment_signal": 0.4},
            "current_expression_representation": {"gene_count": 834, "mean": 1.2},
            "previous_memory": None,
            "previous_executed_action": None,
        }
    )
    base2565 = {
        "current_time": 12.0,
        "agent_id": "AT1",
        "biological_cell_type": "AT1",
        "worker_count": 5,
        "represented_abundance": 0.1,
        "current_worker_state_statistics": {"damage_memory": {"mean": 0.2}},
        "previous_agent_output": None,
        "previous_executed_action": None,
        "recent_state_trends": [{"from_minute": 480, "to_minute": 720, "mean_changes": {"damage_memory": -0.1}}],
        "cell_type_abundance_change": 0.0,
        "allowed_program_enum": ["repair_signal"],
        "biological_boundaries": ["no future real expression and no target time"],
        "stable_baseline_evidence": False,
        "repair_evidence": True,
        "normalized_program_evidence": {"repair_signal": 0.5},
        "evidence_scale": "0 to 1",
        "cell_type_decision_guidance": "Use compatible repair evidence.",
        "decision_semantics": {"requirements": ["Use supplied evidence."]},
    }
    p2565, step = sanitize_gse2565_request(
        base2565,
        {"minute": 720, "run_id": "contains_phosgene_but_trusted", "condition": "virtual_phosgene_injury"},
    )
    p267 = sanitize_gse267904_payload(
        {
            "experiment": "GSE267904",
            "interval": 1,
            "model_time": 125,
            "agent_id": "micro_01",
            "cell_type": "fibroblast_myofibroblast",
            "position": {"x": 1.0, "y": 2.0},
            "local_physicell_environment": {"fibrosis_signal": 0.1},
            "recent_memory": {},
            "calibration_only_biology": {
                "source": "GSE267904 d7 bleomycin",
                "allowed_prior": ["fibroblast cells can secrete extracellular-matrix-associated signals"],
                "heldout_real_d21_values_visible": False,
            },
            "allowed_output_schema": {
                "dominant_program": "short string",
                "communication_program": {"fibrosis_signal": "0.0-0.5"},
            },
        }
    )
    checks = {
        "negative_controls_rejected": True,
        "gse120575_clean": not find_forbidden(p120),
        "gse120575_outcome_fields_absent": not {
            "experiment", "sample_id", "known_response_condition", "therapy"
        }.intersection(p120),
        "gse2565_clean": not find_forbidden(p2565),
        "gse2565_normalized_step": step == 5 / 7,
        "gse2565_absolute_time_absent": "current_time" not in p2565,
        "gse2565_trend_minutes_absent": "minute" not in json.dumps(p2565["recent_state_trends"]),
        "gse267904_clean": not find_forbidden(p267),
        "gse267904_neutral_remodeling_field": (
            "remodeling_signal" in p267["local_environment"]
            and "fibrosis_signal" not in json.dumps(p267)
        ),
        "gse267904_stage_fields_absent": not {
            "experiment", "interval", "model_time", "calibration_only_biology"
        }.intersection(p267),
    }
    status = "PASS_STATIC_PROMPT_FIREWALL_QUALIFICATION" if all(checks.values()) else "FAIL"
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"status": status, "checks": checks, "example_payloads": {
        "GSE120575": p120, "GSE2565": p2565, "GSE267904": p267,
    }}
    (OUT / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
