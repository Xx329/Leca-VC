#!/usr/bin/env python3
"""Strict model-facing prompt firewall for Leca-VC contamination controls.

Trusted run metadata may still be retained by the orchestrator, but only the
objects returned by the dataset-specific sanitizers in this module may be sent
to an LLM.  The audit checks exact serialized messages, including retry
messages, rather than checking a larger trusted request envelope.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "LECAVC_DEIDENTIFIED_PROMPT_V1"

# These patterns intentionally cover identifiers, endpoint labels and the
# dataset-specific context combinations identified in the historical audit.
FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("geo_series", re.compile(r"\bGSE\s*[-_]?\s*\d+\b", re.I)),
    ("geo_sample", re.compile(r"\bGSM\s*[-_]?\s*\d+\b", re.I)),
    ("patient_sample", re.compile(r"\b(?:Pre|Post)[-_ ]?P\d+(?:[_-]\d+)?\b", re.I)),
    ("response_label", re.compile(r"\b(?:non[- ]?responder|responder)\b", re.I)),
    ("anti_pd1", re.compile(r"\banti[- ]?PD[- ]?1\b", re.I)),
    ("anti_ctla4", re.compile(r"\banti[- ]?CTLA[- ]?4\b", re.I)),
    ("phosgene", re.compile(r"\bphosgene\b", re.I)),
    ("bleomycin", re.compile(r"\b(?:bleomycin|bleo)\b", re.I)),
    ("melanoma", re.compile(r"\bmelanoma\b", re.I)),
    ("lung_injury", re.compile(r"\blung\s+injury\b", re.I)),
    ("fibrosis", re.compile(r"\bfibrosis\b", re.I)),
    ("immunotherapy", re.compile(r"\bimmunotherap(?:y|ies)\b", re.I)),
    ("study_author", re.compile(r"\bSade[- ]?Feldman\b", re.I)),
    ("absolute_day", re.compile(r"\b(?:d|day\s*)\s*(?:7|21)\b", re.I)),
    ("endpoint_wording", re.compile(r"\b(?:held[- ]?out|late[- ]?stage|future\s+(?:real\s+)?(?:target|expression|measurement))\b", re.I)),
)


class PromptFirewallError(RuntimeError):
    """Raised before an API request when model-visible text is not clean."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def find_forbidden(value: Any) -> list[dict[str, str]]:
    text = value if isinstance(value, str) else canonical_json(value)
    hits: list[dict[str, str]] = []
    for name, pattern in FORBIDDEN_PATTERNS:
        for match in pattern.finditer(text):
            hits.append({"rule": name, "match": match.group(0)})
    return hits


def assert_clean(value: Any, *, label: str) -> None:
    hits = find_forbidden(value)
    if hits:
        raise PromptFirewallError(f"{label} contains forbidden model-visible content: {hits}")


def assert_top_level_keys(payload: dict[str, Any], allowed: Iterable[str], dataset: str) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise PromptFirewallError(f"{dataset} model payload has non-allowlisted fields: {extra}")


def _clean_previous_action(value: Any) -> Any:
    """Keep numerical recurrence while excluding unconstrained free text."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PromptFirewallError("previous action must be a mapping or null")
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"brief_rationale", "evidence_summary", "memory_update", "reasoning_brief"}:
            continue
        if isinstance(item, dict):
            safe[key] = _clean_previous_action(item)
        elif isinstance(item, (int, float, bool, type(None), list, str)):
            safe[key] = item
    return safe


def sanitize_gse120575_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "normalized_treatment_checkpoint",
        "cell_type",
        "current_physicell_state",
        "current_expression_representation",
        "previous_memory",
        "previous_executed_action",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise PromptFirewallError(f"GSE120575 trusted payload missing fields: {missing}")
    result: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "normalized_step": float(payload["normalized_treatment_checkpoint"]) / 100.0,
        "cell_type": payload["cell_type"],
        "current_agent_state": payload["current_physicell_state"],
        "current_expression_summary": payload["current_expression_representation"],
        "previous_memory": payload["previous_memory"],
        "previous_executed_action": _clean_previous_action(payload["previous_executed_action"]),
        "biological_constraints": [
            "act only for the supplied cell type",
            "do not propose a cross-lineage transition",
            "use only the supplied current and recurrent state",
        ],
    }
    if "scgpt_runtime" in payload:
        result["representation_context"] = payload["scgpt_runtime"]
    assert_top_level_keys(
        result,
        {
            "protocol", "normalized_step", "cell_type", "current_agent_state",
            "current_expression_summary", "previous_memory", "previous_executed_action",
            "biological_constraints", "representation_context",
        },
        "GSE120575",
    )
    assert_clean(result, label="GSE120575 sanitized payload")
    return result


def sanitize_gse2565_request(model_request: dict[str, Any], trusted_request: dict[str, Any]) -> tuple[dict[str, Any], float]:
    minute = int(trusted_request["minute"])
    checkpoints = (0, 30, 60, 240, 480, 720, 1440, 2880)
    if minute not in checkpoints:
        raise PromptFirewallError(f"unexpected trusted checkpoint: {minute}")
    normalized_step = checkpoints.index(minute) / (len(checkpoints) - 1)
    result = dict(model_request)
    result.pop("current_time", None)
    normalized_trends = []
    for trend in result.get("recent_state_trends", []):
        if not isinstance(trend, dict):
            raise PromptFirewallError("GSE2565 trend must be an object")
        start = int(trend["from_minute"])
        end = int(trend["to_minute"])
        if start not in checkpoints or end not in checkpoints:
            raise PromptFirewallError("GSE2565 trend contains an unexpected physical checkpoint")
        normalized_trends.append(
            {
                "from_step": checkpoints.index(start) / (len(checkpoints) - 1),
                "to_step": checkpoints.index(end) / (len(checkpoints) - 1),
                "mean_changes": trend["mean_changes"],
            }
        )
    result["recent_state_trends"] = normalized_trends
    result["protocol"] = PROTOCOL_ID
    result["normalized_step"] = normalized_step
    result["previous_agent_output"] = _clean_previous_action(result.get("previous_agent_output"))
    result["biological_boundaries"] = [
        "use only the supplied current and recurrent state",
        "respect the supplied cell-type program constraints",
        "stable baseline states require bounded activity",
        "actionable local evidence requires a non-zero compatible response",
    ]
    allowed = {
        "protocol", "normalized_step", "agent_id", "biological_cell_type",
        "worker_count", "represented_abundance", "current_worker_state_statistics",
        "previous_agent_output", "previous_executed_action", "recent_state_trends",
        "cell_type_abundance_change", "allowed_program_enum", "biological_boundaries",
        "stable_baseline_evidence", "repair_evidence", "normalized_program_evidence",
        "evidence_scale", "cell_type_decision_guidance", "decision_semantics",
    }
    assert_top_level_keys(result, allowed, "GSE2565")
    assert_clean(result, label="GSE2565 sanitized payload")
    return result, normalized_step


def sanitize_gse267904_payload(payload: dict[str, Any]) -> dict[str, Any]:
    calibration = payload.get("calibration_only_biology", {})
    environment = dict(payload["local_physicell_environment"])
    if "fibrosis_signal" in environment:
        environment["remodeling_signal"] = environment.pop("fibrosis_signal")
    output_schema = json.loads(json.dumps(payload["allowed_output_schema"]))
    communication_schema = output_schema.get("communication_program", {})
    if isinstance(communication_schema, dict) and "fibrosis_signal" in communication_schema:
        communication_schema["remodeling_signal"] = communication_schema.pop("fibrosis_signal")
    priors = []
    for prior in calibration.get("allowed_prior", []):
        priors.append(re.sub(r"\bfibrosis\b", "matrix-remodeling", str(prior), flags=re.I))
    result: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "normalized_step": float(payload["interval"]),
        "agent_id": payload["agent_id"],
        "cell_type": payload["cell_type"],
        "local_position": payload["position"],
        "local_environment": environment,
        "recent_memory": payload.get("recent_memory", {}),
        "generic_cell_role_constraints": priors,
        "allowed_output_schema": output_schema,
    }
    assert_top_level_keys(
        result,
        {
            "protocol", "normalized_step", "agent_id", "cell_type", "local_position",
            "local_environment", "recent_memory", "generic_cell_role_constraints",
            "allowed_output_schema",
        },
        "GSE267904",
    )
    assert_clean(result, label="GSE267904 sanitized payload")
    return result


def restore_gse267904_execution_response(value: dict[str, Any]) -> dict[str, Any]:
    """Translate the neutral model-facing field back to the frozen executor field."""
    restored = json.loads(json.dumps(value))
    communication = restored.get("communication_program")
    if isinstance(communication, dict) and "remodeling_signal" in communication:
        communication["fibrosis_signal"] = communication.pop("remodeling_signal")
    return restored


def write_message_audit(
    directory: Path,
    messages: list[dict[str, str]],
    *,
    dataset_role: str,
    trusted_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    assert_clean(messages, label=f"{dataset_role} exact API messages")
    serialized = canonical_json(messages)
    audit = {
        "status": "PASS_MODEL_FACING_PROMPT_FIREWALL",
        "protocol": PROTOCOL_ID,
        "dataset_role": dataset_role,
        "message_sha256": sha256_text(serialized),
        "forbidden_hits": [],
        "trusted_envelope_sha256": sha256_text(canonical_json(trusted_envelope or {})),
        "trusted_envelope_sent_to_model": False,
    }
    (directory / "model_messages.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "prompt_firewall_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def validate_response_text(value: Any, *, dataset_role: str) -> None:
    """Prevent generated identifiers from re-entering recurrent model context."""
    assert_clean(value, label=f"{dataset_role} generated response")
