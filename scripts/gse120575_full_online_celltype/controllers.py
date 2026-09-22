#!/usr/bin/env python3
"""Frozen Traditional/LLM controllers and scGPT-causal action mapper."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib import request

import numpy as np


PROGRAMS = (
    "proliferation", "survival", "apoptosis", "recruitment", "clearance",
    "activation", "exhaustion", "inflammatory_secretion", "immune_suppression",
)
MULTIPLIERS = (
    "birth_rate", "death_rate", "motility", "secretion", "uptake", "recruitment_rate",
)
CELL_TYPES = (
    "B cell", "Plasma cell", "Monocyte/Macrophage",
    "Dendritic cell", "T cell", "NK cell",
)


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def template(cell_type: str) -> dict[str, object]:
    return {
        "cell_type": cell_type,
        "dominant_program": "survival",
        "program_strengths": {key: 0.0 for key in PROGRAMS},
        "phenotype_multipliers": {
            "birth_rate": 1.0, "death_rate": 1.0, "motility": 1.0,
            "secretion": 1.0, "uptake": 1.0, "recruitment_rate": 1.0,
        },
        "memory_update": "brief current-state memory",
        "confidence": 0.5,
        "evidence_summary": "brief evidence summary",
    }


def validate_action(action: dict[str, object], cell_type: str) -> dict[str, object]:
    expected = {
        "cell_type", "dominant_program", "program_strengths",
        "phenotype_multipliers", "memory_update", "confidence", "evidence_summary",
    }
    if not isinstance(action, dict) or set(action) != expected:
        raise ValueError("Agent top-level schema mismatch")
    if action["cell_type"] != cell_type:
        raise ValueError("Agent cell_type mismatch")
    if action["dominant_program"] not in PROGRAMS:
        raise ValueError("Invalid dominant_program")
    strengths = action["program_strengths"]
    multipliers = action["phenotype_multipliers"]
    if not isinstance(strengths, dict) or set(strengths) != set(PROGRAMS):
        raise ValueError("program_strength schema mismatch")
    if not isinstance(multipliers, dict) or set(multipliers) != set(MULTIPLIERS):
        raise ValueError("phenotype_multiplier schema mismatch")
    if not all(np.isfinite(float(v)) and 0 <= float(v) <= 1 for v in strengths.values()):
        raise ValueError("program strengths outside [0,1]")
    if not all(np.isfinite(float(v)) and 0.5 <= float(v) <= 1.5 for v in multipliers.values()):
        raise ValueError("phenotype multipliers outside [0.5,1.5]")
    if not np.isfinite(float(action["confidence"])) or not 0 <= float(action["confidence"]) <= 1:
        raise ValueError("confidence outside [0,1]")
    if not isinstance(action["memory_update"], str) or not isinstance(action["evidence_summary"], str):
        raise ValueError("memory/evidence must be strings")
    normalized = json.loads(json.dumps(action))
    normalized["program_strengths"] = {key: float(strengths[key]) for key in PROGRAMS}
    normalized["phenotype_multipliers"] = {key: float(multipliers[key]) for key in MULTIPLIERS}
    normalized["confidence"] = float(action["confidence"])
    return normalized


def traditional_action(cell_type: str, state: dict[str, object]) -> dict[str, object]:
    action = template(cell_type)
    treatment = float(state["biofvm"]["treatment_signal"])
    stress = float(state["biofvm"]["stress_signal"])
    suppressive = float(state["biofvm"]["suppressive_signal"])
    immune = cell_type in {"T cell", "NK cell", "Dendritic cell", "Monocyte/Macrophage"}
    action["dominant_program"] = "activation" if immune else "survival"
    action["program_strengths"].update({
        "survival": 0.45,
        "activation": min(1.0, treatment * (0.65 if immune else 0.15)),
        "apoptosis": min(1.0, 0.15 + 0.35 * stress),
        "exhaustion": min(1.0, 0.20 + 0.45 * suppressive) if cell_type == "T cell" else 0.05,
        "inflammatory_secretion": 0.35 if immune else 0.10,
        "recruitment": 0.20 if immune else 0.05,
        "clearance": 0.08,
        "proliferation": 0.20,
        "immune_suppression": 0.35 if cell_type == "Monocyte/Macrophage" else 0.05,
    })
    action["phenotype_multipliers"].update({
        "birth_rate": 0.95,
        "death_rate": 1.0 + 0.2 * stress,
        "motility": 1.15 if immune else 1.0,
        "secretion": 1.15 if immune else 1.0,
        "uptake": 1.0,
        "recruitment_rate": 1.10 if immune else 1.0,
    })
    action["memory_update"] = "traditional frozen controller"
    action["evidence_summary"] = "pre-registered rule based on current PhysiCell/BioFVM state"
    action["confidence"] = 1.0
    return validate_action(action, cell_type)


def mock_agent_action(cell_type: str, state: dict[str, object]) -> dict[str, object]:
    action = template(cell_type)
    index = CELL_TYPES.index(cell_type)
    action["dominant_program"] = "activation"
    action["program_strengths"].update({
        "proliferation": 0.20 + 0.02 * index,
        "survival": 0.45,
        "apoptosis": 0.15,
        "recruitment": 0.20,
        "clearance": 0.08,
        "activation": 0.35 + 0.03 * index,
        "exhaustion": 0.12,
        "inflammatory_secretion": 0.30,
        "immune_suppression": 0.10,
    })
    action["phenotype_multipliers"].update({
        "birth_rate": 0.90 + 0.02 * index,
        "death_rate": 1.05,
        "motility": 1.10,
        "secretion": 1.12,
        "uptake": 1.0,
        "recruitment_rate": 1.08,
    })
    action["memory_update"] = f"mock protocol state checkpoint {state['checkpoint']}"
    action["evidence_summary"] = "protocol mock only"
    return validate_action(action, cell_type)


def call_llm(
    prompt_payload: dict[str, object],
    cell_type: str,
    raw_dir: Path,
    cache_path: Path,
    cache_identity: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; no fallback")
    system = (
        "You are one biological cell-type Agent in an online immunotherapy simulation. "
        "Use only the current/past state supplied. Return one strict JSON object exactly matching "
        "the template. All program strengths are [0,1]; every phenotype multiplier is [0.5,1.5]. "
        "Do not invent cross-lineage transitions and do not include hidden reasoning. Template: "
        + json.dumps(template(cell_type), ensure_ascii=False, sort_keys=True)
    )
    prompt = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    prompt_hash = hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()
    expected_cache = {**cache_identity, "prompt_hash": prompt_hash, "model": "deepseek-chat"}
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if all(metadata.get(key) == value for key, value in expected_cache.items()):
            action = validate_action(json.loads(cache_path.read_text()), cell_type)
            return action, {**metadata, "fresh_call": False, "cache_reused": True}

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.joinpath("prompt_payload.json").write_text(json.dumps(prompt_payload, ensure_ascii=False, indent=2) + "\n")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(1, 5):
        content = None
        try:
            body = {
                "model": "deepseek-chat", "temperature": 0.1,
                "response_format": {"type": "json_object"}, "messages": messages,
            }
            req = request.Request(
                "https://api.deepseek.com/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=120) as response:
                raw = response.read()
            raw_dir.joinpath(f"attempt_{attempt}_raw.json").write_bytes(raw + b"\n")
            content = json.loads(raw)["choices"][0]["message"]["content"]
            raw_dir.joinpath(f"attempt_{attempt}_content.json").write_text(content + "\n")
            action = validate_action(json.loads(content), cell_type)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(action, ensure_ascii=False, indent=2) + "\n")
            metadata = {
                **expected_cache, "fresh_call": True, "cache_reused": False,
                "fallback_used": False, "attempt": attempt, "timestamp": time.time(),
                "response_hash": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            }
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            return action, metadata
        except Exception as exc:
            last_error = exc
            raw_dir.joinpath(f"attempt_{attempt}_error.txt").write_text(type(exc).__name__ + ": " + str(exc) + "\n")
            if content is not None:
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Schema validation failed: " + str(exc) + ". Return a complete corrected JSON object only."},
                ])
    raise RuntimeError(f"LLM validation failed after four attempts for {cell_type}: {last_error}; no fallback")


def map_action(
    validated: dict[str, object],
    model: str,
    scgpt_output: dict[str, object] | None,
) -> tuple[dict[str, float], dict[str, object]]:
    strengths = validated["program_strengths"]
    multipliers = validated["phenotype_multipliers"]
    executed = {
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
    causal = {
        "scgpt_fields_read": [], "identity_gate": 1.0,
        "activation_gate": 0.0, "stress_gate": 0.0, "exhaustion_gate": 0.0,
        "scgpt_causal_feedback_used": False,
    }
    if model == "full":
        if scgpt_output is None:
            raise RuntimeError("Full model missing scGPT output")
        drift = float(scgpt_output["identity_drift"])
        activation = max(0.0, float(scgpt_output["activation_shift"]))
        stress = max(0.0, float(scgpt_output["stress_shift"]))
        exhaustion = max(0.0, float(scgpt_output["exhaustion_shift"]))
        identity_gate = float(np.clip(1.0 - 0.5 * drift, 0.70, 1.0))
        activation_gate = float(np.tanh(activation))
        stress_gate = float(np.tanh(stress))
        exhaustion_gate = float(np.tanh(exhaustion))
        executed["birth_rate_multiplier"] *= identity_gate * (1.0 - 0.10 * exhaustion_gate)
        executed["death_rate_multiplier"] *= 1.0 + 0.20 * stress_gate + 0.15 * exhaustion_gate
        executed["secretion_multiplier"] *= 1.0 + 0.20 * activation_gate
        executed["motility_multiplier"] *= identity_gate
        executed["recruitment_rate"] *= 1.0 + 0.15 * activation_gate
        causal = {
            "scgpt_fields_read": ["identity_drift", "activation_shift", "stress_shift", "exhaustion_shift"],
            "identity_gate": identity_gate, "activation_gate": activation_gate,
            "stress_gate": stress_gate, "exhaustion_gate": exhaustion_gate,
            "scgpt_causal_feedback_used": True,
        }
    for key in (
        "birth_rate_multiplier", "death_rate_multiplier", "motility_multiplier",
        "secretion_multiplier", "uptake_multiplier",
    ):
        executed[key] = float(np.clip(executed[key], 0.25, 2.0))
    executed["recruitment_rate"] = float(np.clip(executed["recruitment_rate"], 0.0, 0.05))
    executed["clearance_rate"] = float(np.clip(executed["clearance_rate"], 0.0, 0.05))
    return executed, causal
