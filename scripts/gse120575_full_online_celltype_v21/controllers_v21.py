#!/usr/bin/env python3
"""Frozen trusted Population Actuator V2.1 three-channel budget mapper."""

from __future__ import annotations

import math


INTRINSIC_MIN, INTRINSIC_MAX = -0.20, 0.20
EXTERNAL_RECRUITMENT_MIN, EXTERNAL_RECRUITMENT_MAX = 0.0, 0.20
CLEARANCE_MIN, CLEARANCE_MAX = 0.0, 0.20

# These preserve the V2 program weights. V2.1 only separates their dimensions.
WEIGHTS = {
    "proliferation": 0.08,
    "survival_centered": 0.05,
    "recruitment": 0.07,
    "apoptosis": 0.10,
    "clearance": 0.07,
}


def _clip(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError("Non-finite V2.1 population budget")
    return min(upper, max(lower, value))


def trusted_budgets(
    validated: dict[str, object], model: str, scgpt: dict[str, object] | None
) -> tuple[dict[str, float], dict[str, object]]:
    """Map one frozen cell program to dimensionally distinct population budgets.

    The LLM never supplies a budget directly. scGPT can affect only the Full
    intrinsic channel, preserving the already-qualified V2 causal path without
    creating a target-aware recruitment rule.
    """
    strengths = validated["program_strengths"]
    intrinsic_components = {
        "proliferation": WEIGHTS["proliferation"] * float(strengths["proliferation"]),
        "survival_centered": WEIGHTS["survival_centered"]
        * (float(strengths["survival"]) - 0.5),
        "apoptosis": -WEIGHTS["apoptosis"] * float(strengths["apoptosis"]),
    }
    scgpt_components = {
        "activation": 0.0,
        "stress": 0.0,
        "exhaustion": 0.0,
        "identity_drift": 0.0,
    }
    if model == "full":
        if scgpt is None:
            raise RuntimeError("Full V2.1 mapper missing scGPT runtime output")
        scgpt_components = {
            "activation": 0.025 * math.tanh(max(0.0, float(scgpt["activation_shift"]))),
            "stress": -0.030 * math.tanh(max(0.0, float(scgpt["stress_shift"]))),
            "exhaustion": -0.030 * math.tanh(max(0.0, float(scgpt["exhaustion_shift"]))),
            "identity_drift": -0.025
            * min(1.0, max(0.0, float(scgpt["identity_drift"]))),
        }

    raw_intrinsic = sum(intrinsic_components.values()) + sum(scgpt_components.values())
    raw_external = WEIGHTS["recruitment"] * float(strengths["recruitment"])
    raw_clearance = WEIGHTS["clearance"] * float(strengths["clearance"])
    budgets = {
        "intrinsic_population_budget": _clip(raw_intrinsic, INTRINSIC_MIN, INTRINSIC_MAX),
        "external_recruitment_budget": _clip(
            raw_external, EXTERNAL_RECRUITMENT_MIN, EXTERNAL_RECRUITMENT_MAX
        ),
        "clearance_budget": _clip(raw_clearance, CLEARANCE_MIN, CLEARANCE_MAX),
    }
    audit = {
        "formula_version": "trusted_three_channel_population_budget_v2_1",
        "weights": WEIGHTS,
        "intrinsic_components": intrinsic_components,
        "scgpt_components": scgpt_components,
        "raw_intrinsic_population_budget": raw_intrinsic,
        "raw_external_recruitment_budget": raw_external,
        "raw_clearance_budget": raw_clearance,
        "executed_budgets": budgets,
        "intrinsic_clipped": abs(raw_intrinsic - budgets["intrinsic_population_budget"]) > 1e-12,
        "external_recruitment_clipped": abs(raw_external - budgets["external_recruitment_budget"]) > 1e-12,
        "clearance_clipped": abs(raw_clearance - budgets["clearance_budget"]) > 1e-12,
        "direct_llm_budget_used": False,
        "post_target_used": False,
        "cell_type_special_rule_used": False,
        "scgpt_causal_budget_feedback_used": model == "full",
    }
    return budgets, audit
