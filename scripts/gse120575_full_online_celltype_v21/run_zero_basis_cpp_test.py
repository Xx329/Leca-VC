#!/usr/bin/env python3
"""Compile and audit the real C++ V2.1 zero-basis recruitment scheduler."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_full_online_celltype_v21"
PROJECT = ROOT / "scenarios/gse120575_full_online_celltype_v21/physicell_project"
BUDGETS = (0.0, 0.001, 0.002, 0.005, 0.01)


def main() -> int:
    source = PROJECT / "zero_basis_actuator_test.cpp"
    binary = PROJECT / "zero_basis_actuator_test"
    build = subprocess.run(
        ["g++", "-O2", "-std=c++11", source.name, "-o", str(binary)],
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )
    if build.returncode:
        raise RuntimeError(build.stderr)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    audit = OUT / "zero_basis_cpp_test"
    audit.mkdir(parents=True, exist_ok=True)
    curve_path = audit / "zero_basis_recruitment_response_curve.csv"
    curve_path.write_text(result.stdout)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row["external_recruitment_budget"]), row["recruitment_enabled"] == "true")].append(row)

    checks = {
        "all_requested_budgets_tested": set(b for b, enabled in groups if enabled) == set(BUDGETS),
        "zero_budget_no_cells": True,
        "positive_budget_recruits_from_zero": True,
        "response_monotone": True,
        "no_minimum_one_recruitment": False,
        "residual_carry_correct": True,
        "cumulative_error_lt_one_worker": True,
        "disabled_zero_type_stays_zero": True,
        "no_false_emergence": True,
        "event_balance_zero": True,
    }
    endpoints = {}
    for (budget, enabled), rr in groups.items():
        rr.sort(key=lambda x: int(x["interval"]))
        cumulative_requested = sum(float(x["requested_recruitment"]) for x in rr)
        cumulative_executed = sum(int(x["scheduled_recruitment"]) for x in rr)
        endpoint = int(rr[-1]["end_type_workers"])
        endpoints[(budget, enabled)] = endpoint
        checks["zero_budget_no_cells"] &= budget != 0 or endpoint == 0
        if enabled and budget > 0:
            checks["positive_budget_recruits_from_zero"] &= endpoint > 0
        checks["cumulative_error_lt_one_worker"] &= abs(cumulative_requested - cumulative_executed) < 1.0
        checks["disabled_zero_type_stays_zero"] &= enabled or endpoint == 0
        checks["no_false_emergence"] &= all(int(x["false_emergence"]) == 0 for x in rr)
        checks["event_balance_zero"] &= all(int(x["event_balance"]) == 0 for x in rr)
        if enabled:
            last_residual = float(rr[-1]["residual_after"])
            checks["residual_carry_correct"] &= abs(last_residual - (cumulative_requested - cumulative_executed)) < 1e-9
    enabled_endpoints = [endpoints[(b, True)] for b in BUDGETS]
    checks["response_monotone"] = all(a <= b for a, b in zip(enabled_endpoints, enabled_endpoints[1:]))
    # At budget 0.001, repeated positive requests include a zero-scheduled interval.
    checks["no_minimum_one_recruitment"] = any(
        float(x["external_recruitment_budget"]) > 0
        and float(x["requested_recruitment"]) > 0
        and int(x["scheduled_recruitment"]) == 0
        and x["recruitment_enabled"] == "true"
        for x in rows
    )
    status = "PASS_REAL_CPP_V21_ZERO_BASIS_TEST" if all(checks.values()) else "FAIL_REAL_CPP_V21_ZERO_BASIS_TEST"
    summary = {
        "status": status,
        "checks": checks,
        "compiled_cpp": str(source.resolve()),
        "response_rows": len(rows),
        "frozen_cumulative_error_tolerance_workers": "<1",
        "post_target_used": False,
        "llm_called": False,
        "scgpt_called": False,
        "normalized_internal_simulation_time": True,
    }
    (audit / "zero_basis_cpp_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
