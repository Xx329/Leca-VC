#!/usr/bin/env python3
"""Write the offline Population Actuator V2 design and synthetic response curve."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_full_online_celltype"
DIAG = OUT / "diagnostics/population_coupling_v1"
COUNTS = (1, 3, 5, 10, 50, 100, 500, 600)
BUDGETS = (-0.20, -0.10, -0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
DT = 2.0
POSITIVE_DIVISION_FRACTION = 0.5
NEGATIVE_APOPTOSIS_FRACTION = 0.5


def half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def split_integer(total: int, first_fraction: float) -> tuple[int, int]:
    desired_first = total * first_fraction
    first = int(math.floor(desired_first))
    second = total - first
    if desired_first - first >= 0.5 and second > 0:
        first += 1; second -= 1
    return first, second


def map_budget(n: int, budget: float) -> dict[str, float | int]:
    desired_delta = budget * n
    total_events = half_up(abs(desired_delta))
    divisions = recruited = apoptosis = cleared = 0
    birth_rate = death_rate = recruitment_flux = clearance_flux = 0.0
    if budget > 0:
        divisions, recruited = split_integer(total_events, POSITIVE_DIVISION_FRACTION)
        birth_fraction = budget * POSITIVE_DIVISION_FRACTION
        birth_rate = math.log1p(birth_fraction) / DT
        recruitment_flux = budget * (1.0 - POSITIVE_DIVISION_FRACTION) * n / DT
    elif budget < 0:
        apoptosis, cleared = split_integer(total_events, NEGATIVE_APOPTOSIS_FRACTION)
        death_fraction = abs(budget) * NEGATIVE_APOPTOSIS_FRACTION
        death_rate = -math.log1p(-death_fraction) / DT
        clearance_flux = abs(budget) * (1.0 - NEGATIVE_APOPTOSIS_FRACTION) * n / DT
    actual_delta = divisions + recruited - apoptosis - cleared
    return {
        "initial_workers": n, "interval_minutes": DT,
        "interval_population_budget": budget,
        "desired_delta_workers": desired_delta,
        "mapped_additive_birth_hazard_per_min": birth_rate,
        "mapped_additive_death_hazard_per_min": death_rate,
        "mapped_recruitment_flux_workers_per_min": recruitment_flux,
        "mapped_clearance_flux_workers_per_min": clearance_flux,
        "executed_divisions_synthetic": divisions,
        "executed_recruitment_synthetic": recruited,
        "executed_apoptosis_synthetic": apoptosis,
        "executed_clearance_synthetic": cleared,
        "actual_delta_workers_synthetic": actual_delta,
        "actual_fractional_response_synthetic": actual_delta / n,
        "single_interval_budget_error_workers": actual_delta - desired_delta,
    }


def residual_carry_response(n0: int, budget: float, intervals: int = 5) -> dict[str, float | int]:
    current = n0; residual = 0.0; cumulative_actual = 0; cumulative_desired = 0.0
    for _ in range(intervals):
        desired = abs(budget) * current
        due = desired + residual
        events = min(current, half_up(due)) if budget < 0 else half_up(due)
        residual = due - events
        signed = events if budget >= 0 else -events
        current += signed; cumulative_actual += signed
        cumulative_desired += budget * (current - signed)
    return {
        "five_interval_final_workers_with_residual_carry": current,
        "five_interval_cumulative_desired_delta_workers": cumulative_desired,
        "five_interval_cumulative_actual_delta_workers": cumulative_actual,
        "five_interval_cumulative_error_workers": cumulative_actual - cumulative_desired,
        "final_fractional_change_from_initial": (current - n0) / n0,
        "final_residual_worker_budget": residual,
    }


def main() -> int:
    curve = []
    for n in COUNTS:
        for budget in BUDGETS:
            curve.append({**map_budget(n, budget), **residual_carry_response(n, budget)})
    curve_path = DIAG / "population_actuator_v2_synthetic_response_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)

    tests = {
        "zero_budget_zero_response": all(row["actual_delta_workers_synthetic"] == 0 for row in curve if row["interval_population_budget"] == 0),
        "single_interval_sign_correct": all(
            row["actual_delta_workers_synthetic"] == 0
            or math.copysign(1, row["actual_delta_workers_synthetic"]) == math.copysign(1, row["interval_population_budget"])
            for row in curve if row["interval_population_budget"] != 0
        ),
        "single_interval_integer_error_at_most_half_worker": all(abs(row["single_interval_budget_error_workers"]) <= 0.5000000001 for row in curve),
        "five_interval_residual_carry_error_below_one_worker": all(abs(row["five_interval_cumulative_error_workers"]) < 1.0 for row in curve),
        "nonnegative_dimensionally_correct_rates_and_fluxes": all(
            row[key] >= 0 for row in curve for key in (
                "mapped_additive_birth_hazard_per_min", "mapped_additive_death_hazard_per_min",
                "mapped_recruitment_flux_workers_per_min", "mapped_clearance_flux_workers_per_min",
            )
        ),
        "monotone_one_interval_response_with_budget": all(
            all(a <= b for a, b in zip(
                [row["actual_fractional_response_synthetic"] for row in curve if row["initial_workers"] == n][:-1],
                [row["actual_fractional_response_synthetic"] for row in curve if row["initial_workers"] == n][1:],
            )) for n in COUNTS
        ),
        "post_target_or_expression_used": False,
        "real_physicell_experiment_run": False,
    }
    result = {
        "status": "PASS_SYNTHETIC_ACTUATOR_V2_UNIT_TEST" if all(value for key, value in tests.items() if key not in {"post_target_or_expression_used", "real_physicell_experiment_run"}) else "FAIL_SYNTHETIC_ACTUATOR_V2_UNIT_TEST",
        "tests": tests,
        "synthetic_calibration": {
            "budget_range": [min(BUDGETS), max(BUDGETS)],
            "worker_counts": list(COUNTS), "interval_minutes": DT,
            "positive_division_fraction": POSITIVE_DIVISION_FRACTION,
            "negative_apoptosis_fraction": NEGATIVE_APOPTOSIS_FRACTION,
            "integer_policy": "round total budget once, split by largest remainder, carry subworker residual across intervals",
        },
        "response_curve": str(curve_path.resolve()),
    }
    (DIAG / "population_actuator_v2_unit_test.json").write_text(json.dumps(result, indent=2) + "\n")

    summary = json.loads((DIAG / "population_coupling_summary.json").read_text())
    channel = {(row["model"], row["channel"]): row for row in summary["channel_summary"]}
    metric = {row["model"]: row for row in summary["composition_metric_summary"]}
    coupling_report = f"""# GSE120575 offline Agent-to-PhysiCell population coupling diagnosis

## Scope and status

- Status: **{summary['status']}**
- Source: existing frozen 54-run quick logs only.
- PhysiCell, LLM and scGPT rerun: **False**.
- Expression evaluation used: **False**.
- Every interval is 2 PhysiCell minutes (0–20, 20–40, 40–60, 60–80, 80–100%).
- Layer rows: action={summary['rows']['layer1']}, events={summary['rows']['layer2']}, population={summary['rows']['layer3']}.

## Principal coupling findings

1. The C++ `cumulative_births` field is not a reliable division counter in 186/1620 interval/type rows. It reuses cumulative recruitment/death after the first interval and counts clearance both as cleared and dead. Actual divisions were reconstructed by the exact interval conservation equation; all 1620 balances close exactly and separated apoptosis is never negative.
2. Birth baselines are {summary['baseline_rate_near_zero']['birth_range_per_min'][0]}–{summary['baseline_rate_near_zero']['birth_range_per_min'][1]} per minute and death baselines are {summary['baseline_rate_near_zero']['death_range_per_min'][0]}–{summary['baseline_rate_near_zero']['death_range_per_min'][1]} per minute. Over two minutes every baseline contributes below 0.01 expected event per worker, so even visibly different multipliers have weak population leverage.
3. Recruitment is strongly upward biased: current C++ forces at least one recruited worker whenever the fraction is positive. Traditional requested {channel[('traditional','recruitment')]['requested_workers_total']:.3f} workers but executed {channel[('traditional','recruitment')]['actual_events_total']}; Agent-only requested {channel[('agent_only','recruitment')]['requested_workers_total']:.3f} but executed {channel[('agent_only','recruitment')]['actual_events_total']}; Full requested {channel[('full','recruitment')]['requested_workers_total']:.3f} but executed {channel[('full','recruitment')]['actual_events_total']}.
4. Clearance is downward biased by rounding: Traditional had {channel[('traditional','clearance')]['integer_rounding_to_zero_count']} positive requests rounded to zero, Agent-only {channel[('agent_only','clearance')]['integer_rounding_to_zero_count']}, and Full {channel[('full','clearance')]['integer_rounding_to_zero_count']}.
5. Agent-only and Full assign similar net growth across cell types. Their mean within-interval cell-type growth SD is about 0.0096 and 0.0091 per minute versus 0.0658 for Traditional. This weak differential growth explains why Agent/Full composition remains close to initialization.

## Composition diagnostics (sample-level after three-seed mean)

| Model | Macro-MAE ↓ | JSD ↓ | Aitchison ↓ | Direction ↑ | Magnitude ratio (1 ideal) | Progress-to-target ↑ | RMSE ↓ | Pearson ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Traditional | {metric['traditional']['mean_macro_mae']:.5f} | {metric['traditional']['mean_jsd_base2']:.5f} | {metric['traditional']['mean_aitchison_distance_epsilon_1e_6']:.3f} | {metric['traditional']['mean_change_direction_agreement']:.3f} | {metric['traditional']['mean_change_magnitude_ratio_l1']:.3f} | {metric['traditional']['mean_progress_to_target_l1']:.3f} | {metric['traditional']['mean_rmse_auxiliary']:.5f} | {metric['traditional']['mean_pearson_auxiliary']:.3f} |
| Agent-only | {metric['agent_only']['mean_macro_mae']:.5f} | {metric['agent_only']['mean_jsd_base2']:.5f} | {metric['agent_only']['mean_aitchison_distance_epsilon_1e_6']:.3f} | {metric['agent_only']['mean_change_direction_agreement']:.3f} | {metric['agent_only']['mean_change_magnitude_ratio_l1']:.3f} | {metric['agent_only']['mean_progress_to_target_l1']:.3f} | {metric['agent_only']['mean_rmse_auxiliary']:.5f} | {metric['agent_only']['mean_pearson_auxiliary']:.3f} |
| Full | {metric['full']['mean_macro_mae']:.5f} | {metric['full']['mean_jsd_base2']:.5f} | {metric['full']['mean_aitchison_distance_epsilon_1e_6']:.3f} | {metric['full']['mean_change_direction_agreement']:.3f} | {metric['full']['mean_change_magnitude_ratio_l1']:.3f} | {metric['full']['mean_progress_to_target_l1']:.3f} | {metric['full']['mean_rmse_auxiliary']:.5f} | {metric['full']['mean_pearson_auxiliary']:.3f} |

Aitchison distance uses closure after a fixed 1e-6 zero replacement. Progress-to-target is `1 - L1(pred,target)/L1(pre,target)`; change-magnitude ratio is `L1(pred,pre)/L1(target,pre)`.

## Interpretation

The dominant failure is actuator semantics, not absence of Agent/scGPT action differences. Small rate multipliers operate on near-zero hazards, recruitment has a forced minimum-one discontinuity, clearance is usually rounded away, and cell types receive insufficiently differentiated realized growth. Therefore action differences do not translate proportionally into composition changes.
"""
    (DIAG / "population_coupling_report.md").write_text(coupling_report)

    design = f"""# Population Actuator V2 design

## Objective

Replace population control through rate multipliers alone with a dimensionless signed interval budget that has direct, auditable population meaning. This design does not modify or reinterpret the frozen 54-run results.

## Agent schema

For each `sample × seed × model × checkpoint × cell_type`, the Agent adds:

```json
{{
  "interval_population_budget": 0.05,
  "budget_units": "fraction_of_current_live_workers_over_this_interval",
  "budget_direction": "increase",
  "population_evidence_summary": "current/past state only"
}}
```

- `interval_population_budget = b` is signed and dimensionless: desired `ΔN/N` over the next frozen interval.
- Initial synthetic safety range: `-0.20 ≤ b ≤ 0.20`; this is tested synthetically and is not chosen from post-treatment performance.
- Post target, observed post composition and post expression are forbidden from the prompt and mapper.
- Secretion, uptake and motility remain separate phenotype channels and do not contribute to population budget strength.

## Trusted mapper

Given current live workers `N`, duration `Δt`, and signed budget `b`:

1. Desired signed worker change is `B = b × N`.
2. A pre-registered mechanism allocation splits positive B into division/recruitment and negative B into apoptosis/clearance. The synthetic reference curve uses 50/50 only as a unit-test calibration; lineage-specific allocations must come from external biology or synthetic calibration, never held-out results.
3. Positive division hazard is `λ_budget = log(1 + b_division) / Δt` in `1/min`; negative apoptosis hazard is `μ_budget = -log(1 - |b_apoptosis|) / Δt` in `1/min`.
4. Recruitment and clearance are fluxes in `workers/min`: `flux = desired_channel_workers / Δt`, distributed across the interval rather than applied as an instantaneous checkpoint fraction.
5. Budget-derived hazards are additive to frozen baseline hazards. They are not multipliers of near-zero baselines.
6. Integer execution rounds the total requested budget once, then allocates integer events by largest remainder. A signed residual-worker accumulator is carried to the next checkpoint, keeping cumulative error below one worker without forcing minimum-one recruitment.
7. Removal is capped by current live workers; additions/removals are logged as requested real-valued budget, scheduled events, executed events, residual and realized `ΔN/N`.

## Required runtime audit

- current N, interval duration and input budget;
- desired worker delta and channel allocation;
- baseline hazard, additive budget hazard and final absolute hazard;
- recruitment/clearance flux in workers/min;
- integer event quota and residual carry;
- divisions, apoptosis, recruited and cleared events from explicit event counters;
- realized count/proportion change and budget error;
- cross-cell-type differential growth.

The V2 C++ executor must replace the current inferred `cumulative_births` with explicit division callbacks/counters and must distinguish natural apoptosis from actuator clearance.

## Synthetic unit test and response curve

- Unit-test status: **{result['status']}**.
- Tested N: {', '.join(map(str, COUNTS))} workers.
- Tested budgets: {', '.join(map(str, BUDGETS))}.
- Duration: {DT} minutes per interval, plus five-interval residual-carry tests.
- Tests: sign, monotonicity, dimensional non-negativity, ≤0.5-worker single-interval error and <1-worker cumulative residual error.
- Real PhysiCell run: False.
- Post target/expression used: False.
- Response curve: `{curve_path}`.

## Advancement boundary

V2 is currently a design plus pure synthetic actuator calibration. Before any new biological experiment, implement it in a new code/output version, run explicit C++ actuator unit tests with synthetic initial counts, verify response curves, then request separate authorization for a new smoke. The frozen 54-run quick remains unchanged.
"""
    (DIAG / "population_actuator_v2_design.md").write_text(design)
    print(json.dumps({"status": result["status"], "curve_rows": len(curve), "post_target_used": False, "real_experiment_run": False}, indent=2))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
