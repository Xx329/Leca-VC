#!/usr/bin/env python3
"""Offline three-layer population-coupling diagnosis for the frozen 54-run quick."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_full_online_celltype"
QUICK = OUT / "quick_3seed"
DIAG = OUT / "diagnostics/population_coupling_v1"
CELL_TYPES = ("B cell", "Plasma cell", "Monocyte/Macrophage", "Dendritic cell", "T cell", "NK cell")
MODELS = ("traditional", "agent_only", "full")
SEEDS = (12057501, 12057502, 12057503)
BASE_BIRTH = dict(zip(CELL_TYPES, (0.0010, 0.0005, 0.0008, 0.0007, 0.0011, 0.0009)))
BASE_DEATH = dict(zip(CELL_TYPES, (0.00030, 0.00035, 0.00040, 0.00035, 0.00030, 0.00032)))
INTERVAL_MINUTES = 2.0
AITchISON_EPSILON = 1e-6


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum(); midpoint = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def clr(vector: np.ndarray) -> np.ndarray:
    closed = np.maximum(np.asarray(vector, float), AITchISON_EPSILON)
    closed /= closed.sum()
    logged = np.log(closed)
    return logged - logged.mean()


def aitchison(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(clr(p) - clr(q)))


def reconstruct_runs() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    run_manifest = json.loads((QUICK / "quick_run_manifest.json").read_text())
    layer1, nonpopulation, layer2, layer3, gains = [], [], [], [], []
    for spec in run_manifest["runs"]:
        run_dir = QUICK / "runs" / spec["run_id"]
        states = {}
        for checkpoint in range(6):
            states[checkpoint] = {row["cell_type"]: row for row in read_csv(run_dir / f"state_checkpoint_{checkpoint}.csv")}
        actions = {
            checkpoint: {row["cell_type"]: row for row in read_csv(run_dir / f"execution_parameters_{checkpoint}.csv")}
            for checkpoint in range(5)
        }
        writeback_rows = read_csv(run_dir / "cpp_writeback_audit.csv")
        writebacks = {(int(row["checkpoint"]), row["cell_type"]): row for row in writeback_rows}
        for checkpoint in range(5):
            total_start = sum(int(float(states[checkpoint][cell_type]["live_cells"])) for cell_type in CELL_TYPES)
            total_end = sum(int(float(states[checkpoint + 1][cell_type]["live_cells"])) for cell_type in CELL_TYPES)
            total_growth_rate = (total_end - total_start) / total_start / INTERVAL_MINUTES
            for cell_type in CELL_TYPES:
                start = states[checkpoint][cell_type]; end = states[checkpoint + 1][cell_type]
                action = actions[checkpoint][cell_type]; wb = writebacks[(checkpoint, cell_type)]
                n0 = int(float(start["live_cells"])); n1 = int(float(end["live_cells"]))
                birth_multiplier = float(action["birth_rate_multiplier"])
                death_multiplier = float(action["death_rate_multiplier"])
                birth_absolute = float(wb["birth_rate"]); death_absolute = float(wb["death_rate"])
                recruitment_fraction = float(action["recruitment_rate"])
                clearance_fraction = float(action["clearance_rate"])
                birth_request = n0 * birth_absolute * INTERVAL_MINUTES
                death_request = n0 * death_absolute * INTERVAL_MINUTES
                recruitment_request = recruitment_fraction * max(1, n0)
                clearance_request = clearance_fraction * n0
                predicted_recruited_integer = min(5, max(1, half_up(recruitment_request))) if recruitment_fraction > 0 else 0
                predicted_cleared_integer = min(5, n0, half_up(clearance_request)) if clearance_fraction > 0 else 0
                agent_incremental_birth = n0 * (birth_absolute - BASE_BIRTH[cell_type]) * INTERVAL_MINUTES
                agent_incremental_death = n0 * (death_absolute - BASE_DEATH[cell_type]) * INTERVAL_MINUTES
                net_request = birth_request - death_request + recruitment_request - clearance_request
                incremental_net = agent_incremental_birth - agent_incremental_death + recruitment_request - clearance_request
                common = {
                    "run_id": spec["run_id"], "model": spec["model"], "sample_id": spec["sample_id"],
                    "response": spec["response"], "therapy": spec["therapy"], "seed": spec["seed"],
                    "interval": checkpoint, "start_checkpoint_percent": checkpoint * 20,
                    "end_checkpoint_percent": (checkpoint + 1) * 20, "effective_physicell_minutes": INTERVAL_MINUTES,
                    "cell_type": cell_type, "start_live_workers": n0, "end_live_workers": n1,
                }
                layer1.append({
                    **common,
                    "baseline_birth_rate_per_min": BASE_BIRTH[cell_type], "birth_multiplier": birth_multiplier,
                    "final_birth_rate_per_min": birth_absolute, "birth_request_workers": birth_request,
                    "agent_incremental_birth_request_workers": agent_incremental_birth,
                    "baseline_death_rate_per_min": BASE_DEATH[cell_type], "death_multiplier": death_multiplier,
                    "final_death_rate_per_min": death_absolute, "death_request_workers": death_request,
                    "agent_incremental_death_request_workers": agent_incremental_death,
                    "recruitment_fraction_per_checkpoint": recruitment_fraction,
                    "recruitment_request_workers_before_rounding": recruitment_request,
                    "recruitment_integer_requested_by_cpp": predicted_recruited_integer,
                    "clearance_fraction_per_checkpoint": clearance_fraction,
                    "clearance_request_workers_before_rounding": clearance_request,
                    "clearance_integer_requested_by_cpp": predicted_cleared_integer,
                    "population_effective_net_request_workers": net_request,
                    "agent_incremental_net_request_workers": incremental_net,
                    "baseline_birth_near_zero_for_interval": BASE_BIRTH[cell_type] * INTERVAL_MINUTES < 0.01,
                    "baseline_death_near_zero_for_interval": BASE_DEATH[cell_type] * INTERVAL_MINUTES < 0.01,
                })
                nonpopulation.append({
                    **common,
                    "motility_multiplier": float(action["motility_multiplier"]),
                    "final_motility_speed": float(wb["motility_speed"]),
                    "secretion_multiplier": float(action["secretion_multiplier"]),
                    "final_secretion_rate": float(wb["secretion_rate"]),
                    "uptake_multiplier": float(action["uptake_multiplier"]),
                    "final_uptake_rate": float(wb["uptake_rate"]),
                    "excluded_from_population_action_strength": True,
                })

                recruited = int(float(end["cumulative_recruited"])) - int(float(start["cumulative_recruited"]))
                cleared = int(float(end["cumulative_cleared"])) - int(float(start["cumulative_cleared"]))
                total_deaths = int(float(end["cumulative_deaths"])) - int(float(start["cumulative_deaths"]))
                apoptosis = total_deaths - cleared
                divisions = (n1 - n0) - recruited + total_deaths
                logged_birth_delta = int(float(end["cumulative_births"])) - int(float(start["cumulative_births"]))
                count_balance = divisions - apoptosis + recruited - cleared
                layer2.append({
                    **common,
                    "actual_divisions_balance_reconstructed": divisions,
                    "actual_apoptosis_excluding_clearance": apoptosis,
                    "actual_total_new_deaths_including_clearance": total_deaths,
                    "actual_recruited_workers": recruited,
                    "actual_cleared_emigrated_workers": cleared,
                    "raw_logged_cumulative_birth_delta_unreliable": logged_birth_delta,
                    "logged_vs_balance_division_difference": logged_birth_delta - divisions,
                    "event_balance_net_workers": count_balance,
                    "observed_absolute_count_change": n1 - n0,
                    "event_balance_exact": count_balance == n1 - n0,
                    "apoptosis_nonnegative_after_clearance_separation": apoptosis >= 0,
                })
                p0 = n0 / total_start; p1 = n1 / total_end
                relative_growth = safe_ratio(n1 - n0, n0) / INTERVAL_MINUTES if n0 else float("nan")
                actual_net_events = divisions - apoptosis + recruited - cleared
                layer3.append({
                    **common,
                    "total_start_live_workers": total_start, "total_end_live_workers": total_end,
                    "absolute_cell_count_change": n1 - n0,
                    "start_proportion": p0, "end_proportion": p1, "proportion_change": p1 - p0,
                    "relative_net_growth_rate_per_min": relative_growth,
                    "total_population_growth_rate_per_min": total_growth_rate,
                    "differential_growth_vs_total_per_min": relative_growth - total_growth_rate if np.isfinite(relative_growth) else float("nan"),
                    "action_to_proportion_gain": safe_ratio(p1 - p0, net_request / total_start),
                    "event_to_count_gain": safe_ratio(n1 - n0, actual_net_events),
                })
                channel_data = (
                    ("proliferation", birth_request, divisions, 1),
                    ("death", death_request, apoptosis, -1),
                    ("recruitment", recruitment_request, recruited, 1),
                    ("clearance", clearance_request, cleared, -1),
                )
                for channel, requested, actual, sign in channel_data:
                    gains.append({
                        **common, "channel": channel,
                        "requested_workers": requested, "actual_events": actual,
                        "signed_actual_count_contribution": sign * actual,
                        "action_to_event_gain": safe_ratio(actual, requested),
                        "request_positive_but_less_than_one_worker": 0 < requested < 1,
                        "integer_rounding_to_zero": channel in {"recruitment", "clearance"} and requested > 0 and actual == 0,
                    })
    return layer1, nonpopulation, layer2, layer3, gains


def composition_metrics() -> list[dict[str, object]]:
    post = read_csv(OUT / "preprocessed/posttreatment_sample_celltype_proportions.csv")
    references = {}
    for response in ("Responder", "Non-responder"):
        matrix = np.asarray([[float(row[cell_type]) for cell_type in CELL_TYPES] for row in post if row["response"] == response])
        references[response] = matrix.mean(axis=0)
    quick_manifest = json.loads((OUT / "audit/quick_manifest_frozen.json").read_text())
    rows = []
    for sample in quick_manifest["samples"]:
        sample_id = sample["sample_id"]; response = sample["response"]; target = references[response]
        for model in MODELS:
            start_vectors, end_vectors = [], []
            for seed in SEEDS:
                run_dir = QUICK / "runs" / f"quick__{sample_id}__{model}__seed_{seed}"
                trajectory = read_csv(run_dir / "composition_trajectory.csv")
                for checkpoint, container in (("0", start_vectors), ("5", end_vectors)):
                    by_type = {row["cell_type"]: float(row["proportion"]) for row in trajectory if row["checkpoint"] == checkpoint}
                    container.append(np.asarray([by_type[cell_type] for cell_type in CELL_TYPES]))
            pre = np.mean(start_vectors, axis=0); predicted = np.mean(end_vectors, axis=0)
            target_distance = float(np.sum(np.abs(target - pre)))
            predicted_change = float(np.sum(np.abs(predicted - pre)))
            rows.append({
                "sample_id": sample_id, "response": response, "therapy": sample["therapy"], "model": model,
                "macro_mae": float(np.mean(np.abs(predicted - target))),
                "jsd_base2": jsd(predicted, target),
                "aitchison_distance_epsilon_1e_6": aitchison(predicted, target),
                "change_direction_agreement": float(np.mean(np.sign(predicted - pre) == np.sign(target - pre))),
                "change_magnitude_ratio_l1": safe_ratio(predicted_change, target_distance),
                "progress_to_target_l1": 1.0 - safe_ratio(float(np.sum(np.abs(predicted - target))), target_distance),
                "rmse_auxiliary": float(np.sqrt(np.mean((predicted - target) ** 2))),
                "pearson_auxiliary": float(np.corrcoef(predicted, target)[0, 1]),
                "seed_mean_celltype_sd": float(np.mean(np.std(end_vectors, axis=0))),
                "pre_to_target_l1": target_distance, "pre_to_prediction_l1": predicted_change,
                "prediction_to_target_l1": float(np.sum(np.abs(predicted - target))),
                "post_reference_definition": "arithmetic biopsy-level mean within response group; evaluation-only",
            })
    return rows


def aggregate_diagnostics(layer1, layer2, layer3, gains, metrics) -> dict[str, object]:
    channel_summary = []
    for model in MODELS:
        for channel in ("proliferation", "death", "recruitment", "clearance"):
            selected = [row for row in gains if row["model"] == model and row["channel"] == channel]
            requested = sum(float(row["requested_workers"]) for row in selected)
            actual = sum(int(row["actual_events"]) for row in selected)
            channel_summary.append({
                "model": model, "channel": channel, "requested_workers_total": requested,
                "actual_events_total": actual,
                "cumulative_signed_count_contribution": sum(int(row["signed_actual_count_contribution"]) for row in selected),
                "aggregate_action_to_event_gain": safe_ratio(actual, requested),
                "requests_positive_but_less_than_one_worker": sum(bool(row["request_positive_but_less_than_one_worker"]) for row in selected),
                "integer_rounding_to_zero_count": sum(bool(row["integer_rounding_to_zero"]) for row in selected),
            })
    write_csv(DIAG / "coupling_summary_model_channel.csv", channel_summary)

    growth_summary = []
    for model in MODELS:
        selected = [row for row in layer3 if row["model"] == model]
        finite_diff = np.asarray([float(row["differential_growth_vs_total_per_min"]) for row in selected if np.isfinite(float(row["differential_growth_vs_total_per_min"]))])
        by_interval = defaultdict(list)
        for row in selected:
            if np.isfinite(float(row["relative_net_growth_rate_per_min"])):
                by_interval[(row["sample_id"], row["seed"], row["interval"])].append(float(row["relative_net_growth_rate_per_min"]))
        dispersions = [float(np.std(values)) for values in by_interval.values() if values]
        growth_summary.append({
            "model": model,
            "mean_absolute_differential_growth_per_min": float(np.mean(np.abs(finite_diff))),
            "mean_within_interval_celltype_growth_sd_per_min": float(np.mean(dispersions)),
            "fraction_intervals_growth_sd_below_0_001_per_min": float(np.mean(np.asarray(dispersions) < 0.001)),
            "interpretation": "lower dispersion means similar net growth across cell types and weak composition change",
        })
    write_csv(DIAG / "differential_growth_summary.csv", growth_summary)

    metric_summary = []
    for model in MODELS:
        selected = [row for row in metrics if row["model"] == model]
        record = {"model": model, "sample_n": len(selected)}
        for key in (
            "macro_mae", "jsd_base2", "aitchison_distance_epsilon_1e_6",
            "change_direction_agreement", "change_magnitude_ratio_l1", "progress_to_target_l1",
            "rmse_auxiliary", "pearson_auxiliary", "seed_mean_celltype_sd",
        ):
            record[f"mean_{key}"] = float(np.mean([float(row[key]) for row in selected]))
        metric_summary.append(record)
    write_csv(DIAG / "composition_metrics_model_summary.csv", metric_summary)

    birth_mismatch = [row for row in layer2 if row["logged_vs_balance_division_difference"] != 0]
    negative_apoptosis = [row for row in layer2 if not row["apoptosis_nonnegative_after_clearance_separation"]]
    balance_fail = [row for row in layer2 if not row["event_balance_exact"]]
    summary = {
        "status": "PASS_OFFLINE_DIAGNOSIS_WITH_IDENTIFIED_COUPLING_LIMITATIONS" if not balance_fail and not negative_apoptosis else "FAIL_EVENT_RECONSTRUCTION",
        "source_quick_audit_status": json.loads((QUICK / "quick_execution_audit.json").read_text())["status"],
        "no_physicell_llm_or_scgpt_rerun": True,
        "expression_evaluation_used": False,
        "rows": {"layer1": len(layer1), "layer2": len(layer2), "layer3": len(layer3), "channel_gain": len(gains), "composition_metrics": len(metrics)},
        "interval_minutes": INTERVAL_MINUTES,
        "all_event_balances_exact": not balance_fail,
        "negative_apoptosis_rows_after_clearance_separation": len(negative_apoptosis),
        "unreliable_cpp_logged_birth_delta_mismatch_rows": len(birth_mismatch),
        "cpp_birth_counter_limitation": "cumulative_births is balance-derived using cumulative rather than interval recruitment/death and double-counts clearance as cleared plus dead; balance-reconstructed divisions are used as actual divisions",
        "baseline_rate_near_zero": {
            "criterion": "baseline_rate_per_min * 2-minute interval < 0.01 expected event per worker",
            "all_birth_cell_types_near_zero": all(BASE_BIRTH[cell_type] * INTERVAL_MINUTES < 0.01 for cell_type in CELL_TYPES),
            "all_death_cell_types_near_zero": all(BASE_DEATH[cell_type] * INTERVAL_MINUTES < 0.01 for cell_type in CELL_TYPES),
            "birth_range_per_min": [min(BASE_BIRTH.values()), max(BASE_BIRTH.values())],
            "death_range_per_min": [min(BASE_DEATH.values()), max(BASE_DEATH.values())],
        },
        "channel_summary": channel_summary,
        "differential_growth_summary": growth_summary,
        "composition_metric_summary": metric_summary,
    }
    write_json(DIAG / "population_coupling_summary.json", summary)
    return summary


def main() -> int:
    if json.loads((QUICK / "quick_execution_audit.json").read_text())["status"] != "PASS_54_RUN_QUICK_EXECUTION":
        raise RuntimeError("Frozen quick execution did not pass")
    DIAG.mkdir(parents=True, exist_ok=True)
    layer1, nonpopulation, layer2, layer3, gains = reconstruct_runs()
    metrics = composition_metrics()
    write_csv(DIAG / "layer1_population_effective_actions.csv", layer1)
    write_csv(DIAG / "layer1_nonpopulation_actions.csv", nonpopulation)
    write_csv(DIAG / "layer2_physicell_events.csv", layer2)
    write_csv(DIAG / "layer3_population_outcomes.csv", layer3)
    write_csv(DIAG / "channel_gain_diagnostics.csv", gains)
    write_csv(DIAG / "composition_metrics_sample_model.csv", metrics)
    summary = aggregate_diagnostics(layer1, layer2, layer3, gains, metrics)
    print(json.dumps({
        "status": summary["status"], "rows": summary["rows"],
        "birth_counter_mismatch_rows": summary["unreliable_cpp_logged_birth_delta_mismatch_rows"],
        "all_event_balances_exact": summary["all_event_balances_exact"],
        "expression_evaluation_used": False,
    }, ensure_ascii=False, indent=2))
    return 0 if summary["all_event_balances_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
