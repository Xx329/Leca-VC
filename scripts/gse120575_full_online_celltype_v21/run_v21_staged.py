#!/usr/bin/env python3
"""Run V2.1 smoke then unseen-sample 18-run validation; never run 54 or plot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from physicell_executor_v21 import BINARY, compile_executor, runtime_config


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_full_online_celltype_v21"
V1 = ROOT / "outputs/GSE120575_full_online_celltype"
CELL_TYPES = ("B cell", "Plasma cell", "Monocyte/Macrophage", "Dendritic cell", "T cell", "NK cell")
MODELS = ("traditional", "agent_only", "full")
SEEDS = (12057501, 12057502, 12057503)
FORBIDDEN_TARGET_KEYS = {"observed_post", "post_target", "target_composition", "evaluation_reference"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specs(samples: tuple[str, ...], seeds: tuple[int, ...], phase: str) -> list[dict[str, object]]:
    metadata = {r["sample_id"]: r for r in read_csv(V1 / "preprocessed/pretreatment_sample_celltype_proportions.csv")}
    return [
        {
            "run_id": f"v21_{phase}__{sample}__{model}__seed_{seed}",
            "sample_id": sample,
            "patient_id": metadata[sample]["patient_id"],
            "response": metadata[sample]["response"],
            "therapy": metadata[sample]["therapy"],
            "model": model,
            "seed": seed,
        }
        for sample in samples for seed in seeds for model in MODELS
    ]


def select_unseen_validation_samples() -> tuple[str, str]:
    source = V1 / "audit/quick_manifest_frozen.csv"
    rows = read_csv(source)
    excluded = {"Pre_P1", "Pre_P12"}
    candidates = [r for r in rows if r["sample_id"] not in excluded and r["therapy"] == "anti-PD1"]
    by_response = defaultdict(list)
    for row in candidates:
        by_response[row["response"]].append(row)
    if not by_response["Responder"] or not by_response["Non-responder"]:
        audit = {
            "status": "STOP_NO_THERAPY_MATCHED_UNSEEN_PAIR",
            "source": str(source.resolve()),
            "excluded": sorted(excluded),
            "candidates": candidates,
            "post_target_read": False,
        }
        write_json(OUT / "audit/unseen_sample_selection.json", audit)
        raise RuntimeError("No unused Responder/Non-responder anti-PD1 pair")
    sort_key = lambda r: int(r["patient_id"].removeprefix("P"))
    responder = sorted(by_response["Responder"], key=sort_key)[0]
    nonresponder = sorted(by_response["Non-responder"], key=sort_key)[0]
    selected = (responder["sample_id"], nonresponder["sample_id"])
    registry_paths = [
        V1 / "preprocessed/worker_registries" / f"{sample}__seed_{seed}.csv"
        for sample in selected for seed in SEEDS
    ]
    if not all(path.is_file() for path in registry_paths):
        raise RuntimeError("Selected unseen sample lacks a frozen worker registry")
    audit = {
        "status": "PASS_FROZEN_UNSEEN_THERAPY_MATCHED_SELECTION",
        "selection_scope": "unused_by_V2_and_V2.1_scientific_validation",
        "source": str(source.resolve()),
        "source_sha256": digest(source),
        "excluded": sorted(excluded),
        "therapy_required": "anti-PD1",
        "selection_rule": "minimum numeric patient_id within each response stratum; no post data read",
        "candidate_rows": candidates,
        "selected_rows": [responder, nonresponder],
        "selected_samples": list(selected),
        "registry_hashes": [{"path": str(p.resolve()), "sha256": digest(p)} for p in registry_paths],
        "post_target_read": False,
    }
    selection_path = OUT / "audit/unseen_sample_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != audit:
            raise RuntimeError("Frozen unseen sample selection audit changed")
    else:
        write_json(selection_path, audit)
    return selected


def contains_target_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(str(k).lower() in FORBIDDEN_TARGET_KEYS or contains_target_key(v) for k, v in value.items())
    if isinstance(value, list):
        return any(contains_target_key(v) for v in value)
    return False


def run_phase(phase: str, specifications: list[dict[str, object]]) -> dict[str, object]:
    phase_dir = OUT / phase
    run_root = phase_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(phase_dir / f"{phase}_manifest.json", {
        "phase": phase,
        "runs": specifications,
        "population_actuator": "V2.1",
        "normalized_internal_simulation_time": True,
        "clinical_minutes_claimed": False,
        "post_target_runtime_access": False,
    })
    progress = []
    for index, specification in enumerate(specifications, 1):
        run_dir = run_root / str(specification["run_id"])
        if run_dir.exists() and any(run_dir.iterdir()):
            if (run_dir / "RUN_COMPLETE.json").is_file():
                progress.append({**specification, "status": "PASS_EXISTING"})
                continue
            raise RuntimeError(f"Refusing incomplete V2.1 run overwrite: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        registry = V1 / "preprocessed/worker_registries" / f"{specification['sample_id']}__seed_{specification['seed']}.csv"
        config = runtime_config(
            run_dir, registry, str(specification["sample_id"]), str(specification["model"]),
            str(specification["response"]), str(specification["therapy"]), int(specification["seed"]),
        )
        write_json(run_dir / "run_manifest.json", {
            **specification,
            "run_dir": str(run_dir.resolve()),
            "registry": str(registry.resolve()),
            "registry_sha256": digest(registry),
            "runner_sha256": digest(Path(__file__)),
            "config_sha256": digest(config),
            "code_hash": digest(Path(__file__)),
            "config_hash": digest(config),
            "POPULATION_ACTUATOR_VERSION": "V2.1",
            "normalized_internal_simulation_time": True,
            "post_target_runtime_access": False,
        })
        print(f"[{index}/{len(specifications)}] starting {specification['run_id']}", flush=True)
        start = time.time()
        result = subprocess.run([str(BINARY), str(config)], cwd=BINARY.parent, capture_output=True, text=True, timeout=1800)
        (run_dir / "stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            write_json(phase_dir / "FAILED.json", {**specification, "exit_code": result.returncode})
            raise RuntimeError(f"V2.1 {phase} failed: {specification['run_id']}")
        write_json(run_dir / "RUN_COMPLETE.json", {"status": "PASS", "elapsed_seconds": time.time() - start})
        progress.append({**specification, "status": "PASS"})
        write_json(phase_dir / f"{phase}_progress.json", {"completed": progress, "target": len(specifications)})
        print(f"[{index}/{len(specifications)}] PASS {specification['run_id']}", flush=True)
    return audit_phase(phase, specifications)


def audit_phase(phase: str, specifications: list[dict[str, object]]) -> dict[str, object]:
    root = OUT / phase / "runs"
    decisions, scgpt, writebacks, budgets, events, counts, compositions = [], [], [], [], [], [], []
    checkpoint_ok = True
    leakage_paths = []
    initial_bytes = {}
    full_budget_causal = []
    for specification in specifications:
        run_dir = root / str(specification["run_id"])
        decisions += read_csv(run_dir / "decision_call_audit.csv")
        writebacks += read_csv(run_dir / "cpp_writeback_audit.csv")
        budgets += [{**r, "run_id": specification["run_id"]} for r in read_csv(run_dir / "v21_budget_audit.csv")]
        events += [{**r, "run_id": specification["run_id"]} for r in read_csv(run_dir / "v21_interval_event_audit.csv")]
        counts += [{**r, "run_id": specification["run_id"]} for r in read_csv(run_dir / "cell_count_trajectory.csv")]
        compositions += [{**r, "run_id": specification["run_id"]} for r in read_csv(run_dir / "composition_trajectory.csv")]
        if (run_dir / "scgpt_call_audit.csv").is_file():
            scgpt += read_csv(run_dir / "scgpt_call_audit.csv")
        checkpoint_ok &= all((run_dir / f"execution_payload_{k}.json").is_file() for k in range(5)) and (run_dir / "state_checkpoint_5.csv").is_file()
        initial_bytes[(specification["sample_id"], specification["seed"], specification["model"])] = (run_dir / "state_checkpoint_0.csv").read_bytes()
        for path in run_dir.glob("prompts/checkpoint_*/*.json"):
            if contains_target_key(json.loads(path.read_text())):
                leakage_paths.append(str(path))
        if specification["model"] == "full":
            for checkpoint in range(5):
                payload = json.loads((run_dir / f"execution_payload_{checkpoint}.json").read_text())
                full_budget_causal += [bool(r.get("scgpt_causal_budget_feedback_used")) for r in payload["records"]]

    agent = [r for r in decisions if r["model"] in {"agent_only", "full"}]
    fallback = [r for r in decisions if r["fallback_used"].lower() == "true"]
    sample_seed_pairs = {(s["sample_id"], s["seed"]) for s in specifications}
    expected_agent = len(sample_seed_pairs) * 2 * 6 * 5
    expected_scgpt = len(sample_seed_pairs) * 6 * 5
    expected_writebacks = len(specifications) * 30
    composition_sums = defaultdict(float)
    for row in compositions:
        composition_sums[(row["run_id"], row["checkpoint"])] += float(row["proportion"])
    totals = [int(r["total_live"]) for r in counts]
    fair = all(
        initial_bytes[(sample, seed, "traditional")] == initial_bytes[(sample, seed, "agent_only")] == initial_bytes[(sample, seed, "full")]
        for sample, seed in sample_seed_pairs
    )
    conservation_errors = [int(r["conservation_error"]) for r in events]
    initial_zero = defaultdict(set)
    final_count = {}
    requested_external = defaultdict(float)
    recruited = defaultdict(int)
    for row in compositions:
        key = (row["run_id"], row["cell_type"])
        if row["checkpoint"] == "0" and int(row["live_cells"]) == 0:
            initial_zero[row["run_id"]].add(row["cell_type"])
        if row["checkpoint"] == "5":
            final_count[key] = int(row["live_cells"])
    for row in budgets:
        requested_external[(row["run_id"], row["cell_type"])] += float(row["requested_recruitment_workers"])
    for row in events:
        recruited[(row["run_id"], row["cell_type"])] += int(row["actuator_recruitment"])
    eligible_zero = [(run_id, cell_type) for run_id, types in initial_zero.items() for cell_type in types if requested_external[(run_id, cell_type)] > 0]
    recovered_zero = [(run_id, cell_type) for run_id, cell_type in eligible_zero if final_count[(run_id, cell_type)] > 0 and recruited[(run_id, cell_type)] > 0]
    false_emergence = [(run_id, cell_type) for run_id, types in initial_zero.items() for cell_type in types if requested_external[(run_id, cell_type)] == 0 and final_count[(run_id, cell_type)] > 0]
    requested_increase = sum(float(r["requested_increase_workers"]) for r in budgets)
    requested_decrease = sum(float(r["requested_decrease_workers"]) for r in budgets)
    executed_increase = sum(int(r["actuator_divisions"]) + int(r["actuator_recruitment"]) for r in events)
    executed_decrease = sum(int(r["actuator_apoptosis"]) + int(r["actuator_clearance"]) for r in events)
    baseline_drift = sum(int(r["baseline_divisions"]) - int(r["baseline_apoptosis"]) for r in events)
    checks = {
        "all_runs_complete": all((root / str(s["run_id"]) / "RUN_COMPLETE.json").is_file() for s in specifications),
        "all_checkpoints_complete": checkpoint_ok,
        "agent_calls_expected": len(agent) == expected_agent,
        "scgpt_calls_expected": len(scgpt) == expected_scgpt,
        "writebacks_expected": len(writebacks) == expected_writebacks,
        "budget_logs_complete": len(budgets) == expected_writebacks,
        "event_logs_complete": len(events) == expected_writebacks,
        "split_event_sources_present": all(all(k in r for k in ("baseline_divisions", "actuator_divisions", "baseline_apoptosis", "actuator_apoptosis", "actuator_recruitment", "actuator_clearance")) for r in events),
        "quantity_conservation_zero": all(error == 0 for error in conservation_errors),
        "no_fallback": not fallback,
        "no_target_leakage": not leakage_paths,
        "stable_cell_counts": min(totals) >= 300 and max(totals) <= 1200,
        "valid_compositions": all(abs(value - 1) < 1e-6 for value in composition_sums.values()),
        "fair_initial_states": fair,
        "zero_to_positive_recovery_when_requested": bool(eligible_zero) and len(recovered_zero) == len(eligible_zero),
        "no_false_emergence": not false_emergence,
        "positive_gain_defined": requested_increase > 0,
        "negative_gain_defined": requested_decrease > 0,
        "scgpt_causal_budget_feedback": len(full_budget_causal) == expected_scgpt and all(full_budget_causal),
    }
    status = f"PASS_V21_{phase.upper()}" if all(checks.values()) else f"FAIL_V21_{phase.upper()}"
    audit = {
        "status": status,
        "all_passed": all(checks.values()),
        "checks": checks,
        "actual": {
            "runs": len(specifications), "agent_calls": len(agent), "scgpt_calls": len(scgpt),
            "writebacks": len(writebacks), "budget_rows": len(budgets), "event_rows": len(events),
            "cell_count_range": [min(totals), max(totals)],
            "conservation_error_count": sum(error != 0 for error in conservation_errors),
            "zero_basis_eligible": len(eligible_zero), "zero_basis_recovered": len(recovered_zero),
            "false_emergence_count": len(false_emergence),
            "requested_increase_workers": requested_increase,
            "executed_increase_workers": executed_increase,
            "increase_gain": executed_increase / requested_increase if requested_increase else None,
            "requested_decrease_workers": requested_decrease,
            "executed_decrease_workers": executed_decrease,
            "decrease_gain": executed_decrease / requested_decrease if requested_decrease else None,
            "baseline_drift": baseline_drift,
            "actuator_net_effect": executed_increase - executed_decrease,
        },
        "leakage_paths": leakage_paths,
        "false_emergence": false_emergence,
        "normalized_internal_simulation_time": True,
        "clinical_minutes_claimed": False,
    }
    write_json(OUT / phase / f"{phase}_audit.json", audit)
    return audit


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = (p + q) / 2
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        keep = a > 0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def clr(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, float), 1e-6)
    values /= values.sum()
    logged = np.log(values)
    return logged - logged.mean()


def evaluate_validation(specifications: list[dict[str, object]], phase: str) -> dict[str, object]:
    """Read post-treatment references only here, after all runtime runs completed."""
    root = OUT / phase / "runs"
    post = read_csv(V1 / "preprocessed/posttreatment_sample_celltype_proportions.csv")
    references = {
        response: np.mean([[float(r[c]) for c in CELL_TYPES] for r in post if r["response"] == response], axis=0)
        for response in ("Responder", "Non-responder")
    }
    samples = tuple(dict.fromkeys(str(s["sample_id"]) for s in specifications))
    rows = []
    for sample in samples:
        response = next(str(s["response"]) for s in specifications if s["sample_id"] == sample)
        target = references[response]
        for model in MODELS:
            starts, ends, per_seed_budget_gain = [], [], []
            zero_eligible = zero_recovered = false_emergence = 0
            baseline_drifts = []
            for seed in SEEDS:
                run_dir = root / f"v21_{phase}__{sample}__{model}__seed_{seed}"
                trajectory = read_csv(run_dir / "composition_trajectory.csv")
                starts.append(np.array([next(float(r["proportion"]) for r in trajectory if r["checkpoint"] == "0" and r["cell_type"] == c) for c in CELL_TYPES]))
                ends.append(np.array([next(float(r["proportion"]) for r in trajectory if r["checkpoint"] == "5" and r["cell_type"] == c) for c in CELL_TYPES]))
                budget = read_csv(run_dir / "v21_budget_audit.csv")
                events = read_csv(run_dir / "v21_interval_event_audit.csv")
                requested_inc = sum(float(r["requested_increase_workers"]) for r in budget)
                requested_dec = sum(float(r["requested_decrease_workers"]) for r in budget)
                executed_inc = sum(int(r["actuator_divisions"]) + int(r["actuator_recruitment"]) for r in events)
                executed_dec = sum(int(r["actuator_apoptosis"]) + int(r["actuator_clearance"]) for r in events)
                per_seed_budget_gain.append((executed_inc / requested_inc if requested_inc else np.nan, executed_dec / requested_dec if requested_dec else np.nan))
                baseline_drifts.append(sum(int(r["baseline_divisions"]) - int(r["baseline_apoptosis"]) for r in events))
                for cell_type in CELL_TYPES:
                    start_count = next(int(r["live_cells"]) for r in trajectory if r["checkpoint"] == "0" and r["cell_type"] == cell_type)
                    end_count = next(int(r["live_cells"]) for r in trajectory if r["checkpoint"] == "5" and r["cell_type"] == cell_type)
                    requested_external = sum(float(r["requested_recruitment_workers"]) for r in budget if r["cell_type"] == cell_type)
                    if start_count == 0 and requested_external > 0:
                        zero_eligible += 1
                        zero_recovered += end_count > 0
                    if start_count == 0 and requested_external == 0 and end_count > 0:
                        false_emergence += 1
            pre = np.mean(starts, axis=0)
            pred = np.mean(ends, axis=0)
            target_distance = np.sum(np.abs(target - pre))
            change = np.sum(np.abs(pred - pre))
            rows.append({
                "sample_id": sample, "response": response, "therapy": "anti-PD1", "model": model,
                "macro_mae": float(np.mean(np.abs(pred - target))),
                "jsd": jsd(pred, target),
                "aitchison_distance": float(np.linalg.norm(clr(pred) - clr(target))),
                "direction_agreement": float(np.mean(np.sign(pred - pre) == np.sign(target - pre))),
                "change_magnitude_ratio": float(change / target_distance),
                "progress_to_target": float(1 - np.sum(np.abs(pred - target)) / target_distance),
                "zero_to_positive_recovery": float(zero_recovered / zero_eligible) if zero_eligible else None,
                "zero_to_positive_eligible": zero_eligible,
                "false_emergence_rate": float(false_emergence / (len(SEEDS) * len(CELL_TYPES))),
                "positive_actuator_gain": float(np.nanmean([x[0] for x in per_seed_budget_gain])),
                "negative_actuator_gain": float(np.nanmean([x[1] for x in per_seed_budget_gain])),
                "baseline_drift_mean_workers": float(np.mean(baseline_drifts)),
                "seed_stability_mean_celltype_sd": float(np.mean(np.std(ends, axis=0))),
            })
    metrics_path = OUT / phase / "validation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [k for k in rows[0] if k not in {"sample_id", "response", "therapy", "model"}]
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for key in numeric:
            values = [float(r[key]) for r in rows if r["model"] == model and r[key] is not None]
            summary[model][key] = float(np.mean(values)) if values else None
    result = {
        "status": "COMPLETE_V21_UNSEEN_18_RUN_VALIDATION",
        "selected_samples": list(samples),
        "therapy_matched": all(str(s["therapy"]) == "anti-PD1" for s in specifications),
        "metrics": summary,
        "post_reference_read_only_after_runtime": True,
        "expression_evaluation_used": False,
        "new_54_run_started": False,
        "formal_plot_created": False,
        "automatic_parameter_tuning_after_results": False,
    }
    write_json(OUT / phase / "validation_evaluation.json", result)
    print(json.dumps(result, indent=2))
    return result


def freeze_v21(suffix: str) -> None:
    files = sorted((ROOT / "scripts/gse120575_full_online_celltype_v21").glob("*.py"))
    files += sorted((ROOT / "scenarios/gse120575_full_online_celltype_v21/physicell_project").glob("*.cpp"))
    files += [ROOT / "scenarios/gse120575_full_online_celltype_v21/physicell_project/custom.h", ROOT / "scenarios/gse120575_full_online_celltype_v21/physicell_project/Makefile", BINARY]
    freeze_name = f"v21_frozen_hashes{suffix}.json" if suffix else "v21_frozen_hashes.json"
    write_json(OUT / "audit" / freeze_name, {
        "population_actuator": "V2.1",
        "files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in files if p.is_file()],
        "post_target_used_for_freeze": False,
    })


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY unavailable")
    static = json.loads((OUT / "audit/static_validation_v21.json").read_text())
    zero = json.loads((OUT / "zero_basis_cpp_test/zero_basis_cpp_test_summary.json").read_text())
    if not static["status"].startswith("PASS") or not zero["status"].startswith("PASS"):
        raise RuntimeError("V2.1 static or zero-basis C++ gate failed")
    suffix = os.getenv("V21_ATTEMPT_SUFFIX", "")
    compile_executor(OUT)
    freeze_v21(suffix)
    smoke_phase = "smoke" + suffix
    smoke_specifications = specs(("Pre_P1", "Pre_P12"), (12057501,), smoke_phase)
    smoke = run_phase(smoke_phase, smoke_specifications)
    print(json.dumps(smoke, indent=2))
    if not smoke["all_passed"]:
        return 2
    selected = select_unseen_validation_samples()
    validation_phase = "validation18_unseen" + suffix
    validation_specifications = specs(selected, SEEDS, validation_phase)
    validation = run_phase(validation_phase, validation_specifications)
    print(json.dumps(validation, indent=2))
    if not validation["all_passed"]:
        return 2
    evaluate_validation(validation_specifications, validation_phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
