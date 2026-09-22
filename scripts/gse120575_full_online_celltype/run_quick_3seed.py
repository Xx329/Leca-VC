#!/usr/bin/env python3
"""Run the explicitly authorized frozen 54-run GSE120575 quick experiment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from controllers import CELL_TYPES
from physicell_executor import BINARY, runtime_config
from run_smoke import OUT, read_csv, structured_leakage_audit, write_json


QUICK = OUT / "quick_3seed"
MODELS = ("traditional", "agent_only", "full")
SEEDS = (12057501, 12057502, 12057503)
EXPECTED = {"runs": 54, "agent_calls": 1080, "scgpt_calls": 540, "cpp_writebacks": 1620}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze() -> dict[str, object]:
    corrected = json.loads((OUT / "smoke/smoke_audit_corrected.json").read_text())
    if not corrected.get("qualified_for_54_run_quick"):
        raise RuntimeError("Corrected smoke audit does not qualify for 54-run quick")
    freeze = json.loads((OUT / "audit/quick_freeze_hashes.json").read_text())
    mismatches = []
    for item in freeze["files"]:
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        if actual != item["sha256"]:
            mismatches.append({"path": str(path), "expected": item["sha256"], "actual": actual})
    if mismatches:
        raise RuntimeError("Frozen file hash mismatch; quick not started:\n" + json.dumps(mismatches, indent=2))
    if sha256(BINARY) != freeze["category_sha256"]["binary"]:
        # Category hashes include path framing, so verify against the explicit binary entry instead.
        binary_entries = [item for item in freeze["files"] if item["category"] == "binary"]
        if len(binary_entries) != 1 or sha256(BINARY) != binary_entries[0]["sha256"]:
            raise RuntimeError("Frozen PhysiCell binary mismatch; quick not started")
    if freeze["design_expected_counts"] != {
        "expected_runs": EXPECTED["runs"],
        "expected_agent_calls": EXPECTED["agent_calls"],
        "expected_scgpt_calls": EXPECTED["scgpt_calls"],
        "expected_cpp_writebacks": EXPECTED["cpp_writebacks"],
    }:
        raise RuntimeError("Frozen expected counts differ from runner constants")
    return freeze


def completed_run_valid(run_dir: Path) -> bool:
    marker = run_dir / "RUN_COMPLETE.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("status") != "PASS":
        return False
    required = [
        run_dir / "decision_call_audit.csv", run_dir / "cpp_writeback_audit.csv",
        run_dir / "composition_trajectory.csv", run_dir / "cell_count_trajectory.csv",
        run_dir / "biofvm_trajectory.csv", run_dir / "state_checkpoint_5.csv",
    ] + [run_dir / f"execution_payload_{checkpoint}.json" for checkpoint in range(5)]
    return all(path.is_file() for path in required)


def build_specs(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    specs = []
    for sample in samples:
        for seed in SEEDS:
            for model in MODELS:
                sample_id = str(sample["sample_id"])
                specs.append({
                    "run_id": f"quick__{sample_id}__{model}__seed_{seed}",
                    "sample_id": sample_id,
                    "patient_id": sample["patient_id"],
                    "response": sample["response"],
                    "therapy": sample["therapy"],
                    "model": model,
                    "seed": seed,
                })
    if len(specs) != EXPECTED["runs"]:
        raise RuntimeError(f"Expected 54 run specs, got {len(specs)}")
    return specs


def evaluate(specs: list[dict[str, object]]) -> dict[str, object]:
    decisions, scgpt_rows, writeback = [], [], []
    compositions, counts, biofvm = [], [], []
    checkpoint_status, run_records = [], []
    initial_states: dict[tuple[str, int, str], bytes] = {}
    for spec in specs:
        run_dir = QUICK / "runs" / str(spec["run_id"])
        record = {**spec, "run_dir": str(run_dir.resolve()), "exit_code": 0}
        run_records.append(record)
        decisions.extend(read_csv(run_dir / "decision_call_audit.csv"))
        if (run_dir / "scgpt_call_audit.csv").is_file():
            scgpt_rows.extend(read_csv(run_dir / "scgpt_call_audit.csv"))
        writeback.extend(read_csv(run_dir / "cpp_writeback_audit.csv"))
        for row in read_csv(run_dir / "composition_trajectory.csv"):
            compositions.append({**row, "sample_id": spec["sample_id"], "model": spec["model"], "seed": spec["seed"]})
        for row in read_csv(run_dir / "cell_count_trajectory.csv"):
            counts.append({**row, "sample_id": spec["sample_id"], "model": spec["model"], "seed": spec["seed"]})
        for row in read_csv(run_dir / "biofvm_trajectory.csv"):
            biofvm.append({**row, "sample_id": spec["sample_id"], "model": spec["model"], "seed": spec["seed"]})
        completed = [
            checkpoint for checkpoint in range(5)
            if (run_dir / f"execution_payload_{checkpoint}.json").is_file()
            and (run_dir / f"execution_parameters_{checkpoint}.csv").is_file()
        ]
        checkpoint_status.append({
            "run_id": spec["run_id"], "completed": completed,
            "endpoint": (run_dir / "state_checkpoint_5.csv").is_file(),
        })
        initial_states[(str(spec["sample_id"]), int(spec["seed"]), str(spec["model"]))] = (run_dir / "state_checkpoint_0.csv").read_bytes()

    agent = [row for row in decisions if row["model"] in {"agent_only", "full"}]
    full = [row for row in decisions if row["model"] == "full"]
    fallback = [row for row in decisions if row["fallback_used"].lower() == "true"]
    composition_sums: dict[tuple[str, str, str, str], float] = {}
    for row in compositions:
        key = (str(row["sample_id"]), str(row["seed"]), str(row["model"]), str(row["checkpoint"]))
        composition_sums[key] = composition_sums.get(key, 0.0) + float(row["proportion"])
    totals = [int(row["total_live"]) for row in counts]
    finite_biofvm = all(
        np.isfinite(float(row[field]))
        for row in biofvm
        for field in ("treatment_signal", "inflammatory_signal", "suppressive_signal", "survival_signal", "stress_signal")
    )
    fairness = all(
        initial_states[(sample, seed, "traditional")] == initial_states[(sample, seed, "agent_only")] == initial_states[(sample, seed, "full")]
        for sample in {str(spec["sample_id"]) for spec in specs}
        for seed in SEEDS
    )
    leakage = structured_leakage_audit(run_records)
    checks = {
        "fifty_four_runs_complete": len(specs) == EXPECTED["runs"] and all(completed_run_valid(QUICK / "runs" / str(spec["run_id"])) for spec in specs),
        "agent_calls_equal_1080": len(agent) == EXPECTED["agent_calls"],
        "scgpt_calls_equal_540": len(scgpt_rows) == EXPECTED["scgpt_calls"],
        "cpp_writebacks_equal_1620": len(writeback) == EXPECTED["cpp_writebacks"] and all(row["writeback_executed"].lower() == "true" for row in writeback),
        "all_checkpoints_and_endpoints_complete": all(row["completed"] == [0, 1, 2, 3, 4] and row["endpoint"] for row in checkpoint_status),
        "no_fallback": not fallback,
        "no_real_target_leakage": not leakage["real_target_leakage_detected"],
        "full_scgpt_causal_feedback_used": len(full) == EXPECTED["scgpt_calls"] and all(
            row["scgpt_prompt_field_present"].lower() == "true"
            and row["scgpt_causal_feedback_used"].lower() == "true" for row in full
        ),
        "identical_initial_state_across_models": fairness,
        "valid_compositions": len(compositions) == EXPECTED["runs"] * 6 * len(CELL_TYPES) and all(abs(value - 1.0) < 1e-6 for value in composition_sums.values()),
        "stable_cell_counts": len(counts) == EXPECTED["runs"] * 6 and min(totals) >= 300 and max(totals) <= 1200,
        "finite_biofvm": len(biofvm) == EXPECTED["runs"] * 6 and finite_biofvm,
    }
    return {
        "status": "PASS_54_RUN_QUICK_EXECUTION" if all(checks.values()) else "FAIL_54_RUN_QUICK_EXECUTION",
        "all_passed": all(checks.values()),
        "checks": checks,
        "runs_actual": len(specs), "runs_expected": EXPECTED["runs"],
        "agent_calls_actual": len(agent), "agent_calls_expected": EXPECTED["agent_calls"],
        "scgpt_calls_actual": len(scgpt_rows), "scgpt_calls_expected": EXPECTED["scgpt_calls"],
        "cpp_writebacks_actual": len(writeback), "cpp_writebacks_expected": EXPECTED["cpp_writebacks"],
        "cell_count_min": min(totals), "cell_count_max": max(totals),
        "checkpoint_status": checkpoint_status,
        "fallback_rows": fallback,
        "structured_leakage_audit": leakage,
        "formal_box_plot_started": False,
    }


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY unavailable; quick not started")
    freeze = verify_freeze()
    quick_manifest = json.loads((OUT / "audit/quick_manifest_frozen.json").read_text())
    if quick_manifest["design"]["response_and_therapy_completely_confounded"]:
        raise RuntimeError("Frozen quick manifest reports complete response/therapy confounding")
    specs = build_specs(quick_manifest["samples"])
    QUICK.mkdir(parents=True, exist_ok=True)
    (QUICK / "runs").mkdir(exist_ok=True)
    write_json(QUICK / "quick_run_manifest.json", {
        "status": "AUTHORIZED_54_RUN_QUICK",
        "freeze_sha256": freeze["overall_freeze_sha256"],
        "expected": EXPECTED,
        "runs": specs,
        "formal_box_plot_authorized": False,
    })

    progress = []
    progress_path = QUICK / "quick_progress.json"
    for index, spec in enumerate(specs, 1):
        run_dir = QUICK / "runs" / str(spec["run_id"])
        if run_dir.exists() and any(run_dir.iterdir()):
            if completed_run_valid(run_dir):
                progress.append({**spec, "status": "PASS_EXISTING", "ordinal": index})
                write_json(progress_path, {"completed": progress, "target": EXPECTED["runs"]})
                print(f"[{index}/54] existing PASS {spec['run_id']}", flush=True)
                continue
            raise RuntimeError(f"Incomplete existing run requires audit before resume; refusing overwrite: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        registry = OUT / "preprocessed/worker_registries" / f"{spec['sample_id']}__seed_{spec['seed']}.csv"
        config = runtime_config(
            run_dir, registry, str(spec["sample_id"]), str(spec["model"]),
            str(spec["response"]), str(spec["therapy"]), int(spec["seed"]),
        )
        write_json(run_dir / "run_manifest.json", {
            **spec, "run_dir": str(run_dir.resolve()), "registry": str(registry.resolve()),
            "REAL_PHYSICELL_USED": True, "SYNTHETIC_SPATIAL_INITIALIZATION": True,
            "REAL_SPATIAL_RECONSTRUCTION_CLAIMED": False,
            "freeze_sha256": freeze["overall_freeze_sha256"],
            "code_hash": freeze["category_sha256"]["code"],
            "config_hash": sha256(config),
            "binary_sha256": sha256(BINARY),
            "decision_checkpoints": [0, 1, 2, 3, 4], "endpoint_checkpoint": 5,
            "normalized_percent": [0, 20, 40, 60, 80, 100],
        })
        print(f"[{index}/54] starting {spec['run_id']}", flush=True)
        start = time.time()
        result = subprocess.run(
            [str(BINARY), str(config)], cwd=BINARY.parent,
            capture_output=True, text=True, timeout=1800,
        )
        (run_dir / "stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr)
        elapsed = time.time() - start
        if result.returncode != 0:
            failure = {**spec, "status": "FAIL", "exit_code": result.returncode, "elapsed_seconds": elapsed, "ordinal": index}
            write_json(QUICK / "QUICK_FAILED.json", failure)
            raise RuntimeError(f"Quick failed at {spec['run_id']}; stopped immediately")
        write_json(run_dir / "RUN_COMPLETE.json", {"status": "PASS", "exit_code": 0, "elapsed_seconds": elapsed})
        progress.append({**spec, "status": "PASS", "elapsed_seconds": elapsed, "ordinal": index})
        write_json(progress_path, {"completed": progress, "target": EXPECTED["runs"]})
        print(f"[{index}/54] PASS {spec['run_id']} ({elapsed:.1f}s)", flush=True)

    audit = evaluate(specs)
    write_json(QUICK / "quick_execution_audit.json", audit)
    report = [
        "# GSE120575 54-run quick execution report", "",
        f"- Status: **{audit['status']}**",
        f"- Runs: {audit['runs_actual']} / {audit['runs_expected']}",
        f"- Agent calls: {audit['agent_calls_actual']} / {audit['agent_calls_expected']}",
        f"- scGPT calls: {audit['scgpt_calls_actual']} / {audit['scgpt_calls_expected']}",
        f"- C++ writebacks: {audit['cpp_writebacks_actual']} / {audit['cpp_writebacks_expected']}",
        f"- Cell-count range: {audit['cell_count_min']}–{audit['cell_count_max']}",
        "- Formal box plot was not started.", "", "## Gates", "",
    ] + [f"- {key}: {value}" for key, value in audit["checks"].items()]
    (QUICK / "quick_execution_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({
        "status": audit["status"],
        "runs": audit["runs_actual"],
        "agent_calls": audit["agent_calls_actual"],
        "scgpt_calls": audit["scgpt_calls_actual"],
        "cpp_writebacks": audit["cpp_writebacks_actual"],
        "formal_box_plot_started": False,
    }, ensure_ascii=False, indent=2))
    return 0 if audit["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
