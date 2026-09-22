#!/usr/bin/env python3
"""Locked GSE2565 DeepSeek/PhysiCell rerun with model-facing de-identification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE2565_deidentified_rerun_v1"
SOURCE = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v4_online_agent"
BASE_ENGINE = ROOT / "scripts/gse2565_bulk_macro_benchmark_v1/physicell_engine.py"
V4_ENGINE = ROOT / "scripts/gse2565_bulk_macro_benchmark_v4_online_agent/v4_online_engine.py"
BRIDGE = ROOT / "scripts/gse2565_bulk_macro_benchmark_v4_online_agent/runtime_bridge.py"
CONTROLLER = ROOT / "scripts/gse2565_bulk_macro_benchmark_v4_online_agent/online_controller.py"
FIREWALL = ROOT / "scripts/leca_vc_prompt_isolation_v1/prompt_firewall.py"
PROJECT = ROOT / "scenarios/gse2565_bulk_macro_benchmark_v4_online_agent/physicell_project"
SEEDS = (256501, 256502, 256503)
CONDITIONS = ("virtual_air_control", "virtual_phosgene_injury")
CHECKPOINTS = (0, 30, 60, 240, 480, 720, 1440, 2880)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_inputs() -> list[dict[str, object]]:
    (OUT / "initialization").mkdir(parents=True, exist_ok=True)
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for relative in (
        "initialization/worker_registry.csv",
        "initialization/healthy_expression_prototypes.npz",
        "audit/online_agent_config.json",
    ):
        source = SOURCE / relative
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"frozen input changed while copying {relative}")
        copied[relative] = sha256(target)
    specs = [
        {
            "run_id": f"agent_physicell_v4_online__{condition}__seed_{seed}__to_72h",
            "condition": condition,
            "seed": seed,
        }
        for seed in SEEDS
        for condition in CONDITIONS
    ]
    write_json(
        OUT / "audit/input_freeze.json",
        {
            "status": "FROZEN_LOCKED_POST_HOC_CONTAMINATION_CONTROL",
            "runs": specs,
            "expected_runs": 6,
            "expected_llm_calls": 480,
            "frozen_input_hashes": copied,
            "controller_sha256": sha256(CONTROLLER),
            "bridge_sha256": sha256(BRIDGE),
            "firewall_sha256": sha256(FIREWALL),
            "real_outcome_available_to_generation": False,
        },
    )
    return specs


def compile_executor() -> Path:
    env = {**os.environ, "LECAVC_OUTPUT_ROOT": str(OUT)}
    result = subprocess.run(
        [sys.executable, str(V4_ENGINE)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    (OUT / "audit/compile_stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError("GSE2565 PhysiCell compile failed")
    binary = PROJECT / "gse2565_bulk_macro_v4_online"
    if not binary.is_file():
        raise RuntimeError(f"compiled binary missing: {binary}")
    return binary


def run_all(specs: list[dict[str, object]]) -> None:
    incomplete = [
        item for item in specs
        if not (
            OUT / "simulation/runs" / str(item["run_id"]) / "RUN_COMPLETE.json"
        ).is_file()
    ]
    if incomplete and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY unavailable; production cannot start")
    if not incomplete:
        print("GSE2565: all six simulation runs already complete; no DeepSeek API key required.", flush=True)
        return
    base = load(BASE_ENGINE, "gse2565_deidentified_base")
    binary = compile_executor()
    write_json(
        OUT / "pilot/run_manifest.json",
        [
            {
                **item,
                "run_dir": str((OUT / "simulation/runs" / str(item["run_id"])).relative_to(ROOT)),
                "exit_code": 0,
                "REAL_PHYSICELL_USED": True,
                "dynamic_decisions": 80,
                "checkpoints_complete": 8,
            }
            for item in specs
        ],
    )
    for index, item in enumerate(specs, start=1):
        run_dir = OUT / "simulation/runs" / str(item["run_id"])
        if (run_dir / "RUN_COMPLETE.json").is_file():
            print(f"[{index}/6] existing PASS {run_dir.name}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite incomplete run {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = run_dir / "PhysiCell_settings.xml"
        base.runtime_xml(
            PROJECT / "PhysiCell_settings_base.xml",
            config,
            run_dir,
            OUT / "initialization/worker_registry.csv",
            OUT / "unused_policy.csv",
            str(item["condition"]),
            int(item["seed"]),
            72,
        )
        write_json(
            run_dir / "trusted_run_manifest.json",
            {
                **item,
                "experiment_role": "locked post-hoc contamination-control rerun",
                "trusted_condition_sent_to_model": False,
                "physical_time_sent_to_model": False,
                "normalized_step_sent_to_model": True,
                "config_sha256": sha256(config),
            },
        )
        env = {
            **os.environ,
            "LECAVC_OUTPUT_ROOT": str(OUT),
            "LECAVC_PROMPT_MODE": "deidentified_v1",
            "LECAVC_LLM_MODEL": "deepseek-chat",
            "V4_RUNTIME_BRIDGE_PATH": str(BRIDGE),
            "V4_ONLINE_TIMEOUT_SECONDS": "900",
        }
        print(f"[{index}/6] starting {run_dir.name}", flush=True)
        started = time.time()
        result = subprocess.run(
            [str(binary), str(config)], cwd=PROJECT, env=env,
            capture_output=True, text=True, timeout=7200,
        )
        (run_dir / "stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            write_json(OUT / "BLOCKED.json", {"stage": "simulation", "run": run_dir.name, "exit_code": result.returncode})
            raise RuntimeError(f"PhysiCell failed: {run_dir.name}")
        write_json(run_dir / "RUN_COMPLETE.json", {"status": "PASS", "elapsed_seconds": time.time() - started})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(specs: list[dict[str, object]]) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "scripts/leca_vc_prompt_isolation_v1"))
    from prompt_firewall import find_forbidden

    failures: list[str] = []
    complete = calls = writebacks = exact_messages = 0
    for item in specs:
        run_dir = OUT / "simulation/runs" / str(item["run_id"])
        complete += int((run_dir / "RUN_COMPLETE.json").is_file())
        for path in run_dir.glob("agent_decisions/*/*/model_messages.json"):
            exact_messages += 1
            hits = find_forbidden(json.loads(path.read_text()))
            if hits:
                failures.append(f"{path}: {hits}")
        call_path = OUT / "audit/deepseek_runtime_call_audit.csv"
        action_path = run_dir / "agent_action_execution_audit.csv"
        if action_path.exists():
            writebacks += len(read_csv(action_path))
    call_path = OUT / "audit/deepseek_runtime_call_audit.csv"
    if call_path.exists():
        rows = read_csv(call_path)
        calls = len(rows)
        if any(row.get("fallback_used", "").lower() == "true" for row in rows):
            failures.append("fallback recorded in DeepSeek audit")
    checks = {
        "runs_6_of_6": complete == 6,
        "calls_480_of_480": calls == 480,
        "writebacks_480_of_480": writebacks == 480,
        "exact_messages_480_of_480": exact_messages == 480,
        "forbidden_hits_zero": not failures,
    }
    result = {"status": "PASS_GSE2565_DEIDENTIFIED_EXECUTION" if all(checks.values()) else "FAIL_GSE2565_DEIDENTIFIED_EXECUTION", "checks": checks, "failures": failures}
    write_json(OUT / "audit/execution_audit.json", result)
    return result


def consolidate_action_audits() -> None:
    """Rebuild the historical controller-level audit from six per-run audits."""
    manifest = json.loads((OUT / "pilot/run_manifest.json").read_text())
    rows: list[dict[str, object]] = []
    air_rows: list[dict[str, object]] = []
    all_minutes = (*CHECKPOINTS, 4320)
    for run in manifest:
        run_dir = ROOT / str(run["run_dir"])
        run_id = str(run["run_id"])
        cpp = pd.read_csv(run_dir / "agent_action_execution_audit.csv")
        if len(cpp) != 80 or len(cpp.drop_duplicates(["minute", "agent_id"])) != 80:
            raise RuntimeError(f"incomplete per-run action audit: {run_id}")
        for minute in CHECKPOINTS:
            response_path = (
                run_dir / "runtime_responses" / run_id / str(minute) / "full_response.json"
            )
            response = json.loads(response_path.read_text())
            if response.get("fallback_used") is not False or response.get("schema_valid") is not True:
                raise RuntimeError(f"invalid saved response envelope: {response_path}")
            agents = {item["agent_id"]: item for item in response["agents"]}
            current = cpp.loc[cpp.minute.eq(minute)]
            if set(current.agent_id.astype(str)) != set(agents):
                raise RuntimeError(f"agent identity mismatch at {run_id}/{minute}")
            for _, item in current.iterrows():
                agent = agents[str(item.agent_id)]
                rows.append(
                    {
                        "run_id": run_id,
                        "condition": run["condition"],
                        "seed": run["seed"],
                        "minute": minute,
                        "agent_id": item.agent_id,
                        "request_hash": item.request_hash,
                        "response_hash": item.response_hash,
                        "raw_program_strengths": json.dumps(
                            agent["raw_decision"]["program_strengths"], sort_keys=True
                        ),
                        "smoothed_program_strengths": json.dumps(
                            agent["executed_action"], sort_keys=True
                        ),
                        "actual_secretion_parameters": json.dumps(
                            {
                                "oxidative": item.oxidative_stress,
                                "injury": item.epithelial_injury,
                                "inflammation": item.inflammation,
                                "edema": item.edema_proxy,
                                "death": item.death_signal,
                                "repair": item.repair_signal,
                            },
                            sort_keys=True,
                        ),
                        "actual_uptake_parameter": item.toxicant_uptake,
                        "actual_motility_parameter": item.motility,
                        "death_tendency": item.death_tendency,
                        "repair_tendency": item.repair_tendency,
                        "abundance_request": json.dumps(
                            agent["raw_decision"]["abundance_transition_request"],
                            sort_keys=True,
                        ),
                        "worker_count": item.worker_count,
                        "represented_abundance": item.represented_abundance,
                        "executed": True,
                        "fallback_used": False,
                        "confidence": agent["raw_decision"]["confidence"],
                    }
                )
        if run["condition"] == "virtual_air_control":
            baseline = (
                pd.read_csv(run_dir / "work_cells_minute_0.csv")
                .groupby("agent_id")
                .represented_abundance.sum()
            )
            for minute in all_minutes:
                cells = pd.read_csv(run_dir / f"work_cells_minute_{minute}.csv")
                abundance = cells.groupby("agent_id").represented_abundance.sum()
                state_ok = all(
                    cells[column].mean() <= 0.01 and cells[column].max() <= 0.03
                    for column in (
                        "damage_memory",
                        "injury_state",
                        "inflammatory_memory",
                        "inflammation",
                    )
                )
                maximum = float((abundance - baseline).abs().max())
                l1 = float((abundance - baseline).abs().sum())
                total = float(cells.represented_abundance.sum())
                air_rows.append(
                    {
                        "run_id": run_id,
                        "seed": run["seed"],
                        "minute": minute,
                        "state_threshold_passed": state_ok,
                        "max_cell_type_abundance_drift": maximum,
                        "abundance_l1_drift": l1,
                        "abundance_sum": total,
                        "abundance_passed": maximum <= 0.01 and l1 <= 0.02 and abs(total - 1) <= 1e-6,
                    }
                )
    action = pd.DataFrame(rows)
    air = pd.DataFrame(air_rows)
    if len(action) != 480 or action.fallback_used.any():
        raise RuntimeError("consolidated action audit failed 480-row/no-fallback gate")
    if len(air) != 27 or not air.state_threshold_passed.all() or not air.abundance_passed.all():
        raise RuntimeError("air-control stability gate failed")
    action_path = OUT / "audit/agent_action_execution_audit.csv"
    air_path = OUT / "audit/air_stability_audit.csv"
    action.to_csv(action_path, index=False)
    air.to_csv(air_path, index=False)
    write_json(
        OUT / "audit/action_consolidation_audit.json",
        {
            "status": "PASS_480_ACTIONS_AND_AIR_STABILITY",
            "action_rows": len(action),
            "air_rows": len(air),
            "source_run_count": len(manifest),
            "no_new_llm_calls": True,
            "action_sha256": sha256(action_path),
            "air_sha256": sha256(air_path),
        },
    )


def postprocess() -> None:
    if (OUT / "postprocess/COMPLETE.json").is_file():
        print("GSE2565 deterministic postprocess already complete", flush=True)
        return
    consolidate_action_audits()
    env = {**os.environ, "LECAVC_OUTPUT_ROOT": str(OUT)}
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/gse2565_bulk_macro_benchmark_v4_online_agent/v4_online_expression_amend004.py"),
            "--stage", "pilot", "--manifest", str(OUT / "pilot/run_manifest.json"),
        ],
        cwd=ROOT, env=env, check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/gse2565_bulk_macro_benchmark_v4_online_agent/evaluate_amend004.py")],
        cwd=ROOT, env=env, check=True,
    )
    old_v52 = ROOT / "outputs/GSE2565_bulk_master_comparison_v5_2_bop_gpr"
    new_v52 = OUT / "postprocess/bulk_master_comparison_v5_2"
    new_v52.mkdir(parents=True, exist_ok=True)
    for name in (
        "BOPDMD_FORMAL_AUDIT.json", "FROZEN_PROTOCOL_V5_2.json",
        "bopdmd_rank_sensitivity_even_ranks.csv", "bopdmd_reconstructed_trajectory.npz",
        "bopdmd_dnb_bootstrap_trajectories_B20.npz", "bopdmd_dnb_bootstrap_curve.csv",
    ):
        shutil.copy2(old_v52 / name, new_v52 / name)
    figure_out = OUT / "figures/bulk_time_resolved_benchmark"
    render_env = {
        **env,
        "LECAVC_DEIDENTIFIED_POSTPROCESS": "1",
        "GSE2565_V52_OUT": str(new_v52),
        "GSE2565_PAPER_FIGURE_OUT": str(figure_out),
    }
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/gse2565_bulk_master_comparison_v5_2_bop_gpr/build_v5_2.py")],
        cwd=ROOT, env=render_env, check=True,
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("render_paper_figure.py"))],
        cwd=ROOT, env=render_env, check=True,
    )
    write_json(
        OUT / "postprocess/COMPLETE.json",
        {
            "status": "PASS_EXPRESSION_AND_METRICS_REBUILT",
            "external_baselines_reused": True,
            "paper_figure_root": str(figure_out),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resume-postprocess",
        action="store_true",
        help="Use six audited completed runs; make no DeepSeek or PhysiCell calls.",
    )
    args = parser.parse_args()
    specs = freeze_inputs()
    if args.preflight_only:
        result = {
            "status": "PASS_PREFLIGHT_READY_FOR_DEEPSEEK",
            "expected_runs": 6,
            "expected_llm_calls": 480,
            "deepseek_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
            "production_started": False,
        }
        write_json(OUT / "audit/preflight.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.resume_postprocess:
        result = audit(specs)
        if not result["status"].startswith("PASS"):
            raise RuntimeError("cannot resume postprocess: execution audit is not PASS")
        postprocess()
        print(json.dumps(result, indent=2))
        return 0
    run_all(specs)
    result = audit(specs)
    if result["status"].startswith("PASS"):
        postprocess()
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
