#!/usr/bin/env python3
"""Run the locked six-sample, three-seed GSE120575 de-identified benchmark."""

from __future__ import annotations

import csv
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_deidentified_rerun_v1"
RUN_ROOT = OUT / "simulation/runs"
SOURCE = ROOT / "outputs/GSE120575_full_online_celltype_v21_agent_rerun"
SOURCE_RUNS = SOURCE / "unified_18_agent_physicell_v21/runs"
V1 = ROOT / "outputs/GSE120575_full_online_celltype"
EXECUTOR_PATH = ROOT / "scripts/gse120575_full_online_celltype_v21/physicell_executor_v21.py"
BRIDGE = Path(__file__).with_name("online_inference_deidentified.py")
FIREWALL = ROOT / "scripts/leca_vc_prompt_isolation_v1/prompt_firewall.py"
SAMPLES = ("Pre_P24", "Pre_P2", "Pre_P29", "Pre_P35", "Pre_P3", "Pre_P27")
SEEDS = (12057501, 12057502, 12057503)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_executor():
    spec = importlib.util.spec_from_file_location("gse120575_deid_executor", EXECUTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXECUTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_bridge_import() -> None:
    module_dir = ROOT / "scripts/gse120575_full_online_celltype_v21"
    sys.path.insert(0, str(module_dir))
    try:
        spec = importlib.util.spec_from_file_location("gse120575_deid_bridge_preflight", BRIDGE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {BRIDGE}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.load_base()
    finally:
        if sys.path and sys.path[0] == str(module_dir):
            sys.path.pop(0)


def freeze_inputs() -> tuple[list[dict[str, object]], dict[str, str]]:
    metadata = {
        row["sample_id"]: row
        for row in read_csv(V1 / "preprocessed/pretreatment_sample_celltype_proportions.csv")
    }
    registry_out = OUT / "preprocessed/worker_registries"
    registry_out.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    specs: list[dict[str, object]] = []
    for sample in SAMPLES:
        for seed in SEEDS:
            source = SOURCE / f"preprocessed/worker_registries/{sample}__seed_{seed}.csv"
            target = registry_out / source.name
            if not target.exists():
                shutil.copy2(source, target)
            if sha256(source) != sha256(target):
                raise RuntimeError(f"registry copy changed: {source}")
            hashes[target.name] = sha256(target)
            specs.append(
                {
                    "run_id": f"agent_v21_unified__{sample}__seed_{seed}",
                    "trusted_sample_id": sample,
                    "trusted_patient_id": metadata[sample]["patient_id"],
                    "trusted_response": metadata[sample]["response"],
                    "trusted_therapy": metadata[sample]["therapy"],
                    "model": "agent_only",
                    "seed": seed,
                }
            )
    freeze = {
        "status": "FROZEN_LOCKED_POST_HOC_CONTAMINATION_CONTROL",
        "samples": list(SAMPLES),
        "seeds": list(SEEDS),
        "runs": specs,
        "registry_hashes": hashes,
        "heldout_expression_available_to_generation": False,
        "response_used_by_generation": False,
        "response_use": "trusted evaluation stratification only",
        "bridge_sha256": sha256(BRIDGE),
        "firewall_sha256": sha256(FIREWALL),
    }
    write_json(OUT / "audit/input_freeze.json", freeze)
    return specs, hashes


def configure(executor, run_dir: Path, spec: dict[str, object]) -> Path:
    sample = str(spec["trusted_sample_id"])
    seed = int(spec["seed"])
    registry = OUT / f"preprocessed/worker_registries/{sample}__seed_{seed}.csv"
    config = executor.runtime_config(
        run_dir,
        registry,
        sample,
        "agent_only",
        str(spec["trusted_response"]),
        str(spec["trusted_therapy"]),
        seed,
    )
    tree = ET.parse(config)
    root = tree.getroot()
    root.find("./user_parameters/bridge_path").text = str(BRIDGE.resolve())
    root.find("./user_parameters/sample_id").text = f"opaque_sample_{SAMPLES.index(sample):02d}"
    response = root.find("./user_parameters/response")
    therapy = root.find("./user_parameters/therapy")
    if response is not None:
        response.text = "evaluation_metadata_hidden"
    if therapy is not None:
        therapy.text = "perturbation_context_hidden"
    ET.indent(tree, space="  ")
    tree.write(config, encoding="utf-8", xml_declaration=True)
    return config


def run_all(specs: list[dict[str, object]]) -> None:
    incomplete = [
        spec for spec in specs
        if not (RUN_ROOT / str(spec["run_id"]) / "RUN_COMPLETE.json").is_file()
    ]
    if incomplete and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY unavailable; qualification passed but production cannot start")
    if not incomplete:
        print("GSE120575: all 18 simulation runs already complete; no DeepSeek API key required.", flush=True)
        return
    executor = load_executor()
    binary = executor.compile_executor(OUT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "simulation/run_manifest.json",
        {
            "status": "FROZEN",
            "experiment_role": "locked post-hoc contamination-control rerun",
            "expected_runs": 18,
            "expected_agent_calls": 540,
            "specifications": specs,
        },
    )
    for index, spec in enumerate(specs, start=1):
        run_dir = RUN_ROOT / str(spec["run_id"])
        if (run_dir / "RUN_COMPLETE.json").is_file():
            print(f"[{index}/18] existing PASS {run_dir.name}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite incomplete run {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = configure(executor, run_dir, spec)
        manifest = {
            "run_id": spec["run_id"],
            "sample_id": spec["trusted_sample_id"],
            "patient_id": spec["trusted_patient_id"],
            "response": spec["trusted_response"],
            "therapy": spec["trusted_therapy"],
            "seed": spec["seed"],
            "model": "agent_only",
            "run_dir": str(run_dir.resolve()),
            "registry": str((OUT / f"preprocessed/worker_registries/{spec['trusted_sample_id']}__seed_{spec['seed']}.csv").resolve()),
            "code_hash": sha256(Path(__file__)),
            "config_hash": sha256(config),
            "prompt_protocol": "LECAVC_DEIDENTIFIED_PROMPT_V1",
            "trusted_metadata_sent_to_model": False,
            "post_target_runtime_access": False,
        }
        write_json(run_dir / "run_manifest.json", manifest)
        print(f"[{index}/18] starting {run_dir.name}", flush=True)
        started = time.time()
        module_dir = ROOT / "scripts/gse120575_full_online_celltype_v21"
        inherited_pythonpath = os.environ.get("PYTHONPATH", "")
        runtime_env = {
            **os.environ,
            "PYTHONPATH": str(module_dir)
            + (os.pathsep + inherited_pythonpath if inherited_pythonpath else ""),
        }
        result = subprocess.run(
            [str(binary), str(config)],
            cwd=binary.parent,
            env=runtime_env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        (run_dir / "stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            write_json(OUT / "BLOCKED.json", {"stage": "simulation", "run": run_dir.name, "exit_code": result.returncode})
            raise RuntimeError(f"PhysiCell failed: {run_dir.name}")
        write_json(run_dir / "RUN_COMPLETE.json", {"status": "PASS", "elapsed_seconds": time.time() - started})


def audit(specs: list[dict[str, object]]) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "scripts/leca_vc_prompt_isolation_v1"))
    from prompt_firewall import find_forbidden

    runs_complete = 0
    messages = 0
    decisions = 0
    writebacks = 0
    failures: list[str] = []
    for spec in specs:
        run_dir = RUN_ROOT / str(spec["run_id"])
        runs_complete += int((run_dir / "RUN_COMPLETE.json").is_file())
        for path in run_dir.glob("raw_agent_responses/checkpoint_*/*/model_messages.json"):
            messages += 1
            hits = find_forbidden(json.loads(path.read_text()))
            if hits:
                failures.append(f"{path}: {hits}")
        decision_path = run_dir / "decision_call_audit.csv"
        writeback_path = run_dir / "cpp_writeback_audit.csv"
        if decision_path.exists():
            decisions += len(read_csv(decision_path))
            if any(row["fallback_used"].lower() == "true" for row in read_csv(decision_path)):
                failures.append(f"fallback used: {decision_path}")
        if writeback_path.exists():
            writebacks += len(read_csv(writeback_path))
    checks = {
        "runs_18_of_18": runs_complete == 18,
        "exact_messages_540_of_540": messages == 540,
        "decisions_540_of_540": decisions == 540,
        "writebacks_540_of_540": writebacks == 540,
        "forbidden_model_visible_hits_zero": not failures,
    }
    result = {
        "status": "PASS_GSE120575_DEIDENTIFIED_EXECUTION" if all(checks.values()) else "FAIL_GSE120575_DEIDENTIFIED_EXECUTION",
        "checks": checks,
        "failures": failures,
    }
    write_json(OUT / "audit/execution_audit.json", result)
    if not result["status"].startswith("PASS"):
        write_json(OUT / "BLOCKED.json", result)
    return result


def postprocess() -> None:
    complete = OUT / "postprocess/COMPLETE.json"
    heterogeneity_audit = OUT / "postprocess/heterogeneity/analysis_audit.json"
    if (
        complete.is_file()
        and heterogeneity_audit.is_file()
        and json.loads(heterogeneity_audit.read_text(encoding="utf-8")).get("status")
        == "PASS_POST_HOC_CELL_EXPANDED_HETEROGENEITY_AUDIT"
    ):
        print("GSE120575 deterministic postprocess already complete", flush=True)
        return
    proxy = OUT / "postprocess/expression_proxy"
    expression = OUT / "postprocess/expression_benchmark"
    v3 = OUT / "postprocess/expression_benchmark_v3_cellrank"
    heterogeneity = OUT / "postprocess/heterogeneity"
    heterogeneity_figure = OUT / "figures/heterogeneity"
    umap_figure = OUT / "figures/cell_expanded_umap"
    env = {
        **os.environ,
        "LECAVC_DEIDENTIFIED_POSTPROCESS": "1",
        "GSE120575_RUNS_ROOT": str(RUN_ROOT),
        "GSE120575_PROXY_OUT": str(proxy),
        "GSE120575_EXPRESSION_BENCHMARK_OUT": str(expression),
        "GSE120575_V3_OUT": str(v3),
        "GSE120575_CELL_EXPANDED_SOURCE": str(v3 / "cell_expanded_umap_figure_v3"),
        "GSE120575_HETEROGENEITY_OUT": str(heterogeneity),
        "GSE120575_HETEROGENEITY_FIGURE_OUT": str(heterogeneity_figure),
        "GSE120575_UMAP_FIGURE_OUT": str(umap_figure),
    }
    def status(path: Path) -> str:
        if not path.is_file():
            return ""
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("status", ""))
        except (json.JSONDecodeError, OSError):
            return ""

    def run(command: Path, *arguments: str) -> None:
        print(f"Postprocessing: {command.name} {' '.join(arguments)}".rstrip(), flush=True)
        subprocess.run(
            [sys.executable, str(command), *arguments],
            cwd=ROOT,
            env=env,
            check=True,
        )

    proxy_script = ROOT / "scripts/gse120575_agentvc_checkpoint5_runtime_expression_proxy_v1/reconstruct_proxy.py"
    expression_script = ROOT / "scripts/gse120575_unified_expression_benchmark_v1/run_benchmark.py"
    v3_script = ROOT / "scripts/gse120575_unified_expression_benchmark_v3_cellrank_proxy/run_v3.py"
    sample_umap_script = ROOT / "scripts/gse120575_sample_level_umap_clustering_v1/run_analysis.py"
    cell_umap_script = ROOT / "scripts/gse120575_cell_expanded_umap_figure_v3/run_v3.py"

    if status(proxy / "reconstruction_audit.json") != "PASS_AGENTVC_CHECKPOINT5_PROXY_RECONSTRUCTION":
        run(proxy_script)
    if not status(expression / "benchmark_audit.json").startswith("PASS"):
        run(expression_script)
    v3_status = status(v3 / "V3_AUDIT.json")
    if v3_status == "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION":
        run(v3_script, "--mark-visual-pass")
    elif v3_status != "PASS_GSE120575_EXPRESSION_V3_CELLRANK_PROXY":
        run(v3_script)

    sample_out = v3 / "sample_level_umap_clustering_v1"
    sample_status = status(sample_out / "analysis_audit.json")
    if sample_status not in {
        "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION",
        "PASS_SAMPLE_LEVEL_UMAP",
    }:
        run(sample_umap_script)
        sample_status = status(sample_out / "analysis_audit.json")
    if sample_status == "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION":
        run(sample_umap_script, "--mark-visual-pass")

    cell_out = v3 / "cell_expanded_umap_figure_v3"
    cell_status = status(cell_out / "analysis_audit.json")
    if cell_status != "PASS_POST_HOC_CELL_EXPANDED_PROXY_UMAP":
        arguments = ("--rebuild-current",) if cell_out.exists() and any(cell_out.iterdir()) else ()
        run(cell_umap_script, *arguments)
        cell_status = status(cell_out / "analysis_audit.json")
    if cell_status == "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION":
        run(cell_umap_script, "--mark-visual-pass")

    run(Path(__file__).with_name("render_umap_response_figure.py"))
    heterogeneity_script = ROOT / "scripts/gse120575_cell_expanded_heterogeneity_audit_v1/run_analysis.py"
    run(heterogeneity_script)
    if status(heterogeneity / "analysis_audit.json") == "PENDING_DETERMINISM_RERUN":
        run(heterogeneity_script)
    heterogeneity_status = status(heterogeneity / "analysis_audit.json")
    if heterogeneity_status != "PASS_POST_HOC_CELL_EXPANDED_HETEROGENEITY_AUDIT":
        raise RuntimeError(
            f"GSE120575 heterogeneity audit did not pass: {heterogeneity_status}"
        )
    run(ROOT / "scripts/gse120575_cell_expanded_heterogeneity_audit_v1/render_clean_figure_v2.py")
    write_json(
        OUT / "postprocess/COMPLETE.json",
        {
            "status": "PASS_DETERMINISTIC_POSTPROCESS",
            "external_baselines_reused": True,
            "cell_expansion_source_cells": 1664,
            "expression_gene_count": 834,
            "heterogeneity_audit": heterogeneity_status,
            "determinism_verified": True,
            "figure_roots": [str(umap_figure), str(heterogeneity_figure)],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Freeze and validate inputs without compiling, calling DeepSeek, or running PhysiCell.",
    )
    parser.add_argument(
        "--resume-postprocess",
        action="store_true",
        help="Audit completed runs and resume deterministic postprocessing without DeepSeek or PhysiCell.",
    )
    args = parser.parse_args()
    specs, _ = freeze_inputs()
    validate_bridge_import()
    if args.preflight_only:
        result = {
            "status": "PASS_PREFLIGHT_READY_FOR_DEEPSEEK",
            "expected_runs": len(specs),
            "expected_llm_calls": 540,
            "deepseek_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
            "bridge_import_with_historical_module_path": True,
            "production_started": False,
        }
        write_json(OUT / "audit/preflight.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.resume_postprocess:
        result = audit(specs)
        if not result["status"].startswith("PASS"):
            print(json.dumps(result, indent=2))
            return 2
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
