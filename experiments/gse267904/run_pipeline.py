#!/usr/bin/env python3
"""Locked K=20/50/100 de-identified GSE267904 PhysiCell/COMMOT rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE267904_deidentified_rerun_v1"
OLD = ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"
BASE = ROOT / "scripts/gse267904_spatial_commot_agent"
WORKER_OUT = OUT / "worker_cci_dynamic_parents_v2"
LRRA_OUT = OUT / "worker_cci_lrra_dynamic_parents_v2"
K_VALUES = (20, 50, 100)
COMMOT_PYTHON_VALUE = os.environ.get("COMMOT_PYTHON", "").strip()
COMMOT_PYTHON = Path(COMMOT_PYTHON_VALUE).resolve() if COMMOT_PYTHON_VALUE else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def commot_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "NUMBA_CACHE_DIR": "/tmp/gse267904_deidentified_commot_numba_cache",
        "MPLCONFIGDIR": "/tmp/gse267904_deidentified_commot_matplotlib",
        "OMP_NUM_THREADS": "1",
    }
    if extra:
        env.update(extra)
    return env


def call(
    script: Path,
    *args: object,
    env: dict[str, str] | None = None,
    python: Path | str | None = None,
) -> None:
    interpreter = str(python) if python is not None else sys.executable
    command = [interpreter, str(script), *map(str, args)]
    print("Running:", " ".join(command), flush=True)
    effective_env = dict(env) if env is not None else dict(os.environ)
    effective_env.setdefault(
        "NUMBA_CACHE_DIR", "/tmp/gse267904_deidentified_runtime_numba_cache"
    )
    effective_env.setdefault(
        "MPLCONFIGDIR", "/tmp/gse267904_deidentified_runtime_matplotlib"
    )
    subprocess.run(command, cwd=ROOT, env=effective_env, check=True)


def frozen_hashes() -> dict[str, str]:
    paths = {
        "d7_ctrl": OLD / "real_benchmark/d7_ctrl.h5ad",
        "d7_bleo": OLD / "real_benchmark/d7_bleo.h5ad",
        "d21_ctrl": OLD / "real_benchmark/d21_ctrl.h5ad",
        "d21_bleo": OLD / "real_benchmark/d21_bleo.h5ad",
        "real_commot": OLD / "real_commot/real_commot_edges_long.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen GSE267904 inputs: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def preflight() -> dict[str, object]:
    hashes = frozen_hashes()
    failures: list[str] = []
    required_scripts = [
        BASE / "build_micro_agent_registry.py",
        BASE / "run_physicell_micro_agent_simulation.py",
        BASE / "build_virtual_commot_input.py",
        BASE / "run_virtual_commot.py",
        BASE / "evaluate_real_vs_virtual_commot.py",
        ROOT / "scripts/gse267904_agentvc_worker_scgpt_cci_main_v1/prepare_workers.py",
        ROOT / "scripts/gse267904_agentvc_worker_scgpt_cci_main_v1/benchmark.py",
        ROOT / "scripts/gse267904_multimethod_cci_benchmark_v5_agentvc_lrra/pipeline.py",
    ]
    missing_scripts = [str(path) for path in required_scripts if not path.is_file()]
    if missing_scripts:
        failures.append(f"missing scripts: {missing_scripts}")
    if not Path(sys.executable).is_file():
        failures.append(f"runtime Python missing: {sys.executable}")
    if COMMOT_PYTHON is None or not COMMOT_PYTHON.is_file():
        failures.append(f"COMMOT Python missing: {COMMOT_PYTHON}")

    runtime_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import anndata,numpy,pandas,scipy,sklearn,matplotlib,h5py,openai; print('PASS')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if runtime_probe.returncode or "PASS" not in runtime_probe.stdout:
        failures.append(f"runtime Python dependency probe failed: {runtime_probe.stderr.strip()}")

    commot_version = ""
    if COMMOT_PYTHON is not None and COMMOT_PYTHON.is_file():
        commot_probe = subprocess.run(
            [
                str(COMMOT_PYTHON),
                "-c",
                (
                    "import importlib.metadata as m; "
                    "import anndata,commot,scanpy,numpy,pandas,scipy,sklearn,matplotlib,h5py; "
                    "print(m.version('commot'))"
                ),
            ],
            cwd=ROOT,
            env=commot_environment(),
            capture_output=True,
            text=True,
        )
        commot_version = commot_probe.stdout.strip()
        if commot_probe.returncode:
            failures.append(f"COMMOT environment probe failed: {commot_probe.stderr.strip()}")
        elif commot_version != "0.0.3":
            failures.append(f"COMMOT version must be 0.0.3, found {commot_version}")

    for executable in ("make", "g++"):
        if shutil.which(executable) is None:
            failures.append(f"required executable missing: {executable}")

    resume_checks: dict[str, object] = {}
    for k in K_VALUES:
        k_out = OUT / f"K_{k}"
        generated = (k_out / "GENERATION_COMPLETE.json").is_file()
        heldout = [str(path) for path in k_out.rglob("*d21*")] if k_out.exists() else []
        resume_checks[str(k)] = {
            "generation_complete": generated,
            "heldout_files_present": len(heldout),
            "heldout_allowed_after_generation": generated,
        }
        if heldout and not generated:
            failures.append(f"K={k} has held-out files before generation completion: {heldout}")

    result = {
        "status": "PASS_PREFLIGHT_READY_FOR_DEEPSEEK" if not failures else "FAIL_PREFLIGHT",
        "experiment_role": "locked post-hoc contamination-control rerun",
        "K_values": list(K_VALUES),
        "historical_expected_llm_calls_approx": 267,
        "actual_calls_depend_on_live_agents": True,
        "deepseek_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "runtime_python": sys.executable,
        "commot_python": str(COMMOT_PYTHON),
        "commot_version": commot_version,
        "commot_cache_directory": "/tmp/gse267904_deidentified_commot_numba_cache",
        "resume_checks": resume_checks,
        "failures": failures,
        "generation_input_policy": "only d7 calibration files are staged before each PhysiCell run",
        "frozen_observed_hashes": hashes,
        "production_started": False,
    }
    write_json(OUT / "audit/preflight.json", result)
    return result


def stage_d7_only(k_out: Path) -> None:
    target = k_out / "real_benchmark"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("d7_ctrl.h5ad", "d7_bleo.h5ad"):
        shutil.copy2(OLD / "real_benchmark" / name, target / name)
    forbidden = list(k_out.rglob("*d21*"))
    if forbidden:
        raise RuntimeError(f"held-out d21 file present before generation: {forbidden}")
    write_json(
        k_out / "audit/generation_input_gate.json",
        {
            "status": "PASS_D7_ONLY_GENERATION_GATE",
            "d7_hashes": {p.name: sha256(p) for p in sorted(target.glob("*.h5ad"))},
            "d21_files_present": [],
        },
    )


def attach_frozen_evaluation_inputs(k_out: Path) -> None:
    if not (k_out / "GENERATION_COMPLETE.json").is_file():
        raise RuntimeError("cannot attach held-out evaluation inputs before generation completion")
    for directory in ("real_benchmark", "celltype_annotation", "real_commot"):
        shutil.copytree(OLD / directory, k_out / directory, dirs_exist_ok=True)
    hashes = frozen_hashes()
    observed = {
        "d7_ctrl": k_out / "real_benchmark/d7_ctrl.h5ad",
        "d7_bleo": k_out / "real_benchmark/d7_bleo.h5ad",
        "d21_ctrl": k_out / "real_benchmark/d21_ctrl.h5ad",
        "d21_bleo": k_out / "real_benchmark/d21_bleo.h5ad",
        "real_commot": k_out / "real_commot/real_commot_edges_long.csv",
    }
    if {name: sha256(path) for name, path in observed.items()} != hashes:
        raise RuntimeError("frozen Observed input hashes changed")


def run_k(k: int) -> None:
    k_out = OUT / f"K_{k}"
    if (k_out / "RUN_COMPLETE.json").is_file():
        print(f"K={k} existing PASS", flush=True)
        return
    generated = (k_out / "GENERATION_COMPLETE.json").is_file()
    if not generated:
        heldout = list(k_out.rglob("*d21*")) if k_out.exists() else []
        if heldout:
            raise RuntimeError(f"cannot resume generation with held-out files present: {heldout}")
        stage_d7_only(k_out)
    env = {
        **os.environ,
        "LECAVC_PROMPT_MODE": "deidentified_v1",
        "DEEPSEEK_MODEL": "deepseek-chat",
        "LLM_MAX_RETRIES": "4",
    }
    if not generated:
        call(BASE / "build_micro_agent_registry.py", "--project-root", ROOT, "--out-dir", k_out, "--num-agents", k)
        call(BASE / "run_physicell_micro_agent_simulation.py", "--project-root", ROOT, "--out-dir", k_out, "--max-time", 125, env=env)
        phys_audit = json.loads((k_out / "audit/physicell_writeback_audit.json").read_text())
        if not phys_audit.get("REAL_PHYSICELL_USED") or not phys_audit.get("LLM_MICRO_AGENT_RUNTIME_USED"):
            raise RuntimeError(f"K={k} lacks real PhysiCell/LLM completion evidence")
        write_json(k_out / "GENERATION_COMPLETE.json", {"status": "PASS", "K": k, "llm_calls": phys_audit.get("number_of_llm_calls")})
    attach_frozen_evaluation_inputs(k_out)
    input_manifest = k_out / "virtual_commot_input/virtual_commot_input_manifest.json"
    input_h5ads = list((k_out / "virtual_commot_input").glob("*.h5ad"))
    if not input_manifest.is_file() or len(input_h5ads) != 2:
        call(BASE / "build_virtual_commot_input.py", "--project-root", ROOT, "--out-dir", k_out)
    commot_status = k_out / "virtual_commot/virtual_commot_run_status.json"
    if not commot_status.is_file():
        call(
            BASE / "run_virtual_commot.py",
            "--project-root", ROOT,
            "--out-dir", k_out,
            env=commot_environment(),
            python=COMMOT_PYTHON,
        )
    evaluation_audit = k_out / "evaluation/commot_evaluation_audit.json"
    if not evaluation_audit.is_file():
        call(BASE / "evaluate_real_vs_virtual_commot.py", "--project-root", ROOT, "--out-dir", k_out)
    write_json(k_out / "RUN_COMPLETE.json", {"status": "PASS_REAL_PHYSICELL_BIOFVM_COMMOT", "K": k})


def audit_spatial() -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "scripts/leca_vc_prompt_isolation_v1"))
    from prompt_firewall import find_forbidden

    failures: list[str] = []
    counts: dict[str, int] = {}
    for k in K_VALUES:
        k_out = OUT / f"K_{k}"
        if not (k_out / "RUN_COMPLETE.json").is_file():
            failures.append(f"K={k} incomplete")
        paths = list(k_out.glob("micro_agents/llm_calls/*_audit/model_messages.json"))
        counts[str(k)] = len(paths)
        for path in paths:
            hits = find_forbidden(json.loads(path.read_text()))
            if hits:
                failures.append(f"{path}: {hits}")
        if not paths:
            failures.append(f"K={k} has no exact-message audits")
    result = {
        "status": "PASS_GSE267904_DEIDENTIFIED_EXECUTION" if not failures else "FAIL_GSE267904_DEIDENTIFIED_EXECUTION",
        "exact_message_counts_by_K": counts,
        "forbidden_hits": failures,
        "fallback_used": False,
    }
    write_json(OUT / "audit/execution_audit.json", result)
    return result


def render_granularity_figure() -> None:
    postprocess_complete = OUT / "postprocess/COMPLETE.json"
    if postprocess_complete.is_file():
        print("GSE267904 agent-granularity postprocess already complete", flush=True)
        return
    shared = OUT / "shared_real"
    for directory in ("real_benchmark", "real_commot"):
        shutil.copytree(OLD / directory, shared / directory, dirs_exist_ok=True)
    heterogeneity = OUT / "postprocess/agent_granularity_heterogeneity"
    module_maps = OUT / "postprocess/module_expression_maps"
    combined = OUT / "figures/agent_granularity_spatial_combined"
    if not (heterogeneity / "COMPLETE.json").is_file():
        call(
            ROOT / "scripts/gse267904_agent_granularity_heterogeneity/run_analysis.py",
            "--project-root", ROOT,
            "--frozen-root", OUT,
            "--output-dir", heterogeneity,
            "--paper-title-display",
            "--allow-new-agent-counts",
            "--resume-failed-validation",
        )
    module_map_source = (
        module_maps
        / "GSE267904_module_expression_K20_K50_K100_old_style_panel_normalized.png"
    )
    if not module_map_source.is_file():
        call(
            ROOT / "scripts/gse267904_spatial_agent_granularity/plot_module_expression_k20_k50_k100_old_style_horizontal.py",
            "--project-root", ROOT,
            "--input-root", OUT,
            "--output-dir", module_maps,
        )
    env = {
        **os.environ,
        "GSE267904_HETEROGENEITY_DATA": str(heterogeneity / "data"),
        "GSE267904_MODULE_MAP_SOURCE": str(module_map_source),
        "GSE267904_COMBINED_FIGURE_OUT": str(combined),
    }
    if not (combined / "COMPLETE.json").is_file():
        call(ROOT / "scripts/gse267904_spatial_agent_granularity/render_balanced_combined_v2.py", env=env)
    write_json(OUT / "postprocess/COMPLETE.json", {"status": "PASS_AGENT_GRANULARITY_FIGURE_REBUILT", "combined_figure_root": str(combined)})


def run_worker_cci() -> None:
    env = commot_environment({
        "GSE267904_K100_ROOT": str(OUT / "K_100"),
        "GSE267904_WORKER_CCI_OUT": str(WORKER_OUT),
        "GSE267904_LRRA_OUT": str(LRRA_OUT),
        "GSE267904_DYNAMIC_PARENT_COUNT": "1",
    })
    worker_scripts = ROOT / "scripts/gse267904_agentvc_worker_scgpt_cci_main_v1"
    call(worker_scripts / "prepare_workers.py", "all-workers", env=env, python=COMMOT_PYTHON)
    call(worker_scripts / "benchmark.py", "all-evaluation", env=env, python=COMMOT_PYTHON)
    lrra = ROOT / "scripts/gse267904_multimethod_cci_benchmark_v5_agentvc_lrra/pipeline.py"
    for stage in (
        "freeze", "prepare-d7", "smoke", "run-cv-base", "prepare-candidates",
        "run-cv-candidates", "evaluate-cv", "refit-freeze", "prepare-late",
        "commot-affine", "evaluate-affine",
    ):
        call(lrra, stage, env=env, python=COMMOT_PYTHON)
    gate = json.loads((LRRA_OUT / "05_metrics/affine_metrics_gate.json").read_text())
    if not str(gate.get("status", "")).startswith("PASS"):
        blocked = {
            "status": "BLOCKED_FROZEN_CCI_LRRA_AFFINE_GATE",
            "stage": "CCI_LRRA_GATE",
            "reason": "The frozen scientific success criteria were not all met; this is not an execution failure.",
            "failed_checks": [
                name for name, passed in gate.get("checks", {}).items() if not passed
            ],
            "gate": gate,
            "figure_generated": False,
            "deepseek_rerun_required": False,
            "physicell_rerun_required": False,
        }
        write_json(OUT / "BLOCKED.json", blocked)
        raise RuntimeError(
            "frozen CCI LRRA affine scientific gate did not pass; see "
            f"{OUT / 'BLOCKED.json'}"
        )
    for stage in ("commot-median", "evaluate-median", "finalize"):
        call(lrra, stage, env=env, python=COMMOT_PYTHON)
    call(ROOT / "scripts/gse267904_multimethod_cci_benchmark_v5_agentvc_lrra/plot_main.py", env=env)
    composite_env = {
        **env,
        "LECAVC_DEIDENTIFIED_POSTPROCESS": "1",
        "GSE267904_DEID_CCI_SOURCE": str(LRRA_OUT / "06_figure/source_data"),
        "GSE120575_DEID_EXPRESSION_ROOT": str(
            ROOT
            / "outputs/GSE120575_deidentified_rerun_v1/"
            "postprocess/expression_benchmark_v3_cellrank"
        ),
        "LECAVC_DEID_PAPER_FIGURE_OUT": str(
            ROOT / "outputs/LecaVC_paper_figures_deidentified_v1/multibenchmark"
        ),
    }
    call(
        ROOT / "scripts/leca_vc_multibenchmark_deidentified_v1/render.py",
        env=composite_env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-worker-cci", action="store_true")
    args = parser.parse_args()
    result = preflight()
    if args.preflight_only:
        print(json.dumps(result, indent=2))
        return 0
    if not result["status"].startswith("PASS"):
        write_json(OUT / "BLOCKED.json", {"stage": "preflight", **result})
        print(json.dumps(result, indent=2))
        return 2
    generation_incomplete = any(
        not (OUT / f"K_{k}/GENERATION_COMPLETE.json").is_file()
        for k in K_VALUES
    )
    if generation_incomplete and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY unavailable; production cannot start")
    for k in K_VALUES:
        run_k(k)
    execution = audit_spatial()
    if not execution["status"].startswith("PASS"):
        write_json(OUT / "BLOCKED.json", execution)
        return 2
    render_granularity_figure()
    if not args.skip_worker_cci:
        run_worker_cci()
    write_json(OUT / "COMPLETE.json", {"status": "PASS", "worker_cci_completed": not args.skip_worker_cci})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
