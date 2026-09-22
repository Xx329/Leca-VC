#!/usr/bin/env python3
"""Preflight or launch a locked full benchmark; never substitutes a fallback."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "fibrosis_application": ROOT / "experiments/fibrosis_application/application/run_pipeline.py",
    "gse2565": ROOT / "experiments/gse2565/run_pipeline.py",
    "gse120575": ROOT / "experiments/gse120575/run_pipeline.py",
    "gse267904": ROOT / "experiments/gse267904/run_pipeline.py",
}


def check_imports(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def preflight(dataset: str, require_online: bool) -> dict:
    failures: list[str] = []
    entrypoint = ENTRYPOINTS[dataset]
    if not entrypoint.is_file(): failures.append(f"missing entrypoint: {entrypoint.relative_to(ROOT)}")
    if dataset == "fibrosis_application":
        failures.append(
            "collaborator V3 is paused: qualification multiplier anomaly and failed scientific gate require contributor resolution"
        )
        failures.append(
            "Figure 6 manual paper assembly is available, but a clean full rerun remains blocked by the truncated output archive and unreconciled model-lock/completion evidence"
        )
    marker = ROOT / f"artifacts/processed_inputs/{dataset}/READY.json"
    if not marker.is_file(): failures.append(f"processed-input artifact not installed: {marker.relative_to(ROOT)}")
    modules = ("numpy", "pandas", "scipy", "matplotlib", "anndata", "sklearn", "h5py")
    missing_modules = check_imports(modules)
    if missing_modules: failures.append(f"missing Python modules: {missing_modules}")
    physicell = Path(os.environ.get("PHYSICELL_ROOT", str(ROOT / "vendor/PhysiCell"))).resolve()
    if not (physicell / "VERSION.txt").is_file(): failures.append("PhysiCell 1.14.2 checkout not found; set PHYSICELL_ROOT")
    if dataset == "gse267904":
        commot_python = os.environ.get("COMMOT_PYTHON", "").strip()
        if not commot_python or not Path(commot_python).is_file(): failures.append("COMMOT_PYTHON is not set to the dedicated COMMOT environment")
    if require_online and not os.environ.get("DEEPSEEK_API_KEY", "").strip(): failures.append("DEEPSEEK_API_KEY is not set; no fallback is permitted")
    return {"dataset": dataset, "status": "PASS" if not failures else "BLOCKED", "full_run_requested": require_online, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(ENTRYPOINTS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    report = preflight(args.dataset, require_online=args.run)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS": return 2
    if args.preflight: return 0
    if args.dataset == "fibrosis_application":
        raise RuntimeError("unreachable: the paused fibrosis application cannot pass release preflight")
    env = {**os.environ, "LECAVC_ROOT": str(ROOT), "V4_RUNTIME_BRIDGE_PATH": str(ROOT / "scripts/gse2565_bulk_macro_benchmark_v4_online_agent/runtime_bridge.py")}
    command = [sys.executable, str(ENTRYPOINTS[args.dataset])]
    if args.dataset == "gse267904": env["COMMOT_PYTHON"] = str(Path(env["COMMOT_PYTHON"]).resolve())
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__": raise SystemExit(main())
