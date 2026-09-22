#!/usr/bin/env python3
"""Stage-oriented command line entrypoint for the V3 paper experiment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_OUT


SCRIPTS = {
    "prepare-d7": "prepare_d7.py",
    "import-frozen-t0-policies": "import_frozen_t0_policies.py",
    "calibrate-dynamics": "calibrate.py",
    "generate-adaptive-policies": "generate_adaptive_policies.py",
    "qualify-pre-d21": "qualify_pre_benchmark.py",
    "simulate-and-freeze": "physicell_engine.py",
    "evaluate-locked-benchmark": "evaluate_locked_benchmark.py",
    "report": "report.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=[*SCRIPTS, "all"])
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--external-summary", type=Path)
    parser.add_argument("--allow-prior-only", action="store_true")
    parser.add_argument("--download-external", action="store_true")
    parser.add_argument("--sample-id")
    parser.add_argument("--run-id")
    parser.add_argument("--prepare-project-only", action="store_true")
    parser.add_argument("--smoke-rule", action="store_true")
    parser.add_argument("--sensitivity-only", action="store_true")
    args = parser.parse_args()
    phases = list(SCRIPTS) if args.phase == "all" else [args.phase]
    for phase_index, phase in enumerate(phases, start=1):
        script = Path(__file__).resolve().parent / SCRIPTS[phase]
        cmd = [sys.executable, str(script), "--project-root", str(args.project_root), "--out-dir", str(args.out_dir)]
        if phase == "calibrate-dynamics":
            if args.external_summary:
                cmd += ["--external-summary", str(args.external_summary)]
            if args.allow_prior_only and args.phase != "all":
                cmd += ["--allow-prior-only"]
            if args.download_external:
                cmd += ["--download-external"]
        if phase == "simulate-and-freeze" and args.phase != "all":
            if args.run_id:
                cmd += ["--run-id", args.run_id]
            if args.prepare_project_only:
                cmd += ["--prepare-project-only"]
            if args.smoke_rule:
                cmd += ["--smoke-rule"]
            if args.sensitivity_only:
                cmd += ["--sensitivity-only"]
        if args.phase == "all":
            print(f"\n=== V3 phase {phase_index}/{len(phases)}: {phase} ===", flush=True)
        print("+ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode:
            print(f"\nV3 pipeline stopped at phase '{phase}' (exit {result.returncode}). Later phases were not run.", flush=True)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
