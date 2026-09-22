#!/usr/bin/env python3
"""Offline verification or guarded online execution of the contamination probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def offline() -> int:
    source = ROOT / "source_data/figS7/deidentified_exact_runtime_call_results.csv"
    calls = pd.read_csv(source)
    if len(calls) != 30:
        raise RuntimeError(f"Expected 30 probe calls, found {len(calls)}")
    result = {
        "calls": len(calls),
        "correct": int(calls.heldout_answer_correct.sum()),
        "unknown": int(calls.abstained_unknown.sum()),
    }
    result["incorrect"] = result["calls"] - result["correct"] - result["unknown"]
    result["accuracy"] = result["correct"] / result["calls"]
    expected = {"calls": 30, "correct": 1, "incorrect": 6, "unknown": 23, "accuracy": 1 / 30}
    if result != expected:
        raise RuntimeError(f"Frozen probe result changed: {result}")
    mplconfig = ROOT / "build/matplotlib"
    mplconfig.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "figures/supplementary/figS7/render.py")],
        cwd=ROOT, check=True, env={**os.environ, "MPLCONFIGDIR": str(mplconfig)},
    )
    print(json.dumps(result, indent=2))
    return 0


def online() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("DEEPSEEK_API_KEY is not set; online probe stopped with no fallback")
    contexts = ROOT / "outputs/GSE2565_deidentified_rerun_v1"
    if not contexts.is_dir():
        raise RuntimeError("Exact production audit contexts are absent. Install the verified runtime-audit artifact first; held-out answers must remain outside model messages.")
    runner = ROOT / "scripts/leca_vc_prompt_isolation_v1/run_exact_runtime_probe_v2.py"
    if not runner.is_file():
        raise RuntimeError("Exact online probe runner has not yet been synchronized into the private RC")
    return subprocess.run([sys.executable, str(runner)], cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--online", action="store_true")
    args = parser.parse_args()
    return offline() if args.offline else online()


if __name__ == "__main__": raise SystemExit(main())
