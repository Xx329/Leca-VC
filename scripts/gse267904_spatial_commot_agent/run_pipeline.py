#!/usr/bin/env python3
"""End-to-end runner for GSE267904 Spatial COMMOT + 50 Micro-Agent experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import dump_json, ensure_dirs, outpath


HERE = Path(__file__).resolve().parent


def call(root: Path, script: str, *args: object) -> None:
    cmd = [sys.executable, str(HERE / script), "--project-root", str(root), *map(str, args)]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--num-agents", type=int, default=50)
    p.add_argument("--max-samples-per-stage", type=int, default=1)
    p.add_argument("--max-commot-spots", type=int, default=800)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-real-commot", action="store_true")
    p.add_argument("--skip-virtual-commot", action="store_true")
    p.add_argument("--allow-debug-model-spatial-generator", action="store_true", help="Debug only; not real PhysiCell.")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    started = datetime.now().isoformat(timespec="seconds")

    if not a.skip_download:
        call(root, "download_gse267904.py", "--out-dir", out)

    call(root, "preflight_commot.py", "--out-dir", out)
    if a.preflight_only:
        return

    call(root, "build_real_spatial_benchmark.py", "--out-dir", out, "--max-samples-per-stage", a.max_samples_per_stage)
    call(root, "annotate_spots_by_markers.py", "--out-dir", out)

    if not a.skip_real_commot:
        call(root, "run_real_commot.py", "--out-dir", out, "--max-spots", a.max_commot_spots)

    call(root, "build_micro_agent_registry.py", "--out-dir", out, "--num-agents", a.num_agents)
    phys_args = ["--out-dir", out]
    if a.allow_debug_model_spatial_generator:
        phys_args.append("--allow-debug-model-spatial-generator")
    call(root, "run_physicell_micro_agent_simulation.py", *phys_args)
    call(root, "build_virtual_commot_input.py", "--out-dir", out)

    if not a.skip_virtual_commot:
        call(root, "run_virtual_commot.py", "--out-dir", out)
        call(root, "evaluate_real_vs_virtual_commot.py", "--out-dir", out)

    audit = {
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "REAL_SPATIAL_DATA_USED": True,
        "GSE267904_USED": True,
        "COMMOT_USED_FOR_REAL": not a.skip_real_commot,
        "COMMOT_USED_FOR_VIRTUAL": not a.skip_virtual_commot,
        "NUM_MICRO_AGENTS": a.num_agents,
        "SPOT_LEVEL_NOT_SINGLE_CELL_LEVEL": True,
        "CELL_TYPE_LABELS_INFERRED_BY_MARKERS": True,
        "REAL_D21_HELDOUT_USED_IN_AGENT_PROMPT": False,
        "OLD_VIRTUAL_CELLCHAT_USED": False,
        "debug_model_spatial_generator_allowed": a.allow_debug_model_spatial_generator,
    }
    dump_json(out / "audit/gse267904_spatial_commot_audit.json", audit)
    call(root, "write_report.py", "--out-dir", out)
    print(f"Wrote GSE267904 spatial COMMOT micro-agent experiment to {out}")


if __name__ == "__main__":
    main()
