#!/usr/bin/env python3
"""Preflight for GSE267904 + COMMOT spatial communication experiment."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import traceback
from pathlib import Path

from common import discover_10x_samples, dump_json, ensure_dirs, find_raw_root, outpath


def try_import(name: str) -> dict:
    try:
        mod = importlib.import_module(name)
        return {"available": True, "version": getattr(mod, "__version__", "unknown"), "error": None}
    except Exception as e:
        return {"available": False, "version": None, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--require-data", action="store_true")
    a = p.parse_args()

    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    raw_root = find_raw_root(root, out)
    samples = discover_10x_samples(raw_root)
    imports = {name: try_import(name) for name in ["anndata", "scanpy", "commot", "squidpy", "numpy", "scipy", "sklearn"]}
    rscript = shutil.which("Rscript")
    physicell_root = Path(
        os.environ.get("PHYSICELL_ROOT", str(root / "vendor/PhysiCell"))
    ).expanduser().resolve()
    result = {
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "n_discovered_spatial_samples": len(samples),
        "discovered_samples": samples[:20],
        "imports": imports,
        "Rscript_available": bool(rscript),
        "Rscript_path": rscript,
        "physicell_root_exists": physicell_root.exists(),
        "physicell_makefile_exists": (physicell_root / "Makefile").exists(),
        "commot_available": imports["commot"]["available"],
        "scanpy_available": imports["scanpy"]["available"],
        "anndata_available": imports["anndata"]["available"],
        "squidpy_available": imports["squidpy"]["available"],
    }
    blocking = []
    if not imports["commot"]["available"]:
        blocking.append("COMMOT is not importable. Install commot in the active Python environment.")
    if not imports["scanpy"]["available"]:
        blocking.append("scanpy is not importable or currently errors during import.")
    if not imports["anndata"]["available"]:
        blocking.append("anndata is not importable.")
    if a.require_data and not samples:
        blocking.append("No GSE267904 10x/Visium-like raw samples discovered.")
    if not physicell_root.exists():
        blocking.append("PhysiCell checkout not found; set PHYSICELL_ROOT.")
    result["can_run_real_commot"] = imports["commot"]["available"] and imports["scanpy"]["available"] and imports["anndata"]["available"] and (len(samples) > 0 or not a.require_data)
    result["can_run_full_spatial_mvp"] = result["can_run_real_commot"] and physicell_root.exists()
    result["blocking_issues"] = blocking
    dump_json(out / "audit/commot_preflight.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if blocking:
        print("\nBlocking issues:")
        for b in blocking:
            print(f"- {b}")


if __name__ == "__main__":
    main()
