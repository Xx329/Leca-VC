#!/usr/bin/env python3
"""Rebuild ready paper figures from frozen release source data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/paper_figures.yaml"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def run_one(figure: str, record: dict) -> dict:
    if not str(record.get("status", "")).startswith("READY"):
        return {"figure": figure, "status": record.get("status", "SOURCE_GAP"), "reason": record.get("reason", "source not synchronized")}
    renderer = ROOT / record["renderer"]
    missing = [name for name in record.get("inputs", []) if not (ROOT / name).is_file()]
    if missing:
        return {"figure": figure, "status": "BLOCKED_MISSING_INPUT", "missing": missing}
    mplconfig = ROOT / "build/matplotlib"
    mplconfig.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(renderer)], cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, "MPLCONFIGDIR": str(mplconfig)},
    )
    if completed.returncode == 0:
        # Captions are transcribed from the final PDFs, independently of the
        # earlier plotting templates retained for scientific provenance.
        caption = renderer.parent / "caption.txt"
        if caption.is_file():
            target = ROOT / "build/figures" / figure
            target.mkdir(parents=True, exist_ok=True)
            text = caption.read_text(encoding="utf-8")
            for destination in {target / "caption.txt", *target.rglob("caption.txt")}:
                destination.write_text(text, encoding="utf-8")
    return {"figure": figure, "status": "PASS" if completed.returncode == 0 else "FAILED", "returncode": completed.returncode, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-4000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--figure")
    parser.add_argument("--allow-source-gaps", action="store_true", help="Return success after rendering all READY figures while reporting gaps.")
    args = parser.parse_args()
    figures = load_manifest()["figures"]
    selected = ([f"fig{i}" for i in range(1, 7)] +
                [f"figS{i}" for i in range(1, 9)]) if args.all else [args.figure]
    unknown = [name for name in selected if name not in figures]
    if unknown:
        parser.error(f"Unknown figure(s): {unknown}")
    results = [run_one(name, figures[name]) for name in selected]
    report = ROOT / "build/figure_reproduction_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    failures = [row for row in results if row["status"] == "FAILED" or row["status"].startswith("BLOCKED")]
    gaps = [row for row in results if "SOURCE_GAP" in row["status"]]
    if failures or (gaps and not args.allow_source_gaps):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
