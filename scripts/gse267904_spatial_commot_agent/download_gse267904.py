#!/usr/bin/env python3
"""Download and unpack GSE267904 RAW files.

Network is intentionally isolated in this script so the rest of the pipeline can
be run with --skip-download once data are present.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import RAW_URL, discover_10x_samples, dump_json, ensure_dirs, extract_tar, outpath


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--data-dir", type=Path, default=Path("data/GSE267904"))
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    a = p.parse_args()

    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    data_dir = outpath(root, a.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "GSE267904_RAW.tar"
    raw_dir = data_dir / "raw"

    actions = {"raw_url": RAW_URL, "tar_path": str(tar_path), "raw_dir": str(raw_dir)}
    if not a.skip_download and not tar_path.exists():
        subprocess.run(["wget", "-O", str(tar_path), RAW_URL], check=True)
        actions["downloaded"] = True
    else:
        actions["downloaded"] = False

    if not a.skip_extract and tar_path.exists() and not any(raw_dir.glob("**/*")):
        actions["extract"] = extract_tar(tar_path, raw_dir)
    else:
        actions["extract"] = {"skipped": True, "reason": "raw_dir already populated or tar missing"}

    samples = discover_10x_samples(raw_dir if raw_dir.exists() else data_dir)
    actions["discovered_samples"] = samples
    actions["n_discovered_samples"] = len(samples)
    dump_json(out / "audit/gse267904_download_audit.json", actions)
    print(f"Wrote download audit to {out/'audit/gse267904_download_audit.json'}")
    print(f"Discovered {len(samples)} 10x/Visium-like sample directories.")


if __name__ == "__main__":
    main()

