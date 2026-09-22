#!/usr/bin/env python3
"""Download and verify release artifacts after Zenodo draft URLs are frozen."""

from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"manifests/artifact_sha256.json"

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("artifact", choices=("processed_inputs","runtime_audit","reference_results","all")); parser.add_argument("--download-dir",type=Path,default=ROOT/"artifacts/downloads"); args=parser.parse_args()
    records=json.loads(MANIFEST.read_text(encoding="utf-8"))["external_artifacts"]
    selected=list(records) if args.artifact=="all" else [args.artifact]
    args.download_dir.mkdir(parents=True,exist_ok=True)
    for name in selected:
        record=records[name]
        if str(record.get("url","")).startswith("TBD") or str(record.get("sha256","")).startswith("TBD"):
            raise RuntimeError(f"{name}: Zenodo draft URL/SHA256 has not been frozen")
        target=args.download_dir/record["filename"]
        urllib.request.urlretrieve(record["url"],target)
        actual=sha256(target)
        if actual!=record["sha256"]: raise RuntimeError(f"{name}: SHA256 mismatch")
        print(f"verified {target.name}: {actual}")
    return 0

if __name__=="__main__": raise SystemExit(main())

