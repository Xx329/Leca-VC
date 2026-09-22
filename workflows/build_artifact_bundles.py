#!/usr/bin/env python3
"""Build staged Zenodo bundles after privacy checks; never uploads them."""

from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPECS={
    "processed_inputs": (ROOT/"artifacts/processed_inputs", "lecavc_processed_inputs_v1.tar.zst"),
    "runtime_audit": (ROOT/"artifacts/runtime_audit", "lecavc_runtime_audit_v1.tar.zst"),
    "reference_results": (ROOT/"artifacts/reference_results", "lecavc_reference_results_v1.tar.zst"),
}
PATTERNS=(re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}"),re.compile(rb"Authorization\s*:\s*Bearer\s+\S+",re.I),re.compile(rb"/home/[^/\s]+/"))

def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def scan(stage: Path) -> None:
    if not stage.is_dir(): raise RuntimeError(f"staging directory absent: {stage}")
    for path in stage.rglob("*"):
        if path.is_file() and path.stat().st_size<20*1024*1024:
            data=path.read_bytes()
            if any(pattern.search(data) for pattern in PATTERNS): raise RuntimeError(f"privacy scan failed: {path.relative_to(stage)}")

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("artifact",choices=(*SPECS,"all")); args=parser.parse_args()
    selected=list(SPECS) if args.artifact=="all" else [args.artifact]
    output=ROOT/"artifacts/bundles"; output.mkdir(parents=True,exist_ok=True)
    records={}
    for name in selected:
        stage,filename=SPECS[name]; scan(stage); target=output/filename
        subprocess.run(["tar","--zstd","-cf",str(target),"-C",str(stage),"."],check=True)
        records[name]={"path":str(target.relative_to(ROOT)),"sha256":sha(target),"size_bytes":target.stat().st_size}
    (output/"bundle_build.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(records,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())

