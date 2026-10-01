#!/usr/bin/env python3
"""Audit repository hygiene and maintain deterministic repository-file hashes."""

from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HASH_MANIFEST=ROOT/"manifests/artifact_sha256.json"
EXCLUDED_PARTS={".git","build","vendor","data","outputs","runtime","__pycache__",".pytest_cache"}
EXCLUDED_FILES={HASH_MANIFEST.resolve()}
BANNED_SUFFIXES={".pyc",".pyo",".o",".so",".dylib",".exe"}
SECRET_PATTERNS=(
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\b(?:DEEPSEEK_API_KEY|OPENAI_API_KEY)\s*=\s*['\"]?(?!\$|enter-key|your-key|<)[A-Za-z0-9_-]{16,}",re.I),
    re.compile(r"Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]{16,}",re.I),
)

def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def files():
    for path in sorted(ROOT.rglob("*")):
        if path.is_symlink() or not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts) or path.resolve() in EXCLUDED_FILES: continue
        yield path

def scan() -> dict:
    absolute_paths=[]; secrets=[]; banned=[]; large=[]
    for path in files():
        rel=str(path.relative_to(ROOT))
        if path.suffix in BANNED_SUFFIXES or path.name.endswith("Zone.Identifier"): banned.append(rel)
        if path.stat().st_size>50*1024*1024: large.append({"path":rel,"bytes":path.stat().st_size})
        if path.stat().st_size<=10*1024*1024:
            try: text=path.read_text(encoding="utf-8")
            except (UnicodeDecodeError,OSError): continue
            if path.name not in {"audit_release.py", "build_artifact_bundles.py"} and "/home/" in text:
                absolute_paths.append(rel)
            if any(pattern.search(text) for pattern in SECRET_PATTERNS): secrets.append(rel)
    return {"absolute_private_paths":absolute_paths,"possible_secrets":secrets,"banned_files":banned,"files_over_50MiB":large}

def verify_recorded_hashes() -> list[str]:
    if not HASH_MANIFEST.is_file(): return ["missing manifests/artifact_sha256.json"]
    recorded=json.loads(HASH_MANIFEST.read_text(encoding="utf-8")).get("repository_files",{})
    if not recorded: return []
    mismatches=[]
    current={str(path.relative_to(ROOT)):path for path in files()}
    for name,record in recorded.items():
        path=current.get(name)
        if path is None or sha(path)!=record["sha256"] or path.stat().st_size!=record["size_bytes"]: mismatches.append(name)
    mismatches.extend(sorted(set(current)-set(recorded)))
    return sorted(set(mismatches))

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--write-hashes",action="store_true"); args=parser.parse_args()
    findings=scan()
    if args.write_hashes:
        manifest=json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
        manifest["repository_files"]={str(path.relative_to(ROOT)): {"sha256":sha(path),"size_bytes":path.stat().st_size} for path in files()}
        HASH_MANIFEST.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    else:
        findings["hash_mismatches"]=verify_recorded_hashes()
    findings["status"]="PASS" if not any(findings.values()) else "FAIL"
    print(json.dumps(findings,indent=2))
    return 0 if findings["status"]=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
