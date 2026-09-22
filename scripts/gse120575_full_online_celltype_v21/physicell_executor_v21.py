#!/usr/bin/env python3
"""Compile and configure the independent real PhysiCell V2.1 executor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "scenarios/gse120575_full_online_celltype_v21/physicell_project"
BINARY = PROJECT / "gse120575_celltype_v21_online"
V1_EXECUTOR = ROOT / "scripts/gse120575_full_online_celltype/physicell_executor.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_executor(output_root: Path) -> Path:
    result = subprocess.run(
        ["make", "-B", "gse120575_celltype_v21_online"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )
    log = output_root / "logs/physicell_v21_build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"V2.1 PhysiCell build failed; inspect {log}")
    evidence = {
        "REAL_PHYSICELL_USED": True,
        "POPULATION_ACTUATOR_VERSION": "V2.1",
        "binary": str(BINARY.resolve()),
        "binary_sha256": sha256(BINARY),
        "custom_cpp_sha256": sha256(PROJECT / "custom.cpp"),
        "normalized_internal_simulation_time": True,
        "clinical_minutes_claimed": False,
    }
    (output_root / "audit").mkdir(parents=True, exist_ok=True)
    (output_root / "audit/physicell_v21_build_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    return BINARY


def runtime_config(
    run_dir: Path,
    registry: Path,
    sample_id: str,
    model: str,
    response: str,
    therapy: str,
    seed: int,
) -> Path:
    spec = importlib.util.spec_from_file_location("v1_physicell_executor", V1_EXECUTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = module.runtime_config(run_dir, registry, sample_id, model, response, therapy, seed)
    tree = ET.parse(path)
    root = tree.getroot()
    root.find("./user_parameters/bridge_path").text = str(
        (ROOT / "scripts/gse120575_full_online_celltype_v21/online_inference_v21.py").resolve()
    )
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
