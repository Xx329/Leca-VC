#!/usr/bin/env python3
"""Build and configure the real GSE120575 PhysiCell/BioFVM executor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHYSICELL_ROOT = Path(
    os.environ.get("PHYSICELL_ROOT", str(ROOT / "vendor/PhysiCell"))
).expanduser().resolve()
PROJECT = ROOT / "scenarios/gse120575_full_online_celltype/physicell_project"
BINARY = PROJECT / "gse120575_celltype_online"
CELL_TYPES = (
    "B cell", "Plasma cell", "Monocyte/Macrophage",
    "Dendritic cell", "T cell", "NK cell",
)
SUBSTRATES = (
    ("treatment_signal", 1.0),
    ("inflammatory_signal", 0.05),
    ("suppressive_signal", 0.05),
    ("survival_signal", 0.50),
    ("stress_signal", 0.05),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_text(root: ET.Element, path: str, value: object) -> None:
    node = root.find(path)
    if node is None:
        raise RuntimeError(f"Missing PhysiCell XML node: {path}")
    node.text = str(value)


def compile_executor(output_root: Path) -> Path:
    result = subprocess.run(
        ["make", "-B", "gse120575_celltype_online"], cwd=PROJECT,
        capture_output=True, text=True,
    )
    log = output_root / "logs/physicell_build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Real PhysiCell build failed; inspect {log}")
    evidence = {
        "REAL_PHYSICELL_USED": True,
        "PHYSICELL_VERSION": (PHYSICELL_ROOT / "VERSION.txt").read_text().strip(),
        "binary": str(BINARY.resolve()),
        "binary_sha256": sha256(BINARY),
        "custom_cpp": str((PROJECT / "custom.cpp").resolve()),
        "custom_cpp_sha256": sha256(PROJECT / "custom.cpp"),
        "makefile_sha256": sha256(PROJECT / "Makefile"),
        "SYNTHETIC_SPATIAL_INITIALIZATION": True,
        "REAL_SPATIAL_RECONSTRUCTION_CLAIMED": False,
        "python_spatial_executor_used": False,
    }
    (output_root / "audit/physicell_build_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return BINARY


def runtime_config(
    run_dir: Path, registry_path: Path, sample_id: str, model: str,
    response: str, therapy: str, seed: int,
) -> Path:
    tree = ET.parse(PHYSICELL_ROOT / "config/PhysiCell_settings.xml")
    root = tree.getroot()
    for path, value in (
        ("./domain/x_min", -1000), ("./domain/x_max", 1000),
        ("./domain/y_min", -1000), ("./domain/y_max", 1000),
        ("./domain/z_min", -10), ("./domain/z_max", 10),
        ("./domain/dx", 100), ("./domain/dy", 100), ("./domain/dz", 20),
        ("./overall/max_time", 10), ("./overall/dt_diffusion", 0.2),
        ("./overall/dt_mechanics", 0.2), ("./overall/dt_phenotype", 0.2),
        ("./save/folder", str((run_dir / "physicell_raw").resolve())),
        ("./save/full_data/interval", 2), ("./save/SVG/enable", "false"),
        ("./parallel/omp_num_threads", 1), ("./options/random_seed", seed),
        ("./domain/use_2D", "true"),
    ):
        set_text(root, path, value)

    setup = root.find("./microenvironment_setup")
    original_variable = setup.find("./variable")
    for variable in list(setup.findall("./variable")):
        setup.remove(variable)
    for index, (name, initial) in enumerate(SUBSTRATES):
        variable = copy.deepcopy(original_variable)
        variable.attrib.update({"name": name, "ID": str(index), "units": "dimensionless"})
        variable.find("./physical_parameter_set/diffusion_coefficient").text = "1000"
        variable.find("./physical_parameter_set/decay_rate").text = "0.01"
        variable.find("./initial_condition").text = str(initial)
        boundary = variable.find("./Dirichlet_boundary_condition")
        boundary.text = str(initial); boundary.attrib["enabled"] = "false"
        setup.insert(index, variable)

    definitions = root.find("./cell_definitions")
    original = definitions.find("./cell_definition")
    for definition in list(definitions):
        definitions.remove(definition)
    default = copy.deepcopy(original); default.attrib.update({"name": "default", "ID": "0"})
    definitions.append(default)
    for index, cell_type in enumerate(CELL_TYPES, 1):
        definition = copy.deepcopy(original)
        definition.attrib.update({"name": cell_type, "ID": str(index)})
        cycle = definition.find("./phenotype/cycle")
        cycle.clear(); cycle.attrib.update({"code": "5", "name": "live"})
        rates = ET.SubElement(cycle, "phase_transition_rates", {"units": "1/min"})
        ET.SubElement(rates, "rate", {"start_index": "0", "end_index": "0", "fixed_duration": "false"}).text = "0.001"
        secretion = definition.find("./phenotype/secretion")
        source_substrate = secretion.find("./substrate")
        for substrate in list(secretion): secretion.remove(substrate)
        for substrate_index, (name, _) in enumerate(SUBSTRATES):
            substrate = copy.deepcopy(source_substrate)
            substrate.attrib.update({"name": name, "ID": str(substrate_index)})
            for tag, value in (
                ("secretion_rate", 0), ("secretion_target", 1),
                ("uptake_rate", 0), ("net_export_rate", 0),
            ):
                node = substrate.find("./" + tag)
                if node is not None: node.text = str(value)
            secretion.append(substrate)
        motility = definition.find("./phenotype/motility")
        motility.find("./speed").text = "0.35"
        motility.find("./options/enabled").text = "true"
        definitions.append(definition)

    # PhysiCell validates substrate names even for disabled chemotaxis and for
    # the unused default definition, so every definition must be rebound.
    for definition in definitions.findall("./cell_definition"):
        motility = definition.find("./phenotype/motility")
        chemotaxis = motility.find("./options/chemotaxis")
        if chemotaxis is not None:
            chemotaxis.find("./enabled").text = "false"
            chemotaxis.find("./substrate").text = "treatment_signal"
        for sensitivity in motility.findall("./options/advanced_chemotaxis/chemotactic_sensitivities/chemotactic_sensitivity"):
            sensitivity.attrib["substrate"] = "treatment_signal"
        if definition.attrib.get("name") == "default":
            secretion = definition.find("./phenotype/secretion")
            source_substrate = secretion.find("./substrate")
            for substrate in list(secretion):
                secretion.remove(substrate)
            for substrate_index, (name, _) in enumerate(SUBSTRATES):
                substrate = copy.deepcopy(source_substrate)
                substrate.attrib.update({"name": name, "ID": str(substrate_index)})
                secretion.append(substrate)

    parameters = root.find("./user_parameters")
    if parameters is None:
        parameters = ET.SubElement(root, "user_parameters")
    parameters.clear()
    values = {
        "run_dir": str(run_dir.resolve()), "registry_path": str(registry_path.resolve()),
        "bridge_path": str((ROOT / "scripts/gse120575_full_online_celltype/runtime_bridge.py").resolve()),
        "sample_id": sample_id, "model_name": model, "response": response, "therapy": therapy,
    }
    for name, value in values.items():
        node = ET.SubElement(parameters, name, {"type": "string", "units": "dimensionless"})
        node.text = str(value)
    config = run_dir / "PhysiCell_settings.xml"
    run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "physicell_raw").mkdir(exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(config, encoding="utf-8", xml_declaration=True)
    return config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    print(compile_executor(a.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
