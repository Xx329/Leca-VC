#!/usr/bin/env python3
"""Audit, freeze, generate, run, and parse the real PhysiCell worker layer."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial import cKDTree

from common import (
    AFFINE_AUDIT,
    BASE_PHYSICELL_CONFIG,
    CELL_AGENT_MAP,
    COMMOT_CONFIG,
    D7,
    D7_SCALEFACTORS,
    DIRS,
    FAMILIES,
    K100,
    LATE_AGENT,
    MAX_TIME,
    N_PER_TYPE,
    OUT,
    PAIR_UNIVERSE,
    PANEL,
    PARENT_PROGRAM,
    ROOT,
    SAMPLE_SEEDS,
    SCALED_REGISTRY,
    SCENARIO_DIR,
    SCGPT,
    SCGPT_VALIDATION,
    SCRIPT_DIR,
    SPOT_DIAMETER_FULLRES,
    SPOT_RADIUS_FULLRES,
    TYPES,
    WORKER_SEED_FOR_SAMPLE,
    WORKER_SEEDS,
    affine,
    ensure_dirs,
    frozen_pairs_and_genes,
    model_to_pixel,
    pixel_to_model,
    require_files,
    sha256_file,
    sha256_text,
    spatial_sample_indices,
    write_json,
)


ALLOW_DYNAMIC_PARENT_COUNT = os.environ.get("GSE267904_DYNAMIC_PARENT_COUNT") == "1"


REQUIRED_INPUTS = [
    D7,
    LATE_AGENT,
    PARENT_PROGRAM,
    CELL_AGENT_MAP,
    SCALED_REGISTRY,
    AFFINE_AUDIT,
    SCGPT,
    SCGPT_VALIDATION,
    PANEL,
    PAIR_UNIVERSE,
    D7_SCALEFACTORS,
    BASE_PHYSICELL_CONFIG,
]

PROGRAM_COLUMNS = {
    "damage_secretion": "damage",
    "inflammatory_secretion": "inflammatory",
    "fibrosis_secretion": "fibrosis",
    "resolution_secretion": "resolution",
    "TGFb_secretion": "tgfb",
    "SPP1_secretion": "spp1",
    "chemokine_secretion": "chemokine",
    "motility": "motility",
    "death_multiplier": "death_multiplier",
    "proliferation_multiplier": "proliferation_multiplier",
}


def gate0_audit() -> dict:
    ensure_dirs()
    require_files(REQUIRED_INPUTS)
    late = ad.read_h5ad(LATE_AGENT)
    cell_map = pd.read_csv(CELL_AGENT_MAP)
    registry = pd.read_csv(SCALED_REGISTRY)
    programs = pd.read_csv(PARENT_PROGRAM)
    late.obs["frozen_type"] = late.obs["cell_type"].astype(str)
    primary = late.obs[late.obs["frozen_type"].isin(TYPES)].copy()
    all_xy = np.asarray(late.obsm["spatial"], dtype=float)
    primary_xy = all_xy[late.obs["frozen_type"].isin(TYPES).to_numpy()]
    live_ids = set(late.obs["agent_id"].astype(str))
    primary_ids = set(primary["agent_id"].astype(str))
    program_ids = set(programs["agent_id"].astype(str))

    inventory_paths = [
        SCALED_REGISTRY,
        CELL_AGENT_MAP,
        K100 / "physicell/virtual_cell_positions_by_stage.csv",
        K100 / "physicell/output00000002.xml",
        K100 / "physicell/output00000002_cells.mat",
        LATE_AGENT,
        K100 / "micro_agents/micro_agent_expression_prototypes.csv",
        PARENT_PROGRAM,
    ]
    inventory = []
    for path in inventory_paths:
        inventory.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
                "role": (
                    "historical PhysiCell/Agent endpoint input; read-only"
                    if path.exists()
                    else "missing"
                ),
            }
        )
    pd.DataFrame(inventory).to_csv(
        OUT / "01_worker_audit/worker_file_inventory.csv", index=False
    )

    findings = {
        "status": "GATE0_REQUIRES_NEW_WORKER_LAYER",
        "historical_late_entities": int(late.n_obs),
        "historical_late_primary_entities": int(len(primary)),
        "historical_late_other_entities": int((late.obs["frozen_type"] == "other").sum()),
        "historical_late_unique_coordinates_all": int(len(np.unique(all_xy, axis=0))),
        "historical_late_unique_coordinates_primary": int(
            len(np.unique(primary_xy, axis=0))
        ),
        "cell_agent_map_rows": int(len(cell_map)),
        "cell_agent_map_one_to_one": bool(
            cell_map["cell_id"].nunique() == len(cell_map)
            and cell_map["agent_id"].nunique() == len(cell_map)
        ),
        "scaled_registry_rows": int(len(registry)),
        "historical_parent_agent_id_field": False,
        "historical_represented_abundance_field": False,
        "historical_worker_expression_field": False,
        "primary_parent_programs_covered_at_interval_1": int(
            len(primary_ids & program_ids)
        ),
        "primary_parent_programs_missing": sorted(primary_ids - program_ids),
        "all_live_programs_covered": int(len(live_ids & program_ids)),
        "interpretation": (
            "The K=100 endpoint is a one-Agent-to-one-PhysiCell-cell historical "
            "micro-agent run, not an existing one-to-many worker population. "
            "The formal benchmark must create a new, explicitly labeled worker layer."
        ),
        "other_policy": (
            "Historical Other entities are invalid as formal workers because 51 late "
            "Other entities include fill artifacts and coincident geometry."
        ),
    }
    if findings["primary_parent_programs_missing"]:
        findings["status"] = "BLOCKED_MISSING_PRIMARY_PARENT_PROGRAM"
    write_json(OUT / "01_worker_audit/worker_level_availability_audit.json", findings)
    md = f"""# Worker-level availability audit

Status: `{findings['status']}`

- Historical late endpoint: {late.n_obs} entities; {len(primary)} primary and {findings['historical_late_other_entities']} Other.
- Unique endpoint coordinates: {findings['historical_late_unique_coordinates_all']} overall and {findings['historical_late_unique_coordinates_primary']} in the primary nine types.
- Historical mapping is one Agent to one PhysiCell cell: `{findings['cell_agent_map_one_to_one']}`.
- `parent_agent_id`, `represented_abundance`, and native worker transcriptomes are absent.
- All {len(primary)} surviving primary parent Agents have a frozen interval-1 program: `{not findings['primary_parent_programs_missing']}`.

The old endpoint is retained read-only. The new benchmark will build a real
PhysiCell worker layer from d7 in-tissue spatial anchors and frozen late parent
programs. This is a prospective worker-layer benchmark after historical
Agent-level results, not a reinterpretation of the old 83 entities.
"""
    (OUT / "01_worker_audit/worker_level_availability_audit.md").write_text(
        md, encoding="utf-8"
    )
    return findings


def protocol_object() -> dict:
    pairs, genes = frozen_pairs_and_genes()
    with gzip.open(D7_SCALEFACTORS, "rt") as f:
        scalefactors = json.load(f)
    validation = json.loads(SCGPT_VALIDATION.read_text())
    if validation.get("status") != "PASS_EXTERNAL_VALIDATION_BEATS_PERSISTENCE":
        raise RuntimeError("External scGPT validation gate is not PASS")
    if validation.get("gse267904_d21_read") is not False:
        raise RuntimeError("scGPT training/prediction audit does not prove d21 exclusion")
    center, scale = affine()
    input_hashes = {str(p): sha256_file(p) for p in REQUIRED_INPUTS}
    code_files = sorted(
        list(SCRIPT_DIR.glob("*.py"))
        + list(SCRIPT_DIR.glob("*.sh"))
        + list(SCENARIO_DIR.glob("*.cpp"))
        + list(SCENARIO_DIR.glob("*.h"))
        + list(SCENARIO_DIR.glob("Makefile"))
    )
    late = ad.read_h5ad(LATE_AGENT)
    primary_parent_count = int(
        late.obs["cell_type"].astype(str).isin(TYPES).sum()
    )
    if not ALLOW_DYNAMIC_PARENT_COUNT and primary_parent_count != 32:
        raise RuntimeError(
            f"Expected 32 historical primary parent Agents, got {primary_parent_count}"
        )
    return {
        "experiment_id": "GSE267904_agentvc_worker_scgpt_cci_main_v1",
        "protocol_semantics": (
            "prospectively frozen worker benchmark after historical Agent-level "
            "analyses; not a de novo first preregistration"
        ),
        "created_from_historical_results": True,
        "primary_cell_types": TYPES,
        "other_policy": (
            "excluded from primary analysis; AgentVC Other sensitivity is "
            "NOT_ESTIMABLE_INVALID_HISTORICAL_OTHER"
        ),
        "parent_agents": {
            "scope": f"{primary_parent_count} surviving primary late Agents",
            "count": primary_parent_count,
            "count_source": "current frozen K=100 late endpoint",
            "program_source": str(PARENT_PROGRAM),
            "llm_calls": 0,
        },
        "worker_layer": {
            "initialization_source": "GSE267904 d7 bleomycin in_tissue=1 only",
            "expected_initial_primary_workers": 3410,
            "parent_assignment": "within-cell-type nearest frozen late parent in pixel-like coordinates",
            "represented_weight": (
                "one over 3410 per initial worker; conserved through division; "
                "live per-parent secretion re-normalized to frozen parent mass"
            ),
            "simulation_minutes": MAX_TIME,
            "dt_diffusion": 1.0,
            "dt_mechanics": 1.0,
            "dt_phenotype": 1.0,
            "seeds": WORKER_SEEDS,
            "python_spatial_fallback_allowed": False,
        },
        "capacity": {
            "entities_per_type": N_PER_TYPE,
            "primary_types": len(TYPES),
            "entities_per_method": N_PER_TYPE * len(TYPES),
            "sampling_seeds": SAMPLE_SEEDS,
            "worker_seed_for_sampling_seed": WORKER_SEED_FOR_SAMPLE,
            "sampling": "cell-type and 6x6 spatial-grid stratified without replacement",
        },
        "geometry": {
            "primary": "frozen affine pixel-like scale",
            "center_xy": center.tolist(),
            "scale": scale,
            "worker_forward": "model_xy=(pixel_xy-center_xy)*scale without clipping",
            "worker_inverse": "pixel_xy=model_xy/scale+center_xy",
            "dis_thr": 500.0,
            "sensitivity": "median nearest-neighbor normalized to 100",
            "entity_self_distance": 501.0,
        },
        "pseudo_spot": {
            "spot_diameter_fullres": SPOT_DIAMETER_FULLRES,
            "spot_radius_fullres": SPOT_RADIUS_FULLRES,
            "scalefactors_file_value": float(scalefactors["spot_diameter_fullres"]),
            "aggregation": "sum worker parent-expression pseudo-counts within radius, then CP10K+log1p",
        },
        "expression": {
            "panel_genes": len(genes),
            "normalization": "log1p(CP10K) over frozen 1200-gene common panel",
            "observed_input": "raw count",
            "scgpt_input": (
                "nonnegative pseudo-count from frozen scGPT encoder plus externally "
                "trained conditional expression head"
            ),
            "agentvc_input": (
                "expm1 of frozen late parent expression proxy; no new LR boost/no noise"
            ),
            "worker_name": "AgentVC worker-expanded parent-expression proxy",
        },
        "lr_panel": {
            "families": FAMILIES,
            "exact_pairs": int(len(pairs)),
            "pair_counts": pairs["pathway"].value_counts().to_dict(),
        },
        "commot": COMMOT_CONFIG,
        "metrics": {
            "primary": [
                "81-edge Spearman",
                "informative-union-positive-edge Spearman",
                "deterministic positive Top-10 Jaccard",
                "five-pathway flow JS similarity",
                "sender/receiver role mean Spearman",
                "pathway allocation absolute error",
                "cell-type local error",
                "same-type share",
                "cross-type positive-edge coverage",
            ],
            "secondary": {
                "GNRS": "mean((edge_spearman+1)/2, positive_top10_jaccard)",
                "LBSS": "mean(five_flow_js_similarity, (role_mean_spearman+1)/2)",
            },
            "overall_score": None,
            "inference": "sampling/simulation stability only; no biological CI or p-value",
        },
        "hard_gates": {
            "minimum_live_workers_per_type": 90,
            "minimum_nonempty_pseudospots_per_type": 90,
            "minimum_candidate_type_pairs": 75,
            "max_agentvc_observed_candidate_coverage_difference_percentage_points": 10,
            "max_zero_neighbor_fraction": 0.05,
            "all_sampling_combinations": 10,
            "all_worker_realizations": 3,
            "all_pairs_together_per_commot_call": True,
        },
        "input_hashes": input_hashes,
        "code_hashes": {str(p): sha256_file(p) for p in code_files},
    }


def freeze_protocol() -> dict:
    ensure_dirs()
    gate = gate0_audit()
    if gate["status"].startswith("BLOCKED"):
        raise RuntimeError(gate["status"])
    protocol = protocol_object()
    text = json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path = OUT / "00_protocol/frozen_worker_cci_protocol_v1.yaml"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(
            "Frozen protocol already exists with different content; create a new version"
        )
    path.write_text(text, encoding="utf-8")
    protocol_hash = sha256_text(text)
    leakage = """# Data leakage rules

- Worker audit, parent assignment, worker generation, worker expression and geometry
  may read d7, frozen K=100 late outputs and frozen external-scGPT artifacts.
- They must not open GSE267904 d21.
- The evaluation process is the only stage allowed to open d21.
- No LLM call, scGPT retraining, LR boost, target-driven coordinate change or
  result-driven distance change is allowed.
- Historical Agent-level results were already observed; this is disclosed and the
  present protocol is prospective only for the new worker benchmark.
"""
    metrics = """# Metric prospective specification

Raw components are primary. GNRS and LBSS are secondary descriptive indices and
are never combined. Edge weights are normalized within each 9x9 network.
Positive Top-10 sets use descending weight and sender|receiver lexical tie-breaks;
if fewer than ten positive edges exist, all positive edges are used and effective
k is recorded. Informative Spearman uses the union of edges positive in either
Observed or the method. Ranges are sampling/simulation stability ranges only.
"""
    (OUT / "00_protocol/data_leakage_rules.md").write_text(leakage, encoding="utf-8")
    (OUT / "00_protocol/metric_preregistration.md").write_text(
        metrics, encoding="utf-8"
    )
    frozen = {
        "status": "PASS_PROTOCOL_FROZEN",
        "protocol": str(path),
        "protocol_sha256": protocol_hash,
        "input_hashes": protocol["input_hashes"],
        "code_hashes": protocol["code_hashes"],
    }
    write_json(OUT / "00_protocol/frozen_protocol_hash.json", frozen)
    return frozen


def build_worker_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol_path = OUT / "00_protocol/frozen_worker_cci_protocol_v1.yaml"
    if not protocol_path.exists():
        raise RuntimeError("Protocol must be frozen before worker generation")
    d7 = ad.read_h5ad(D7)
    if "in_tissue" not in d7.obs or "dominant_cell_type" not in d7.obs:
        raise RuntimeError("d7 input lacks tissue flag or frozen cell-type labels")
    mask = (
        d7.obs["in_tissue"].astype(int).eq(1)
        & d7.obs["dominant_cell_type"].astype(str).isin(TYPES)
    )
    d7p = d7[mask].copy()
    if d7p.n_obs != 3410:
        raise RuntimeError(f"Expected 3410 d7 in-tissue primary anchors, got {d7p.n_obs}")

    late = ad.read_h5ad(LATE_AGENT)
    late.obs["cell_type"] = late.obs["cell_type"].astype(str)
    latep = late[late.obs["cell_type"].isin(TYPES)].copy()
    if not ALLOW_DYNAMIC_PARENT_COUNT and latep.n_obs != 32:
        raise RuntimeError(f"Expected 32 surviving primary parents, got {latep.n_obs}")
    parent_xy_pixel = model_to_pixel(np.asarray(latep.obsm["spatial"], dtype=float))
    parents = pd.DataFrame(
        {
            "parent_agent_id": latep.obs["agent_id"].astype(str).to_numpy(),
            "cell_type": latep.obs["cell_type"].astype(str).to_numpy(),
            "parent_x_pixel": parent_xy_pixel[:, 0],
            "parent_y_pixel": parent_xy_pixel[:, 1],
        }
    ).sort_values("parent_agent_id", kind="mergesort")
    parents["parent_index"] = np.arange(len(parents), dtype=int)

    coords_pixel = np.asarray(d7p.obsm["spatial"], dtype=float)
    labels = d7p.obs["dominant_cell_type"].astype(str).to_numpy()
    parent_idx = np.full(d7p.n_obs, -1, dtype=int)
    for cell_type in TYPES:
        worker_rows = np.where(labels == cell_type)[0]
        p = parents[parents["cell_type"].eq(cell_type)]
        if p.empty:
            raise RuntimeError(f"No live late parent for {cell_type}")
        pxy = p[["parent_x_pixel", "parent_y_pixel"]].to_numpy(float)
        delta = coords_pixel[worker_rows, None, :] - pxy[None, :, :]
        nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        parent_idx[worker_rows] = p["parent_index"].to_numpy(int)[nearest]
    if (parent_idx < 0).any():
        raise RuntimeError("At least one d7 worker anchor lacks a parent")

    model_xy = pixel_to_model(coords_pixel)
    workers = pd.DataFrame(
        {
            "worker_id": [f"worker_init_{i:06d}" for i in range(d7p.n_obs)],
            "initial_worker_index": np.arange(d7p.n_obs, dtype=int),
            "parent_index": parent_idx,
            "source_spot_id": d7p.obs_names.astype(str),
            "cell_type": labels,
            "x_model": model_xy[:, 0],
            "y_model": model_xy[:, 1],
            "represented_weight": np.full(d7p.n_obs, 1.0 / d7p.n_obs),
            "x_pixel": coords_pixel[:, 0],
            "y_pixel": coords_pixel[:, 1],
        }
    )
    workers = workers.merge(
        parents[["parent_index", "parent_agent_id"]], on="parent_index", how="left"
    )
    workers = workers[
        [
            "worker_id",
            "initial_worker_index",
            "parent_index",
            "parent_agent_id",
            "cell_type",
            "x_model",
            "y_model",
            "represented_weight",
            "source_spot_id",
            "x_pixel",
            "y_pixel",
        ]
    ]
    if workers["source_spot_id"].duplicated().any():
        raise RuntimeError("Worker initialization duplicates a d7 spatial anchor")

    allocation = (
        workers.groupby(["cell_type", "parent_index", "parent_agent_id"], as_index=False)
        .agg(
            allocated_initial_workers=("worker_id", "size"),
            parent_mass=("represented_weight", "sum"),
        )
        .sort_values(["cell_type", "parent_agent_id"], kind="mergesort")
    )
    if not np.isclose(allocation["parent_mass"].sum(), 1.0, atol=1e-12):
        raise RuntimeError("Represented worker mass is not globally normalized")

    historical = pd.read_csv(PARENT_PROGRAM)
    historical = historical[historical["agent_id"].astype(str).isin(parents["parent_agent_id"])]
    if historical["agent_id"].nunique() != len(parents):
        raise RuntimeError(
            "Interval-1 program does not cover all current frozen primary parents"
        )
    programs = parents.merge(
        historical[
            ["agent_id", "cell_type", *PROGRAM_COLUMNS.keys()]
        ].rename(columns={"agent_id": "parent_agent_id"}),
        on=["parent_agent_id", "cell_type"],
        how="left",
        validate="one_to_one",
    ).merge(
        allocation[["parent_index", "parent_mass"]],
        on="parent_index",
        how="left",
        validate="one_to_one",
    )
    programs = programs.rename(columns=PROGRAM_COLUMNS)
    programs["program_source_sha256"] = sha256_file(PARENT_PROGRAM)
    programs = programs[
        [
            "parent_index",
            "parent_agent_id",
            "cell_type",
            "damage",
            "inflammatory",
            "fibrosis",
            "resolution",
            "tgfb",
            "spp1",
            "chemokine",
            "motility",
            "death_multiplier",
            "proliferation_multiplier",
            "parent_mass",
            "program_source_sha256",
        ]
    ]
    numeric = programs.iloc[:, 3:14].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise RuntimeError("Parent program table contains missing/non-numeric values")

    base = OUT / "02_worker_generation"
    workers.to_csv(base / "worker_initialization_registry.csv", index=False)
    programs.to_csv(base / "frozen_parent_programs.csv", index=False)
    allocation.to_csv(base / "worker_count_allocation.csv", index=False)
    centers = workers[
        ["source_spot_id", "cell_type", "x_pixel", "y_pixel", "parent_agent_id"]
    ].copy()
    centers.to_csv(base / "d7_in_tissue_primary_capture_centers.csv", index=False)
    audit = {
        "status": "PASS_WORKER_REGISTRY_PREPARED",
        "d7_read": True,
        "d21_read": False,
        "initial_workers": int(len(workers)),
        "parent_agents": int(programs["parent_agent_id"].nunique()),
        "cell_type_counts": workers["cell_type"].value_counts().to_dict(),
        "worker_coordinate_unique": int(
            len(np.unique(workers[["x_model", "y_model"]].to_numpy(), axis=0))
        ),
        "represented_weight_sum": float(workers["represented_weight"].sum()),
        "parent_mass_sum": float(programs["parent_mass"].sum()),
        "llm_calls": 0,
        "worker_registry_sha256": sha256_file(
            base / "worker_initialization_registry.csv"
        ),
        "parent_programs_sha256": sha256_file(base / "frozen_parent_programs.csv"),
    }
    write_json(base / "worker_registry_audit.json", audit)
    return workers, programs


def make_smoke_registry(workers: pd.DataFrame) -> Path:
    idx, _ = spatial_sample_indices(
        workers["cell_type"].to_numpy(str),
        workers[["x_pixel", "y_pixel"]].to_numpy(float),
        workers["worker_id"].to_numpy(str),
        1701,
        n_per_type=10,
    )
    smoke = workers.iloc[idx].copy().reset_index(drop=True)
    smoke["initial_worker_index"] = np.arange(len(smoke), dtype=int)
    # Preserve global mass normalization in the technical smoke.
    smoke["represented_weight"] = 1.0 / len(smoke)
    path = OUT / "02_worker_generation/smoke_worker_initialization_registry.csv"
    smoke.to_csv(path, index=False)
    return path


def build_config(run_dir: Path, seed: int, smoke: bool) -> Path:
    tree = ET.parse(BASE_PHYSICELL_CONFIG)
    root = tree.getroot()
    root.find("./overall/max_time").text = str(MAX_TIME)
    root.find("./overall/dt_diffusion").text = "1"
    root.find("./overall/dt_mechanics").text = "1"
    root.find("./overall/dt_phenotype").text = "1"
    root.find("./parallel/omp_num_threads").text = "1"
    root.find("./save/folder").text = str(run_dir)
    root.find("./save/full_data/interval").text = str(MAX_TIME)
    root.find("./save/full_data/enable").text = "true"
    root.find("./save/SVG/enable").text = "false"
    root.find("./options/random_seed").text = str(seed)
    # No old initial conditions or rule files may inject historical cells.
    cell_positions = root.find("./initial_conditions/cell_positions")
    if cell_positions is not None:
        cell_positions.attrib["enabled"] = "false"
    rules = root.find("./cell_rules/rulesets/ruleset")
    if rules is not None:
        rules.attrib["enabled"] = "false"
    config = run_dir / ("smoke_config.xml" if smoke else "PhysiCell_settings.xml")
    ET.indent(tree, space="  ")
    tree.write(config, encoding="utf-8", xml_declaration=True)
    return config


def compile_scenario() -> Path:
    binary = SCENARIO_DIR / "agentvc_worker_cci"
    proc = subprocess.run(
        ["make", "agentvc_worker_cci"],
        cwd=SCENARIO_DIR,
        text=True,
        capture_output=True,
    )
    (OUT / "logs/physicell_compile.log").write_text(
        proc.stdout + "\n" + proc.stderr, encoding="utf-8"
    )
    if proc.returncode != 0 or not binary.exists():
        raise RuntimeError(
            "Real PhysiCell compilation failed; see logs/physicell_compile.log"
        )
    return binary


def best_mat_array(path: Path) -> np.ndarray:
    mats = loadmat(path)
    arrays = [
        v
        for key, v in mats.items()
        if not key.startswith("__") and hasattr(v, "shape") and v.size
    ]
    if not arrays:
        raise RuntimeError(f"No matrix array in {path}")
    return max(arrays, key=lambda x: x.size)


def label_indices(xml_root: ET.Element) -> dict[str, tuple[int, int]]:
    result = {}
    for label in xml_root.findall(
        ".//cellular_information//simplified_data/labels/label"
    ):
        result[(label.text or "").strip()] = (
            int(label.attrib["index"]),
            int(label.attrib.get("size", "1")),
        )
    return result


def parse_endpoint(run_dir: Path, seed: int, formal: bool) -> dict:
    xml_path = run_dir / "output00000001.xml"
    if not xml_path.exists():
        candidates = sorted(run_dir.glob("output*.xml"))
        if len(candidates) < 2:
            raise RuntimeError(f"No final PhysiCell snapshot in {run_dir}")
        xml_path = candidates[-1]
    xml = ET.parse(xml_path).getroot()
    cell_file = xml.findtext(".//cellular_information//simplified_data/filename")
    micro_file = xml.findtext(".//microenvironment//domain/data/filename")
    if not cell_file or not micro_file:
        raise RuntimeError("Final MCDS XML lacks cell or microenvironment matrix")
    cells = best_mat_array(run_dir / cell_file)
    micro = best_mat_array(run_dir / micro_file)
    labels = label_indices(xml)
    required = [
        "ID",
        "position",
        "cell_type",
        "dead",
        "current_phase",
        "migration_speed",
        "secretion_rates",
        "parent_index",
        "initial_worker_index",
        "represented_weight",
    ]
    missing = [x for x in required if x not in labels]
    if missing:
        raise RuntimeError(f"PhysiCell output lacks custom/phenotype labels: {missing}")
    programs = pd.read_csv(OUT / "02_worker_generation/frozen_parent_programs.csv")
    by_parent = programs.set_index("parent_index")
    initial = pd.read_csv(
        OUT
        / "02_worker_generation"
        / (
            "worker_initialization_registry.csv"
            if formal
            else "smoke_worker_initialization_registry.csv"
        )
    )
    initial_id = initial.set_index("initial_worker_index")["worker_id"].to_dict()

    def row(name: str) -> int:
        return labels[name][0]

    rows = []
    secretion_start, secretion_size = labels["secretion_rates"]
    for j in range(cells.shape[1]):
        cid = int(round(float(cells[row("ID"), j])))
        parent = int(round(float(cells[row("parent_index"), j])))
        initial_index = int(round(float(cells[row("initial_worker_index"), j])))
        if parent not in by_parent.index:
            raise RuntimeError(f"Endpoint cell {cid} has invalid parent {parent}")
        pr = by_parent.loc[parent]
        is_initial = cid < len(initial)
        worker_id = (
            str(initial_id.get(initial_index, f"worker_initial_unknown_{initial_index}"))
            if is_initial
            else f"worker_seed_{seed}_daughter_cell_{cid}"
        )
        secretion = cells[secretion_start : secretion_start + secretion_size, j]
        rows.append(
            {
                "worker_id": worker_id,
                "physicell_cell_id": cid,
                "initial_worker_index": initial_index,
                "parent_index": parent,
                "parent_agent_id": str(pr["parent_agent_id"]),
                "cell_type": str(pr["cell_type"]),
                "x_model": float(cells[row("position"), j]),
                "y_model": float(cells[row("position") + 1, j]),
                "alive": bool(float(cells[row("dead"), j]) <= 0),
                "represented_weight": float(cells[row("represented_weight"), j]),
                "current_phase": int(round(float(cells[row("current_phase"), j]))),
                "migration_speed": float(cells[row("migration_speed"), j]),
                "secretion_sum": float(np.sum(secretion)),
                "generation_seed": int(seed),
                "is_initial_worker": bool(is_initial),
            }
        )
    endpoint = pd.DataFrame(rows)
    pixel = model_to_pixel(endpoint[["x_model", "y_model"]].to_numpy(float))
    endpoint["x_pixel"] = pixel[:, 0]
    endpoint["y_pixel"] = pixel[:, 1]
    endpoint.to_csv(run_dir / "workers_endpoint_all.csv", index=False)
    live = endpoint[endpoint["alive"]].copy()
    live.to_csv(run_dir / "workers_endpoint_live.csv", index=False)

    var_ids = {
        v.attrib.get("name", ""): int(v.attrib.get("ID", i))
        for i, v in enumerate(xml.findall(".//microenvironment//variables/variable"))
    }
    field_stats = {}
    for name, idx in var_ids.items():
        values = micro[4 + idx] if 4 + idx < micro.shape[0] else np.array([])
        field_stats[name] = {
            "min": float(np.min(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "finite": bool(np.all(np.isfinite(values))) if values.size else False,
            "nonnegative": bool(np.min(values) >= -1e-12) if values.size else False,
        }
    counts = live["cell_type"].value_counts().reindex(TYPES, fill_value=0)
    parent_effective = []
    for parent, pr in by_parent.iterrows():
        sub = live[live["parent_index"].eq(parent)]
        parent_effective.append(
            {
                "parent_index": int(parent),
                "parent_agent_id": str(pr["parent_agent_id"]),
                "cell_type": str(pr["cell_type"]),
                "live_workers": int(len(sub)),
                "live_conserved_weight": float(sub["represented_weight"].sum()),
                "frozen_parent_mass": float(pr["parent_mass"]),
                "effective_secretion_mass_if_live": (
                    float(pr["parent_mass"]) if len(sub) else 0.0
                ),
            }
        )
    parent_effective_df = pd.DataFrame(parent_effective)
    parent_effective_df.to_csv(run_dir / "parent_source_conservation.csv", index=False)
    status = "PASS_REAL_PHYSICELL_WORKER_ENDPOINT"
    blockers = []
    if formal:
        for cell_type, n in counts.items():
            if int(n) < N_PER_TYPE:
                blockers.append(f"BLOCKED_INSUFFICIENT_WORKERS_{cell_type}:{int(n)}")
    if any(not x["finite"] or not x["nonnegative"] for x in field_stats.values()):
        blockers.append("BLOCKED_INVALID_BIOFVM_FIELD")
    if blockers:
        status = "FAIL_REAL_PHYSICELL_WORKER_ENDPOINT"
    audit = {
        "status": status,
        "blockers": blockers,
        "REAL_PHYSICELL_USED": True,
        "PHYSICELL_XML_PARSED": str(xml_path),
        "PHYSICELL_CELLS_MAT_PARSED": str(run_dir / cell_file),
        "PHYSICELL_MICROENVIRONMENT_MAT_PARSED": str(run_dir / micro_file),
        "generation_seed": int(seed),
        "initial_workers": int(len(initial)),
        "endpoint_cells_total": int(len(endpoint)),
        "endpoint_live_workers": int(len(live)),
        "births": int(max(0, len(endpoint) - len(initial))),
        "deaths_or_removed": int(max(0, len(initial) - live["is_initial_worker"].sum())),
        "live_counts_by_type": counts.astype(int).to_dict(),
        "live_parent_programs": int(live["parent_agent_id"].nunique()),
        "llm_calls": 0,
        "worker_coordinate_unique": int(
            len(np.unique(live[["x_model", "y_model"]].to_numpy(), axis=0))
        ),
        "source_conservation": {
            "parents_with_no_live_worker": int(
                (parent_effective_df["live_workers"] == 0).sum()
            ),
            "effective_parent_mass_sum": float(
                parent_effective_df["effective_secretion_mass_if_live"].sum()
            ),
            "frozen_parent_mass_sum": float(
                parent_effective_df["frozen_parent_mass"].sum()
            ),
        },
        "biofvm_fields": field_stats,
        "endpoint_csv_sha256": sha256_file(run_dir / "workers_endpoint_live.csv"),
    }
    write_json(run_dir / "worker_generation_audit.json", audit)
    if status.startswith("PASS"):
        (run_dir / "PHYSICELL_WORKER_COMPLETE.marker").write_text(
            f"{status}\n", encoding="utf-8"
        )
    return audit


def run_one(seed: int, smoke: bool) -> dict:
    workers_path = OUT / "02_worker_generation/worker_initialization_registry.csv"
    programs_path = OUT / "02_worker_generation/frozen_parent_programs.csv"
    if not workers_path.exists() or not programs_path.exists():
        workers, _ = build_worker_registry()
    else:
        workers = pd.read_csv(workers_path)
    registry = make_smoke_registry(workers) if smoke else workers_path
    run_dir = OUT / "02_worker_generation" / (
        "smoke_seed_1701" if smoke else f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "PHYSICELL_WORKER_COMPLETE.marker"
    if marker.exists():
        audit = json.loads((run_dir / "worker_generation_audit.json").read_text())
        if audit.get("status") == "PASS_REAL_PHYSICELL_WORKER_ENDPOINT":
            return audit
    binary = compile_scenario()
    config = build_config(run_dir, seed, smoke)
    env = os.environ.copy()
    env.update(
        {
            "AGENTVC_WORKER_REGISTRY": str(registry),
            "AGENTVC_PARENT_PROGRAMS": str(programs_path),
            "AGENTVC_WORKER_OUT": str(run_dir),
            "AGENTVC_SMOKE": "1" if smoke else "0",
        }
    )
    start = time.time()
    proc = subprocess.run(
        [str(binary), str(config)],
        cwd=SCENARIO_DIR,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - start
    (OUT / "logs" / f"physicell_{'smoke' if smoke else seed}.log").write_text(
        proc.stdout + "\n" + proc.stderr, encoding="utf-8"
    )
    if proc.returncode != 0:
        audit = {
            "status": "FAIL_REAL_PHYSICELL_EXECUTION",
            "returncode": int(proc.returncode),
            "elapsed_seconds": elapsed,
            "log": str(OUT / "logs" / f"physicell_{'smoke' if smoke else seed}.log"),
        }
        write_json(run_dir / "worker_generation_audit.json", audit)
        raise RuntimeError(f"Real PhysiCell failed for seed {seed}: rc={proc.returncode}")
    audit = parse_endpoint(run_dir, seed, formal=not smoke)
    audit["elapsed_seconds"] = elapsed
    audit["binary_sha256"] = sha256_file(binary)
    audit["config_sha256"] = sha256_file(config)
    write_json(run_dir / "worker_generation_audit.json", audit)
    if audit["status"] != "PASS_REAL_PHYSICELL_WORKER_ENDPOINT":
        raise RuntimeError(";".join(audit.get("blockers", [audit["status"]])))
    return audit


def run_formal() -> dict:
    audits = []
    for seed in WORKER_SEEDS:
        audits.append(run_one(seed, smoke=False))
    summary = {
        "status": "PASS_ALL_REAL_PHYSICELL_WORKER_REALIZATIONS",
        "worker_seeds": WORKER_SEEDS,
        "realizations": audits,
        "llm_calls": 0,
    }
    write_json(OUT / "02_worker_generation/formal_worker_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["audit-freeze", "registry", "smoke", "formal", "all-workers"],
    )
    args = parser.parse_args()
    if args.stage == "audit-freeze":
        print(json.dumps(freeze_protocol(), indent=2))
    elif args.stage == "registry":
        workers, programs = build_worker_registry()
        print(json.dumps({"workers": len(workers), "parents": len(programs)}, indent=2))
    elif args.stage == "smoke":
        print(json.dumps(run_one(1701, smoke=True), indent=2))
    elif args.stage == "formal":
        print(json.dumps(run_formal(), indent=2))
    else:
        freeze_protocol()
        build_worker_registry()
        run_one(1701, smoke=True)
        print(json.dumps(run_formal(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
