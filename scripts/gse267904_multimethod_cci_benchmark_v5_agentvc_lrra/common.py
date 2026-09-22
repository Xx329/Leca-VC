#!/usr/bin/env python3
"""Frozen constants and utilities for GSE267904 AgentVC-LRRA V5."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import distance_matrix


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = Path(os.environ.get("GSE267904_LRRA_OUT", str(ROOT / "outputs/GSE267904_multimethod_cci_benchmark_v5_agentvc_lrra"))).resolve()
V1 = Path(os.environ.get("GSE267904_WORKER_CCI_OUT", str(ROOT / "outputs/GSE267904_agentvc_worker_scgpt_cci_main_v1"))).resolve()
V2 = ROOT / "outputs/GSE267904_multimethod_cci_benchmark_v2"
V3 = ROOT / "outputs/GSE267904_multimethod_cci_benchmark_v3_agentvc_hw"
V4 = ROOT / "outputs/GSE267904_multimethod_cci_benchmark_v4_agentvc_srhw"
K100 = Path(os.environ.get("GSE267904_K100_ROOT", str(ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"))).resolve()

D7 = K100 / "real_benchmark/d7_bleo.h5ad"
EARLY_PARENT = K100 / "virtual_commot_input/virtual_early_bleo.h5ad"
REGISTRY = V1 / "02_worker_generation/worker_initialization_registry.csv"
PANEL = ROOT / "outputs/GSE267904_scgpt_cci_v1/frozen/scgpt_1200_mouse_human_panel.csv"
PAIRS = ROOT / "outputs/GSE267904_scgpt_cci_v1/frozen/common_mouse_cellchat_lr_pairs.csv"

METHOD_ID = "worker_lrra"
METHOD_NAME = "AgentVC d7-calibrated LR-role expression adapter (AgentVC-LRRA)"
TYPES = [
    "alveolar_epithelial_AT1_AT2",
    "activated_Krt8_ADI_epithelial",
    "airway_epithelial",
    "macrophage",
    "recruited_monocyte_macrophage",
    "fibroblast_myofibroblast",
    "endothelial",
    "dendritic",
    "lymphoid",
]
TYPE_SHORT = {
    "alveolar_epithelial_AT1_AT2": "AT1/AT2",
    "activated_Krt8_ADI_epithelial": "Krt8/ADI",
    "airway_epithelial": "Airway",
    "macrophage": "Macrophage",
    "recruited_monocyte_macrophage": "Mono/Mac",
    "fibroblast_myofibroblast": "Fib/Myofib",
    "endothelial": "Endothelial",
    "dendritic": "Dendritic",
    "lymphoid": "Lymphoid",
}
FAMILIES = ["TGFb", "CCL", "CXCL", "PDGF", "VEGF"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CV_FOLDS = [0, 1, 2]
CV_TRAIN_N = 90
CV_VALIDATION_N = 45
CV_SEEDS = {0: 2679061, 1: 2679062, 2: 2679063}
REFIT_SEED = 2679060
SMOKE_SEED = 2679069
SAMPLE_SEEDS = list(range(2679041, 2679051))
WORKER_SEEDS = [1701, 1702, 1703]
WORKER_SEED_FOR_SAMPLE = {
    seed: WORKER_SEEDS[index % 3] for index, seed in enumerate(SAMPLE_SEEDS)
}

TOTAL_KEY = "commot-cellchat-total-total"
EXPECTED_KEYS = [TOTAL_KEY] + [f"commot-cellchat-{x}" for x in FAMILIES]
DIS_THR = 500.0
COMMOT_CONFIG = {
    "package_version": "0.0.3",
    "database_name": "cellchat",
    "dis_thr": DIS_THR,
    "heteromeric": True,
    "heteromeric_rule": "min",
    "pathway_sum": True,
    "cost_type": "euc",
    "cot_eps_p": 0.1,
    "cot_eps_mu": None,
    "cot_eps_nu": None,
    "cot_rho": 10.0,
    "cot_nitermax": 10000,
    "cot_weights": [0.25, 0.25, 0.25, 0.25],
    "smooth": False,
}
ROLE_EPS = 1e-6
MULTIPLIER_MIN = 0.5
MULTIPLIER_MAX = 2.0
TOL = 1e-12

SUCCESS_THRESHOLDS = {
    "cell_type_local_mean_absolute_error": 0.00400031932730395,
    "GNRS_secondary": 0.4919722089377756,
    "LBSS_secondary": 0.7882435584215964,
    "five_pathway_flow_js_similarity": 0.964409414374817,
    "sender_receiver_role_mean_spearman": 0.22666666666666666,
    "pathway_allocation_mean_absolute_error": 0.09252239488733124,
}

DIRS = [
    "00_protocol",
    "01_d7_cv/inputs",
    "01_d7_cv/commot",
    "01_d7_cv/factors",
    "02_refit/inputs",
    "02_refit/commot",
    "02_refit/frozen",
    "03_late_expression/full",
    "03_late_expression/sampled",
    "04_commot_runs/affine_pixel",
    "04_commot_runs/median_nn",
    "05_metrics",
    "06_figure/source_data",
    "07_audits",
    "08_reports",
    "logs",
]


def ensure_dirs() -> None:
    for relative in DIRS:
        (OUT / relative).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dense(value) -> np.ndarray:
    return np.asarray(
        value.toarray() if sparse.issparse(value) else value, dtype=np.float64
    )


def cp10k(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    if not np.isfinite(value).all() or (value < 0).any():
        raise RuntimeError("Expression contains a negative or non-finite value")
    total = value.sum(axis=1)
    if (total <= 0).any():
        raise RuntimeError("Expression contains a zero-library row")
    return value * (10000.0 / total[:, None])


def make_h5ad(
    pseudo_count: np.ndarray,
    obs: pd.DataFrame,
    coordinates: np.ndarray,
    genes: list[str],
    provenance: dict,
) -> ad.AnnData:
    pseudo_count = cp10k(pseudo_count)
    result = ad.AnnData(
        X=sparse.csr_matrix(np.log1p(pseudo_count)),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    result.layers["pseudo_count"] = pseudo_count
    result.obsm["spatial"] = np.asarray(coordinates, np.float64)
    result.uns["expression_preprocessing"] = provenance
    return result


def genes_and_pairs() -> tuple[list[str], pd.DataFrame]:
    genes = pd.read_csv(PANEL)["mouse_symbol"].astype(str).tolist()
    if len(genes) != 1200 or len(set(genes)) != 1200:
        raise RuntimeError("Frozen gene panel is not 1,200 unique genes")
    pairs = pd.read_csv(PAIRS)
    pairs = (
        pairs[pairs["pathway"].isin(FAMILIES)]
        .sort_values(["pathway", "database_row"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected = {"CCL": 24, "TGFb": 12, "CXCL": 8, "VEGF": 8, "PDGF": 6}
    if len(pairs) != 58 or pairs["pathway"].value_counts().to_dict() != expected:
        raise RuntimeError("Frozen LR universe is not the expected 58-pair panel")
    return genes, pairs


def no_self_distance(coordinates: np.ndarray) -> np.ndarray:
    values = distance_matrix(coordinates, coordinates)
    np.fill_diagonal(values, DIS_THR + 1.0)
    return values.astype(np.float32)


def median_nn_coords(coordinates: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    coordinates = np.asarray(coordinates, np.float64)
    nearest = cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
    median = float(np.median(nearest[nearest > 0]))
    if not np.isfinite(median) or median <= 0:
        raise RuntimeError("Cannot derive median nearest-neighbor scale")
    return (coordinates - coordinates.mean(axis=0)) * (100.0 / median)


def old_worker_full(worker_seed: int) -> Path:
    return V1 / f"03_worker_expression/worker_seed_{worker_seed}_common_normalized.h5ad"


def old_worker_sample(seed: int) -> Path:
    return V1 / f"05_commot_inputs/worker/seed_{seed}/commot_input.h5ad"


def late_full_path(worker_seed: int) -> Path:
    return OUT / f"03_late_expression/full/worker_lrra_seed_{worker_seed}.h5ad"


def late_sample_path(seed: int) -> Path:
    return OUT / f"03_late_expression/sampled/seed_{seed}/commot_input.h5ad"


def formal_output(seed: int, mode: str) -> Path:
    return OUT / f"04_commot_runs/{mode}/seed_{seed}/edges_long.csv"
