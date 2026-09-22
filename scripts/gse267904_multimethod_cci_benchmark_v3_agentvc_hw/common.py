#!/usr/bin/env python3
"""Constants and utilities for GSE267904 AgentVC-HW CCI V3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import distance_matrix


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = ROOT / "outputs/GSE267904_multimethod_cci_benchmark_v3_agentvc_hw"
V2 = ROOT / "outputs/GSE267904_multimethod_cci_benchmark_v2"
WORKER_V1 = ROOT / "outputs/GSE267904_agentvc_worker_scgpt_cci_main_v1"
K100 = ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"

D7 = K100 / "real_benchmark/d7_bleo.h5ad"
EARLY_PARENT = K100 / "virtual_commot_input/virtual_early_bleo.h5ad"
LATE_PARENT = K100 / "virtual_commot_input/virtual_late_bleo.h5ad"
REGISTRY = WORKER_V1 / "02_worker_generation/worker_initialization_registry.csv"
PANEL = ROOT / "outputs/GSE267904_scgpt_cci_v1/frozen/scgpt_1200_mouse_human_panel.csv"
PAIRS = ROOT / "outputs/GSE267904_scgpt_cci_v1/frozen/common_mouse_cellchat_lr_pairs.csv"

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
SAMPLE_SEEDS = list(range(2679041, 2679051))
QUALIFICATION_SEEDS = SAMPLE_SEEDS[:3]
WORKER_SEEDS = [1701, 1702, 1703]
WORKER_SEED_FOR_SAMPLE = {
    seed: WORKER_SEEDS[i % 3] for i, seed in enumerate(SAMPLE_SEEDS)
}

OLD_METHODS = ["scgpt", "scgen", "wot_external", "cellrank_proxy", "worker"]
PREDICTED_METHODS = OLD_METHODS + ["worker_hw"]
ALL_SOURCES = ["observed", "scgpt", "scgen", "wot_external", "cellrank_proxy", "worker_hw"]
METHOD_LABELS = {
    "observed": "Observed d21 reference",
    "scgpt": "Frozen scGPT + external conditional head",
    "scgen": "scGen 2.1.0 external expression prediction",
    "wot_external": "external-WOT transport projection*",
    "cellrank_proxy": "CellRank-derived fate-weighted expression proxy†",
    "worker": "Original AgentVC worker-expression proxy",
    "worker_hw": "AgentVC heterogeneous-worker expression proxy (AgentVC-HW)",
}
COLORS = {
    "observed": "#333333",
    "scgpt": "#0072B2",
    "scgen": "#E69F00",
    "wot_external": "#009E73",
    "cellrank_proxy": "#56B4E9",
    "worker": "#B8A0B0",
    "worker_hw": "#CC3F8D",
}

TOTAL_KEY = "commot-cellchat-total-total"
EXPECTED_KEYS = [TOTAL_KEY] + [f"commot-cellchat-{family}" for family in FAMILIES]
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
SUPPORT_EPS = 1e-8
PROJECTION_TOL = 1e-10
PROJECTION_MAX_ITER = 10000
GNRS_TARGET = 0.4919722089377756
LOCAL_TARGET = 0.00400031932730395

DIRS = [
    "00_protocol",
    "01_expression/early",
    "01_expression/late",
    "02_d7_qualification/inputs",
    "02_d7_qualification/commot",
    "03_late_inputs/full",
    "03_late_inputs/sampled",
    "04_commot_runs",
    "05_metrics",
    "06_figure/source_data",
    "07_audits",
    "logs",
]


def ensure_dirs() -> None:
    for rel in DIRS:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float64)


def genes_and_pairs() -> tuple[list[str], pd.DataFrame]:
    genes = pd.read_csv(PANEL)["mouse_symbol"].astype(str).tolist()
    if len(genes) != 1200 or len(set(genes)) != 1200:
        raise RuntimeError("Frozen panel is not 1,200 unique genes")
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


def panel_matrix(a: ad.AnnData, genes: list[str]) -> np.ndarray:
    index = {str(gene): i for i, gene in enumerate(a.var_names)}
    missing = [gene for gene in genes if gene not in index]
    if missing:
        raise RuntimeError(f"Missing panel genes: {missing[:10]}")
    return dense(a.X[:, [index[gene] for gene in genes]])


def cp10k(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise RuntimeError("Expression is negative or non-finite")
    totals = x.sum(axis=1)
    if (totals <= 0).any():
        raise RuntimeError("At least one row has zero panel library")
    return x * (10000.0 / totals[:, None])


def make_h5ad(
    pseudo: np.ndarray,
    obs: pd.DataFrame,
    coords: np.ndarray,
    genes: list[str],
    provenance: dict,
) -> ad.AnnData:
    pseudo = np.asarray(pseudo, dtype=np.float64)
    if not np.isfinite(pseudo).all() or (pseudo < 0).any():
        raise RuntimeError("Projected pseudo-count is invalid")
    result = ad.AnnData(
        X=sparse.csr_matrix(np.log1p(pseudo)),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    result.layers["pseudo_count"] = pseudo
    result.obsm["spatial"] = np.asarray(coords, dtype=np.float64)
    result.uns["expression_preprocessing"] = provenance
    return result


def no_self_distance(coords: np.ndarray) -> np.ndarray:
    values = distance_matrix(np.asarray(coords, float), np.asarray(coords, float))
    np.fill_diagonal(values, DIS_THR + 1.0)
    return values.astype(np.float32)


def median_nn_coords(coords: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, float)
    nn = cKDTree(coords).query(coords, k=2)[0][:, 1]
    median = float(np.median(nn[nn > 0]))
    if not np.isfinite(median) or median <= 0:
        raise RuntimeError("Cannot derive median-NN scale")
    return (coords - coords.mean(axis=0)) * (100.0 / median)


def old_worker_sample(seed: int) -> Path:
    return WORKER_V1 / f"05_commot_inputs/worker/seed_{seed}/commot_input.h5ad"


def late_full_path(worker_seed: int) -> Path:
    return OUT / f"03_late_inputs/full/worker_hw_seed_{worker_seed}.h5ad"


def late_sample_path(seed: int) -> Path:
    return OUT / f"03_late_inputs/sampled/seed_{seed}/commot_input.h5ad"


def formal_output(seed: int, mode: str) -> Path:
    return OUT / f"04_commot_runs/{mode}/worker_hw/seed_{seed}/edges_long.csv"
