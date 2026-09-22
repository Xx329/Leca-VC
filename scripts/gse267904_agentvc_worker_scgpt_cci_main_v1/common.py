#!/usr/bin/env python3
"""Shared frozen constants and utilities for the GSE267904 worker CCI benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import distance_matrix


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "scenarios/gse267904_agentvc_worker_scgpt_cci_main_v1"
OUT = Path(os.environ.get("GSE267904_WORKER_CCI_OUT", str(ROOT / "outputs/GSE267904_agentvc_worker_scgpt_cci_main_v1"))).resolve()
K100 = Path(os.environ.get("GSE267904_K100_ROOT", str(ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"))).resolve()
SCGPT_V1 = ROOT / "outputs/GSE267904_scgpt_cci_v1"

D7 = K100 / "real_benchmark/d7_bleo.h5ad"
D21 = K100 / "real_benchmark/d21_bleo.h5ad"
LATE_AGENT = K100 / "virtual_commot_input/virtual_late_bleo.h5ad"
PARENT_PROGRAM = K100 / "physicell/applied_decision_1.csv"
CELL_AGENT_MAP = K100 / "physicell/cell_agent_map.csv"
SCALED_REGISTRY = K100 / "physicell/micro_agent_registry_physicell_scaled.csv"
AFFINE_AUDIT = (
    ROOT
    / "outputs/GSE267904_scgpt_cci_capacity_matched_v1/audit/coordinate_scale_audit.json"
)
SCGPT = SCGPT_V1 / "scgpt_prediction/scgpt_predicted_d21_from_d7_spots.h5ad"
SCGPT_VALIDATION = SCGPT_V1 / "audit/external_transition_validation.json"
PANEL = SCGPT_V1 / "frozen/scgpt_1200_mouse_human_panel.csv"
PAIR_UNIVERSE = SCGPT_V1 / "frozen/common_mouse_cellchat_lr_pairs.csv"
D7_SCALEFACTORS = (
    ROOT
    / "data/GSE267904/raw/GSM8281387_V10A20-072-B1_scalefactors_json.json.gz"
)
BASE_PHYSICELL_CONFIG = (
    ROOT / "scenarios/gse267904_spatial_commot_agent/PhysiCell_settings.xml"
)

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
FAMILIES = ["TGFb", "CCL", "CXCL", "PDGF", "VEGF"]
SAMPLE_SEEDS = list(range(2679041, 2679051))
WORKER_SEEDS = [1701, 1702, 1703]
WORKER_SEED_FOR_SAMPLE = {
    sample_seed: WORKER_SEEDS[i % len(WORKER_SEEDS)]
    for i, sample_seed in enumerate(SAMPLE_SEEDS)
}
N_PER_TYPE = 90
GRID_N = 6
DIS_THR = 500.0
SPOT_DIAMETER_FULLRES = 129.2138
SPOT_RADIUS_FULLRES = SPOT_DIAMETER_FULLRES / 2.0
MAX_TIME = 125.0

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

DIRS = [
    "00_protocol",
    "01_worker_audit",
    "02_worker_generation",
    "03_worker_expression",
    "04_geometry_normalization",
    "05_commot_inputs",
    "06_commot_runs",
    "07_metrics",
    "08_figures",
    "09_supplementary",
    "10_audits",
    "logs",
]


def ensure_dirs() -> None:
    for rel in DIRS:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def affine() -> tuple[np.ndarray, float]:
    obj = json.loads(AFFINE_AUDIT.read_text())
    if obj.get("status") != "PASS_EXACT_AFFINE_INVERSE":
        raise RuntimeError("Frozen affine audit did not pass")
    return np.asarray(obj["center_xy"], dtype=float), float(obj["scale"])


def pixel_to_model(xy: np.ndarray) -> np.ndarray:
    center, scale = affine()
    return (np.asarray(xy, dtype=float) - center) * scale


def model_to_pixel(xy: np.ndarray) -> np.ndarray:
    center, scale = affine()
    return np.asarray(xy, dtype=float) / scale + center


def frozen_pairs_and_genes() -> tuple[pd.DataFrame, list[str]]:
    pairs = pd.read_csv(PAIR_UNIVERSE)
    pairs = (
        pairs[pairs["pathway"].isin(FAMILIES)]
        .copy()
        .sort_values(["pathway", "database_row"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected = {"CCL": 24, "TGFb": 12, "CXCL": 8, "VEGF": 8, "PDGF": 6}
    if len(pairs) != 58 or pairs["pathway"].value_counts().to_dict() != expected:
        raise RuntimeError("Frozen focused panel is not the expected 58 exact pairs")
    panel = pd.read_csv(PANEL)
    genes = panel["mouse_symbol"].astype(str).tolist()
    if len(genes) != 1200 or len(set(genes)) != 1200:
        raise RuntimeError("Frozen scGPT common panel is not 1,200 unique genes")
    components = {
        g
        for value in pairs["all_mouse_components"].astype(str)
        for g in value.split(";")
        if g
    }
    if not components.issubset(set(genes)):
        raise RuntimeError("At least one focused LR component is outside the common panel")
    return pairs, genes


def dense_panel(a: ad.AnnData, genes: list[str]) -> np.ndarray:
    index = {str(g): i for i, g in enumerate(a.var_names)}
    missing = sorted(set(genes) - set(index))
    if missing:
        raise RuntimeError(f"Input lacks {len(missing)} common-panel genes: {missing[:10]}")
    x = a.X[:, [index[g] for g in genes]]
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)) or (x < 0).any():
        raise RuntimeError("Expression contains negative or non-finite values")
    return x


def cp10k_log1p(pseudo_count: np.ndarray) -> np.ndarray:
    x = np.asarray(pseudo_count, dtype=np.float64)
    if not np.all(np.isfinite(x)) or (x < 0).any():
        raise RuntimeError("Pseudo-count matrix must be finite and nonnegative")
    totals = x.sum(axis=1)
    if (totals <= 0).any():
        bad = np.where(totals <= 0)[0][:10].tolist()
        raise RuntimeError(f"Zero common-panel library in rows {bad}")
    out = np.log1p(x * (10000.0 / totals[:, None]))
    return out.astype(np.float32)


def grid_ids(coords: np.ndarray, grid_n: int = GRID_N) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    bins = np.floor((coords - lo) / span * grid_n).astype(int)
    bins = bins.clip(0, grid_n - 1)
    return bins[:, 0] * grid_n + bins[:, 1]


def largest_remainder(counts: pd.Series, total: int) -> pd.Series:
    counts = counts.astype(int).sort_index()
    quota = counts / counts.sum() * total
    base = np.floor(quota).astype(int)
    remaining = total - int(base.sum())
    if remaining:
        ranking = pd.DataFrame(
            {
                "position": np.arange(len(quota), dtype=int),
                "fraction": (quota - base).to_numpy(),
                "key": quota.index.astype(str),
            }
        )
        order = ranking.sort_values(
            ["fraction", "key"], ascending=[False, True], kind="mergesort"
        )["position"].to_numpy()
        for position in order[:remaining]:
            base.iloc[position] += 1
    if (base > counts).any() or int(base.sum()) != total:
        raise RuntimeError("Largest-remainder allocation is infeasible")
    return base


def spatial_sample_indices(
    labels: np.ndarray,
    coords: np.ndarray,
    entity_ids: np.ndarray,
    seed: int,
    n_per_type: int = N_PER_TYPE,
) -> tuple[np.ndarray, pd.DataFrame]:
    labels = np.asarray(labels, dtype=str)
    coords = np.asarray(coords, dtype=float)
    entity_ids = np.asarray(entity_ids, dtype=str)
    rng = np.random.default_rng(seed)
    gids = grid_ids(coords)
    selected: list[int] = []
    rows: list[dict] = []
    for cell_type in TYPES:
        idx = np.where(labels == cell_type)[0]
        if len(idx) < n_per_type:
            raise RuntimeError(
                f"BLOCKED_INSUFFICIENT_ENTITIES_{cell_type}: {len(idx)} < {n_per_type}"
            )
        counts = pd.Series(gids[idx]).value_counts().sort_index()
        allocation = largest_remainder(counts, n_per_type)
        for gid, n in allocation.items():
            candidates = idx[gids[idx] == int(gid)]
            take = np.sort(rng.choice(candidates, size=int(n), replace=False))
            selected.extend(take.tolist())
            for i in take:
                rows.append(
                    {
                        "sampling_seed": int(seed),
                        "entity_id": entity_ids[i],
                        "cell_type": cell_type,
                        "grid_id": int(gid),
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                    }
                )
    selected_array = np.asarray(sorted(selected), dtype=int)
    manifest = pd.DataFrame(rows).sort_values(
        ["cell_type", "grid_id", "entity_id"], kind="mergesort"
    )
    if len(selected_array) != n_per_type * len(TYPES):
        raise RuntimeError("Capacity-matched sample size mismatch")
    if manifest["entity_id"].duplicated().any():
        raise RuntimeError("Sampling manifest contains duplicated entities")
    return selected_array, manifest


def no_self_distance(coords: np.ndarray, dis_thr: float = DIS_THR) -> np.ndarray:
    d = distance_matrix(np.asarray(coords, dtype=float), np.asarray(coords, dtype=float))
    np.fill_diagonal(d, dis_thr + 1.0)
    return d.astype(np.float32)


def geometry_stats(
    coords: np.ndarray, labels: np.ndarray, dis_thr: float = DIS_THR
) -> dict:
    labels = np.asarray(labels, dtype=str)
    d = distance_matrix(coords, coords)
    np.fill_diagonal(d, np.inf)
    neighbors = (d <= dis_thr).sum(axis=1)
    supported = set()
    for i, j in zip(*np.where(d <= dis_thr)):
        supported.add((labels[i], labels[j]))
    type_supported = {
        (s, r) for s, r in supported if s in TYPES and r in TYPES
    }
    finite_nn = np.min(d, axis=1)
    return {
        "x_min": float(np.min(coords[:, 0])),
        "x_max": float(np.max(coords[:, 0])),
        "y_min": float(np.min(coords[:, 1])),
        "y_max": float(np.max(coords[:, 1])),
        "median_nearest_neighbor": float(np.median(finite_nn)),
        "mean_neighbors_within_threshold": float(np.mean(neighbors)),
        "zero_neighbor_fraction": float(np.mean(neighbors == 0)),
        "candidate_type_pairs": int(len(type_supported)),
        "candidate_type_pair_fraction": float(len(type_supported) / 81.0),
    }


def normalized_h5ad(
    pseudo_count: np.ndarray,
    obs: pd.DataFrame,
    coords: np.ndarray,
    genes: list[str],
    provenance: dict,
) -> ad.AnnData:
    obs = obs.copy()
    if obs.index.has_duplicates:
        raise RuntimeError("Entity IDs must be unique")
    x = cp10k_log1p(pseudo_count)
    a = ad.AnnData(
        # COMMOT 0.0.3 calls ``.toarray()`` on expression slices, so keep the
        # normalized matrix sparse even though this 1,200-gene panel is small.
        X=sparse.csr_matrix(x),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    a.layers["pseudo_count"] = np.asarray(pseudo_count, dtype=np.float32)
    a.obsm["spatial"] = np.asarray(coords, dtype=np.float64)
    a.uns["expression_preprocessing"] = {
        "target": "log1p(CP10K) on frozen 1,200-gene common panel",
        **provenance,
    }
    return a
