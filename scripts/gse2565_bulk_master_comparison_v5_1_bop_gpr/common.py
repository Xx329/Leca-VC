#!/usr/bin/env python3
"""Shared frozen paths and helpers for stable BOP-DMD comparison V5.1."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE2565_bulk_master_comparison_v5_1_bop_gpr"
PLOT_DATA = OUT / "plot_input_tables"
ENV_DIR = OUT / "environment"
LOG_DIR = OUT / "logs"

OLD = ROOT / "outputs/GSE2565_external_bulk_baselines_v1"
V21 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v2_1"
V4 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v4_online_agent"
GPR_OUT = ROOT / "outputs/GSE2565_bulk_agent_gpr_chronode_main_v1"
HELDOUT = ROOT / "outputs/GSE2565_bulk_agent_gpr_chronode_heldout_v2"
FORMAL_DNB = ROOT / "outputs/GSE2565_bulk_dnb_scope_corrected_v3"
COMMON_DNB = OLD / "revision_panelA_chronodeBS_B20_v1"

REAL_PATH = V21 / "real_data/biological_sample_gene_expression.csv.gz"
MODULE_PATH = V21 / "metrics/methods/functional_modules.csv"
GENE_PATH = OLD / "audit/frozen_gene_space.txt"
CHRON_PATH = OLD / "chronode/formal/chronode_monotonic_fitted_expression.npz"
CHRON_PARAMETERS = OLD / "chronode/formal/chronode_monotonic_parameters.csv"
RVA_PATH = OLD / "rvagene/formal/rvagene_reconstruction_mean.npz"
RVA_SEEDS_PATH = OLD / "rvagene/formal/rvagene_reconstruction_by_seed.npz"
GPR_PATH = GPR_OUT / "gpr_predicted_trajectory.npz"
GPR_DNB_PATH = GPR_OUT / "gpr_dnb_bootstrap_curve.csv"
FROZEN_HELDOUT_PATH = HELDOUT / "heldout_unified_metrics.csv"
FORMAL_DNB_PATH = FORMAL_DNB / "formal_fullspace_dnb_metrics.csv"

TIMES = np.asarray([0, 0.5, 1, 4, 8, 12, 24, 48, 72], float)
STATE_INDICES = np.arange(4, 9)
VELOCITY_INDICES = np.arange(3, 8)

BOP_PRIMARY_SEED = 2565201
BOP_BOOTSTRAP_SEED_BASE = 2565300
BOP_RANK = 4
BOP_NUM_TRIALS = 100
BOP_TRIAL_SIZE = 0.8
BOP_BOOTSTRAPS = 20
BOP_EIG_CONSTRAINTS = {"stable", "conjugate_pairs"}
BOP_VARPRO_TOL = 0.2
BOP_RANK_SENSITIVITY = [2, 4, 6]
COMPLEX_TOLERANCE = 1e-10
DETERMINISM_TOLERANCE = 1e-10


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = import_module(
    "gse2565_v5_1_base",
    ROOT / "scripts/gse2565_bulk_agent_gpr_chronode_main_v1/run_benchmark.py",
)
EVAL = BASE.E
REVISION = BASE.R


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def ensure_dirs() -> None:
    for path in [OUT, PLOT_DATA, ENV_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
