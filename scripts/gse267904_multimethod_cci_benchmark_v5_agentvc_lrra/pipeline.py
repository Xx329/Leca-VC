#!/usr/bin/env python3
"""Gated GSE267904 AgentVC-LRRA V5 calibration and CCI benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from common import (
    ALPHAS,
    COMMOT_CONFIG,
    CV_FOLDS,
    CV_SEEDS,
    CV_TRAIN_N,
    CV_VALIDATION_N,
    D7,
    EARLY_PARENT,
    EXPECTED_KEYS,
    FAMILIES,
    METHOD_ID,
    METHOD_NAME,
    MULTIPLIER_MAX,
    MULTIPLIER_MIN,
    OUT,
    PAIRS,
    PANEL,
    REFIT_SEED,
    REGISTRY,
    ROLE_EPS,
    ROOT,
    SAMPLE_SEEDS,
    SCRIPT_DIR,
    SMOKE_SEED,
    SUCCESS_THRESHOLDS,
    TOL,
    TOTAL_KEY,
    TYPES,
    V1,
    V2,
    V3,
    V4,
    WORKER_SEEDS,
    WORKER_SEED_FOR_SAMPLE,
    cp10k,
    dense,
    ensure_dirs,
    formal_output,
    genes_and_pairs,
    late_full_path,
    late_sample_path,
    make_h5ad,
    median_nn_coords,
    no_self_distance,
    old_worker_full,
    old_worker_sample,
    sha256_file,
    write_json,
)


PROTOCOL = OUT / "00_protocol/frozen_protocol.json"
PROTOCOL_HASH = OUT / "00_protocol/frozen_protocol_hash.json"
CV_PASS = OUT / "01_d7_cv/PASS_D7_LRRA_CV_GATE.marker"
FACTOR_PASS = OUT / "02_refit/PASS_LRRA_FACTORS_FROZEN.marker"
LATE_PASS = OUT / "03_late_expression/PASS_LATE_EXPRESSION_FROZEN.marker"
AFFINE_PASS = OUT / "05_metrics/PASS_AFFINE_METRICS_GATE.marker"
MEDIAN_PASS = OUT / "05_metrics/PASS_MEDIAN_NN_GATE.marker"
FINAL_PASS = OUT / "PASS_AGENTVC_LRRA_TOP1_LOCAL_NO_REGRESSION.marker"
NO_FIGURE = OUT / "06_figure/NO_FIGURE_GENERATED.md"
EPS = 1e-15


def require_marker(path: Path, name: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{name} marker is absent: {path}")


def no_figure(status: str, reason: str) -> None:
    NO_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    NO_FIGURE.write_text(
        "# NO FIGURE GENERATED\n\n"
        f"Status: `{status}`\n\nReason: {reason}\n\n"
        "The frozen protocol prohibits PNG, PDF and SVG generation after a failed gate.\n",
        encoding="utf-8",
    )


def required_paths() -> list[Path]:
    paths = [
        D7,
        EARLY_PARENT,
        REGISTRY,
        PANEL,
        PAIRS,
        SCRIPT_DIR / "common.py",
        SCRIPT_DIR / "pipeline.py",
        SCRIPT_DIR / "plot_main.py",
        V2 / "08_audits/final_validation.json",
        V2 / "06_metrics/commot_aggregated_edges.csv",
        V2 / "06_metrics/seed_level_metrics.csv",
        V2 / "06_metrics/pathway_allocation.csv",
        V2 / "06_metrics/sender_receiver_roles.csv",
        V2 / "06_metrics/edge_level_values.csv",
        V2 / "06_metrics/cell_type_local_errors.csv",
        V3 / "07_audits/final_validation.json",
        V4 / "07_audits/final_validation.json",
    ]
    for worker_seed in WORKER_SEEDS:
        paths.extend(
            [
                old_worker_full(worker_seed),
                V1 / f"02_worker_generation/seed_{worker_seed}/workers_endpoint_live.csv",
                V1 / f"02_worker_generation/seed_{worker_seed}/PHYSICELL_WORKER_COMPLETE.marker",
            ]
        )
    for seed in SAMPLE_SEEDS:
        paths.append(old_worker_sample(seed))
    return paths


def freeze() -> dict:
    ensure_dirs()
    missing = [str(path) for path in required_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing V5 inputs:\n" + "\n".join(missing))
    if PROTOCOL.exists() or PROTOCOL_HASH.exists():
        if not (PROTOCOL.exists() and PROTOCOL_HASH.exists()):
            raise RuntimeError("Protocol freeze is incomplete")
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        audit = json.loads(PROTOCOL_HASH.read_text(encoding="utf-8"))
        if sha256_file(PROTOCOL) != audit["protocol_sha256"]:
            raise RuntimeError("Frozen V5 protocol changed")
        changed = [
            path
            for path, expected in protocol["input_sha256"].items()
            if not Path(path).is_file() or sha256_file(Path(path)) != expected
        ]
        if changed:
            raise RuntimeError("Frozen input changed: " + ", ".join(changed))
        return protocol

    protocol = {
        "experiment_id": "GSE267904_multimethod_cci_benchmark_v5_agentvc_lrra",
        "method_id": METHOD_ID,
        "method_name": METHOD_NAME,
        "semantics": "post-hoc GSE267904 d7-calibrated LR-role adapter",
        "historical_d21_results_already_known": True,
        "GSE267904_d21_parameter_selection": "forbidden",
        "llm_calls": 0,
        "physicell_runs": 0,
        "fold_formula": "(grid_x + 2*grid_y) mod 3 on equal-width 6x6 grid",
        "cv_train_per_type": CV_TRAIN_N,
        "cv_validation_per_type": CV_VALIDATION_N,
        "alphas": ALPHAS,
        "role_epsilon": ROLE_EPS,
        "multiplier_bounds": [MULTIPLIER_MIN, MULTIPLIER_MAX],
        "non_lr_direct_modification": False,
        "commot": COMMOT_CONFIG,
        "success_thresholds": SUCCESS_THRESHOLDS,
        "input_sha256": {str(path): sha256_file(path) for path in required_paths()},
    }
    write_json(PROTOCOL, protocol)
    write_json(
        PROTOCOL_HASH,
        {"status": "PASS_PROTOCOL_FROZEN", "protocol_sha256": sha256_file(PROTOCOL)},
    )
    write_json(
        OUT / "07_audits/d21_access_audit.json",
        {
            "status": "PARAMETER_FREEZE_PENDING_D21_ACCESS_DENIED",
            "GSE267904_d21_opened": False,
            "allowed_after": "PASS_LRRA_FACTORS_FROZEN",
        },
    )
    print(json.dumps(protocol, indent=2))
    return protocol


def panel_matrix(source: ad.AnnData, genes: list[str]) -> np.ndarray:
    index = {str(gene): i for i, gene in enumerate(source.var_names)}
    missing = [gene for gene in genes if gene not in index]
    if missing:
        raise RuntimeError(f"Missing panel genes: {missing[:10]}")
    return dense(source.X[:, [index[gene] for gene in genes]])


def parent_targets(genes: list[str]) -> dict[str, np.ndarray]:
    source = ad.read_h5ad(EARLY_PARENT)
    source.obs["agent_id"] = source.obs["agent_id"].astype(str)
    values = cp10k(np.expm1(panel_matrix(source, genes)))
    return {
        str(agent): values[index].copy()
        for index, agent in enumerate(source.obs["agent_id"])
    }


def stable_rank(seed: int, entity: str) -> int:
    return int(hashlib.sha256(f"{seed}|{entity}".encode()).hexdigest()[:16], 16)


def spatial_quota_sample(table: pd.DataFrame, per_type: int, seed: int) -> pd.DataFrame:
    selected = []
    for cell_type in TYPES:
        local = table[table["cell_type"].eq(cell_type)].copy()
        if len(local) < per_type:
            raise RuntimeError(f"{cell_type}: {len(local)} < required {per_type}")
        counts = local.groupby("grid_id").size().sort_index()
        exact = counts * (per_type / counts.sum())
        quota = np.floor(exact).astype(int)
        remainder = per_type - int(quota.sum())
        order = sorted(counts.index, key=lambda key: (-(exact[key] - quota[key]), key))
        for key in order[:remainder]:
            quota[key] += 1
        pieces = []
        for grid_id, number in quota.items():
            if number <= 0:
                continue
            group = local[local["grid_id"].eq(grid_id)].copy()
            group["_rank"] = [stable_rank(seed, value) for value in group["spot_id"]]
            pieces.append(group.sort_values(["_rank", "spot_id"]).head(number))
        combined = pd.concat(pieces, ignore_index=True)
        if len(combined) != per_type:
            raise RuntimeError(f"Quota sampler failed for {cell_type}")
        selected.append(combined)
    result = pd.concat(selected, ignore_index=True)
    return result.sort_values(["cell_type", "grid_id", "spot_id"]).reset_index(drop=True)


def build_d7_base() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    genes, _ = genes_and_pairs()
    registry = pd.read_csv(REGISTRY)
    source = ad.read_h5ad(D7)
    source = source[registry["source_spot_id"].astype(str).tolist()].copy()
    observed = cp10k(panel_matrix(source, genes))
    targets = parent_targets(genes)
    parent = np.vstack([targets[str(value)] for value in registry["parent_agent_id"]])
    coordinates = registry[["x_pixel", "y_pixel"]].to_numpy(np.float64)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    gx = np.minimum(((x - x.min()) / (x.max() - x.min() + EPS) * 6).astype(int), 5)
    gy = np.minimum(((y - y.min()) / (y.max() - y.min() + EPS) * 6).astype(int), 5)
    table = pd.DataFrame(
        {
            "spot_id": registry["source_spot_id"].astype(str),
            "worker_id": registry["worker_id"].astype(str),
            "parent_agent_id": registry["parent_agent_id"].astype(str),
            "cell_type": registry["cell_type"].astype(str),
            "x_pixel": x,
            "y_pixel": y,
            "grid_x": gx,
            "grid_y": gy,
            "grid_id": gx * 6 + gy,
            "spatial_fold": (gx + 2 * gy) % 3,
            "row_index": np.arange(len(registry)),
        }
    )
    if len(table) != 3410 or set(table["cell_type"]) != set(TYPES):
        raise RuntimeError("d7 base is not the frozen 3,410-spot nine-type input")
    return table, observed, parent, genes


def subset_h5ad(
    meta: pd.DataFrame,
    matrix: np.ndarray,
    genes: list[str],
    method: str,
    stage: str,
) -> ad.AnnData:
    rows = meta["row_index"].to_numpy(int)
    obs = pd.DataFrame(
        {
            "cell_type": meta["cell_type"].to_numpy(),
            "method": method,
            "source_spot_id": meta["spot_id"].to_numpy(),
            "spatial_fold": meta["spatial_fold"].to_numpy(int),
        },
        index=pd.Index(meta["spot_id"], name="entity_id"),
    )
    return make_h5ad(
        matrix[rows],
        obs,
        meta[["x_pixel", "y_pixel"]].to_numpy(float),
        genes,
        {"method": method, "stage": stage, "GSE267904_d21_used": False},
    )


def prepare_d7() -> None:
    freeze()
    table, observed, parent, genes = build_d7_base()
    table.to_csv(OUT / "01_d7_cv/spatial_fold_assignments.csv", index=False)
    counts = table.groupby(["cell_type", "spatial_fold"]).size().unstack()
    if counts.min().min() < CV_VALIDATION_N:
        raise RuntimeError("At least one spatial fold lacks 45 entities per type")
    manifest = []
    for fold in CV_FOLDS:
        training = spatial_quota_sample(
            table[~table["spatial_fold"].eq(fold)], CV_TRAIN_N, CV_SEEDS[fold]
        )
        validation = spatial_quota_sample(
            table[table["spatial_fold"].eq(fold)], CV_VALIDATION_N, CV_SEEDS[fold]
        )
        for split, meta in [("train", training), ("validation", validation)]:
            meta.to_csv(OUT / f"01_d7_cv/fold_{fold}_{split}_spots.csv", index=False)
            for method, matrix in [("observed_d7", observed), ("parent_repeat", parent)]:
                obj = subset_h5ad(meta, matrix, genes, method, f"d7_cv_{split}")
                path = OUT / f"01_d7_cv/inputs/fold_{fold}/{split}/{method}.h5ad"
                path.parent.mkdir(parents=True, exist_ok=True)
                obj.write_h5ad(path)
                manifest.append(
                    {
                        "fold": fold,
                        "split": split,
                        "method": method,
                        "n": obj.n_obs,
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                )
    refit = spatial_quota_sample(table, CV_TRAIN_N, REFIT_SEED)
    refit.to_csv(OUT / "02_refit/refit_spots.csv", index=False)
    for method, matrix in [("observed_d7", observed), ("parent_repeat", parent)]:
        obj = subset_h5ad(refit, matrix, genes, method, "d7_refit")
        path = OUT / f"02_refit/inputs/{method}.h5ad"
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.write_h5ad(path)
    pd.DataFrame(manifest).to_csv(OUT / "01_d7_cv/input_manifest.csv", index=False)
    write_json(
        OUT / "07_audits/d7_structure_audit.json",
        {
            "status": "PASS_D7_CV_INPUTS_PREPARED",
            "spots": len(table),
            "genes": len(genes),
            "minimum_fold_cell_type_count": int(counts.min().min()),
            "GSE267904_d21_opened": False,
        },
    )
    print("PASS_D7_CV_INPUTS_PREPARED")


def commot_database(pairs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": pairs["ligand_mouse"].astype(str),
            "receptor": pairs["receptor_mouse"].astype(str),
            "pathway": pairs["pathway"].astype(str),
        }
    )


def run_commot(
    source: ad.AnnData,
    method: str,
    seed: int,
    mode: str,
    pairs: pd.DataFrame,
    output: Path,
    input_path: Path,
    stage: str,
) -> dict:
    import commot as ct

    version = importlib.metadata.version("commot")
    if version != "0.0.3":
        raise RuntimeError(f"Expected COMMOT 0.0.3, got {version}")
    pair_count = len(pairs)
    input_hash = sha256_file(input_path)
    audit_path = output.parent / "run_audit.json"
    if output.is_file() and audit_path.is_file():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            old.get("status") == "PASS_COMMOT"
            and old.get("input_sha256") == input_hash
            and old.get("pair_count_single_call") == pair_count
            and old.get("edges_sha256") == sha256_file(output)
        ):
            return old
    source = source.copy()
    coordinates = np.asarray(source.obsm["spatial"], np.float64)
    if mode == "median_nn":
        coordinates = median_nn_coords(coordinates)
        source.obsm["spatial"] = coordinates
    elif mode != "affine_pixel":
        raise ValueError(mode)
    source.obsp["spatial_distance"] = no_self_distance(coordinates)
    started = time.time()
    ct.tl.spatial_communication(
        source,
        database_name="cellchat",
        df_ligrec=commot_database(pairs),
        pathway_sum=True,
        heteromeric=True,
        heteromeric_rule=COMMOT_CONFIG["heteromeric_rule"],
        dis_thr=COMMOT_CONFIG["dis_thr"],
        cost_type=COMMOT_CONFIG["cost_type"],
        cot_eps_p=COMMOT_CONFIG["cot_eps_p"],
        cot_eps_mu=COMMOT_CONFIG["cot_eps_mu"],
        cot_eps_nu=COMMOT_CONFIG["cot_eps_nu"],
        cot_rho=COMMOT_CONFIG["cot_rho"],
        cot_nitermax=COMMOT_CONFIG["cot_nitermax"],
        cot_weights=tuple(COMMOT_CONFIG["cot_weights"]),
        smooth=COMMOT_CONFIG["smooth"],
    )
    expected = [TOTAL_KEY] + [
        f"commot-cellchat-{family}" for family in sorted(pairs["pathway"].unique())
    ]
    labels = source.obs["cell_type"].astype(str).to_numpy()
    ids = source.obs_names.astype(str).to_numpy()
    rows = []
    self_weight = 0.0
    for key in expected:
        if key not in source.obsp:
            raise RuntimeError(f"COMMOT missing {key}")
        matrix = sparse.coo_matrix(source.obsp[key])
        for sender, receiver, value in zip(matrix.row, matrix.col, matrix.data):
            if not value:
                continue
            if sender == receiver:
                self_weight += float(value)
                continue
            rows.append(
                {
                    "geometry_mode": mode,
                    "method": method,
                    "sampling_seed": seed,
                    "sender_entity": ids[sender],
                    "receiver_entity": ids[receiver],
                    "sender": labels[sender],
                    "receiver": labels[receiver],
                    "commot_key": key,
                    "weight": float(value),
                }
            )
    if self_weight > TOL or not rows:
        raise RuntimeError(f"Invalid COMMOT result: self={self_weight}, rows={len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    audit = {
        "status": "PASS_COMMOT",
        "stage": stage,
        "method": method,
        "sampling_seed": seed,
        "geometry_mode": mode,
        "input": str(input_path),
        "input_sha256": input_hash,
        "commot_version": version,
        "pair_count_single_call": pair_count,
        "pair_blocks": 1,
        "retained_entity_self_weight": self_weight,
        "retained_edge_rows": len(rows),
        "elapsed_seconds": float(time.time() - started),
        "edges_sha256": sha256_file(output),
        "pair_panel_sha256": sha256_file(PAIRS),
        "commot_parameters": COMMOT_CONFIG,
    }
    write_json(audit_path, audit)
    return audit


def aggregate_entity_edges(path: Path) -> pd.DataFrame:
    values = pd.read_csv(path)
    return (
        values.groupby(["commot_key", "sender", "receiver"], sort=False)["weight"]
        .sum()
        .reset_index()
    )


def full_network(edges: pd.DataFrame, method: str, seed: int, key: str) -> pd.Series:
    index = pd.MultiIndex.from_product([TYPES, TYPES], names=["sender", "receiver"])
    chosen = edges[
        edges["method"].eq(method)
        & edges["sampling_seed"].eq(seed)
        & edges["commot_key"].eq(key)
    ]
    return (
        chosen.groupby(["sender", "receiver"])["weight"]
        .sum()
        .reindex(index, fill_value=0.0)
        .astype(float)
    )


def smoke() -> None:
    freeze()
    genes, pairs = genes_and_pairs()
    pieces = [pairs[pairs["pathway"].eq(family)].head(2) for family in FAMILIES]
    smoke_pairs = pd.concat(pieces, ignore_index=True)
    source = ad.read_h5ad(OUT / "02_refit/inputs/parent_repeat.h5ad")
    chosen = []
    for cell_type in TYPES:
        chosen.extend(
            source.obs.index[source.obs["cell_type"].astype(str).eq(cell_type)][:10]
        )
    source = source[chosen].copy()
    path = OUT / "01_d7_cv/inputs/smoke_10_per_type.h5ad"
    source.write_h5ad(path)
    output = OUT / "01_d7_cv/commot/smoke/edges_long.csv"
    audit = run_commot(
        source, "parent_repeat_smoke", SMOKE_SEED, "affine_pixel", smoke_pairs,
        output, path, "d7_interface_smoke",
    )
    if audit["pair_count_single_call"] != 10:
        raise RuntimeError("Smoke did not use exactly ten LR pairs")
    write_json(
        OUT / "07_audits/smoke_audit.json",
        {"status": "PASS_D7_90_ENTITY_10_PAIR_SMOKE", **audit},
    )
    print("PASS_D7_90_ENTITY_10_PAIR_SMOKE")


def cv_base_output(fold: int, split: str, method: str) -> Path:
    return OUT / f"01_d7_cv/commot/fold_{fold}/{split}/{method}/edges_long.csv"


def run_cv_base() -> None:
    freeze()
    require_marker(OUT / "07_audits/smoke_audit.json", "smoke audit")
    _, pairs = genes_and_pairs()
    for fold in CV_FOLDS:
        for split in ["train", "validation"]:
            for method in ["observed_d7", "parent_repeat"]:
                path = OUT / f"01_d7_cv/inputs/fold_{fold}/{split}/{method}.h5ad"
                run_commot(
                    ad.read_h5ad(path), method, CV_SEEDS[fold], "affine_pixel", pairs,
                    cv_base_output(fold, split, method), path, f"d7_cv_{split}_base",
                )
    print("PASS_D7_CV_BASE_COMMOT")


def role_parameters(observed_path: Path, parent_path: Path) -> pd.DataFrame:
    observed = aggregate_entity_edges(observed_path)
    parent = aggregate_entity_edges(parent_path)
    rows = []
    observed_flows = []
    parent_flows = []
    for family in FAMILIES:
        key = f"commot-cellchat-{family}"
        observed_flows.append(observed.loc[observed["commot_key"].eq(key), "weight"].sum())
        parent_flows.append(parent.loc[parent["commot_key"].eq(key), "weight"].sum())
    observed_share = np.asarray(observed_flows) / max(sum(observed_flows), EPS)
    parent_share = np.asarray(parent_flows) / max(sum(parent_flows), EPS)
    for family_index, family in enumerate(FAMILIES):
        key = f"commot-cellchat-{family}"
        oo = observed[observed["commot_key"].eq(key)]
        pp = parent[parent["commot_key"].eq(key)]
        global_log_ratio = float(
            np.log((observed_share[family_index] + ROLE_EPS) / (parent_share[family_index] + ROLE_EPS))
        )
        for role, level in [("sender", "sender"), ("receiver", "receiver")]:
            orole = oo.groupby(level)["weight"].sum().reindex(TYPES, fill_value=0.0)
            prole = pp.groupby(level)["weight"].sum().reindex(TYPES, fill_value=0.0)
            orole = orole / max(float(orole.sum()), EPS)
            prole = prole / max(float(prole.sum()), EPS)
            for cell_type in TYPES:
                role_log_ratio = float(
                    np.log((orole[cell_type] + ROLE_EPS) / (prole[cell_type] + ROLE_EPS))
                )
                rows.append(
                    {
                        "pathway": family,
                        "cell_type": cell_type,
                        "role": role,
                        "observed_pathway_share": observed_share[family_index],
                        "parent_pathway_share": parent_share[family_index],
                        "global_log_ratio": global_log_ratio,
                        "observed_role_share": orole[cell_type],
                        "parent_role_share": prole[cell_type],
                        "role_log_ratio": role_log_ratio,
                        "base_log_correction": 0.5 * global_log_ratio + role_log_ratio,
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 90 or not np.isfinite(result.select_dtypes(float)).all().all():
        raise RuntimeError("Role-parameter table is invalid")
    return result


def gene_memberships(pairs: pd.DataFrame) -> tuple[dict[str, set[tuple[str, str]]], list[str]]:
    membership: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in pairs.itertuples():
        for gene in str(row.ligand_mouse).split("_"):
            membership[gene].add((str(row.pathway), "sender"))
        for gene in str(row.receptor_mouse).split("_"):
            membership[gene].add((str(row.pathway), "receiver"))
    genes = sorted(membership)
    if len(genes) != 54:
        raise RuntimeError(f"Expected 54 LR component genes, got {len(genes)}")
    return membership, genes


def gene_multipliers(
    parameters: pd.DataFrame, genes: list[str], pairs: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    membership, lr_genes = gene_memberships(pairs)
    lookup = {
        (row.cell_type, row.pathway, row.role): float(row.base_log_correction)
        for row in parameters.itertuples()
    }
    rows = []
    for cell_type in TYPES:
        for gene in lr_genes:
            contributions = [
                lookup[(cell_type, pathway, role)]
                for pathway, role in sorted(membership[gene])
            ]
            raw = float(np.mean(contributions))
            multiplier = float(
                np.clip(np.exp(alpha * raw), MULTIPLIER_MIN, MULTIPLIER_MAX)
            )
            rows.append(
                {
                    "cell_type": cell_type,
                    "gene": gene,
                    "alpha": alpha,
                    "membership_count": len(contributions),
                    "mean_base_log_correction": raw,
                    "multiplier": multiplier,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 9 * 54:
        raise RuntimeError("Gene-multiplier table is incomplete")
    return result


def apply_multipliers(
    source: ad.AnnData,
    multipliers: pd.DataFrame,
    method: str,
    provenance: dict,
) -> tuple[ad.AnnData, dict]:
    genes = source.var_names.astype(str).tolist()
    index = {gene: position for position, gene in enumerate(genes)}
    values = cp10k(np.expm1(dense(source.X)))
    original = values.copy()
    lr_genes = sorted(multipliers["gene"].unique())
    lr_index = np.asarray([index[gene] for gene in lr_genes], int)
    labels = source.obs["cell_type"].astype(str).to_numpy()
    for cell_type in TYPES:
        row = multipliers[multipliers["cell_type"].eq(cell_type)].set_index("gene")
        factor = row.loc[lr_genes, "multiplier"].to_numpy(float)
        chosen = np.flatnonzero(labels == cell_type)
        values[np.ix_(chosen, lr_index)] *= factor[None, :]
    values = cp10k(values)
    non_lr = np.ones(len(genes), bool)
    non_lr[lr_index] = False
    old_non_lr = original[:, non_lr]
    new_non_lr = values[:, non_lr]
    old_share = old_non_lr / np.maximum(old_non_lr.sum(axis=1, keepdims=True), EPS)
    new_share = new_non_lr / np.maximum(new_non_lr.sum(axis=1, keepdims=True), EPS)
    non_lr_error = float(np.max(np.abs(old_share - new_share)))
    row_error = float(np.max(np.abs(values.sum(axis=1) - 10000.0)))
    obs = source.obs.copy()
    obs["method"] = method
    result = make_h5ad(values, obs, source.obsm["spatial"], genes, provenance)
    audit = {
        "rows": result.n_obs,
        "genes": result.n_vars,
        "lr_genes_directly_modified": len(lr_genes),
        "non_lr_genes_directly_modified": 0,
        "non_lr_within_group_share_max_abs_error": non_lr_error,
        "row_total_max_abs_error": row_error,
        "finite": bool(np.isfinite(values).all()),
        "nonnegative": bool((values >= 0).all()),
    }
    return result, audit


def prepare_candidates() -> None:
    freeze()
    _, pairs = genes_and_pairs()
    audits = []
    for fold in CV_FOLDS:
        parameters = role_parameters(
            cv_base_output(fold, "train", "observed_d7"),
            cv_base_output(fold, "train", "parent_repeat"),
        )
        parameters.to_csv(
            OUT / f"01_d7_cv/factors/fold_{fold}_role_parameters.csv", index=False
        )
        parent_path = OUT / f"01_d7_cv/inputs/fold_{fold}/validation/parent_repeat.h5ad"
        parent = ad.read_h5ad(parent_path)
        for alpha in ALPHAS[1:]:
            multipliers = gene_multipliers(
                parameters, parent.var_names.astype(str).tolist(), pairs, alpha
            )
            multipliers.to_csv(
                OUT / f"01_d7_cv/factors/fold_{fold}_alpha_{alpha:.2f}_gene_multipliers.csv",
                index=False,
            )
            method = f"lrra_alpha_{alpha:.2f}"
            adapted, audit = apply_multipliers(
                parent,
                multipliers,
                method,
                {
                    "method": METHOD_NAME,
                    "stage": "d7_spatial_cv_validation",
                    "fold": fold,
                    "alpha": alpha,
                    "GSE267904_d21_used": False,
                },
            )
            path = OUT / f"01_d7_cv/inputs/fold_{fold}/validation/{method}.h5ad"
            adapted.write_h5ad(path)
            audits.append({"fold": fold, "alpha": alpha, "path": str(path), **audit})
    audit_table = pd.DataFrame(audits)
    audit_table.to_csv(OUT / "01_d7_cv/candidate_expression_audit.csv", index=False)
    if (
        not audit_table["finite"].all()
        or not audit_table["nonnegative"].all()
        or audit_table["row_total_max_abs_error"].max() > 1e-10
        or audit_table["non_lr_within_group_share_max_abs_error"].max() > 1e-12
    ):
        raise RuntimeError("Candidate expression structural audit failed")
    print("PASS_D7_CV_CANDIDATES_PREPARED")


def cv_candidate_output(fold: int, alpha: float) -> Path:
    method = "parent_repeat" if alpha == 0 else f"lrra_alpha_{alpha:.2f}"
    return OUT / f"01_d7_cv/commot/fold_{fold}/validation/{method}/edges_long.csv"


def run_cv_candidates() -> None:
    freeze()
    _, pairs = genes_and_pairs()
    for fold in CV_FOLDS:
        for alpha in ALPHAS[1:]:
            method = f"lrra_alpha_{alpha:.2f}"
            path = OUT / f"01_d7_cv/inputs/fold_{fold}/validation/{method}.h5ad"
            run_commot(
                ad.read_h5ad(path), method, CV_SEEDS[fold], "affine_pixel", pairs,
                cv_candidate_output(fold, alpha), path, "d7_spatial_cv_validation",
            )
    print("PASS_D7_CV_CANDIDATE_COMMOT")


def normalize(values: pd.Series) -> pd.Series:
    return values / max(float(values.sum()), EPS)


def safe_spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(spearmanr(x, y).statistic)


def positive_top(values: pd.Series, k: int = 10) -> set[str]:
    table = values.reset_index(name="weight")
    table["edge"] = table["sender"].astype(str) + "|" + table["receiver"].astype(str)
    table = table[table["weight"] > 0].sort_values(
        ["weight", "edge"], ascending=[False, True], kind="mergesort"
    )
    return set(table.head(min(k, len(table)))["edge"])


def network_metrics(real_raw: pd.Series, predicted_raw: pd.Series) -> tuple[float, float, dict[str, float]]:
    real = normalize(real_raw)
    predicted = normalize(predicted_raw)
    rho = safe_spearman(real, predicted)
    real_top = positive_top(real)
    predicted_top = positive_top(predicted)
    jaccard = len(real_top & predicted_top) / max(len(real_top | predicted_top), 1)
    local = {}
    for cell_type in TYPES:
        positions = [
            index
            for index, (sender, receiver) in enumerate(real.index)
            if sender == cell_type or receiver == cell_type
        ]
        local[cell_type] = float(
            np.mean(np.abs(predicted.iloc[positions] - real.iloc[positions]))
        )
    return float(np.nanmean([(rho + 1) / 2, jaccard])), float(np.mean(list(local.values()))), local


def evaluate_cv() -> None:
    freeze()
    rows = []
    local_rows = []
    for fold in CV_FOLDS:
        seed = CV_SEEDS[fold]
        observed = aggregate_entity_edges(
            cv_base_output(fold, "validation", "observed_d7")
        )
        observed.insert(0, "sampling_seed", seed)
        observed.insert(0, "method", "observed_d7")
        candidates = [observed]
        for alpha in ALPHAS:
            method = "parent_repeat" if alpha == 0 else f"lrra_alpha_{alpha:.2f}"
            values = aggregate_entity_edges(cv_candidate_output(fold, alpha))
            values.insert(0, "sampling_seed", seed)
            values.insert(0, "method", method)
            candidates.append(values)
        edges = pd.concat(candidates, ignore_index=True)
        real = full_network(edges, "observed_d7", seed, TOTAL_KEY)
        for alpha in ALPHAS:
            method = "parent_repeat" if alpha == 0 else f"lrra_alpha_{alpha:.2f}"
            predicted = full_network(edges, method, seed, TOTAL_KEY)
            gnrs, local, cell_types = network_metrics(real, predicted)
            rows.append(
                {
                    "fold": fold,
                    "sampling_seed": seed,
                    "alpha": alpha,
                    "method": method,
                    "GNRS": gnrs,
                    "cell_type_local_MAE": local,
                }
            )
            for cell_type, value in cell_types.items():
                local_rows.append(
                    {
                        "fold": fold,
                        "alpha": alpha,
                        "method": method,
                        "cell_type": cell_type,
                        "local_MAE": value,
                    }
                )
    metrics = pd.DataFrame(rows)
    cell_metrics = pd.DataFrame(local_rows)
    metrics.to_csv(OUT / "01_d7_cv/cv_fold_metrics.csv", index=False)
    cell_metrics.to_csv(OUT / "01_d7_cv/cv_cell_type_metrics.csv", index=False)
    summary = (
        metrics.groupby("alpha")[["GNRS", "cell_type_local_MAE"]]
        .agg(["mean", "median", "min", "max"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index().sort_values(
        ["cell_type_local_MAE_mean", "alpha"], kind="mergesort"
    )
    summary.to_csv(OUT / "01_d7_cv/cv_alpha_summary.csv", index=False)
    selected_alpha = float(summary.iloc[0]["alpha"])
    baseline = metrics[metrics["alpha"].eq(0)].set_index("fold")
    selected = metrics[metrics["alpha"].eq(selected_alpha)].set_index("fold")
    fold_improvements = selected["cell_type_local_MAE"] < baseline["cell_type_local_MAE"] - TOL
    reduction = 1.0 - float(selected["cell_type_local_MAE"].median()) / float(
        baseline["cell_type_local_MAE"].median()
    )
    local_summary = (
        cell_metrics[cell_metrics["alpha"].isin([0.0, selected_alpha])]
        .groupby(["alpha", "cell_type"])["local_MAE"]
        .median()
        .unstack("alpha")
    )
    type_ratio = local_summary[selected_alpha] / local_summary[0.0]
    type_wins = int((type_ratio < 1.0 - TOL).sum())
    max_worsening = float((type_ratio - 1.0).max())
    structural = pd.read_csv(OUT / "01_d7_cv/candidate_expression_audit.csv")
    checks = {
        "alpha_nonzero": selected_alpha > 0,
        "all_three_folds_local_improved": bool(fold_improvements.all()),
        "median_local_reduction_at_least_10pct": reduction >= 0.10 - TOL,
        "median_GNRS_not_lower": float(selected["GNRS"].median())
        >= float(baseline["GNRS"].median()) - TOL,
        "at_least_6_cell_types_improve": type_wins >= 6,
        "no_cell_type_worse_than_5pct": max_worsening <= 0.05 + TOL,
        "row_total_error": float(structural["row_total_max_abs_error"].max()) <= 1e-10,
        "non_lr_not_directly_changed": bool((structural["non_lr_genes_directly_modified"] == 0).all()),
        "GSE267904_d21_not_opened": True,
    }
    passed = all(checks.values())
    status = "PASS_D7_LRRA_CV_GATE" if passed else "BLOCKED_D7_LRRA_CV_GATE"
    audit = {
        "status": status,
        "selected_alpha": selected_alpha,
        "checks": checks,
        "baseline_median_local_MAE": float(baseline["cell_type_local_MAE"].median()),
        "selected_median_local_MAE": float(selected["cell_type_local_MAE"].median()),
        "median_local_reduction_fraction": reduction,
        "baseline_median_GNRS": float(baseline["GNRS"].median()),
        "selected_median_GNRS": float(selected["GNRS"].median()),
        "fold_local_improvements": {str(key): bool(value) for key, value in fold_improvements.items()},
        "cell_type_wins": type_wins,
        "maximum_cell_type_worsening_fraction": max_worsening,
        "GSE267904_d21_opened": False,
    }
    write_json(OUT / "01_d7_cv/d7_cv_gate.json", audit)
    if passed:
        CV_PASS.write_text(status + "\n", encoding="utf-8")
    else:
        no_figure(status, "The frozen three-fold d7 LR-role calibration gate failed.")
        write_json(
            OUT / "07_audits/final_validation.json",
            {
                "status": status,
                "stopped_at": "d7_spatial_cross_validation",
                "GSE267904_d21_opened": False,
                "commot_jobs_late": 0,
                "figure_generated": False,
                "fibrosis_application_v3": "paused_unchanged",
            },
        )
    print(json.dumps(audit, indent=2))


def refit_freeze() -> None:
    freeze()
    require_marker(CV_PASS, "d7 cross-validation")
    _, pairs = genes_and_pairs()
    for method in ["observed_d7", "parent_repeat"]:
        path = OUT / f"02_refit/inputs/{method}.h5ad"
        output = OUT / f"02_refit/commot/{method}/edges_long.csv"
        run_commot(
            ad.read_h5ad(path), method, REFIT_SEED, "affine_pixel", pairs,
            output, path, "d7_full_refit",
        )
    parameters = role_parameters(
        OUT / "02_refit/commot/observed_d7/edges_long.csv",
        OUT / "02_refit/commot/parent_repeat/edges_long.csv",
    )
    selected_alpha = float(
        json.loads((OUT / "01_d7_cv/d7_cv_gate.json").read_text())["selected_alpha"]
    )
    genes, pairs = genes_and_pairs()
    multipliers = gene_multipliers(parameters, genes, pairs, selected_alpha)
    parameters.to_csv(OUT / "02_refit/frozen/final_role_parameters_90.csv", index=False)
    multipliers.to_csv(OUT / "02_refit/frozen/final_gene_multipliers_486.csv", index=False)
    manifest = {
        "status": "PASS_LRRA_FACTORS_FROZEN",
        "selected_alpha": selected_alpha,
        "role_parameters": 90,
        "gene_multipliers": 486,
        "lr_genes": 54,
        "GSE267904_d21_opened": False,
        "sha256": {
            "role_parameters": sha256_file(OUT / "02_refit/frozen/final_role_parameters_90.csv"),
            "gene_multipliers": sha256_file(OUT / "02_refit/frozen/final_gene_multipliers_486.csv"),
        },
    }
    write_json(OUT / "02_refit/frozen/factor_manifest.json", manifest)
    FACTOR_PASS.write_text("PASS_LRRA_FACTORS_FROZEN\n", encoding="utf-8")
    write_json(
        OUT / "07_audits/d21_access_audit.json",
        {
            "status": "PARAMETERS_FROZEN_D21_EVALUATION_NOW_PERMITTED",
            "GSE267904_d21_opened": False,
            "factor_manifest_sha256": sha256_file(OUT / "02_refit/frozen/factor_manifest.json"),
        },
    )
    print(json.dumps(manifest, indent=2))


def prepare_late() -> None:
    freeze()
    require_marker(FACTOR_PASS, "frozen LRRA factors")
    multipliers = pd.read_csv(OUT / "02_refit/frozen/final_gene_multipliers_486.csv")
    audits = []
    full_objects = {}
    for worker_seed in WORKER_SEEDS:
        historical_path = old_worker_full(worker_seed)
        historical = ad.read_h5ad(historical_path)
        adapted, audit = apply_multipliers(
            historical,
            multipliers,
            METHOD_ID,
            {
                "method": METHOD_NAME,
                "stage": "late_locked_evaluation_input",
                "factor_manifest": str(OUT / "02_refit/frozen/factor_manifest.json"),
                "GSE267904_d21_used_for_construction": False,
            },
        )
        path = late_full_path(worker_seed)
        adapted.write_h5ad(path)
        full_objects[worker_seed] = adapted
        audits.append(
            {
                "worker_seed": worker_seed,
                "source": str(historical_path),
                "source_sha256": sha256_file(historical_path),
                "output": str(path),
                "output_sha256": sha256_file(path),
                **audit,
            }
        )
    sample_rows = []
    for seed in SAMPLE_SEEDS:
        worker_seed = WORKER_SEED_FOR_SAMPLE[seed]
        historical = ad.read_h5ad(old_worker_sample(seed))
        ids = historical.obs_names.astype(str).tolist()
        sample = full_objects[worker_seed][ids].copy()
        coordinate_difference = float(
            np.max(
                np.abs(
                    np.asarray(sample.obsm["spatial"], float)
                    - np.asarray(historical.obsm["spatial"], float)
                )
            )
        )
        if sample.obs_names.tolist() != historical.obs_names.tolist() or coordinate_difference != 0:
            raise RuntimeError("Late sampled worker alignment changed")
        path = late_sample_path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        sample.write_h5ad(path)
        sample_rows.append(
            {
                "sampling_seed": seed,
                "worker_seed": worker_seed,
                "n_entities": sample.n_obs,
                "ids_and_order_exact": True,
                "coordinate_max_abs_difference": coordinate_difference,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    audit_table = pd.DataFrame(audits)
    sampling_table = pd.DataFrame(sample_rows)
    audit_table.to_csv(OUT / "03_late_expression/full_expression_audit.csv", index=False)
    sampling_table.to_csv(OUT / "03_late_expression/sampling_alignment.csv", index=False)
    passed = bool(
        audit_table["finite"].all()
        and audit_table["nonnegative"].all()
        and audit_table["row_total_max_abs_error"].max() <= 1e-10
        and audit_table["non_lr_within_group_share_max_abs_error"].max() <= 1e-12
        and sampling_table["ids_and_order_exact"].all()
        and sampling_table["coordinate_max_abs_difference"].max() == 0
        and (sampling_table["n_entities"] == 810).all()
    )
    status = "PASS_LATE_EXPRESSION_FROZEN" if passed else "BLOCKED_LATE_EXPRESSION_STRUCTURE"
    manifest = {
        "status": status,
        "full_worker_realizations": 3,
        "sampled_inputs": 10,
        "llm_calls": 0,
        "physicell_runs": 0,
        "GSE267904_observed_d21_used_for_construction": False,
        "factor_manifest_sha256": sha256_file(OUT / "02_refit/frozen/factor_manifest.json"),
    }
    write_json(OUT / "03_late_expression/frozen_late_expression_manifest.json", manifest)
    if not passed:
        no_figure(status, "Late LRRA expression structure or historical alignment failed.")
        raise RuntimeError(status)
    LATE_PASS.write_text(status + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def run_commot_mode(mode: str) -> None:
    freeze()
    require_marker(LATE_PASS, "late expression")
    if mode == "median_nn":
        require_marker(AFFINE_PASS, "affine metrics")
    _, pairs = genes_and_pairs()
    rows = []
    for seed in SAMPLE_SEEDS:
        path = late_sample_path(seed)
        output = formal_output(seed, mode)
        audit = run_commot(
            ad.read_h5ad(path), METHOD_ID, seed, mode, pairs, output, path,
            f"late_locked_{mode}",
        )
        rows.append(
            {
                "geometry_mode": mode,
                "sampling_seed": seed,
                "status": audit["status"],
                "pair_count_single_call": audit["pair_count_single_call"],
                "self_weight": audit["retained_entity_self_weight"],
                "path": str(output),
                "sha256": sha256_file(output),
            }
        )
    inventory_path = OUT / "07_audits/formal_commot_inventory.csv"
    new = pd.DataFrame(rows)
    if inventory_path.is_file():
        old = pd.read_csv(inventory_path)
        old = old[~old["geometry_mode"].eq(mode)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(inventory_path, index=False)
    print(f"PASS_{mode.upper()}_10_COMMOT_TASKS")


def js_similarity(x, y) -> float:
    return float(1.0 - jensenshannon(np.asarray(x) + EPS, np.asarray(y) + EPS) ** 2)


def evaluate_new_mode(
    edges: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, pathway_rows, role_rows, edge_rows, local_rows = [], [], [], [], []
    mode_edges = edges[edges["geometry_mode"].eq(mode)]
    for seed in SAMPLE_SEEDS:
        real_raw = full_network(mode_edges, "observed", seed, TOTAL_KEY)
        predicted_raw = full_network(mode_edges, METHOD_ID, seed, TOTAL_KEY)
        real, predicted = normalize(real_raw), normalize(predicted_raw)
        informative = (real_raw > 0) | (predicted_raw > 0)
        real_top, predicted_top = positive_top(real), positive_top(predicted)
        metrics = {
            "edge_spearman_81": safe_spearman(real, predicted),
            "informative_edge_spearman": safe_spearman(real[informative], predicted[informative]),
            "positive_top10_jaccard": len(real_top & predicted_top) / max(len(real_top | predicted_top), 1),
            "positive_top10_real_effective_k": float(min(10, (real > 0).sum())),
            "positive_top10_method_effective_k": float(min(10, (predicted > 0).sum())),
            "same_type_share": float(sum(predicted.loc[(cell_type, cell_type)] for cell_type in TYPES)),
            "cross_type_positive_edge_coverage": float(
                np.mean([
                    predicted_raw.loc[(sender, receiver)] > 0
                    for sender in TYPES for receiver in TYPES if sender != receiver
                ])
            ),
        }
        observed_flows, predicted_flows, roles = [], [], []
        for family in FAMILIES:
            key = f"commot-cellchat-{family}"
            observed = full_network(mode_edges, "observed", seed, key)
            pred = full_network(mode_edges, METHOD_ID, seed, key)
            observed_flows.append(float(observed.sum()))
            predicted_flows.append(float(pred.sum()))
            for role, level in [("sender", 0), ("receiver", 1)]:
                oo = observed.groupby(level=level).sum().reindex(TYPES, fill_value=0)
                pp = pred.groupby(level=level).sum().reindex(TYPES, fill_value=0)
                rho = safe_spearman(oo, pp)
                roles.append(rho)
                role_rows.append(
                    {
                        "geometry_mode": mode,
                        "sampling_seed": seed,
                        "method": METHOD_ID,
                        "pathway": family,
                        "role": role,
                        "spearman": rho,
                    }
                )
        observed_flow = np.asarray(observed_flows) / max(sum(observed_flows), EPS)
        predicted_flow = np.asarray(predicted_flows) / max(sum(predicted_flows), EPS)
        metrics["five_pathway_flow_js_similarity"] = js_similarity(observed_flow, predicted_flow)
        metrics["sender_receiver_role_mean_spearman"] = float(np.nanmean(roles))
        metrics["pathway_allocation_mean_absolute_error"] = float(
            np.mean(np.abs(predicted_flow - observed_flow))
        )
        local = []
        for cell_type in TYPES:
            positions = [
                index for index, (sender, receiver) in enumerate(real.index)
                if sender == cell_type or receiver == cell_type
            ]
            value = float(np.mean(np.abs(predicted.iloc[positions] - real.iloc[positions])))
            local.append(value)
            local_rows.append(
                {
                    "geometry_mode": mode,
                    "sampling_seed": seed,
                    "method": METHOD_ID,
                    "cell_type": cell_type,
                    "local_mean_absolute_error": value,
                }
            )
        metrics["cell_type_local_mean_absolute_error"] = float(np.mean(local))
        metrics["GNRS_secondary"] = float(
            np.nanmean([(metrics["edge_spearman_81"] + 1) / 2, metrics["positive_top10_jaccard"]])
        )
        metrics["LBSS_secondary"] = float(
            np.nanmean([
                metrics["five_pathway_flow_js_similarity"],
                (metrics["sender_receiver_role_mean_spearman"] + 1) / 2,
            ])
        )
        for metric, value in metrics.items():
            metric_rows.append(
                {
                    "geometry_mode": mode,
                    "sampling_seed": seed,
                    "worker_seed": WORKER_SEED_FOR_SAMPLE[seed],
                    "method": METHOD_ID,
                    "metric": metric,
                    "value": value,
                    "interpretation": "sampling_stability_not_biological_inference",
                }
            )
        for family, observed_value, predicted_value in zip(FAMILIES, observed_flow, predicted_flow):
            pathway_rows.append(
                {
                    "geometry_mode": mode,
                    "sampling_seed": seed,
                    "method": METHOD_ID,
                    "pathway": family,
                    "observed_share": observed_value,
                    "method_share": predicted_value,
                    "absolute_error": abs(predicted_value - observed_value),
                }
            )
        for (sender, receiver), observed_value, predicted_value in zip(real.index, real, predicted):
            edge_rows.append(
                {
                    "geometry_mode": mode,
                    "sampling_seed": seed,
                    "method": METHOD_ID,
                    "sender": sender,
                    "receiver": receiver,
                    "observed_normalized_weight": observed_value,
                    "method_normalized_weight": predicted_value,
                    "absolute_error": abs(predicted_value - observed_value),
                }
            )
    return tuple(pd.DataFrame(value) for value in [metric_rows, pathway_rows, role_rows, edge_rows, local_rows])


def aggregate_formal_mode(mode: str) -> pd.DataFrame:
    rows = []
    for seed in SAMPLE_SEEDS:
        output = formal_output(seed, mode)
        audit = json.loads((output.parent / "run_audit.json").read_text(encoding="utf-8"))
        if (
            audit.get("status") != "PASS_COMMOT"
            or audit.get("pair_count_single_call") != 58
            or audit.get("retained_entity_self_weight", 1) > TOL
            or audit.get("edges_sha256") != sha256_file(output)
        ):
            raise RuntimeError(f"Formal COMMOT audit failed: {output.parent}")
        grouped = aggregate_entity_edges(output)
        grouped.insert(0, "sampling_seed", seed)
        grouped.insert(0, "method", METHOD_ID)
        grouped.insert(0, "geometry_mode", mode)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def evaluate_mode(mode: str) -> None:
    freeze()
    if mode == "affine_pixel":
        require_marker(LATE_PASS, "late expression")
    else:
        require_marker(AFFINE_PASS, "affine metrics")
    historical_edges = pd.read_csv(V2 / "06_metrics/commot_aggregated_edges.csv")
    new_edges = aggregate_formal_mode(mode)
    edge_path = OUT / "05_metrics/commot_aggregated_edges.csv"
    if edge_path.is_file():
        existing = pd.read_csv(edge_path)
        existing_new = existing[
            existing["method"].eq(METHOD_ID) & ~existing["geometry_mode"].eq(mode)
        ]
        combined_edges = pd.concat([historical_edges, existing_new, new_edges], ignore_index=True)
    else:
        combined_edges = pd.concat([historical_edges, new_edges], ignore_index=True)
    combined_edges.to_csv(edge_path, index=False)

    new_tables = evaluate_new_mode(combined_edges, mode)
    names = [
        "seed_level_metrics.csv",
        "pathway_allocation.csv",
        "sender_receiver_roles.csv",
        "edge_level_values.csv",
        "cell_type_local_errors.csv",
    ]
    for name, new in zip(names, new_tables):
        historical = pd.read_csv(V2 / f"06_metrics/{name}")
        output = OUT / f"05_metrics/{name}"
        other_new = pd.DataFrame()
        if output.is_file():
            current = pd.read_csv(output)
            other_new = current[
                current["method"].eq(METHOD_ID) & ~current["geometry_mode"].eq(mode)
            ]
        pd.concat([historical, other_new, new], ignore_index=True).to_csv(output, index=False)

    metrics = pd.read_csv(OUT / "05_metrics/seed_level_metrics.csv")
    local = pd.read_csv(OUT / "05_metrics/cell_type_local_errors.csv")
    chosen = metrics[
        metrics["geometry_mode"].eq(mode)
        & metrics["method"].isin(["worker", METHOD_ID])
    ]
    medians = chosen.groupby(["method", "metric"])["value"].median().unstack("metric")
    new = medians.loc[METHOD_ID]
    old = medians.loc["worker"]
    if mode == "affine_pixel":
        pivot = chosen.pivot_table(
            index=["sampling_seed", "metric"], columns="method", values="value"
        ).reset_index()
        local_metric = pivot[pivot["metric"].eq("cell_type_local_mean_absolute_error")]
        seed_wins = int((local_metric[METHOD_ID] < local_metric["worker"] - TOL).sum())
        local_summary = (
            local[
                local["geometry_mode"].eq(mode)
                & local["method"].isin(["worker", METHOD_ID])
            ]
            .groupby(["method", "cell_type"])["local_mean_absolute_error"]
            .median()
            .unstack("method")
        )
        ratio = local_summary[METHOD_ID] / local_summary["worker"]
        type_wins = int((ratio < 1.0 - TOL).sum())
        maximum_worsening = float((ratio - 1.0).max())
        checks = {
            "local_MAE_top1": bool(
                new["cell_type_local_mean_absolute_error"]
                < SUCCESS_THRESHOLDS["cell_type_local_mean_absolute_error"] - TOL
            ),
            "at_least_6_of_10_seed_local_wins": seed_wins >= 6,
            "at_least_6_of_9_cell_types_improve": type_wins >= 6,
            "no_cell_type_worse_than_5pct": maximum_worsening <= 0.05 + TOL,
            "GNRS_not_worse": bool(new["GNRS_secondary"] >= SUCCESS_THRESHOLDS["GNRS_secondary"] - TOL),
            "LBSS_not_worse": bool(new["LBSS_secondary"] >= SUCCESS_THRESHOLDS["LBSS_secondary"] - TOL),
            "flow_JS_not_worse": bool(
                new["five_pathway_flow_js_similarity"]
                >= SUCCESS_THRESHOLDS["five_pathway_flow_js_similarity"] - TOL
            ),
            "role_not_worse": bool(
                new["sender_receiver_role_mean_spearman"]
                >= SUCCESS_THRESHOLDS["sender_receiver_role_mean_spearman"] - TOL
            ),
            "pathway_MAE_not_worse": bool(
                new["pathway_allocation_mean_absolute_error"]
                <= SUCCESS_THRESHOLDS["pathway_allocation_mean_absolute_error"] + TOL
            ),
        }
        passed = all(checks.values())
        status = "PASS_AFFINE_METRICS_GATE" if passed else "COMPLETED_AFFINE_GATE_FAILED_NO_FIGURE"
        audit = {
            "status": status,
            "checks": checks,
            "AgentVC_LRRA_medians": new.to_dict(),
            "old_AgentVC_medians": old.to_dict(),
            "local_seed_wins": seed_wins,
            "cell_type_wins": type_wins,
            "maximum_cell_type_worsening_fraction": maximum_worsening,
            "locked_d21_evaluation_executions": 1,
            "parameters_changed_after_evaluation": False,
        }
        write_json(OUT / "05_metrics/affine_metrics_gate.json", audit)
        write_json(
            OUT / "07_audits/d21_access_audit.json",
            {
                "status": "LOCKED_D21_AFFINE_EVALUATION_COMPLETED_ONCE",
                "GSE267904_d21_reference_metrics_opened": True,
                "parameters_changed_after_evaluation": False,
            },
        )
        if passed:
            AFFINE_PASS.write_text(status + "\n", encoding="utf-8")
        else:
            no_figure(status, "At least one frozen affine success criterion failed.")
    else:
        checks = {
            "median_nn_local_MAE_improved": bool(
                new["cell_type_local_mean_absolute_error"]
                < old["cell_type_local_mean_absolute_error"] - TOL
            ),
            "median_nn_GNRS_not_worse": bool(
                new["GNRS_secondary"] >= old["GNRS_secondary"] - TOL
            ),
        }
        passed = all(checks.values())
        status = "PASS_MEDIAN_NN_GATE" if passed else "BLOCKED_MEDIANNN_NO_FIGURE"
        audit = {
            "status": status,
            "checks": checks,
            "AgentVC_LRRA_medians": new.to_dict(),
            "old_AgentVC_medians": old.to_dict(),
        }
        write_json(OUT / "05_metrics/median_nn_gate.json", audit)
        if passed:
            MEDIAN_PASS.write_text(status + "\n", encoding="utf-8")
        else:
            no_figure(status, "Median-NN local MAE or GNRS sensitivity gate failed.")
    print(json.dumps(audit, indent=2))


def historical_immutability_audit() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    paths = {
        path: expected
        for path, expected in protocol["input_sha256"].items()
        if "/GSE267904_multimethod_cci_benchmark_v2/" in path
        or "/GSE267904_multimethod_cci_benchmark_v3_agentvc_hw/" in path
        or "/GSE267904_multimethod_cci_benchmark_v4_agentvc_srhw/" in path
    }
    changed = [path for path, expected in paths.items() if sha256_file(Path(path)) != expected]
    result = {
        "status": "PASS_HISTORICAL_RESULTS_UNCHANGED" if not changed else "FAIL_HISTORICAL_RESULTS_CHANGED",
        "changed": changed,
        "checked": paths,
    }
    write_json(OUT / "07_audits/historical_immutability_audit.json", result)
    return result


def write_report(status: str) -> None:
    cv = json.loads((OUT / "01_d7_cv/d7_cv_gate.json").read_text(encoding="utf-8"))
    lines = [
        "# GSE267904 AgentVC-LRRA V5",
        "",
        f"最终状态：`{status}`",
        "",
        "本版本是GSE267904 d7校准的post-hoc LR角色适配器，不是runtime-blind预测。",
        f"d7三折选择alpha：`{cv['selected_alpha']}`；local MAE中位数变化："
        f"`{cv['baseline_median_local_MAE']:.8f}` → `{cv['selected_median_local_MAE']:.8f}`。",
    ]
    affine_path = OUT / "05_metrics/affine_metrics_gate.json"
    if affine_path.is_file():
        affine = json.loads(affine_path.read_text(encoding="utf-8"))
        new = affine["AgentVC_LRRA_medians"]
        lines.extend(
            [
                "",
                "## Locked d21 affine评价",
                "",
                f"- local MAE：`{new['cell_type_local_mean_absolute_error']:.12f}`",
                f"- GNRS：`{new['GNRS_secondary']:.12f}`",
                f"- LBSS：`{new['LBSS_secondary']:.12f}`",
                f"- Gate：`{affine['status']}`",
            ]
        )
    if status != "PASS_AGENTVC_LRRA_TOP1_LOCAL_NO_REGRESSION":
        lines.extend(["", "未通过全部冻结门，因此没有生成论文图。"])
    (OUT / "08_reports/results_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize() -> None:
    freeze()
    require_marker(AFFINE_PASS, "affine metrics")
    require_marker(MEDIAN_PASS, "median-NN sensitivity")
    immutable = historical_immutability_audit()
    if immutable["changed"]:
        raise RuntimeError("Historical results changed")
    status = "PASS_AGENTVC_LRRA_TOP1_LOCAL_NO_REGRESSION"
    FINAL_PASS.write_text(status + "\n", encoding="utf-8")
    write_json(
        OUT / "07_audits/final_metric_status.json",
        {"status": status, "figure_authorized": True},
    )
    write_json(
        OUT / "07_audits/final_validation.json",
        {
            "status": status,
            "d7_spatial_cv": "PASS",
            "affine_metrics": "PASS",
            "median_nn_sensitivity": "PASS",
            "late_commot_jobs": 20,
            "llm_calls": 0,
            "physicell_runs": 0,
            "historical_outputs_modified": False,
            "fibrosis_application_v3": "paused_unchanged",
        },
    )
    write_report(status)
    print(status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "freeze",
            "prepare-d7",
            "smoke",
            "run-cv-base",
            "prepare-candidates",
            "run-cv-candidates",
            "evaluate-cv",
            "refit-freeze",
            "prepare-late",
            "commot-affine",
            "evaluate-affine",
            "commot-median",
            "evaluate-median",
            "finalize",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = {
        "freeze": freeze,
        "prepare-d7": prepare_d7,
        "smoke": smoke,
        "run-cv-base": run_cv_base,
        "prepare-candidates": prepare_candidates,
        "run-cv-candidates": run_cv_candidates,
        "evaluate-cv": evaluate_cv,
        "refit-freeze": refit_freeze,
        "prepare-late": prepare_late,
        "commot-affine": lambda: run_commot_mode("affine_pixel"),
        "evaluate-affine": lambda: evaluate_mode("affine_pixel"),
        "commot-median": lambda: run_commot_mode("median_nn"),
        "evaluate-median": lambda: evaluate_mode("median_nn"),
        "finalize": finalize,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
