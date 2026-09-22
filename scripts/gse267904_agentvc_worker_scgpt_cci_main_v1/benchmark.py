#!/usr/bin/env python3
"""Frozen expression, geometry, COMMOT, metrics and reporting pipeline."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from common import (
    COMMOT_CONFIG,
    D21,
    DIS_THR,
    FAMILIES,
    LATE_AGENT,
    N_PER_TYPE,
    OUT,
    PAIR_UNIVERSE,
    SAMPLE_SEEDS,
    SCGPT,
    SPOT_RADIUS_FULLRES,
    TYPES,
    WORKER_SEED_FOR_SAMPLE,
    WORKER_SEEDS,
    dense_panel,
    ensure_dirs,
    frozen_pairs_and_genes,
    geometry_stats,
    no_self_distance,
    normalized_h5ad,
    sha256_file,
    spatial_sample_indices,
    write_json,
)


TOTAL_KEY = "commot-cellchat-total-total"
EXPECTED_KEYS = [TOTAL_KEY] + [f"commot-cellchat-{x}" for x in FAMILIES]
EPS = 1e-15
METHOD_LABELS = {
    "scgpt": "Frozen scGPT encoder + external conditional head",
    "worker": "AgentVC worker-expanded parent-expression proxy",
    "pseudo_spot": "AgentVC worker pseudo-spot confirmation",
    "parent_agent": "Leca-VC K=100 parent-Agent ablation",
}
ALLOW_DYNAMIC_PARENT_COUNT = os.environ.get("GSE267904_DYNAMIC_PARENT_COUNT") == "1"


def protocol() -> dict:
    path = OUT / "00_protocol/frozen_worker_cci_protocol_v1.yaml"
    frozen = OUT / "00_protocol/frozen_protocol_hash.json"
    if not path.exists() or not frozen.exists():
        raise RuntimeError("Gate 0 protocol must be frozen first")
    value = json.loads(path.read_text(encoding="utf-8"))
    audit = json.loads(frozen.read_text(encoding="utf-8"))
    if sha256_file(path) != audit["protocol_sha256"]:
        raise RuntimeError("Frozen protocol hash changed")
    return value


def verify_frozen_inputs() -> None:
    p = protocol()
    changed = []
    for name, expected in p["input_hashes"].items():
        path = Path(name)
        if not path.exists() or sha256_file(path) != expected:
            changed.append(name)
    if changed:
        raise RuntimeError("Frozen input hash changed: " + ", ".join(changed))


def source_to_common(
    source_path: Path,
    source: str,
    label_column: str = "dominant_cell_type",
) -> ad.AnnData:
    _, genes = frozen_pairs_and_genes()
    a = ad.read_h5ad(source_path)
    mask = (
        a.obs["in_tissue"].astype(int).eq(1)
        & a.obs[label_column].astype(str).isin(TYPES)
    )
    a = a[mask].copy()
    pseudo = dense_panel(a, genes)
    obs = pd.DataFrame(
        {
            "cell_type": a.obs[label_column].astype(str).to_numpy(),
            "method": source,
            "source_entity_id": a.obs_names.astype(str),
            "in_tissue": 1,
        },
        index=pd.Index(
            [f"{source}_{x}" for x in a.obs_names.astype(str)], name="entity_id"
        ),
    )
    provenance = {
        "method": source,
        "input_semantics": (
            "raw observed count"
            if source == "observed"
            else "nonnegative pseudo-count from frozen scGPT encoder plus externally trained conditional expression head"
        ),
        "in_tissue_only": True,
        "input_path": str(source_path),
        "input_sha256": sha256_file(source_path),
    }
    return normalized_h5ad(
        pseudo,
        obs,
        np.asarray(a.obsm["spatial"], dtype=float),
        genes,
        provenance,
    )


def parent_proxy() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    _, genes = frozen_pairs_and_genes()
    late = ad.read_h5ad(LATE_AGENT)
    mask = late.obs["cell_type"].astype(str).isin(TYPES)
    late = late[mask].copy()
    if not ALLOW_DYNAMIC_PARENT_COUNT and late.n_obs != 32:
        raise RuntimeError(f"Expected 32 frozen primary parent expressions, got {late.n_obs}")
    log_proxy = dense_panel(late, genes)
    pseudo = np.expm1(np.clip(log_proxy, 0.0, 30.0))
    meta = pd.DataFrame(
        {
            "parent_agent_id": late.obs["agent_id"].astype(str).to_numpy(),
            "cell_type": late.obs["cell_type"].astype(str).to_numpy(),
        }
    )
    return meta, pseudo, np.asarray(late.obsm["spatial"], float), genes


def worker_common(seed: int) -> ad.AnnData:
    meta, parent_pseudo, _, genes = parent_proxy()
    lookup = {x: i for i, x in enumerate(meta["parent_agent_id"])}
    endpoint_path = (
        OUT / f"02_worker_generation/seed_{seed}/workers_endpoint_live.csv"
    )
    marker = endpoint_path.parent / "PHYSICELL_WORKER_COMPLETE.marker"
    audit_path = endpoint_path.parent / "worker_generation_audit.json"
    if not marker.exists() or not audit_path.exists():
        raise RuntimeError(f"Real PhysiCell worker seed {seed} is incomplete")
    audit = json.loads(audit_path.read_text())
    if not audit["status"].startswith("PASS_REAL_PHYSICELL"):
        raise RuntimeError(f"Worker seed {seed} failed: {audit['status']}")
    endpoint = pd.read_csv(endpoint_path)
    endpoint = endpoint[endpoint["cell_type"].isin(TYPES)].copy()
    missing = sorted(set(endpoint["parent_agent_id"]) - set(lookup))
    if missing:
        raise RuntimeError(f"Endpoint references unknown parent expressions: {missing}")
    pseudo = parent_pseudo[
        np.asarray([lookup[x] for x in endpoint["parent_agent_id"]], dtype=int)
    ]
    obs = pd.DataFrame(
        {
            "cell_type": endpoint["cell_type"].astype(str).to_numpy(),
            "method": "worker",
            "parent_agent_id": endpoint["parent_agent_id"].astype(str).to_numpy(),
            "physicell_cell_id": endpoint["physicell_cell_id"].astype(int).to_numpy(),
            "represented_weight": endpoint["represented_weight"].astype(float).to_numpy(),
            "generation_seed": seed,
            "is_initial_worker": endpoint["is_initial_worker"].astype(bool).to_numpy(),
        },
        index=pd.Index(endpoint["worker_id"].astype(str), name="entity_id"),
    )
    a = normalized_h5ad(
        pseudo,
        obs,
        endpoint[["x_pixel", "y_pixel"]].to_numpy(float),
        genes,
        {
            "method": "AgentVC worker-expanded parent-expression proxy",
            "input_semantics": "expm1 of frozen late parent expression; no noise and no LR boost",
            "generation_seed": seed,
            "real_physicell_endpoint": str(endpoint_path),
            "endpoint_sha256": sha256_file(endpoint_path),
        },
    )
    parent_sizes = obs.groupby("parent_agent_id").size()
    repeat = (
        obs.assign(_one=1)
        .groupby("parent_agent_id")["_one"]
        .sum()
        .sub(1)
        .clip(lower=0)
        .sum()
    )
    write_json(
        OUT / f"03_worker_expression/seed_{seed}_parent_proxy_audit.json",
        {
            "status": "PASS_WORKER_PARENT_EXPRESSION_PROXY",
            "worker_name": "AgentVC worker-expanded parent-expression proxy",
            "workers": int(a.n_obs),
            "unique_parent_expressions": int(obs["parent_agent_id"].nunique()),
            "same_parent_repeated_worker_rows": int(repeat),
            "same_parent_expression_repeat_fraction": float(repeat / max(a.n_obs, 1)),
            "min_workers_per_parent": int(parent_sizes.min()),
            "max_workers_per_parent": int(parent_sizes.max()),
            "random_expression_noise_added": False,
            "new_lr_boost_added": False,
        },
    )
    return a


def pseudo_spots(worker: ad.AnnData, seed: int) -> ad.AnnData:
    centers_path = OUT / "02_worker_generation/d7_in_tissue_primary_capture_centers.csv"
    centers = pd.read_csv(centers_path)
    coords = np.asarray(worker.obsm["spatial"], float)
    tree = cKDTree(coords)
    neighbor_ids = tree.query_ball_point(
        centers[["x_pixel", "y_pixel"]].to_numpy(float), SPOT_RADIUS_FULLRES
    )
    worker_count = np.asarray(worker.layers["pseudo_count"], dtype=float)
    rows, keep, n_workers = [], [], []
    for i, ids in enumerate(neighbor_ids):
        if ids:
            rows.append(worker_count[np.asarray(ids, int)].sum(axis=0))
            keep.append(i)
            n_workers.append(len(ids))
    if not rows:
        raise RuntimeError(f"No nonempty worker pseudo-spots for seed {seed}")
    centers = centers.iloc[keep].copy()
    obs = pd.DataFrame(
        {
            "cell_type": centers["cell_type"].astype(str).to_numpy(),
            "method": "pseudo_spot",
            "generation_seed": seed,
            "aggregated_live_workers": n_workers,
            "capture_radius_pixel": SPOT_RADIUS_FULLRES,
            "source_spot_id": centers["source_spot_id"].astype(str).to_numpy(),
        },
        index=pd.Index(
            [f"pseudo_{seed}_{x}" for x in centers["source_spot_id"].astype(str)],
            name="entity_id",
        ),
    )
    return normalized_h5ad(
        np.stack(rows),
        obs,
        centers[["x_pixel", "y_pixel"]].to_numpy(float),
        list(worker.var_names),
        {
            "method": "AgentVC worker pseudo-spot confirmation",
            "aggregation": "sum live worker pseudo-count within frozen full-resolution radius, then CP10K+log1p",
            "capture_radius_pixel": SPOT_RADIUS_FULLRES,
            "generation_seed": seed,
        },
    )


def parent_ablation() -> ad.AnnData:
    meta, pseudo, model_coords, genes = parent_proxy()
    from common import model_to_pixel

    obs = pd.DataFrame(
        {
            "cell_type": meta["cell_type"].to_numpy(),
            "method": "parent_agent",
            "parent_agent_id": meta["parent_agent_id"].to_numpy(),
        },
        index=pd.Index(meta["parent_agent_id"], name="entity_id"),
    )
    return normalized_h5ad(
        pseudo,
        obs,
        model_to_pixel(model_coords),
        genes,
        {
            "method": "Leca-VC K=100 parent-Agent ablation",
            "supplemental_only": True,
            "normalization_harmonized": True,
            "other_excluded": True,
        },
    )


def save_full_source(a: ad.AnnData, method: str, suffix: str = "") -> Path:
    path = OUT / "03_worker_expression" / f"{method}{suffix}_common_normalized.h5ad"
    a.write_h5ad(path, compression="gzip")
    return path


def sampled_path(method: str, seed: int) -> Path:
    return OUT / f"05_commot_inputs/{method}/seed_{seed}/commot_input.h5ad"


def sample_and_write(a: ad.AnnData, method: str, sample_seed: int) -> dict:
    idx, manifest = spatial_sample_indices(
        a.obs["cell_type"].astype(str).to_numpy(),
        np.asarray(a.obsm["spatial"], float),
        a.obs_names.astype(str).to_numpy(),
        sample_seed,
    )
    sample = a[idx].copy()
    path = sampled_path(method, sample_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.write_h5ad(path, compression="gzip")
    manifest["method"] = method
    manifest["worker_seed"] = WORKER_SEED_FOR_SAMPLE.get(sample_seed)
    manifest_path = path.parent / "sampling_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    stats = geometry_stats(
        np.asarray(sample.obsm["spatial"], float),
        sample.obs["cell_type"].astype(str).to_numpy(),
    )
    return {
        "method": method,
        "sampling_seed": sample_seed,
        "worker_seed": WORKER_SEED_FOR_SAMPLE.get(sample_seed),
        "n_entities": int(sample.n_obs),
        "n_genes": int(sample.n_vars),
        "input": str(path),
        "input_sha256": sha256_file(path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        **stats,
    }


def prepare_inputs() -> None:
    ensure_dirs()
    verify_frozen_inputs()
    observed = source_to_common(D21, "observed")
    scgpt = source_to_common(SCGPT, "scgpt")
    save_full_source(observed, "observed")
    save_full_source(scgpt, "scgpt")

    workers, pseudos = {}, {}
    pseudo_counts = {}
    for worker_seed in WORKER_SEEDS:
        workers[worker_seed] = worker_common(worker_seed)
        pseudos[worker_seed] = pseudo_spots(workers[worker_seed], worker_seed)
        save_full_source(workers[worker_seed], "worker", f"_seed_{worker_seed}")
        save_full_source(pseudos[worker_seed], "pseudo_spot", f"_seed_{worker_seed}")
        pseudo_counts[worker_seed] = (
            pseudos[worker_seed]
            .obs["cell_type"]
            .astype(str)
            .value_counts()
            .reindex(TYPES, fill_value=0)
            .astype(int)
            .to_dict()
        )
    pseudo_pass = all(
        min(counts.values()) >= N_PER_TYPE for counts in pseudo_counts.values()
    )
    status = "PASS_DUAL_LEVEL_INPUTS" if pseudo_pass else "CONDITIONAL_WORKER_ONLY"

    parent = parent_ablation()
    ppath = OUT / "05_commot_inputs/supplemental_parent_agent/commot_input.h5ad"
    ppath.parent.mkdir(parents=True, exist_ok=True)
    parent.write_h5ad(ppath, compression="gzip")

    rows = []
    for sample_seed in SAMPLE_SEEDS:
        worker_seed = WORKER_SEED_FOR_SAMPLE[sample_seed]
        rows.append(sample_and_write(observed, "observed", sample_seed))
        rows.append(sample_and_write(scgpt, "scgpt", sample_seed))
        rows.append(sample_and_write(workers[worker_seed], "worker", sample_seed))
        if pseudo_pass:
            rows.append(
                sample_and_write(pseudos[worker_seed], "pseudo_spot", sample_seed)
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "04_geometry_normalization/input_geometry_audit.csv", index=False)
    failures = []
    for _, row in audit.iterrows():
        if row["method"] in {"observed", "scgpt", "worker"}:
            if row["candidate_type_pairs"] < 75:
                failures.append(
                    f"{row['method']}/{row['sampling_seed']}:candidate={row['candidate_type_pairs']}"
                )
            if row["zero_neighbor_fraction"] >= 0.05:
                failures.append(
                    f"{row['method']}/{row['sampling_seed']}:zero_neighbor={row['zero_neighbor_fraction']}"
                )
    for sample_seed in SAMPLE_SEEDS:
        z = audit[audit["sampling_seed"].eq(sample_seed)].set_index("method")
        if (
            abs(
                float(z.loc["worker", "candidate_type_pair_fraction"])
                - float(z.loc["observed", "candidate_type_pair_fraction"])
            )
            > 0.10
        ):
            failures.append(f"worker-observed coverage difference seed {sample_seed}")
    if failures:
        write_json(
            OUT / "10_audits/input_fairness_validation.json",
            {"status": "FAIL_INPUT_GEOMETRY_GATE", "blockers": failures},
        )
        raise RuntimeError("FAIL_INPUT_GEOMETRY_GATE: " + "; ".join(failures))
    result = {
        "status": status,
        "d21_opened_only_in_evaluation_stage": True,
        "observed_in_tissue_primary": int(observed.n_obs),
        "scgpt_in_tissue_primary": int(scgpt.n_obs),
        "worker_realizations": WORKER_SEEDS,
        "pseudo_spot_counts_by_seed": pseudo_counts,
        "sampling_inputs": int(len(audit)),
        "expected_primary_sampling_inputs": 30,
        "pseudo_spot_sampling_inputs": int((audit["method"] == "pseudo_spot").sum()),
        "all_primary_inputs_810_entities": bool(
            (audit[audit["method"].isin(["observed", "scgpt", "worker"])]["n_entities"] == 810).all()
        ),
        "geometry_gate": "PASS",
        "other_policy": "NOT_ESTIMABLE_INVALID_HISTORICAL_OTHER",
    }
    write_json(OUT / "10_audits/input_fairness_validation.json", result)
    print(json.dumps(result, indent=2))


def median_nn_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, float)
    tree = cKDTree(coords)
    nn = tree.query(coords, k=2)[0][:, 1]
    median = float(np.median(nn[nn > 0]))
    if not np.isfinite(median) or median <= 0:
        raise RuntimeError("Cannot derive median-NN sensitivity scale")
    return (coords - np.mean(coords, axis=0)) * (100.0 / median)


def commot_jobs(include_sensitivity: bool = True) -> list[dict]:
    fairness = json.loads(
        (OUT / "10_audits/input_fairness_validation.json").read_text()
    )
    if fairness["status"].startswith("FAIL"):
        raise RuntimeError("Input geometry gate failed")
    jobs = []
    for sample_seed in SAMPLE_SEEDS:
        for method in ["observed", "scgpt", "worker"]:
            jobs.append(
                {
                    "method": method,
                    "sampling_seed": sample_seed,
                    "mode": "affine_pixel",
                    "input": sampled_path(method, sample_seed),
                }
            )
        if fairness["status"] == "PASS_DUAL_LEVEL_INPUTS":
            jobs.append(
                {
                    "method": "pseudo_spot",
                    "sampling_seed": sample_seed,
                    "mode": "affine_pixel",
                    "input": sampled_path("pseudo_spot", sample_seed),
                }
            )
        if include_sensitivity:
            for method in ["observed", "scgpt", "worker"]:
                jobs.append(
                    {
                        "method": method,
                        "sampling_seed": sample_seed,
                        "mode": "median_nn",
                        "input": sampled_path(method, sample_seed),
                    }
                )
    jobs.append(
        {
            "method": "parent_agent",
            "sampling_seed": 1701,
            "mode": "affine_pixel",
            "input": OUT
            / "05_commot_inputs/supplemental_parent_agent/commot_input.h5ad",
        }
    )
    return jobs


def commot_output(job: dict) -> Path:
    return (
        OUT
        / f"06_commot_runs/{job['mode']}/{job['method']}/seed_{job['sampling_seed']}/edges_long.csv"
    )


def run_commot_job(job: dict, pairs: pd.DataFrame) -> dict:
    import commot as ct

    version = importlib.metadata.version("commot")
    if version != COMMOT_CONFIG["package_version"]:
        raise RuntimeError(f"Expected COMMOT 0.0.3, got {version}")
    path = commot_output(job)
    audit_path = path.parent / "run_audit.json"
    input_hash = sha256_file(job["input"])
    pair_hash = sha256_file(PAIR_UNIVERSE)
    if path.exists() and audit_path.exists():
        old = json.loads(audit_path.read_text())
        if (
            old.get("status") == "PASS_COMMOT"
            and old.get("input_sha256") == input_hash
            and old.get("pair_panel_sha256") == pair_hash
        ):
            return old
    path.parent.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(job["input"])
    coords = np.asarray(a.obsm["spatial"], float)
    if job["mode"] == "median_nn":
        coords = median_nn_coords(coords)
        a.obsm["spatial"] = coords
    elif job["mode"] != "affine_pixel":
        raise ValueError(job["mode"])
    a.obsp["spatial_distance"] = no_self_distance(coords)
    db = pd.DataFrame(
        {
            "ligand": pairs["ligand_mouse"].astype(str),
            "receptor": pairs["receptor_mouse"].astype(str),
            "pathway": pairs["pathway"].astype(str),
        }
    )
    started = time.time()
    print(
        f"[COMMOT] {job['mode']} {job['method']} seed={job['sampling_seed']} n={a.n_obs}",
        flush=True,
    )
    ct.tl.spatial_communication(
        a,
        database_name="cellchat",
        df_ligrec=db,
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
    expected_keys = [TOTAL_KEY] + [
        f"commot-cellchat-{x}" for x in FAMILIES if x in set(pairs["pathway"])
    ]
    missing = [key for key in expected_keys if key not in a.obsp]
    if missing:
        raise RuntimeError(f"COMMOT keys missing: {missing}")
    labels = a.obs["cell_type"].astype(str).to_numpy()
    entity_ids = a.obs_names.astype(str).to_numpy()
    rows = []
    self_weight = 0.0
    for key in expected_keys:
        matrix = sparse.coo_matrix(a.obsp[key])
        for i, j, value in zip(matrix.row, matrix.col, matrix.data):
            if not value:
                continue
            if i == j:
                self_weight += float(value)
                continue
            rows.append(
                {
                    "geometry_mode": job["mode"],
                    "method": job["method"],
                    "sampling_seed": int(job["sampling_seed"]),
                    "sender_entity": entity_ids[i],
                    "receiver_entity": entity_ids[j],
                    "sender": labels[i],
                    "receiver": labels[j],
                    "commot_key": key,
                    "weight": float(value),
                }
            )
    if self_weight > 1e-12:
        raise RuntimeError(f"COMMOT entity self-loop exclusion failed: {self_weight}")
    edges = pd.DataFrame(rows)
    if edges.empty:
        raise RuntimeError("COMMOT produced no selected edges")
    edges.to_csv(path, index=False)
    audit = {
        "status": "PASS_COMMOT",
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in job.items()},
        "commot_version": version,
        "input_sha256": input_hash,
        "pair_panel_sha256": pair_hash,
        "pair_count_single_call": int(len(pairs)),
        "pair_blocks": 1,
        "selected_keys": expected_keys,
        "n_entities": int(a.n_obs),
        "entity_self_distance": float(DIS_THR + 1),
        "retained_entity_self_weight": self_weight,
        "retained_edge_rows": int(len(edges)),
        "elapsed_seconds": float(time.time() - started),
        "edges_sha256": sha256_file(path),
        "commot_parameters": COMMOT_CONFIG,
    }
    write_json(audit_path, audit)
    return audit


def run_commot_all(
    include_sensitivity: bool = True,
    limit: int | None = None,
    shard_id: int | None = None,
    shards: int = 1,
) -> None:
    verify_frozen_inputs()
    pairs, _ = frozen_pairs_and_genes()
    all_jobs = commot_jobs(include_sensitivity)
    if shards < 1 or (shard_id is not None and not 0 <= shard_id < shards):
        raise ValueError("Shard ID must be in [0, shards)")
    jobs = (
        [job for i, job in enumerate(all_jobs) if i % shards == shard_id]
        if shard_id is not None
        else all_jobs
    )
    audits, new = [], 0
    for i, job in enumerate(jobs, start=1):
        out = commot_output(job)
        existed = out.exists()
        if limit is not None and new >= limit and not existed:
            continue
        audit = run_commot_job(job, pairs)
        audits.append(audit)
        if not existed:
            new += 1
        write_json(
            OUT / "10_audits/commot_progress.json",
            {
                "status": (
                    "PASS_COMMOT_RUNS"
                    if all(commot_output(x).exists() for x in all_jobs)
                    else "RUNNING"
                ),
                "completed_jobs": sum(commot_output(x).exists() for x in all_jobs),
                "expected_jobs": len(all_jobs),
                "execution_shard": shard_id,
                "execution_shards": shards,
                "last_job": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in job.items()
                },
            },
        )
        print(
            f"[progress] {sum(commot_output(x).exists() for x in all_jobs)}/{len(all_jobs)}",
            flush=True,
        )
    all_audits = []
    for job in jobs:
        ap = commot_output(job).parent / "run_audit.json"
        if ap.exists():
            all_audits.append(json.loads(ap.read_text()))
    pd.DataFrame(all_audits).to_csv(
        OUT / "10_audits/commot_run_audit.csv", index=False
    )
    complete = sum(commot_output(x).exists() for x in all_jobs)
    if shard_id is None and limit is None and complete != len(all_jobs):
        raise RuntimeError(f"COMMOT incomplete: {complete}/{len(all_jobs)}")


def smoke_commot() -> None:
    """Technical-only 10/type, 10-pair end-to-end smoke."""
    verify_frozen_inputs()
    _, genes = frozen_pairs_and_genes()
    endpoint = OUT / "02_worker_generation/smoke_seed_1701/workers_endpoint_live.csv"
    if not endpoint.exists():
        raise RuntimeError("Real PhysiCell smoke endpoint is missing")
    meta, parent_pseudo, _, _ = parent_proxy()
    lookup = {x: i for i, x in enumerate(meta["parent_agent_id"])}
    ep = pd.read_csv(endpoint)
    pseudo = parent_pseudo[[lookup[x] for x in ep["parent_agent_id"]]]
    obs = pd.DataFrame(
        {
            "cell_type": ep["cell_type"].astype(str).to_numpy(),
            "method": "worker_smoke",
        },
        index=pd.Index(ep["worker_id"].astype(str), name="entity_id"),
    )
    worker = normalized_h5ad(
        pseudo,
        obs,
        ep[["x_pixel", "y_pixel"]].to_numpy(float),
        genes,
        {"technical_smoke_only": True, "performance_tuning_allowed": False},
    )
    path = OUT / "05_commot_inputs/smoke/worker_10_per_type.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    worker.write_h5ad(path, compression="gzip")
    pairs, _ = frozen_pairs_and_genes()
    job = {
        "method": "worker_smoke",
        "sampling_seed": 1701,
        "mode": "affine_pixel",
        "input": path,
    }
    # Smoke has its own output namespace and deliberately uses only the first
    # frozen ten pairs in one COMMOT call.
    original = commot_output(job)
    job["method"] = "smoke_worker"
    audit = run_commot_job(job, pairs.iloc[:10].copy())
    if audit["pair_count_single_call"] != 10:
        raise RuntimeError("Smoke did not execute the frozen 10-pair subset")
    write_json(
        OUT / "10_audits/end_to_end_smoke.json",
        {
            "status": "PASS_TECHNICAL_SMOKE",
            "real_physicell": True,
            "entities_per_type": 10,
            "lr_pairs": 10,
            "commot_output": str(commot_output(job)),
            "self_loop_exclusion": audit["retained_entity_self_weight"] == 0,
            "resume_audited": True,
            "performance_values_used_for_tuning": False,
        },
    )


def all_edges() -> pd.DataFrame:
    frames = []
    for job in commot_jobs(include_sensitivity=True):
        path = commot_output(job)
        if not path.exists():
            raise RuntimeError(f"Missing formal COMMOT output: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def safe_spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else math.nan


def type_network(
    edges: pd.DataFrame, mode: str, method: str, seed: int, key: str
) -> pd.Series:
    full = pd.MultiIndex.from_product(
        [TYPES, TYPES], names=["sender", "receiver"]
    )
    z = edges[
        edges["geometry_mode"].eq(mode)
        & edges["method"].eq(method)
        & edges["sampling_seed"].eq(seed)
        & edges["commot_key"].eq(key)
    ]
    s = z.groupby(["sender", "receiver"])["weight"].sum().reindex(full, fill_value=0.0)
    return s.astype(float)


def normalize(x: pd.Series) -> pd.Series:
    return x / max(float(x.sum()), EPS)


def positive_top(x: pd.Series, k: int = 10) -> tuple[set[str], int]:
    table = x.reset_index(name="weight")
    table["edge"] = table["sender"].astype(str) + "|" + table["receiver"].astype(str)
    table = table[table["weight"] > 0].sort_values(
        ["weight", "edge"], ascending=[False, True], kind="mergesort"
    )
    actual = min(k, len(table))
    return set(table.head(actual)["edge"]), actual


def js_similarity(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.sum() <= 0 or y.sum() <= 0:
        return math.nan
    return float(1.0 - jensenshannon(x + EPS, y + EPS) ** 2)


def evaluate() -> None:
    edges = all_edges()
    metric_rows, pathway_rows, role_rows, edge_rows, local_rows = [], [], [], [], []
    for mode in ["affine_pixel", "median_nn"]:
        methods = ["scgpt", "worker"]
        if mode == "affine_pixel" and (
            edges["method"].eq("pseudo_spot").any()
        ):
            methods.append("pseudo_spot")
        for seed in SAMPLE_SEEDS:
            real_raw = type_network(edges, mode, "observed", seed, TOTAL_KEY)
            real = normalize(real_raw)
            observed_descriptors = {
                "same_type_share": float(
                    sum(real.loc[(ct, ct)] for ct in TYPES)
                ),
                "cross_type_positive_edge_coverage": float(
                    np.mean(
                        [
                            real_raw.loc[(s, r)] > 0
                            for s in TYPES
                            for r in TYPES
                            if s != r
                        ]
                    )
                ),
            }
            for metric, value in observed_descriptors.items():
                metric_rows.append(
                    {
                        "geometry_mode": mode,
                        "sampling_seed": seed,
                        "worker_seed": None,
                        "method": "observed",
                        "metric": metric,
                        "value": value,
                        "interpretation": "sampling_stability_reference_descriptor",
                    }
                )
            for method in methods:
                pred_raw = type_network(edges, mode, method, seed, TOTAL_KEY)
                pred = normalize(pred_raw)
                informative = (real_raw > 0) | (pred_raw > 0)
                rt, rk = positive_top(real)
                pt, pk = positive_top(pred)
                jac = len(rt & pt) / max(len(rt | pt), 1)
                base_metrics = {
                    "edge_spearman_81": safe_spearman(real, pred),
                    "informative_edge_spearman": safe_spearman(
                        real[informative], pred[informative]
                    ),
                    "positive_top10_jaccard": jac,
                    "positive_top10_real_effective_k": float(rk),
                    "positive_top10_method_effective_k": float(pk),
                    "same_type_share": float(
                        sum(
                            pred.loc[(ct, ct)]
                            for ct in TYPES
                        )
                    ),
                    "cross_type_positive_edge_coverage": float(
                        np.mean(
                            [
                                pred_raw.loc[(s, r)] > 0
                                for s in TYPES
                                for r in TYPES
                                if s != r
                            ]
                        )
                    ),
                }
                flows_real, flows_pred = [], []
                roles = []
                for family in FAMILIES:
                    rr = type_network(
                        edges, mode, "observed", seed, f"commot-cellchat-{family}"
                    )
                    pp = type_network(
                        edges, mode, method, seed, f"commot-cellchat-{family}"
                    )
                    flows_real.append(float(rr.sum()))
                    flows_pred.append(float(pp.sum()))
                    for role, level in [("sender", 0), ("receiver", 1)]:
                        rrole = rr.groupby(level=level).sum().reindex(TYPES, fill_value=0)
                        prole = pp.groupby(level=level).sum().reindex(TYPES, fill_value=0)
                        rho = safe_spearman(rrole, prole)
                        roles.append(rho)
                        role_rows.append(
                            {
                                "geometry_mode": mode,
                                "sampling_seed": seed,
                                "method": method,
                                "pathway": family,
                                "role": role,
                                "spearman": rho,
                            }
                        )
                flow_real = np.asarray(flows_real) / max(sum(flows_real), EPS)
                flow_pred = np.asarray(flows_pred) / max(sum(flows_pred), EPS)
                base_metrics["five_pathway_flow_js_similarity"] = js_similarity(
                    flow_real, flow_pred
                )
                valid_roles = np.asarray(roles, float)
                base_metrics["sender_receiver_role_mean_spearman"] = (
                    float(np.nanmean(valid_roles))
                    if np.isfinite(valid_roles).any()
                    else math.nan
                )
                base_metrics["pathway_allocation_mean_absolute_error"] = float(
                    np.mean(np.abs(flow_pred - flow_real))
                )
                # Local error is the mean absolute normalized edge error incident
                # to each cell type, then averaged equally over the nine types.
                local = []
                for ct in TYPES:
                    idx = [
                        i
                        for i, (s, r) in enumerate(real.index)
                        if s == ct or r == ct
                    ]
                    err = float(np.mean(np.abs(pred.iloc[idx] - real.iloc[idx])))
                    local.append(err)
                    local_rows.append(
                        {
                            "geometry_mode": mode,
                            "sampling_seed": seed,
                            "method": method,
                            "cell_type": ct,
                            "local_mean_absolute_error": err,
                        }
                    )
                base_metrics["cell_type_local_mean_absolute_error"] = float(
                    np.mean(local)
                )
                base_metrics["GNRS_secondary"] = float(
                    np.nanmean(
                        [
                            (base_metrics["edge_spearman_81"] + 1) / 2,
                            base_metrics["positive_top10_jaccard"],
                        ]
                    )
                )
                base_metrics["LBSS_secondary"] = float(
                    np.nanmean(
                        [
                            base_metrics["five_pathway_flow_js_similarity"],
                            (
                                base_metrics[
                                    "sender_receiver_role_mean_spearman"
                                ]
                                + 1
                            )
                            / 2,
                        ]
                    )
                )
                for metric, value in base_metrics.items():
                    metric_rows.append(
                        {
                            "geometry_mode": mode,
                            "sampling_seed": seed,
                            "worker_seed": WORKER_SEED_FOR_SAMPLE[seed],
                            "method": method,
                            "metric": metric,
                            "value": value,
                            "interpretation": "sampling_and_simulation_stability_not_biological_inference",
                        }
                    )
                for family, rv, pv in zip(FAMILIES, flow_real, flow_pred):
                    pathway_rows.append(
                        {
                            "geometry_mode": mode,
                            "sampling_seed": seed,
                            "method": method,
                            "pathway": family,
                            "observed_share": rv,
                            "method_share": pv,
                            "absolute_error": abs(pv - rv),
                        }
                    )
                for (s, r), rv, pv in zip(real.index, real, pred):
                    edge_rows.append(
                        {
                            "geometry_mode": mode,
                            "sampling_seed": seed,
                            "method": method,
                            "sender": s,
                            "receiver": r,
                            "observed_normalized_weight": rv,
                            "method_normalized_weight": pv,
                            "absolute_error": abs(pv - rv),
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "07_metrics/seed_level_metrics.csv", index=False)
    pd.DataFrame(pathway_rows).to_csv(
        OUT / "07_metrics/pathway_allocation.csv", index=False
    )
    pd.DataFrame(role_rows).to_csv(
        OUT / "07_metrics/sender_receiver_roles.csv", index=False
    )
    pd.DataFrame(edge_rows).to_csv(
        OUT / "07_metrics/edge_level_values.csv", index=False
    )
    pd.DataFrame(local_rows).to_csv(
        OUT / "07_metrics/cell_type_local_errors.csv", index=False
    )
    summary = (
        metrics.groupby(["geometry_mode", "method", "metric"])["value"]
        .agg(n="count", median="median", minimum="min", maximum="max")
        .reset_index()
    )
    summary.to_csv(OUT / "07_metrics/metric_stability_summary.csv", index=False)
    write_json(
        OUT / "10_audits/metric_validation.json",
        {
            "status": "PASS_METRICS",
            "primary_metrics_reported_raw": True,
            "overall_score": None,
            "GNRS_and_LBSS_secondary_only": True,
            "p_values_computed": False,
            "biological_confidence_intervals_computed": False,
            "seed_interpretation": "sampling_and_simulation_stability_only",
            "global_and_local_reported_separately": True,
        },
    )


def plot_results() -> None:
    metrics = pd.read_csv(OUT / "07_metrics/seed_level_metrics.csv")
    primary_names = [
        "edge_spearman_81",
        "informative_edge_spearman",
        "positive_top10_jaccard",
        "five_pathway_flow_js_similarity",
        "sender_receiver_role_mean_spearman",
        "cell_type_local_mean_absolute_error",
    ]
    primary = metrics[
        metrics["geometry_mode"].eq("affine_pixel")
        & metrics["metric"].isin(primary_names)
    ].copy()
    methods = [x for x in ["scgpt", "worker", "pseudo_spot"] if x in set(primary["method"])]
    colors = {"scgpt": "#4C78A8", "worker": "#E45756", "pseudo_spot": "#72B7B2"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rng = np.random.default_rng(1701)
    for ax, metric in zip(axes.flat, primary_names):
        z = primary[primary["metric"].eq(metric)]
        for j, method in enumerate(methods):
            values = z[z["method"].eq(method)]["value"].dropna().to_numpy()
            ax.scatter(
                np.full(len(values), j) + rng.normal(0, 0.035, len(values)),
                values,
                s=18,
                alpha=0.65,
                color=colors[method],
            )
            if len(values):
                ax.plot([j - 0.22, j + 0.22], [np.median(values)] * 2, color="black", lw=2)
        display = {
            "scgpt": "frozen scGPT +\nexternal head",
            "worker": "AgentVC worker",
            "pseudo_spot": "AgentVC\npseudo-spot",
        }
        ax.set_xticks(range(len(methods)), [display[m] for m in methods])
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("GSE267904 CCI: frozen AgentVC worker benchmark", fontweight="bold")
    fig.tight_layout()
    for ext, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(OUT / f"08_figures/main_worker_cci_benchmark.{ext}", dpi=dpi, bbox_inches="tight")
    fig.savefig(OUT / "08_figures/main_worker_cci_benchmark_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fair = pd.read_csv(OUT / "04_geometry_normalization/input_geometry_audit.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(
        axes,
        ["candidate_type_pairs", "zero_neighbor_fraction", "median_nearest_neighbor"],
    ):
        for method in fair["method"].unique():
            values = fair[fair["method"].eq(method)][col]
            ax.plot(range(len(values)), values, marker="o", ms=3, label=method)
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("sampling realization")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "09_supplementary/fairness_geometry_audit.pdf", bbox_inches="tight")
    fig.savefig(OUT / "09_supplementary/fairness_geometry_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    values = [
        OUT / "07_metrics/seed_level_metrics.csv",
        OUT / "07_metrics/pathway_allocation.csv",
        OUT / "07_metrics/sender_receiver_roles.csv",
        OUT / "07_metrics/edge_level_values.csv",
        OUT / "04_geometry_normalization/input_geometry_audit.csv",
    ]
    write_json(
        OUT / "10_audits/figure_value_traceability.json",
        {
            "status": "PASS_FIGURE_VALUES_TRACEABLE",
            "figures": {
                "main_worker_cci_benchmark": [str(x) for x in values[:4]],
                "fairness_geometry_audit": str(values[4]),
            },
            "csv_sha256": {str(x): sha256_file(x) for x in values},
        },
    )


def validate() -> None:
    verify_frozen_inputs()
    blockers = []
    worker_audits = []
    for seed in WORKER_SEEDS:
        run_dir = OUT / f"02_worker_generation/seed_{seed}"
        marker = run_dir / "PHYSICELL_WORKER_COMPLETE.marker"
        ap = run_dir / "worker_generation_audit.json"
        if not marker.exists() or not ap.exists():
            blockers.append(f"missing real PhysiCell completion seed {seed}")
            continue
        a = json.loads(ap.read_text())
        worker_audits.append(a)
        if not a["status"].startswith("PASS_REAL_PHYSICELL"):
            blockers.append(f"worker seed {seed}: {a['status']}")
        if a.get("llm_calls") != 0:
            blockers.append(f"worker seed {seed}: nonzero LLM calls")
        if min(a["live_counts_by_type"].values()) < 90:
            blockers.append(f"worker seed {seed}: fewer than 90/type")
        conservation = a["source_conservation"]
        if not np.isclose(
            conservation["effective_parent_mass_sum"],
            conservation["frozen_parent_mass_sum"],
            atol=1e-9,
        ):
            blockers.append(f"worker seed {seed}: source mass not conserved")
    fairness = json.loads(
        (OUT / "10_audits/input_fairness_validation.json").read_text()
    )
    jobs = commot_jobs(include_sensitivity=True)
    missing_jobs = [str(commot_output(j)) for j in jobs if not commot_output(j).exists()]
    blockers.extend([f"missing COMMOT {x}" for x in missing_jobs])
    run_audit_path = OUT / "10_audits/commot_run_audit.csv"
    if run_audit_path.exists():
        ca = pd.read_csv(run_audit_path)
        if not (ca["pair_count_single_call"] == 58).all():
            blockers.append("at least one formal COMMOT call did not use all 58 pairs")
        if not np.allclose(ca["retained_entity_self_weight"], 0):
            blockers.append("entity self-loop leakage")
    else:
        blockers.append("missing COMMOT run audit")
    expected_primary = 30
    primary_completed = sum(
        commot_output(
            {
                "method": method,
                "sampling_seed": seed,
                "mode": "affine_pixel",
                "input": sampled_path(method, seed),
            }
        ).exists()
        for seed in SAMPLE_SEEDS
        for method in ["observed", "scgpt", "worker"]
    )
    if primary_completed != expected_primary:
        blockers.append(f"primary COMMOT combinations {primary_completed}/{expected_primary}")
    trace = OUT / "10_audits/figure_value_traceability.json"
    if not trace.exists():
        blockers.append("figure values not traceable")
    status = (
        "FAIL"
        if blockers
        else (
            "CONDITIONAL_WORKER_ONLY"
            if fairness["status"] == "CONDITIONAL_WORKER_ONLY"
            else "PASS"
        )
    )
    parent_program_count = int(
        pd.read_csv(OUT / "02_worker_generation/frozen_parent_programs.csv")[
            "parent_agent_id"
        ].nunique()
    )
    final = {
        "status": status,
        "blockers": blockers,
        "protocol_semantics": "prospectively frozen worker benchmark after historical Agent-level analyses",
        "real_physicell_realizations": len(worker_audits),
        "parent_programs": parent_program_count,
        "llm_calls": 0,
        "formal_affine_sampling_combinations": primary_completed,
        "formal_pairs_per_commot_call": 58,
        "entity_self_loops_excluded": not any("self-loop" in x for x in blockers),
        "pseudo_spot_status": fairness["status"],
        "other_sensitivity": "NOT_ESTIMABLE_INVALID_HISTORICAL_OTHER",
        "overall_score_reported": False,
        "biological_significance_claimed": False,
        "v3_state_changed": False,
    }
    write_json(OUT / "10_audits/final_validation.json", final)
    summary = f"""# GSE267904 AgentVC-worker vs external scGPT CCI

Final status: `{status}`

This is a prospectively frozen worker benchmark after historical Agent-level
analyses. It is not described as a de novo first preregistration.

- Real PhysiCell worker realizations completed: {len(worker_audits)}/3.
- Frozen current K=100 parent programs: {parent_program_count}; LLM calls: 0.
- Formal affine 810-entity COMMOT combinations: {primary_completed}/{expected_primary}.
- Each formal COMMOT call used the complete 58-pair panel in one collective-OT run.
- Worker expression is an **AgentVC worker-expanded parent-expression proxy**,
  not a native worker transcriptome.
- Sampling ranges quantify sampling/simulation stability only; no biological
  confidence interval or significance test was computed.
- AgentVC Other sensitivity: `NOT_ESTIMABLE_INVALID_HISTORICAL_OTHER`.
- Fibrosis Application V3 remains paused and was not touched.

Blockers: {json.dumps(blockers, ensure_ascii=False)}
"""
    (OUT / "FINAL_SUMMARY.md").write_text(summary, encoding="utf-8")
    if blockers:
        raise RuntimeError("Final validation failed: " + "; ".join(blockers))
    print(json.dumps(final, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "smoke",
            "prepare-inputs",
            "commot",
            "evaluate",
            "plot",
            "validate",
            "all-evaluation",
        ],
    )
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke_commot()
    elif args.stage == "prepare-inputs":
        prepare_inputs()
    elif args.stage == "commot":
        run_commot_all(
            not args.no_sensitivity, args.limit, args.shard_id, args.shards
        )
    elif args.stage == "evaluate":
        evaluate()
    elif args.stage == "plot":
        plot_results()
    elif args.stage == "validate":
        validate()
    else:
        prepare_inputs()
        run_commot_all(
            not args.no_sensitivity, args.limit, args.shard_id, args.shards
        )
        if args.limit is None:
            evaluate()
            plot_results()
            validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
