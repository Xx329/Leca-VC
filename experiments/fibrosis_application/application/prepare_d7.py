#!/usr/bin/env python3
"""Prepare leakage-free d7 fields and exactly 50 cell-centric agents per section."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree, distance_matrix
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    DEFAULT_OUT, FIELDS, coordinate_transform, discover_manifest, dump_json, ensure_cell2location,
    ensure_dirs, estimate_spot_spacing, load_cell2location, load_config, read_visium_tissue,
    resolve_out, sha256, transform_xy,
)


def module_scores(adata, config: dict) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    matrix = sparse.csr_matrix(adata.X, dtype=np.float64).copy()
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    matrix = sparse.diags(1e4 / np.maximum(library_size, 1.0)) @ matrix
    matrix.data = np.log1p(matrix.data)
    gene_means = np.asarray(matrix.mean(axis=0)).ravel()
    # Match controls by average-expression rank. This is the same statistical
    # contract as score_genes without importing scanpy/numba at runtime.
    order = np.argsort(gene_means, kind="stable")
    expression_bin = np.empty(len(order), dtype=int)
    expression_bin[order] = np.minimum(np.arange(len(order)) * 25 // max(len(order), 1), 24)
    scores = pd.DataFrame(index=adata.obs_names)
    coverage = {}
    lookup = {str(g).split("-", 1)[0].lower(): j for j, g in enumerate(adata.var_names)}
    all_module_indices = {
        lookup[g.lower()] for genes in config["module_genes"].values() for g in genes if g.lower() in lookup
    }
    for i, (field, genes) in enumerate(config["module_genes"].items()):
        present_indices = sorted({lookup[g.lower()] for g in genes if g.lower() in lookup})
        present = [str(adata.var_names[j]).split("-", 1)[0] for j in present_indices]
        coverage[field] = present
        if len(present_indices) < 4:
            raise RuntimeError(f"{field} has only {len(present_indices)} present genes: {present}")
        rng = np.random.default_rng(1701 + i)
        controls: set[int] = set()
        ctrl_size = max(25, len(present_indices))
        for bin_id in sorted(set(expression_bin[present_indices])):
            candidates = np.where(expression_bin == bin_id)[0]
            candidates = np.asarray([j for j in candidates if j not in all_module_indices], dtype=int)
            if len(candidates) == 0:
                raise RuntimeError(f"No matched background genes for {field} expression bin {bin_id}")
            selected = rng.choice(candidates, size=min(ctrl_size, len(candidates)), replace=False)
            controls.update(int(j) for j in selected)
        module_mean = np.asarray(matrix[:, present_indices].mean(axis=1)).ravel()
        control_mean = np.asarray(matrix[:, sorted(controls)].mean(axis=1)).ravel()
        scores[field + "_module_raw"] = module_mean - control_mean
    return scores, coverage


def smooth_scores(scores: pd.DataFrame, xy: np.ndarray, alpha: float) -> pd.DataFrame:
    n_neighbors = min(7, len(scores))
    idx = NearestNeighbors(n_neighbors=n_neighbors).fit(xy).kneighbors(return_distance=False)
    out = scores.copy()
    for col in scores:
        values = scores[col].to_numpy(dtype=float)
        neighbor_mean = values[idx[:, 1:]].mean(axis=1) if n_neighbors > 1 else values
        out[col.replace("_raw", "_smoothed")] = (1.0 - alpha) * values + alpha * neighbor_mean
    return out


def fit_ctrl_normalization(sample_frames: dict[str, pd.DataFrame], manifest: pd.DataFrame) -> dict[str, dict[str, float]]:
    ctrl_ids = manifest.loc[manifest.stage.eq("d7_ctrl"), "sample_id"]
    pooled = pd.concat([sample_frames[s] for s in ctrl_ids], ignore_index=True)
    params = {}
    for field in FIELDS:
        col = field + "_module_smoothed"
        median = float(np.nanmedian(pooled[col]))
        mad = float(np.nanmedian(np.abs(pooled[col] - median)))
        params[field] = {"d7_ctrl_median": median, "d7_ctrl_mad": max(mad, 1e-6), "transform": "expit((score-median)/(1.4826*MAD))"}
    return params


def apply_frozen_fields(df: pd.DataFrame, params: dict, abundance_scale: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for field in FIELDS:
        p = params[field]
        z = (out[field + "_module_smoothed"] - p["d7_ctrl_median"]) / (1.4826 * p["d7_ctrl_mad"])
        out[field + "_robust_z"] = z.clip(-8, 8)
        out[field] = expit(out[field + "_robust_z"])
    epithelial = out[["alveolar_epithelial_AT1_AT2", "activated_Krt8_ADI_epithelial", "airway_epithelial"]].sum(axis=1)
    macro = out[["macrophage", "recruited_monocyte_macrophage"]].sum(axis=1)
    fibro = out["fibroblast_myofibroblast"]
    out["epithelial_homeostasis"] *= epithelial / (epithelial + abundance_scale["epithelial"])
    out["macrophage_APOE_SPP1_signal"] *= macro / (macro + abundance_scale["macrophage"])
    out["ECM_fibrosis"] *= fibro / (fibro + abundance_scale["fibroblast"])
    return out


def largest_remainder_allocation(totals: pd.Series, config: dict, n_agents: int) -> dict[str, int]:
    cell_types = list(config["cell_types"])
    minimum = {ct: int(config["cell_types"][ct]["minimum_agents"]) for ct in cell_types}
    if sum(minimum.values()) > n_agents:
        raise RuntimeError("Cell-type minimum allocation exceeds agent count")
    remaining = n_agents - sum(minimum.values())
    weights = totals.reindex(cell_types).fillna(0).clip(lower=0)
    weights = weights / max(float(weights.sum()), 1e-12)
    quotas = weights * remaining
    alloc = {ct: minimum[ct] + int(math.floor(quotas[ct])) for ct in cell_types}
    left = n_agents - sum(alloc.values())
    order = sorted(cell_types, key=lambda ct: (quotas[ct] - math.floor(quotas[ct]), weights[ct], ct), reverse=True)
    for ct in order[:left]:
        alloc[ct] += 1
    if sum(alloc.values()) != n_agents or any(v < 1 for v in alloc.values()):
        raise RuntimeError(f"Invalid agent allocation {alloc}")
    return alloc


def weighted_medoids(xy: np.ndarray, weights: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic weighted PAM approximation returning real spot indices and assignments."""
    rng = np.random.default_rng(seed)
    w = np.maximum(np.asarray(weights, dtype=float), 1e-9)
    first = int(rng.choice(len(xy), p=w / w.sum()))
    medoids = [first]
    nearest = np.linalg.norm(xy - xy[first], axis=1)
    while len(medoids) < k:
        score = w * nearest * nearest
        score[medoids] = 0
        nxt = int(np.argmax(score))
        medoids.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(xy - xy[nxt], axis=1))
    medoids = np.asarray(medoids, dtype=int)
    for _ in range(4):
        dist = distance_matrix(xy, xy[medoids])
        assignment = np.argmin(dist, axis=1)
        changed = False
        for cluster in range(k):
            members = np.where(assignment == cluster)[0]
            if len(members) == 0:
                continue
            candidates = members if len(members) <= 250 else members[np.argsort(w[members])[-250:]]
            cost = distance_matrix(xy[candidates], xy[members]) @ w[members]
            best = int(candidates[int(np.argmin(cost))])
            if best != medoids[cluster]:
                medoids[cluster] = best
                changed = True
        if not changed:
            break
    assignment = np.argmin(distance_matrix(xy, xy[medoids]), axis=1)
    return medoids, assignment


def build_agents(df: pd.DataFrame, config: dict, sample_id: str) -> pd.DataFrame:
    cell_types = list(config["cell_types"])
    alloc = largest_remainder_allocation(df[cell_types].sum(), config, int(config["agent_count_per_section"]))
    xy = df[["x", "y"]].to_numpy(dtype=float)
    rows = []
    agent_no = 0
    for type_no, cell_type in enumerate(cell_types):
        abundance = df[cell_type].to_numpy(dtype=float)
        medoids, assignment = weighted_medoids(xy, abundance, alloc[cell_type], 1701 + type_no)
        for local_no, medoid in enumerate(medoids):
            represented = float(abundance[assignment == local_no].sum())
            spot = df.iloc[int(medoid)]
            rows.append({
                "agent_id": f"{sample_id}_agent_{agent_no:03d}", "sample_id": sample_id,
                "cell_type": cell_type, "spot_id": spot["spot_id"], "x": float(spot["x"]), "y": float(spot["y"]),
                "represented_abundance": represented, "initial_represented_abundance": represented,
                "fibrosis_memory": 0.0, "agent_semantics": "one_cell_type_representative_cell_not_controller",
                **{field: float(spot[field]) for field in FIELDS},
            })
            agent_no += 1
    agents = pd.DataFrame(rows)
    if len(agents) != 50 or agents["agent_id"].nunique() != 50:
        raise RuntimeError(f"{sample_id} did not produce exactly 50 unique agents")
    return agents


def interpolate_mesh(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    mesh = config["mesh"]
    values = np.arange(float(mesh["min"]) + float(mesh["spacing"]) / 2, float(mesh["max"]), float(mesh["spacing"]))
    grid = np.array([(x, y) for y in values for x in values], dtype=float)
    xy = df[["x", "y"]].to_numpy(dtype=float)
    spacing = estimate_spot_spacing(xy)
    tree = cKDTree(xy)
    distances, idx = tree.query(grid, k=min(7, len(xy)))
    if distances.ndim == 1:
        distances, idx = distances[:, None], idx[:, None]
    mask = distances[:, 0] <= spacing * float(mesh["tissue_radius_in_spot_spacings"])
    weights = np.exp(-0.5 * (distances / max(spacing, 1e-8)) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    out = pd.DataFrame({"voxel_id": np.arange(len(grid)), "x": grid[:, 0], "y": grid[:, 1], "tissue_mask": mask.astype(int)})
    for field in FIELDS:
        interpolated = np.sum(weights * df[field].to_numpy(dtype=float)[idx], axis=1)
        out[field] = np.where(mask, interpolated, 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    config = load_config()
    assigned: dict[str, str] = {}
    for field, genes in config["module_genes"].items():
        for gene in genes:
            key = str(gene).lower()
            if key in assigned:
                raise RuntimeError(f"Module gene {gene} overlaps between {assigned[key]} and {field}")
            assigned[key] = field
    ensure_dirs(out)
    manifest = discover_manifest(root)
    manifest.to_csv(out / "manifests/gse267904_24_sections.csv", index=False)
    c2l_dir, c2l_audit = ensure_cell2location(root, config)

    d7_manifest = manifest[manifest["day"].eq(7)].copy()
    frames: dict[str, pd.DataFrame] = {}
    coverage = {}
    transforms = {}
    for _, sample in d7_manifest.iterrows():
        adata, obs = read_visium_tissue(sample)
        scores, sample_coverage = module_scores(adata, config)
        xy_raw = np.asarray(adata.obsm["spatial"], dtype=float)
        smoothed = smooth_scores(scores, xy_raw, float(config["module_smoothing_alpha"]))
        c2l = load_cell2location(sample["sample_id"], c2l_dir, config).reindex(adata.obs_names)
        if c2l.isna().any().any():
            raise RuntimeError(f"Official cell2location does not cover all in-tissue spots for {sample['sample_id']}")
        transform = coordinate_transform(xy_raw)
        xy = transform_xy(xy_raw, transform)
        frame = pd.DataFrame({
            "spot_id": adata.obs_names.astype(str), "sample_id": sample["sample_id"], "stage": sample["stage"],
            "animal_id": sample["animal_id"], "technical_section": sample["technical_section"],
            "in_tissue": 1, "x_raw": xy_raw[:, 0], "y_raw": xy_raw[:, 1], "x": xy[:, 0], "y": xy[:, 1],
        }, index=adata.obs_names)
        frame = pd.concat([frame, smoothed, c2l], axis=1)
        frames[sample["sample_id"]] = frame.reset_index(drop=True)
        coverage[sample["sample_id"]] = sample_coverage
        transforms[sample["sample_id"]] = transform

    params = fit_ctrl_normalization(frames, d7_manifest)
    ctrl = pd.concat([frames[s] for s in d7_manifest.loc[d7_manifest.stage.eq("d7_ctrl"), "sample_id"]])
    abundance_scale = {
        "epithelial": float(np.nanmedian(ctrl[["alveolar_epithelial_AT1_AT2", "activated_Krt8_ADI_epithelial", "airway_epithelial"]].sum(axis=1))),
        "macrophage": float(np.nanmedian(ctrl[["macrophage", "recruited_monocyte_macrophage"]].sum(axis=1))),
        "fibroblast": float(np.nanmedian(ctrl["fibroblast_myofibroblast"])),
    }
    abundance_scale = {key: max(value, 1e-6) for key, value in abundance_scale.items()}

    agent_audit = {}
    for sample_id, frame in frames.items():
        final = apply_frozen_fields(frame, params, abundance_scale)
        final.to_csv(out / f"fields/d7/{sample_id}_spot_fields.csv.gz", index=False, compression="gzip")
        interpolate_mesh(final, config).to_csv(out / f"fields/d7/{sample_id}_physicell_mesh.csv", index=False)
        agents = build_agents(final, config, sample_id)
        agents.to_csv(out / f"agents/{sample_id}_cell_agents.csv", index=False)
        represented_by_type = agents.groupby("cell_type")["represented_abundance"].sum()
        source_by_type = final[list(config["cell_types"])].sum()
        conservation_error = (represented_by_type - source_by_type).abs()
        agent_audit[sample_id] = {
            "n_agents": int(len(agents)), "all_in_tissue": True,
            "cell_type_allocation": agents["cell_type"].value_counts().to_dict(),
            "represented_abundance_sum": float(agents["represented_abundance"].sum()),
            "represented_abundance_by_cell_type": represented_by_type.to_dict(),
            "cell2location_abundance_by_cell_type": source_by_type.to_dict(),
            "maximum_abundance_conservation_error": float(conservation_error.max()),
            "semantics": "each LLM agent is one cell-type representative PhysiCell cell; no controller or worker cells",
        }

    dump_json(out / "audit/d7_preparation_audit.json", {
        "experiment": config["experiment"], "d21_files_opened": False,
        "processed_stages": ["d7_ctrl", "d7_bleo"], "n_d7_sections": len(d7_manifest),
        "official_cell2location": c2l_audit, "module_gene_coverage": coverage,
        "normalization": params, "abundance_scales_from_d7_ctrl": abundance_scale,
        "coordinate_transforms": transforms, "agents": agent_audit,
        "module_smoothing": "0.7 raw score + 0.3 mean of six nearest in-tissue Visium neighbors",
    })
    print(f"Prepared 12 leakage-free d7 sections and 50 cell-centric agents per section in {out}")


if __name__ == "__main__":
    main()
