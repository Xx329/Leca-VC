#!/usr/bin/env python3
"""Build AnnData real spatial benchmark files from GSE267904 raw Visium data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse

from common import (
    STAGES,
    discover_10x_samples,
    dump_json,
    ensure_dirs,
    find_raw_root,
    outpath,
    read_barcodes,
    read_features,
    read_positions,
)


def import_anndata():
    try:
        import anndata as ad
        return ad
    except Exception as e:
        raise RuntimeError(f"anndata is required to build h5ad benchmark files: {e}") from e


def build_one(sample: dict, stage: str, out_dir: Path) -> dict:
    ad = import_anndata()
    matrix = Path(sample["matrix"])
    features = Path(sample["features"]) if sample.get("features") else None
    barcodes = Path(sample["barcodes"]) if sample.get("barcodes") else None
    positions = Path(sample["positions"]) if sample.get("positions") else None
    if not features or not features.exists() or not barcodes or not barcodes.exists() or not positions or not positions.exists():
        raise RuntimeError(f"Sample {sample['sample_id']} is missing features/barcodes/positions: {sample}")
    X = mmread(matrix).tocsr()
    genes = read_features(features)
    bcs = read_barcodes(barcodes)
    if X.shape[0] == len(genes) and X.shape[1] == len(bcs):
        X = X.T.tocsr()
    elif X.shape[0] == len(bcs) and X.shape[1] == len(genes):
        X = X.tocsr()
    else:
        raise RuntimeError(f"Matrix shape {X.shape} does not match genes {len(genes)} and barcodes {len(bcs)} for {sample['sample_id']}")
    pos = read_positions(positions)
    obs = pd.DataFrame(index=pd.Index(bcs, name="spot_id"))
    obs["barcode"] = obs.index.astype(str)
    obs["sample_id"] = sample["sample_id"]
    obs["condition"] = "bleomycin" if "bleo" in stage else "control"
    obs["day"] = 21 if "21" in stage else 7
    obs["stage"] = stage
    obs = obs.merge(pos, on="barcode", how="left").set_index("barcode")
    keep = obs["pxl_row"].notna() & obs["pxl_col"].notna()
    obs = obs.loc[keep].copy()
    X = X[keep.to_numpy(), :]
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    var["gene_symbol"] = genes
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obsm["spatial"] = obs[["pxl_col", "pxl_row"]].to_numpy(dtype=float)
    out_path = out_dir / f"{stage}.h5ad"
    adata.write_h5ad(out_path)
    return {"stage": stage, "sample_id": sample["sample_id"], "h5ad": str(out_path), "n_spots": int(adata.n_obs), "n_genes": int(adata.n_vars)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--max-samples-per-stage", type=int, default=1)
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    raw_root = find_raw_root(root, out)
    samples = discover_10x_samples(raw_root)
    by_stage: dict[str, list[dict]] = {s: [] for s in STAGES}
    for s in samples:
        if s.get("stage") in by_stage:
            by_stage[s["stage"]].append(s)
    rows = []
    for stage in STAGES:
        for sample in by_stage.get(stage, [])[: a.max_samples_per_stage]:
            rows.append(build_one(sample, stage, out / "real_benchmark"))
    if not rows:
        raise RuntimeError(f"No usable GSE267904 Visium samples found under {raw_root}. Run download_gse267904.py first or inspect sample names.")
    pd.DataFrame(rows).to_csv(out / "real_benchmark/real_spatial_benchmark_manifest.csv", index=False)
    dump_json(out / "audit/gse267904_data_lineage_audit.json", {
        "GSE267904_RAW_ROOT": str(raw_root),
        "REAL_SPATIAL_DATA_USED": True,
        "SPOT_LEVEL_NOT_SINGLE_CELL_LEVEL": True,
        "CELL_TYPE_LABELS_INFERRED_BY_MARKERS": True,
        "built_h5ad_files": rows,
        "all_discovered_samples": samples,
    })
    print(f"Wrote {len(rows)} real benchmark h5ad files to {out/'real_benchmark'}")


if __name__ == "__main__":
    main()
