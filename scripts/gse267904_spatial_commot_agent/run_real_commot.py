#!/usr/bin/env python3
"""Run COMMOT on real GSE267904 spatial data.

This script fails if COMMOT is unavailable. It does not generate proxy COMMOT
outputs. The exact COMMOT API has changed across versions, so the script uses
the documented high-level functions when present and records API details.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import BLEO_STAGES, CELL_TYPES, COMMOT_PATHWAYS, dump_json, ensure_dirs, outpath, pearson


def import_dependencies():
    try:
        import anndata as ad
        import commot as ct
    except Exception as e:
        raise RuntimeError(f"COMMOT real run requires importable anndata and commot: {e}") from e
    return ad, ct


def ligand_receptor_database(ct, species: str = "mouse"):
    if not hasattr(ct, "pp") or not hasattr(ct.pp, "ligand_receptor_database"):
        raise RuntimeError("commot.pp.ligand_receptor_database is not available in this COMMOT installation.")
    return ct.pp.ligand_receptor_database(database="CellChat", species=species)


def subset_adata(adata, max_spots: int, seed: int):
    if max_spots <= 0 or adata.n_obs <= max_spots:
        return adata
    rng = np.random.default_rng(seed)
    labels = adata.obs["dominant_cell_type"].astype(str).to_numpy()
    keep = []
    unique_labels = pd.Series(labels).nunique()
    per_group = max(10, max_spots // max(1, unique_labels))
    for lab in sorted(set(labels)):
        idx = np.where(labels == lab)[0]
        n = min(len(idx), per_group)
        keep.extend(rng.choice(idx, size=n, replace=False).tolist())
    if len(keep) < max_spots:
        remaining = np.setdiff1d(np.arange(adata.n_obs), np.array(keep, dtype=int), assume_unique=False)
        add = min(len(remaining), max_spots - len(keep))
        if add > 0:
            keep.extend(rng.choice(remaining, size=add, replace=False).tolist())
    keep = np.array(sorted(set(map(int, keep))))[:max_spots]
    return adata[keep].copy()


def focus_ligrec_for_lung_fibrosis(df_ligrec: pd.DataFrame, adata, max_lr_pairs: int) -> tuple[pd.DataFrame, dict]:
    """Restrict CellChat LR DB to lung-injury/fibrosis-relevant, expressed pairs.

    This is a computational filter, not a target-fitting step. It uses pathway
    names and whether ligand/receptor genes are present in the expression matrix.
    """
    original_n = int(len(df_ligrec))
    df = df_ligrec.copy()
    pathway_col = "2" if "2" in df.columns else (2 if 2 in df.columns else None)
    ligand_col = "0" if "0" in df.columns else (0 if 0 in df.columns else None)
    receptor_col = "1" if "1" in df.columns else (1 if 1 in df.columns else None)
    focus_regex = r"TGF|SPP1|MIF|CCL|CXCL|TNF|IL1|PDGF|VEGF|COLLAGEN|COL|LAMININ|FN1|GALECTIN|ANNEXIN"
    used_focus = False
    if pathway_col is not None:
        mask = df[pathway_col].astype(str).str.contains(focus_regex, case=False, regex=True, na=False)
        if mask.any():
            df = df.loc[mask].copy()
            used_focus = True
    genes = set(map(str, adata.var_names))
    used_expression_filter = False
    if ligand_col is not None and receptor_col is not None:
        def gene_tokens(x: object) -> list[str]:
            return [t for t in str(x).replace("_", " ").replace("-", " ").split() if t]

        expr_mask = []
        for _, row in df.iterrows():
            lig_ok = any(g in genes for g in gene_tokens(row[ligand_col]))
            rec_ok = any(g in genes for g in gene_tokens(row[receptor_col]))
            expr_mask.append(lig_ok and rec_ok)
        expr_mask = pd.Series(expr_mask, index=df.index)
        if expr_mask.any():
            df = df.loc[expr_mask].copy()
            used_expression_filter = True
    if max_lr_pairs > 0 and len(df) > max_lr_pairs:
        df = df.head(max_lr_pairs).copy()
    info = {
        "original_lr_pairs": original_n,
        "filtered_lr_pairs": int(len(df)),
        "focus_pathway_filter_used": bool(used_focus),
        "expression_presence_filter_used": bool(used_expression_filter),
        "max_lr_pairs": int(max_lr_pairs),
    }
    if len(df) == 0:
        raise RuntimeError(f"LR filtering removed all pairs; filter info={info}")
    return df, info


def run_commot_one(adata, ct, stage: str, out: Path, max_spots: int = 0, seed: int = 42, max_lr_pairs: int = 120) -> dict:
    adata = subset_adata(adata, max_spots=max_spots, seed=seed)
    df_ligrec = ligand_receptor_database(ct, "mouse")
    # Keep secreted/signaling rows if COMMOT DB exposes a category column.
    if "annotation" in df_ligrec.columns:
        mask = df_ligrec["annotation"].astype(str).str.contains("Secreted|ECM|Cell-Cell", case=False, na=False)
        if mask.any():
            df_ligrec = df_ligrec.loc[mask].copy()
    df_ligrec, filter_info = focus_ligrec_for_lung_fibrosis(df_ligrec, adata, max_lr_pairs=max_lr_pairs)
    # COMMOT expects raw-like counts and spatial in obsm.
    ct.tl.spatial_communication(
        adata,
        database_name="cellchat",
        df_ligrec=df_ligrec,
        dis_thr=500,
        heteromeric=True,
        pathway_sum=True,
    )
    # Aggregate any generated communication summaries to sender/receiver cell type.
    rows = []
    keys = list(getattr(adata, "obsp", {}).keys())
    # Common COMMOT stores pairwise spot communication in obsp keys containing database name.
    comm_keys = [k for k in keys if "cellchat" in k.lower() and ("sum" in k.lower() or "total" in k.lower() or "comm" in k.lower())]
    if not comm_keys:
        comm_keys = [k for k in keys if "cellchat" in k.lower()]
    labels = adata.obs["dominant_cell_type"].astype(str).to_numpy()
    for key in comm_keys:
        mat = adata.obsp[key]
        if hasattr(mat, "tocoo"):
            coo = mat.tocoo()
            for i, j, v in zip(coo.row, coo.col, coo.data):
                rows.append({"stage": stage, "sender": labels[i], "receiver": labels[j], "pathway": key, "weight": float(v)})
        else:
            arr = np.asarray(mat)
            nz = np.argwhere(arr != 0)
            for i, j in nz:
                rows.append({"stage": stage, "sender": labels[i], "receiver": labels[j], "pathway": key, "weight": float(arr[i, j])})
    if not rows:
        raise RuntimeError(f"COMMOT ran for {stage}, but no communication matrix was found in adata.obsp keys: {keys}")
    edges = pd.DataFrame(rows)
    agg = edges.groupby(["stage", "sender", "receiver"], as_index=False)["weight"].sum()
    agg.to_csv(out / f"real_commot/{stage}_sender_receiver_matrix.csv", index=False)
    edges.to_csv(out / f"real_commot/{stage}_commot_edges_long.csv", index=False)
    return {"stage": stage, "n_spots_used": int(adata.n_obs), "n_edges_raw": len(edges), "n_sender_receiver_edges": len(agg), "obsp_keys": keys, "comm_keys_used": comm_keys, **filter_info}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--stages", nargs="*", default=BLEO_STAGES)
    p.add_argument("--max-spots", type=int, default=800, help="Subsample spots per stage for practical COMMOT smoke/MVP runs; <=0 means full sample.")
    p.add_argument("--max-lr-pairs", type=int, default=120, help="Limit focused lung-fibrosis LR pairs for tractable COMMOT runs; <=0 keeps all focused pairs.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    ad, ct = import_dependencies()
    summaries = []
    all_agg = []
    all_edges = []
    for stage in a.stages:
        h5 = out / f"real_benchmark/{stage}.h5ad"
        if not h5.exists():
            print(f"Skipping {stage}; missing {h5}")
            continue
        adata = ad.read_h5ad(h5)
        if "dominant_cell_type" not in adata.obs:
            raise RuntimeError(f"{h5} lacks obs['dominant_cell_type']; run annotate_spots_by_markers.py")
        s = run_commot_one(adata, ct, stage, out, max_spots=a.max_spots, seed=a.seed, max_lr_pairs=a.max_lr_pairs)
        summaries.append(s)
        all_agg.append(pd.read_csv(out / f"real_commot/{stage}_sender_receiver_matrix.csv"))
        all_edges.append(pd.read_csv(out / f"real_commot/{stage}_commot_edges_long.csv"))
    if not summaries:
        raise RuntimeError("No real COMMOT stages were run.")
    pd.concat(all_agg, ignore_index=True).to_csv(out / "real_commot/real_sender_receiver_matrix_by_stage.csv", index=False)
    pd.concat(all_edges, ignore_index=True).to_csv(out / "real_commot/real_commot_edges_long.csv", index=False)
    dump_json(out / "real_commot/real_commot_run_status.json", {"COMMOT_USED_FOR_REAL": True, "summaries": summaries})
    print(f"Wrote real COMMOT outputs to {out/'real_commot'}")


if __name__ == "__main__":
    main()
