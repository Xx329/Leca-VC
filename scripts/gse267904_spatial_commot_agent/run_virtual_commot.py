#!/usr/bin/env python3
"""Run COMMOT on virtual GSE267904 micro-agent spatial data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import dump_json, ensure_dirs, outpath


def import_dependencies():
    try:
        import anndata as ad
        import commot as ct
    except Exception as e:
        raise RuntimeError(f"Virtual COMMOT requires importable anndata and commot: {e}") from e
    return ad, ct


def ligand_receptor_database(ct, species: str = "mouse"):
    if not hasattr(ct, "pp") or not hasattr(ct.pp, "ligand_receptor_database"):
        raise RuntimeError("commot.pp.ligand_receptor_database is not available.")
    return ct.pp.ligand_receptor_database(database="CellChat", species=species)


def focus_ligrec_for_lung_fibrosis(df_ligrec: pd.DataFrame, adata, max_lr_pairs: int) -> tuple[pd.DataFrame, dict]:
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


def run_one(h5: Path, stage: str, out: Path, max_cells: int = 0, seed: int = 42, max_lr_pairs: int = 120) -> dict:
    ad, ct = import_dependencies()
    adata = ad.read_h5ad(h5)
    if max_cells > 0 and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_cells, replace=False))
        adata = adata[idx].copy()
    adata.obs["dominant_cell_type"] = adata.obs["cell_type"].astype(str)
    df_ligrec = ligand_receptor_database(ct, "mouse")
    df_ligrec, filter_info = focus_ligrec_for_lung_fibrosis(df_ligrec, adata, max_lr_pairs=max_lr_pairs)
    ct.tl.spatial_communication(
        adata,
        database_name="cellchat",
        df_ligrec=df_ligrec,
        dis_thr=500,
        heteromeric=True,
        pathway_sum=True,
    )
    labels = adata.obs["dominant_cell_type"].astype(str).to_numpy()
    rows = []
    keys = list(getattr(adata, "obsp", {}).keys())
    comm_keys = [k for k in keys if "cellchat" in k.lower() and ("sum" in k.lower() or "total" in k.lower() or "comm" in k.lower())]
    if not comm_keys:
        comm_keys = [k for k in keys if "cellchat" in k.lower()]
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
        raise RuntimeError(f"COMMOT ran for virtual {stage}, but no communication matrix was found in adata.obsp keys: {keys}")
    edges = pd.DataFrame(rows)
    agg = edges.groupby(["stage", "sender", "receiver"], as_index=False)["weight"].sum()
    edges.to_csv(out / f"virtual_commot/{stage}_commot_edges_long.csv", index=False)
    agg.to_csv(out / f"virtual_commot/{stage}_sender_receiver_matrix.csv", index=False)
    return {"stage": stage, "h5ad": str(h5), "n_cells_used": int(adata.n_obs), "n_edges_raw": len(edges), "n_sender_receiver_edges": len(agg), "comm_keys_used": comm_keys, **filter_info}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--max-cells", type=int, default=0)
    p.add_argument("--max-lr-pairs", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    h5ads = sorted((out / "virtual_commot_input").glob("*.h5ad"))
    if not h5ads:
        raise RuntimeError("No virtual_commot_input/*.h5ad files found.")
    summaries = []
    all_agg, all_edges = [], []
    for h in h5ads:
        s = run_one(h, h.stem, out, max_cells=a.max_cells, seed=a.seed, max_lr_pairs=a.max_lr_pairs)
        summaries.append(s)
        all_agg.append(pd.read_csv(out / f"virtual_commot/{h.stem}_sender_receiver_matrix.csv"))
        all_edges.append(pd.read_csv(out / f"virtual_commot/{h.stem}_commot_edges_long.csv"))
    pd.concat(all_agg, ignore_index=True).to_csv(out / "virtual_commot/virtual_sender_receiver_matrix_by_stage.csv", index=False)
    pd.concat(all_edges, ignore_index=True).to_csv(out / "virtual_commot/virtual_commot_edges_long.csv", index=False)
    dump_json(out / "virtual_commot/virtual_commot_run_status.json", {"COMMOT_USED_FOR_VIRTUAL": True, "summaries": summaries})
    print(f"Wrote virtual COMMOT outputs to {out/'virtual_commot'}")


if __name__ == "__main__":
    main()
