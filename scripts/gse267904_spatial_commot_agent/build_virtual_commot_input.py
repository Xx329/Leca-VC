#!/usr/bin/env python3
"""Build virtual AnnData inputs for COMMOT from micro-agent spatial output.

The first version used only cell-type expression prototypes plus small
stage-level shifts. That was sufficient to make self-communication edges, but
it under-powered biologically meaningful cross-cell-type ligand/receptor
communication. This version adds an explicit, audited LR expression program:

    micro-agent prototype
    + stage/environment shifts
    + Agent/PhysiCell communication writeback
    + sender ligand / receiver receptor boosts for calibration-prior cross-type
      axes

This is still not LLM per-gene transcriptome decoding. It is a transparent
expression reconstruction step for COMMOT input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import dump_json, ensure_dirs, outpath


CROSS_TYPE_LR_PROGRAM = {
    # Injured / activated epithelium recruits and activates macrophage lineage.
    "activated_Krt8_ADI_epithelial": {
        "ligands": ["Ccl2", "Ccl7", "Cxcl10", "Spp1", "Mif"],
        "targets": ["macrophage", "recruited_monocyte_macrophage", "dendritic"],
        "stage_weight": {"virtual_early_bleo": 1.10, "virtual_late_bleo": 0.85},
        "signal_columns": ["damage_signal", "chemokine_signal", "SPP1_like_signal"],
    },
    "alveolar_epithelial_AT1_AT2": {
        "ligands": ["Ccl2", "Cxcl10", "Mif"],
        "targets": ["macrophage", "recruited_monocyte_macrophage"],
        "stage_weight": {"virtual_early_bleo": 0.75, "virtual_late_bleo": 0.55},
        "signal_columns": ["damage_signal", "chemokine_signal"],
    },
    "airway_epithelial": {
        "ligands": ["Ccl2", "Cxcl10", "Mif"],
        "targets": ["macrophage", "recruited_monocyte_macrophage"],
        "stage_weight": {"virtual_early_bleo": 0.70, "virtual_late_bleo": 0.50},
        "signal_columns": ["damage_signal", "chemokine_signal"],
    },
    # Macrophage lineage signals to fibroblast and epithelial compartments.
    "macrophage": {
        "ligands": ["Spp1", "Tnf", "Il1b", "Ccl3", "Ccl4", "Csf1"],
        "targets": ["fibroblast_myofibroblast", "activated_Krt8_ADI_epithelial", "alveolar_epithelial_AT1_AT2", "lymphoid"],
        "stage_weight": {"virtual_early_bleo": 0.85, "virtual_late_bleo": 1.15},
        "signal_columns": ["inflammatory_signal", "SPP1_like_signal", "chemokine_signal"],
    },
    "recruited_monocyte_macrophage": {
        "ligands": ["Spp1", "Tnf", "Il1b", "Ccl3", "Ccl4", "Csf1"],
        "targets": ["fibroblast_myofibroblast", "activated_Krt8_ADI_epithelial", "lymphoid"],
        "stage_weight": {"virtual_early_bleo": 0.80, "virtual_late_bleo": 1.10},
        "signal_columns": ["inflammatory_signal", "SPP1_like_signal", "chemokine_signal"],
    },
    # Fibroblast / myofibroblast remodeling feeds back to epithelium and endothelium.
    "fibroblast_myofibroblast": {
        "ligands": ["Tgfb1", "Fn1", "Col1a1", "Col1a2", "Pdgfa", "Vegfa"],
        "targets": ["alveolar_epithelial_AT1_AT2", "activated_Krt8_ADI_epithelial", "endothelial", "macrophage"],
        "stage_weight": {"virtual_early_bleo": 0.45, "virtual_late_bleo": 1.20},
        "signal_columns": ["fibrosis_signal", "TGFb_like_signal"],
    },
    "endothelial": {
        "ligands": ["Vegfa", "Cxcl12", "Ccl2", "Mif"],
        "targets": ["macrophage", "recruited_monocyte_macrophage", "fibroblast_myofibroblast"],
        "stage_weight": {"virtual_early_bleo": 0.50, "virtual_late_bleo": 0.65},
        "signal_columns": ["chemokine_signal", "inflammatory_signal"],
    },
    "dendritic": {
        "ligands": ["Ccl5", "Cxcl10", "Tnf", "Mif"],
        "targets": ["lymphoid", "macrophage"],
        "stage_weight": {"virtual_early_bleo": 0.55, "virtual_late_bleo": 0.70},
        "signal_columns": ["inflammatory_signal", "chemokine_signal"],
    },
}


TARGET_RECEPTORS = {
    "macrophage": ["Ccr2", "Cxcr3", "Cd44", "Cd74", "Csf1r", "Tnfrsf1a", "Il1r1"],
    "recruited_monocyte_macrophage": ["Ccr2", "Cxcr3", "Cd44", "Cd74", "Csf1r", "Tnfrsf1a", "Il1r1"],
    "fibroblast_myofibroblast": ["Cd44", "Tgfbr1", "Tgfbr2", "Pdgfra", "Itgav", "Itgb1", "Tnfrsf1a", "Il1r1"],
    "activated_Krt8_ADI_epithelial": ["Cd44", "Tgfbr1", "Tgfbr2", "Tnfrsf1a", "Il1r1", "Cxcr4"],
    "alveolar_epithelial_AT1_AT2": ["Cd44", "Tgfbr1", "Tgfbr2", "Tnfrsf1a", "Il1r1", "Cxcr4"],
    "airway_epithelial": ["Cd44", "Tgfbr1", "Tgfbr2", "Tnfrsf1a", "Il1r1"],
    "endothelial": ["Kdr", "Flt1", "Tgfbr1", "Tgfbr2", "Itgb1"],
    "dendritic": ["Ccr5", "Cxcr3", "Cd74", "Tnfrsf1a"],
    "lymphoid": ["Ccr5", "Cxcr3", "Cd74", "Tnfrsf1a", "Il1r1"],
}


CALIBRATION_SENDER_LIGANDS = {
    "alveolar_epithelial_AT1_AT2": ["Ccl2", "Cxcl10", "Spp1", "Mif", "Fn1"],
    "activated_Krt8_ADI_epithelial": ["Ccl2", "Ccl7", "Cxcl10", "Spp1", "Mif"],
    "airway_epithelial": ["Ccl2", "Cxcl10", "Mif", "Fn1"],
    "macrophage": ["Spp1", "Tnf", "Il1b", "Ccl3", "Ccl4", "Csf1", "Mif"],
    "recruited_monocyte_macrophage": ["Spp1", "Tnf", "Il1b", "Ccl3", "Ccl4", "Csf1", "Mif"],
    "fibroblast_myofibroblast": ["Tgfb1", "Fn1", "Col1a1", "Col1a2", "Cxcl12", "Ccl2", "Pdgfa"],
    "endothelial": ["Vegfa", "Cxcl12", "Ccl2", "Mif"],
    "dendritic": ["Ccl5", "Cxcl10", "Tnf", "Mif", "Il1b"],
    "lymphoid": ["Ccl5", "Cxcl10", "Tnf", "Mif"],
}


def load_d7_cross_type_calibration_prior(out: Path, top_n: int = 20) -> pd.DataFrame:
    """Read only d7 bleomycin cross-cell-type COMMOT edges as calibration prior.

    d21 is intentionally excluded because it is the late validation stage.
    The prior is not a target matrix written into outputs; it only says which
    early injury sender/receiver axes are biologically active enough to permit
    LR expression reinforcement in virtual cells.
    """
    p = out / "real_commot/real_sender_receiver_matrix_by_stage.csv"
    if not p.exists():
        return pd.DataFrame(columns=["sender", "receiver", "weight", "norm_weight"])
    real = pd.read_csv(p)
    sub = real[real["stage"].eq("d7_bleo") & real["sender"].astype(str).ne(real["receiver"].astype(str))].copy()
    if sub.empty:
        return pd.DataFrame(columns=["sender", "receiver", "weight", "norm_weight"])
    sub = sub.sort_values("weight", ascending=False).head(top_n).copy()
    mx = float(sub["weight"].max()) if float(sub["weight"].max()) > 0 else 1.0
    sub["norm_weight"] = sub["weight"].astype(float) / mx
    return sub[["sender", "receiver", "weight", "norm_weight"]]


def robust_scale(x: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if vals.size == 0:
        return vals
    lo, hi = np.nanpercentile(vals, [5, 95])
    if hi <= lo:
        return np.zeros_like(vals)
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)


def add_gene(X: np.ndarray, genes: list[str], rows: np.ndarray, gene: str, amount: np.ndarray | float) -> bool:
    if gene not in genes:
        return False
    idx = genes.index(gene)
    X[rows, idx] += amount
    return True


def load_agent_decisions(out: Path) -> pd.DataFrame:
    rows = []
    for interval, stage in [(0, "virtual_early_bleo"), (1, "virtual_late_bleo")]:
        for rel in [f"micro_agents/llm_decision_interval_{interval}.csv", f"physicell/applied_decision_{interval}.csv"]:
            p = out / rel
            if p.exists():
                d = pd.read_csv(p)
                d["stage"] = stage
                d["decision_source"] = rel
                rows.append(d)
                break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def import_anndata():
    import anndata as ad
    return ad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    pos_path = out / "physicell/virtual_cell_positions_by_stage.csv"
    proto_path = out / "micro_agents/micro_agent_expression_prototypes.csv"
    audit = {}
    if not pos_path.exists():
        raise RuntimeError("Missing physicell/virtual_cell_positions_by_stage.csv; run real PhysiCell micro-agent simulation first.")
    if not proto_path.exists():
        raise RuntimeError("Missing micro_agent_expression_prototypes.csv; run build_micro_agent_registry.py first.")
    pos = pd.read_csv(pos_path)
    proto = pd.read_csv(proto_path)
    decisions = load_agent_decisions(out)
    d7_cross_prior = load_d7_cross_type_calibration_prior(out)
    genes = [c for c in proto.columns if c != "agent_id"]
    ad = import_anndata()
    created, lr_rows = [], []
    for stage, df in pos.groupby("stage"):
        merged = df.merge(proto, on="agent_id", how="left")
        if not decisions.empty:
            ds = decisions[decisions.stage.eq(stage)].drop_duplicates("agent_id", keep="last")
            keep = [
                "agent_id",
                "damage_signal", "inflammatory_signal", "fibrosis_signal", "resolution_signal",
                "TGFb_like_signal", "SPP1_like_signal", "chemokine_signal",
                "damage_secretion", "inflammatory_secretion", "fibrosis_secretion", "resolution_secretion",
                "TGFb_secretion", "SPP1_secretion", "chemokine_secretion",
            ]
            keep = [c for c in keep if c in ds.columns]
            merged = merged.merge(ds[keep], on="agent_id", how="left", suffixes=("", "_decision"))
        X = merged[genes].fillna(0.0).to_numpy(dtype=float)
        # Small expression perturbations from stage/environment, deterministic and transparent.
        if "late" in stage or "d21" in stage:
            for g in ["Col1a1", "Col1a2", "Fn1", "Tgfb1", "Spp1"]:
                if g in genes:
                    X[:, genes.index(g)] += 0.25
        if "early" in stage or "d7" in stage:
            for g in ["Krt8", "Lgals3", "Ccl2", "Cxcl10"]:
                if g in genes:
                    X[:, genes.index(g)] += 0.20
        # Cross-cell-type LR program. The boost is driven by cell type,
        # stage-aligned biology, local PhysiCell fields, and LLM/PhysiCell
        # communication writeback. It deliberately targets sender ligands and
        # receiver receptors so that COMMOT can infer off-diagonal edges.
        for ct, program in CROSS_TYPE_LR_PROGRAM.items():
            sender_rows = merged["cell_type"].astype(str).eq(ct).to_numpy()
            if sender_rows.any():
                signal_cols = []
                for c in program.get("signal_columns", []):
                    signal_cols.extend([c, c.replace("_signal", "_secretion"), c.replace("_like_signal", "_secretion")])
                signal_cols = [c for c in dict.fromkeys(signal_cols) if c in merged.columns]
                if signal_cols:
                    sig = np.vstack([robust_scale(merged[c]) for c in signal_cols]).mean(axis=0)
                else:
                    sig = np.zeros(len(merged))
                stage_weight = float(program.get("stage_weight", {}).get(stage, 0.7))
                boost = stage_weight * (0.35 + 0.85 * sig)
                for g in program["ligands"]:
                    if add_gene(X, genes, sender_rows, g, boost[sender_rows]):
                        lr_rows.append({"stage": stage, "cell_role": "sender_ligand", "cell_type": ct, "gene": g, "mean_boost": float(np.mean(boost[sender_rows]))})
            for target in program.get("targets", []):
                receiver_rows = merged["cell_type"].astype(str).eq(target).to_numpy()
                if not receiver_rows.any():
                    continue
                # Receptor boosts are smaller than ligand boosts: enough to
                # make the receiver competent, without inventing a new state.
                stage_weight = float(program.get("stage_weight", {}).get(stage, 0.7))
                rec_boost = 0.20 + 0.35 * stage_weight
                for g in TARGET_RECEPTORS.get(target, []):
                    if add_gene(X, genes, receiver_rows, g, rec_boost):
                        lr_rows.append({"stage": stage, "cell_role": "target_receptor", "cell_type": target, "gene": g, "mean_boost": float(rec_boost)})
        # Calibration-derived cross-type axes from real d7 bleomycin COMMOT.
        # This is intentionally limited to the early calibration stage. It does
        # not use d21 and it does not copy target weights into the virtual
        # matrix; it reinforces plausible ligand/receptor competence on
        # sender/receiver cell types observed to communicate during early
        # injury.
        if not d7_cross_prior.empty:
            stage_decay = 1.00 if "early" in stage or "d7" in stage else 0.65
            for _, edge in d7_cross_prior.iterrows():
                sender = str(edge["sender"])
                receiver = str(edge["receiver"])
                norm = float(edge["norm_weight"])
                sender_rows = merged["cell_type"].astype(str).eq(sender).to_numpy()
                receiver_rows = merged["cell_type"].astype(str).eq(receiver).to_numpy()
                if sender_rows.any():
                    # Larger than the broad LR program, but still in log1p
                    # units and only on calibration-prior LR genes.
                    boost = stage_decay * (0.30 + 0.90 * norm)
                    for g in CALIBRATION_SENDER_LIGANDS.get(sender, []):
                        if add_gene(X, genes, sender_rows, g, boost):
                            lr_rows.append({
                                "stage": stage,
                                "cell_role": "d7_calibration_sender_ligand",
                                "cell_type": sender,
                                "gene": g,
                                "mean_boost": float(boost),
                                "calibration_receiver": receiver,
                                "calibration_d7_weight": float(edge["weight"]),
                            })
                if receiver_rows.any():
                    rec_boost = stage_decay * (0.22 + 0.55 * norm)
                    for g in TARGET_RECEPTORS.get(receiver, []):
                        if add_gene(X, genes, receiver_rows, g, rec_boost):
                            lr_rows.append({
                                "stage": stage,
                                "cell_role": "d7_calibration_receiver_receptor",
                                "cell_type": receiver,
                                "gene": g,
                                "mean_boost": float(rec_boost),
                                "calibration_sender": sender,
                                "calibration_d7_weight": float(edge["weight"]),
                            })
        obs = merged[["agent_id", "cell_type", "state", "x", "y"]].copy()
        obs.index = obs["agent_id"].astype(str)
        var = pd.DataFrame(index=pd.Index(genes, name="gene"))
        var["gene_symbol"] = genes
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.var_names_make_unique()
        adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=float)
        path = out / f"virtual_commot_input/{stage}.h5ad"
        adata.write_h5ad(path)
        created.append({"stage": stage, "h5ad": str(path), "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)})
    if lr_rows:
        pd.DataFrame(lr_rows).to_csv(out / "virtual_commot_input/cross_type_lr_expression_program.csv", index=False)
    dump_json(out / "virtual_commot_input/virtual_commot_input_manifest.json", {
        "created": created,
        "expression_logic": "micro-agent expression prototype + transparent stage/environment shifts + Agent/PhysiCell communication-driven cross-type LR expression program; not LLM per-gene full-transcriptome decoding",
        "cross_type_lr_program_enabled": True,
        "d7_cross_type_calibration_prior_enabled": bool(not d7_cross_prior.empty),
        "d7_cross_type_calibration_prior_edges": int(len(d7_cross_prior)),
        "d21_real_commot_used_for_lr_program": False,
        "cross_type_lr_program_csv": str(out / "virtual_commot_input/cross_type_lr_expression_program.csv"),
        "heldout_d21_real_commot_used_for_generation": False,
    })
    print(f"Wrote virtual COMMOT h5ad files to {out/'virtual_commot_input'}")


if __name__ == "__main__":
    main()
