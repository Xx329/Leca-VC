#!/usr/bin/env python3
"""Create healthy PBS cell-type prototypes and a normalized worker registry."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v1"
AUDIT = OUT / "audit"
INIT = OUT / "initialization"
G141 = ROOT / "data/GSE141259"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt") as f:
        return [x.strip() for x in f if x.strip()]


def map_type(cell_type: str, meta: str) -> str | None:
    ct, mt = str(cell_type).lower(), str(meta).lower()
    text = ct + " " + mt
    if "krt8" in text or "activated at2" in text:
        return "injured_Krt8_epithelial"
    if "at1" in ct:
        return "AT1"
    if "at2" in ct:
        return "AT2"
    if "am (pbs)" in ct or ("macrophage" in text and "recruited" not in text and "fn1+" not in text):
        return "resident_macrophage"
    if "monocyte" in text or "recruited macrophage" in text or "fn1+ macrophage" in text:
        return "recruited_monocyte_macrophage"
    if "neutrophil" in text:
        return "neutrophil"
    if "endothelial" in text or ct in {"vecs", "cecs", "lecs", "vcam1+ vecs"}:
        return "endothelial"
    if "fibroblast" in text or "myofibroblast" in text or "smcs" in ct:
        return "fibroblast_stromal"
    if any(k in text for k in ["t-lymph", "b-lymph", "nk cells", "plasma cells", "t cell subset"]):
        return "lymphoid_immune"
    if any(k in text for k in ["club cells", "ciliated", "goblet"]):
        return "airway_epithelial"
    return None


def main() -> None:
    config = yaml.safe_load((HERE / "experiment.yaml").read_text())
    agent_types = config["agent_types"]
    INIT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(AUDIT / "gene_mapping_report.csv")
    common = sorted(set(mapping.loc[mapping.present_in_gse141259_wholelung & mapping.primary_gene_symbol.notna(), "primary_gene_symbol"].astype(str)))
    common_upper = {g.upper(): g for g in common}

    genes = read_lines(G141 / "GSE141259_WholeLung_genes.txt.gz")
    barcodes = read_lines(G141 / "GSE141259_WholeLung_barcodes.txt.gz")
    meta = pd.read_csv(G141 / "GSE141259_WholeLung_cellinfo.csv.gz")
    if "Unnamed: 0" in meta:
        meta = meta.rename(columns={"Unnamed: 0": "cell_id"})
    meta = meta.set_index("cell_id").reindex(barcodes)
    pbs = meta.grouping.eq("PBS").to_numpy()
    labels = np.array([map_type(a, b) for a, b in zip(meta["cell.type"], meta["metacelltype"])], dtype=object)

    mat = mmread(G141 / "GSE141259_WholeLung_rawcounts.mtx.gz").tocsc().astype(float)
    gene_by_upper = {g.upper(): i for i, g in enumerate(genes)}
    selected_symbols = [g for g in common if g.upper() in gene_by_upper]
    selected_idx = [gene_by_upper[g.upper()] for g in selected_symbols]
    sub = mat[selected_idx, :]
    library = np.asarray(mat.sum(axis=0)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    sub = sub @ sparse.diags(scale)
    sub.data = np.log1p(sub.data)

    prototypes = []
    counts = {}
    for agent in agent_types:
        idx = np.where(pbs & (labels == agent))[0]
        counts[agent] = int(len(idx))
        if len(idx):
            prototypes.append(np.asarray(sub[:, idx].mean(axis=1)).ravel())
        else:
            prototypes.append(np.zeros(len(selected_symbols)))
    prototypes = np.vstack(prototypes)
    np.savez_compressed(INIT / "healthy_expression_prototypes.npz", genes=np.array(selected_symbols), agent_types=np.array(agent_types), expression=prototypes)

    raw_abundance = np.array([max(counts[a], 5) for a in agent_types], dtype=float)
    raw_abundance /= raw_abundance.sum()
    workers_per = int(config["workers_per_agent"])
    rng = np.random.default_rng(config["seeds"]["smoke"][0])
    rows = []
    for agent, abundance in zip(agent_types, raw_abundance):
        for j in range(workers_per):
            angle = rng.uniform(0, 2 * np.pi)
            radius = 520 * np.sqrt(rng.uniform())
            rows.append({
                "worker_id": f"{agent}__worker_{j+1:02d}",
                "agent_id": agent,
                "cell_type": agent,
                "x": radius * np.cos(angle),
                "y": radius * np.sin(angle),
                "represented_abundance": abundance / workers_per,
            })
    registry = pd.DataFrame(rows)
    registry.to_csv(INIT / "worker_registry.csv", index=False)
    pd.DataFrame({"agent_type": agent_types, "pbs_cells": [counts[a] for a in agent_types], "represented_abundance": raw_abundance}).to_csv(AUDIT / "healthy_reference_agent_audit.csv", index=False)
    lineage = {
        "reference": "GSE141259 WholeLung PBS only",
        "raw_counts_sha256": sha256(G141 / "GSE141259_WholeLung_rawcounts.mtx.gz"),
        "cellinfo_sha256": sha256(G141 / "GSE141259_WholeLung_cellinfo.csv.gz"),
        "pbs_cells_used": int(pbs.sum()),
        "agent_mapping_counts": counts,
        "common_genes": len(selected_symbols),
        "workers": len(registry),
        "represented_abundance_sum": float(registry.represented_abundance.sum()),
        "gse267904_spatial_reference_used_in_smoke": False,
        "injury_dynamics_borrowed_from_gse141259_or_gse267904": False,
    }
    (AUDIT / "healthy_reference_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    if len(selected_symbols) < config["qualification"]["common_gene_coverage_min"]:
        raise RuntimeError(f"Common gene coverage {len(selected_symbols)} below gate")
    if abs(registry.represented_abundance.sum() - 1.0) > config["qualification"]["abundance_relative_error_max"]:
        raise RuntimeError("represented_abundance does not sum to one")
    print(json.dumps(lineage))


if __name__ == "__main__":
    main()
