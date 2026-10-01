#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODULES = {
    "CD8_T_cell": ["CD8A", "CD8B", "GZMK", "GZMA"],
    "Cytotoxic_T_cell": ["GZMB", "PRF1", "IFNG", "NKG7", "GNLY"],
    "Memory_like_T_cell": ["TCF7", "IL7R", "LEF1", "CCR7", "SELL"],
    "Exhausted_T_cell": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "Macrophage_Monocyte": ["LYZ", "CD14", "FCGR3A", "MS4A7", "LST1"],
    "Dendritic_cell": ["FCER1A", "CLEC10A", "LILRA4", "IRF8", "ITGAX"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "CD74"],
    "NK_cell": ["NKG7", "GNLY", "KLRD1", "FCGR3A"],
}
STATES = list(MODULES)
RESPONSE_GROUPS = ["Responder", "Non-responder"]
RESPONSE_SCORE_MODULES = {
    "positive": ["Cytotoxic_T_cell", "Memory_like_T_cell"],
    "negative": ["Exhausted_T_cell", "Treg"],
}
RAW_EXPR_DEFAULT = Path("data/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO(1).txt/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt")


def outpath(root: Path, out: Path) -> Path:
    return out if out.is_absolute() else root / out


def dump(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


def resolve_expr_path(root: Path, expr: Path | None = None) -> Path:
    p = expr or RAW_EXPR_DEFAULT
    p = p if p.is_absolute() else root / p
    if p.is_file():
        return p
    gz = Path(str(p) + ".gz")
    if gz.is_file():
        return gz
    raise FileNotFoundError(f"GSE120575 raw expression matrix not found: {p}")


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=0)) / df.std(axis=0, ddof=0).replace(0, np.nan)


def normalize_props(d: dict[str, float]) -> dict[str, float]:
    vals = {s: max(0.0, float(d.get(s, 0.0))) for s in STATES}
    total = sum(vals.values())
    if total <= 0:
        return {s: 1.0 / len(STATES) for s in STATES}
    return {s: v / total for s, v in vals.items()}


def response_score(module_scores: dict[str, float]) -> float:
    return float(
        sum(module_scores.get(m, 0.0) for m in RESPONSE_SCORE_MODULES["positive"])
        - sum(module_scores.get(m, 0.0) for m in RESPONSE_SCORE_MODULES["negative"])
    )


def module_scores_from_gene_means(gene_means: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for module, genes in MODULES.items():
        usable = [g for g in genes if g in gene_means.index]
        out[module] = float(gene_means[usable].mean()) if usable else float("nan")
    out["response_score"] = response_score(out)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

