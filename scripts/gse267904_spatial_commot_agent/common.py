#!/usr/bin/env python3
"""Shared utilities for the GSE267904 spatial COMMOT micro-agent experiment.

This module intentionally keeps the COMMOT and PhysiCell-backed experiment
strict: no COMMOT-like proxy is reported as COMMOT, and no Python toy spatial
simulator is reported as real PhysiCell.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


GSE = "GSE267904"
RAW_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE267904&format=file"

STAGES = ["d7_ctrl", "d7_bleo", "d21_ctrl", "d21_bleo"]
BLEO_STAGES = ["d7_bleo", "d21_bleo"]
STAGE_ALIASES = {
    "d7_ctrl": ["d7_ctrl", "d7_control", "day7_ctrl", "day7_control", "d7_saline", "day7_saline", "ctrl_d7", "saline_d7"],
    "d7_bleo": ["d7_bleo", "d7_bleomycin", "day7_bleo", "day7_bleomycin", "bleo_d7", "bleomycin_d7"],
    "d21_ctrl": ["d21_ctrl", "d21_control", "day21_ctrl", "day21_control", "d21_saline", "day21_saline", "ctrl_d21", "saline_d21"],
    "d21_bleo": ["d21_bleo", "d21_bleomycin", "day21_bleo", "day21_bleomycin", "bleo_d21", "bleomycin_d21"],
}

# Verified from NCBI GEO GSE267904 sample table:
# GSM8281384-1386 d7 ctrl; GSM8281387-1395 d7 bleo;
# GSM8281396-1398 d21 ctrl; GSM8281399-1407 d21 bleo.
GSM_TO_STAGE = {
    **{f"GSM828{i}": "d7_ctrl" for i in range(1384, 1387)},
    **{f"GSM828{i}": "d7_bleo" for i in range(1387, 1396)},
    **{f"GSM828{i}": "d21_ctrl" for i in range(1396, 1399)},
    **{f"GSM828{i}": "d21_bleo" for i in range(1399, 1408)},
}

CELL_TYPES = [
    "alveolar_epithelial_AT1_AT2",
    "activated_Krt8_ADI_epithelial",
    "airway_epithelial",
    "macrophage",
    "recruited_monocyte_macrophage",
    "fibroblast_myofibroblast",
    "endothelial",
    "dendritic",
    "lymphoid",
    "other",
]

CELL_TYPE_LABELS = {
    "alveolar_epithelial_AT1_AT2": "Alveolar epithelial",
    "activated_Krt8_ADI_epithelial": "Activated Krt8/ADI epithelial",
    "airway_epithelial": "Airway epithelial",
    "macrophage": "Macrophage",
    "recruited_monocyte_macrophage": "Recruited mono/mac",
    "fibroblast_myofibroblast": "Fibro/myofibroblast",
    "endothelial": "Endothelial",
    "dendritic": "Dendritic",
    "lymphoid": "Lymphoid",
    "other": "Other",
}

CELL_TYPE_COLORS = {
    "alveolar_epithelial_AT1_AT2": "#4e79a7",
    "activated_Krt8_ADI_epithelial": "#f28e2b",
    "airway_epithelial": "#ff9da7",
    "macrophage": "#59a14f",
    "recruited_monocyte_macrophage": "#e15759",
    "fibroblast_myofibroblast": "#b07aa1",
    "endothelial": "#76b7b2",
    "dendritic": "#edc948",
    "lymphoid": "#9c755f",
    "other": "#bab0ac",
}

MARKERS = {
    "alveolar_epithelial_AT1_AT2": ["Ager", "Hopx", "Sftpc", "Sftpa1", "Sftpa2", "Sftpb"],
    "activated_Krt8_ADI_epithelial": ["Krt8", "Krt18", "Krt19", "Clu", "Lgals3", "Sox4"],
    "airway_epithelial": ["Scgb1a1", "Foxj1", "Muc5b", "Krt5"],
    "macrophage": ["Lyz2", "Adgre1", "C1qa", "C1qb", "C1qc", "Mrc1"],
    "recruited_monocyte_macrophage": ["S100a8", "S100a9", "Ccr2", "Ly6c2", "Lgals3"],
    "fibroblast_myofibroblast": ["Col1a1", "Col1a2", "Col3a1", "Acta2", "Tagln", "Fn1"],
    "endothelial": ["Pecam1", "Kdr", "Cdh5", "Vwf"],
    "dendritic": ["Itgax", "Flt3", "H2-Ab1", "Cd74"],
    "lymphoid": ["Cd3d", "Cd3e", "Cd79a", "Ms4a1", "Nkg7"],
}

COMMOT_PATHWAYS = ["TGFb", "SPP1", "MIF", "CCL", "CXCL", "TNF", "IL1", "PDGF", "VEGF", "COLLAGEN"]
KEY_INTERACTIONS = [
    ("activated_Krt8_ADI_epithelial", "macrophage"),
    ("activated_Krt8_ADI_epithelial", "recruited_monocyte_macrophage"),
    ("macrophage", "fibroblast_myofibroblast"),
    ("fibroblast_myofibroblast", "alveolar_epithelial_AT1_AT2"),
    ("endothelial", "macrophage"),
    ("macrophage", "lymphoid"),
]

MICRO_AGENT_ALLOCATION = {
    "alveolar_epithelial_AT1_AT2": 8,
    "activated_Krt8_ADI_epithelial": 6,
    "macrophage": 8,
    "recruited_monocyte_macrophage": 6,
    "fibroblast_myofibroblast": 6,
    "endothelial": 4,
    "dendritic": 3,
    "lymphoid": 3,
    "airway_epithelial": 4,
    "other": 2,
}


def outpath(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def ensure_dirs(out: Path) -> None:
    for d in [
        "data",
        "real_benchmark",
        "celltype_annotation",
        "real_commot",
        "micro_agents",
        "physicell",
        "virtual_commot_input",
        "virtual_commot",
        "evaluation",
        "figures",
        "audit",
        "reports",
        "logs",
    ]:
        (out / d).mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def clean_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def infer_stage_from_path(path: Path) -> str | None:
    text = clean_token("_".join(path.parts[-6:]))
    for gsm, stage in GSM_TO_STAGE.items():
        if gsm.lower() in text:
            return stage
    for stage, aliases in STAGE_ALIASES.items():
        for alias in aliases:
            if clean_token(alias) in text:
                return stage
    # permissive fallback
    is_bleo = "bleo" in text or "bleomycin" in text
    is_ctrl = any(k in text for k in ["ctrl", "control", "saline", "pbs"])
    if ("d7" in text or "day7" in text or "7d" in text) and is_bleo:
        return "d7_bleo"
    if ("d7" in text or "day7" in text or "7d" in text) and is_ctrl:
        return "d7_ctrl"
    if ("d21" in text or "day21" in text or "21d" in text) and is_bleo:
        return "d21_bleo"
    if ("d21" in text or "day21" in text or "21d" in text) and is_ctrl:
        return "d21_ctrl"
    return None


def find_raw_root(project_root: Path, out: Path) -> Path:
    candidates = [
        project_root / "data/GSE267904/raw",
        project_root / "data/GSE267904",
        out / "data/raw",
        out / "data/GSE267904/raw",
    ]
    for c in candidates:
        if c.exists():
            return c
    return project_root / "data/GSE267904/raw"


def discover_10x_samples(raw_root: Path) -> list[dict[str, Any]]:
    """Discover Visium/10x-like sample directories.

    Supports flat GEO tar extracts and nested SpaceRanger-like folders.
    """
    if not raw_root.exists():
        return []
    # GEO often extracts Visium supplementary files flat into one directory:
    # GSMxxxx_sample_matrix.mtx.gz, GSMxxxx_sample_features.tsv.gz, ...
    # Detect and group those first. This avoids accidentally pairing a matrix
    # from one GSM with barcodes/features from another GSM.
    flat_files = [p for p in raw_root.glob("*") if p.is_file()]
    flat_groups: dict[str, dict[str, Path]] = {}
    for p in flat_files:
        m = re.match(r"^(GSM\d+_.+?)_(matrix\.mtx|features\.tsv|genes\.tsv|barcodes\.tsv|tissue_positions_list\.csv|tissue_positions\.csv|scalefactors_json\.json)(\.gz)?$", p.name)
        if not m:
            continue
        prefix, kind, _gz = m.groups()
        g = flat_groups.setdefault(prefix, {})
        if kind == "matrix.mtx":
            g["matrix"] = p
        elif kind in {"features.tsv", "genes.tsv"}:
            g["features"] = p
        elif kind == "barcodes.tsv":
            g["barcodes"] = p
        elif kind in {"tissue_positions_list.csv", "tissue_positions.csv"}:
            g["positions"] = p
        elif kind == "scalefactors_json.json":
            g["scalefactors"] = p
    flat_samples = []
    for prefix, g in sorted(flat_groups.items()):
        if {"matrix", "features", "barcodes", "positions"}.issubset(g):
            flat_samples.append(
                {
                    "sample_id": prefix,
                    "sample_dir": str(raw_root),
                    "stage": infer_stage_from_path(Path(prefix)),
                    "matrix": str(g["matrix"]),
                    "features": str(g["features"]),
                    "barcodes": str(g["barcodes"]),
                    "positions": str(g["positions"]),
                    "scalefactors": str(g["scalefactors"]) if "scalefactors" in g else None,
                    "flat_geo_files": True,
                }
            )
    if flat_samples:
        return flat_samples

    matrix_files = []
    for pat in ["**/matrix.mtx", "**/matrix.mtx.gz", "**/*matrix.mtx", "**/*matrix.mtx.gz"]:
        matrix_files.extend(raw_root.glob(pat))
    samples: list[dict[str, Any]] = []
    seen = set()
    for matrix in matrix_files:
        parent = matrix.parent
        key = str(parent.resolve())
        if key in seen:
            continue
        seen.add(key)
        files = list(parent.iterdir()) if parent.exists() else []
        feature = first_existing(parent, ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz", "*features*.tsv", "*features*.tsv.gz"])
        barcode = first_existing(parent, ["barcodes.tsv", "barcodes.tsv.gz", "*barcodes*.tsv", "*barcodes*.tsv.gz"])
        spatial_dir = parent / "spatial"
        position = first_existing(spatial_dir, ["tissue_positions_list.csv", "tissue_positions.csv", "tissue_positions_list.csv.gz", "tissue_positions.csv.gz", "*positions*.csv", "*positions*.csv.gz"]) if spatial_dir.exists() else None
        if position is None:
            position = first_existing(parent, ["tissue_positions_list.csv", "tissue_positions.csv", "tissue_positions_list.csv.gz", "tissue_positions.csv.gz", "*positions*.csv", "*positions*.csv.gz", "**/tissue_positions*.csv", "**/tissue_positions*.csv.gz"])
        stage = infer_stage_from_path(parent) or infer_stage_from_path(matrix)
        samples.append(
            {
                "sample_id": parent.name,
                "sample_dir": str(parent),
                "stage": stage,
                "matrix": str(matrix),
                "features": str(feature) if feature else None,
                "barcodes": str(barcode) if barcode else None,
                "positions": str(position) if position else None,
                "n_files_in_dir": len(files),
            }
        )
    # Fallback: GEO sometimes extracts sample-prefixed files into one dir.
    if not samples:
        all_files = list(raw_root.glob("**/*"))
        for matrix in [p for p in all_files if p.name.endswith(("matrix.mtx", "matrix.mtx.gz"))]:
            prefix = re.sub(r"(matrix\.mtx(\.gz)?)$", "", matrix.name)
            parent = matrix.parent
            stage = infer_stage_from_path(matrix)
            candidates = [p for p in parent.iterdir() if p.name.startswith(prefix)]
            feature = next((p for p in candidates if "feature" in p.name.lower() or "gene" in p.name.lower()), None)
            barcode = next((p for p in candidates if "barcode" in p.name.lower()), None)
            position = next((p for p in candidates if "position" in p.name.lower()), None)
            samples.append({"sample_id": prefix.strip("_."), "sample_dir": str(parent), "stage": stage, "matrix": str(matrix), "features": str(feature) if feature else None, "barcodes": str(barcode) if barcode else None, "positions": str(position) if position else None})
    return samples


def first_existing(base: Path, patterns: Iterable[str]) -> Path | None:
    for pat in patterns:
        direct = base / pat
        if "*" not in pat and direct.exists():
            return direct
        hits = list(base.glob(pat))
        if hits:
            return hits[0]
    return None


def read_features(path: Path) -> list[str]:
    genes = []
    with open_text(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                genes.append(parts[1])
            elif parts:
                genes.append(parts[0])
    return genes


def read_barcodes(path: Path) -> list[str]:
    with open_text(path) as fh:
        return [line.strip().split("\t")[0] for line in fh if line.strip()]


def read_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    # SpaceRanger v1 tissue_positions_list: barcode,in_tissue,array_row,array_col,pxl_row,pxl_col
    # SpaceRanger v2 has a header. Try both.
    if df.shape[1] >= 6 and str(df.iloc[0, 0]).lower() in {"barcode", "barcodes"}:
        df = pd.read_csv(path)
    if "barcode" in df.columns:
        out = pd.DataFrame(
            {
                "barcode": df["barcode"].astype(str),
                "in_tissue": pd.to_numeric(df.get("in_tissue", 1), errors="coerce").fillna(1).astype(int),
                "array_row": pd.to_numeric(df.get("array_row", df.get("row", 0)), errors="coerce"),
                "array_col": pd.to_numeric(df.get("array_col", df.get("col", 0)), errors="coerce"),
                "pxl_row": pd.to_numeric(df.get("pxl_row_in_fullres", df.get("pxl_row", df.get("imagerow", 0))), errors="coerce"),
                "pxl_col": pd.to_numeric(df.get("pxl_col_in_fullres", df.get("pxl_col", df.get("imagecol", 0))), errors="coerce"),
            }
        )
    else:
        out = pd.DataFrame(
            {
                "barcode": df.iloc[:, 0].astype(str),
                "in_tissue": pd.to_numeric(df.iloc[:, 1] if df.shape[1] > 1 else 1, errors="coerce").fillna(1).astype(int),
                "array_row": pd.to_numeric(df.iloc[:, 2] if df.shape[1] > 2 else 0, errors="coerce"),
                "array_col": pd.to_numeric(df.iloc[:, 3] if df.shape[1] > 3 else 0, errors="coerce"),
                "pxl_row": pd.to_numeric(df.iloc[:, 4] if df.shape[1] > 4 else 0, errors="coerce"),
                "pxl_col": pd.to_numeric(df.iloc[:, 5] if df.shape[1] > 5 else 0, errors="coerce"),
            }
        )
    return out


def normalize_rows(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    arr = arr - np.nanmin(arr, axis=1, keepdims=True)
    den = np.nanmax(arr, axis=1, keepdims=True) + eps
    return pd.DataFrame(arr / den, index=df.index, columns=df.columns)


def pearson(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    x = pd.Series(list(a), dtype=float).rank().to_numpy()
    y = pd.Series(list(b), dtype=float).rank().to_numpy()
    return pearson(x, y)


def l1_normalize(d: dict[str, float]) -> dict[str, float]:
    vals = {k: max(0.0, float(v)) for k, v in d.items()}
    s = sum(vals.values())
    if s <= 0:
        return {k: 1.0 / len(vals) for k in vals}
    return {k: v / s for k, v in vals.items()}


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_tar(tar_path: Path, raw_dir: Path) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(raw_dir)
    return {"tar_path": str(tar_path), "raw_dir": str(raw_dir), "n_files": sum(1 for _ in raw_dir.glob("**/*"))}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = ["empty"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
