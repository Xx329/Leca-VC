#!/usr/bin/env python3
"""Shared, leakage-aware utilities for fibrosis application V3."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.io import mmread


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = Path("outputs/GSE267904_fibrosis_application_v3")


def gse267904_root(project_root: Path) -> Path:
    """Resolve teammate data root without embedding a workstation path."""
    return Path(os.environ.get("GSE267904_DATA_ROOT", project_root / "data/GSE267904")).expanduser().resolve()
FIELDS = [
    "injury_signal",
    "epithelial_homeostasis",
    "inflammatory_signal",
    "TGFB_signal",
    "ECM_fibrosis",
    "macrophage_APOE_SPP1_signal",
]

SAMPLE_SPECS = {
    **{f"GSM828{i}": ("d7_ctrl", f"d7_ctrl_{i-1383}", "a") for i in range(1384, 1387)},
    "GSM8281387": ("d7_bleo", "d7_bleo_1", "a"),
    "GSM8281388": ("d7_bleo", "d7_bleo_2", "a"),
    "GSM8281389": ("d7_bleo", "d7_bleo_3", "a"),
    "GSM8281390": ("d7_bleo", "d7_bleo_4", "a"),
    "GSM8281391": ("d7_bleo", "d7_bleo_5", "a"),
    "GSM8281392": ("d7_bleo", "d7_bleo_6", "a"),
    "GSM8281393": ("d7_bleo", "d7_bleo_4", "b"),
    "GSM8281394": ("d7_bleo", "d7_bleo_5", "b"),
    "GSM8281395": ("d7_bleo", "d7_bleo_6", "b"),
    **{f"GSM828{i}": ("d21_ctrl", f"d21_ctrl_{i-1395}", "a") for i in range(1396, 1399)},
    "GSM8281399": ("d21_bleo", "d21_bleo_1", "a"),
    "GSM8281400": ("d21_bleo", "d21_bleo_2", "a"),
    "GSM8281401": ("d21_bleo", "d21_bleo_3", "a"),
    "GSM8281402": ("d21_bleo", "d21_bleo_4", "a"),
    "GSM8281403": ("d21_bleo", "d21_bleo_5", "a"),
    "GSM8281404": ("d21_bleo", "d21_bleo_6", "a"),
    "GSM8281405": ("d21_bleo", "d21_bleo_4", "b"),
    "GSM8281406": ("d21_bleo", "d21_bleo_5", "b"),
    "GSM8281407": ("d21_bleo", "d21_bleo_6", "b"),
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or HERE / "experiment.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_out(root: Path, out_dir: Path) -> Path:
    return out_dir if out_dir.is_absolute() else root / out_dir


def ensure_dirs(out: Path) -> None:
    for rel in [
        "audit", "manifests", "data/d7", "data/locked_benchmark", "fields/d7", "fields/locked_benchmark",
        "agents", "llm/policies", "llm/raw", "llm/checkpoints", "calibration", "physicell_project",
        "checkpoint_runs", "qualification", "runs", "evaluation", "figures", "reports", "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        h.update(str(path).encode())
        h.update(sha256(path).encode())
    return h.hexdigest()


def sample_id_from_name(name: str) -> str:
    match = re.match(r"^(GSM\d+_[^_]+)", name)
    if not match:
        raise ValueError(f"Cannot parse sample id from {name}")
    return match.group(1)


def short_sample(sample_id: str) -> str:
    return sample_id.split("_", 1)[1]


def discover_manifest(root: Path) -> pd.DataFrame:
    raw = gse267904_root(root) / "raw"
    rows = []
    for matrix in sorted(raw.glob("GSM*_matrix.mtx.gz")):
        sample_id = matrix.name[: -len("_matrix.mtx.gz")]
        gsm = sample_id.split("_", 1)[0]
        if gsm not in SAMPLE_SPECS:
            continue
        stage, animal_id, section = SAMPLE_SPECS[gsm]
        prefix = raw / sample_id
        required = {
            "matrix": matrix,
            "features": Path(str(prefix) + "_features.tsv.gz"),
            "barcodes": Path(str(prefix) + "_barcodes.tsv.gz"),
            "positions": Path(str(prefix) + "_tissue_positions_list.csv.gz"),
        }
        missing = [str(p) for p in required.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete raw sample {sample_id}: {missing}")
        rows.append({
            "sample_id": sample_id, "gsm": gsm, "section_id": short_sample(sample_id),
            "stage": stage, "day": 7 if stage.startswith("d7") else 21,
            "condition": "bleomycin" if stage.endswith("bleo") else "control",
            "animal_id": animal_id, "technical_section": section,
            **{key: str(value.resolve()) for key, value in required.items()},
        })
    manifest = pd.DataFrame(rows)
    expected = {"d7_ctrl": 3, "d7_bleo": 9, "d21_ctrl": 3, "d21_bleo": 9}
    if len(manifest) != 24 or manifest.groupby("stage").size().to_dict() != expected:
        raise RuntimeError(f"Expected 24 GSE267904 sections {expected}; found {manifest.groupby('stage').size().to_dict()}")
    return manifest


def ensure_cell2location(root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    data_root = gse267904_root(root)
    archive = data_root / "cell2location_strunz2020.zip"
    if not archive.exists():
        raise FileNotFoundError(
            f"Missing official cell2location archive {archive}. Download from {config['cell2location']['url']}"
        )
    actual = sha256(archive)
    expected = str(config["cell2location"]["archive_sha256"])
    if actual != expected:
        raise RuntimeError(f"Official cell2location SHA256 mismatch: expected {expected}, got {actual}")
    target = data_root / "cell2location_official"
    files = [p for p in target.glob("**/*_spot_cell_abundances_5pc.csv") if "__MACOSX" not in p.parts]
    if len(files) != 24:
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            safe = [n for n in zf.namelist() if not n.startswith("__MACOSX/") and ".." not in Path(n).parts]
            for name in safe:
                zf.extract(name, target)
        files = [p for p in target.glob("**/*_spot_cell_abundances_5pc.csv") if "__MACOSX" not in p.parts]
    if len(files) != 24:
        raise RuntimeError(f"Expected 24 official cell2location CSVs, found {len(files)}")
    return target, {
        "accession": config["cell2location"]["accession"], "url": config["cell2location"]["url"],
        "archive": str(archive.resolve()), "sha256": actual, "n_files": len(files),
    }


def read_tissue_positions(path: Path) -> pd.DataFrame:
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    return pd.read_csv(path, compression="gzip", header=None, names=cols)


def read_visium_tissue(row: pd.Series) -> tuple[Any, pd.DataFrame]:
    import anndata as ad

    with gzip.open(row["features"], "rt", encoding="utf-8") as handle:
        features = pd.read_csv(handle, sep="\t", header=None)
    with gzip.open(row["barcodes"], "rt", encoding="utf-8") as handle:
        barcodes = pd.read_csv(handle, sep="\t", header=None)[0].astype(str)
    with gzip.open(row["matrix"], "rb") as handle:
        matrix = mmread(handle).tocsr()
    if matrix.shape == (len(features), len(barcodes)):
        matrix = matrix.T.tocsr()
    if matrix.shape != (len(barcodes), len(features)):
        raise RuntimeError(f"Unexpected matrix shape for {row['sample_id']}: {matrix.shape}")
    positions = read_tissue_positions(Path(row["positions"])).set_index("barcode").reindex(barcodes)
    keep = positions["in_tissue"].fillna(0).astype(int).to_numpy() == 1
    obs = positions.loc[keep].copy()
    obs.index = barcodes[keep]
    obs["sample_id"] = row["sample_id"]
    obs["stage"] = row["stage"]
    obs["animal_id"] = row["animal_id"]
    raw_names = (features.iloc[:, 1] if features.shape[1] > 1 else features.iloc[:, 0]).astype(str).tolist()
    seen: dict[str, int] = {}
    unique_names = []
    for name in raw_names:
        count = seen.get(name, 0)
        unique_names.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    var_names = pd.Index(unique_names, name=None)
    adata = ad.AnnData(matrix[keep], obs=obs, var=pd.DataFrame(index=var_names))
    spatial = obs[["pxl_col", "pxl_row"]].to_numpy(dtype=float)
    adata.obsm["spatial"] = spatial
    return adata, obs


def load_cell2location(sample_id: str, source_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    section = short_sample(sample_id)
    matches = [p for p in source_dir.glob(f"**/{section}_spot_cell_abundances_5pc.csv") if "__MACOSX" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one official cell2location file for {section}, found {matches}")
    raw = pd.read_csv(matches[0])
    raw["barcode"] = raw["spot_id"].astype(str).str.replace(section + "_", "", regex=False)
    result = pd.DataFrame({"barcode": raw["barcode"]})
    prefix = "q05cell_abundance_w_sf_"
    for cell_type, spec in config["cell_types"].items():
        wanted = [prefix + name for name in spec["cell2location"]]
        missing = [name for name in wanted if name not in raw.columns]
        if missing:
            raise RuntimeError(f"Official cell2location file {matches[0]} missing columns {missing}")
        result[cell_type] = raw[wanted].sum(axis=1).clip(lower=0)
    return result.set_index("barcode")


def coordinate_transform(xy: np.ndarray, target_radius: float = 2200.0) -> dict[str, float]:
    low, high = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
    center = (low + high) / 2.0
    scale = 2.0 * target_radius / max(float(np.max(high - low)), 1.0)
    return {"center_x": float(center[0]), "center_y": float(center[1]), "scale": float(scale)}


def transform_xy(xy: np.ndarray, transform: dict[str, float]) -> np.ndarray:
    return (xy - np.array([transform["center_x"], transform["center_y"]])) * transform["scale"]


def estimate_spot_spacing(xy: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(distances[:, 1]))
