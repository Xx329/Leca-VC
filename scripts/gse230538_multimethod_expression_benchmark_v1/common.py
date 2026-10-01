#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/GSE230538_multimethod_expression_benchmark_v1_v7_frozen"
EXTERNAL_ROOT = ROOT / "data/GSE230538 others/vcmodelreport"
RAW_ROOT = ROOT / "data/GSE230538/raw"

SCGEN_ROOT = EXTERNAL_ROOT / "scgen_gse230538_pre_to_days"
WOT_ROOT = EXTERNAL_ROOT / "wot_gse230538_untreated_to_days_native"
CELLRANK_ROOT = EXTERNAL_ROOT / "cellrank_gse230538_pre_to_days_native"
V7_ROOT = ROOT / "outputs/GSE230538_v7_dynamic_agent_benchmark"

SCGEN_AGGREGATE = (
    SCGEN_ROOT / "figures/backing_csv/aggregate_gene_expression.csv.gz"
)
CELLRANK_H5AD = CELLRANK_ROOT / "native_cellrank_pre_terminal_status_all_genes.h5ad"

CELL_LINES = ("IPC-298", "M20", "MEL-JUSO", "SK-MEL-30")
HELDOUT_DAYS = (4, 33)
METHODS = ("scGen", "WOT", "CellRank", "AgentVC")
METHOD_COLORS = {
    "scGen": "#F58518",
    "WOT": "#54A24B",
    "CellRank": "#4C9ED9",
    "AgentVC": "#A66FA0",
}
CELL_LINE_COLORS = {
    "IPC-298": "#0072B2",
    "M20": "#D55E00",
    "MEL-JUSO": "#009E73",
    "SK-MEL-30": "#CC79A7",
}

STATES = (
    "proliferative_sensitive",
    "early_drug_response",
    "P2RX7_ion_signal_response",
    "senescent_like",
    "immune_like_resistant",
    "intrinsic_resistant",
    "stress_adaptive",
    "apoptotic_or_dying",
)

SAMPLE_MAP = {
    "GSM7226481": ("MEL-JUSO", 0),
    "GSM7226482": ("MEL-JUSO", 1),
    "GSM7226483": ("MEL-JUSO", 4),
    "GSM7226484": ("MEL-JUSO", 33),
    "GSM7226485": ("IPC-298", 0),
    "GSM7226486": ("IPC-298", 1),
    "GSM7226487": ("IPC-298", 4),
    "GSM7226488": ("IPC-298", 33),
    "GSM7226489": ("M20", 0),
    "GSM7226490": ("M20", 1),
    "GSM7226491": ("M20", 4),
    "GSM7226492": ("M20", 33),
    "GSM7226493": ("SK-MEL-30", 0),
    "GSM7226494": ("SK-MEL-30", 1),
    "GSM7226495": ("SK-MEL-30", 4),
    "GSM7226496": ("SK-MEL-30", 33),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def raw_files() -> list[Path]:
    paths = sorted(RAW_ROOT.glob("GSM*_DGE.txt.gz"))
    require(len(paths) == 16, f"Expected 16 GSE230538 DGE files, found {len(paths)}")
    return paths


def sample_info(path: Path) -> tuple[str, str, int]:
    gsm = path.name.split("_", 1)[0]
    require(gsm in SAMPLE_MAP, f"Unregistered GSE230538 sample: {gsm}")
    cell_line, day = SAMPLE_MAP[gsm]
    return gsm, cell_line, day


def wot_h5ads(days: Iterable[int] = HELDOUT_DAYS) -> list[Path]:
    result: list[Path] = []
    for day in days:
        result.extend(
            sorted(
                WOT_ROOT.glob(
                    f"wot/pairs/*__day_{day}/*_transport_projected_status_all_genes.h5ad"
                )
            )
        )
    expected = len(tuple(days)) * len(CELL_LINES)
    require(len(result) == expected, f"Expected {expected} WOT H5ADs, found {len(result)}")
    return result


def read_gene_panel(out: Path) -> list[str]:
    path = out / "00_freeze/frozen_7576_gene_panel.txt"
    require(path.exists(), f"Missing frozen gene panel: {path}")
    genes = path.read_text(encoding="utf-8").splitlines()
    require(len(genes) == 7576 and len(set(genes)) == 7576, "Frozen gene panel is invalid")
    return genes


def read_dge_logcp10k(path: Path, genes: list[str]) -> tuple[list[str], np.ndarray, dict]:
    """Read selected genes and return cells x genes mean-compatible log1p(CP10K)."""
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        cell_names = header[1:]
        n_cells = len(cell_names)
        require(n_cells > 0, f"No cells in {path}")
        selected = np.zeros((n_cells, len(genes)), dtype=np.float32)
        library_sum = np.zeros(n_cells, dtype=np.float64)
        seen: set[str] = set()
        for line in handle:
            gene, values_text = line.rstrip("\r\n").split("\t", 1)
            values = np.fromstring(values_text, sep="\t", dtype=np.float64)
            require(len(values) == n_cells, f"Malformed DGE row {gene} in {path}")
            library_sum += values
            index = gene_to_index.get(gene)
            if index is not None:
                selected[:, index] = values.astype(np.float32)
                seen.add(gene)
    missing_genes = [gene for gene in genes if gene not in seen]
    positive = library_sum > 0
    scale = np.zeros(n_cells, dtype=np.float32)
    scale[positive] = (10000.0 / library_sum[positive]).astype(np.float32)
    selected *= scale[:, None]
    np.log1p(selected, out=selected)
    require(np.isfinite(selected).all() and np.all(selected >= 0), f"Invalid logCP10K in {path}")
    sample_id = path.name.removesuffix("_DGE.txt.gz")
    return [f"{sample_id}:{name}" for name in cell_names], selected, {
        "cell_count": n_cells,
        "positive_library_cells": int(positive.sum()),
        "zero_library_cells_retained_as_zero": int((~positive).sum()),
        "panel_genes_present": len(seen),
        "panel_genes_zero_filled": len(missing_genes),
        "zero_filled_gene_examples": missing_genes[:20],
        "normalization": "positive libraries scaled to CP10K; zero libraries retained as all-zero",
    }


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = x - x.mean()
    dy = y - y.mean()
    denominator = np.mean(dx * dx) + np.mean(dy * dy) + (x.mean() - y.mean()) ** 2
    return float(2.0 * np.mean(dx * dy) / denominator) if denominator > 0 else float("nan")


def simplex_projection_columns(matrix: np.ndarray, target_means: np.ndarray) -> np.ndarray:
    """Project each nonnegative column onto a simplex with an exact target mean."""
    x = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(target_means, dtype=np.float64)
    require(x.ndim == 2 and targets.shape == (x.shape[1],), "Simplex shape mismatch")
    require(np.isfinite(x).all() and np.all(x >= 0), "Invalid simplex source")
    require(np.isfinite(targets).all() and np.all(targets >= 0), "Invalid simplex target")
    n_rows = x.shape[0]
    sums = targets * n_rows
    order = np.sort(x, axis=0)[::-1]
    cssv = np.cumsum(order, axis=0) - sums[None, :]
    indices = np.arange(1, n_rows + 1, dtype=np.float64)[:, None]
    active = order - cssv / indices > 0
    rho = active.sum(axis=0) - 1
    rho = np.maximum(rho, 0)
    theta = cssv[rho, np.arange(x.shape[1])] / (rho + 1.0)
    projected = np.maximum(x - theta[None, :], 0.0)
    # Correct the tiny floating residual without changing rank ordering.
    residual = sums - projected.sum(axis=0)
    positive = projected > 0
    counts = positive.sum(axis=0)
    for column in np.where(np.abs(residual) > 1e-12)[0]:
        if counts[column] > 0:
            projected[positive[:, column], column] += residual[column] / counts[column]
        elif sums[column] > 0:
            projected[:, column] = sums[column] / n_rows
    projected = np.maximum(projected, 0.0)
    error = np.max(np.abs(projected.mean(axis=0) - targets))
    require(error <= 1e-10, f"Simplex target mean error {error} exceeds 1e-10")
    return projected
