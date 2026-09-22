#!/usr/bin/env python3
"""Build leakage-safe early time-course calibration targets from public raw data."""
from __future__ import annotations

import gzip
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.io import mmread

from common import dump_json, sha256


GSE264278_FILES = {
    0: ("GSM8215373", "GSM8215373_matrix_inflection_demulti_day00UT.txt.gz"),
    3: ("GSM8215375", "GSM8215375_matrix_inflection_demulti_day03BLM.txt.gz"),
    7: ("GSM8215377", "GSM8215377_matrix_inflection_demulti_day07BLM.txt.gz"),
    14: ("GSM8215379", "GSM8215379_matrix_inflection_demulti_day14BLM.txt.gz"),
}


def _module_lookup(config: dict) -> tuple[dict[str, str], dict[str, set[str]]]:
    by_gene = {}
    modules = {}
    for field, genes in config["module_genes"].items():
        modules[field] = {str(g).lower() for g in genes}
        for gene in genes:
            by_gene[str(gene).lower()] = field
    return by_gene, modules


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    last_error = None
    for attempt in range(1, 7):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, stream=True, headers=headers, timeout=(30, 600)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                with partial.open("ab" if append else "wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            partial.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            if attempt < 6:
                time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"Download failed after 6 resumable attempts: {url}") from last_error


def ensure_gse264278(root: Path, download: bool) -> tuple[dict[int, Path], list[dict]]:
    directory = root / "data/GSE264278"
    paths, audit = {}, []
    for day, (gsm, name) in GSE264278_FILES.items():
        path = directory / name
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8215nnn/{gsm}/suppl/{name}"
        if not path.exists() and download:
            print(f"Downloading official GSE264278 day {day}: {url}", flush=True)
            _download(url, path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Re-run calibrate with --download-external, or download the official GEO file from {url}"
            )
        paths[day] = path
        audit.append({"dataset": "GSE264278", "day": day, "path": str(path.resolve()), "url": url, "sha256": sha256(path)})
    return paths, audit


def gse141259_targets(root: Path, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    directory = root / "data/GSE141259"
    paths = {
        "matrix": directory / "GSE141259_WholeLung_rawcounts.mtx.gz",
        "genes": directory / "GSE141259_WholeLung_genes.txt.gz",
        "barcodes": directory / "GSE141259_WholeLung_barcodes.txt.gz",
        "metadata": directory / "GSE141259_WholeLung_cellinfo.csv.gz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing official GSE141259 files: {missing}")
    genes = pd.read_csv(paths["genes"], sep="\t", header=None)[0].astype(str)
    barcodes = pd.read_csv(paths["barcodes"], sep="\t", header=None)[0].astype(str)
    metadata = pd.read_csv(paths["metadata"]).set_index("Unnamed: 0").reindex(barcodes)
    if metadata["identifier"].isna().any():
        raise RuntimeError("GSE141259 barcodes and cell metadata do not align")
    with gzip.open(paths["matrix"], "rb") as handle:
        matrix = mmread(handle).tocsr()
    if matrix.shape != (len(genes), len(barcodes)):
        raise RuntimeError(f"Unexpected GSE141259 matrix shape {matrix.shape}")
    cells_by_genes = matrix.T.tocsr().astype(np.float64)
    library = np.asarray(cells_by_genes.sum(axis=1)).ravel()
    normalized = sparse.diags(1e4 / np.maximum(library, 1.0)) @ cells_by_genes
    normalized.data = np.log1p(normalized.data)
    lower = pd.Series(np.arange(len(genes)), index=genes.str.lower()).groupby(level=0).first()
    sample = metadata["identifier"].astype(str)
    grouping = metadata["grouping"].astype(str)
    actual_day = sample.str.extract(r"_d(\d+)$")[0].astype(int)
    keep = actual_day.le(14) & (grouping.eq("PBS") | grouping.str.fullmatch(r"d(?:3|7|10|14)"))
    rows = []
    for field, wanted in config["module_genes"].items():
        indices = [int(lower[g.lower()]) for g in wanted if g.lower() in lower.index]
        if len(indices) < 4:
            raise RuntimeError(f"GSE141259 {field} has only {len(indices)} module genes")
        value = np.asarray(normalized[:, indices].mean(axis=1)).ravel()
        frame = pd.DataFrame({"sample": sample, "grouping": grouping, "keep": keep, "activity": value})
        grouped = frame[frame.keep].groupby(["sample", "grouping"], as_index=False).activity.mean()
        grouped["day"] = grouped["grouping"].map(lambda x: 0 if x == "PBS" else int(x[1:]))
        grouped["field"] = field
        rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    result["dataset"] = "GSE141259"
    audit = [{"dataset": "GSE141259", "role": key, "path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()]
    return result[["dataset", "sample", "day", "field", "activity"]], audit


def _dense_text_target(path: Path, config: dict, day: int) -> list[dict]:
    by_gene, modules = _module_lookup(config)
    header = pd.read_csv(path, sep="\t", nrows=0).columns.astype(str).tolist()
    module_columns = {field: [col for col in header if col.lower() in genes] for field, genes in modules.items()}
    cell_rows = sum(len(v) for v in module_columns.values()) >= 4
    totals = 0.0
    module_totals = {field: 0.0 for field in modules}
    present = {field: set() for field in modules}
    if cell_rows:
        for chunk in pd.read_csv(path, sep="\t", chunksize=2000):
            numeric = chunk.select_dtypes(include=[np.number])
            totals += float(numeric.to_numpy(dtype=float).sum())
            for field, columns in module_columns.items():
                existing = [col for col in columns if col in chunk]
                present[field].update(col.lower() for col in existing)
                module_totals[field] += float(chunk[existing].to_numpy(dtype=float).sum())
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            handle.readline()
            for line_number, line in enumerate(handle, start=2):
                gene, separator, values = line.partition("\t")
                if not separator:
                    continue
                numeric = np.fromstring(values, sep="\t", dtype=np.float64)
                if numeric.size == 0:
                    raise RuntimeError(f"Could not parse numeric GSE264278 row {line_number} in {path}")
                row_total = float(numeric.sum())
                totals += row_total
                field = by_gene.get(gene.lower())
                if field:
                    present[field].add(gene.lower())
                    module_totals[field] += row_total
    rows = []
    for field, genes in present.items():
        if len(genes) < 4:
            raise RuntimeError(f"GSE264278 day {day} {field} has only {len(genes)} module genes")
        activity = np.log1p(1e4 * module_totals[field] / max(totals, 1.0) / len(genes))
        rows.append({"dataset": "GSE264278", "sample": f"pooled_day{day}", "day": day, "field": field, "activity": activity})
    return rows


def build_external_summary(root: Path, out: Path, config: dict, download: bool = False) -> Path:
    print("Parsing GSE141259 day0/3/7/10/14 calibration data", flush=True)
    gse141, audit141 = gse141259_targets(root, config)
    paths264, audit264 = ensure_gse264278(root, download)
    rows264 = []
    for day, path in paths264.items():
        print(f"Parsing GSE264278 day {day}: {path.name}", flush=True)
        rows264.extend(_dense_text_target(path, config, day))
    gse264 = pd.DataFrame(rows264)
    table = pd.concat([gse141, gse264], ignore_index=True)
    baselines = table[table.day.eq(0)].groupby(["dataset", "field"])["activity"].mean().rename("baseline")
    table = table.join(baselines, on=["dataset", "field"])
    if table["baseline"].isna().any():
        raise RuntimeError("Each external dataset/field requires a day-0 baseline")
    table["relative_activity"] = np.exp(table["activity"] - table["baseline"])
    destination = out / "calibration/external_day0_to14_module_trajectories.csv"
    table.to_csv(destination, index=False)
    dump_json(out / "audit/external_calibration_sources.json", {
        "day21_or_later_opened": False, "allowed_days": [0, 3, 7, 10, 14],
        "sources": audit141 + audit264, "summary_sha256": sha256(destination),
    })
    return destination
