#!/usr/bin/env python3
"""GSE230538 old-scale multimethod expression comparison.

This visualization-only/post-hoc V2 keeps the frozen V7 AgentVC trajectories
and re-harmonizes expression to the GSE120575 ordering:

    aggregate CP10K-like expression first, then apply log1p.

The local scGen handoff does not contain its generated-cell H5ADs.  Therefore
scGen is represented by an explicitly audited Day0-dispersion-adjusted
pseudo-bulk proxy derived from its frozen mean-log output.  No held-out Day4 or
Day33 expression is used for that adjustment.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gse230538-v2-old-scale-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/gse230538-v2-old-scale-numba")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
V1_SCRIPT = ROOT / "scripts/gse230538_multimethod_expression_benchmark_v1"
sys.path.insert(0, str(V1_SCRIPT))
import common as v1c  # noqa: E402
import figures as v1f  # noqa: E402


OUT = ROOT / "outputs/GSE230538_multimethod_expression_benchmark_v2_old_scale_v7_frozen"
V1_OUT = ROOT / "outputs/GSE230538_multimethod_expression_benchmark_v1_v7_frozen"
PANEL_NAME = "frozen_6336_exact_common_gene_panel.txt"
ARCHIVE_NAME = "harmonized_6336_old_scale_expression.npz"
METHODS = v1c.METHODS
COLORS = v1c.METHOD_COLORS
LABELS = {"scGen": "scGen‡", "WOT": "WOT*", "CellRank": "CellRank†", "AgentVC": "AgentVC"}
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 2305386336


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx, dy = x - x.mean(), y - y.mean()
    denominator = np.mean(dx * dx) + np.mean(dy * dy) + (x.mean() - y.mean()) ** 2
    return float(2 * np.mean(dx * dy) / denominator) if denominator > 0 else float("nan")


def read_cp10k(path: Path, genes: list[str]) -> tuple[list[str], np.ndarray, dict]:
    positions = {gene: index for index, gene in enumerate(genes)}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        cells = header[1:]
        matrix = np.zeros((len(cells), len(genes)), dtype=np.float32)
        library = np.zeros(len(cells), dtype=np.float64)
        seen: set[str] = set()
        for line in handle:
            gene, text = line.rstrip("\r\n").split("\t", 1)
            values = np.fromstring(text, sep="\t", dtype=np.float64)
            require(len(values) == len(cells), f"Malformed row {gene}: {path}")
            library += values
            index = positions.get(gene)
            if index is not None:
                matrix[:, index] = values.astype(np.float32)
                seen.add(gene)
    positive = library > 0
    matrix[positive] *= (10000.0 / library[positive]).astype(np.float32)[:, None]
    require(np.isfinite(matrix).all() and np.all(matrix >= 0), f"Invalid CP10K: {path}")
    sample = path.name.removesuffix("_DGE.txt.gz")
    return [f"{sample}:{cell}" for cell in cells], matrix, {
        "cells": len(cells),
        "zero_library_cells": int((~positive).sum()),
        "present_genes": len(seen),
    }


def exact_common_panel() -> list[str]:
    union = v1c.read_gene_panel(V1_OUT)
    frame = pd.read_csv(v1c.SCGEN_AGGREGATE)
    masks = []
    for day in v1c.HELDOUT_DAYS:
        for cell_line in v1c.CELL_LINES:
            selected = frame[
                frame["target_condition"].eq(f"day_{day}")
                & frame["held_out_group"].eq(cell_line)
            ].drop_duplicates("gene").set_index("gene")["native_scgen_predicted_mean"]
            masks.append(selected.reindex(union).notna().to_numpy())
    common = np.logical_and.reduce(masks)
    genes = [gene for gene, keep in zip(union, common) if keep]
    require(len(genes) == 6336 and len(set(genes)) == 6336, f"Expected 6,336 exact genes, got {len(genes)}")
    return genes


def prepare(out: Path) -> list[str]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    for directory in ("00_audit", "00_freeze", "01_harmonized", "02_metrics", "03_figures", "04_umap_models"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    genes = exact_common_panel()
    panel_path = out / "00_freeze" / PANEL_NAME
    panel_path.write_text("\n".join(genes) + "\n", encoding="utf-8")
    inputs = [
        v1c.SCGEN_AGGREGATE,
        v1c.SCGEN_ROOT / "workflow_summary.json",
        v1c.CELLRANK_H5AD,
        v1c.CELLRANK_ROOT / "workflow_summary.json",
        v1c.WOT_ROOT / "workflow_summary.json",
        v1c.V7_ROOT / "framework_manifest.json",
        v1c.V7_ROOT / "real_benchmark/metadata_checked.csv",
        V1_OUT / "00_audit/FINAL_VALIDATION.json",
        V1_OUT / "00_audit/output_sha256_manifest.json",
    ]
    inputs += v1c.raw_files() + v1c.wot_h5ads(days=(4, 33))
    inputs += [
        v1c.V7_ROOT / f"runs/v7_cell_like_agent_dynamic/seed_{seed}/simulation_summary.csv"
        for seed in (0, 1, 2)
    ]
    require(all(path.is_file() for path in inputs), "A frozen input is missing")
    manifest = [
        {"relative_path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in inputs
    ]
    write_json(out / "00_freeze/input_manifest.json", {"hash_algorithm": "SHA256", "files": manifest})
    write_json(out / "00_freeze/frozen_protocol.json", {
        "experiment_id": "GSE230538_multimethod_expression_benchmark_v2_old_scale_v7_frozen",
        "status": "FROZEN_BEFORE_REHARMONIZATION",
        "comparison": "visualization-only post-hoc old-scale sensitivity",
        "gene_panel": "6,336 genes modeled in every scGen LOCO fold, ordered by the frozen V1 CellRank order",
        "gene_panel_sha256": sha256_file(panel_path),
        "common_scale": "log1p(cell-line-level pseudo-bulk CP10K-like expression); aggregation before log1p",
        "cell_line_weighting": "four cell lines equally weighted after cell-line pseudo-bulk construction",
        "WOT": "wot_row_mass-weighted target-derived transport projection; aggregation before log1p",
        "CellRank": "untreated-only terminal CP10K centroids fate-weighted before log1p",
        "AgentVC": "frozen V7 state-prototype CP10K-like reconstruction; seeds averaged before log1p",
        "scGen_limitation": (
            "generated-cell H5ADs referenced by the handoff are absent locally; frozen mean-log output is converted "
            "to a pseudo-bulk proxy with a gene- and cell-line-specific Jensen gap estimated from untreated Day0 only"
        ),
        "heldout_expression_used_for_scgen_adjustment": False,
        "new_model_runs": 0,
        "new_llm_calls": 0,
        "new_physicell_runs": 0,
        "overall_winner_score": False,
    })
    write_json(out / "00_audit/PREPARED.json", {"status": "PASS_OLD_SCALE_INPUTS_FROZEN", "gene_count": 6336})
    return genes


def load_observed_agent_inputs(out: Path, genes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata = pd.read_csv(v1c.V7_ROOT / "real_benchmark/metadata_checked.csv")
    require(len(metadata) == 48000, "Unexpected metadata cell count")
    state_by_cell = metadata.set_index("cell_id")["inferred_state"]
    days = (0, 1, 4, 33)
    linear_means = np.empty((4, 4, len(genes)), dtype=np.float64)
    mean_logs = np.empty_like(linear_means)
    state_sums = {day: {state: np.zeros(len(genes)) for state in v1c.STATES} for day in (0, 1)}
    state_counts = {day: {state: 0 for state in v1c.STATES} for day in (0, 1)}
    rows = []
    for path in v1c.raw_files():
        gsm, cell_line, day = v1c.sample_info(path)
        cell_ids, cp10k, audit = read_cp10k(path, genes)
        line_index, day_index = v1c.CELL_LINES.index(cell_line), days.index(day)
        linear_means[line_index, day_index] = cp10k.mean(axis=0, dtype=np.float64)
        mean_logs[line_index, day_index] = np.log1p(cp10k).mean(axis=0, dtype=np.float64)
        if day in (0, 1):
            labels = state_by_cell.loc[cell_ids].astype(str).to_numpy()
            for state in v1c.STATES:
                selected = labels == state
                if selected.any():
                    state_sums[day][state] += cp10k[selected].sum(axis=0, dtype=np.float64)
                    state_counts[day][state] += int(selected.sum())
        rows.append({"gsm": gsm, "cell_line": cell_line, "day": day, **audit})
        del cp10k
        gc.collect()
    pd.DataFrame(rows).to_csv(out / "01_harmonized/observed_sample_registry.csv", index=False)
    global_day0 = linear_means[:, 0].mean(axis=0)
    prototypes = []
    sources = []
    for state in v1c.STATES:
        source_day = 0 if state_counts[0][state] > 0 else 1
        count = state_counts[source_day][state]
        prototypes.append(state_sums[source_day][state] / count if count else global_day0)
        sources.append({"state": state, "source_day": source_day, "cell_count": count})
    pd.DataFrame(sources).to_csv(out / "01_harmonized/agentvc_state_prototype_registry.csv", index=False)
    shift = linear_means[:, 1].mean(axis=0) - global_day0
    observed = np.log1p(linear_means[:, [2, 3], :]).transpose(1, 0, 2)
    day0_log_pseudobulk = np.log1p(linear_means[:, 0])
    jensen_gap = day0_log_pseudobulk - mean_logs[:, 0]
    require(jensen_gap.min() >= -1e-10, "Day0 Jensen gap unexpectedly negative")
    return observed, day0_log_pseudobulk, jensen_gap, np.vstack(prototypes), shift, {
        "state_counts": state_counts,
        "heldout_access_for_agentvc_construction": 0,
    }


def load_scgen(genes: list[str], jensen_gap: np.ndarray) -> tuple[np.ndarray, dict]:
    frame = pd.read_csv(v1c.SCGEN_AGGREGATE)
    result = np.empty((2, 4, len(genes)), dtype=np.float64)
    native_min = float("inf")
    clipped = 0
    for day_index, day in enumerate(v1c.HELDOUT_DAYS):
        for line_index, cell_line in enumerate(v1c.CELL_LINES):
            selected = frame[
                frame["target_condition"].eq(f"day_{day}")
                & frame["held_out_group"].eq(cell_line)
            ].drop_duplicates("gene").set_index("gene")["native_scgen_predicted_mean"].reindex(genes)
            require(selected.notna().all(), f"scGen exact-panel gap: {cell_line}/day{day}")
            native = selected.to_numpy(dtype=np.float64)
            native_min = min(native_min, float(native.min()))
            clipped += int((native < 0).sum())
            result[day_index, line_index] = np.maximum(native, 0.0) + jensen_gap[line_index]
    require(np.isfinite(result).all() and np.all(result >= 0), "Invalid scGen proxy")
    return result, {
        "native_mean_log_minimum": native_min,
        "negative_mean_values_clipped": clipped,
        "formula": "max(native_scGen_mean_log1p,0) + [log1p(mean Day0 CP10K) - mean Day0 log1p(CP10K)]",
        "adjustment_data": "untreated Day0 cells from the held-out cell line only",
        "heldout_day4_day33_expression_used": False,
        "native_generated_cell_h5ad_available_locally": False,
        "interpretation": "Day0-dispersion-adjusted scGen pseudo-bulk proxy; not an exact native generated-cell pseudo-bulk",
    }


def load_wot(genes: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    result = np.empty((2, 4, len(genes)), dtype=np.float64)
    rows = []
    for path in v1c.wot_h5ads():
        data = ad.read_h5ad(path, backed="r")
        cell_line = data.obs["cell_line"].astype(str).unique().tolist()
        day_values = data.obs["target_day"].astype(float).unique().tolist()
        require(len(cell_line) == len(day_values) == 1, f"Ambiguous WOT file: {path}")
        day = int(round(day_values[0]))
        if day not in v1c.HELDOUT_DAYS:
            data.file.close()
            continue
        indices = data.var_names.get_indexer(genes)
        require(np.all(indices >= 0), f"WOT missing genes: {path}")
        matrix = data.X[:, indices]
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float64)
        weights = data.obs["wot_row_mass"].to_numpy(dtype=np.float64)
        require(np.isfinite(matrix).all() and np.all(matrix >= 0), f"Invalid WOT matrix: {path}")
        require(np.isfinite(weights).all() and np.all(weights > 0), f"Invalid WOT weights: {path}")
        vector = np.log1p(np.average(matrix, axis=0, weights=weights))
        result[v1c.HELDOUT_DAYS.index(day), v1c.CELL_LINES.index(cell_line[0])] = vector
        rows.append({"cell_line": cell_line[0], "day": day, "rows": len(matrix), "row_mass_sum": weights.sum()})
        data.file.close()
        del matrix
        gc.collect()
    require(np.isfinite(result).all() and np.all(result >= 0), "Invalid WOT output")
    return result, pd.DataFrame(rows)


def load_cellrank(genes: list[str]) -> tuple[np.ndarray, dict]:
    data = ad.read_h5ad(v1c.CELLRANK_H5AD)
    indices = data.var_names.get_indexer(genes)
    require(np.all(indices >= 0), "CellRank missing exact-panel genes")
    matrix = data.X[:, indices]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float64)
    terminal = data.obs["cellrank_terminal_state_membership"].astype(str).to_numpy()
    lineages = [
        column.removeprefix("fate_probability_")
        for column in data.obs.columns
        if column.startswith("fate_probability_terminal_lineage_")
    ]
    centroids = np.vstack([matrix[terminal == lineage].mean(axis=0) for lineage in lineages])
    probabilities = data.obs[[f"fate_probability_{lineage}" for lineage in lineages]].to_numpy(float)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = data.obs["cell_line"].astype(str).to_numpy()
    vectors, weights = [], []
    for cell_line in v1c.CELL_LINES:
        fate = probabilities[labels == cell_line].mean(axis=0)
        vectors.append(np.log1p(fate @ centroids))
        weights.append({"cell_line": cell_line, **dict(zip(lineages, fate))})
    result = np.repeat(np.vstack(vectors)[None, :, :], 2, axis=0)
    return result, {
        "formula": "mean untreated fate weights x untreated terminal mean CP10K, then log1p",
        "post_expression_used": False,
        "best_post_alignment_used": False,
        "fate_weights": weights,
    }


def load_agentvc(prototypes: np.ndarray, shift: np.ndarray) -> np.ndarray:
    result = np.empty((2, 3, prototypes.shape[1]), dtype=np.float64)
    for seed in (0, 1, 2):
        summary = pd.read_csv(v1c.V7_ROOT / f"runs/v7_cell_like_agent_dynamic/seed_{seed}/simulation_summary.csv")
        for day_index, day in enumerate(v1c.HELDOUT_DAYS):
            selected = summary[summary["aligned_real_day"].astype(int).eq(day)]
            require(len(selected) == 1, f"Missing V7 AgentVC seed{seed}/day{day}")
            composition = selected.iloc[0][list(v1c.STATES)].to_numpy(float)
            require(np.isclose(composition.sum(), 1.0, atol=1e-8), "AgentVC composition sum mismatch")
            result[day_index, seed] = np.maximum(composition @ prototypes + shift, 0.0)
    require(np.isfinite(result).all() and np.all(result >= 0), "Invalid AgentVC linear proxy")
    return result


def point_prediction(method: str, values: np.ndarray) -> np.ndarray:
    if method == "AgentVC":
        return np.log1p(values.mean(axis=0))
    return values.mean(axis=0)


def analyze(observed: np.ndarray, methods: dict[str, np.ndarray], genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, boot_rows, gene_rows, error_rows, decile_rows = [], [], [], [], []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for day_index, day in enumerate(v1c.HELDOUT_DAYS):
        truth = observed[day_index].mean(axis=0)
        order = np.argsort(truth, kind="stable")
        groups = np.array_split(order, 10)
        decile_labels = np.empty(len(genes), dtype=int)
        for decile, indexes in enumerate(groups, 1):
            decile_labels[indexes] = decile
        line_draws = rng.integers(0, 4, size=(BOOTSTRAP_ITERATIONS, 4))
        seed_draws = rng.integers(0, 3, size=(BOOTSTRAP_ITERATIONS, 3))
        for method in METHODS:
            prediction = point_prediction(method, methods[method][day_index])
            slope, intercept = np.polyfit(truth, prediction, 1)
            metric_rows.append({
                "day": day, "method": method, "gene_count": len(genes),
                "Pearson": pearsonr(prediction, truth).statistic,
                "Spearman": spearmanr(prediction, truth).statistic,
                "CCC": ccc(prediction, truth),
                "RMSE": np.sqrt(np.mean((prediction - truth) ** 2)),
                "MAE": np.mean(np.abs(prediction - truth)),
                "slope": slope, "intercept": intercept,
                "expression_scale": "log1p(cell-line pseudo-bulk CP10K-like); aggregation before log1p",
            })
            for gene, x, y, decile in zip(genes, truth, prediction, decile_labels):
                gene_rows.append({"day": day, "method": method, "gene": gene, "observed": x, "predicted": y})
                error_rows.append({
                    "day": day, "method": method, "gene": gene,
                    "signed_error": y - x, "absolute_error": abs(y - x), "observed_decile": decile,
                })
            for decile, indexes in enumerate(groups, 1):
                decile_rows.append({
                    "day": day, "method": method, "decile": decile, "gene_count": len(indexes),
                    "observed_mean": truth[indexes].mean(), "predicted_mean": prediction[indexes].mean(),
                })
            for iteration in range(BOOTSTRAP_ITERATIONS):
                line_indexes = line_draws[iteration]
                boot_truth = observed[day_index, line_indexes].mean(axis=0)
                if method == "AgentVC":
                    boot_prediction = np.log1p(methods[method][day_index, seed_draws[iteration]].mean(axis=0))
                    unit = "Observed cell lines and frozen V7 seeds independently"
                else:
                    boot_prediction = methods[method][day_index, line_indexes].mean(axis=0)
                    unit = "paired cell line"
                boot_rows.append({
                    "day": day, "method": method, "iteration": iteration,
                    "Pearson": pearsonr(boot_prediction, boot_truth).statistic,
                    "bootstrap_unit": unit, "bootstrap_seed": BOOTSTRAP_SEED,
                })
    return tuple(pd.DataFrame(rows) for rows in (metric_rows, boot_rows, gene_rows, error_rows, decile_rows))


def build_benchmark(out: Path, genes: list[str]) -> None:
    observed, day0_log, jensen_gap, prototypes, shift, agent_audit = load_observed_agent_inputs(out, genes)
    scgen, scgen_audit = load_scgen(genes, jensen_gap)
    wot, wot_rows = load_wot(genes)
    cellrank, cellrank_audit = load_cellrank(genes)
    agent_linear = load_agentvc(prototypes, shift)
    methods = {"scGen": scgen, "WOT": wot, "CellRank": cellrank, "AgentVC": agent_linear}
    metrics, bootstrap, gene_table, errors, deciles = analyze(observed, methods, genes)

    agent_point = np.stack([np.log1p(agent_linear[index].mean(axis=0)) for index in range(2)])
    agent_line = np.empty((2, 4, len(genes)), dtype=np.float64)
    global_day0 = day0_log.mean(axis=0)
    target_errors = []
    for day_index in range(2):
        initial = np.maximum(day0_log + (agent_point[day_index] - global_day0)[None, :], 0.0)
        agent_line[day_index] = v1c.simplex_projection_columns(initial, agent_point[day_index])
        target_errors.append(np.max(np.abs(agent_line[day_index].mean(axis=0) - agent_point[day_index])))

    np.savez_compressed(
        out / "01_harmonized" / ARCHIVE_NAME,
        genes=np.asarray(genes), cell_lines=np.asarray(v1c.CELL_LINES), days=np.asarray(v1c.HELDOUT_DAYS),
        observed=observed, source_day0=day0_log, scgen=scgen, wot=wot, cellrank=cellrank,
        agentvc=agent_point, agentvc_linear=agent_linear, agentvc_line=agent_line,
        agentvc_line_day33=agent_line[1],
    )
    metrics.to_csv(out / "02_metrics/expression_metrics.csv", index=False, float_format="%.17g")
    bootstrap.to_csv(out / "02_metrics/bootstrap_pearson_values.csv.gz", index=False, compression="gzip")
    gene_table.to_csv(out / "01_harmonized/gene_level_plotting_source.csv.gz", index=False, compression="gzip")
    errors.to_csv(out / "02_metrics/signed_error_by_gene.csv.gz", index=False, compression="gzip")
    deciles.to_csv(out / "02_metrics/decile_calibration.csv", index=False, float_format="%.17g")
    wot_rows.to_csv(out / "01_harmonized/wot_weighting_audit.csv", index=False)
    pd.DataFrame(cellrank_audit["fate_weights"]).to_csv(out / "01_harmonized/cellrank_fate_weights.csv", index=False)
    write_json(out / "01_harmonized/harmonization_audit.json", {
        "status": "PASS_OLD_SCALE_6336_EXACT_COMMON_HARMONIZATION",
        "gene_count": len(genes),
        "scale": "log1p(cell-line-level pseudo-bulk CP10K-like expression); aggregation before log1p",
        "observed": "mean CP10K per cell line, then log1p",
        "scGen": scgen_audit,
        "WOT": "positive wot_row_mass-weighted projected CP10K mean, then log1p; target-derived",
        "CellRank": cellrank_audit,
        "AgentVC": {**agent_audit, "seed_aggregation": "mean linear CP10K-like seed proxy, then log1p"},
        "agentvc_line_target_max_error": float(max(target_errors)),
        "finite_nonnegative": True,
        "method_information_regimes_equal": False,
    })
    write_json(out / "00_audit/BENCHMARK_COMPLETE.json", {
        "status": "PASS_OLD_SCALE_BENCHMARK", "metric_rows": len(metrics),
        "bootstrap_rows": len(bootstrap), "gene_rows": len(gene_table), "overall_winner_score": False,
    })


def save_bundle(figure: plt.Figure, directory: Path, stem: str) -> list[str]:
    return v1f._save_bundle(figure, directory, stem)


def draw_scatter(out: Path, metrics: pd.DataFrame, source: pd.DataFrame) -> list[str]:
    figure, axes = plt.subplots(2, 4, figsize=(14.8, 9.35), sharex=True, sharey=True)
    letters = iter("ABCDEFGH")
    values = np.r_[source.observed.to_numpy(), source.predicted.to_numpy()]
    lower, upper = 0.0, float(values.max() * 1.035)
    for row, day in enumerate(v1c.HELDOUT_DAYS):
        for column, method in enumerate(METHODS):
            axis = axes[row, column]
            selected = source[source.day.eq(day) & source.method.eq(method)]
            metric = metrics[metrics.day.eq(day) & metrics.method.eq(method)].iloc[0]
            color = COLORS[method]
            axis.scatter(selected.observed, selected.predicted, s=8, alpha=.24, color=color, edgecolors="none", rasterized=True)
            x = np.asarray([lower, upper])
            axis.plot(x, x, "--", color="#92999F", lw=1)
            axis.plot(x, metric.intercept + metric.slope * x, color=color, lw=1.8)
            axis.set_xlim(lower, upper); axis.set_ylim(lower, upper); axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{next(letters)}. Day{day} — {LABELS[method]}", loc="left", fontsize=10.2)
            axis.text(.035, .96, f"Pearson r = {metric.Pearson:.3f}\nCCC = {metric.CCC:.3f}\nRMSE = {metric.RMSE:.3f}\nslope = {metric.slope:.3f}",
                      transform=axis.transAxes, va="top", fontsize=7.5,
                      bbox={"boxstyle":"round,pad=.25","facecolor":"white","edgecolor":"#D5D8DA","alpha":.9})
            axis.set_xlabel("Observed Post expression", fontsize=8.2)
            if column == 0: axis.set_ylabel("Predicted / proxy expression", fontsize=8.2)
    figure.suptitle("GSE230538 expanded gene-expression benchmark — old unified scale", y=.985, fontsize=18, fontweight="bold")
    figure.text(.5,.947,"6,336 exact common genes · log1p cell-line pseudo-bulk CP10K-like expression · held-out Day4/Day33",ha="center",fontsize=10.2,color="#444")
    figure.text(.5,.018,"* WOT is target-derived. † CellRank is an untreated target-blind proxy. ‡ scGen is a Day0-dispersion-adjusted pseudo-bulk proxy because its generated-cell H5AD is absent locally.\nAgentVC reuses frozen V7 trajectories; method inputs are not information-equivalent.",ha="center",va="bottom",fontsize=7.6,color="#444")
    figure.tight_layout(rect=(.035,.07,.995,.925),w_pad=1.1,h_pad=4.2); figure.subplots_adjust(hspace=.30)
    files=save_bundle(figure,out/"03_figures/figure_1_scatter","gse230538_old_scale_expression_scatter_2x4"); plt.close(figure); return files


def draw_diagnostics(out: Path, metrics: pd.DataFrame, boot: pd.DataFrame, errors: pd.DataFrame, deciles: pd.DataFrame) -> list[str]:
    figure, axes = plt.subplots(2,3,figsize=(15.1,9.0)); letters=iter("ABCDEF")
    palette=[COLORS[m] for m in METHODS]
    for row,day in enumerate(v1c.HELDOUT_DAYS):
        axis=axes[row,0]; b=boot[boot.day.eq(day)]
        sns.violinplot(data=b,x="method",y="Pearson",order=METHODS,palette=palette,inner="box",cut=0,linewidth=.8,ax=axis)
        axis.set_xticklabels([LABELS[m] for m in METHODS]); axis.set_xlabel(""); axis.set_ylabel("Bootstrap Pearson r")
        axis.set_title(f"{next(letters)}. Day{day}: bootstrap Pearson",loc="left")
        lo,hi=axis.get_ylim()
        for i,m in enumerate(METHODS):
            point=metrics[metrics.day.eq(day)&metrics.method.eq(m)].iloc[0].Pearson; vals=b[b.method.eq(m)].Pearson
            q=np.quantile(vals,[.025,.975]); axis.text(i,hi-.025*(hi-lo),f"r={point:.3f}\nCI {q[0]:.3f}–{q[1]:.3f}",ha="center",va="top",fontsize=6.6,color="#555")
        axis=axes[row,1]; e=errors[errors.day.eq(day)]
        sns.violinplot(data=e,x="method",y="signed_error",order=METHODS,palette=palette,inner="box",cut=0,linewidth=.8,ax=axis)
        axis.axhline(0,color="#777",ls="--",lw=1); axis.set_xticklabels([LABELS[m] for m in METHODS]); axis.set_xlabel(""); axis.set_ylabel("Predicted / proxy − Observed")
        axis.set_title(f"{next(letters)}. Day{day}: signed error",loc="left"); lo,hi=axis.get_ylim()
        for i,m in enumerate(METHODS):
            z=e[e.method.eq(m)]; axis.text(i,hi-.025*(hi-lo),f"median {z.signed_error.median():.3f}\nMAE {z.absolute_error.mean():.3f}",ha="center",va="top",fontsize=6.6,color="#555")
        axis=axes[row,2]; obs=deciles[deciles.day.eq(day)&deciles.method.eq("scGen")]
        axis.plot(obs.decile,obs.observed_mean,color="#222",marker="o",lw=2,label="Observed")
        for m in METHODS:
            z=deciles[deciles.day.eq(day)&deciles.method.eq(m)]; axis.plot(z.decile,z.predicted_mean,color=COLORS[m],marker="o",lw=1.7,label=LABELS[m])
        axis.set_xticks(range(1,11),[f"D{i}" for i in range(1,11)]); axis.set_xlabel("Observed-expression decile"); axis.set_ylabel("Mean pseudo-bulk expression")
        axis.set_title(f"{next(letters)}. Day{day}: decile calibration",loc="left")
        if row==0: axis.legend(ncol=3,fontsize=7,frameon=False)
    figure.suptitle("GSE230538 expanded gene-expression diagnostics — old unified scale",y=.982,fontsize=18,fontweight="bold")
    figure.text(.5,.944,"6,336 exact common genes · aggregation before log1p · 1,000 cell-line/seed bootstrap iterations",ha="center",fontsize=10.2,color="#444")
    figure.text(.5,.018,"* WOT is target-derived. † CellRank is target-blind. ‡ scGen uses an untreated-Day0 dispersion correction because native generated cells are unavailable. No overall winner is constructed.",ha="center",va="bottom",fontsize=7.6,color="#444")
    figure.tight_layout(rect=(.035,.065,.995,.91),w_pad=2,h_pad=2.6)
    files=save_bundle(figure,out/"03_figures/figure_2_diagnostics","gse230538_old_scale_expression_diagnostics_2x3"); plt.close(figure); return files


def draw_umap(out: Path, coordinates: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    figure,axes=plt.subplots(2,3,figsize=(17.2,10.2)); panels=["Observed","scGen","WOT","CellRank","AgentVC"]
    titles={"Observed":"A. Observed Day33","scGen":"B. scGen‡","WOT":"C. WOT*","CellRank":"D. CellRank†","AgentVC":"E. AgentVC"}
    reference=coordinates[coordinates.source.eq("Observed")]; xlim=(coordinates.UMAP1.min(),coordinates.UMAP1.max()); ylim=(coordinates.UMAP2.min(),coordinates.UMAP2.max())
    for panel,axis in zip(panels,axes.flat[:5]):
        axis.scatter(reference.UMAP1,reference.UMAP2,s=1.3,color="#D5DADF",alpha=.32,edgecolors="none",rasterized=True)
        shown=reference[reference.day.eq(33)] if panel=="Observed" else coordinates[coordinates.source.eq(panel)]
        for line in v1c.CELL_LINES:
            z=shown[shown.cell_line.eq(line)]; axis.scatter(z.UMAP1,z.UMAP2,s=3,color=v1c.CELL_LINE_COLORS[line],alpha=.7,edgecolors="none",rasterized=True)
        axis.set_title(titles[panel],loc="left"); axis.set_xlim(*xlim);axis.set_ylim(*ylim);axis.set_xticks([]);axis.set_yticks([]);axis.set_xlabel("UMAP1");axis.set_ylabel("UMAP2")
        if panel!="Observed": axis.text(.99,.01,"deterministic proxy\ntransform only",transform=axis.transAxes,ha="right",va="bottom",fontsize=7.2,color="#666")
    old=axes.flat[5]; spec=old.get_subplotspec();old.remove();grid=spec.subgridspec(1,2,wspace=.34);ca=figure.add_subplot(grid[0,0]);ra=figure.add_subplot(grid[0,1],sharey=ca)
    pos=np.arange(len(METHODS)); offsets={4:-.1,33:.1}; markers={4:"o",33:"^"}
    cmin=max(-.05,float(metrics.CCC.min()-0.08)); rmax=float(metrics.RMSE.max()*1.28)
    for day in v1c.HELDOUT_DAYS:
        z=metrics[metrics.day.eq(day)].set_index("method").loc[list(METHODS)]
        for i,m in enumerate(METHODS):
            cv=z.loc[m,"CCC"];rv=z.loc[m,"RMSE"]
            ca.scatter(cv,i+offsets[day],s=44,marker=markers[day],color=COLORS[m],edgecolor="white",lw=.6);ca.text(cv+.008,i+offsets[day],f"{cv:.3f}",va="center",fontsize=6.5)
            ra.scatter(rv,i+offsets[day],s=44,marker=markers[day],color=COLORS[m],edgecolor="white",lw=.6);ra.text(rv+.008,i+offsets[day],f"{rv:.3f}",va="center",fontsize=6.5)
    ca.set_yticks(pos,[LABELS[m] for m in METHODS]);ca.invert_yaxis();ca.set_xlim(cmin,1.03);ra.set_xlim(0,rmax);ca.set_xlabel("CCC ↑");ra.set_xlabel("RMSE ↓");ca.set_title("F. Day4/Day33 fidelity",loc="left",pad=14);ra.tick_params(labelleft=False)
    ca.legend(handles=[Line2D([0],[0],marker="o",ls="",color="#555",label="Day4"),Line2D([0],[0],marker="^",ls="",color="#555",label="Day33")],loc="upper left",bbox_to_anchor=(0,1.06),ncol=2,frameon=False,fontsize=7.5)
    figure.legend(handles=[Line2D([0],[0],marker="o",ls="",color=v1c.CELL_LINE_COLORS[line],label=line) for line in v1c.CELL_LINES],ncol=4,loc="lower center",bbox_to_anchor=(.5,.052),frameon=False)
    figure.suptitle("GSE230538 Day33 cell-expanded expression — old unified scale",y=1.01)
    figure.text(.5,.972,"Observed Day0+Day33 fit frozen PCA/UMAP · proxies transform only · 6,336 exact common genes",ha="center",fontsize=10.5,color="#555")
    figure.text(.5,.008,"UMAP proxy points are deterministic expansions, not native single-cell predictions; colors are source-inherited cell-line labels.\n* WOT is target-derived; † CellRank is target-blind; ‡ scGen is Day0-dispersion-adjusted. AgentVC is frozen V7, not V9.",ha="center",fontsize=8,color="#555")
    figure.tight_layout(rect=(0,.10,1,.95));files=save_bundle(figure,out/"03_figures/figure_3_umap","gse230538_old_scale_day33_cell_expanded_umap");plt.close(figure);return files


def build_figures(out: Path) -> None:
    v1f._style()
    archive=np.load(out/"01_harmonized"/ARCHIVE_NAME,allow_pickle=False);genes=archive["genes"].astype(str).tolist();arrays={k:archive[k] for k in archive.files if k!="genes"}
    metrics=pd.read_csv(out/"02_metrics/expression_metrics.csv");source=pd.read_csv(out/"01_harmonized/gene_level_plotting_source.csv.gz")
    boot=pd.read_csv(out/"02_metrics/bootstrap_pearson_values.csv.gz");errors=pd.read_csv(out/"02_metrics/signed_error_by_gene.csv.gz");deciles=pd.read_csv(out/"02_metrics/decile_calibration.csv")
    f1=draw_scatter(out,metrics,source);f2=draw_diagnostics(out,metrics,boot,errors,deciles)
    coordinates,audit=v1f._umap_sources(out,genes,arrays);f3=draw_umap(out,coordinates,metrics)
    write_json(out/"00_audit/FIGURES_COMPLETE.json",{"status":"PASS_THREE_OLD_SCALE_FIGURE_BUNDLES","figure_1":f1,"figure_2":f2,"figure_3":f3,"proxy_max_target_mean_error":float(audit.maximum_gene_target_mean_error.max())})


def validate(out: Path) -> None:
    panel=(out/"00_freeze"/PANEL_NAME).read_text().splitlines(); require(len(panel)==6336 and len(set(panel))==6336,"Panel validation failed")
    archive=np.load(out/"01_harmonized"/ARCHIVE_NAME,allow_pickle=False)
    for key in ("observed","scgen","wot","cellrank","agentvc","agentvc_linear","agentvc_line"):
        require(np.isfinite(archive[key]).all() and np.all(archive[key]>=0),f"Invalid archive array: {key}")
    metrics=pd.read_csv(out/"02_metrics/expression_metrics.csv"); require(len(metrics)==8,"Metric completeness failed")
    recomputed=[]
    observed=archive["observed"]
    for di,day in enumerate(v1c.HELDOUT_DAYS):
        truth=observed[di].mean(0)
        for method,key in (("scGen","scgen"),("WOT","wot"),("CellRank","cellrank")):
            pred=archive[key][di].mean(0); recomputed.append(abs(ccc(pred,truth)-metrics[metrics.day.eq(day)&metrics.method.eq(method)].iloc[0].CCC))
        pred=np.log1p(archive["agentvc_linear"][di].mean(0)); recomputed.append(abs(ccc(pred,truth)-metrics[metrics.day.eq(day)&metrics.method.eq("AgentVC")].iloc[0].CCC))
    require(max(recomputed)<=1e-12,f"Metric recomputation error: {max(recomputed)}")
    proxy=pd.read_csv(out/"04_umap_models/proxy_target_mean_audit.csv"); require(proxy.maximum_gene_target_mean_error.max()<=1e-10,"UMAP target error")
    figures=list((out/"03_figures").rglob("*")); require(sum(p.is_file() and p.suffix in {".png",".pdf",".svg"} for p in figures)==12,"Figure bundle incomplete")
    frozen=json.loads((out/"00_freeze/input_manifest.json").read_text())
    changed=[]
    for row in frozen["files"]:
        path=ROOT/row["relative_path"]
        if not path.is_file() or sha256_file(path)!=row["sha256"]: changed.append(row["relative_path"])
    require(not changed,f"Frozen inputs changed: {changed[:3]}")
    output_files=sorted(p for p in out.rglob("*") if p.is_file() and p.name not in {"FINAL_VALIDATION.json","output_sha256_manifest.json"})
    write_json(out/"00_audit/output_sha256_manifest.json",{"hash_algorithm":"SHA256","files":[{"relative_path":str(p.relative_to(out)),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in output_files]})
    write_json(out/"00_audit/FINAL_VALIDATION.json",{
        "status":"PASS_POST_HOC_GSE230538_OLD_SCALE_THREE_FIGURES_V2","gene_count":6336,"metric_rows":8,
        "max_metric_recomputation_error":max(recomputed),"max_proxy_target_mean_error":float(proxy.maximum_gene_target_mean_error.max()),
        "frozen_inputs_unchanged":True,"figure_files":12,"scgen_native_single_cell_pseudobulk_claimed":False,
    })
    print("PASS_POST_HOC_GSE230538_OLD_SCALE_THREE_FIGURES_V2")


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=OUT);parser.add_argument("--resume",action="store_true");args=parser.parse_args();out=args.out.resolve()
    if args.resume:
        marker=out/"00_audit/FINAL_VALIDATION.json"
        if marker.exists(): print(json.loads(marker.read_text())["status"]+" (resume)");return 0
        require((out/"00_audit/PREPARED.json").exists(),"Partial output cannot resume before PREPARED marker")
        genes=(out/"00_freeze"/PANEL_NAME).read_text().splitlines()
    else:
        genes=prepare(out)
    if not (out/"00_audit/BENCHMARK_COMPLETE.json").exists(): build_benchmark(out,genes)
    if not (out/"00_audit/FIGURES_COMPLETE.json").exists(): build_figures(out)
    validate(out);return 0


if __name__=="__main__":
    raise SystemExit(main())
