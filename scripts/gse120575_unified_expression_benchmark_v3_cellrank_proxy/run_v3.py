#!/usr/bin/env python3
"""Build the post-hoc GSE120575 expression benchmark with CellRank proxy.

This script does not run CellRank or any other model. It derives a target-blind
expression proxy from the existing Pre-only CellRank terminal states and fate
probabilities, then evaluates it with the frozen V1 benchmark protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/matplotlib-gse120575-expression-v3-cellrank"
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
CELLRANK_ROOT = (
    ROOT
    / "data/wot_scgen_cellrank_native_pre_post_reports"
    / "cellrank_gse120575_pre_to_post_native"
)
CELLRANK_H5AD = CELLRANK_ROOT / "native_cellrank_pre_terminal_status_all_genes.h5ad"
PROXY_ROOT = Path(os.environ.get("GSE120575_PROXY_OUT", str(ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"))).resolve()
V1_ROOT = Path(os.environ.get("GSE120575_EXPRESSION_BENCHMARK_OUT", str(ROOT / "outputs/GSE120575_unified_expression_benchmark_v1"))).resolve()
OUT = Path(os.environ.get("GSE120575_V3_OUT", str(ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy"))).resolve()

PANEL = PROXY_ROOT / "frozen_834_gene_panel.txt"
HARMONIZED = PROXY_ROOT / "harmonized_834_expression_by_method_sample.npz"
SCALE_AUDIT = PROXY_ROOT / "scale_harmonization_audit.json"
RECON_AUDIT = PROXY_ROOT / "reconstruction_audit.json"
V1_AUDIT = V1_ROOT / "benchmark_audit.json"
V1_METRICS = V1_ROOT / "unified_expression_metrics.csv"
V1_BOOTSTRAP = V1_ROOT / "bootstrap_confidence_intervals.csv"

EXPECTED_HASHES = {
    "cellrank_h5ad": "9af4c5a4ae2fa3167c9bc0fcbcec340bc71266264c43aca171332c2ed7fe9b45",
    "gene_panel": "1a9e02f8bb52cdea852d6fe30992339ad8bc2fa0ab807b81848dce574a7bce41",
    "harmonized_npz": "1a0645902aac33e6ddefdb9bfd32e6af4e628ba2502abf2b376ceb3457ffb36a",
    "v1_metrics": "b88af75b61a585776481dd9c03fec6fead97064ae462ef080be7503214f507b6",
    "v1_bootstrap": "606ebb187193e6638839eb6231ecbe1eb44145ed43f0c1b1cfbf051ae753cd8b",
}

RESPONSES = ("Responder", "Non-responder")
SAMPLES_BY_RESPONSE = {
    "Responder": ("Pre_P24", "Pre_P29", "Pre_P35"),
    "Non-responder": ("Pre_P2", "Pre_P3", "Pre_P27"),
}
SAMPLES = tuple(sample for response in RESPONSES for sample in SAMPLES_BY_RESPONSE[response])
LINEAGES = tuple(f"terminal_lineage_0{i}_dpt_rank_{i}" for i in range(1, 5))
FATE_COLUMNS = tuple(f"fate_probability_{lineage}" for lineage in LINEAGES)
TERMINAL_COLUMN = "cellrank_terminal_state_membership"

METHODS = (
    "scGen",
    "WOT_weighted",
    "CellRank_terminal_proxy",
    "AgentVC_proxy",
)
LEGACY_METHODS = ("scGen", "WOT_weighted", "AgentVC_proxy")
METHOD_LABELS = {
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "CellRank_terminal_proxy": "CellRank\u2020",
    "AgentVC_proxy": "AgentVC",
}
METHOD_ROLES = {
    "scGen": "generated_expression_baseline",
    "WOT_weighted": "primary_target_derived_transport_baseline",
    "CellRank_terminal_proxy": "pre_only_fate_weighted_terminal_state_expression_proxy",
    "AgentVC_proxy": "offline_reconstructed_runtime_expression_proxy",
}
COLORS = {
    "Observed_Post": "#222222",
    "scGen": "#F58518",
    "WOT_weighted": "#54A24B",
    "CellRank_terminal_proxy": "#4C9ED9",
    "AgentVC_proxy": "#A66FA0",
}
MARKERS = {
    "Observed_Post": "o",
    "scGen": "o",
    "WOT_weighted": "s",
    "CellRank_terminal_proxy": "^",
    "AgentVC_proxy": "D",
}

METRICS = (
    "RMSE",
    "MAE",
    "CCC",
    "Pearson",
    "Spearman",
    "mean_signed_bias",
    "high_expression_gene_bias",
)
EXPRESSION_SCALE = "log1p(sample-level pseudo-bulk TPM-like expression)"
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 120575834
EXPECTED_OBSERVED_N = {"Responder": 5, "Non-responder": 18}

SCATTER_PNG = OUT / "GSE120575_expression_benchmark_with_CellRank_scatter_600dpi.png"
SCATTER_PDF = OUT / "GSE120575_expression_benchmark_with_CellRank_scatter_vector.pdf"
SCORE_PNG = OUT / "GSE120575_expression_benchmark_with_CellRank_scores_600dpi.png"
SCORE_PDF = OUT / "GSE120575_expression_benchmark_with_CellRank_scores_vector.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def concordance_correlation(predicted: np.ndarray, observed: np.ndarray) -> float:
    mean_predicted = float(np.mean(predicted))
    mean_observed = float(np.mean(observed))
    centered_predicted = predicted - mean_predicted
    centered_observed = observed - mean_observed
    denominator = (
        float(np.mean(centered_predicted**2))
        + float(np.mean(centered_observed**2))
        + (mean_predicted - mean_observed) ** 2
    )
    if denominator == 0:
        return np.nan
    return float(
        2.0 * np.mean(centered_predicted * centered_observed) / denominator
    )


def observed_deciles(observed: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    order = np.argsort(observed, kind="stable")
    groups = [np.asarray(group, dtype=int) for group in np.array_split(order, 10)]
    labels = np.empty(len(observed), dtype=np.int8)
    for decile, indexes in enumerate(groups, start=1):
        labels[indexes] = decile
    return labels, groups


def compute_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    high_expression_indexes: np.ndarray,
) -> dict[str, float]:
    error = predicted - observed
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "CCC": concordance_correlation(predicted, observed),
        "Pearson": float(pearsonr(predicted, observed).statistic),
        "Spearman": float(spearmanr(predicted, observed).statistic),
        "mean_signed_bias": float(np.mean(error)),
        "high_expression_gene_bias": float(
            np.mean(error[high_expression_indexes])
        ),
    }


def validate_sources() -> tuple[list[str], dict[str, str]]:
    source_paths = {
        "cellrank_h5ad": CELLRANK_H5AD,
        "gene_panel": PANEL,
        "harmonized_npz": HARMONIZED,
        "v1_metrics": V1_METRICS,
        "v1_bootstrap": V1_BOOTSTRAP,
        "scale_audit": SCALE_AUDIT,
        "reconstruction_audit": RECON_AUDIT,
        "v1_benchmark_audit": V1_AUDIT,
    }
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    frozen_names = EXPECTED_HASHES if os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") != "1" else {"cellrank_h5ad": EXPECTED_HASHES["cellrank_h5ad"], "gene_panel": EXPECTED_HASHES["gene_panel"]}
    for name, expected in frozen_names.items():
        if source_hashes[name] != expected:
            raise RuntimeError(
                f"Frozen input hash changed for {name}: "
                f"{source_hashes[name]} != {expected}"
            )

    scale_audit = json.loads(SCALE_AUDIT.read_text(encoding="utf-8"))
    reconstruction_audit = json.loads(RECON_AUDIT.read_text(encoding="utf-8"))
    v1_audit = json.loads(V1_AUDIT.read_text(encoding="utf-8"))
    if scale_audit.get("status") != "PASS_FOUR_METHOD_834_GENE_SCALE_HARMONIZATION":
        raise RuntimeError("Frozen scale audit did not pass")
    if reconstruction_audit.get("status") != "PASS_AGENTVC_CHECKPOINT5_PROXY_RECONSTRUCTION":
        raise RuntimeError("Frozen AgentVC reconstruction audit did not pass")
    if v1_audit.get("status") != "PASS_GSE120575_UNIFIED_EXPRESSION_BENCHMARK_V1":
        raise RuntimeError("Frozen V1 benchmark audit did not pass")

    genes = PANEL.read_text(encoding="utf-8").splitlines()
    if len(genes) != 834 or len(set(genes)) != 834:
        raise RuntimeError("Frozen panel is not 834 unique genes")
    return genes, source_hashes


def build_cellrank_proxy(
    genes: list[str],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cellrank = ad.read_h5ad(CELLRANK_H5AD)
    if cellrank.shape != (5928, 55737):
        raise RuntimeError(f"Unexpected CellRank H5AD shape: {cellrank.shape}")
    if not cellrank.var_names.is_unique:
        raise RuntimeError("CellRank var_names are not unique")
    missing_genes = [gene for gene in genes if gene not in cellrank.var_names]
    if missing_genes:
        raise RuntimeError(f"CellRank is missing frozen genes: {missing_genes[:10]}")
    required_obs = {"sample_label", TERMINAL_COLUMN, *FATE_COLUMNS}
    if not required_obs.issubset(cellrank.obs.columns):
        raise RuntimeError("CellRank H5AD lacks required Pre-only fields")

    expression = cellrank[:, genes].X
    if sparse.issparse(expression):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=np.float64)
    if expression.shape != (5928, 834):
        raise RuntimeError("CellRank panel matrix shape mismatch")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("CellRank log1p(TPM) expression is invalid")
    tpm = np.expm1(expression)

    terminal_labels = cellrank.obs[TERMINAL_COLUMN].astype(str).to_numpy()
    centroid_tpm = np.empty((len(LINEAGES), len(genes)), dtype=np.float64)
    terminal_counts: dict[str, int] = {}
    centroid_rows: list[dict[str, object]] = []
    for lineage_index, lineage in enumerate(LINEAGES):
        selected = terminal_labels == lineage
        terminal_counts[lineage] = int(selected.sum())
        if terminal_counts[lineage] != 30:
            raise RuntimeError(
                f"{lineage} has {terminal_counts[lineage]} terminal cells, expected 30"
            )
        centroid_tpm[lineage_index] = tpm[selected].mean(axis=0)
        for gene_index, gene in enumerate(genes):
            centroid_rows.append(
                {
                    "terminal_lineage": lineage,
                    "terminal_cell_count": terminal_counts[lineage],
                    "gene": gene,
                    "terminal_centroid_TPM": centroid_tpm[lineage_index, gene_index],
                    "terminal_centroid_log1p_TPM": np.log1p(
                        centroid_tpm[lineage_index, gene_index]
                    ),
                }
            )

    probabilities = cellrank.obs.loc[:, list(FATE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise RuntimeError("CellRank fate probabilities are invalid")
    raw_row_sums = probabilities.sum(axis=1)
    if np.any(raw_row_sums <= 0) or np.max(np.abs(raw_row_sums - 1.0)) > 5e-5:
        raise RuntimeError("CellRank fate probability row sums are outside tolerance")
    normalized_probabilities = probabilities / raw_row_sums[:, None]
    normalized_row_sums = normalized_probabilities.sum(axis=1)
    if not np.allclose(normalized_row_sums, 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("Normalized fate probability rows do not sum to one")

    sample_labels = cellrank.obs["sample_label"].astype(str).to_numpy()
    proxy_expression = np.empty((len(SAMPLES), len(genes)), dtype=np.float64)
    fate_rows: list[dict[str, object]] = []
    for sample_index, sample in enumerate(SAMPLES):
        selected = sample_labels == sample
        n_cells = int(selected.sum())
        if n_cells <= 0:
            raise RuntimeError(f"CellRank has no cells for {sample}")
        weights = normalized_probabilities[selected].mean(axis=0)
        weights = weights / weights.sum()
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise RuntimeError(f"Invalid sample fate weights for {sample}")
        if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise RuntimeError(f"Sample fate weights do not sum to one: {sample}")
        proxy_tpm = weights @ centroid_tpm
        proxy_expression[sample_index] = np.log1p(proxy_tpm)
        response = next(
            response
            for response, samples in SAMPLES_BY_RESPONSE.items()
            if sample in samples
        )
        row: dict[str, object] = {
            "sample_id": sample,
            "response": response,
            "n_pre_cells": n_cells,
            "raw_probability_row_sum_min": float(raw_row_sums[selected].min()),
            "raw_probability_row_sum_max": float(raw_row_sums[selected].max()),
            "normalized_sample_weight_sum": float(weights.sum()),
        }
        row.update(
            {
                f"weight_{lineage}": float(weights[index])
                for index, lineage in enumerate(LINEAGES)
            }
        )
        fate_rows.append(row)

    if (
        proxy_expression.shape != (6, 834)
        or not np.isfinite(proxy_expression).all()
        or np.any(proxy_expression < 0)
    ):
        raise RuntimeError("CellRank proxy failed the 6 x 834 finite nonnegative gate")

    checks: dict[str, object] = {
        "input_shape": list(cellrank.shape),
        "panel_shape": list(expression.shape),
        "proxy_shape": list(proxy_expression.shape),
        "terminal_cell_counts": terminal_counts,
        "raw_fate_probability_row_sum_min": float(raw_row_sums.min()),
        "raw_fate_probability_row_sum_max": float(raw_row_sums.max()),
        "normalized_fate_probability_row_sum_min": float(
            normalized_row_sums.min()
        ),
        "normalized_fate_probability_row_sum_max": float(
            normalized_row_sums.max()
        ),
        "sample_cell_counts": {
            sample: int((sample_labels == sample).sum()) for sample in SAMPLES
        },
        "forbidden_post_alignment_fields_used": False,
        "observed_post_used_to_select_lineage": False,
        "cellrank_model_rerun": False,
    }
    return (
        proxy_expression,
        pd.DataFrame(fate_rows),
        pd.DataFrame(centroid_rows),
        checks,
    )


def load_augmented_data(
    genes: list[str],
    cellrank_proxy: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(HARMONIZED) as source:
        loaded = {key: source[key].copy() for key in source.files}
    if loaded["genes"].astype(str).tolist() != genes:
        raise RuntimeError("Harmonized matrix gene order differs from frozen panel")
    expression = loaded["expression_log1p_sample_pseudobulk"].astype(np.float64)
    if expression.shape != (47, 834):
        raise RuntimeError("Unexpected frozen harmonized expression shape")

    cellrank_responses = np.asarray(
        [
            next(
                response
                for response, samples in SAMPLES_BY_RESPONSE.items()
                if sample in samples
            )
            for sample in SAMPLES
        ]
    )
    augmented = {
        "genes": loaded["genes"].astype(str),
        "methods": np.concatenate(
            [
                loaded["methods"].astype(str),
                np.repeat("CellRank_terminal_proxy", len(SAMPLES)),
            ]
        ),
        "sample_ids": np.concatenate(
            [loaded["sample_ids"].astype(str), np.asarray(SAMPLES)]
        ),
        "responses": np.concatenate(
            [loaded["responses"].astype(str), cellrank_responses]
        ),
        "therapy": np.concatenate(
            [loaded["therapy"].astype(str), np.repeat("anti-PD1", len(SAMPLES))]
        ),
        "roles": np.concatenate(
            [
                loaded["roles"].astype(str),
                np.repeat(METHOD_ROLES["CellRank_terminal_proxy"], len(SAMPLES)),
            ]
        ),
        "expression": np.vstack([expression, cellrank_proxy]),
    }
    if augmented["expression"].shape != (53, 834):
        raise RuntimeError("Augmented expression matrix is not 53 x 834")
    if not np.isfinite(augmented["expression"]).all() or np.any(
        augmented["expression"] < 0
    ):
        raise RuntimeError("Augmented expression matrix is invalid")

    for response in RESPONSES:
        observed = (
            (augmented["methods"] == "Observed_Post")
            & (augmented["responses"] == response)
        )
        if int(observed.sum()) != EXPECTED_OBSERVED_N[response]:
            raise RuntimeError(f"Observed biopsy count changed for {response}")
        for method in METHODS:
            selected = (
                (augmented["methods"] == method)
                & (augmented["responses"] == response)
            )
            expected_samples = set(SAMPLES_BY_RESPONSE[response])
            if int(selected.sum()) != 3:
                raise RuntimeError(f"Expected three rows: {response}/{method}")
            if set(augmented["sample_ids"][selected]) != expected_samples:
                raise RuntimeError(f"Sample scope changed: {response}/{method}")
    return augmented


def export_cellrank_sample_matrix(
    genes: list[str],
    proxy: np.ndarray,
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "sample_id": list(SAMPLES),
            "response": [
                next(
                    response
                    for response, samples in SAMPLES_BY_RESPONSE.items()
                    if sample in samples
                )
                for sample in SAMPLES
            ],
            "method": "CellRank_terminal_proxy",
            "method_label": METHOD_LABELS["CellRank_terminal_proxy"],
            "expression_scale": EXPRESSION_SCALE,
            "method_semantics": METHOD_ROLES["CellRank_terminal_proxy"],
        }
    )
    expression_frame = pd.DataFrame(proxy, columns=genes)
    return pd.concat([metadata, expression_frame], axis=1)


def analyze(
    data: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genes = data["genes"]
    methods = data["methods"]
    responses = data["responses"]
    expression = data["expression"]
    metric_rows: list[dict[str, object]] = []
    bootstrap_value_rows: list[dict[str, object]] = []
    bootstrap_summary_rows: list[dict[str, object]] = []
    gene_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for response in RESPONSES:
        observed_rows = expression[
            (methods == "Observed_Post") & (responses == response)
        ]
        observed_mean = observed_rows.mean(axis=0)
        decile_labels, decile_groups = observed_deciles(observed_mean)
        high_indexes = decile_groups[-1]
        n_observed = len(observed_rows)
        observed_draws = rng.integers(
            0,
            n_observed,
            size=(BOOTSTRAP_ITERATIONS, n_observed),
        )
        virtual_draws = rng.integers(
            0,
            3,
            size=(BOOTSTRAP_ITERATIONS, 3),
        )

        for method in METHODS:
            predicted_rows = expression[
                (methods == method) & (responses == response)
            ]
            predicted_mean = predicted_rows.mean(axis=0)
            point = compute_metrics(predicted_mean, observed_mean, high_indexes)
            for metric in METRICS:
                metric_rows.append(
                    {
                        "response": response,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "method_semantics": METHOD_ROLES[method],
                        "metric": metric,
                        "value": point[metric],
                        "n_virtual_source_samples": len(predicted_rows),
                        "n_observed_post_biopsies": n_observed,
                        "gene_count": len(genes),
                        "expression_scale": EXPRESSION_SCALE,
                    }
                )

            error = predicted_mean - observed_mean
            for index, gene in enumerate(genes):
                gene_rows.append(
                    {
                        "response": response,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "method_semantics": METHOD_ROLES[method],
                        "gene": gene,
                        "observed_expression": observed_mean[index],
                        "predicted_expression": predicted_mean[index],
                        "signed_error": error[index],
                        "absolute_error": abs(error[index]),
                        "observed_expression_decile": int(decile_labels[index]),
                    }
                )

            distributions = {metric: np.empty(BOOTSTRAP_ITERATIONS) for metric in METRICS}
            for iteration in range(BOOTSTRAP_ITERATIONS):
                observed_boot = observed_rows[observed_draws[iteration]].mean(axis=0)
                predicted_boot = predicted_rows[virtual_draws[iteration]].mean(axis=0)
                values = compute_metrics(
                    predicted_boot,
                    observed_boot,
                    high_indexes,
                )
                for metric in METRICS:
                    distributions[metric][iteration] = values[metric]
                    bootstrap_value_rows.append(
                        {
                            "response": response,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "metric": metric,
                            "bootstrap_iteration": iteration,
                            "value": values[metric],
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "bootstrap_unit": "sample/biopsy",
                        }
                    )
            for metric in METRICS:
                values = distributions[metric]
                bootstrap_summary_rows.append(
                    {
                        "response": response,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "metric": metric,
                        "point_estimate": point[metric],
                        "bootstrap_mean": float(values.mean()),
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_unit": "sample/biopsy",
                        "virtual_resampling": (
                            "3 source samples with replacement within response"
                        ),
                        "observed_resampling": (
                            f"{n_observed} Post biopsies with replacement within response"
                        ),
                        "high_expression_gene_set": (
                            "fixed top Observed-expression rank decile from "
                            "point-estimate group mean"
                        ),
                    }
                )

        observed_decile_means = {
            decile: float(observed_mean[indexes].mean())
            for decile, indexes in enumerate(decile_groups, start=1)
        }
        for method in ("Observed_Post", *METHODS):
            if method == "Observed_Post":
                values = observed_mean
                label = "Observed"
            else:
                values = expression[
                    (methods == method) & (responses == response)
                ].mean(axis=0)
                label = METHOD_LABELS[method]
            for decile, indexes in enumerate(decile_groups, start=1):
                selected = values[indexes]
                decile_rows.append(
                    {
                        "response": response,
                        "method": method,
                        "method_label": label,
                        "observed_expression_decile": decile,
                        "decile_label": f"D{decile}",
                        "gene_count": len(indexes),
                        "observed_mean_expression": observed_decile_means[decile],
                        "mean_post_expression": float(selected.mean()),
                        "sem_across_genes": float(
                            selected.std(ddof=1) / np.sqrt(len(selected))
                        ),
                        "decile_definition": (
                            "rank-based decile defined only by Observed Post "
                            "group-mean expression"
                        ),
                    }
                )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(bootstrap_value_rows),
        pd.DataFrame(bootstrap_summary_rows),
        pd.DataFrame(gene_rows),
        pd.DataFrame(decile_rows),
    )


def validate_legacy_reproduction(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, object]:
    frozen_metrics = pd.read_csv(V1_METRICS)
    frozen_bootstrap = pd.read_csv(V1_BOOTSTRAP)
    metric_differences: list[float] = []
    bootstrap_differences: list[float] = []
    records: list[dict[str, object]] = []
    for response in RESPONSES:
        for method in LEGACY_METHODS:
            for metric in METRICS:
                new_metric = metrics[
                    metrics["response"].eq(response)
                    & metrics["method"].eq(method)
                    & metrics["metric"].eq(metric)
                ]
                old_metric = frozen_metrics[
                    frozen_metrics["response"].eq(response)
                    & frozen_metrics["method"].eq(method)
                    & frozen_metrics["metric"].eq(metric)
                ]
                new_boot = bootstrap[
                    bootstrap["response"].eq(response)
                    & bootstrap["method"].eq(method)
                    & bootstrap["metric"].eq(metric)
                ]
                old_boot = frozen_bootstrap[
                    frozen_bootstrap["response"].eq(response)
                    & frozen_bootstrap["method"].eq(method)
                    & frozen_bootstrap["metric"].eq(metric)
                ]
                if not all(len(frame) == 1 for frame in (new_metric, old_metric, new_boot, old_boot)):
                    raise RuntimeError(
                        f"Missing legacy validation row: {response}/{method}/{metric}"
                    )
                metric_difference = abs(
                    float(new_metric.iloc[0]["value"])
                    - float(old_metric.iloc[0]["value"])
                )
                metric_differences.append(metric_difference)
                row_boot_differences = {
                    column: abs(
                        float(new_boot.iloc[0][column])
                        - float(old_boot.iloc[0][column])
                    )
                    for column in (
                        "point_estimate",
                        "bootstrap_mean",
                        "ci95_low",
                        "ci95_high",
                    )
                }
                bootstrap_differences.extend(row_boot_differences.values())
                records.append(
                    {
                        "response": response,
                        "method": method,
                        "metric": metric,
                        "point_metric_absolute_difference": metric_difference,
                        "bootstrap_absolute_differences": row_boot_differences,
                    }
                )
    max_metric_difference = max(metric_differences)
    max_bootstrap_difference = max(bootstrap_differences)
    if max_metric_difference > 1e-12 or max_bootstrap_difference > 1e-12:
        raise RuntimeError(
            "Legacy result reproduction failed: "
            f"metrics={max_metric_difference}, bootstrap={max_bootstrap_difference}"
        )
    return {
        "status": "PASS",
        "legacy_methods": list(LEGACY_METHODS),
        "metric_rows_checked": len(metric_differences),
        "max_metric_absolute_difference": max_metric_difference,
        "bootstrap_values_checked": len(bootstrap_differences),
        "max_bootstrap_summary_absolute_difference": max_bootstrap_difference,
        "tolerance": 1e-12,
        "records": records,
    }


def draw_scatter(
    gene_values: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    values = np.concatenate(
        [
            gene_values["observed_expression"].to_numpy(float),
            gene_values["predicted_expression"].to_numpy(float),
        ]
    )
    span = float(values.max() - values.min())
    lower = float(values.min() - 0.015 * span)
    upper = float(values.max() + 0.035 * span)
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 9.35), sharex=True, sharey=True)
    panel_letters = "ABCDEFGH"
    panel_index = 0
    for row, response in enumerate(RESPONSES):
        for column, method in enumerate(METHODS):
            ax = axes[row, column]
            selected = (
                gene_values[
                    gene_values["response"].eq(response)
                    & gene_values["method"].eq(method)
                ]
                .sort_values("gene")
                .reset_index(drop=True)
            )
            observed = selected["observed_expression"].to_numpy(float)
            predicted = selected["predicted_expression"].to_numpy(float)
            slope, intercept = np.polyfit(observed, predicted, 1)
            metric_map = (
                metrics[
                    metrics["response"].eq(response)
                    & metrics["method"].eq(method)
                ]
                .set_index("metric")["value"]
                .to_dict()
            )
            color = COLORS[method]
            ax.scatter(
                observed,
                predicted,
                s=9,
                alpha=0.30,
                color=color,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            ax.plot(
                [lower, upper],
                [lower, upper],
                "--",
                color="#92999F",
                lw=1.0,
                zorder=1,
            )
            x_line = np.asarray([lower, upper])
            ax.plot(
                x_line,
                slope * x_line + intercept,
                color=color,
                lw=1.8,
                zorder=3,
            )
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                f"{panel_letters[panel_index]}. {response} \u2014 {METHOD_LABELS[method]}",
                loc="left",
                fontsize=10.2,
                fontweight="bold",
            )
            panel_index += 1
            ax.text(
                0.035,
                0.96,
                (
                    f"Pearson r = {metric_map['Pearson']:.3f}\n"
                    f"CCC = {metric_map['CCC']:.3f}\n"
                    f"RMSE = {metric_map['RMSE']:.3f}\n"
                    f"slope = {slope:.3f}"
                ),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#D5D8DA",
                    "alpha": 0.90,
                },
            )
            ax.grid(True, color="#E7EAEC", lw=0.65, alpha=0.75)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7.5)
            ax.set_xlabel("Observed Post expression", fontsize=8.2)
            if column == 0:
                ax.set_ylabel("Predicted / proxy expression", fontsize=8.2)

    fig.suptitle(
        "GSE120575 expanded gene-expression benchmark",
        y=0.985,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.947,
        (
            "834 exact common genes \u00b7 log1p sample-level pseudo-bulk TPM-like "
            "expression \u00b7 observed anti-PD1 Post reference"
        ),
        ha="center",
        fontsize=10.2,
        color="#444444",
    )
    fig.text(
        0.5,
        0.020,
        (
            "WOT* is target-derived.  \u2020CellRank is not a native Post-expression "
            "forecast; it is a target-blind fate-weighted proxy from Pre terminal states.\n"
            "AgentVC is an offline reconstructed checkpoint-5 runtime-expression proxy."
        ),
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#333333",
    )
    fig.tight_layout(rect=(0.035, 0.070, 0.995, 0.925), w_pad=1.1, h_pad=4.2)
    fig.subplots_adjust(hspace=0.30)
    fig.savefig(SCATTER_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(SCATTER_PDF, bbox_inches="tight")
    plt.close(fig)


def draw_violin(
    ax: plt.Axes,
    values_by_method: dict[str, np.ndarray],
    annotation_by_method: dict[str, str],
    y_limits: tuple[float, float],
    jitter_seed: int,
) -> None:
    positions = np.arange(len(METHODS))
    rng = np.random.default_rng(jitter_seed)
    for position, method in zip(positions, METHODS):
        values = np.asarray(values_by_method[method], dtype=float)
        violin = ax.violinplot(
            [values],
            positions=[position],
            widths=0.74,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor(COLORS[method])
            body.set_alpha(0.23)
        ax.boxplot(
            [values],
            positions=[position],
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": "white",
                "edgecolor": COLORS[method],
                "linewidth": 1.0,
            },
            medianprops={"color": COLORS[method], "linewidth": 1.3},
            whiskerprops={"color": COLORS[method], "linewidth": 0.8},
            capprops={"color": COLORS[method], "linewidth": 0.8},
        )
        show_n = min(180, len(values))
        selected = rng.choice(len(values), size=show_n, replace=False)
        jitter = rng.normal(position, 0.055, size=show_n)
        ax.scatter(
            jitter,
            values[selected],
            s=5,
            alpha=0.15,
            color=COLORS[method],
            edgecolors="none",
            rasterized=True,
        )
        ax.text(
            position,
            y_limits[1] - 0.035 * (y_limits[1] - y_limits[0]),
            annotation_by_method[method],
            ha="center",
            va="top",
            fontsize=6.2,
            color="#444444",
        )
    ax.set_xlim(-0.55, len(METHODS) - 0.45)
    ax.set_ylim(*y_limits)
    ax.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS])
    ax.tick_params(axis="x", labelsize=7.8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(True, axis="y", color="#E3E7E9", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_score_figure(
    metrics: pd.DataFrame,
    bootstrap_values: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    gene_values: pd.DataFrame,
    decile_values: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.1, 9.0))
    for row, response in enumerate(RESPONSES):
        pearson_values: dict[str, np.ndarray] = {}
        pearson_annotations: dict[str, str] = {}
        for method in METHODS:
            values = bootstrap_values[
                bootstrap_values["response"].eq(response)
                & bootstrap_values["method"].eq(method)
                & bootstrap_values["metric"].eq("Pearson")
            ]["value"].to_numpy(float)
            pearson_values[method] = values
            point = float(
                metrics[
                    metrics["response"].eq(response)
                    & metrics["method"].eq(method)
                    & metrics["metric"].eq("Pearson")
                ].iloc[0]["value"]
            )
            ci = bootstrap_summary[
                bootstrap_summary["response"].eq(response)
                & bootstrap_summary["method"].eq(method)
                & bootstrap_summary["metric"].eq("Pearson")
            ].iloc[0]
            pearson_annotations[method] = (
                f"r={point:.3f}\n"
                f"CI {float(ci['ci95_low']):.3f}\u2013{float(ci['ci95_high']):.3f}"
            )
        all_pearson = np.concatenate(list(pearson_values.values()))
        pearson_limits = (
            min(0.0, float(all_pearson.min()) - 0.05),
            min(1.08, float(all_pearson.max()) + 0.10),
        )
        draw_violin(
            axes[row, 0],
            pearson_values,
            pearson_annotations,
            pearson_limits,
            BOOTSTRAP_SEED + row,
        )
        axes[row, 0].set_title(
            f"{'A' if row == 0 else 'D'}. {response}: bootstrap Pearson",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
        axes[row, 0].set_ylabel("Bootstrap Pearson r", fontsize=8.5)

        signed_values: dict[str, np.ndarray] = {}
        signed_annotations: dict[str, str] = {}
        for method in METHODS:
            selected = gene_values[
                gene_values["response"].eq(response)
                & gene_values["method"].eq(method)
            ]
            values = selected["signed_error"].to_numpy(float)
            signed_values[method] = values
            signed_annotations[method] = (
                f"median {np.median(values):.3f}\n"
                f"MAE {selected['absolute_error'].mean():.3f}"
            )
        signed_max = max(
            1.0,
            float(
                np.quantile(
                    np.abs(np.concatenate(list(signed_values.values()))),
                    0.995,
                )
                * 1.22
            ),
        )
        draw_violin(
            axes[row, 1],
            signed_values,
            signed_annotations,
            (-signed_max, signed_max),
            BOOTSTRAP_SEED + 10 + row,
        )
        axes[row, 1].axhline(0, color="#656D72", ls="--", lw=1.0)
        axes[row, 1].set_title(
            f"{'B' if row == 0 else 'E'}. {response}: signed error",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
        axes[row, 1].set_ylabel(
            "Signed error (proxy/predicted \u2212 Observed)",
            fontsize=8.5,
        )

        ax = axes[row, 2]
        for method in ("Observed_Post", *METHODS):
            selected = decile_values[
                decile_values["response"].eq(response)
                & decile_values["method"].eq(method)
            ].sort_values("observed_expression_decile")
            ax.plot(
                selected["observed_expression_decile"],
                selected["mean_post_expression"],
                color=COLORS[method],
                marker=MARKERS[method],
                ms=4.2,
                lw=1.7,
                label=(
                    "Observed"
                    if method == "Observed_Post"
                    else METHOD_LABELS[method]
                ),
            )
        ax.set_xticks(range(1, 11), [f"D{i}" for i in range(1, 11)])
        ax.set_xlabel("Observed-expression decile", fontsize=8.5)
        ax.set_ylabel("Mean Post / proxy expression", fontsize=8.5)
        ax.set_title(
            f"{'C' if row == 0 else 'F'}. {response}: decile calibration",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.grid(True, color="#E3E7E9", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            lw=1.8,
            ms=5,
            label=("Observed" if method == "Observed_Post" else METHOD_LABELS[method]),
        )
        for method in ("Observed_Post", *METHODS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.910),
        ncol=5,
        frameon=False,
        fontsize=8.7,
    )
    fig.suptitle(
        "GSE120575 expanded gene-expression benchmark",
        y=0.982,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.944,
        (
            "834 exact common genes \u00b7 log1p sample-level pseudo-bulk TPM-like "
            "expression \u00b7 observed anti-PD1 Post reference"
        ),
        ha="center",
        fontsize=10.2,
        color="#444444",
    )
    fig.text(
        0.5,
        0.020,
        (
            "WOT* is target-derived.  \u2020CellRank is not a native Post-expression "
            "forecast; it is a target-blind fate-weighted proxy from Pre terminal states.\n"
            "AgentVC is an offline reconstructed checkpoint-5 runtime-expression proxy."
        ),
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#333333",
    )
    fig.tight_layout(rect=(0.035, 0.065, 0.995, 0.875), w_pad=2.0, h_pad=2.6)
    fig.savefig(SCORE_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(SCORE_PDF, bbox_inches="tight")
    plt.close(fig)


def write_summary(metrics: pd.DataFrame) -> None:
    point = metrics.pivot_table(
        index=["response", "method_label"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    lines = [
        "# GSE120575基因表达CellRank扩展比较V3",
        "",
        "状态：`PASS_POST_HOC_EXPANDED_BASELINE_COMPARISON`。",
        "",
        "## 语义",
        "",
        "- CellRank没有原生生成Post表达。本版本只用Pre terminal-state表达和"
        "fate probabilities构建无Post选择的表达proxy。",
        "- WOT*仍是target-derived transport baseline。",
        "- AgentVC仍是checkpoint-5 offline reconstructed runtime-expression proxy。",
        "- 本版本是post-hoc扩展比较，不是预注册主benchmark。",
        "",
        "## 主要指标",
        "",
        "| Response | Method | Pearson | CCC | RMSE | MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in point.itertuples(index=False):
        lines.append(
            f"| {row.response} | {row.method_label} | "
            f"{row.Pearson:.3f} | {row.CCC:.3f} | "
            f"{row.RMSE:.3f} | {row.MAE:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "AgentVC相对CellRank-derived proxy和scGen在两个response group中具有更高"
            "的一致性；Responder中AgentVC的CCC、RMSE和MAE最佳。Non-responder中"
            "target-derived WOT*仍最佳，因此不构造overall winner分数。",
            "",
        ]
    )
    (OUT / "RESULTS_SUMMARY_CN.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def refresh_manifest() -> None:
    files = sorted(
        path for path in OUT.iterdir() if path.is_file() and path.name != "figure_manifest.json"
    )
    manifest = {
        "version": "GSE120575_unified_expression_benchmark_v3_cellrank_proxy",
        "status": "POST_HOC_EXPANDED_BASELINE_COMPARISON",
        "files": [
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
        "figure_source_data": {
            SCATTER_PNG.name: "unified_gene_level_plotting_source.csv",
            SCATTER_PDF.name: "unified_gene_level_plotting_source.csv",
            SCORE_PNG.name: [
                "bootstrap_metric_values.csv",
                "unified_gene_level_plotting_source.csv",
                "expression_decile_calibration.csv",
            ],
            SCORE_PDF.name: [
                "bootstrap_metric_values.csv",
                "unified_gene_level_plotting_source.csv",
                "expression_decile_calibration.csv",
            ],
        },
    }
    write_json(OUT / "figure_manifest.json", manifest)


def mark_visual_pass() -> None:
    audit_path = OUT / "V3_AUDIT.json"
    if not audit_path.exists() or not all(
        path.exists() for path in (SCATTER_PNG, SCATTER_PDF, SCORE_PNG, SCORE_PDF)
    ):
        raise RuntimeError("V3 outputs do not exist; build before visual finalization")
    visual = {
        "status": "PASS_NO_CLIPPING_OVERLAP_OR_UNREADABLE_LABELS",
        "inspection_scope": [
            SCATTER_PNG.name,
            SCORE_PNG.name,
        ],
        "inspection_type": "manual visual inspection after generation",
    }
    write_json(OUT / "visual_inspection.json", visual)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["visual_inspection"] = visual["status"]
    audit["status"] = "PASS_GSE120575_EXPRESSION_V3_CELLRANK_PROXY"
    write_json(audit_path, audit)
    refresh_manifest()
    print("VISUAL INSPECTION FINALIZED")


def build(allow_rebuild: bool = False) -> None:
    if OUT.exists():
        existing = list(OUT.iterdir())
        if existing and not allow_rebuild:
            raise FileExistsError(
                f"Refusing to overwrite non-empty V3 output directory: {OUT}"
            )
        if existing and allow_rebuild:
            audit_path = OUT / "V3_AUDIT.json"
            if not audit_path.exists():
                raise RuntimeError("Cannot identify existing V3 output for safe rebuild")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("status") != "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION":
                raise RuntimeError(
                    "Safe rebuild is allowed only before visual finalization"
                )
    OUT.mkdir(parents=True, exist_ok=True)

    genes, source_hashes = validate_sources()
    protocol = {
        "status": "FROZEN_FORMULA_BEFORE_V3_OUTPUT_GENERATION",
        "version": "GSE120575_unified_expression_benchmark_v3_cellrank_proxy",
        "comparison_semantics": "post-hoc expanded baseline comparison",
        "gene_count": 834,
        "samples_by_response": {
            response: list(samples)
            for response, samples in SAMPLES_BY_RESPONSE.items()
        },
        "cellrank_method_id": "CellRank_terminal_proxy",
        "cellrank_label": METHOD_LABELS["CellRank_terminal_proxy"],
        "formula": [
            "read Pre-only CellRank log1p(TPM) on frozen 834 genes",
            "expm1 to TPM",
            "mean exactly 30 terminal-state cells per each of four lineages",
            "normalize each Pre-cell fate probability row to sum one",
            "mean normalized fate weights within each of six source samples",
            "mix four terminal-state TPM centroids by sample fate weights",
            "log1p mixed TPM to common sample-level scale",
        ],
        "forbidden": [
            "best_observed_post_aligned_terminal_lineage",
            "is_best_observed_post_aligned_terminal_lineage",
            "Observed Post distance or performance for lineage selection",
            "CellRank model rerun",
            "parameter changes based on V3 performance",
        ],
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "unit": "sample/biopsy",
            "common_draws_across_methods": True,
        },
        "source_hashes": source_hashes,
    }
    write_json(OUT / "protocol_frozen.json", protocol)
    write_json(
        OUT / "source_hashes.json",
        {
            "hash_algorithm": "SHA256",
            "sources": source_hashes,
        },
    )

    proxy, fate_weights, terminal_centroids, cellrank_checks = build_cellrank_proxy(
        genes
    )
    sample_matrix = export_cellrank_sample_matrix(genes, proxy)
    sample_matrix.to_csv(OUT / "cellrank_expression_proxy_by_sample.csv", index=False)
    fate_weights.to_csv(OUT / "cellrank_fate_weights.csv", index=False)
    terminal_centroids.to_csv(OUT / "cellrank_terminal_centroids.csv", index=False)
    np.savez_compressed(
        OUT / "cellrank_expression_proxy_834.npz",
        genes=np.asarray(genes),
        sample_ids=np.asarray(SAMPLES),
        responses=sample_matrix["response"].to_numpy(str),
        method=np.asarray(["CellRank_terminal_proxy"]),
        expression_log1p_sample_pseudobulk=proxy,
    )

    data = load_augmented_data(genes, proxy)
    (
        metrics,
        bootstrap_values,
        bootstrap_summary,
        gene_values,
        decile_values,
    ) = analyze(data)
    legacy_reproduction = validate_legacy_reproduction(metrics, bootstrap_summary)

    metrics.to_csv(OUT / "unified_expression_metrics.csv", index=False)
    bootstrap_values.to_csv(OUT / "bootstrap_metric_values.csv", index=False)
    bootstrap_summary.to_csv(
        OUT / "bootstrap_confidence_intervals.csv",
        index=False,
    )
    gene_values.to_csv(
        OUT / "unified_gene_level_plotting_source.csv",
        index=False,
    )
    decile_values.to_csv(
        OUT / "expression_decile_calibration.csv",
        index=False,
    )

    draw_scatter(gene_values, metrics)
    draw_score_figure(
        metrics,
        bootstrap_values,
        bootstrap_summary,
        gene_values,
        decile_values,
    )
    write_summary(metrics)

    output_checks = {
        "cellrank_proxy_shape_6_by_834": proxy.shape == (6, 834),
        "cellrank_proxy_finite": bool(np.isfinite(proxy).all()),
        "cellrank_proxy_nonnegative": bool(np.all(proxy >= 0)),
        "metrics_rows": len(metrics),
        "bootstrap_value_rows": len(bootstrap_values),
        "bootstrap_summary_rows": len(bootstrap_summary),
        "gene_plotting_rows": len(gene_values),
        "decile_rows": len(decile_values),
        "methods": list(METHODS),
        "responses": list(RESPONSES),
        "legacy_reproduction": legacy_reproduction,
    }
    if (
        len(metrics) != 56
        or len(bootstrap_values) != 56000
        or len(bootstrap_summary) != 56
        or len(gene_values) != 6672
        or len(decile_values) != 100
    ):
        raise RuntimeError("V3 output table dimensions failed")
    if not all(path.exists() and path.stat().st_size > 0 for path in (
        SCATTER_PNG,
        SCATTER_PDF,
        SCORE_PNG,
        SCORE_PDF,
    )):
        raise RuntimeError("One or more V3 figure files are missing")

    visual = {
        "status": "PENDING_MANUAL_VISUAL_INSPECTION",
        "inspection_scope": [SCATTER_PNG.name, SCORE_PNG.name],
    }
    write_json(OUT / "visual_inspection.json", visual)
    audit = {
        "status": "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION",
        "comparison_semantics": "post-hoc expanded baseline comparison",
        "models_rerun": False,
        "cellrank_native_post_expression_claimed": False,
        "cellrank_proxy_semantics": METHOD_ROLES["CellRank_terminal_proxy"],
        "cellrank_checks": cellrank_checks,
        "output_checks": output_checks,
        "source_hashes": source_hashes,
        "historical_outputs_deleted_or_overwritten": False,
        "visual_inspection": visual["status"],
    }
    write_json(OUT / "V3_AUDIT.json", audit)
    refresh_manifest()

    point = metrics.pivot_table(
        index=["response", "method_label"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    print("GSE120575 CELLRANK EXPRESSION PROXY V3")
    print(f"- Output: {OUT}")
    print("- Numeric validation: PASS")
    print("- Visual validation: PENDING")
    for response in RESPONSES:
        row = point.loc[(response, METHOD_LABELS["CellRank_terminal_proxy"])]
        print(
            f"- {response} CellRank\u2020: Pearson={row['Pearson']:.6f}, "
            f"CCC={row['CCC']:.6f}, RMSE={row['RMSE']:.6f}, MAE={row['MAE']:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mark-visual-pass",
        action="store_true",
        help="Finalize the audit after manual inspection of both PNG figures.",
    )
    parser.add_argument(
        "--rebuild-current",
        action="store_true",
        help="Rebuild only an existing non-finalized V3 output directory.",
    )
    args = parser.parse_args()
    if args.mark_visual_pass:
        mark_visual_pass()
    else:
        build(allow_rebuild=args.rebuild_current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
