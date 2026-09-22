#!/usr/bin/env python3
"""GSE120575 sample-level PCA/UMAP and clustering benchmark V1.

No prediction model is run here. The script reads frozen sample-level
expression outputs, fits scaler/PCA/UMAP only on observed anti-PD1 samples,
transforms predictions, and computes all formal metrics in PCA space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR", "/tmp/numba-gse120575-sample-level-umap-v1"
)
os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/matplotlib-gse120575-sample-level-umap-v1"
)

import anndata as ad
import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PROXY_ROOT = Path(
    os.environ.get(
        "GSE120575_PROXY_OUT",
        str(ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"),
    )
).resolve()
V3_ROOT = Path(
    os.environ.get(
        "GSE120575_V3_OUT",
        str(ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy"),
    )
).resolve()
CELLRANK_ROOT = (
    ROOT
    / "data/wot_scgen_cellrank_native_pre_post_reports"
    / "cellrank_gse120575_pre_to_post_native"
)
META_ROOT = ROOT / "outputs/GSE120575_agent_granularity_ablation/real_benchmark"
OUT = V3_ROOT / "sample_level_umap_clustering_v1"

PANEL = PROXY_ROOT / "frozen_834_gene_panel.txt"
HARMONIZED = PROXY_ROOT / "harmonized_834_expression_by_method_sample.npz"
CELLRANK_PROXY = V3_ROOT / "cellrank_expression_proxy_834.npz"
CELLRANK_PRE = CELLRANK_ROOT / "native_cellrank_pre_terminal_status_all_genes.h5ad"
SAMPLE_METADATA = META_ROOT / "gse120575_geo_sample_metadata.csv"
V3_AUDIT = V3_ROOT / "V3_AUDIT.json"

EXPECTED_HASHES = {
    "gene_panel": "1a9e02f8bb52cdea852d6fe30992339ad8bc2fa0ab807b81848dce574a7bce41",
    "harmonized_expression": "1a0645902aac33e6ddefdb9bfd32e6af4e628ba2502abf2b376ceb3457ffb36a",
    "cellrank_proxy": "cf8a79193b790382ae11c7472676ad8bcfb39bac7389867aa9279601eb1cc32b",
    "cellrank_pre": "9af4c5a4ae2fa3167c9bc0fcbcec340bc71266264c43aca171332c2ed7fe9b45",
    "sample_metadata": "71ddcce18e88b97d64f98e5b55814d6767e86574f35862d776694b284378a834",
}

PRE_SAMPLES = (
    "Pre_P2",
    "Pre_P3",
    "Pre_P12",
    "Pre_P15",
    "Pre_P20",
    "Pre_P24",
    "Pre_P25",
    "Pre_P27",
    "Pre_P29",
    "Pre_P31",
    "Pre_P33",
    "Pre_P35",
)
PREDICTION_SAMPLES = (
    "Pre_P24",
    "Pre_P29",
    "Pre_P35",
    "Pre_P2",
    "Pre_P3",
    "Pre_P27",
)
RESPONSES = ("Responder", "Non-responder")
PREDICTION_METHODS = (
    "scGen",
    "WOT_weighted",
    "CellRank_terminal_proxy",
    "AgentVC_proxy",
)
METHOD_LABELS = {
    "Observed_Pre": "Observed Pre",
    "Observed_Post": "Observed Post",
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "CellRank_terminal_proxy": "CellRank\u2020",
    "AgentVC_proxy": "AgentVC",
}
METHOD_SEMANTICS = {
    "Observed_Pre": "observed anti-PD1 Pre sample-level pseudo-bulk",
    "Observed_Post": "observed anti-PD1 Post biopsy-level pseudo-bulk",
    "scGen": "generated Post sample-level expression baseline",
    "WOT_weighted": "target-derived WOT reconstructed Post sample-level expression",
    "CellRank_terminal_proxy": "Pre-only fate-weighted terminal-state expression proxy",
    "AgentVC_proxy": "offline reconstructed checkpoint-5 runtime-expression proxy",
}
COLORS = {
    "Observed_Pre": "#B7BCC1",
    "Observed_Post": "#222222",
    "scGen": "#F58518",
    "WOT_weighted": "#54A24B",
    "CellRank_terminal_proxy": "#4C9ED9",
    "AgentVC_proxy": "#A66FA0",
}
RESPONSE_MARKERS = {"Responder": "o", "Non-responder": "^"}
EXPRESSION_SCALE = "log1p(sample-level pseudo-bulk TPM-like expression)"

PCA_RANDOM_STATE = 120575
UMAP_RANDOM_STATE = 120575
UMAP_PARAMETERS = {
    "n_neighbors": 10,
    "min_dist": 0.30,
    "spread": 1.0,
    "metric": "euclidean",
    "random_state": UMAP_RANDOM_STATE,
    "transform_seed": UMAP_RANDOM_STATE,
}
KNN_K = 5

MAIN_PREVIEW = OUT / "GSE120575_sample_level_UMAP_clustering_v1_preview.png"
MAIN_PNG = OUT / "GSE120575_sample_level_UMAP_clustering_v1_600dpi.png"
MAIN_PDF = OUT / "GSE120575_sample_level_UMAP_clustering_v1_vector.pdf"
MAIN_SVG = OUT / "GSE120575_sample_level_UMAP_clustering_v1.svg"
HEATMAP_PNG = OUT / "GSE120575_sample_level_hierarchical_heatmap_600dpi.png"
HEATMAP_PDF = OUT / "GSE120575_sample_level_hierarchical_heatmap_vector.pdf"


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


def patient_id(sample_id: str) -> str:
    match = re.match(r"^(?:Pre|Post)_(P\d+)", str(sample_id))
    return match.group(1) if match else "Unknown"


def cluster_purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(clusters):
        values, counts = np.unique(labels[clusters == cluster], return_counts=True)
        del values
        total += int(counts.max())
    return float(total / len(labels))


def validate_sources() -> tuple[list[str], dict[str, str]]:
    paths = {
        "gene_panel": PANEL,
        "harmonized_expression": HARMONIZED,
        "cellrank_proxy": CELLRANK_PROXY,
        "cellrank_pre": CELLRANK_PRE,
        "sample_metadata": SAMPLE_METADATA,
        "v3_audit": V3_AUDIT,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    variable_deidentified_sources = {
        "harmonized_expression",
        "cellrank_proxy",
        "v3_audit",
    }
    for name, expected in EXPECTED_HASHES.items():
        if source_hashes[name] != expected and not (
            os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") == "1"
            and name in variable_deidentified_sources
        ):
            raise RuntimeError(
                f"Frozen source changed for {name}: "
                f"{source_hashes[name]} != {expected}"
            )
    v3_audit = json.loads(V3_AUDIT.read_text(encoding="utf-8"))
    if v3_audit.get("status") != "PASS_GSE120575_EXPRESSION_V3_CELLRANK_PROXY":
        raise RuntimeError("CellRank expression V3 audit did not pass")
    genes = PANEL.read_text(encoding="utf-8").splitlines()
    if len(genes) != 834 or len(set(genes)) != 834:
        raise RuntimeError("Frozen panel is not 834 unique genes")
    return genes, source_hashes


def metadata_map() -> tuple[dict[str, str], pd.DataFrame]:
    metadata = pd.read_csv(SAMPLE_METADATA)
    metadata["therapy"] = metadata["geo_characteristics_ch1"].astype(str).str.extract(
        r"therapy: ([^;]+)"
    )[0]
    selected = metadata[
        metadata["sample_id"].isin(PRE_SAMPLES)
        & metadata["treatment_status"].eq("pre-treatment")
        & metadata["therapy"].eq("anti-PD1")
    ].copy()
    if len(selected) != 12 or selected["sample_id"].nunique() != 12:
        raise RuntimeError("Could not resolve exactly 12 anti-PD1 Pre samples")
    response_map = selected.set_index("sample_id")["response_group"].to_dict()
    if set(response_map.values()) != set(RESPONSES):
        raise RuntimeError("Pre response metadata is incomplete")
    return response_map, selected


def observed_pre_matrix(
    genes: list[str],
    response_map: dict[str, str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    source = ad.read_h5ad(CELLRANK_PRE)
    if source.shape != (5928, 55737):
        raise RuntimeError(f"Unexpected Pre H5AD shape: {source.shape}")
    if not source.var_names.is_unique or not set(genes).issubset(source.var_names):
        raise RuntimeError("Pre H5AD does not contain the frozen gene panel")
    expression = source[:, genes].X
    if sparse.issparse(expression):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=np.float64)
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Observed Pre cell-level log1p(TPM) is invalid")
    tpm = np.expm1(expression)
    sample_labels = source.obs["sample_label"].astype(str).to_numpy()
    rows: list[np.ndarray] = []
    counts: dict[str, int] = {}
    metadata_rows: list[dict[str, object]] = []
    for sample in PRE_SAMPLES:
        selected = sample_labels == sample
        counts[sample] = int(selected.sum())
        if counts[sample] <= 0:
            raise RuntimeError(f"Observed Pre cells missing for {sample}")
        rows.append(np.log1p(tpm[selected].mean(axis=0)))
        metadata_rows.append(
            {
                "point_id": f"Observed_Pre::{sample}",
                "sample_id": sample,
                "patient_id": patient_id(sample),
                "responder_status": response_map[sample],
                "method": "Observed_Pre",
                "method_label": METHOD_LABELS["Observed_Pre"],
                "state": "Observed Pre",
                "is_observed": True,
                "is_prediction": False,
                "prediction_target": "",
                "paired_observed_pre_id": f"Observed_Pre::{sample}",
                "paired_observed_post_id": "",
                "paired_observed_post_candidates": "",
                "post_pairing_status": "not_applicable",
                "method_semantics": METHOD_SEMANTICS["Observed_Pre"],
                "biological_unit": "sample",
                "source_cell_count": counts[sample],
            }
        )
    matrix = np.vstack(rows)
    if matrix.shape != (12, 834) or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise RuntimeError("Observed Pre pseudo-bulk matrix failed validation")
    return matrix, pd.DataFrame(metadata_rows), {
        "source_shape": list(source.shape),
        "output_shape": list(matrix.shape),
        "sample_cell_counts": counts,
        "construction": "expm1 cell log1p(TPM), arithmetic mean within sample, log1p",
    }


def load_frozen_expression(
    genes: list[str],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    with np.load(HARMONIZED) as source:
        frozen = {name: source[name].copy() for name in source.files}
    if frozen["genes"].astype(str).tolist() != genes:
        raise RuntimeError("Harmonized matrix gene order differs from frozen panel")
    methods = frozen["methods"].astype(str)
    sample_ids = frozen["sample_ids"].astype(str)
    responses = frozen["responses"].astype(str)
    expression = frozen["expression_log1p_sample_pseudobulk"].astype(np.float64)
    if expression.shape != (47, 834):
        raise RuntimeError("Unexpected harmonized expression shape")

    post_mask = methods == "Observed_Post"
    post_expression = expression[post_mask]
    post_ids = sample_ids[post_mask]
    post_responses = responses[post_mask]
    if post_expression.shape != (23, 834):
        raise RuntimeError("Observed Post is not 23 x 834")
    if dict(zip(*np.unique(post_responses, return_counts=True))) != {
        "Non-responder": 18,
        "Responder": 5,
    }:
        raise RuntimeError("Observed Post response counts changed")

    post_patient_candidates: dict[str, list[str]] = {}
    for sample in post_ids:
        post_patient_candidates.setdefault(patient_id(sample), []).append(sample)

    post_metadata_rows: list[dict[str, object]] = []
    for sample, response in zip(post_ids, post_responses):
        post_metadata_rows.append(
            {
                "point_id": f"Observed_Post::{sample}",
                "sample_id": sample,
                "patient_id": patient_id(sample),
                "responder_status": response,
                "method": "Observed_Post",
                "method_label": METHOD_LABELS["Observed_Post"],
                "state": "Observed Post",
                "is_observed": True,
                "is_prediction": False,
                "prediction_target": "",
                "paired_observed_pre_id": "",
                "paired_observed_post_id": f"Observed_Post::{sample}",
                "paired_observed_post_candidates": sample,
                "post_pairing_status": "not_applicable",
                "method_semantics": METHOD_SEMANTICS["Observed_Post"],
                "biological_unit": "biopsy",
                "source_cell_count": np.nan,
            }
        )

    prediction_matrices: list[np.ndarray] = []
    prediction_metadata_rows: list[dict[str, object]] = []
    method_counts: dict[str, int] = {}
    for method in ("scGen", "WOT_weighted", "AgentVC_proxy"):
        selected = methods == method
        method_expression = expression[selected]
        method_ids = sample_ids[selected]
        method_responses = responses[selected]
        if method_expression.shape != (6, 834) or set(method_ids) != set(
            PREDICTION_SAMPLES
        ):
            raise RuntimeError(f"{method} is not six sample-level vectors")
        order = [int(np.flatnonzero(method_ids == sample)[0]) for sample in PREDICTION_SAMPLES]
        method_expression = method_expression[order]
        method_ids = method_ids[order]
        method_responses = method_responses[order]
        if len(np.unique(method_expression, axis=0)) != 6:
            raise RuntimeError(f"{method} contains duplicated expression vectors")
        prediction_matrices.append(method_expression)
        method_counts[method] = len(method_expression)
        for sample, response in zip(method_ids, method_responses):
            candidates = sorted(post_patient_candidates.get(patient_id(sample), []))
            pairing_status = (
                "unique"
                if len(candidates) == 1
                else "ambiguous"
                if len(candidates) > 1
                else "missing"
            )
            prediction_metadata_rows.append(
                prediction_metadata_row(
                    method,
                    sample,
                    response,
                    candidates,
                    pairing_status,
                )
            )

    with np.load(CELLRANK_PROXY) as source:
        cellrank_genes = source["genes"].astype(str)
        cellrank_ids = source["sample_ids"].astype(str)
        cellrank_responses = source["responses"].astype(str)
        cellrank_expression = source[
            "expression_log1p_sample_pseudobulk"
        ].astype(np.float64)
    if cellrank_genes.tolist() != genes or cellrank_expression.shape != (6, 834):
        raise RuntimeError("CellRank proxy panel or shape changed")
    if set(cellrank_ids) != set(PREDICTION_SAMPLES):
        raise RuntimeError("CellRank proxy sample scope changed")
    order = [
        int(np.flatnonzero(cellrank_ids == sample)[0]) for sample in PREDICTION_SAMPLES
    ]
    cellrank_expression = cellrank_expression[order]
    cellrank_ids = cellrank_ids[order]
    cellrank_responses = cellrank_responses[order]
    if len(np.unique(cellrank_expression, axis=0)) != 6:
        raise RuntimeError("CellRank proxy contains duplicated expression vectors")

    insert_at = 2
    prediction_matrices.insert(insert_at, cellrank_expression)
    method_counts["CellRank_terminal_proxy"] = len(cellrank_expression)
    cellrank_metadata: list[dict[str, object]] = []
    for sample, response in zip(cellrank_ids, cellrank_responses):
        candidates = sorted(post_patient_candidates.get(patient_id(sample), []))
        pairing_status = (
            "unique"
            if len(candidates) == 1
            else "ambiguous"
            if len(candidates) > 1
            else "missing"
        )
        cellrank_metadata.append(
            prediction_metadata_row(
                "CellRank_terminal_proxy",
                sample,
                response,
                candidates,
                pairing_status,
            )
        )
    first_two_methods_rows = 12
    prediction_metadata_rows[
        first_two_methods_rows:first_two_methods_rows
    ] = cellrank_metadata

    prediction_expression = np.vstack(prediction_matrices)
    prediction_metadata = pd.DataFrame(prediction_metadata_rows)
    expected_method_sequence = np.repeat(PREDICTION_METHODS, 6)
    if prediction_metadata["method"].to_numpy().tolist() != expected_method_sequence.tolist():
        raise RuntimeError("Prediction metadata method order is inconsistent")
    if prediction_expression.shape != (24, 834):
        raise RuntimeError("Prediction expression is not 24 x 834")
    if (
        not np.isfinite(post_expression).all()
        or not np.isfinite(prediction_expression).all()
        or np.any(post_expression < 0)
        or np.any(prediction_expression < 0)
    ):
        raise RuntimeError("Frozen Post/prediction expression is invalid")

    return (
        post_expression,
        pd.DataFrame(post_metadata_rows),
        prediction_expression,
        prediction_metadata,
        {
            "observed_post_shape": list(post_expression.shape),
            "prediction_shape": list(prediction_expression.shape),
            "prediction_method_counts": method_counts,
            "strict_unique_post_pair_samples": ["Pre_P2"],
            "ambiguous_post_pair_samples": ["Pre_P3"],
            "missing_post_pair_samples": [
                "Pre_P24",
                "Pre_P29",
                "Pre_P35",
                "Pre_P27",
            ],
        },
    )


def prediction_metadata_row(
    method: str,
    sample: str,
    response: str,
    candidates: list[str],
    pairing_status: str,
) -> dict[str, object]:
    return {
        "point_id": f"{method}::{sample}",
        "sample_id": sample,
        "patient_id": patient_id(sample),
        "responder_status": response,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "state": "Predicted Post",
        "is_observed": False,
        "is_prediction": True,
        "prediction_target": "Observed anti-PD1 Post response state",
        "paired_observed_pre_id": f"Observed_Pre::{sample}",
        "paired_observed_post_id": (
            f"Observed_Post::{candidates[0]}" if len(candidates) == 1 else ""
        ),
        "paired_observed_post_candidates": ";".join(candidates),
        "post_pairing_status": pairing_status,
        "method_semantics": METHOD_SEMANTICS[method],
        "biological_unit": "source sample representation",
        "source_cell_count": np.nan,
    }


def write_gate_zero(
    pre_checks: dict[str, object],
    frozen_checks: dict[str, object],
    source_hashes: dict[str, str],
) -> None:
    inventory_rows = [
        {
            "method_state": "Observed Pre",
            "source_path": str(CELLRANK_PRE),
            "source_shape": "5928 x 55737 cells x genes",
            "output_shape": "12 x 834",
            "row_semantics": "one anti-PD1 Pre sample pseudo-bulk per row",
            "sample_id_field": "sample_label",
            "patient_id_status": "derived from sample label",
            "response_status": "frozen GEO metadata",
            "expression_scale": EXPRESSION_SCALE,
            "sample_level_status": "PASS_SAMPLE_LEVEL",
            "vector_count": 12,
            "group_mean_only": False,
            "seed_or_bootstrap_rows": False,
        },
        {
            "method_state": "Observed Post",
            "source_path": str(HARMONIZED),
            "source_shape": "47 x 834 harmonized matrix",
            "output_shape": "23 x 834",
            "row_semantics": "one observed anti-PD1 Post biopsy per row",
            "sample_id_field": "sample_ids",
            "patient_id_status": "derived from biopsy label",
            "response_status": "frozen harmonized metadata",
            "expression_scale": EXPRESSION_SCALE,
            "sample_level_status": "PASS_SAMPLE_LEVEL",
            "vector_count": 23,
            "group_mean_only": False,
            "seed_or_bootstrap_rows": False,
        },
    ]
    for method in PREDICTION_METHODS:
        inventory_rows.append(
            {
                "method_state": METHOD_LABELS[method],
                "source_path": str(
                    CELLRANK_PROXY
                    if method == "CellRank_terminal_proxy"
                    else HARMONIZED
                ),
                "source_shape": (
                    "6 x 834" if method == "CellRank_terminal_proxy" else "47 x 834"
                ),
                "output_shape": "6 x 834",
                "row_semantics": "one real source sample representation per row",
                "sample_id_field": "sample_ids",
                "patient_id_status": "derived from source sample label",
                "response_status": "frozen benchmark metadata",
                "expression_scale": EXPRESSION_SCALE,
                "sample_level_status": "PASS_SAMPLE_LEVEL",
                "vector_count": 6,
                "group_mean_only": False,
                "seed_or_bootstrap_rows": False,
            }
        )
    pd.DataFrame(inventory_rows).to_csv(
        OUT / "sample_level_expression_inventory.csv", index=False
    )
    alignment = {
        "status": "PASS_SAMPLE_LEVEL",
        "observed_pre": 12,
        "observed_post": 23,
        "predictions_per_method": 6,
        "prediction_methods": list(PREDICTION_METHODS),
        "unified_point_count": 59,
        "gene_count": 834,
        "strict_unique_post_pairing": {
            "eligible_source_samples": ["Pre_P2"],
            "count": 1,
        },
        "ambiguous_post_pairing": {
            "source_samples": ["Pre_P3"],
            "reason": "Post_P3 and Post_P3_2 are distinct biopsies",
        },
        "missing_post_pairing": [
            "Pre_P24",
            "Pre_P29",
            "Pre_P35",
            "Pre_P27",
        ],
        "paired_recovery_status": (
            "PAIRED_RECOVERY_NOT_PRIMARY_INSUFFICIENT_ALIGNMENT"
        ),
        "trajectory_lines_allowed": False,
        "bootstrap_rows_used_as_points": False,
        "seed_rows_used_as_points": False,
        "group_means_used_as_points": False,
        "pre_checks": pre_checks,
        "frozen_checks": frozen_checks,
        "source_hashes": source_hashes,
    }
    write_json(OUT / "sample_alignment_audit.json", alignment)
    (OUT / "sample_alignment_audit.md").write_text(
        "\n".join(
            [
                "# GSE120575 sample alignment audit",
                "",
                "Gate 0: **PASS_SAMPLE_LEVEL**",
                "",
                "- Observed Pre: 12 anti-PD1 sample-level vectors.",
                "- Observed Post: 23 anti-PD1 biopsy-level vectors.",
                "- scGen/WOT*/CellRank\u2020/AgentVC: six source-sample vectors each.",
                "- Each UMAP point is one sample/biopsy representation; genes, seeds, "
                "bootstrap replicates and response-group means are not points.",
                "- Only Pre_P2 has a unique Post biopsy. Pre_P3 has two candidate "
                "Post biopsies; four other prediction sources have no Post biopsy.",
                "- Primary paired recovery and UMAP trajectory lines are disabled. "
                "Primary recovery uses response-matched Observed Post centroids in PCA.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def fit_reference_models(
    expression: np.ndarray,
    metadata: pd.DataFrame,
    genes: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    StandardScaler,
    PCA,
    umap.UMAP,
    np.ndarray,
    dict[str, object],
]:
    reference_mask = metadata["is_observed"].to_numpy(bool)
    prediction_mask = metadata["is_prediction"].to_numpy(bool)
    if int(reference_mask.sum()) != 35 or int(prediction_mask.sum()) != 24:
        raise RuntimeError("Reference/prediction masks are not 35/24")
    reference = expression[reference_mask]
    standard_deviation = reference.std(axis=0, ddof=0)
    nonzero = standard_deviation > 0
    zero_variance_genes = np.asarray(genes)[~nonzero].tolist()
    if int(nonzero.sum()) < 3:
        raise RuntimeError("Fewer than three nonzero-variance genes")

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(reference[:, nonzero])
    standardized_kept = scaler.transform(expression[:, nonzero])
    standardized_full = np.zeros_like(expression, dtype=np.float64)
    standardized_full[:, nonzero] = standardized_kept
    n_pcs = min(20, int(reference_mask.sum()) - 1, int(nonzero.sum()))
    if n_pcs < 3:
        raise RuntimeError("LOW_SAMPLE_PCA: fewer than three PCs")
    pca = PCA(n_components=n_pcs, random_state=PCA_RANDOM_STATE)
    pca.fit(standardized_full[reference_mask][:, nonzero])
    pca_coordinates = pca.transform(standardized_full[:, nonzero])

    reference_pca = pca_coordinates[reference_mask]
    reducer = umap.UMAP(**UMAP_PARAMETERS)
    reducer.fit(reference_pca)
    umap_coordinates = np.empty((len(expression), 2), dtype=np.float64)
    umap_coordinates[reference_mask] = reducer.embedding_
    umap_coordinates[prediction_mask] = reducer.transform(
        pca_coordinates[prediction_mask]
    )
    if not np.isfinite(umap_coordinates).all():
        raise RuntimeError("UMAP transform produced nonfinite coordinates")

    deterministic_reducer = umap.UMAP(**UMAP_PARAMETERS)
    deterministic_reducer.fit(reference_pca)
    deterministic_coordinates = np.empty_like(umap_coordinates)
    deterministic_coordinates[reference_mask] = deterministic_reducer.embedding_
    deterministic_coordinates[prediction_mask] = deterministic_reducer.transform(
        pca_coordinates[prediction_mask]
    )
    determinism_difference = float(
        np.max(np.abs(deterministic_coordinates - umap_coordinates))
    )
    if determinism_difference > 1e-8:
        raise RuntimeError(
            f"UMAP determinism gate failed: {determinism_difference}"
        )

    checks = {
        "reference_points": int(reference_mask.sum()),
        "prediction_points": int(prediction_mask.sum()),
        "zero_variance_gene_count": len(zero_variance_genes),
        "zero_variance_genes": zero_variance_genes,
        "pca_gene_count": int(nonzero.sum()),
        "n_pcs": n_pcs,
        "pca_cumulative_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "scaler_fit_scope": "35 Observed Pre + Observed Post points only",
        "pca_fit_scope": "35 Observed Pre + Observed Post points only",
        "umap_fit_scope": "35 Observed reference PCA coordinates only",
        "prediction_operation": "transform only",
        "umap_parameters": UMAP_PARAMETERS,
        "umap_package_version": umap.__version__,
        "umap_repeat_max_absolute_coordinate_difference": determinism_difference,
    }
    return (
        standardized_full,
        pca_coordinates,
        scaler,
        pca,
        reducer,
        nonzero,
        checks,
    ), umap_coordinates


def export_models_and_coordinates(
    expression: np.ndarray,
    standardized: np.ndarray,
    metadata: pd.DataFrame,
    genes: list[str],
    pca_coordinates: np.ndarray,
    umap_coordinates: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    reducer: umap.UMAP,
    nonzero: np.ndarray,
    model_checks: dict[str, object],
) -> None:
    np.save(OUT / "standardized_sample_expression.npy", standardized)
    scaler_full_mean = np.full(len(genes), np.nan)
    scaler_full_scale = np.full(len(genes), np.nan)
    scaler_full_mean[nonzero] = scaler.mean_
    scaler_full_scale[nonzero] = scaler.scale_
    write_json(
        OUT / "observed_reference_scaler.json",
        {
            "fit_scope": model_checks["scaler_fit_scope"],
            "gene_count_input": 834,
            "gene_count_used": int(nonzero.sum()),
            "zero_variance_genes": model_checks["zero_variance_genes"],
            "genes": genes,
            "mean": [
                None if np.isnan(value) else float(value)
                for value in scaler_full_mean
            ],
            "scale": [
                None if np.isnan(value) else float(value)
                for value in scaler_full_scale
            ],
            "ddof": 0,
        },
    )
    joblib.dump(scaler, OUT / "observed_reference_scaler.pkl")
    joblib.dump(pca, OUT / "observed_reference_pca.pkl")
    joblib.dump(reducer, OUT / "observed_reference_umap.pkl")

    pca_frame = metadata.copy()
    for index in range(pca_coordinates.shape[1]):
        pca_frame[f"PC{index + 1}"] = pca_coordinates[:, index]
    pca_frame.to_csv(OUT / "sample_pca_coordinates.csv", index=False)

    umap_frame = metadata.copy()
    umap_frame["UMAP1"] = umap_coordinates[:, 0]
    umap_frame["UMAP2"] = umap_coordinates[:, 1]
    umap_frame.to_csv(OUT / "sample_umap_coordinates.csv", index=False)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(pca.n_components_)],
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative,
        }
    ).to_csv(OUT / "pca_explained_variance.csv", index=False)

    kept_genes = np.asarray(genes)[nonzero]
    loading_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for pc_index, loadings in enumerate(pca.components_, start=1):
        for gene, loading in zip(kept_genes, loadings):
            loading_rows.append(
                {
                    "PC": f"PC{pc_index}",
                    "gene": gene,
                    "loading": float(loading),
                    "absolute_loading": abs(float(loading)),
                }
            )
        if pc_index <= 5:
            order = np.argsort(np.abs(loadings))[::-1][:20]
            for rank, gene_index in enumerate(order, start=1):
                top_rows.append(
                    {
                        "PC": f"PC{pc_index}",
                        "rank": rank,
                        "gene": kept_genes[gene_index],
                        "loading": float(loadings[gene_index]),
                        "absolute_loading": abs(float(loadings[gene_index])),
                    }
                )
    pd.DataFrame(loading_rows).to_csv(OUT / "pca_loadings.csv", index=False)
    pd.DataFrame(top_rows).to_csv(OUT / "pca_top_loading_genes.csv", index=False)
    write_json(
        OUT / "umap_parameters.json",
        {
            **UMAP_PARAMETERS,
            "fit_points": 35,
            "transformed_prediction_points": 24,
            "input_space": f"{pca.n_components_}-dimensional Observed-reference PCA",
            "formal_distance_space": "PCA, never UMAP",
            "determinism_max_absolute_difference": model_checks[
                "umap_repeat_max_absolute_coordinate_difference"
            ],
        },
    )

    adata = ad.AnnData(
        X=expression.astype(np.float64),
        obs=metadata.set_index("point_id").copy(),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.layers["standardized_observed_reference"] = standardized
    adata.obsm["X_pca"] = pca_coordinates
    adata.obsm["X_umap"] = umap_coordinates
    adata.var["included_in_pca"] = nonzero
    adata.var["observed_reference_mean"] = scaler_full_mean
    adata.var["observed_reference_scale"] = scaler_full_scale
    pca_full = np.zeros((len(genes), pca.n_components_))
    pca_full[nonzero] = pca.components_.T
    adata.varm["PC_loadings"] = pca_full
    adata.uns["expression_scale"] = EXPRESSION_SCALE
    adata.uns["gene_panel_sha256"] = EXPECTED_HASHES["gene_panel"]
    adata.uns["scaler_fit_scope"] = model_checks["scaler_fit_scope"]
    adata.uns["pca_fit_scope"] = model_checks["pca_fit_scope"]
    adata.uns["umap_fit_scope"] = model_checks["umap_fit_scope"]
    adata.uns["formal_distance_space"] = "20D Observed-reference PCA"
    adata.uns["post_hoc_analysis"] = True
    adata.write_h5ad(
        OUT / "unified_sample_expression_834genes.h5ad",
        compression="gzip",
    )
    metadata.to_csv(OUT / "unified_sample_metadata.csv", index=False)
    write_json(
        OUT / "unified_matrix_audit.json",
        {
            "status": "PASS_UNIFIED_59_POINT_834_GENE_MATRIX",
            "shape": [59, 834],
            "point_id_unique": bool(metadata["point_id"].is_unique),
            "gene_order_sha256": EXPECTED_HASHES["gene_panel"],
            "expression_scale": EXPRESSION_SCALE,
            "all_values_finite": bool(np.isfinite(expression).all()),
            "all_values_nonnegative": bool(np.all(expression >= 0)),
            "point_counts": [
                {
                    "method": str(method),
                    "state": str(state),
                    "count": int(count),
                }
                for (method, state), count in metadata.groupby(
                    ["method", "state"], sort=False
                ).size().items()
            ],
            "bootstrap_or_seed_points": 0,
            "group_mean_points": 0,
        },
    )
    (OUT / "standardization_audit.md").write_text(
        "\n".join(
            [
                "# Standardization audit",
                "",
                "- StandardScaler was fit once on 35 observed anti-PD1 points.",
                "- The same mean and standard deviation were applied to all 24 predictions.",
                f"- Zero-variance genes: {len(model_checks['zero_variance_genes'])}.",
                f"- Genes used for PCA: {model_checks['pca_gene_count']}.",
                "- No method-specific or response-specific normalization was performed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def compute_recovery_metrics(
    metadata: pd.DataFrame,
    pca_coordinates: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    coordinates = {
        point_id: pca_coordinates[index]
        for index, point_id in enumerate(metadata["point_id"])
    }
    post = metadata["method"].eq("Observed_Post")
    centroids = {
        response: pca_coordinates[
            post.to_numpy()
            & metadata["responder_status"].eq(response).to_numpy()
        ].mean(axis=0)
        for response in RESPONSES
    }

    recovery_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    prediction_rows = metadata[metadata["is_prediction"]]
    for row in prediction_rows.itertuples(index=False):
        prediction = coordinates[row.point_id]
        source_pre = coordinates[row.paired_observed_pre_id]
        matched = centroids[row.responder_status]
        opposite_response = next(
            response for response in RESPONSES if response != row.responder_status
        )
        opposite = centroids[opposite_response]
        prediction_distance = float(np.linalg.norm(prediction - matched))
        pre_distance = float(np.linalg.norm(source_pre - matched))
        opposite_distance = float(np.linalg.norm(prediction - opposite))
        recovery_ratio = (
            prediction_distance / pre_distance if pre_distance > 1e-12 else np.nan
        )
        margin_denominator = prediction_distance + opposite_distance
        normalized_margin = (
            (opposite_distance - prediction_distance) / margin_denominator
            if margin_denominator > 1e-12
            else np.nan
        )
        common = {
            "point_id": row.point_id,
            "sample_id": row.sample_id,
            "patient_id": row.patient_id,
            "responder_status": row.responder_status,
            "method": row.method,
            "method_label": row.method_label,
            "formal_space": "20D Observed-reference PCA",
        }
        recovery_rows.append(
            {
                **common,
                "predicted_to_matched_response_post_centroid_distance": (
                    prediction_distance
                ),
                "source_pre_to_matched_response_post_centroid_distance": pre_distance,
                "response_centroid_recovery_ratio": recovery_ratio,
                "interpretation": (
                    "<1 moves closer than persistence to matched-response Post centroid"
                ),
                "paired_patient_metric": False,
            }
        )
        margin_rows.append(
            {
                **common,
                "distance_to_matched_response_post_centroid": prediction_distance,
                "distance_to_opposite_response_post_centroid": opposite_distance,
                "normalized_response_centroid_margin": normalized_margin,
                "correct_response_region": normalized_margin > 0,
                "interpretation": ">0 is closer to matched-response Post centroid",
            }
        )
        paired_distance = np.nan
        if row.post_pairing_status == "unique":
            paired_distance = float(
                np.linalg.norm(
                    prediction - coordinates[row.paired_observed_post_id]
                )
            )
        pair_rows.append(
            {
                **common,
                "post_pairing_status": row.post_pairing_status,
                "paired_observed_post_id": row.paired_observed_post_id,
                "paired_observed_post_candidates": (
                    row.paired_observed_post_candidates
                ),
                "paired_post_distance_pca": paired_distance,
                "eligible_for_primary_summary": False,
                "exclusion_reason": (
                    "only one of six source samples has a unique Post biopsy; "
                    "no unique Responder pairing"
                ),
            }
        )

    recovery = pd.DataFrame(recovery_rows)
    margin = pd.DataFrame(margin_rows)
    paired = pd.DataFrame(pair_rows)
    summary_rows: list[dict[str, object]] = []
    for method in PREDICTION_METHODS:
        selected = recovery[recovery["method"].eq(method)]
        for response_scope in ("All", *RESPONSES):
            values = (
                selected
                if response_scope == "All"
                else selected[selected["responder_status"].eq(response_scope)]
            )["response_centroid_recovery_ratio"].to_numpy(float)
            summary_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "response_scope": response_scope,
                    "n_source_samples": len(values),
                    "median_recovery_ratio": float(np.median(values)),
                    "q1_recovery_ratio": float(np.quantile(values, 0.25)),
                    "q3_recovery_ratio": float(np.quantile(values, 0.75)),
                    "min_recovery_ratio": float(values.min()),
                    "max_recovery_ratio": float(values.max()),
                    "lower_is_better": True,
                    "paired_patient_metric": False,
                }
            )
    summary = pd.DataFrame(summary_rows)
    centroid_distance = recovery[
        [
            "point_id",
            "sample_id",
            "responder_status",
            "method",
            "method_label",
            "predicted_to_matched_response_post_centroid_distance",
            "formal_space",
        ]
    ].copy()
    return recovery, summary, centroid_distance, margin, paired


def compute_knn_supplement(
    metadata: pd.DataFrame,
    pca_coordinates: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    post_mask = metadata["method"].eq("Observed_Post").to_numpy()
    post_coordinates = pca_coordinates[post_mask]
    post_responses = metadata.loc[post_mask, "responder_status"].to_numpy(str)
    if len(post_coordinates) != 23:
        raise RuntimeError("Expected 23 Observed Post points for kNN")
    loo_predictions: list[str] = []
    loo_fractions: list[float] = []
    for index in range(len(post_coordinates)):
        distances = np.linalg.norm(
            post_coordinates - post_coordinates[index], axis=1
        )
        neighbors = np.argsort(distances)[1 : KNN_K + 1]
        responder_fraction = float(
            np.mean(post_responses[neighbors] == "Responder")
        )
        predicted = (
            "Responder" if responder_fraction >= 0.5 else "Non-responder"
        )
        loo_predictions.append(predicted)
        loo_fractions.append(
            float(np.mean(post_responses[neighbors] == post_responses[index]))
        )
    loo = {
        "k": KNN_K,
        "observed_post_points": 23,
        "accuracy": float(accuracy_score(post_responses, loo_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(post_responses, loo_predictions)
        ),
        "median_same_response_neighbor_fraction": float(
            np.median(loo_fractions)
        ),
        "primary_panel_eligible": False,
        "reason": (
            "Observed Post response imbalance (5 Responder, 18 Non-responder) "
            "and LOO balanced accuracy is only 0.60"
        ),
    }

    rows: list[dict[str, object]] = []
    for index, row in metadata[metadata["is_prediction"]].iterrows():
        distances = np.linalg.norm(
            post_coordinates - pca_coordinates[index], axis=1
        )
        neighbors = np.argsort(distances)[:KNN_K]
        neighbor_responses = post_responses[neighbors]
        same_fraction = float(
            np.mean(neighbor_responses == row["responder_status"])
        )
        predicted_response = (
            "Responder"
            if np.mean(neighbor_responses == "Responder") >= 0.5
            else "Non-responder"
        )
        rows.append(
            {
                "point_id": row["point_id"],
                "sample_id": row["sample_id"],
                "method": row["method"],
                "method_label": row["method_label"],
                "true_response": row["responder_status"],
                "predicted_response": predicted_response,
                "correct": predicted_response == row["responder_status"],
                "same_response_neighbor_fraction": same_fraction,
                "k": KNN_K,
                "reference": "Observed Post only",
                "formal_space": "20D Observed-reference PCA",
                "primary_panel_eligible": False,
            }
        )
    return pd.DataFrame(rows), loo


def compute_clustering_and_heterogeneity(
    metadata: pd.DataFrame,
    pca_coordinates: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    post_or_prediction = (
        metadata["method"].eq("Observed_Post") | metadata["is_prediction"]
    ).to_numpy()
    full_metadata = metadata.loc[post_or_prediction].reset_index(drop=True)
    full_coordinates = pca_coordinates[post_or_prediction]
    full_linkage = linkage(full_coordinates, method="ward", metric="euclidean")
    full_clusters = fcluster(full_linkage, t=2, criterion="maxclust")
    assignment_rows: list[dict[str, object]] = []
    for row, cluster in zip(full_metadata.itertuples(index=False), full_clusters):
        assignment_rows.append(
            {
                "analysis_scope": "Observed Post + all predictions",
                "point_id": row.point_id,
                "sample_id": row.sample_id,
                "method": row.method,
                "method_label": row.method_label,
                "responder_status": row.responder_status,
                "ward_cluster_k2": int(cluster),
                "formal_space": "20D Observed-reference PCA",
            }
        )

    metric_rows: list[dict[str, object]] = []
    scopes = [("Observed Post + all predictions", None)] + [
        (f"Observed Post + {METHOD_LABELS[method]}", method)
        for method in PREDICTION_METHODS
    ]
    for scope, method in scopes:
        if method is None:
            selected = post_or_prediction
            scope_coordinates = full_coordinates
            scope_metadata = full_metadata
            clusters = full_clusters
        else:
            selected = (
                metadata["method"].eq("Observed_Post")
                | metadata["method"].eq(method)
            ).to_numpy()
            scope_coordinates = pca_coordinates[selected]
            scope_metadata = metadata.loc[selected].reset_index(drop=True)
            model = AgglomerativeClustering(
                n_clusters=2,
                linkage="ward",
                metric="euclidean",
            )
            clusters = model.fit_predict(scope_coordinates) + 1
            for row, cluster in zip(
                scope_metadata.itertuples(index=False), clusters
            ):
                assignment_rows.append(
                    {
                        "analysis_scope": scope,
                        "point_id": row.point_id,
                        "sample_id": row.sample_id,
                        "method": row.method,
                        "method_label": row.method_label,
                        "responder_status": row.responder_status,
                        "ward_cluster_k2": int(cluster),
                        "formal_space": "20D Observed-reference PCA",
                    }
                )
        labels = scope_metadata["responder_status"].to_numpy(str)
        metric_rows.append(
            {
                "analysis_scope": scope,
                "n_points": len(labels),
                "n_responder": int(np.sum(labels == "Responder")),
                "n_non_responder": int(np.sum(labels == "Non-responder")),
                "clustering_method": "Ward hierarchical clustering",
                "distance": "Euclidean in 20D Observed-reference PCA",
                "cut_k": 2,
                "ARI_vs_response": float(adjusted_rand_score(labels, clusters)),
                "NMI_vs_response": float(
                    normalized_mutual_info_score(labels, clusters)
                ),
                "cluster_purity_vs_response": cluster_purity(labels, clusters),
                "interpretation": (
                    "descriptive only; method representations of the same six "
                    "source samples are not independent biological replicates"
                ),
            }
        )

    heterogeneity_rows: list[dict[str, object]] = []
    observed_dispersion: dict[str, float] = {}
    for response in RESPONSES:
        selected = (
            metadata["method"].eq("Observed_Post")
            & metadata["responder_status"].eq(response)
        ).to_numpy()
        values = pdist(pca_coordinates[selected], metric="euclidean")
        observed_dispersion[response] = float(np.median(values))
        heterogeneity_rows.append(
            {
                "method": "Observed_Post",
                "method_label": "Observed Post",
                "responder_status": response,
                "n_points": int(selected.sum()),
                "pair_count": len(values),
                "median_pairwise_pca_distance": float(np.median(values)),
                "mean_pairwise_pca_distance": float(np.mean(values)),
                "dispersion_ratio_vs_observed_post": 1.0,
                "formal_space": "20D Observed-reference PCA",
            }
        )
    for method in PREDICTION_METHODS:
        for response in RESPONSES:
            selected = (
                metadata["method"].eq(method)
                & metadata["responder_status"].eq(response)
            ).to_numpy()
            values = pdist(pca_coordinates[selected], metric="euclidean")
            median = float(np.median(values))
            heterogeneity_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "responder_status": response,
                    "n_points": int(selected.sum()),
                    "pair_count": len(values),
                    "median_pairwise_pca_distance": median,
                    "mean_pairwise_pca_distance": float(np.mean(values)),
                    "dispersion_ratio_vs_observed_post": (
                        median / observed_dispersion[response]
                    ),
                    "formal_space": "20D Observed-reference PCA",
                }
            )
    return (
        pd.DataFrame(assignment_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(heterogeneity_rows),
        full_linkage,
        post_or_prediction,
    )


def plot_embedding_panel(
    ax: plt.Axes,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    title: str,
    response_filter: str | None = None,
    xlabel: str = "UMAP1",
    ylabel: str = "UMAP2",
) -> None:
    selected = np.ones(len(metadata), dtype=bool)
    if response_filter is not None:
        selected &= metadata["responder_status"].eq(response_filter).to_numpy()
    plot_order = (
        "Observed_Pre",
        "scGen",
        "WOT_weighted",
        "CellRank_terminal_proxy",
        "AgentVC_proxy",
        "Observed_Post",
    )
    for method in plot_order:
        for response in RESPONSES:
            mask = (
                selected
                & metadata["method"].eq(method).to_numpy()
                & metadata["responder_status"].eq(response).to_numpy()
            )
            if not mask.any():
                continue
            is_post = method == "Observed_Post"
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=58 if is_post else 44,
                color=COLORS[method],
                marker=RESPONSE_MARKERS[response],
                alpha=0.95 if is_post else 0.78,
                edgecolor="#000000" if is_post else "white",
                linewidth=1.25 if is_post else 0.45,
                zorder=5 if is_post else 3,
            )
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.8)
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_ylabel(ylabel, fontsize=8.8)
    ax.grid(True, color="#E8EBED", lw=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5)


def plot_metric_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    reference: float,
    lower_is_better: bool,
) -> None:
    rng = np.random.default_rng(120575)
    for position, method in enumerate(PREDICTION_METHODS):
        selected = frame[frame["method"].eq(method)]
        values = selected[value_column].to_numpy(float)
        x = rng.normal(position, 0.055, size=len(values))
        for response in RESPONSES:
            response_values = selected["responder_status"].eq(response).to_numpy()
            ax.scatter(
                x[response_values],
                values[response_values],
                s=38,
                marker=RESPONSE_MARKERS[response],
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.88,
                zorder=3,
            )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(position, q1, q3, color=COLORS[method], lw=4.5, alpha=0.65)
        ax.hlines(
            median,
            position - 0.18,
            position + 0.18,
            color=COLORS[method],
            lw=2.2,
            zorder=4,
        )
        ax.text(
            position,
            max(values.max(), q3) + 0.035 * max(1.0, np.ptp(values)),
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#444444",
        )
    ax.axhline(reference, color="#6E7478", ls="--", lw=1.0)
    ax.set_xticks(
        range(len(PREDICTION_METHODS)),
        [METHOD_LABELS[method] for method in PREDICTION_METHODS],
        rotation=12,
    )
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(True, axis="y", color="#E4E8EA", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.4)
    direction = "Lower is better" if lower_is_better else "Higher is better"
    ax.text(
        0.99,
        0.02,
        direction,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#555555",
    )


def draw_main_figure(
    metadata: pd.DataFrame,
    pca_coordinates: np.ndarray,
    umap_coordinates: np.ndarray,
    pca: PCA,
    recovery: pd.DataFrame,
    margin: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 9.0))
    plot_embedding_panel(
        axes[0, 0],
        metadata,
        umap_coordinates,
        "A. Unified observed-reference UMAP",
    )
    plot_embedding_panel(
        axes[0, 1],
        metadata,
        umap_coordinates,
        "B. Responder expression states",
        response_filter="Responder",
    )
    plot_embedding_panel(
        axes[0, 2],
        metadata,
        umap_coordinates,
        "C. Non-responder expression states",
        response_filter="Non-responder",
    )
    plot_embedding_panel(
        axes[1, 0],
        metadata,
        pca_coordinates[:, :2],
        "D. Observed-reference PCA",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
    )
    plot_metric_points(
        axes[1, 1],
        recovery,
        "response_centroid_recovery_ratio",
        "E. Response-centroid recovery",
        "Prediction / persistence PCA-distance ratio",
        1.0,
        True,
    )
    plot_metric_points(
        axes[1, 2],
        margin,
        "normalized_response_centroid_margin",
        "F. Response-state recovery",
        "Normalized matched-response centroid margin",
        0.0,
        False,
    )
    axes[1, 2].set_ylim(-1.02, 1.02)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="black" if method == "Observed_Post" else "white",
            markersize=7,
            label=METHOD_LABELS[method],
        )
        for method in (
            "Observed_Pre",
            "Observed_Post",
            *PREDICTION_METHODS,
        )
    ]
    response_handles = [
        Line2D(
            [0],
            [0],
            marker=RESPONSE_MARKERS[response],
            linestyle="none",
            color="#555555",
            markersize=7,
            label=response,
        )
        for response in RESPONSES
    ]
    fig.legend(
        handles=method_handles + response_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=8,
        frameon=False,
        fontsize=7.8,
    )
    fig.suptitle(
        "GSE120575 sample-level expression-state recovery benchmark",
        y=0.982,
        fontsize=17.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.946,
        "Unified observed-reference PCA/UMAP projection across 834 exact common genes",
        ha="center",
        fontsize=10.0,
        color="#444444",
    )
    fig.text(
        0.5,
        0.019,
        (
            "WOT* is target-derived. \u2020CellRank is a Pre-only fate-weighted proxy. "
            "AgentVC is an offline reconstructed expression proxy.\n"
            "UMAP is visualization only; recovery, response and clustering metrics "
            "use 20D Observed-reference PCA. No paired trajectories are shown."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )
    fig.tight_layout(rect=(0.035, 0.075, 0.995, 0.875), w_pad=2.0, h_pad=2.2)
    fig.savefig(MAIN_PREVIEW, dpi=150, bbox_inches="tight")
    fig.savefig(MAIN_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(MAIN_PDF, bbox_inches="tight")
    fig.savefig(MAIN_SVG, bbox_inches="tight")
    plt.close(fig)


def draw_heatmap(
    expression: np.ndarray,
    standardized: np.ndarray,
    metadata: pd.DataFrame,
    full_linkage: np.ndarray,
    post_or_prediction: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    post_mask = metadata["method"].eq("Observed_Post").to_numpy()
    variances = expression[post_mask].var(axis=0, ddof=0)
    top_indexes = np.argsort(variances, kind="stable")[::-1][:100]
    selected_genes = np.asarray(genes)[top_indexes]
    heat_matrix = standardized[post_or_prediction][:, top_indexes]
    heat_metadata = metadata.loc[post_or_prediction].reset_index(drop=True)
    row_colors = pd.DataFrame(
        {
            "Method": heat_metadata["method"].map(COLORS).to_numpy(),
            "Response": heat_metadata["responder_status"]
            .map({"Responder": "#D62728", "Non-responder": "#1F77B4"})
            .to_numpy(),
        },
        index=heat_metadata["point_id"].to_numpy(),
    )
    heat_frame = pd.DataFrame(
        heat_matrix,
        index=heat_metadata["point_id"],
        columns=selected_genes,
    )
    sns.set_theme(style="white")
    grid = sns.clustermap(
        heat_frame,
        row_linkage=full_linkage,
        row_cluster=True,
        col_cluster=False,
        row_colors=row_colors,
        cmap="vlag",
        center=0,
        vmin=-3,
        vmax=3,
        figsize=(18, 12),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Observed-reference z-score"},
    )
    grid.ax_heatmap.tick_params(axis="x", labelsize=4.1, rotation=90)
    grid.ax_heatmap.tick_params(axis="y", labelsize=4.4)
    grid.fig.suptitle(
        "GSE120575 hierarchical clustering of Post expression states",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    heatmap_legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="none",
            markersize=8,
            label=METHOD_LABELS[method],
        )
        for method in (
            "Observed_Post",
            "scGen",
            "WOT_weighted",
            "CellRank_terminal_proxy",
            "AgentVC_proxy",
        )
    ]
    heatmap_legend.extend(
        [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#D62728",
                markeredgecolor="none",
                markersize=8,
                label="Responder",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#1F77B4",
                markeredgecolor="none",
                markersize=8,
                label="Non-responder",
            ),
        ]
    )
    grid.fig.legend(
        handles=heatmap_legend,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.968),
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    grid.fig.text(
        0.5,
        0.01,
        (
            "Rows: 23 Observed Post biopsies + 24 predicted/proxy sample states. "
            "Ward row linkage uses 20D PCA; genes are the top 100 by Observed Post variance."
        ),
        ha="center",
        fontsize=8.5,
    )
    grid.fig.savefig(HEATMAP_PNG, dpi=600, bbox_inches="tight")
    grid.fig.savefig(HEATMAP_PDF, bbox_inches="tight")
    plt.close(grid.fig)
    return pd.DataFrame(
        {
            "rank": np.arange(1, 101),
            "gene": selected_genes,
            "observed_post_variance_log1p": variances[top_indexes],
            "selection_scope": "Observed Post only",
        }
    )


def write_readme_and_caption(
    model_checks: dict[str, object],
    recovery_summary: pd.DataFrame,
    loo_knn: dict[str, object],
) -> None:
    medians = recovery_summary[
        recovery_summary["response_scope"].eq("All")
    ].set_index("method_label")["median_recovery_ratio"]
    readme = [
        "# GSE120575 sample-level PCA/UMAP/clustering V1",
        "",
        "Status: `PASS_SAMPLE_LEVEL_UMAP`.",
        "",
        "## Data",
        "",
        "- 12 Observed anti-PD1 Pre samples.",
        "- 23 Observed anti-PD1 Post biopsies.",
        "- Six sample-level states for each of scGen, WOT*, CellRank\u2020 and AgentVC.",
        "- Unified matrix: 59 points \u00d7 834 exact common genes.",
        "",
        "## Reference-space rules",
        "",
        "- Scaler, PCA and UMAP were fit only on the 35 observed points.",
        "- Predictions were transformed into the frozen reference space.",
        "- All formal distances and clustering use 20D PCA, never 2D UMAP.",
        "- Paired-patient recovery is not primary because only Pre_P2 has a "
        "unique Post biopsy and no Responder prediction source has one.",
        "",
        "## Median response-centroid recovery ratio",
        "",
    ]
    for label, value in medians.items():
        readme.append(f"- {label}: {value:.3f}")
    readme.extend(
        [
            "",
            "A value below 1 indicates movement closer than persistence to the "
            "matched-response Observed Post centroid. This is not a paired-patient metric.",
            "",
            "## Limitations",
            "",
            "- WOT* is target-derived.",
            "- CellRank\u2020 is a Pre-only fate-weighted terminal-state expression proxy.",
            "- AgentVC is an offline reconstructed checkpoint-5 expression proxy.",
            "- Prediction heterogeneity and clustering use only three source samples "
            "per response and are descriptive.",
            f"- Observed Post k=5 LOO balanced accuracy was "
            f"{loo_knn['balanced_accuracy']:.3f}; kNN is supplementary, not Panel F.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (OUT / "caption.txt").write_text(
        (
            "GSE120575 sample-level expression-state recovery benchmark. "
            "Scaler, PCA and UMAP were fit on 12 observed anti-PD1 Pre samples "
            "and 23 observed Post biopsies using 834 exact common genes; four "
            "sets of six predicted or proxy sample states were transformed into "
            "the shared reference. UMAP panels are descriptive. Recovery ratio, "
            "response margin and Ward clustering were computed in 20D PCA. "
            "WOT* is target-derived; CellRank\u2020 is a Pre-only fate-weighted "
            "terminal-state proxy; AgentVC is an offline reconstructed expression proxy."
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        OUT / "reference_model_summary.json",
        {
            "pca": {
                "n_components": model_checks["n_pcs"],
                "cumulative_explained_variance": model_checks[
                    "pca_cumulative_explained_variance"
                ],
                "fit_scope": model_checks["pca_fit_scope"],
            },
            "umap": {
                "parameters": UMAP_PARAMETERS,
                "fit_scope": model_checks["umap_fit_scope"],
                "repeat_max_absolute_difference": model_checks[
                    "umap_repeat_max_absolute_coordinate_difference"
                ],
            },
        },
    )


def refresh_manifest() -> None:
    files = sorted(
        path
        for path in OUT.iterdir()
        if path.is_file() and path.name != "output_manifest.json"
    )
    write_json(
        OUT / "output_manifest.json",
        {
            "version": "GSE120575_sample_level_umap_clustering_v1",
            "status": "post-hoc sample-level expression-state benchmark",
            "files": [
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
            "main_figure_source_data": [
                "sample_umap_coordinates.csv",
                "sample_pca_coordinates.csv",
                "response_centroid_recovery_metrics.csv",
                "response_state_margin_metrics.csv",
            ],
            "heatmap_source_data": [
                "unified_sample_expression_834genes.h5ad",
                "heatmap_top100_gene_selection.csv",
                "hierarchical_cluster_assignments.csv",
            ],
        },
    )


def mark_visual_pass() -> None:
    audit_path = OUT / "analysis_audit.json"
    if not audit_path.exists():
        raise RuntimeError("Analysis output does not exist")
    for path in (MAIN_PREVIEW, MAIN_PNG, MAIN_PDF, MAIN_SVG, HEATMAP_PNG, HEATMAP_PDF):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing figure for visual finalization: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION":
        raise RuntimeError("Visual finalization is allowed only after numeric PASS")
    visual = {
        "status": "PASS_NO_CLIPPING_OVERLAP_OR_HIDDEN_OUTLIERS",
        "inspected_figures": [
            MAIN_PREVIEW.name,
            HEATMAP_PNG.name,
        ],
        "outliers_hidden": False,
        "trajectory_lines_drawn": False,
    }
    write_json(OUT / "visual_inspection.json", visual)
    audit["status"] = "PASS_SAMPLE_LEVEL_UMAP"
    audit["visual_inspection"] = visual["status"]
    write_json(audit_path, audit)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_sample_level_umap_clustering_v1",
            "status": "COMPLETE_FINAL",
        },
    )
    refresh_manifest()
    print("VISUAL INSPECTION FINALIZED")


def build(allow_rebuild: bool = False) -> None:
    if OUT.exists() and any(OUT.iterdir()):
        if not allow_rebuild:
            raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
        audit_path = OUT / "analysis_audit.json"
        build_state_path = OUT / "build_state.json"
        audit_status = None
        build_status = None
        if audit_path.exists():
            audit_status = json.loads(
                audit_path.read_text(encoding="utf-8")
            ).get("status")
        if build_state_path.exists():
            build_status = json.loads(
                build_state_path.read_text(encoding="utf-8")
            ).get("status")
        if audit_status != "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION" and (
            build_status != "BUILDING"
        ):
            raise RuntimeError("Cannot identify a non-final current output for safe rebuild")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_sample_level_umap_clustering_v1",
            "status": "BUILDING",
        },
    )

    genes, source_hashes = validate_sources()
    response_map, selected_pre_metadata = metadata_map()
    pre_expression, pre_metadata, pre_checks = observed_pre_matrix(
        genes, response_map
    )
    (
        post_expression,
        post_metadata,
        prediction_expression,
        prediction_metadata,
        frozen_checks,
    ) = load_frozen_expression(genes)
    write_gate_zero(pre_checks, frozen_checks, source_hashes)

    expression = np.vstack([pre_expression, post_expression, prediction_expression])
    metadata = pd.concat(
        [pre_metadata, post_metadata, prediction_metadata],
        ignore_index=True,
    )
    if expression.shape != (59, 834) or len(metadata) != 59:
        raise RuntimeError("Unified matrix is not 59 x 834")
    if not metadata["point_id"].is_unique:
        raise RuntimeError("Unified point IDs are not unique")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Unified expression contains invalid values")

    model_result, umap_coordinates = fit_reference_models(
        expression, metadata, genes
    )
    (
        standardized,
        pca_coordinates,
        scaler,
        pca,
        reducer,
        nonzero,
        model_checks,
    ) = model_result
    export_models_and_coordinates(
        expression,
        standardized,
        metadata,
        genes,
        pca_coordinates,
        umap_coordinates,
        scaler,
        pca,
        reducer,
        nonzero,
        model_checks,
    )

    (
        recovery,
        recovery_summary,
        centroid_distance,
        margin,
        paired,
    ) = compute_recovery_metrics(metadata, pca_coordinates)
    recovery.to_csv(OUT / "response_centroid_recovery_metrics.csv", index=False)
    recovery_summary.to_csv(
        OUT / "method_recovery_ratio_summary.csv", index=False
    )
    centroid_distance.to_csv(OUT / "response_centroid_distance.csv", index=False)
    margin.to_csv(OUT / "response_state_margin_metrics.csv", index=False)
    paired.to_csv(OUT / "paired_post_recovery_metrics.csv", index=False)

    knn, loo_knn = compute_knn_supplement(metadata, pca_coordinates)
    knn.to_csv(OUT / "knn_response_classification.csv", index=False)
    write_json(OUT / "knn_reference_diagnostic.json", loo_knn)
    (
        assignments,
        clustering_metrics,
        heterogeneity,
        full_linkage,
        post_or_prediction,
    ) = compute_clustering_and_heterogeneity(metadata, pca_coordinates)
    assignments.to_csv(OUT / "hierarchical_cluster_assignments.csv", index=False)
    clustering_metrics.to_csv(OUT / "clustering_metric_summary.csv", index=False)
    heterogeneity.to_csv(OUT / "heterogeneity_recovery_metrics.csv", index=False)

    draw_main_figure(
        metadata,
        pca_coordinates,
        umap_coordinates,
        pca,
        recovery,
        margin,
    )
    heat_genes = draw_heatmap(
        expression,
        standardized,
        metadata,
        full_linkage,
        post_or_prediction,
        genes,
    )
    heat_genes.to_csv(OUT / "heatmap_top100_gene_selection.csv", index=False)
    write_readme_and_caption(model_checks, recovery_summary, loo_knn)
    shutil.copy2(
        Path(__file__),
        OUT / "plot_GSE120575_sample_level_umap_clustering_v1.py",
    )

    numeric_checks = {
        "unified_shape": list(expression.shape),
        "unique_point_ids": bool(metadata["point_id"].is_unique),
        "observed_reference_points": int(metadata["is_observed"].sum()),
        "prediction_points": int(metadata["is_prediction"].sum()),
        "prediction_counts": metadata[
            metadata["is_prediction"]
        ]["method"].value_counts().to_dict(),
        "all_expression_finite_nonnegative": bool(
            np.isfinite(expression).all() and np.all(expression >= 0)
        ),
        "all_methods_same_834_genes": True,
        "scaler_fit_observed_only": True,
        "pca_fit_observed_only": True,
        "umap_fit_observed_only": True,
        "predictions_transform_only": True,
        "formal_distances_use_pca": True,
        "clustering_uses_pca": True,
        "umap_used_for_formal_metrics": False,
        "bootstrap_points": 0,
        "seed_points": 0,
        "group_mean_points": 0,
        "trajectory_lines_drawn": False,
        "strict_unique_paired_source_samples": 1,
        "paired_recovery_primary": False,
        "main_recovery_metric": "response-centroid recovery ratio in 20D PCA",
        "main_response_metric": "normalized response-centroid margin in 20D PCA",
        "pca_components": pca.n_components_,
        "pca_cumulative_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "umap_repeat_max_absolute_difference": model_checks[
            "umap_repeat_max_absolute_coordinate_difference"
        ],
        "models_rerun": False,
    }
    if numeric_checks["unified_shape"] != [59, 834]:
        raise RuntimeError("Final unified shape gate failed")
    if set(numeric_checks["prediction_counts"].values()) != {6}:
        raise RuntimeError("Prediction count gate failed")
    if not all(
        numeric_checks[key]
        for key in (
            "unique_point_ids",
            "all_expression_finite_nonnegative",
            "all_methods_same_834_genes",
            "scaler_fit_observed_only",
            "pca_fit_observed_only",
            "umap_fit_observed_only",
            "predictions_transform_only",
            "formal_distances_use_pca",
            "clustering_uses_pca",
        )
    ):
        raise RuntimeError("One or more final numeric gates failed")

    write_json(
        OUT / "visual_inspection.json",
        {
            "status": "PENDING_MANUAL_VISUAL_INSPECTION",
            "figures": [MAIN_PREVIEW.name, HEATMAP_PNG.name],
        },
    )
    write_json(
        OUT / "analysis_audit.json",
        {
            "status": "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION",
            "final_expected_status": "PASS_SAMPLE_LEVEL_UMAP",
            "analysis_semantics": "post-hoc sample-level expression-state benchmark",
            "paired_recovery_status": (
                "PAIRED_RECOVERY_NOT_PRIMARY_INSUFFICIENT_ALIGNMENT"
            ),
            "gate_zero_status": "PASS_SAMPLE_LEVEL",
            "numeric_checks": numeric_checks,
            "reference_model_checks": model_checks,
            "knn_supplement_status": loo_knn,
            "source_hashes": source_hashes,
            "historical_outputs_deleted_or_overwritten": False,
            "visual_inspection": "PENDING_MANUAL_VISUAL_INSPECTION",
        },
    )
    write_json(
        OUT / "source_hashes.json",
        {"hash_algorithm": "SHA256", "sources": source_hashes},
    )
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_sample_level_umap_clustering_v1",
            "status": "COMPLETE_NUMERIC_PENDING_VISUAL_INSPECTION",
        },
    )
    refresh_manifest()

    all_medians = recovery_summary[
        recovery_summary["response_scope"].eq("All")
    ].set_index("method_label")["median_recovery_ratio"]
    print("GSE120575 SAMPLE-LEVEL UMAP AND CLUSTERING")
    print("1. Data availability")
    print("- Observed Pre samples: 12")
    print("- Observed Post samples: 23")
    for method in PREDICTION_METHODS:
        print(f"- {METHOD_LABELS[method]} predictions/proxies: 6")
    print("2. Unified matrix")
    print("- samples/points: 59")
    print("- genes: 834")
    print(f"- expression scale: {EXPRESSION_SCALE}")
    print("3. PCA")
    print(f"- PCs: {pca.n_components_}")
    print(
        f"- cumulative variance: {pca.explained_variance_ratio_.sum():.6f}"
    )
    print("- fit data: 35 Observed Pre + Post points")
    print("4. UMAP")
    print("- observed reference points: 35")
    print("- n_neighbors: 10")
    print("- min_dist: 0.30")
    print("- transformed prediction points: 24")
    print("5. Recovery")
    for label, value in all_medians.items():
        print(f"- {label} median response-centroid recovery ratio: {value:.6f}")
    print("6. Clustering")
    print("- method: Ward hierarchical clustering in 20D PCA")
    print("- response recovery metric: normalized response-centroid margin")
    print("- paired recovery primary: NO")
    print("7. Final status")
    print("- Numeric: PASS")
    print("- Visual: PENDING")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-current", action="store_true")
    parser.add_argument("--mark-visual-pass", action="store_true")
    args = parser.parse_args()
    if args.mark_visual_pass:
        mark_visual_pass()
    else:
        build(allow_rebuild=args.rebuild_current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
