#!/usr/bin/env python3
"""GSE120575 post-hoc cell-expanded expression-proxy UMAP V3.

The script deterministically expands each frozen sample-level target over the
same real Pre cells by per-gene nonnegative simplex projection. These are not
native single-cell predictions. Observed Post is never read by the expansion
operator; it is used only to fit the observed reference and evaluate recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR", "/tmp/numba-gse120575-cell-expanded-umap-v3"
)
os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/matplotlib-gse120575-cell-expanded-umap-v3"
)

import anndata as ad
import joblib
import matplotlib
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("GSE120575_V3_OUT", str(ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy"))).resolve()
OUT = BASE / "cell_expanded_umap_figure_v3"

PANEL = Path(os.environ.get("GSE120575_PROXY_OUT", str(ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"))).resolve() / "frozen_834_gene_panel.txt"
RAW_CELLS = (
    ROOT
    / "data/wot_scgen_cellrank_native_pre_post_reports"
    / "wot_gse120575_pre_to_post_native"
    / "wot_expression_cost_genes.h5ad"
)
ANNOTATIONS = (
    ROOT
    / "outputs/GSE120575_full_online_celltype"
    / "annotation/cell_level_annotations.csv"
)
V1 = BASE / "sample_level_umap_clustering_v1"
FROZEN_TARGETS = V1 / "unified_sample_expression_834genes.h5ad"
V1_AUDIT = V1 / "analysis_audit.json"
AGENTVC_AUDIT = Path(os.environ.get("GSE120575_PROXY_OUT", str(ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"))).resolve() / "reconstruction_audit.json"
CELLRANK_V3_AUDIT = BASE / "V3_AUDIT.json"

EXPECTED_HASHES = {
    "gene_panel": "1a9e02f8bb52cdea852d6fe30992339ad8bc2fa0ab807b81848dce574a7bce41",
    "raw_cells": "b19c7ff0aaa2c7c49de54bb918ce95b0d8c8b4fd5459644515d42b3ad44b791d",
    "cell_annotations": "e63d6b185bacfbbbf7689b9b8043cf4ccda6f52f935e1d67efaa545bfd2b1614",
    "frozen_targets": "206dee47c582bfe1498c87adab530f3bbe942063cb1a38dcdd6310c7d998918a",
    "v1_audit": "e67f4568ad726c38b9010f53fefbbbc9237985a91624e1a955adca939a22fe21",
    "agentvc_audit": "a966de88ce0f3459ffc36eb70eac53ab0f06c06a1f7c590f495008eaac8cc3f4",
    "cellrank_v3_audit": "c41bb0bbc21ff19571ecc5533a4453330d8da3a88c5b54c78717465aa9e61568",
}

SAMPLES = (
    "Pre_P24",
    "Pre_P29",
    "Pre_P35",
    "Pre_P2",
    "Pre_P3",
    "Pre_P27",
)
METHODS = (
    "scGen",
    "WOT_weighted",
    "CellRank_terminal_proxy",
    "AgentVC_proxy",
)
METHOD_LABELS = {
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "CellRank_terminal_proxy": "CellRank\u2020",
    "AgentVC_proxy": "AgentVC",
}
METHOD_COLORS = {
    "scGen": "#E69F00",
    "WOT_weighted": "#009E73",
    "CellRank_terminal_proxy": "#56B4E9",
    "AgentVC_proxy": "#CC79A7",
}
RESPONSES = ("Responder", "Non-responder")
RESPONSE_MARKERS = {"Responder": "o", "Non-responder": "^"}
BROAD_TYPES = (
    "B cell",
    "Plasma cell",
    "Monocyte/Macrophage",
    "Dendritic cell",
    "T cell",
    "NK cell",
)
FINE_POPULATIONS = (
    "B cells",
    "Cytotoxicity lymphocytes",
    "Dendritic cells",
    "Exhausted CD8+ T cells",
    "Exhausted/heat-shock CD8+ T cells",
    "Lymphocytes",
    "Lymphocytes exhausted/cell cycle",
    "Memory T cells",
    "Monocytes/Macrophages",
    "Plasma cells",
    "Regulatory T cells",
)
POPULATION_COLORS = {
    "B cells": "#0000DD",
    "Cytotoxicity lymphocytes": "#FFA52F",
    "Dendritic cells": "#018700",
    "Exhausted CD8+ T cells": "#D60000",
    "Exhausted/heat-shock CD8+ T cells": "#8C3BFF",
    "Lymphocytes": "#573B00",
    "Lymphocytes exhausted/cell cycle": "#FF7ED1",
    "Memory T cells": "#00ACC6",
    "Monocytes/Macrophages": "#91BF00",
    "Plasma cells": "#6B004F",
    "Regulatory T cells": "#005659",
}

PCA_COMPONENTS = 50
RANDOM_SEED = 120575
UMAP_PARAMETERS = {
    "n_neighbors": 25,
    "min_dist": 0.12,
    "spread": 1.0,
    "metric": "euclidean",
    "random_state": RANDOM_SEED,
    "transform_seed": RANDOM_SEED,
}
MAX_MEAN_ERROR = 1e-10

FIGURE_STEM = "GSE120575_cell_expanded_expression_proxy_UMAP_v3"
PREVIEW = OUT / f"{FIGURE_STEM}_preview.png"
PNG_600 = OUT / f"{FIGURE_STEM}_600dpi.png"
PDF = OUT / f"{FIGURE_STEM}_vector.pdf"
SVG = OUT / f"{FIGURE_STEM}.svg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def patient_id(sample_id: str) -> str:
    pieces = sample_id.split("_")
    return pieces[1] if len(pieces) > 1 else sample_id


def validate_hashes() -> dict[str, str]:
    paths = {
        "gene_panel": PANEL,
        "raw_cells": RAW_CELLS,
        "cell_annotations": ANNOTATIONS,
        "frozen_targets": FROZEN_TARGETS,
        "v1_audit": V1_AUDIT,
        "agentvc_audit": AGENTVC_AUDIT,
        "cellrank_v3_audit": CELLRANK_V3_AUDIT,
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
        variable = {"frozen_targets", "v1_audit", "agentvc_audit", "cellrank_v3_audit"}
        if hashes[name] != EXPECTED_HASHES[name] and not (os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") == "1" and name in variable):
            raise RuntimeError(
                f"Frozen source hash changed for {name}: {hashes[name]}"
            )
    if json.loads(V1_AUDIT.read_text())["status"] != "PASS_SAMPLE_LEVEL_UMAP":
        raise RuntimeError("Frozen sample-level V1 is not finalized")
    if (
        json.loads(AGENTVC_AUDIT.read_text())["status"]
        != "PASS_AGENTVC_CHECKPOINT5_PROXY_RECONSTRUCTION"
    ):
        raise RuntimeError("Frozen AgentVC reconstruction did not pass")
    if (
        json.loads(CELLRANK_V3_AUDIT.read_text())["status"]
        != "PASS_GSE120575_EXPRESSION_V3_CELLRANK_PROXY"
    ):
        raise RuntimeError("Frozen CellRank V3 did not pass")
    return hashes


def load_sources() -> tuple[
    list[str],
    np.ndarray,
    pd.DataFrame,
    ad.AnnData,
    dict[str, object],
]:
    genes = [
        line.strip()
        for line in PANEL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(genes) != 834 or len(set(genes)) != 834:
        raise RuntimeError("Frozen panel is not 834 unique genes")

    raw = ad.read_h5ad(RAW_CELLS)
    if raw.shape != (16291, 5000):
        raise RuntimeError(f"Unexpected raw cell matrix: {raw.shape}")
    if raw.uns.get("matrix_units") != "log1p(TPM)":
        raise RuntimeError("Raw cells are not declared log1p(TPM)")
    gene_indexes = raw.var_names.get_indexer(genes)
    if np.any(gene_indexes < 0):
        missing = np.asarray(genes)[gene_indexes < 0].tolist()
        raise RuntimeError(f"Missing panel genes: {missing[:10]}")
    selected = raw[:, gene_indexes].X
    expression = (
        selected.toarray().astype(np.float64)
        if sparse.issparse(selected)
        else np.asarray(selected, dtype=np.float64)
    )
    if expression.shape != (16291, 834):
        raise RuntimeError("Selected cell expression has wrong shape")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Observed log1p(TPM) is invalid")

    annotations = pd.read_csv(ANNOTATIONS)
    if len(annotations) != 16291 or annotations["cell_id"].duplicated().any():
        raise RuntimeError("Cell annotation table is not one row per raw cell")
    annotation_by_id = annotations.set_index("cell_id")
    metadata = raw.obs.copy()
    metadata.index = metadata["unique_cell_id"].astype(str)
    if set(metadata.index) != set(annotation_by_id.index.astype(str)):
        raise RuntimeError("Raw cell IDs and annotations are not one-to-one")
    join_columns = [
        "patient_sample",
        "response",
        "therapy",
        "broad_cell_type",
        "published_population",
    ]
    metadata = metadata.join(annotation_by_id[join_columns], how="left")
    if metadata[join_columns].isna().any().any():
        raise RuntimeError("Cell annotation join produced missing values")
    if set(metadata["published_population"].astype(str)) != set(FINE_POPULATIONS):
        raise RuntimeError("Published population vocabulary changed")
    if set(metadata["broad_cell_type"].astype(str)) != set(BROAD_TYPES):
        raise RuntimeError("Broad cell-type vocabulary changed")

    targets = ad.read_h5ad(FROZEN_TARGETS)
    if targets.shape != (59, 834):
        raise RuntimeError("Frozen target H5AD shape changed")
    if targets.var_names.tolist() != genes:
        raise RuntimeError("Frozen target gene order changed")
    target_values = np.asarray(targets.X, dtype=np.float64)
    if not np.isfinite(target_values).all() or np.any(target_values < 0):
        raise RuntimeError("Frozen sample targets are invalid")
    target_checks: dict[str, object] = {}
    for method in METHODS:
        selected_method = targets.obs["method"].astype(str).eq(method)
        sample_set = set(
            targets.obs.loc[selected_method, "sample_id"].astype(str)
        )
        if int(selected_method.sum()) != 6 or sample_set != set(SAMPLES):
            raise RuntimeError(f"Frozen targets changed for {method}")
        target_checks[method] = {
            "rows": 6,
            "samples": list(SAMPLES),
        }

    checks = {
        "raw_shape": [16291, 5000],
        "annotation_rows": 16291,
        "annotation_joined_rows": 16291,
        "panel_gene_count": 834,
        "raw_expression_scale": "log1p(TPM)",
        "published_population_count": 11,
        "broad_cell_type_count": 6,
        "frozen_target_shape": [59, 834],
        "frozen_target_methods": target_checks,
    }
    return genes, expression, metadata, targets, checks


def simplex_projection(values: np.ndarray, target_sum: float) -> np.ndarray:
    """Euclidean projection of values onto {x >= 0, sum(x) = target_sum}."""
    if target_sum <= 0:
        return np.zeros_like(values)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indexes = np.arange(1, len(values) + 1)
    support = np.flatnonzero(ordered * indexes > cumulative - target_sum)
    if len(support) == 0:
        return np.full_like(values, target_sum / len(values))
    last = int(support[-1])
    theta = (cumulative[last] - target_sum) / (last + 1)
    return np.maximum(values - theta, 0.0)


def build_cell_expansion(
    genes: list[str],
    expression: np.ndarray,
    metadata: pd.DataFrame,
    targets: ad.AnnData,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    source_mask = metadata["patient_sample"].astype(str).isin(SAMPLES).to_numpy()
    post_mask = (
        metadata["therapy"].astype(str).eq("anti-PD1")
        & metadata["patient_sample"].astype(str).str.startswith("Post_")
    ).to_numpy()
    source_expression = expression[source_mask]
    post_expression = expression[post_mask]
    source_metadata = metadata.loc[source_mask].copy()
    post_metadata = metadata.loc[post_mask].copy()
    if len(source_metadata) != 1664 or len(post_metadata) != 7858:
        raise RuntimeError(
            f"Expected 1664 selected Pre and 7858 Post cells, got "
            f"{len(source_metadata)} and {len(post_metadata)}"
        )
    if source_metadata["patient_sample"].nunique() != 6:
        raise RuntimeError("Selected Pre cells do not cover six source samples")
    if post_metadata["patient_sample"].nunique() != 23:
        raise RuntimeError("Observed Post does not cover 23 biopsies")

    target_meta = targets.obs.copy()
    target_values = np.asarray(targets.X, dtype=np.float64)
    prediction_blocks: list[np.ndarray] = []
    prediction_metadata_blocks: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    mapping_rows: list[pd.DataFrame] = []
    maximum_mean_error = 0.0
    all_rank_preserved = True

    for method in METHODS:
        method_blocks: list[np.ndarray] = []
        method_metadata: list[pd.DataFrame] = []
        for sample in SAMPLES:
            local = source_metadata["patient_sample"].astype(str).eq(sample).to_numpy()
            source_log = source_expression[local]
            source_tpm = np.expm1(source_log)
            target_mask = (
                target_meta["method"].astype(str).eq(method)
                & target_meta["sample_id"].astype(str).eq(sample)
            ).to_numpy()
            if int(target_mask.sum()) != 1:
                raise RuntimeError(f"Missing target: {method}/{sample}")
            target_tpm = np.expm1(target_values[target_mask][0])
            expanded_tpm = np.empty_like(source_tpm)
            n_cells = len(source_tpm)
            for gene_index, gene in enumerate(genes):
                source_gene = source_tpm[:, gene_index]
                expanded_gene = simplex_projection(
                    source_gene,
                    float(n_cells * target_tpm[gene_index]),
                )
                expanded_tpm[:, gene_index] = expanded_gene
                actual_mean = float(expanded_gene.mean())
                target_mean = float(target_tpm[gene_index])
                absolute_error = abs(actual_mean - target_mean)
                maximum_mean_error = max(maximum_mean_error, absolute_error)
                order = np.argsort(source_gene, kind="stable")
                rank_preserved = bool(
                    np.all(np.diff(expanded_gene[order]) >= -1e-12)
                )
                all_rank_preserved &= rank_preserved
                audit_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "sample_id": sample,
                        "gene": gene,
                        "source_cell_count": n_cells,
                        "target_mean_tpm": target_mean,
                        "expanded_mean_tpm": actual_mean,
                        "absolute_mean_error": absolute_error,
                        "rank_order_non_decreasing": rank_preserved,
                        "expansion_operator": (
                            "per-gene nonnegative Euclidean simplex projection"
                        ),
                    }
                )
            if not np.isfinite(expanded_tpm).all() or np.any(expanded_tpm < 0):
                raise RuntimeError(f"Invalid expansion: {method}/{sample}")
            expanded_log = np.log1p(expanded_tpm)
            method_blocks.append(expanded_log)

            local_meta = source_metadata.loc[local].copy()
            local_meta["method"] = method
            local_meta["method_label"] = METHOD_LABELS[method]
            local_meta["state"] = "Cell-expanded Predicted Post"
            local_meta["source_cell_id"] = local_meta.index.astype(str)
            local_meta["sample_id"] = sample
            local_meta["patient_id"] = patient_id(sample)
            local_meta["is_observed"] = False
            local_meta["is_prediction"] = True
            local_meta["annotation_semantics"] = (
                "source-inherited published population; not model-predicted"
            )
            local_meta["expression_semantics"] = (
                "cell-expanded frozen sample-level expression proxy"
            )
            local_meta["point_id"] = [
                f"{method}::{cell_id}" for cell_id in local_meta.index
            ]
            method_metadata.append(local_meta)
            mapping_rows.append(
                local_meta[
                    [
                        "point_id",
                        "source_cell_id",
                        "sample_id",
                        "patient_id",
                        "response",
                        "method",
                        "method_label",
                        "broad_cell_type",
                        "published_population",
                        "annotation_semantics",
                        "expression_semantics",
                    ]
                ].copy()
            )
        method_block = np.vstack(method_blocks)
        method_meta = pd.concat(method_metadata, axis=0)
        if method_block.shape != (1664, 834) or len(method_meta) != 1664:
            raise RuntimeError(f"Wrong expanded method shape: {method}")
        if method_meta["source_cell_id"].duplicated().any():
            raise RuntimeError(f"Duplicated source cell IDs: {method}")
        prediction_blocks.append(method_block)
        prediction_metadata_blocks.append(method_meta)

    if maximum_mean_error > MAX_MEAN_ERROR:
        raise RuntimeError(
            f"Expanded mean error exceeds gate: {maximum_mean_error}"
        )
    if not all_rank_preserved:
        raise RuntimeError("Simplex expansion introduced an expression inversion")

    prediction_expression = np.vstack(prediction_blocks)
    prediction_metadata = pd.concat(prediction_metadata_blocks, axis=0)
    if prediction_expression.shape != (6656, 834):
        raise RuntimeError("Expanded prediction matrix is not 6656x834")
    method_source_sets = {
        method: set(
            prediction_metadata.loc[
                prediction_metadata["method"].eq(method),
                "source_cell_id",
            ]
        )
        for method in METHODS
    }
    if any(
        method_source_sets[method] != method_source_sets[METHODS[0]]
        for method in METHODS[1:]
    ):
        raise RuntimeError("Methods do not use identical source cells")

    mean_audit = pd.DataFrame(audit_rows)
    mapping = pd.concat(mapping_rows, ignore_index=True)
    checks = {
        "selected_pre_cells": 1664,
        "observed_post_cells": 7858,
        "observed_post_biopsies": 23,
        "expanded_cells_per_method": {
            method: int(
                prediction_metadata["method"].astype(str).eq(method).sum()
            )
            for method in METHODS
        },
        "expanded_prediction_cells": 6656,
        "methods_use_identical_source_cell_ids": True,
        "source_cell_ids_unique_within_method": True,
        "maximum_sample_gene_mean_absolute_error_tpm": maximum_mean_error,
        "mean_error_gate": MAX_MEAN_ERROR,
        "all_gene_rank_orders_non_decreasing": all_rank_preserved,
        "random_noise_added": False,
        "cell_rows_replicated": False,
        "coordinate_jitter": False,
        "observed_post_used_in_expansion": False,
    }
    return (
        source_expression,
        source_metadata,
        post_expression,
        post_metadata,
        prediction_expression,
        prediction_metadata,
        mean_audit,
        mapping,
        checks,
    )


def observed_metadata(
    frame: pd.DataFrame,
    method: str,
    state: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["method"] = method
    result["method_label"] = state
    result["state"] = state
    result["source_cell_id"] = result.index.astype(str)
    result["sample_id"] = result["patient_sample"].astype(str)
    result["patient_id"] = result["sample_id"].map(patient_id)
    result["is_observed"] = True
    result["is_prediction"] = False
    result["annotation_semantics"] = "published observed-cell annotation"
    result["expression_semantics"] = "observed log1p(TPM)"
    result["point_id"] = [
        f"{method}::{cell_id}" for cell_id in result.index
    ]
    return result


def fit_reference_models(
    observed_expression: np.ndarray,
    prediction_expression: np.ndarray,
) -> tuple[
    StandardScaler,
    PCA,
    umap.UMAP,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    scaler = StandardScaler()
    observed_standardized = scaler.fit_transform(observed_expression)
    prediction_standardized = scaler.transform(prediction_expression)
    zero_variance = np.asarray(scaler.var_) == 0.0

    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )
    # Fit strictly on Observed, then transform both sets through the frozen
    # model so every persisted coordinate is exactly reproducible from it.
    pca.fit(observed_standardized)
    observed_pca = pca.transform(observed_standardized)
    prediction_pca = pca.transform(prediction_standardized)

    model = umap.UMAP(
        n_neighbors=UMAP_PARAMETERS["n_neighbors"],
        min_dist=UMAP_PARAMETERS["min_dist"],
        spread=UMAP_PARAMETERS["spread"],
        metric=UMAP_PARAMETERS["metric"],
        random_state=UMAP_PARAMETERS["random_state"],
        transform_seed=UMAP_PARAMETERS["transform_seed"],
        n_jobs=1,
        low_memory=True,
    )
    observed_umap = model.fit_transform(observed_pca)
    prediction_umap = model.transform(prediction_pca)
    if (
        observed_pca.shape != (9522, 50)
        or prediction_pca.shape != (6656, 50)
        or observed_umap.shape != (9522, 2)
        or prediction_umap.shape != (6656, 2)
    ):
        raise RuntimeError("Reference model output shapes are invalid")
    if not all(
        np.isfinite(values).all()
        for values in (
            observed_pca,
            prediction_pca,
            observed_umap,
            prediction_umap,
        )
    ):
        raise RuntimeError("Reference coordinates contain non-finite values")
    checks = {
        "scaler_fit_points": 9522,
        "pca_fit_points": 9522,
        "umap_fit_points": 9522,
        "prediction_transform_points": 6656,
        "fit_scope": "1664 selected Observed Pre + 7858 Observed Post only",
        "prediction_operation": "transform only",
        "pca_components": 50,
        "pca_cumulative_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "zero_variance_gene_count": int(zero_variance.sum()),
        "umap_parameters": UMAP_PARAMETERS,
        "formal_metric_space": "50D Observed-reference PCA",
        "umap_used_for_formal_metrics": False,
    }
    return (
        scaler,
        pca,
        model,
        np.vstack([observed_pca, prediction_pca]),
        np.vstack([observed_umap, prediction_umap]),
        checks,
    )


def local_recovery_metrics(
    metadata: pd.DataFrame,
    pca_coordinates: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    pre = metadata["method"].astype(str).eq("Observed_Pre").to_numpy()
    post = metadata["method"].astype(str).eq("Observed_Post").to_numpy()
    for method in METHODS:
        predicted = metadata["method"].astype(str).eq(method).to_numpy()
        for response in RESPONSES:
            for cell_type in BROAD_TYPES:
                group = (
                    metadata["response"].astype(str).eq(response)
                    & metadata["broad_cell_type"].astype(str).eq(cell_type)
                ).to_numpy()
                pre_group = pre & group
                post_group = post & group
                prediction_group = predicted & group
                if min(
                    int(pre_group.sum()),
                    int(post_group.sum()),
                    int(prediction_group.sum()),
                ) == 0:
                    raise RuntimeError(
                        f"Missing local recovery group: "
                        f"{method}/{response}/{cell_type}"
                    )
                post_centroid = pca_coordinates[post_group].mean(axis=0)
                source_distance = float(
                    np.linalg.norm(
                        pca_coordinates[pre_group].mean(axis=0) - post_centroid
                    )
                )
                prediction_distance = float(
                    np.linalg.norm(
                        pca_coordinates[prediction_group].mean(axis=0)
                        - post_centroid
                    )
                )
                if source_distance <= 0:
                    raise RuntimeError("Zero persistence denominator")
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "response": response,
                        "broad_cell_type": cell_type,
                        "n_source_pre_cells": int(pre_group.sum()),
                        "n_observed_post_cells": int(post_group.sum()),
                        "n_predicted_proxy_cells": int(
                            prediction_group.sum()
                        ),
                        "prediction_to_matched_response_post_centroid_distance": (
                            prediction_distance
                        ),
                        "source_pre_to_matched_response_post_centroid_distance": (
                            source_distance
                        ),
                        "local_recovery_ratio": (
                            prediction_distance / source_distance
                        ),
                        "lower_is_better": True,
                        "formal_space": "50D Observed-reference PCA",
                        "biological_replicate_interpretation": (
                            "descriptive response x broad-cell-type group; "
                            "pseudo-cells are not independent biological replicates"
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    if len(metrics) != 48 or not np.isfinite(
        metrics["local_recovery_ratio"]
    ).all():
        raise RuntimeError("Local recovery metrics are not 48 finite rows")
    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        values = metrics.loc[
            metrics["method"].eq(method), "local_recovery_ratio"
        ].to_numpy(float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        summary_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_response_celltype_groups": len(values),
                "median_local_recovery_ratio": float(median),
                "q1_local_recovery_ratio": float(q1),
                "q3_local_recovery_ratio": float(q3),
                "mean_local_recovery_ratio": float(values.mean()),
                "min_local_recovery_ratio": float(values.min()),
                "max_local_recovery_ratio": float(values.max()),
                "lower_is_better": True,
                "overall_winner_score": False,
            }
        )
    return metrics, pd.DataFrame(summary_rows)


def normalized_limits(
    coordinates: np.ndarray,
) -> tuple[float, float, float, float]:
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_pad = (x_max - x_min) * 0.045
    y_pad = (y_max - y_min) * 0.045
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def label_positions(
    anchors: np.ndarray,
    limits: tuple[float, float, float, float],
) -> np.ndarray:
    """Deterministic soft repulsion in normalized axes coordinates."""
    x_min, x_max, y_min, y_max = limits
    scale = np.array([x_max - x_min, y_max - y_min])
    origin = np.array([x_min, y_min])
    anchor_norm = (anchors - origin) / scale
    positions = anchor_norm.copy()
    for _ in range(500):
        movement = 0.018 * (anchor_norm - positions)
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                delta = positions[left] - positions[right]
                weighted = np.array([delta[0] / 0.065, delta[1] / 0.08])
                distance = float(np.linalg.norm(weighted))
                if distance < 1.0:
                    if distance < 1e-8:
                        direction = np.array(
                            [1.0 if (left + right) % 2 == 0 else -1.0, 1.0]
                        )
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = weighted / distance
                    push = (1.0 - distance) * 0.0035
                    movement[left] += direction * push
                    movement[right] -= direction * push
        positions += movement
        positions = np.clip(positions, 0.035, 0.965)
    return positions * scale + origin


def add_umap_axes(ax: plt.Axes) -> None:
    arrow = {
        "arrowstyle": "-|>",
        "color": "#4F5558",
        "lw": 0.75,
        "mutation_scale": 7,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate(
        "",
        xy=(0.18, 0.065),
        xytext=(0.06, 0.065),
        xycoords="axes fraction",
        arrowprops=arrow,
    )
    ax.annotate(
        "",
        xy=(0.06, 0.185),
        xytext=(0.06, 0.065),
        xycoords="axes fraction",
        arrowprops=arrow,
    )
    ax.text(
        0.12,
        0.012,
        "UMAP1",
        transform=ax.transAxes,
        ha="center",
        fontsize=5.8,
        color="#4F5558",
    )
    ax.text(
        0.01,
        0.13,
        "UMAP2",
        transform=ax.transAxes,
        va="center",
        rotation=90,
        fontsize=5.8,
        color="#4F5558",
    )


def plot_umap_panel(
    ax: plt.Axes,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    limits: tuple[float, float, float, float],
    letter: str,
    title: str,
    method: str | None,
) -> None:
    post = metadata["method"].astype(str).eq("Observed_Post").to_numpy()
    if method is None:
        for population in FINE_POPULATIONS:
            selected = post & metadata["published_population"].astype(str).eq(
                population
            ).to_numpy()
            ax.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                s=4.2,
                color=POPULATION_COLORS[population],
                alpha=0.72,
                linewidth=0,
                rasterized=True,
                zorder=2,
            )
        anchors = np.vstack(
            [
                np.median(
                    coordinates[
                        post
                        & metadata["published_population"]
                        .astype(str)
                        .eq(population)
                        .to_numpy()
                    ],
                    axis=0,
                )
                for population in FINE_POPULATIONS
            ]
        )
        positions = label_positions(anchors, limits)
        x_span = limits[1] - limits[0]
        y_span = limits[3] - limits[2]
        for index, (anchor, position) in enumerate(
            zip(anchors, positions),
            start=1,
        ):
            displacement = np.linalg.norm(
                (position - anchor) / np.array([x_span, y_span])
            )
            if displacement > 0.012:
                ax.plot(
                    [anchor[0], position[0]],
                    [anchor[1], position[1]],
                    color="#555555",
                    lw=0.45,
                    alpha=0.65,
                    zorder=4,
                )
            text = ax.text(
                position[0],
                position[1],
                str(index),
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="bold",
                color="#111111",
                zorder=5,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=2.2, foreground="white")]
            )
        count_text = "7,858 Observed Post cells"
    else:
        ax.scatter(
            coordinates[post, 0],
            coordinates[post, 1],
            s=2.2,
            color="#BFC3C5",
            alpha=0.13,
            linewidth=0,
            rasterized=True,
            zorder=1,
        )
        for population in FINE_POPULATIONS:
            selected = (
                metadata["method"].astype(str).eq(method)
                & metadata["published_population"].astype(str).eq(population)
            ).to_numpy()
            ax.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                s=5.2,
                color=POPULATION_COLORS[population],
                alpha=0.78,
                linewidth=0,
                rasterized=True,
                zorder=2,
            )
        count_text = "7,858 reference + 1,664 proxy cells"

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(
        f"{letter}. {title}",
        loc="left",
        fontsize=11.0,
        fontweight="bold",
        pad=4,
    )
    ax.text(
        0.98,
        0.97,
        count_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color="#686E72",
    )
    add_umap_axes(ax)


def plot_metric_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
) -> None:
    y_positions = {
        method: len(METHODS) - 1 - index
        for index, method in enumerate(METHODS)
    }
    values_all = metrics["local_recovery_ratio"].to_numpy(float)
    x_min = min(0.45, float(values_all.min()) - 0.12)
    x_max = float(values_all.max()) + 0.25
    for method in METHODS:
        selected = metrics[metrics["method"].eq(method)].copy()
        selected["_order"] = pd.Categorical(
            selected["broad_cell_type"],
            categories=BROAD_TYPES,
            ordered=True,
        )
        selected["_response_order"] = pd.Categorical(
            selected["response"],
            categories=RESPONSES,
            ordered=True,
        )
        selected = selected.sort_values(
            ["_response_order", "_order"]
        ).reset_index(drop=True)
        values = selected["local_recovery_ratio"].to_numpy(float)
        base_y = y_positions[method]
        offsets = np.linspace(-0.17, 0.17, len(values))
        for response in RESPONSES:
            mask = selected["response"].astype(str).eq(response).to_numpy()
            ax.scatter(
                values[mask],
                base_y + offsets[mask],
                s=28,
                marker=RESPONSE_MARKERS[response],
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=3,
            )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.hlines(
            base_y,
            q1,
            q3,
            color=METHOD_COLORS[method],
            lw=6.0,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            median,
            base_y,
            s=50,
            marker="D",
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            median,
            base_y + 0.30,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#3F4447",
        )
    ax.axvline(1.0, color="#6F767A", lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.60, 3.62)
    ax.set_yticks(
        [y_positions[method] for method in METHODS],
        [METHOD_LABELS[method] for method in METHODS],
    )
    ax.set_xlabel(
        "Prediction / persistence centroid-distance ratio",
        fontsize=7.5,
    )
    ax.set_title(
        "F. Cell-type local recovery in 50D PCA",
        loc="left",
        fontsize=11.0,
        fontweight="bold",
        pad=10,
    )
    ax.text(
        0.99,
        0.02,
        "Lower is better \u00b7 12 response \u00d7 broad-cell-type groups",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555B5F",
    )
    ax.grid(axis="x", color="#E4E7E9", lw=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#777D80")
    ax.tick_params(axis="x", labelsize=6.8)
    ax.tick_params(axis="y", labelsize=7.5, length=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker=RESPONSE_MARKERS[response],
            linestyle="none",
            markerfacecolor="#73787B",
            markeredgecolor="white",
            markersize=6,
            label=response,
        )
        for response in RESPONSES
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=6.7,
        loc="upper right",
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.8,
    )


def draw_figure(
    metadata: pd.DataFrame,
    umap_coordinates: np.ndarray,
    metrics: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    limits = normalized_limits(umap_coordinates)
    fig = plt.figure(figsize=(15.4, 9.55), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.985,
        bottom=0.225,
        top=0.855,
        wspace=0.15,
        hspace=0.22,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    specifications = (
        ("A", "Observed Post reference", None),
        ("B", "scGen", "scGen"),
        ("C", "WOT* (target-derived)", "WOT_weighted"),
        ("D", "CellRank\u2020 proxy", "CellRank_terminal_proxy"),
        ("E", "AgentVC proxy", "AgentVC_proxy"),
    )
    for ax, (letter, title, method) in zip(axes, specifications):
        plot_umap_panel(
            ax,
            metadata,
            umap_coordinates,
            limits,
            letter,
            title,
            method,
        )
    metric_ax = fig.add_subplot(grid[1, 2])
    plot_metric_panel(metric_ax, metrics)

    population_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=POPULATION_COLORS[population],
            markeredgecolor="none",
            markersize=6.0,
            label=f"{index}  {population}",
        )
        for index, population in enumerate(FINE_POPULATIONS, start=1)
    ]
    fig.legend(
        handles=population_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=4,
        frameon=False,
        fontsize=7.1,
        columnspacing=1.25,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Cell-expanded expression-state recovery across immune populations",
        y=0.975,
        fontsize=18.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.936,
        (
            "GSE120575 \u00b7 9,522 observed reference cells \u00b7 "
            "1,664 deterministic proxy cells per method \u00b7 834 genes"
        ),
        ha="center",
        fontsize=9.6,
        color="#4D5357",
    )
    fig.text(
        0.5,
        0.030,
        (
            "Post-hoc cell-expanded sample-level expression proxies; not native "
            "single-cell predictions. Population labels are inherited from each "
            "real source Pre cell.\n"
            "No cell replication, random noise or coordinate jitter was used. "
            "UMAP is visualization only; Panel F uses 50D Observed-reference PCA. "
            "WOT* is target-derived."
        ),
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#3F4447",
    )
    fig.savefig(PREVIEW, dpi=150, facecolor="white")
    fig.savefig(PNG_600, dpi=600, facecolor="white")
    fig.savefig(PDF, facecolor="white")
    fig.savefig(SVG, facecolor="white")
    plt.close(fig)


def save_unified_h5ad(
    genes: list[str],
    expression: np.ndarray,
    metadata: pd.DataFrame,
    pca_coordinates: np.ndarray,
    umap_coordinates: np.ndarray,
    pca: PCA,
) -> None:
    obs_columns = [
        "point_id",
        "source_cell_id",
        "sample_id",
        "patient_id",
        "response",
        "therapy",
        "method",
        "method_label",
        "state",
        "is_observed",
        "is_prediction",
        "broad_cell_type",
        "published_population",
        "annotation_semantics",
        "expression_semantics",
    ]
    obs = metadata[obs_columns].copy()
    obs.index = obs["point_id"].astype(str)
    obs.index.name = None
    adata = ad.AnnData(
        # Keep float64 here: the frozen contract requires the persisted
        # pseudo-cell TPM means (after expm1) to match their sample targets
        # within 1e-10. Casting log1p(TPM) to float32 breaks that gate.
        X=expression.astype(np.float64),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name=None)),
    )
    adata.obsm["X_pca"] = pca_coordinates.astype(np.float64)
    adata.obsm["X_umap"] = umap_coordinates.astype(np.float32)
    adata.uns["expression_scale"] = "log1p(TPM)"
    adata.uns["analysis_semantics"] = (
        "post-hoc cell-expanded sample-level expression proxy UMAP"
    )
    adata.uns["native_single_cell_predictions"] = False
    adata.uns["population_annotation_semantics"] = (
        "source-inherited for proxy cells"
    )
    adata.uns["reference_fit_points"] = 9522
    adata.uns["prediction_transform_points"] = 6656
    adata.uns["formal_metric_space"] = "50D Observed-reference PCA"
    adata.uns["pca_explained_variance_ratio"] = (
        pca.explained_variance_ratio_.astype(float)
    )
    adata.write_h5ad(
        OUT / "unified_cell_expanded_expression_834genes.h5ad",
        compression="gzip",
    )


def write_coordinate_files(
    metadata: pd.DataFrame,
    pca_coordinates: np.ndarray,
    umap_coordinates: np.ndarray,
) -> None:
    identity = metadata[
        [
            "point_id",
            "source_cell_id",
            "sample_id",
            "patient_id",
            "response",
            "method",
            "method_label",
            "state",
            "broad_cell_type",
            "published_population",
            "is_observed",
            "is_prediction",
        ]
    ].reset_index(drop=True)
    umap_frame = identity.copy()
    umap_frame["UMAP1"] = umap_coordinates[:, 0]
    umap_frame["UMAP2"] = umap_coordinates[:, 1]
    umap_frame.to_csv(
        OUT / "cell_expanded_umap_coordinates.csv",
        index=False,
        float_format="%.17g",
    )
    pca_columns = pd.DataFrame(
        pca_coordinates,
        columns=[
            f"PC{component + 1}" for component in range(PCA_COMPONENTS)
        ],
    )
    pca_frame = pd.concat([identity, pca_columns], axis=1)
    pca_frame.to_csv(
        OUT / "cell_expanded_pca_coordinates.csv",
        index=False,
        float_format="%.17g",
    )


def write_readme(
    model_checks: dict[str, object],
    expansion_checks: dict[str, object],
    summary: pd.DataFrame,
) -> None:
    medians = summary.set_index("method_label")[
        "median_local_recovery_ratio"
    ]
    lines = [
        "# GSE120575 cell-expanded expression-proxy UMAP V3",
        "",
        "This is a post-hoc visualization and local-state analysis.",
        "The expanded rows are not native single-cell predictions.",
        "",
        "## Data",
        "",
        "- 1,664 real selected Pre cells.",
        "- 7,858 real anti-PD1 Post cells.",
        "- 1,664 deterministic expanded proxy cells per method.",
        "- 834 frozen common genes.",
        "",
        "## Expansion",
        "",
        "Each method/sample/gene uses nonnegative Euclidean simplex projection",
        "over the same source Pre cells. No random noise, row replication or",
        "coordinate jitter is used. Source-cell population labels are inherited.",
        f"Maximum target-mean error: "
        f"{expansion_checks['maximum_sample_gene_mean_absolute_error_tpm']:.3e}.",
        "",
        "## Reference models",
        "",
        "- Scaler, 50D PCA and UMAP fit only 9,522 observed cells.",
        "- 6,656 proxy cells use transform only.",
        f"- PCA cumulative variance: "
        f"{model_checks['pca_cumulative_explained_variance']:.6f}.",
        "",
        "## Median local recovery ratio",
        "",
    ]
    for method in ("scGen", "WOT*", "CellRank\u2020", "AgentVC"):
        lines.append(f"- {method}: {medians[method]:.6f}")
    lines.extend(
        [
            "",
            "Lower is better. These are 12 descriptive response x broad-cell-type",
            "groups, not independent biological replicates and not an overall score.",
            "",
            "WOT* is target-derived. CellRank\u2020 and AgentVC remain expression proxies.",
        ]
    )
    (OUT / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    caption = (
        "Cell-expanded expression-state recovery across GSE120575 immune "
        "populations. Panel A shows 7,858 observed anti-PD1 Post cells colored "
        "by 11 published populations. Panels B-E show the identical Observed "
        "Post reference in gray with 1,664 deterministic cell-expanded proxies "
        "from the named method. Each proxy inherits a real Pre cell's population "
        "label; these are not native single-cell predictions. Panel F shows "
        "response-by-broad-cell-type recovery ratios in 50D Observed-reference "
        "PCA; lower than one indicates recovery beyond persistence. No random "
        "noise, row replication or coordinate jitter was used. WOT* is "
        "target-derived."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")


def refresh_manifest() -> None:
    files = sorted(
        path
        for path in OUT.iterdir()
        if path.is_file() and path.name != "output_manifest.json"
    )
    write_json(
        OUT / "output_manifest.json",
        {
            "version": "GSE120575_CELL_EXPANDED_PROXY_UMAP_V3",
            "status": "post-hoc cell-expanded expression-proxy analysis",
            "hash_algorithm": "SHA256",
            "files": [
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
            "main_figure_source_data": [
                "cell_expanded_umap_coordinates.csv",
                "celltype_local_recovery_metrics.csv",
                "celltype_local_recovery_summary.csv",
            ],
            "expression_source_data": [
                "unified_cell_expanded_expression_834genes.h5ad",
                "cell_expansion_mapping.csv",
                "cell_expansion_mean_audit.csv",
            ],
        },
    )


def build(rebuild_current: bool = False) -> None:
    if OUT.exists() and any(OUT.iterdir()):
        if not rebuild_current:
            raise FileExistsError(f"Refusing to overwrite V3: {OUT}")
        audit_path = OUT / "analysis_audit.json"
        build_path = OUT / "build_state.json"
        audit_status = None
        build_status = None
        if audit_path.exists():
            audit_status = json.loads(audit_path.read_text()).get("status")
        if build_path.exists():
            build_status = json.loads(build_path.read_text()).get("status")
        if (
            audit_status != "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION"
            and build_status != "BUILDING"
        ):
            raise RuntimeError("Current V3 is not safe to rebuild")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_CELL_EXPANDED_PROXY_UMAP_V3",
            "status": "BUILDING",
        },
    )

    source_hashes = validate_hashes()
    genes, expression, metadata, targets, source_checks = load_sources()
    (
        source_expression,
        source_metadata,
        post_expression,
        post_metadata,
        prediction_expression,
        prediction_metadata,
        mean_audit,
        mapping,
        expansion_checks,
    ) = build_cell_expansion(genes, expression, metadata, targets)

    observed_pre_metadata = observed_metadata(
        source_metadata,
        "Observed_Pre",
        "Observed Pre",
    )
    observed_post_metadata = observed_metadata(
        post_metadata,
        "Observed_Post",
        "Observed Post",
    )
    unified_expression = np.vstack(
        [source_expression, post_expression, prediction_expression]
    )
    unified_metadata = pd.concat(
        [
            observed_pre_metadata,
            observed_post_metadata,
            prediction_metadata,
        ],
        axis=0,
    ).reset_index(drop=True)
    if unified_expression.shape != (16178, 834):
        raise RuntimeError("Unified expression is not 16178x834")
    if len(unified_metadata) != 16178:
        raise RuntimeError("Unified metadata is not 16178 rows")
    if unified_metadata["point_id"].duplicated().any():
        raise RuntimeError("Unified point IDs are not unique")
    if not np.isfinite(unified_expression).all() or np.any(
        unified_expression < 0
    ):
        raise RuntimeError("Unified expression is invalid")

    (
        scaler,
        pca,
        umap_model,
        pca_coordinates,
        umap_coordinates,
        model_checks,
    ) = fit_reference_models(
        unified_expression[:9522],
        unified_expression[9522:],
    )
    metrics, metric_summary = local_recovery_metrics(
        unified_metadata,
        pca_coordinates,
    )

    mean_audit.to_csv(
        OUT / "cell_expansion_mean_audit.csv",
        index=False,
        float_format="%.17g",
    )
    mapping.to_csv(OUT / "cell_expansion_mapping.csv", index=False)
    metrics.to_csv(
        OUT / "celltype_local_recovery_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    metric_summary.to_csv(
        OUT / "celltype_local_recovery_summary.csv",
        index=False,
        float_format="%.17g",
    )
    write_coordinate_files(
        unified_metadata,
        pca_coordinates,
        umap_coordinates,
    )
    save_unified_h5ad(
        genes,
        unified_expression,
        unified_metadata,
        pca_coordinates,
        umap_coordinates,
        pca,
    )
    joblib.dump(scaler, OUT / "observed_reference_scaler.pkl")
    joblib.dump(pca, OUT / "observed_reference_pca50.pkl")
    joblib.dump(umap_model, OUT / "observed_reference_umap.pkl")
    pd.DataFrame(
        {
            "PC": np.arange(1, PCA_COMPONENTS + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    ).to_csv(
        OUT / "pca_explained_variance.csv",
        index=False,
        float_format="%.17g",
    )
    write_json(
        OUT / "umap_parameters.json",
        {
            **UMAP_PARAMETERS,
            "fit_scope": model_checks["fit_scope"],
            "prediction_operation": "transform only",
            "package_version": umap.__version__,
        },
    )

    draw_figure(unified_metadata, umap_coordinates, metrics)
    write_readme(model_checks, expansion_checks, metric_summary)
    shutil.copy2(__file__, OUT / "run_GSE120575_cell_expanded_umap_v3.py")

    panel_counts = {
        "A_observed_post": 7858,
        **{
            f"{METHOD_LABELS[method]}_observed_background": 7858
            for method in METHODS
        },
        **{
            f"{METHOD_LABELS[method]}_proxy_cells": 1664
            for method in METHODS
        },
    }
    numeric_checks = {
        "unified_shape": [16178, 834],
        "unified_point_ids_unique": True,
        "all_expression_finite_nonnegative": True,
        "observed_reference_cells": 9522,
        "expanded_prediction_cells": 6656,
        "expanded_cells_per_method": 1664,
        "published_population_count": 11,
        "local_metric_rows": len(metrics),
        "local_groups_per_method": {
            method: int(metrics["method"].eq(method).sum())
            for method in METHODS
        },
        "panel_point_counts": panel_counts,
        "formal_metrics_use_50d_pca": True,
        "umap_used_for_formal_metrics": False,
        "significance_tests": 0,
        "overall_winner_score": False,
        "native_single_cell_prediction_claim": False,
    }
    write_json(
        OUT / "gate0_source_audit.json",
        {
            "status": "PASS_CELL_LEVEL_SOURCE_AVAILABILITY",
            "source_hashes": source_hashes,
            "source_checks": source_checks,
        },
    )
    write_json(
        OUT / "visual_inspection.json",
        {
            "status": "PENDING_MANUAL_VISUAL_INSPECTION",
            "figures": [PREVIEW.name, PNG_600.name, PDF.name, SVG.name],
        },
    )
    write_json(
        OUT / "analysis_audit.json",
        {
            "status": "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION",
            "final_expected_status": (
                "PASS_POST_HOC_CELL_EXPANDED_PROXY_UMAP"
            ),
            "analysis_semantics": (
                "post-hoc cell-expanded sample-level expression proxy UMAP"
            ),
            "source_hashes": source_hashes,
            "source_checks": source_checks,
            "expansion_checks": expansion_checks,
            "reference_model_checks": model_checks,
            "numeric_checks": numeric_checks,
            "historical_v1_v2_modified": False,
        },
    )
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_CELL_EXPANDED_PROXY_UMAP_V3",
            "status": "COMPLETE_NUMERIC_PENDING_VISUAL_INSPECTION",
        },
    )
    refresh_manifest()

    medians = metric_summary.set_index("method_label")[
        "median_local_recovery_ratio"
    ]
    print("GSE120575 CELL-EXPANDED PROXY UMAP V3")
    print("- Observed reference cells: 9522")
    print("- Proxy cells per method: 1664")
    print("- Unified shape: 16178 x 834")
    print(
        "- Maximum target mean error: "
        f"{expansion_checks['maximum_sample_gene_mean_absolute_error_tpm']:.3e}"
    )
    print(
        "- PCA50 cumulative variance: "
        f"{model_checks['pca_cumulative_explained_variance']:.6f}"
    )
    for method in ("scGen", "WOT*", "CellRank\u2020", "AgentVC"):
        print(f"- {method} median local recovery: {medians[method]:.6f}")
    print("- Numeric status: PASS")
    print("- Visual status: PENDING")


def mark_visual_pass() -> None:
    audit_path = OUT / "analysis_audit.json"
    if not audit_path.exists():
        raise RuntimeError("Numeric V3 output does not exist")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION":
        raise RuntimeError("Visual finalization requires pending numeric PASS")
    for path in (PREVIEW, PNG_600, PDF, SVG):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing figure: {path}")
    visual = {
        "status": "PASS_DENSE_UMAP_LABELS_AND_OUTLIERS_VISIBLE",
        "inspected_files": [PREVIEW.name, PNG_600.name, PDF.name],
        "eleven_population_colors_readable": True,
        "cluster_numbers_readable": True,
        "observed_background_readable": True,
        "panel_f_readable": True,
        "legend_and_footnote_readable": True,
        "outliers_hidden": False,
        "random_jitter_present": False,
        "native_single_cell_claim_present": False,
    }
    write_json(OUT / "visual_inspection.json", visual)
    audit["status"] = "PASS_POST_HOC_CELL_EXPANDED_PROXY_UMAP"
    audit["visual_inspection"] = visual["status"]
    write_json(audit_path, audit)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_CELL_EXPANDED_PROXY_UMAP_V3",
            "status": "COMPLETE_FINAL",
        },
    )
    refresh_manifest()
    print("VISUAL INSPECTION FINALIZED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-current", action="store_true")
    parser.add_argument("--mark-visual-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mark_visual_pass:
        mark_visual_pass()
    else:
        build(rebuild_current=args.rebuild_current)


if __name__ == "__main__":
    main()
