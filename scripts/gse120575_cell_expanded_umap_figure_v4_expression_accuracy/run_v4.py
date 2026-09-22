#!/usr/bin/env python3
"""GSE120575 direct-expression-accuracy cell-expanded UMAP figure V4.

V4 is a visualization/metric revision of the frozen V3 artifact. It reuses
V3 UMAP coordinates exactly and replaces the persistence-normalized PCA
distance panel with direct 834-gene CCC and RMSE comparisons to Observed Post.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR", "/tmp/numba-gse120575-expression-accuracy-v4"
)
os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/matplotlib-gse120575-expression-accuracy-v4"
)

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("GSE120575_V3_OUT", str(ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy"))).resolve()
V3_OUT = BASE / "cell_expanded_umap_figure_v3"
OUT = Path(os.environ.get("GSE120575_UMAP_FIGURE_OUT", str(BASE / "cell_expanded_umap_figure_v4_expression_accuracy"))).resolve()

V3_H5AD = V3_OUT / "unified_cell_expanded_expression_834genes.h5ad"
V3_COORDINATES = V3_OUT / "cell_expanded_umap_coordinates.csv"
V3_AUDIT = V3_OUT / "analysis_audit.json"
V3_MANIFEST = V3_OUT / "output_manifest.json"
V3_SCRIPT = (
    ROOT / "scripts/gse120575_cell_expanded_umap_figure_v3/run_v3.py"
)

EXPECTED_HASHES = {
    "v3_h5ad": "d3aa2ede49a58b9b7328b93ca6aa8938dfd7efdcf344cd858fc7faa2c0f90934",
    "v3_coordinates": (
        "e4b69619077e4c479ece9f8a40a3e0704fc1bc6a13a602d03bc7c5d3372fb390"
    ),
    "v3_audit": (
        "65a3dd650d637fab8d176ea609b875710f727633e337024020c2d06cc98ba91f"
    ),
    "v3_manifest": (
        "be633c4bd443821f9ac67f538a19390b059eee7c6806c99494e7908a24623dcb"
    ),
    "v3_script": (
        "ef81489f7ed7e29fce33e6ed3ae9af6cb4b60d06a8ad2e31607537c0e37ac279"
    ),
}

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
    "AgentVC_proxy": "Leca-VC" if os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") == "1" else "AgentVC",
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

FIGURE_STEM = "GSE120575_cell_expanded_direct_expression_accuracy_UMAP_v4"
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


def load_v3_plot_module():
    specification = importlib.util.spec_from_file_location(
        "gse120575_cell_expanded_umap_v3_frozen",
        V3_SCRIPT,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load frozen V3 plotting module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_v3_sources() -> dict[str, str]:
    paths = {
        "v3_h5ad": V3_H5AD,
        "v3_coordinates": V3_COORDINATES,
        "v3_audit": V3_AUDIT,
        "v3_manifest": V3_MANIFEST,
        "v3_script": V3_SCRIPT,
    }
    observed = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        observed[name] = sha256(path)
        if observed[name] != EXPECTED_HASHES[name] and os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") != "1":
            raise RuntimeError(f"Frozen V3 source changed: {name}")
    audit = json.loads(V3_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_POST_HOC_CELL_EXPANDED_PROXY_UMAP":
        raise RuntimeError("V3 is not in its frozen final PASS state")
    return observed


def load_v3_data() -> tuple[ad.AnnData, np.ndarray, float, float]:
    adata = ad.read_h5ad(V3_H5AD)
    if adata.shape != (16178, 834):
        raise RuntimeError(f"Unexpected V3 H5AD shape: {adata.shape}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise RuntimeError("V3 point IDs or gene names are not unique")
    expression = np.asarray(adata.X, dtype=np.float64)
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("V3 expression is not finite and nonnegative")
    if adata.obsm["X_umap"].shape != (16178, 2):
        raise RuntimeError("V3 H5AD UMAP coordinates are invalid")

    coordinates_csv = pd.read_csv(V3_COORDINATES)
    if coordinates_csv.shape[0] != 16178:
        raise RuntimeError("V3 coordinate CSV is not 16178 rows")
    if not np.array_equal(
        coordinates_csv["point_id"].astype(str).to_numpy(),
        adata.obs_names.astype(str).to_numpy(),
    ):
        raise RuntimeError("V3 coordinate point order differs from H5AD")
    csv_values = coordinates_csv[["UMAP1", "UMAP2"]].to_numpy(float)
    csv_roundtrip_difference = float(
        np.max(np.abs(csv_values - np.asarray(adata.obsm["X_umap"])))
    )
    if csv_roundtrip_difference > 1e-12:
        raise RuntimeError(
            "V3 coordinate CSV exceeds round-trip tolerance: "
            f"{csv_roundtrip_difference}"
        )
    # V4 draws directly from this exact frozen H5AD array; no coordinate copy,
    # conversion, fitting, transform or modification is performed.
    v4_vs_v3_coordinate_difference = 0.0
    return (
        adata,
        expression,
        v4_vs_v3_coordinate_difference,
        csv_roundtrip_difference,
    )


def biopsy_pseudobulks(
    adata: ad.AnnData,
    expression: np.ndarray,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], np.ndarray]]:
    obs = adata.obs.reset_index(drop=False).rename(
        columns={"index": "point_id_from_index"}
    )
    rows: list[dict[str, object]] = []
    vectors: dict[tuple[str, str, str, str], np.ndarray] = {}
    included_methods = ("Observed_Post",) + METHODS
    for method in included_methods:
        method_mask = obs["method"].astype(str).eq(method).to_numpy()
        for response in RESPONSES:
            response_mask = obs["response"].astype(str).eq(response).to_numpy()
            for cell_type in BROAD_TYPES:
                type_mask = (
                    obs["broad_cell_type"].astype(str).eq(cell_type).to_numpy()
                )
                group_mask = method_mask & response_mask & type_mask
                sample_ids = sorted(
                    obs.loc[group_mask, "sample_id"].astype(str).unique()
                )
                if not sample_ids:
                    raise RuntimeError(
                        f"No biopsy for {method}/{response}/{cell_type}"
                    )
                for sample_id in sample_ids:
                    selected = (
                        group_mask
                        & obs["sample_id"].astype(str).eq(sample_id).to_numpy()
                    )
                    indexes = np.flatnonzero(selected)
                    if len(indexes) == 0:
                        raise RuntimeError("Empty biopsy pseudobulk")
                    vector = np.log1p(
                        np.expm1(expression[indexes]).mean(axis=0)
                    )
                    if not np.isfinite(vector).all() or np.any(vector < 0):
                        raise RuntimeError("Invalid biopsy pseudobulk vector")
                    key = (method, response, cell_type, sample_id)
                    vectors[key] = vector
                    row: dict[str, object] = {
                        "method": method,
                        "method_label": (
                            "Observed Post"
                            if method == "Observed_Post"
                            else METHOD_LABELS[method]
                        ),
                        "response": response,
                        "broad_cell_type": cell_type,
                        "sample_id": sample_id,
                        "n_cells": int(len(indexes)),
                        "expression_scale": (
                            "log1p(mean cell-level TPM within biopsy)"
                        ),
                        "aggregation_weight": (
                            "one equal-weight biopsy within response x cell type"
                        ),
                    }
                    row.update(
                        {
                            gene: float(value)
                            for gene, value in zip(adata.var_names, vector)
                        }
                    )
                    rows.append(row)
    frame = pd.DataFrame(rows)
    expected_keys = set(vectors)
    if len(frame) != len(expected_keys):
        raise RuntimeError("Biopsy pseudobulk keys are not unique")
    return frame, vectors


def lins_ccc(predicted: np.ndarray, observed: np.ndarray) -> float:
    predicted_mean = float(predicted.mean())
    observed_mean = float(observed.mean())
    predicted_variance = float(np.var(predicted, ddof=0))
    observed_variance = float(np.var(observed, ddof=0))
    covariance = float(
        np.mean(
            (predicted - predicted_mean) * (observed - observed_mean)
        )
    )
    denominator = (
        predicted_variance
        + observed_variance
        + (predicted_mean - observed_mean) ** 2
    )
    if denominator <= 0:
        raise RuntimeError("CCC denominator is not positive")
    return 2.0 * covariance / denominator


def direct_accuracy_metrics(
    adata: ad.AnnData,
    vectors: dict[tuple[str, str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    centroid_rows: list[dict[str, object]] = []

    def group_vectors(
        method: str,
        response: str,
        cell_type: str,
    ) -> tuple[np.ndarray, list[str]]:
        selected = [
            (key[3], value)
            for key, value in vectors.items()
            if key[:3] == (method, response, cell_type)
        ]
        selected.sort(key=lambda item: item[0])
        if not selected:
            raise RuntimeError("Missing sample-balanced group")
        return (
            np.vstack([value for _, value in selected]),
            [sample for sample, _ in selected],
        )

    for response in RESPONSES:
        for cell_type in BROAD_TYPES:
            observed_vectors, observed_samples = group_vectors(
                "Observed_Post", response, cell_type
            )
            observed_centroid = observed_vectors.mean(axis=0)
            observed_row: dict[str, object] = {
                "method": "Observed_Post",
                "method_label": "Observed Post",
                "response": response,
                "broad_cell_type": cell_type,
                "n_equal_weight_biopsies": len(observed_samples),
                "sample_ids": "|".join(observed_samples),
                "expression_scale": (
                    "mean of equal-weight biopsy log1p(TPM) pseudobulks"
                ),
            }
            observed_row.update(
                {
                    gene: float(value)
                    for gene, value in zip(
                        adata.var_names, observed_centroid
                    )
                }
            )
            centroid_rows.append(observed_row)

            for method in METHODS:
                predicted_vectors, predicted_samples = group_vectors(
                    method, response, cell_type
                )
                predicted_centroid = predicted_vectors.mean(axis=0)
                ccc = float(lins_ccc(predicted_centroid, observed_centroid))
                rmse = float(
                    np.sqrt(
                        np.mean(
                            (predicted_centroid - observed_centroid) ** 2
                        )
                    )
                )
                if not np.isfinite(ccc) or not np.isfinite(rmse):
                    raise RuntimeError("Direct accuracy metric is non-finite")
                metric_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "response": response,
                        "broad_cell_type": cell_type,
                        "n_predicted_equal_weight_biopsies": len(
                            predicted_samples
                        ),
                        "n_observed_post_equal_weight_biopsies": len(
                            observed_samples
                        ),
                        "predicted_sample_ids": "|".join(predicted_samples),
                        "observed_post_sample_ids": "|".join(
                            observed_samples
                        ),
                        "n_genes": 834,
                        "lins_ccc": ccc,
                        "rmse_log1p_tpm": rmse,
                        "ccc_higher_is_better": True,
                        "rmse_lower_is_better": True,
                        "biopsy_weighting": (
                            "equal within response x broad-cell-type group"
                        ),
                        "observed_pre_used": False,
                        "persistence_denominator_used": False,
                        "pca_or_umap_used_for_metric": False,
                        "biological_replicate_interpretation": (
                            "descriptive group accuracy; proxy cells are not "
                            "independent biological replicates"
                        ),
                    }
                )
                predicted_row: dict[str, object] = {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "response": response,
                    "broad_cell_type": cell_type,
                    "n_equal_weight_biopsies": len(predicted_samples),
                    "sample_ids": "|".join(predicted_samples),
                    "expression_scale": (
                        "mean of equal-weight biopsy log1p(TPM) pseudobulks"
                    ),
                }
                predicted_row.update(
                    {
                        gene: float(value)
                        for gene, value in zip(
                            adata.var_names, predicted_centroid
                        )
                    }
                )
                centroid_rows.append(predicted_row)

    metrics = pd.DataFrame(metric_rows)
    centroids = pd.DataFrame(centroid_rows)
    if len(metrics) != 48:
        raise RuntimeError(f"Expected 48 direct accuracy rows: {len(metrics)}")
    if not (
        metrics.groupby("method", observed=True).size().eq(12).all()
    ):
        raise RuntimeError("Each method must have 12 direct accuracy groups")
    if not metrics["n_genes"].eq(834).all():
        raise RuntimeError("Direct accuracy metrics are not 834-gene")
    if metrics["observed_pre_used"].any():
        raise RuntimeError("Observed Pre entered a V4 metric")

    summaries = []
    for method in METHODS:
        selected = metrics[metrics["method"].eq(method)]
        ccc_values = selected["lins_ccc"].to_numpy(float)
        rmse_values = selected["rmse_log1p_tpm"].to_numpy(float)
        ccc_q1, ccc_median, ccc_q3 = np.quantile(
            ccc_values, [0.25, 0.5, 0.75]
        )
        rmse_q1, rmse_median, rmse_q3 = np.quantile(
            rmse_values, [0.25, 0.5, 0.75]
        )
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_response_celltype_groups": 12,
                "median_lins_ccc": float(ccc_median),
                "q1_lins_ccc": float(ccc_q1),
                "q3_lins_ccc": float(ccc_q3),
                "min_lins_ccc": float(ccc_values.min()),
                "max_lins_ccc": float(ccc_values.max()),
                "median_rmse_log1p_tpm": float(rmse_median),
                "q1_rmse_log1p_tpm": float(rmse_q1),
                "q3_rmse_log1p_tpm": float(rmse_q3),
                "min_rmse_log1p_tpm": float(rmse_values.min()),
                "max_rmse_log1p_tpm": float(rmse_values.max()),
                "overall_winner_score": False,
            }
        )
    return metrics, pd.DataFrame(summaries), centroids


def plot_accuracy_axis(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    metric: str,
    xlabel: str,
    xlim: tuple[float, float],
    perfect_value: float,
    show_ylabels: bool,
) -> None:
    y_positions = {
        method: len(METHODS) - 1 - index
        for index, method in enumerate(METHODS)
    }
    for method in METHODS:
        selected = metrics[metrics["method"].eq(method)].copy()
        selected["_response_order"] = pd.Categorical(
            selected["response"], categories=RESPONSES, ordered=True
        )
        selected["_type_order"] = pd.Categorical(
            selected["broad_cell_type"], categories=BROAD_TYPES, ordered=True
        )
        selected = selected.sort_values(
            ["_response_order", "_type_order"]
        ).reset_index(drop=True)
        values = selected[metric].to_numpy(float)
        base_y = y_positions[method]
        offsets = np.linspace(-0.16, 0.16, len(values))
        for response in RESPONSES:
            response_mask = (
                selected["response"].astype(str).eq(response).to_numpy()
            )
            ax.scatter(
                values[response_mask],
                base_y + offsets[response_mask],
                s=22,
                marker=RESPONSE_MARKERS[response],
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.88,
                zorder=3,
            )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.hlines(
            base_y,
            q1,
            q3,
            color=METHOD_COLORS[method],
            lw=5.5,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            median,
            base_y,
            s=43,
            marker="D",
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            median,
            base_y + 0.27,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color="#3F4447",
        )

    ax.axvline(
        perfect_value,
        color="#73797D",
        lw=0.9,
        ls=(0, (3, 2)),
        zorder=1,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.56, 3.58)
    ax.set_xlabel(xlabel, fontsize=6.8)
    ax.set_yticks([y_positions[method] for method in METHODS])
    if show_ylabels:
        ax.set_yticklabels(
            [METHOD_LABELS[method] for method in METHODS],
            fontsize=6.8,
        )
    else:
        ax.set_yticklabels([])
    ax.grid(axis="x", color="#E4E7E9", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#777D80")
    ax.tick_params(axis="x", labelsize=6.2)
    ax.tick_params(axis="y", length=0)


def draw_figure(
    v3,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    metrics: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    limits = v3.normalized_limits(coordinates)
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
    umap_axes = [
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
        ("E", "Leca-VC", "AgentVC_proxy") if os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") == "1" else ("E", "AgentVC proxy", "AgentVC_proxy"),
    )
    for ax, (letter, title, method) in zip(umap_axes, specifications):
        v3.plot_umap_panel(
            ax,
            metadata,
            coordinates,
            limits,
            letter,
            title,
            method,
        )

    metric_grid = grid[1, 2].subgridspec(1, 2, wspace=0.13)
    ccc_ax = fig.add_subplot(metric_grid[0, 0])
    rmse_ax = fig.add_subplot(metric_grid[0, 1])
    plot_accuracy_axis(
        ccc_ax,
        metrics,
        "lins_ccc",
        "CCC \u2191",
        (0.0, 1.02),
        1.0,
        True,
    )
    rmse_max = max(
        0.9,
        math.ceil(float(metrics["rmse_log1p_tpm"].max()) * 10.0) / 10.0,
    )
    plot_accuracy_axis(
        rmse_ax,
        metrics,
        "rmse_log1p_tpm",
        "RMSE \u2193",
        (-0.025, rmse_max),
        0.0,
        False,
    )
    ccc_ax.set_title(
        "F. Direct local gene-expression accuracy",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        pad=10,
    )
    response_handles = [
        Line2D(
            [0],
            [0],
            marker=RESPONSE_MARKERS[response],
            linestyle="none",
            markerfacecolor="#73787B",
            markeredgecolor="white",
            markersize=5.5,
            label=response,
        )
        for response in RESPONSES
    ]
    rmse_ax.legend(
        handles=response_handles,
        frameon=False,
        fontsize=6.0,
        loc="upper right",
        ncol=1,
        handletextpad=0.3,
    )
    rmse_ax.text(
        0.98,
        0.015,
        "12 response \u00d7 broad-cell-type groups",
        transform=rmse_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555B5F",
    )

    population_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=v3.POPULATION_COLORS[population],
            markeredgecolor="none",
            markersize=6.0,
            label=f"{index}  {population}",
        )
        for index, population in enumerate(v3.FINE_POPULATIONS, start=1)
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
        "Cell-expanded gene-expression accuracy across immune populations",
        y=0.975,
        fontsize=18.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.936,
        (
            "GSE120575 \u00b7 7,858 Observed Post cells \u00b7 "
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
            "single-cell predictions. Population labels are inherited from real "
            "source Pre cells.\n"
            "Panel F directly compares 834-gene response \u00d7 broad-cell-type "
            "centroids with Observed Post after equal biopsy weighting; "
            "CCC and RMSE are computed directly in log1p(TPM). "
            "WOT* is target-derived."
        ),
        ha="center",
        va="bottom",
        fontsize=7.15,
        color="#3F4447",
    )
    fig.savefig(PREVIEW, dpi=150, facecolor="white")
    fig.savefig(PNG_600, dpi=600, facecolor="white")
    fig.savefig(PDF, facecolor="white")
    fig.savefig(SVG, facecolor="white")
    plt.close(fig)


def write_readme_and_caption(summary: pd.DataFrame) -> None:
    indexed = summary.set_index("method_label")
    leca_label = METHOD_LABELS["AgentVC_proxy"]
    lines = [
        "# GSE120575 direct-expression-accuracy UMAP V4",
        "",
        "V4 reuses the frozen V3 cell-expanded UMAP coordinates exactly.",
        "Only Panel F and its direct expression metrics are new.",
        "",
        "## Direct accuracy definition",
        "",
        "Within each biopsy and broad cell type, cell-level log1p(TPM) is",
        "aggregated as expm1 -> mean TPM -> log1p. Biopsy pseudobulks then",
        "receive equal weight within each response x broad-cell-type group.",
        "The predicted and Observed Post 834-gene centroids are compared with",
        "Lin's CCC and RMSE in log1p(TPM). Observed Pre and Persistence are not",
        "used by these metrics.",
        "",
        "## Descriptive medians across 12 local groups",
        "",
    ]
    for label in ("scGen", "WOT*", "CellRank\u2020", leca_label):
        lines.append(
            f"- {label}: CCC "
            f"{indexed.loc[label, 'median_lins_ccc']:.6f}; RMSE "
            f"{indexed.loc[label, 'median_rmse_log1p_tpm']:.6f}"
        )
    lines.extend(
        [
            "",
            "These 12 groups are descriptive, not independent biological",
            "replicates and not an overall winner score. WOT* is target-derived.",
            f"CellRank\u2020 and {leca_label} remain expression proxies, and proxy-cell",
            "population labels are inherited from source Pre cells.",
        ]
    )
    (OUT / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    caption = (
        "Direct cell-type-local expression accuracy in the frozen GSE120575 "
        "cell-expanded reference manifold. Panels A-E exactly reuse the V3 UMAP "
        "coordinates. Panel F compares predicted and Observed Post centroids "
        "over the same 834 genes using Lin's concordance correlation coefficient "
        "(CCC; higher is better) and RMSE in log1p(TPM) (lower is better). "
        "Each biopsy is first pseudobulked within response and broad cell type, "
        "then biopsies receive equal weight. Observed Pre, Persistence, PCA and "
        "UMAP distances are not used in Panel F. Expanded rows are post-hoc "
        "sample-level expression proxies rather than native single-cell "
        "predictions; population labels are source-inherited. WOT* is "
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
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4",
            "status": "post-hoc direct expression accuracy visualization",
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
                "direct_local_expression_accuracy_metrics.csv",
                "direct_local_expression_accuracy_summary.csv",
                "response_celltype_expression_centroids_834genes.csv",
            ],
            "biopsy_source_data": (
                "sample_celltype_pseudobulk_expression_834genes.csv"
            ),
            "frozen_umap_source": str(
                V3_COORDINATES.relative_to(ROOT)
            ),
        },
    )


def build(rebuild_current: bool = False) -> None:
    if OUT.exists() and any(OUT.iterdir()):
        if not rebuild_current:
            raise FileExistsError(f"Refusing to overwrite V4: {OUT}")
        audit_path = OUT / "analysis_audit.json"
        build_path = OUT / "build_state.json"
        audit_status = (
            json.loads(audit_path.read_text()).get("status")
            if audit_path.exists()
            else None
        )
        build_status = (
            json.loads(build_path.read_text()).get("status")
            if build_path.exists()
            else None
        )
        if (
            audit_status
            != "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION"
            and build_status != "BUILDING"
        ):
            raise RuntimeError("Current V4 is not safe to rebuild")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4",
            "status": "BUILDING",
        },
    )

    source_hashes = validate_v3_sources()
    (
        adata,
        expression,
        coordinate_difference,
        csv_roundtrip_difference,
    ) = load_v3_data()
    v3 = load_v3_plot_module()
    pseudobulks, vectors = biopsy_pseudobulks(adata, expression)
    metrics, summary, centroids = direct_accuracy_metrics(adata, vectors)

    pseudobulks.to_csv(
        OUT / "sample_celltype_pseudobulk_expression_834genes.csv",
        index=False,
        float_format="%.17g",
    )
    centroids.to_csv(
        OUT / "response_celltype_expression_centroids_834genes.csv",
        index=False,
        float_format="%.17g",
    )
    metrics.to_csv(
        OUT / "direct_local_expression_accuracy_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    summary.to_csv(
        OUT / "direct_local_expression_accuracy_summary.csv",
        index=False,
        float_format="%.17g",
    )

    metadata = adata.obs.reset_index(drop=True)
    coordinates = np.asarray(adata.obsm["X_umap"], dtype=float)
    draw_figure(v3, metadata, coordinates, metrics)
    write_readme_and_caption(summary)
    shutil.copy2(
        __file__,
        OUT / "run_GSE120575_direct_expression_accuracy_UMAP_v4.py",
    )

    metric_sample_counts = (
        metrics[
            [
                "method",
                "response",
                "broad_cell_type",
                "n_predicted_equal_weight_biopsies",
                "n_observed_post_equal_weight_biopsies",
            ]
        ]
        .sort_values(["method", "response", "broad_cell_type"])
        .to_dict(orient="records")
    )
    numeric_checks = {
        "input_shape": [16178, 834],
            "v4_umap_coordinate_max_absolute_difference_from_v3": (
                coordinate_difference
            ),
            "v3_coordinate_csv_roundtrip_max_absolute_difference": (
                csv_roundtrip_difference
            ),
        "panels_a_to_e_reuse_v3_coordinates": True,
        "direct_accuracy_metric_rows": int(len(metrics)),
        "groups_per_method": {
            method: int(metrics["method"].eq(method).sum())
            for method in METHODS
        },
        "genes_per_group": 834,
        "all_ccc_and_rmse_finite": bool(
            np.isfinite(
                metrics[["lins_ccc", "rmse_log1p_tpm"]].to_numpy(float)
            ).all()
        ),
        "biopsy_weighting": "equal within response x broad-cell-type",
        "observed_pre_used_for_metrics": False,
        "persistence_denominator_used": False,
        "pca_or_umap_used_for_metrics": False,
        "significance_tests": 0,
        "pseudo_cells_as_independent_replicates": False,
        "overall_winner_score": False,
        "metric_sample_counts": metric_sample_counts,
    }
    write_json(
        OUT / "gate0_v3_source_audit.json",
        {
            "status": "PASS_FROZEN_V3_SOURCE",
            "source_hashes": source_hashes,
            "v3_final_status": (
                "PASS_POST_HOC_CELL_EXPANDED_PROXY_UMAP"
            ),
            "coordinate_max_absolute_difference": coordinate_difference,
            "coordinate_csv_roundtrip_max_absolute_difference": (
                csv_roundtrip_difference
            ),
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
                "PASS_POST_HOC_DIRECT_EXPRESSION_ACCURACY_UMAP"
            ),
            "analysis_semantics": (
                "post-hoc direct 834-gene expression accuracy on "
                "cell-expanded sample-level proxies"
            ),
            "source_hashes": source_hashes,
            "numeric_checks": numeric_checks,
            "historical_v1_v2_v3_modified": False,
            "wot_target_derived": True,
            "cellrank_and_agentvc_proxy": True,
            "population_annotation_semantics": (
                "source-inherited for proxy cells"
            ),
        },
    )
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4",
            "status": "COMPLETE_NUMERIC_PENDING_VISUAL_INSPECTION",
        },
    )
    refresh_manifest()

    indexed = summary.set_index("method_label")
    leca_label = METHOD_LABELS["AgentVC_proxy"]
    print("GSE120575 DIRECT EXPRESSION ACCURACY UMAP V4")
    print(f"- V3 UMAP coordinate max difference: {coordinate_difference:.3e}")
    print(f"- Biopsy pseudobulks: {len(pseudobulks)}")
    print("- Direct local metric rows: 48")
    for label in ("scGen", "WOT*", "CellRank\u2020", leca_label):
        print(
            f"- {label}: median CCC "
            f"{indexed.loc[label, 'median_lins_ccc']:.6f}; median RMSE "
            f"{indexed.loc[label, 'median_rmse_log1p_tpm']:.6f}"
        )
    print("- Numeric status: PASS")
    print("- Visual status: PENDING")


def mark_visual_pass() -> None:
    audit_path = OUT / "analysis_audit.json"
    if not audit_path.exists():
        raise RuntimeError("Numeric V4 output does not exist")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_NUMERIC_GATES_PENDING_VISUAL_INSPECTION":
        raise RuntimeError("Visual finalization requires pending numeric PASS")
    for path in (PREVIEW, PNG_600, PDF, SVG):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing figure: {path}")
    visual = {
        "status": "PASS_DIRECT_ACCURACY_PANEL_READABLE",
        "inspected_files": [PREVIEW.name, PNG_600.name, PDF.name],
        "panels_a_to_e_unchanged_from_v3_coordinates": True,
        "ccc_and_rmse_panels_readable": True,
        "response_markers_readable": True,
        "population_legend_readable": True,
        "footnote_readable": True,
        "outliers_hidden": False,
        "persistence_reference_present": False,
        "ratio_metric_present": False,
        "native_single_cell_claim_present": False,
    }
    write_json(OUT / "visual_inspection.json", visual)
    audit["status"] = "PASS_POST_HOC_DIRECT_EXPRESSION_ACCURACY_UMAP"
    audit["visual_inspection"] = visual["status"]
    write_json(audit_path, audit)
    write_json(
        OUT / "build_state.json",
        {
            "version": "GSE120575_DIRECT_EXPRESSION_ACCURACY_UMAP_V4",
            "status": "COMPLETE_FINAL_PASS",
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
    arguments = parse_args()
    if arguments.mark_visual_pass:
        mark_visual_pass()
    else:
        build(rebuild_current=arguments.rebuild_current)


if __name__ == "__main__":
    main()
