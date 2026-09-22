#!/usr/bin/env python3
"""Post-hoc GSE120575 cell-expanded proxy heterogeneity audit V1.

This analysis does not rerun or refit scGen, CellRank, Leca-VC, the cell
expansion operator, or the observed-reference PCA. It quantifies distribution
fidelity of uniformly expanded sample-level expression proxies. It must not be
interpreted as a benchmark of native single-cell generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse120575-heterogeneity-v1")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get(
    "GSE120575_CELL_EXPANDED_SOURCE",
    str(ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy/cell_expanded_umap_figure_v3"),
)).resolve()
H5AD = SOURCE / "unified_cell_expanded_expression_834genes.h5ad"
PCA_MODEL = SOURCE / "observed_reference_pca50.pkl"
SOURCE_AUDIT = SOURCE / "analysis_audit.json"
SOURCE_MANIFEST = SOURCE / "output_manifest.json"
OUT = Path(os.environ.get("GSE120575_HETEROGENEITY_OUT", str(ROOT / "outputs/GSE120575_cell_expanded_heterogeneity_audit_v1"))).resolve()

EXPECTED_HASHES = {
    "h5ad": "d3aa2ede49a58b9b7328b93ca6aa8938dfd7efdcf344cd858fc7faa2c0f90934",
    "pca_model": "1d1197b30122f09349fecbf5c0b7876b33b8a3e319cef690bae70a758062cac7",
    "source_audit": "65a3dd650d637fab8d176ea609b875710f727633e337024020c2d06cc98ba91f",
}
EXPECTED_SHAPE = (16178, 834)
EXPECTED_PCA_SHAPE = (16178, 50)
EXPECTED_PROXY_CELLS = 1664
EXPECTED_OBSERVED_POST_CELLS = 7858

OBSERVED = "Observed_Post"
METHODS = ("scGen", "CellRank_terminal_proxy", "AgentVC_proxy")
METHOD_LABELS = {
    "scGen": "scGen",
    "CellRank_terminal_proxy": "CellRank",
    "AgentVC_proxy": "Leca-VC",
}
METHOD_COLORS = {
    "scGen": "#E69F00",
    "CellRank_terminal_proxy": "#56B4E9",
    "AgentVC_proxy": "#CC79A7",
}
RESPONSES = ("Responder", "Non-responder")
BROAD_TYPES = (
    "B cell",
    "Plasma cell",
    "Monocyte/Macrophage",
    "Dendritic cell",
    "T cell",
    "NK cell",
)
EXPECTED_ELIGIBLE_TYPES = (
    "B cell",
    "Monocyte/Macrophage",
    "NK cell",
    "T cell",
)

MIN_CELLS_PER_UNIT = 2
MIN_UNITS_PER_STRATUM = 2
MIN_TOTAL_CELLS_PER_STRATUM = 10
SD_EPSILON = 1e-6
BOOTSTRAP_REPLICATES = 1000
RANDOM_SEED = 120575
STATUS = "PASS_POST_HOC_CELL_EXPANDED_HETEROGENEITY_AUDIT"

FIGURE_STEM = "GSE120575_cell_expanded_proxy_heterogeneity_audit_v1"
PREVIEW = OUT / f"{FIGURE_STEM}_preview.png"
PNG_600 = OUT / f"{FIGURE_STEM}_600dpi.png"
PDF = OUT / f"{FIGURE_STEM}_vector.pdf"
SVG = OUT / f"{FIGURE_STEM}.svg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def manifest_hash(path: Path, manifest: dict[str, Any]) -> str:
    matches = [
        item["sha256"]
        for item in manifest["files"]
        if item["relative_path"] == path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Manifest does not contain one entry for {path.name}")
    return str(matches[0])


def validate_source() -> tuple[ad.AnnData, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    for path in (H5AD, PCA_MODEL, SOURCE_AUDIT, SOURCE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes = {
        "h5ad": sha256(H5AD),
        "pca_model": sha256(PCA_MODEL),
        "source_audit": sha256(SOURCE_AUDIT),
        "source_manifest": sha256(SOURCE_MANIFEST),
    }
    expected_items = EXPECTED_HASHES.items() if os.environ.get("LECAVC_DEIDENTIFIED_POSTPROCESS") != "1" else []
    for name, expected in expected_items:
        if hashes[name] != expected:
            raise RuntimeError(
                f"Frozen source hash changed for {name}: {hashes[name]} != {expected}"
            )

    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source_audit.get("status") != "PASS_POST_HOC_CELL_EXPANDED_PROXY_UMAP":
        raise RuntimeError("Frozen V3 source did not pass its audit")
    if manifest_hash(H5AD, source_manifest) != hashes["h5ad"]:
        raise RuntimeError("H5AD hash disagrees with frozen output manifest")
    if manifest_hash(PCA_MODEL, source_manifest) != hashes["pca_model"]:
        raise RuntimeError("PCA model hash disagrees with frozen output manifest")
    expansion = source_audit.get("expansion_checks", {})
    reference = source_audit.get("reference_model_checks", {})
    if expansion.get("observed_post_used_in_expansion") is not False:
        raise RuntimeError("Observed Post was not excluded from the expansion operator")
    if expansion.get("methods_use_identical_source_cell_ids") is not True:
        raise RuntimeError("Frozen methods do not share identical source cells")
    if reference.get("pca_components") != 50:
        raise RuntimeError("Frozen reference PCA is not 50-dimensional")
    if reference.get("formal_metric_space") != "50D Observed-reference PCA":
        raise RuntimeError("Frozen formal metric space changed")

    data = ad.read_h5ad(H5AD)
    if data.shape != EXPECTED_SHAPE:
        raise RuntimeError(f"Unexpected H5AD shape: {data.shape}")
    if data.obsm["X_pca"].shape != EXPECTED_PCA_SHAPE:
        raise RuntimeError(f"Unexpected PCA coordinate shape: {data.obsm['X_pca'].shape}")
    if data.uns.get("expression_scale") != "log1p(TPM)":
        raise RuntimeError("Frozen expression scale is not log1p(TPM)")
    if data.uns.get("formal_metric_space") != "50D Observed-reference PCA":
        raise RuntimeError("H5AD formal metric space changed")
    if bool(data.uns.get("native_single_cell_predictions")):
        raise RuntimeError("H5AD no longer declares proxy rather than native predictions")

    expression = np.asarray(data.X, dtype=np.float64)
    pca = np.asarray(data.obsm["X_pca"], dtype=np.float64)
    obs = data.obs.reset_index(drop=True).copy()
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Expression contains invalid values")
    if not np.isfinite(pca).all():
        raise RuntimeError("PCA coordinates contain invalid values")
    if obs["point_id"].duplicated().any():
        raise RuntimeError("point_id is not unique")

    counts: dict[str, int] = {}
    for method in (OBSERVED,) + METHODS:
        count = int(obs["method"].astype(str).eq(method).sum())
        counts[method] = count
        expected = EXPECTED_OBSERVED_POST_CELLS if method == OBSERVED else EXPECTED_PROXY_CELLS
        if count != expected:
            raise RuntimeError(f"Unexpected row count for {method}: {count}")

    source_sets = {}
    for method in METHODS:
        local = obs.loc[obs["method"].astype(str).eq(method), "source_cell_id"].astype(str)
        if local.duplicated().any() or len(local) != EXPECTED_PROXY_CELLS:
            raise RuntimeError(f"Source IDs are invalid for {method}")
        source_sets[method] = set(local)
    if any(source_sets[method] != source_sets[METHODS[0]] for method in METHODS[1:]):
        raise RuntimeError("Selected methods do not use identical source cell IDs")

    checks = {
        "input_hashes": hashes,
        "unified_shape": list(data.shape),
        "pca_shape": list(data.obsm["X_pca"].shape),
        "expression_scale": data.uns["expression_scale"],
        "formal_metric_space": data.uns["formal_metric_space"],
        "native_single_cell_predictions": False,
        "observed_post_used_in_expansion": False,
        "methods_use_identical_source_cell_ids": True,
        "selected_method_counts": counts,
        "selected_methods": list(METHODS),
        "excluded_methods": ["WOT_weighted", "RVAgene"],
    }
    return data, expression, pca, obs, checks


def eligibility_table(obs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    datasets = (OBSERVED,) + METHODS
    for response in RESPONSES:
        for cell_type in BROAD_TYPES:
            for method in datasets:
                selected = (
                    obs["method"].astype(str).eq(method)
                    & obs["response"].astype(str).eq(response)
                    & obs["broad_cell_type"].astype(str).eq(cell_type)
                )
                counts = (
                    obs.loc[selected]
                    .groupby("sample_id", observed=True)
                    .size()
                    .astype(int)
                )
                valid = counts[counts >= MIN_CELLS_PER_UNIT]
                passed = bool(
                    len(valid) >= MIN_UNITS_PER_STRATUM
                    and int(valid.sum()) >= MIN_TOTAL_CELLS_PER_STRATUM
                )
                rows.append(
                    {
                        "response": response,
                        "broad_cell_type": cell_type,
                        "method": method,
                        "method_label": "Observed Post" if method == OBSERVED else METHOD_LABELS[method],
                        "all_units": int(len(counts)),
                        "all_cells": int(counts.sum()),
                        "valid_units": int(len(valid)),
                        "valid_cells": int(valid.sum()),
                        "minimum_cells_per_unit": MIN_CELLS_PER_UNIT,
                        "minimum_units_per_stratum": MIN_UNITS_PER_STRATUM,
                        "minimum_total_cells_per_stratum": MIN_TOTAL_CELLS_PER_STRATUM,
                        "stratum_pass": passed,
                        "valid_unit_ids": ";".join(map(str, valid.index.tolist())),
                    }
                )
    table = pd.DataFrame(rows)
    eligible = []
    for cell_type in BROAD_TYPES:
        local = table[table["broad_cell_type"].eq(cell_type)]
        if len(local) == len(RESPONSES) * len(datasets) and local["stratum_pass"].all():
            eligible.append(cell_type)
    table["cell_type_in_primary_analysis"] = table["broad_cell_type"].isin(eligible)
    table["exclusion_reason"] = np.where(
        table["cell_type_in_primary_analysis"],
        "included",
        "at least one response-method stratum failed the preregistered biological-unit/cell-count gate",
    )
    if set(eligible) != set(EXPECTED_ELIGIBLE_TYPES):
        raise RuntimeError(f"Eligibility result changed: {eligible}")
    eligible = list(EXPECTED_ELIGIBLE_TYPES)
    return table, eligible


def unit_arrays(
    values: np.ndarray,
    obs: pd.DataFrame,
    method: str,
    response: str,
    cell_type: str,
) -> dict[str, np.ndarray]:
    selected = (
        obs["method"].astype(str).eq(method)
        & obs["response"].astype(str).eq(response)
        & obs["broad_cell_type"].astype(str).eq(cell_type)
    )
    indexes = np.flatnonzero(selected.to_numpy())
    local_obs = obs.iloc[indexes]
    result: dict[str, np.ndarray] = {}
    for unit_id, positions in local_obs.groupby("sample_id", observed=True).indices.items():
        local_indexes = indexes[np.asarray(positions, dtype=int)]
        if len(local_indexes) >= MIN_CELLS_PER_UNIT:
            result[str(unit_id)] = values[local_indexes]
    if len(result) < MIN_UNITS_PER_STRATUM:
        raise RuntimeError(f"Insufficient units for {method}/{response}/{cell_type}")
    if sum(len(value) for value in result.values()) < MIN_TOTAL_CELLS_PER_STRATUM:
        raise RuntimeError(f"Insufficient cells for {method}/{response}/{cell_type}")
    return dict(sorted(result.items()))


def unit_variances(units: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    unit_ids = list(units)
    variances = np.vstack([np.var(units[unit], axis=0, ddof=1) for unit in unit_ids])
    if not np.isfinite(variances).all() or np.any(variances < -1e-12):
        raise RuntimeError("Invalid per-unit variances")
    return unit_ids, np.maximum(variances, 0.0)


def center_units(units: dict[str, np.ndarray]) -> tuple[list[str], list[np.ndarray]]:
    unit_ids = list(units)
    centered = [units[unit] - units[unit].mean(axis=0, keepdims=True) for unit in unit_ids]
    return unit_ids, centered


def mean_distance_matrix(
    left: list[np.ndarray],
    right: list[np.ndarray],
    symmetric: bool = False,
) -> np.ndarray:
    matrix = np.empty((len(left), len(right)), dtype=np.float64)
    if symmetric:
        if len(left) != len(right):
            raise ValueError("Symmetric distance matrix must be square")
        for i, left_values in enumerate(left):
            for j in range(i, len(right)):
                value = float(cdist(left_values, right[j], metric="euclidean").mean())
                matrix[i, j] = value
                matrix[j, i] = value
    else:
        for i, left_values in enumerate(left):
            for j, right_values in enumerate(right):
                matrix[i, j] = float(cdist(left_values, right_values, metric="euclidean").mean())
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise RuntimeError("Invalid unit-pair mean distance matrix")
    return matrix


def energy_distance_squared(
    pred_within: np.ndarray,
    obs_within: np.ndarray,
    cross: np.ndarray,
    pred_weights: np.ndarray,
    obs_weights: np.ndarray,
) -> tuple[float, float, float]:
    cross_term = float(pred_weights @ cross @ obs_weights)
    pred_term = float(pred_weights @ pred_within @ pred_weights)
    obs_term = float(obs_weights @ obs_within @ obs_weights)
    energy = max(0.0, 2.0 * cross_term - pred_term - obs_term)
    normalized = energy / obs_term if obs_term > 0 else math.nan
    return energy, normalized, obs_term


def build_metric_inputs(
    expression: np.ndarray,
    pca: np.ndarray,
    obs: pd.DataFrame,
    eligible_types: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    structures: dict[tuple[str, str], dict[str, Any]] = {}
    for response in RESPONSES:
        for cell_type in eligible_types:
            obs_expression = unit_arrays(expression, obs, OBSERVED, response, cell_type)
            obs_pca = unit_arrays(pca, obs, OBSERVED, response, cell_type)
            obs_unit_ids, obs_variance = unit_variances(obs_expression)
            obs_pca_ids, obs_centered = center_units(obs_pca)
            if obs_unit_ids != obs_pca_ids:
                raise RuntimeError("Observed expression/PCA unit order mismatch")
            obs_within = mean_distance_matrix(obs_centered, obs_centered, symmetric=True)
            structure: dict[str, Any] = {
                "observed_unit_ids": obs_unit_ids,
                "observed_unit_cells": [len(obs_expression[unit]) for unit in obs_unit_ids],
                "observed_variance": obs_variance,
                "observed_within": obs_within,
                "methods": {},
            }
            common_proxy_ids: list[str] | None = None
            for method in METHODS:
                pred_expression = unit_arrays(expression, obs, method, response, cell_type)
                pred_pca = unit_arrays(pca, obs, method, response, cell_type)
                pred_ids, pred_variance = unit_variances(pred_expression)
                pred_pca_ids, pred_centered = center_units(pred_pca)
                if pred_ids != pred_pca_ids:
                    raise RuntimeError("Proxy expression/PCA unit order mismatch")
                if common_proxy_ids is None:
                    common_proxy_ids = pred_ids
                elif pred_ids != common_proxy_ids:
                    raise RuntimeError(
                        f"Methods have different valid units for {response}/{cell_type}"
                    )
                pred_within = mean_distance_matrix(pred_centered, pred_centered, symmetric=True)
                cross = mean_distance_matrix(pred_centered, obs_centered)
                structure["methods"][method] = {
                    "unit_ids": pred_ids,
                    "unit_cells": [len(pred_expression[unit]) for unit in pred_ids],
                    "variance": pred_variance,
                    "within": pred_within,
                    "cross": cross,
                }
            structures[(response, cell_type)] = structure
    return structures


def point_estimates(
    structures: dict[tuple[str, str], dict[str, Any]],
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    dispersion_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    method_abs_values: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    method_energy_values: dict[str, list[float]] = {method: [] for method in METHODS}

    for (response, cell_type), structure in structures.items():
        obs_variance = structure["observed_variance"]
        obs_weights = np.full(len(structure["observed_unit_ids"]), 1.0 / len(structure["observed_unit_ids"]))
        obs_sd = np.sqrt(obs_weights @ obs_variance)
        eligible_genes = obs_sd > SD_EPSILON
        if not eligible_genes.any():
            raise RuntimeError(f"No eligible genes for {response}/{cell_type}")
        for method in METHODS:
            local = structure["methods"][method]
            pred_weights = np.full(len(local["unit_ids"]), 1.0 / len(local["unit_ids"]))
            pred_sd = np.sqrt(pred_weights @ local["variance"])
            log_ratio = np.log2((pred_sd + SD_EPSILON) / (obs_sd + SD_EPSILON))
            selected_ratio = log_ratio[eligible_genes]
            selected_abs = np.abs(selected_ratio)
            method_abs_values[method].append(selected_abs)
            for gene_index in np.flatnonzero(eligible_genes):
                dispersion_rows.append(
                    {
                        "response": response,
                        "broad_cell_type": cell_type,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "gene": genes[gene_index],
                        "observed_sd": float(obs_sd[gene_index]),
                        "predicted_sd": float(pred_sd[gene_index]),
                        "log2_predicted_to_observed_sd_ratio": float(log_ratio[gene_index]),
                        "absolute_log2_sd_ratio": float(abs(log_ratio[gene_index])),
                        "observed_unit_count": len(structure["observed_unit_ids"]),
                        "predicted_unit_count": len(local["unit_ids"]),
                    }
                )
            energy, normalized, obs_scale = energy_distance_squared(
                local["within"],
                structure["observed_within"],
                local["cross"],
                pred_weights,
                obs_weights,
            )
            method_energy_values[method].append(normalized)
            stratum_rows.append(
                {
                    "response": response,
                    "broad_cell_type": cell_type,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "eligible_gene_count": int(eligible_genes.sum()),
                    "median_log2_sd_ratio": float(np.median(selected_ratio)),
                    "median_absolute_log2_sd_ratio": float(np.median(selected_abs)),
                    "observed_unit_count": len(structure["observed_unit_ids"]),
                    "observed_valid_cell_count": int(sum(structure["observed_unit_cells"])),
                    "predicted_unit_count": len(local["unit_ids"]),
                    "predicted_valid_cell_count": int(sum(local["unit_cells"])),
                }
            )
            energy_rows.append(
                {
                    "response": response,
                    "broad_cell_type": cell_type,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "energy_distance_squared": energy,
                    "observed_internal_mean_distance": obs_scale,
                    "normalized_energy_distance": normalized,
                    "pca_dimensions": 50,
                    "within_unit_centering": True,
                    "biological_unit_weighting": "equal",
                }
            )

    overall: dict[str, dict[str, float]] = {}
    for method in METHODS:
        overall[method] = {
            "median_absolute_log2_sd_ratio": float(
                np.median(np.concatenate(method_abs_values[method]))
            ),
            "normalized_energy_distance": float(np.mean(method_energy_values[method])),
        }
    return (
        pd.DataFrame(dispersion_rows),
        pd.DataFrame(stratum_rows),
        pd.DataFrame(energy_rows),
        overall,
    )


def bootstrap_metrics(
    structures: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    detailed_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    strata = list(structures)

    for replicate in range(BOOTSTRAP_REPLICATES):
        global_abs: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        global_energy: dict[str, list[float]] = {method: [] for method in METHODS}
        for response, cell_type in strata:
            structure = structures[(response, cell_type)]
            n_obs = len(structure["observed_unit_ids"])
            first_method = structure["methods"][METHODS[0]]
            n_pred = len(first_method["unit_ids"])
            obs_weights = rng.multinomial(n_obs, np.full(n_obs, 1.0 / n_obs)) / n_obs
            pred_weights = rng.multinomial(n_pred, np.full(n_pred, 1.0 / n_pred)) / n_pred
            obs_sd = np.sqrt(obs_weights @ structure["observed_variance"])
            eligible_genes = obs_sd > SD_EPSILON
            for method in METHODS:
                local = structure["methods"][method]
                if len(local["unit_ids"]) != n_pred:
                    raise RuntimeError("Paired bootstrap unit count mismatch")
                pred_sd = np.sqrt(pred_weights @ local["variance"])
                ratios = np.log2((pred_sd + SD_EPSILON) / (obs_sd + SD_EPSILON))
                abs_ratios = np.abs(ratios[eligible_genes])
                dispersion_value = float(np.median(abs_ratios))
                _, energy_value, _ = energy_distance_squared(
                    local["within"],
                    structure["observed_within"],
                    local["cross"],
                    pred_weights,
                    obs_weights,
                )
                global_abs[method].append(abs_ratios)
                global_energy[method].append(energy_value)
                detailed_rows.extend(
                    [
                        {
                            "replicate": replicate,
                            "scope": "stratum",
                            "response": response,
                            "broad_cell_type": cell_type,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "metric": "median_absolute_log2_sd_ratio",
                            "value": dispersion_value,
                        },
                        {
                            "replicate": replicate,
                            "scope": "stratum",
                            "response": response,
                            "broad_cell_type": cell_type,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "metric": "normalized_energy_distance",
                            "value": energy_value,
                        },
                    ]
                )
        for method in METHODS:
            values = {
                "median_absolute_log2_sd_ratio": float(
                    np.median(np.concatenate(global_abs[method]))
                ),
                "normalized_energy_distance": float(np.mean(global_energy[method])),
            }
            for metric, value in values.items():
                row = {
                    "replicate": replicate,
                    "scope": "overall",
                    "response": "All",
                    "broad_cell_type": "All eligible",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "value": value,
                }
                detailed_rows.append(row)
                overall_rows.append(row)

    detailed = pd.DataFrame(detailed_rows)
    overall_bootstrap = pd.DataFrame(overall_rows)
    ci_rows: list[dict[str, Any]] = []
    group_columns = ["scope", "response", "broad_cell_type", "method", "method_label", "metric"]
    for keys, local in detailed.groupby(group_columns, observed=True, sort=False):
        values = local["value"].to_numpy(dtype=float)
        ci_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "bootstrap_replicates": len(values),
                "bootstrap_median": float(np.median(values)),
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
            }
        )
    return detailed, overall_bootstrap, pd.DataFrame(ci_rows)


def paired_differences(
    overall: dict[str, dict[str, float]],
    overall_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    leca = "AgentVC_proxy"
    for comparator in ("scGen", "CellRank_terminal_proxy"):
        for metric in ("median_absolute_log2_sd_ratio", "normalized_energy_distance"):
            leca_boot = (
                overall_bootstrap[
                    overall_bootstrap["method"].eq(leca)
                    & overall_bootstrap["metric"].eq(metric)
                ]
                .sort_values("replicate")["value"]
                .to_numpy()
            )
            comparator_boot = (
                overall_bootstrap[
                    overall_bootstrap["method"].eq(comparator)
                    & overall_bootstrap["metric"].eq(metric)
                ]
                .sort_values("replicate")["value"]
                .to_numpy()
            )
            differences = leca_boot - comparator_boot
            lower = float(np.quantile(differences, 0.025))
            upper = float(np.quantile(differences, 0.975))
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"Leca-VC minus {METHOD_LABELS[comparator]}",
                    "point_difference": float(overall[leca][metric] - overall[comparator][metric]),
                    "bootstrap_median_difference": float(np.median(differences)),
                    "ci_2_5": lower,
                    "ci_97_5": upper,
                    "lower_values_are_better": True,
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "direction_if_ci_excludes_zero": (
                        "favours Leca-VC" if upper < 0 else "favours comparator" if lower > 0 else "inconclusive"
                    ),
                    "formal_p_value_computed": False,
                }
            )
    return pd.DataFrame(rows)


def metric_summary(
    overall: dict[str, dict[str, float]],
    bootstrap_ci: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    ci = bootstrap_ci[bootstrap_ci["scope"].eq("overall")]
    for method in METHODS:
        for metric, value in overall[method].items():
            local = ci[ci["method"].eq(method) & ci["metric"].eq(metric)]
            if len(local) != 1:
                raise RuntimeError("Missing unique overall bootstrap interval")
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "point_estimate": value,
                    "bootstrap_median": float(local.iloc[0]["bootstrap_median"]),
                    "ci_2_5": float(local.iloc[0]["ci_2_5"]),
                    "ci_97_5": float(local.iloc[0]["ci_97_5"]),
                    "lower_values_are_better": True,
                }
            )
    return pd.DataFrame(rows)


def plot_figure(
    stratum_summary: pd.DataFrame,
    energy: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gse120575-heterogeneity-v1",
        }
    )
    fig = plt.figure(figsize=(13.6, 6.8), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.12, 1.48],
        left=0.07,
        right=0.985,
        bottom=0.18,
        top=0.83,
        wspace=0.25,
    )
    ax_heat = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(1, 2, wspace=0.35)
    ax_disp = fig.add_subplot(right[0, 0])
    ax_energy = fig.add_subplot(right[0, 1])

    row_pairs = [(response, cell_type) for response in RESPONSES for cell_type in EXPECTED_ELIGIBLE_TYPES]
    heat = np.empty((len(row_pairs), len(METHODS)), dtype=float)
    for row_index, (response, cell_type) in enumerate(row_pairs):
        for column_index, method in enumerate(METHODS):
            local = stratum_summary[
                stratum_summary["response"].eq(response)
                & stratum_summary["broad_cell_type"].eq(cell_type)
                & stratum_summary["method"].eq(method)
            ]
            if len(local) != 1:
                raise RuntimeError("Heatmap source row is not unique")
            heat[row_index, column_index] = float(local.iloc[0]["median_log2_sd_ratio"])
    color_limit = max(0.5, math.ceil(float(np.max(np.abs(heat))) / 0.25) * 0.25)
    image = ax_heat.imshow(heat, cmap="RdBu_r", vmin=-color_limit, vmax=color_limit, aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            color = "white" if abs(heat[i, j]) > 0.58 * color_limit else "#202020"
            ax_heat.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax_heat.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    row_labels = [
        f"{'R' if response == 'Responder' else 'NR'}  ·  {cell_type.replace('Monocyte/Macrophage', 'Mono/Mac')}"
        for response, cell_type in row_pairs
    ]
    ax_heat.set_yticks(range(len(row_labels)), row_labels)
    ax_heat.axhline(3.5, color="white", linewidth=2.5)
    ax_heat.set_title("A   Within-class dispersion ratio", loc="left", fontweight="bold", pad=10)
    ax_heat.set_xlabel("Cell-expanded expression proxy")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.048, pad=0.035)
    colorbar.set_label(r"Median log$_2$(predicted SD / observed SD)")
    colorbar.ax.axhline(colorbar.norm(0), color="black", linewidth=0.7)

    metric_axes = [
        (
            ax_disp,
            "median_absolute_log2_sd_ratio",
            "Dispersion error",
            r"Median |log$_2$ SD ratio|",
            stratum_summary,
            "median_absolute_log2_sd_ratio",
        ),
        (
            ax_energy,
            "normalized_energy_distance",
            "Distribution distance",
            "Normalized energy distance",
            energy,
            "normalized_energy_distance",
        ),
    ]
    jitter = np.linspace(-0.09, 0.09, len(row_pairs))
    for axis, metric, title, ylabel, source, source_column in metric_axes:
        for method_index, method in enumerate(METHODS):
            local_points = []
            for response, cell_type in row_pairs:
                local = source[
                    source["method"].eq(method)
                    & source["response"].eq(response)
                    & source["broad_cell_type"].eq(cell_type)
                ]
                local_points.append(float(local.iloc[0][source_column]))
            axis.scatter(
                method_index + jitter,
                local_points,
                s=21,
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.62,
                zorder=2,
            )
            local_summary = summary[
                summary["method"].eq(method) & summary["metric"].eq(metric)
            ]
            row = local_summary.iloc[0]
            point = float(row["bootstrap_median"])
            lower = float(row["ci_2_5"])
            upper = float(row["ci_97_5"])
            axis.errorbar(
                method_index,
                point,
                yerr=np.asarray([[point - lower], [upper - point]]),
                fmt="D",
                markersize=7,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="#222222",
                markeredgewidth=0.65,
                ecolor="#222222",
                elinewidth=1.35,
                capsize=3,
                zorder=4,
            )
        axis.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS], rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold", pad=8)
        axis.set_xlim(-0.35, len(METHODS) - 0.65)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(0.98, 0.98, "Lower is better", transform=axis.transAxes, ha="right", va="top", fontsize=7.5, color="#555555")

    fig.text(0.535, 0.862, "B", fontsize=12, fontweight="bold", ha="right", va="bottom")
    fig.text(0.745, 0.862, "Across-stratum summary", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#777777", markeredgecolor="white", markersize=5, alpha=0.65, label="Response × cell-type stratum"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="#777777", markeredgecolor="#222222", markersize=6, label="Bootstrap median (95% cluster-bootstrap CI)"),
    ]
    ax_energy.legend(handles=legend, loc="upper center", bbox_to_anchor=(-0.18, -0.25), frameon=False, fontsize=7.5, ncol=1)

    fig.suptitle(
        "GSE120575 within-cell-type distribution fidelity of post-hoc cell-expanded expression proxies",
        fontsize=14,
        fontweight="bold",
        y=0.955,
    )
    fig.text(
        0.5,
        0.09,
        "B cell, Mono/Mac, NK cell and T cell passed the frozen coverage gate. "
        "PCA distributions were centered within each biological unit and cell type.",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
    )
    fig.text(
        0.5,
        0.055,
        "Important: proxies were deterministically expanded from sample-level predictions; labels were inherited from source Pre cells. "
        "This is not a native single-cell generation benchmark.",
        ha="center",
        va="center",
        fontsize=8,
        color="#7A1F1F",
        fontweight="bold",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_metadata = {"Creator": "GSE120575 heterogeneity audit V1", "CreationDate": None, "ModDate": None}
    fig.savefig(PREVIEW, dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(PNG_600, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(PDF, facecolor="white", bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(
        SVG,
        facecolor="white",
        bbox_inches="tight",
        metadata={
            "Creator": "GSE120575 heterogeneity audit V1",
            "Date": "2026-09-06",
        },
    )
    plt.close(fig)
    return {
        "heatmap_color_limit": color_limit,
        "figure_size_inches": [13.6, 6.8],
        "raster_dpi": 600,
        "conceptual_panels": 2,
        "panel_b_subaxes": 2,
    }


def write_caption() -> str:
    caption = (
        "Within-cell-type distribution fidelity of post-hoc cell-expanded expression proxies in GSE120575. "
        "(A) Median gene-wise log2 ratio of predicted to observed Post within-biopsy standard deviation for "
        "scGen, CellRank and Leca-VC, stratified by response and broad cell type. Zero denotes matched dispersion; "
        "negative and positive values denote under- and over-dispersion, respectively. (B) Across-stratum summaries "
        "of the median absolute log2 dispersion ratio and normalized energy distance in the frozen 50-dimensional "
        "Observed-reference PCA space. PCA coordinates were centered within each biological unit and cell type before "
        "energy-distance calculation. Small points denote the eight response-by-cell-type strata; diamonds and whiskers "
        "denote bootstrap medians and 95% biological-unit cluster-bootstrap confidence intervals (1,000 replicates). "
        "B cell, Monocyte/Macrophage, NK cell and T cell passed the prespecified coverage gate; Dendritic and Plasma "
        "cells were excluded for insufficient proxy coverage. All proxy cells were deterministically expanded from "
        "frozen sample-level expression predictions, and their cell-type labels were inherited from source Pre cells. "
        "The analysis therefore evaluates cell-expanded proxy distribution fidelity, not native single-cell transcriptome generation."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    return caption


def run_synthetic_checks() -> dict[str, Any]:
    within = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    weights = np.asarray([0.5, 0.5])
    identical, normalized, scale = energy_distance_squared(within, within, within, weights, weights)
    base = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    doubled = base.mean(axis=0) + 2.0 * (base - base.mean(axis=0))
    ratio = np.log2((doubled.std(axis=0, ddof=1) + SD_EPSILON) / (base.std(axis=0, ddof=1) + SD_EPSILON))
    checks = {
        "identical_distribution_energy_distance_squared": identical,
        "identical_distribution_normalized_energy_distance": normalized,
        "identical_distribution_observed_scale": scale,
        "identical_distribution_energy_is_zero": bool(np.isclose(identical, 0.0, atol=1e-12)),
        "double_spread_log2_ratio": ratio.tolist(),
        "double_spread_log2_ratio_is_one": bool(np.allclose(ratio, 1.0, atol=1e-6)),
    }
    if not checks["identical_distribution_energy_is_zero"]:
        raise RuntimeError("Synthetic identical-distribution energy check failed")
    if not checks["double_spread_log2_ratio_is_one"]:
        raise RuntimeError("Synthetic doubled-spread dispersion check failed")
    return checks


def write_readme(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    eligible_types: list[str],
) -> None:
    lines = [
        "# GSE120575 cell-expanded proxy heterogeneity audit V1",
        "",
        "## Scope",
        "",
        "This is a locked post-hoc analysis of uniformly cell-expanded sample-level expression proxies. "
        "It does not benchmark native single-cell generation. No model or dimensionality-reduction model was rerun or refit.",
        "",
        "- Methods: scGen, CellRank and Leca-VC.",
        "- WOT and RVAgene are excluded by design.",
        "- Eligible broad types: " + ", ".join(eligible_types) + ".",
        "- Dendritic and Plasma cells failed the frozen coverage gate.",
        "- Expression: 834 genes on the frozen log1p(TPM) scale.",
        "- Distribution space: frozen Observed-reference 50D PCA, centered within biological unit and cell type.",
        "- Uncertainty: 1,000 biological-unit cluster-bootstrap replicates; no pseudo-cell p-values.",
        "",
        "## Overall results",
        "",
        "Lower values are better.",
        "",
        "| Method | Median absolute log2 SD ratio (95% CI) | Normalized energy distance (95% CI) |",
        "| --- | ---: | ---: |",
    ]
    for method in METHODS:
        local = summary[summary["method"].eq(method)].set_index("metric")
        dispersion = local.loc["median_absolute_log2_sd_ratio"]
        energy = local.loc["normalized_energy_distance"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {dispersion.bootstrap_median:.3f} "
            f"({dispersion.ci_2_5:.3f}–{dispersion.ci_97_5:.3f}) | "
            f"{energy.bootstrap_median:.3f} ({energy.ci_2_5:.3f}–{energy.ci_97_5:.3f}) |"
        )
    lines.extend(
        [
            "",
            "Paired Leca-VC-minus-comparator bootstrap contrasts are recorded in `paired_method_differences.csv`. "
            "A contrast is described as supported only when its 95% interval excludes zero.",
            "",
            "## Interpretation boundary",
            "",
            "Allowed: the uniformly expanded Leca-VC proxy can be compared with the uniformly expanded scGen and "
            "CellRank proxies for agreement with observed within-cell-type dispersion and centered distribution structure.",
            "",
            "Not allowed: claiming that Leca-VC generated native single-cell transcriptomes or recovered intrinsic "
            "cell-to-cell heterogeneity. Proxy expression comes from deterministic simplex expansion, and proxy labels "
            "are inherited from source Pre cells.",
            "",
            "## Files",
            "",
            f"- `{PNG_600.name}`, `{PDF.name}`, `{SVG.name}`: publication figure.",
            "- `dispersion_ratio_by_gene.csv`: gene-level standard-deviation ratios.",
            "- `dispersion_stratum_summary.csv`: response-by-cell-type summaries.",
            "- `energy_distance_by_stratum.csv`: centered 50D energy distances.",
            "- `overall_metric_summary.csv`: overall point estimates and intervals.",
            "- `bootstrap_confidence_intervals.csv`: stratum and overall intervals.",
            "- `paired_method_differences.csv`: paired Leca-VC contrasts.",
            "- `celltype_eligibility_audit.csv`: coverage gate and exclusions.",
            "- `analysis_audit.json`: inputs, parameters, checks and interpretation limits.",
            "",
            "## Paired contrasts",
            "",
        ]
    )
    for row in paired.itertuples(index=False):
        lines.append(
            f"- {row.comparison}, {row.metric}: {row.point_difference:.3f} "
            f"(95% CI {row.ci_2_5:.3f} to {row.ci_97_5:.3f}); {row.direction_if_ci_excludes_zero}."
        )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def deterministic_gate(key_files: list[Path]) -> dict[str, Any]:
    reference_path = OUT / "determinism_reference_hashes.json"
    current = {path.name: sha256(path) for path in key_files}
    code_hash = sha256(Path(__file__))
    prior: dict[str, Any] | None = None
    if reference_path.exists():
        loaded = json.loads(reference_path.read_text(encoding="utf-8"))
        if "analysis_code_sha256" in loaded and "file_hashes" in loaded:
            prior = loaded
    if prior is None or prior["analysis_code_sha256"] != code_hash:
        write_json(
            reference_path,
            {"analysis_code_sha256": code_hash, "file_hashes": current},
        )
        result = {
            "status": "BASELINE_RECORDED; RERUN_REQUIRED",
            "key_file_count": len(current),
            "all_key_files_byte_identical": None,
            "analysis_code_sha256": code_hash,
            "baseline_reason": "new or changed analysis code",
        }
    else:
        reference = prior["file_hashes"]
        matches = {name: current.get(name) == expected for name, expected in reference.items()}
        extra = sorted(set(current) - set(reference))
        missing = sorted(set(reference) - set(current))
        passed = bool(matches and all(matches.values()) and not extra and not missing)
        result = {
            "status": "PASS_BYTE_IDENTICAL_RERUN" if passed else "FAIL_NONDETERMINISTIC_RERUN",
            "key_file_count": len(current),
            "all_key_files_byte_identical": passed,
            "analysis_code_sha256": code_hash,
            "per_file_match": matches,
            "extra_files": extra,
            "missing_files": missing,
        }
        if not passed:
            write_json(OUT / "determinism_check.json", result)
            raise RuntimeError("Deterministic rerun gate failed")
    write_json(OUT / "determinism_check.json", result)
    return result


def write_manifest() -> None:
    excluded = {"output_manifest.json"}
    files = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in excluded:
            files.append(
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(
        OUT / "output_manifest.json",
        {
            "version": "GSE120575_CELL_EXPANDED_HETEROGENEITY_AUDIT_V1",
            "status": STATUS,
            "hash_algorithm": "SHA256",
            "files": files,
            "main_figure_source_data": [
                "dispersion_stratum_summary.csv",
                "energy_distance_by_stratum.csv",
                "overall_metric_summary.csv",
                "bootstrap_confidence_intervals.csv",
            ],
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_COMPLETE.json", {"status": "RUNNING"})
    data, expression, pca, obs, source_checks = validate_source()
    eligibility, eligible_types = eligibility_table(obs)
    eligibility.to_csv(OUT / "celltype_eligibility_audit.csv", index=False)

    structures = build_metric_inputs(expression, pca, obs, eligible_types)
    genes = data.var_names.astype(str).tolist()
    dispersion, stratum_summary, energy, overall = point_estimates(structures, genes)
    dispersion.to_csv(OUT / "dispersion_ratio_by_gene.csv", index=False)
    stratum_summary.to_csv(OUT / "dispersion_stratum_summary.csv", index=False)
    energy.to_csv(OUT / "energy_distance_by_stratum.csv", index=False)

    bootstrap_values, overall_bootstrap, bootstrap_ci = bootstrap_metrics(structures)
    bootstrap_values.to_csv(OUT / "bootstrap_metric_values.csv", index=False)
    bootstrap_ci.to_csv(OUT / "bootstrap_confidence_intervals.csv", index=False)
    paired = paired_differences(overall, overall_bootstrap)
    paired.to_csv(OUT / "paired_method_differences.csv", index=False)
    summary = metric_summary(overall, bootstrap_ci)
    summary.to_csv(OUT / "overall_metric_summary.csv", index=False)

    figure_checks = plot_figure(stratum_summary, energy, summary)
    caption = write_caption()
    write_readme(summary, paired, eligible_types)
    shutil.copy2(__file__, OUT / "run_GSE120575_cell_expanded_heterogeneity_audit_v1.py")
    synthetic_checks = run_synthetic_checks()

    numeric_checks = {
        "eligible_cell_types": eligible_types,
        "eligible_strata": len(structures),
        "dispersion_gene_rows": len(dispersion),
        "dispersion_stratum_rows": len(stratum_summary),
        "energy_stratum_rows": len(energy),
        "bootstrap_value_rows": len(bootstrap_values),
        "bootstrap_ci_rows": len(bootstrap_ci),
        "paired_difference_rows": len(paired),
        "all_dispersion_values_finite": bool(
            np.isfinite(dispersion["log2_predicted_to_observed_sd_ratio"]).all()
        ),
        "all_energy_values_finite_nonnegative": bool(
            np.isfinite(energy["normalized_energy_distance"]).all()
            and (energy["normalized_energy_distance"] >= 0).all()
        ),
        "all_bootstrap_values_finite_nonnegative": bool(
            np.isfinite(bootstrap_values["value"]).all()
            and (bootstrap_values["value"] >= 0).all()
        ),
        "formal_p_values": 0,
        "pseudo_cells_treated_as_independent_replicates": False,
    }
    if not all(
        [
            numeric_checks["all_dispersion_values_finite"],
            numeric_checks["all_energy_values_finite_nonnegative"],
            numeric_checks["all_bootstrap_values_finite_nonnegative"],
        ]
    ):
        raise RuntimeError("Numeric quality gate failed")

    key_files = [
        OUT / "dispersion_ratio_by_gene.csv",
        OUT / "dispersion_stratum_summary.csv",
        OUT / "energy_distance_by_stratum.csv",
        OUT / "bootstrap_metric_values.csv",
        OUT / "bootstrap_confidence_intervals.csv",
        OUT / "paired_method_differences.csv",
        OUT / "overall_metric_summary.csv",
        PREVIEW,
        PNG_600,
        PDF,
        SVG,
    ]
    determinism = deterministic_gate(key_files)
    final_status = STATUS if determinism["status"] == "PASS_BYTE_IDENTICAL_RERUN" else "PENDING_DETERMINISM_RERUN"
    audit = {
        "status": final_status,
        "analysis_semantics": "post-hoc within-cell-type distribution fidelity of uniformly cell-expanded sample-level expression proxies",
        "source_checks": source_checks,
        "coverage_gate": {
            "minimum_cells_per_unit": MIN_CELLS_PER_UNIT,
            "minimum_units_per_stratum": MIN_UNITS_PER_STRATUM,
            "minimum_total_cells_per_stratum": MIN_TOTAL_CELLS_PER_STRATUM,
            "both_response_groups_required": True,
            "eligible_cell_types": eligible_types,
            "excluded_cell_types": [cell_type for cell_type in BROAD_TYPES if cell_type not in eligible_types],
        },
        "dispersion_metric": {
            "expression_scale": "log1p(TPM)",
            "unit_variance_ddof": 1,
            "unit_weighting": "equal within response-by-cell-type stratum",
            "formula": "log2((sqrt(mean_b Var_b(pred))) + 1e-6) / (sqrt(mean_b Var_b(observed Post)) + 1e-6))",
            "observed_gene_sd_gate": SD_EPSILON,
            "overall_error": "median absolute log2 SD ratio across eligible genes and eight strata",
        },
        "energy_metric": {
            "space": "frozen 50D Observed-reference PCA",
            "pca_refit": False,
            "within_biological_unit_and_cell_type_centering": True,
            "unit_weighting": "equal",
            "cell_downsampling": False,
            "formula": "max(0, 2E||X-Y|| - E||X-X'|| - E||Y-Y'||)",
            "normalization": "divide by equal-unit-weighted Observed Post internal mean pairwise distance",
            "implementation": "exact unit-pair mean Euclidean distance matrices",
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": RANDOM_SEED,
            "resampling_unit": "biopsy/source sample within response-by-cell-type stratum",
            "paired_source_sample_weights_across_methods": True,
            "interval": "2.5th and 97.5th percentiles",
            "formal_p_values": False,
        },
        "numeric_checks": numeric_checks,
        "synthetic_checks": synthetic_checks,
        "determinism": determinism,
        "figure_checks": figure_checks,
        "overall_results": overall,
        "paired_results": paired.to_dict(orient="records"),
        "caption": caption,
        "interpretation_limits": {
            "native_single_cell_generation_benchmark": False,
            "proxy_cells_deterministically_expanded_from_sample_level_predictions": True,
            "proxy_cell_type_labels_source_pre_inherited": True,
            "allowed_claim": "comparison of uniformly expanded expression proxies for agreement with observed within-cell-type dispersion and centered distribution structure",
            "forbidden_claim": "Leca-VC generates native single-cell transcriptomes or reconstructs intrinsic cell-to-cell heterogeneity",
        },
        "historical_outputs_modified": False,
    }
    write_json(OUT / "analysis_audit.json", audit)

    if final_status == STATUS:
        write_json(
            OUT / "RUN_COMPLETE.json",
            {
                "status": "PASS",
                "analysis_status": STATUS,
                "evidence": "analysis_audit.json",
                "main_figure": PNG_600.name,
                "determinism": determinism["status"],
            },
        )
    else:
        write_json(
            OUT / "RUN_COMPLETE.json",
            {
                "status": "PENDING",
                "analysis_status": final_status,
                "next_step": "rerun the same script once for byte-identical determinism verification",
            },
        )
    write_manifest()
    data.file.close() if getattr(data, "file", None) is not None else None
    print(json.dumps({"status": final_status, "out": str(OUT), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
