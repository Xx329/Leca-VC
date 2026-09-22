#!/usr/bin/env python3
"""Add the frozen CellRank terminal-lineage proxy to the GSE120575 plot.

The four previously published bars and intervals are read unchanged from the
completed V2 benchmark. CellRank is shown as a dagger-marked post-hoc proxy:
each of its four Pre-only terminal lineages is mapped to a broad cell type by
the same frozen Pre-only classifier used by V1, then sample-level fate weights
are summed by mapped broad type. It is not represented as native generated
Post expression.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse120575-composition-v3")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "outputs/GSE120575_external_baseline_unified_composition_v2"
V3_EXPRESSION = ROOT / "outputs/GSE120575_unified_expression_benchmark_v3_cellrank_proxy"
OUT = ROOT / "outputs/GSE120575_external_baseline_unified_composition_v3_cellrank_proxy"
DATA = ROOT / "data/wot_scgen_cellrank_native_pre_post_reports"
SCGEN = DATA / "scgen_gse120575_pre_to_post/adapter/generated_post_from_pre.h5ad"
CELLRANK = DATA / "cellrank_gse120575_pre_to_post_native/native_cellrank_pre_terminal_status_all_genes.h5ad"
ANNOTATION = ROOT / "outputs/GSE120575_full_online_celltype/annotation/cell_level_annotations.csv"
FATE_WEIGHTS = V3_EXPRESSION / "cellrank_fate_weights.csv"

RESPONSES = ("Responder", "Non-responder")
SOURCE_SAMPLES = ("Pre_P24", "Pre_P29", "Pre_P35", "Pre_P2", "Pre_P3", "Pre_P27")
CELL_TYPES = (
    "B cell",
    "Plasma cell",
    "Monocyte/Macrophage",
    "Dendritic cell",
    "T cell",
    "NK cell",
)
METHODS = ("Real_Post", "scGen", "WOT_weighted", "CellRank_terminal_proxy", "Our_Agent_PhysiCell_V2.1")
METHOD_LABELS = {
    "Real_Post": "Observed",
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "CellRank_terminal_proxy": "CellRank\N{DAGGER}",
    "Our_Agent_PhysiCell_V2.1": "Leca-VC",
}
METHOD_COLORS = {
    "Real_Post": "#4C78A8",
    "scGen": "#F58518",
    "WOT_weighted": "#54A24B",
    "CellRank_terminal_proxy": "#4C9ED9",
    "Our_Agent_PhysiCell_V2.1": "#B279A2",
}
LINEAGES = tuple(f"terminal_lineage_0{i}_dpt_rank_{i}" for i in range(1, 5))
RANDOM_SEED = 120575
N_BOOTSTRAP = 1000

STEM = "GSE120575_celltype_composition_with_CellRank_proxy"
PREVIEW = OUT / f"{STEM}_preview.png"
PNG = OUT / f"{STEM}_600dpi.png"
PDF = OUT / f"{STEM}_vector.pdf"
SVG = OUT / f"{STEM}.svg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense(matrix: object) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def derive_lineage_mapping() -> pd.DataFrame:
    """Rebuild the frozen V1 classifier and audit each terminal lineage."""
    scgen = ad.read_h5ad(SCGEN, backed="r")
    genes = list(map(str, scgen.var_names))
    scgen.file.close()

    cellrank = ad.read_h5ad(CELLRANK, backed="r")
    positions = pd.Index(cellrank.var_names.astype(str)).get_indexer(genes)
    if (positions < 0).any():
        raise RuntimeError("CellRank is missing frozen V1 classifier genes")
    expression = np.maximum(dense(cellrank[:, positions].X).astype(np.float32), 0.0)
    ids = pd.Index(cellrank.obs_names.astype(str))
    samples = cellrank.obs["sample_label"].astype(str).to_numpy()
    terminal = cellrank.obs["cellrank_terminal_state_membership"].astype(str).to_numpy()

    annotations = pd.read_csv(ANNOTATION).set_index("cell_id")
    labels = annotations.reindex(ids)["broad_cell_type"]
    if labels.isna().any():
        raise RuntimeError("CellRank cells failed the frozen Stage 0 annotation join")
    labels_array = labels.astype(str).to_numpy()
    train = ~np.isin(samples, SOURCE_SAMPLES)
    scaler = StandardScaler().fit(expression[train])
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
        n_jobs=1,
    ).fit(scaler.transform(expression[train]), labels_array[train])

    records: list[dict[str, object]] = []
    for lineage in LINEAGES:
        selected = terminal == lineage
        if int(selected.sum()) != 30:
            raise RuntimeError(f"{lineage} does not contain exactly 30 frozen terminal cells")
        predicted = classifier.predict(scaler.transform(expression[selected]))
        observed_labels = labels_array[selected]
        predicted_counts = pd.Series(predicted).value_counts()
        observed_counts = pd.Series(observed_labels).value_counts()
        mapped = str(predicted_counts.index[0])
        if int(predicted_counts.iloc[0]) != 30 or observed_counts.to_dict() != {mapped: 30}:
            raise RuntimeError(f"Terminal lineage mapping is not unanimous: {lineage}")
        records.append(
            {
                "terminal_lineage": lineage,
                "mapped_broad_cell_type": mapped,
                "terminal_cell_count": 30,
                "classifier_vote_fraction": 1.0,
                "stage0_annotation_fraction": 1.0,
                "mapping_semantics": "post-hoc Pre-only terminal-lineage proxy; not native generated Post composition",
            }
        )
    cellrank.file.close()
    return pd.DataFrame(records)


def build_cellrank_samples(mapping: pd.DataFrame) -> pd.DataFrame:
    weights = pd.read_csv(FATE_WEIGHTS)
    if set(weights["sample_id"].astype(str)) != set(SOURCE_SAMPLES):
        raise RuntimeError("CellRank fate-weight sample scope changed")
    lineage_to_type = mapping.set_index("terminal_lineage")["mapped_broad_cell_type"].to_dict()
    rows: list[dict[str, object]] = []
    for sample in SOURCE_SAMPLES:
        record = weights.loc[weights["sample_id"].astype(str).eq(sample)]
        if len(record) != 1:
            raise RuntimeError(f"Expected one CellRank fate row for {sample}")
        record = record.iloc[0]
        vector = {cell_type: 0.0 for cell_type in CELL_TYPES}
        for lineage in LINEAGES:
            vector[lineage_to_type[lineage]] += float(record[f"weight_{lineage}"])
        if not np.isclose(sum(vector.values()), 1.0, atol=1e-10):
            raise RuntimeError(f"CellRank composition does not sum to one for {sample}")
        for cell_type in CELL_TYPES:
            rows.append(
                {
                    "method": "CellRank_terminal_proxy",
                    "sample_id": sample,
                    "response": str(record["response"]),
                    "therapy": "anti-PD1",
                    "cell_type": cell_type,
                    "proportion": vector[cell_type],
                    "statistical_unit": "source Pre sample",
                    "semantics": "fate-weighted broad-type mapping of Pre-only terminal lineages",
                }
            )
    return pd.DataFrame(rows)


def bootstrap_group_means(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for response_index, response in enumerate(RESPONSES):
        subset = samples[samples["response"].eq(response)]
        sample_ids = sorted(subset["sample_id"].unique())
        matrix = np.asarray(
            [
                [
                    float(
                        subset[
                            subset["sample_id"].eq(sample)
                            & subset["cell_type"].eq(cell_type)
                        ]["proportion"].iloc[0]
                    )
                    for cell_type in CELL_TYPES
                ]
                for sample in sample_ids
            ],
            dtype=float,
        )
        rng = np.random.default_rng(RANDOM_SEED + 310 + response_index)
        bootstrap = np.asarray(
            [matrix[rng.integers(0, len(matrix), len(matrix))].mean(axis=0) for _ in range(N_BOOTSTRAP)]
        )
        means = matrix.mean(axis=0)
        for index, cell_type in enumerate(CELL_TYPES):
            rows.append(
                {
                    "response": response,
                    "method": "CellRank_terminal_proxy",
                    "cell_type": cell_type,
                    "proportion": float(means[index]),
                    "bootstrap_ci95_low": float(np.quantile(bootstrap[:, index], 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(bootstrap[:, index], 0.975)),
                    "n_statistical_units": len(matrix),
                    "statistical_unit": "source Pre sample",
                }
            )
    return pd.DataFrame(rows)


def calculate_rmse(source: pd.DataFrame) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for response in RESPONSES:
        observed = (
            source[source["response"].eq(response) & source["method"].eq("Real_Post")]
            .set_index("cell_type")
            .reindex(CELL_TYPES)["proportion"]
            .to_numpy(float)
        )
        for method in METHODS[1:]:
            predicted = (
                source[source["response"].eq(response) & source["method"].eq(method)]
                .set_index("cell_type")
                .reindex(CELL_TYPES)["proportion"]
                .to_numpy(float)
            )
            result[(response, method)] = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    return result


def render(source: pd.DataFrame, rmse: dict[tuple[str, str], float]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.0), sharey=True)
    x = np.arange(len(CELL_TYPES), dtype=float)
    width = 0.155
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    for ax, response, panel in zip(axes, RESPONSES, ("A", "B")):
        for method, offset in zip(METHODS, offsets):
            subset = (
                source[source["response"].eq(response) & source["method"].eq(method)]
                .set_index("cell_type")
                .reindex(CELL_TYPES)
            )
            values = subset["proportion"].to_numpy(float)
            lows = subset["bootstrap_ci95_low"].to_numpy(float)
            highs = subset["bootstrap_ci95_high"].to_numpy(float)
            errors = np.vstack((values - lows, highs - values))
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.5,
                yerr=errors,
                error_kw={"ecolor": "#303030", "elinewidth": 0.75, "capsize": 1.9, "capthick": 0.75},
                zorder=3,
            )
            for bar, value, high in zip(bars, values, highs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(max(value, high) + 0.016, 1.055),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    color="#252525",
                    clip_on=False,
                )

        ax.set_title(f"{panel}. {response} post-treatment", loc="left", fontsize=12.2, y=1.135, pad=0)
        first_line = "   ".join(
            f"{METHOD_LABELS[m]} RMSE {rmse[(response, m)]:.3f}" for m in METHODS[1:3]
        )
        second_line = "   ".join(
            f"{METHOD_LABELS[m]} RMSE {rmse[(response, m)]:.3f}" for m in METHODS[3:]
        )
        ax.text(0.5, 1.063, first_line, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.text(0.5, 1.018, second_line, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.set_xticks(x, CELL_TYPES)
        ax.tick_params(axis="x", rotation=22, pad=5)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.set_xlim(-0.62, len(CELL_TYPES) - 0.38)
        ax.set_ylim(0, 1.10)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#D9D9D9", alpha=0.55, linewidth=0.65, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Broad cell-type proportion")
    handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="none", label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.925))
    fig.suptitle("GSE120575 cell-type composition benchmark", fontsize=14.5, fontweight="bold", y=0.987)
    fig.text(
        0.5,
        0.050,
        "Error bars represent 95% sample/biopsy bootstrap confidence intervals. "
        "* WOT is target-derived; \N{DAGGER} CellRank is a Pre-only fate-weighted terminal-lineage proxy.",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#444444",
    )
    fig.text(
        0.5,
        0.023,
        "Leca-VC denotes the frozen Online Agent\N{EN DASH}PhysiCell/BioFVM V2.1 model without scGPT.",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#444444",
    )
    fig.subplots_adjust(left=0.062, right=0.99, top=0.775, bottom=0.225, wspace=0.13)
    fig.savefig(PREVIEW, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    outputs = (
        PREVIEW,
        PNG,
        PDF,
        SVG,
        OUT / "plotting_source.csv",
        OUT / "cellrank_sample_level_compositions.csv",
        OUT / "cellrank_terminal_lineage_mapping.csv",
        OUT / "rmse_summary.csv",
        OUT / "caption.txt",
        OUT / "audit.json",
    )
    for output in outputs:
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite existing output: {output}")

    mapping = derive_lineage_mapping()
    cellrank_samples = build_cellrank_samples(mapping)
    cellrank_means = bootstrap_group_means(cellrank_samples)
    v2_means = pd.read_csv(V2 / "unified_composition_group_means.csv")
    v2_means = v2_means[
        v2_means["response"].isin(RESPONSES)
        & v2_means["method"].isin(("Real_Post", "scGen", "WOT_weighted", "Our_Agent_PhysiCell_V2.1"))
        & v2_means["cell_type"].isin(CELL_TYPES)
    ].copy()
    source = pd.concat([v2_means, cellrank_means], ignore_index=True, sort=False)
    expected = len(RESPONSES) * len(METHODS) * len(CELL_TYPES)
    if len(source) != expected or source.duplicated(["response", "method", "cell_type"]).any():
        raise RuntimeError("Final plotting table is incomplete or duplicated")
    source["method_label"] = source["method"].map(METHOD_LABELS)
    source["method_order"] = source["method"].map({m: i for i, m in enumerate(METHODS)})
    source["cell_type_order"] = source["cell_type"].map({c: i for i, c in enumerate(CELL_TYPES)})
    source["response_order"] = source["response"].map({r: i for i, r in enumerate(RESPONSES)})
    source = source.sort_values(["response_order", "cell_type_order", "method_order"]).reset_index(drop=True)

    rmse = calculate_rmse(source)
    rmse_table = pd.DataFrame(
        [
            {"response": response, "method": method, "method_label": METHOD_LABELS[method], "RMSE": value}
            for (response, method), value in rmse.items()
        ]
    )
    mapping.to_csv(OUT / "cellrank_terminal_lineage_mapping.csv", index=False)
    cellrank_samples.to_csv(OUT / "cellrank_sample_level_compositions.csv", index=False)
    source.to_csv(OUT / "plotting_source.csv", index=False)
    rmse_table.to_csv(OUT / "rmse_summary.csv", index=False)
    render(source, rmse)

    caption = (
        "GSE120575 cell-type composition benchmark. Broad cell-type proportions are shown for observed "
        "post-treatment biopsies, scGen, target-derived WOT, a CellRank terminal-lineage proxy, and the "
        "frozen Leca-VC prediction under responder and non-responder scenarios. Error bars denote 95% "
        "sample/biopsy bootstrap confidence intervals. CellRank does not natively generate Post expression; "
        "its dagger-marked composition is a post-hoc Pre-only proxy obtained by mapping four terminal lineages "
        "to broad cell types with the frozen Pre-only classifier and summing sample-level fate probabilities."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    audit = {
        "status": "PASS_VISUALIZATION_WITH_DISCLOSED_CELLRANK_TERMINAL_LINEAGE_PROXY",
        "historical_v2_outputs_overwritten": False,
        "model_rerun": False,
        "existing_v2_bar_values_and_intervals_reused_unchanged": True,
        "cellrank_is_native_post_composition": False,
        "cellrank_proxy_semantics": "Pre-only fate-weighted terminal-lineage-to-broad-type mapping",
        "lineage_mapping_unanimous_terminal_cells": True,
        "lineage_count": 4,
        "terminal_cells_per_lineage": 30,
        "bootstrap_iterations": N_BOOTSTRAP,
        "bootstrap_unit": "source Pre sample for CellRank; frozen V2 units retained for other methods",
        "source_hashes": {
            "v2_group_means": sha256(V2 / "unified_composition_group_means.csv"),
            "v2_metrics": sha256(V2 / "unified_composition_metrics.csv"),
            "cellrank_fate_weights": sha256(FATE_WEIGHTS),
            "cellrank_h5ad": sha256(CELLRANK),
            "scgen_gene_panel_container": sha256(SCGEN),
            "stage0_annotations": sha256(ANNOTATION),
        },
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "pdf": str(PDF), "rmse": rmse_table.to_dict("records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
