#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gse230538-matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/gse230538-numba-cache")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import IncrementalPCA

from common import (
    CELL_LINES,
    CELL_LINE_COLORS,
    DEFAULT_OUT,
    HELDOUT_DAYS,
    METHODS,
    METHOD_COLORS,
    raw_files,
    read_dge_logcp10k,
    read_gene_panel,
    require,
    sample_info,
    simplex_projection_columns,
    write_json,
)


METHOD_LABELS = {
    "scGen": "scGen",
    "WOT": "WOT*",
    "CellRank": "CellRank†",
    "AgentVC": "AgentVC",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "figure.titlesize": 18,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_bundle(figure: plt.Figure, directory: Path, stem: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "preview": directory / f"{stem}_preview.png",
        "png_600dpi": directory / f"{stem}_600dpi.png",
        "pdf": directory / f"{stem}.pdf",
        "svg": directory / f"{stem}.svg",
    }
    figure.savefig(paths["preview"], dpi=180, bbox_inches="tight")
    figure.savefig(paths["png_600dpi"], dpi=600, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")
    return [str(path.name) for path in paths.values()]


def _load_arrays(out: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    path = out / "01_harmonized/harmonized_7576_expression.npz"
    require(path.exists(), "Run benchmark before figures")
    archive = np.load(path, allow_pickle=False)
    arrays = {key: archive[key] for key in archive.files if key != "genes"}
    genes = archive["genes"].astype(str).tolist()
    require(len(genes) == 7576, "Harmonized archive gene panel is invalid")
    return genes, arrays


def _scatter_figure(out: Path, metrics: pd.DataFrame, source: pd.DataFrame) -> list[str]:
    figure, axes = plt.subplots(2, 4, figsize=(18, 9.6), sharex="row", sharey="row")
    letters = iter("ABCDEFGH")
    for row_index, day in enumerate(HELDOUT_DAYS):
        day_source = source[source.day.eq(day)]
        maximum = float(max(day_source.observed.max(), day_source.predicted.max())) * 1.03
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            selected = day_source[day_source.method.eq(method)]
            metric = metrics[metrics.day.eq(day) & metrics.method.eq(method)].iloc[0]
            color = METHOD_COLORS[method]
            axis.scatter(
                selected.observed,
                selected.predicted,
                s=7,
                alpha=0.20,
                color=color,
                edgecolors="none",
                rasterized=True,
            )
            x = np.array([0.0, maximum])
            axis.plot(x, x, color="#7B8794", linestyle="--", linewidth=1.0)
            axis.plot(x, metric.intercept + metric.slope * x, color=color, linewidth=2.0)
            letter = next(letters)
            axis.set_title(f"{letter}. Day{day} — {METHOD_LABELS[method]}", loc="left")
            axis.text(
                0.03,
                0.96,
                "\n".join(
                    [
                        f"Pearson r = {metric.Pearson:.3f}",
                        f"CCC = {metric.CCC:.3f}",
                        f"RMSE = {metric.RMSE:.3f}",
                        f"slope = {metric.slope:.3f}",
                    ]
                ),
                transform=axis.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82},
            )
            axis.set_xlim(0, maximum)
            axis.set_ylim(0, maximum)
            if row_index == 1:
                axis.set_xlabel("Observed mean log1p(CP10K)")
            if column_index == 0:
                axis.set_ylabel("Predicted / proxy mean log1p(CP10K)")
    figure.suptitle("GSE230538 post-hoc expanded gene-expression benchmark", y=1.01)
    figure.text(
        0.5,
        0.972,
        "7,576-gene expanded union panel · four cell lines equally weighted · held-out Day4 and Day33",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    figure.text(
        0.5,
        0.002,
        "* WOT is target-derived. † CellRank is an untreated, target-blind fate-weighted proxy; it is not a native Post forecast. "
        "AgentVC is reconstructed read-only from the frozen three-seed V7 state trajectories. Method inputs are not information-equivalent.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    files = _save_bundle(figure, out / "03_figures/figure_1_scatter", "gse230538_expression_scatter_2x4")
    plt.close(figure)
    return files


def _diagnostic_figure(
    out: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    errors: pd.DataFrame,
    deciles: pd.DataFrame,
) -> list[str]:
    figure, axes = plt.subplots(2, 3, figsize=(17.5, 9.8))
    letters = iter("ABCDEF")
    palette = [METHOD_COLORS[method] for method in METHODS]
    for row_index, day in enumerate(HELDOUT_DAYS):
        boot = bootstrap[bootstrap.day.eq(day)]
        axis = axes[row_index, 0]
        sns.violinplot(
            data=boot,
            x="method",
            y="Pearson",
            order=METHODS,
            palette=palette,
            inner="box",
            cut=0,
            linewidth=0.8,
            ax=axis,
        )
        axis.set_xticklabels([METHOD_LABELS[method] for method in METHODS])
        axis.set_xlabel("")
        axis.set_ylabel("Bootstrap Pearson r")
        axis.set_title(f"{next(letters)}. Day{day}: bootstrap Pearson", loc="left")
        y_max = axis.get_ylim()[1]
        y_min = axis.get_ylim()[0]
        for index, method in enumerate(METHODS):
            metric = metrics[metrics.day.eq(day) & metrics.method.eq(method)].iloc[0]
            values = boot[boot.method.eq(method)].Pearson.to_numpy()
            low, high = np.quantile(values, [0.025, 0.975])
            axis.text(
                index,
                y_max - 0.025 * (y_max - y_min),
                f"r={metric.Pearson:.3f}\nCI {low:.3f}–{high:.3f}",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#555555",
            )

        error = errors[errors.day.eq(day)]
        axis = axes[row_index, 1]
        sns.violinplot(
            data=error,
            x="method",
            y="signed_error",
            order=METHODS,
            palette=palette,
            inner="box",
            cut=0,
            linewidth=0.8,
            ax=axis,
        )
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_xticklabels([METHOD_LABELS[method] for method in METHODS])
        axis.set_xlabel("")
        axis.set_ylabel("Predicted / proxy − Observed")
        axis.set_title(f"{next(letters)}. Day{day}: signed error", loc="left")
        y_max = axis.get_ylim()[1]
        y_min = axis.get_ylim()[0]
        for index, method in enumerate(METHODS):
            selected = error[error.method.eq(method)]
            axis.text(
                index,
                y_max - 0.025 * (y_max - y_min),
                f"median {selected.signed_error.median():.3f}\nMAE {selected.absolute_error.mean():.3f}",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#555555",
            )

        axis = axes[row_index, 2]
        observed = deciles[deciles.day.eq(day) & deciles.method.eq("scGen")]
        axis.plot(
            observed.decile,
            observed.observed_mean,
            color="#222222",
            marker="o",
            linewidth=2,
            label="Observed",
        )
        for method in METHODS:
            selected = deciles[deciles.day.eq(day) & deciles.method.eq(method)]
            axis.plot(
                selected.decile,
                selected.predicted_mean,
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=1.8,
                label=METHOD_LABELS[method],
            )
        axis.set_xticks(range(1, 11), [f"D{index}" for index in range(1, 11)])
        axis.set_xlabel("Observed-expression decile")
        axis.set_ylabel("Mean log1p(CP10K)")
        axis.set_title(f"{next(letters)}. Day{day}: decile calibration", loc="left")
        if row_index == 0:
            axis.legend(ncol=3, fontsize=7.5, frameon=False)
    figure.suptitle("GSE230538 expression diagnostics", y=1.01)
    figure.text(
        0.5,
        0.972,
        "1,000 bootstrap iterations · external methods pair-resample cell lines · AgentVC independently resamples cell lines and three frozen V7 seeds",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    figure.text(
        0.5,
        0.002,
        "* WOT is target-derived. † CellRank is a target-blind deterministic proxy. AgentVC reuses the frozen data-driven V7 main result, not PhysiCell-backed V9. No overall winner is constructed.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    files = _save_bundle(
        figure,
        out / "03_figures/figure_2_diagnostics",
        "gse230538_expression_diagnostics_2x3",
    )
    plt.close(figure)
    return files


def _reference_pca(
    out: Path,
    genes: list[str],
) -> tuple[IncrementalPCA, np.ndarray, pd.DataFrame]:
    model_path = out / "04_umap_models/observed_day0_day33_incremental_pca.joblib"
    pca = IncrementalPCA(n_components=50, batch_size=3000)
    reference_paths = [path for path in raw_files() if sample_info(path)[2] in (0, 33)]
    for path in reference_paths:
        _, matrix, _ = read_dge_logcp10k(path, genes)
        pca.partial_fit(matrix)
        del matrix
        gc.collect()
    joblib.dump(pca, model_path)

    coordinate_blocks = []
    metadata_blocks = []
    for path in reference_paths:
        cell_ids, matrix, _ = read_dge_logcp10k(path, genes)
        coordinate_blocks.append(pca.transform(matrix).astype(np.float32))
        _, cell_line, day = sample_info(path)
        metadata_blocks.append(
            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "cell_line": cell_line,
                    "day": day,
                    "source": "Observed",
                    "source_inherited_cell_line": False,
                }
            )
        )
        del matrix
        gc.collect()
    return pca, np.vstack(coordinate_blocks), pd.concat(metadata_blocks, ignore_index=True)


def _umap_sources(
    out: Path,
    genes: list[str],
    arrays: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = out / "04_umap_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pca, reference_pca, reference_meta = _reference_pca(out, genes)
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.25,
        metric="euclidean",
        random_state=230538,
        transform_seed=230538,
        n_jobs=1,
        verbose=False,
    )
    reference_umap = reducer.fit_transform(reference_pca).astype(np.float32)
    joblib.dump(reducer, model_dir / "observed_day0_day33_umap.joblib")
    reference_meta["UMAP1"] = reference_umap[:, 0]
    reference_meta["UMAP2"] = reference_umap[:, 1]

    method_targets = {
        "scGen": arrays["scgen"][1],
        "WOT": arrays["wot"][1],
        "CellRank": arrays["cellrank"][1],
        "AgentVC": arrays["agentvc_line_day33"],
    }
    proxy_blocks = []
    audit_rows = []
    day0_paths = {
        sample_info(path)[1]: path for path in raw_files() if sample_info(path)[2] == 0
    }
    for line_index, cell_line in enumerate(CELL_LINES):
        cell_ids, source, _ = read_dge_logcp10k(day0_paths[cell_line], genes)
        for method in METHODS:
            target = method_targets[method][line_index]
            proxy = simplex_projection_columns(source, target)
            maximum_error = float(np.max(np.abs(proxy.mean(axis=0) - target)))
            require(maximum_error <= 1e-10, f"{method}/{cell_line} proxy mean mismatch")
            proxy_pca = pca.transform(proxy)
            coordinates = reducer.transform(proxy_pca)
            proxy_blocks.append(
                pd.DataFrame(
                    {
                        "cell_id": [f"{method}:{cell_id}" for cell_id in cell_ids],
                        "cell_line": cell_line,
                        "day": 33,
                        "source": method,
                        "source_inherited_cell_line": True,
                        "UMAP1": coordinates[:, 0],
                        "UMAP2": coordinates[:, 1],
                    }
                )
            )
            audit_rows.append(
                {
                    "method": method,
                    "cell_line": cell_line,
                    "proxy_cell_count": len(proxy),
                    "gene_count": len(genes),
                    "maximum_gene_target_mean_error": maximum_error,
                    "source_cells": "Observed untreated cells from the same cell line",
                    "target": "method Day33 cell-line mean expression",
                    "native_single_cell_prediction": False,
                }
            )
            del proxy, proxy_pca, coordinates
            gc.collect()
        del source
        gc.collect()
    proxies = pd.concat(proxy_blocks, ignore_index=True)
    coordinates = pd.concat([reference_meta, proxies], ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    coordinates.to_csv(
        out / "04_umap_models/day33_cell_expanded_umap_coordinates.csv.gz",
        index=False,
        compression="gzip",
    )
    audit.to_csv(out / "04_umap_models/proxy_target_mean_audit.csv", index=False)
    write_json(
        out / "04_umap_models/frozen_embedding_audit.json",
        {
            "status": "PASS_OBSERVED_ONLY_FROZEN_UMAP",
            "fit_sources": ["Observed untreated", "Observed Day33"],
            "fit_cell_count": len(reference_meta),
            "proxy_fit_access": False,
            "proxy_operation": "transform_only",
            "pca_components": 50,
            "umap_random_state": 230538,
            "maximum_proxy_gene_target_mean_error": float(
                audit.maximum_gene_target_mean_error.max()
            ),
            "cell_colors": "source-inherited cell-line labels",
            "cell_type_prediction_claimed": False,
        },
    )
    return coordinates, audit


def _umap_figure(out: Path, coordinates: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    figure, axes = plt.subplots(2, 3, figsize=(17.8, 10.4))
    panels = ["Observed", "scGen", "WOT", "CellRank", "AgentVC"]
    titles = {
        "Observed": "A. Observed reference",
        "scGen": "B. scGen",
        "WOT": "C. WOT*",
        "CellRank": "D. CellRank†",
        "AgentVC": "E. AgentVC",
    }
    reference = coordinates[coordinates.source.eq("Observed")]
    limits = (
        (coordinates.UMAP1.min(), coordinates.UMAP1.max()),
        (coordinates.UMAP2.min(), coordinates.UMAP2.max()),
    )
    for panel, axis in zip(panels, axes.flat[:5]):
        axis.scatter(
            reference.UMAP1,
            reference.UMAP2,
            s=1.4,
            color="#D5DADF",
            alpha=0.34,
            edgecolors="none",
            rasterized=True,
        )
        if panel == "Observed":
            displayed = reference[reference.day.eq(33)]
        else:
            displayed = coordinates[coordinates.source.eq(panel)]
        for cell_line in CELL_LINES:
            selected = displayed[displayed.cell_line.eq(cell_line)]
            axis.scatter(
                selected.UMAP1,
                selected.UMAP2,
                s=3.0,
                color=CELL_LINE_COLORS[cell_line],
                alpha=0.70,
                edgecolors="none",
                rasterized=True,
            )
        axis.set_title(titles[panel], loc="left")
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        if panel != "Observed":
            axis.text(
                0.99,
                0.01,
                "deterministic proxy\ntransform only",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#666666",
            )

    old_axis = axes.flat[5]
    panel_spec = old_axis.get_subplotspec()
    old_axis.remove()
    metric_grid = panel_spec.subgridspec(1, 2, wspace=0.32)
    ccc_axis = figure.add_subplot(metric_grid[0, 0])
    rmse_axis = figure.add_subplot(metric_grid[0, 1], sharey=ccc_axis)
    positions = np.arange(len(METHODS))
    offsets = {4: -0.10, 33: 0.10}
    markers = {4: "o", 33: "^"}
    for day in HELDOUT_DAYS:
        selected = metrics[metrics.day.eq(day)].set_index("method").loc[list(METHODS)]
        for index, method in enumerate(METHODS):
            ccc_value = selected.loc[method, "CCC"]
            rmse_value = selected.loc[method, "RMSE"]
            ccc_axis.scatter(
                ccc_value,
                index + offsets[day],
                s=48,
                marker=markers[day],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.6,
            )
            ccc_axis.text(
                ccc_value + 0.008,
                index + offsets[day],
                f"{ccc_value:.3f}",
                va="center",
                fontsize=6.8,
            )
            rmse_axis.scatter(
                rmse_value,
                index + offsets[day],
                s=48,
                marker=markers[day],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.6,
            )
            rmse_axis.text(
                rmse_value + 0.008,
                index + offsets[day],
                f"{rmse_value:.3f}",
                va="center",
                fontsize=6.8,
            )
    ccc_axis.set_yticks(positions, [METHOD_LABELS[method] for method in METHODS])
    ccc_axis.invert_yaxis()
    ccc_axis.set_xlim(0.65, 1.03)
    rmse_axis.set_xlim(0.0, 0.36)
    ccc_axis.set_xlabel("CCC ↑")
    rmse_axis.set_xlabel("RMSE ↓")
    ccc_axis.set_title("F. Day4/Day33 fidelity", loc="left", pad=14)
    rmse_axis.tick_params(labelleft=False)
    ccc_axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color="#555555", label="Day4"),
            Line2D([0], [0], marker="^", linestyle="", color="#555555", label="Day33"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.06),
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=CELL_LINE_COLORS[line],
            label=line,
        )
        for line in CELL_LINES
    ]
    figure.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        frameon=False,
    )
    figure.suptitle("GSE230538 Day33 cell-expanded expression across melanoma cell lines", y=1.01)
    figure.text(
        0.5,
        0.972,
        "Observed Day0+Day33 fit the frozen PCA/UMAP · all method proxies transform only · 7,576 genes",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    figure.text(
        0.5,
        0.008,
        "Proxy points are deterministic sample-level expression expansions, not native single-cell predictions; colors are inherited source cell-line labels, not predicted cell types.\n"
        "* WOT is target-derived; † CellRank is target-blind. AgentVC is the frozen V7 state-prototype result, not V9. Method inputs are not information-equivalent.",
        ha="center",
        fontsize=8.3,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    files = _save_bundle(
        figure,
        out / "03_figures/figure_3_umap",
        "gse230538_day33_cell_expanded_umap",
    )
    plt.close(figure)
    return files


def build_figures(out: Path = DEFAULT_OUT, resume: bool = False) -> None:
    marker = out / "00_audit/BENCHMARK_COMPLETE.json"
    require(marker.exists(), "Run benchmark before figures")
    complete = out / "00_audit/FIGURES_COMPLETE.json"
    if resume and complete.exists():
        print("PASS_THREE_FIGURE_BUNDLES (resume)")
        return
    _style()
    genes, arrays = _load_arrays(out)
    metrics = pd.read_csv(out / "02_metrics/expression_metrics.csv")
    source = pd.read_csv(out / "01_harmonized/gene_level_plotting_source.csv.gz")
    bootstrap = pd.read_csv(out / "02_metrics/bootstrap_pearson_values.csv.gz")
    errors = pd.read_csv(out / "02_metrics/signed_error_by_gene.csv.gz")
    deciles = pd.read_csv(out / "02_metrics/decile_calibration.csv")
    figure_1 = _scatter_figure(out, metrics, source)
    figure_2 = _diagnostic_figure(out, metrics, bootstrap, errors, deciles)
    coordinates, proxy_audit = _umap_sources(out, genes, arrays)
    figure_3 = _umap_figure(out, coordinates, metrics)
    write_json(
        complete,
        {
            "status": "PASS_THREE_FIGURE_BUNDLES",
            "figure_1": figure_1,
            "figure_2": figure_2,
            "figure_3": figure_3,
            "formats": ["preview PNG", "600-DPI PNG", "PDF", "SVG"],
            "proxy_max_gene_target_mean_error": float(
                proxy_audit.maximum_gene_target_mean_error.max()
            ),
        },
    )
    print("PASS_THREE_FIGURE_BUNDLES")
