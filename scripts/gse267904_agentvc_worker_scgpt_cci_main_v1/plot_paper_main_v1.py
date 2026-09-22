#!/usr/bin/env python3
"""Create the single paper main figure from frozen GSE267904 CCI results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, SAMPLE_SEEDS, TYPES


METRICS = OUT / "07_metrics"
FIGURE_DIR = OUT / "08_figures/paper_main_v1"
SOURCE_DIR = FIGURE_DIR / "source_data"

EDGE_INPUT = METRICS / "edge_level_values.csv"
SEED_INPUT = METRICS / "seed_level_metrics.csv"
LOCAL_INPUT = METRICS / "cell_type_local_errors.csv"

METHODS = ["scgpt", "worker"]
METHOD_LABELS = {
    "scgpt": "Frozen scGPT +\nexternal head",
    "worker": "AgentVC worker-\nexpression proxy",
}
SOURCE_LABELS = {
    "observed": "Observed d21\nreference",
    "scgpt": "Frozen scGPT +\nexternal head",
    "worker": "AgentVC worker-\nexpression proxy",
}
COLORS = {
    "observed": "#333333",
    "scgpt": "#0072B2",
    "worker": "#D55E00",
}
TYPE_SHORT = {
    "alveolar_epithelial_AT1_AT2": "AT1/AT2",
    "activated_Krt8_ADI_epithelial": "Krt8/ADI",
    "airway_epithelial": "Airway",
    "macrophage": "Macrophage",
    "recruited_monocyte_macrophage": "Mono/Mac",
    "fibroblast_myofibroblast": "Fib/Myofib",
    "endothelial": "Endothelial",
    "dendritic": "Dendritic",
    "lymphoid": "Lymphoid",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"{name} lacks required columns: {missing}")


def prepare_network_source(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    edges = edges[
        edges["geometry_mode"].eq("affine_pixel")
        & edges["method"].isin(METHODS)
    ].copy()
    expected_rows = len(SAMPLE_SEEDS) * len(TYPES) ** 2 * len(METHODS)
    if len(edges) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} affine edge rows, found {len(edges)}"
        )
    if sorted(edges["sampling_seed"].unique()) != SAMPLE_SEEDS:
        raise RuntimeError("Edge table does not contain the ten frozen sampling seeds")

    observed_consistency = (
        edges.groupby(["sampling_seed", "sender", "receiver"])[
            "observed_normalized_weight"
        ]
        .agg(lambda x: float(np.max(x) - np.min(x)))
        .max()
    )
    if observed_consistency > 1e-12:
        raise RuntimeError("Observed edge values differ between comparison copies")

    observed = (
        edges.drop_duplicates(["sampling_seed", "sender", "receiver"])
        .loc[
            :,
            [
                "sampling_seed",
                "sender",
                "receiver",
                "observed_normalized_weight",
            ],
        ]
        .rename(columns={"observed_normalized_weight": "normalized_weight"})
    )
    observed["source"] = "observed"
    predicted = edges[
        ["sampling_seed", "method", "sender", "receiver", "method_normalized_weight"]
    ].rename(
        columns={"method": "source", "method_normalized_weight": "normalized_weight"}
    )
    long = pd.concat([observed, predicted], ignore_index=True)
    counts = long.groupby(["source", "sampling_seed"]).size()
    if not (counts == 81).all() or len(counts) != 30:
        raise RuntimeError("Each source/seed network must contain exactly 81 edges")
    sums = long.groupby(["source", "sampling_seed"])["normalized_weight"].sum()
    if not np.allclose(sums, 1.0, atol=1e-9):
        raise RuntimeError("At least one 81-edge network is not normalized")

    summary = (
        long.groupby(["source", "sender", "receiver"])["normalized_weight"]
        .agg(n="count", median="median", minimum="min", maximum="max")
        .reset_index()
    )
    full = pd.MultiIndex.from_product(
        [["observed", "scgpt", "worker"], TYPES, TYPES],
        names=["source", "sender", "receiver"],
    )
    if len(summary.set_index(["source", "sender", "receiver"]).reindex(full).dropna()) != 243:
        raise RuntimeError("Three complete 9x9 network summaries were not produced")
    summary.to_csv(SOURCE_DIR / "panels_A-C_network_heatmaps.csv", index=False)
    return summary, {
        "observed_copy_max_difference": float(observed_consistency),
        "networks": 30,
        "edges_per_network": 81,
    }


def prepare_score_source(
    metrics: pd.DataFrame, metric: str, filename: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = metrics[
        metrics["geometry_mode"].eq("affine_pixel")
        & metrics["method"].isin(METHODS)
        & metrics["metric"].eq(metric)
    ][["sampling_seed", "method", "value"]].copy()
    if len(table) != 20:
        raise RuntimeError(f"{metric}: expected 20 method/seed values, got {len(table)}")
    if sorted(table["sampling_seed"].unique()) != SAMPLE_SEEDS:
        raise RuntimeError(f"{metric}: frozen sampling seeds are incomplete")
    if not (table.groupby("method").size().reindex(METHODS) == 10).all():
        raise RuntimeError(f"{metric}: each method must have ten values")
    summary = (
        table.groupby("method")["value"]
        .agg(n="count", median="median", minimum="min", maximum="max")
        .reindex(METHODS)
        .reset_index()
    )
    table = table.merge(summary, on="method", validate="many_to_one")
    table.to_csv(SOURCE_DIR / filename, index=False)
    return table, summary


def prepare_local_source(local: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    table = local[
        local["geometry_mode"].eq("affine_pixel")
        & local["method"].isin(METHODS)
        & local["cell_type"].isin(TYPES)
    ][
        ["sampling_seed", "method", "cell_type", "local_mean_absolute_error"]
    ].copy()
    if len(table) != 180:
        raise RuntimeError(f"Expected 180 local-error rows, found {len(table)}")
    counts = table.groupby(["method", "cell_type"]).size()
    if len(counts) != 18 or not (counts == 10).all():
        raise RuntimeError("Each method/cell type must have ten local-error values")
    summary = (
        table.groupby(["method", "cell_type"])["local_mean_absolute_error"]
        .agg(n="count", median="median", minimum="min", maximum="max")
        .reset_index()
    )
    wide = summary.pivot(index="cell_type", columns="method", values="median")
    better = int((wide["worker"] < wide["scgpt"]).sum())
    table = table.merge(summary, on=["method", "cell_type"], validate="many_to_one")
    table.to_csv(SOURCE_DIR / "panel_F_cell_type_local_error.csv", index=False)
    summary.to_csv(
        SOURCE_DIR / "panel_F_cell_type_local_error_summary.csv", index=False
    )
    return table, summary, better


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )


def heatmap_matrix(networks: pd.DataFrame, source: str) -> np.ndarray:
    z = networks[networks["source"].eq(source)].pivot(
        index="sender", columns="receiver", values="median"
    )
    return z.reindex(index=TYPES, columns=TYPES).to_numpy(float)


def plot_score(
    ax: plt.Axes,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    title: str,
    letter: str,
    win_text: str,
) -> None:
    pivot = table.pivot(index="sampling_seed", columns="method", values="value")
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row["scgpt"], row["worker"]], color="#B6B6B6", lw=0.9, zorder=1)
        ax.scatter(0, row["scgpt"], color=COLORS["scgpt"], s=25, alpha=0.72, zorder=2)
        ax.scatter(1, row["worker"], color=COLORS["worker"], s=25, alpha=0.72, zorder=2)
    for x, method in enumerate(METHODS):
        median = float(summary.set_index("method").loc[method, "median"])
        ax.plot([x - 0.22, x + 0.22], [median, median], color="black", lw=2.4, zorder=3)
        ax.text(
            x,
            median + 0.025,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.30, 0.90)
    ax.set_xticks([0, 1], [METHOD_LABELS[x] for x in METHODS], fontsize=8)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=9)
    ax.text(
        0.98,
        0.03,
        win_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLORS["worker"],
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    add_panel_letter(ax, letter)


def plot_local(
    ax: plt.Axes,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    better: int,
) -> None:
    lookup = summary.set_index(["method", "cell_type"])
    ypos = np.arange(len(TYPES))[::-1]
    for y, cell_type in zip(ypos, TYPES):
        s = lookup.loc[("scgpt", cell_type)]
        w = lookup.loc[("worker", cell_type)]
        ax.plot(
            [s["median"], w["median"]],
            [y, y],
            color="#AFAFAF",
            lw=1.5,
            zorder=1,
        )
        for row, method in [(s, "scgpt"), (w, "worker")]:
            x = float(row["median"])
            ax.errorbar(
                x,
                y,
                xerr=np.asarray(
                    [[x - float(row["minimum"])], [float(row["maximum"]) - x]]
                ),
                fmt="o",
                ms=5.0,
                color=COLORS[method],
                ecolor=COLORS[method],
                elinewidth=1.0,
                capsize=2.0,
                alpha=0.9,
                zorder=2,
            )
    overall_lookup = overall.set_index("method")
    ax.set_yticks(ypos, [TYPE_SHORT[x] for x in TYPES], fontsize=8)
    ax.set_xlabel("Cell-type local MAE (lower is better)")
    ax.set_title(
        "Cell-type local error\n"
        f"AgentVC lower in {better}/9 types; overall "
        f"{overall_lookup.loc['scgpt', 'median']:.5f} → "
        f"{overall_lookup.loc['worker', 'median']:.5f}",
        fontsize=10.5,
        fontweight="bold",
        pad=9,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS[method],
            linestyle="none",
            markersize=6,
            label=METHOD_LABELS[method].replace("\n", " "),
        )
        for method in METHODS
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=7.5,
        borderaxespad=0,
    )
    ax.grid(axis="x", color="#E6E6E6", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    add_panel_letter(ax, "F")


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for path in [EDGE_INPUT, SEED_INPUT, LOCAL_INPUT]:
        if not path.exists():
            raise FileNotFoundError(path)

    edges = pd.read_csv(EDGE_INPUT)
    metrics = pd.read_csv(SEED_INPUT)
    local = pd.read_csv(LOCAL_INPUT)
    require_columns(
        edges,
        {
            "geometry_mode",
            "sampling_seed",
            "method",
            "sender",
            "receiver",
            "observed_normalized_weight",
            "method_normalized_weight",
        },
        "edge input",
    )
    require_columns(
        metrics,
        {"geometry_mode", "sampling_seed", "method", "metric", "value"},
        "seed metric input",
    )
    require_columns(
        local,
        {
            "geometry_mode",
            "sampling_seed",
            "method",
            "cell_type",
            "local_mean_absolute_error",
        },
        "local-error input",
    )

    networks, network_audit = prepare_network_source(edges)
    gnrs, gnrs_summary = prepare_score_source(
        metrics, "GNRS_secondary", "panel_D_GNRS.csv"
    )
    lbss, lbss_summary = prepare_score_source(
        metrics, "LBSS_secondary", "panel_E_LBSS.csv"
    )
    _, local_summary, better = prepare_local_source(local)
    overall_local = (
        metrics[
            metrics["geometry_mode"].eq("affine_pixel")
            & metrics["method"].isin(METHODS)
            & metrics["metric"].eq("cell_type_local_mean_absolute_error")
        ]
        .groupby("method")["value"]
        .agg(n="count", median="median", minimum="min", maximum="max")
        .reindex(METHODS)
        .reset_index()
    )
    if not (overall_local["n"] == 10).all():
        raise RuntimeError("Overall local MAE must have ten values per method")

    lbss_pivot = lbss.pivot(index="sampling_seed", columns="method", values="value")
    gnrs_pivot = gnrs.pivot(index="sampling_seed", columns="method", values="value")
    lbss_wins = int((lbss_pivot["worker"] > lbss_pivot["scgpt"]).sum())
    gnrs_wins = int((gnrs_pivot["worker"] > gnrs_pivot["scgpt"]).sum())
    if lbss_wins != 10 or better != 7:
        raise RuntimeError(
            f"Frozen expected visual summary changed: LBSS wins={lbss_wins}, local={better}"
        )

    matrices = {
        source: heatmap_matrix(networks, source)
        for source in ["observed", "scgpt", "worker"]
    }
    shared_vmin = 0.0
    shared_vmax = float(max(np.max(x) for x in matrices.values()))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(16.5, 10.0), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.90,
        bottom=0.105,
        top=0.88,
        wspace=0.36,
        hspace=0.50,
    )
    top_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    bottom_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]

    image = None
    for i, (ax, source, letter) in enumerate(
        zip(top_axes, ["observed", "scgpt", "worker"], ["A", "B", "C"])
    ):
        image = ax.imshow(
            matrices[source],
            cmap="YlGnBu",
            vmin=shared_vmin,
            vmax=shared_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xticks(
            np.arange(len(TYPES)),
            [TYPE_SHORT[x] for x in TYPES],
            rotation=55,
            ha="right",
            fontsize=7,
        )
        if i == 0:
            ax.set_yticks(
                np.arange(len(TYPES)), [TYPE_SHORT[x] for x in TYPES], fontsize=7
            )
            ax.set_ylabel("Sender cell type")
        else:
            ax.set_yticks(np.arange(len(TYPES)), [])
        ax.set_xlabel("Receiver cell type")
        ax.set_title(
            SOURCE_LABELS[source],
            fontsize=11,
            fontweight="bold",
            color=COLORS[source],
            pad=9,
        )
        for spine in ax.spines.values():
            spine.set_color(COLORS[source])
            spine.set_linewidth(1.5)
        add_panel_letter(ax, letter)

    cbar_ax = fig.add_axes([0.925, 0.585, 0.012, 0.26])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Median normalized CCI edge weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plot_score(
        bottom_axes[0],
        gnrs,
        gnrs_summary,
        "Global network reconstruction\nGNRS (secondary descriptive)",
        "D",
        f"AgentVC higher in {gnrs_wins}/10 mappings",
    )
    plot_score(
        bottom_axes[1],
        lbss,
        lbss_summary,
        "Local biological signaling\nLBSS (secondary descriptive)",
        "E",
        f"AgentVC higher in {lbss_wins}/10 mappings",
    )
    plot_local(bottom_axes[2], local_summary, overall_local, better)

    fig.suptitle(
        "AgentVC workers improve reconstruction of observed cell–cell communication",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.075,
        0.025,
        "Observed d21 is the reference and is not assigned a self-score. "
        "Points are the 10 frozen sampling/simulation mappings; black bars are medians "
        "and local-error whiskers show min–max stability ranges. No biological CI or p-value is implied.",
        fontsize=8.5,
        color="#444444",
        ha="left",
    )

    output_paths = {
        "pdf": FIGURE_DIR / "AgentVC_vs_external_scGPT_main_figure.pdf",
        "svg": FIGURE_DIR / "AgentVC_vs_external_scGPT_main_figure.svg",
        "png_600dpi": FIGURE_DIR / "AgentVC_vs_external_scGPT_main_figure_600dpi.png",
        "preview": FIGURE_DIR / "AgentVC_vs_external_scGPT_main_figure_preview.png",
    }
    fig.savefig(output_paths["pdf"], bbox_inches="tight")
    fig.savefig(output_paths["svg"], bbox_inches="tight")
    fig.savefig(output_paths["png_600dpi"], dpi=600, bbox_inches="tight")
    fig.savefig(output_paths["preview"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    caption = """Figure X. AgentVC workers improve reconstruction of observed cell-cell communication.
(A-C) Median normalized 9x9 cell-type communication networks for observed d21,
the frozen scGPT encoder plus externally trained conditional head, and the
AgentVC worker-expanded parent-expression proxy. All matrices share one color
scale. (D) Global network reconstruction score (GNRS). (E) Local biological
signaling score (LBSS). GNRS and LBSS are secondary descriptive scores, not a
combined overall score. (F) Cell-type local mean absolute error; lower values
indicate closer agreement with observed d21. AgentVC improves aggregate global
and local fidelity, while dendritic and lymphoid local errors remain limitations.
Ranges reflect sampling/simulation stability, not biological confidence intervals.
"""
    (FIGURE_DIR / "caption.txt").write_text(caption, encoding="utf-8")

    source_paths = sorted(SOURCE_DIR.glob("*.csv"))
    manifest = {
        "status": "PASS_SINGLE_PAPER_MAIN_FIGURE",
        "figure_semantics": {
            "observed": "Observed d21 reference; no self-score assigned",
            "baseline": "Frozen scGPT encoder + externally trained conditional head",
            "agentvc": "AgentVC worker-expanded parent-expression proxy",
            "inference": "sampling/simulation stability only; no biological CI or p-value",
        },
        "validation": {
            **network_audit,
            "sampling_seeds": SAMPLE_SEEDS,
            "sampling_seed_count": len(SAMPLE_SEEDS),
            "cell_types": len(TYPES),
            "shared_heatmap_vmin": shared_vmin,
            "shared_heatmap_vmax": shared_vmax,
            "GNRS_agentvc_higher_mappings": gnrs_wins,
            "LBSS_agentvc_higher_mappings": lbss_wins,
            "local_cell_types_agentvc_lower_error": better,
            "expected_local_limitations_retained": ["dendritic", "lymphoid"],
        },
        "input_sha256": {
            str(path): sha256(path) for path in [EDGE_INPUT, SEED_INPUT, LOCAL_INPUT]
        },
        "source_data_sha256": {str(path): sha256(path) for path in source_paths},
        "outputs_sha256": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
        "caption": str(FIGURE_DIR / "caption.txt"),
        "formal_results_recomputed": False,
        "physicell_or_commot_rerun": False,
    }
    write_json(FIGURE_DIR / "figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
