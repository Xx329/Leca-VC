#!/usr/bin/env python3
"""Publication-oriented GSE267904 Real–Virtual communication comparison V4.

This plotting-only revision reads frozen K=100 COMMOT outputs. It does not
rerun Agent, PhysiCell, COMMOT, or any upstream analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gse267904-communication-main-v4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
K100 = ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"
OUT = K100 / "figures/communication_concordance_main_v4"
V2_SCRIPT = (
    ROOT
    / "scripts/gse267904_spatial_agent_granularity/"
    "plot_communication_concordance_visual_v2.py"
)
OLD_OUTPUT_DIRS = [
    K100 / "figures/communication_concordance_visual_v2",
    K100 / "figures/communication_concordance_matrix_v3",
]

REAL_MATRIX_PATH = K100 / "real_commot/real_sender_receiver_matrix_by_stage.csv"
VIRTUAL_MATRIX_PATH = K100 / "virtual_commot/virtual_sender_receiver_matrix_by_stage.csv"
REAL_EDGES_PATH = K100 / "real_commot/real_commot_edges_long.csv"
VIRTUAL_EDGES_PATH = K100 / "virtual_commot/virtual_commot_edges_long.csv"
SUMMARY_PATH = K100 / "evaluation/commot_summary_metrics.csv"
INPUT_MANIFEST_PATH = K100 / "virtual_commot_input/virtual_commot_input_manifest.json"
FROZEN_V3_ROLE_PATH = (
    K100
    / "figures/communication_concordance_matrix_v3/"
    "five_pathway_dotplot_values.csv"
)

PSEUDOCOUNT = 1e-5
REAL_COLOR = "#4B5563"
VIRTUAL_COLOR = "#2F6F9F"
CROSS_EDGE_COLOR = "#78A9CF"
SELF_EDGE_COLOR = "#6E57A5"
TOP_EDGE_COLOR = "#D97706"
MATRIX_CMAP = "YlGnBu"
DIFF_CMAP = "RdBu_r"
ROLE_CMAP = "RdBu_r"


def import_v2():
    spec = importlib.util.spec_from_file_location("communication_visual_v2", V2_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V = import_v2()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def short_labels() -> list[str]:
    return [V.CELL_LABELS[cell] for cell in V.CELL_ORDER]


def paracrine_row_normalize(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    paracrine = matrix.copy().astype(float)
    for cell_type in V.CELL_ORDER:
        paracrine.loc[cell_type, cell_type] = 0.0
    row_totals = paracrine.sum(axis=1)
    normalized = paracrine.div(row_totals.replace(0, np.nan), axis=0).fillna(0.0)
    zero_rows = row_totals.index[row_totals.eq(0)].astype(str).tolist()
    return normalized, row_totals, zero_rows


def stage_total_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    total = float(matrix.to_numpy(float).sum())
    return matrix / total if total > 0 else matrix * 0.0


def build_paracrine_sources(
    real_long: pd.DataFrame,
    virtual_long: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, dict[str, list[str]]],
]:
    real_norm: dict[str, pd.DataFrame] = {}
    virtual_norm: dict[str, pd.DataFrame] = {}
    differences: dict[str, pd.DataFrame] = {}
    zero_rows: dict[str, dict[str, list[str]]] = {}
    rows: list[dict] = []

    for stage in V.STAGES:
        short = stage["short"]
        real_raw = V.matrix_for(real_long, stage["real"])
        virtual_raw = V.matrix_for(virtual_long, stage["virtual"])
        real, real_totals, real_zero = paracrine_row_normalize(real_raw)
        virtual, virtual_totals, virtual_zero = paracrine_row_normalize(virtual_raw)
        difference = virtual - real
        real_norm[short] = real
        virtual_norm[short] = virtual
        differences[short] = difference
        zero_rows[short] = {"Real": real_zero, "Virtual": virtual_zero}

        for sender in V.CELL_ORDER:
            for receiver in V.CELL_ORDER:
                is_autocrine = sender == receiver
                for source, raw, normalized, totals in [
                    ("Real", real_raw, real, real_totals),
                    ("Virtual", virtual_raw, virtual, virtual_totals),
                ]:
                    rows.append(
                        {
                            "stage": short,
                            "stage_semantic": stage["semantic"],
                            "real_stage": stage["real"],
                            "virtual_stage": stage["virtual"],
                            "source": source,
                            "sender": sender,
                            "receiver": receiver,
                            "is_autocrine_excluded": is_autocrine,
                            "raw_weight": float(raw.loc[sender, receiver]),
                            "paracrine_sender_total": float(totals.loc[sender]),
                            "paracrine_sender_row_probability": float(
                                normalized.loc[sender, receiver]
                            ),
                            "virtual_minus_real_probability": float(
                                difference.loc[sender, receiver]
                            ),
                        }
                    )
    return real_norm, virtual_norm, differences, pd.DataFrame(rows), zero_rows


def build_edge_source(
    real_long: pd.DataFrame,
    virtual_long: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    rows: list[dict] = []
    all_logs: list[float] = []
    for stage in V.STAGES:
        real = stage_total_normalize(V.matrix_for(real_long, stage["real"]))
        virtual = stage_total_normalize(V.matrix_for(virtual_long, stage["virtual"]))
        p = real.to_numpy(float).ravel()
        q = virtual.to_numpy(float).ravel()
        x = np.log10(p + PSEUDOCOUNT)
        y = np.log10(q + PSEUDOCOUNT)
        all_logs.extend(x.tolist())
        all_logs.extend(y.tolist())
        self_mask = np.eye(len(V.CELL_ORDER), dtype=bool).ravel()
        top_mask = np.zeros_like(p, dtype=bool)
        top_mask[np.argsort(p)[-10:]] = True

        frozen = summary.loc[
            summary.real_stage.eq(stage["real"])
            & summary.virtual_stage.eq(stage["virtual"])
        ]
        if len(frozen) != 1:
            raise RuntimeError(f"Missing frozen edge metrics for {stage['short']}")
        frozen = frozen.iloc[0]
        pearson = float(np.corrcoef(p, q)[0, 1])
        rank = float(spearmanr(p, q).statistic)
        if not np.isclose(pearson, float(frozen.sender_receiver_pearson), atol=1e-12):
            raise RuntimeError(f"Pearson mismatch for {stage['short']}")
        if not np.isclose(rank, float(frozen.sender_receiver_spearman), atol=1e-12):
            raise RuntimeError(f"Spearman mismatch for {stage['short']}")

        for idx, (sender, receiver) in enumerate(
            (s, r) for s in V.CELL_ORDER for r in V.CELL_ORDER
        ):
            rows.append(
                {
                    "stage": stage["short"],
                    "stage_semantic": stage["semantic"],
                    "real_stage": stage["real"],
                    "virtual_stage": stage["virtual"],
                    "sender": sender,
                    "receiver": receiver,
                    "real_stage_total_normalized_weight": float(p[idx]),
                    "virtual_stage_total_normalized_weight": float(q[idx]),
                    "real_log10_weight_plus_pseudocount": float(x[idx]),
                    "virtual_log10_weight_plus_pseudocount": float(y[idx]),
                    "is_self_edge": bool(self_mask[idx]),
                    "is_real_top10_edge": bool(top_mask[idx]),
                    "frozen_pearson": pearson,
                    "frozen_spearman": rank,
                    "frozen_top10_jaccard": float(frozen.top10_edge_jaccard),
                }
            )
    lo = float(np.floor(min(all_logs) * 10) / 10)
    hi = float(np.ceil(max(all_logs) * 10) / 10)
    return pd.DataFrame(rows), (lo, hi)


def build_information_flow(
    real_edges: pd.DataFrame,
    virtual_edges: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for stage in V.STAGES:
        stage_rows: list[dict] = []
        for source, edges, stage_name in [
            ("Real", real_edges, stage["real"]),
            ("Virtual", virtual_edges, stage["virtual"]),
        ]:
            totals = {}
            for family, exact_path in V.FAMILY_PATHS.items():
                totals[family] = float(
                    edges.loc[
                        edges.stage.astype(str).eq(stage_name)
                        & edges.pathway.astype(str).eq(exact_path),
                        "weight",
                    ].sum()
                )
            denominator = float(sum(totals.values()))
            if denominator <= 0:
                raise RuntimeError(f"No aggregate-family information flow: {stage_name}")
            for family, total in totals.items():
                stage_rows.append(
                    {
                        "stage": stage["short"],
                        "stage_semantic": stage["semantic"],
                        "real_stage": stage["real"],
                        "virtual_stage": stage["virtual"],
                        "source": source,
                        "family": family,
                        "family_exact_commot_path": V.FAMILY_PATHS[family],
                        "aggregate_family_total_weight": total,
                        "five_family_total_weight": denominator,
                        "five_family_information_flow_share": total / denominator,
                    }
                )
        rows.extend(stage_rows)
    return pd.DataFrame(rows)


def validate_role_metrics(
    role: pd.DataFrame,
    frozen_path: Path | None,
) -> dict:
    expected = {
        (row.stage, row.family, row.mode): float(row.spearman)
        for row in role.itertuples()
    }
    result = {
        "formula": (
            "Spearman rank correlation between Real and Virtual normalized "
            "cell-type sender/receiver distributions for each exact aggregate family"
        ),
        "recomputed_from_frozen_edge_tables": True,
        "matched_preexisting_v3_values": None,
        "preexisting_v3_path": str(frozen_path) if frozen_path else None,
        "preexisting_v3_sha256": sha256(frozen_path) if frozen_path else None,
    }
    if frozen_path is not None:
        frozen = pd.read_csv(frozen_path)
        frozen_values = {
            (row.stage, row.family, row.mode): float(row.spearman)
            for row in frozen.itertuples()
        }
        if set(expected) != set(frozen_values):
            raise RuntimeError("V4/V3 sender-receiver role keys differ")
        if not all(np.isclose(expected[key], frozen_values[key], atol=1e-12) for key in expected):
            raise RuntimeError("V4 sender-receiver role metrics differ from frozen V3")
        result["matched_preexisting_v3_values"] = True
    return result


def masked_for_plot(values: pd.DataFrame) -> np.ma.MaskedArray:
    array = values.to_numpy(float)
    return np.ma.masked_where(np.eye(array.shape[0], dtype=bool), array)


def draw_matrix(
    ax,
    values: pd.DataFrame,
    title: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    show_y: bool,
    show_x: bool,
):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#ECEFF3")
    image = ax.imshow(
        masked_for_plot(values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        rasterized=True,
    )
    labels = short_labels()
    if show_x:
        ax.set_xticks(range(len(labels)), labels, rotation=58, ha="right", fontsize=6.5)
        ax.set_xlabel("Receiver", fontsize=7.4)
    else:
        ax.set_xticks([])
    if show_y:
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_ylabel("Sender", fontsize=7.4)
    else:
        ax.set_yticks([])
    ax.set_title(title, fontsize=9.2, fontweight="semibold", pad=5)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.82)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def draw_scatter(
    ax,
    edge_source: pd.DataFrame,
    stage: str,
    limits: tuple[float, float],
):
    part = edge_source.loc[edge_source.stage.eq(stage)].copy()
    x = part.real_log10_weight_plus_pseudocount.to_numpy(float)
    y = part.virtual_log10_weight_plus_pseudocount.to_numpy(float)
    self_mask = part.is_self_edge.to_numpy(bool)
    top_mask = part.is_real_top10_edge.to_numpy(bool)
    ax.scatter(
        x[~self_mask],
        y[~self_mask],
        s=20,
        color=CROSS_EDGE_COLOR,
        alpha=0.57,
        edgecolor="none",
        label="Cross-type",
    )
    ax.scatter(
        x[self_mask],
        y[self_mask],
        s=46,
        marker="D",
        color=SELF_EDGE_COLOR,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.45,
        label="Self-edge",
        zorder=4,
    )
    ax.scatter(
        x[top_mask],
        y[top_mask],
        s=66,
        facecolor="none",
        edgecolor=TOP_EDGE_COLOR,
        linewidth=1.25,
        label="Real top-10",
        zorder=5,
    )
    lo, hi = limits
    ax.plot([lo, hi], [lo, hi], "--", color="#7A838C", lw=0.95)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E5E7EB", lw=0.62)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.7)
    ax.set_xlabel("Real normalized edge weight (log10)", fontsize=7.0)
    ax.set_ylabel("Virtual normalized edge weight (log10)", fontsize=7.0)
    ax.set_title(stage, fontsize=9.2, fontweight="semibold", pad=5)
    metrics = part.iloc[0]
    ax.text(
        0.04,
        0.96,
        (
            f"Pearson {metrics.frozen_pearson:.2f}\n"
            f"Spearman {metrics.frozen_spearman:.2f}\n"
            f"Top-10 Jaccard {metrics.frozen_top10_jaccard:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox=dict(
            boxstyle="round,pad=0.27",
            facecolor="white",
            edgecolor="#D1D5DB",
            alpha=0.92,
        ),
    )


def draw_dumbbell(
    ax,
    flow: pd.DataFrame,
    stage: str,
    x_max: float,
):
    part = flow.loc[flow.stage.eq(stage)]
    families = list(V.FAMILY_PATHS)
    y = np.arange(len(families))
    real = (
        part.loc[part.source.eq("Real")]
        .set_index("family")
        .reindex(families)
        .five_family_information_flow_share.to_numpy(float)
    )
    virtual = (
        part.loc[part.source.eq("Virtual")]
        .set_index("family")
        .reindex(families)
        .five_family_information_flow_share.to_numpy(float)
    )
    for yi, rv, vv in zip(y, real, virtual):
        ax.plot([rv, vv], [yi, yi], color="#B8C0C8", lw=2.0, zorder=1)
    ax.scatter(real, y, s=48, color=REAL_COLOR, label="Real", zorder=3)
    ax.scatter(
        virtual,
        y,
        s=55,
        facecolor="white",
        edgecolor=VIRTUAL_COLOR,
        linewidth=1.8,
        label="Virtual",
        zorder=4,
    )
    for yi, rv, vv in zip(y, real, virtual):
        ax.text(rv, yi - 0.23, f"{rv:.2f}", ha="center", va="bottom", fontsize=6.1, color=REAL_COLOR)
        ax.text(vv, yi + 0.25, f"{vv:.2f}", ha="center", va="top", fontsize=6.1, color=VIRTUAL_COLOR)
    ax.set_yticks(y, families, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Share of five-family information flow", fontsize=7.0)
    ax.set_title(stage, fontsize=9.2, fontweight="semibold", pad=5)
    ax.grid(axis="x", color="#E5E7EB", lw=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.tick_params(axis="y", length=0)


def draw_role_heatmap(ax, role: pd.DataFrame):
    rows = [
        ("Early", "sender", "Early sender"),
        ("Early", "receiver", "Early receiver"),
        ("Late", "sender", "Late sender"),
        ("Late", "receiver", "Late receiver"),
    ]
    families = list(V.FAMILY_PATHS)
    values = np.empty((len(rows), len(families)), dtype=float)
    for yi, (stage, mode, _label) in enumerate(rows):
        for xi, family in enumerate(families):
            match = role.loc[
                role.stage.eq(stage)
                & role["mode"].eq(mode)
                & role.family.eq(family)
            ]
            if len(match) != 1:
                raise RuntimeError(f"Missing role metric: {stage}/{mode}/{family}")
            values[yi, xi] = float(match.iloc[0].spearman)
    image = ax.imshow(
        values,
        cmap=ROLE_CMAP,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xticks(range(len(families)), families, fontsize=7.6)
    ax.set_yticks(range(len(rows)), [row[2] for row in rows], fontsize=7.4)
    ax.tick_params(length=0)
    for yi in range(values.shape[0]):
        for xi in range(values.shape[1]):
            value = values[yi, xi]
            color = "white" if abs(value) >= 0.55 else "#1F2933"
            ax.text(
                xi,
                yi,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.3,
                fontweight="semibold",
                color=color,
            )
    ax.set_xticks(np.arange(-0.5, len(families), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def make_figure(
    real_norm: dict[str, pd.DataFrame],
    virtual_norm: dict[str, pd.DataFrame],
    differences: dict[str, pd.DataFrame],
    edge_source: pd.DataFrame,
    scatter_limits: tuple[float, float],
    flow: pd.DataFrame,
    role: pd.DataFrame,
    difference_limit: float,
) -> tuple[Path, Path, Path]:
    fig = plt.figure(figsize=(18.6, 10.8), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 0.70],
        width_ratios=[1.0, 1.0, 1.0, 1.13],
        left=0.062,
        right=0.95,
        top=0.86,
        bottom=0.11,
        hspace=0.55,
        wspace=0.35,
    )
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(4)]
    middle_axes = [fig.add_subplot(grid[1, idx]) for idx in range(4)]
    early_flow_ax = fig.add_subplot(grid[2, 0])
    late_flow_ax = fig.add_subplot(grid[2, 1])
    role_ax = fig.add_subplot(grid[2, 2:])

    prob_image = draw_matrix(
        top_axes[0], real_norm["Early"], "Real", MATRIX_CMAP, 0, 1, True, False
    )
    draw_matrix(
        top_axes[1], virtual_norm["Early"], "Virtual", MATRIX_CMAP, 0, 1, False, False
    )
    diff_image = draw_matrix(
        top_axes[2],
        differences["Early"],
        "Virtual − Real",
        DIFF_CMAP,
        -difference_limit,
        difference_limit,
        False,
        False,
    )
    draw_scatter(top_axes[3], edge_source, "Early", scatter_limits)

    draw_matrix(
        middle_axes[0], real_norm["Late"], "Real", MATRIX_CMAP, 0, 1, True, True
    )
    draw_matrix(
        middle_axes[1], virtual_norm["Late"], "Virtual", MATRIX_CMAP, 0, 1, False, True
    )
    draw_matrix(
        middle_axes[2],
        differences["Late"],
        "Virtual − Real",
        DIFF_CMAP,
        -difference_limit,
        difference_limit,
        False,
        True,
    )
    draw_scatter(middle_axes[3], edge_source, "Late", scatter_limits)

    flow_max = float(flow.five_family_information_flow_share.max())
    x_max = max(0.4, np.ceil((flow_max + 0.03) * 20) / 20)
    draw_dumbbell(early_flow_ax, flow, "Early", x_max)
    draw_dumbbell(late_flow_ax, flow, "Late", x_max)
    role_image = draw_role_heatmap(role_ax, role)

    fig.text(
        0.062,
        0.875,
        "A. Early paracrine communication",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.062,
        0.555,
        "B. Late paracrine communication",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.783,
        0.875,
        "C. Edge-weight concordance",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.062,
        0.267,
        "D. Pathway information-flow recovery",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.535,
        0.267,
        "E. Sender/receiver role agreement",
        fontsize=10.5,
        fontweight="bold",
    )

    prob_cax = fig.add_axes([0.285, 0.905, 0.105, 0.011])
    prob_bar = fig.colorbar(prob_image, cax=prob_cax, orientation="horizontal")
    prob_bar.set_label("Paracrine receiver probability", fontsize=6.8, labelpad=2)
    prob_bar.ax.xaxis.set_label_position("top")
    prob_bar.ax.tick_params(labelsize=6.0, length=2)

    diff_cax = fig.add_axes([0.486, 0.905, 0.105, 0.011])
    diff_bar = fig.colorbar(diff_image, cax=diff_cax, orientation="horizontal")
    diff_bar.set_label("Virtual − Real", fontsize=6.8, labelpad=2)
    diff_bar.ax.xaxis.set_label_position("top")
    diff_bar.ax.tick_params(labelsize=6.0, length=2)

    role_cax = role_ax.inset_axes([1.015, 0.04, 0.016, 0.92])
    role_bar = fig.colorbar(role_image, cax=role_cax)
    role_bar.set_label("Spearman", fontsize=6.7)
    role_bar.ax.tick_params(labelsize=6.0, length=2)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=REAL_COLOR, label="Real"),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=VIRTUAL_COLOR,
            markeredgewidth=1.5,
            label="Virtual",
        ),
    ]
    early_flow_ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=6.5,
        handletextpad=0.3,
        borderpad=0.2,
    )

    fig.suptitle(
        "GSE267904 observed–virtual cell–cell communication concordance",
        fontsize=17.0,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.952,
        "Paracrine communication recovery across early reconstruction and held-out late comparison",
        ha="center",
        fontsize=9.1,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.028,
        "Early: d7 calibration/reconstruction; Late: held-out d21 comparison. "
        "Matrices exclude autocrine edges and normalize non-self receivers within each sender. "
        "Single-tissue descriptive evaluation only; no biological-sample CI.",
        ha="center",
        fontsize=7.1,
        color="#374151",
    )

    preview = OUT / "GSE267904_communication_concordance_main_v4_preview.png"
    png = OUT / "GSE267904_communication_concordance_main_v4_600dpi.png"
    pdf = OUT / "GSE267904_communication_concordance_main_v4_vector.pdf"
    fig.savefig(preview, dpi=170, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return preview, png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finalize-delete-old",
        action="store_true",
        help="After all hard gates pass, delete the two explicitly authorized old figure directories.",
    )
    parser.add_argument(
        "--visual-inspection-pass",
        action="store_true",
        help="Record that the generated preview was visually inspected for clipping and readability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        REAL_MATRIX_PATH,
        VIRTUAL_MATRIX_PATH,
        REAL_EDGES_PATH,
        VIRTUAL_EDGES_PATH,
        SUMMARY_PATH,
        INPUT_MANIFEST_PATH,
        V2_SCRIPT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    frozen_v3_path = FROZEN_V3_ROLE_PATH if FROZEN_V3_ROLE_PATH.exists() else None
    real_matrix_long = pd.read_csv(REAL_MATRIX_PATH)
    virtual_matrix_long = pd.read_csv(VIRTUAL_MATRIX_PATH)
    real_edges = pd.read_csv(REAL_EDGES_PATH)
    virtual_edges = pd.read_csv(VIRTUAL_EDGES_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    input_manifest = json.loads(INPUT_MANIFEST_PATH.read_text())

    real_norm, virtual_norm, differences, paracrine_source, zero_rows = (
        build_paracrine_sources(real_matrix_long, virtual_matrix_long)
    )
    edge_source, scatter_limits = build_edge_source(
        real_matrix_long, virtual_matrix_long, summary
    )
    flow = build_information_flow(real_edges, virtual_edges)
    role = V.family_metrics(real_edges, virtual_edges)
    role_validation = validate_role_metrics(role, frozen_v3_path)
    lr_inventory = V.lr_inventory(real_edges, virtual_edges)
    difference_limit = max(
        float(np.abs(differences[stage["short"]].to_numpy(float)).max())
        for stage in V.STAGES
    )

    paracrine_csv = OUT / "paracrine_matrix_plotting_source.csv"
    edge_csv = OUT / "edge_concordance_plotting_source.csv"
    flow_csv = OUT / "pathway_information_flow_values.csv"
    role_csv = OUT / "sender_receiver_role_agreement.csv"
    lr_csv = OUT / "complete_lr_pair_inventory_and_recovery.csv"
    paracrine_source.to_csv(paracrine_csv, index=False)
    edge_source.to_csv(edge_csv, index=False)
    flow.to_csv(flow_csv, index=False)
    role.to_csv(role_csv, index=False)
    lr_inventory.to_csv(lr_csv, index=False)

    preview, png, pdf = make_figure(
        real_norm,
        virtual_norm,
        differences,
        edge_source,
        scatter_limits,
        flow,
        role,
        difference_limit,
    )
    output_script = OUT / "plot_communication_concordance_main_v4.py"
    shutil.copy2(Path(__file__).resolve(), output_script)

    row_checks: dict[str, dict[str, dict[str, float | int | list[str]]]] = {}
    difference_checks: dict[str, bool] = {}
    for stage in V.STAGES:
        short = stage["short"]
        row_checks[short] = {}
        for source, matrix in [
            ("Real", real_norm[short]),
            ("Virtual", virtual_norm[short]),
        ]:
            sums = matrix.sum(axis=1)
            nonzero = ~sums.eq(0)
            if not np.allclose(sums.loc[nonzero], 1.0, atol=1e-12):
                raise RuntimeError(f"Paracrine row normalization failed: {short}/{source}")
            row_checks[short][source] = {
                "nonzero_sender_rows": int(nonzero.sum()),
                "zero_sender_rows": zero_rows[short][source],
                "nonzero_row_sum_min": float(sums.loc[nonzero].min()),
                "nonzero_row_sum_max": float(sums.loc[nonzero].max()),
            }
        expected = virtual_norm[short] - real_norm[short]
        difference_checks[short] = bool(
            np.allclose(expected.to_numpy(), differences[short].to_numpy(), atol=0, rtol=0)
        )
        if not difference_checks[short]:
            raise RuntimeError(f"Difference matrix mismatch: {short}")

    flow_sums = (
        flow.groupby(["stage", "source"]).five_family_information_flow_share.sum()
    )
    if not np.allclose(flow_sums.to_numpy(float), 1.0, atol=1e-12):
        raise RuntimeError("Five-family information-flow shares do not sum to one")
    if len(role) != 20:
        raise RuntimeError(f"Expected 20 sender/receiver role metrics, found {len(role)}")

    required_outputs = [
        preview,
        png,
        pdf,
        output_script,
        paracrine_csv,
        edge_csv,
        flow_csv,
        role_csv,
        lr_csv,
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required_outputs):
        raise RuntimeError("One or more V4 outputs are missing or empty")

    deleted_old_dirs: list[str] = []
    absent_old_dirs: list[str] = []
    if args.finalize_delete_old:
        if not args.visual_inspection_pass:
            raise RuntimeError(
                "--finalize-delete-old requires --visual-inspection-pass"
            )
        for old_dir in OLD_OUTPUT_DIRS:
            if old_dir.exists():
                shutil.rmtree(old_dir)
                deleted_old_dirs.append(str(old_dir))
            else:
                absent_old_dirs.append(str(old_dir))

    source_files = [
        {"path": str(path), "sha256": sha256(path)}
        for path in required
        if path.exists()
    ]
    if frozen_v3_path is not None:
        source_files.append(
            {
                "path": str(frozen_v3_path),
                "sha256": role_validation["preexisting_v3_sha256"],
                "status_after_finalize": (
                    "deleted with authorized old V3 output directory"
                    if args.finalize_delete_old
                    else "retained pending finalization"
                ),
            }
        )

    audit = {
        "status": "PASS_COMMUNICATION_CONCORDANCE_MAIN_V4",
        "models_rerun": False,
        "agent_rerun": False,
        "physicell_rerun": False,
        "commot_rerun": False,
        "upstream_values_modified": False,
        "supplementary_spatial_figure_generated": False,
        "layout": {
            "orientation": "landscape",
            "rows": 3,
            "columns": 4,
            "panels": {
                "A": "Early paracrine Real, Virtual, Virtual-minus-Real matrices",
                "B": "Late paracrine Real, Virtual, Virtual-minus-Real matrices",
                "C": "Early and Late all-edge concordance",
                "D": "Early and Late five-family information-flow dumbbells",
                "E": "Sender/receiver Spearman role-agreement heatmap",
            },
            "cartoon_workflow_present": False,
            "circular_network_present": False,
            "bubble_size_encoding_present": False,
            "agent_count_displayed": False,
        },
        "paracrine_matrix": {
            "shape": [10, 10],
            "cell_type_order": V.CELL_ORDER,
            "formula": (
                "set sender==receiver weights to zero; divide each nonzero sender "
                "row by its remaining non-self receiver total"
            ),
            "zero_sender_rows_remain_zero": True,
            "autocrine_diagonal_masked_in_plot": True,
            "real_virtual_shared_range": [0.0, 1.0],
            "difference_definition": "Virtual minus Real paracrine row probability",
            "difference_shared_symmetric_range": [
                -difference_limit,
                difference_limit,
            ],
            "row_checks": row_checks,
            "difference_exact_checks": difference_checks,
        },
        "edge_concordance": {
            "normalization": "each complete stage matrix divided by its total weight",
            "transform": f"log10(normalized weight + {PSEUDOCOUNT})",
            "shared_axis_limits": list(scatter_limits),
            "frozen_metrics_revalidated": True,
            "early_pearson": float(
                edge_source.loc[edge_source.stage.eq("Early"), "frozen_pearson"].iloc[0]
            ),
            "late_pearson": float(
                edge_source.loc[edge_source.stage.eq("Late"), "frozen_pearson"].iloc[0]
            ),
            "early_top10_jaccard": float(
                edge_source.loc[
                    edge_source.stage.eq("Early"), "frozen_top10_jaccard"
                ].iloc[0]
            ),
            "late_top10_jaccard": float(
                edge_source.loc[
                    edge_source.stage.eq("Late"), "frozen_top10_jaccard"
                ].iloc[0]
            ),
        },
        "pathway_information_flow": {
            "families": list(V.FAMILY_PATHS),
            "exact_aggregate_paths": V.FAMILY_PATHS,
            "definition": (
                "sum the exact aggregate-family COMMOT weight over all sender-receiver "
                "pairs, then divide by the sum across the five frozen families within "
                "the same stage and source"
            ),
            "lr_pair_rows_not_reaggregated": True,
            "group_share_sums": {
                f"{stage}/{source}": float(value)
                for (stage, source), value in flow_sums.items()
            },
        },
        "sender_receiver_role_agreement": {
            **role_validation,
            "item_count": int(len(role)),
            "heatmap_value": "Spearman",
            "heatmap_range": [-1.0, 1.0],
            "js_similarity_retained_in_csv_only": True,
        },
        "stage_semantics": {
            "early": {
                "real": "d7_bleo",
                "virtual": "virtual_early_bleo",
                "interpretation": "calibration/reconstruction",
                "d7_cross_type_calibration_prior_enabled": bool(
                    input_manifest.get("d7_cross_type_calibration_prior_enabled", False)
                ),
            },
            "late": {
                "real": "d21_bleo",
                "virtual": "virtual_late_bleo",
                "interpretation": "held-out",
                "d21_real_commot_used_for_generation": bool(
                    input_manifest.get("d21_real_commot_used_for_lr_program", False)
                ),
            },
        },
        "source_files": source_files,
        "outputs": {
            "preview_png": str(preview),
            "png_600dpi": str(png),
            "vector_pdf": str(pdf),
            "script": str(output_script),
            "paracrine_source": str(paracrine_csv),
            "edge_source": str(edge_csv),
            "information_flow_source": str(flow_csv),
            "role_agreement_source": str(role_csv),
            "lr_inventory": str(lr_csv),
        },
        "visual_inspection": {
            "status": "PASS" if args.visual_inspection_pass else "PENDING",
            "no_label_clipping": bool(args.visual_inspection_pass),
            "readable_at_preview_scale": bool(args.visual_inspection_pass),
        },
        "old_output_cleanup": {
            "authorized_by_user": True,
            "finalization_requested": bool(args.finalize_delete_old),
            "deleted_directories": deleted_old_dirs,
            "already_absent_directories": absent_old_dirs,
            "other_historical_outputs_deleted": False,
        },
    }
    audit_path = OUT / "communication_concordance_main_v4_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    readme = OUT / "README.md"
    cleanup_text = (
        "The two explicitly authorized V2/V3 output directories were deleted "
        "after V4 hard gates and visual inspection passed."
        if args.finalize_delete_old
        else "The old V2/V3 output directories remain pending final visual inspection."
    )
    readme.write_text(
        f"""# GSE267904 communication concordance main V4

This publication-oriented figure was generated only from existing frozen
COMMOT outputs. Agent, PhysiCell, COMMOT, and upstream analyses were not rerun.

Panels A/B use paracrine-only sender-row normalization: autocrine diagonal
weights are excluded and each nonzero sender row is normalized across its
remaining receivers. The Late Virtual AT1/AT2 sender has no nonzero paracrine
edge and therefore remains a zero row rather than receiving an imputed value.

Panel C uses the frozen stage-total normalized all-edge comparison. Panel D
shows each exact aggregate family's share of the total across TGFβ, CCL, CXCL,
PDGF, and VEGF within the same stage/source. Panel E reports frozen
sender/receiver Spearman role agreement; JS similarity is retained only in
`{role_csv}`.

Early is a d7 calibration/reconstruction comparison. Late is the held-out d21
comparison. This is a single-tissue descriptive evaluation; no
biological-sample confidence interval is claimed.

{cleanup_text}

- Preview: `{preview}`
- 600 dpi PNG: `{png}`
- Vector PDF: `{pdf}`
- Audit: `{audit_path}`
"""
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
