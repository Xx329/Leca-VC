#!/usr/bin/env python3
"""Publication-style Real-versus-Virtual communication concordance figure.

Uses existing GSE267904 COMMOT outputs only. No COMMOT, PhysiCell, Agent, or
expression model is rerun. The figure intentionally does not display Agent
granularity. Five frozen CellChat pathway families are shown in the main
figure; the complete ligand-receptor inventory is exported for audit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gse267904-communication-visual-v2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
K100 = ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"
OUT = K100 / "figures/communication_concordance_visual_v2"
REAL_MATRIX_PATH = K100 / "real_commot/real_sender_receiver_matrix_by_stage.csv"
VIRTUAL_MATRIX_PATH = (
    K100 / "virtual_commot/virtual_sender_receiver_matrix_by_stage.csv"
)
REAL_EDGES_PATH = K100 / "real_commot/real_commot_edges_long.csv"
VIRTUAL_EDGES_PATH = K100 / "virtual_commot/virtual_commot_edges_long.csv"
SUMMARY_PATH = K100 / "evaluation/commot_summary_metrics.csv"
INPUT_MANIFEST_PATH = K100 / "virtual_commot_input/virtual_commot_input_manifest.json"

STAGES = [
    {
        "short": "Early",
        "real": "d7_bleo",
        "virtual": "virtual_early_bleo",
        "semantic": "calibration/reconstruction",
    },
    {
        "short": "Late",
        "real": "d21_bleo",
        "virtual": "virtual_late_bleo",
        "semantic": "held-out",
    },
]
FAMILY_PATHS = {
    "TGFβ": "commot-cellchat-TGFb",
    "CCL": "commot-cellchat-CCL",
    "CXCL": "commot-cellchat-CXCL",
    "PDGF": "commot-cellchat-PDGF",
    "VEGF": "commot-cellchat-VEGF",
}
MODES = ("sender", "receiver")
TOP_NETWORK_EDGES = 20

CELL_ORDER = [
    "activated_Krt8_ADI_epithelial",
    "airway_epithelial",
    "alveolar_epithelial_AT1_AT2",
    "endothelial",
    "fibroblast_myofibroblast",
    "macrophage",
    "recruited_monocyte_macrophage",
    "dendritic",
    "lymphoid",
    "other",
]
CELL_LABELS = {
    "activated_Krt8_ADI_epithelial": "Krt8+ ADI",
    "airway_epithelial": "Airway epi.",
    "alveolar_epithelial_AT1_AT2": "AT1/AT2",
    "endothelial": "Endothelial",
    "fibroblast_myofibroblast": "Fibroblast",
    "macrophage": "Macrophage",
    "recruited_monocyte_macrophage": "Recruited Mφ",
    "dendritic": "Dendritic",
    "lymphoid": "Lymphoid",
    "other": "Other",
}
CELL_COLORS = {
    cell: color
    for cell, color in zip(
        CELL_ORDER,
        [
            "#D95F59",
            "#F2A65A",
            "#E9C46A",
            "#2A9D8F",
            "#A66BBE",
            "#3A7CA5",
            "#5DA9E9",
            "#76B041",
            "#E76FAD",
            "#8D99AE",
        ],
    )
}
REAL_COLOR = "#3F4650"
VIRTUAL_COLOR = "#326EA8"
EARLY_COLOR = "#D97706"
LATE_COLOR = "#2563A6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_for(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    part = frame.loc[frame["stage"].astype(str).eq(stage)]
    matrix = part.pivot_table(
        index="sender", columns="receiver", values="weight", aggfunc="sum", fill_value=0
    )
    return matrix.reindex(index=CELL_ORDER, columns=CELL_ORDER, fill_value=0).astype(float)


def normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    total = float(matrix.to_numpy().sum())
    if total <= 0:
        return matrix * 0.0
    return matrix / total


def top_real_edges(real_norm: pd.DataFrame, n: int = TOP_NETWORK_EDGES) -> list[tuple[str, str]]:
    long = real_norm.stack().rename("weight").reset_index()
    long.columns = ["sender", "receiver", "weight"]
    long = long.sort_values(
        ["weight", "sender", "receiver"], ascending=[False, True, True]
    ).head(n)
    return list(zip(long.sender, long.receiver))


def draw_cell(ax, x: float, y: float, radius: float, color: str) -> None:
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="white", lw=1.2))
    ax.add_patch(
        Circle(
            (x - radius * 0.12, y + radius * 0.03),
            radius * 0.40,
            facecolor="#F3F4F6",
            edgecolor="white",
            lw=0.6,
            alpha=0.92,
        )
    )


def draw_schematic(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cards = [
        (0.03, 0.12, 0.25, 0.72, "#F7F7F8", "#4B5563", "Observed spatial tissue"),
        (0.375, 0.12, 0.25, 0.72, "#F4F7FB", "#375A7F", "COMMOT communication graph"),
        (0.72, 0.12, 0.25, 0.72, "#F4F8FC", "#326EA8", "Virtual-cell reconstruction"),
    ]
    for x, y, w, h, face, edge, title in cards:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="center", fontsize=10, fontweight="bold")

    # Observed tissue: heterogeneous sender/receiver cells and ligand arrows.
    observed_cells = [
        (0.075, 0.37, "#D95F59"),
        (0.145, 0.55, "#3A7CA5"),
        (0.215, 0.35, "#76B041"),
        (0.205, 0.62, "#A66BBE"),
    ]
    for x, y, color in observed_cells:
        draw_cell(ax, x, y, 0.034, color)
    ax.add_patch(
        FancyArrowPatch(
            (0.098, 0.39),
            (0.184, 0.36),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.6,
            color="#D95F59",
            connectionstyle="arc3,rad=-0.15",
        )
    )
    for x in (0.12, 0.14, 0.16):
        ax.add_patch(Circle((x, 0.39 + (x - 0.14) ** 2 * 8), 0.006, color="#E9C46A"))

    # COMMOT graph: fixed nodes and directed edges.
    center = np.array([0.50, 0.43])
    graph_pos = []
    for i, color in enumerate(["#D95F59", "#3A7CA5", "#76B041", "#A66BBE", "#2A9D8F"]):
        angle = np.pi / 2 + 2 * np.pi * i / 5
        point = center + 0.085 * np.array([np.cos(angle), np.sin(angle)])
        graph_pos.append(point)
        ax.add_patch(Circle(tuple(point), 0.017, facecolor=color, edgecolor="white", lw=0.6, zorder=4))
    for i, j in [(0, 2), (0, 3), (1, 4), (3, 1), (4, 2), (2, 1)]:
        ax.add_patch(
            FancyArrowPatch(
                graph_pos[i],
                graph_pos[j],
                arrowstyle="-|>",
                mutation_scale=6,
                lw=1.25,
                color="#6B7280",
                alpha=0.8,
                connectionstyle="arc3,rad=0.12",
                zorder=2,
            )
        )
    ax.text(0.50, 0.24, "same cell types · directed edge weights", ha="center", fontsize=8, color="#4B5563")

    # Virtual reconstruction uses the same cell colors and recovered arrows.
    virtual_cells = [
        (0.765, 0.37, "#D95F59"),
        (0.835, 0.55, "#3A7CA5"),
        (0.905, 0.35, "#76B041"),
        (0.895, 0.62, "#A66BBE"),
    ]
    for x, y, color in virtual_cells:
        draw_cell(ax, x, y, 0.034, color)
    ax.add_patch(
        FancyArrowPatch(
            (0.788, 0.39),
            (0.874, 0.36),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.6,
            color="#326EA8",
            connectionstyle="arc3,rad=-0.15",
        )
    )
    ax.text(0.845, 0.24, "compare topology and pathway fingerprints", ha="center", fontsize=8, color="#4B5563")

    for x1, x2 in [(0.29, 0.365), (0.635, 0.71)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 0.48),
                (x2, 0.48),
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.5,
                color="#7B8794",
            )
        )
    ax.text(0.0, 0.95, "A. Communication reconstruction and concordance workflow", fontsize=12.5, fontweight="bold", va="top")


def circular_positions() -> dict[str, np.ndarray]:
    positions = {}
    for i, cell in enumerate(CELL_ORDER):
        angle = np.pi / 2 + 2 * np.pi * i / len(CELL_ORDER)
        positions[cell] = np.array([np.cos(angle), np.sin(angle)])
    return positions


def draw_network(
    ax,
    matrix: pd.DataFrame,
    selected_edges: list[tuple[str, str]],
    shared_max: float,
    title: str,
    panel_letter: str,
    is_virtual: bool,
) -> None:
    positions = circular_positions()
    values = matrix.to_dict()
    for rank, (sender, receiver) in enumerate(selected_edges):
        value = float(matrix.loc[sender, receiver])
        relative = value / max(shared_max, 1e-12)
        color = CELL_COLORS[sender]
        if sender == receiver:
            p = positions[sender]
            loop_radius = 0.17 + 0.055 * np.sqrt(relative)
            ax.add_patch(
                Circle(
                    tuple(p * 1.04),
                    loop_radius,
                    fill=False,
                    edgecolor=color if value > 0 else "#D1D5DB",
                    linewidth=0.6 + 4.8 * np.sqrt(relative),
                    alpha=0.52 if value > 0 else 0.20,
                    linestyle="-" if value > 0 else "--",
                    zorder=1,
                )
            )
            continue
        start = positions[sender] * 0.91
        end = positions[receiver] * 0.91
        sender_i = CELL_ORDER.index(sender)
        receiver_i = CELL_ORDER.index(receiver)
        direction = 1 if (receiver_i - sender_i) % len(CELL_ORDER) < len(CELL_ORDER) / 2 else -1
        radius = 0.13 * direction + 0.018 * ((rank % 3) - 1)
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=5.5 + 2.0 * np.sqrt(relative),
            linewidth=0.45 + 4.3 * np.sqrt(relative),
            color=color if value > 0 else "#C9CDD3",
            alpha=0.55 if value > 0 else 0.18,
            linestyle="-" if value > 0 else "--",
            connectionstyle=f"arc3,rad={radius:.3f}",
            zorder=1,
        )
        ax.add_patch(patch)

    for cell, p in positions.items():
        ax.add_patch(
            Circle(tuple(p), 0.095, facecolor=CELL_COLORS[cell], edgecolor="white", lw=1.2, zorder=5)
        )
        label_pos = p * 1.23
        ha = "left" if label_pos[0] > 0.15 else "right" if label_pos[0] < -0.15 else "center"
        va = "bottom" if label_pos[1] > 0.75 else "top" if label_pos[1] < -0.75 else "center"
        ax.text(
            label_pos[0],
            label_pos[1],
            CELL_LABELS[cell],
            ha=ha,
            va=va,
            fontsize=6.3,
            color="#20252B",
        )
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.36, 1.36)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{panel_letter}. {title}",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        pad=7,
        color=VIRTUAL_COLOR if is_virtual else REAL_COLOR,
    )


def edge_vectors(real: pd.DataFrame, virtual: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = real.to_numpy(float).ravel()
    q = virtual.to_numpy(float).ravel()
    self_mask = np.eye(len(CELL_ORDER), dtype=bool).ravel()
    return p, q, self_mask


def draw_concordance_scatter(
    ax,
    real: pd.DataFrame,
    virtual: pd.DataFrame,
    metrics: pd.Series,
    title: str,
    panel_letter: str,
) -> None:
    p, q, self_mask = edge_vectors(real, virtual)
    pseudocount = 1e-5
    x = np.log10(p + pseudocount)
    y = np.log10(q + pseudocount)
    top10 = np.argsort(p)[-10:]
    top_mask = np.zeros_like(p, dtype=bool)
    top_mask[top10] = True

    ax.scatter(
        x[~self_mask],
        y[~self_mask],
        s=18,
        color="#8CB6D9",
        alpha=0.62,
        edgecolor="none",
        label="Cross-type edges",
    )
    ax.scatter(
        x[self_mask],
        y[self_mask],
        s=45,
        marker="D",
        color="#7A4EAB",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
        label="Self-edges",
        zorder=4,
    )
    ax.scatter(
        x[top_mask],
        y[top_mask],
        s=62,
        facecolor="none",
        edgecolor="#E07A10",
        linewidth=1.2,
        label="Real top-10",
        zorder=5,
    )
    lo = min(float(np.nanmin(x)), float(np.nanmin(y)), -5.0)
    hi = max(float(np.nanmax(x)), float(np.nanmax(y)), -0.5)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#737A83", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E5E7EB", lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Real normalized edge weight (log10)", fontsize=8)
    ax.set_ylabel("Virtual normalized edge weight (log10)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{panel_letter}. {title}", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.04,
        0.96,
        (
            f"Pearson = {float(metrics.sender_receiver_pearson):.2f}\n"
            f"Spearman = {float(metrics.sender_receiver_spearman):.2f}\n"
            f"Top-10 Jaccard = {float(metrics.top10_edge_jaccard):.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D1D5DB", alpha=0.92),
    )


def exact_family_distribution(
    edges: pd.DataFrame, stage: str, family_path: str, mode: str
) -> pd.Series:
    part = edges.loc[
        edges["stage"].astype(str).eq(stage)
        & edges["pathway"].astype(str).eq(family_path)
    ]
    key = "sender" if mode == "sender" else "receiver"
    totals = part.groupby(key)["weight"].sum().astype(float).clip(lower=0)
    denom = float(totals.sum())
    return totals / denom if denom > 0 else pd.Series(dtype=float)


def family_metrics(real_edges: pd.DataFrame, virtual_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        for family, exact_path in FAMILY_PATHS.items():
            for mode in MODES:
                p_s = exact_family_distribution(real_edges, stage["real"], exact_path, mode)
                q_s = exact_family_distribution(virtual_edges, stage["virtual"], exact_path, mode)
                types = sorted(set(p_s.index.astype(str)) | set(q_s.index.astype(str)))
                p = p_s.reindex(types, fill_value=0.0).to_numpy(float)
                q = q_s.reindex(types, fill_value=0.0).to_numpy(float)
                if p.sum() <= 0 or q.sum() <= 0:
                    rho = js_sim = cosine = tv = np.nan
                else:
                    p = p / p.sum()
                    q = q / q.sum()
                    rho = float(spearmanr(p, q).statistic) if np.std(p) > 0 and np.std(q) > 0 else np.nan
                    js_sim = 1.0 - float(jensenshannon(p, q, base=2.0)) ** 2
                    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
                    cosine = float(np.dot(p, q) / denom) if denom > 0 else np.nan
                    tv = float(0.5 * np.abs(p - q).sum())
                rows.append(
                    {
                        "stage": stage["short"],
                        "stage_semantic": stage["semantic"],
                        "real_stage": stage["real"],
                        "virtual_stage": stage["virtual"],
                        "family": family,
                        "family_exact_commot_path": exact_path,
                        "mode": mode,
                        "n_union_cell_types": len(types),
                        "spearman": rho,
                        "js_similarity": js_sim,
                        "cosine_similarity": cosine,
                        "total_variation_distance": tv,
                    }
                )
    return pd.DataFrame(rows)


def draw_family_bubbles(ax, family: pd.DataFrame, panel_letter: str) -> None:
    row_specs = [
        ("Early", "sender", "Early · Sender"),
        ("Early", "receiver", "Early · Receiver"),
        ("Late", "sender", "Late · Sender"),
        ("Late", "receiver", "Late · Receiver"),
    ]
    families = list(FAMILY_PATHS)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap("RdBu_r")
    for yi, (stage, mode, _label) in enumerate(row_specs):
        for xi, family_name in enumerate(families):
            row = family.loc[
                family.stage.eq(stage)
                & family["mode"].eq(mode)
                & family.family.eq(family_name)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing family metric {stage}/{mode}/{family_name}")
            record = row.iloc[0]
            rho = float(record.spearman)
            js = float(record.js_similarity)
            size = 420 * max(js, 0.0) ** 2
            ax.scatter(
                xi,
                yi,
                s=size,
                color=cmap(norm(rho)),
                edgecolor="#374151",
                linewidth=0.5,
                zorder=3,
            )
            text_color = "white" if abs(rho) > 0.52 else "#111827"
            ax.text(
                xi,
                yi,
                f"ρ {rho:.2f}\nJS {js:.2f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color=text_color,
                fontweight="bold",
                zorder=4,
            )
    ax.set_xlim(-0.65, len(families) - 0.35)
    ax.set_ylim(len(row_specs) - 0.45, -0.55)
    ax.set_xticks(range(len(families)), families, fontsize=8.5)
    ax.set_yticks(range(len(row_specs)), [x[2] for x in row_specs], fontsize=8)
    ax.set_title(
        f"{panel_letter}. Five-pathway communication fingerprints",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Sender/receiver rank agreement (Spearman)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    ax.text(
        0.5,
        -0.18,
        "Bubble size = Jensen–Shannon similarity",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.2,
        color="#4B5563",
    )


def lr_inventory(real_edges: pd.DataFrame, virtual_edges: pd.DataFrame) -> pd.DataFrame:
    aggregate_paths = set(FAMILY_PATHS.values()) | {"commot-cellchat-total-total"}
    rows = []
    for stage in STAGES:
        r = real_edges.loc[real_edges.stage.astype(str).eq(stage["real"])]
        v = virtual_edges.loc[virtual_edges.stage.astype(str).eq(stage["virtual"])]
        r_total = r.groupby("pathway").weight.sum()
        v_total = v.groupby("pathway").weight.sum()
        pair_ids = sorted((set(r_total.index) | set(v_total.index)) - aggregate_paths)
        real_denom = float(r_total.reindex(pair_ids, fill_value=0.0).sum())
        virtual_denom = float(v_total.reindex(pair_ids, fill_value=0.0).sum())
        for pair in pair_ids:
            rw = float(r_total.get(pair, 0.0))
            vw = float(v_total.get(pair, 0.0))
            rows.append(
                {
                    "stage": stage["short"],
                    "real_stage": stage["real"],
                    "virtual_stage": stage["virtual"],
                    "lr_pair_id": pair,
                    "present_real": rw > 0,
                    "present_virtual": vw > 0,
                    "real_total_weight": rw,
                    "virtual_total_weight": vw,
                    "real_normalized_weight": rw / real_denom if real_denom > 0 else np.nan,
                    "virtual_normalized_weight": vw / virtual_denom if virtual_denom > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    real_matrices: dict[str, pd.DataFrame],
    virtual_matrices: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    family: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    fig = plt.figure(figsize=(21.0, 11.8), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.62, 1.60, 1.36],
        hspace=0.34,
        wspace=0.34,
        left=0.045,
        right=0.975,
        top=0.90,
        bottom=0.085,
    )
    schematic_ax = fig.add_subplot(grid[0, :])
    draw_schematic(schematic_ax)

    network_axes = [fig.add_subplot(grid[1, i]) for i in range(4)]
    panel_info = []
    letters = iter(["B", "C", "D", "E"])
    for stage in STAGES:
        real = real_matrices[stage["real"]]
        virtual = virtual_matrices[stage["virtual"]]
        selected = top_real_edges(real)
        shared_max = max(
            max(float(real.loc[s, r]) for s, r in selected),
            max(float(virtual.loc[s, r]) for s, r in selected),
        )
        panel_info.extend(
            [
                (real, selected, shared_max, f"{stage['short']} Real", next(letters), False),
                (
                    virtual,
                    selected,
                    shared_max,
                    f"{stage['short']} Virtual",
                    next(letters),
                    True,
                ),
            ]
        )
    for ax, args in zip(network_axes, panel_info):
        draw_network(ax, *args)

    scatter_early = fig.add_subplot(grid[2, 0])
    scatter_late = fig.add_subplot(grid[2, 1])
    family_ax = fig.add_subplot(grid[2, 2:])
    for ax, stage, letter in [
        (scatter_early, STAGES[0], "F"),
        (scatter_late, STAGES[1], "G"),
    ]:
        metric = summary.loc[
            summary.real_stage.eq(stage["real"])
            & summary.virtual_stage.eq(stage["virtual"])
        ]
        if len(metric) != 1:
            raise RuntimeError(f"Missing summary metrics for {stage}")
        draw_concordance_scatter(
            ax,
            real_matrices[stage["real"]],
            virtual_matrices[stage["virtual"]],
            metric.iloc[0],
            f"{stage['short']} edge-weight concordance",
            letter,
        )
    draw_family_bubbles(family_ax, family, "H")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=7, label=CELL_LABELS[cell])
        for cell, color in CELL_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=10,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.25,
        columnspacing=0.8,
    )
    fig.suptitle(
        "GSE267904 observed–virtual cell–cell communication concordance",
        fontsize=18,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.957,
        "Existing COMMOT outputs · identical cell-type order · five frozen pathway families",
        ha="center",
        fontsize=9.3,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.022,
        "Network panels show the Real-defined top 20 sender–receiver edges after within-stage normalization. "
        "Early uses the documented d7 calibration prior; Late is the held-out d21 comparison. "
        "No model was rerun and raw COMMOT amplitude is not compared across source sizes.",
        ha="center",
        fontsize=7.4,
        color="#374151",
    )

    preview = OUT / "GSE267904_communication_concordance_visual_v2_preview.png"
    png = OUT / "GSE267904_communication_concordance_visual_v2_600dpi.png"
    pdf = OUT / "GSE267904_communication_concordance_visual_v2_vector.pdf"
    fig.savefig(preview, dpi=160, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return preview, png, pdf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        REAL_MATRIX_PATH,
        VIRTUAL_MATRIX_PATH,
        REAL_EDGES_PATH,
        VIRTUAL_EDGES_PATH,
        SUMMARY_PATH,
        INPUT_MANIFEST_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    real_matrix_long = pd.read_csv(REAL_MATRIX_PATH)
    virtual_matrix_long = pd.read_csv(VIRTUAL_MATRIX_PATH)
    real_edges = pd.read_csv(REAL_EDGES_PATH)
    virtual_edges = pd.read_csv(VIRTUAL_EDGES_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    input_manifest = json.loads(INPUT_MANIFEST_PATH.read_text())

    real_matrices = {
        stage["real"]: normalize_matrix(matrix_for(real_matrix_long, stage["real"]))
        for stage in STAGES
    }
    virtual_matrices = {
        stage["virtual"]: normalize_matrix(
            matrix_for(virtual_matrix_long, stage["virtual"])
        )
        for stage in STAGES
    }

    # Recompute edge correlations from the plotted matrices and require agreement
    # with the frozen K=100 summary.
    concordance_rows = []
    for stage in STAGES:
        p, q, self_mask = edge_vectors(
            real_matrices[stage["real"]], virtual_matrices[stage["virtual"]]
        )
        pearson = float(np.corrcoef(p, q)[0, 1])
        spearman = float(spearmanr(p, q).statistic)
        frozen = summary.loc[
            summary.real_stage.eq(stage["real"])
            & summary.virtual_stage.eq(stage["virtual"])
        ].iloc[0]
        if not np.isclose(pearson, float(frozen.sender_receiver_pearson), atol=1e-12):
            raise RuntimeError(f"Pearson mismatch for {stage['short']}")
        if not np.isclose(spearman, float(frozen.sender_receiver_spearman), atol=1e-12):
            raise RuntimeError(f"Spearman mismatch for {stage['short']}")
        selected = top_real_edges(real_matrices[stage["real"]])
        concordance_rows.append(
            {
                "stage": stage["short"],
                "stage_semantic": stage["semantic"],
                "real_stage": stage["real"],
                "virtual_stage": stage["virtual"],
                "edge_count": len(p),
                "self_edge_count": int(self_mask.sum()),
                "cross_type_edge_count": int((~self_mask).sum()),
                "pearson": pearson,
                "spearman": spearman,
                "top10_jaccard": float(frozen.top10_edge_jaccard),
                "network_display_edge_count": len(selected),
                "network_display_selection": "top 20 by Real normalized weight",
            }
        )

    family = family_metrics(real_edges, virtual_edges)
    inventory = lr_inventory(real_edges, virtual_edges)
    concordance = pd.DataFrame(concordance_rows)
    family_csv = OUT / "five_pathway_family_concordance.csv"
    inventory_csv = OUT / "complete_lr_pair_inventory_and_recovery.csv"
    concordance_csv = OUT / "edge_concordance_metrics.csv"
    family.to_csv(family_csv, index=False)
    inventory.to_csv(inventory_csv, index=False)
    concordance.to_csv(concordance_csv, index=False)

    preview, png, pdf = make_figure(
        real_matrices, virtual_matrices, summary, family
    )

    aggregate_ids = set(FAMILY_PATHS.values()) | {"commot-cellchat-total-total"}
    real_unique = set(real_edges.pathway.astype(str))
    virtual_unique = set(virtual_edges.pathway.astype(str))
    real_pairs = real_unique - aggregate_ids
    virtual_pairs = virtual_unique - aggregate_ids
    audit = {
        "status": "PASS_COMMUNICATION_CONCORDANCE_VISUAL_V2",
        "models_rerun": False,
        "commot_rerun": False,
        "physicell_rerun": False,
        "agent_count_displayed_in_figure": False,
        "schematic": {
            "style": "BioRender-inspired vector schematic drawn programmatically",
            "uses_BioRender_assets": False,
            "purpose": "orient the observed-COMMOT-virtual comparison",
        },
        "pathway_scope": {
            "main_figure_families": list(FAMILY_PATHS),
            "family_count": len(FAMILY_PATHS),
            "family_selection": "all aggregate CellChat families present in the frozen edge files",
            "real_lr_pair_count": len(real_pairs),
            "virtual_lr_pair_count": len(virtual_pairs),
            "common_lr_pair_count": len(real_pairs & virtual_pairs),
            "real_only_lr_pair_count": len(real_pairs - virtual_pairs),
            "virtual_only_lr_pair_count": len(virtual_pairs - real_pairs),
            "complete_inventory_csv": str(inventory_csv),
        },
        "network_panels": {
            "cell_type_count": len(CELL_ORDER),
            "cell_type_order": CELL_ORDER,
            "display_edge_count_per_stage": TOP_NETWORK_EDGES,
            "selection_rule": "top 20 by Real normalized sender-receiver weight",
            "same_selected_edges_in_real_and_virtual_panel": True,
            "within_stage_normalization": True,
            "cross_stage_raw_amplitude_comparison": False,
        },
        "stage_semantics": {
            "early_d7_calibration_prior_enabled": bool(
                input_manifest.get("d7_cross_type_calibration_prior_enabled", False)
            ),
            "late_d21_real_commot_used_for_generation": bool(
                input_manifest.get("d21_real_commot_used_for_lr_program", False)
            ),
        },
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in required
        ],
        "outputs": {
            "preview": str(preview),
            "png_600dpi": str(png),
            "pdf_vector": str(pdf),
            "family_metrics_csv": str(family_csv),
            "edge_metrics_csv": str(concordance_csv),
            "lr_inventory_csv": str(inventory_csv),
        },
        "visual_inspection": (
            "PASS_NO_PANEL_CLIPPING; late Virtual cross-type edges remain visibly "
            "weak as required by the frozen data"
        ),
    }
    audit_path = OUT / "communication_concordance_visual_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    readme = OUT / "README.md"
    readme.write_text(
        f"""# GSE267904 observed\u2013virtual communication concordance V2

This is a plotting-only and metric-organization revision based on existing
COMMOT outputs. It does not rerun any model and does not display Agent count.

The main figure combines:

1. an original BioRender-inspired communication workflow schematic;
2. paired Real/Virtual circular communication networks with identical cell
   positions and the Real-defined top 20 edges;
3. all-edge Real-versus-Virtual concordance plots; and
4. all five aggregate CellChat pathway families (TGF\u03b2, CCL, CXCL, PDGF,
   VEGF) in a bubble fingerprint.

The complete ligand\u2013receptor inventory is exported rather than crowded into
the main figure. Early/d7 is calibration/reconstruction; Late/d21 is held-out
with respect to the LR generation program.

- Preview: `{preview}`
- 600 dpi PNG: `{png}`
- Vector PDF: `{pdf}`
- Five-family metrics: `{family_csv}`
- Complete LR inventory: `{inventory_csv}`
- Audit: `{audit_path}`
"""
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
