#!/usr/bin/env python3
"""Plot updated spatial COMMOT hotspot maps using current PhysiCell coordinates."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_PAIRS = [
    ("d7_bleo", "virtual_early_bleo", "early"),
    ("d21_bleo", "virtual_late_bleo", "late"),
]


CELL_COLORS = {
    "alveolar_epithelial_AT1_AT2": "#4E79A7",
    "activated_Krt8_ADI_epithelial": "#F28E2B",
    "airway_epithelial": "#FF9DA7",
    "macrophage": "#59A14F",
    "recruited_monocyte_macrophage": "#E15759",
    "fibroblast_myofibroblast": "#B07AA1",
    "endothelial": "#76B7B2",
    "dendritic": "#EDC948",
    "lymphoid": "#9C755F",
    "other": "#BAB0AC",
}

LABELS = {
    "alveolar_epithelial_AT1_AT2": "Alv epi",
    "activated_Krt8_ADI_epithelial": "Krt8/ADI",
    "airway_epithelial": "Airway",
    "macrophage": "Macro",
    "recruited_monocyte_macrophage": "Rec mono",
    "fibroblast_myofibroblast": "Fibro",
    "endothelial": "Endo",
    "dendritic": "DC",
    "lymphoid": "Lymph",
    "other": "Other",
}


def outpath(root: Path, out_dir: Path) -> Path:
    return out_dir if out_dir.is_absolute() else root / out_dir


def load_adata(out: Path, stage: str) -> ad.AnnData:
    if stage.startswith("virtual"):
        p = out / "virtual_commot_input" / f"{stage}.h5ad"
    else:
        p = out / "real_benchmark" / f"{stage}.h5ad"
    if not p.exists():
        raise FileNotFoundError(p)
    return ad.read_h5ad(p)


def get_xy_and_celltype(a: ad.AnnData, virtual: bool) -> tuple[np.ndarray, pd.Series]:
    xy = np.asarray(a.obsm["spatial"])
    if virtual:
        ct = a.obs["cell_type"].astype(str)
    else:
        ct = a.obs["dominant_cell_type"].astype(str)
    return xy, ct


def strength_by_type_from_matrix(mat: pd.DataFrame, stage: str, mode: str) -> dict[str, float]:
    d = mat[mat["stage"].eq(stage)].copy()
    if mode == "sender":
        return d.groupby("sender")["weight"].sum().to_dict()
    return d.groupby("receiver")["weight"].sum().to_dict()


def strength_by_type_from_edges(edges: pd.DataFrame, stage: str, family: str, mode: str) -> dict[str, float]:
    d = edges[edges["stage"].eq(stage)].copy()
    d = d[d["pathway"].astype(str).str.upper().str.contains(family.upper(), na=False)]
    if mode == "sender":
        return d.groupby("sender")["weight"].sum().to_dict()
    return d.groupby("receiver")["weight"].sum().to_dict()


def project_scores(ct: pd.Series, score_map: dict[str, float]) -> np.ndarray:
    return ct.map(lambda x: float(score_map.get(x, 0.0))).to_numpy()


def plot_celltype_map(ax, xy: np.ndarray, ct: pd.Series, title: str, s: float) -> None:
    for ctype in sorted(ct.unique()):
        idx = ct.eq(ctype).to_numpy()
        ax.scatter(xy[idx, 0], xy[idx, 1], s=s, c=CELL_COLORS.get(ctype, "#888888"), label=LABELS.get(ctype, ctype), alpha=0.9, linewidths=0)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_score_map(ax, xy: np.ndarray, scores: np.ndarray, title: str, cmap: str, vmax: float | None, s: float) -> None:
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=scores, s=s, cmap=cmap, vmin=0, vmax=vmax, alpha=0.92, linewidths=0)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return sc


def make_total_hotspot_figure(out: Path) -> Path:
    real_mat = pd.read_csv(out / "real_commot/real_sender_receiver_matrix_by_stage.csv")
    virt_mat = pd.read_csv(out / "virtual_commot/virtual_sender_receiver_matrix_by_stage.csv")
    fig = plt.figure(figsize=(23, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        8,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.055, 1.0, 1.0, 0.055],
        wspace=0.18,
        hspace=0.18,
    )
    axes = np.empty((2, 6), dtype=object)
    cbar_axes = {}
    for r in range(2):
        axes[r, 0] = fig.add_subplot(gs[r, 0])
        axes[r, 1] = fig.add_subplot(gs[r, 1])
        axes[r, 2] = fig.add_subplot(gs[r, 2])
        axes[r, 3] = fig.add_subplot(gs[r, 3])
        cbar_axes[(r, "sender")] = fig.add_subplot(gs[r, 4])
        axes[r, 4] = fig.add_subplot(gs[r, 5])
        axes[r, 5] = fig.add_subplot(gs[r, 6])
        cbar_axes[(r, "receiver")] = fig.add_subplot(gs[r, 7])
    for row, (real_stage, virt_stage, label) in enumerate(STAGE_PAIRS):
        real = load_adata(out, real_stage)
        virt = load_adata(out, virt_stage)
        rxy, rct = get_xy_and_celltype(real, virtual=False)
        vxy, vct = get_xy_and_celltype(virt, virtual=True)
        plot_celltype_map(axes[row, 0], rxy, rct, f"Real cell-type map\n{real_stage}", s=3)
        plot_celltype_map(axes[row, 1], vxy, vct, f"Virtual micro-agent map\n{virt_stage}", s=50)
        rsend = project_scores(rct, strength_by_type_from_matrix(real_mat, real_stage, "sender"))
        vsend = project_scores(vct, strength_by_type_from_matrix(virt_mat, virt_stage, "sender"))
        rrecv = project_scores(rct, strength_by_type_from_matrix(real_mat, real_stage, "receiver"))
        vrecv = project_scores(vct, strength_by_type_from_matrix(virt_mat, virt_stage, "receiver"))
        send_vmax = max(float(np.nanmax(rsend)), float(np.nanmax(vsend)), 1.0)
        recv_vmax = max(float(np.nanmax(rrecv)), float(np.nanmax(vrecv)), 1.0)
        im1 = plot_score_map(axes[row, 2], rxy, rsend, "Real sender hotspot\noutgoing strength", "YlOrRd", send_vmax, s=3)
        plot_score_map(axes[row, 3], vxy, vsend, "Virtual sender hotspot\noutgoing strength", "YlOrRd", send_vmax, s=50)
        im2 = plot_score_map(axes[row, 4], rxy, rrecv, "Real receiver hotspot\nincoming strength", "YlGnBu", recv_vmax, s=3)
        plot_score_map(axes[row, 5], vxy, vrecv, "Virtual receiver hotspot\nincoming strength", "YlGnBu", recv_vmax, s=50)
        cbar1 = fig.colorbar(im1, cax=cbar_axes[(row, "sender")])
        cbar1.set_label("Outgoing\nweight", fontsize=8)
        cbar1.ax.tick_params(labelsize=8)
        cbar2 = fig.colorbar(im2, cax=cbar_axes[(row, "receiver")])
        cbar2.set_label("Incoming\nweight", fontsize=8)
        cbar2.ax.tick_params(labelsize=8)
        axes[row, 0].set_ylabel(f"{label}\nreal/virtual", fontsize=11, weight="bold")
    handles = []
    labels = []
    for ctype, color in CELL_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=6))
        labels.append(LABELS.get(ctype, ctype))
    fig.legend(handles, labels, loc="outside lower center", ncol=10, frameon=False, fontsize=8)
    fig.suptitle("GSE267904 spatial COMMOT validation: real Visium spots vs moving PhysiCell-backed virtual micro-agents", fontsize=17, weight="bold")
    fig.text(
        0.5,
        0.025,
        "Sender/receiver hotspot scores are cell-type aggregated COMMOT outgoing/incoming strengths projected onto real spots or virtual micro-agents. Virtual coordinates are parsed from the updated PhysiCell XML outputs.",
        ha="center",
        fontsize=9,
        color="#4a5568",
    )
    p = out / "figures/gse267904_spatial_position_total_sender_receiver_FIXED.png"
    fig.savefig(p, dpi=260)
    plt.close(fig)
    return p


def make_pathway_figure(out: Path) -> Path:
    real_edges = pd.read_csv(out / "real_commot/real_commot_edges_long.csv")
    virt_edges = pd.read_csv(out / "virtual_commot/virtual_commot_edges_long.csv")
    families = ["TGFB", "CCL", "CXCL"]
    fig, axes = plt.subplots(2, len(families) * 2, figsize=(18, 7.8))
    for row, (real_stage, virt_stage, label) in enumerate(STAGE_PAIRS):
        real = load_adata(out, real_stage)
        virt = load_adata(out, virt_stage)
        rxy, rct = get_xy_and_celltype(real, virtual=False)
        vxy, vct = get_xy_and_celltype(virt, virtual=True)
        for col, fam in enumerate(families):
            rsend = project_scores(rct, strength_by_type_from_edges(real_edges, real_stage, fam, "sender"))
            vsend = project_scores(vct, strength_by_type_from_edges(virt_edges, virt_stage, fam, "sender"))
            rrecv = project_scores(rct, strength_by_type_from_edges(real_edges, real_stage, fam, "receiver"))
            vrecv = project_scores(vct, strength_by_type_from_edges(virt_edges, virt_stage, fam, "receiver"))
            # Show sender and receiver as real+virtual overlays by using the
            # same color scale per pathway/stage pair.
            send_vmax = max(float(np.nanmax(rsend)), float(np.nanmax(vsend)), 1.0)
            recv_vmax = max(float(np.nanmax(rrecv)), float(np.nanmax(vrecv)), 1.0)
            ax1 = axes[row, col * 2]
            ax2 = axes[row, col * 2 + 1]
            im1 = plot_score_map(ax1, rxy, rsend, f"Real {fam} sender\n{real_stage}", "YlOrRd", send_vmax, s=3)
            # Overlay virtual as larger outlined points in same panel? Instead
            # use separate right panel for clarity.
            ax1.scatter(vxy[:, 0], vxy[:, 1], c=vsend, s=32, cmap="YlOrRd", vmin=0, vmax=send_vmax, edgecolors="#374151", linewidths=0.25, alpha=0.85)
            im2 = plot_score_map(ax2, rxy, rrecv, f"Real {fam} receiver\n{real_stage}", "YlGnBu", recv_vmax, s=3)
            ax2.scatter(vxy[:, 0], vxy[:, 1], c=vrecv, s=32, cmap="YlGnBu", vmin=0, vmax=recv_vmax, edgecolors="#374151", linewidths=0.25, alpha=0.85)
            fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.01)
            fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.01)
        axes[row, 0].set_ylabel(f"{label}", fontsize=11, weight="bold")
    fig.suptitle("GSE267904 pathway-specific spatial hotspots: real spots with virtual micro-agent overlays", fontsize=16, weight="bold", y=0.985)
    fig.text(0.5, 0.035, "Small background dots are real Visium spots. Larger outlined dots are virtual PhysiCell micro-agents at updated positions. Families shown only when present in COMMOT output.", ha="center", fontsize=9, color="#4a5568")
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.93])
    p = out / "figures/gse267904_spatial_position_pathway_hotspots_UPDATED.png"
    fig.savefig(p, dpi=260)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = outpath(root, args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    p1 = make_total_hotspot_figure(out)
    p2 = make_pathway_figure(out)
    print(f"Wrote updated spatial COMMOT position figures:\\n- {p1}\\n- {p2}")


if __name__ == "__main__":
    main()
