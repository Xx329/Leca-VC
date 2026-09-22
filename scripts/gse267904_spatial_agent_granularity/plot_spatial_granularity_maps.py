#!/usr/bin/env python3
"""Spatial visual comparison for GSE267904 Agent granularity.

This script does not rerun LLM, PhysiCell, or COMMOT.  It reads the existing
GSE267904 spatial Agent-count outputs and generates smoothed spatial hotspot
figures comparing:

    real Visium spots vs K=20 virtual micro-agents vs K=50 virtual micro-agents

The plots are intentionally hotspot/region comparisons rather than spot-to-cell
matching, because real Visium spots and virtual micro-agents live at different
resolutions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


STAGE_PAIRS = [
    ("d7_bleo", "virtual_early_bleo", "d7 / virtual early"),
    ("d21_bleo", "virtual_late_bleo", "d21 / virtual late"),
]

CELL_TYPES = [
    "alveolar_epithelial_AT1_AT2",
    "activated_Krt8_ADI_epithelial",
    "airway_epithelial",
    "macrophage",
    "recruited_monocyte_macrophage",
    "fibroblast_myofibroblast",
    "endothelial",
    "dendritic",
    "lymphoid",
    "other",
]

STATE_FEATURES = {
    "Macrophage / recruited mono-mac": {
        "real_score_cols": ["cell_type_score_macrophage", "cell_type_score_recruited_monocyte_macrophage"],
        "virtual_cell_types": ["macrophage", "recruited_monocyte_macrophage"],
    },
    "Fibroblast / myofibroblast": {
        "real_score_cols": ["cell_type_score_fibroblast_myofibroblast"],
        "virtual_cell_types": ["fibroblast_myofibroblast"],
    },
    "Krt8 / ADI epithelial repair": {
        "real_score_cols": ["cell_type_score_activated_Krt8_ADI_epithelial"],
        "virtual_cell_types": ["activated_Krt8_ADI_epithelial"],
    },
}

MODULE_GENES = {
    "Inflammation": ["Il1b", "Tnf", "Nfkbia", "Ccl2", "Cxcl2", "Cxcl10"],
    "Fibrosis / ECM": ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Acta2", "Tagln"],
    "TGFβ response": ["Tgfb1", "Tgfbr1", "Tgfbr2", "Serpine1", "Col1a1", "Fn1"],
    "CCL/CXCL chemokine": ["Ccl2", "Ccl7", "Ccl8", "Cxcl1", "Cxcl2", "Cxcl10"],
    "Krt8 / ADI repair": ["Krt8", "Krt18", "Krt19", "Clu", "Lgals3", "Sox4"],
    "Macrophage activation": ["Lyz2", "Adgre1", "C1qa", "C1qb", "C1qc", "Mrc1", "Spp1"],
}

PATHWAY_FAMILIES = {
    "TGFβ": ["TGFB", "TGFb", "Tgfb"],
    "CCL": ["CCL", "Ccl"],
    "CXCL": ["CXCL", "Cxcl"],
    "SPP1": ["SPP1", "Spp1"],
    "MIF": ["MIF", "Mif"],
}


def resolve_out(project_root: Path, out_dir: Path) -> Path:
    return out_dir if out_dir.is_absolute() else project_root / out_dir


def read_adata(path: Path) -> ad.AnnData:
    if not path.exists():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def norm_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    span = np.where((hi - lo) <= 0, 1.0, hi - lo)
    return (xy - lo) / span


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = np.nanpercentile(x, 1)
    hi = np.nanpercentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = np.nanmax(x)
        lo = np.nanmin(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def smooth_grid(xy01: np.ndarray, values: np.ndarray, grid_n: int = 90, sigma: float = 2.0) -> np.ndarray:
    """Weighted spatial binning followed by Gaussian smoothing."""
    xy01 = np.asarray(xy01, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(xy01).all(axis=1) & np.isfinite(values)
    xy01 = xy01[ok]
    values = values[ok]
    if len(values) == 0:
        return np.zeros((grid_n, grid_n), dtype=float)
    ix = np.clip((xy01[:, 0] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    iy = np.clip((xy01[:, 1] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    weight = np.zeros((grid_n, grid_n), dtype=float)
    count = np.zeros((grid_n, grid_n), dtype=float)
    np.add.at(weight, (iy, ix), values)
    np.add.at(count, (iy, ix), 1.0)
    weight = gaussian_filter(weight, sigma=sigma, mode="nearest")
    count = gaussian_filter(count, sigma=sigma, mode="nearest")
    grid = np.divide(weight, count, out=np.zeros_like(weight), where=count > 1e-9)
    return minmax01(grid)


def grid_similarity(a: np.ndarray, b: np.ndarray, top_frac: float = 0.15) -> tuple[float, float]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        pear = np.nan
    else:
        pear = float(np.corrcoef(a, b)[0, 1])
    k = max(1, int(np.ceil(len(a) * top_frac)))
    aa = set(np.argsort(a)[-k:].tolist())
    bb = set(np.argsort(b)[-k:].tolist())
    overlap = float(len(aa & bb) / max(1, len(aa | bb)))
    return pear, overlap


def values_state(adata: ad.AnnData, feature: dict, virtual: bool) -> np.ndarray:
    if virtual:
        ct = adata.obs["cell_type"].astype(str)
        return ct.isin(feature["virtual_cell_types"]).astype(float).to_numpy()
    vals = np.zeros(adata.n_obs, dtype=float)
    for col in feature["real_score_cols"]:
        if col in adata.obs:
            vals += pd.to_numeric(adata.obs[col], errors="coerce").fillna(0).to_numpy(dtype=float)
    return minmax01(vals)


def values_module(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return np.zeros(adata.n_obs, dtype=float)
    X = adata[:, present].X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    arr = np.asarray(arr, dtype=float)
    if arr.size and np.nanmax(arr) > 20:
        arr = np.log1p(arr)
    return minmax01(arr.mean(axis=1))


def load_real(out: Path, stage: str) -> ad.AnnData:
    candidates = [
        out / "shared_real" / "real_benchmark" / f"{stage}.h5ad",
        out / "K_50" / "real_benchmark" / f"{stage}.h5ad",
        out / "K_20" / "real_benchmark" / f"{stage}.h5ad",
    ]
    for p in candidates:
        if p.exists():
            return read_adata(p)
    raise FileNotFoundError(f"Could not find real h5ad for {stage}")


def load_virtual(out: Path, k: int, stage: str) -> ad.AnnData:
    return read_adata(out / f"K_{k}" / "virtual_commot_input" / f"{stage}.h5ad")


def plot_grid(ax, grid: np.ndarray, title: str, cmap: str = "magma") -> None:
    ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.set_title(title, fontsize=9.5, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def build_grids(
    out: Path,
    feature_name: str,
    value_fn,
    k_values: list[int],
    grid_n: int,
    sigma_real: float,
    sigma_virtual: float,
) -> tuple[dict, list[dict]]:
    grids: dict[tuple[str, str, int | str], np.ndarray] = {}
    rows: list[dict] = []
    for real_stage, virtual_stage, stage_label in STAGE_PAIRS:
        real = load_real(out, real_stage)
        rgrid = smooth_grid(norm_xy(real.obsm["spatial"]), value_fn(real, False), grid_n=grid_n, sigma=sigma_real)
        grids[(stage_label, "real", "real")] = rgrid
        for k in k_values:
            try:
                virt = load_virtual(out, k, virtual_stage)
                vgrid = smooth_grid(
                    norm_xy(virt.obsm["spatial"]),
                    value_fn(virt, True),
                    grid_n=grid_n,
                    sigma=sigma_virtual,
                )
                note = "ok"
            except FileNotFoundError:
                vgrid = np.full_like(rgrid, np.nan)
                note = f"missing K={k} virtual h5ad"
            grids[(stage_label, "virtual", k)] = vgrid
            pear, hot = grid_similarity(rgrid, vgrid) if np.isfinite(vgrid).any() else (np.nan, np.nan)
            rows.append(
                {
                    "stage": stage_label,
                    "agent_count": k,
                    "map_type": "state_or_module",
                    "feature": feature_name,
                    "spatial_pearson": pear,
                    "hotspot_overlap": hot,
                    "notes": note,
                }
            )
    return grids, rows


def make_state_figure(out: Path, k_values: list[int], fig_dir: Path, summary_rows: list[dict]) -> Path:
    suffix = "_".join(f"K{k}" for k in k_values)
    nrows = len(STATE_FEATURES) * len(STAGE_PAIRS)
    fig, axes = plt.subplots(nrows, 1 + len(k_values), figsize=(3.5 * (1 + len(k_values)), 3.1 * nrows), dpi=220)
    if nrows == 1:
        axes = np.asarray([axes])
    row = 0
    for feat_name, feat in STATE_FEATURES.items():
        grids, rows = build_grids(
            out,
            feat_name,
            lambda a, virtual, feat=feat: values_state(a, feat, virtual),
            k_values,
            grid_n=95,
            sigma_real=1.6,
            sigma_virtual=4.0,
        )
        summary_rows.extend(rows)
        for _, _, stage_label in STAGE_PAIRS:
            plot_grid(axes[row, 0], grids[(stage_label, "real", "real")], f"Real {stage_label}\n{feat_name}", "viridis")
            for j, k in enumerate(k_values, start=1):
                plot_grid(axes[row, j], grids[(stage_label, "virtual", k)], f"K={k} virtual\n{feat_name}", "viridis")
            row += 1
    fig.suptitle(f"GSE267904 spatial cell-state abundance recovery: real vs {' vs '.join(f'K{k}' for k in k_values)}", fontsize=16, weight="bold", y=0.995)
    fig.text(0.5, 0.01, "Maps are normalized smoothed hotspots. Real = Visium spot-level; virtual = PhysiCell micro-agent KDE/grid smoothing; not spot-to-spot matching.", ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=[0, 0.025, 1, 0.985])
    p = fig_dir / f"gse267904_spatial_state_abundance_{suffix}.png"
    fig.savefig(p, dpi=260)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def make_module_figure(out: Path, k_values: list[int], fig_dir: Path, summary_rows: list[dict]) -> Path:
    suffix = "_".join(f"K{k}" for k in k_values)
    nrows = len(MODULE_GENES) * len(STAGE_PAIRS)
    fig, axes = plt.subplots(nrows, 1 + len(k_values), figsize=(3.5 * (1 + len(k_values)), 2.8 * nrows), dpi=220)
    if nrows == 1:
        axes = np.asarray([axes])
    row = 0
    for module, genes in MODULE_GENES.items():
        grids, rows = build_grids(
            out,
            module,
            lambda a, virtual, genes=genes: values_module(a, genes),
            k_values,
            grid_n=95,
            sigma_real=1.6,
            sigma_virtual=4.0,
        )
        for r in rows:
            r["map_type"] = "module_hotspot"
        summary_rows.extend(rows)
        for _, _, stage_label in STAGE_PAIRS:
            plot_grid(axes[row, 0], grids[(stage_label, "real", "real")], f"Real {stage_label}\n{module}", "magma")
            for j, k in enumerate(k_values, start=1):
                plot_grid(axes[row, j], grids[(stage_label, "virtual", k)], f"K={k} virtual\n{module}", "magma")
            row += 1
    fig.suptitle("GSE267904 spatial module-expression hotspot recovery", fontsize=16, weight="bold", y=0.997)
    fig.text(0.5, 0.008, "Module maps are normalized per feature/stage to compare spatial hotspot patterns rather than absolute expression scale.", ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=[0, 0.02, 1, 0.988])
    p = fig_dir / f"gse267904_spatial_module_hotspots_{suffix}.png"
    fig.savefig(p, dpi=260)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def read_sender_receiver_edges(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"stage", "sender", "receiver", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    return df


def strength_map_from_edges(
    edges: pd.DataFrame,
    stage: str,
    mode: str,
    pathway_patterns: list[str] | None = None,
) -> dict[str, float]:
    d = edges[edges["stage"].astype(str).eq(stage)].copy()
    if pathway_patterns and "pathway" in d.columns:
        pattern = "|".join(pathway_patterns)
        d = d[d["pathway"].astype(str).str.contains(pattern, case=False, na=False, regex=True)]
    if d.empty:
        return {}
    key = "sender" if mode == "sender" else "receiver"
    return d.groupby(key)["weight"].sum().to_dict()


def values_projected_commot(adata: ad.AnnData, strength: dict[str, float], virtual: bool) -> np.ndarray:
    label_col = "cell_type" if virtual else "dominant_cell_type"
    labels = adata.obs[label_col].astype(str)
    return minmax01(labels.map(lambda x: float(strength.get(x, 0.0))).to_numpy())


def build_commot_grids(
    out: Path,
    k_values: list[int],
    mode: str,
    pathway_name: str,
    pathway_patterns: list[str] | None,
    grid_n: int,
) -> tuple[dict, list[dict]]:
    grids: dict[tuple[str, str, int | str], np.ndarray] = {}
    rows: list[dict] = []
    real_edges = read_sender_receiver_edges(out / "shared_real" / "real_commot" / "real_commot_edges_long.csv")
    # Fallback for older runs.
    if real_edges.empty:
        real_edges = read_sender_receiver_edges(out / "K_50" / "real_commot" / "real_commot_edges_long.csv")
    for real_stage, virtual_stage, stage_label in STAGE_PAIRS:
        real = load_real(out, real_stage)
        r_strength = strength_map_from_edges(real_edges, real_stage, mode, pathway_patterns)
        rgrid = smooth_grid(
            norm_xy(real.obsm["spatial"]),
            values_projected_commot(real, r_strength, virtual=False),
            grid_n=grid_n,
            sigma=1.6,
        )
        grids[(stage_label, "real", "real")] = rgrid
        for k in k_values:
            try:
                virt = load_virtual(out, k, virtual_stage)
                virt_edges = read_sender_receiver_edges(out / f"K_{k}" / "virtual_commot" / "virtual_commot_edges_long.csv")
                v_strength = strength_map_from_edges(virt_edges, virtual_stage, mode, pathway_patterns)
                if pathway_patterns and (not r_strength or not v_strength):
                    note = "pathway skipped/weak because insufficient real or virtual edges"
                else:
                    note = "ok"
                vgrid = smooth_grid(
                    norm_xy(virt.obsm["spatial"]),
                    values_projected_commot(virt, v_strength, virtual=True),
                    grid_n=grid_n,
                    sigma=4.0,
                )
            except FileNotFoundError:
                vgrid = np.full_like(rgrid, np.nan)
                note = f"missing K={k} virtual input or COMMOT edges"
            grids[(stage_label, "virtual", k)] = vgrid
            pear, hot = grid_similarity(rgrid, vgrid) if np.isfinite(vgrid).any() else (np.nan, np.nan)
            rows.append(
                {
                    "stage": stage_label,
                    "agent_count": k,
                    "map_type": f"commot_{mode}",
                    "feature": pathway_name,
                    "spatial_pearson": pear,
                    "hotspot_overlap": hot,
                    "notes": note,
                }
            )
    return grids, rows


def make_commot_hotspot_figure(out: Path, k_values: list[int], fig_dir: Path, summary_rows: list[dict]) -> Path:
    suffix = "_".join(f"K{k}" for k in k_values)
    specs = [("Total sender", "sender", None), ("Total receiver", "receiver", None)]
    nrows = len(specs) * len(STAGE_PAIRS)
    fig, axes = plt.subplots(nrows, 1 + len(k_values), figsize=(3.6 * (1 + len(k_values)), 3.0 * nrows), dpi=220)
    if nrows == 1:
        axes = np.asarray([axes])
    row = 0
    for feat_name, mode, patterns in specs:
        grids, rows = build_commot_grids(out, k_values, mode, feat_name, patterns, grid_n=95)
        summary_rows.extend(rows)
        cmap = "YlOrRd" if mode == "sender" else "YlGnBu"
        for _, _, stage_label in STAGE_PAIRS:
            plot_grid(axes[row, 0], grids[(stage_label, "real", "real")], f"Real {stage_label}\n{feat_name}", cmap)
            for j, k in enumerate(k_values, start=1):
                plot_grid(axes[row, j], grids[(stage_label, "virtual", k)], f"K={k} virtual\n{feat_name}", cmap)
            row += 1
    fig.suptitle("GSE267904 spatial COMMOT sender/receiver hotspot recovery", fontsize=16, weight="bold", y=0.995)
    fig.text(0.5, 0.012, "COMMOT outgoing/incoming strengths are aggregated by cell type and projected to real spots or virtual micro-agents, then smoothed on the same normalized coordinate grid.", ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])
    p = fig_dir / f"gse267904_spatial_commot_hotspots_{suffix}.png"
    fig.savefig(p, dpi=260)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def make_pathway_figure(out: Path, k_values: list[int], fig_dir: Path, summary_rows: list[dict]) -> Path:
    suffix = "_".join(f"K{k}" for k in k_values)
    available: list[tuple[str, list[str]]] = []
    real_edges = read_sender_receiver_edges(out / "shared_real" / "real_commot" / "real_commot_edges_long.csv")
    for fam, patterns in PATHWAY_FAMILIES.items():
        pattern = "|".join(patterns)
        if real_edges["pathway"].astype(str).str.contains(pattern, case=False, na=False, regex=True).any():
            available.append((fam, patterns))
        else:
            summary_rows.append(
                {
                    "stage": "all",
                    "agent_count": np.nan,
                    "map_type": "commot_pathway",
                    "feature": fam,
                    "spatial_pearson": np.nan,
                    "hotspot_overlap": np.nan,
                    "notes": "pathway skipped because no real COMMOT edges were found",
                }
            )
    if not available:
        raise RuntimeError("No requested pathway families found in real COMMOT edges.")

    # Keep this figure readable: sender and receiver are separate rows.
    specs = []
    for fam, patterns in available:
        specs.append((f"{fam} sender", "sender", patterns))
        specs.append((f"{fam} receiver", "receiver", patterns))

    nrows = len(specs) * len(STAGE_PAIRS)
    fig, axes = plt.subplots(nrows, 1 + len(k_values), figsize=(3.5 * (1 + len(k_values)), 2.7 * nrows), dpi=220)
    if nrows == 1:
        axes = np.asarray([axes])
    row = 0
    for feat_name, mode, patterns in specs:
        grids, rows = build_commot_grids(out, k_values, mode, feat_name, patterns, grid_n=95)
        for r in rows:
            r["map_type"] = f"commot_pathway_{mode}"
        summary_rows.extend(rows)
        cmap = "YlOrRd" if mode == "sender" else "YlGnBu"
        for _, _, stage_label in STAGE_PAIRS:
            plot_grid(axes[row, 0], grids[(stage_label, "real", "real")], f"Real {stage_label}\n{feat_name}", cmap)
            for j, k in enumerate(k_values, start=1):
                plot_grid(axes[row, j], grids[(stage_label, "virtual", k)], f"K={k} virtual\n{feat_name}", cmap)
            row += 1
    fig.suptitle("GSE267904 pathway-specific COMMOT spatial hotspot recovery", fontsize=16, weight="bold", y=0.998)
    fig.text(0.5, 0.006, "Pathway families are selected from COMMOT pathway names. SPP1/MIF are recorded as skipped if absent from the COMMOT output.", ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=[0, 0.018, 1, 0.99])
    p = fig_dir / f"gse267904_pathway_specific_commot_hotspots_{suffix}.png"
    fig.savefig(p, dpi=260)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def mean_metric(summary: pd.DataFrame, feature: str, map_type_contains: str, k: int, stage_contains: str = "d7") -> float:
    d = summary[
        summary["agent_count"].eq(k)
        & summary["feature"].astype(str).str.contains(feature, case=False, na=False)
        & summary["map_type"].astype(str).str.contains(map_type_contains, case=False, na=False)
        & summary["stage"].astype(str).str.contains(stage_contains, case=False, na=False)
    ]
    if d.empty:
        return np.nan
    return float(np.nanmean(d["spatial_pearson"]))


def make_visual_summary(out: Path, k_values: list[int], fig_dir: Path, summary_df: pd.DataFrame) -> Path:
    """Compact main-paper style panel focused on d7, where metrics are strongest."""
    suffix = "_".join(f"K{k}" for k in k_values)
    chosen = [
        ("Macrophage / recruited mono-mac", "state", "viridis"),
        ("Fibrosis / ECM", "module", "magma"),
        ("Total sender", "commot_sender", "YlOrRd"),
        ("TGFβ receiver", "commot_pathway_receiver", "YlGnBu"),
    ]
    ncols = 1 + len(k_values)
    fig = plt.figure(figsize=(4.2 * ncols, 12.2), dpi=230)
    gs = fig.add_gridspec(5, ncols, height_ratios=[1, 1, 1, 1, 0.72], hspace=0.34, wspace=0.08)

    plot_specs = []
    # Precompute by reusing the same helpers as above.
    state_grids, _ = build_grids(
        out,
        "Macrophage / recruited mono-mac",
        lambda a, virtual: values_state(a, STATE_FEATURES["Macrophage / recruited mono-mac"], virtual),
        k_values,
        grid_n=95,
        sigma_real=1.6,
        sigma_virtual=4.0,
    )
    module_grids, _ = build_grids(
        out,
        "Fibrosis / ECM",
        lambda a, virtual: values_module(a, MODULE_GENES["Fibrosis / ECM"]),
        k_values,
        grid_n=95,
        sigma_real=1.6,
        sigma_virtual=4.0,
    )
    sender_grids, _ = build_commot_grids(out, k_values, "sender", "Total sender", None, grid_n=95)
    tgf_grids, _ = build_commot_grids(out, k_values, "receiver", "TGFβ receiver", PATHWAY_FAMILIES["TGFβ"], grid_n=95)
    plot_specs = [
        ("A. Macrophage-lineage abundance", state_grids, "viridis"),
        ("B. Fibrosis / ECM module", module_grids, "magma"),
        ("C. Total COMMOT sender hotspot", sender_grids, "YlOrRd"),
        ("D. TGFβ receiver hotspot", tgf_grids, "YlGnBu"),
    ]
    cols = [("Real", ("d7 / virtual early", "real", "real"))] + [
        (f"K={k}", ("d7 / virtual early", "virtual", k)) for k in k_values
    ]
    for r, (title, grids, cmap) in enumerate(plot_specs):
        for c, (col_name, key) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            plot_grid(ax, grids[key], f"{title}\n{col_name}", cmap)

    ax = fig.add_subplot(gs[4, :])
    metric = pd.read_csv(out / "evaluation" / "spatial_agent_granularity_metrics.csv")
    metric = metric[metric["real_stage"].astype(str).eq("d7_bleo") & metric["agent_count"].isin(k_values)].copy()
    metric = metric.sort_values("agent_count")
    x = np.arange(len(metric))
    vals = metric["overall_spatial_fidelity_score"].to_numpy(dtype=float)
    best_k = int(metric.loc[metric["overall_spatial_fidelity_score"].idxmax(), "agent_count"]) if not metric.empty else None
    colors = ["#F59E0B" if int(k) == best_k else "#6BAED6" for k in metric["agent_count"]]
    ax.bar(x, vals, color=colors, alpha=0.88)
    ax.set_xticks(x, [f"K={int(k)}" for k in metric["agent_count"]])
    ax.set_ylabel("Spatial fidelity score ↑")
    ax.set_title("E. Early-stage spatial fidelity across Agent granularities", fontsize=11, weight="bold")
    for xi, yi in zip(x, vals):
        if np.isfinite(yi):
            ax.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"GSE267904 spatial granularity visualization: real vs {' vs '.join(f'K{k}' for k in k_values)}", fontsize=17, weight="bold", y=0.99)
    fig.text(0.5, 0.012, "All maps are smoothed hotspot comparisons on normalized coordinates. Real Visium spots and virtual micro-agents are not matched one-to-one.", ha="center", fontsize=9.5, color="#4b5563")
    p = fig_dir / f"gse267904_spatial_granularity_visual_summary_{suffix}.png"
    fig.savefig(p, dpi=280)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_agent_granularity"))
    parser.add_argument("--agent-counts", default="20,50")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    k_values = [int(x) for x in str(args.agent_counts).split(",") if str(x).strip()]
    fig_dir = out / "figures" / "spatial_maps"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    written = []
    written.append(make_state_figure(out, k_values, fig_dir, summary_rows))
    written.append(make_module_figure(out, k_values, fig_dir, summary_rows))
    written.append(make_commot_hotspot_figure(out, k_values, fig_dir, summary_rows))
    written.append(make_pathway_figure(out, k_values, fig_dir, summary_rows))
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "evaluation" / "spatial_visualization_summary.csv"
    summary.to_csv(summary_path, index=False)
    written.append(make_visual_summary(out, k_values, fig_dir, summary))

    manifest = {
        "note": "Spatial maps compare smoothed hotspot patterns, not spot-to-cell matching.",
        "agent_counts": k_values,
        "figures": [str(p) for p in written],
        "summary_csv": str(summary_path),
    }
    manifest_path = fig_dir / "spatial_visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Wrote GSE267904 spatial granularity figures:")
    for p in written:
        print(f"- {p}")
    print(f"Summary CSV: {summary_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
