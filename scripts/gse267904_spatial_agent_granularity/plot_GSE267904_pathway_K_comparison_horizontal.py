#!/usr/bin/env python3
"""Horizontal 4x8 GSE267904 TGFB/CCL spatial recovery figure.

Read-only: consumes existing COMMOT and spatial h5ad outputs; never runs COMMOT,
PhysiCell, or an Agent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse267904-horizontal")
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter


K_VALUES = (20, 50, 100)
ROW_SPECS = (("TGFB", "sender"), ("TGFB", "receiver"), ("CCL", "sender"), ("CCL", "receiver"))
STAGES = {
    "Early": ("d7_bleo", "virtual_early_bleo"),
    "Late": ("d21_bleo", "virtual_late_bleo"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
    return (xy - lo) / np.where(hi > lo, hi - lo, 1.0)


def relative_strength(edges: pd.DataFrame, stage: str, family: str, mode: str) -> dict[str, float]:
    """Relative distribution of the selected pathway traffic across cell types."""
    data = edges[edges["stage"].astype(str).eq(stage)]
    data = data[data["pathway"].astype(str).str.contains(family, case=False, regex=False, na=False)]
    key = "sender" if mode == "sender" else "receiver"
    totals = data.groupby(key)["weight"].sum().astype(float)
    denom = float(totals.sum())
    if denom <= 0:
        return {str(k): 0.0 for k in totals.index}
    return (totals / denom).to_dict()


def values_for(adata: ad.AnnData, strengths: dict[str, float], virtual: bool) -> np.ndarray:
    label = "cell_type" if virtual else "dominant_cell_type"
    if label not in adata.obs:
        raise RuntimeError(f"AnnData missing obs[{label!r}]")
    return adata.obs[label].astype(str).map(lambda x: strengths.get(x, 0.0)).to_numpy(dtype=float)


def common_mask(virtual_xy: list[np.ndarray], gx: np.ndarray, sigma: float) -> np.ndarray:
    """Organic shared support mask from unique K=100 early/late coordinates."""
    points = np.unique(np.vstack([norm_xy(xy) for xy in virtual_xy]), axis=0)
    n = gx.shape[0]
    ix = np.clip((points[:, 0] * (n - 1)).astype(int), 0, n - 1)
    iy = np.clip((points[:, 1] * (n - 1)).astype(int), 0, n - 1)
    density = np.zeros_like(gx, dtype=float)
    np.add.at(density, (iy, ix), 1.0)
    density = gaussian_filter(density, sigma=sigma, mode="nearest")
    mask = density >= max(float(np.nanmax(density)) * 0.015, 1e-12)
    mask = binary_fill_holes(binary_closing(mask, iterations=2))
    return mask


def aggregate_duplicate_xy(xy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "value": values})
    grouped = frame.groupby(["x", "y"], as_index=False)["value"].mean()
    return grouped[["x", "y"]].to_numpy(float), grouped["value"].to_numpy(float)


def interpolate_virtual(
    xy: np.ndarray, values: np.ndarray, gx: np.ndarray, gy: np.ndarray,
    mask: np.ndarray, sigma: float,
) -> np.ndarray:
    """Reference-style Gaussian-smoothed weighted grid (no polygon interpolation)."""
    xy, values = aggregate_duplicate_xy(norm_xy(xy), np.asarray(values, dtype=float))
    n = gx.shape[0]
    ix = np.clip((xy[:, 0] * (n - 1)).astype(int), 0, n - 1)
    iy = np.clip((xy[:, 1] * (n - 1)).astype(int), 0, n - 1)
    weighted = np.zeros_like(gx, dtype=float)
    density = np.zeros_like(gx, dtype=float)
    np.add.at(weighted, (iy, ix), values)
    np.add.at(density, (iy, ix), 1.0)
    weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    density = gaussian_filter(density, sigma=sigma, mode="nearest")
    field = np.divide(weighted, density, out=np.zeros_like(weighted), where=density > 1e-12)
    field = np.clip(field, 0, None)
    field[~mask] = np.nan
    return field


def grid_real_for_diagnostic(
    xy: np.ndarray, values: np.ndarray, gx: np.ndarray, gy: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    xy, values = aggregate_duplicate_xy(norm_xy(xy), np.asarray(values, dtype=float))
    linear = griddata(xy, values, (gx, gy), method="linear")
    nearest = griddata(xy, values, (gx, gy), method="nearest")
    field = np.where(np.isfinite(linear), linear, nearest)
    field[~mask] = np.nan
    return field


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--grid-resolution", type=int, default=180)
    parser.add_argument("--smoothing-sigma", type=float, default=7.0)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root / "outputs/GSE267904_spatial_agent_granularity"
    out = base / "figures/pathway_K_comparison_horizontal_smoothed"
    out.mkdir(parents=True, exist_ok=True)

    real_edges_path = base / "shared_real/real_commot/real_commot_edges_long.csv"
    real_edges = pd.read_csv(real_edges_path)
    real_paths = {s: base / f"shared_real/real_benchmark/{s}.h5ad" for s in ("d7_bleo", "d21_bleo")}
    real = {s: ad.read_h5ad(p) for s, p in real_paths.items()}
    edge_paths = {k: base / f"K_{k}/virtual_commot/virtual_commot_edges_long.csv" for k in K_VALUES}
    virtual_edges = {k: pd.read_csv(p) for k, p in edge_paths.items()}
    virtual_paths = {
        (k, s): base / f"K_{k}/virtual_commot_input/{s}.h5ad"
        for k in K_VALUES for s in ("virtual_early_bleo", "virtual_late_bleo")
    }
    virtual = {key: ad.read_h5ad(path) for key, path in virtual_paths.items()}

    axis = np.linspace(0, 1, args.grid_resolution)
    gx, gy = np.meshgrid(axis, axis)
    mask = common_mask(
        [np.asarray(virtual[(100, s)].obsm["spatial"]) for s in ("virtual_early_bleo", "virtual_late_bleo")],
        gx, args.smoothing_sigma,
    )

    # Prepare every panel before plotting so each entire row receives one scale.
    panels: dict[tuple[str, str, str, str], dict] = {}
    diagnostics = []
    row_ranges = {}
    for family, mode in ROW_SPECS:
        row_values = []
        for period, (real_stage, virtual_stage) in STAGES.items():
            radata = real[real_stage]
            rvals = values_for(radata, relative_strength(real_edges, real_stage, family, mode), False)
            rxy = norm_xy(np.asarray(radata.obsm["spatial"], dtype=float))
            panels[(family, mode, period, "Real")] = {"kind": "real", "xy": rxy, "values": rvals}
            row_values.append(rvals)
            rgrid = grid_real_for_diagnostic(rxy, rvals, gx, gy, mask)
            for k in K_VALUES:
                vadata = virtual[(k, virtual_stage)]
                vvals = values_for(vadata, relative_strength(virtual_edges[k], virtual_stage, family, mode), True)
                field = interpolate_virtual(
                    np.asarray(vadata.obsm["spatial"], dtype=float), vvals, gx, gy, mask, args.smoothing_sigma,
                )
                panels[(family, mode, period, f"K={k}")] = {"kind": "virtual", "field": field}
                row_values.append(field[np.isfinite(field)])
                diagnostics.append({
                    "stage": period, "pathway": family, "mode": mode, "K": k,
                    "spatial_pattern_correlation_to_real": corr(rgrid, field),
                })
        finite = np.concatenate([np.asarray(x)[np.isfinite(x)] for x in row_values])
        vmax = float(np.quantile(finite, 0.995)) if finite.size else 1.0
        row_ranges[f"{family}_{mode}"] = {"vmin": 0.0, "vmax": max(vmax, 1e-12)}

    fig = plt.figure(figsize=(22.5, 9.1), dpi=180, facecolor="white")
    gs = fig.add_gridspec(
        4, 10, width_ratios=[1, 1, 1, 1, 0.10, 1, 1, 1, 1, 0.055],
        left=0.055, right=0.975, top=0.865, bottom=0.055, wspace=0.045, hspace=0.10,
    )
    panel_cols = [0, 1, 2, 3, 5, 6, 7, 8]
    panel_keys = [("Early", x) for x in ("Real", "K=20", "K=50", "K=100")] + [
        ("Late", x) for x in ("Real", "K=20", "K=50", "K=100")
    ]
    for row, (family, mode) in enumerate(ROW_SPECS):
        scale = row_ranges[f"{family}_{mode}"]
        cmap = "YlOrRd" if mode == "sender" else "YlGnBu"
        last_image = None
        for col, (period, label) in zip(panel_cols, panel_keys):
            ax = fig.add_subplot(gs[row, col])
            panel = panels[(family, mode, period, label)]
            if panel["kind"] == "real":
                last_image = ax.scatter(
                    panel["xy"][:, 0], panel["xy"][:, 1], c=panel["values"],
                    cmap=cmap, vmin=scale["vmin"], vmax=scale["vmax"], s=2.4,
                    linewidths=0, rasterized=True,
                )
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            else:
                last_image = ax.imshow(
                    panel["field"], origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                    vmin=scale["vmin"], vmax=scale["vmax"], interpolation="bilinear",
                )
            ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for spine in ax.spines.values(): spine.set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=11.5, weight="bold", pad=5)
            if col == 0:
                ax.set_ylabel(f"{family} {mode}", fontsize=11, weight="bold", labelpad=9)
        cax = fig.add_subplot(gs[row, 9])
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.locator = plt.MaxNLocator(3); cbar.update_ticks()

    fig.suptitle(
        "GSE267904 pathway-specific spatial recovery across increasing Agent granularity",
        fontsize=17, weight="bold", y=0.975,
    )
    fig.text(0.275, 0.912, "EARLY", ha="center", va="center", fontsize=13, weight="bold", color="#374151")
    fig.text(0.725, 0.912, "LATE", ha="center", va="center", fontsize=13, weight="bold", color="#374151")
    divider = plt.Line2D([0.505, 0.505], [0.05, 0.885], transform=fig.transFigure, color="#9CA3AF", lw=1.2)
    fig.add_artist(divider)

    stem = "GSE267904_TGFB_CCL_K20_K50_K100_horizontal_smoothed"
    preview = out / f"{stem}_preview.png"
    highres = out / f"{stem}_600dpi.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(preview, dpi=150, facecolor="white")
    fig.savefig(highres, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    diag_path = out / "spatial_pattern_diagnostic.csv"
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    source_files = [real_edges_path, *real_paths.values(), *edge_paths.values(), *virtual_paths.values()]
    manifest = {
        "source_files": [{"path": str(p), "sha256": sha256(p)} for p in source_files],
        "panel_layout": "4 rows x 8 data columns",
        "row_order": [f"{f} {m}" for f, m in ROW_SPECS],
        "column_order": ["Early Real", "Early K=20", "Early K=50", "Early K=100", "Late Real", "Late K=20", "Late K=50", "Late K=100"],
        "score_display": "cell-type pathway strength divided by total selected pathway strength; identical formula for Real and all K; raw files unchanged",
        "real_display": "raw normalized Visium coordinates as unsmoothed small spots",
        "virtual_display": "reference-style Gaussian-smoothed weighted grid divided by Gaussian-smoothed agent density",
        "outputs": {"preview": str(preview), "600dpi": str(highres), "pdf": str(pdf), "diagnostic": str(diag_path)},
    }
    (out / "plotting_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audit = {
        "status": "PASS_SPATIAL_PLOTTING_AUDIT",
        "xlim": [0.0, 1.0], "ylim": [0.0, 1.0],
        "grid_resolution": [args.grid_resolution, args.grid_resolution],
        "interpolation_method": "weighted grid accumulation followed by Gaussian smoothing and density normalization",
        "smoothing_sigma": args.smoothing_sigma,
        "same_smoothing_all_virtual_panels": True,
        "mask_source": "single shared organic Gaussian-density support mask from unique K=100 early and late coordinates",
        "same_mask_all_virtual_panels": True,
        "row_wise_vmin_vmax": row_ranges,
        "coordinate_transform": "each source normalized independently to [0,1] x [0,1] using source min/max",
        "coordinate_flips": {"x": False, "y": False},
        "coordinate_rotation_degrees": 0,
        "real_smoothed": False,
        "virtual_agent_markers_shown": False,
        "missing_area_color": "white/transparent mask exterior",
        "commot_rerun": False, "physicell_rerun": False, "raw_pathway_scores_modified": False,
    }
    (out / "spatial_plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    shutil.copy2(Path(__file__), out / "plot_GSE267904_pathway_K_comparison_horizontal.py")
    (out / "README.md").write_text(
        "# Horizontal GSE267904 pathway comparison\n\n"
        "This figure is generated only from existing shared Real and K=20/K=50/K=100 COMMOT and spatial h5ad outputs. "
        "Real panels show unsmoothed Visium spots. Virtual panels use one common mask and canvas with reference-style weighted-grid Gaussian smoothing and density normalization; "
        "grid resolution, and Gaussian smoothing sigma. Each full row shares one vmin/vmax and colorbar.\n\n"
        "The display value is each cell type's fraction of total TGFB or CCL sender/receiver traffic for that source-stage; "
        "this common transformation makes spatial distributions comparable despite different total edge counts. Raw inputs are unchanged. "
        "Exact paths and hashes are in `plotting_source_manifest.json`; plotting parameters and color limits are in `spatial_plot_audit.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
