#!/usr/bin/env python3
"""Create the 8x4 TGFB/CCL Real–K20–K50–K100 spatial comparison.

Read-only plotting utility: it does not run COMMOT, PhysiCell, or any Agent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse267904-k-comparison")
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


K_VALUES = (20, 50, 100)
ROWS = [
    ("early", "d7_bleo", "virtual_early_bleo", "TGFB", "sender"),
    ("early", "d7_bleo", "virtual_early_bleo", "TGFB", "receiver"),
    ("early", "d7_bleo", "virtual_early_bleo", "CCL", "sender"),
    ("early", "d7_bleo", "virtual_early_bleo", "CCL", "receiver"),
    ("late", "d21_bleo", "virtual_late_bleo", "TGFB", "sender"),
    ("late", "d21_bleo", "virtual_late_bleo", "TGFB", "receiver"),
    ("late", "d21_bleo", "virtual_late_bleo", "CCL", "sender"),
    ("late", "d21_bleo", "virtual_late_bleo", "CCL", "receiver"),
]


def norm_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
    return (xy - lo) / np.where(hi > lo, hi - lo, 1.0)


def robust01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def spatial_grid(xy: np.ndarray, values: np.ndarray, grid_n: int, sigma: float) -> np.ndarray:
    """Identical weighted-bin/Gaussian procedure for Real and every K."""
    xy, values = norm_xy(xy), robust01(values)
    ix = np.clip((xy[:, 0] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    iy = np.clip((xy[:, 1] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    weighted = np.zeros((grid_n, grid_n), dtype=float)
    density = np.zeros_like(weighted)
    np.add.at(weighted, (iy, ix), values)
    np.add.at(density, (iy, ix), 1.0)
    weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    density = gaussian_filter(density, sigma=sigma, mode="nearest")
    grid = np.divide(weighted, density, out=np.full_like(weighted, np.nan), where=density > 1e-10)
    # Absolute support threshold avoids hiding isolated agents when another
    # coordinate contains many coincident workers (notably in the K=100 file).
    grid[density < 1e-4] = np.nan
    return grid


def pathway_strength(edges: pd.DataFrame, stage: str, family: str, mode: str) -> dict[str, float]:
    subset = edges[edges["stage"].astype(str).eq(stage)]
    subset = subset[subset["pathway"].astype(str).str.contains(family, case=False, regex=False, na=False)]
    key = "sender" if mode == "sender" else "receiver"
    return subset.groupby(key)["weight"].sum().astype(float).to_dict()


def projected_values(adata: ad.AnnData, strengths: dict[str, float], virtual: bool) -> np.ndarray:
    label = "cell_type" if virtual else "dominant_cell_type"
    if label not in adata.obs:
        raise RuntimeError(f"Missing {label!r} in AnnData.obs")
    return adata.obs[label].astype(str).map(lambda x: strengths.get(x, 0.0)).to_numpy(dtype=float)


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--grid-n", type=int, default=120)
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root / "outputs/GSE267904_spatial_agent_granularity"
    out = base / "figures/pathway_K_comparison"
    out.mkdir(parents=True, exist_ok=True)

    real_edges_path = base / "shared_real/real_commot/real_commot_edges_long.csv"
    real_edges = pd.read_csv(real_edges_path)
    virtual_edges = {
        k: pd.read_csv(base / f"K_{k}/virtual_commot/virtual_commot_edges_long.csv") for k in K_VALUES
    }
    real_adata = {
        stage: ad.read_h5ad(base / f"shared_real/real_benchmark/{stage}.h5ad")
        for stage in ("d7_bleo", "d21_bleo")
    }
    virtual_adata = {
        (k, stage): ad.read_h5ad(base / f"K_{k}/virtual_commot_input/{stage}.h5ad")
        for k in K_VALUES for stage in ("virtual_early_bleo", "virtual_late_bleo")
    }

    fig, axes = plt.subplots(8, 4, figsize=(12.2, 18.4), dpi=180)
    columns = ("Real", "K=20", "K=50", "K=100")
    diagnostics, inputs = [], [str(real_edges_path)]
    cmap = "magma"

    for row, (period, real_stage, virtual_stage, family, mode) in enumerate(ROWS):
        real_strength = pathway_strength(real_edges, real_stage, family, mode)
        radata = real_adata[real_stage]
        rgrid = spatial_grid(
            np.asarray(radata.obsm["spatial"]), projected_values(radata, real_strength, False),
            args.grid_n, args.sigma,
        )
        grids = [rgrid]
        for k in K_VALUES:
            strengths = pathway_strength(virtual_edges[k], virtual_stage, family, mode)
            vadata = virtual_adata[(k, virtual_stage)]
            grids.append(spatial_grid(
                np.asarray(vadata.obsm["spatial"]), projected_values(vadata, strengths, True),
                args.grid_n, args.sigma,
            ))
            inputs.extend([
                str(base / f"K_{k}/virtual_commot/virtual_commot_edges_long.csv"),
                str(base / f"K_{k}/virtual_commot_input/{virtual_stage}.h5ad"),
            ])

        for col, (ax, grid) in enumerate(zip(axes[row], grids)):
            image = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for spine in ax.spines.values(): spine.set_visible(False)
            if row == 0:
                ax.set_title(columns[col], fontsize=12, weight="bold", pad=7)
        axes[row, 0].set_ylabel(f"{family} {mode}", fontsize=10.5, weight="bold", labelpad=8)
        for k, grid in zip(K_VALUES, grids[1:]):
            diagnostics.append({
                "period": period, "pathway": family, "mode": mode, "K": k,
                "spatial_pattern_correlation_to_real": correlation(rgrid, grid),
            })

    fig.suptitle(
        "GSE267904 pathway-specific spatial patterns across increasing Agent granularity",
        fontsize=15.5, weight="bold", y=0.995,
    )
    fig.text(0.018, 0.745, "EARLY", rotation=90, va="center", ha="center", fontsize=12, weight="bold", color="#374151")
    fig.text(0.018, 0.285, "LATE", rotation=90, va="center", ha="center", fontsize=12, weight="bold", color="#374151")
    line = plt.Line2D([0.04, 0.92], [0.505, 0.505], transform=fig.transFigure, color="#9CA3AF", lw=1.1)
    fig.add_artist(line)
    fig.subplots_adjust(left=0.105, right=0.92, top=0.955, bottom=0.025, hspace=0.08, wspace=0.06)
    cax = fig.add_axes([0.94, 0.16, 0.012, 0.68])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_ticks([0, 0.5, 1]); cbar.ax.tick_params(labelsize=8, length=2)
    cbar.set_label("Relative pathway hotspot strength", fontsize=9)

    stem = "GSE267904_TGFB_CCL_K20_K50_K100_comparison"
    preview, highres, pdf = out / f"{stem}_preview.png", out / f"{stem}_600dpi.png", out / f"{stem}.pdf"
    fig.savefig(preview, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(highres, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(out / f"{stem}_diagnostic.csv", index=False)
    monotonic = []
    for key, group in diag.groupby(["period", "pathway", "mode"], sort=False):
        vals = group.set_index("K")["spatial_pattern_correlation_to_real"]
        monotonic.append({"row": " / ".join(key), "K20_to_K50_to_K100_monotonic": bool(vals[20] <= vals[50] <= vals[100])})
    manifest = {
        "inputs": sorted(set(inputs + [str(base / f"shared_real/real_benchmark/{s}.h5ad") for s in ("d7_bleo", "d21_bleo")])),
        "panel_order": [f"{p}: {f} {m}" for p, _, _, f, m in ROWS],
        "columns": list(columns),
        "display_scale": "vmin=0, vmax=1 for all four panels in every row",
        "normalization": "identical within-panel 1st–99th percentile normalization for relative hotspot pattern comparison",
        "smoothing": f"identical weighted grid and Gaussian sigma={args.sigma}, grid_n={args.grid_n}",
        "monotonic_diagnostic": monotonic,
        "outputs": {"preview": str(preview), "600dpi": str(highres), "pdf": str(pdf)},
    }
    (out / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = out / f"{stem}_README.md"
    readme.write_text(
        "# GSE267904 TGFB/CCL K comparison\n\n"
        "This read-only figure uses existing shared Real COMMOT outputs and existing K=20, K=50, and K=100 virtual COMMOT outputs. "
        "Rows are early TGFB sender/receiver, early CCL sender/receiver, late TGFB sender/receiver, and late CCL sender/receiver; columns are Real, K=20, K=50, and K=100.\n\n"
        "All panels use the same relative-hotspot preprocessing, grid, Gaussian smoothing, colormap, and displayed 0–1 color range. "
        "No simulations or COMMOT analyses were rerun. Exact input and output paths are recorded in the adjacent manifest JSON.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
