#!/usr/bin/env python3
"""Plot K=100 pathway-specific spatial COMMOT maps with smoothed virtual fields."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


STAGE_ROWS = [
    ("real", "d7_bleo", "early\nreal"),
    ("virtual", "virtual_early_bleo", "early\nvirtual"),
    ("real", "d21_bleo", "late\nreal"),
    ("virtual", "virtual_late_bleo", "late\nvirtual"),
]

PATHWAY_FAMILIES = ["TGFB", "CCL", "CXCL"]
MODES = ["sender", "receiver"]


def resolve_out(project_root: Path, out_dir: Path) -> Path:
    return out_dir if out_dir.is_absolute() else project_root / out_dir


def read_adata(out: Path, stage: str, source: str) -> ad.AnnData:
    if source == "virtual":
        path = out / "virtual_commot_input" / f"{stage}.h5ad"
    else:
        path = out / "real_benchmark" / f"{stage}.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def read_edges(out: Path, source: str) -> pd.DataFrame:
    if source == "virtual":
        path = out / "virtual_commot" / "virtual_commot_edges_long.csv"
    else:
        path = out / "real_commot" / "real_commot_edges_long.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    edges = pd.read_csv(path)
    required = {"stage", "sender", "receiver", "pathway", "weight"}
    missing = required - set(edges.columns)
    if missing:
        raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    return edges


def pathway_strength_by_type(edges: pd.DataFrame, stage: str, family: str, mode: str) -> dict[str, float]:
    data = edges[edges["stage"].astype(str).eq(stage)].copy()
    data = data[data["pathway"].astype(str).str.contains(family, case=False, na=False, regex=False)]
    if data.empty:
        return {}
    group_col = "sender" if mode == "sender" else "receiver"
    return data.groupby(group_col)["weight"].sum().to_dict()


def projected_scores(adata: ad.AnnData, source: str, strengths: dict[str, float]) -> np.ndarray:
    label_col = "cell_type" if source == "virtual" else "dominant_cell_type"
    if label_col not in adata.obs:
        raise RuntimeError(f"AnnData obs is missing {label_col!r}")
    labels = adata.obs[label_col].astype(str)
    return labels.map(lambda x: float(strengths.get(x, 0.0))).to_numpy(dtype=float)


def expanded_extent(xy: np.ndarray, pad_frac: float = 0.035) -> tuple[float, float, float, float]:
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    span = np.where((hi - lo) <= 0, 1.0, hi - lo)
    lo = lo - span * pad_frac
    hi = hi + span * pad_frac
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def smooth_weighted_grid(
    xy: np.ndarray,
    values: np.ndarray,
    grid_n: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Weighted spatial binning followed by Gaussian smoothing, preserving score scale."""
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(values)
    xy = xy[ok]
    values = values[ok]
    extent = expanded_extent(xy)
    xmin, xmax, ymin, ymax = extent
    if len(values) == 0:
        empty = np.full((grid_n, grid_n), np.nan)
        return empty, np.zeros_like(empty), extent

    x01 = (xy[:, 0] - xmin) / max(xmax - xmin, 1e-12)
    y01 = (xy[:, 1] - ymin) / max(ymax - ymin, 1e-12)
    ix = np.clip((x01 * (grid_n - 1)).astype(int), 0, grid_n - 1)
    iy = np.clip((y01 * (grid_n - 1)).astype(int), 0, grid_n - 1)

    weighted_sum = np.zeros((grid_n, grid_n), dtype=float)
    density = np.zeros((grid_n, grid_n), dtype=float)
    np.add.at(weighted_sum, (iy, ix), values)
    np.add.at(density, (iy, ix), 1.0)

    weighted_sum = gaussian_filter(weighted_sum, sigma=sigma, mode="nearest")
    density = gaussian_filter(density, sigma=sigma, mode="nearest")
    grid = np.divide(weighted_sum, density, out=np.full_like(weighted_sum, np.nan), where=density > 1e-10)

    finite_density = density[np.isfinite(density)]
    if finite_density.size:
        threshold = max(float(np.nanmax(finite_density)) * 0.015, 1e-12)
        grid = np.where(density >= threshold, grid, np.nan)
    return grid, density, extent


def plot_real_points(ax, xy: np.ndarray, scores: np.ndarray, title: str, cmap: str):
    vmax = max(float(np.nanmax(scores)) if scores.size else 0.0, 1.0)
    image = ax.scatter(xy[:, 0], xy[:, 1], c=scores, s=2.0, cmap=cmap, vmin=0, vmax=vmax, alpha=0.93, linewidths=0)
    ax.set_title(title, fontsize=9.5, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return image, vmax


def plot_virtual_smoothed(ax, xy: np.ndarray, scores: np.ndarray, title: str, cmap: str, grid_n: int, sigma: float):
    grid, density, extent = smooth_weighted_grid(xy, scores, grid_n=grid_n, sigma=sigma)
    vmax = max(float(np.nanmax(grid)) if np.isfinite(grid).any() else 0.0, 1.0)
    image = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.scatter(xy[:, 0], xy[:, 1], s=5, c="#111827", alpha=0.16, linewidths=0)
    ax.set_title(title, fontsize=9.5, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return image, vmax, grid, density


def make_figure(out: Path, fig_dir: Path, grid_n: int, sigma_virtual: float) -> tuple[Path, Path]:
    real_edges = read_edges(out, "real")
    virtual_edges = read_edges(out, "virtual")
    edge_by_source = {"real": real_edges, "virtual": virtual_edges}
    adata_cache: dict[tuple[str, str], ad.AnnData] = {}

    fig, axes = plt.subplots(4, 6, figsize=(21, 13), dpi=220)
    summary_rows: list[dict] = []

    for row_idx, (source, stage, row_label) in enumerate(STAGE_ROWS):
        adata = adata_cache.setdefault((source, stage), read_adata(out, stage, source))
        xy = np.asarray(adata.obsm["spatial"], dtype=float)
        edges = edge_by_source[source]

        for fam_idx, family in enumerate(PATHWAY_FAMILIES):
            for mode_idx, mode in enumerate(MODES):
                col_idx = fam_idx * 2 + mode_idx
                ax = axes[row_idx, col_idx]
                cmap = "YlOrRd" if mode == "sender" else "YlGnBu"
                strengths = pathway_strength_by_type(edges, stage, family, mode)
                scores = projected_scores(adata, source, strengths)
                title = f"{family} {mode}\n{stage}"

                if source == "virtual":
                    image, vmax, grid, density = plot_virtual_smoothed(
                        ax,
                        xy,
                        scores,
                        title,
                        cmap,
                        grid_n=grid_n,
                        sigma=sigma_virtual,
                    )
                    finite_grid = grid[np.isfinite(grid)]
                    nonzero_cells = int(np.count_nonzero(finite_grid > 0)) if finite_grid.size else 0
                else:
                    image, vmax = plot_real_points(ax, xy, scores, title, cmap)
                    nonzero_cells = int(np.count_nonzero(scores > 0))

                cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.012)
                cbar.ax.tick_params(labelsize=7)
                ax.set_ylabel(row_label if col_idx == 0 else "", fontsize=12, weight="bold")
                summary_rows.append(
                    {
                        "source": source,
                        "stage": stage,
                        "family": family,
                        "mode": mode,
                        "n_observations": int(adata.n_obs),
                        "n_types_with_strength": int(len(strengths)),
                        "n_nonzero_projected_points_or_grid_cells": nonzero_cells,
                        "max_projected_or_smoothed_score": vmax,
                        "virtual_grid_n": grid_n if source == "virtual" else np.nan,
                        "virtual_gaussian_sigma": sigma_virtual if source == "virtual" else np.nan,
                    }
                )

    fig.suptitle(
        "GSE267904 K=100 pathway-specific spatial COMMOT hotspots: real spots and smoothed virtual agent fields",
        fontsize=16,
        weight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.018,
        "Real panels use Visium spot coordinates. Virtual panels use K=100 PhysiCell micro-agent coordinates and Gaussian-smoothed weighted grids; faint dots mark available virtual agents.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0.02, 0.035, 0.985, 0.965], h_pad=1.0, w_pad=1.25)

    png = fig_dir / "gse267904_k100_pathway_specific_spatial_commot_hotspots_smoothed.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary_path = fig_dir / "gse267904_k100_pathway_specific_spatial_commot_hotspots_smoothed_manifest.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "summary_csv": str(summary_path),
        "out_dir": str(out),
        "virtual_smoothing": {
            "method": "weighted grid binning followed by scipy.ndimage.gaussian_filter",
            "grid_n": grid_n,
            "sigma": sigma_virtual,
            "scale": "absolute projected COMMOT strength, not min-max normalized",
        },
    }
    (fig_dir / "gse267904_k100_pathway_specific_spatial_commot_hotspots_smoothed_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_agent_granularity/K_100"))
    parser.add_argument("--grid-n", type=int, default=170)
    parser.add_argument("--sigma-virtual", type=float, default=7.0)
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = make_figure(out, fig_dir, grid_n=args.grid_n, sigma_virtual=args.sigma_virtual)
    print("Wrote K=100 smoothed pathway COMMOT figure:")
    print(f"- {png}")
    print(f"- {pdf}")


if __name__ == "__main__":
    main()
