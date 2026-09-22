#!/usr/bin/env python3
"""Horizontal module-expression figure using the original visualization style.

Reads existing h5ad files only. Produces (1) the original panel-normalized
style and (2) a diagnostic row-shared absolute-scale style.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse267904-old-module-style")
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter


MODULES = {
    "TGF-beta response": ["Tgfb1", "Tgfbr1", "Tgfbr2", "Serpine1", "Col1a1", "Fn1"],
    "Inflammation": ["Il1b", "Tnf", "Nfkbia", "Ccl2", "Cxcl2", "Cxcl10"],
    "Fibrosis / ECM": ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Acta2", "Tagln"],
}
STAGES = (
    ("d7 / virtual early", "d7_bleo", "virtual_early_bleo"),
    ("d21 / virtual late", "d21_bleo", "virtual_late_bleo"),
)
K_VALUES = (20, 50, 100)
GRID_N = 95
SIGMA_REAL = 1.6
SIGMA_VIRTUAL = 4.0


def norm_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
    return (xy - lo) / np.where((hi - lo) > 0, hi - lo, 1.0)


def minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def module_score(path: Path, genes: list[str], panel_normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    a = ad.read_h5ad(path, backed="r")
    present = [g for g in genes if g in a.var_names]
    if len(present) != len(genes):
        a.file.close()
        raise RuntimeError(f"{path} missing genes: {sorted(set(genes) - set(present))}")
    x = a[:, present].to_memory().X
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    x = np.asarray(x, dtype=float)
    if x.size and np.nanmax(x) > 20:
        x = np.log1p(np.clip(x, 0, None))
    values = np.nanmean(x, axis=1)
    if panel_normalize:
        values = minmax01(values)
    xy = norm_xy(np.asarray(a.obsm["spatial"], dtype=float))
    a.file.close()
    return xy, values


def original_smooth_grid(
    xy: np.ndarray, values: np.ndarray, sigma: float, panel_normalize: bool,
) -> np.ndarray:
    ix = np.clip((xy[:, 0] * (GRID_N - 1)).astype(int), 0, GRID_N - 1)
    iy = np.clip((xy[:, 1] * (GRID_N - 1)).astype(int), 0, GRID_N - 1)
    weighted = np.zeros((GRID_N, GRID_N), dtype=float)
    count = np.zeros_like(weighted)
    np.add.at(weighted, (iy, ix), values)
    np.add.at(count, (iy, ix), 1.0)
    weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    count = gaussian_filter(count, sigma=sigma, mode="nearest")
    grid = np.divide(weighted, count, out=np.zeros_like(weighted), where=count > 1e-9)
    return minmax01(grid) if panel_normalize else grid


def build_panels(base: Path, panel_normalize: bool) -> tuple[dict, dict]:
    real_paths = {s: base / f"shared_real/real_benchmark/{s}.h5ad" for _, s, _ in STAGES}
    virtual_paths = {
        (k, s): base / f"K_{k}/virtual_commot_input/{s}.h5ad"
        for k in K_VALUES for _, _, s in STAGES
    }
    panels, scales = {}, {}
    for module, genes in MODULES.items():
        for stage_label, real_stage, virtual_stage in STAGES:
            xy, values = module_score(real_paths[real_stage], genes, panel_normalize)
            panels[(module, stage_label, "Real")] = original_smooth_grid(
                xy, values, SIGMA_REAL, panel_normalize,
            )
            for k in K_VALUES:
                xy, values = module_score(virtual_paths[(k, virtual_stage)], genes, panel_normalize)
                panels[(module, stage_label, f"K={k}")] = original_smooth_grid(
                    xy, values, SIGMA_VIRTUAL, panel_normalize,
                )
            if panel_normalize:
                scales[f"{module}|{stage_label}"] = {"vmin": 0.0, "vmax": 1.0}
            else:
                all_values = np.concatenate([
                    panels[(module, stage_label, label)].ravel()
                    for label in ("Real", "K=20", "K=50", "K=100")
                ])
                scales[f"{module}|{stage_label}"] = {
                    "vmin": 0.0,
                    "vmax": max(float(np.nanpercentile(all_values, 99)), 1e-12),
                }
    return panels, scales


def plot_version(panels: dict, scales: dict, output: Path, shared_colorbars: bool) -> None:
    fig, axes = plt.subplots(6, 4, figsize=(16.8, 12.4), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.145, right=0.91 if shared_colorbars else 0.96, top=0.94, bottom=0.04, hspace=0.10, wspace=0.08)
    columns = ("Real", "K=20", "K=50", "K=100")
    images = []
    row = 0
    for module in MODULES:
        for stage_label, _, _ in STAGES:
            scale = scales[f"{module}|{stage_label}"]
            for col, label in enumerate(columns):
                ax = axes[row, col]
                image = ax.imshow(
                    panels[(module, stage_label, label)], origin="lower", cmap="magma",
                    vmin=scale["vmin"], vmax=scale["vmax"], interpolation="bilinear",
                )
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                for spine in ax.spines.values(): spine.set_visible(False)
                if row == 0:
                    ax.set_title(label, fontsize=12.5, weight="bold", pad=6)
            axes[row, 0].set_ylabel(f"{module}\n{stage_label}", fontsize=10.5, weight="bold", labelpad=10)
            images.append(image); row += 1
    if shared_colorbars:
        for row, image in enumerate(images):
            box = axes[row, -1].get_position()
            cax = fig.add_axes([0.93, box.y0, 0.008, box.height])
            cb = fig.colorbar(image, cax=cax); cb.locator = plt.MaxNLocator(3); cb.update_ticks()
            cb.ax.tick_params(labelsize=7, length=2)
    for i in (1, 3):
        y = (axes[i, 0].get_position().y0 + axes[i + 1, 0].get_position().y1) / 2
        fig.add_artist(plt.Line2D([0.055, 0.96], [y, y], transform=fig.transFigure, color="#B7BDC5", lw=0.9))
    fig.suptitle(
        "GSE267904 module-expression recovery across increasing Agent granularity",
        fontsize=16.5, weight="bold", y=0.985,
    )
    fig.savefig(output, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = args.input_root.resolve() if args.input_root is not None else root / "outputs/GSE267904_spatial_agent_granularity"
    out = args.output_dir.resolve() if args.output_dir is not None else base / "figures/module_expression_K20_K50_K100_old_style_horizontal"
    out.mkdir(parents=True, exist_ok=True)

    old_panels, old_scales = build_panels(base, panel_normalize=True)
    shared_panels, shared_scales = build_panels(base, panel_normalize=False)
    old_png = out / "GSE267904_module_expression_K20_K50_K100_old_style_panel_normalized.png"
    shared_png = out / "GSE267904_module_expression_K20_K50_K100_old_style_row_shared_scale.png"
    plot_version(old_panels, old_scales, old_png, shared_colorbars=False)
    plot_version(shared_panels, shared_scales, shared_png, shared_colorbars=True)

    audit = {
        "upstream_models_rerun": False,
        "old_style_parameters": {"grid_n": GRID_N, "sigma_real": SIGMA_REAL, "sigma_virtual": SIGMA_VIRTUAL, "cmap": "magma"},
        "panel_normalized_version": {
            "module_score": "old minmax01 after expression-scale handling",
            "grid": "old weighted/count Gaussian grid followed by minmax01",
            "color_limits": old_scales,
            "path": str(old_png),
        },
        "row_shared_scale_version": {
            "module_score": "expression-scale handling without minmax normalization",
            "grid": "old weighted/count Gaussian grid without minmax normalization",
            "color_limits": shared_scales,
            "path": str(shared_png),
        },
    }
    (out / "style_comparison_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Old-style horizontal module-expression diagnostics\n\n"
        "Both versions reuse the original script's grid_n=95, Real sigma=1.6, Virtual sigma=4.0, weighted/count Gaussian grid, and magma colormap.\n\n"
        "The panel-normalized version additionally reproduces the original 1st–99th percentile min-max normalization per panel. "
        "It is the closest visual match to the historical figure and best exposes increasing spatial detail, but color intensity is relative within each panel.\n\n"
        "The row-shared version removes per-panel min-max normalization and shares one absolute effective-expression scale across Real/K20/K50/K100 in each row. "
        "It is stricter quantitatively but visually suppresses some granularity differences.\n",
        encoding="utf-8",
    )
    print(json.dumps({"panel_normalized": str(old_png), "row_shared_scale": str(shared_png), "audit": str(out / 'style_comparison_audit.json')}, indent=2))


if __name__ == "__main__":
    main()
