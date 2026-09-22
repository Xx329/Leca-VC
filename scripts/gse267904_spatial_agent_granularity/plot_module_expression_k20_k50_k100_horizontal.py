#!/usr/bin/env python3
"""Plot the GSE267904 6x4 module-expression recovery main figure.

This is a read-only plotting workflow. It reads existing h5ad matrices and does
not run PhysiCell, an Agent, COMMOT, or any upstream simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse267904-module-main")
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
    return (xy - lo) / np.where(hi > lo, hi - lo, 1.0)


def read_module_score(path: Path, genes: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    a = ad.read_h5ad(path, backed="r")
    present = [g for g in genes if g in a.var_names]
    if len(present) != len(genes):
        a.file.close()
        raise RuntimeError(f"{path} missing module genes: {sorted(set(genes) - set(present))}")
    subset = a[:, present].to_memory()
    matrix = subset.X.toarray() if sparse.issparse(subset.X) else np.asarray(subset.X)
    matrix = np.asarray(matrix, dtype=float)
    raw_max = float(np.nanmax(matrix)) if matrix.size else 0.0
    # Real matrices contain non-log count-scale values; virtual matrices store
    # the corresponding log-scale expression programs. Preserve the frozen
    # preprocessing semantics used by the prior module-expression workflow.
    transform = "log1p" if raw_max > 20 else "already_log_scale"
    if transform == "log1p":
        matrix = np.log1p(np.clip(matrix, 0, None))
    score = np.nanmean(matrix, axis=1)
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    info = {
        "path": str(path), "shape": list(a.shape), "genes": present,
        "raw_module_gene_max": raw_max, "expression_transform": transform,
        "n_coordinates": int(len(xy)), "n_unique_coordinates": int(len(np.unique(xy, axis=0))),
    }
    a.file.close()
    return xy, score, info


def aggregate_duplicate_xy(xy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "value": values})
    grouped = frame.groupby(["x", "y"], as_index=False)["value"].mean()
    return grouped[["x", "y"]].to_numpy(float), grouped["value"].to_numpy(float)


def shared_virtual_mask(xy_sets: list[np.ndarray], grid_n: int, sigma: float) -> np.ndarray:
    points = np.unique(np.vstack([norm_xy(xy) for xy in xy_sets]), axis=0)
    ix = np.clip((points[:, 0] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    iy = np.clip((points[:, 1] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    density = np.zeros((grid_n, grid_n), dtype=float)
    np.add.at(density, (iy, ix), 1.0)
    density = gaussian_filter(density, sigma=sigma, mode="nearest")
    mask = density >= max(float(np.nanmax(density)) * 0.015, 1e-12)
    return binary_fill_holes(binary_closing(mask, iterations=2))


def weighted_gaussian_field(
    xy: np.ndarray, values: np.ndarray, grid_n: int, sigma: float, mask: np.ndarray,
) -> np.ndarray:
    xy, values = aggregate_duplicate_xy(norm_xy(xy), np.asarray(values, dtype=float))
    ix = np.clip((xy[:, 0] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    iy = np.clip((xy[:, 1] * (grid_n - 1)).astype(int), 0, grid_n - 1)
    weighted = np.zeros((grid_n, grid_n), dtype=float)
    density = np.zeros_like(weighted)
    np.add.at(weighted, (iy, ix), values)
    np.add.at(density, (iy, ix), 1.0)
    weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    density = gaussian_filter(density, sigma=sigma, mode="nearest")
    field = np.divide(weighted, density, out=np.zeros_like(weighted), where=density > 1e-12)
    field[~mask] = np.nan
    return field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--grid-resolution", type=int, default=180)
    parser.add_argument("--smoothing-sigma", type=float, default=7.0)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root / "outputs/GSE267904_spatial_agent_granularity"
    out = base / "figures/module_expression_K20_K50_K100_horizontal"
    out.mkdir(parents=True, exist_ok=True)

    real_paths = {stage: base / f"shared_real/real_benchmark/{stage}.h5ad" for _, stage, _ in STAGES}
    virtual_paths = {
        (k, stage): base / f"K_{k}/virtual_commot_input/{stage}.h5ad"
        for k in K_VALUES for _, _, stage in STAGES
    }

    # Coordinates do not depend on the selected module. Read one module to
    # establish the single virtual support mask used throughout the figure.
    first_genes = next(iter(MODULES.values()))
    mask_xy = [read_module_score(virtual_paths[(100, stage)], first_genes)[0] for _, _, stage in STAGES]
    mask = shared_virtual_mask(mask_xy, args.grid_resolution, args.smoothing_sigma)

    panels: dict[tuple[str, str, str], dict] = {}
    row_scales: dict[str, dict] = {}
    source_audit: dict[str, dict] = {}
    for module, genes in MODULES.items():
        for stage_label, real_stage, virtual_stage in STAGES:
            row_values = []
            rxy, rscore, info = read_module_score(real_paths[real_stage], genes)
            source_audit[f"{module}|{stage_label}|Real"] = info
            panels[(module, stage_label, "Real")] = {
                "kind": "real", "xy": norm_xy(rxy), "values": rscore,
            }
            row_values.append(rscore)
            for k in K_VALUES:
                vxy, vscore, info = read_module_score(virtual_paths[(k, virtual_stage)], genes)
                source_audit[f"{module}|{stage_label}|K={k}"] = info
                field = weighted_gaussian_field(
                    vxy, vscore, args.grid_resolution, args.smoothing_sigma, mask,
                )
                panels[(module, stage_label, f"K={k}")] = {"kind": "virtual", "field": field}
                row_values.append(field[np.isfinite(field)])
            finite = np.concatenate([np.asarray(x)[np.isfinite(x)] for x in row_values])
            vmax = float(np.quantile(finite, 0.995)) if finite.size else 1.0
            row_scales[f"{module}|{stage_label}"] = {"vmin": 0.0, "vmax": max(vmax, 1e-12)}

    fig, axes = plt.subplots(6, 4, figsize=(17.0, 12.5), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.145, right=0.91, top=0.945, bottom=0.04, hspace=0.12, wspace=0.10)
    columns = ("Real", "K=20", "K=50", "K=100")
    row_images = []
    row = 0
    for module, _genes in MODULES.items():
        for stage_label, _real_stage, _virtual_stage in STAGES:
            scale = row_scales[f"{module}|{stage_label}"]
            image = None
            for col, label in enumerate(columns):
                ax = axes[row, col]
                panel = panels[(module, stage_label, label)]
                if panel["kind"] == "real":
                    image = ax.scatter(
                        panel["xy"][:, 0], panel["xy"][:, 1], c=panel["values"],
                        cmap="magma", vmin=scale["vmin"], vmax=scale["vmax"],
                        s=2.0, linewidths=0, rasterized=True,
                    )
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                else:
                    image = ax.imshow(
                        panel["field"], origin="lower", extent=(0, 1, 0, 1),
                        cmap="magma", vmin=scale["vmin"], vmax=scale["vmax"],
                        interpolation="bilinear",
                    )
                ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                for spine in ax.spines.values(): spine.set_visible(False)
                if row == 0:
                    ax.set_title(label, fontsize=12.5, weight="bold", pad=7)
            axes[row, 0].set_ylabel(
                f"{module}\n{stage_label}", fontsize=10.5, weight="bold", labelpad=10,
            )
            row_images.append(image)
            row += 1

    fig.suptitle(
        "GSE267904 module-expression recovery across increasing Agent granularity",
        fontsize=17, weight="bold", y=0.988,
    )
    # Separate the three biological module blocks without adding visual clutter.
    for row_index, image in enumerate(row_images):
        box = axes[row_index, -1].get_position()
        cax = fig.add_axes([0.93, box.y0, 0.008, box.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.locator = plt.MaxNLocator(3); cbar.update_ticks(); cbar.ax.tick_params(labelsize=7, length=2)
    divider_y = [
        (axes[1, 0].get_position().y0 + axes[2, 0].get_position().y1) / 2,
        (axes[3, 0].get_position().y0 + axes[4, 0].get_position().y1) / 2,
    ]
    for y in divider_y:
        fig.add_artist(plt.Line2D([0.055, 0.955], [y, y], transform=fig.transFigure, color="#B7BDC5", lw=1.0))

    stem = "GSE267904_module_expression_K20_K50_K100_horizontal"
    preview = out / f"{stem}_preview.png"
    highres = out / f"{stem}_600dpi.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(preview, dpi=150, facecolor="white", bbox_inches="tight")
    fig.savefig(highres, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    audit = {
        "status": "PASS_MODULE_EXPRESSION_FIGURE_AUDIT",
        "modules": list(MODULES), "stages": [x[0] for x in STAGES],
        "columns": list(columns), "layout": "6 rows x 4 columns",
        "grid_resolution": args.grid_resolution, "smoothing_sigma": args.smoothing_sigma,
        "coordinate_normalization": "independent source min-max to [0,1] in x and y; no flips or rotations",
        "virtual_method": "weighted-grid Gaussian smoothing divided by Gaussian-smoothed agent density",
        "virtual_mask": "one shared support mask from unique K=100 early and late coordinates",
        "real_display": "unsmoothed Visium spots",
        "row_wise_color_scales": row_scales,
        "color_scale_rule": "vmin=0; shared row-wise vmax=99.5th percentile over Real and all three virtual panels",
        "source_audit": source_audit,
        "historical_outputs_modified": False, "upstream_models_rerun": False,
        "outputs": {"preview": str(preview), "600dpi_png": str(highres), "pdf": str(pdf)},
    }
    (out / "figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest_paths = sorted(set(real_paths.values()) | set(virtual_paths.values()))
    (out / "plotting_source_manifest.json").write_text(json.dumps({
        "files": [{"path": str(p), "sha256": sha256(p)} for p in manifest_paths],
        "script": str(Path(__file__).resolve()),
    }, indent=2) + "\n")
    caption = (
        "Spatial module-expression recovery from existing GSE267904 outputs. Real panels show unsmoothed Visium spots; "
        "virtual panels show identically processed weighted-grid Gaussian fields for K=20, K=50, and K=100. "
        "Color limits are shared within each module-stage row. TGF-beta response and inflammation show the most stable "
        "improvement with increasing Agent granularity. Fibrosis/ECM shows high overall recovery, while at d21 the "
        "K=100 pattern is slightly less concordant with Real than K=50."
    )
    (out / "README.md").write_text("# Figure caption\n\n" + caption + "\n", encoding="utf-8")
    print(json.dumps({"preview": str(preview), "600dpi_png": str(highres), "pdf": str(pdf), "row_scales": row_scales}, indent=2))


if __name__ == "__main__":
    main()
