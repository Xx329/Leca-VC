#!/usr/bin/env python3
"""Infer dominant spot cell type/state using marker module scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import CELL_TYPES, MARKERS, dump_json, ensure_dirs, outpath


def import_anndata():
    import anndata as ad
    return ad


def score_h5ad(path: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    adata = import_anndata().read_h5ad(path)
    genes = list(map(str, adata.var_names))
    lower = {g.lower(): i for i, g in enumerate(genes)}
    X = adata.X
    if hasattr(X, "toarray"):
        # marker scoring only; keep small dense slice per marker set
        pass
    score = pd.DataFrame(index=adata.obs_names)
    used_markers = {}
    for ct in CELL_TYPES:
        if ct == "other":
            continue
        idx = sorted(set(lower[g.lower()] for g in MARKERS.get(ct, []) if g.lower() in lower))
        used_markers[ct] = [genes[i] for i in idx]
        if idx:
            sub = X[:, idx]
            arr = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
            score[ct] = np.log1p(arr).mean(axis=1)
        else:
            score[ct] = 0.0
    z = score.copy()
    for c in z.columns:
        sd = float(z[c].std())
        z[c] = 0.0 if sd == 0 else (z[c] - float(z[c].mean())) / sd
    dominant = z.idxmax(axis=1)
    maxscore = z.max(axis=1)
    dominant = dominant.where(maxscore > -0.25, "other")
    obs = adata.obs.copy()
    obs["dominant_cell_type"] = dominant.reindex(obs.index).fillna("other").values
    for c in z.columns:
        obs[f"cell_type_score_{c}"] = z[c].reindex(obs.index).values
    adata.obs = obs
    adata.write_h5ad(path)
    stage = path.stem
    z2 = z.copy()
    z2.insert(0, "spot_id", z2.index)
    z2.insert(1, "stage", stage)
    ann = pd.DataFrame({"spot_id": obs.index, "stage": stage, "dominant_cell_type": obs["dominant_cell_type"].values})
    dump_json(out / f"celltype_annotation/{stage}_marker_coverage.json", used_markers)
    return z2.reset_index(drop=True), ann


def plot_maps(out: Path, ann_all: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import anndata as ad
    from common import CELL_TYPE_COLORS, CELL_TYPE_LABELS

    h5ads = sorted((out / "real_benchmark").glob("*.h5ad"))
    n = len(h5ads)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for ax, h in zip(axes[0], h5ads):
        a = ad.read_h5ad(h)
        xy = a.obsm["spatial"]
        colors = [CELL_TYPE_COLORS.get(x, "#999999") for x in a.obs["dominant_cell_type"]]
        ax.scatter(xy[:, 0], -xy[:, 1], c=colors, s=8, linewidths=0)
        ax.set_title(h.stem)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=CELL_TYPE_LABELS.get(k, k), markerfacecolor=v, markersize=6)
        for k, v in CELL_TYPE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("GSE267904 real spatial dominant cell-type maps (marker-score inferred)")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures/gse267904_real_spatial_celltype_maps.png", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    h5ads = sorted((out / "real_benchmark").glob("*.h5ad"))
    if not h5ads:
        raise RuntimeError("No real_benchmark/*.h5ad files found. Run build_real_spatial_benchmark.py first.")
    scores, anns = [], []
    for h in h5ads:
        s, ann = score_h5ad(h, out)
        scores.append(s)
        anns.append(ann)
    score_all = pd.concat(scores, ignore_index=True)
    ann_all = pd.concat(anns, ignore_index=True)
    score_all.to_csv(out / "celltype_annotation/spot_celltype_scores.csv", index=False)
    ann_all.to_csv(out / "celltype_annotation/spot_dominant_celltype.csv", index=False)
    plot_maps(out, ann_all)
    print(f"Wrote marker annotation to {out/'celltype_annotation'}")


if __name__ == "__main__":
    main()

