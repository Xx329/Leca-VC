#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import RESPONSE_GROUPS, STATES, open_text, outpath, resolve_expr_path


def safe_mean(vals: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return np.nan
    return float(np.nanmean(vals[idx]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-mean-expression", type=float, default=0.02)
    args = ap.parse_args()

    root = args.project_root.resolve()
    out = outpath(root, args.out_dir)
    real_dir = out / "real_benchmark"
    eval_dir = out / "evaluation"
    fig_dir = out / "figures"
    expr_dir = out / "expression"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    expr_dir.mkdir(parents=True, exist_ok=True)

    expr_path = resolve_expr_path(root)
    meta = pd.read_csv(real_dir / "metadata_checked.csv")
    cell_scores = pd.read_csv(real_dir / "cell_module_scores.csv")[["cell_id", "inferred_state"]]
    meta = meta.merge(cell_scores, on="cell_id", how="left")
    sim = pd.read_csv(out / "simulation/gse120575_v7_virtual_state_proportions.csv")

    with open_text(expr_path) as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
    matrix_cells = header[1:]
    index = {c: i for i, c in enumerate(matrix_cells)}
    meta = meta[meta.cell_id.astype(str).isin(index)].copy()
    meta["matrix_index"] = meta.cell_id.astype(str).map(index).astype(int)

    real_post_idx = {
        group: meta.loc[
            meta.response_group.eq(group) & meta.treatment_status.eq("post-treatment"),
            "matrix_index",
        ].to_numpy(dtype=int)
        for group in RESPONSE_GROUPS
    }
    pre_state_idx = {
        (group, state): meta.loc[
            meta.response_group.eq(group)
            & meta.treatment_status.eq("pre-treatment")
            & meta.inferred_state.eq(state),
            "matrix_index",
        ].to_numpy(dtype=int)
        for group in RESPONSE_GROUPS
        for state in STATES
    }
    weights = {
        group: {
            r.cell_state: float(r.proportion)
            for r in sim[
                sim.response_group.eq(group)
                & sim.phase.eq("virtual_post-treatment")
            ].itertuples(index=False)
        }
        for group in RESPONSE_GROUPS
    }

    rows = []
    n_genes = 0
    with open_text(expr_path) as handle:
        handle.readline()  # header
        handle.readline()  # embedded sample label row
        for line in handle:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                continue
            gene = parts[0]
            try:
                vals = np.array(parts[1:], dtype=float)
            except ValueError:
                vals = np.array([float(x) if x else np.nan for x in parts[1:]], dtype=float)
            n_genes += 1
            row = {"gene": gene}
            keep = False
            for group in RESPONSE_GROUPS:
                real_post = safe_mean(vals, real_post_idx[group])
                virtual_post = 0.0
                total_w = 0.0
                for state in STATES:
                    idx = pre_state_idx[(group, state)]
                    if idx.size == 0:
                        continue
                    w = weights[group].get(state, 0.0)
                    if w <= 0:
                        continue
                    m = safe_mean(vals, idx)
                    if np.isfinite(m):
                        virtual_post += w * m
                        total_w += w
                virtual_post = virtual_post / total_w if total_w > 0 else np.nan
                row[f"{group}_real_post"] = real_post
                row[f"{group}_virtual_post"] = virtual_post
                if np.nanmax([real_post, virtual_post]) >= args.min_mean_expression:
                    keep = True
            if keep:
                rows.append(row)

    comp = pd.DataFrame(rows)
    comp.to_csv(expr_dir / "gse120575_v7_full_gene_pseudobulk_real_virtual.csv", index=False)

    metric_rows = []
    for group in RESPONSE_GROUPS:
        x = np.log1p(comp[f"{group}_real_post"].astype(float).to_numpy())
        y = np.log1p(comp[f"{group}_virtual_post"].astype(float).to_numpy())
        mask = np.isfinite(x) & np.isfinite(y)
        metric_rows.append(
            {
                "response_group": group,
                "n_genes_tested_in_raw_matrix": n_genes,
                "n_genes_after_expression_filter": int(mask.sum()),
                "comparison": "log1p real post-treatment pseudo-bulk vs virtual post-treatment pseudo-bulk",
                "pearson": float(pearsonr(x[mask], y[mask]).statistic) if mask.sum() > 2 else np.nan,
                "spearman": float(spearmanr(x[mask], y[mask]).statistic) if mask.sum() > 2 else np.nan,
                "note": "Virtual full-gene pseudo-bulk is state-mixture projection from pre-treatment state prototypes; not a per-cell full-transcriptome decoder.",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(eval_dir / "gse120575_v7_full_gene_pseudobulk_correlation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220)
    for ax, group in zip(axes, RESPONSE_GROUPS):
        x = np.log1p(comp[f"{group}_real_post"].astype(float).to_numpy())
        y = np.log1p(comp[f"{group}_virtual_post"].astype(float).to_numpy())
        mask = np.isfinite(x) & np.isfinite(y)
        m = metrics[metrics.response_group.eq(group)].iloc[0]
        hb = ax.hexbin(x[mask], y[mask], gridsize=55, mincnt=1, cmap="viridis", linewidths=0)
        lo = min(float(x[mask].min()), float(y[mask].min()))
        hi = max(float(x[mask].max()), float(y[mask].max()))
        ax.plot([lo, hi], [lo, hi], "--", color="white", lw=1.2, alpha=0.9)
        ax.set_title(
            f"{group}: full-gene pseudo-bulk\\nPearson={m.pearson:.3f}, Spearman={m.spearman:.3f}, n={int(m.n_genes_after_expression_filter)} genes",
            fontsize=11,
            weight="bold",
        )
        ax.set_xlabel("Real post-treatment log1p mean expression")
        ax.set_ylabel("Virtual post-treatment log1p mean expression")
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Gene count")
    fig.suptitle("GSE120575 v7: full-gene pseudo-bulk real-vs-virtual expression comparison", fontsize=14, weight="bold")
    fig.text(
        0.01,
        0.01,
        "Note: This is full-gene pseudo-bulk comparison from pre-treatment state prototypes + virtual state mixture. It is not a full-transcriptome per-cell decoder.",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(fig_dir / "gse120575_v7_full_gene_pseudobulk_real_vs_virtual.png", bbox_inches="tight")
    plt.close(fig)
    print(fig_dir / "gse120575_v7_full_gene_pseudobulk_real_vs_virtual.png")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

