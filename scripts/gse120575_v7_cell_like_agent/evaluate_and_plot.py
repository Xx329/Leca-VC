#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import MODULES, RESPONSE_GROUPS, STATES, dump, module_scores_from_gene_means, outpath, pearson, response_score, zscore_frame


COLORS = {
    "CD8_T_cell": "#4e79a7",
    "Cytotoxic_T_cell": "#59a14f",
    "Memory_like_T_cell": "#f28e2b",
    "Exhausted_T_cell": "#e15759",
    "Treg": "#b07aa1",
    "Macrophage_Monocyte": "#8cd17d",
    "Dendritic_cell": "#76b7b2",
    "B_cell": "#edc948",
    "NK_cell": "#af7aa1",
}


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def real_post_props(real_dir: Path) -> pd.DataFrame:
    props = pd.read_csv(real_dir / "cell_state_proportions_by_sample.csv")
    return (
        props[props.treatment_status.eq("post-treatment")]
        .groupby(["response_group", "cell_state"])["proportion"]
        .mean()
        .reset_index()
    )


def real_post_modules(real_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(real_dir / "sample_module_scores.csv")
    return (
        scores[scores.treatment_status.eq("post-treatment")]
        .groupby("response_group")[[f"{m}_score" for m in STATES] + ["response_score"]]
        .mean()
        .reset_index()
    )


def state_comparison(real_dir: Path, out: Path) -> pd.DataFrame:
    real = real_post_props(real_dir)
    virt = pd.read_csv(out / "simulation/gse120575_v7_virtual_state_proportions.csv")
    virt = virt[virt.phase.eq("virtual_post-treatment")]
    rows = []
    for group in RESPONSE_GROUPS:
        r = real[real.response_group.eq(group)].set_index("cell_state").reindex(STATES).proportion.fillna(0)
        v = virt[virt.response_group.eq(group)].set_index("cell_state").reindex(STATES).proportion.fillna(0)
        rows.append(
            {
                "scenario_name": f"virtual_{group.lower().replace('-', '_')}_like",
                "comparison": f"real_post_{group}_vs_virtual",
                "state_proportion_rmse": rmse(v, r),
                "state_proportion_mae": float(np.mean(np.abs(v.to_numpy(float) - r.to_numpy(float)))),
                "state_proportion_pearson": pearson(v.to_numpy(float), r.to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def module_comparison(real_dir: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = real_post_modules(real_dir)
    virt_expr = pd.read_csv(out / "expression/gse120575_v7_virtual_marker_expression.csv")
    rows = []
    for group in RESPONSE_GROUPS:
        rr = real[real.response_group.eq(group)].iloc[0]
        vv = virt_expr[virt_expr.response_group.eq(group)].iloc[0]
        for m in STATES + ["response_score"]:
            real_col = f"{m}_score" if m != "response_score" else "response_score"
            virt_col = m if m == "response_score" else m
            # virtual expression table stores module name without _score plus response_score
            virtual_score = float(vv[virt_col])
            real_score = float(rr[real_col])
            rows.append(
                {
                    "response_group": group,
                    "module": m,
                    "real_post_score": real_score,
                    "virtual_score": virtual_score,
                    "abs_error": abs(real_score - virtual_score),
                }
            )
    comp = pd.DataFrame(rows)
    response_rows = []
    real_delta = float(real[real.response_group.eq("Responder")].response_score.iloc[0] - real[real.response_group.eq("Non-responder")].response_score.iloc[0])
    virt_delta = float(virt_expr[virt_expr.response_group.eq("Responder")].response_score.iloc[0] - virt_expr[virt_expr.response_group.eq("Non-responder")].response_score.iloc[0])
    response_rows.append(
        {
            "metric": "response_score_direction_agreement",
            "value": bool(np.sign(real_delta) == np.sign(virt_delta)),
            "real_post_delta": real_delta,
            "virtual_delta": virt_delta,
        }
    )
    response_rows.append(
        {
            "metric": "mean_state_proportion_rmse",
            "value": float(state_comparison(real_dir, out).state_proportion_rmse.mean()),
            "real_post_delta": real_delta,
            "virtual_delta": virt_delta,
        }
    )
    return comp, pd.DataFrame(response_rows)


def expression_correlation(out: Path) -> pd.DataFrame:
    real = pd.read_csv(out / "real_benchmark/v7_marker_gene_means_by_group.csv")
    virt = pd.read_csv(out / "expression/gse120575_v7_virtual_marker_expression.csv")
    gene_cols = sorted(set(g for genes in MODULES.values() for g in genes))
    rows = []
    for group in RESPONSE_GROUPS:
        r = real[real.response_group.eq(group) & real.treatment_status.eq("post-treatment")].iloc[0]
        v = virt[virt.response_group.eq(group)].iloc[0]
        usable = [g for g in gene_cols if g in real.columns and g in virt.columns]
        rows.append({"response_group": group, "pearson": pearson(r[usable].to_numpy(float), v[usable].to_numpy(float)), "n_genes": len(usable)})
    return pd.DataFrame(rows)


def make_figures(real_dir: Path, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    real_props = real_post_props(real_dir)
    virt_props = pd.read_csv(out / "simulation/gse120575_v7_virtual_state_proportions.csv")
    virt_post = virt_props[virt_props.phase.eq("virtual_post-treatment")]

    # State composition
    plot_rows = []
    for group in RESPONSE_GROUPS:
        for label, df in [(f"Real post {group}", real_props[real_props.response_group.eq(group)]), (f"Virtual post {group}", virt_post[virt_post.response_group.eq(group)])]:
            for s in STATES:
                val = float(df[df.cell_state.eq(s)].proportion.iloc[0]) if (df.cell_state.eq(s)).any() else 0.0
                plot_rows.append({"group": label, "state": s, "proportion": val})
    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    order = [f"Real post {g}" for g in RESPONSE_GROUPS] + [f"Virtual post {g}" for g in RESPONSE_GROUPS]
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for s in STATES:
        vals = plot_df[plot_df.state.eq(s)].set_index("group").reindex(order).proportion.fillna(0).to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=s, color=COLORS[s])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cell-state proportion")
    ax.set_title("GSE120575 v7: real vs virtual post-treatment immune-state composition", weight="bold")
    ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.23), loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "gse120575_v7_real_vs_virtual_state_composition.png", bbox_inches="tight")
    plt.close(fig)

    # Response score
    real_mod = real_post_modules(real_dir)
    virt = pd.read_csv(out / "expression/gse120575_v7_virtual_marker_expression.csv")
    resp_rows = []
    for group in RESPONSE_GROUPS:
        resp_rows.append({"group": group, "source": "Real post", "response_score": float(real_mod[real_mod.response_group.eq(group)].response_score.iloc[0])})
        resp_rows.append({"group": group, "source": "Virtual post", "response_score": float(virt[virt.response_group.eq(group)].response_score.iloc[0])})
    resp = pd.DataFrame(resp_rows)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    for i, src in enumerate(["Real post", "Virtual post"]):
        vals = resp[resp.source.eq(src)].set_index("group").reindex(RESPONSE_GROUPS).response_score.to_numpy(float)
        ax.bar(np.arange(len(RESPONSE_GROUPS)) + (i - 0.5) * 0.36, vals, width=0.36, label=src)
    ax.set_xticks(np.arange(len(RESPONSE_GROUPS)))
    ax.set_xticklabels(RESPONSE_GROUPS)
    ax.set_ylabel("Response score")
    ax.set_title("GSE120575 v7: response score real vs virtual", weight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gse120575_v7_response_score_real_vs_virtual.png", bbox_inches="tight")
    plt.close(fig)

    # Module heatmap
    module_comp, _ = module_comparison(real_dir, out)
    mat_rows = []
    for group in RESPONSE_GROUPS:
        for source, col in [("Real post", "real_post_score"), ("Virtual post", "virtual_score")]:
            row = {"label": f"{source} {group}"}
            sub = module_comp[module_comp.response_group.eq(group) & module_comp.module.isin(STATES)]
            for m in STATES:
                row[m] = float(sub[sub.module.eq(m)][col].iloc[0])
            mat_rows.append(row)
    mat = pd.DataFrame(mat_rows).set_index("label")
    z = zscore_frame(mat)
    fig, ax = plt.subplots(figsize=(10, 4.3), dpi=180)
    im = ax.imshow(z.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(np.arange(len(z.index)))
    ax.set_yticklabels(z.index)
    ax.set_xticks(np.arange(len(z.columns)))
    ax.set_xticklabels(z.columns, rotation=35, ha="right")
    ax.set_title("GSE120575 v7: module score heatmap", weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "gse120575_v7_module_score_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # Marker correlation / signature recovery
    corr = expression_correlation(out)
    real_gene = pd.read_csv(out / "real_benchmark/v7_marker_gene_means_by_group.csv")
    virt_gene = pd.read_csv(out / "expression/gse120575_v7_virtual_marker_expression.csv")
    gene_cols = sorted(set(g for genes in MODULES.values() for g in genes))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=180)
    for ax, group in zip(axes, RESPONSE_GROUPS):
        r = real_gene[real_gene.response_group.eq(group) & real_gene.treatment_status.eq("post-treatment")].iloc[0]
        v = virt_gene[virt_gene.response_group.eq(group)].iloc[0]
        usable = [g for g in gene_cols if g in real_gene.columns and g in virt_gene.columns]
        x, y = r[usable].to_numpy(float), v[usable].to_numpy(float)
        ax.scatter(x, y, s=28, alpha=0.85)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#888")
        pr = float(corr[corr.response_group.eq(group)].pearson.iloc[0])
        ax.set_title(f"{group}: marker Pearson={pr:.3f}, n={len(usable)}", weight="bold")
        ax.set_xlabel("Real post mean expression")
        ax.set_ylabel("Virtual post mean expression")
    fig.tight_layout()
    fig.savefig(fig_dir / "gse120575_v7_signature_gene_recovery.png", bbox_inches="tight")
    plt.close(fig)

    # Summary panel
    fig = plt.figure(figsize=(15, 9), dpi=180)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    bottom = np.zeros(len(order))
    for s in STATES:
        vals = plot_df[plot_df.state.eq(s)].set_index("group").reindex(order).proportion.fillna(0).to_numpy(float)
        ax1.bar(np.arange(len(order)), vals, bottom=bottom, color=COLORS[s], label=s)
        bottom += vals
    ax1.set_title("A. State composition", loc="left", weight="bold")
    ax1.set_xticks(np.arange(len(order)))
    ax1.set_xticklabels(order, rotation=22, ha="right", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.legend(ncol=3, fontsize=7, frameon=False)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, src in enumerate(["Real post", "Virtual post"]):
        vals = resp[resp.source.eq(src)].set_index("group").reindex(RESPONSE_GROUPS).response_score.to_numpy(float)
        ax2.bar(np.arange(len(RESPONSE_GROUPS)) + (i - 0.5) * 0.36, vals, width=0.36, label=src)
    ax2.set_title("B. Response score", loc="left", weight="bold")
    ax2.set_xticks(np.arange(len(RESPONSE_GROUPS)))
    ax2.set_xticklabels(RESPONSE_GROUPS)
    ax2.legend(frameon=False)
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(z.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax3.set_title("C. Module score pattern", loc="left", weight="bold")
    ax3.set_yticks(np.arange(len(z.index)))
    ax3.set_yticklabels(z.index, fontsize=8)
    ax3.set_xticks(np.arange(len(z.columns)))
    ax3.set_xticklabels(z.columns, rotation=35, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.02)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    metrics = pd.read_csv(out / "evaluation/gse120575_v7_real_vs_virtual_metrics.csv")
    txt = "\n".join([f"{r.metric}: {r.value}" for r in metrics.itertuples(index=False)])
    ax4.text(0, 0.9, "D. External cancer benchmark metrics", weight="bold", fontsize=12)
    ax4.text(0, 0.72, txt, fontsize=11)
    ax4.text(0, 0.28, "Post-treatment real data are evaluation-only.\nThis is cross-dataset generalization, not dynamic-memory proof.", fontsize=10)
    fig.suptitle("GSE120575 v7 cell-like Agent external melanoma immunotherapy benchmark", weight="bold")
    fig.savefig(fig_dir / "gse120575_v7_external_validation_summary_panel.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    eval_dir = out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    real_dir = out / "real_benchmark"
    state = state_comparison(real_dir, out)
    state.to_csv(eval_dir / "gse120575_v7_state_proportion_comparison.csv", index=False)
    module_comp, metrics = module_comparison(real_dir, out)
    module_comp.to_csv(eval_dir / "gse120575_v7_module_score_comparison.csv", index=False)
    corr = expression_correlation(out)
    corr.to_csv(eval_dir / "gse120575_v7_signature_gene_recovery.csv", index=False)
    corr_mean = float(corr.pearson.mean())
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame([{"metric": "mean_marker_gene_pearson", "value": corr_mean, "real_post_delta": np.nan, "virtual_delta": np.nan}]),
        ],
        ignore_index=True,
    )
    metrics.to_csv(eval_dir / "gse120575_v7_real_vs_virtual_metrics.csv", index=False)
    make_figures(real_dir, out)
    dump(out / "audit/gse120575_v7_evaluation_audit.json", {"REAL_POST_TREATMENT_USED_FOR_EVALUATION_ONLY": True, "OLD_FULLSTACK_OUTPUT_USED_AS_INPUT": False})


if __name__ == "__main__":
    main()
