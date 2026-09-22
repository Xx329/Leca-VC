#!/usr/bin/env python3
"""Evaluate real-vs-virtual COMMOT communication agreement."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import KEY_INTERACTIONS, dump_json, ensure_dirs, outpath, pearson, spearman


STAGE_MAP = {"d7_bleo": "virtual_early_bleo", "d21_bleo": "virtual_late_bleo"}


def load_matrix(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing {name}: {path}")
    df = pd.read_csv(path)
    need = {"stage", "sender", "receiver", "weight"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"{path} must contain columns {need}")
    return df


def compare_stage(real: pd.DataFrame, virt: pd.DataFrame, real_stage: str, virt_stage: str) -> dict:
    r = real[real.stage.eq(real_stage)].groupby(["sender", "receiver"], as_index=False)["weight"].sum()
    v = virt[virt.stage.eq(virt_stage)].groupby(["sender", "receiver"], as_index=False)["weight"].sum()
    m = r.merge(v, on=["sender", "receiver"], how="outer", suffixes=("_real", "_virtual")).fillna(0)
    m["edge_type"] = np.where(m["sender"].astype(str).eq(m["receiver"].astype(str)), "self", "cross_cell_type")
    top_r = set(tuple(x) for x in r.sort_values("weight", ascending=False)[["sender", "receiver"]].head(10).to_numpy())
    top_v = set(tuple(x) for x in v.sort_values("weight", ascending=False)[["sender", "receiver"]].head(10).to_numpy())
    cross = m[m.edge_type.eq("cross_cell_type")]
    rc = r[r["sender"].astype(str).ne(r["receiver"].astype(str))]
    vc = v[v["sender"].astype(str).ne(v["receiver"].astype(str))]
    top_rc = set(tuple(x) for x in rc.sort_values("weight", ascending=False)[["sender", "receiver"]].head(10).to_numpy())
    top_vc = set(tuple(x) for x in vc.sort_values("weight", ascending=False)[["sender", "receiver"]].head(10).to_numpy())
    key_rows = []
    for s, t in KEY_INTERACTIONS:
        rr = float(m.loc[m.sender.eq(s) & m.receiver.eq(t), "weight_real"].sum())
        vv = float(m.loc[m.sender.eq(s) & m.receiver.eq(t), "weight_virtual"].sum())
        key_rows.append({"real_stage": real_stage, "virtual_stage": virt_stage, "sender": s, "receiver": t, "real_weight": rr, "virtual_weight": vv, "recovered": bool(rr > 0 and vv > 0)})
    return {
        "real_stage": real_stage,
        "virtual_stage": virt_stage,
        "n_common_edges_nonzero": int(((m.weight_real > 0) & (m.weight_virtual > 0)).sum()),
        "n_union_edges": int(len(m)),
        "sender_receiver_pearson": pearson(m.weight_real, m.weight_virtual),
        "sender_receiver_spearman": spearman(m.weight_real, m.weight_virtual),
        "self_edge_pearson": pearson(m.loc[m.edge_type.eq("self"), "weight_real"], m.loc[m.edge_type.eq("self"), "weight_virtual"]),
        "self_edge_spearman": spearman(m.loc[m.edge_type.eq("self"), "weight_real"], m.loc[m.edge_type.eq("self"), "weight_virtual"]),
        "cross_cell_type_pearson": pearson(cross.weight_real, cross.weight_virtual),
        "cross_cell_type_spearman": spearman(cross.weight_real, cross.weight_virtual),
        "cross_cell_type_real_total_weight": float(cross.weight_real.sum()),
        "cross_cell_type_virtual_total_weight": float(cross.weight_virtual.sum()),
        "cross_cell_type_common_nonzero_edges": int(((cross.weight_real > 0) & (cross.weight_virtual > 0)).sum()),
        "top10_edge_overlap": len(top_r & top_v),
        "top10_edge_jaccard": len(top_r & top_v) / max(1, len(top_r | top_v)),
        "top10_self_edges_real": int(sum(s == t for s, t in top_r)),
        "top10_self_edges_virtual": int(sum(s == t for s, t in top_v)),
        "cross_top10_edge_overlap": len(top_rc & top_vc),
        "cross_top10_edge_jaccard": len(top_rc & top_vc) / max(1, len(top_rc | top_vc)),
        "key_interactions_recovered": int(sum(x["recovered"] for x in key_rows)),
        "matrix": m,
        "key_rows": key_rows,
    }


def make_figures(out: Path, metrics: pd.DataFrame, residuals: list[pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for m in residuals:
        stage = m.attrs.get("stage", "stage")
        pivot_r = m.pivot_table(index="sender", columns="receiver", values="weight_real", fill_value=0)
        pivot_v = m.pivot_table(index="sender", columns="receiver", values="weight_virtual", fill_value=0)
        senders = sorted(set(pivot_r.index) | set(pivot_v.index))
        receivers = sorted(set(pivot_r.columns) | set(pivot_v.columns))
        pivot_r = pivot_r.reindex(index=senders, columns=receivers, fill_value=0)
        pivot_v = pivot_v.reindex(index=senders, columns=receivers, fill_value=0)
        vmax = max(float(pivot_r.values.max()), float(pivot_v.values.max()), 1e-9)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        sns.heatmap(pivot_r, ax=axes[0], cmap="viridis", vmin=0, vmax=vmax)
        axes[0].set_title(f"Real {stage}")
        sns.heatmap(pivot_v, ax=axes[1], cmap="viridis", vmin=0, vmax=vmax)
        axes[1].set_title(f"Virtual {stage}")
        sns.heatmap(pivot_v - pivot_r, ax=axes[2], cmap="coolwarm", center=0)
        axes[2].set_title("Virtual − real")
        for ax in axes:
            ax.set_xlabel("Receiver")
            ax.set_ylabel("Sender")
        fig.suptitle("GSE267904 COMMOT sender–receiver network")
        fig.tight_layout()
        fig.savefig(fig_dir / f"gse267904_real_vs_virtual_sender_receiver_heatmap_{stage}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(m.weight_real, m.weight_virtual, s=24, alpha=0.75)
        mx = max(float(m.weight_real.max()), float(m.weight_virtual.max()), 1e-9)
        ax.plot([0, mx], [0, mx], "--", color="gray")
        ax.set_xlabel("Real COMMOT edge weight")
        ax.set_ylabel("Virtual COMMOT edge weight")
        mm = metrics[metrics.real_stage.eq(stage)].iloc[0]
        ax.set_title(f"{stage}: edge agreement r={mm.sender_receiver_pearson:.2f}, top10 overlap={int(mm.top10_edge_overlap)}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"gse267904_commot_edge_agreement_{stage}.png", dpi=220)
        plt.close(fig)
        # Cross-cell-type diagnostic scatter: this is the real communication
        # validation target, separated from self-loops.
        cross = m[m["sender"].astype(str).ne(m["receiver"].astype(str))].copy()
        fig, ax = plt.subplots(figsize=(5.8, 5.1))
        ax.scatter(cross.weight_real, cross.weight_virtual, s=28, alpha=0.78, color="#e76f51", edgecolor="white", linewidth=0.4)
        mx = max(float(cross.weight_real.max()), float(cross.weight_virtual.max()), 1e-9)
        ax.plot([0, mx], [0, mx], "--", color="gray")
        ax.set_xlabel("Real cross-cell-type COMMOT edge weight")
        ax.set_ylabel("Virtual cross-cell-type COMMOT edge weight")
        mm = metrics[metrics.real_stage.eq(stage)].iloc[0]
        ax.set_title(
            f"{stage}: cross-cell-type edge agreement\n"
            f"r={mm.cross_cell_type_pearson:.2f}, top10 overlap={int(mm.cross_top10_edge_overlap)}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gse267904_commot_cross_type_edge_agreement_{stage}.png", dpi=220)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics))
    ax.bar(x - 0.2, metrics.sender_receiver_pearson, width=0.18, label="Pearson")
    ax.bar(x, metrics.sender_receiver_spearman, width=0.18, label="Spearman")
    ax.bar(x + 0.2, metrics.top10_edge_jaccard, width=0.18, label="Top10 Jaccard")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.real_stage)
    ax.set_ylim(-0.2, 1.0)
    ax.set_title("Real-vs-virtual COMMOT summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "gse267904_commot_metric_summary.png", dpi=220)
    plt.close(fig)

    # Paper-facing comparison that does not hide behind all-edge Pearson.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(metrics))
    w = 0.16
    ax.bar(x - 1.5*w, metrics.sender_receiver_pearson, width=w, label="All-edge Pearson", color="#4e79a7")
    ax.bar(x - 0.5*w, metrics.self_edge_pearson, width=w, label="Self-edge Pearson", color="#59a14f")
    ax.bar(x + 0.5*w, metrics.cross_cell_type_pearson, width=w, label="Cross-type Pearson", color="#e76f51")
    ax.bar(x + 1.5*w, metrics.cross_top10_edge_jaccard, width=w, label="Cross-type top10 Jaccard", color="#f28e2b")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.real_stage)
    ax.set_ylim(-0.35, 1.0)
    ax.set_ylabel("Agreement")
    ax.set_title("COMMOT agreement separated into self vs cross-cell-type communication")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gse267904_commot_self_vs_cross_type_metric_summary.png", dpi=240)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    real = load_matrix(out / "real_commot/real_sender_receiver_matrix_by_stage.csv", "real sender-receiver matrix")
    virt = load_matrix(out / "virtual_commot/virtual_sender_receiver_matrix_by_stage.csv", "virtual sender-receiver matrix")
    rows, matrices, key_rows = [], [], []
    for rs, vs in STAGE_MAP.items():
        comp = compare_stage(real, virt, rs, vs)
        m = comp.pop("matrix")
        kr = comp.pop("key_rows")
        rows.append(comp)
        m.attrs["stage"] = rs
        matrices.append(m)
        key_rows.extend(kr)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "evaluation/commot_sender_receiver_metrics.csv", index=False)
    pd.DataFrame(key_rows).to_csv(out / "evaluation/key_interaction_recovery.csv", index=False)
    summary = metrics.copy()
    summary.to_csv(out / "evaluation/commot_summary_metrics.csv", index=False)
    make_figures(out, metrics, matrices)
    dump_json(out / "evaluation/commot_evaluation_audit.json", {"stages_compared": STAGE_MAP, "metrics_csv": str(out / "evaluation/commot_summary_metrics.csv")})
    print(f"Wrote COMMOT evaluation to {out/'evaluation'} and figures to {out/'figures'}")


if __name__ == "__main__":
    main()
