#!/usr/bin/env python3
"""GSE267904 spatial Agent-count / granularity ablation.

This wrapper reuses the validated GSE267904 spatial COMMOT pipeline but changes
only the number of PhysiCell-backed LLM micro-agents.  Real spatial benchmark
and real COMMOT are built once, then copied into each K-specific run so the
comparison is:

    same real spatial COMMOT reference
    same real calibration data
    same evaluation code
    different number of virtual micro-agents

The output is intentionally compact: one metrics CSV and one summary figure.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gse267904_spatial_commot_agent"

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

MODULE_GENES = {
    "ECM_collagen": ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Acta2", "Tagln"],
    "inflammation": ["Il1b", "Tnf", "Nfkbia", "Ccl2", "Cxcl2", "Cxcl10"],
    "chemokine_CCL_CXCL": ["Ccl2", "Ccl7", "Ccl8", "Cxcl1", "Cxcl2", "Cxcl10"],
    "TGFb_response": ["Tgfb1", "Tgfbr1", "Tgfbr2", "Serpine1", "Col1a1", "Fn1"],
    "Krt8_ADI_repair": ["Krt8", "Krt18", "Krt19", "Clu", "Lgals3", "Sox4"],
    "macrophage_activation": ["Lyz2", "Adgre1", "C1qa", "C1qb", "C1qc", "Mrc1", "Spp1"],
}

STAGE_PAIRS = [
    ("d7_bleo", "virtual_early_bleo"),
    ("d21_bleo", "virtual_late_bleo"),
]


def abs_out(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def call(root: Path, script: str, out: Path, *args: object) -> None:
    cmd = [
        sys.executable,
        str(BASE / script),
        "--project-root",
        str(root),
        "--out-dir",
        str(out),
        *map(str, args),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"Required shared directory does not exist: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def count_llm_calls(k_out: Path) -> int:
    return len(list(k_out.glob("**/*_prompt.json")))


def parse_intish(x: object) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x)
    if "/" in s:
        s = s.split("/", 1)[0]
    try:
        return float(s)
    except Exception:
        return np.nan


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def norm_spatial(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return (xy - lo) / span


def grid_codes(xy01: np.ndarray, grid_n: int = 5) -> list[str]:
    ij = np.clip((xy01 * grid_n).astype(int), 0, grid_n - 1)
    return [f"g{int(i)}_{int(j)}" for i, j in ij]


def top_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.2) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = len(a)
    if n == 0:
        return np.nan
    k = max(1, int(np.ceil(n * frac)))
    aa = set(np.argsort(a)[-k:].tolist())
    bb = set(np.argsort(b)[-k:].tolist())
    return float(len(aa & bb) / max(1, len(aa | bb)))


def read_h5ad(path: Path):
    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(f"anndata is required for spatial abundance/module metrics: {e}") from e
    if not path.exists():
        raise RuntimeError(f"Missing h5ad file: {path}")
    return ad.read_h5ad(path)


def grid_state_abundance(adata, label_col: str, grid_n: int = 5) -> pd.DataFrame:
    xy = norm_spatial(adata.obsm["spatial"])
    grids = grid_codes(xy, grid_n)
    labels = adata.obs[label_col].astype(str).to_numpy()
    all_grids = [f"g{i}_{j}" for i in range(grid_n) for j in range(grid_n)]
    tab = pd.crosstab(pd.Series(grids, name="grid"), pd.Series(labels, name="cell_type"), normalize="index")
    tab = tab.reindex(index=all_grids, columns=CELL_TYPES, fill_value=0.0)
    return tab


def matrix_from_adata(adata, genes: list[str]) -> np.ndarray:
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return np.zeros((adata.n_obs, 0))
    X = adata[:, present].X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    arr = np.asarray(arr, dtype=float)
    # Real Visium matrices are count-like. Virtual matrices are already
    # reconstructed on a log-like scale. This heuristic keeps both comparable
    # without pretending that we have exact per-cell expression.
    if np.nanmax(arr) > 20:
        arr = np.log1p(arr)
    return arr


def grid_module_scores(adata, grid_n: int = 5) -> pd.DataFrame:
    xy = norm_spatial(adata.obsm["spatial"])
    grids = np.asarray(grid_codes(xy, grid_n))
    all_grids = [f"g{i}_{j}" for i in range(grid_n) for j in range(grid_n)]
    rows = []
    for module, genes in MODULE_GENES.items():
        arr = matrix_from_adata(adata, genes)
        if arr.shape[1] == 0:
            score = np.zeros(adata.n_obs)
        else:
            score = arr.mean(axis=1)
        df = pd.DataFrame({"grid": grids, "score": score})
        g = df.groupby("grid")["score"].mean().reindex(all_grids, fill_value=0.0)
        rows.append(pd.Series(g.to_numpy(), index=all_grids, name=module))
    return pd.DataFrame(rows).T


def compute_spatial_state_module_metrics(k_out: Path, k: int) -> pd.DataFrame:
    rows: list[dict] = []
    for real_stage, virtual_stage in STAGE_PAIRS:
        real_path = k_out / f"real_benchmark/{real_stage}.h5ad"
        virt_path = k_out / f"virtual_commot_input/{virtual_stage}.h5ad"
        if not real_path.exists() or not virt_path.exists():
            rows.append(
                {
                    "agent_count": k,
                    "real_stage": real_stage,
                    "virtual_stage": virtual_stage,
                    "spatial_state_abundance_pearson": np.nan,
                    "spatial_state_abundance_rmse": np.nan,
                    "macrophage_hotspot_overlap": np.nan,
                    "fibroblast_hotspot_overlap": np.nan,
                    "spatial_module_pearson_mean": np.nan,
                    "spatial_module_rmse_mean": np.nan,
                    "spatial_module_hotspot_overlap_mean": np.nan,
                    "spatial_metric_status": f"missing {'real' if not real_path.exists() else 'virtual'} h5ad",
                }
            )
            continue
        real = read_h5ad(real_path)
        virt = read_h5ad(virt_path)

        real_ab = grid_state_abundance(real, "dominant_cell_type")
        virt_ab = grid_state_abundance(virt, "cell_type")
        macro_cols = ["macrophage", "recruited_monocyte_macrophage"]
        fib_cols = ["fibroblast_myofibroblast"]
        real_macro = real_ab[macro_cols].sum(axis=1).to_numpy()
        virt_macro = virt_ab[macro_cols].sum(axis=1).to_numpy()
        real_fib = real_ab[fib_cols].sum(axis=1).to_numpy()
        virt_fib = virt_ab[fib_cols].sum(axis=1).to_numpy()

        real_mod = grid_module_scores(real)
        virt_mod = grid_module_scores(virt)
        mod_pearsons = []
        mod_rmses = []
        mod_hotspots = []
        for m in MODULE_GENES:
            mod_pearsons.append(pearson_safe(real_mod[m].to_numpy(), virt_mod[m].to_numpy()))
            mod_rmses.append(rmse(real_mod[m].to_numpy(), virt_mod[m].to_numpy()))
            mod_hotspots.append(top_overlap(real_mod[m].to_numpy(), virt_mod[m].to_numpy()))

        rows.append(
            {
                "agent_count": k,
                "real_stage": real_stage,
                "virtual_stage": virtual_stage,
                "spatial_state_abundance_pearson": pearson_safe(real_ab.to_numpy(), virt_ab.to_numpy()),
                "spatial_state_abundance_rmse": rmse(real_ab.to_numpy(), virt_ab.to_numpy()),
                "macrophage_hotspot_overlap": top_overlap(real_macro, virt_macro),
                "fibroblast_hotspot_overlap": top_overlap(real_fib, virt_fib),
                "spatial_module_pearson_mean": float(np.nanmean(mod_pearsons)),
                "spatial_module_rmse_mean": float(np.nanmean(mod_rmses)),
                "spatial_module_hotspot_overlap_mean": float(np.nanmean(mod_hotspots)),
                "spatial_metric_status": "ok",
            }
        )
    return pd.DataFrame(rows)


def collect_metrics(out: Path, agent_counts: list[int], runtimes: dict[int, float]) -> pd.DataFrame:
    rows: list[dict] = []
    for k in agent_counts:
        k_out = out / f"K_{k}"
        f = k_out / "evaluation/commot_summary_metrics.csv"
        if not f.exists():
            raise RuntimeError(f"Missing evaluation metrics for K={k}: {f}")
        df = pd.read_csv(f)
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "agent_count": k,
                    "real_stage": r.get("real_stage"),
                    "virtual_stage": r.get("virtual_stage"),
                    "sender_receiver_pearson": pd.to_numeric(r.get("sender_receiver_pearson"), errors="coerce"),
                    "sender_receiver_spearman": pd.to_numeric(r.get("sender_receiver_spearman"), errors="coerce"),
                    "cross_cell_type_pearson": pd.to_numeric(r.get("cross_cell_type_pearson"), errors="coerce"),
                    "cross_cell_type_spearman": pd.to_numeric(r.get("cross_cell_type_spearman"), errors="coerce"),
                    "top10_edge_overlap": parse_intish(r.get("top10_edge_overlap")),
                    "cross_top10_edge_overlap": parse_intish(r.get("cross_top10_edge_overlap")),
                    "key_interactions_recovered": parse_intish(r.get("key_interactions_recovered")),
                    "n_common_edges_nonzero": parse_intish(r.get("n_common_edges_nonzero")),
                    "llm_call_count": count_llm_calls(k_out),
                    "runtime_minutes": runtimes.get(k, np.nan),
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(f"No K-specific COMMOT metrics found under {out}/K_*/evaluation.")
    spatial_rows = []
    for k in agent_counts:
        spatial_rows.append(compute_spatial_state_module_metrics(out / f"K_{k}", k))
    spatial = pd.concat(spatial_rows, ignore_index=True) if spatial_rows else pd.DataFrame()
    if not spatial.empty:
        metrics = metrics.merge(spatial, on=["agent_count", "real_stage", "virtual_stage"], how="outer")
        for c in ["llm_call_count", "runtime_minutes"]:
            if c in metrics.columns:
                metrics[c] = metrics.groupby("agent_count")[c].transform(lambda s: s.ffill().bfill())
    # A simple, transparent score for visual comparison only. It is not used
    # for training or model selection.
    metrics["spatial_commot_score"] = (
        metrics["sender_receiver_pearson"].fillna(0) * 0.35
        + metrics["cross_cell_type_pearson"].clip(lower=0).fillna(0) * 0.35
        + (metrics["top10_edge_overlap"].fillna(0) / 10.0) * 0.15
        + (metrics["cross_top10_edge_overlap"].fillna(0) / 10.0) * 0.10
        + (metrics["key_interactions_recovered"].fillna(0) / 6.0) * 0.05
    )
    metrics["spatial_state_module_score"] = (
        metrics["spatial_state_abundance_pearson"].clip(lower=0).fillna(0) * 0.35
        + metrics["spatial_module_pearson_mean"].clip(lower=0).fillna(0) * 0.35
        + metrics["macrophage_hotspot_overlap"].fillna(0) * 0.15
        + metrics["spatial_module_hotspot_overlap_mean"].fillna(0) * 0.15
    )
    metrics["overall_spatial_fidelity_score"] = (
        metrics["spatial_commot_score"] * 0.50 + metrics["spatial_state_module_score"] * 0.50
    )
    ensure(out / "evaluation")
    metrics.to_csv(out / "evaluation/spatial_agent_granularity_metrics.csv", index=False)
    return metrics


def plot_summary(out: Path, metrics: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure(out / "figures")
    stages = list(dict.fromkeys(metrics["real_stage"].astype(str).tolist()))
    ks = sorted(metrics["agent_count"].unique())
    labels = [f"K={k}" for k in ks]
    colors = {"d7_bleo": "#4C78A8", "d21_bleo": "#F58518"}
    fallback = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    fig, axs = plt.subplots(2, 3, figsize=(16.2, 8.8), dpi=220)
    fig.suptitle(
        "GSE267904 spatial Agent granularity: state, module and COMMOT recovery",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    width = 0.34 if len(stages) <= 2 else 0.25
    x = np.arange(len(ks))

    def values(stage: str, col: str) -> list[float]:
        sub = metrics[metrics["real_stage"].astype(str).eq(stage)].set_index("agent_count")
        return [float(sub[col].get(k, np.nan)) if k in sub.index else np.nan for k in ks]

    # A. Spatial state abundance recovery
    ax = axs[0, 0]
    for i, st in enumerate(stages):
        off = (i - (len(stages) - 1) / 2) * width
        vals = values(st, "spatial_state_abundance_pearson")
        ax.bar(x + off, vals, width=width, color=colors.get(st, fallback[i % len(fallback)]), alpha=0.86, label=st)
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Grid state-abundance Pearson ↑")
    ax.set_title("A. Spatial cell-state abundance", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # B. Spatial module hotspot/expression recovery
    ax = axs[0, 1]
    for i, st in enumerate(stages):
        off = (i - (len(stages) - 1) / 2) * width
        vals = values(st, "spatial_module_pearson_mean")
        ax.bar(x + off, vals, width=width, color=colors.get(st, fallback[i % len(fallback)]), alpha=0.86, label=st)
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean module Pearson ↑")
    ax.set_title("B. Spatial module expression", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # C. Overall sender-receiver recovery
    ax = axs[0, 2]
    for i, st in enumerate(stages):
        off = (i - (len(stages) - 1) / 2) * width
        vals = values(st, "sender_receiver_pearson")
        ax.bar(x + off, vals, width=width, color=colors.get(st, fallback[i % len(fallback)]), alpha=0.86, label=st)
        for xx, yy in zip(x + off, vals):
            if np.isfinite(yy):
                ax.text(xx, yy + 0.015, f"{yy:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overall sender-receiver Pearson ↑")
    ax.set_title("C. Overall COMMOT edge recovery", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # D. Cross-cell-type recovery
    ax = axs[1, 0]
    for i, st in enumerate(stages):
        off = (i - (len(stages) - 1) / 2) * width
        vals = values(st, "cross_cell_type_pearson")
        ax.bar(x + off, vals, width=width, color=colors.get(st, fallback[i % len(fallback)]), alpha=0.86, label=st)
        for xx, yy in zip(x + off, vals):
            if np.isfinite(yy):
                ax.text(xx, yy + (0.025 if yy >= 0 else -0.055), f"{yy:.2f}", ha="center", va="bottom" if yy >= 0 else "top", fontsize=8)
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cross-cell-type Pearson ↑")
    ax.set_title("D. Cross-cell-type communication", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # E. Top edge / key interaction recovery
    ax = axs[1, 1]
    agg = (
        metrics.groupby("agent_count", as_index=False)
        .agg(
            top10=("top10_edge_overlap", "mean"),
            cross_top10=("cross_top10_edge_overlap", "mean"),
            key=("key_interactions_recovered", "mean"),
        )
        .set_index("agent_count")
    )
    top_vals = [agg.loc[k, "top10"] if k in agg.index else np.nan for k in ks]
    cross_vals = [agg.loc[k, "cross_top10"] if k in agg.index else np.nan for k in ks]
    key_vals = [agg.loc[k, "key"] if k in agg.index else np.nan for k in ks]
    ax.bar(x - width, top_vals, width=width, color="#72B7B2", label="Overall top-10 overlap /10")
    ax.bar(x, cross_vals, width=width, color="#E45756", label="Cross-type top-10 overlap /10")
    ax.bar(x + width, key_vals, width=width, color="#54A24B", label="Key interactions /6")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Recovered count ↑")
    ax.set_title("E. Top-edge and key interaction recovery", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # F. Score-cost trade-off
    ax = axs[1, 2]
    score = metrics.groupby("agent_count")["overall_spatial_fidelity_score"].mean()
    calls = metrics.groupby("agent_count")["llm_call_count"].max()
    score_vals = [score.get(k, np.nan) for k in ks]
    call_vals = [calls.get(k, np.nan) for k in ks]
    best_k = ks[int(np.nanargmax(score_vals))] if any(np.isfinite(score_vals)) else None
    bar_colors = ["#4C78A8" if k != best_k else "#F2A23A" for k in ks]
    ax.bar(x, score_vals, color=bar_colors, alpha=0.9)
    for xx, yy, k in zip(x, score_vals, ks):
        if np.isfinite(yy):
            ax.text(xx, yy + 0.01, f"{yy:.2f}", ha="center", va="bottom", fontsize=8)
            if k == best_k:
                ax.text(xx, max(0.02, yy * 0.50), "best", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(0.25, np.nanmax(score_vals) * 1.25 if any(np.isfinite(score_vals)) else 1))
    ax.set_ylabel("Composite spatial fidelity score ↑")
    ax.set_title("F. Fidelity-cost trade-off", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, call_vals, color="#7A5195", marker="o", lw=2.0, label="LLM calls")
    ax2.set_ylabel("LLM calls ↑", color="#7A5195")
    ax2.tick_params(axis="y", colors="#7A5195")

    fig.text(
        0.5,
        0.02,
        "K is the number of PhysiCell-backed LLM micro-agents. Real d7/d21 bleomycin spatial data are fixed; only virtual Agent granularity changes. "
        "State/module metrics are grid-level spatial comparisons; COMMOT metrics are sender-receiver network comparisons.",
        ha="center",
        fontsize=9,
        color="#4A5568",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.995, 0.955))
    out_png = fig_dir / "gse267904_spatial_agent_granularity_summary.png"
    out_pdf = fig_dir / "gse267904_spatial_agent_granularity_summary.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def build_shared_real(root: Path, shared: Path, args: argparse.Namespace) -> None:
    if not args.skip_download:
        call(root, "download_gse267904.py", shared)
    call(root, "preflight_commot.py", shared)
    manifest = shared / "real_benchmark/real_spatial_benchmark_manifest.csv"
    if args.reuse_existing_real and manifest.exists() and (shared / "real_commot/real_sender_receiver_matrix_by_stage.csv").exists():
        print(f"Reusing existing shared real benchmark and real COMMOT under {shared}", flush=True)
        return
    call(root, "build_real_spatial_benchmark.py", shared, "--max-samples-per-stage", args.max_samples_per_stage)
    call(root, "annotate_spots_by_markers.py", shared)
    call(root, "run_real_commot.py", shared, "--max-spots", args.max_commot_spots)


def run_one_k(root: Path, shared: Path, k_out: Path, k: int, args: argparse.Namespace) -> float:
    ensure(k_out)
    for name in ["real_benchmark", "celltype_annotation", "real_commot", "audit"]:
        src = shared / name
        if src.exists():
            copy_dir(src, k_out / name)
    t0 = time.time()
    call(root, "build_micro_agent_registry.py", k_out, "--num-agents", k)
    phys_args: list[object] = ["--max-time", args.max_time]
    if args.allow_debug_model_spatial_generator:
        phys_args.append("--allow-debug-model-spatial-generator")
    call(root, "run_physicell_micro_agent_simulation.py", k_out, *phys_args)
    call(root, "build_virtual_commot_input.py", k_out)
    call(root, "run_virtual_commot.py", k_out)
    call(root, "evaluate_real_vs_virtual_commot.py", k_out)
    return (time.time() - t0) / 60.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_agent_granularity"))
    p.add_argument("--agent-counts", default="5,20,50")
    p.add_argument("--max-samples-per-stage", type=int, default=1)
    p.add_argument("--max-commot-spots", type=int, default=800)
    p.add_argument("--max-time", type=float, default=125.0)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--reuse-existing-real", action="store_true", help="Reuse shared real benchmark/real COMMOT if already present.")
    p.add_argument("--allow-debug-model-spatial-generator", action="store_true", help="Debug only; not for final real PhysiCell-backed run.")
    p.add_argument("--summary-only", action="store_true", help="Do not rerun simulations; rebuild summary CSV/figure from existing K_* outputs.")
    args = p.parse_args()

    root = args.project_root.resolve()
    out = abs_out(root, args.out_dir).resolve()
    ensure(out)
    agent_counts = [int(x.strip()) for x in str(args.agent_counts).split(",") if x.strip()]
    if not agent_counts:
        raise ValueError("--agent-counts is empty")

    runtimes: dict[int, float] = {}
    if not args.summary_only:
        shared = out / "shared_real"
        ensure(shared)
        build_shared_real(root, shared, args)

        for k in agent_counts:
            print(f"\n=== Running spatial micro-agent granularity K={k} ===", flush=True)
            runtimes[k] = run_one_k(root, shared, out / f"K_{k}", k, args)
    else:
        print("Summary-only mode: reusing existing K_* outputs and regenerating metrics/figure.", flush=True)

    metrics = collect_metrics(out, agent_counts, runtimes)
    fig = plot_summary(out, metrics)
    audit = {
        "dataset_id": "GSE267904",
        "experiment": "spatial_micro_agent_granularity_ablation",
        "agent_counts": agent_counts,
        "real_spatial_benchmark_shared_once": True,
        "real_commot_shared_once": True,
        "virtual_commot_recomputed_for_each_K": True,
        "physicell_backed_micro_agent_simulation": not args.allow_debug_model_spatial_generator,
        "debug_model_spatial_generator_allowed": bool(args.allow_debug_model_spatial_generator),
        "heldout_d21_used_in_agent_prompt": False,
        "metrics_csv": str(out / "evaluation/spatial_agent_granularity_metrics.csv"),
        "main_figure": str(fig),
    }
    ensure(out / "audit")
    (out / "audit/spatial_agent_granularity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"\nWrote spatial Agent granularity metrics to {out/'evaluation/spatial_agent_granularity_metrics.csv'}")
    print(f"Main figure: {fig}")


if __name__ == "__main__":
    main()
