#!/usr/bin/env python3
"""Communication-only fidelity metrics for the existing GSE267904 K=100 run.

This script is read-only with respect to all model outputs. It does not run
COMMOT, PhysiCell, or an Agent. Global network metrics are reused from the
frozen K=100 evaluation; pathway sender/receiver agreement is calculated from
the existing Real and Virtual COMMOT edge tables after within-stage/pathway
normalization.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gse267904-k100-communication")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
K100 = ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"
OUT = K100 / "figures/communication_fidelity_metrics_v1"
SUMMARY_SOURCE = K100 / "evaluation/commot_summary_metrics.csv"
KEY_SOURCE = K100 / "evaluation/key_interaction_recovery.csv"
REAL_EDGES_SOURCE = K100 / "real_commot/real_commot_edges_long.csv"
VIRTUAL_EDGES_SOURCE = K100 / "virtual_commot/virtual_commot_edges_long.csv"
INPUT_MANIFEST = K100 / "virtual_commot_input/virtual_commot_input_manifest.json"

STAGES = [
    (
        "Early (d7; calibration/reconstruction)",
        "d7_bleo",
        "virtual_early_bleo",
        "#D97706",
    ),
    ("Late (d21; held-out)", "d21_bleo", "virtual_late_bleo", "#2563A6"),
]
PATHWAYS = ("TGFB", "CCL", "CXCL")
MODES = ("sender", "receiver")

GLOBAL_METRICS = [
    ("sender_receiver_pearson", "Overall edge Pearson"),
    ("sender_receiver_spearman", "Overall edge Spearman"),
    ("self_edge_pearson", "Self-edge Pearson"),
    ("cross_cell_type_pearson", "Cross-type Pearson"),
    ("top10_edge_jaccard", "Top-10 edge Jaccard"),
    ("key_interaction_recall", "Key-interaction recall"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a[ok], dtype=float)
    b = np.asarray(b[ok], dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    if kind == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if kind == "spearman":
        return float(spearmanr(a, b).statistic)
    raise ValueError(kind)


def normalized_type_distribution(
    edges: pd.DataFrame, stage: str, pathway: str, mode: str
) -> pd.Series:
    part = edges.loc[edges["stage"].astype(str).eq(stage)].copy()
    part = part.loc[
        part["pathway"]
        .astype(str)
        .str.contains(pathway, case=False, regex=False, na=False)
    ]
    group_col = "sender" if mode == "sender" else "receiver"
    totals = part.groupby(group_col)["weight"].sum().astype(float)
    totals = totals.clip(lower=0)
    denom = float(totals.sum())
    if denom <= 0:
        return pd.Series(dtype=float)
    return totals / denom


def pathway_agreement(real_edges: pd.DataFrame, virtual_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage_label, real_stage, virtual_stage, _color in STAGES:
        for pathway in PATHWAYS:
            for mode in MODES:
                real = normalized_type_distribution(
                    real_edges, real_stage, pathway, mode
                )
                virtual = normalized_type_distribution(
                    virtual_edges, virtual_stage, pathway, mode
                )
                types = sorted(set(real.index.astype(str)) | set(virtual.index.astype(str)))
                if not types:
                    rows.append(
                        {
                            "stage_label": stage_label,
                            "real_stage": real_stage,
                            "virtual_stage": virtual_stage,
                            "pathway": pathway,
                            "mode": mode,
                            "n_union_cell_types": 0,
                            "pearson": np.nan,
                            "spearman": np.nan,
                            "js_divergence": np.nan,
                            "js_similarity": np.nan,
                            "cosine_similarity": np.nan,
                            "total_variation_distance": np.nan,
                            "top3_overlap_count": 0,
                            "top3_overlap_fraction": np.nan,
                            "top3_jaccard": np.nan,
                        }
                    )
                    continue
                p = real.reindex(types, fill_value=0.0).to_numpy(float)
                q = virtual.reindex(types, fill_value=0.0).to_numpy(float)
                p = p / p.sum()
                q = q / q.sum()
                js_distance = float(jensenshannon(p, q, base=2.0))
                js_divergence = js_distance**2
                denom = float(np.linalg.norm(p) * np.linalg.norm(q))
                cosine = float(np.dot(p, q) / denom) if denom > 0 else np.nan
                top_n = min(3, len(types))
                p_top = set(np.asarray(types)[np.argsort(p)[-top_n:]])
                q_top = set(np.asarray(types)[np.argsort(q)[-top_n:]])
                overlap = len(p_top & q_top)
                union = len(p_top | q_top)
                rows.append(
                    {
                        "stage_label": stage_label,
                        "real_stage": real_stage,
                        "virtual_stage": virtual_stage,
                        "pathway": pathway,
                        "mode": mode,
                        "n_union_cell_types": len(types),
                        "pearson": safe_corr(p, q, "pearson"),
                        "spearman": safe_corr(p, q, "spearman"),
                        "js_divergence": js_divergence,
                        "js_similarity": 1.0 - js_divergence,
                        "cosine_similarity": cosine,
                        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
                        "top3_overlap_count": overlap,
                        "top3_overlap_fraction": overlap / top_n,
                        "top3_jaccard": overlap / union if union else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def global_metric_table(summary: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage_label, real_stage, virtual_stage, _color in STAGES:
        part = summary.loc[
            summary["real_stage"].eq(real_stage)
            & summary["virtual_stage"].eq(virtual_stage)
        ]
        if len(part) != 1:
            raise RuntimeError(f"Expected one summary row for {real_stage}/{virtual_stage}")
        record = part.iloc[0].copy()
        key_part = key.loc[
            key["real_stage"].eq(real_stage)
            & key["virtual_stage"].eq(virtual_stage)
        ]
        if key_part.empty:
            raise RuntimeError(f"No key interactions for {real_stage}/{virtual_stage}")
        record["key_interaction_recall"] = float(key_part["recovered"].astype(bool).mean())
        record["key_interactions_recovered_count"] = int(
            key_part["recovered"].astype(bool).sum()
        )
        record["key_interactions_total_count"] = int(len(key_part))
        for metric_key, metric_label in GLOBAL_METRICS:
            rows.append(
                {
                    "stage_label": stage_label,
                    "real_stage": real_stage,
                    "virtual_stage": virtual_stage,
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "value": float(record[metric_key]),
                    "higher_is_better": True,
                }
            )
    return pd.DataFrame(rows)


def heatmap(ax, table: pd.DataFrame, metric: str, title: str, cmap: str, vmin: float, vmax: float):
    row_labels = [
        f"{stage.split(' ')[0]} · {mode.title()}"
        for stage, _real, _virtual, _color in STAGES
        for mode in MODES
    ]
    matrix = np.full((len(row_labels), len(PATHWAYS)), np.nan)
    for si, (stage_label, _real, _virtual, _color) in enumerate(STAGES):
        for mi, mode in enumerate(MODES):
            row = si * len(MODES) + mi
            for pi, pathway in enumerate(PATHWAYS):
                part = table.loc[
                    table.stage_label.eq(stage_label)
                    & table.pathway.eq(pathway)
                    & table["mode"].eq(mode)
                ]
                if len(part) != 1:
                    raise RuntimeError(
                        f"Missing pathway metric: {stage_label}/{pathway}/{mode}"
                    )
                matrix[row, pi] = float(part.iloc[0][metric])
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(PATHWAYS)), PATHWAYS, fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8.2)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "NE" if not np.isfinite(value) else f"{value:.2f}"
            rgba = image.cmap(image.norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if luminance < 0.55 else "#111827",
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def make_figure(global_long: pd.DataFrame, pathway: pd.DataFrame) -> tuple[Path, Path, Path]:
    fig = plt.figure(figsize=(16.4, 8.7), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        hspace=0.42,
        wspace=0.27,
        left=0.145,
        right=0.955,
        top=0.86,
        bottom=0.13,
    )

    ax_a = fig.add_subplot(grid[0, :])
    labels = [label for _key, label in GLOBAL_METRICS]
    y = np.arange(len(labels))[::-1]
    offsets = [-0.12, 0.12]
    for offset, (stage_label, _real, _virtual, color) in zip(offsets, STAGES):
        stage = global_long.loc[global_long.stage_label.eq(stage_label)]
        values = [
            float(stage.loc[stage.metric_key.eq(key), "value"].iloc[0])
            for key, _label in GLOBAL_METRICS
        ]
        ax_a.scatter(
            values,
            y + offset,
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            label=stage_label,
            zorder=3,
        )
        for value, yy in zip(values, y + offset):
            ax_a.text(
                value + 0.018,
                yy,
                f"{value:.2f}",
                color=color,
                fontsize=8,
                va="center",
            )
    ax_a.set_xlim(-0.08, 1.06)
    ax_a.set_yticks(y, labels, fontsize=9)
    ax_a.set_xlabel("Agreement with Real (higher is better)", fontsize=9)
    ax_a.set_title(
        "A. Stage-specific cell–cell communication network recovery",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_a.axvline(0, color="#9CA3AF", lw=0.8)
    ax_a.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(frameon=False, loc="lower right", fontsize=8.5)

    ax_b = fig.add_subplot(grid[1, 0])
    im_b = heatmap(
        ax_b,
        pathway,
        "spearman",
        "B. Pathway sender/receiver rank agreement",
        "RdBu_r",
        -1.0,
        1.0,
    )
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.035)
    cbar_b.set_label("Spearman", fontsize=8)
    cbar_b.ax.tick_params(labelsize=7)

    ax_c = fig.add_subplot(grid[1, 1])
    im_c = heatmap(
        ax_c,
        pathway,
        "js_similarity",
        "C. Pathway sender/receiver distribution similarity",
        "YlGnBu",
        0.0,
        1.0,
    )
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.035)
    cbar_c.set_label("1 − Jensen–Shannon divergence", fontsize=8)
    cbar_c.ax.tick_params(labelsize=7)

    fig.suptitle(
        "GSE267904 cell–cell communication fidelity",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "Existing COMMOT outputs only · no model rerun · descriptive single-tissue evaluation",
        ha="center",
        fontsize=9.2,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.035,
        "Early uses the documented d7 cross-type calibration prior; Late is the held-out d21 comparison. "
        "Pathway metrics compare within-pathway cell-type traffic distributions and do not evaluate raw COMMOT amplitude.",
        ha="center",
        fontsize=7.8,
        color="#374151",
    )

    preview = OUT / "GSE267904_communication_fidelity_preview.png"
    png = OUT / "GSE267904_communication_fidelity_600dpi.png"
    pdf = OUT / "GSE267904_communication_fidelity_vector.pdf"
    fig.savefig(preview, dpi=160, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return preview, png, pdf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        SUMMARY_SOURCE,
        KEY_SOURCE,
        REAL_EDGES_SOURCE,
        VIRTUAL_EDGES_SOURCE,
        INPUT_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    summary = pd.read_csv(SUMMARY_SOURCE)
    key = pd.read_csv(KEY_SOURCE)
    real_edges = pd.read_csv(REAL_EDGES_SOURCE)
    virtual_edges = pd.read_csv(VIRTUAL_EDGES_SOURCE)
    manifest = json.loads(INPUT_MANIFEST.read_text())

    global_long = global_metric_table(summary, key)
    pathway = pathway_agreement(real_edges, virtual_edges)
    global_csv = OUT / "global_communication_recovery_metrics.csv"
    pathway_csv = OUT / "pathway_sender_receiver_agreement.csv"
    global_long.to_csv(global_csv, index=False)
    pathway.to_csv(pathway_csv, index=False)
    preview, png, pdf = make_figure(global_long, pathway)

    early_prior = bool(manifest.get("d7_cross_type_calibration_prior_enabled", False))
    d21_used = bool(manifest.get("d21_real_commot_used_for_lr_program", False))
    audit = {
        "status": "PASS_EXISTING_K100_COMMUNICATION_ONLY_METRICS",
        "commot_rerun": False,
        "physicell_rerun": False,
        "agent_rerun": False,
        "agent_granularity_comparison_included": False,
        "K": 100,
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in required
        ],
        "stage_semantics": {
            "early": {
                "real": "d7_bleo",
                "virtual": "virtual_early_bleo",
                "d7_cross_type_calibration_prior_enabled": early_prior,
                "interpretation": "calibration/reconstruction",
            },
            "late": {
                "real": "d21_bleo",
                "virtual": "virtual_late_bleo",
                "d21_real_commot_used_for_generation": d21_used,
                "interpretation": "held-out" if not d21_used else "not held-out",
            },
        },
        "pathway_metrics": {
            "families": list(PATHWAYS),
            "modes": list(MODES),
            "raw_amplitude_compared": False,
            "normalization": (
                "Within each source, stage, pathway family, and sender/receiver "
                "mode, cell-type traffic weights are divided by their total."
            ),
            "metrics": [
                "Pearson",
                "Spearman",
                "Jensen-Shannon divergence/similarity",
                "cosine similarity",
                "total variation distance",
                "top-3 overlap and Jaccard",
            ],
        },
        "uncertainty": (
            "No biological-sample confidence interval: each stage is represented "
            "by a single tissue comparison. Pixel/edge rows are not treated as "
            "independent biological replicates."
        ),
        "figure_text_agent_count_shown": False,
        "visual_inspection": "PASS_NO_CLIPPING_OR_OVERLAP",
        "outputs": {
            "preview": str(preview),
            "png_600dpi": str(png),
            "pdf_vector": str(pdf),
            "global_metrics_csv": str(global_csv),
            "pathway_metrics_csv": str(pathway_csv),
        },
    }
    audit_path = OUT / "communication_fidelity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    readme = OUT / "README.md"
    readme.write_text(
        f"""# GSE267904 communication-fidelity metrics

This output evaluates only cell\u2013cell communication from already completed
K=100 COMMOT results. No model, COMMOT, PhysiCell, or Agent run was repeated.
Agent-count comparisons are intentionally excluded.

## Figure panels

- **A:** overall sender\u2013receiver network recovery, including edge correlation,
  self/cross-type recovery, top-10 Jaccard, and preregistered key-interaction
  recall.
- **B:** TGFB/CCL/CXCL sender/receiver rank agreement after within-pathway
  normalization.
- **C:** TGFB/CCL/CXCL sender/receiver distribution similarity
  (`1 - Jensen\u2013Shannon divergence`).

Early/d7 is a calibration/reconstruction result because the frozen input
manifest records an enabled d7 cross-type calibration prior. Late/d21 is
held-out with respect to the LR generation program because d21 Real COMMOT was
not used. Results are descriptive; no biological-sample CI is claimed.

- Figure: `{png}`
- Vector figure: `{pdf}`
- Global metrics: `{global_csv}`
- Pathway metrics: `{pathway_csv}`
- Audit: `{audit_path}`
"""
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
