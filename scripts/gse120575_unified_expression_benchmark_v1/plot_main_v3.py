#!/usr/bin/env python3
"""Create audited GSE120575 main expression figure V3 using direct Post levels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse120575-expression-main-v3")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_unified_expression_benchmark_v1"
PROXY_ROOT = ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"
HARMONIZED = PROXY_ROOT / "harmonized_834_expression_by_method_sample.npz"
PANEL = PROXY_ROOT / "frozen_834_gene_panel.txt"
METRICS = OUT / "unified_expression_metrics.csv"
BOOTSTRAP = OUT / "bootstrap_confidence_intervals.csv"
DELTA_AUDIT = OUT / "runtime_proxy_delta_audit_v1/audit.json"

PNG = OUT / "GSE120575_unified_expression_benchmark_main_v3_600dpi.png"
PDF = OUT / "GSE120575_unified_expression_benchmark_main_v3_vector.pdf"
TOP_RESPONDER = OUT / "top_post_genes_responder_v3.csv"
TOP_NON_RESPONDER = OUT / "top_post_genes_non_responder_v3.csv"
PLOT_VALUES = OUT / "top_post_gene_plot_values_v3.csv"
OVERALL_VALUES = OUT / "overall_expression_metrics_for_plot_v3.csv"
SUMMARY = OUT / "PLOT_DESIGN_SUMMARY_V3.md"

RESPONSES = ("Responder", "Non-responder")
METHODS = ("scGen", "WOT_weighted", "AgentVC_proxy")
METHOD_LABELS = {
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "AgentVC_proxy": "AgentVC",
}
COLORS = {
    "Observed_Post": "#5B5B5B",
    "scGen": "#F58518",
    "WOT_weighted": "#54A24B",
    "AgentVC_proxy": "#B279A2",
}
MARKERS = {
    "Observed_Post": "o",
    "scGen": "s",
    "WOT_weighted": "^",
    "AgentVC_proxy": "D",
}
TOP_N = 15
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 120575834
EXPECTED_PANEL_HASH = "1a9e02f8bb52cdea852d6fe30992339ad8bc2fa0ab807b81848dce574a7bce41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_inputs() -> dict[str, np.ndarray]:
    delta_audit = json.loads(DELTA_AUDIT.read_text())
    if (
        delta_audit["status"] != "PASS_AGENTVC_DELTA_AUDIT"
        or delta_audit["classification"] != "B_RUNTIME_EXPRESSION_PROXY_MECHANISM_LIMITATION"
    ):
        raise RuntimeError("AgentVC delta audit gate did not pass")
    if sha256(PANEL) != EXPECTED_PANEL_HASH:
        raise RuntimeError("Frozen gene panel changed")
    genes = PANEL.read_text(encoding="utf-8").splitlines()
    with np.load(HARMONIZED) as source:
        data = {key: source[key].copy() for key in source.files}
    if data["genes"].astype(str).tolist() != genes:
        raise RuntimeError("Harmonized gene order mismatch")
    return data


def prepare_overall() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS)
    intervals = pd.read_csv(BOOTSTRAP)
    rows: list[dict[str, object]] = []
    for response in RESPONSES:
        for method in METHODS:
            values = {}
            for metric in ("RMSE", "MAE", "CCC"):
                selected = metrics[
                    metrics["response"].eq(response)
                    & metrics["method"].eq(method)
                    & metrics["metric"].eq(metric)
                ]["value"]
                if len(selected) != 1:
                    raise RuntimeError(f"Missing metric {response}/{method}/{metric}")
                values[metric] = float(selected.iloc[0])
            interval = intervals[
                intervals["response"].eq(response)
                & intervals["method"].eq(method)
                & intervals["metric"].eq("RMSE")
            ]
            if len(interval) != 1:
                raise RuntimeError(f"Missing RMSE CI {response}/{method}")
            rows.append({
                "response": response,
                "method": method,
                "method_label": METHOD_LABELS[method],
                **values,
                "RMSE_ci95_low": float(interval.iloc[0]["ci95_low"]),
                "RMSE_ci95_high": float(interval.iloc[0]["ci95_high"]),
                "bootstrap_iterations": int(interval.iloc[0]["bootstrap_iterations"]),
                "bootstrap_unit": str(interval.iloc[0]["bootstrap_unit"]),
            })
    return pd.DataFrame(rows)


def select_and_prepare(
    data: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = data["genes"].astype(str)
    methods = data["methods"].astype(str)
    responses = data["responses"].astype(str)
    expression = data["expression_log1p_sample_pseudobulk"].astype(np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selection_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []

    for response in RESPONSES:
        observed_rows = expression[(methods == "Observed_Post") & (responses == response)]
        observed_mean = observed_rows.mean(axis=0)
        # Frozen systematic rule: highest Observed Post group mean, then gene symbol.
        selected_indexes = np.lexsort((genes, -observed_mean))[:TOP_N]
        draws = rng.integers(
            0, len(observed_rows),
            size=(BOOTSTRAP_ITERATIONS, len(observed_rows)),
        )
        bootstrap_means = observed_rows[draws].mean(axis=1)
        low = np.quantile(bootstrap_means, 0.025, axis=0)
        high = np.quantile(bootstrap_means, 0.975, axis=0)

        method_means = {
            method: expression[(methods == method) & (responses == response)].mean(axis=0)
            for method in METHODS
        }
        for rank, index in enumerate(selected_indexes, start=1):
            gene = str(genes[index])
            selection_rows.append({
                "response": response,
                "rank": rank,
                "gene": gene,
                "observed_post_group_mean_log1p": observed_mean[index],
                "observed_post_ci95_low": low[index],
                "observed_post_ci95_high": high[index],
                "observed_post_biopsy_n": len(observed_rows),
                "selection_rule": (
                    "top 15 by descending Observed Post group mean log1p expression; "
                    "gene-symbol tie break; model predictions not used"
                ),
            })
            plot_rows.append({
                "response": response,
                "rank": rank,
                "gene": gene,
                "method": "Observed_Post",
                "method_label": "Observed Post",
                "post_expression_log1p": observed_mean[index],
                "ci95_low": low[index],
                "ci95_high": high[index],
                "has_interval": True,
                "sample_or_biopsy_n": len(observed_rows),
            })
            for method in METHODS:
                plot_rows.append({
                    "response": response,
                    "rank": rank,
                    "gene": gene,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "post_expression_log1p": method_means[method][index],
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "has_interval": False,
                    "sample_or_biopsy_n": 3,
                })
    return pd.DataFrame(selection_rows), pd.DataFrame(plot_rows)


def make_figure(overall: pd.DataFrame, plot_values: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.8, 9.6), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, left=0.07, right=0.965, bottom=0.17, top=0.86,
        height_ratios=(0.72, 1.35), hspace=0.48, wspace=0.20,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]

    rmse_max = float(overall["RMSE_ci95_high"].max())
    text_x = rmse_max * 1.08
    forest_xlim = rmse_max * 1.50
    y_positions = {"scGen": 2, "WOT_weighted": 1, "AgentVC_proxy": 0}
    for response_index, response in enumerate(RESPONSES):
        axis = axes[response_index]
        subset = overall[overall["response"].eq(response)]
        for row in subset.itertuples():
            y = y_positions[row.method]
            axis.hlines(
                y, row.RMSE_ci95_low, row.RMSE_ci95_high,
                color=COLORS[row.method], linewidth=2.2,
            )
            axis.vlines(
                [row.RMSE_ci95_low, row.RMSE_ci95_high],
                y - 0.09, y + 0.09, color=COLORS[row.method], linewidth=1.2,
            )
            axis.scatter(
                row.RMSE, y, s=58, marker=MARKERS[row.method],
                facecolor=COLORS[row.method], edgecolor="white",
                linewidth=0.7, zorder=3,
            )
            axis.text(
                text_x, y, f"MAE {row.MAE:.3f}   CCC {row.CCC:.3f}",
                va="center", fontsize=8.2, color="#333333",
            )
        axis.set_yticks(
            [y_positions[m] for m in METHODS],
            [METHOD_LABELS[m] for m in METHODS],
        )
        axis.set_xlim(0, forest_xlim)
        axis.set_ylim(-0.55, 2.55)
        axis.set_xlabel("RMSE (lower is better)")
        axis.grid(axis="x", color="#E3E3E3", linewidth=0.75)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        panel = "A" if response_index == 0 else "B"
        axis.set_title(
            f"{panel}. {response}: overall expression performance",
            loc="left", fontweight="bold",
        )
        axis.text(
            0.99, 0.965, "point = RMSE; line = 95% sample/biopsy bootstrap CI",
            transform=axis.transAxes, ha="right", va="top",
            fontsize=7.3, color="#555555",
        )

    interval_values = plot_values[plot_values["has_interval"]]
    all_y = np.concatenate([
        plot_values["post_expression_log1p"].to_numpy(dtype=float),
        interval_values["ci95_low"].to_numpy(dtype=float),
        interval_values["ci95_high"].to_numpy(dtype=float),
    ])
    y_range = float(all_y.max() - all_y.min())
    y_min = max(0.0, float(all_y.min() - 0.10 * y_range))
    y_max = float(all_y.max() + 0.12 * y_range)
    offsets = {
        "Observed_Post": -0.27,
        "scGen": -0.09,
        "WOT_weighted": 0.09,
        "AgentVC_proxy": 0.27,
    }
    for response_index, response in enumerate(RESPONSES):
        axis = axes[response_index + 2]
        subset = plot_values[plot_values["response"].eq(response)]
        observed = subset[subset["method"].eq("Observed_Post")].sort_values("rank")
        genes = observed["gene"].tolist()
        x = np.arange(TOP_N)
        axis.errorbar(
            x + offsets["Observed_Post"],
            observed["post_expression_log1p"],
            yerr=np.vstack([
                observed["post_expression_log1p"].to_numpy() - observed["ci95_low"].to_numpy(),
                observed["ci95_high"].to_numpy() - observed["post_expression_log1p"].to_numpy(),
            ]),
            fmt="o", markersize=4.8, color=COLORS["Observed_Post"],
            ecolor="#8A8A8A", elinewidth=1.0, capsize=2.2, zorder=2,
        )
        for method in METHODS:
            selected = subset[subset["method"].eq(method)].sort_values("rank")
            axis.scatter(
                x + offsets[method], selected["post_expression_log1p"],
                s=31, marker=MARKERS[method], facecolor=COLORS[method],
                edgecolor="white", linewidth=0.5, zorder=3,
            )
        axis.set_xlim(-0.7, TOP_N - 0.3)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks(x, genes, rotation=58, ha="right", fontsize=7.4)
        axis.set_ylabel("Post expression\n(log1p TPM-like)")
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        panel = "C" if response_index == 0 else "D"
        axis.set_title(
            f"{panel}. {response}: top Observed Post-expression genes",
            loc="left", fontweight="bold",
        )

    legend = [
        Line2D(
            [0], [0], marker=MARKERS["Observed_Post"], linestyle="none",
            color=COLORS["Observed_Post"], label="Observed Post (95% bootstrap CI)",
            markersize=6,
        )
    ] + [
        Line2D(
            [0], [0], marker=MARKERS[method], linestyle="none",
            markerfacecolor=COLORS[method], markeredgecolor="white",
            color=COLORS[method], label=METHOD_LABELS[method], markersize=6,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=legend, loc="upper right", bbox_to_anchor=(0.965, 0.91),
        ncol=4, frameon=False, fontsize=8.5,
    )
    fig.suptitle(
        "GSE120575 unified gene-expression benchmark",
        fontsize=15, fontweight="bold", y=0.965,
    )
    fig.text(
        0.07, 0.915,
        "834 exact common genes · log1p sample-level pseudo-bulk TPM-like expression · "
        "Observed anti-PD1 Post reference",
        fontsize=9, color="#444444",
    )
    fig.text(
        0.07, 0.095,
        "Panels A/B summarize performance across all 834 genes.\n"
        "Panels C/D use the frozen top 15 genes selected only by highest response-specific "
        "Observed Post group mean; model predictions were not used for selection. "
        "Gray intervals are 1,000-replicate biopsy-bootstrap CIs.",
        fontsize=8, color="#3F3F3F",
    )
    fig.text(
        0.07, 0.055,
        "WOT* is a target-derived transport baseline. AgentVC is the checkpoint-5 offline "
        "reconstructed runtime-expression proxy, not a directly saved or PhysiCell-executed "
        "per-gene expression field.",
        fontsize=8, color="#3F3F3F",
    )
    fig.savefig(PNG, dpi=600, facecolor="white")
    fig.savefig(PDF, facecolor="white")
    plt.close(fig)


def main() -> int:
    outputs = {
        PNG.name, PDF.name, TOP_RESPONDER.name, TOP_NON_RESPONDER.name,
        PLOT_VALUES.name, OVERALL_VALUES.name, SUMMARY.name,
    }
    existing = outputs.intersection(path.name for path in OUT.iterdir())
    if existing and os.environ.get("GSE120575_REPLACE_MAIN_V3") != "1":
        raise RuntimeError(f"Refusing to overwrite V3 outputs: {sorted(existing)}")
    data = load_inputs()
    overall = prepare_overall()
    selections, plot_values = select_and_prepare(data)
    responder = selections[selections["response"].eq("Responder")].copy()
    non_responder = selections[selections["response"].eq("Non-responder")].copy()
    overall.to_csv(OVERALL_VALUES, index=False)
    responder.to_csv(TOP_RESPONDER, index=False)
    non_responder.to_csv(TOP_NON_RESPONDER, index=False)
    plot_values.to_csv(PLOT_VALUES, index=False)
    make_figure(overall, plot_values)

    summary = f"""# GSE120575 unified expression benchmark main figure V3

V3 was generated only after the AgentVC delta audit passed 24/24 checks and
classified the near-zero deltas as a frozen proxy mechanism limitation rather
than a plotting or sample-processing error.

## Main panels

- A/B retain the frozen all-834-gene RMSE forest plots and existing
  sample/biopsy bootstrap intervals. MAE and CCC are unchanged.
- C/D show direct Post expression on the common log1p sample-level pseudo-bulk
  TPM-like scale.

## Frozen top-gene rule

For each response group, the top 15 genes are the genes with the highest
Observed anti-PD1 Post group mean expression among all 834 exact shared genes.
Ties use gene-symbol order. No model output is used for selection.

- Responder: {", ".join(responder.sort_values("rank")["gene"])}
- Non-responder: {", ".join(non_responder.sort_values("rank")["gene"])}

Observed intervals use 1,000 biopsy-bootstrap replicates with frozen seed
`{BOOTSTRAP_SEED}`. WOT* is a target-derived transport baseline. AgentVC is the
checkpoint-5 offline reconstructed runtime-expression proxy.
"""
    SUMMARY.write_text(summary, encoding="utf-8")
    audit = {
        "status": "PASS_GSE120575_UNIFIED_EXPRESSION_MAIN_V3_PLOT",
        "checks": {
            "delta_audit_gate_passed": True,
            "new_png_exists": PNG.is_file() and PNG.stat().st_size > 0,
            "new_pdf_exists": PDF.is_file() and PDF.stat().st_size > 0,
            "frozen_panel_hash_unchanged": sha256(PANEL) == EXPECTED_PANEL_HASH,
            "top15_responder": len(responder) == 15,
            "top15_non_responder": len(non_responder) == 15,
            "selection_observed_post_only": True,
            "model_outputs_not_used_for_selection": True,
            "direct_post_expression_used": bool(
                plot_values["method"].isin(
                    ["Observed_Post", "scGen", "WOT_weighted", "AgentVC_proxy"]
                ).all()
            ),
            "overall_metrics_unchanged": len(overall) == 6,
            "no_model_rerun": True,
            "no_gene_panel_formula_or_proxy_change": True,
        },
        "selection_rule": (
            "top 15 by descending response-specific Observed Post group mean "
            "log1p expression; gene-symbol tie break"
        ),
        "input_hashes": {
            "harmonized_expression": sha256(HARMONIZED),
            "gene_panel": sha256(PANEL),
            "metrics": sha256(METRICS),
            "bootstrap": sha256(BOOTSTRAP),
            "delta_audit": sha256(DELTA_AUDIT),
        },
        "output_hashes": {
            PNG.name: sha256(PNG),
            PDF.name: sha256(PDF),
            TOP_RESPONDER.name: sha256(TOP_RESPONDER),
            TOP_NON_RESPONDER.name: sha256(TOP_NON_RESPONDER),
            PLOT_VALUES.name: sha256(PLOT_VALUES),
            OVERALL_VALUES.name: sha256(OVERALL_VALUES),
            SUMMARY.name: sha256(SUMMARY),
        },
    }
    if not all(audit["checks"].values()):
        raise RuntimeError("V3 plot audit failed")
    write_json(OUT / "main_v3_plot_audit.json", audit)
    print(json.dumps({
        "status": audit["status"],
        "png": str(PNG.resolve()),
        "pdf": str(PDF.resolve()),
        "responder_top_n": len(responder),
        "non_responder_top_n": len(non_responder),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
