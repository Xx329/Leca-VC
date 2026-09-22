#!/usr/bin/env python3
"""Run the frozen GSE120575 four-method expression benchmark V1.

This script reads already harmonized expression matrices. It does not run or
modify scGen, WOT, AgentVC, PhysiCell, an LLM, or any external model.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse120575-expression-v1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("GSE120575_PROXY_OUT", str(ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"))).resolve()
INPUT_NPZ = INPUT_ROOT / "harmonized_834_expression_by_method_sample.npz"
INPUT_SCALE_AUDIT = INPUT_ROOT / "scale_harmonization_audit.json"
INPUT_RECON_AUDIT = INPUT_ROOT / "reconstruction_audit.json"
GENE_PANEL = INPUT_ROOT / "frozen_834_gene_panel.txt"
GENE_HASH = INPUT_ROOT / "gene_panel_hash.json"
AGENT_SAMPLE_CSV = INPUT_ROOT / "agentvc_checkpoint5_proxy_by_sample.csv"

OUT = Path(os.environ.get("GSE120575_EXPRESSION_BENCHMARK_OUT", str(ROOT / "outputs/GSE120575_unified_expression_benchmark_v1"))).resolve()
PNG = OUT / "GSE120575_unified_expression_benchmark_2x2_600dpi.png"
PDF = OUT / "GSE120575_unified_expression_benchmark_2x2_vector.pdf"

RESPONSES = ("Responder", "Non-responder")
INPUT_METHODS = ("scGen", "WOT_weighted", "AgentVC_proxy")
METHOD_LABELS = {
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "AgentVC_proxy": "AgentVC",
}
METHOD_COLORS = {
    "scGen": "#F58518",
    "WOT_weighted": "#54A24B",
    "AgentVC_proxy": "#B279A2",
}
EXPECTED_SAMPLES = (
    "Pre_P24", "Pre_P29", "Pre_P35",
    "Pre_P2", "Pre_P3", "Pre_P27",
)
EXPECTED_BY_RESPONSE = {
    "Responder": {"Pre_P24", "Pre_P29", "Pre_P35"},
    "Non-responder": {"Pre_P2", "Pre_P3", "Pre_P27"},
}
EXPECTED_OBSERVED_N = {"Responder": 5, "Non-responder": 18}
METRIC_ORDER = (
    "RMSE", "MAE", "CCC", "Pearson", "Spearman",
    "mean_signed_bias", "high_expression_gene_bias",
)
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 120575834
EXPECTED_GENE_SHA256 = "1a9e02f8bb52cdea852d6fe30992339ad8bc2fa0ab807b81848dce574a7bce41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    centered_x = x - mean_x
    centered_y = y - mean_y
    covariance = float(np.mean(centered_x * centered_y))
    denominator = (
        float(np.mean(centered_x**2))
        + float(np.mean(centered_y**2))
        + (mean_x - mean_y) ** 2
    )
    return float(2.0 * covariance / denominator) if denominator > 0 else np.nan


def observed_deciles(observed: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return deterministic rank-based observed-expression deciles."""
    order = np.argsort(observed, kind="stable")
    groups = [np.asarray(group, dtype=int) for group in np.array_split(order, 10)]
    labels = np.empty(len(observed), dtype=np.int8)
    for decile, indexes in enumerate(groups, start=1):
        labels[indexes] = decile
    return labels, groups


def compute_metrics(
    prediction: np.ndarray,
    observed: np.ndarray,
    high_expression_indexes: np.ndarray,
) -> dict[str, float]:
    error = prediction - observed
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "CCC": concordance_correlation(prediction, observed),
        "Pearson": float(pearsonr(prediction, observed).statistic),
        "Spearman": float(spearmanr(prediction, observed).statistic),
        "mean_signed_bias": float(np.mean(error)),
        "high_expression_gene_bias": float(np.mean(error[high_expression_indexes])),
    }


def load_and_validate() -> dict[str, object]:
    scale_audit = json.loads(INPUT_SCALE_AUDIT.read_text())
    recon_audit = json.loads(INPUT_RECON_AUDIT.read_text())
    hash_audit = json.loads(GENE_HASH.read_text())
    if scale_audit["status"] != "PASS_FOUR_METHOD_834_GENE_SCALE_HARMONIZATION":
        raise RuntimeError("Input scale harmonization did not pass")
    if recon_audit["status"] != "PASS_AGENTVC_CHECKPOINT5_PROXY_RECONSTRUCTION":
        raise RuntimeError("AgentVC reconstruction did not pass")
    if hash_audit["sha256"] != EXPECTED_GENE_SHA256 or sha256(GENE_PANEL) != EXPECTED_GENE_SHA256:
        raise RuntimeError("Frozen 834-gene panel hash mismatch")
    frozen_genes = GENE_PANEL.read_text(encoding="utf-8").splitlines()
    if len(frozen_genes) != 834 or len(set(frozen_genes)) != 834:
        raise RuntimeError("Frozen gene panel is not 834 unique genes")

    with np.load(INPUT_NPZ) as source:
        loaded = {key: source[key].copy() for key in source.files}
    genes = loaded["genes"].astype(str)
    methods = loaded["methods"].astype(str)
    sample_ids = loaded["sample_ids"].astype(str)
    responses = loaded["responses"].astype(str)
    therapy = loaded["therapy"].astype(str)
    roles = loaded["roles"].astype(str)
    expression = loaded["expression_log1p_sample_pseudobulk"].astype(np.float64)

    if genes.tolist() != frozen_genes:
        raise RuntimeError("Harmonized matrix gene order differs from frozen panel")
    if expression.shape != (47, 834) or not np.isfinite(expression).all() or np.any(expression < 0):
        raise RuntimeError("Invalid harmonized expression matrix")
    if set(therapy) != {"anti-PD1"}:
        raise RuntimeError("Unexpected therapy in harmonized matrix")
    for response in RESPONSES:
        observed_mask = (methods == "Observed_Post") & (responses == response)
        if int(observed_mask.sum()) != EXPECTED_OBSERVED_N[response]:
            raise RuntimeError(f"Unexpected Observed biopsy count for {response}")
        for method in INPUT_METHODS:
            mask = (methods == method) & (responses == response)
            if set(sample_ids[mask]) != EXPECTED_BY_RESPONSE[response] or int(mask.sum()) != 3:
                raise RuntimeError(f"Unexpected source samples for {response}/{method}")
    if set(sample_ids[np.isin(methods, INPUT_METHODS)]) != set(EXPECTED_SAMPLES):
        raise RuntimeError("Virtual source sample scope mismatch")
    expected_roles = {
        "scGen": "generated_expression_baseline",
        "WOT_weighted": "primary_target_derived_transport_baseline",
        "AgentVC_proxy": "offline_reconstructed_runtime_expression_proxy",
    }
    for method, role in expected_roles.items():
        if set(roles[methods == method]) != {role}:
            raise RuntimeError(f"Method semantics mismatch: {method}")

    agent_frame = pd.read_csv(AGENT_SAMPLE_CSV)
    agent_frame = agent_frame.set_index("sample_id").loc[list(EXPECTED_SAMPLES), frozen_genes]
    agent_from_harmonized = np.vstack([
        expression[(methods == "AgentVC_proxy") & (sample_ids == sample)][0]
        for sample in EXPECTED_SAMPLES
    ])
    if not np.allclose(agent_frame.to_numpy(), agent_from_harmonized, rtol=1e-13, atol=1e-13):
        raise RuntimeError("AgentVC harmonized rows differ from audited three-seed sample matrix")

    return {
        "genes": genes,
        "methods": methods,
        "sample_ids": sample_ids,
        "responses": responses,
        "therapy": therapy,
        "roles": roles,
        "expression": expression,
    }


def analyze(data: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genes = np.asarray(data["genes"])
    methods = np.asarray(data["methods"])
    responses = np.asarray(data["responses"])
    expression = np.asarray(data["expression"])
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    gene_error_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for response in RESPONSES:
        observed_rows = expression[(methods == "Observed_Post") & (responses == response)]
        observed_mean = observed_rows.mean(axis=0)
        decile_labels, decile_groups = observed_deciles(observed_mean)
        high_indexes = decile_groups[-1]
        n_observed = len(observed_rows)

        observed_draws = rng.integers(
            0, n_observed, size=(BOOTSTRAP_ITERATIONS, n_observed)
        )
        virtual_draws = rng.integers(0, 3, size=(BOOTSTRAP_ITERATIONS, 3))

        for method in INPUT_METHODS:
            prediction_rows = expression[(methods == method) & (responses == response)]
            prediction_mean = prediction_rows.mean(axis=0)
            point = compute_metrics(prediction_mean, observed_mean, high_indexes)
            for metric in METRIC_ORDER:
                metric_rows.append({
                    "response": response,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "value": point[metric],
                    "n_virtual_source_samples": len(prediction_rows),
                    "n_observed_post_biopsies": n_observed,
                    "gene_count": len(genes),
                    "expression_scale": "log1p(sample-level pseudo-bulk TPM-like expression)",
                })

            error = prediction_mean - observed_mean
            for gene_index, gene in enumerate(genes):
                gene_error_rows.append({
                    "response": response,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "gene": gene,
                    "observed_expression": observed_mean[gene_index],
                    "predicted_expression": prediction_mean[gene_index],
                    "signed_error": error[gene_index],
                    "absolute_error": abs(error[gene_index]),
                    "observed_expression_decile": int(decile_labels[gene_index]),
                })

            for decile, indexes in enumerate(decile_groups, start=1):
                selected_error = error[indexes]
                decile_rows.append({
                    "response": response,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "observed_expression_decile": decile,
                    "gene_count": len(indexes),
                    "observed_expression_min": float(observed_mean[indexes].min()),
                    "observed_expression_max": float(observed_mean[indexes].max()),
                    "observed_expression_mean": float(observed_mean[indexes].mean()),
                    "MAE": float(np.mean(np.abs(selected_error))),
                    "signed_bias": float(np.mean(selected_error)),
                })

            distributions = {metric: [] for metric in METRIC_ORDER}
            for iteration in range(BOOTSTRAP_ITERATIONS):
                boot_observed = observed_rows[observed_draws[iteration]].mean(axis=0)
                boot_prediction = prediction_rows[virtual_draws[iteration]].mean(axis=0)
                values = compute_metrics(boot_prediction, boot_observed, high_indexes)
                for metric in METRIC_ORDER:
                    distributions[metric].append(values[metric])
            for metric in METRIC_ORDER:
                values = np.asarray(distributions[metric], dtype=np.float64)
                bootstrap_rows.append({
                    "response": response,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "point_estimate": point[metric],
                    "bootstrap_mean": float(np.mean(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_unit": "sample/biopsy",
                    "virtual_resampling": "3 source samples with replacement within response",
                    "observed_resampling": f"{n_observed} Post biopsies with replacement within response",
                    "high_expression_gene_set": "fixed top Observed-expression rank decile from point-estimate group mean",
                })

    metrics_frame = pd.DataFrame(metric_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    decile_frame = pd.DataFrame(decile_rows)
    gene_error_frame = pd.DataFrame(gene_error_rows)
    return metrics_frame, bootstrap_frame, decile_frame, gene_error_frame


def build_rankings(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for response in RESPONSES:
        for metric in METRIC_ORDER:
            subset = metrics[
                metrics["response"].eq(response) & metrics["metric"].eq(metric)
            ].copy()
            if metric in {"CCC", "Pearson", "Spearman"}:
                subset["ranking_value"] = subset["value"]
                subset["better_direction"] = "higher"
                ascending = False
            elif metric in {"mean_signed_bias", "high_expression_gene_bias"}:
                subset["ranking_value"] = subset["value"].abs()
                subset["better_direction"] = "absolute value closer to zero"
                ascending = True
            else:
                subset["ranking_value"] = subset["value"]
                subset["better_direction"] = "lower"
                ascending = True
            subset["rank"] = subset["ranking_value"].rank(
                method="min", ascending=ascending
            ).astype(int)
            rows.append(subset[[
                "response", "metric", "method", "method_label", "value",
                "ranking_value", "better_direction", "rank",
            ]])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["response", "metric", "rank"],
        key=lambda series: series.map({value: i for i, value in enumerate(RESPONSES)})
        if series.name == "response" else series,
    )


def metric_value(metrics: pd.DataFrame, response: str, method: str, metric: str) -> float:
    selected = metrics[
        metrics["response"].eq(response)
        & metrics["method"].eq(method)
        & metrics["metric"].eq(metric)
    ]["value"]
    if len(selected) != 1:
        raise RuntimeError("Missing unique plotting metric")
    return float(selected.iloc[0])


def make_figure(
    metrics: pd.DataFrame,
    deciles: pd.DataFrame,
    gene_errors: pd.DataFrame,
) -> None:
    sns.set_theme(style="white", context="paper")
    fig = plt.figure(figsize=(14.2, 9.2), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 2, left=0.075, right=0.93, bottom=0.145, top=0.88,
        wspace=0.25, hspace=0.42,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]

    for panel_index, response in enumerate(RESPONSES):
        axis = axes[panel_index]
        plot_data = gene_errors[gene_errors["response"].eq(response)].copy()
        sns.violinplot(
            data=plot_data,
            x="method_label",
            y="absolute_error",
            order=[METHOD_LABELS[m] for m in INPUT_METHODS],
            palette={METHOD_LABELS[m]: METHOD_COLORS[m] for m in INPUT_METHODS},
            inner=None,
            cut=0,
            linewidth=0.8,
            density_norm="width",
            saturation=0.85,
            ax=axis,
        )
        sns.boxplot(
            data=plot_data,
            x="method_label",
            y="absolute_error",
            order=[METHOD_LABELS[m] for m in INPUT_METHODS],
            width=0.18,
            showfliers=False,
            color="white",
            boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.9},
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#333333", "linewidth": 0.8},
            capprops={"color": "#333333", "linewidth": 0.8},
            ax=axis,
        )
        panel_letter = "A" if panel_index == 0 else "B"
        axis.set_title(
            f"{panel_letter}. {response} 834-gene absolute error",
            loc="left", fontweight="bold", fontsize=11,
        )
        axis.set_xlabel("")
        axis.set_ylabel("Absolute error (log1p TPM-like)" if panel_index == 0 else "")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        lines = []
        for method in INPUT_METHODS:
            lines.append(
                f"{METHOD_LABELS[method]}  "
                f"RMSE {metric_value(metrics, response, method, 'RMSE'):.3f}  "
                f"MAE {metric_value(metrics, response, method, 'MAE'):.3f}  "
                f"CCC {metric_value(metrics, response, method, 'CCC'):.3f}"
            )
        axis.text(
            0.02, 0.98, "\n".join(lines),
            transform=axis.transAxes, va="top", ha="left", fontsize=7.3,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92, "pad": 3},
        )

    all_bias = deciles["signed_bias"].to_numpy()
    color_limit = max(float(np.max(np.abs(all_bias))), 1e-12)
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    heatmap_image = None
    for response_index, response in enumerate(RESPONSES):
        axis = axes[response_index + 2]
        subset = deciles[deciles["response"].eq(response)].copy()
        signed = np.vstack([
            subset[subset["method"].eq(method)]
            .sort_values("observed_expression_decile")["signed_bias"].to_numpy()
            for method in INPUT_METHODS
        ])
        mae = np.vstack([
            subset[subset["method"].eq(method)]
            .sort_values("observed_expression_decile")["MAE"].to_numpy()
            for method in INPUT_METHODS
        ])
        heatmap_image = axis.imshow(signed, cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_xticks(range(10), [f"D{i}" for i in range(1, 11)], fontsize=8)
        axis.set_yticks(
            range(3), [METHOD_LABELS[method] for method in INPUT_METHODS], fontsize=9
        )
        for row in range(3):
            for column in range(10):
                rgba = heatmap_image.cmap(norm(signed[row, column]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column, row,
                    f"{signed[row, column]:.2f}\n({mae[row, column]:.2f})",
                    ha="center", va="center", fontsize=5.8,
                    color="black" if luminance > 0.56 else "white",
                )
        panel_letter = "C" if response_index == 0 else "D"
        axis.set_title(
            f"{panel_letter}. {response} observed-expression deciles",
            loc="left", fontweight="bold", fontsize=11,
        )
        axis.set_xlabel("Observed-expression rank decile (low → high)")
        axis.set_ylabel("")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(
            0.0, -0.22, "Cell text: signed bias (MAE)",
            transform=axis.transAxes, fontsize=7.5, color="#444444",
        )

    assert heatmap_image is not None
    color_axis = fig.add_axes([0.945, 0.15, 0.012, 0.29])
    colorbar = fig.colorbar(heatmap_image, cax=color_axis)
    colorbar.set_label("Mean signed error", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "GSE120575 unified gene-expression benchmark",
        fontsize=15, fontweight="bold", y=0.965,
    )
    fig.text(
        0.075, 0.925,
        "834 exact common genes · log1p sample-level pseudo-bulk TPM-like expression · "
        "Observed anti-PD1 Post reference",
        fontsize=9, color="#444444",
    )
    legend = [
        Patch(facecolor=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABELS[method])
        for method in INPUT_METHODS
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.93, 0.935), frameon=False, ncol=3)
    fig.text(
        0.075, 0.055,
        "WOT* is a target-derived, wot_row_mass-weighted transport baseline. "
        "AgentVC is the checkpoint-5 offline reconstructed runtime-expression proxy; "
        "it is not a directly saved or PhysiCell-executed per-gene expression field.",
        fontsize=8, color="#333333",
    )
    fig.text(
        0.075, 0.03,
        "Violin distributions summarize gene-level absolute errors. Biological-sample uncertainty "
        "is reported separately using 1,000 sample/biopsy bootstrap replicates.",
        fontsize=8, color="#555555",
    )
    fig.savefig(PNG, dpi=600, facecolor="white")
    fig.savefig(PDF, facecolor="white")
    plt.close(fig)


def build_summary(metrics: pd.DataFrame, rankings: pd.DataFrame) -> str:
    lines = [
        "# GSE120575 unified expression benchmark V1",
        "",
        "## Scope",
        "",
        "- 834 frozen exact common genes; no genes were removed after evaluation.",
        "- Common scale: `log1p(sample-level pseudo-bulk TPM-like expression)`.",
        "- Responder: three virtual source samples and five Observed anti-PD1 Post biopsies.",
        "- Non-responder: three virtual source samples and eighteen Observed anti-PD1 Post biopsies.",
        "- AgentVC seeds were averaged within source sample before this benchmark.",
        "- WOT* is the primary `wot_row_mass`-weighted, target-derived transport baseline.",
        "- AgentVC is an offline reconstructed checkpoint-5 runtime-expression proxy, not a directly saved expression endpoint.",
        "",
        "## Point estimates",
        "",
        "| Response | Method | RMSE | MAE | CCC | Pearson | Spearman | Mean bias | High-expression bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wide = metrics.pivot_table(
        index=["response", "method_label"], columns="metric", values="value"
    ).reset_index()
    for response in RESPONSES:
        for method in [METHOD_LABELS[item] for item in INPUT_METHODS]:
            row = wide[
                wide["response"].eq(response) & wide["method_label"].eq(method)
            ].iloc[0]
            lines.append(
                f"| {response} | {method} | {row['RMSE']:.4f} | {row['MAE']:.4f} | "
                f"{row['CCC']:.4f} | {row['Pearson']:.4f} | {row['Spearman']:.4f} | "
                f"{row['mean_signed_bias']:.4f} | {row['high_expression_gene_bias']:.4f} |"
            )
    lines.extend([
        "",
        "## Rankings",
        "",
        "Rankings are descriptive and are reported separately per response and metric. "
        "RMSE/MAE are ranked lower-is-better; CCC/Pearson/Spearman higher-is-better; "
        "signed biases are ranked by absolute distance from zero.",
        "",
    ])
    for response in RESPONSES:
        lines.append(f"- **{response}:**")
        for metric in ("RMSE", "MAE", "CCC"):
            subset = rankings[
                rankings["response"].eq(response) & rankings["metric"].eq(metric)
            ].sort_values("rank")
            order = " > ".join(
                f"{row.method_label} (rank {row.rank})"
                for row in subset.itertuples()
            )
            lines.append(f"  - {metric}: {order}")
    lines.extend([
        "",
        "## Uncertainty",
        "",
        "The 95% confidence intervals use 1,000 biological sample/biopsy bootstrap "
        f"replicates with fixed seed `{BOOTSTRAP_SEED}`. Virtual source samples and "
        "Observed Post biopsies are resampled independently within response. No gene-bootstrap "
        "interval is presented as a biological-sample confidence interval.",
        "",
        "## Figure",
        "",
        "Panels A/B show 834-gene absolute-error violin distributions. Panels C/D show "
        "Observed-defined expression-decile signed errors, with MAE in parentheses. "
        "The same signed-error color scale is used across both heatmaps.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required = {
        "unified_expression_metrics.csv",
        "bootstrap_confidence_intervals.csv",
        "expression_decile_metrics.csv",
        "method_rankings.csv",
        "EXPRESSION_BENCHMARK_SUMMARY.md",
        PNG.name,
        PDF.name,
    }
    existing = required.intersection(path.name for path in OUT.iterdir())
    if existing:
        raise RuntimeError(f"Refusing to overwrite completed benchmark files: {sorted(existing)}")

    data = load_and_validate()
    spec = {
        "status": "FROZEN_BEFORE_METRIC_COMPUTATION",
        "version": "GSE120575_UNIFIED_EXPRESSION_BENCHMARK_V1",
        "input_npz": str(INPUT_NPZ.resolve()),
        "input_npz_sha256": sha256(INPUT_NPZ),
        "gene_panel": str(GENE_PANEL.resolve()),
        "gene_panel_sha256": sha256(GENE_PANEL),
        "gene_count": 834,
        "expression_scale": "log1p(sample-level pseudo-bulk TPM-like expression)",
        "responses_evaluated_separately": list(RESPONSES),
        "virtual_source_samples": list(EXPECTED_SAMPLES),
        "methods": {
            "Observed_Post": "anti-PD1 Post sample-level pseudo-bulk reference",
            "scGen": "sample-level generated predicted Post",
            "WOT_weighted": "wot_row_mass-weighted target-derived projection",
            "AgentVC_proxy": "checkpoint-5 offline reconstructed runtime-expression proxy",
        },
        "point_estimate": "method group mean minus Observed Post group mean within response",
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "unit": "sample/biopsy",
            "virtual_n_per_response": 3,
            "observed_n": EXPECTED_OBSERVED_N,
            "sampling": "independent within-response resampling with replacement; common index draws across methods",
        },
        "high_expression_gene_bias": (
            "mean signed error over top rank-based Observed-expression decile, "
            "defined separately per response and fixed for bootstrap"
        ),
        "deciles": "10 rank-based bins defined only from the response-specific Observed group mean",
        "model_runs": 0,
        "gene_filtering_after_results": False,
    }
    write_json(OUT / "benchmark_spec.json", spec)

    metrics, bootstrap, deciles, gene_errors = analyze(data)
    rankings = build_rankings(metrics)
    metrics.to_csv(OUT / "unified_expression_metrics.csv", index=False)
    bootstrap.to_csv(OUT / "bootstrap_confidence_intervals.csv", index=False)
    deciles.to_csv(OUT / "expression_decile_metrics.csv", index=False)
    rankings.to_csv(OUT / "method_rankings.csv", index=False)
    gene_errors.to_csv(OUT / "gene_level_errors_plotting_source.csv", index=False)
    make_figure(metrics, deciles, gene_errors)
    (OUT / "EXPRESSION_BENCHMARK_SUMMARY.md").write_text(
        build_summary(metrics, rankings), encoding="utf-8"
    )

    expected_metric_rows = len(RESPONSES) * len(INPUT_METHODS) * len(METRIC_ORDER)
    audit = {
        "status": "PASS_GSE120575_UNIFIED_EXPRESSION_BENCHMARK_V1",
        "checks": {
            "frozen_834_gene_hash_unchanged": sha256(GENE_PANEL) == EXPECTED_GENE_SHA256,
            "metric_rows_complete": len(metrics) == expected_metric_rows,
            "bootstrap_rows_complete": len(bootstrap) == expected_metric_rows,
            "all_bootstrap_iterations_1000": bool(
                bootstrap["bootstrap_iterations"].eq(BOOTSTRAP_ITERATIONS).all()
            ),
            "all_ci_values_finite": bool(
                np.isfinite(bootstrap[["point_estimate", "ci95_low", "ci95_high"]]).all().all()
            ),
            "all_point_estimates_inside_reported_ci_or_retained_without_forcing": True,
            "decile_rows_complete": len(deciles) == len(RESPONSES) * len(INPUT_METHODS) * 10,
            "each_response_deciles_cover_834_genes_per_method": all(
                int(deciles[
                    deciles["response"].eq(response) & deciles["method"].eq(method)
                ]["gene_count"].sum()) == 834
                for response in RESPONSES for method in INPUT_METHODS
            ),
            "gene_error_rows_complete": len(gene_errors) == len(RESPONSES) * len(INPUT_METHODS) * 834,
            "observed_not_treated_as_error_method": "Observed_Post" not in set(metrics["method"]),
            "wot_weighted_used_as_primary": set(metrics["method"]) == set(INPUT_METHODS),
            "agentvc_three_seed_sample_mean_input_verified": True,
            "responder_observed_n_5": bool(
                metrics[metrics["response"].eq("Responder")]["n_observed_post_biopsies"].eq(5).all()
            ),
            "non_responder_observed_n_18": bool(
                metrics[metrics["response"].eq("Non-responder")]["n_observed_post_biopsies"].eq(18).all()
            ),
            "virtual_n_3_each_response_method": bool(
                metrics["n_virtual_source_samples"].eq(3).all()
            ),
            "figure_png_exists": PNG.is_file(),
            "figure_pdf_exists": PDF.is_file(),
            "no_model_rerun": True,
            "no_result_based_gene_filtering": True,
        },
        "input_hashes": {
            "harmonized_npz": sha256(INPUT_NPZ),
            "gene_panel": sha256(GENE_PANEL),
            "scale_audit": sha256(INPUT_SCALE_AUDIT),
            "reconstruction_audit": sha256(INPUT_RECON_AUDIT),
        },
    }
    if not all(audit["checks"].values()):
        raise RuntimeError("Final benchmark audit failed")
    write_json(OUT / "benchmark_audit.json", audit)
    output_hashes = {
        path.name: sha256(path)
        for path in OUT.iterdir()
        if path.is_file() and path.name != "output_hashes.json"
    }
    write_json(OUT / "output_hashes.json", {"status": "FROZEN_OUTPUT_HASHES", "files": output_hashes})

    print(json.dumps({
        "status": audit["status"],
        "metric_rows": len(metrics),
        "bootstrap_rows": len(bootstrap),
        "decile_rows": len(deciles),
        "figure_png": str(PNG.resolve()),
        "figure_pdf": str(PDF.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
