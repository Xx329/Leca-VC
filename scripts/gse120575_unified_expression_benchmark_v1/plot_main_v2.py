#!/usr/bin/env python3
"""Create the frozen GSE120575 unified-expression benchmark main figure V2.

This is plotting and deterministic derived-table generation only. It does not
rerun a model or alter the frozen 834-gene panel, expression proxy, metrics, or
bootstrap procedure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse120575-expression-main-v2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE120575_unified_expression_benchmark_v1"
PROXY_ROOT = ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"
HARMONIZED = PROXY_ROOT / "harmonized_834_expression_by_method_sample.npz"
GENE_PANEL = PROXY_ROOT / "frozen_834_gene_panel.txt"
PRE_PROFILES = ROOT / (
    "outputs/GSE120575_full_online_celltype/preprocessed/"
    "pretreatment_celltype_expression_profiles.npz"
)
METRICS = OUT / "unified_expression_metrics.csv"
BOOTSTRAP = OUT / "bootstrap_confidence_intervals.csv"

PNG = OUT / "GSE120575_unified_expression_benchmark_main_v2_600dpi.png"
PDF = OUT / "GSE120575_unified_expression_benchmark_main_v2_vector.pdf"
TOP_RESPONDER = OUT / "top_response_genes_responder.csv"
TOP_NON_RESPONDER = OUT / "top_response_genes_non_responder.csv"
TOP_VALUES = OUT / "top_gene_plot_values.csv"
OVERALL_VALUES = OUT / "overall_expression_metrics_for_plot.csv"
SUMMARY = OUT / "PLOT_DESIGN_SUMMARY.md"

RESPONSES = ("Responder", "Non-responder")
METHODS = ("scGen", "WOT_weighted", "AgentVC_proxy")
METHOD_LABELS = {
    "scGen": "scGen",
    "WOT_weighted": "WOT*",
    "AgentVC_proxy": "AgentVC",
}
METHOD_COLORS = {
    "Observed_delta": "#5B5B5B",
    "scGen": "#F58518",
    "WOT_weighted": "#54A24B",
    "AgentVC_proxy": "#B279A2",
}
METHOD_MARKERS = {
    "Observed_delta": "o",
    "scGen": "s",
    "WOT_weighted": "^",
    "AgentVC_proxy": "D",
}
SOURCE_SAMPLES = (
    "Pre_P24", "Pre_P29", "Pre_P35",
    "Pre_P2", "Pre_P3", "Pre_P27",
)
SAMPLES_BY_RESPONSE = {
    "Responder": ("Pre_P24", "Pre_P29", "Pre_P35"),
    "Non-responder": ("Pre_P2", "Pre_P3", "Pre_P27"),
}
TOP_N = 15
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


def load_inputs() -> dict[str, object]:
    genes = GENE_PANEL.read_text(encoding="utf-8").splitlines()
    if sha256(GENE_PANEL) != EXPECTED_GENE_SHA256 or len(genes) != 834:
        raise RuntimeError("Frozen 834-gene panel changed")
    with np.load(HARMONIZED) as source:
        harmonized = {key: source[key].copy() for key in source.files}
    if harmonized["genes"].astype(str).tolist() != genes:
        raise RuntimeError("Harmonized gene order differs from frozen panel")

    methods = harmonized["methods"].astype(str)
    sample_ids = harmonized["sample_ids"].astype(str)
    responses = harmonized["responses"].astype(str)
    values = harmonized["expression_log1p_sample_pseudobulk"].astype(np.float64)
    if values.shape != (47, 834) or not np.isfinite(values).all():
        raise RuntimeError("Invalid harmonized expression matrix")

    with np.load(PRE_PROFILES) as source:
        profile = {key: source[key].copy() for key in source.files}
    panel_genes = profile["genes"].astype(str)
    panel_index = {gene: index for index, gene in enumerate(panel_genes)}
    if not set(genes).issubset(panel_index):
        raise RuntimeError("Pre-only profiles do not cover the frozen panel")
    selected = np.asarray([panel_index[gene] for gene in genes], dtype=int)
    profile_values = profile["profiles"].astype(np.float64)[:, selected]
    profile_samples = profile["sample_ids"].astype(str)
    profile_counts = profile["n_cells"].astype(np.int64)

    pre_by_sample: dict[str, np.ndarray] = {}
    pre_counts: dict[str, int] = {}
    for sample in SOURCE_SAMPLES:
        mask = profile_samples == sample
        if int(mask.sum()) != 6:
            raise RuntimeError(f"Expected six frozen cell-type profiles for {sample}")
        counts = profile_counts[mask]
        if int(counts.sum()) <= 0:
            raise RuntimeError(f"No Pre cells for {sample}")
        pre_tpm = np.average(profile_values[mask], axis=0, weights=counts)
        pre_by_sample[sample] = np.log1p(pre_tpm)
        pre_counts[sample] = int(counts.sum())

    expected_observed = {"Responder": 5, "Non-responder": 18}
    for response in RESPONSES:
        mask = (methods == "Observed_Post") & (responses == response)
        if int(mask.sum()) != expected_observed[response]:
            raise RuntimeError(f"Observed Post count mismatch for {response}")
        for method in METHODS:
            method_mask = (methods == method) & (responses == response)
            if set(sample_ids[method_mask]) != set(SAMPLES_BY_RESPONSE[response]):
                raise RuntimeError(f"Virtual source sample mismatch for {response}/{method}")

    return {
        "genes": np.asarray(genes),
        "methods": methods,
        "sample_ids": sample_ids,
        "responses": responses,
        "values": values,
        "pre_by_sample": pre_by_sample,
        "pre_counts": pre_counts,
    }


def prepare_overall_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS)
    bootstrap = pd.read_csv(BOOTSTRAP)
    rows: list[dict[str, object]] = []
    for response in RESPONSES:
        for method in METHODS:
            def point(metric: str) -> float:
                selected = metrics[
                    metrics["response"].eq(response)
                    & metrics["method"].eq(method)
                    & metrics["metric"].eq(metric)
                ]["value"]
                if len(selected) != 1:
                    raise RuntimeError(f"Missing frozen metric: {response}/{method}/{metric}")
                return float(selected.iloc[0])

            rmse_ci = bootstrap[
                bootstrap["response"].eq(response)
                & bootstrap["method"].eq(method)
                & bootstrap["metric"].eq("RMSE")
            ]
            if len(rmse_ci) != 1 or int(rmse_ci.iloc[0]["bootstrap_iterations"]) != 1000:
                raise RuntimeError(f"Missing frozen RMSE bootstrap: {response}/{method}")
            rows.append({
                "response": response,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "RMSE": point("RMSE"),
                "RMSE_ci95_low": float(rmse_ci.iloc[0]["ci95_low"]),
                "RMSE_ci95_high": float(rmse_ci.iloc[0]["ci95_high"]),
                "MAE": point("MAE"),
                "CCC": point("CCC"),
                "bootstrap_unit": "sample/biopsy",
                "bootstrap_iterations": 1000,
            })
    return pd.DataFrame(rows)


def prepare_top_genes(
    data: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    genes = np.asarray(data["genes"])
    methods = np.asarray(data["methods"])
    sample_ids = np.asarray(data["sample_ids"])
    responses = np.asarray(data["responses"])
    values = np.asarray(data["values"])
    pre_by_sample = data["pre_by_sample"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selection_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    wrong_counts: dict[str, int] = {}

    for response in RESPONSES:
        response_samples = SAMPLES_BY_RESPONSE[response]
        pre_rows = np.vstack([pre_by_sample[sample] for sample in response_samples])
        post_rows = values[(methods == "Observed_Post") & (responses == response)]
        observed_delta = post_rows.mean(axis=0) - pre_rows.mean(axis=0)
        # Stable tie-breaking: descending absolute Observed delta, then gene symbol.
        selected_indexes = np.lexsort((genes, -np.abs(observed_delta)))[:TOP_N]

        post_draws = rng.integers(
            0, len(post_rows), size=(BOOTSTRAP_ITERATIONS, len(post_rows))
        )
        pre_draws = rng.integers(
            0, len(pre_rows), size=(BOOTSTRAP_ITERATIONS, len(pre_rows))
        )
        boot_delta = (
            post_rows[post_draws].mean(axis=1)
            - pre_rows[pre_draws].mean(axis=1)
        )
        ci_low = np.quantile(boot_delta, 0.025, axis=0)
        ci_high = np.quantile(boot_delta, 0.975, axis=0)

        method_deltas: dict[str, np.ndarray] = {}
        for method in METHODS:
            predicted_rows = np.vstack([
                values[(methods == method) & (sample_ids == sample)][0]
                for sample in response_samples
            ])
            method_deltas[method] = (predicted_rows - pre_rows).mean(axis=0)
            wrong_counts[f"{response}|{method}"] = int(np.sum(
                method_deltas[method][selected_indexes] * observed_delta[selected_indexes] < 0
            ))

        for rank, gene_index in enumerate(selected_indexes, start=1):
            gene = str(genes[gene_index])
            selection_rows.append({
                "response": response,
                "rank": rank,
                "gene": gene,
                "observed_pre_group_mean_log1p": float(pre_rows[:, gene_index].mean()),
                "observed_post_group_mean_log1p": float(post_rows[:, gene_index].mean()),
                "observed_group_mean_delta": float(observed_delta[gene_index]),
                "absolute_observed_group_mean_delta": float(abs(observed_delta[gene_index])),
                "observed_delta_ci95_low": float(ci_low[gene_index]),
                "observed_delta_ci95_high": float(ci_high[gene_index]),
                "pre_source_sample_n": len(pre_rows),
                "post_biopsy_n": len(post_rows),
                "selection_rule": (
                    "top 15 by descending absolute Observed group-level mean "
                    "Post-minus-Pre log1p expression change; gene-symbol tie break"
                ),
                "patient_paired": False,
            })
            plot_rows.append({
                "response": response,
                "rank": rank,
                "gene": gene,
                "method": "Observed_delta",
                "method_label": "Observed Δ",
                "delta_expression": float(observed_delta[gene_index]),
                "ci95_low": float(ci_low[gene_index]),
                "ci95_high": float(ci_high[gene_index]),
                "has_interval": True,
                "wrong_direction": False,
            })
            for method in METHODS:
                predicted_delta = float(method_deltas[method][gene_index])
                plot_rows.append({
                    "response": response,
                    "rank": rank,
                    "gene": gene,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "delta_expression": predicted_delta,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "has_interval": False,
                    "wrong_direction": bool(predicted_delta * observed_delta[gene_index] < 0),
                })

    return pd.DataFrame(selection_rows), pd.DataFrame(plot_rows), wrong_counts


def make_figure(
    overall: pd.DataFrame,
    top_values: pd.DataFrame,
    wrong_counts: dict[str, int],
) -> None:
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
    forest_text_x = rmse_max * 1.08
    forest_xlim = rmse_max * 1.50
    y_positions = {"scGen": 2, "WOT_weighted": 1, "AgentVC_proxy": 0}
    for response_index, response in enumerate(RESPONSES):
        axis = axes[response_index]
        subset = overall[overall["response"].eq(response)]
        for row in subset.itertuples():
            y = y_positions[row.method]
            axis.hlines(
                y, row.RMSE_ci95_low, row.RMSE_ci95_high,
                color=METHOD_COLORS[row.method], linewidth=2.2, zorder=2,
            )
            axis.vlines(
                [row.RMSE_ci95_low, row.RMSE_ci95_high],
                y - 0.09, y + 0.09,
                color=METHOD_COLORS[row.method], linewidth=1.2,
            )
            axis.scatter(
                row.RMSE, y, s=58, marker=METHOD_MARKERS[row.method],
                facecolor=METHOD_COLORS[row.method], edgecolor="white",
                linewidth=0.7, zorder=3,
            )
            axis.text(
                forest_text_x, y, f"MAE {row.MAE:.3f}   CCC {row.CCC:.3f}",
                va="center", ha="left", fontsize=8.2, color="#333333",
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

    all_delta_limits = np.concatenate([
        top_values["delta_expression"].to_numpy(dtype=float),
        top_values.loc[top_values["has_interval"], "ci95_low"].to_numpy(dtype=float),
        top_values.loc[top_values["has_interval"], "ci95_high"].to_numpy(dtype=float),
    ])
    delta_limit = max(float(np.max(np.abs(all_delta_limits))) * 1.13, 0.1)
    offsets = {
        "Observed_delta": -0.27,
        "scGen": -0.09,
        "WOT_weighted": 0.09,
        "AgentVC_proxy": 0.27,
    }
    for response_index, response in enumerate(RESPONSES):
        axis = axes[response_index + 2]
        subset = top_values[top_values["response"].eq(response)].copy()
        genes = (
            subset[subset["method"].eq("Observed_delta")]
            .sort_values("rank")["gene"].tolist()
        )
        base_x = np.arange(TOP_N)
        observed = subset[subset["method"].eq("Observed_delta")].sort_values("rank")
        observed_x = base_x + offsets["Observed_delta"]
        axis.errorbar(
            observed_x,
            observed["delta_expression"],
            yerr=np.vstack([
                observed["delta_expression"].to_numpy() - observed["ci95_low"].to_numpy(),
                observed["ci95_high"].to_numpy() - observed["delta_expression"].to_numpy(),
            ]),
            fmt="o", markersize=4.8, color=METHOD_COLORS["Observed_delta"],
            ecolor="#8A8A8A", elinewidth=1.0, capsize=2.2, zorder=2,
        )
        for method in METHODS:
            method_values = subset[subset["method"].eq(method)].sort_values("rank")
            axis.scatter(
                base_x + offsets[method],
                method_values["delta_expression"],
                s=31, marker=METHOD_MARKERS[method],
                facecolor=METHOD_COLORS[method], edgecolor="white",
                linewidth=0.5, zorder=3,
            )
        axis.axhline(0, color="#6E6E6E", linewidth=0.9, linestyle="--", zorder=1)
        axis.set_xlim(-0.7, TOP_N - 0.3)
        axis.set_ylim(-delta_limit, delta_limit)
        axis.set_xticks(base_x, genes, rotation=58, ha="right", fontsize=7.4)
        axis.set_ylabel("Δexpression: Post − Pre\n(log1p TPM-like)")
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        panel = "C" if response_index == 0 else "D"
        axis.set_title(
            f"{panel}. {response}: top observed response genes",
            loc="left", fontweight="bold",
        )
        wrong_text = "\n".join(
            f"{METHOD_LABELS[method]}: {wrong_counts[f'{response}|{method}']} / {TOP_N}"
            for method in METHODS
        )
        axis.text(
            0.985, 0.975, "Wrong direction\n" + wrong_text,
            transform=axis.transAxes, ha="right", va="top", fontsize=7.3,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92, "pad": 3},
        )

    legend_handles = [
        Line2D(
            [0], [0], marker=METHOD_MARKERS["Observed_delta"], linestyle="none",
            color=METHOD_COLORS["Observed_delta"], label="Observed Δ (95% bootstrap CI)",
            markersize=6,
        )
    ] + [
        Line2D(
            [0], [0], marker=METHOD_MARKERS[method], linestyle="none",
            markerfacecolor=METHOD_COLORS[method], markeredgecolor="white",
            color=METHOD_COLORS[method], label=METHOD_LABELS[method], markersize=6,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=legend_handles, loc="upper right", bbox_to_anchor=(0.965, 0.91),
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
        0.07, 0.085,
        "Panels A/B summarize all 834 genes. Panels C/D show the frozen top 15 genes selected "
        "only by descending |Observed group-level mean Post − Pre change|; no patient-level pairing "
        "was available. Gray intervals are 1,000-replicate sample/biopsy bootstrap CIs.",
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
    required_new = {
        PNG.name, PDF.name, TOP_RESPONDER.name, TOP_NON_RESPONDER.name,
        TOP_VALUES.name, OVERALL_VALUES.name, SUMMARY.name,
    }
    existing = required_new.intersection(path.name for path in OUT.iterdir())
    if existing and os.environ.get("GSE120575_REPLACE_MAIN_V2") != "1":
        raise RuntimeError(f"Refusing to overwrite existing V2 plot outputs: {sorted(existing)}")

    data = load_inputs()
    overall = prepare_overall_metrics()
    selections, top_values, wrong_counts = prepare_top_genes(data)
    responder = selections[selections["response"].eq("Responder")].copy()
    non_responder = selections[selections["response"].eq("Non-responder")].copy()
    if len(responder) != TOP_N or len(non_responder) != TOP_N:
        raise RuntimeError("Top-gene selection is incomplete")

    overall.to_csv(OVERALL_VALUES, index=False)
    responder.to_csv(TOP_RESPONDER, index=False)
    non_responder.to_csv(TOP_NON_RESPONDER, index=False)
    top_values.to_csv(TOP_VALUES, index=False)
    make_figure(overall, top_values, wrong_counts)

    summary = f"""# GSE120575 unified expression benchmark main figure V2

## Design

- Panels A/B: frozen RMSE point estimates and 95% sample/biopsy bootstrap
  intervals from the completed 834-gene benchmark. MAE and CCC are shown as
  supporting values.
- Panels C/D: Observed and predicted group-level expression changes for a
  frozen top-15 gene set selected separately by response.

## Top-gene selection

For each response group, genes were ranked by descending absolute Observed
group-level mean expression change on the frozen log1p pseudo-bulk scale:

`mean(Observed anti-PD1 Post biopsies) - mean(the three source Pre samples)`.

Ties use gene-symbol order. Model predictions were not used for selection.
No gene was manually selected or removed. Patient-level pairing was not
available, so this is explicitly an unpaired group-level mean change.

- Responder top 15: {", ".join(responder.sort_values("rank")["gene"])}
- Non-responder top 15: {", ".join(non_responder.sort_values("rank")["gene"])}

For scGen, WOT* and AgentVC, each source sample's predicted Post expression was
reduced by that same source sample's frozen Pre expression, followed by a mean
over the three source samples.

## Uncertainty and semantics

Observed delta intervals use 1,000 sample/biopsy bootstrap replicates, resampling
the three Pre source samples and the 5 Responder or 18 Non-responder Post
biopsies independently within response (seed `{BOOTSTRAP_SEED}`).

WOT* is a target-derived transport baseline. AgentVC is the checkpoint-5
offline reconstructed runtime-expression proxy and is not a directly saved or
PhysiCell-executed per-gene expression field.

## Wrong-direction counts

- Responder: scGen {wrong_counts['Responder|scGen']}/{TOP_N}; WOT*
  {wrong_counts['Responder|WOT_weighted']}/{TOP_N}; AgentVC
  {wrong_counts['Responder|AgentVC_proxy']}/{TOP_N}.
- Non-responder: scGen {wrong_counts['Non-responder|scGen']}/{TOP_N}; WOT*
  {wrong_counts['Non-responder|WOT_weighted']}/{TOP_N}; AgentVC
  {wrong_counts['Non-responder|AgentVC_proxy']}/{TOP_N}.

## Frozen inputs

- Harmonized expression: `{HARMONIZED}`
- Pre-only sample×cell-type profiles: `{PRE_PROFILES}`
- Frozen metrics: `{METRICS}`
- Frozen bootstrap intervals: `{BOOTSTRAP}`
- Gene panel SHA256: `{sha256(GENE_PANEL)}`
"""
    SUMMARY.write_text(summary, encoding="utf-8")

    audit = {
        "status": "PASS_GSE120575_UNIFIED_EXPRESSION_MAIN_V2_PLOT",
        "checks": {
            "new_png_exists": PNG.is_file() and PNG.stat().st_size > 0,
            "new_pdf_exists": PDF.is_file() and PDF.stat().st_size > 0,
            "frozen_gene_panel_hash_unchanged": sha256(GENE_PANEL) == EXPECTED_GENE_SHA256,
            "top_15_responder": len(responder) == 15,
            "top_15_non_responder": len(non_responder) == 15,
            "selection_uses_observed_delta_only": True,
            "no_patient_pairing_claim": True,
            "group_level_delta_used": True,
            "all_four_plot_entities_present": set(top_values["method"]) == {
                "Observed_delta", "scGen", "WOT_weighted", "AgentVC_proxy"
            },
            "overall_values_read_from_frozen_metrics": len(overall) == 6,
            "bootstrap_rule_unchanged": bool(
                overall["bootstrap_iterations"].eq(1000).all()
                and overall["bootstrap_unit"].eq("sample/biopsy").all()
            ),
            "no_model_rerun": True,
            "no_gene_filtering_by_model_result": True,
        },
        "top_gene_selection": (
            "response-specific top 15 by descending absolute Observed unpaired "
            "group-level mean Post-minus-Pre change; gene-symbol tie break"
        ),
        "input_hashes": {
            "harmonized_expression": sha256(HARMONIZED),
            "gene_panel": sha256(GENE_PANEL),
            "pre_profiles": sha256(PRE_PROFILES),
            "metrics": sha256(METRICS),
            "bootstrap": sha256(BOOTSTRAP),
        },
        "output_hashes": {
            PNG.name: sha256(PNG),
            PDF.name: sha256(PDF),
            TOP_RESPONDER.name: sha256(TOP_RESPONDER),
            TOP_NON_RESPONDER.name: sha256(TOP_NON_RESPONDER),
            TOP_VALUES.name: sha256(TOP_VALUES),
            OVERALL_VALUES.name: sha256(OVERALL_VALUES),
            SUMMARY.name: sha256(SUMMARY),
        },
    }
    if not all(audit["checks"].values()):
        raise RuntimeError("V2 plot audit failed")
    write_json(OUT / "main_v2_plot_audit.json", audit)
    print(json.dumps({
        "status": audit["status"],
        "png": str(PNG.resolve()),
        "pdf": str(PDF.resolve()),
        "responder_top_n": len(responder),
        "non_responder_top_n": len(non_responder),
        "wrong_direction_counts": wrong_counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
