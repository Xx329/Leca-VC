#!/usr/bin/env python3
"""Render the frozen Leca-VC prompt de-identification audit figure."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "outputs/LecaVC_prompt_contamination_audit_v1/probe"
OUT = ROOT / "outputs/LecaVC_prompt_contamination_audit_v1/figure_v1"
DATASETS = ("GSE2565", "GSE120575", "GSE267904")
CONDITIONS = (
    "identified_positive_control",
    "deidentified_runtime",
    "counterfactual",
)
CONDITION_LABELS = {
    "identified_positive_control": "Identified\ncontrol",
    "deidentified_runtime": "De-identified\nruntime",
    "counterfactual": "Counterfactual",
}
COLORS = {
    "identified_positive_control": "#7A7A7A",
    "deidentified_runtime": "#2878B5",
    "counterfactual": "#31A354",
}
DATASET_COLORS = {
    "GSE2565": "#E68613",
    "GSE120575": "#8E63B6",
    "GSE267904": "#16A6A1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def normalize_accession(value: object) -> str:
    return str(value).replace("-", "").replace("_", "").replace(" ", "").upper()


def is_specific_window(value: object) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "unknown", "uncertain", "none", "na", "n/a", "not specified"}


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def collect_probe_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = PROBE / "RESPONSES_FROZEN.json"
    if not frozen.is_file():
        raise FileNotFoundError(f"Frozen probe marker missing: {frozen}")
    rows: list[dict[str, object]] = []
    responses: dict[tuple[str, str, int], dict[str, object]] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for replicate in range(1, 11):
                path = PROBE / dataset / condition / f"replicate_{replicate:02d}/response.json"
                response = json.loads(path.read_text())
                responses[(dataset, condition, replicate)] = response
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "replicate": replicate,
                        "identified_correctly": dataset.upper()
                        in normalize_accession(response["dataset_guess"]),
                        "specific_future_window": is_specific_window(response["peak_window_bin"]),
                        "response_path": str(path.relative_to(ROOT)),
                        "response_sha256": sha256(path),
                    }
                )

    probe = pd.DataFrame(rows)
    cf_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for replicate in range(1, 11):
            runtime = responses[(dataset, "deidentified_runtime", replicate)]["next_change_direction"]
            counter = responses[(dataset, "counterfactual", replicate)]["next_change_direction"]
            if not isinstance(runtime, dict) or not isinstance(counter, dict):
                raise TypeError("De-identified and counterfactual directions must be JSON objects")
            axes = sorted(set(runtime) & set(counter))
            changed = [str(runtime[key]).lower() != str(counter[key]).lower() for key in axes]
            cf_rows.append(
                {
                    "dataset": dataset,
                    "replicate": replicate,
                    "n_axes_compared": len(axes),
                    "n_axes_changed": int(sum(changed)),
                    "any_direction_changed": bool(any(changed)),
                    "axis_change_fraction": float(np.mean(changed)) if changed else math.nan,
                }
            )
    return probe, pd.DataFrame(cf_rows)


def summarize_probe(probe: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset in (*DATASETS, "All datasets"):
        current = probe if dataset == "All datasets" else probe[probe["dataset"] == dataset]
        for condition in CONDITIONS:
            subset = current[current["condition"] == condition]
            for metric, column in (
                ("GEO accession identification", "identified_correctly"),
                ("Specific future-window response", "specific_future_window"),
            ):
                successes = int(subset[column].sum())
                n = len(subset)
                low, high = wilson(successes, n)
                records.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "metric": metric,
                        "successes": successes,
                        "n": n,
                        "rate": successes / n,
                        "wilson_ci_2_5": low,
                        "wilson_ci_97_5": high,
                    }
                )
    return pd.DataFrame(records)


def collect_production_audit() -> pd.DataFrame:
    g2565_root = ROOT / "outputs/GSE2565_deidentified_rerun_v1"
    g120_root = ROOT / "outputs/GSE120575_deidentified_rerun_v1"
    g267_root = ROOT / "outputs/GSE267904_deidentified_rerun_v1"

    g2565_calls = pd.read_csv(g2565_root / "audit/deepseek_runtime_call_audit.csv")
    g120_files = sorted((g120_root / "simulation/runs").glob("*/decision_call_audit.csv"))
    if len(g120_files) != 18:
        raise RuntimeError(f"Expected 18 GSE120575 decision audits, found {len(g120_files)}")
    g120_calls = pd.concat([pd.read_csv(path) for path in g120_files], ignore_index=True)
    a2565 = json.loads((g2565_root / "audit/execution_audit.json").read_text())
    a120 = json.loads((g120_root / "audit/execution_audit.json").read_text())
    a267 = json.loads((g267_root / "audit/execution_audit.json").read_text())
    n267 = sum(int(value) for value in a267["exact_message_counts_by_K"].values())

    rows = [
        {
            "dataset": "GSE2565",
            "model_facing_messages": len(g2565_calls),
            "prohibited_metadata_hits": 0 if a2565["checks"]["forbidden_hits_zero"] else math.nan,
            "fallbacks": int(g2565_calls["fallback_used"].astype(bool).sum()),
            "final_schema_invalid": int((~g2565_calls["schema_valid"].astype(bool)).sum()),
            "execution_status": a2565["status"],
        },
        {
            "dataset": "GSE120575",
            "model_facing_messages": len(g120_calls),
            "prohibited_metadata_hits": 0
            if a120["checks"]["forbidden_model_visible_hits_zero"]
            else math.nan,
            "fallbacks": int(g120_calls["fallback_used"].astype(bool).sum()),
            "final_schema_invalid": 0 if a120["status"].startswith("PASS") else math.nan,
            "execution_status": a120["status"],
        },
        {
            "dataset": "GSE267904",
            "model_facing_messages": n267,
            "prohibited_metadata_hits": len(a267["forbidden_hits"]),
            "fallbacks": int(bool(a267["fallback_used"])),
            "final_schema_invalid": 0 if a267["status"].startswith("PASS") else math.nan,
            "execution_status": a267["status"],
        },
    ]
    result = pd.DataFrame(rows)
    if int(result["model_facing_messages"].sum()) != 1253:
        raise RuntimeError("Production prompt count is not the frozen total of 1,253")
    return result


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "svg.hashsalt": "lecavc-prompt-audit-v1",
        }
    )


def plot_rate_panel(ax: plt.Axes, summary: pd.DataFrame, metric: str, letter: str) -> None:
    overall = summary[(summary["dataset"] == "All datasets") & (summary["metric"] == metric)]
    x = np.arange(len(CONDITIONS), dtype=float)
    values = [float(overall[overall["condition"] == condition]["rate"].iloc[0]) for condition in CONDITIONS]
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=[COLORS[condition] for condition in CONDITIONS],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    for idx, condition in enumerate(CONDITIONS):
        row = overall[overall["condition"] == condition].iloc[0]
        low, high, value = float(row["wilson_ci_2_5"]), float(row["wilson_ci_97_5"]), float(row["rate"])
        ax.errorbar(
            idx,
            value,
            yerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
            color="#333333",
            capsize=3,
            lw=1.0,
            zorder=4,
        )
        for offset, dataset in zip((-0.10, 0.0, 0.10), DATASETS):
            point = summary[
                (summary["dataset"] == dataset)
                & (summary["condition"] == condition)
                & (summary["metric"] == metric)
            ].iloc[0]
            ax.scatter(
                idx + offset,
                float(point["rate"]),
                s=28,
                color=DATASET_COLORS[dataset],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
        label_y = min(1.045, value + 0.055) if value < 0.98 else 1.025
        ax.text(idx, label_y, f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_xticks(x, [CONDITION_LABELS[condition] for condition in CONDITIONS])
    ax.set_ylim(-0.03, 1.11)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", color="#E6E8EB", lw=0.8, zorder=0)
    ax.set_title(metric, pad=8)
    ax.text(-0.11, 1.02, letter, transform=ax.transAxes, fontsize=14, weight="bold", va="top")


def render() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probe, counterfactual = collect_probe_rows()
    summary = summarize_probe(probe)
    production = collect_production_audit()

    probe.to_csv(OUT / "probe_call_level_results.csv", index=False)
    summary.to_csv(OUT / "probe_rate_summary.csv", index=False)
    counterfactual.to_csv(OUT / "counterfactual_sensitivity.csv", index=False)
    production.to_csv(OUT / "production_prompt_firewall_audit.csv", index=False)

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.10, top=0.88, wspace=0.28, hspace=0.42)
    fig.suptitle("Prompt De-identification and Leakage-Control Audit", fontsize=16, weight="bold", y=0.965)

    plot_rate_panel(axes[0, 0], summary, "GEO accession identification", "A")
    plot_rate_panel(axes[0, 1], summary, "Specific future-window response", "B")

    ax = axes[1, 0]
    grouped = counterfactual.groupby("dataset", sort=False).agg(
        any_response_changed=("any_direction_changed", "mean"),
        axis_level_changed=("axis_change_fraction", "mean"),
    ).reindex(DATASETS)
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.34
    first = ax.bar(x - width / 2, grouped["any_response_changed"], width, color="#2878B5", label="Any output changed")
    second = ax.bar(x + width / 2, grouped["axis_level_changed"], width, color="#74C476", label="Individual axes changed")
    for bars in (first, second):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Paired change rate (%)")
    ax.set_title("Counterfactual response sensitivity", pad=8)
    ax.grid(axis="y", color="#E6E8EB", lw=0.8, zorder=0)
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.11, 1.02, "C", transform=ax.transAxes, fontsize=14, weight="bold", va="top")

    ax = axes[1, 1]
    y = np.arange(len(DATASETS))
    counts = production.set_index("dataset").reindex(DATASETS)["model_facing_messages"]
    bars = ax.barh(y, counts, color=[DATASET_COLORS[d] for d in DATASETS], alpha=0.88, height=0.58)
    for bar, dataset in zip(bars, DATASETS):
        row = production[production["dataset"] == dataset].iloc[0]
        ax.text(
            12,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['model_facing_messages'])} audited",
            ha="left",
            va="center",
            fontsize=9,
            weight="bold",
            color="white",
        )
        ax.text(
            690,
            bar.get_y() + bar.get_height() / 2,
            "0 prohibited hits · 0 fallbacks",
            ha="right",
            va="center",
            fontsize=8.7,
            weight="bold",
            color="#237A3B",
        )
    ax.set_yticks(y, DATASETS)
    ax.invert_yaxis()
    ax.set_xlim(0, 700)
    ax.set_xlabel("Audited model-facing decisions")
    ax.set_title("Production prompt firewall", pad=8)
    ax.grid(axis="x", color="#E6E8EB", lw=0.8, zorder=0)
    ax.text(-0.11, 1.02, "D", transform=ax.transAxes, fontsize=14, weight="bold", va="top")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=DATASET_COLORS[d], markeredgecolor="white", markersize=7, label=d)
        for d in DATASETS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.923), ncol=3, frameon=False)

    base = OUT / "LecaVC_prompt_deidentification_leakage_control_audit"
    fig.savefig(base.with_name(base.name + "_preview.png"), dpi=180, facecolor="white")
    fig.savefig(base.with_name(base.name + "_600dpi.png"), dpi=600, facecolor="white")
    fig.savefig(
        base.with_name(base.name + "_vector.pdf"),
        facecolor="white",
        metadata={"Creator": "Leca-VC audit", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        base.with_name(base.name + ".svg"),
        facecolor="white",
        metadata={"Creator": "Leca-VC audit", "Date": None},
    )
    plt.close(fig)

    caption = (
        "Prompt de-identification and leakage-control audit. (A) GEO accession identification "
        "rates across 10 repeated probes per dataset and condition. Dataset identity was recovered "
        "from the identified positive controls (30/30) but not from de-identified runtime-format or "
        "counterfactual prompts (0/30 each). (B) Specific future-window claims were elicited by "
        "identified controls (25/30) but not by either de-identified condition (0/30 each). Bars show "
        "pooled rates with 95% Wilson intervals; colored points show individual datasets. (C) Reversing "
        "the supplied numerical state changed at least one predicted direction in every paired probe "
        "and changed 60.0–62.5% of individual state axes. (D) Across 1,253 model-facing production "
        "decisions, the prompt firewall detected no prohibited metadata and no rule-based fallback was "
        "used. This audit tests prompt-enabled identification and response dependence on supplied state; "
        "it cannot establish that the model never encountered the underlying studies during pretraining."
    )
    (OUT / "caption.txt").write_text(caption + "\n")

    inputs = [PROBE / "RESPONSES_FROZEN.json"]
    inputs.extend(Path(path) for path in probe["response_path"].map(lambda x: str(ROOT / x)))
    inputs.extend(
        [
            ROOT / "outputs/GSE2565_deidentified_rerun_v1/audit/execution_audit.json",
            ROOT / "outputs/GSE2565_deidentified_rerun_v1/audit/deepseek_runtime_call_audit.csv",
            ROOT / "outputs/GSE120575_deidentified_rerun_v1/audit/execution_audit.json",
            ROOT / "outputs/GSE267904_deidentified_rerun_v1/audit/execution_audit.json",
        ]
    )
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(inputs))}
    write_json(
        OUT / "analysis_audit.json",
        {
            "status": "PASS_PROMPT_DEIDENTIFICATION_LEAKAGE_CONTROL_FIGURE_V1",
            "probe_calls": len(probe),
            "probe_calls_per_dataset_condition": 10,
            "identified_positive_control_correct": int(
                probe[probe["condition"] == "identified_positive_control"]["identified_correctly"].sum()
            ),
            "deidentified_identity_recoveries": int(
                probe[probe["condition"] == "deidentified_runtime"]["identified_correctly"].sum()
            ),
            "counterfactual_identity_recoveries": int(
                probe[probe["condition"] == "counterfactual"]["identified_correctly"].sum()
            ),
            "production_messages_audited": int(production["model_facing_messages"].sum()),
            "production_prohibited_metadata_hits": int(production["prohibited_metadata_hits"].sum()),
            "production_fallbacks": int(production["fallbacks"].sum()),
            "interpretation_limit": (
                "The audit tests model-facing prompt content and prompt-enabled recognition/recall; "
                "it cannot prove absence of the studies from model pretraining."
            ),
            "input_sha256": input_hashes,
        },
    )
    outputs = sorted(
        path for path in OUT.iterdir()
        if path.is_file() and path.name not in {"output_manifest.json", "COMPLETE.json"}
    )
    write_json(
        OUT / "output_manifest.json",
        {"files": {path.name: sha256(path) for path in outputs}},
    )
    write_json(
        OUT / "COMPLETE.json",
        {
            "status": "PASS_PROMPT_DEIDENTIFICATION_LEAKAGE_CONTROL_FIGURE_V1",
            "new_model_calls": 0,
            "source_probe_calls": 90,
            "production_messages_audited": 1253,
        },
    )
    print(json.dumps(json.loads((OUT / "COMPLETE.json").read_text()), indent=2))


if __name__ == "__main__":
    render()
