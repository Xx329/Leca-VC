#!/usr/bin/env python3
"""Render the runtime-only Supplement figure from completed exact-runtime probes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from render_contamination_control_figure import DATASET_COLORS, setup_style, wilson, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/LecaVC_prompt_contamination_audit_v2_exact_runtime"
OUT = ROOT / "outputs/LecaVC_prompt_contamination_audit_v2_runtime_only_supp"
DATASETS = ("GSE2565", "GSE120575", "GSE267904")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = json.loads((SOURCE / "COMPLETE.json").read_text())
    if complete.get("status") != "PASS_EXACT_RUNTIME_PROMPT_CONTAMINATION_PROBE_V2":
        raise RuntimeError("The exact-runtime V2 probe is not complete")
    calls = pd.read_csv(SOURCE / "metrics/call_level_results.csv")
    calls = calls[calls["condition"] == "deidentified_exact_runtime"].copy()
    production = pd.read_csv(SOURCE / "metrics/production_prompt_firewall_audit.csv")
    if len(calls) != 30 or set(calls["dataset"]) != set(DATASETS):
        raise RuntimeError("Expected 30 completed de-identified exact-runtime probes")
    if int(calls["dataset_identified_correctly"].sum()) != 0:
        raise RuntimeError("Frozen GEO-identification total changed")
    if int(calls["heldout_answer_correct"].sum()) != 1:
        raise RuntimeError("Frozen held-out-answer total changed")
    if int(production["prohibited_metadata_hits"].sum()) != 0 or int(production["fallbacks"].sum()) != 0:
        raise RuntimeError("Production firewall audit is not clean")
    return calls, production


def rate_rows(calls: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        current = calls[calls["dataset"] == dataset]
        successes = int(current[column].sum())
        low, high = wilson(successes, len(current))
        rows.append(
            {
                "dataset": dataset,
                "successes": successes,
                "n": len(current),
                "rate": successes / len(current),
                "wilson_ci_2_5": low,
                "wilson_ci_97_5": high,
            }
        )
    return pd.DataFrame(rows)


def plot_rate(ax: plt.Axes, data: pd.DataFrame, title: str, letter: str, pooled: str) -> None:
    x = np.arange(len(DATASETS), dtype=float)
    for index, dataset in enumerate(DATASETS):
        row = data[data["dataset"] == dataset].iloc[0]
        value = float(row["rate"])
        low = float(row["wilson_ci_2_5"])
        high = float(row["wilson_ci_97_5"])
        ax.errorbar(
            index,
            value,
            yerr=[[value - low], [high - value]],
            fmt="o",
            ms=9,
            color=DATASET_COLORS[dataset],
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=4,
            lw=1.5,
            zorder=3,
        )
        ax.text(index, min(0.47, high + 0.035), f"{int(row['successes'])}/{int(row['n'])}",
                ha="center", va="bottom", fontsize=9, weight="bold", color=DATASET_COLORS[dataset])
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(-0.025, 0.52)
    ax.set_yticks(np.arange(0, 0.51, 0.1), [f"{int(v * 100)}" for v in np.arange(0, 0.51, 0.1)])
    ax.set_ylabel("Rate (%)")
    ax.set_title(title, pad=9)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.text(0.98, 0.94, pooled, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#444444")
    ax.text(-0.12, 1.03, letter, transform=ax.transAxes, fontsize=14, weight="bold", va="top")


def main() -> int:
    calls, production = load_inputs()
    identity = rate_rows(calls, "dataset_identified_correctly")
    outcome = rate_rows(calls, "heldout_answer_correct")

    source_dir = OUT / "source_data"
    figure_dir = OUT / "figure"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    calls.to_csv(source_dir / "deidentified_exact_runtime_call_results.csv", index=False)
    identity.to_csv(source_dir / "geo_identification_rates.csv", index=False)
    outcome.to_csv(source_dir / "heldout_outcome_accuracy.csv", index=False)
    production.to_csv(source_dir / "production_prompt_firewall_audit.csv", index=False)

    setup_style()
    mpl.rcParams["svg.hashsalt"] = "lecavc-runtime-only-prompt-isolation-supp"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.78, wspace=0.34)
    fig.suptitle("Operational Audit of De-identified Runtime Prompts", fontsize=16, weight="bold", y=0.94)

    plot_rate(axes[0], identity, "GEO accession identification", "A", "Pooled: 0/30")
    plot_rate(axes[1], outcome, "Held-out later-outcome accuracy", "B", "Pooled: 1/30 (3.3%)")

    ax = axes[2]
    current = production.set_index("dataset").reindex(DATASETS)
    y = np.arange(len(DATASETS))
    counts = current["model_facing_messages"].astype(int)
    bars = ax.barh(y, counts, height=0.56, color=[DATASET_COLORS[d] for d in DATASETS], alpha=0.9)
    maximum = max(counts)
    for bar, count in zip(bars, counts):
        ax.text(count + maximum * 0.025, bar.get_y() + bar.get_height() / 2,
                f"{count}", ha="left", va="center", fontsize=9, weight="bold")
    ax.set_yticks(y, DATASETS)
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.23)
    ax.set_xlabel("Audited decisions (0 prohibited hits; 0 fallbacks)")
    ax.set_title("Production prompt firewall audit", pad=9)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.text(-0.12, 1.03, "C", transform=ax.transAxes, fontsize=14, weight="bold", va="top")

    base = figure_dir / "LecaVC_deidentified_runtime_prompt_isolation_audit"
    fig.savefig(base.with_name(base.name + "_preview.png"), dpi=180, facecolor="white")
    fig.savefig(base.with_name(base.name + "_600dpi.png"), dpi=600, facecolor="white")
    fig.savefig(base.with_name(base.name + "_vector.pdf"), facecolor="white",
                metadata={"Creator": "Leca-VC audit", "CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".svg"), facecolor="white",
                metadata={"Creator": "Leca-VC audit", "Date": None})
    plt.close(fig)

    caption = (
        "Operational audit of de-identified runtime prompts. Ten normalized-step-zero production contexts "
        "were selected deterministically for each dataset and presented in their exact model-facing format "
        "to a separate DeepSeek audit query. The query asked for the GEO accession and one pre-specified "
        "later-course outcome; frozen reference answers were withheld from the model and scored only after "
        "response generation. (A) The model identified none of the 30 dataset contexts. (B) It correctly "
        "answered 1 of 30 held-out outcome questions (GSE2565 DNB peak window, GSE120575 dominant Post broad "
        "cell population, or GSE267904 dominant late aggregate CCI pathway). Points show rates and 95% Wilson "
        "intervals. (C) Independent inspection of all 1,253 production model-facing decisions found no "
        "prohibited dataset identifiers or endpoint cues and no rule-based fallback. These results support "
        "operational prompt isolation but do not establish that the underlying studies were absent from model pretraining."
    )
    (figure_dir / "caption.txt").write_text(caption + "\n")
    audit = {
        "status": "PASS_RUNTIME_ONLY_PROMPT_ISOLATION_SUPP_FIGURE",
        "source_probe_status": "PASS_EXACT_RUNTIME_PROMPT_CONTAMINATION_PROBE_V2",
        "new_model_calls": 0,
        "deidentified_exact_runtime_calls": 30,
        "geo_identification_correct": int(calls["dataset_identified_correctly"].sum()),
        "heldout_outcomes_correct": int(calls["heldout_answer_correct"].sum()),
        "production_decisions_audited": int(production["model_facing_messages"].sum()),
        "production_forbidden_hits": int(production["prohibited_metadata_hits"].sum()),
        "production_fallbacks": int(production["fallbacks"].sum()),
        "interpretation_limit": (
            "Supports operational prompt isolation; does not prove absence of study exposure during pretraining."
        ),
    }
    write_json(OUT / "analysis_audit.json", audit)
    write_json(OUT / "COMPLETE.json", {"status": audit["status"], "figure_generated": True})
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
