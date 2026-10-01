#!/usr/bin/env python3
"""Paper-style visualization of the exact-runtime held-out-outcome probe."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from render_contamination_control_figure import DATASET_COLORS, setup_style, wilson, write_json


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "source_data/figS8"
OUT = ROOT / "build/figures/figS8"
DATASETS = ("GSE2565", "GSE120575", "GSE267904")
DISPLAY_ROWS = (*DATASETS, "Pooled")
CLASSES = ("Correct", "Incorrect", "Unknown")
CLASS_COLORS = {
    "Correct": "#2E8B57",
    "Incorrect": "#D45B5B",
    "Unknown": "#B9BEC6",
}


def load_calls() -> pd.DataFrame:
    calls = pd.read_csv(SOURCE / "deidentified_exact_runtime_call_results.csv")
    if len(calls) != 30:
        raise RuntimeError(f"Expected 30 de-identified calls, found {len(calls)}")
    calls["response_class"] = np.where(
        calls["heldout_answer_correct"],
        "Correct",
        np.where(calls["abstained_unknown"], "Unknown", "Incorrect"),
    )
    expected = {
        "GSE2565": {"Correct": 0, "Incorrect": 0, "Unknown": 10},
        "GSE120575": {"Correct": 1, "Incorrect": 6, "Unknown": 3},
        "GSE267904": {"Correct": 0, "Incorrect": 0, "Unknown": 10},
    }
    observed = pd.crosstab(calls["dataset"], calls["response_class"])
    for dataset, counts in expected.items():
        for response_class, expected_count in counts.items():
            actual = int(observed.get(response_class, pd.Series(dtype=int)).get(dataset, 0))
            if actual != expected_count:
                raise RuntimeError(
                    f"Frozen result changed for {dataset}/{response_class}: {actual} != {expected_count}"
                )
    return calls


def summaries(calls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    composition_rows = []
    accuracy_rows = []
    for label in DISPLAY_ROWS:
        current = calls if label == "Pooled" else calls[calls["dataset"] == label]
        n = len(current)
        for response_class in CLASSES:
            count = int((current["response_class"] == response_class).sum())
            composition_rows.append(
                {
                    "dataset": label,
                    "response_class": response_class,
                    "count": count,
                    "n": n,
                    "proportion": count / n,
                }
            )
        correct = int(current["heldout_answer_correct"].sum())
        low, high = wilson(correct, n)
        accuracy_rows.append(
            {
                "dataset": label,
                "correct": correct,
                "n": n,
                "accuracy": correct / n,
                "wilson_ci_2_5": low,
                "wilson_ci_97_5": high,
            }
        )
    return pd.DataFrame(composition_rows), pd.DataFrame(accuracy_rows)


def plot_composition(ax: plt.Axes, composition: pd.DataFrame) -> None:
    y = np.arange(len(DISPLAY_ROWS))
    left = np.zeros(len(DISPLAY_ROWS), dtype=float)
    for response_class in CLASSES:
        current = composition[composition["response_class"] == response_class].set_index("dataset").reindex(DISPLAY_ROWS)
        values = current["proportion"].to_numpy(float)
        counts = current["count"].to_numpy(int)
        bars = ax.barh(
            y, values, left=left, height=0.58,
            color=CLASS_COLORS[response_class], edgecolor="white", linewidth=0.8,
            label=response_class,
        )
        for bar, count, value in zip(bars, counts, values):
            if count == 0:
                continue
            x = bar.get_x() + bar.get_width() / 2
            text_color = "white" if response_class != "Unknown" else "#25282C"
            ax.text(x, bar.get_y() + bar.get_height() / 2, f"{count}",
                    ha="center", va="center", fontsize=9.5, weight="bold", color=text_color)
        left += values
    ax.set_yticks(y, DISPLAY_ROWS)
    for tick, label in zip(ax.get_yticklabels(), DISPLAY_ROWS):
        if label in DATASET_COLORS:
            tick.set_color(DATASET_COLORS[label])
        if label == "Pooled":
            tick.set_weight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v * 100)}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Responses (%)")
    ax.set_title("Response composition", pad=9)
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.8, zorder=0)
    ax.axhline(2.5, color="#4E535A", linewidth=1.0)
    ax.text(-0.13, 1.04, "A", transform=ax.transAxes, fontsize=14, weight="bold", va="top")


def plot_accuracy(ax: plt.Axes, accuracy: pd.DataFrame) -> None:
    x = np.arange(len(DISPLAY_ROWS), dtype=float)
    for index, label in enumerate(DISPLAY_ROWS):
        row = accuracy[accuracy["dataset"] == label].iloc[0]
        value = float(row["accuracy"])
        low = float(row["wilson_ci_2_5"])
        high = float(row["wilson_ci_97_5"])
        color = DATASET_COLORS.get(label, "#202124")
        marker = "D" if label == "Pooled" else "o"
        size = 8.5 if label == "Pooled" else 8
        ax.errorbar(
            index, value,
            yerr=[[value - low], [high - value]],
            fmt=marker, markersize=size, color=color,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8,
            capsize=4, linewidth=1.5, zorder=3,
        )
        label_y = min(0.49, high + 0.032)
        ax.text(index, label_y, f"{int(row['correct'])}/{int(row['n'])}",
                ha="center", va="bottom", fontsize=9.5, weight="bold", color=color)
    ax.set_xticks(x, DISPLAY_ROWS, rotation=18, ha="right")
    for tick, label in zip(ax.get_xticklabels(), DISPLAY_ROWS):
        if label in DATASET_COLORS:
            tick.set_color(DATASET_COLORS[label])
        if label == "Pooled":
            tick.set_weight("bold")
    ax.set_ylim(-0.02, 0.52)
    ax.set_yticks(np.arange(0, 0.51, 0.1), [f"{int(v * 100)}" for v in np.arange(0, 0.51, 0.1)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Held-out outcome accuracy", pad=9)
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.8, zorder=0)
    ax.text(-0.13, 1.04, "B", transform=ax.transAxes, fontsize=14, weight="bold", va="top")


def main() -> int:
    calls = load_calls()
    composition, accuracy = summaries(calls)

    setup_style()
    mpl.rcParams["svg.hashsalt"] = "lecavc-exact-runtime-contamination-probe-paper-v1"
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), gridspec_kw={"width_ratios": [1.20, 1.0]})
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.19, top=0.72, wspace=0.34)
    fig.suptitle("Exact-Runtime Prompt Contamination Probe", fontsize=15, weight="bold", y=0.95)
    plot_composition(axes[0], composition)
    plot_accuracy(axes[1], accuracy)
    fig.legend(
        handles=[Patch(facecolor=CLASS_COLORS[name], edgecolor="none", label=name) for name in CLASSES],
        loc="upper center", bbox_to_anchor=(0.36, 0.855), ncol=3, frameon=False,
    )

    figure_dir = OUT / "figure"
    source_dir = OUT / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    base = figure_dir / "LecaVC_exact_runtime_prompt_contamination_probe"
    fig.savefig(base.with_name(base.name + "_preview.png"), dpi=180, facecolor="white")
    fig.savefig(base.with_name(base.name + "_600dpi.png"), dpi=600, facecolor="white")
    fig.savefig(base.with_name(base.name + "_vector.pdf"), facecolor="white",
                metadata={"Creator": "Leca-VC audit", "CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".svg"), facecolor="white",
                metadata={"Creator": "Leca-VC audit", "Date": None})
    plt.close(fig)

    calls.to_csv(source_dir / "deidentified_exact_runtime_call_results.csv", index=False)
    composition.to_csv(source_dir / "response_composition.csv", index=False)
    accuracy.to_csv(source_dir / "heldout_outcome_accuracy_with_wilson_ci.csv", index=False)
    (figure_dir / "caption.txt").write_text(
        (Path(__file__).resolve().parent / "caption.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    audit = {
        "status": "PASS_EXACT_RUNTIME_CONTAMINATION_PROBE_PAPER_FIGURE_V1",
        "source_probe_status": "PASS_EXACT_RUNTIME_PROMPT_CONTAMINATION_PROBE_V2",
        "visualization_only": True,
        "new_model_calls": 0,
        "probe_calls": len(calls),
        "correct": int(calls["heldout_answer_correct"].sum()),
        "incorrect": int(((~calls["heldout_answer_correct"]) & (~calls["abstained_unknown"])).sum()),
        "unknown": int(calls["abstained_unknown"].sum()),
    }
    write_json(OUT / "analysis_audit.json", audit)
    write_json(OUT / "COMPLETE.json", {"status": audit["status"], "figure_generated": True})
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
