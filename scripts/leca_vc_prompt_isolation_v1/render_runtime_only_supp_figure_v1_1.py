#!/usr/bin/env python3
"""Clean evidence-chain Supplement figure for the runtime prompt-isolation audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from render_contamination_control_figure import DATASET_COLORS, setup_style, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/LecaVC_prompt_contamination_audit_v2_exact_runtime"
OUT = ROOT / "outputs/LecaVC_prompt_contamination_audit_v2_runtime_only_supp_v1_1"
DATASETS = ("GSE2565", "GSE120575", "GSE267904")


def rounded_box(ax, xy, width, height, face, edge, title, lines, title_color="#202124", dashed=False):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.25, edgecolor=edge, facecolor=face,
        linestyle="--" if dashed else "-", transform=ax.transAxes,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + 0.035 * width, y + height - 0.040, title,
            transform=ax.transAxes, ha="left", va="top", fontsize=10.2,
            weight="bold", color=title_color)
    ax.text(x + 0.035 * width, y + height - 0.100, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=8.7,
            color="#404348", linespacing=1.45)
    return box


def arrow(ax, start, end, color="#7A7F87"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.3, color=color, transform=ax.transAxes))


def load_data():
    complete = json.loads((SOURCE / "COMPLETE.json").read_text())
    if complete.get("status") != "PASS_EXACT_RUNTIME_PROMPT_CONTAMINATION_PROBE_V2":
        raise RuntimeError("Completed exact-runtime V2 probe required")
    calls = pd.read_csv(SOURCE / "metrics/call_level_results.csv")
    calls = calls[calls["condition"] == "deidentified_exact_runtime"].copy()
    production = pd.read_csv(SOURCE / "metrics/production_prompt_firewall_audit.csv")
    if len(calls) != 30:
        raise RuntimeError(f"Expected 30 de-identified calls, found {len(calls)}")
    if int(calls["dataset_identified_correctly"].sum()) != 0:
        raise RuntimeError("GEO-identification count changed")
    if int(calls["heldout_answer_correct"].sum()) != 1:
        raise RuntimeError("Held-out-answer count changed")
    if int(production["prohibited_metadata_hits"].sum()) or int(production["fallbacks"].sum()):
        raise RuntimeError("Production prompt audit is not clean")
    return calls, production


def panel_a(ax):
    ax.set_axis_off()
    ax.set_title("Isolation design", fontsize=12, weight="bold", pad=10)
    rounded_box(
        ax, (0.03, 0.66), 0.94, 0.24,
        "#F7F8FA", "#B7BCC5", "Trusted simulation state",
        ["Numerical cell state + local environment", "Dataset metadata retained by orchestration only"],
    )
    arrow(ax, (0.50, 0.66), (0.50, 0.57))
    rounded_box(
        ax, (0.27, 0.44), 0.46, 0.12,
        "#E8F2FB", "#2878B5", "Prompt firewall", ["Allowlist + prohibited-term audit"],
        title_color="#1F5E91",
    )
    arrow(ax, (0.38, 0.44), (0.23, 0.34), color="#C75C5C")
    arrow(ax, (0.62, 0.44), (0.77, 0.34), color="#2F8C72")
    rounded_box(
        ax, (0.02, 0.09), 0.43, 0.24,
        "#FFF1F0", "#D77A72", "Excluded",
        ["GEO/GSM and sample IDs", "Disease, treatment, response labels", "Future times and held-out outcomes"],
        title_color="#A44640",
    )
    rounded_box(
        ax, (0.55, 0.09), 0.43, 0.24,
        "#EEF8F4", "#52A58B", "Visible to the model",
        ["Cell/state identity", "Current numerical state", "Local environment + prior memory"],
        title_color="#237A62",
    )
    ax.text(-0.02, 1.03, "A", transform=ax.transAxes, fontsize=15, weight="bold", va="top")


def panel_b(ax, calls):
    ax.set_axis_off()
    ax.set_title("Exact-runtime probe results", fontsize=12, weight="bold", pad=10)
    rows = list(DATASETS) + ["Pooled"]
    numerators = []
    denominators = []
    for dataset in DATASETS:
        current = calls[calls["dataset"] == dataset]
        numerators.append([
            int(current["dataset_identified_correctly"].sum()),
            int(current["heldout_answer_correct"].sum()),
        ])
        denominators.append(len(current))
    numerators.append([
        int(calls["dataset_identified_correctly"].sum()),
        int(calls["heldout_answer_correct"].sum()),
    ])
    denominators.append(len(calls))

    x0, y0 = 0.30, 0.20
    cell_w, cell_h = 0.32, 0.145
    headers = ("GEO accession\nidentified", "Held-out outcome\ncorrect")
    for col, header in enumerate(headers):
        ax.text(x0 + (col + 0.5) * cell_w, y0 + 4.37 * cell_h, header,
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, weight="bold")
    for row_idx, (label, values, n) in enumerate(zip(rows, numerators, denominators)):
        y = y0 + (3 - row_idx) * cell_h
        label_color = DATASET_COLORS.get(label, "#202124")
        ax.text(x0 - 0.035, y + cell_h / 2, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=9.8,
                weight="bold" if label == "Pooled" else "normal", color=label_color)
        for col, value in enumerate(values):
            rate = value / n
            if rate == 0:
                face, edge = "#EAF6EF", "#77B98E"
            elif rate <= 0.10:
                face, edge = "#FFF4DE", "#D9A441"
            else:
                face, edge = "#FCE8E6", "#D77A72"
            ax.add_patch(Rectangle((x0 + col * cell_w, y), cell_w, cell_h,
                                   transform=ax.transAxes, facecolor=face,
                                   edgecolor=edge, linewidth=1.0))
            ax.text(x0 + (col + 0.5) * cell_w, y + cell_h / 2, f"{value}/{n}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    weight="bold", color="#27313A")
        if label == "Pooled":
            ax.plot([x0 - 0.01, x0 + 2 * cell_w + 0.01], [y + cell_h, y + cell_h],
                    transform=ax.transAxes, color="#555B63", linewidth=1.4)
    ax.text(x0 + cell_w, 0.095, "Frozen reference answers were withheld until offline scoring",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.6, color="#5A5F66")
    ax.text(-0.02, 1.03, "B", transform=ax.transAxes, fontsize=15, weight="bold", va="top")


def kpi(ax, y, number, label, color):
    box = FancyBboxPatch((0.07, y), 0.86, 0.19,
                         boxstyle="round,pad=0.02,rounding_size=0.03",
                         transform=ax.transAxes, facecolor="#F8F9FA",
                         edgecolor="#D5D9DE", linewidth=1.0)
    ax.add_patch(box)
    ax.text(0.17, y + 0.095, number, transform=ax.transAxes, ha="center", va="center",
            fontsize=19, weight="bold", color=color)
    ax.text(0.44, y + 0.095, label, transform=ax.transAxes, ha="left", va="center",
            fontsize=9.2, color="#34383D", linespacing=1.25)


def panel_c(ax, production):
    ax.set_axis_off()
    ax.set_title("Production audit", fontsize=12, weight="bold", pad=10)
    total = int(production["model_facing_messages"].sum())
    kpi(ax, 0.66, "30", "de-identified\nexact-runtime probes", "#2878B5")
    kpi(ax, 0.40, f"{total:,}", "production decisions\naudited", "#6A55A3")
    kpi(ax, 0.14, "0", "prohibited hits\nand fallbacks", "#237A62")
    ax.text(-0.02, 1.03, "C", transform=ax.transAxes, fontsize=15, weight="bold", va="top")


def main() -> int:
    calls, production = load_data()
    setup_style()
    mpl.rcParams["svg.hashsalt"] = "lecavc-runtime-prompt-isolation-supp-v1-1"
    fig = plt.figure(figsize=(13.6, 5.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.30, 0.82],
                           left=0.045, right=0.98, bottom=0.12, top=0.80, wspace=0.26)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    fig.suptitle("Runtime Prompt-Isolation Audit", fontsize=16, weight="bold", y=0.94)
    panel_a(axes[0])
    panel_b(axes[1], calls)
    panel_c(axes[2], production)

    figure_dir = OUT / "figure"
    source_dir = OUT / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    base = figure_dir / "LecaVC_runtime_prompt_isolation_audit"
    fig.savefig(base.with_name(base.name + "_preview.png"), dpi=180, facecolor="white")
    fig.savefig(base.with_name(base.name + "_600dpi.png"), dpi=600, facecolor="white")
    fig.savefig(base.with_name(base.name + "_vector.pdf"), facecolor="white",
                metadata={"Creator": "Leca-VC audit", "CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".svg"), facecolor="white",
                metadata={"Creator": "Leca-VC audit", "Date": None})
    plt.close(fig)

    calls.to_csv(source_dir / "deidentified_exact_runtime_call_results.csv", index=False)
    production.to_csv(source_dir / "production_prompt_firewall_audit.csv", index=False)
    caption = (
        "Runtime prompt-isolation audit. (A) Dataset identifiers, sample identifiers, disease and treatment "
        "context, response labels, future times, and held-out outcomes were retained only in the trusted "
        "orchestration layer and excluded from model-facing messages; the model received numerical cell-state, "
        "local-environment, and recurrent-memory information. (B) Ten normalized-step-zero production contexts "
        "per dataset were presented in their exact de-identified runtime format to a separate query. Frozen "
        "reference answers were not supplied to the model and were scored only after response generation. The "
        "model identified no GEO accession (0/30) and correctly answered 1/30 pre-specified later-outcome "
        "questions: the GSE2565 DNB peak window, the GSE120575 dominant observed Post broad-cell population, or "
        "the GSE267904 dominant late aggregate CCI pathway. (C) Across 1,253 production model-facing decisions, "
        "the prompt firewall detected no prohibited metadata or endpoint cues and no rule-based fallback. This "
        "audit supports operational prompt isolation but cannot establish that the studies were absent from the "
        "model's pretraining corpus."
    )
    (figure_dir / "caption.txt").write_text(caption + "\n")
    audit = {
        "status": "PASS_RUNTIME_PROMPT_ISOLATION_SUPP_FIGURE_V1_1",
        "source_probe_status": "PASS_EXACT_RUNTIME_PROMPT_CONTAMINATION_PROBE_V2",
        "visualization_only": True,
        "new_model_calls": 0,
        "deidentified_probe_calls": len(calls),
        "geo_identified": int(calls["dataset_identified_correctly"].sum()),
        "heldout_correct": int(calls["heldout_answer_correct"].sum()),
        "production_decisions": int(production["model_facing_messages"].sum()),
        "prohibited_hits": int(production["prohibited_metadata_hits"].sum()),
        "fallbacks": int(production["fallbacks"].sum()),
    }
    write_json(OUT / "analysis_audit.json", audit)
    write_json(OUT / "COMPLETE.json", {"status": audit["status"], "figure_generated": True})
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
