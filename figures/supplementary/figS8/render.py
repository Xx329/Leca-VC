"""Clean-layout Supplementary Figure for frozen GSE267904 Agent traces."""
from pathlib import Path
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gse267904-agent-trace-supp-v2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import base_trace as base


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "source_data/figS8"
OUT = ROOT / "build/figures/figS8"
FIG_OUT = OUT / "figure"


def draw_trace_panel(ax, decision, spec):
    selected = decision.loc[decision.agent_id.eq(spec["agent_id"])].sort_values("interval")
    if len(selected) != 2:
        raise RuntimeError(f"Expected two decisions for {spec['agent_id']}; found {len(selected)}")
    x = np.array([0.0, 1.0])
    plotted = []
    for label, field, color in spec["series"]:
        values = selected[field].to_numpy(float)
        ax.plot(x, values, marker="o", ms=8, lw=2.5, color=color, label=label, zorder=3)
        plotted.append((label, color, values))
    for series_index, (_, color, values) in enumerate(plotted):
        for point_index, (xx, value) in enumerate(zip(x, values)):
            other = plotted[1 - series_index][2][point_index]
            offset = -18 if abs(value - other) < 0.035 and value < other else 9
            ax.annotate(
                f"{value:.2f}",
                (xx, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color=color,
                weight="bold",
            )
    ax.set_xticks(x, ["Decision 1", "Decision 2"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Validated program output")
    ax.grid(axis="y", color="#DFE4EA", lw=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 1.26, spec["clean_title"], transform=ax.transAxes, fontsize=12.5, weight="bold", ha="center", va="center")
    ax.text(-0.12, 1.26, spec["panel"], transform=ax.transAxes, fontsize=15, weight="bold", va="center")
    ax.text(0.0, 1.13, spec["cell_label"], transform=ax.transAxes, fontsize=10.5, weight="bold", va="center")
    p0 = base.program_label(selected.iloc[0].dominant_program)
    p1 = base.program_label(selected.iloc[1].dominant_program)
    box = dict(boxstyle="round,pad=0.32", facecolor="#F4F6F8", edgecolor="#9AA4B2", lw=0.9)
    ax.text(0.02, 1.03, p0, transform=ax.transAxes, fontsize=9.1, ha="left", va="center", bbox=box)
    ax.text(0.98, 1.03, p1, transform=ax.transAxes, fontsize=9.1, ha="right", va="center", bbox=box)
    arrow_start, arrow_end = (0.49, 0.70) if spec["panel"] == "C" else (0.33, 0.67)
    ax.annotate(
        "",
        xy=(arrow_end, 1.03),
        xytext=(arrow_start, 1.03),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#4B5563", lw=1.5),
    )
    ax.legend(frameon=False, fontsize=9.2, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2)


def render(decision, writeback):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(15.8, 9.2))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.965,
        bottom=0.09,
        top=0.79,
        wspace=0.28,
        hspace=0.62,
    )
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    for ax, spec in zip(axes[:3], base.TRACE_SPECS):
        draw_trace_panel(ax, decision, spec)

    ax = axes[3]
    colors = {"Decision 1": "#D55E00", "Decision 2": "#0072B2"}
    for label in ("Decision 1", "Decision 2"):
        subset = writeback.loc[writeback.decision.eq(label)]
        ax.scatter(
            subset.agent_output,
            subset.physicell_writeback,
            s=25,
            alpha=0.38,
            color=colors[label],
            edgecolor="none",
            label=label,
            zorder=2,
        )
    ax.plot([0, 0.5], [0, 0.5], ls="--", lw=1.7, color="#343A40", label="Identity line", zorder=3)
    ax.set_xlim(-0.015, 0.515)
    ax.set_ylim(-0.015, 0.515)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Validated Agent output")
    ax.set_ylabel("PhysiCell-applied value")
    ax.text(0.5, 1.15, "Agent-to-PhysiCell writeback", transform=ax.transAxes, fontsize=12.5, weight="bold", ha="center", va="center")
    ax.text(-0.24, 1.15, "D", transform=ax.transAxes, fontsize=15, weight="bold", va="center")
    ax.grid(color="#DFE4EA", lw=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.04,
        0.94,
        f"{len(writeback)}/{len(writeback)} exact\nmax |difference| = 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#AAB2BD", alpha=0.94),
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(
        "Traceable Cellular Program Changes and PhysiCell Execution in GSE267904",
        fontsize=18,
        weight="bold",
        y=0.965,
    )
    stem = FIG_OUT / "GSE267904_traceable_cellular_program_changes_clean"
    fig.savefig(str(stem) + "_preview.png", dpi=150)
    fig.savefig(str(stem) + "_600dpi.png", dpi=600)
    fig.savefig(str(stem) + ".pdf", metadata={"CreationDate": None})
    fig.savefig(str(stem) + ".svg", metadata={"Date": None})
    plt.close(fig)
    return stem


def main():
    long_traces = pd.read_csv(SOURCE / "selected_agent_traces.csv")
    decision = long_traces.pivot_table(
        index=["agent_id", "cell_type", "decision", "interval", "dominant_program"],
        columns="agent_output_field", values="agent_output", aggfunc="first",
    ).reset_index()
    decision.columns.name = None
    writeback = pd.read_csv(SOURCE / "all_K100_communication_writebacks.csv")
    if len(writeback) != 665 or float(writeback.absolute_difference.max()) != 0.0:
        raise RuntimeError("Frozen exact-writeback contract changed")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    clean_titles = {
        "A": "Fibroblast matrix-remodeling activation",
        "B": "Macrophage inflammatory-to-resolution transition",
        "C": "Epithelial homeostatic-to-repair transition",
    }
    for spec in base.TRACE_SPECS:
        spec["clean_title"] = clean_titles[spec["panel"]]
    render(decision, writeback)
    print(OUT)


if __name__ == "__main__":
    main()
