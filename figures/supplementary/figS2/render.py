#!/usr/bin/env python3
"""Render final Supplementary Fig. S1 from its frozen plotting table."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-lecavc-figs1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "source_data/figS2"
OUT = ROOT / "build/figures/figS2"
STEM = "GSE120575_celltype_composition_with_CellRank_proxy"
RESPONSES = ("Responder", "Non-responder")
CELL_TYPES = ("B cell", "Plasma cell", "Monocyte/Macrophage", "Dendritic cell", "T cell", "NK cell")
METHODS = ("Real_Post", "scGen", "WOT_weighted", "CellRank_terminal_proxy", "Our_Agent_PhysiCell_V2.1")
LABELS = {"Real_Post":"Observed", "scGen":"scGen", "WOT_weighted":"WOT*", "CellRank_terminal_proxy":"CellRank\N{DAGGER}", "Our_Agent_PhysiCell_V2.1":"Leca-VC"}
COLORS = {"Real_Post":"#4C78A8", "scGen":"#F58518", "WOT_weighted":"#54A24B", "CellRank_terminal_proxy":"#4C9ED9", "Our_Agent_PhysiCell_V2.1":"#B279A2"}

def main() -> None:
    source = pd.read_csv(SOURCE / "plotting_source.csv")
    rmse_table = pd.read_csv(SOURCE / "rmse_summary.csv")
    rmse = {(r.response, r.method): float(r.RMSE) for r in rmse_table.itertuples()}
    expected = len(RESPONSES) * len(METHODS) * len(CELL_TYPES)
    if len(source) != expected or source.duplicated(["response", "method", "cell_type"]).any():
        raise RuntimeError("Frozen S1 plotting table is incomplete or duplicated")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9.2, "axes.linewidth":0.9, "axes.titleweight":"bold", "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.0), sharey=True)
    x = np.arange(len(CELL_TYPES), dtype=float)
    width = 0.155
    offsets = (np.arange(len(METHODS)) - (len(METHODS)-1)/2) * width
    for ax, response, panel in zip(axes, RESPONSES, ("A", "B")):
        for method, offset in zip(METHODS, offsets):
            selected = source[source.response.eq(response) & source.method.eq(method)].set_index("cell_type").reindex(CELL_TYPES)
            values = selected.proportion.to_numpy(float)
            lows = selected.bootstrap_ci95_low.to_numpy(float)
            highs = selected.bootstrap_ci95_high.to_numpy(float)
            bars = ax.bar(x+offset, values, width=width, color=COLORS[method], edgecolor="white", linewidth=.5,
                          yerr=np.vstack((values-lows, highs-values)), error_kw={"ecolor":"#303030","elinewidth":.75,"capsize":1.9,"capthick":.75}, zorder=3)
            for bar, value, high in zip(bars, values, highs):
                ax.text(bar.get_x()+bar.get_width()/2, min(max(value, high)+.016, 1.055), f"{value:.2f}", ha="center", va="bottom", fontsize=5.7, color="#252525", clip_on=False)
        ax.set_title(f"{panel}. {response} post-treatment", loc="left", fontsize=12.2, y=1.135, pad=0)
        ax.text(.5, 1.063, "   ".join(f"{LABELS[m]} RMSE {rmse[(response,m)]:.3f}" for m in METHODS[1:3]), transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.text(.5, 1.018, "   ".join(f"{LABELS[m]} RMSE {rmse[(response,m)]:.3f}" for m in METHODS[3:]), transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.set_xticks(x, CELL_TYPES); ax.tick_params(axis="x", rotation=22, pad=5)
        for label in ax.get_xticklabels(): label.set_ha("right")
        ax.set_xlim(-.62, len(CELL_TYPES)-.38); ax.set_ylim(0,1.10); ax.set_yticks(np.arange(0,1.01,.2))
        ax.grid(axis="y", color="#D9D9D9", alpha=.55, linewidth=.65, zorder=0); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Broad cell-type proportion")
    fig.legend(handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=LABELS[m]) for m in METHODS], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5,.925))
    fig.suptitle("GSE120575 cell-type composition benchmark", fontsize=14.5, fontweight="bold", y=.987)
    fig.text(.5,.050,"Error bars represent 95% sample/biopsy bootstrap confidence intervals. * WOT is target-derived; \N{DAGGER} CellRank is a Pre-only fate-weighted terminal-lineage proxy.",ha="center",va="bottom",fontsize=8.1,color="#444444")
    fig.subplots_adjust(left=.062,right=.99,top=.775,bottom=.225,wspace=.13)
    fig.savefig(OUT/f"{STEM}_preview.png",dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{STEM}_600dpi.png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{STEM}_vector.pdf",bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{STEM}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)

if __name__ == "__main__": main()
