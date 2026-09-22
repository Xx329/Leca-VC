#!/usr/bin/env python3
"""Render final Supplementary Fig. S2 from frozen composition summaries."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-lecavc-figs2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/"source_data/figS2"
OUT=ROOT/"build/figures/figS2"
STEM="GSE230538_cell_state_composition_multimethod_proxy"
STATES=("proliferative_sensitive","early_drug_response","P2RX7_ion_signal_response","senescent_like","immune_like_resistant","intrinsic_resistant","stress_adaptive","apoptotic_or_dying")
STATE_LABELS=("Proliferative\nsensitive","Early drug\nresponse","P2RX7/ion\nresponse","Senescent-like","Immune-like\nresistant","Intrinsic\nresistant","Stress\nadaptive","Apoptotic/\ndying")
DAYS=(4,33)
METHODS=("Observed","scGen","WOT","CellRank","Leca-VC")
LABELS={"Observed":"Observed","scGen":"scGen\N{DOUBLE DAGGER}","WOT":"WOT*\N{DOUBLE DAGGER}","CellRank":"CellRank\N{DAGGER}\N{DOUBLE DAGGER}","Leca-VC":"Leca-VC"}
COLORS={"Observed":"#4C78A8","scGen":"#F58518","WOT":"#54A24B","CellRank":"#4C9ED9","Leca-VC":"#B279A2"}

def main() -> None:
    summary=pd.read_csv(SOURCE/"plotting_source.csv")
    rmse=pd.read_csv(SOURCE/"bar_mean_rmse.csv")
    expected=len(DAYS)*len(METHODS)*len(STATES)
    if len(summary)!=expected or summary.duplicated(["day","method","state"]).any():
        raise RuntimeError("Frozen S2 plotting table is incomplete or duplicated")
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.2,"axes.linewidth":.9,"axes.titleweight":"bold","pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(17.0,7.7),sharey=True)
    x=np.arange(len(STATES),dtype=float); width=.15
    offsets=(np.arange(len(METHODS))-(len(METHODS)-1)/2)*width
    for ax,day,panel in zip(axes,DAYS,("A","B")):
        for method,offset in zip(METHODS,offsets):
            selected=summary[summary.day.eq(day)&summary.method.eq(method)].set_index("state").loc[list(STATES)]
            means=selected.mean_proportion.to_numpy(float); errors=selected.unit_sd.to_numpy(float)
            bars=ax.bar(x+offset,means,width=width,color=COLORS[method],edgecolor="white",linewidth=.4,yerr=errors,error_kw={"ecolor":"#4A4A4A","elinewidth":.7,"capsize":1.5,"capthick":.7},zorder=3)
            for bar,value,error in zip(bars,means,errors):
                if value>=.015:
                    ax.text(bar.get_x()+bar.get_width()/2,min(value+error+.011,1.03),f"{value:.2f}",ha="center",va="bottom",fontsize=5.2,color="#333333",rotation=90,clip_on=False)
        day_rmse=rmse[rmse.day.eq(day)].set_index("method").bar_mean_RMSE
        ax.set_title(f"{panel}. Day {day} held-out",loc="left",fontsize=12.4,y=1.17,pad=0)
        ax.text(.5,1.095,"   ".join(f"{LABELS[m]} {day_rmse[m]:.3f}" for m in ("scGen","WOT","CellRank")),transform=ax.transAxes,ha="center",va="bottom",fontsize=7.3,color="#444444")
        ax.text(.5,1.045,f"Bar-mean RMSE: {LABELS['Leca-VC']} {day_rmse['Leca-VC']:.3f}",transform=ax.transAxes,ha="center",va="bottom",fontsize=7.3,color="#444444")
        ax.set_xticks(x,STATE_LABELS); ax.tick_params(axis="x",rotation=27,pad=5,labelsize=7.2)
        for label in ax.get_xticklabels(): label.set_ha("right")
        ax.set_xlim(-.58,len(STATES)-.42); ax.set_ylim(0,1.06); ax.set_yticks(np.arange(0,1.01,.2))
        ax.grid(axis="y",color="#D9D9D9",linewidth=.65,alpha=.62,zorder=0); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Cell-state proportion")
    fig.legend(handles=[Patch(facecolor=COLORS[m],edgecolor="none",label=LABELS[m]) for m in METHODS],loc="upper center",bbox_to_anchor=(.5,.915),ncol=5,frameon=False)
    fig.suptitle("GSE230538 cell-state composition benchmark",fontsize=17,fontweight="bold",y=.986)
    fig.text(.5,.052,"Bars show means; error bars are \N{PLUS-MINUS SIGN}1 population SD across four cell lines (Observed and external proxies) or three frozen seeds (simulations).",ha="center",fontsize=8.0,color="#4A4A4A")
    fig.text(.5,.025,"\N{DOUBLE DAGGER} Expression-derived deterministic composition proxy, not a native single-cell prediction. * WOT is target-derived; \N{DAGGER} CellRank is an untreated-only, day-invariant terminal-expression proxy.",ha="center",fontsize=8.0,color="#4A4A4A")
    fig.subplots_adjust(left=.058,right=.992,top=.73,bottom=.245,wspace=.12)
    fig.savefig(OUT/f"{STEM}_preview.png",dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{STEM}_600dpi.png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{STEM}_vector.pdf",bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{STEM}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)

if __name__=="__main__": main()
