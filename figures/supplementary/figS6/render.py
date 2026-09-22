#!/usr/bin/env python3
"""Render final Supplementary Fig. S6 from its canonical published values."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-lecavc-figs6")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/"source_data/figS6/representative_agent_step60.csv"
OUT=ROOT/"build/figures/figS6"

def panel(ax,data,title,xlabel,cmap):
    data=data.sort_values("order",ascending=False); y=np.arange(len(data))
    colors=plt.get_cmap(cmap)(np.linspace(.25,.82,len(data)))
    bars=ax.barh(y,data.value,color=colors,edgecolor="#303030",linewidth=.8,height=.66)
    ax.set_yticks(y,data.label,fontsize=10,fontweight="bold"); ax.set_xlim(0,1.0); ax.set_xticks(np.arange(0,1.01,.2))
    ax.set_xlabel(xlabel,fontsize=10,fontweight="bold"); ax.set_title(title,fontsize=12,fontweight="bold",pad=12)
    ax.spines[["top","right"]].set_visible(False); ax.grid(False)
    for bar,value in zip(bars,data.value):
        ax.text(value+.025,bar.get_y()+bar.get_height()/2,f"{value:.2f}",va="center",ha="left",fontsize=10,fontweight="bold",color="#111111" if cmap=="Blues" else "#e31a1c")

def main():
    frame=pd.read_csv(SOURCE)
    if len(frame)!=10 or set(frame.side)!={"input","output"}: raise RuntimeError("Unexpected frozen S6 source-data contract")
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.8))
    panel(axes[0],frame[frame.side.eq("input")],"Input: Microenvironment & Memory\n(Step 60)","Signal Strength (0.0 - 1.0)","Blues")
    panel(axes[1],frame[frame.side.eq("output")],"Output: Generated Biological Program\n(LLM Decision Engine)","Action Policy Score (0.0 - 1.0)","Reds")
    fig.subplots_adjust(left=.18,right=.98,top=.82,bottom=.18,wspace=.55)
    stem=OUT/"representative_myofibroblast_agent_step60"
    fig.savefig(stem.with_name(stem.name+"_preview.png"),dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_600dpi.png"),dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_vector.pdf"),bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_vector.svg"),bbox_inches="tight",facecolor="white"); plt.close(fig)

if __name__=="__main__": main()
