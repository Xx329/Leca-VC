"""Frozen source-data renderers for final Supplementary Figs. S1 and S4."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CCI_SOURCES=["observed","scgpt","scgen","wot_external","cellrank_proxy","worker"]
CCI_TITLES={"observed":"Observed","scgpt":"scGPT","scgen":"scGen","wot_external":"WOT","cellrank_proxy":"CellRank","worker":"Leca-VC"}
CCI_COLORS={"observed":"#333333","scgpt":"#0072B2","scgen":"#E69F00","wot_external":"#009E73","cellrank_proxy":"#56B4E9","worker":"#CC79A7"}
CELL_TYPES=["alveolar_epithelial_AT1_AT2","activated_Krt8_ADI_epithelial","airway_epithelial","macrophage","recruited_monocyte_macrophage","fibroblast_myofibroblast","endothelial","dendritic","lymphoid"]
CELL_SHORT={"alveolar_epithelial_AT1_AT2":"AT1/AT2","activated_Krt8_ADI_epithelial":"Krt8/ADI","airway_epithelial":"Airway","macrophage":"Macrophage","recruited_monocyte_macrophage":"Mono/Mac","fibroblast_myofibroblast":"Fib/Myofib","endothelial":"Endothelial","dendritic":"Dendritic","lymphoid":"Lymphoid"}
BULK_METHODS=["Real","chronODE-M","RVAgene","BOP-DMD","GPR","AgentVC Online Agent pilot"]
BULK_TITLES={"Real":"Observed","chronODE-M":"chronODE-M","RVAgene":"RVAgene","BOP-DMD":"BOP-DMD","GPR":"GPR","AgentVC Online Agent pilot":"Leca-VC"}

def setup() -> None:
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none","axes.linewidth":.8})

def plot_cci(frame: pd.DataFrame, out: Path) -> None:
    if frame.shape!=(486,8) or set(frame.source)!=set(CCI_SOURCES):
        raise RuntimeError("Unexpected frozen S4 source-data contract")
    matrices={}
    for source in CCI_SOURCES:
        table=frame[frame.source.eq(source)].pivot(index="sender",columns="receiver",values="median")
        matrices[source]=table.reindex(index=CELL_TYPES,columns=CELL_TYPES).to_numpy(float)
    vmax=max(float(x.max()) for x in matrices.values())
    fig,axes=plt.subplots(2,3,figsize=(14.5,9.0),facecolor="white")
    fig.subplots_adjust(left=.065,right=.925,bottom=.10,top=.89,hspace=.44,wspace=.30)
    image=None
    for index,(ax,source) in enumerate(zip(axes.ravel(),CCI_SOURCES)):
        image=ax.imshow(matrices[source],cmap="YlGnBu",vmin=0,vmax=vmax,interpolation="nearest",aspect="equal")
        ax.set_xticks(np.arange(9),[CELL_SHORT[x] for x in CELL_TYPES],rotation=48,ha="right",fontsize=7.4)
        ax.set_yticks(np.arange(9),[CELL_SHORT[x] for x in CELL_TYPES],fontsize=7.4)
        ax.set_xlabel("Receiver cell type",fontsize=8.2); ax.set_ylabel("Sender cell type",fontsize=8.2)
        ax.set_title(CCI_TITLES[source],fontsize=12,fontweight="bold",color=CCI_COLORS[source],pad=7)
        ax.text(-.12,1.08,chr(65+index),transform=ax.transAxes,fontsize=14,fontweight="bold",va="top")
        for spine in ax.spines.values(): spine.set_color(CCI_COLORS[source]); spine.set_linewidth(1.35)
    cax=fig.add_axes([.945,.19,.013,.61]); cbar=fig.colorbar(image,cax=cax)
    cbar.set_label("Median normalized CCI edge weight",fontsize=8.5); cbar.ax.tick_params(labelsize=7.5)
    fig.suptitle("GSE267904 cell\N{EN DASH}cell communication network reconstruction",fontsize=18,fontweight="bold",y=.965)
    stem=out/"GSE267904_CCI_network_heatmaps_Leca_VC"
    fig.savefig(stem.with_name(stem.name+"_preview.png"),dpi=180,facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_600dpi.png"),dpi=600,facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_vector.pdf"),facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_vector.svg"),facecolor="white"); plt.close(fig)

def pretty_module(label: str) -> str:
    return {"oxidative_stress":"Oxidative stress","inflammation":"Inflammation","NFkB_TNF":"NF\N{GREEK SMALL LETTER KAPPA}B/TNF","apoptosis_cell_death":"Apoptosis/cell death","hypoxia":"Hypoxia","alveolar_epithelial_injury_barrier":"Alveolar injury/barrier","vascular_permeability_edema_proxy":"Vascular permeability/edema","immune_recruitment":"Immune recruitment","tissue_repair":"Tissue repair","ECM_remodeling":"ECM remodeling"}.get(label,label.replace("_"," "))

def plot_bulk(frame: pd.DataFrame, out: Path) -> None:
    if frame.shape!=(540,6) or set(frame.method)!=set(BULK_METHODS):
        raise RuntimeError("Unexpected frozen S1 source-data contract")
    modules=frame[frame.method.eq("Real")].module.drop_duplicates().tolist(); times=sorted(frame.time_hours.unique().astype(float).tolist())
    matrices={m:frame[frame.method.eq(m)].pivot(index="module",columns="time_hours",values="temporal_z_score").reindex(index=modules,columns=times).to_numpy(float) for m in BULK_METHODS}
    vmax=max(float(np.abs(x).max()) for x in matrices.values())
    fig,axes=plt.subplots(2,3,figsize=(15.5,8.7),facecolor="white")
    fig.subplots_adjust(left=.11,right=.93,bottom=.10,top=.88,hspace=.34,wspace=.20)
    image=None
    for index,(ax,method) in enumerate(zip(axes.ravel(),BULK_METHODS)):
        image=ax.imshow(matrices[method],aspect="auto",cmap="coolwarm",vmin=-vmax,vmax=vmax,interpolation="nearest")
        ax.set_xticks(range(len(times)),[f"{x:g}" for x in times],fontsize=7.5); ax.set_xlabel("Time (h)",fontsize=8.2)
        ax.set_yticks(range(len(modules)),[pretty_module(x) for x in modules] if index%3==0 else [],fontsize=7.1)
        ax.set_title(BULK_TITLES[method],fontsize=11.5,fontweight="bold",pad=6)
        ax.text(-.12,1.08,chr(65+index),transform=ax.transAxes,fontsize=14,fontweight="bold",va="top")
    cax=fig.add_axes([.947,.18,.014,.63]); cbar=fig.colorbar(image,cax=cax)
    cbar.set_label("Temporal z-score",fontsize=8.5); cbar.ax.tick_params(labelsize=7.5)
    fig.suptitle("GSE2565 functional-program response profiles",fontsize=18,fontweight="bold",y=.96)
    stem=out/"GSE2565_functional_program_heatmaps_Leca_VC"
    fig.savefig(stem.with_name(stem.name+"_preview.png"),dpi=180,facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_600dpi.png"),dpi=600,facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_vector.pdf"),facecolor="white")
    fig.savefig(stem.with_name(stem.name+"_vector.svg"),facecolor="white"); plt.close(fig)
