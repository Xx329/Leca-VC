#!/usr/bin/env python3
"""Render final Supplementary Fig. S5 from compact frozen spatial arrays."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-lecavc-figs5")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/"source_data/figS5/spatial_hotspot_arrays.npz"
OUT=ROOT/"build/figures/figS5"
STAGES=[("real","d7_bleo","early\nreal"),("virtual","virtual_early_bleo","early\nvirtual"),("real","d21_bleo","late\nreal"),("virtual","virtual_late_bleo","late\nvirtual")]
FAMILIES=["TGFB","CCL","CXCL"]; MODES=["sender","receiver"]

def extent(xy,pad=.035):
    lo=np.nanmin(xy,axis=0); hi=np.nanmax(xy,axis=0); span=np.where((hi-lo)<=0,1.,hi-lo)
    lo-=span*pad; hi+=span*pad; return float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1])

def smooth(xy,values,grid_n=170,sigma=7.0):
    ok=np.isfinite(xy).all(axis=1)&np.isfinite(values); xy=xy[ok]; values=values[ok]
    bounds=extent(xy); xmin,xmax,ymin,ymax=bounds
    ix=np.clip((((xy[:,0]-xmin)/max(xmax-xmin,1e-12))*(grid_n-1)).astype(int),0,grid_n-1)
    iy=np.clip((((xy[:,1]-ymin)/max(ymax-ymin,1e-12))*(grid_n-1)).astype(int),0,grid_n-1)
    weighted=np.zeros((grid_n,grid_n)); density=np.zeros((grid_n,grid_n))
    np.add.at(weighted,(iy,ix),values); np.add.at(density,(iy,ix),1.)
    weighted=gaussian_filter(weighted,sigma=sigma,mode="nearest"); density=gaussian_filter(density,sigma=sigma,mode="nearest")
    grid=np.divide(weighted,density,out=np.full_like(weighted,np.nan),where=density>1e-10)
    threshold=max(float(np.nanmax(density))*.015,1e-12); grid=np.where(density>=threshold,grid,np.nan)
    return grid,bounds

def main() -> None:
    data=np.load(SOURCE); OUT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(4,6,figsize=(21,13),dpi=220); rows=[]
    for row,(source,stage,row_label) in enumerate(STAGES):
        xy=np.asarray(data[f"xy_{row}"],float)
        for fi,family in enumerate(FAMILIES):
            for mi,mode in enumerate(MODES):
                col=fi*2+mi; ax=axes[row,col]; scores=np.asarray(data[f"score_{row}_{family}_{mode}"],float)
                cmap="YlOrRd" if mode=="sender" else "YlGnBu"; title=f"{family} {mode}\n{stage}"
                if source=="virtual":
                    grid,bounds=smooth(xy,scores); vmax=max(float(np.nanmax(grid)) if np.isfinite(grid).any() else 0.,1.)
                    image=ax.imshow(grid,origin="lower",extent=bounds,cmap=cmap,vmin=0,vmax=vmax,interpolation="bilinear")
                    ax.scatter(xy[:,0],xy[:,1],s=5,c="#111827",alpha=.16,linewidths=0); nonzero=int(np.count_nonzero(grid[np.isfinite(grid)]>0))
                else:
                    vmax=max(float(np.nanmax(scores)) if scores.size else 0.,1.)
                    image=ax.scatter(xy[:,0],xy[:,1],c=scores,s=2.0,cmap=cmap,vmin=0,vmax=vmax,alpha=.93,linewidths=0); nonzero=int(np.count_nonzero(scores>0))
                ax.set_title(title,fontsize=9.5,weight="bold"); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal",adjustable="box")
                cbar=fig.colorbar(image,ax=ax,fraction=.046,pad=.012); cbar.ax.tick_params(labelsize=7)
                ax.set_ylabel(row_label if col==0 else "",fontsize=12,weight="bold")
                rows.append({"source":source,"stage":stage,"family":family,"mode":mode,"n_observations":len(xy),"n_nonzero_projected_points_or_grid_cells":nonzero,"max_projected_or_smoothed_score":vmax})
    fig.suptitle("GSE267904 K=100 pathway-specific spatial COMMOT hotspots: real spots and smoothed virtual agent fields",fontsize=16,weight="bold",y=.992)
    fig.text(.5,.018,"Real panels use Visium spot coordinates. Virtual panels use K=100 PhysiCell micro-agent coordinates and Gaussian-smoothed weighted grids; faint dots mark available virtual agents.",ha="center",fontsize=9.5,color="#4b5563")
    fig.tight_layout(rect=[.02,.035,.985,.965],h_pad=1.0,w_pad=1.25)
    stem=OUT/"gse267904_k100_pathway_specific_spatial_commot_hotspots_smoothed"
    fig.savefig(stem.with_suffix(".png"),dpi=260); fig.savefig(stem.with_suffix(".pdf")); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/(stem.name+"_manifest.csv"),index=False)

if __name__=="__main__": main()
