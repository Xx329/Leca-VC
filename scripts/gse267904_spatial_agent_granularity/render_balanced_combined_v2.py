"""Equal-width scientific figure using frozen metrics and original map pixels."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-balanced-gse267904')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('GSE267904_HETEROGENEITY_DATA', str(ROOT / 'outputs/GSE267904_agent_granularity_heterogeneity_v1_5_paper_title/data'))).resolve()
SOURCE = Path(os.environ.get('GSE267904_MODULE_MAP_SOURCE', str(ROOT / 'outputs/GSE267904_spatial_agent_granularity/figures/module_expression_K20_K50_K100_old_style_horizontal/GSE267904_module_expression_K20_K50_K100_old_style_panel_normalized.png'))).resolve()
OUT = Path(os.environ.get('GSE267904_COMBINED_FIGURE_OUT', str(ROOT / 'outputs/GSE267904_agent_granularity_spatial_combined_v2_balanced'))).resolve()

def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    module = pd.read_csv(DATA / 'module_heterogeneity_metrics.csv')
    cci = pd.read_csv(DATA / 'cci_heterogeneity_metrics_primary.csv')
    pixels = np.asarray(Image.open(SOURCE).convert('RGB'))
    chromatic = np.ptp(pixels.astype(np.int16), axis=2) > 30
    columns = [r for r in runs(chromatic.sum(axis=0) > 400) if r[1]-r[0] > 300]
    rows = [r for r in runs(chromatic.sum(axis=1) > 400) if r[1]-r[0] > 300]
    assert len(columns) == 4 and len(rows) == 6, (columns, rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig = plt.figure(figsize=(22, 11))
    # Both blocks occupy exactly 46% of the canvas width and the same height.
    left = fig.add_gridspec(2, 3, left=.035, right=.495, bottom=.09, top=.79,
                           wspace=.38, hspace=.48)
    right = fig.add_gridspec(6, 4, left=.600, right=.985, bottom=.09, top=.79,
                            wspace=.06, hspace=.18)
    specs = [
        (module,'energy_distance_squared','Module-state distribution distance','Squared energy distance'),
        (module,'observed_state_mass_coverage','Module-state coverage','Observed state mass covered'),
        (module,'state_js_similarity','Module-state proportional fidelity','1 − Jensen–Shannon divergence'),
        (cci,'observed_mass_on_predicted_support','CCI-state coverage','Observed CCI-state mass covered'),
        (cci,'core_80_mass_recovery','Core CCI-state recovery','Observed core mass recovered'),
        (cci,'effective_diversity_ratio','CCI-state diversity recovery','Effective diversity / observed')]
    for i,(table,key,title,ylabel) in enumerate(specs):
        ax = fig.add_subplot(left[i//3,i%3])
        for stage,color,marker,style in [('Early','#D55E00','o','-'),('Late','#0072B2','s','--')]:
            subset=table[table.stage.eq(stage)].sort_values('K')
            ax.plot(subset.K,subset[key],color=color,marker=marker,ls=style,lw=2,ms=6,
                    label='d7 (early)' if stage=='Early' else 'd21 (late)')
            for k,v in zip(subset.K,subset[key]):
                dy=11 if stage=='Early' else -17
                if i==0:
                    dy=12 if stage=='Early' else -17
                    if k==50: dy=-20 if stage=='Early' else 14
                if v==0: dy=8
                ax.annotate(f'{v:.2f}' if i==0 else f'{v:.3f}',(k,v),
                            xytext=(0,dy),textcoords='offset points',ha='center',
                            fontsize=9,color=color,weight='bold')
        ax.set_title(title,fontsize=10.5,weight='bold',pad=14)
        ax.text(-.20,1.06,'ABCDEF'[i],transform=ax.transAxes,fontsize=13,weight='bold')
        ax.set_xticks([20,50,100]); ax.set_xlim(13,107)
        ax.set_xlabel('Nominal Agent granularity (K)'); ax.set_ylabel(ylabel)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#e1e5eb'); ax.set_axisbelow(True)
        if i==0: ax.set_yscale('log'); ax.set_ylim(.8,11)
        else: ax.set_ylim(-.045,1.065)
        if i==5: ax.axhline(1,color='#7c8390',ls=':',lw=1)
        if i==0: handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.265,.875),ncol=2,frameon=False,fontsize=12)
    row_names=['TGF-β response\nd7 (early)','TGF-β response\nd21 (late)',
               'Inflammation\nd7 (early)','Inflammation\nd21 (late)',
               'Fibrosis / ECM\nd7 (early)','Fibrosis / ECM\nd21 (late)']
    boxes=[]
    for r,(y0,y1) in enumerate(rows):
        for c,(x0,x1) in enumerate(columns):
            ax=fig.add_subplot(right[r,c])
            # Square extraction includes black background at map edges.
            size=max(y1-y0,x1-x0)
            xc=(x0+x1)//2; yc=(y0+y1)//2
            box=(xc-size//2,yc-size//2,xc-size//2+size,yc-size//2+size)
            crop=pixels[box[1]:box[3],box[0]:box[2]]
            ax.imshow(crop,interpolation='none'); ax.set_xticks([]); ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if r==0: ax.set_title(['Real','K=20','K=50','K=100'][c],fontsize=13,weight='bold',pad=10)
            if c==0:
                pos=ax.get_position()
                fig.text(.585,(pos.y0+pos.y1)/2,row_names[r],fontsize=11,
                         ha='right',va='center',weight='bold')
            boxes.append([int(v) for v in box])
    fig.text(.265,.965,'GSE267904 State-Recovery Benchmark\nacross Leca-VC Agent Granularities',ha='center',va='top',fontsize=16,weight='bold')
    fig.text(.755,.965,'GSE267904 Module-Expression Recovery\nacross Agent Granularities',ha='center',va='top',fontsize=16,weight='bold')
    stem=OUT/'GSE267904_balanced_metrics_and_spatial_maps'
    fig.savefig(str(stem)+'_preview.png',dpi=140)
    fig.savefig(str(stem)+'_600dpi.png',dpi=600)
    fig.savefig(str(stem)+'.pdf',metadata={'CreationDate':None})
    fig.savefig(str(stem)+'.svg',metadata={'Date':None})
    plt.close(fig)
    audit={'status':'PASS','equal_panel_width':.46,'scientific_metrics_recomputed':False,
           'spatial_source':str(SOURCE),'spatial_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           'map_crop_boxes':boxes,'pdf':'vector text and curves with raster spatial maps',
           'svg':str(stem)+'.svg','source_outputs_modified':False}
    (OUT/'COMPLETE.json').write_text(json.dumps(audit,indent=2))
    print(OUT)

if __name__=='__main__': main()
