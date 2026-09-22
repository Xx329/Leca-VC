#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, hashlib, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('LECAVC_OUTPUT_ROOT',str(ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent'))).resolve()
V21=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v2_1'
MINUTES=[0,30,60,240,480,720,1440,2880,4320]
CHECKPOINTS=MINUTES[:-1]
TH={'cumulative_exposure':(.005,.060),'oxidative_stress':(.002,.020),'damage_memory':(.010,.080),'epithelial_injury':(.002,.030),'inflammatory_memory':(.001,.020),'inflammation_state':(.001,.020),'inflammation':(.002,.020),'death_signal':(.002,.020),'resolution_memory':(.005,.025),'repair_state':(.002,.020),'repair_signal':(.002,.020),'edema_proxy':(.002,.020)}
HALF_LIFE_H={'oxidative_stress':2.0,'epithelial_injury':5.0,'inflammation':8.0,'immune_recruitment':8.0,'edema_proxy':4.0,'death_signal':5.0,'repair_signal':10.0,'ecm_remodeling':14.0}
PROGRAM_MODULE_WEIGHTS={
 'oxidative_stress':{'oxidative_stress':1.0,'hypoxia':0.35},
 'epithelial_injury':{'alveolar_epithelial_injury_barrier':-1.0,'apoptosis_cell_death':0.35,'tissue_repair':-0.20,'ECM_remodeling':-0.20},
 'inflammation':{'inflammation':1.0,'NFkB_TNF':0.8},
 'immune_recruitment':{'immune_recruitment':1.0},
 'edema_proxy':{'vascular_permeability_edema_proxy':-1.0},
 'death_signal':{'apoptosis_cell_death':1.0},
 'repair_signal':{'tissue_repair':1.0},
 'ecm_remodeling':{'ECM_remodeling':1.0},
}
COEF=0.42
NOISE_SIGMA=0.0

def norm(x,name):
 off,on=TH[name]; return np.clip((np.asarray(x,float)-off)/(on-off),0,1)
def compat(agent,program):
 epi=('AT' in agent or 'epithelial' in agent); immune=any(x in agent for x in ['macrophage','monocyte','neutrophil','lymphoid']); fib='fibroblast' in agent; endo=agent=='endothelial'
 return {'oxidative_stress':epi or endo,'epithelial_injury':epi or endo,'inflammation':immune,'immune_recruitment':immune,'edema_proxy':endo,'death_signal':epi or endo or immune,'repair_signal':fib or agent in {'AT2','resident_macrophage'},'ecm_remodeling':fib}.get(program,False)
def evidence(c,program):
 if program=='oxidative_stress': return np.maximum(norm(c.cumulative_exposure,'cumulative_exposure'),norm(c.oxidative_stress,'oxidative_stress'))
 if program=='epithelial_injury': return np.maximum(norm(c.damage_memory,'damage_memory'),norm(c.epithelial_injury,'epithelial_injury'))
 if program in {'inflammation','immune_recruitment'}: return np.maximum.reduce([norm(c.inflammatory_memory,'inflammatory_memory'),norm(c.inflammation_state,'inflammation_state'),norm(c.inflammation,'inflammation')])
 if program=='edema_proxy': return np.maximum(norm(c.edema_proxy,'edema_proxy'),norm(c.damage_memory,'damage_memory'))
 if program=='death_signal': return np.maximum(norm(c.death_signal,'death_signal'),norm(c.damage_memory,'damage_memory'))
 if program in {'repair_signal','ecm_remodeling'}: return np.maximum.reduce([norm(c.resolution_memory,'resolution_memory'),norm(c.repair_state,'repair_state'),norm(c.repair_signal,'repair_signal')])*np.asarray(c.post_exposure,float)
 return np.zeros(len(c))
def fixed_noise(seed,agent,program):
 h=hashlib.sha256(f'{seed}|{agent}|{program}|amend004'.encode()).digest()
 u=(int.from_bytes(h[:8],'big')+0.5)/(2**64)
 v=(int.from_bytes(h[8:16],'big')+0.5)/(2**64)
 return float(np.sqrt(-2*np.log(max(u,1e-12)))*np.cos(2*np.pi*v))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--stage',default='pilot'); p.add_argument('--manifest'); a=p.parse_args()
 manifest=json.loads(Path(a.manifest or OUT/a.stage/'run_manifest.json').read_text())
 proto=np.load(OUT/'initialization/healthy_expression_prototypes.npz',allow_pickle=False)
 genes=list(map(str,proto['genes'])); gidx={g:i for i,g in enumerate(genes)}
 bases={str(x):proto['expression'][i].astype(float) for i,x in enumerate(proto['agent_types'])}
 ref=pd.read_csv(V21/'published_extended_module/normal_reference/reference_gene_statistics.csv').set_index('gene')
 refsd=np.array([float(ref.loc[g,'reference_SD']) if g in ref.index and np.isfinite(ref.loc[g,'reference_SD']) else 0.0 for g in genes]); refsd=np.maximum(refsd,1e-4)
 mods=pd.read_csv(V21/'metrics/methods/functional_modules.csv'); sets={r.module:[gidx[g] for g in str(r.genes).split('|') if g in gidx] for _,r in mods.iterrows()}
 overlap=np.zeros(len(genes),float)
 for pm in PROGRAM_MODULE_WEIGHTS.values():
  for module,w in pm.items():
   for j in sets.get(module,[]): overlap[j]+=abs(w)
 overlap=np.maximum(overlap,1.0)
 # Precompute per-seed agent-level raw drives, then ensemble means. LLM seed deviations are retained only near state-derived criticality.
 drive_rows=[]; criticality_by_run_time={}
 for run in manifest:
  rid=run['run_id']; rd=ROOT/run['run_dir']
  for minute in MINUTES:
   c=pd.read_csv(rd/f'work_cells_minute_{minute}.csv').sort_values('worker_id').reset_index(drop=True); checkpoint=max(q for q in CHECKPOINTS if q<=minute)
   ep=json.loads((rd/'runtime_responses'/rid/str(checkpoint)/'execution_payload.json').read_text()); actions={z['agent_id']:z for z in ep['agents']}
   damage=np.maximum(norm(c.damage_memory,'damage_memory'),norm(c.epithelial_injury,'epithelial_injury')); resolution=np.maximum.reduce([norm(c.resolution_memory,'resolution_memory'),norm(c.repair_state,'repair_state'),norm(c.repair_signal,'repair_signal')])*np.asarray(c.post_exposure,float)
   epi_mask=np.array([('AT' in z or 'epithelial' in z or z=='endothelial') for z in c.agent_id]); dbar=float(damage[epi_mask].mean()) if epi_mask.any() else 0.; rbar=float(resolution[epi_mask].mean()) if epi_mask.any() else 0.; crit=4*dbar*(1-dbar)*(1-rbar); criticality_by_run_time[(rid,minute)]=crit
   for agent,g in c.groupby('agent_id'):
    ii=g.index.to_numpy()
    for program in PROGRAM_MODULE_WEIGHTS:
     ev=evidence(c,program); ex=float(actions[agent]['executed_'+program]); raw=float(ex*ev[ii].mean()*(1.0 if compat(agent,program) else 0.0))
     # biological gating
     res=float(resolution[ii].mean())
     
     if program=='oxidative_stress': raw*=max(0.,1-.25*res)
     elif program in {'epithelial_injury','inflammation','immune_recruitment','edema_proxy','death_signal'}: raw*=max(0.,1-.75*res)
     if program in {'repair_signal','ecm_remodeling'}: raw*=res
     drive_rows.append({'run_id':rid,'condition':run['condition'],'seed':run['seed'],'minute':minute,'agent_id':agent,'program':program,'raw_drive':raw})
 drives=pd.DataFrame(drive_rows); means=drives.groupby(['condition','minute','agent_id','program']).raw_drive.mean().rename('ensemble_drive').reset_index(); drives=drives.merge(means,on=['condition','minute','agent_id','program']); drive_lookup={(r.run_id,int(r.minute),r.agent_id,r.program):(float(r.raw_drive),float(r.ensemble_drive)) for r in drives.itertuples()}
 cellroot=OUT/a.stage/'virtual_expression_amend004/cell_level'; bulkroot=OUT/a.stage/'virtual_expression_amend004/pseudobulk'; cellroot.mkdir(parents=True,exist_ok=True); bulkroot.mkdir(parents=True,exist_ok=True)
 coupling=[]; weights=[]
 for run in manifest:
  rid=run['run_id']; rd=ROOT/run['run_dir']; memory={}; rows=[]; prev_minute=None; (cellroot/rid).mkdir(parents=True,exist_ok=True)
  for minute in MINUTES:
   c=pd.read_csv(rd/f'work_cells_minute_{minute}.csv').sort_values('worker_id').reset_index(drop=True); x=np.vstack([bases[z].copy() for z in c.agent_id]); dt_h=0.0 if prev_minute is None else (minute-prev_minute)/60.; prev_minute=minute
   damage=np.maximum(norm(c.damage_memory,'damage_memory'),norm(c.epithelial_injury,'epithelial_injury')); resolution=np.maximum.reduce([norm(c.resolution_memory,'resolution_memory'),norm(c.repair_state,'repair_state'),norm(c.repair_signal,'repair_signal')])*np.asarray(c.post_exposure,float); criticality=criticality_by_run_time[(rid,minute)]
   delta=np.zeros_like(x,float)
   for program,module_weights in PROGRAM_MODULE_WEIGHTS.items():
    target=np.zeros(len(c),float); raw_seed=np.zeros(len(c),float); ens=np.zeros(len(c),float)
    for agent,g in c.groupby('agent_id'):
     ii=g.index.to_numpy(); rs,em=drive_lookup[(rid,minute,agent,program)]; raw_seed[ii]=rs; ens[ii]=em; target[ii]=em+criticality*(rs-em)
    prev=memory.get(program,np.zeros(len(c))); decay=0.0 if dt_h<=0 else math.exp(-math.log(2)*dt_h/HALF_LIFE_H[program]); mem=target.copy() if dt_h<=0 else decay*prev+(1-decay)*target; memory[program]=mem
    for module,wgt in module_weights.items():
     idx=sets.get(module,[])
     if not idx: continue
     delta[:,idx]+=mem[:,None]*(wgt/overlap[idx])[None,:]
     for agent,g in c.groupby('agent_id'):
      ii=g.index.to_numpy(); coupling.append({'run_id':rid,'seed':run['seed'],'condition':run['condition'],'minute':minute,'agent_id':agent,'module':module,'program':program,'seed_raw_drive':float(raw_seed[ii].mean()),'ensemble_drive':float(ens[ii].mean()),'criticality':criticality,'effective_drive':float(target[ii].mean()),'memory':float(mem[ii].mean()),'signed_weight':wgt,'affected_gene_count':len(idx)})
   acute=np.clip(damage*(1-resolution),0,1)
   for module,wgt in {'alveolar_epithelial_injury_barrier':-0.30,'tissue_repair':-0.12,'ECM_remodeling':-0.12}.items():
    idx=sets.get(module,[])
    if idx: delta[:,idx]+=acute[:,None]*(wgt/overlap[idx])[None,:]
   offset=(delta*refsd[None,:]).sum(axis=1,keepdims=True)/refsd.sum(); delta=delta-offset; x=np.maximum(x+COEF*delta*refsd[None,:],0).astype('float32'); w=c.represented_abundance.to_numpy(float); pb=np.average(x,axis=0,weights=w)
   np.savez_compressed(cellroot/rid/f'worker_expression_minute_{minute}.npz',genes=np.array(genes),worker_ids=c.worker_id.astype(str).to_numpy(),agent_ids=c.agent_id.astype(str).to_numpy(),represented_abundance=w,expression=x); rows.append({'minute':minute,'hour':minute/60,**dict(zip(genes,map(float,pb)))}); weights.append({'run_id':rid,'seed':run['seed'],'condition':run['condition'],'minute':minute,'sum_abundance':float(w.sum())})
  pd.DataFrame(rows).to_csv(bulkroot/f'{rid}.csv.gz',index=False,compression='gzip')
 pd.DataFrame(coupling).to_csv(OUT/'audit/amend004_expression_coupling_audit.csv',index=False); pd.DataFrame(weights).to_csv(OUT/'audit/amend004_abundance.csv',index=False); print(json.dumps({'runs':len(manifest),'genes':len(genes),'decoder':'amend004_ensemble_criticality','uses_real_outcome_for_decoder':False}))
if __name__=='__main__': main()
