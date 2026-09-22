#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent';V21=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v2_1';MINUTES=[0,30,60,240,480,720,1440,2880,4320];CHECKPOINTS=MINUTES[:-1]
PROGRAM_MODULES={'oxidative_stress':['oxidative_stress'],'epithelial_injury':['alveolar_epithelial_injury_barrier'],'inflammation':['inflammation','NFkB_TNF'],'immune_recruitment':['immune_recruitment'],'edema_proxy':['vascular_permeability_edema_proxy'],'death_signal':['apoptosis_cell_death'],'repair_signal':['tissue_repair'],'ecm_remodeling':['ECM_remodeling']}
TH={'cumulative_exposure':(.005,.060),'oxidative_stress':(.002,.020),'damage_memory':(.010,.080),'epithelial_injury':(.002,.030),'inflammatory_memory':(.001,.020),'inflammation_state':(.001,.020),'inflammation':(.002,.020),'death_signal':(.002,.020),'resolution_memory':(.005,.025),'repair_state':(.002,.020),'repair_signal':(.002,.020),'edema_proxy':(.002,.020)}
def norm(x,name):off,on=TH[name];return np.clip((np.asarray(x,float)-off)/(on-off),0,1)
def compat(agent,program):
 epi=('AT' in agent or 'epithelial' in agent);immune=any(x in agent for x in ['macrophage','monocyte','neutrophil','lymphoid']);fib='fibroblast' in agent;endo=agent=='endothelial'
 return {'oxidative_stress':epi or endo,'epithelial_injury':epi or endo,'inflammation':immune,'immune_recruitment':immune,'edema_proxy':endo,'death_signal':epi or endo or immune,'repair_signal':fib or agent in {'AT2','resident_macrophage'},'ecm_remodeling':fib}.get(program,False)
def evidence(c,program):
 if program=='oxidative_stress':return np.maximum(norm(c.cumulative_exposure,'cumulative_exposure'),norm(c.oxidative_stress,'oxidative_stress'))
 if program=='epithelial_injury':return np.maximum(norm(c.damage_memory,'damage_memory'),norm(c.epithelial_injury,'epithelial_injury'))
 if program in {'inflammation','immune_recruitment'}:return np.maximum.reduce([norm(c.inflammatory_memory,'inflammatory_memory'),norm(c.inflammation_state,'inflammation_state'),norm(c.inflammation,'inflammation')])
 if program=='edema_proxy':return np.maximum(norm(c.edema_proxy,'edema_proxy'),norm(c.damage_memory,'damage_memory'))
 if program=='death_signal':return np.maximum(norm(c.death_signal,'death_signal'),norm(c.damage_memory,'damage_memory'))
 if program in {'repair_signal','ecm_remodeling'}:return np.maximum.reduce([norm(c.resolution_memory,'resolution_memory'),norm(c.repair_state,'repair_state'),norm(c.repair_signal,'repair_signal')])*np.asarray(c.post_exposure,float)
 return np.zeros(len(c))
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['protocol_mock','pilot'],required=True);p.add_argument('--manifest');p.add_argument('--minutes',default=','.join(map(str,MINUTES)));a=p.parse_args();minutes=[int(x) for x in a.minutes.split(',')];manifest=json.loads(Path(a.manifest or OUT/a.stage/'run_manifest.json').read_text());proto=np.load(OUT/'initialization/healthy_expression_prototypes.npz',allow_pickle=False);genes=list(map(str,proto['genes']));bases={str(x):proto['expression'][i] for i,x in enumerate(proto['agent_types'])};mods=pd.read_csv(V21/'metrics/methods/functional_modules.csv');sets={r.module:[genes.index(g) for g in str(r.genes).split('|') if g in genes] for _,r in mods.iterrows()};cellroot=OUT/a.stage/'virtual_expression/cell_level';bulkroot=OUT/a.stage/'virtual_expression/pseudobulk';cellroot.mkdir(parents=True,exist_ok=True);bulkroot.mkdir(parents=True,exist_ok=True);coupling=[];weights=[]
 for run in manifest:
  rid=run['run_id'];rd=ROOT/run['run_dir'];memory={};rows=[];(cellroot/rid).mkdir(parents=True,exist_ok=True)
  for minute in minutes:
   c=pd.read_csv(rd/f'work_cells_minute_{minute}.csv').sort_values('worker_id').reset_index(drop=True);x=np.vstack([bases[z].copy() for z in c.agent_id]);checkpoint=max(q for q in CHECKPOINTS if q<=minute);ep=json.loads((rd/'runtime_responses'/rid/str(checkpoint)/'execution_payload.json').read_text());actions={z['agent_id']:z for z in ep['agents']}
   for program,modules in PROGRAM_MODULES.items():
    ev=evidence(c,program);raw=np.array([actions[z]['raw_'+program] for z in c.agent_id]);executed=np.array([actions[z]['executed_'+program] for z in c.agent_id]);mask=np.array([compat(z,program) for z in c.agent_id]);drive=executed*ev*mask;key=program;prev=memory.get(key,np.zeros(len(c)));mem=.4*prev+.6*drive;memory[key]=mem
    for module in modules:
     idx=sets.get(module,[])
     if idx:x[:,idx]+=0.03*ev[:,None]*mask[:,None]+0.20*mem[:,None]
     for agent,g in c.groupby('agent_id'):ii=g.index.to_numpy();coupling.append({'run_id':rid,'seed':run['seed'],'condition':run['condition'],'minute':minute,'agent_id':agent,'module':module,'raw_agent_strength':float(raw[ii].mean()),'smoothed_agent_strength':float(executed[ii].mean()),'state_evidence':float(ev[ii].mean()),'final_expression_drive':float(mem[ii].mean()),'affected_gene_count':len(idx),'cell_type_compatible':bool(mask[ii].all()),'applied':bool(len(idx) and mask[ii].any())})
   x=np.maximum(x,0).astype('float32');w=c.represented_abundance.to_numpy(float);pb=np.average(x,axis=0,weights=w);np.savez_compressed(cellroot/rid/f'worker_expression_minute_{minute}.npz',genes=np.array(genes),worker_ids=c.worker_id.astype(str).to_numpy(),agent_ids=c.agent_id.astype(str).to_numpy(),represented_abundance=w,expression=x);rows.append({'minute':minute,'hour':minute/60,**dict(zip(genes,map(float,pb)))});weights.append({'run_id':rid,'seed':run['seed'],'condition':run['condition'],'minute':minute,'sum_abundance':float(w.sum()),'minimum_abundance':float(w.min()),'worker_count_extra_weight':False})
  pd.DataFrame(rows).to_csv(bulkroot/f'{rid}.csv.gz',index=False,compression='gzip')
 append=pd.DataFrame(coupling);append.to_csv(OUT/'audit/online_expression_coupling_audit.csv',index=False);pd.DataFrame(weights).to_csv(OUT/'audit/abundance_conservation.csv',index=False);print(json.dumps({'runs':len(manifest),'genes':len(genes),'online_action_coupled':True}))
if __name__=='__main__':main()
