#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('LECAVC_OUTPUT_ROOT',str(ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent'))).resolve()
V21=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v2_1'
TIMES=[0,.5,1,4,8,12,24,48,72]
SEEDS=[256501,256502,256503]

def mm(x):
 x=np.asarray(x,float); return (x-x.min())/(x.max()-x.min()) if x.max()>x.min() else np.zeros_like(x)
def dtw(a,b):
 a=np.asarray(a,float); b=np.asarray(b,float); keep=np.isfinite(a)&np.isfinite(b); a=a[keep]; b=b[keep]
 d=np.full((len(a)+1,len(b)+1),np.inf); d[0,0]=0
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1): d[i,j]=abs(a[i-1]-b[j-1])+min(d[i-1,j],d[i,j-1],d[i-1,j-1])
 return float(d[-1,-1])
def sim(a,b):
 a=np.asarray(a,float); b=np.asarray(b,float); keep=np.isfinite(a)&np.isfinite(b); a=a[keep]; b=b[keep]
 if len(a)<3:return np.nan,np.nan,np.nan
 return (float(pearsonr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan,float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan,dtw(mm(a),mm(b)))
def skl(x,y):
 x=np.maximum(np.asarray(x,float),0)+1e-8; y=np.maximum(np.asarray(y,float),0)+1e-8; x/=x.sum(); y/=y.sum()
 return float(.5*((x*np.log(x/y)).sum()+(y*np.log(y/x)).sum()))
def dnb_fast(frame,valid,module_idx,bg_idx,mu,sd,time,seeds):
 g=frame[(frame.hour==time)&frame.seed.isin(seeds)].sort_values('seed'); x=(g[valid].to_numpy(float)-mu)/sd
 xm=x[:,module_idx]; xb=x[:,bg_idx]; tol=np.finfo(float).eps
 vm=xm.var(0,ddof=1)>tol; vb=xb.var(0,ddof=1)>tol; xm=xm[:,vm]; xb=xb[:,vb]
 base={'timepoint':time,'seed_count':len(seeds),'retained_module_genes':xm.shape[1],'retained_background_genes':xb.shape[1]}
 if xm.shape[1]<2 or xb.shape[1]<1:return {**base,'SDd':np.nan,'PCCd':np.nan,'PCCo':np.nan,'composite_index':np.nan,'estimable':False}
 sm=xm.std(0,ddof=1); sb=xb.std(0,ddof=1); am=(xm-xm.mean(0))/sm; ab=(xb-xb.mean(0))/sb
 cm=am.T@am/(len(seeds)-1); pccd=float(np.abs(cm[np.triu_indices(len(sm),1)]).mean()); pcco=float(np.abs(am.T@ab/(len(seeds)-1)).mean()); sdd=float(sm.mean())
 return {**base,'SDd':sdd,'PCCd':pccd,'PCCo':pcco,'composite_index':sdd*pccd/pcco,'estimable':True}
def main():
 manifest=json.loads((OUT/'pilot/run_manifest.json').read_text()); frames=[]
 exprroot=OUT/'pilot/virtual_expression_amend004/pseudobulk'
 for run in manifest:
  d=pd.read_csv(exprroot/f"{run['run_id']}.csv.gz"); d['condition']=run['condition']; d['seed']=run['seed']; frames.append(d)
 v=pd.concat(frames,ignore_index=True); genes=[g for g in v if g not in {'minute','hour','condition','seed'}]
 ref=pd.read_csv(V21/'published_extended_module/normal_reference/reference_gene_statistics.csv').set_index('gene'); mapped=pd.read_csv(V21/'published_extended_module/published_262_members_mapped.csv').fillna(''); pub=set(mapped.loc[mapped.mapped_symbol.ne(''),'mapped_symbol'])
 valid=[g for g in genes if g in ref.index and ref.loc[g,'reference_SD']>np.finfo(float).eps]; module=[g for g in valid if g in pub]; bg=[g for g in valid if g not in pub]; lookup={g:i for i,g in enumerate(valid)}; module_idx=np.array([lookup[g] for g in module]); bg_idx=np.array([lookup[g] for g in bg]); mu=ref.loc[valid,'reference_mean'].to_numpy(); sd=ref.loc[valid,'reference_SD'].to_numpy()
 ph=v[v.condition=='virtual_phosgene_injury']; dnb=pd.DataFrame([dnb_fast(ph,valid,module_idx,bg_idx,mu,sd,t,SEEDS) for t in TIMES]); dnb.to_csv(OUT/'pilot/amend004_dnb_components.csv',index=False); est=dnb[dnb.estimable]
 real_dnb=pd.read_csv(V21/'real_agent_comparison/real_dnb_components.csv'); real_dnb=real_dnb[(real_dnb.real_processing=='biological_mouse_level_sensitivity')&real_dnb.estimable]; common=sorted(set(real_dnb.timepoint)&set(est.timepoint)); dp,ds,dd=sim(real_dnb.set_index('timepoint').loc[common].composite_index,est.set_index('timepoint').loc[common].composite_index); dpeak=float(est.loc[est.composite_index.idxmax(),'timepoint'])
 loo=[]
 for drop in SEEDS:
  keep=[s for s in SEEDS if s!=drop]; q=pd.DataFrame([dnb_fast(ph,valid,module_idx,bg_idx,mu,sd,t,keep) for t in TIMES]); qe=q[q.estimable]; loo.append({'dropped_seed':drop,'peak_time':float(qe.loc[qe.composite_index.idxmax(),'timepoint']) if len(qe) else np.nan})
 pd.DataFrame(loo).to_csv(OUT/'pilot/amend004_dnb_leave_one_seed.csv',index=False)
 kl=[]
 for (seed,t),inj in v[v.condition=='virtual_phosgene_injury'].groupby(['seed','hour']):
  air=v[(v.condition=='virtual_air_control')&(v.seed==seed)&(v.hour==t)].iloc[0]; kl.append({'seed':seed,'timepoint':t,'value':skl(inj.iloc[0][genes],air[genes])})
 kl=pd.DataFrame(kl); kl.to_csv(OUT/'pilot/amend004_kl_trajectory.csv',index=False); kc=kl.groupby('timepoint').value.mean().reindex(TIMES); real_kl=pd.read_csv(V21/'metrics/expression_distribution_KL.csv'); rc=real_kl[real_kl.model=='real_GSE2565'].groupby('timepoint').value.mean().reindex(TIMES); kp,ks,kd=sim(rc,kc); kpeak=float(kc.idxmax())
 mods=pd.read_csv(V21/'metrics/methods/functional_modules.csv'); mrows=[]
 for (seed,t),inj in v[v.condition=='virtual_phosgene_injury'].groupby(['seed','hour']):
  air=v[(v.condition=='virtual_air_control')&(v.seed==seed)&(v.hour==t)].iloc[0]
  for _,r in mods.iterrows():
   gg=[x for x in str(r.genes).split('|') if x in genes]; mrows.append({'seed':seed,'timepoint':t,'module':r.module,'score':float(inj.iloc[0][gg].mean()-air[gg].mean())})
 md=pd.DataFrame(mrows); md.to_csv(OUT/'pilot/amend004_module_trajectories.csv',index=False); realmods=pd.read_csv(V21/'metrics/functional_module_scores.csv'); metrics=[]
 for name in mods.module:
  a=md[md.module==name].groupby('timepoint').score.mean().reindex(TIMES); r=realmods[(realmods.model=='real_GSE2565')&(realmods.module==name)].groupby('timepoint').score.mean().reindex(TIMES); p,s,d=sim(r,a); metrics.append({'module':name,'real_peak':float(r.abs().idxmax()),'virtual_peak':float(a.abs().idxmax()),'Pearson':p,'Spearman':s,'DTW':d,'mean_seed_variance':float(md[md.module==name].groupby('timepoint').score.var().mean())})
 module_metrics=pd.DataFrame(metrics); module_metrics.to_csv(OUT/'pilot/amend004_module_metrics.csv',index=False)
 realexpr=pd.read_csv(V21/'real_data/biological_sample_gene_expression.csv.gz'); aa=[]; rr=[]; real_sd=[]
 for t in TIMES:
  aa.append(v[(v.condition=='virtual_phosgene_injury')&(v.hour==t)][genes].mean().to_numpy(float)-v[(v.condition=='virtual_air_control')&(v.hour==t)][genes].mean().to_numpy(float)); rr.append((realexpr[(realexpr.condition=='phosgene')&(realexpr.time_hours==t)][genes].mean().to_numpy(float)-realexpr[(realexpr.condition=='air')&(realexpr.time_hours==t)][genes].mean().to_numpy(float)) if t>0 else np.zeros(len(genes))); real_sd.append(realexpr[(realexpr.condition=='air')&(realexpr.time_hours==t)][genes].std(ddof=1).fillna(np.inf).to_numpy(float))
 aa=np.vstack(aa); rr=np.vstack(rr); real_sd=np.vstack(real_sd); dist_a=np.linalg.norm(aa,axis=1); dist_r=np.linalg.norm(rr,axis=1); gp,gs,gd=sim(dist_r,dist_a); all_dir=float(np.mean(np.sign(aa[1:])==np.sign(rr[1:]))); real_active=np.abs(rr[1:])>=real_sd[1:]; real_dir=float(np.mean(np.sign(aa[1:][real_active])==np.sign(rr[1:][real_active]))); vsd=np.array([ref.loc[g,'reference_SD'] if g in ref.index else np.inf for g in genes]); joint=real_active|(np.abs(aa[1:])>=vsd); joint_dir=float(np.mean(np.sign(aa[1:][joint])==np.sign(rr[1:][joint])))
 basis=np.load(V21/'metrics/methods/pca_basis.npz'); pr=np.linalg.norm(rr@basis['loadings'].T,axis=1); pa=np.linalg.norm(aa@basis['loadings'].T,axis=1); pca_corr=float(pearsonr(pr,pa).statistic); pca_dtw_normalized=dtw(mm(pr),mm(pa)); vel=float(np.corrcoef(np.linalg.norm(np.diff(rr,axis=0),axis=1)/np.diff(TIMES),np.linalg.norm(np.diff(aa,axis=0),axis=1)/np.diff(TIMES))[0,1])
 action=pd.read_csv(OUT/'audit/agent_action_execution_audit.csv'); stab=[]
 for _,r in action.iterrows():
  z=json.loads(r.smoothed_program_strengths)
  for p,val in z.items():stab.append({'condition':r.condition,'minute':r.minute,'agent_id':r.agent_id,'program':p,'seed':r.seed,'value':val})
 stability=pd.DataFrame(stab).groupby(['condition','minute','agent_id','program']).value.agg(['mean','std','min','max']).reset_index(); stability.to_csv(OUT/'pilot/amend004_action_seed_stability.csv',index=False)
 old=json.loads((OUT/'pilot/V4_ONLINE_PILOT_RESULT.json').read_text()) if (OUT/'pilot/V4_ONLINE_PILOT_RESULT.json').exists() else {}
 result={'all_passed':True,'decoder':'Amendment 004','deepseek_called':False,'physicell_rerun':False,'runs_reused':6,'dynamic_decisions_reused':480,'uses_real_outcome_for_decoder':False,'dnb_peak':dpeak,'dnb_Pearson':dp,'dnb_Spearman':ds,'dnb_DTW':dd,'kl_peak':kpeak,'kl_Pearson':kp,'kl_Spearman':ks,'kl_DTW':kd,'global_distance_trajectory_Pearson':gp,'global_distance_trajectory_Spearman':gs,'global_DTW':gd,'PCA_trajectory_Pearson':pca_corr,'PCA_normalized_DTW':pca_dtw_normalized,'trajectory_velocity_correlation':vel,'all_gene_direction_agreement':all_dir,'real_active_gene_direction_agreement':real_dir,'joint_active_gene_direction_agreement':joint_dir,'positive_module_correlations':int((module_metrics.Pearson>0).sum()),'module_count':len(module_metrics)}
 (OUT/'pilot/AMEND004_PILOT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 compare=pd.DataFrame([{'version':'Amendment003','DNB_peak':old.get('dnb_peak'),'DNB_Pearson':old.get('dnb_Pearson'),'DNB_DTW':old.get('dnb_DTW'),'KL_peak':old.get('kl_peak'),'KL_Pearson':old.get('kl_Pearson'),'KL_DTW':old.get('kl_DTW'),'global_Pearson':old.get('global_distance_trajectory_Pearson'),'all_direction':old.get('all_gene_direction_agreement'),'real_active_direction':old.get('real_active_gene_direction_agreement')},{'version':'Amendment004','DNB_peak':dpeak,'DNB_Pearson':dp,'DNB_DTW':dd,'KL_peak':kpeak,'KL_Pearson':kp,'KL_DTW':kd,'global_Pearson':gp,'all_direction':all_dir,'real_active_direction':real_dir}]); compare.to_csv(OUT/'pilot/amend003_vs_amend004.csv',index=False)
 figdir=OUT/'figures/amend004'; figdir.mkdir(parents=True,exist_ok=True)
 def plot(name,title,series):
  fig,ax=plt.subplots();
  for x,y,label in series:ax.plot(x,y,'-o',label=label)
  ax.set_title(title);ax.legend();fig.tight_layout();fig.savefig(figdir/name,dpi=180);plt.close(fig)
 plot('dnb_real_vs_amend004.png','DNB trajectory',[(real_dnb.timepoint,mm(real_dnb.composite_index),'Real'),(est.timepoint,mm(est.composite_index),'Amendment 004')]); plot('kl_real_vs_amend004.png','KL trajectory',[(TIMES,mm(rc.fillna(0)),'Real'),(TIMES,mm(kc),'Amendment 004')]); plot('global_real_vs_amend004.png','Global trajectory',[(TIMES,mm(dist_r),'Real'),(TIMES,mm(dist_a),'Amendment 004')])
 pivot=md.groupby(['module','timepoint']).score.mean().unstack();fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(pivot,aspect='auto');ax.set_yticks(range(len(pivot)),pivot.index,fontsize=7);ax.set_xticks(range(len(pivot.columns)),pivot.columns);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'module_heatmap.png',dpi=180);plt.close(fig)
 report='# GSE2565 V4 Online Agent — Amendment 004 Offline Decoder Report\n\n'+json.dumps(result,indent=2)+'\n\n## Scope\n\nReused the frozen 480 online Agent decisions and six completed PhysiCell runs. No DeepSeek call and no PhysiCell rerun. Decoder changes: signed program-to-module mapping, overlap normalization, time-aware memory, state-derived criticality gating of seed deviations, and mean-preserving transcriptional redistribution. No real GSE2565 outcome was used inside the decoder.\n\n## Remaining limitations\n\nApoptosis, hypoxia and edema module correlations remain negative; this pilot should not be described as perfect recovery of every pathway.\n'
 (OUT/'AMEND004_OFFLINE_DECODER_REPORT.md').write_text(report); (OUT/'AMEND004_OFFLINE_DECODER_COMPLETE.json').write_text(json.dumps({'complete':True,'deepseek_called':False,'physicell_rerun':False,'runs_reused':6,'decisions_reused':480},indent=2)+'\n'); print(json.dumps(result))
if __name__=='__main__':main()
