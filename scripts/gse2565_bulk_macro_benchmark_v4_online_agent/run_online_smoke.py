#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,os,subprocess,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent';PROJ=ROOT/'scenarios/gse2565_bulk_macro_benchmark_v4_online_agent/physicell_project'
def load(p,n):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 if not os.getenv('DEEPSEEK_API_KEY'):raise RuntimeError('DEEPSEEK_API_KEY is not set')
 validation=json.loads((OUT/'audit/ONLINE_AGENT_ARCHITECTURE_VALIDATION.json').read_text());
 if not validation['all_passed']:raise RuntimeError('online architecture validation failed')
 base=load(ROOT/'scripts/gse2565_bulk_macro_benchmark_v1/physicell_engine.py','base');all_audit=[];runs=[]
 for condition in ['virtual_air_control','virtual_phosgene_injury']:
  rid=f'agent_physicell_v4_online__{condition}__seed_256501__to_72h';rd=OUT/'smoke/runs'/rid;rd.mkdir(parents=True,exist_ok=True);cfg=rd/'PhysiCell_settings.xml';base.runtime_xml(PROJ/'PhysiCell_settings_base.xml',cfg,rd,OUT/'initialization/worker_registry.csv',OUT/'unused_policy.csv',condition,256501,72);env={**os.environ,'V4_ONLINE_TIMEOUT_SECONDS':'900'};r=subprocess.run([str(PROJ/'gse2565_bulk_macro_v4_online'),str(cfg)],cwd=PROJ,env=env);runs.append({'run_id':rid,'condition':condition,'seed':256501,'exit_code':r.returncode});
  if r.returncode:raise RuntimeError('online PhysiCell run failed; no fallback')
  a=pd.read_csv(rd/'agent_action_execution_audit.csv');all_audit.append(a)
 audit=pd.concat(all_audit,ignore_index=True);audit.to_csv(OUT/'audit/agent_action_execution_audit.csv',index=False);result={'all_passed':all(x['exit_code']==0 for x in runs) and len(audit)==160,'runs':runs,'dynamic_decisions':len(audit),'deepseek_expected':160,'fixed_policy_lookup':False};(OUT/'smoke/ONLINE_AGENT_SMOKE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
