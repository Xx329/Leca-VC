#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent';V21=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v2_1'
def main():
 mods=pd.read_csv(V21/'metrics/methods/functional_modules.csv');mapped=pd.read_csv(V21/'published_extended_module/published_262_members_mapped.csv').fillna('');dnb=sorted(set(mapped.loc[mapped.mapped_symbol.ne(''),'mapped_symbol']));rows=[]
 for g in dnb:
  hits=[r.module for _,r in mods.iterrows() if g in str(r.genes).split('|')];rows.append({'gene':g,'covered':bool(hits),'module_count':len(hits),'modules':'|'.join(hits),'duplicate_module_mapping':len(hits)>1})
 pd.DataFrame(rows).to_csv(OUT/'audit/dnb_gene_program_coverage.csv',index=False)
 active={'real_active':'abs(phosgene-air)>=matched-air biological replicate sample SD','virtual_active':'abs(injury-air)>=GSE141259 mouse-level reference SD','machine_precision_used':False,'frozen_before_pilot':True};(OUT/'audit/active_gene_definition.json').write_text(json.dumps(active,indent=2)+'\n')
 print(json.dumps({'dnb_genes':len(rows),'covered':sum(x['covered'] for x in rows)}))
if __name__=='__main__':main()
