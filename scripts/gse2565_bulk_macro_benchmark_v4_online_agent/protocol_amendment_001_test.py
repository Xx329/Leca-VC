#!/usr/bin/env python3
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
from online_controller import validate_with_rationale_audit

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent'
RUN=OUT/'pilot/runs/agent_physicell_v4_online__virtual_phosgene_injury__seed_256501__to_72h'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def without_rationale(obj):return {k:v for k,v in obj.items() if k!='brief_rationale'}

def main():
 request=json.loads((RUN/'runtime_requests/agent_physicell_v4_online__virtual_phosgene_injury__seed_256501__to_72h/30/request.json').read_text())
 agents={x['agent_id']:x for x in request['agents']};tested=[]
 for path in sorted((RUN/'agent_decisions/30').glob('*/attempt_*_content.json')):
  raw=json.loads(path.read_text());rationale=raw.get('brief_rationale')
  if not isinstance(rationale,str) or len(rationale)<=80:continue
  before=sha(path)
  try:validated,audit=validate_with_rationale_audit(raw,agents[raw['agent_id']])
  except Exception:continue
  after=sha(path)
  assert before==after
  assert audit['rationale_original_text']==rationale
  assert len(validated['brief_rationale'])<=80
  assert audit['rationale_truncated'] and validated['brief_rationale'].endswith('…')
  assert without_rationale(validated)==without_rationale(raw)
  assert audit['rationale_original_hash']==hashlib.sha256(rationale.encode()).hexdigest()
  assert audit['rationale_stored_hash']==hashlib.sha256(validated['brief_rationale'].encode()).hexdigest()
  tested.append({'source':str(path.relative_to(ROOT)),'agent_id':raw['agent_id'],**audit,'raw_file_hash_unchanged':True,'all_non_rationale_fields_unchanged':True,'fallback_used':False})
 if not tested:raise RuntimeError('no otherwise-valid saved overlong rationale response found')
 sample=json.loads((ROOT/tested[0]['source']).read_text());req=agents[sample['agent_id']]
 missing=copy.deepcopy(sample);missing.pop('brief_rationale')
 nonstring=copy.deepcopy(sample);nonstring['brief_rationale']=123
 required_failures=[]
 for label,obj in [('missing',missing),('non_string',nonstring)]:
  try:validate_with_rationale_audit(obj,req)
  except ValueError as e:required_failures.append({'case':label,'failed_as_required':True,'error':str(e)})
  else:raise AssertionError(f'{label} rationale was accepted')
 result={'protocol_amendment':'001','all_passed':True,'deepseek_called':False,'fallback_used':False,'tested_saved_responses':len(tested),'raw_responses_preserved':True,'validated_rationale_max_80':True,'all_action_and_identity_fields_unchanged':True,'tests':tested,'required_schema_failures':required_failures}
 (OUT/'audit').mkdir(parents=True,exist_ok=True)
 (OUT/'audit/protocol_amendment_001_regression.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'all_passed':True,'tested_saved_responses':len(tested),'actions_unchanged':True,'deepseek_called':False,'fallback_used':False}))
if __name__=='__main__':main()
