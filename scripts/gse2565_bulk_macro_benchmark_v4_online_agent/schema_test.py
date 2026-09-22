#!/usr/bin/env python3
from __future__ import annotations
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent';BRIDGE=ROOT/'scripts/gse2565_bulk_macro_benchmark_v4_online_agent/runtime_bridge.py'
def main():
 if not os.getenv('DEEPSEEK_API_KEY'):raise RuntimeError('DEEPSEEK_API_KEY is not set')
 rd=OUT/'protocol_mock/runs/agent_physicell_v4_online__virtual_air_control__seed_256501__mock_air';r=subprocess.run(['python',str(BRIDGE),'--run-dir',str(rd),'--minute','0','--schema-test-agent','AT1'],cwd=ROOT,capture_output=True,text=True);passed=r.returncode==0 and (rd/'schema_test_response.json').exists();result={'all_passed':passed,'agent_id':'AT1','minute':0,'phase':'schema_test','exit_code':r.returncode,'stdout':r.stdout[-1000:],'stderr':r.stderr[-1000:],'reusable_for_pilot':False};(OUT/'audit/schema_test_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0 if passed else 2)
if __name__=='__main__':main()
