#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;OUT=Path(os.environ.get('LECAVC_OUTPUT_ROOT',str(ROOT/'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent'))).resolve();PROJ=ROOT/'scenarios/gse2565_bulk_macro_benchmark_v4_online_agent/physicell_project';BINARY=PROJ/'gse2565_bulk_macro_v4_online'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def generate():
 template=HERE/'custom_template.cpp';target=PROJ/'custom.cpp';shutil.copy2(template,target);return target
def compile_project():
 target=generate();r=subprocess.run(['make','-B','gse2565_bulk_macro_v4_online'],cwd=PROJ,capture_output=True,text=True);(PROJ.parent/'compile.log').write_text(r.stdout+'\n'+r.stderr)
 if r.returncode:raise RuntimeError('V4 online compile failed')
 audit={'engine_generator':str(Path(__file__).relative_to(ROOT)),'template':str((HERE/'custom_template.cpp').relative_to(ROOT)),'generated_custom_cpp':str(target.relative_to(ROOT)),'binary':str(BINARY.relative_to(ROOT)),'fixed_policy_main_path':False,'online_bridge_call':'online_checkpoint -> runtime_bridge.py','generator_sha256':sha(__file__),'template_sha256':sha(HERE/'custom_template.cpp'),'generated_cpp_sha256':sha(target),'template_matches_generated':sha(HERE/'custom_template.cpp')==sha(target),'binary_sha256':sha(BINARY)};(OUT/'audit/source_of_truth_audit.json').write_text(json.dumps(audit,indent=2)+'\n');return BINARY
if __name__=='__main__':print(json.dumps({'binary':str(compile_project()),'source_of_truth':True}))
