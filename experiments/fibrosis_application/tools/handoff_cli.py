#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
APP = ROOT / "application"
OUT_DEFAULT = REPO_ROOT / "outputs/GSE267904_fibrosis_application_v3"

def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    for key, env in [("gse267904_data_root","GSE267904_DATA_ROOT"),("physicell_root","PHYSICELL_ROOT"),("physicell_binary","PHYSICELL_BINARY"),("scgpt_checkpoint","SCGPT_CHECKPOINT")]:
        if data.get(key): os.environ[env] = str(Path(data[key]).expanduser())
    return data

def out_path(c: dict) -> Path:
    return Path(c.get("output_root") or OUT_DEFAULT).expanduser().resolve()

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2)+"\n")

def run_cmd(cmd: list[str], stage: str, dry: bool=False) -> int:
    print("+", " ".join(cmd))
    if dry: return 0
    result = subprocess.run(cmd)
    if result.returncode:
        write_json(OUT_DEFAULT/"audit/LAST_STAGE_FAILURE.json", {"stage":stage,"exit_code":result.returncode,"command":cmd})
    return result.returncode

def pipeline(c: dict, *args: str, dry: bool=False) -> int:
    out=out_path(c); cmd=[sys.executable,str(APP/"run_pipeline.py"),*args,"--project-root",str(REPO_ROOT),"--out-dir",str(out)]
    return run_cmd(cmd,args[0],dry)

def preflight(c: dict) -> int:
    checks={}
    checks["python_ge_3_10"]=sys.version_info[:2]>=(3,10)
    for m in ["numpy","pandas","scipy","sklearn","yaml","requests","matplotlib","anndata"]: checks[f"python_{m}"]=importlib.util.find_spec(m) is not None
    for x in ["bash","g++","make"]: checks[f"command_{x}"]=shutil.which(x) is not None
    pr=Path(os.environ.get("PHYSICELL_ROOT","")); dr=Path(os.environ.get("GSE267904_DATA_ROOT","")); cp=Path(os.environ.get("SCGPT_CHECKPOINT",""))
    checks.update({"physicell_root":pr.is_dir(),"physicell_headers":(pr/"core/PhysiCell.h").is_file(),"gse267904_raw":(dr/"raw").is_dir(),"cell2location_archive":(dr/"cell2location_strunz2020.zip").is_file(),"scgpt_checkpoint_optional":not str(cp) or cp.exists(),"config":True,"application_source":(APP/"physicell_engine.py").is_file(),"xml":(REPO_ROOT/"physicell/fibrosis_application/PhysiCell_settings_base.xml").is_file(),"policy_schedules":len(list((OUT_DEFAULT/"llm/policies").glob("*.csv")))==12,"api_variable_present":bool(os.environ.get("DEEPSEEK_API_KEY")),"disk_free_ge_20gb":shutil.disk_usage(out_path(c).parent).free>=20*1024**3})
    binary=Path(os.environ.get("PHYSICELL_BINARY",REPO_ROOT/"build/fibrosis_application_v3")); checks["application_binary"]=binary.is_file()
    write_json(out_path(c)/"audit/PREFLIGHT.json",{"checks":checks,"note":"API value was not printed"})
    failed=[k for k,v in checks.items() if not v and k not in {"api_variable_present","application_binary"}]
    print("Preflight failures:",failed,"; binary may be built after this check")
    return 1 if failed else 0

def formal_gate(out: Path) -> tuple[bool,str]:
    q=out/"audit/pre_benchmark_qualification.json"; lock=out/"pre_benchmark_model_lock.json"
    if not q.exists(): return False,"qualification audit missing"
    if not json.loads(q.read_text()).get("all_passed"): return False,"qualification all_passed=false"
    if not lock.exists(): return False,"pre_benchmark_model_lock.json missing"
    return True,"passed"

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("action",choices=["preflight","smoke","pilot","qualification","formal","resume","collect","matrix-dry-run"]); p.add_argument("--config",type=Path,required=True); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(); c=load(a.config); out=out_path(c)
    if a.action=="preflight": return preflight(c)
    if a.action=="smoke": return pipeline(c,"simulate-and-freeze","--smoke-rule",dry=a.dry_run)
    if a.action=="qualification": return pipeline(c,"qualify-pre-d21",dry=a.dry_run)
    if a.action=="pilot":
        import pandas as pd
        m=pd.read_csv(out/"manifests/physicell_418_run_matrix.csv"); ids=m[(m.condition=="natural_progression")&(m.agent_mode=="llm_agent")].run_id.head(3)
        for rid in ids:
            rc=pipeline(c,"simulate-and-freeze","--run-id",str(rid),dry=a.dry_run)
            if rc:return rc
        return 0
    if a.action in {"formal","resume"}:
        ok,reason=formal_gate(out)
        if not ok: print("FORMAL REFUSED:",reason); return 2
        return pipeline(c,"simulate-and-freeze",dry=a.dry_run)
    if a.action=="collect":
        if not (out/"model_frozen.json").exists(): print("COLLECT REFUSED: model_frozen.json missing"); return 2
        rc=pipeline(c,"evaluate-locked-benchmark",dry=a.dry_run); return rc or pipeline(c,"report",dry=a.dry_run)
    import pandas as pd
    m=pd.read_csv(out/"manifests/physicell_418_run_matrix.csv"); print(json.dumps({"runs":len(m),"seeds":sorted(m.seed.unique().tolist()),"would_execute":False})); return 0 if len(m)==418 else 1
if __name__=="__main__": raise SystemExit(main())
