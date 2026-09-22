#!/usr/bin/env python3
"""Qualification gates for smoke and pilot without real late-expression access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v1"


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["smoke","pilot"],required=True);a=p.parse_args()
    cfg=yaml.safe_load((HERE/"experiment.yaml").read_text()); gates=cfg["qualification"]
    manifest=json.loads((OUT/f"simulation/{a.stage}_run_manifest.json").read_text())
    checks={"real_physicell_runs":True,"standard_output_present":True,"finite_nonnegative_substrates":True,"worker_count_and_updates":True,"agent_policy_worker_coverage":True,"represented_abundance_conserved":True,"gene_coverage":True,"rule_agent_not_identical":True,"virtual_air_stability":True,"virtual_injury_early_direction":True}
    details={"runs":len(manifest),"expected_runs":4 if a.stage=="smoke" else 12,"real_late_expression_read":False}
    summaries=[]
    for run in manifest:
        run_dir=OUT/"simulation"/run["run_id"]; audit=json.loads((run_dir/"run_audit.json").read_text())
        checks["real_physicell_runs"] &= audit["exit_code"]==0 and audit["REAL_PHYSICELL_USED"] and not audit["PYTHON_SPATIAL_EXECUTOR_USED"]
        checks["standard_output_present"] &= audit["standard_output_xml_count"]>0 and audit["required_work_cell_snapshots_present"]
        first=pd.read_csv(run_dir/"work_cells_minute_0.csv"); one=pd.read_csv(run_dir/"work_cells_minute_60.csv"); final=pd.read_csv(run_dir/"work_cells_minute_720.csv")
        checks["worker_count_and_updates"] &= len(first)==50 and len(final)==50 and np.isfinite(final[["injury_state","inflammation_state","repair_state"]]).all().all()
        checks["agent_policy_worker_coverage"] &= bool(final.AGENT_POLICY_APPLIED.all()) and final.agent_id.nunique()==10
        checks["represented_abundance_conserved"] &= abs(first.represented_abundance.sum()-1)<=gates["abundance_relative_error_max"] and abs(final.represented_abundance.sum()-1)<=gates["abundance_relative_error_max"]
        for path in run_dir.glob("substrates_minute_*.csv"):
            x=pd.read_csv(path); values=x.iloc[:,4:].to_numpy(float); checks["finite_nonnegative_substrates"] &= np.isfinite(values).all() and (values>=-1e-12).all()
        delta=float(one.injury_state.mean()-first.injury_state.mean()); final_delta=float(final.injury_state.mean()-first.injury_state.mean())
        summaries.append({"run_id":run["run_id"],"model":run["model"],"condition":run["condition"],"seed":run["seed"],"injury_delta_1h":delta,"injury_delta_12h":final_delta,"final_inflammation":float(final.inflammation_state.mean()),"final_repair":float(final.repair_state.mean())})
        if run["condition"]=="virtual_air_control": checks["virtual_air_stability"] &= abs(final_delta)<=gates["virtual_air_max_mean_injury_change"]
        else: checks["virtual_injury_early_direction"] &= delta>=gates["virtual_injury_min_early_injury_change"]
    cov=pd.read_csv(OUT/"virtual_expression/gene_coverage_audit.csv")
    current={r["run_id"] for r in manifest}; cov=cov[cov.run_id.isin(current)]
    checks["gene_coverage"] &= len(cov)==len(manifest) and (cov.common_genes>=gates["common_gene_coverage_min"]).all()
    s=pd.DataFrame(summaries);pd.DataFrame(summaries).to_csv(OUT/f"audit/{a.stage}_engineering_summaries.csv",index=False)
    paired_model_differences = []
    for condition in cfg["conditions"]:
        for seed in cfg["seeds"][a.stage]:
            z=s[(s.condition==condition)&(s.seed==seed)].set_index("model")
            if len(z)==2:
                rule = z.loc["traditional_rule_physicell",["injury_delta_12h","final_inflammation","final_repair"]].to_numpy(dtype=float)
                agent = z.loc["agent_physicell_full",["injury_delta_12h","final_inflammation","final_repair"]].to_numpy(dtype=float)
                paired_model_differences.append(not np.allclose(rule, agent, atol=1e-10))
    checks["rule_agent_not_identical"] = any(paired_model_differences)
    if a.stage=="pilot":
        checks["pilot_seed_variance_nonzero"]=True;checks["pilot_seed_cv_bounded"]=True
        for (model,condition),g in s.groupby(["model","condition"]):
            vals=g.injury_delta_12h.to_numpy(); checks["pilot_seed_variance_nonzero"] &= float(np.var(vals))>=gates["pilot_seed_variance_min"];mean=max(abs(float(np.mean(vals))),1e-12);checks["pilot_seed_cv_bounded"] &= float(np.std(vals)/mean)<=gates["pilot_seed_cv_max"]
    result={"stage":a.stage,"checks":{k:bool(v) for k,v in checks.items()},"all_passed":all(checks.values()),"details":details,"qualification_thresholds":gates}
    (OUT/f"audit/{a.stage}_qualification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    if not result["all_passed"]: raise SystemExit(2)


if __name__=="__main__": main()
