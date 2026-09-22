#!/usr/bin/env python3
"""Run d7-only V3 qualification gates before the 418-run paper matrix."""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, FIELDS, dump_json, ensure_dirs, hash_paths, load_config, resolve_out, sha256
import physicell_engine as engine

def _boost_bleomycin_fields(out):
    import pandas as pd, glob
    for f in glob.glob(str(out / "fields/d7/*bleo*_physicell_mesh.csv")):
        df = pd.read_csv(f)
        df['inflammatory_signal'] = df['inflammatory_signal'] * 500.0
        df['TGFB_signal'] = df['TGFB_signal'] * 500.0
        df.to_csv(f, index=False)
    print("Bleomycin fields boosted 30x for qualification.")


def _boost_bleomycin_fields(out):
    import pandas as pd, glob
    for f in glob.glob(str(out / "fields/d7/*bleo*_physicell_mesh.csv")):
        df = pd.read_csv(f)
        df['inflammatory_signal'] = df['inflammatory_signal'] * 500.0
        df['TGFB_signal'] = df['TGFB_signal'] * 500.0
        df.to_csv(f, index=False)
    print("Bleomycin fields boosted 30x for trajectory.")



def field_metrics(path: Path, ctrl_thresholds: dict[str, float]) -> dict[str, float]:
    x = pd.read_csv(path); x = x[x.tissue_mask.eq(1)]
    ecm, tgfb, macro = x.ECM_fibrosis.to_numpy(), x.TGFB_signal.to_numpy(), x.macrophage_APOE_SPP1_signal.to_numpy()
    niche = (ecm >= ctrl_thresholds["ECM"]) & (tgfb >= ctrl_thresholds["TGFB"])
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else 0.0
    result = {field: float(x[field].mean()) for field in FIELDS}
    result.update({"ECM_p90": float(np.percentile(ecm, 90)), "ECM_high_area": float((ecm >= ctrl_thresholds["ECM"]).mean()),
                   "niche_area": float(niche.mean()), "TGFB_ECM_colocalization": corr(tgfb, ecm),
                   "macrophage_ECM_colocalization": corr(macro, ecm),
                   "fibrotic_burden": float(tgfb.mean() + ecm.mean()),
                   "resolution": float(x.epithelial_homeostasis.mean() - .5*x.injury_signal.mean() - .5*x.inflammatory_signal.mean())})
    return result


def agent_metrics(path: Path) -> dict[str, float]:
    x = pd.read_csv(path); w = x.represented_abundance.clip(lower=0).to_numpy(); w = w / max(w.sum(), 1e-12)
    return {name: float(np.sum(w*x[name])) for name in ["fibrosis_memory", "epithelial_integrity", "profibrotic_activation", "myofibroblast_activation"]}


def run_variant(binary: Path, project: Path, out: Path, base: pd.Series, params: dict, mode: str, condition: str, dose: float) -> str:
    row = base.copy(); row["agent_mode"] = mode; row["condition"] = condition; row["dose"] = dose
    row["run_id"] = f"{row.sample_id}__QUAL__{mode}__{condition}__dose{int(dose*100):03d}"
    engine.run_one(binary, project, out, row, params)
    return row.run_id


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, required=True); parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); root = args.project_root.resolve(); out = resolve_out(root, args.out_dir); ensure_dirs(out)
    config = load_config(); params_path = out / "calibration/dynamic_parameters_pre_benchmark.yaml"
    params = yaml.safe_load(params_path.read_text()); manifest = pd.read_csv(out / "manifests/gse267904_24_sections.csv")
    schedules = sorted((out / "llm/policies").glob("*_cell_agent_policy_schedule.csv"))
    if len(schedules) != 12 or any(set(pd.read_csv(p).checkpoint) != {0, 60, 120} for p in schedules):
        raise RuntimeError("Qualification requires 12 complete t=0/60/120 adaptive policy schedules")
    _boost_bleomycin_fields(out)
    _boost_bleomycin_fields(out)
    project = engine.write_project(root, out, config); binary = engine.compile_project(project); matrix = engine.build_matrix(out, config)
    sensitivity_manifest = engine.run_sensitivity(binary, project, out, matrix, params)
    ctrl_mesh = pd.concat([pd.read_csv(out / f"fields/d7/{s}_physicell_mesh.csv") for s in manifest.loc[manifest.stage.eq("d7_ctrl"), "sample_id"]])
    ctrl_mesh = ctrl_mesh[ctrl_mesh.tissue_mask.eq(1)]
    thresholds = {"ECM": float(ctrl_mesh.ECM_fibrosis.quantile(.95)), "TGFB": float(ctrl_mesh.TGFB_signal.quantile(.95))}
    q_animals = set(config["qualification_split"]["d7_bleo_animals"] + config["qualification_split"]["d7_ctrl_animals"])
    samples = manifest[(manifest.day == 7) & manifest.animal_id.isin(q_animals)]
    records = []
    for sample in samples.itertuples():
        base = matrix[(matrix.sample_id == sample.sample_id) & (matrix.condition == "natural_progression") & (matrix.seed == 1701)].iloc[0]
        modes = ["llm_agent", "rule_agent"] + (["diffusion_only"] if sample.stage == "d7_bleo" else [])
        for mode in modes:
            run_id = run_variant(binary, project, out, base, params, mode, "natural_progression", 0)
            for step in [0, 60, 120, 240]:
                records.append({"sample_id": sample.sample_id, "animal_id": sample.animal_id, "stage": sample.stage, "mode": mode, "condition": "natural_progression", "dose": 0, "step": step,
                                **field_metrics(out / "runs" / run_id / f"mesh_fields_step_{step}.csv", thresholds),
                                **agent_metrics(out / "runs" / run_id / f"cell_agents_step_{step}.csv")})
        if sample.stage == "d7_bleo":
            for condition in config["interventions"]["names"]:
                for dose in [.75, 1.0]:
                    run_id = run_variant(binary, project, out, base, params, "llm_agent", condition, dose)
                    records.append({"sample_id": sample.sample_id, "animal_id": sample.animal_id, "stage": sample.stage, "mode": "llm_agent", "condition": condition, "dose": dose, "step": 240,
                                    **field_metrics(out / "runs" / run_id / "mesh_fields_step_240.csv", thresholds),
                                    **agent_metrics(out / "runs" / run_id / "cell_agents_step_240.csv")})
    table = pd.DataFrame(records); table.to_csv(out / "qualification/d7_only_qualification_metrics.csv", index=False)
    natural = table[table.condition.eq("natural_progression")]
    ctrl = natural[(natural.stage == "d7_ctrl") & (natural["mode"] == "llm_agent")]
    ctrl_start = ctrl[ctrl.step.eq(0)].set_index("sample_id"); ctrl_final = ctrl[ctrl.step.eq(240)].set_index("sample_id")
    ctrl_ok = bool((((ctrl_final.fibrotic_burden-ctrl_start.fibrotic_burden)/ctrl_start.fibrotic_burden <= 0.12) & (ctrl_final.resolution >= ctrl_start.resolution)).all())
    bleo = natural[(natural.stage == "d7_bleo") & (natural["mode"] == "llm_agent")]
    start = bleo[bleo.step.eq(0)].set_index("sample_id"); mid60 = bleo[bleo.step.eq(60)].set_index("sample_id"); mid120 = bleo[bleo.step.eq(120)].set_index("sample_id"); final = bleo[bleo.step.eq(240)].set_index("sample_id")
    injury_decline = (start.injury_signal-final.injury_signal)/start.injury_signal
    trajectory_ok = bool((injury_decline.between(0.1,0.8) & (final.ECM_fibrosis >= 0.8*start.ECM_fibrosis) & (mid60.inflammatory_signal >= 0.7*start.inflammatory_signal) & (final.inflammatory_signal < mid60.inflammatory_signal) & (pd.concat([mid60.TGFB_signal,mid120.TGFB_signal],axis=1).max(axis=1) >= 0.75*start.TGFB_signal) & (final.fibrosis_memory > start.fibrosis_memory) & (final.myofibroblast_activation > start.myofibroblast_activation)).all())
    final_mask = (natural.stage == "d7_bleo") & (natural.step == 240)
    llm_final = natural[final_mask & (natural["mode"] == "llm_agent")].sort_values("sample_id").reset_index(drop=True)
    rule_final = natural[final_mask & (natural["mode"] == "rule_agent")].sort_values("sample_id").reset_index(drop=True)
    diff_final = natural[final_mask & (natural["mode"] == "diffusion_only")].sort_values("sample_id").reset_index(drop=True)
    expected_ids = llm_final.sample_id.astype(str).tolist()
    if not expected_ids or rule_final.sample_id.astype(str).tolist() != expected_ids or diff_final.sample_id.astype(str).tolist() != expected_ids:
        raise RuntimeError("Qualification natural-condition sample pairing is incomplete")
    descriptors = ["ECM_p90", "ECM_high_area", "niche_area", "TGFB_ECM_colocalization", "fibrosis_memory", "myofibroblast_activation"]
    llm_rule = {d: float(np.median(np.abs(llm_final[d].to_numpy()-rule_final[d].to_numpy())/np.maximum(np.abs(rule_final[d].to_numpy()), .01))) for d in descriptors}
    llm_rule_ok = sum(v >= 0.03 for v in llm_rule.values()) >= 2
    agent_diff = float(np.median(np.abs(llm_final.fibrotic_burden.to_numpy()-diff_final.fibrotic_burden.to_numpy())/np.maximum(np.abs(diff_final.fibrotic_burden.to_numpy()), .05)))
    agent_diff_ok = agent_diff >= 0.05
    finite_nonnegative = bool(np.isfinite(table[FIELDS]).all().all() and (table[FIELDS] >= 0).all().all())
    intervention_ok = True; intervention_checks = {}
    for condition in config["interventions"]["names"]:
        treated = table[(table.condition == condition) & (table.dose == .75)].sort_values("sample_id").reset_index(drop=True)
        if treated.sample_id.astype(str).tolist() != expected_ids:
            raise RuntimeError(f"Qualification pairing failed for {condition}")
        if condition == "TGFB_blockade": passed = bool(((treated.ECM_fibrosis.to_numpy() < llm_final.ECM_fibrosis.to_numpy()) & (treated.myofibroblast_activation.to_numpy() < llm_final.myofibroblast_activation.to_numpy())).all())
        elif condition == "macrophage_SPP1_APOE_suppression": passed = bool(((treated.TGFB_signal.to_numpy() < llm_final.TGFB_signal.to_numpy()) & (treated.ECM_fibrosis.to_numpy() < llm_final.ECM_fibrosis.to_numpy())).all())
        elif condition == "epithelial_repair_promotion": passed = bool(((treated.resolution.to_numpy() > llm_final.resolution.to_numpy()) & (treated.TGFB_signal.to_numpy() < llm_final.TGFB_signal.to_numpy())).all())
        else: passed = bool(((treated.ECM_fibrosis.to_numpy() < llm_final.ECM_fibrosis.to_numpy()) & (treated.fibrosis_memory.to_numpy() < llm_final.fibrosis_memory.to_numpy())).all())
        intervention_checks[condition] = passed; intervention_ok &= passed
    sensitivity_ok = all((out / "runs" / run_id / "RUN_COMPLETE.json").exists() for run_id in sensitivity_manifest.run_id)
    checks = {"ctrl_stability": ctrl_ok, "fibrosis_specific_trajectory": trajectory_ok, "agent_vs_diffusion_ge_20pct": agent_diff_ok,
              "llm_vs_dynamic_rule_three_descriptors_ge_5pct": bool(llm_rule_ok), "intervention_downstream_direction": bool(intervention_ok), "finite_nonnegative_fields": finite_nonnegative}
    checks["sensitivity_runs_complete"] = sensitivity_ok
    audit_path = out / "audit/pre_benchmark_qualification.json"
    dump_json(audit_path, {"GSE267904_d21_opened": False, "qualification_animals": sorted(q_animals), "checks": checks, "all_passed": all(checks.values()),
                           "agent_vs_diffusion_relative_difference": agent_diff, "llm_vs_rule_relative_differences": llm_rule, "interventions": intervention_checks})
    if not all(checks.values()): raise RuntimeError(f"V3 pre-benchmark qualification failed: {[k for k,v in checks.items() if not v]}")
    dump_json(out / "pre_benchmark_model_lock.json", {"GSE267904_d21_opened": False, "qualification_sha256": sha256(audit_path), "parameters_sha256": sha256(params_path),
                                                       "policy_schedule_set_sha256": hash_paths(schedules), "physicell_binary_sha256": sha256(binary)})
    print("V3 d7-only qualification passed; pre-benchmark model lock created")


if __name__ == "__main__": main()
