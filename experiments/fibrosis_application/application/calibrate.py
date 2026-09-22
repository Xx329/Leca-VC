#!/usr/bin/env python3
"""Create V3 pre-benchmark dynamic calibration and sensitivity contracts."""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, dump_json, ensure_dirs, hash_paths, load_config, read_json, resolve_out, sha256
from external_calibration import build_external_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--external-summary", type=Path, help="Preprocessed GSE141259/GSE264278 day<=14 field trajectory CSV")
    parser.add_argument("--download-external", action="store_true", help="Download only the four official GSE264278 day0/3/7/14 files")
    parser.add_argument("--allow-prior-only", action="store_true", help="Prepare code/smoke calibration only; cannot freeze the paper model")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    ensure_dirs(out)
    config = load_config()
    d7_files = sorted((out / "fields/d7").glob("*_spot_fields.csv.gz"))
    if len(d7_files) != 12:
        raise RuntimeError(f"Calibration requires all 12 d7 sections; found {len(d7_files)}")
    external = args.external_summary
    if external is None and not args.allow_prior_only:
        cached = out / "calibration/external_day0_to14_module_trajectories.csv"
        source_audit = read_json(out / "audit/external_calibration_sources.json", {})
        cache_valid = (
            cached.exists()
            and len(source_audit.get("sources", [])) == 8
            and all(Path(item["path"]).exists() for item in source_audit.get("sources", []))
        )
        if cache_valid:
            print(f"Reusing hash-validated external calibration summary {cached}", flush=True)
            external = cached
        else:
            external = build_external_summary(root, out, config, download=args.download_external)
    external_ok = False
    external_audit = None
    table = None
    if external:
        external = external.resolve()
        table = pd.read_csv(external)
        required = {"dataset", "day", "field", "relative_activity"}
        if not required.issubset(table.columns):
            raise RuntimeError(f"External calibration summary must have columns {sorted(required)}")
        forbidden = set(config["external_calibration"]["forbidden_days"])
        if set(pd.to_numeric(table["day"]).astype(int)) & forbidden:
            raise RuntimeError("External calibration includes forbidden day21-or-later observations")
        expected_datasets = {"GSE141259", "GSE264278"}
        if set(table["dataset"]) != expected_datasets:
            raise RuntimeError(f"Expected external datasets {expected_datasets}, found {set(table['dataset'])}")
        external_ok = True
        day_field_coverage = table.groupby("dataset")["day"].apply(lambda x: sorted(set(map(int, x)))).to_dict()
        expected_days = {"GSE141259": [0, 3, 7, 10, 14], "GSE264278": [0, 3, 7, 14]}
        if day_field_coverage != expected_days:
            raise RuntimeError(f"External calibration day coverage must be {expected_days}; found {day_field_coverage}")
        field_coverage = table.groupby("dataset")["field"].apply(lambda x: sorted(set(x))).to_dict()
        if any(set(fields) != set(config["module_genes"]) for fields in field_coverage.values()):
            raise RuntimeError(f"External calibration does not cover all six modules: {field_coverage}")
        canonical_external = out / "calibration/external_day0_to14_module_trajectories.csv"
        if external.resolve() != canonical_external.resolve():
            table.to_csv(canonical_external, index=False)
        source_manifest = read_json(out / "audit/external_calibration_sources.json", {})
        if source_manifest:
            source_manifest["summary_sha256"] = sha256(canonical_external)
            dump_json(out / "audit/external_calibration_sources.json", source_manifest)
        external_audit = {"source_path": str(external), "source_sha256": sha256(external), "path": str(canonical_external.resolve()), "sha256": sha256(canonical_external), "n_rows": len(table), "max_day": int(table.day.max()), "day_coverage": day_field_coverage}
    elif not args.allow_prior_only:
        raise FileNotFoundError(
            "External day<=14 calibration summary is required for paper-model freezing. "
            "Use --allow-prior-only only for compilation/smoke tests."
        )

    base = copy.deepcopy(config["substrates"])
    calibration_selection = {"method": "prior_only", "decay_multipliers": {}, "source_coupling_multiplier": 1.0}
    if external_ok and table is not None:
        trajectory = table.groupby(["field", "day"], as_index=False)["relative_activity"].median()
        trajectory.to_csv(out / "calibration/external_relative_trajectory_medians.csv", index=False)
        for field in base:
            values = trajectory[trajectory.field.eq(field)].set_index("day")["relative_activity"]
            early_peak = float(values.reindex([3, 7, 10]).dropna().max())
            persistence = float(values.loc[14] / max(early_peak, 1e-8))
            multiplier = 0.5 if persistence >= 0.75 else (1.5 if persistence <= 0.35 else 1.0)
            base[field]["decay"] *= multiplier
            calibration_selection["decay_multipliers"][field] = {
                "day14_to_early_peak": persistence, "selected_multiplier": multiplier,
            }
        risk_fields = ["injury_signal", "inflammatory_signal", "TGFB_signal", "ECM_fibrosis", "macrophage_APOE_SPP1_signal"]
        amplification = float(trajectory[trajectory.field.isin(risk_fields) & trajectory.day.isin([3, 7, 10])].groupby("field")["relative_activity"].max().median())
        source_multiplier = 1.5 if amplification >= 1.5 else (0.5 if amplification <= 0.8 else 1.0)
        calibration_selection.update({"method": "GSE141259_and_GSE264278_day0_to14_relative_dynamics", "early_risk_amplification": amplification, "source_coupling_multiplier": source_multiplier})
    else:
        source_multiplier = 1.0
    # External day14-to-early-peak persistence is high for these axes; enforce
    # conservative persistence caps before any GSE267904 d21 access.
    for field, cap in {"injury_signal": .0012, "inflammatory_signal": .0010, "TGFB_signal": .0005, "macrophage_APOE_SPP1_signal": .0005}.items():
        base[field]["decay"] = min(float(base[field]["decay"]), cap)
    for field, cap in {"inflammatory_signal": 600.0, "TGFB_signal": 400.0, "macrophage_APOE_SPP1_signal": 300.0}.items():
        base[field]["diffusion"] = min(float(base[field]["diffusion"]), cap)
    sensitivity_rows = []
    for family, multiplier in itertools.product(["diffusion", "decay", "source_coupling"], [0.5, 1.0, 1.5]):
        sensitivity_rows.append({"family": family, "multiplier": multiplier, "uses_d21": False})
    pd.DataFrame(sensitivity_rows).to_csv(out / "calibration/sensitivity_grid.csv", index=False)
    frozen_params = out / "calibration/dynamic_parameters_pre_benchmark.yaml"
    frozen_params.write_text(yaml.safe_dump({
        "progression_time_label": config["progression_time_label"],
        "substrates": base,
        "mechanism": {
            "kernel_rate": 0.002,
            "kernel_sigma": config["mesh"]["agent_kernel_sigma"],
            "kernel_radius": config["mesh"]["agent_kernel_radius"],
            "macrophage_source_injury_exponent": 2.0,
            "macrophage_source_multiplier": 2.0,
            "fibroblast_source_injury_exponent": 1.5,
            "field_saturation_target": 1.5,
            "memory_gain_rate": 0.006, "memory_resolution_rate": 0.0015,
            "epithelial_integrity_gain": 0.006, "epithelial_integrity_loss": 0.004,
            "macrophage_activation_rate": 0.007, "macrophage_resolution_rate": 0.0015,
            "myofibroblast_activation_rate": 0.006, "myofibroblast_resolution_rate": 0.0008,
            "injury_clearance_rate": 0.003, "inflammation_clearance_rate": 0.0015,
            "ecm_degradation_rate": 0.00008,
            "repair_intervention_multiplier": 3.0, "ecm_intervention_multiplier": 4.0,
            "abundance_growth_rate": 0.00015,
        },
        "d7_physicell_mechanism_pilot": {
            "selected_source_rate_before_external_multiplier": 0.1,
            "selected_fibrotic_source_multiplier": 10.0,
            "selection_data": "GSE267904 d7_ctrl stability and d7_bleo rule-agent versus diffusion-only; no d21 access",
            "selection_status": "requires automatic d7-only pilot revalidation before model freeze",
            "full_matrix_revalidation_required_before_model_freeze": True,
        },
        "selection_constraints": {
            "d7_ctrl_max_pathological_burden_increase": 0.10,
            "d7_ctrl_min_resolution_change": -0.10,
            "agent_vs_diffusion_min_burden_difference": 0.20,
            "not_pure_exponential_decay": True, "positive_control_direction_required": True,
        },
        "external_calibration": external_audit,
        "external_parameter_selection": calibration_selection,
        "paper_freeze_eligible": external_ok,
        "GSE267904_d21_used": False, "final_virtual_step": 240,
    }, sort_keys=False), encoding="utf-8")
    dump_json(out / "audit/calibration_audit.json", {
        "GSE267904_d21_opened": False, "external_day21_or_later_used": False,
        "external_calibration_complete": external_ok, "prior_only_smoke_mode": not external_ok,
        "kinetic_parameter_sha256": sha256(frozen_params), "d7_input_hash": hash_paths(d7_files),
    })
    print(f"Wrote V3 pre-benchmark dynamic calibration to {frozen_params}; freeze_eligible={external_ok}")


if __name__ == "__main__":
    main()
