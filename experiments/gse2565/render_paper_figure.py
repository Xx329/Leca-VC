#!/usr/bin/env python3
"""Render the current A--G GSE2565 template with newly rebuilt Leca-VC rows."""
from __future__ import annotations
import hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
V52 = Path(os.environ["GSE2565_V52_OUT"]).resolve()
OUT = Path(os.environ["GSE2565_PAPER_FIGURE_OUT"]).resolve()
BASE_SCRIPT = ROOT / "figures/main/fig2/base_plot.py"
METHOD = "AgentVC Online Agent pilot"

def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else np.zeros_like(values)

def dtw(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float); right = np.asarray(right, float)
    matrix = np.full((len(left) + 1, len(right) + 1), np.inf); matrix[0, 0] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            matrix[i, j] = abs(left[i - 1] - right[j - 1]) + min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
    return float(matrix[-1, -1])

def formal_dnb_metrics() -> dict[str, float]:
    real_path = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v2_1/real_agent_comparison/real_dnb_components.csv"
    agent_path = Path(os.environ["LECAVC_OUTPUT_ROOT"]) / "pilot/amend004_dnb_components.csv"
    real = pd.read_csv(real_path)
    real = real.loc[real.real_processing.eq("biological_mouse_level_sensitivity") & real.estimable].sort_values("timepoint")
    agent = pd.read_csv(agent_path)
    agent = agent.loc[agent.estimable].sort_values("timepoint")
    common = sorted(set(real.timepoint.astype(float)) & set(agent.timepoint.astype(float)))
    if common != [4.0, 8.0, 12.0, 24.0, 48.0, 72.0]:
        raise RuntimeError(f"formal DNB common time grid changed: {common}")
    real_values = real.set_index("timepoint").loc[common, "composite_index"].to_numpy(float)
    agent_values = agent.set_index("timepoint").loc[common, "composite_index"].to_numpy(float)
    real_curve = minmax(real_values); agent_curve = minmax(agent_values)
    real_peak = common[int(np.argmax(real_curve))]; agent_peak = common[int(np.argmax(agent_curve))]
    result = {
        "dnb_peak_error": float(abs(agent_peak - real_peak)),
        "dnb_pearson": float(pearsonr(real_curve, agent_curve).statistic),
        "dnb_dtw": dtw(real_curve, agent_curve),
        "real_peak_hours": float(real_peak),
        "leca_vc_peak_hours": float(agent_peak),
        "formal_gene_space": 11171,
    }
    return result

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_base():
    spec = importlib.util.spec_from_file_location("gse2565_deidentified_paper_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def main() -> None:
    profiles_path = V52 / "plot_input_tables/panels_A_to_D_trajectories.csv"
    metrics_path = V52 / "plot_input_tables/panel_F_raw_values.csv"
    profiles = pd.read_csv(profiles_path); metrics = pd.read_csv(metrics_path)
    profiles = profiles[~profiles.method.eq("RVAgene")].copy()
    metrics = metrics[~metrics.method.eq("RVAgene")].copy()
    if profiles.shape != (45, 8) or metrics.shape[0] != 32:
        raise RuntimeError(f"unexpected paper source shapes: {profiles.shape}, {metrics.shape}")
    formal = formal_dnb_metrics()
    # The final manuscript keeps common-space timing and formal-space
    # Leca-VC profile agreement. Record those scopes rather than conflating them.
    real_common = profiles.loc[profiles.method.eq("Real")].dropna(subset=["normalized_DNB"])
    observed_peak = float(real_common.loc[real_common.normalized_DNB.idxmax(), "time_hours"])
    for method in profiles.method.unique():
        if method == "Real":
            continue
        curve = profiles.loc[profiles.method.eq(method)].dropna(subset=["normalized_DNB"])
        peak = float(curve.loc[curve.normalized_DNB.idxmax(), "time_hours"])
        mask = metrics.method.eq(method) & metrics.metric_key.eq("dnb_peak_error")
        if int(mask.sum()) != 1:
            raise RuntimeError(f"Missing common-space peak-error row for {method}")
        metrics.loc[mask, "raw_value"] = abs(peak - observed_peak)
    for key in ("dnb_pearson", "dnb_dtw"):
        mask = metrics.method.eq(METHOD) & metrics.metric_key.eq(key)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"missing unique Leca-VC metric row: {key}")
        metrics.loc[mask, "raw_value"] = formal[key]
    base = load_base()
    base.OUT = OUT; base.SOURCE_OUT = OUT / "source_data"
    base.FILE_STEM = "GSE2565_bulk_time_resolved_transcriptomic_response_benchmark_deidentified_v1"
    base.STATUS = "PASS_GSE2565_DEIDENTIFIED_PAPER_FIGURE_V1"
    base.OVERALL_TITLE = "GSE2565 bulk time-resolved transcriptomic response benchmark"
    base.OVERALL_SUBTITLE = ""
    base.DISPLAY_NAMES[METHOD] = "Leca-VC"
    base.PANEL_SPECS = [
        ("A", "DNB peak timing", "normalized_DNB"),
        ("B", "Expression distribution shift over time", "normalized_distribution_shift"),
        ("C", "Transcriptomic deviation from baseline over time", "normalized_progression"),
        ("D", "Transcriptomic change rate over time", "normalized_velocity"),
    ]
    base.GROUP_SPECS = [
        ("E. DNB timing and temporal-profile accuracy", "F1", ["dnb_peak_error", "dnb_pearson", "dnb_dtw"]),
        ("F. Late-time global transcriptomic agreement", "F2", ["distribution_pearson", "progression_pearson", "velocity_pearson"]),
        ("G. Late-time functional and gene-direction agreement", "F3", ["mean_module_pearson", "gene_direction_agreement"]),
    ]
    OUT.mkdir(parents=True, exist_ok=True); base.SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    ep = profiles.copy(); ep.insert(0, "source_method_id", ep.method); ep["method"] = ep.method.map(base.DISPLAY_NAMES)
    em = metrics.copy(); em.insert(0, "source_method_id", em.method); em["method"] = em.method.map(base.DISPLAY_NAMES)
    ep.to_csv(base.SOURCE_OUT / "panels_A_to_D_time_resolved_profiles.csv", index=False)
    em.to_csv(base.SOURCE_OUT / "panels_E_to_G_raw_metrics.csv", index=False)
    (base.SOURCE_OUT / "leca_vc_formal_fullspace_dnb_metrics.json").write_text(
        json.dumps(formal, indent=2) + "\n", encoding="utf-8"
    )
    paths = base.render(profiles, metrics)
    (OUT / "caption.txt").write_text(
        (ROOT / "figures/main/fig2/caption.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest = {
        "status": base.STATUS, "template": "current A-G layout", "external_baselines_reused": True,
        "source_hashes": {str(profiles_path): sha256(profiles_path), str(metrics_path): sha256(metrics_path)},
        "formal_fullspace_dnb": formal,
        "displayed_peak_error_scope": "common 1601-gene curves relative to common-space Observed peak",
        "leca_vc_profile_metric_scope": "formal 11171-gene curves",
        "outputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
