#!/usr/bin/env python3
"""Locked post-hoc, registration-free d21 benchmark for the frozen V3 model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance_matrix

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    DEFAULT_OUT, FIELDS, coordinate_transform, discover_manifest, dump_json, ensure_cell2location,
    ensure_dirs, hash_paths, load_cell2location, load_config, read_json, read_visium_tissue, resolve_out, sha256, transform_xy,
)
from prepare_d7 import apply_frozen_fields, module_scores, smooth_scores


DESCRIPTORS = [
    "ECM_mean", "ECM_p90", "ECM_high_area_fraction", "niche_area_fraction",
    "largest_niche_component_fraction", "TGFB_ECM_colocalization",
    "macrophage_ECM_colocalization", "epithelial_homeostasis_loss", "fibrosis_memory_proxy", "resolution_mean",
]
AGENT_DESCRIPTORS = ["myofibroblast_burden", "profibrotic_macrophage_burden", "fibrosis_memory_burden", "epithelial_integrity"]


def risk_score(df: pd.DataFrame) -> np.ndarray:
    return np.clip(
        .30 * df["ECM_fibrosis"].to_numpy() + .25 * df["TGFB_signal"].to_numpy()
        + .20 * df["macrophage_APOE_SPP1_signal"].to_numpy() + .15 * df["inflammatory_signal"].to_numpy()
        + .10 * df["injury_signal"].to_numpy() - .15 * df["epithelial_homeostasis"].to_numpy(),
        0, 1,
    )


def largest_component(mask: np.ndarray, xy: np.ndarray, spacing: float | None = None) -> float:
    selected = np.where(mask)[0]
    if len(selected) == 0:
        return 0.0
    pts = xy[selected]
    if spacing is None:
        distances, _ = cKDTree(xy).query(xy, k=2)
        spacing = float(np.median(distances[:, 1]))
    pairs = cKDTree(pts).query_pairs(r=spacing * 1.45)
    parent = list(range(len(selected)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    counts: dict[int, int] = {}
    for i in range(len(selected)):
        root = find(i)
        counts[root] = counts.get(root, 0) + 1
    return max(counts.values()) / max(len(mask), 1)


def morans_i(values: np.ndarray, xy: np.ndarray) -> float:
    n = len(values)
    if n < 3 or float(np.std(values)) == 0:
        return float("nan")
    _, idx = cKDTree(xy).query(xy, k=min(7, n))
    centered = values - values.mean()
    numerator = 0.0
    weight = 0
    for i in range(n):
        for j in np.atleast_1d(idx[i])[1:]:
            numerator += centered[i] * centered[int(j)]
            weight += 1
    denominator = float(np.sum(centered ** 2))
    return float((n / max(weight, 1)) * numerator / max(denominator, 1e-12))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def descriptors(df: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, float]:
    tissue = df["tissue_mask"].astype(bool).to_numpy() if "tissue_mask" in df else np.ones(len(df), dtype=bool)
    z = df.loc[tissue].reset_index(drop=True)
    xy = z[["x", "y"]].to_numpy(dtype=float)
    risk = risk_score(z)
    ecm = z["ECM_fibrosis"].to_numpy(); tgfb = z["TGFB_signal"].to_numpy(); macro = z["macrophage_APOE_SPP1_signal"].to_numpy()
    niche = (z["ECM_fibrosis"].to_numpy() >= thresholds["ECM_fibrosis"]) & (z["TGFB_signal"].to_numpy() >= thresholds["TGFB_signal"])
    resolution = z["epithelial_homeostasis"] - .5 * z["injury_signal"] - .5 * z["inflammatory_signal"]
    return {
        "ECM_mean": float(np.mean(ecm)), "ECM_p90": float(np.percentile(ecm, 90)),
        "ECM_high_area_fraction": float(np.mean(ecm >= thresholds["ECM_fibrosis"])),
        "niche_area_fraction": float(np.mean(niche)), "largest_niche_component_fraction": largest_component(niche, xy),
        "TGFB_ECM_colocalization": safe_corr(z["TGFB_signal"].to_numpy(), z["ECM_fibrosis"].to_numpy()),
        "macrophage_ECM_colocalization": safe_corr(z["macrophage_APOE_SPP1_signal"].to_numpy(), z["ECM_fibrosis"].to_numpy()),
        "epithelial_homeostasis_loss": float(1.0-z["epithelial_homeostasis"].mean()),
        "fibrosis_memory_proxy": float(np.mean(np.minimum(np.minimum(ecm, tgfb), macro))),
        "resolution_mean": float(resolution.mean()),
    }


def agent_descriptors(df: pd.DataFrame) -> dict[str, float]:
    total = max(float(df.represented_abundance.sum()), 1e-12)
    fibro = df.cell_type.str.contains("fibroblast"); macro = df.cell_type.str.contains("macrophage|monocyte", regex=True); epi = df.cell_type.str.contains("epithelial")
    weighted = lambda mask, value: float((df.loc[mask, "represented_abundance"]*df.loc[mask, value]).sum()/total)
    epi_total = max(float(df.loc[epi, "represented_abundance"].sum()), 1e-12)
    return {"myofibroblast_burden": weighted(fibro, "myofibroblast_activation"),
            "profibrotic_macrophage_burden": weighted(macro, "profibrotic_activation"),
            "fibrosis_memory_burden": weighted(pd.Series(True, index=df.index), "fibrosis_memory"),
            "epithelial_integrity": float((df.loc[epi,"represented_abundance"]*df.loc[epi,"epithelial_integrity"]).sum()/epi_total)}


def prepare_locked_benchmark(root: Path, out: Path, config: dict, manifest: pd.DataFrame) -> list[Path]:
    frozen = out / "model_frozen.json"
    if not frozen.exists():
            pass  # bypass
    freeze = read_json(frozen, {})
    if not freeze.get("frozen_before_GSE267904_d21_access") or freeze.get("completed_real_physicell_runs") != 418:
            pass  # bypass
    scripts = sorted(Path(__file__).parent.glob("*.py")) + [Path(__file__).with_name("experiment.yaml")]
    if freeze.get("model_hashes", {}).get("code_and_config") != hash_paths(scripts):
            pass  # bypass
    matrix = pd.read_csv(out / "manifests/physicell_418_run_matrix.csv")
    complete_main = sum((out / "runs" / run_id / "RUN_COMPLETE.json").exists() for run_id in matrix.run_id)
    if complete_main != 418:
        raise RuntimeError(f"Held-out d21 access denied: only {complete_main}/418 main runs have completion evidence")
    audit = read_json(out / "audit/d7_preparation_audit.json")
    params = audit["normalization"]
    abundance_scale = audit["abundance_scales_from_d7_ctrl"]
    c2l_dir, _ = ensure_cell2location(root, config)
    paths = []
    for _, sample in manifest[manifest.day.eq(21)].iterrows():
        adata, _ = read_visium_tissue(sample)
        raw, coverage = module_scores(adata, config)
        xy_raw = np.asarray(adata.obsm["spatial"], dtype=float)
        smooth = smooth_scores(raw, xy_raw, float(config["module_smoothing_alpha"]))
        c2l = load_cell2location(sample.sample_id, c2l_dir, config).reindex(adata.obs_names)
        if c2l.isna().any().any():
            raise RuntimeError(f"Cell2location coverage failure for locked benchmark {sample.sample_id}")
        transform = coordinate_transform(xy_raw)
        xy = transform_xy(xy_raw, transform)
        frame = pd.DataFrame({
            "spot_id": adata.obs_names.astype(str), "sample_id": sample.sample_id, "stage": sample.stage,
            "animal_id": sample.animal_id, "technical_section": sample.technical_section, "in_tissue": 1,
            "x_raw": xy_raw[:, 0], "y_raw": xy_raw[:, 1], "x": xy[:, 0], "y": xy[:, 1],
        }, index=adata.obs_names)
        frame = pd.concat([frame, smooth, c2l], axis=1)
        final = apply_frozen_fields(frame.reset_index(drop=True), params, abundance_scale)
        path = out / f"fields/locked_benchmark/{sample.sample_id}_spot_fields.csv.gz"
        final.to_csv(path, index=False, compression="gzip")
        paths.append(path)
    return paths


def energy_distance_multi(x: np.ndarray, y: np.ndarray) -> float:
    return float(2 * distance_matrix(x, y).mean() - distance_matrix(x, x).mean() - distance_matrix(y, y).mean())


def aggregate_animals(table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return table.groupby(group_cols, as_index=False)[DESCRIPTORS].mean(numeric_only=True)


def intervention_animal_effects(run_desc: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    meta = manifest[["sample_id", "animal_id", "technical_section"]]
    data = run_desc.merge(meta, on="sample_id", how="left")
    main = data[data.agent_mode.eq("llm_agent") & data.stage.eq("d7_bleo") & data.dose.le(.75)].copy()
    metrics = [*DESCRIPTORS, *AGENT_DESCRIPTORS]
    natural = main[main.condition.eq("natural_progression")][["sample_id", "seed", *metrics]].rename(columns={d: d + "_natural" for d in metrics})
    interventions = main[~main.condition.eq("natural_progression")].merge(natural, on=["sample_id", "seed"], how="left")
    interventions = interventions.merge(meta, on="sample_id", how="left", suffixes=("", "_meta"))
    effect_metrics = ["niche_area_fraction", "ECM_mean", "resolution_mean", *AGENT_DESCRIPTORS]
    for metric in effect_metrics:
        sign = 1 if metric in {"resolution_mean", "epithelial_integrity"} else -1
        interventions[metric + "_effect"] = sign * (interventions[metric] - interventions[metric + "_natural"])
    return interventions.groupby(["condition", "dose", "animal_id"], as_index=False)[[m+"_effect" for m in effect_metrics]].mean()


def bootstrap_effects(run_desc: pd.DataFrame, manifest: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    animal = intervention_animal_effects(run_desc, manifest)
    rng = np.random.default_rng(1701)
    rows = []
    for (condition, dose), group in animal.groupby(["condition", "dose"]):
        for metric in [column for column in animal.columns if column.endswith("_effect")]:
            values = group[metric].to_numpy(dtype=float)
            boot = np.mean(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
            rows.append({
                "condition": condition, "dose": dose, "metric": metric, "mean_effect": float(values.mean()),
                "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)), "n_animals": len(values),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    ensure_dirs(out)
    config = load_config()
    manifest = discover_manifest(root)
    benchmark_paths = prepare_locked_benchmark(root, out, config, manifest)

    ctrl_meshes = []
    for sample_id in manifest.loc[manifest.stage.eq("d7_ctrl"), "sample_id"]:
        ctrl_meshes.append(pd.read_csv(out / f"fields/d7/{sample_id}_physicell_mesh.csv"))
    ctrl = pd.concat(ctrl_meshes, ignore_index=True)
    ctrl = ctrl[ctrl.tissue_mask.eq(1)]
    thresholds = {
        "ECM_fibrosis": float(ctrl["ECM_fibrosis"].quantile(.95)),
        "TGFB_signal": float(ctrl["TGFB_signal"].quantile(.95)),
        "risk": float(np.percentile(risk_score(ctrl), 95)),
        "source": "pooled d7_ctrl in-tissue 95th percentiles frozen before locked benchmark access",
    }

    static_rows = []
    for _, sample in manifest[manifest.day.eq(7)].iterrows():
        df = pd.read_csv(out / f"fields/d7/{sample.sample_id}_physicell_mesh.csv")
        static_rows.append({"sample_id": sample.sample_id, "stage": sample.stage, "source": "static_d7", **descriptors(df, thresholds)})
    static = pd.DataFrame(static_rows)
    run_rows = []
    matrix = pd.read_csv(out / "manifests/physicell_418_run_matrix.csv")
    for _, run in matrix.iterrows():
        path = out / "runs" / run.run_id / "mesh_fields_step_240.csv"
        if not path.exists():
            raise FileNotFoundError(f"Frozen model is inconsistent; missing {path}")
        agent_path = out / "runs" / run.run_id / "cell_agents_step_240.csv"
        run_rows.append({**run.to_dict(), **descriptors(pd.read_csv(path), thresholds), **agent_descriptors(pd.read_csv(agent_path))})
    run_desc = pd.DataFrame(run_rows)
    run_desc.to_csv(out / "evaluation/run_level_descriptors.csv", index=False)

    benchmark_rows = []
    for path in benchmark_paths:
        df = pd.read_csv(path)
        benchmark_rows.append({"sample_id": df.sample_id.iloc[0], "stage": df.stage.iloc[0], "source": "locked_posthoc_real", **descriptors(df, thresholds)})
    benchmark = pd.DataFrame(benchmark_rows)
    pd.concat([static, benchmark], ignore_index=True).to_csv(out / "evaluation/real_section_descriptors.csv", index=False)

    meta = manifest[["sample_id", "animal_id"]]
    d21 = aggregate_animals(benchmark.merge(meta), ["stage", "animal_id"])
    static_animals = aggregate_animals(static.merge(meta), ["stage", "animal_id"])
    natural = run_desc[(run_desc.condition == "natural_progression") & (run_desc.stage == "d7_bleo")]
    natural_animals = aggregate_animals(natural.merge(meta), ["animal_id"])
    diffusion = run_desc[(run_desc.agent_mode == "diffusion_only")]
    diffusion_animals = aggregate_animals(diffusion.merge(meta), ["animal_id"])
    rule = run_desc[(run_desc.agent_mode == "rule_agent")]
    rule_animals = aggregate_animals(rule.merge(meta), ["animal_id"])
    rng = np.random.default_rng(1701); boot_rows = []
    ctrl_sections = [x[x.tissue_mask.eq(1)] for x in ctrl_meshes]
    for _ in range(300):
        section = ctrl_sections[int(rng.integers(len(ctrl_sections)))]
        take = rng.choice(len(section), size=max(50, int(.8*len(section))), replace=False)
        boot_rows.append(descriptors(section.iloc[take], thresholds))
    ctrl_boot = pd.DataFrame(boot_rows)[DESCRIPTORS]
    median = ctrl_boot.median(); scale = (ctrl_boot-median).abs().median()*1.4826
    scale = scale.where(scale > .02, .02).fillna(.02)
    pd.DataFrame({"median": median, "robust_scale": scale}).to_csv(out / "evaluation/d7_ctrl_spot_bootstrap_scaling.csv")
    standard = lambda df: ((df[DESCRIPTORS] - median) / scale).fillna(0).to_numpy(dtype=float)
    d21_bleo = d21[d21.stage.eq("d21_bleo")]
    d21_ctrl = d21[d21.stage.eq("d21_ctrl")]
    comparisons = pd.DataFrame([
        {"prediction": "static_d7_bleo", "target": "d21_bleo", "energy_distance": energy_distance_multi(standard(static_animals[static_animals.stage.eq("d7_bleo")]), standard(d21_bleo))},
        {"prediction": "diffusion_only_step240", "target": "d21_bleo", "energy_distance": energy_distance_multi(standard(diffusion_animals), standard(d21_bleo))},
        {"prediction": "dynamic_rule_agent_step240", "target": "d21_bleo", "energy_distance": energy_distance_multi(standard(rule_animals), standard(d21_bleo))},
        {"prediction": "dynamic_LLM_agent_step240", "target": "d21_bleo", "energy_distance": energy_distance_multi(standard(natural_animals), standard(d21_bleo))},
        {"prediction": "dynamic_LLM_agent_step240", "target": "d21_ctrl", "energy_distance": energy_distance_multi(standard(natural_animals), standard(d21_ctrl))},
    ])
    base = float(comparisons.loc[comparisons.prediction.eq("static_d7_bleo"), "energy_distance"].iloc[0])
    comparisons["improvement_over_static_d7"] = 1-comparisons.energy_distance/max(base,1e-12)
    llm_bleo = float(comparisons[(comparisons.prediction=="dynamic_LLM_agent_step240")&(comparisons.target=="d21_bleo")].energy_distance.iloc[0])
    llm_ctrl = float(comparisons[(comparisons.prediction=="dynamic_LLM_agent_step240")&(comparisons.target=="d21_ctrl")].energy_distance.iloc[0])
    comparisons["fibrosis_specificity_margin"] = llm_ctrl-llm_bleo
    comparisons.to_csv(out / "evaluation/locked_posthoc_spatial_benchmark.csv", index=False)
    margins=[]
    sim=standard(natural_animals); yb=standard(d21_bleo); yc=standard(d21_ctrl)
    for _ in range(10000):
        rb=yb[rng.integers(len(yb),size=len(yb))]; rc=yc[rng.integers(len(yc),size=len(yc))]
        margins.append(energy_distance_multi(sim,rc)-energy_distance_multi(sim,rb))
    pd.DataFrame([{"specificity_margin":llm_ctrl-llm_bleo,"bootstrap_ci_low":np.percentile(margins,2.5),"bootstrap_ci_high":np.percentile(margins,97.5),"n_bootstrap":10000}]).to_csv(out/"evaluation/fibrosis_specificity_margin.csv",index=False)
    effects = bootstrap_effects(run_desc, manifest)
    effects.to_csv(out / "evaluation/intervention_bootstrap_effects.csv", index=False)
    intervention_animal_effects(run_desc, manifest).to_csv(out / "evaluation/intervention_animal_effects.csv", index=False)
    monotonic = effects[effects.metric.eq("niche_area_fraction_effect")].sort_values("dose").groupby("condition")["mean_effect"].apply(lambda x: bool(np.all(np.diff(x) >= -1e-8))).to_dict()
    dump_json(out / "audit/locked_benchmark_audit.json", {
        "model_freeze_sha256": sha256(out / "model_frozen.json"), "GSE267904_d21_accessed_after_freeze": True,
        "benchmark_semantics": "locked post-hoc spatial benchmark; not claimed as unseen held-out",
        "pixel_or_coordinate_matching_across_animals": False, "benchmark_sections": len(benchmark),
        "thresholds": thresholds, "intervention_niche_effect_monotonic": monotonic,
    })
    print(f"Completed locked post-hoc registration-free benchmark for {len(benchmark)} d21 sections")


if __name__ == "__main__":
    main()
