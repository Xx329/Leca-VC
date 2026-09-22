#!/usr/bin/env python3
"""Create V3 paper-facing figures and reports after locked benchmark evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, dump_json, ensure_dirs, resolve_out
from evaluate_locked_benchmark import risk_score


COLORS = {"natural_progression": "#4c78a8", "TGFB_blockade": "#e45756", "macrophage_SPP1_APOE_suppression": "#54a24b", "epithelial_repair_promotion": "#f2cf5b", "fibroblast_ECM_suppression": "#b279a2"}
CELL_COLORS = ["#2563eb", "#e45756", "#f59e0b", "#54a24b", "#7c3aed", "#8b5e3c", "#06b6d4", "#ec4899", "#64748b", "#111827"]


def markdown(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = resolve_out(args.project_root.resolve(), args.out_dir)
    ensure_dirs(out)
    pred_path = out / "evaluation/locked_posthoc_spatial_benchmark.csv"
    effect_path = out / "evaluation/intervention_bootstrap_effects.csv"
    if not pred_path.exists() or not effect_path.exists():
        raise FileNotFoundError("Run evaluate-locked-benchmark before report")
    pred = pd.read_csv(pred_path)
    effects = pd.read_csv(effect_path)
    run_desc = pd.read_csv(out / "evaluation/run_level_descriptors.csv")
    real_desc = pd.read_csv(out / "evaluation/real_section_descriptors.csv")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = pred.prediction + "\n→" + pred.target
    ax.bar(labels, pred.energy_distance, color=["#9ca3af", "#f59e0b", "#54a24b", "#2563eb", "#10b981"])
    ax.set_ylabel("Registration-free energy distance")
    ax.set_title("Frozen V3 locked post-hoc d21 spatial benchmark")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "figures/fig_v3_locked_benchmark_energy_distance.png", dpi=240)
    plt.close(fig)

    niche = effects[effects.metric.eq("niche_area_fraction_effect")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for condition, group in niche.groupby("condition"):
        group = group.sort_values("dose")
        yerr = np.vstack([group.mean_effect - group.ci_low, group.ci_high - group.mean_effect])
        ax.errorbar(group.dose * 100, group.mean_effect, yerr=yerr, marker="o", linewidth=2, capsize=3, label=condition, color=COLORS.get(condition))
    ax.axhline(0, color="#111827", linewidth=.8)
    ax.set_xlabel("Abstract target engagement (%)")
    ax.set_ylabel("Fibrotic niche area reduction")
    ax.set_title("Cell-specific virtual intervention dose response")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures/fig_v3_intervention_dose_response.png", dpi=240)
    plt.close(fig)

    agent_files = sorted((out / "agents").glob("*_cell_agents.csv"))
    agents = pd.concat([pd.read_csv(p) for p in agent_files], ignore_index=True)
    cell_types = sorted(agents.cell_type.unique())
    color_by_type = dict(zip(cell_types, CELL_COLORS))
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    for ax, (sample_id, group) in zip(axes.flat, agents.groupby("sample_id", sort=True)):
        for cell_type, cells in group.groupby("cell_type"):
            ax.scatter(cells.x, cells.y, s=12 + 3 * np.sqrt(cells.represented_abundance.clip(lower=0, upper=100)),
                       color=color_by_type[cell_type], alpha=.78, edgecolors="white", linewidths=.25)
        ax.set_title(sample_id.split("_", 1)[-1], fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color_by_type[x], label=x, markersize=6) for x in cell_types]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.01), ncol=5, fontsize=7)
    fig.suptitle("50 cell-centric LLM agents per d7 section: identity, position, represented abundance", fontsize=12)
    fig.savefig(out / "figures/fig_v3_cell_agent_identity_and_positions.png", dpi=240)
    plt.close(fig)

    matrix = pd.read_csv(out / "manifests/physicell_418_run_matrix.csv")
    example = matrix[(matrix.stage.eq("d7_bleo")) & matrix.condition.eq("natural_progression") & matrix.seed.eq(1701)].iloc[0]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, step in zip(axes, [0, 60, 120, 240]):
        mesh = pd.read_csv(out / "runs" / example.run_id / f"mesh_fields_step_{step}.csv")
        tissue = mesh[mesh.tissue_mask.eq(1)]
        image = ax.scatter(tissue.x, tissue.y, c=risk_score(tissue), s=10, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"virtual progression step {step}")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=axes, label="fibrosis risk")
    fig.suptitle("Natural cell-agent progression in real PhysiCell/BioFVM")
    fig.savefig(out / "figures/fig_v3_natural_progression.png", dpi=240)
    plt.close(fig)

    ablation_rows = []
    for label, values in [
        ("static d7", real_desc[(real_desc.stage == "d7_bleo") & (real_desc.source == "static_d7")]),
        ("diffusion only", run_desc[run_desc.agent_mode == "diffusion_only"]),
        ("rule agents", run_desc[run_desc.agent_mode == "rule_agent"]),
        ("LLM cell agents", run_desc[(run_desc.agent_mode == "llm_agent") & (run_desc.condition == "natural_progression") & (run_desc.stage == "d7_bleo")]),
    ]:
        for metric in ["risk_mean", "ECM_mean"]:
            ablation_rows.append({"model": label, "metric": metric, "mean": values[metric].mean(), "sem": values[metric].sem()})
    ablation = pd.DataFrame(ablation_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ["risk_mean", "ECM_mean"]):
        z = ablation[ablation.metric.eq(metric)]
        ax.bar(z.model, z["mean"], yerr=z["sem"], color=["#9ca3af", "#f59e0b", "#54a24b", "#2563eb"], capsize=3)
        ax.set_title(metric); ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Static, diffusion-only, dynamic rule-agent and adaptive LLM-agent ablation at step 240")
    fig.savefig(out / "figures/fig_v3_executor_ablation.png", dpi=240)
    plt.close(fig)

    animal_effects = pd.read_csv(out / "evaluation/intervention_animal_effects.csv")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    rng = np.random.default_rng(1701)
    positions = {(condition, dose): i for i, (condition, dose) in enumerate(animal_effects[["condition", "dose"]].drop_duplicates().itertuples(index=False, name=None))}
    for (condition, dose), group in animal_effects.groupby(["condition", "dose"]):
        x = positions[(condition, dose)] + rng.uniform(-.12, .12, len(group))
        ax.scatter(x, group.ECM_mean_effect, color=COLORS[condition], alpha=.75, s=28)
    labels = [f"{condition.replace('_', ' ')}\n{int(dose*100)}%" for condition, dose in positions]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.axhline(0, color="#111827", linewidth=.8); ax.set_ylabel("Animal-level ECM burden reduction")
    ax.set_title("Paired intervention effects across d7 bleomycin animals")
    fig.tight_layout()
    fig.savefig(out / "figures/fig_v3_cross_animal_intervention_effects.png", dpi=240)
    plt.close(fig)

    sensitivity = pd.read_csv(out / "calibration/parameter_sensitivity_results.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for family, group in sensitivity.groupby("family"):
        group = group.sort_values("multiplier")
        ax.plot(group.multiplier, group.fibrotic_burden, marker="o", linewidth=2, label=family)
    ax.axvline(1, color="#111827", linewidth=.8, linestyle="--")
    ax.set_xlabel("Parameter-family multiplier"); ax.set_ylabel("Step-240 TGFB + ECM burden")
    ax.set_title("Pre-freeze +/-50% parameter sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures/fig_v3_parameter_sensitivity.png", dpi=240)
    plt.close(fig)

    english = f"""# GSE267904 Fibrosis Application V3

This is a cell-centric application experiment. Each of the 50 LLM agents per d7 section represents one cell type and decides only its own response. Real PhysiCell/BioFVM executes spatial progression and writeback.

Progression is labeled abstract PhysiCell progression time / virtual progression step. Step 240 is not real minutes and is not day21.

GSE267904 d21 was accessed only after model freezing and is reported as a locked post-hoc spatial benchmark, not as an unseen held-out dataset. Evaluation is animal-level and registration-free.

## Held-out prediction

{markdown(pred)}

## Intervention effects

{markdown(effects)}
"""
    chinese = f"""# GSE267904 肺纤维化 Application V3

V3坚持cell-centric agent语义：每张d7切片固定50个LLM agent，每个agent代表一种细胞身份，只决定自身反应；空间推进和substrate写回由真实PhysiCell/BioFVM执行。

step 0/60/120/240是abstract PhysiCell progression time，不对应真实分钟或day21。

GSE267904 d21只在模型冻结后读取，并明确称为locked post-hoc spatial benchmark，不声称是全新held-out数据。

## Held-out预测

{markdown(pred)}

## 干预效应

{markdown(effects)}
"""
    (out / "reports/gse267904_fibrosis_application_v3_report.md").write_text(english, encoding="utf-8")
    (out / "reports/gse267904_fibrosis_application_v3_summary_zh.md").write_text(chinese, encoding="utf-8")
    dump_json(out / "audit/report_audit.json", {"reports_written": True, "figures_written": 7, "required_figure_families_complete": True})
    print(f"Wrote V3 figures and reports to {out}")


if __name__ == "__main__":
    main()
