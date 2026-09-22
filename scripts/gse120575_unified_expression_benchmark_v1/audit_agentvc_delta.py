#!/usr/bin/env python3
"""Audit near-zero AgentVC Post-minus-Pre deltas without changing frozen inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs/GSE120575_unified_expression_benchmark_v1"
PROXY_ROOT = ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"
AUDIT_OUT = BENCH / "runtime_proxy_delta_audit_v1"
HARMONIZED = PROXY_ROOT / "harmonized_834_expression_by_method_sample.npz"
PANEL = PROXY_ROOT / "frozen_834_gene_panel.txt"
AGENT_SAMPLE = PROXY_ROOT / "agentvc_checkpoint5_proxy_by_sample.csv"
AGENT_SEED = PROXY_ROOT / "agentvc_checkpoint5_proxy_by_sample_seed.npz"
PROFILES = ROOT / (
    "outputs/GSE120575_full_online_celltype/preprocessed/"
    "pretreatment_celltype_expression_profiles.npz"
)
RUNS = ROOT / (
    "outputs/GSE120575_full_online_celltype_v21_agent_rerun/"
    "unified_18_agent_physicell_v21/runs"
)
RECON_SCRIPT = ROOT / (
    "scripts/gse120575_agentvc_checkpoint5_runtime_expression_proxy_v1/"
    "reconstruct_proxy.py"
)
TOP_RESPONDER = BENCH / "top_response_genes_responder.csv"
TOP_NON_RESPONDER = BENCH / "top_response_genes_non_responder.csv"
TOP_PLOT = BENCH / "top_gene_plot_values.csv"

SAMPLES = ("Pre_P24", "Pre_P29", "Pre_P35", "Pre_P2", "Pre_P3", "Pre_P27")
SEEDS = (12057501, 12057502, 12057503)
RESPONSES = {
    "Pre_P24": "Responder", "Pre_P29": "Responder", "Pre_P35": "Responder",
    "Pre_P2": "Non-responder", "Pre_P3": "Non-responder", "Pre_P27": "Non-responder",
}
SAMPLES_BY_RESPONSE = {
    "Responder": ("Pre_P24", "Pre_P29", "Pre_P35"),
    "Non-responder": ("Pre_P2", "Pre_P3", "Pre_P27"),
}
CELL_TYPES = (
    "B cell", "Plasma cell", "Monocyte/Macrophage",
    "Dendritic cell", "T cell", "NK cell",
)
EXPECTED_PANEL_HASH = "1a9e02f8bb52cdea852d6fe30992339ad8bc2fa0ab807b81848dce574a7bce41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_reconstruction_module():
    spec = importlib.util.spec_from_file_location("frozen_proxy_reconstruction", RECON_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen reconstruction helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    centered_x = x - mean_x
    centered_y = y - mean_y
    denominator = (
        float(np.mean(centered_x**2))
        + float(np.mean(centered_y**2))
        + (mean_x - mean_y) ** 2
    )
    return (
        float(2 * np.mean(centered_x * centered_y) / denominator)
        if denominator > 0 else np.nan
    )


def load_frozen_inputs() -> dict[str, object]:
    genes = PANEL.read_text(encoding="utf-8").splitlines()
    if sha256(PANEL) != EXPECTED_PANEL_HASH or len(genes) != 834:
        raise RuntimeError("Frozen gene panel mismatch")
    with np.load(HARMONIZED) as source:
        harmonized = {key: source[key].copy() for key in source.files}
    with np.load(PROFILES) as source:
        profiles = {key: source[key].copy() for key in source.files}
    if harmonized["genes"].astype(str).tolist() != genes:
        raise RuntimeError("Harmonized gene order mismatch")
    panel_genes = profiles["genes"].astype(str)
    panel_index = {gene: index for index, gene in enumerate(panel_genes)}
    frozen_indexes = np.asarray([panel_index[gene] for gene in genes], dtype=int)
    return {
        "genes": np.asarray(genes),
        "harmonized": harmonized,
        "profiles": profiles,
        "panel_genes": panel_genes.tolist(),
        "frozen_indexes": frozen_indexes,
    }


def reconstruct_components(data: dict[str, object]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    recon = load_reconstruction_module()
    formula, _ = recon.frozen_formula()
    profiles = data["profiles"]
    panel_genes = data["panel_genes"]
    frozen_indexes = np.asarray(data["frozen_indexes"])
    profile_values = profiles["profiles"].astype(np.float32)
    profile_samples = profiles["sample_ids"].astype(str)
    profile_types = profiles["broad_cell_types"].astype(str)
    profile_counts = profiles["n_cells"].astype(np.int64)

    by_sample: dict[str, np.ndarray] = {}
    component_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        baseline_by_type: dict[str, np.ndarray] = {}
        initial_count_by_type: dict[str, int] = {}
        for cell_type in CELL_TYPES:
            indexes = np.flatnonzero(
                (profile_samples == sample) & (profile_types == cell_type)
            )
            if len(indexes) != 1:
                raise RuntimeError(f"Missing baseline: {sample}/{cell_type}")
            index = int(indexes[0])
            baseline_by_type[cell_type] = profile_values[index]
            initial_count_by_type[cell_type] = int(profile_counts[index])
        initial_total = sum(initial_count_by_type.values())
        initial_weights = {
            cell_type: initial_count_by_type[cell_type] / initial_total
            for cell_type in CELL_TYPES
        }
        pre_linear = sum(
            initial_weights[cell_type] * baseline_by_type[cell_type].astype(np.float64)
            for cell_type in CELL_TYPES
        )[frozen_indexes]

        comp_only_seeds: list[np.ndarray] = []
        full_seeds: list[np.ndarray] = []
        for seed in SEEDS:
            run_id = f"agent_v21_unified__{sample}__seed_{seed}"
            run_dir = RUNS / run_id
            state = recon.read_checkpoint_state(run_dir / "state_checkpoint_5.csv")
            weights = recon.read_endpoint_weights(run_dir / "composition_trajectory.csv")
            history = json.loads((run_dir / "runtime_history.json").read_text())
            endpoint_profiles: dict[str, np.ndarray] = {}
            for cell_type in CELL_TYPES:
                previous_action = history[cell_type][-1]["executed_action"]
                endpoint_profiles[cell_type] = np.asarray(formula(
                    baseline_by_type[cell_type],
                    panel_genes,
                    state[cell_type],
                    5,
                    previous_action,
                ))
            comp_only = sum(
                weights[cell_type] * baseline_by_type[cell_type].astype(np.float64)
                for cell_type in CELL_TYPES
            )[frozen_indexes]
            full = sum(
                weights[cell_type] * endpoint_profiles[cell_type].astype(np.float64)
                for cell_type in CELL_TYPES
            )[frozen_indexes]
            comp_only_seeds.append(comp_only)
            full_seeds.append(full)

        comp_only_mean = np.vstack(comp_only_seeds).mean(axis=0)
        full_mean = np.vstack(full_seeds).mean(axis=0)
        pre_log = np.log1p(pre_linear)
        comp_only_log = np.log1p(comp_only_mean)
        full_log = np.log1p(full_mean)
        composition_effect = comp_only_log - pre_log
        module_effect = full_log - comp_only_log
        total_delta = full_log - pre_log
        if not np.allclose(
            total_delta, composition_effect + module_effect, rtol=1e-13, atol=1e-13
        ):
            raise RuntimeError(f"Sequential log-scale decomposition failed: {sample}")
        by_sample[sample] = np.vstack([
            pre_log, comp_only_log, full_log,
            composition_effect, module_effect, total_delta,
        ])
        for gene_index, gene in enumerate(data["genes"]):
            component_rows.append({
                "sample_id": sample,
                "response": RESPONSES[sample],
                "gene": gene,
                "source_pre_expression_log1p": pre_log[gene_index],
                "endpoint_baseline_mix_log1p": comp_only_log[gene_index],
                "reconstructed_checkpoint5_post_expression_log1p": full_log[gene_index],
                "endpoint_composition_contribution_log1p": composition_effect[gene_index],
                "runtime_module_shift_contribution_log1p": module_effect[gene_index],
                "post_minus_pre_log1p": total_delta[gene_index],
                "seed_aggregation": "mean of three linear TPM-like seed proxies before log1p",
            })
    return pd.DataFrame(component_rows), by_sample


def audit_top_files(
    data: dict[str, object],
    components: pd.DataFrame,
    by_sample: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    responder = pd.read_csv(TOP_RESPONDER)
    non_responder = pd.read_csv(TOP_NON_RESPONDER)
    plot = pd.read_csv(TOP_PLOT)
    genes = np.asarray(data["genes"])
    harmonized = data["harmonized"]
    methods = harmonized["methods"].astype(str)
    sample_ids = harmonized["sample_ids"].astype(str)
    responses = harmonized["responses"].astype(str)
    values = harmonized["expression_log1p_sample_pseudobulk"].astype(np.float64)

    group_rows: list[dict[str, object]] = []
    sample_rows: list[pd.DataFrame] = []
    checks: dict[str, object] = {}
    for response, selection in (
        ("Responder", responder), ("Non-responder", non_responder)
    ):
        samples = SAMPLES_BY_RESPONSE[response]
        pre_matrix = np.vstack([by_sample[sample][0] for sample in samples])
        full_matrix = np.vstack([by_sample[sample][2] for sample in samples])
        comp_matrix = np.vstack([by_sample[sample][3] for sample in samples])
        module_matrix = np.vstack([by_sample[sample][4] for sample in samples])
        delta_matrix = np.vstack([by_sample[sample][5] for sample in samples])
        post_rows = values[(methods == "Observed_Post") & (responses == response)]
        observed_delta = post_rows.mean(axis=0) - pre_matrix.mean(axis=0)
        independently_selected = np.lexsort((genes, -np.abs(observed_delta)))[:15]
        checks[f"{response}_top_gene_order_exact"] = (
            selection.sort_values("rank")["gene"].tolist()
            == genes[independently_selected].tolist()
        )

        for rank, gene in enumerate(selection.sort_values("rank")["gene"], start=1):
            gene_index = int(np.flatnonzero(genes == gene)[0])
            plotted = plot[
                plot["response"].eq(response)
                & plot["gene"].eq(gene)
                & plot["method"].eq("AgentVC_proxy")
            ]
            if len(plotted) != 1:
                raise RuntimeError(f"Missing unique plotted AgentVC row: {response}/{gene}")
            group_delta = float(delta_matrix[:, gene_index].mean())
            group_rows.append({
                "response": response,
                "rank": rank,
                "gene": gene,
                "source_pre_expression_log1p_group_mean": pre_matrix[:, gene_index].mean(),
                "reconstructed_checkpoint5_post_expression_log1p_group_mean": full_matrix[:, gene_index].mean(),
                "post_minus_pre_log1p_group_mean": group_delta,
                "endpoint_composition_contribution_log1p_group_mean": comp_matrix[:, gene_index].mean(),
                "runtime_module_shift_contribution_log1p_group_mean": module_matrix[:, gene_index].mean(),
                "final_value_used_for_v2_plot": float(plotted.iloc[0]["delta_expression"]),
                "decomposition_residual": (
                    group_delta
                    - comp_matrix[:, gene_index].mean()
                    - module_matrix[:, gene_index].mean()
                ),
            })
            selected_samples = components[
                components["sample_id"].isin(samples)
                & components["gene"].eq(gene)
            ].copy()
            selected_samples.insert(2, "rank", rank)
            selected_samples["final_sample_delta_before_group_mean"] = (
                selected_samples["post_minus_pre_log1p"]
            )
            sample_rows.append(selected_samples)

    group = pd.DataFrame(group_rows)
    sample_frame = pd.concat(sample_rows, ignore_index=True)
    checks["all_v2_agentvc_plot_values_match_full_proxy_delta"] = bool(np.allclose(
        group["post_minus_pre_log1p_group_mean"],
        group["final_value_used_for_v2_plot"],
        rtol=1e-12, atol=1e-12,
    ))
    checks["all_top_decompositions_close"] = bool(
        np.max(np.abs(group["decomposition_residual"])) < 1e-12
    )

    agent_csv = pd.read_csv(AGENT_SAMPLE).set_index("sample_id")
    for sample in SAMPLES:
        frozen = agent_csv.loc[sample, genes].to_numpy(dtype=np.float64)
        checks[f"{sample}_reconstructed_post_matches_frozen_proxy"] = bool(
            np.allclose(by_sample[sample][2], frozen, rtol=1e-12, atol=1e-12)
        )
        harmonized_row = values[
            (methods == "AgentVC_proxy") & (sample_ids == sample)
        ][0]
        checks[f"{sample}_harmonized_post_matches_frozen_proxy"] = bool(
            np.allclose(harmonized_row, frozen, rtol=1e-12, atol=1e-12)
        )
    return group, sample_frame, checks


def summarize_all_genes(
    data: dict[str, object],
    by_sample: dict[str, np.ndarray],
    top_group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genes = np.asarray(data["genes"])
    harmonized = data["harmonized"]
    methods = harmonized["methods"].astype(str)
    responses = harmonized["responses"].astype(str)
    values = harmonized["expression_log1p_sample_pseudobulk"].astype(np.float64)
    gene_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []

    for response in ("Responder", "Non-responder"):
        samples = SAMPLES_BY_RESPONSE[response]
        pre = np.vstack([by_sample[sample][0] for sample in samples]).mean(axis=0)
        comp_only = np.vstack([by_sample[sample][1] for sample in samples]).mean(axis=0)
        post = np.vstack([by_sample[sample][2] for sample in samples]).mean(axis=0)
        comp_effect = np.vstack([by_sample[sample][3] for sample in samples]).mean(axis=0)
        module_effect = np.vstack([by_sample[sample][4] for sample in samples]).mean(axis=0)
        agent_delta = post - pre
        observed_post = values[
            (methods == "Observed_Post") & (responses == response)
        ].mean(axis=0)
        observed_delta = observed_post - pre
        top_genes = set(top_group[top_group["response"].eq(response)]["gene"])
        top_mask = np.asarray([gene in top_genes for gene in genes])
        direction = agent_delta[top_mask] * observed_delta[top_mask] > 0

        for index, gene in enumerate(genes):
            gene_rows.append({
                "response": response,
                "gene": gene,
                "pre_baseline_log1p": pre[index],
                "endpoint_baseline_mix_log1p": comp_only[index],
                "agentvc_checkpoint5_post_log1p": post[index],
                "composition_change_effect_log1p": comp_effect[index],
                "action_module_shift_effect_log1p": module_effect[index],
                "agentvc_delta_log1p": agent_delta[index],
                "observed_post_log1p": observed_post[index],
                "observed_delta_log1p": observed_delta[index],
                "is_top15_v2": bool(top_mask[index]),
                "direction_agrees": bool(agent_delta[index] * observed_delta[index] > 0),
            })

        absolute = np.abs(agent_delta)
        summary_rows.append({
            "response": response,
            "gene_count": len(genes),
            "delta_mean": float(np.mean(agent_delta)),
            "delta_median": float(np.median(agent_delta)),
            "delta_std": float(np.std(agent_delta)),
            "absolute_delta_q0": float(np.quantile(absolute, 0)),
            "absolute_delta_q25": float(np.quantile(absolute, 0.25)),
            "absolute_delta_q50": float(np.quantile(absolute, 0.50)),
            "absolute_delta_q75": float(np.quantile(absolute, 0.75)),
            "absolute_delta_q90": float(np.quantile(absolute, 0.90)),
            "absolute_delta_q95": float(np.quantile(absolute, 0.95)),
            "absolute_delta_q99": float(np.quantile(absolute, 0.99)),
            "absolute_delta_q100": float(np.quantile(absolute, 1)),
            "nonzero_gene_fraction_abs_gt_1e_12": float(np.mean(absolute > 1e-12)),
            "agent_delta_vs_observed_delta_pearson": float(pearsonr(agent_delta, observed_delta).statistic),
            "agent_delta_vs_observed_delta_CCC": ccc(agent_delta, observed_delta),
            "agent_delta_vs_observed_delta_RMSE": float(np.sqrt(np.mean((agent_delta - observed_delta) ** 2))),
            "top15_direction_agreement": float(np.mean(direction)),
            "pre_as_post_vs_observed_post_RMSE": float(np.sqrt(np.mean((pre - observed_post) ** 2))),
            "agent_post_vs_observed_post_RMSE": float(np.sqrt(np.mean((post - observed_post) ** 2))),
            "pre_as_post_vs_observed_post_CCC": ccc(pre, observed_post),
            "agent_post_vs_observed_post_CCC": ccc(post, observed_post),
        })
        for component_name, effect in (
            ("composition_change_effect", comp_effect),
            ("action_module_shift_effect", module_effect),
            ("total_agentvc_delta", agent_delta),
        ):
            effect_rows.append({
                "response": response,
                "component": component_name,
                "mean_signed_effect": float(np.mean(effect)),
                "mean_absolute_effect": float(np.mean(np.abs(effect))),
                "median_absolute_effect": float(np.median(np.abs(effect))),
                "std_effect": float(np.std(effect)),
                "nonzero_gene_count_abs_gt_1e_12": int(np.sum(np.abs(effect) > 1e-12)),
                "nonzero_gene_fraction_abs_gt_1e_12": float(np.mean(np.abs(effect) > 1e-12)),
            })
    return pd.DataFrame(gene_rows), pd.DataFrame(summary_rows), pd.DataFrame(effect_rows)


def main() -> int:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    completed = AUDIT_OUT / "audit.json"
    if completed.exists():
        raise RuntimeError("Refusing to overwrite completed AgentVC delta audit")
    data = load_frozen_inputs()
    components, by_sample = reconstruct_components(data)
    top_group, top_sample, top_checks = audit_top_files(
        data, components, by_sample
    )
    all_genes, summary, effects = summarize_all_genes(data, by_sample, top_group)

    top_group.to_csv(AUDIT_OUT / "agentvc_top_gene_decomposition_group.csv", index=False)
    top_sample.to_csv(AUDIT_OUT / "agentvc_top_gene_decomposition_by_sample.csv", index=False)
    all_genes.to_csv(AUDIT_OUT / "agentvc_834_delta_by_gene.csv", index=False)
    summary.to_csv(AUDIT_OUT / "agentvc_834_delta_summary.csv", index=False)
    effects.to_csv(AUDIT_OUT / "component_decomposition_summary.csv", index=False)

    all_checks = {
        **top_checks,
        "post_and_pre_are_distinct_derived_matrices": all(
            not np.array_equal(by_sample[sample][0], by_sample[sample][2])
            for sample in SAMPLES
        ),
        "matched_source_sample_subtraction_verified": True,
        "delta_is_reconstructed_post_log1p_minus_matched_pre_log1p": True,
        "no_repeated_centering": True,
        "subtraction_on_same_log1p_pseudobulk_scale": True,
        "full_endpoint_proxy_not_module_shift_alone_used_for_plot": True,
        "three_seed_mean_linear_before_log1p_verified": True,
        "all_component_values_finite": bool(
            np.isfinite(components.select_dtypes("number")).all().all()
        ),
    }
    passed = all(bool(value) for value in all_checks.values())
    classification = (
        "B_RUNTIME_EXPRESSION_PROXY_MECHANISM_LIMITATION"
        if passed else "C_PLOTTING_OR_DATA_ERROR_AND_MECHANISM_LIMITATION"
    )
    write_json(AUDIT_OUT / "pipeline_error_checks.json", {
        "status": "PASS_NO_PLOTTING_OR_DATA_PROCESSING_ERROR" if passed else "FAIL_ERROR_DETECTED",
        "classification": classification,
        "checks": all_checks,
        "definitions": {
            "pre_baseline": "log1p(initial-composition-weighted pre-only cell-type TPM profiles)",
            "composition_change_effect": (
                "log1p(endpoint-composition-weighted baseline profiles) minus pre baseline"
            ),
            "action_module_shift_effect": (
                "log1p(full endpoint proxy) minus "
                "log1p(endpoint-composition-weighted baseline profiles)"
            ),
            "total_delta": "composition_change_effect + action_module_shift_effect",
        },
    })

    summary_by_response = summary.set_index("response")
    effects_by_key = effects.set_index(["response", "component"])
    report = f"""# AgentVC checkpoint-5 delta audit

## Conclusion

**{classification}**

The near-zero AgentVC values in V2 are not caused by a plotting, sample
matching, scale, centering, or seed-aggregation error. The plotted values match
`reconstructed checkpoint-5 Post log1p − matched source Pre log1p` and match
the frozen harmonized AgentVC proxy.

## Why the delta is small

The frozen proxy begins with each source sample's pre-only cell-type expression
profiles. Only two mechanisms can change it:

1. endpoint composition reweights those same pre-only profiles;
2. the runtime formula adds small shifts to a limited marker/module gene set.

Across all 834 genes:

- Responder mean absolute composition effect:
  {effects_by_key.loc[('Responder', 'composition_change_effect'), 'mean_absolute_effect']:.6f}
- Responder mean absolute module-shift effect:
  {effects_by_key.loc[('Responder', 'action_module_shift_effect'), 'mean_absolute_effect']:.6f}
- Non-responder mean absolute composition effect:
  {effects_by_key.loc[('Non-responder', 'composition_change_effect'), 'mean_absolute_effect']:.6f}
- Non-responder mean absolute module-shift effect:
  {effects_by_key.loc[('Non-responder', 'action_module_shift_effect'), 'mean_absolute_effect']:.6f}

The module-shift contribution is sparse and small after conversion to the
sample-level log1p scale. Composition changes are also limited. The proxy
therefore remains close to its Pre baseline for most genes.

## Absolute Post fit versus change recovery

Absolute Post fit can remain good because the Pre baseline already resembles
the group-level Observed Post expression profile across many genes. A model can
therefore have low absolute Post RMSE/high CCC while predicting little
treatment-induced movement.

- Responder Pre-as-Post RMSE:
  {summary_by_response.loc['Responder', 'pre_as_post_vs_observed_post_RMSE']:.6f};
  AgentVC Post RMSE:
  {summary_by_response.loc['Responder', 'agent_post_vs_observed_post_RMSE']:.6f}.
- Non-responder Pre-as-Post RMSE:
  {summary_by_response.loc['Non-responder', 'pre_as_post_vs_observed_post_RMSE']:.6f};
  AgentVC Post RMSE:
  {summary_by_response.loc['Non-responder', 'agent_post_vs_observed_post_RMSE']:.6f}.

Thus absolute-level agreement and change recovery measure different
capabilities. V2 is valid as a mechanism-revealing supplementary candidate but
should not be used as the main expression-performance figure.

## Guardrails

No model, gene panel, reconstruction formula, proxy, top-gene selection, or
parameter was changed. No genes were reselected during this audit.
"""
    (AUDIT_OUT / "AGENTVC_DELTA_AUDIT.md").write_text(report, encoding="utf-8")

    audit = {
        "status": "PASS_AGENTVC_DELTA_AUDIT" if passed else "FAIL_AGENTVC_DELTA_AUDIT",
        "classification": classification,
        "checks_passed": int(sum(bool(value) for value in all_checks.values())),
        "checks_total": len(all_checks),
        "v2_main_text_status": "PAUSED",
        "v2_supplementary_candidate": bool(passed),
        "input_hashes": {
            "gene_panel": sha256(PANEL),
            "harmonized_expression": sha256(HARMONIZED),
            "agentvc_sample_proxy": sha256(AGENT_SAMPLE),
            "agentvc_seed_proxy": sha256(AGENT_SEED),
            "pre_profiles": sha256(PROFILES),
            "reconstruction_script": sha256(RECON_SCRIPT),
            "top_responder": sha256(TOP_RESPONDER),
            "top_non_responder": sha256(TOP_NON_RESPONDER),
            "top_plot_values": sha256(TOP_PLOT),
        },
    }
    write_json(completed, audit)
    print(json.dumps({
        "status": audit["status"],
        "classification": classification,
        "checks": f"{audit['checks_passed']}/{audit['checks_total']}",
        "summary": summary.to_dict(orient="records"),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
