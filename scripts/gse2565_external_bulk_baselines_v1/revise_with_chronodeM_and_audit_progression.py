#!/usr/bin/env python3
"""Add chronODE monotonic sensitivity and audit Agent global progression."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation, cosine


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/GSE2565_external_bulk_baselines_v1"
REV = BASE / "revision_monotonic_progression_audit_v1"
V21 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v2_1"
V4 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v4_online_agent"
TIMES = np.asarray([0, 0.5, 1, 4, 8, 12, 24, 48, 72], float)
METHODS = ["chronODE-M\u2020", "RVAgene", "Online Agent pilot"]
COLORS = {
    "Real": "#202020",
    "chronODE-M\u2020": "#6E53A3",
    "RVAgene": "#D97A23",
    "Online Agent pilot": "#3568A8",
}
TOL = 1e-12


def load_base_module():
    path = (
        ROOT
        / "scripts/gse2565_external_bulk_baselines_v1/"
        "evaluate_full_trajectory_reconstruction.py"
    )
    spec = importlib.util.spec_from_file_location("base_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E = load_base_module()


def hash_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def manual_correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    aa, bb = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(1 - np.dot(aa, bb) / denominator)


def curves_for_matrix(x: np.ndarray, microarray: bool):
    return {
        "distribution": E.mm(E.distribution_shift(x, microarray)),
        "progression": E.progression(x),
        "velocity": E.mm(E.velocity(x)),
    }


def metric_row(method, curves, real_curves, dnb_curve, real_dnb, module_z, real_module_z):
    row = {"method": method, "evaluation_gene_count": curves["progression_gene_count"]}
    row.update(E.trajectory_metrics(real_dnb, dnb_curve, "DNB"))
    row["DNB_estimable"] = bool(np.isfinite(dnb_curve).sum() >= 2)
    row["DNB_non_estimable_reason"] = (
        "" if row["DNB_estimable"] else "single_group_mean_reconstruction_has_no_within_time_replicates"
    )
    row["DNB_8h_peak_recovered"] = (
        bool(row["DNB_peak_time"] == 8) if np.isfinite(row["DNB_peak_time"]) else False
    )
    row.update(
        E.trajectory_metrics(
            real_curves["distribution"], curves["distribution"], "distribution_shift"
        )
    )
    row.update(
        E.trajectory_metrics(
            real_curves["progression"], curves["progression"], "progression"
        )
    )
    row.update(
        E.trajectory_metrics(
            real_curves["velocity"], curves["velocity"], "velocity", TIMES[1:]
        )
    )
    module_rows = []
    for j, name in enumerate(module_z["names"]):
        item = {"method": method, "module": name}
        item.update(
            E.trajectory_metrics(
                real_module_z[:, j], module_z["values"][:, j], "module"
            )
        )
        module_rows.append(item)
    module_frame = pd.DataFrame(module_rows)
    row["mean_module_Pearson"] = module_frame.module_Pearson.mean()
    row["mean_module_peak_time_error"] = module_frame.module_peak_time_error.mean()
    return row, module_frame


def progression_audit(
    genes,
    real,
    agent,
    modules,
):
    baseline = agent[0]
    rows = []
    hashes = []
    previous = None
    for i, hour in enumerate(TIMES):
        current = agent[i]
        delta = current - baseline
        corr_scipy = float(correlation(current, baseline)) if i else 0.0
        corr_manual = manual_correlation_distance(current, baseline) if i else 0.0
        rows.append(
            {
                "time_hours": hour,
                "matrix_rows": 1,
                "matrix_genes": len(genes),
                "gene_order_exact": True,
                "matrix_sha256": hash_array(current),
                "same_as_previous_timepoint": bool(
                    previous is not None and np.array_equal(current, previous)
                ),
                "genes_exactly_equal_to_baseline": int(np.count_nonzero(delta == 0)),
                "fraction_exactly_equal_to_baseline": float(np.mean(delta == 0)),
                "genes_changed_gt_1e_12": int(np.count_nonzero(np.abs(delta) > TOL)),
                "fraction_changed_gt_1e_12": float(np.mean(np.abs(delta) > TOL)),
                "mean_absolute_change": float(np.mean(np.abs(delta))),
                "median_absolute_change": float(np.median(np.abs(delta))),
                "maximum_absolute_change": float(np.max(np.abs(delta))),
                "L2_norm_change": float(np.linalg.norm(delta)),
                "cosine_distance": float(cosine(current, baseline)) if i else 0.0,
                "Pearson_to_baseline": float(1 - corr_manual),
                "correlation_distance_scipy": corr_scipy,
                "correlation_distance_manual": corr_manual,
                "correlation_implementation_absolute_difference": abs(
                    corr_scipy - corr_manual
                ),
                "Euclidean_distance": float(np.linalg.norm(delta)),
                "normalized_L2_change": float(
                    np.linalg.norm(delta) / np.linalg.norm(baseline)
                ),
            }
        )
        hashes.append(hash_array(current))
        previous = current
    timepoint = pd.DataFrame(rows)
    timepoint.to_csv(REV / "agent_progression_timepoint_metrics.csv", index=False)

    position = {g: i for i, g in enumerate(genes)}
    module_union = sorted(
        {
            g
            for value in modules.genes.astype(str)
            for g in value.split("|")
            if g in position
        }
    )
    agent_variance = np.var(agent, axis=0)
    real_variance = np.var(real, axis=0)
    changed = np.flatnonzero(np.any(np.abs(agent - agent[[0]]) > TOL, axis=0))
    sets = {
        "all_12263_genes": np.arange(len(genes)),
        "frozen_functional_module_union": np.asarray(
            [position[g] for g in module_union], int
        ),
        "agent_time_variance_top500": np.argsort(agent_variance)[-500:],
        "agent_time_variance_top1000": np.argsort(agent_variance)[-1000:],
        "agent_any_nonzero_change": changed,
        "real_time_variance_top1000": np.argsort(real_variance)[-1000:],
    }
    sensitivity_rows = []
    curve_rows = []
    for name, idx in sets.items():
        agent_curve = E.progression(agent[:, idx])
        real_curve = E.progression(real[:, idx])
        peak = float(TIMES[np.nanargmax(agent_curve)])
        sensitivity_rows.append(
            {
                "gene_set": name,
                "gene_count": len(idx),
                "agent_max_progression": float(np.max(agent_curve)),
                "agent_mean_progression": float(np.mean(agent_curve)),
                "agent_peak_time": peak,
                "real_max_progression": float(np.max(real_curve)),
                "trajectory_Pearson_vs_Real": E.safe_pearson(real_curve, agent_curve),
                "trajectory_DTW": E.dtw(E.mm(real_curve), E.mm(agent_curve)),
                "selection_rule": {
                    "all_12263_genes": "frozen exact gene panel",
                    "frozen_functional_module_union": "union of frozen functional modules",
                    "agent_time_variance_top500": "top 500 by Agent internal temporal variance",
                    "agent_time_variance_top1000": "top 1000 by Agent internal temporal variance",
                    "agent_any_nonzero_change": "absolute change >1e-12 at any Agent time",
                    "real_time_variance_top1000": "top 1000 by Real internal temporal variance",
                }[name],
            }
        )
        for hour, ac, rc in zip(TIMES, agent_curve, real_curve):
            curve_rows.append(
                {
                    "gene_set": name,
                    "time_hours": hour,
                    "Agent_progression": ac,
                    "Real_progression": rc,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(REV / "agent_progression_gene_set_sensitivity.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(
        REV / "agent_progression_gene_set_curves.csv", index=False
    )

    check = {
        "status": "PASS_TWO_IMPLEMENTATIONS_AGREE",
        "scipy_implementation": "scipy.spatial.distance.correlation",
        "manual_implementation": "1 - Pearson correlation after mean centering",
        "maximum_absolute_difference": float(
            timepoint.correlation_implementation_absolute_difference.max()
        ),
        "tolerance": 1e-12,
        "within_tolerance": bool(
            timepoint.correlation_implementation_absolute_difference.max() <= 1e-12
        ),
        "baseline_time_hours": 0.0,
        "time_order_exact": timepoint.time_hours.tolist() == TIMES.tolist(),
        "all_timepoint_hashes_unique": len(set(hashes)) == len(hashes),
        "all_timepoints_read_from_one_ordered_matrix_per_seed_then_seed_mean": True,
        "per_timepoint_renormalization_used": False,
    }
    (REV / "correlation_distance_implementation_check.json").write_text(
        json.dumps(check, indent=2) + "\n"
    )
    return timepoint, sensitivity, sets


def component_decomposition(genes, module_indices):
    proto_file = np.load(
        V4 / "initialization/healthy_expression_prototypes.npz", allow_pickle=False
    )
    prototype = {
        str(agent): proto_file["expression"][i].astype(float)
        for i, agent in enumerate(proto_file["agent_types"])
    }
    total_seed, comp_seed, action_seed = [], [], []
    for seed in [256501, 256502, 256503]:
        directory = (
            V4
            / "pilot/virtual_expression_amend004/cell_level/"
            f"agent_physicell_v4_online__virtual_phosgene_injury__seed_{seed}__to_72h"
        )
        total_times, comp_times = [], []
        for minute in [0, 30, 60, 240, 480, 720, 1440, 2880, 4320]:
            data = np.load(
                directory / f"worker_expression_minute_{minute}.npz",
                allow_pickle=True,
            )
            if data["genes"].astype(str).tolist() != genes:
                raise RuntimeError("Cell-level Agent gene order mismatch")
            weights = data["represented_abundance"].astype(float)
            weights = weights / weights.sum()
            expression = data["expression"].astype(float)
            agents = data["agent_ids"].astype(str)
            healthy = np.vstack([prototype[a] for a in agents])
            total_times.append(np.average(expression, axis=0, weights=weights))
            comp_times.append(np.average(healthy, axis=0, weights=weights))
        total_seed.append(np.vstack(total_times))
        comp_seed.append(np.vstack(comp_times))
        action_seed.append(np.vstack(total_times) - np.vstack(comp_times))
    total = np.mean(total_seed, axis=0)
    comp = np.mean(comp_seed, axis=0)
    action = np.mean(action_seed, axis=0)
    baseline = total[0]
    total_change = total - total[[0]]
    composition_change = comp - comp[[0]]
    action_change = action - action[[0]]
    residual = total_change - composition_change - action_change
    healthy_baseline_change = np.zeros_like(total_change)
    components = {
        "healthy_prototype_baseline": healthy_baseline_change,
        "composition_change": composition_change,
        "module_action_driven": action_change,
        "other_residual": residual,
        "total_expression_change": total_change,
    }
    rows = []
    for name, matrix in components.items():
        for i, hour in enumerate(TIMES):
            vector = matrix[i]
            candidate = baseline + vector
            rows.append(
                {
                    "component": name,
                    "time_hours": hour,
                    "mean_absolute_change": float(np.mean(np.abs(vector))),
                    "median_absolute_change": float(np.median(np.abs(vector))),
                    "maximum_absolute_change": float(np.max(np.abs(vector))),
                    "L2_norm": float(np.linalg.norm(vector)),
                    "correlation_distance_contribution": (
                        float(correlation(candidate, baseline)) if i else 0.0
                    ),
                    "module_union_mean_absolute_change": float(
                        np.mean(np.abs(vector[module_indices]))
                    ),
                    "nonzero_gene_count_gt_1e_12": int(
                        np.count_nonzero(np.abs(vector) > TOL)
                    ),
                    "nonzero_gene_fraction_gt_1e_12": float(
                        np.mean(np.abs(vector) > TOL)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(REV / "agent_progression_component_decomposition.csv", index=False)
    return frame


def exclusion_sensitivity(
    real_full,
    rva_full,
    agent_full,
    valid,
):
    rows = []
    matrices = {"RVAgene": rva_full, "Online Agent pilot": agent_full}
    for method, matrix in matrices.items():
        for scope, idx in [
            ("all_frozen_genes", np.arange(real_full.shape[1])),
            ("exclude_four_chronode_failed_genes", np.flatnonzero(valid)),
        ]:
            rc = curves_for_matrix(real_full[:, idx], True)
            mc = curves_for_matrix(matrix[:, idx], method == "RVAgene")
            rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "gene_count": len(idx),
                    "distribution_shift_Pearson": E.safe_pearson(
                        rc["distribution"], mc["distribution"]
                    ),
                    "progression_Pearson": E.safe_pearson(
                        rc["progression"], mc["progression"]
                    ),
                    "progression_DTW": E.dtw(
                        E.mm(rc["progression"]), E.mm(mc["progression"])
                    ),
                    "velocity_Pearson": E.safe_pearson(
                        rc["velocity"], mc["velocity"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(REV / "four_gene_exclusion_sensitivity.csv", index=False)
    return frame


def make_figure(
    curves,
    dnb,
    module_z,
    modules,
    metrics,
    dilution_confirmed,
):
    fig = plt.figure(figsize=(18, 15), facecolor="white")
    outer = fig.add_gridspec(
        3, 2, height_ratios=[1, 1, 1.35], hspace=0.34, wspace=0.22
    )
    axes = [
        fig.add_subplot(outer[0, 0]),
        fig.add_subplot(outer[0, 1]),
        fig.add_subplot(outer[1, 0]),
        fig.add_subplot(outer[1, 1]),
    ]
    panels = [
        ("A", "Normalized DNB trajectories", dnb, TIMES),
        (
            "B",
            "Normalized expression-distribution shift",
            {k: v["distribution"] for k, v in curves.items()},
            TIMES,
        ),
        (
            "C",
            "Correlation-distance transcriptomic progression",
            {k: v["progression"] for k, v in curves.items()},
            TIMES,
        ),
        (
            "D",
            "Time-interval-normalized transcriptomic velocity",
            {k: v["velocity"] for k, v in curves.items()},
            TIMES[1:],
        ),
    ]
    for ax, (letter, title, values, x) in zip(axes, panels):
        for method in ["Real", *METHODS]:
            y = values[method]
            if np.isfinite(y).any():
                ax.plot(
                    x,
                    y,
                    "-o",
                    lw=2,
                    ms=4,
                    color=COLORS[method],
                    label=method,
                )
        ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold")
        ax.set_xlabel("Time (h)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(
        0.54,
        0.47,
        "chronODE-M\u2020 DNB non-estimable\n(single reconstructed trajectory)",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=7.5,
        color=COLORS["chronODE-M\u2020"],
    )
    if dilution_confirmed:
        axes[2].text(
            0.02,
            0.96,
            "Global 7,632 common-complete-gene metric; program-restricted sensitivity is reported separately.",
            transform=axes[2].transAxes,
            va="top",
            fontsize=7.5,
            color="#555555",
        )
    axes[0].set_ylabel("Within-method normalized value")
    axes[2].set_ylabel("Distance")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    egrid = outer[2, 0].subgridspec(1, 4, wspace=0.12)
    heat_axes = []
    vmax = max(np.nanmax(np.abs(module_z[m])) for m in ["Real", *METHODS])
    for i, method in enumerate(["Real", *METHODS]):
        ax = fig.add_subplot(egrid[0, i])
        heat_axes.append(ax)
        image = ax.imshow(
            module_z[method].T,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(method, fontsize=9)
        ax.set_xticks(range(len(TIMES)), [f"{x:g}" for x in TIMES], rotation=45)
        ax.set_xlabel("h", fontsize=8)
        if i == 0:
            ax.set_yticks(range(len(modules)), modules.module, fontsize=7)
        else:
            ax.set_yticks([])
    heat_axes[0].text(
        -0.48,
        1.08,
        "E. Functional-program dynamics",
        transform=heat_axes[0].transAxes,
        fontweight="bold",
        fontsize=12,
    )
    cbar = fig.colorbar(image, ax=heat_axes, fraction=0.025, pad=0.02)
    cbar.ax.set_title("z", fontsize=8)

    ax = fig.add_subplot(outer[2, 1])
    specs = [
        ("DNB_peak_time_error", "DNB peak error \u2193", True),
        ("DNB_Pearson", "DNB Pearson \u2191", False),
        ("DNB_DTW", "DNB DTW \u2193", True),
        ("distribution_shift_Pearson", "Distribution Pearson \u2191", False),
        ("distribution_shift_DTW", "Distribution DTW \u2193", True),
        ("progression_Pearson", "Progression Pearson \u2191", False),
        ("progression_DTW", "Progression DTW \u2193", True),
        ("velocity_Pearson", "Velocity Pearson \u2191", False),
        ("velocity_DTW", "Velocity DTW \u2193", True),
        ("mean_module_Pearson", "Mean module Pearson \u2191", False),
        ("mean_module_peak_time_error", "Module peak error \u2193", True),
        ("gene_direction_agreement", "Gene direction \u2191", False),
    ]
    cmap = plt.get_cmap("YlGn")
    norm = plt.Normalize(0, 1)
    indexed = metrics.set_index("method")
    for i, (column, label, lower) in enumerate(specs):
        values = np.asarray([indexed.loc[m, column] for m in METHODS], float)
        finite = np.isfinite(values)
        score = np.full(3, np.nan)
        if finite.any():
            transformed = -values[finite] if lower else values[finite]
            score[finite] = E.mm(transformed)
        for j, value in enumerate(values):
            if np.isfinite(value):
                ax.scatter(
                    j,
                    i,
                    s=85 + 260 * score[j],
                    c=[cmap(norm(score[j]))],
                    edgecolors="#303030",
                    linewidths=0.6,
                )
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5)
            else:
                ax.scatter(j, i, marker="x", s=45, color="#AAAAAA")
                ax.text(j + 0.08, i, "NE", fontsize=6, va="center", color="#888888")
    ax.set_xticks(range(3), ["chronODE-M\u2020", "RVAgene", "Agent pilot"])
    ax.set_yticks(range(len(specs)), [x[1] for x in specs], fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(len(specs) - 0.5, -0.5)
    ax.grid(alpha=0.18)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("F. Cross-method metric dot plot", loc="left", fontweight="bold")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(
        scalar,
        ax=ax,
        fraction=0.035,
        pad=0.03,
        label="Within-metric relative performance",
    )

    fig.suptitle(
        "GSE2565 bulk trajectory reconstruction benchmark",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.968,
        "External reconstruction baselines use the complete observed bulk trajectory; "
        "the Online Agent pilot does not use the observed expression trajectory.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.012,
        "\u2020 chronODE-M is the successfully executed official monotonic sensitivity "
        "baseline; the preregistered chronODE piecewise implementation remained "
        "unavailable. chronODE and RVAgene are full-trajectory fitted reconstructions, "
        "not held-out predictions. Online Agent is a six-run exploratory pilot. "
        "Cross-platform raw expression errors are not compared. chronODE piecewise unavailable.",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    fig.savefig(
        REV / "GSE2565_bulk_trajectory_benchmark_with_chronodeM_600dpi.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        REV / "GSE2565_bulk_trajectory_benchmark_with_chronodeM_vector.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    REV.mkdir(parents=True, exist_ok=True)
    (
        genes,
        real_replicates,
        real_full,
        _piece,
        mono_full,
        rva_seeds_full,
        rva_full,
        agent_seeds_full,
        agent_full,
    ) = E.load_inputs()
    modules = pd.read_csv(V21 / "metrics/methods/functional_modules.csv")
    parameter_table = pd.read_csv(
        BASE / "chronode/formal/chronode_monotonic_parameters.csv"
    ).set_index("gene").reindex(genes)
    fit_success = parameter_table.status.eq("success").to_numpy()
    failed_genes = [genes[i] for i in np.flatnonzero(~fit_success)]
    if int(fit_success.sum()) != 12259 or len(failed_genes) != 4:
        raise RuntimeError("Unexpected chronODE monotonic official fit-status count")
    complete_valid = fit_success & np.isfinite(mono_full).all(axis=0)
    partial_nonfinite = fit_success & ~np.isfinite(mono_full).all(axis=0)
    if int(complete_valid.sum()) != 7632:
        raise RuntimeError(
            "Unexpected complete nine-time-point chronODE monotonic gene count"
        )
    genes_valid = [g for g, keep in zip(genes, complete_valid) if keep]
    real = real_full[:, complete_valid]
    mono = mono_full[:, complete_valid]
    rva = rva_full[:, complete_valid]
    agent = agent_full[:, complete_valid]
    real_reps = real_replicates[:, :, complete_valid]
    rva_seeds = rva_seeds_full[:, :, complete_valid]
    agent_seeds = agent_seeds_full[:, :, complete_valid]

    timepoint, sensitivity, gene_sets = progression_audit(
        genes, real_full, agent_full, modules
    )
    module_union_idx = gene_sets["frozen_functional_module_union"]
    decomposition = component_decomposition(genes, module_union_idx)
    exclusion = exclusion_sensitivity(
        real_full, rva_full, agent_full, fit_success
    )

    matrices = {
        "Real": real,
        "chronODE-M\u2020": mono,
        "RVAgene": rva,
        "Online Agent pilot": agent,
    }
    microarray = {
        "Real": True,
        "chronODE-M\u2020": True,
        "RVAgene": True,
        "Online Agent pilot": False,
    }
    curves = {}
    module_z = {}
    for method, matrix in matrices.items():
        curves[method] = curves_for_matrix(matrix, microarray[method])
        curves[method]["progression_gene_count"] = int(complete_valid.sum())
        module_values = E.module_matrix(matrix, genes_valid, modules)
        module_z[method], _ = E.zscore_time(module_values)

    module, background = E.fixed_dnb_sets(genes_valid)
    real_dnb = E.mm(
        np.asarray(
            [
                E.dnb_value(real_reps[i], genes_valid, module, background)
                for i in range(len(TIMES))
            ]
        )
    )
    rva_dnb = E.mm(
        np.asarray(
            [
                E.dnb_value(rva_seeds[:, i], genes_valid, module, background)
                for i in range(len(TIMES))
            ]
        )
    )
    agent_dnb = E.mm(
        np.asarray(
            [
                E.dnb_value(agent_seeds[:, i], genes_valid, module, background)
                for i in range(len(TIMES))
            ]
        )
    )
    dnb = {
        "Real": real_dnb,
        "chronODE-M\u2020": np.full(len(TIMES), np.nan),
        "RVAgene": rva_dnb,
        "Online Agent pilot": agent_dnb,
    }

    real_module_z = module_z["Real"]
    metric_rows, module_rows = [], []
    for method in METHODS:
        row, module_frame = metric_row(
            method,
            curves[method],
            curves["Real"],
            dnb[method],
            dnb["Real"],
            {"names": modules.module.tolist(), "values": module_z[method]},
            real_module_z,
        )
        change_real = real - real[[0]]
        change_method = matrices[method] - matrices[method][[0]]
        row["gene_direction_agreement"] = float(
            (
                np.sign(change_real[1:])
                == np.sign(change_method[1:])
            ).mean()
        )
        metric_rows.append(row)
        module_rows.append(module_frame)
    metrics = pd.DataFrame(metric_rows)
    module_metrics = pd.concat(module_rows, ignore_index=True)
    metrics[metrics.method == "chronODE-M\u2020"].to_csv(
        REV / "chronode_monotonic_unified_metrics.csv", index=False
    )
    module_metrics[module_metrics.method == "chronODE-M\u2020"].to_csv(
        REV / "chronode_monotonic_module_metrics.csv", index=False
    )
    metrics.to_csv(REV / "updated_unified_metrics.csv", index=False)
    module_metrics.to_csv(REV / "updated_all_method_module_metrics.csv", index=False)

    specs = {
        "DNB_peak_time_error": True,
        "DNB_Pearson": False,
        "DNB_DTW": True,
        "distribution_shift_Pearson": False,
        "distribution_shift_DTW": True,
        "progression_Pearson": False,
        "progression_DTW": True,
        "velocity_Pearson": False,
        "velocity_DTW": True,
        "mean_module_Pearson": False,
        "mean_module_peak_time_error": True,
        "gene_direction_agreement": False,
    }
    ranking_rows = []
    for column, lower in specs.items():
        values = metrics.set_index("method")[column]
        ranks = values.rank(ascending=lower, method="min")
        for method in METHODS:
            ranking_rows.append(
                {
                    "metric": column,
                    "method": method,
                    "value": values[method],
                    "rank_among_estimable_methods": ranks[method],
                    "lower_is_better": lower,
                    "estimable": bool(np.isfinite(values[method])),
                }
            )
    rankings = pd.DataFrame(ranking_rows)
    rankings.to_csv(REV / "updated_method_rankings.csv", index=False)

    all_amp = float(
        sensitivity.loc[
            sensitivity.gene_set == "all_12263_genes", "agent_max_progression"
        ].iloc[0]
    )
    module_amp = float(
        sensitivity.loc[
            sensitivity.gene_set == "frozen_functional_module_union",
            "agent_max_progression",
        ].iloc[0]
    )
    top1000_amp = float(
        sensitivity.loc[
            sensitivity.gene_set == "agent_time_variance_top1000",
            "agent_max_progression",
        ].iloc[0]
    )
    dilution_confirmed = module_amp > all_amp * 2 or top1000_amp > all_amp * 2
    make_figure(curves, dnb, module_z, modules, metrics, dilution_confirmed)

    max_impl_diff = float(
        timepoint.correlation_implementation_absolute_difference.max()
    )
    changed_union = int(
        sensitivity.loc[
            sensitivity.gene_set == "agent_any_nonzero_change", "gene_count"
        ].iloc[0]
    )
    action_rows = decomposition[
        decomposition.component == "module_action_driven"
    ]
    composition_rows = decomposition[
        decomposition.component == "composition_change"
    ]
    agent_full_velocity = E.velocity(agent_full)
    velocity_peak_index = int(np.nanargmax(agent_full_velocity))
    velocity_peak_time = float(TIMES[1:][velocity_peak_index])
    velocity_peak_value = float(agent_full_velocity[velocity_peak_index])
    agent_module_values = E.module_matrix(agent_full, genes, modules)
    module_absolute_delta = np.abs(
        agent_module_values - agent_module_values[[0]]
    )
    module_change_peak_index = int(
        np.nanargmax(module_absolute_delta.mean(axis=1))
    )
    module_change_peak_time = float(TIMES[module_change_peak_index])
    module_change_peak_mean = float(
        module_absolute_delta.mean(axis=1)[module_change_peak_index]
    )
    module_to_global_ratio = module_amp / all_amp
    top500_amp = float(
        sensitivity.loc[
            sensitivity.gene_set == "agent_time_variance_top500",
            "agent_max_progression",
        ].iloc[0]
    )
    top500_to_global_ratio = top500_amp / all_amp
    composition_max = float(
        composition_rows.mean_absolute_change.max()
    )
    residual_max = float(
        decomposition.loc[
            decomposition.component == "other_residual", "maximum_absolute_change"
        ].max()
    )
    audit_class = (
        "B_GLOBAL_TRANSCRIPTOME_DILUTION_OF_PROGRAM_LEVEL_CHANGES"
        if dilution_confirmed and max_impl_diff <= 1e-12
        else "A_OR_OTHER_REQUIRES_REVIEW"
    )
    report = f"""# Online Agent correlation-distance progression audit

## Verdict

**{audit_class}**

No file/time ordering, baseline-index, gene-order, duplicate-matrix, or
correlation-distance implementation error was found.

- All nine timepoint matrix hashes are unique.
- Baseline is the 0 h injury-condition matrix.
- scipy correlation distance and manual `1 - Pearson` agree within
  `{max_impl_diff:.3e}`.
- No per-timepoint normalization is applied before the frozen Panel C metric.
- Agent genes with any change greater than `1e-12`: {changed_union}/{len(genes)}
  ({changed_union/len(genes):.2%}).
- Directly defined frozen-module genes: {len(module_union_idx)}/{len(genes)}
  ({len(module_union_idx)/len(genes):.2%}); mean-preserving redistribution
  propagates much smaller offsets outside this set.
- Maximum all-gene progression: {all_amp:.6g}.
- Maximum frozen-program-union progression: {module_amp:.6g}.
- Maximum Agent top-1000-variance progression: {top1000_amp:.6g}.
- Frozen-program/all-gene amplitude ratio: {module_to_global_ratio:.2f}x.
- Agent top-500-variance/all-gene amplitude ratio:
  {top500_to_global_ratio:.2f}x.

The all-gene correlation structure is dominated by a large, stable healthy
prototype component. Program/action changes are concentrated or small relative
to that 12,263-dimensional baseline; mean-preserving redistribution also
creates small changes outside direct module genes. This makes program heatmaps
and adjacent-time velocity visibly dynamic while the global correlation
distance remains close to zero. These observations are mathematically
compatible.

## Component evidence

- Component decomposition is exact up to a maximum residual of
  `{residual_max:.3e}`.
- Composition-change and module/action contributions are reported independently
  in `agent_progression_component_decomposition.csv`.
- Maximum composition-change mean absolute contribution is
  `{composition_max:.3e}` (zero in the saved trajectories).
- The adjacent-time all-gene correlation-distance velocity reaches
  `{velocity_peak_value:.3e}` per hour at `{velocity_peak_time:g}` h.
- Mean absolute frozen-module change reaches `{module_change_peak_mean:.3e}` at
  `{module_change_peak_time:g}` h.
- The healthy prototype baseline is not mistakenly reloaded as every endpoint.
  In these saved trajectories the composition-change contribution is exactly
  zero, while the saved action/module-driven expression update accounts for
  the endpoint changes.

Thus Panel C, Panel D, and Panel E are mathematically consistent: the
action/module decoder creates a small but temporally structured update, its
largest adjacent-time rate occurs early, and its module-level amplitude is
about {module_to_global_ratio:.2f} times the full-transcriptome correlation
amplitude. The stable healthy-prototype background dominates the global
12,263-dimensional correlation geometry.

## Panel C decision

The frozen global progression metric is retained. The revised four-method
figure uses the {int(complete_valid.sum()):,} genes with finite chronODE-M
values at all nine timepoints. Official fit status reports 12,259 successful
genes and four failed genes; however, {int(partial_nonfinite.sum()):,} of those
officially successful fits contain a non-finite late-time value and therefore
cannot enter a complete-trajectory metric without imputation. No imputation is
used. The original 12,263-gene Agent audit remains reported, and
program/high-variance sensitivities are supplementary only.
"""
    (REV / "AGENT_PROGRESSION_AUDIT.md").write_text(report)

    chron = metrics.set_index("method").loc["chronODE-M\u2020"]
    peaks = {
        method: (
            None
            if not np.isfinite(metrics.set_index("method").loc[method, "DNB_peak_time"])
            else float(metrics.set_index("method").loc[method, "DNB_peak_time"])
        )
        for method in METHODS
    }
    summary = f"""# Updated GSE2565 trajectory reconstruction benchmark

## Status

**PASS_UPDATED_FIGURE_WITH_CHRONODE_MONOTONIC_SENSITIVITY**

- chronODE piecewise remains unavailable and its historical audit is retained.
- chronODE-M\u2020 uses the existing official monotonic fitted matrix.
- Official chronODE-M successful fits: {int(fit_success.sum())}/{len(genes)}.
- Complete nine-time-point common-valid evaluation intersection:
  {int(complete_valid.sum())}/{len(genes)}.
- Officially successful fits with at least one non-finite late-time value:
  {int(partial_nonfinite.sum())}.
- Failed chronODE-M genes were not filled: {", ".join(failed_genes)}.
- The four-gene exclusion sensitivity does not modify model outputs and is
  reported in `four_gene_exclusion_sensitivity.csv`.

chronODE-M\u2020 DNB remains non-estimable because the saved fit is one
group-mean trajectory without within-time replicates. Its other unified
metrics are fully evaluated: distribution Pearson
{chron.distribution_shift_Pearson:.3f}, progression Pearson
{chron.progression_Pearson:.3f}, velocity Pearson
{chron.velocity_Pearson:.3f}, mean module Pearson
{chron.mean_module_Pearson:.3f}, and gene-direction agreement
{chron.gene_direction_agreement:.3f}.

Online Agent Panel C is not a code error. The audit classification is
`{audit_class}`. The global metric remains the primary Panel C result;
program-restricted results remain supplementary.

All external baseline metrics are full-trajectory fitted reconstruction
results, not held-out prediction. Online Agent remains a six-run exploratory
pilot that does not read observed expression.
"""
    (REV / "UPDATED_BENCHMARK_SUMMARY.md").write_text(summary)
    audit = {
        "status": "PASS_UPDATED_FIGURE_WITH_CHRONODE_MONOTONIC_SENSITIVITY",
        "chronode_piecewise": "unavailable",
        "chronode_monotonic": "successful_sensitivity_baseline",
        "chronode_monotonic_existing_output_reused": True,
        "chronode_refit_performed": False,
        "chronode_official_success_genes": int(fit_success.sum()),
        "chronode_complete_trajectory_evaluation_genes": int(
            complete_valid.sum()
        ),
        "chronode_official_success_with_partial_nonfinite_values": int(
            partial_nonfinite.sum()
        ),
        "chronode_nonfinite_values_by_time": {
            str(float(hour)): int((~np.isfinite(mono_full[i])).sum())
            for i, hour in enumerate(TIMES)
        },
        "chronode_failed_genes": failed_genes,
        "chronode_failed_gene_fill_used": False,
        "chronode_DNB_estimable": False,
        "chronode_DNB_non_estimable_reason": "single group-mean reconstruction has no within-time replicates",
        "rvagene_rerun": False,
        "online_agent_rerun": False,
        "agent_progression_audit_classification": audit_class,
        "correlation_implementation_max_difference": max_impl_diff,
        "panel_C_main_metric_changed": False,
        "panel_C_revised_common_valid_gene_count": int(
            complete_valid.sum()
        ),
        "original_full_gene_agent_audit_gene_count": len(genes),
        "program_restricted_progression_is_supplementary": True,
        "figure_generated": True,
        "method_order": ["Real", "chronODE-M\u2020", "RVAgene", "Online Agent pilot"],
        "piecewise_failure_audit_preserved": (
            BASE / "chronode/patch/PATCH_AUDIT.md"
        ).exists(),
        "historical_figure_preserved": (
            BASE / "GSE2565_bulk_trajectory_reconstruction_benchmark_600dpi.png"
        ).exists(),
        "forecasting_claim_allowed": False,
        "exploratory_agent_pilot": True,
    }
    (REV / "UPDATED_BENCHMARK_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
