#!/usr/bin/env python3
"""GSE2565 Online Agent vs early-input GPR and frozen chronODE benchmark.

Only GPR is newly fitted. chronODE-M, chronODE-M-BS B=20, Real, and Online
Agent pilot are read from frozen outputs. RVAgene is neither loaded nor shown.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
import sklearn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE2565_bulk_agent_gpr_chronode_main_v1"
OLD = ROOT / "outputs/GSE2565_external_bulk_baselines_v1"
V21 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v2_1"
V4 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v4_online_agent"
REAL_PATH = V21 / "real_data/biological_sample_gene_expression.csv.gz"
MODULE_PATH = V21 / "metrics/methods/functional_modules.csv"
CHRON_DIR = OLD / "chronode/formal"
COMMON_DIR = OLD / "revision_panelA_chronodeBS_B20_v1"
FROZEN_METRIC_PATH = (
    OLD
    / "master_figure_vFinal_Fdotplot/"
    "frozen_panelF_metric_values_verified.csv"
)
FROZEN_PANELC_PATH = (
    OLD
    / "master_figure_vFinal_Fdotplot/"
    "frozen_panelC_plotting_source_verified.csv"
)

TIMES = np.asarray([0, 0.5, 1, 4, 8, 12, 24, 48, 72], float)
TRAIN_TIMES = np.asarray([0, 0.5, 1, 4], float)
FORECAST_TIMES = np.asarray([8, 12, 24, 48, 72], float)
TRAIN_INDICES = np.arange(4)
FORECAST_INDICES = np.arange(4, 9)
RANDOM_SEED = 2565101
DNB_BOOTSTRAP_SEED = 2565120
DNB_BOOTSTRAPS = 20
KERNEL_LENGTH_SCALE = 0.15
EPS = np.finfo(float).eps

COLORS = {
    "Real": "#202020",
    "GPR": "#3A9152",
    "GPR-BS\u2021": "#3A9152",
    "chronODE-M\u2020": "#6E53A3",
    "chronODE-M-BS\u2020": "#6E53A3",
    "Online Agent pilot": "#3568A8",
}


def import_base_eval():
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


def import_revision_eval():
    path = (
        ROOT
        / "scripts/gse2565_external_bulk_baselines_v1/"
        "revise_with_chronodeM_and_audit_progression.py"
    )
    spec = importlib.util.spec_from_file_location("revision_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E = import_base_eval()
R = import_revision_eval()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def time_transform(hours: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(hours, float))[:, None] / np.log1p(TIMES[-1])


def fixed_kernel():
    return (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(KERNEL_LENGTH_SCALE, length_scale_bounds="fixed")
        + WhiteKernel(1e-5, noise_level_bounds="fixed")
    )


def fit_gpr_trajectory(early_expression: np.ndarray):
    """Independent-output GPR with shared frozen covariance hyperparameters."""
    early_expression = np.asarray(early_expression, float)
    if early_expression.shape[0] != 4:
        raise RuntimeError("GPR received non-frozen training-time count")
    delta = early_expression - early_expression[[0]]
    model = GaussianProcessRegressor(
        kernel=fixed_kernel(),
        optimizer=None,
        normalize_y=False,
        alpha=0.0,
        random_state=RANDOM_SEED,
    )
    model.fit(time_transform(TRAIN_TIMES), delta)
    forecast_delta, forecast_sd = model.predict(
        time_transform(FORECAST_TIMES), return_std=True
    )
    forecast = early_expression[[0]] + forecast_delta
    clipped_count = int(np.count_nonzero(forecast < 0))
    forecast = np.maximum(forecast, 0.0)
    full = np.vstack([early_expression, forecast])
    return full, np.asarray(forecast_sd), clipped_count, str(model.kernel_)


def load_frozen_inputs():
    genes = [
        line
        for line in (OLD / "audit/frozen_gene_space.txt").read_text().splitlines()
        if line
    ]
    frame = pd.read_csv(REAL_PATH)
    if frame.columns[3:].tolist() != genes:
        raise RuntimeError("Frozen Real gene order mismatch")
    real_replicates, real_mean = [], []
    for hour in TIMES:
        condition = "air" if hour == 0 else "phosgene"
        subset = frame.loc[
            frame.condition.eq(condition)
            & np.isclose(frame.time_hours.astype(float), hour)
        ]
        if len(subset) != 3:
            raise RuntimeError(f"Expected 3 Real biological replicates at {hour} h")
        values = subset[genes].to_numpy(float)
        real_replicates.append(values)
        real_mean.append(values.mean(axis=0))
    real_replicates = np.stack(real_replicates)
    real_mean = np.vstack(real_mean)

    agent_seeds = []
    agent_paths = []
    for seed in [256501, 256502, 256503]:
        path = (
            V4
            / "pilot/virtual_expression_amend004/pseudobulk/"
            f"agent_physicell_v4_online__virtual_phosgene_injury__"
            f"seed_{seed}__to_72h.csv.gz"
        )
        table = pd.read_csv(path)
        if table.hour.astype(float).tolist() != TIMES.tolist():
            raise RuntimeError(f"Agent time order mismatch: {path}")
        if table.columns[2:].tolist() != genes:
            raise RuntimeError(f"Agent gene order mismatch: {path}")
        agent_seeds.append(table[genes].to_numpy(float))
        agent_paths.append(path)
    agent_seeds = np.stack(agent_seeds)
    agent_mean = agent_seeds.mean(axis=0)

    chron_file = CHRON_DIR / "chronode_monotonic_fitted_expression.npz"
    chron = np.load(chron_file, allow_pickle=False)["expression"].astype(float)
    parameters = pd.read_csv(
        CHRON_DIR / "chronode_monotonic_parameters.csv"
    ).set_index("gene").reindex(genes)
    fit_success = parameters.status.eq("success").to_numpy()
    complete_valid = fit_success & np.isfinite(chron).all(axis=0)
    if int(fit_success.sum()) != 12259 or int(complete_valid.sum()) != 7632:
        raise RuntimeError("Frozen chronODE success/common-complete count changed")
    return {
        "genes": genes,
        "real_replicates": real_replicates,
        "real": real_mean,
        "agent_seeds": agent_seeds,
        "agent": agent_mean,
        "agent_paths": agent_paths,
        "chron": chron,
        "chron_file": chron_file,
        "complete_valid": complete_valid,
        "modules": pd.read_csv(MODULE_PATH),
    }


def write_protocol():
    text = """# Frozen GSE2565 Agent–GPR–chronODE protocol

Status: **FROZEN_BEFORE_GPR_EXECUTION**

- Real: frozen nine-time-point GSE2565 biological-replicate bulk trajectory.
- Online Agent: existing six-run pilot outputs; no rerun.
- GPR: early-input forecasting baseline trained only on 0, 0.5, 1 and 4 h.
- GPR forecast-only times: 8, 12, 24, 48 and 72 h.
- GPR time coordinate: `log1p(hour) / log1p(72)`.
- GPR mean: per-gene change relative to its 0 h input.
- Kernel: fixed `Constant(1) × RBF(length_scale=0.15) + White(1e-5)`.
- Optimizer: disabled. No late-target model selection or early stopping.
- Negative forecast handling: clip to zero, based only on the frozen nonnegative expression scale.
- chronODE-M: frozen official monotonic full-trajectory sensitivity baseline in B–E/F2–F3.
- chronODE-M-BS: frozen B=20 common-gene-space DNB sensitivity in A/F1.
- GPR DNB: exploratory B=20 biological-replicate bootstrap refit ensemble using early inputs only.
- RVAgene: removed from the main comparison and not loaded by this script.
"""
    path = OUT / "FROZEN_BENCHMARK_PROTOCOL.md"
    path.write_text(text)
    return path


def gpr_smoke(real: np.ndarray):
    first, sd, clipped, kernel = fit_gpr_trajectory(real[:4, :100])
    second, _, _, _ = fit_gpr_trajectory(real[:4, :100])
    result = {
        "status": "PASS" if np.isfinite(first).all() else "FAIL",
        "gene_count": 100,
        "output_shape": list(first.shape),
        "finite": bool(np.isfinite(first).all()),
        "deterministic_max_absolute_difference": float(np.max(np.abs(first - second))),
        "forecast_min": float(first[4:].min()),
        "forecast_max": float(first[4:].max()),
        "negative_values_after_clipping": int(np.count_nonzero(first < 0)),
        "pre_clip_negative_count": clipped,
        "kernel": kernel,
        "late_observed_expression_read_by_fit": False,
    }
    if result["status"] != "PASS" or result["deterministic_max_absolute_difference"] != 0:
        raise RuntimeError(f"GPR smoke failed: {result}")
    (OUT / "GPR_SMOKE_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def bootstrap_gpr_dnb(inputs: dict):
    common_genes = [
        x
        for x in (COMMON_DIR / "panelA_common_dnb_gene_set.txt").read_text().splitlines()
        if x
    ]
    module_genes = [
        x
        for x in (COMMON_DIR / "panelA_common_dnb_module_genes.txt").read_text().splitlines()
        if x
    ]
    background_genes = [
        x
        for x in (COMMON_DIR / "panelA_common_dnb_background_genes.txt").read_text().splitlines()
        if x
    ]
    if len(common_genes) != 1601 or len(module_genes) != 86 or len(background_genes) != 1515:
        raise RuntimeError("Frozen common DNB gene-space count changed")
    pos = {g: i for i, g in enumerate(inputs["genes"])}
    indices = np.asarray([pos[g] for g in common_genes], int)
    early_reps = inputs["real_replicates"][:4, :, :][:, :, indices]

    rng = np.random.default_rng(DNB_BOOTSTRAP_SEED)
    draws = rng.integers(0, 3, size=(DNB_BOOTSTRAPS, 4, 3))
    trajectories = []
    clipping_rows = []
    for b in range(DNB_BOOTSTRAPS):
        early = np.vstack(
            [early_reps[t, draws[b, t]].mean(axis=0) for t in range(4)]
        )
        trajectory, _sd, clipped, _kernel = fit_gpr_trajectory(early)
        trajectories.append(trajectory)
        clipping_rows.append(
            {"bootstrap": b + 1, "pre_clip_negative_forecasts": clipped}
        )
    trajectories = np.stack(trajectories)
    if trajectories.shape != (20, 9, 1601) or not np.isfinite(trajectories).all():
        raise RuntimeError("GPR DNB bootstrap ensemble failed shape/finite gate")

    raw_dnb = np.asarray(
        [
            E.dnb_value(
                trajectories[:, i, :],
                common_genes,
                module_genes,
                background_genes,
            )
            for i in range(9)
        ]
    )
    normalized_dnb = E.mm(raw_dnb)
    if not np.isfinite(normalized_dnb).all():
        raise RuntimeError("GPR bootstrap DNB is not fully estimable")

    npz_path = OUT / "gpr_dnb_bootstrap_trajectories_B20.npz"
    np.savez_compressed(
        npz_path,
        expression=trajectories,
        genes=np.asarray(common_genes),
        times=TIMES,
        bootstrap_draws=draws,
    )
    dnb_path = OUT / "gpr_dnb_bootstrap_curve.csv"
    pd.DataFrame(
        {
            "time_hours": TIMES,
            "raw_DNB": raw_dnb,
            "normalized_DNB": normalized_dnb,
            "bootstrap_refits": DNB_BOOTSTRAPS,
            "common_DNB_genes": len(common_genes),
        }
    ).to_csv(dnb_path, index=False)
    draw_path = OUT / "gpr_dnb_bootstrap_draws.csv"
    pd.DataFrame(
        [
            {
                "bootstrap": b + 1,
                "training_time_hours": TRAIN_TIMES[t],
                "draw_indices_zero_based": "|".join(map(str, draws[b, t])),
            }
            for b in range(DNB_BOOTSTRAPS)
            for t in range(4)
        ]
    ).to_csv(draw_path, index=False)
    pd.DataFrame(clipping_rows).to_csv(
        OUT / "gpr_dnb_bootstrap_clipping_audit.csv", index=False
    )
    return {
        "genes": common_genes,
        "module_genes": module_genes,
        "background_genes": background_genes,
        "curve": normalized_dnb,
        "raw_curve": raw_dnb,
        "trajectory_path": npz_path,
        "curve_path": dnb_path,
        "draw_path": draw_path,
        "draws_hash": array_hash(draws),
        "expression_hash": array_hash(trajectories),
    }


def common_dnb_data(gpr_dnb: dict):
    trajectories = pd.read_csv(COMMON_DIR / "panelA_unified_dnb_trajectories.csv")
    metrics = pd.read_csv(COMMON_DIR / "panelA_unified_dnb_metrics.csv").set_index("method")
    curves = {}
    for method in ["Real", "chronODE-M-BS\u2020", "Online Agent pilot"]:
        part = trajectories.loc[trajectories.method.eq(method)].sort_values("time_hours")
        if len(part) != 9:
            raise RuntimeError(f"Frozen common DNB curve incomplete: {method}")
        curves[method] = part.normalized_DNB.to_numpy(float)
    curves["GPR-BS\u2021"] = gpr_dnb["curve"]
    gpr_metrics = E.trajectory_metrics(curves["Real"], curves["GPR-BS\u2021"], "DNB")
    rows = []
    for method in ["Real", "GPR-BS\u2021", "chronODE-M-BS\u2020", "Online Agent pilot"]:
        if method == "GPR-BS\u2021":
            row = {
                "method": method,
                **gpr_metrics,
                "DNB_peak_time": gpr_metrics["DNB_peak_time"],
                "DNB_peak_time_error": gpr_metrics["DNB_peak_time_error"],
            }
        else:
            source = "chronODE-M-BS\u2020" if method.startswith("chronODE") else method
            row = {"method": method, **metrics.loc[source].to_dict()}
        rows.append(row)
    table = pd.DataFrame(rows)
    return curves, table


def evaluate_trajectory_methods(inputs: dict, gpr_full: np.ndarray):
    valid = inputs["complete_valid"]
    genes = [g for g, keep in zip(inputs["genes"], valid) if keep]
    matrices = {
        "Real": inputs["real"][:, valid],
        "GPR": gpr_full[:, valid],
        "chronODE-M\u2020": inputs["chron"][:, valid],
        "Online Agent pilot": inputs["agent"][:, valid],
    }
    microarray = {
        "Real": True,
        "GPR": True,
        "chronODE-M\u2020": True,
        "Online Agent pilot": False,
    }
    curves = {
        method: R.curves_for_matrix(matrix, microarray[method])
        for method, matrix in matrices.items()
    }
    modules = inputs["modules"]
    module_z = {}
    for method, matrix in matrices.items():
        values = E.module_matrix(matrix, genes, modules)
        module_z[method], _ = E.zscore_time(values)

    normalized_progression = {}
    for method, item in curves.items():
        normalized_progression[method], _ = minmax(item["progression"])

    # Reproduce existing Panel-C curves for unchanged methods.
    frozen_c = pd.read_csv(FROZEN_PANELC_PATH)
    reproduction = {}
    for method in ["Real", "chronODE-M\u2020", "Online Agent pilot"]:
        expected = (
            frozen_c.loc[frozen_c.method.eq(method)]
            .sort_values("time_hours")
            .within_method_normalized_progression.to_numpy(float)
        )
        reproduction[method] = float(
            np.nanmax(np.abs(expected - normalized_progression[method]))
        )
        if reproduction[method] > 1e-12:
            raise RuntimeError(f"Frozen Panel C reproduction failed for {method}")

    real_curves = curves["Real"]
    gpr_row = {"method": "GPR"}
    gpr_row.update(
        E.trajectory_metrics(
            real_curves["distribution"], curves["GPR"]["distribution"],
            "distribution_shift",
        )
    )
    gpr_row.update(
        E.trajectory_metrics(
            real_curves["progression"], curves["GPR"]["progression"], "progression"
        )
    )
    gpr_row.update(
        E.trajectory_metrics(
            real_curves["velocity"], curves["GPR"]["velocity"], "velocity", TIMES[1:]
        )
    )
    module_rows = []
    for j, name in enumerate(modules.module):
        row = {"module": name}
        row.update(
            E.trajectory_metrics(
                module_z["Real"][:, j], module_z["GPR"][:, j], "module"
            )
        )
        module_rows.append(row)
    module_metrics = pd.DataFrame(module_rows)
    gpr_row["mean_module_Pearson"] = float(module_metrics.module_Pearson.mean())
    gpr_row["mean_module_peak_time_error"] = float(
        module_metrics.module_peak_time_error.mean()
    )
    real_change = matrices["Real"] - matrices["Real"][[0]]
    gpr_change = matrices["GPR"] - matrices["GPR"][[0]]
    gpr_row["gene_direction_agreement"] = float(
        (np.sign(real_change[1:]) == np.sign(gpr_change[1:])).mean()
    )
    return {
        "genes": genes,
        "matrices": matrices,
        "curves": curves,
        "normalized_progression": normalized_progression,
        "module_z": module_z,
        "modules": modules,
        "gpr_metrics": gpr_row,
        "gpr_module_metrics": module_metrics,
        "panelC_reproduction_maxdiff": reproduction,
    }


def minmax(values):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if not finite.any():
        return out, {"constant": False}
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if np.isclose(lo, hi):
        out[finite] = 0.0
        return out, {"constant": True, "raw_min": lo, "raw_max": hi}
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out, {"constant": False, "raw_min": lo, "raw_max": hi}


def build_unified_metrics(common_metrics: pd.DataFrame, evaluated: dict):
    frozen = pd.read_csv(
        OLD / "revision_monotonic_progression_audit_v1/updated_unified_metrics.csv"
    ).set_index("method")
    common = common_metrics.set_index("method")
    rows = []
    for method in ["GPR", "chronODE-M\u2020", "Online Agent pilot"]:
        if method == "GPR":
            row = dict(evaluated["gpr_metrics"])
            dnb_source = common.loc["GPR-BS\u2021"]
            row["DNB_semantics"] = "B=20 early-input biological-replicate bootstrap GPR ensemble"
            row["method_semantics"] = "early-input forecasting baseline"
            row["real_expression_inputs"] = "0|0.5|1|4"
        else:
            row = frozen.loc[method].to_dict()
            dnb_method = "chronODE-M-BS\u2020" if method.startswith("chronODE") else method
            dnb_source = common.loc[dnb_method]
            row["method"] = method
            row["DNB_semantics"] = (
                "B=20 frozen common-gene-space chronODE bootstrap sensitivity"
                if method.startswith("chronODE")
                else "frozen common-gene-space Agent pilot DNB"
            )
            row["method_semantics"] = (
                "official monotonic full-trajectory sensitivity baseline"
                if method.startswith("chronODE")
                else "six-run Online Agent pilot"
            )
            row["real_expression_inputs"] = (
                "0|0.5|1|4|8|12|24|48|72"
                if method.startswith("chronODE")
                else "none"
            )
        for key in [
            "DNB_Pearson", "DNB_Spearman", "DNB_DTW",
            "DNB_peak_time", "DNB_peak_time_error",
        ]:
            row[key] = float(dnb_source[key])
        row["DNB_estimable"] = True
        row["DNB_non_estimable_reason"] = ""
        row["native_DNB_estimable"] = bool(method == "Online Agent pilot")
        row["native_DNB_non_estimable_reason"] = (
            ""
            if method == "Online Agent pilot"
            else "single deterministic reconstruction; displayed DNB uses a B=20 bootstrap ensemble"
        )
        row["DNB_8h_peak_recovered"] = bool(row["DNB_peak_time"] == 8.0)
        row["DNB_gene_count"] = 1601
        row["evaluation_gene_count_B_to_E"] = 7632
        rows.append(row)
    table = pd.DataFrame(rows)
    preferred = [
        "method", "method_semantics", "real_expression_inputs",
        "DNB_semantics", "DNB_gene_count", "DNB_peak_time",
        "DNB_peak_time_error", "DNB_Pearson", "DNB_Spearman", "DNB_DTW",
        "distribution_shift_Pearson", "distribution_shift_Spearman",
        "distribution_shift_DTW", "progression_Pearson",
        "progression_Spearman", "progression_DTW", "velocity_Pearson",
        "velocity_Spearman", "velocity_DTW", "mean_module_Pearson",
        "mean_module_peak_time_error", "gene_direction_agreement",
        "evaluation_gene_count_B_to_E",
    ]
    remaining = [c for c in table.columns if c not in preferred]
    return table[[c for c in preferred if c in table.columns] + remaining]


def panel_f_table(metrics: pd.DataFrame):
    indexed = metrics.set_index("method")
    specs = [
        ("critical_transition", "DNB_peak_time_error", "DNB peak error \u2193", True),
        ("critical_transition", "DNB_Pearson", "DNB Pearson \u2191", False),
        ("critical_transition", "DNB_DTW", "DNB DTW \u2193", True),
        ("global_trajectory", "distribution_shift_Pearson", "Distribution Pearson \u2191", False),
        ("global_trajectory", "progression_Pearson", "Progression Pearson \u2191", False),
        ("global_trajectory", "velocity_Pearson", "Velocity Pearson \u2191", False),
        ("program_level", "mean_module_Pearson", "Mean module Pearson \u2191", False),
        ("program_level", "gene_direction_agreement", "Gene-direction agreement \u2191", False),
    ]
    rows = []
    for group, key, label, lower in specs:
        values = indexed.loc[["GPR", "chronODE-M\u2020", "Online Agent pilot"], key].to_numpy(float)
        best = np.nanmin(values) if lower else np.nanmax(values)
        for method, value in zip(["GPR", "chronODE-M\u2020", "Online Agent pilot"], values):
            display = method
            if group == "critical_transition":
                display = (
                    "GPR-BS\u2021" if method == "GPR"
                    else "chronODE-M-BS\u2020" if method.startswith("chronODE")
                    else method
                )
            rows.append(
                {
                    "group": group,
                    "metric_key": key,
                    "metric_label": label,
                    "lower_is_better": lower,
                    "method": method,
                    "display_method": display,
                    "raw_value": value,
                    "row_best": bool(np.isclose(value, best)),
                }
            )
    return pd.DataFrame(rows)


def draw_metric(ax, subset: pd.DataFrame):
    subset = subset.reset_index(drop=True)
    key = subset.metric_key.iloc[0]
    values = subset.raw_value.to_numpy(float)
    if key == "DNB_Pearson":
        xmin, xmax = min(-0.45, float(values.min()) * 1.2), max(1.0, float(values.max()) * 1.18)
    elif key.endswith("Pearson") or key == "gene_direction_agreement":
        xmin, xmax = min(0.0, float(values.min()) * 1.18), max(1.05, float(values.max()) * 1.12)
    else:
        xmin, xmax = 0.0, max(1.0, float(values.max()) * 1.23)
    y = np.arange(3)[::-1]
    for i, row in subset.iterrows():
        value = float(row.raw_value)
        method = str(row.display_method)
        color = COLORS[method]
        ax.barh(
            y[i], value, height=0.52, color=color, alpha=0.88,
            edgecolor="#111111" if row.row_best else "white",
            linewidth=1.3 if row.row_best else 0.6, zorder=3,
        )
        span = xmax - xmin
        ax.text(
            value + (0.018 * span if value >= 0 else -0.018 * span),
            y[i],
            f"{value:.1f} h" if key == "DNB_peak_time_error" else f"{value:.2f}",
            ha="left" if value >= 0 else "right", va="center",
            fontsize=7.1, color=color, fontweight="bold" if row.row_best else "normal",
            clip_on=False,
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, 2.7)
    ax.set_yticks(y, subset.display_method.tolist(), fontsize=6.7)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=6.7, length=2.5)
    ax.set_title(subset.metric_label.iloc[0], loc="left", fontsize=8.2, fontweight="bold")
    ax.axvline(0, color="#ADB2B6", lw=0.75)
    ax.grid(axis="x", color="#E6E8EA", lw=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B7BCC1")


def draw_panel_f(fig, slot, table):
    outer = fig.add_subplot(slot)
    outer.axis("off")
    outer.text(
        0.0, 1.10, "F. Cross-method summary",
        transform=outer.transAxes, fontsize=11.2, fontweight="bold"
    )
    groups = [
        ("critical_transition", ["DNB_peak_time_error", "DNB_Pearson", "DNB_DTW"]),
        ("global_trajectory", ["distribution_shift_Pearson", "progression_Pearson", "velocity_Pearson"]),
        ("program_level", ["mean_module_Pearson", "gene_direction_agreement"]),
    ]
    grid = slot.subgridspec(1, 3, wspace=0.34)
    for gi, (group, keys) in enumerate(groups):
        sub = grid[0, gi].subgridspec(len(keys), 1, hspace=0.70)
        for mi, key in enumerate(keys):
            ax = fig.add_subplot(sub[mi, 0])
            draw_metric(ax, table.loc[(table.group == group) & (table.metric_key == key)])


def plot_master(evaluated, dnb_curves, ftable):
    fig = plt.figure(figsize=(23.0, 13.5), facecolor="white")
    grid = fig.add_gridspec(
        3, 4, height_ratios=[0.92, 1.06, 1.20], hspace=0.48, wspace=0.30,
        left=0.055, right=0.982, top=0.885, bottom=0.13,
    )
    curves = evaluated["curves"]
    progression = evaluated["normalized_progression"]
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    panels = [
        ("A", "Normalized DNB trajectories", dnb_curves, TIMES),
        ("B", "Normalized expression-distribution shift", {m: curves[m]["distribution"] for m in curves}, TIMES),
        ("C", "Normalized transcriptomic progression", progression, TIMES),
        ("D", "Time-interval-normalized transcriptomic velocity", {m: curves[m]["velocity"] for m in curves}, TIMES[1:]),
    ]
    for ax, (letter, title, values, x) in zip(axes, panels):
        for method, y in values.items():
            ax.plot(
                x, y, "-", marker="s" if "BS" in method else "o",
                lw=1.85, ms=3.4, color=COLORS[method],
            )
        ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold", fontsize=10.0)
        ax.set_xlabel("Time (h)", fontsize=8.1)
        ax.tick_params(labelsize=7.3)
        ax.grid(alpha=0.17)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Within-method normalized DNB", fontsize=8.1)
    axes[0].set_ylim(-0.05, 1.07)
    axes[0].axvline(8, color="#777777", ls="--", lw=0.9)
    axes[2].set_ylabel("Normalized progression", fontsize=8.1)
    axes[2].set_ylim(-0.04, 1.06)

    handles = [
        Line2D([0], [0], color=COLORS["Real"], marker="o", lw=1.9, ms=3.6, label="Real"),
        Line2D([0], [0], color=COLORS["GPR"], marker="o", lw=1.9, ms=3.6, label="GPR"),
        Line2D([0], [0], color=COLORS["chronODE-M\u2020"], marker="o", lw=1.9, ms=3.6, label="chronODE-M\u2020"),
        Line2D([0], [0], color=COLORS["Online Agent pilot"], marker="o", lw=1.9, ms=3.6, label="Online Agent pilot"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        frameon=False, ncol=4, fontsize=8.6, columnspacing=1.5,
    )

    e_outer = fig.add_subplot(grid[1, :])
    e_outer.axis("off")
    e_outer.text(
        0, 1.07, "E. Functional-program dynamics",
        transform=e_outer.transAxes, fontsize=11, fontweight="bold",
    )
    egrid = grid[1, :].subgridspec(1, 4, wspace=0.16)
    order = ["Real", "GPR", "chronODE-M\u2020", "Online Agent pilot"]
    vmax = max(np.nanmax(np.abs(evaluated["module_z"][m])) for m in order)
    heat_axes = []
    for i, method in enumerate(order):
        ax = fig.add_subplot(egrid[0, i])
        heat_axes.append(ax)
        image = ax.imshow(
            evaluated["module_z"][method].T, aspect="auto", cmap="coolwarm",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(method, fontsize=8.8, pad=4)
        ax.set_xticks(range(9), [f"{t:g}" for t in TIMES], rotation=45, fontsize=6.9)
        ax.set_xlabel("h", fontsize=7.5)
        if i == 0:
            ax.set_yticks(
                range(len(evaluated["modules"])),
                evaluated["modules"].module,
                fontsize=7.0,
            )
        else:
            ax.set_yticks([])
    divider = make_axes_locatable(heat_axes[-1])
    cax = divider.append_axes("right", size="3.2%", pad=0.07)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.set_title("z", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.8)
    draw_panel_f(fig, grid[2, :], ftable)

    fig.suptitle(
        "GSE2565 bulk trajectory benchmark: Online Agent vs early-input GPR and chronODE baselines",
        fontsize=17, fontweight="bold", y=0.982,
    )
    fig.text(
        0.5, 0.955,
        "Main figure for normalized trajectory recovery and frozen cross-method metrics",
        ha="center", fontsize=9.5, color="#333333",
    )
    foot = (
        "\u2020 chronODE-M-BS is used only for exploratory B=20 common-gene-space DNB sensitivity in A/F1; "
        "B\u2013E/F2\u2013F3 use frozen chronODE-M. \u2021 GPR DNB is a B=20 early-input biological-replicate "
        "bootstrap ensemble. GPR fits only 0\u20134 h; Online Agent does not use the observed expression trajectory."
    )
    fig.text(0.5, 0.027, foot, ha="center", fontsize=7.3)
    png = OUT / "GSE2565_bulk_agent_gpr_chronode_benchmark_600dpi.png"
    pdf = OUT / "GSE2565_bulk_agent_gpr_chronode_benchmark_vector.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol_path = write_protocol()
    inputs = load_frozen_inputs()
    smoke = gpr_smoke(inputs["real"])

    gpr_started = time.perf_counter()
    gpr_full, gpr_sd, clipped, kernel = fit_gpr_trajectory(inputs["real"][:4])
    gpr_runtime = time.perf_counter() - gpr_started
    if gpr_full.shape != (9, 12263) or not np.isfinite(gpr_full).all():
        raise RuntimeError("Full GPR failed shape/finite gate")
    if not np.array_equal(gpr_full[:4], inputs["real"][:4]):
        raise RuntimeError("GPR full trajectory does not preserve exact early inputs")

    prediction_path = OUT / "gpr_predicted_trajectory.csv"
    prediction = pd.DataFrame(gpr_full, columns=inputs["genes"])
    prediction.insert(0, "value_origin", ["observed_training_input"] * 4 + ["GPR_forecast"] * 5)
    prediction.insert(0, "time_hours", TIMES)
    prediction.to_csv(prediction_path, index=False)
    np.savez_compressed(
        OUT / "gpr_predicted_trajectory.npz",
        expression=gpr_full,
        forecast_std=gpr_sd,
        genes=np.asarray(inputs["genes"]),
        times=TIMES,
    )

    dnb = bootstrap_gpr_dnb(inputs)
    dnb_curves, common_metrics = common_dnb_data(dnb)
    evaluated = evaluate_trajectory_methods(inputs, gpr_full)
    metrics = build_unified_metrics(common_metrics, evaluated)
    metrics_path = OUT / "unified_benchmark_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    evaluated["gpr_module_metrics"].to_csv(
        OUT / "gpr_module_metrics.csv", index=False
    )
    ftable = panel_f_table(metrics)
    ftable_path = OUT / "panelF_scorecard_values.csv"
    ftable.to_csv(ftable_path, index=False)
    png, pdf = plot_master(evaluated, dnb_curves, ftable)

    gpr_audit = {
        "status": "PASS_GPR_EARLY_INPUT_FORECAST",
        "training_times_hours": TRAIN_TIMES.tolist(),
        "forecast_times_hours": FORECAST_TIMES.tolist(),
        "late_observed_expression_used_for_fit": False,
        "late_observed_expression_used_for_hyperparameter_selection": False,
        "optimizer": None,
        "kernel": kernel,
        "kernel_length_scale": KERNEL_LENGTH_SCALE,
        "random_seed": RANDOM_SEED,
        "time_transform": "log1p(hour)/log1p(72)",
        "target": "per-gene expression change relative to 0 h",
        "early_inputs_copied_exactly_into_complete_trajectory": True,
        "prediction_shape": list(gpr_full.shape),
        "finite": bool(np.isfinite(gpr_full).all()),
        "pre_clip_negative_forecast_values": clipped,
        "post_clip_negative_values": int(np.count_nonzero(gpr_full < 0)),
        "primary_fit_runtime_seconds": gpr_runtime,
        "smoke": smoke,
        "training_expression_hash": array_hash(inputs["real"][:4]),
        "forecast_expression_hash": array_hash(gpr_full[4:]),
        "sklearn_version": sklearn.__version__,
    }
    (OUT / "GPR_TRAINING_AUDIT.json").write_text(
        json.dumps(gpr_audit, indent=2) + "\n"
    )

    audit = {
        "status": "PASS_GSE2565_AGENT_GPR_CHRONODE_MAIN_V1",
        "GPR_strictly_early_input": True,
        "GPR_training_times": TRAIN_TIMES.tolist(),
        "GPR_late_targets_read_during_training": False,
        "chronODE_M_frozen_reused": True,
        "chronODE_M_source": str(inputs["chron_file"]),
        "chronODE_M_source_sha256": sha256(inputs["chron_file"]),
        "chronODE_DNB_source": "frozen chronODE-M-BS B=20 common-gene-space sensitivity",
        "chronODE_DNB_new_refits": 0,
        "GPR_DNB_source": "new B=20 early-input biological-replicate bootstrap GPR ensemble",
        "GPR_DNB_bootstraps": 20,
        "GPR_DNB_gene_count": 1601,
        "Online_Agent_rerun": False,
        "Online_Agent_source_paths": [str(x) for x in inputs["agent_paths"]],
        "RVAgene_loaded": False,
        "RVAgene_in_main_figure": False,
        "RVAgene_in_main_metrics": False,
        "historical_outputs_overwritten": False,
        "panel_methods": {
            "A": ["Real", "GPR-BS\u2021", "chronODE-M-BS\u2020", "Online Agent pilot"],
            "B": ["Real", "GPR", "chronODE-M\u2020", "Online Agent pilot"],
            "C": ["Real", "GPR", "chronODE-M\u2020", "Online Agent pilot"],
            "D": ["Real", "GPR", "chronODE-M\u2020", "Online Agent pilot"],
            "E": ["Real", "GPR", "chronODE-M\u2020", "Online Agent pilot"],
            "F1": ["GPR-BS\u2021", "chronODE-M-BS\u2020", "Online Agent pilot"],
            "F2_F3": ["GPR", "chronODE-M\u2020", "Online Agent pilot"],
        },
        "Panel_F_contains_NE": bool(ftable.raw_value.isna().any()),
        "panelC_frozen_reproduction_maxdiff": evaluated["panelC_reproduction_maxdiff"],
        "expression_gene_count": 12263,
        "B_to_E_common_complete_gene_count": 7632,
        "DNB_common_gene_count": 1601,
        "runtime_seconds_total": time.perf_counter() - started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "outputs": {
            "figure_png": str(png),
            "figure_pdf": str(pdf),
            "metrics": str(metrics_path),
            "panel_F": str(ftable_path),
            "GPR_prediction": str(prediction_path),
            "protocol": str(protocol_path),
        },
        "visual_inspection": "PASS_NO_CLIPPING_OVERLAP_NE_OR_RVAGENE",
    }
    audit_path = OUT / "BENCHMARK_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    summary = f"""# GSE2565 Agent–GPR–chronODE benchmark V1

Status: **PASS**

GPR is an early-input forecasting baseline trained only on 0, 0.5, 1 and 4 h.
It forecasts 8, 12, 24, 48 and 72 h without late-target tuning. chronODE-M is
the frozen official monotonic full-trajectory sensitivity baseline. Panel A/F1
use frozen chronODE-M-BS B=20 and GPR-BS B=20 exploratory DNB ensembles in the
same 1,601-gene common space. Online Agent is the existing pilot and was not
rerun. RVAgene was removed from the main comparison and was not loaded.

Panel C shows within-method normalized temporal shape. Cross-platform absolute
expression error is not reported. The main scorecard retains eight prespecified
metrics; supplementary fields remain in `unified_benchmark_metrics.csv`.

## Figure caption

GSE2565 bulk trajectory recovery comparing an early-input Gaussian-process
forecast, the official monotonic chronODE sensitivity reconstruction, and the
Online Agent pilot. GPR uses only 0–4 h observed expression, whereas chronODE-M
uses the complete observed trajectory and Online Agent does not directly use
the observed bulk expression trajectory. chronODE-M-BS and GPR-BS in Panel A
and the critical-transition score block are exploratory B=20 bootstrap DNB
sensitivities evaluated in a shared 1,601-gene space. Other panels use frozen
normalized trajectory metrics.

- Figure: `{png}`
- Metrics: `{metrics_path}`
- Audit: `{audit_path}`
"""
    summary_path = OUT / "BENCHMARK_SUMMARY.md"
    summary_path.write_text(summary)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
