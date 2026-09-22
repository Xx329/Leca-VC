#!/usr/bin/env python3
"""Evaluate full-trajectory reconstructions and produce the A--F figure.

chronODE piecewise remains an explicit unavailable method. The completed
chronODE monotonic fit is evaluated only in the native-reconstruction
sensitivity table and is not substituted into the primary A--F figure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial.distance import correlation, cosine
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE2565_external_bulk_baselines_v1"
CHRON = OUT / "chronode/formal"
RVA = OUT / "rvagene/formal"
V21 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v2_1"
V4 = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v4_online_agent"
REAL_PATH = V21 / "real_data/biological_sample_gene_expression.csv.gz"
MODULE_PATH = V21 / "metrics/methods/functional_modules.csv"
TIMES = np.asarray([0, 0.5, 1, 4, 8, 12, 24, 48, 72], dtype=float)
METHODS = ["chronODE piecewise", "RVAgene", "Online Agent pilot"]
COLORS = {
    "Real": "#202020",
    "chronODE piecewise": "#8C8C8C",
    "RVAgene": "#D97A23",
    "Online Agent pilot": "#3568A8",
}
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 2565007
EPS = np.finfo(float).eps


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if finite.sum() == 0:
        return out
    span = np.ptp(x[finite])
    out[finite] = (
        (x[finite] - np.min(x[finite])) / span if span > EPS else 0.0
    )
    return out


def zscore_time(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    mean = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    constant = (~np.isfinite(sd)) | (sd <= EPS)
    z = np.full_like(x, np.nan)
    usable = ~constant
    z[:, usable] = (x[:, usable] - mean[usable]) / sd[usable]
    return z, constant


def robust_scale_change(x: np.ndarray) -> np.ndarray:
    change = x - x[[0], :]
    scale = np.nanmedian(np.abs(change[1:]), axis=0)
    scale[~np.isfinite(scale) | (scale <= EPS)] = 1.0
    return change / scale[None, :]


def safe_pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2 or np.ptp(a[keep]) <= EPS or np.ptp(b[keep]) <= EPS:
        return np.nan
    return float(pearsonr(a[keep], b[keep]).statistic)


def safe_spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2 or np.ptp(a[keep]) <= EPS or np.ptp(b[keep]) <= EPS:
        return np.nan
    return float(spearmanr(a[keep], b[keep]).statistic)


def dtw(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if not len(a):
        return np.nan
    d = np.full((len(a) + 1, len(b) + 1), np.inf)
    d[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = abs(a[i - 1] - b[j - 1]) + min(
                d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]
            )
    return float(d[-1, -1])


def ccc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2:
        return np.nan
    cov = np.mean((a - a.mean()) * (b - b.mean()))
    den = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return float(2 * cov / den) if den > 0 else np.nan


def load_inputs():
    genes = [
        x
        for x in (OUT / "audit/frozen_gene_space.txt").read_text().splitlines()
        if x
    ]
    frame = pd.read_csv(REAL_PATH)
    if frame.columns[3:].tolist() != genes:
        raise RuntimeError("Real gene order mismatch")
    real_replicates = []
    real_mean = []
    for hour in TIMES:
        condition = "air" if hour == 0 else "phosgene"
        selected = frame[
            (frame.condition == condition)
            & np.isclose(frame.time_hours.astype(float), hour)
        ]
        if len(selected) != 3:
            raise RuntimeError(f"Expected three observed replicates at {hour} h")
        values = selected[genes].to_numpy(float)
        real_replicates.append(values)
        real_mean.append(values.mean(axis=0))
    real_replicates = np.stack(real_replicates)
    real_mean = np.vstack(real_mean)

    rva_seed_file = np.load(
        RVA / "rvagene_reconstruction_by_seed.npz", allow_pickle=False
    )
    if rva_seed_file["genes"].astype(str).tolist() != genes:
        raise RuntimeError("RVAgene gene order mismatch")
    rva_seeds = rva_seed_file["expression"].astype(float)
    rva_mean = rva_seeds.mean(axis=0)

    agent_seeds = []
    for seed in [256501, 256502, 256503]:
        path = (
            V4
            / "pilot/virtual_expression_amend004/pseudobulk/"
            f"agent_physicell_v4_online__virtual_phosgene_injury__"
            f"seed_{seed}__to_72h.csv.gz"
        )
        data = pd.read_csv(path)
        if data[["hour"]].to_numpy().ravel().tolist() != TIMES.tolist():
            raise RuntimeError(f"Agent time order mismatch: {path}")
        if data.columns[2:].tolist() != genes:
            raise RuntimeError(f"Agent gene order mismatch: {path}")
        agent_seeds.append(data[genes].to_numpy(float))
    agent_seeds = np.stack(agent_seeds)
    agent_mean = agent_seeds.mean(axis=0)

    piece = np.load(
        CHRON / "chronode_piecewise_fitted_expression.npz", allow_pickle=False
    )["expression"]
    mono_file = np.load(
        CHRON / "chronode_monotonic_fitted_expression.npz", allow_pickle=False
    )
    mono = mono_file["expression"].astype(float)
    return (
        genes,
        real_replicates,
        real_mean,
        piece,
        mono,
        rva_seeds,
        rva_mean,
        agent_seeds,
        agent_mean,
    )


def module_matrix(x: np.ndarray, genes: list[str], modules: pd.DataFrame):
    position = {g: i for i, g in enumerate(genes)}
    columns = []
    for _, row in modules.iterrows():
        idx = [
            position[g]
            for g in str(row.genes).split("|")
            if g in position
        ]
        columns.append(np.nanmean(x[:, idx], axis=1))
    return np.asarray(columns).T


def distribution_shift(x: np.ndarray, microarray: bool) -> np.ndarray:
    transformed = np.exp2(np.clip(x, -50, 50)) if microarray else np.maximum(x, 0)
    reference = transformed[0] + 1e-8
    reference = reference / reference.sum()
    curve = []
    for row in transformed:
        p = row + 1e-8
        p = p / p.sum()
        curve.append(
            0.5
            * (
                np.sum(p * np.log(p / reference))
                + np.sum(reference * np.log(reference / p))
            )
        )
    return np.asarray(curve)


def progression(x: np.ndarray, metric: str = "correlation") -> np.ndarray:
    func = correlation if metric == "correlation" else cosine
    return np.asarray([0.0 if i == 0 else func(x[i], x[0]) for i in range(len(x))])


def velocity(x: np.ndarray) -> np.ndarray:
    return np.asarray(
        [correlation(x[i], x[i - 1]) / (TIMES[i] - TIMES[i - 1]) for i in range(1, len(x))]
    )


def fixed_dnb_sets(genes: list[str]):
    normal = V21 / "published_extended_module/normal_reference"
    stats = pd.read_csv(normal / "reference_gene_statistics.csv")
    valid = set(stats.loc[stats.reference_SD > EPS, "gene"])
    mapped = pd.read_csv(
        V21 / "published_extended_module/published_262_members_mapped.csv"
    ).fillna("")
    published = set(
        mapped.loc[mapped.mapped_symbol.ne(""), "mapped_symbol"]
    )
    module = [g for g in genes if g in valid and g in published]
    background = [g for g in genes if g in valid and g not in published]
    return module, background


def dnb_value(x: np.ndarray, genes: list[str], module: list[str], background: list[str]):
    pos = {g: i for i, g in enumerate(genes)}
    im = [pos[g] for g in module]
    ib = [pos[g] for g in background]
    xm, xb = x[:, im], x[:, ib]
    vm = np.var(xm, axis=0, ddof=1) > EPS
    vb = np.var(xb, axis=0, ddof=1) > EPS
    xm, xb = xm[:, vm], xb[:, vb]
    if xm.shape[1] < 2 or xb.shape[1] < 1:
        return np.nan
    sm, sb = xm.std(axis=0, ddof=1), xb.std(axis=0, ddof=1)
    a = (xm - xm.mean(axis=0)) / sm
    b = (xb - xb.mean(axis=0)) / sb
    internal = (a.T @ a) / (len(x) - 1)
    tri = np.triu_indices(xm.shape[1], 1)
    pccd = np.abs(internal[tri]).mean()
    total = 0.0
    for start in range(0, xb.shape[1], 256):
        total += np.abs((a.T @ b[:, start : start + 256]) / (len(x) - 1)).sum()
    pcco = total / (xm.shape[1] * xb.shape[1])
    return float(sm.mean() * pccd / pcco)


def dnb_curves(
    genes, real_replicates, rva_seeds, agent_seeds
) -> dict[str, np.ndarray]:
    real_table = pd.read_csv(
        V21 / "real_agent_comparison/real_dnb_components.csv"
    )
    real = (
        real_table[
            real_table.real_processing == "biological_mouse_level_sensitivity"
        ]
        .set_index("timepoint")
        .reindex(TIMES)
        .composite_index.to_numpy(float)
    )
    existing_agent = pd.read_csv(
        V4 / "pilot/amend004_dnb_components.csv"
    ).set_index("timepoint").reindex(TIMES).composite_index.to_numpy(float)
    module, background = fixed_dnb_sets(genes)
    rva = np.asarray(
        [
            dnb_value(rva_seeds[:, i, :], genes, module, background)
            for i in range(len(TIMES))
        ]
    )
    return {
        "Real": mm(real),
        "chronODE piecewise": np.full(len(TIMES), np.nan),
        "RVAgene": mm(rva),
        "Online Agent pilot": mm(existing_agent),
    }


def trajectory_metrics(reference, method, prefix, times=TIMES):
    keep = np.isfinite(reference) & np.isfinite(method)
    if keep.sum() < 2:
        return {
            f"{prefix}_Pearson": np.nan,
            f"{prefix}_Spearman": np.nan,
            f"{prefix}_DTW": np.nan,
            f"{prefix}_peak_time": np.nan,
            f"{prefix}_peak_time_error": np.nan,
        }
    r, m, tt = reference[keep], method[keep], np.asarray(times)[keep]
    rp, mp = float(tt[np.argmax(r)]), float(tt[np.argmax(m)])
    return {
        f"{prefix}_Pearson": safe_pearson(r, m),
        f"{prefix}_Spearman": safe_spearman(r, m),
        f"{prefix}_DTW": dtw(mm(r), mm(m)),
        f"{prefix}_peak_time": mp,
        f"{prefix}_peak_time_error": abs(mp - rp),
    }


def secondary_peak(curve: np.ndarray) -> bool:
    keep = np.isfinite(curve)
    if keep.sum() < 3:
        return False
    return len(find_peaks(curve[keep])[0]) >= 2


def native_metrics(real, predicted, method, version):
    keep = np.isfinite(real) & np.isfinite(predicted)
    if not keep.any():
        return {
            "method": method,
            "version": version,
            "common_finite_values": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Pearson": np.nan,
            "CCC": np.nan,
            "data_fitted_reconstruction": True,
        }
    a, b = real[keep], predicted[keep]
    return {
        "method": method,
        "version": version,
        "common_finite_values": int(keep.sum()),
        "RMSE": float(np.sqrt(np.mean((a - b) ** 2))),
        "MAE": float(np.mean(np.abs(a - b))),
        "Pearson": safe_pearson(a, b),
        "CCC": ccc(a, b),
        "data_fitted_reconstruction": True,
    }


def evaluate():
    (
        genes,
        real_replicates,
        real,
        piece,
        mono,
        rva_seeds,
        rva,
        agent_seeds,
        agent,
    ) = load_inputs()
    modules = pd.read_csv(MODULE_PATH)
    primary = {
        "Real": real,
        "chronODE piecewise": piece,
        "RVAgene": rva,
        "Online Agent pilot": agent,
    }
    microarray = {
        "Real": True,
        "chronODE piecewise": True,
        "RVAgene": True,
        "Online Agent pilot": False,
    }

    zscores = {}
    changes = {}
    constant_rows = []
    module_z = {}
    distributions = {}
    progressions = {}
    cosine_progressions = {}
    velocities = {}
    for method, matrix in primary.items():
        z, constant = zscore_time(matrix)
        zscores[method] = z
        changes[method] = robust_scale_change(matrix)
        constant_rows.append(
            {
                "method": method,
                "constant_trajectory_genes": int(constant.sum()),
                "correlation_eligible_genes": int((~constant).sum()),
            }
        )
        scores = module_matrix(matrix, genes, modules)
        module_z[method], _ = zscore_time(scores)
        distributions[method] = mm(distribution_shift(matrix, microarray[method]))
        progressions[method] = progression(matrix)
        cosine_progressions[method] = progression(matrix, "cosine")
        velocities[method] = mm(velocity(matrix))

    np.savez_compressed(
        OUT / "unified_time_zscore_expression.npz",
        genes=np.asarray(genes),
        times=TIMES,
        real=zscores["Real"],
        chronode_piecewise=zscores["chronODE piecewise"],
        rvagene=zscores["RVAgene"],
        online_agent_pilot=zscores["Online Agent pilot"],
    )
    np.savez_compressed(
        OUT / "unified_baseline_relative_robust_scaled_expression.npz",
        genes=np.asarray(genes),
        times=TIMES,
        real=changes["Real"],
        chronode_piecewise=changes["chronODE piecewise"],
        rvagene=changes["RVAgene"],
        online_agent_pilot=changes["Online Agent pilot"],
    )
    pd.DataFrame(constant_rows).to_csv(
        OUT / "constant_trajectory_gene_counts.csv", index=False
    )

    dnb = dnb_curves(genes, real_replicates, rva_seeds, agent_seeds)
    trajectory_rows = []
    module_rows = []
    direction_rows = []
    real_baseline_sd = real_replicates[0].std(axis=0, ddof=1)
    model_seed_matrices = {
        "RVAgene": rva_seeds,
        "Online Agent pilot": agent_seeds,
    }

    for method in METHODS:
        row = {"method": method}
        row.update(trajectory_metrics(dnb["Real"], dnb[method], "DNB"))
        row["DNB_8h_peak_recovered"] = (
            bool(row["DNB_peak_time"] == 8) if np.isfinite(row["DNB_peak_time"]) else False
        )
        row["DNB_secondary_peak_present"] = secondary_peak(dnb[method])
        row.update(
            trajectory_metrics(
                distributions["Real"], distributions[method], "distribution_shift"
            )
        )
        row.update(
            trajectory_metrics(
                progressions["Real"], progressions[method], "progression"
            )
        )
        row["progression_platform_time"] = (
            float(TIMES[np.flatnonzero(progressions[method] >= 0.9 * np.nanmax(progressions[method]))[0]])
            if np.isfinite(progressions[method]).any()
            and np.nanmax(progressions[method]) > 0
            else np.nan
        )
        row.update(
            trajectory_metrics(
                velocities["Real"], velocities[method], "velocity", TIMES[1:]
            )
        )
        trajectory_rows.append(row)

        for j, module in enumerate(modules.module):
            ref, pred = module_z["Real"][:, j], module_z[method][:, j]
            mr = {"method": method, "module": module}
            mr.update(trajectory_metrics(ref, pred, "module"))
            if np.isfinite(pred).any() and np.isfinite(ref).any():
                rp, mp = np.nanargmax(np.abs(ref)), np.nanargmax(np.abs(pred))
                mr["direction_agreement"] = bool(
                    np.sign(ref[rp]) == np.sign(pred[mp])
                )
            else:
                mr["direction_agreement"] = np.nan
            module_rows.append(mr)

        if method not in model_seed_matrices:
            direction_rows.append(
                {
                    "method": method,
                    "all_gene_direction_agreement": np.nan,
                    "real_active_gene_direction_agreement": np.nan,
                    "model_active_gene_direction_agreement": np.nan,
                    "common_active_gene_direction_agreement": np.nan,
                    "direction_agreement_8h": np.nan,
                    "direction_agreement_8_to_24h": np.nan,
                }
            )
            continue
        model_change = primary[method] - primary[method][[0], :]
        real_change = real - real[[0], :]
        all_agree = np.sign(model_change[1:]) == np.sign(real_change[1:])
        real_active = np.abs(real_change[1:]) >= np.maximum(real_baseline_sd, 1e-8)
        seed_sd = model_seed_matrices[method][:, 0, :].std(axis=0, ddof=1)
        model_active = np.abs(model_change[1:]) >= np.maximum(seed_sd, 1e-8)
        common = real_active & model_active
        direction_rows.append(
            {
                "method": method,
                "all_gene_direction_agreement": float(all_agree.mean()),
                "real_active_gene_direction_agreement": float(
                    all_agree[real_active].mean()
                ),
                "model_active_gene_direction_agreement": float(
                    all_agree[model_active].mean()
                ),
                "common_active_gene_direction_agreement": float(
                    all_agree[common].mean()
                )
                if common.any()
                else np.nan,
                "direction_agreement_8h": float(
                    (
                        np.sign(model_change[TIMES.tolist().index(8)])
                        == np.sign(real_change[TIMES.tolist().index(8)])
                    ).mean()
                ),
                "direction_agreement_8_to_24h": float(
                    (
                        np.sign(
                            primary[method][TIMES.tolist().index(24)]
                            - primary[method][TIMES.tolist().index(8)]
                        )
                        == np.sign(
                            real[TIMES.tolist().index(24)]
                            - real[TIMES.tolist().index(8)]
                        )
                    ).mean()
                ),
            }
        )

    trajectory_df = pd.DataFrame(trajectory_rows)
    module_df = pd.DataFrame(module_rows)
    direction_df = pd.DataFrame(direction_rows)
    module_summary = (
        module_df.groupby("method")
        .agg(
            mean_module_Pearson=("module_Pearson", "mean"),
            mean_module_Spearman=("module_Spearman", "mean"),
            mean_module_DTW=("module_DTW", "mean"),
            mean_module_peak_time_error=("module_peak_time_error", "mean"),
            module_direction_agreement=("direction_agreement", "mean"),
        )
        .reset_index()
    )
    unified = (
        trajectory_df.merge(module_summary, on="method", how="left")
        .merge(direction_df, on="method", how="left")
    )
    unified.to_csv(OUT / "unified_normalized_trajectory_metrics.csv", index=False)
    module_df.to_csv(OUT / "module_level_metrics.csv", index=False)
    direction_df.to_csv(OUT / "gene_direction_agreement.csv", index=False)

    native_rows = [
        native_metrics(real, piece, "chronODE", "piecewise primary unavailable"),
        native_metrics(real, mono, "chronODE", "monotonic sensitivity"),
        native_metrics(real, rva, "RVAgene", "three-seed mean"),
    ]
    for i, seed in enumerate([256501, 256502, 256503]):
        native_rows.append(
            native_metrics(real, rva_seeds[i], "RVAgene", f"seed {seed}")
        )
    pd.DataFrame(native_rows).to_csv(
        OUT / "chronode_rvagene_native_reconstruction_metrics.csv", index=False
    )

    # Biological-sample bootstrap: models remain frozen while the three
    # observed biological samples at each time are resampled independently.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for iteration in range(BOOTSTRAPS):
        boot_real = np.vstack(
            [
                block[rng.integers(0, block.shape[0], block.shape[0])].mean(axis=0)
                for block in real_replicates
            ]
        )
        boot_dist = mm(distribution_shift(boot_real, True))
        boot_prog = progression(boot_real)
        boot_vel = mm(velocity(boot_real))
        boot_modules, _ = zscore_time(module_matrix(boot_real, genes, modules))
        boot_change = boot_real - boot_real[[0], :]
        for method in ["RVAgene", "Online Agent pilot"]:
            values = {
                "distribution_shift_Pearson": safe_pearson(
                    boot_dist, distributions[method]
                ),
                "progression_Pearson": safe_pearson(
                    boot_prog, progressions[method]
                ),
                "velocity_Pearson": safe_pearson(
                    boot_vel, velocities[method]
                ),
                "mean_module_Pearson": float(
                    np.nanmean(
                        [
                            safe_pearson(
                                boot_modules[:, j], module_z[method][:, j]
                            )
                            for j in range(len(modules))
                        ]
                    )
                ),
                "all_gene_direction_agreement": float(
                    (
                        np.sign(primary[method][1:] - primary[method][[0], :])
                        == np.sign(boot_change[1:])
                    ).mean()
                ),
            }
            for metric, value in values.items():
                bootstrap_rows.append(
                    {
                        "iteration": iteration,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
    boot = pd.DataFrame(bootstrap_rows)
    ci = (
        boot.groupby(["method", "metric"])
        .value.agg(
            point_bootstrap_mean="mean",
            ci95_low=lambda x: np.nanpercentile(x, 2.5),
            ci95_high=lambda x: np.nanpercentile(x, 97.5),
        )
        .reset_index()
    )
    unavailable = []
    for method in METHODS:
        unavailable.append(
            {
                "method": method,
                "metric": "DNB trajectory metrics",
                "point_bootstrap_mean": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "note": (
                    "DNB sample bootstrap not recomputed; chronODE piecewise has "
                    "no output and model replicate semantics differ"
                ),
            }
        )
    ci["note"] = "biological sample bootstrap; model output held fixed"
    ci = pd.concat([ci, pd.DataFrame(unavailable)], ignore_index=True)
    ci.to_csv(OUT / "bootstrap_confidence_intervals.csv", index=False)

    rank_specs = {
        "DNB_peak_time_error": True,
        "DNB_Pearson": False,
        "DNB_DTW": True,
        "distribution_shift_Pearson": False,
        "distribution_shift_DTW": True,
        "progression_Pearson": False,
        "progression_DTW": True,
        "velocity_Pearson": False,
        "mean_module_Pearson": False,
        "mean_module_peak_time_error": True,
        "all_gene_direction_agreement": False,
    }
    rank_rows = []
    for metric, ascending in rank_specs.items():
        values = unified.set_index("method")[metric]
        ranks = values.rank(ascending=ascending, method="min")
        for method in METHODS:
            rank_rows.append(
                {
                    "metric": metric,
                    "lower_is_better": ascending,
                    "method": method,
                    "value": values.get(method, np.nan),
                    "rank_among_available_methods": ranks.get(method, np.nan),
                    "available": bool(np.isfinite(values.get(method, np.nan))),
                }
            )
    rankings = pd.DataFrame(rank_rows)
    rankings.to_csv(OUT / "method_rankings_by_metric.csv", index=False)

    plot_figure(
        dnb,
        distributions,
        progressions,
        velocities,
        module_z,
        modules,
        unified,
        rankings,
    )
    write_summary(
        genes,
        unified,
        module_df,
        direction_df,
        native_rows,
        constant_rows,
    )


def plot_figure(
    dnb, distributions, progressions, velocities, module_z, modules, unified, rankings
):
    fig = plt.figure(figsize=(18, 15), facecolor="white")
    outer = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 1.35], hspace=0.34, wspace=0.22
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
            distributions,
            TIMES,
        ),
        (
            "C",
            "Correlation-distance transcriptomic progression",
            progressions,
            TIMES,
        ),
        (
            "D",
            "Time-interval-normalized transcriptomic velocity",
            velocities,
            TIMES[1:],
        ),
    ]
    for ax, (letter, title, curves, x) in zip(axes, panels):
        for method in ["Real", *METHODS]:
            y = curves[method]
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
        ax.text(
            0.5,
            0.48,
            "chronODE piecewise\nunavailable",
            transform=ax.transAxes,
            color=COLORS["chronODE piecewise"],
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold", fontsize=12)
        ax.set_xlabel("Time (h)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Within-method normalized value")
    axes[2].set_ylabel("Distance")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    egrid = outer[2, 0].subgridspec(1, 4, wspace=0.12)
    heat_axes = []
    vmax = max(
        np.nanmax(np.abs(module_z[m]))
        for m in ["Real", "RVAgene", "Online Agent pilot"]
    )
    for i, method in enumerate(["Real", *METHODS]):
        ax = fig.add_subplot(egrid[0, i])
        heat_axes.append(ax)
        matrix = module_z[method].T
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(method, fontsize=9)
        ax.set_xticks(range(len(TIMES)), [f"{x:g}" for x in TIMES], rotation=45)
        if i == 0:
            ax.set_yticks(range(len(modules)), modules.module, fontsize=7)
            ax.set_ylabel("Frozen functional program")
        else:
            ax.set_yticks([])
        ax.set_xlabel("h", fontsize=8)
        if method == "chronODE piecewise":
            ax.text(
                0.5,
                0.5,
                "Unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontsize=9,
                fontweight="bold",
            )
    heat_axes[0].text(
        -0.48,
        1.08,
        "E. Functional-program dynamics",
        transform=heat_axes[0].transAxes,
        fontweight="bold",
        fontsize=12,
    )
    cbar = fig.colorbar(im, ax=heat_axes, fraction=0.025, pad=0.02)
    cbar.ax.set_title("z", fontsize=8, pad=3)

    axf = fig.add_subplot(outer[2, 1])
    metric_order = [
        "DNB_peak_time_error",
        "DNB_Pearson",
        "DNB_DTW",
        "distribution_shift_Pearson",
        "distribution_shift_DTW",
        "progression_Pearson",
        "progression_DTW",
        "velocity_Pearson",
        "mean_module_Pearson",
        "mean_module_peak_time_error",
        "all_gene_direction_agreement",
    ]
    labels = [
        "DNB peak error ↓",
        "DNB Pearson ↑",
        "DNB DTW ↓",
        "Distribution Pearson ↑",
        "Distribution DTW ↓",
        "Progression Pearson ↑",
        "Progression DTW ↓",
        "Velocity Pearson ↑",
        "Mean module Pearson ↑",
        "Module peak error ↓",
        "Gene-direction agreement ↑",
    ]
    table = (
        rankings.pivot(index="metric", columns="method", values="value")
        .reindex(index=metric_order, columns=METHODS)
    )
    display = np.full(table.shape, np.nan)
    for i, metric in enumerate(metric_order):
        row = table.loc[metric].to_numpy(float)
        finite = np.isfinite(row)
        if finite.any():
            best = -row if metric.endswith("error") or metric.endswith("DTW") else row
            display[i, finite] = mm(best[finite])
    cmap = plt.get_cmap("YlGn")
    norm = plt.Normalize(0, 1)
    for i in range(len(metric_order)):
        for j in range(len(METHODS)):
            value = table.iloc[i, j]
            score = display[i, j]
            if np.isfinite(value):
                axf.scatter(
                    j,
                    i,
                    s=85 + 260 * score,
                    c=[cmap(norm(score))],
                    edgecolors="#303030",
                    linewidths=0.6,
                    zorder=2,
                )
                axf.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    zorder=3,
                )
            else:
                axf.scatter(j, i, marker="x", s=45, color="#B0B0B0")
                axf.text(j + 0.08, i, "NA", va="center", fontsize=6, color="#888888")
    axf.set_xticks(range(len(METHODS)), ["chronODE", "RVAgene", "Agent pilot"])
    axf.set_yticks(range(len(labels)), labels, fontsize=8)
    axf.set_xlim(-0.5, len(METHODS) - 0.5)
    axf.set_ylim(len(labels) - 0.5, -0.5)
    axf.grid(axis="both", alpha=0.18)
    axf.set_title("F. Cross-method metric dot plot", loc="left", fontweight="bold")
    axf.tick_params(axis="x", labelrotation=20)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(
        scalar,
        ax=axf,
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
        0.967,
        "External methods are fitted to the complete observed bulk trajectory, "
        "whereas the Online Agent pilot does not use the observed expression trajectory.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.012,
        "chronODE and RVAgene are data-fitted reconstruction baselines; the "
        "Online Agent is not an equal-input baseline. Comparisons emphasize "
        "normalized temporal dynamics. Raw cross-platform expression errors "
        "are not reported for the six-run Online Agent pilot. The official "
        "chronODE piecewise primary remained unavailable after the permitted "
        "syntax-only patch.",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    png = OUT / "GSE2565_bulk_trajectory_reconstruction_benchmark_600dpi.png"
    pdf = OUT / "GSE2565_bulk_trajectory_reconstruction_benchmark_vector.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    genes, unified, module_df, direction_df, native_rows, constant_rows
):
    available = unified.copy()
    def best(metric, lower=False):
        x = available[np.isfinite(available[metric])]
        if x.empty:
            return "not estimable"
        row = x.loc[x[metric].idxmin() if lower else x[metric].idxmax()]
        return f"{row.method} ({row[metric]:.3f})"

    peaks = {
        row.method: (
            None if not np.isfinite(row.DNB_peak_time) else float(row.DNB_peak_time)
        )
        for _, row in unified.iterrows()
    }
    summary = f"""# GSE2565 bulk trajectory reconstruction benchmark

## Status

**COMPLETE_WITH_DECLARED_CHRONODE_PIECEWISE_UNAVAILABLE**

The A--F figure and all feasible tables were generated. The official chronODE
piecewise primary is shown as unavailable rather than being replaced by the
monotonic sensitivity fit.

## Method execution

- chronODE piecewise: syntax-only patch failed; 0/{len(genes)} genes fitted.
- chronODE monotonic sensitivity: 12,259/{len(genes)} genes fitted.
- RVAgene: all three seeds completed native 9-point reconstruction.
- Online Agent: reused the existing six-run pilot; no simulation or LLM call.

External baselines used the complete observed trajectory. The Online Agent did
not use observed bulk expression. This is not an equal-input comparison and
none of the external reconstruction results demonstrate forecasting ability.

## DNB peak recovery

- Real reference peak: 8 h.
- chronODE piecewise: unavailable.
- RVAgene peak: {peaks.get("RVAgene")} h.
- Online Agent pilot peak: {peaks.get("Online Agent pilot")} h.

## Available-method metric leaders

- DNB trajectory Pearson: {best("DNB_Pearson")}
- Distribution-shift Pearson: {best("distribution_shift_Pearson")}
- Progression Pearson: {best("progression_Pearson")}
- Velocity Pearson: {best("velocity_Pearson")}
- Mean module Pearson: {best("mean_module_Pearson")}
- Gene-direction agreement: {best("all_gene_direction_agreement")}

Ranks exclude unavailable chronODE piecewise values and Real, which is the
reference. Native reconstruction RMSE/MAE/Pearson/CCC apply only to the
microarray-scale data-fitted baselines and must not be compared with Agent raw
expression.

## Interpretation

This remains an exploratory reconstruction benchmark because the current Agent
is a six-run pilot with `formal_not_run=true`. A formal Agent rerun and a new
external dataset not involved in system development are still required.
"""
    (OUT / "BENCHMARK_SUMMARY.md").write_text(summary)
    audit = {
        "status": "COMPLETE_WITH_DECLARED_CHRONODE_PIECEWISE_UNAVAILABLE",
        "figure_generated": True,
        "figure_contains_panels_A_to_F": True,
        "piecewise_syntax_patch_whitespace_only": True,
        "piecewise_patch_successful": False,
        "piecewise_failure_reason": "missing control-flow cannot be restored by indentation alone; official pipeline hard-codes eight timepoints",
        "chronode_piecewise_success_genes": 0,
        "chronode_piecewise_failed_genes": len(genes),
        "chronode_monotonic_success_genes": 12259,
        "chronode_monotonic_failed_genes": 4,
        "rvagene_three_seeds_successful": True,
        "rvagene_task": "native complete-sequence reconstruction",
        "external_methods_complete_observed_trajectory_input": True,
        "online_agent_observed_expression_input": False,
        "online_agent_runs": 6,
        "online_agent_formal_not_run": True,
        "raw_cross_platform_error_for_agent_reported": False,
        "bootstrap_iterations": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "constant_trajectory_counts": constant_rows,
        "frozen_module_table": str(MODULE_PATH.relative_to(ROOT)),
        "frozen_module_table_sha256": sha(MODULE_PATH),
        "forecasting_interpretation_allowed": False,
        "equal_input_interpretation_allowed": False,
        "exploratory_before_formal_agent_rerun": True,
    }
    (OUT / "BENCHMARK_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    evaluate()
