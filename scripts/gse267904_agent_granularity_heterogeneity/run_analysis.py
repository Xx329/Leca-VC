#!/usr/bin/env python3
"""Quantify GSE267904 heterogeneity recovery across Leca-VC Agent granularity.

This post-hoc analysis is intentionally read-only with respect to all frozen
upstream results. It does not run an LLM, PhysiCell, BioFVM, or COMMOT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse267904-heterogeneity")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist, jensenshannon, pdist, squareform


RANDOM_SEED = 267904
N_REFERENCE_DRAWS = 1000
K_VALUES = (20, 50, 100)
MODULES = {
    "TGF-beta response": ["Tgfb1", "Tgfbr1", "Tgfbr2", "Serpine1", "Col1a1", "Fn1"],
    "Inflammation": ["Il1b", "Tnf", "Nfkbia", "Ccl2", "Cxcl2", "Cxcl10"],
    "Fibrosis / ECM": ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Acta2", "Tagln"],
}
STAGES = {
    "Early": ("d7_bleo", "virtual_early_bleo"),
    "Late": ("d21_bleo", "virtual_late_bleo"),
}
CCI_FAMILIES = ("TGFB", "CCL", "CXCL", "PDGF", "VEGF")
EXPECTED_AGENT_COUNTS = {
    ("Early", 20): 13,
    ("Early", 50): 41,
    ("Early", 100): 93,
    ("Late", 20): 8,
    ("Late", 50): 28,
    ("Late", 100): 83,
}
METHOD_COLORS = {20: "#9CA3AF", 50: "#27A6A1", 100: "#7C3AED"}
STAGE_COLORS = {"Early": "#D55E00", "Late": "#0072B2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_module_scores(path: Path) -> tuple[np.ndarray, dict]:
    data = ad.read_h5ad(path, backed="r")
    scores: list[np.ndarray] = []
    transforms: dict[str, str] = {}
    for module, genes in MODULES.items():
        missing = sorted(set(genes) - set(data.var_names))
        if missing:
            data.file.close()
            raise RuntimeError(f"{path} is missing genes for {module}: {missing}")
        subset = data[:, genes].to_memory()
        matrix = subset.X.toarray() if sparse.issparse(subset.X) else np.asarray(subset.X)
        matrix = np.asarray(matrix, dtype=float)
        raw_max = float(np.nanmax(matrix)) if matrix.size else 0.0
        transform = "log1p" if raw_max > 20 else "already_log_scale"
        if transform == "log1p":
            matrix = np.log1p(np.clip(matrix, 0, None))
        scores.append(np.nanmean(matrix, axis=1))
        transforms[module] = transform
    n_obs = int(data.n_obs)
    data.file.close()
    return np.column_stack(scores), {"n_obs": n_obs, "transforms": transforms}


def energy_squared(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cross = float(cdist(x, y).mean())
    within_x = float(2.0 * pdist(x).sum() / (len(x) ** 2)) if len(x) > 1 else 0.0
    within_y = float(2.0 * pdist(y).sum() / (len(y) ** 2)) if len(y) > 1 else 0.0
    return max(0.0, 2.0 * cross - within_x - within_y)


def energy_reference_draws(
    real_z: np.ndarray, sample_size: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact energy distance of a Real subset against its held-out complement."""
    n_total = len(real_z)
    if sample_size <= 1 or sample_size >= n_total:
        raise ValueError("capacity-matched sample size must be in [2, n_real - 1]")
    distances = squareform(pdist(real_z)).astype(np.float32, copy=False)
    row_sums = distances.sum(axis=1, dtype=np.float64)
    total_sum = float(row_sums.sum())
    n_held = n_total - sample_size
    values = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        selected = rng.choice(n_total, size=sample_size, replace=False)
        within_selected_sum = float(distances[np.ix_(selected, selected)].sum(dtype=np.float64))
        cross_sum = float(row_sums[selected].sum() - within_selected_sum)
        within_held_sum = total_sum - within_selected_sum - 2.0 * cross_sum
        value = (
            2.0 * cross_sum / (sample_size * n_held)
            - within_selected_sum / (sample_size**2)
            - within_held_sum / (n_held**2)
        )
        values[draw] = max(0.0, value)
    return values


def state_ids(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    high = (np.asarray(scores) > np.asarray(thresholds)).astype(np.int8)
    return high[:, 0] * 4 + high[:, 1] * 2 + high[:, 2]


def state_probabilities(ids: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(ids, dtype=int), minlength=8).astype(float) / len(ids)


def js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    return float(1.0 - jensenshannon(p, q, base=2.0) ** 2)


def observed_mass_coverage(observed_p: np.ndarray, predicted_p: np.ndarray) -> float:
    return float(np.asarray(observed_p)[np.asarray(predicted_p) > 0].sum())


def effective_diversity(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    positive = p[p > 0]
    if not len(positive):
        return 0.0
    positive = positive / positive.sum()
    return float(np.exp(-(positive * np.log(positive)).sum()))


def core_mass_recovery(observed_p: np.ndarray, predicted_p: np.ndarray, mass: float = 0.8) -> tuple[float, int]:
    observed_p = np.asarray(observed_p, dtype=float)
    predicted_p = np.asarray(predicted_p, dtype=float)
    order = np.argsort(observed_p)[::-1]
    n_core = int(np.searchsorted(np.cumsum(observed_p[order]), mass, side="left") + 1)
    core = order[:n_core]
    denominator = float(observed_p[core].sum())
    recovery = float(np.minimum(observed_p[core], predicted_p[core]).sum() / denominator)
    return recovery, n_core


def classify_cci_family(pathways: pd.Series) -> pd.Series:
    family = pd.Series(pd.NA, index=pathways.index, dtype="string")
    text = pathways.astype(str)
    for name in CCI_FAMILIES:
        family.loc[text.str.contains(name, case=False, regex=False, na=False)] = name
    return family


def cci_vector(
    edges: pd.DataFrame,
    stage: str,
    cell_types: list[str],
    include_self: bool,
) -> tuple[pd.Series, float]:
    keys = pd.MultiIndex.from_product(
        [CCI_FAMILIES, cell_types, cell_types], names=["family", "sender", "receiver"]
    )
    if not include_self:
        keys = keys[keys.get_level_values("sender") != keys.get_level_values("receiver")]
    selected = edges[
        edges["stage"].astype(str).eq(stage)
        & edges["sender"].astype(str).isin(cell_types)
        & edges["receiver"].astype(str).isin(cell_types)
    ].copy()
    if not include_self:
        selected = selected[selected["sender"].astype(str) != selected["receiver"].astype(str)]
    selected["family"] = classify_cci_family(selected["pathway"])
    selected = selected[selected["family"].notna()].copy()
    selected["weight"] = pd.to_numeric(selected["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weights = (
        selected.groupby(["family", "sender", "receiver"], observed=True)["weight"]
        .sum()
        .reindex(keys, fill_value=0.0)
        .astype(float)
    )
    total = float(weights.sum())
    probabilities = weights / total if total > 0 else weights.copy()
    return probabilities, total


def summarize_reference(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def add_value_labels(
    ax: plt.Axes, table: pd.DataFrame, metric: str, decimals: int, log_scale: bool = False,
    offset_overrides: dict[tuple[str, int], tuple[int, int]] | None = None,
) -> None:
    """Add stage-coloured point labels with opposing offsets."""
    for stage in STAGES:
        group = table[table["stage"] == stage].sort_values("K")
        for row in group.itertuples(index=False):
            value = float(getattr(row, metric))
            if not np.isfinite(value):
                continue
            if stage == "Early":
                offset = (-5, -16) if log_scale else (0, 8)
                vertical_alignment = "top" if log_scale else "bottom"
            else:
                offset = (5, 8) if log_scale or value <= 0.02 else (0, -13)
                vertical_alignment = "bottom" if log_scale or value <= 0.02 else "top"
            if offset_overrides and (stage, int(row.K)) in offset_overrides:
                offset = offset_overrides[(stage, int(row.K))]
                vertical_alignment = "bottom" if offset[1] >= 0 else "top"
            ax.annotate(
                f"{value:.{decimals}f}", (float(row.K), value), xytext=offset,
                textcoords="offset points", ha="center", va=vertical_alignment,
                fontsize=7.6, fontweight="semibold", color=STAGE_COLORS[stage],
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.76},
                zorder=6,
            )


def plot_stage_curves(
    ax: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    log_scale: bool = False,
    reference_table: pd.DataFrame | None = None,
    pooled_reference_band: bool = False,
    explicit_stage_labels: bool = False,
) -> None:
    for stage in STAGES:
        group = table[table["stage"] == stage].sort_values("K")
        ax.plot(
            group["K"], group[metric], marker="o" if stage == "Early" else "s",
            markersize=6.5, linewidth=2.2, color=STAGE_COLORS[stage],
            linestyle="-" if stage == "Early" else "--",
            label=(f"d7 ({stage.lower()})" if stage == "Early" else f"d21 ({stage.lower()})")
            if explicit_stage_labels else f"{stage} Leca-VC",
        )
        if reference_table is not None and not pooled_reference_band:
            ref = reference_table[reference_table["stage"] == stage].sort_values("K")
            ax.plot(
                ref["K"], ref[f"{metric}_median"], linewidth=1.4,
                color=STAGE_COLORS[stage], alpha=0.45, linestyle=":" if stage == "Early" else "-.",
                label=f"{stage} matched Real",
            )
            ax.fill_between(
                ref["K"].to_numpy(float), ref[f"{metric}_lower_95"].to_numpy(float),
                ref[f"{metric}_upper_95"].to_numpy(float), color=STAGE_COLORS[stage], alpha=0.08,
            )
    if reference_table is not None and pooled_reference_band:
        pooled = reference_table.groupby("K", sort=True).agg(
            lower_95=(f"{metric}_lower_95", "min"),
            upper_95=(f"{metric}_upper_95", "max"),
            median=(f"{metric}_median", "median"),
        ).reset_index()
        x = pooled["K"].to_numpy(float)
        ax.fill_between(
            x, pooled["lower_95"].to_numpy(float), pooled["upper_95"].to_numpy(float),
            color="#9CA3AF", alpha=0.20, linewidth=0, label="Equal-size Real reference",
        )
        ax.plot(
            x, pooled["median"].to_numpy(float), color="#6B7280", linewidth=1.25,
            linestyle=":", alpha=0.85,
        )
    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.set_xlabel("Nominal Agent granularity (K)", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log_scale:
        ax.set_yscale("log")
    style_axis(ax)


def save_figure(
    fig: plt.Figure, figure_dir: Path, refined_display: bool, clean_display: bool,
    labeled_display: bool, publication_display: bool, paper_title_display: bool,
) -> dict[str, str]:
    stem = "GSE267904_LecaVC_agent_granularity_heterogeneity"
    if paper_title_display:
        stem += "_paper_title"
    elif publication_display:
        stem += "_publication"
    elif labeled_display:
        stem += "_labeled"
    elif clean_display:
        stem += "_clean"
    elif refined_display:
        stem += "_refined"
    outputs = {
        "preview_png": figure_dir / f"{stem}_preview.png",
        "highres_png": figure_dir / f"{stem}_600dpi.png",
        "vector_pdf": figure_dir / f"{stem}_vector.pdf",
        "vector_svg": figure_dir / f"{stem}_vector.svg",
    }
    png_metadata = {"Software": "Leca-VC GSE267904 heterogeneity audit"}
    fixed_date = datetime(2026, 9, 6, tzinfo=timezone.utc)
    pdf_metadata = {
        "Title": "GSE267904 Leca-VC Agent granularity heterogeneity",
        "Creator": "Leca-VC frozen post-hoc audit",
        "CreationDate": fixed_date,
        "ModDate": fixed_date,
    }
    svg_metadata = {"Title": "GSE267904 Leca-VC Agent granularity heterogeneity", "Date": "2026-09-06"}
    fig.savefig(outputs["preview_png"], dpi=160, bbox_inches="tight", facecolor="white", metadata=png_metadata)
    fig.savefig(outputs["highres_png"], dpi=600, bbox_inches="tight", facecolor="white", metadata=png_metadata)
    fig.savefig(outputs["vector_pdf"], bbox_inches="tight", facecolor="white", metadata=pdf_metadata)
    fig.savefig(outputs["vector_svg"], bbox_inches="tight", facecolor="white", metadata=svg_metadata)
    return {key: str(path) for key, path in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frozen-root", type=Path, default=None)
    parser.add_argument("--allow-new-agent-counts", action="store_true")
    parser.add_argument(
        "--resume-failed-validation",
        action="store_true",
        help="Resume only an output containing the prior validation_failed.json and no completion marker.",
    )
    parser.add_argument("--reference-draws", type=int, default=N_REFERENCE_DRAWS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--refined-display", action="store_true",
        help="Use a pooled grey matched-Real band and show a no-signal CCI case as zero recovery.",
    )
    parser.add_argument(
        "--clean-display", action="store_true",
        help="Hide matched-Real controls and replace CCI JS similarity with state-mass coverage.",
    )
    parser.add_argument(
        "--labeled-display", action="store_true",
        help="Use the clean display with explicit d7/d21 names and point-value labels.",
    )
    parser.add_argument(
        "--publication-display", action="store_true",
        help="Use the labeled display with a conclusion-led title and collision-adjusted layout.",
    )
    parser.add_argument(
        "--paper-title-display", action="store_true",
        help="Use the publication layout with a concise noun-phrase figure title.",
    )
    args = parser.parse_args()
    if args.refined_display and (
        args.clean_display or args.labeled_display or args.publication_display or args.paper_title_display
    ):
        raise ValueError("--refined-display cannot be combined with clean or labeled display")
    publication_mode = args.publication_display or args.paper_title_display
    labeled_mode = args.labeled_display or publication_mode
    clean_mode = args.clean_display or labeled_mode

    root = args.project_root.resolve()
    frozen = args.frozen_root.resolve() if args.frozen_root is not None else root / "outputs/GSE267904_spatial_agent_granularity"
    if args.paper_title_display:
        default_output_name = "GSE267904_agent_granularity_heterogeneity_v1_5_paper_title"
    elif args.publication_display:
        default_output_name = "GSE267904_agent_granularity_heterogeneity_v1_4_publication"
    elif args.labeled_display:
        default_output_name = "GSE267904_agent_granularity_heterogeneity_v1_3_labeled"
    elif args.clean_display:
        default_output_name = "GSE267904_agent_granularity_heterogeneity_v1_2_clean"
    elif args.refined_display:
        default_output_name = "GSE267904_agent_granularity_heterogeneity_v1_1_refined"
    else:
        default_output_name = "GSE267904_agent_granularity_heterogeneity_v1"
    output = args.output_dir.resolve() if args.output_dir is not None else root / f"outputs/{default_output_name}"
    if output.exists() and any(output.iterdir()):
        existing = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
        safe_failed_output = (
            args.resume_failed_validation
            and "audit/validation_failed.json" in existing
            and "COMPLETE.json" not in existing
            and existing <= {"audit/validation_failed.json"}
        )
        if not safe_failed_output:
            raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    data_dir = output / "data"
    figure_dir = output / "figure"
    audit_dir = output / "audit"
    for directory in (data_dir, figure_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    real_h5ad = {
        stage: frozen / f"shared_real/real_benchmark/{real_name}.h5ad"
        for stage, (real_name, _) in STAGES.items()
    }
    virtual_h5ad = {
        (stage, k): frozen / f"K_{k}/virtual_commot_input/{virtual_name}.h5ad"
        for stage, (_, virtual_name) in STAGES.items() for k in K_VALUES
    }
    real_edges_path = frozen / "shared_real/real_commot/real_commot_edges_long.csv"
    virtual_edge_paths = {
        k: frozen / f"K_{k}/virtual_commot/virtual_commot_edges_long.csv" for k in K_VALUES
    }
    input_paths = sorted(set(real_h5ad.values()) | set(virtual_h5ad.values()) | {real_edges_path} | set(virtual_edge_paths.values()))
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing frozen inputs: {missing_inputs}")

    rng = np.random.default_rng(args.random_seed)
    module_metric_rows: list[dict] = []
    module_score_rows: list[dict] = []
    module_state_rows: list[dict] = []
    reference_draw_rows: list[dict] = []
    reference_summary_rows: list[dict] = []
    module_source_audit: dict[str, dict] = {}

    for stage in STAGES:
        real_scores, real_info = read_module_scores(real_h5ad[stage])
        module_source_audit[f"{stage}|Real"] = real_info
        means = real_scores.mean(axis=0)
        standard_deviations = real_scores.std(axis=0, ddof=1)
        if np.any(standard_deviations <= 0):
            raise RuntimeError(f"Non-positive Real module standard deviation in {stage}")
        real_z = (real_scores - means) / standard_deviations
        thresholds = np.median(real_scores, axis=0)
        real_ids = state_ids(real_scores, thresholds)
        real_p = state_probabilities(real_ids)

        for index, values in enumerate(real_scores):
            module_score_rows.append({
                "stage": stage, "source": "Real", "K": np.nan, "entity_index": index,
                **{module: float(values[j]) for j, module in enumerate(MODULES)},
                "module_state_id": int(real_ids[index]),
            })
        for state_id in range(8):
            module_state_rows.append({
                "stage": stage, "source": "Real", "K": np.nan, "state_id": state_id,
                "state_label": f"{state_id:03b}", "count": int((real_ids == state_id).sum()),
                "proportion": float(real_p[state_id]),
            })

        for k in K_VALUES:
            virtual_scores, virtual_info = read_module_scores(virtual_h5ad[(stage, k)])
            module_source_audit[f"{stage}|K={k}"] = virtual_info
            actual_n = int(len(virtual_scores))
            expected_n = EXPECTED_AGENT_COUNTS[(stage, k)]
            if actual_n != expected_n and not args.allow_new_agent_counts:
                raise RuntimeError(f"Unexpected Agent count for {stage}, K={k}: {actual_n} != {expected_n}")
            virtual_z = (virtual_scores - means) / standard_deviations
            virtual_ids = state_ids(virtual_scores, thresholds)
            virtual_p = state_probabilities(virtual_ids)
            energy = energy_squared(virtual_z, real_z)
            coverage = observed_mass_coverage(real_p, virtual_p)
            similarity = js_similarity(real_p, virtual_p)
            module_metric_rows.append({
                "stage": stage, "K": k, "actual_agent_count": actual_n,
                "energy_distance_squared": energy,
                "observed_state_mass_coverage": coverage,
                "state_js_similarity": similarity,
                "n_module_states_detected": int((virtual_p > 0).sum()),
            })
            for index, values in enumerate(virtual_scores):
                module_score_rows.append({
                    "stage": stage, "source": "Leca-VC", "K": k, "entity_index": index,
                    **{module: float(values[j]) for j, module in enumerate(MODULES)},
                    "module_state_id": int(virtual_ids[index]),
                })
            for state_id in range(8):
                module_state_rows.append({
                    "stage": stage, "source": "Leca-VC", "K": k, "state_id": state_id,
                    "state_label": f"{state_id:03b}", "count": int((virtual_ids == state_id).sum()),
                    "proportion": float(virtual_p[state_id]),
                })

            energy_draws = energy_reference_draws(real_z, actual_n, args.reference_draws, rng)
            for draw, energy_draw in enumerate(energy_draws):
                sampled = rng.choice(len(real_ids), size=actual_n, replace=False)
                sampled_p = state_probabilities(real_ids[sampled])
                reference_draw_rows.append({
                    "stage": stage, "K": k, "actual_agent_count": actual_n, "draw": draw,
                    "energy_distance_squared": float(energy_draw),
                    "observed_state_mass_coverage": observed_mass_coverage(real_p, sampled_p),
                    "state_js_similarity": js_similarity(real_p, sampled_p),
                })

    module_metrics = pd.DataFrame(module_metric_rows)
    reference_draws = pd.DataFrame(reference_draw_rows)
    for (stage, k), group in reference_draws.groupby(["stage", "K"], sort=False):
        row = {"stage": stage, "K": int(k)}
        for metric in ("energy_distance_squared", "observed_state_mass_coverage", "state_js_similarity"):
            summary = summarize_reference(group[metric].to_numpy(float))
            row.update({f"{metric}_{key}": value for key, value in summary.items()})
        reference_summary_rows.append(row)
    reference_summary = pd.DataFrame(reference_summary_rows)

    real_edges = pd.read_csv(real_edges_path)
    virtual_edges = {k: pd.read_csv(path) for k, path in virtual_edge_paths.items()}
    cell_types = sorted((set(real_edges["sender"].astype(str)) | set(real_edges["receiver"].astype(str))) - {"other"})
    if len(cell_types) != 9:
        raise RuntimeError(f"Expected nine manuscript CCI cell types after excluding 'other'; found {cell_types}")

    cci_metric_rows: list[dict] = []
    cci_sensitivity_rows: list[dict] = []
    cci_state_rows: list[dict] = []
    for include_self, analysis_name, metric_sink in (
        (True, "primary_all_sender_receiver_states", cci_metric_rows),
        (False, "sensitivity_cross_type_only", cci_sensitivity_rows),
    ):
        expected_states = 405 if include_self else 360
        for stage, (real_stage, virtual_stage) in STAGES.items():
            observed_p, observed_total = cci_vector(real_edges, real_stage, cell_types, include_self)
            if len(observed_p) != expected_states or observed_total <= 0:
                raise RuntimeError(f"Invalid Observed CCI vector for {analysis_name}, {stage}")
            observed_d1 = effective_diversity(observed_p.to_numpy())
            if include_self:
                for key, probability in observed_p.items():
                    cci_state_rows.append({
                        "analysis": analysis_name, "stage": stage, "source": "Real", "K": np.nan,
                        "family": key[0], "sender": key[1], "receiver": key[2],
                        "probability": float(probability),
                    })
            for k in K_VALUES:
                predicted_p, predicted_total = cci_vector(virtual_edges[k], virtual_stage, cell_types, include_self)
                no_signal = predicted_total <= 0
                predicted_array = predicted_p.to_numpy(float)
                observed_array = observed_p.to_numpy(float)
                similarity = js_similarity(observed_array, predicted_array)
                core_recovery, n_core = core_mass_recovery(observed_array, predicted_array)
                predicted_d1 = effective_diversity(predicted_array)
                diversity_ratio = predicted_d1 / observed_d1 if observed_d1 > 0 else float("nan")
                metric_sink.append({
                    "analysis": analysis_name, "stage": stage, "K": k,
                    "state_space_size": expected_states, "include_same_type": include_self,
                    "observed_total_weight": observed_total,
                    "predicted_total_weight": predicted_total, "no_detectable_signal": no_signal,
                    "cci_state_js_similarity": similarity,
                    "core_80_mass_recovery": core_recovery,
                    "n_observed_core_states": n_core,
                    "observed_effective_diversity": observed_d1,
                    "predicted_effective_diversity": predicted_d1,
                    "effective_diversity_ratio": diversity_ratio,
                    "n_predicted_states_with_positive_weight": int((predicted_array > 0).sum()),
                    "observed_mass_on_predicted_support": float(observed_array[predicted_array > 0].sum()),
                })
                if include_self:
                    for key, probability in predicted_p.items():
                        cci_state_rows.append({
                            "analysis": analysis_name, "stage": stage, "source": "Leca-VC", "K": k,
                            "family": key[0], "sender": key[1], "receiver": key[2],
                            "probability": float(probability),
                        })

    cci_metrics = pd.DataFrame(cci_metric_rows)
    cci_sensitivity = pd.DataFrame(cci_sensitivity_rows)

    # Numerical validation before any result is declared complete.
    synthetic = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.5, 1.5]])
    test_p = np.array([0.1, 0.2, 0.3, 0.4])
    identical_core, _ = core_mass_recovery(test_p, test_p)
    validations = {
        "synthetic_identical_energy_distance_squared": energy_squared(synthetic, synthetic),
        "synthetic_identical_js_similarity": js_similarity(test_p, test_p),
        "synthetic_identical_core_recovery": identical_core,
        "synthetic_identical_diversity_ratio": effective_diversity(test_p) / effective_diversity(test_p),
        "primary_cci_state_space_size": int(cci_metrics["state_space_size"].unique()[0]),
        "sensitivity_cci_state_space_size": int(cci_sensitivity["state_space_size"].unique()[0]),
        "module_actual_agent_counts": {
            f"{row.stage}|K={int(row.K)}": int(row.actual_agent_count)
            for row in module_metrics.itertuples()
        },
        "primary_probability_sums": [],
    }
    primary_no_signal_cases = [
        {"stage": str(row.stage), "K": int(row.K)}
        for row in cci_metrics.itertuples()
        if bool(row.no_detectable_signal)
    ]
    validations["primary_no_signal_cases"] = primary_no_signal_cases
    cci_states_frame = pd.DataFrame(cci_state_rows)
    for key, group in cci_states_frame.groupby(["stage", "source", "K"], dropna=False):
        total = float(group["probability"].sum())
        no_signal = bool(group["probability"].eq(0).all())
        validations["primary_probability_sums"].append({
            "stage": key[0], "source": key[1], "K": None if pd.isna(key[2]) else int(key[2]),
            "sum": total, "no_signal": no_signal,
        })

    checks = {
        "identical_energy_is_zero": abs(validations["synthetic_identical_energy_distance_squared"]) < 1e-12,
        "identical_js_is_one": abs(validations["synthetic_identical_js_similarity"] - 1.0) < 1e-12,
        "identical_core_is_one": abs(validations["synthetic_identical_core_recovery"] - 1.0) < 1e-12,
        "identical_diversity_ratio_is_one": abs(validations["synthetic_identical_diversity_ratio"] - 1.0) < 1e-12,
        "primary_state_space_is_405": validations["primary_cci_state_space_size"] == 405,
        "cross_type_state_space_is_360": validations["sensitivity_cci_state_space_size"] == 360,
        "all_nonempty_probability_vectors_sum_to_one": all(
            item["no_signal"] or abs(item["sum"] - 1.0) < 1e-10
            for item in validations["primary_probability_sums"]
        ),
        "no_signal_flags_match_nonpositive_total_weight": all(
            bool(row.no_detectable_signal) == (float(row.predicted_total_weight) <= 0.0)
            for row in cci_metrics.itertuples()
        ),
    }
    validations["checks"] = checks
    if not all(checks.values()):
        write_json(audit_dir / "validation_failed.json", validations)
        raise RuntimeError(f"Validation failed: {[key for key, value in checks.items() if not value]}")

    module_metrics.to_csv(data_dir / "module_heterogeneity_metrics.csv", index=False)
    pd.DataFrame(module_score_rows).to_csv(data_dir / "module_scores_and_states.csv", index=False)
    pd.DataFrame(module_state_rows).to_csv(data_dir / "module_state_distributions.csv", index=False)
    reference_draws.to_csv(data_dir / "capacity_matched_real_draws.csv", index=False)
    reference_summary.to_csv(data_dir / "capacity_matched_real_summary.csv", index=False)
    cci_metrics.to_csv(data_dir / "cci_heterogeneity_metrics_primary.csv", index=False)
    cci_sensitivity.to_csv(data_dir / "cci_heterogeneity_metrics_cross_type_sensitivity.csv", index=False)
    cci_states_frame.to_csv(data_dir / "cci_state_probabilities_primary.csv", index=False)

    matplotlib.rcParams["svg.hashsalt"] = "GSE267904-agent-granularity-heterogeneity-v1"
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.2), dpi=180, facecolor="white")
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.835 if publication_mode else 0.875,
        bottom=0.09, hspace=0.44, wspace=0.30,
    )

    reference_for_plot = None if clean_mode else reference_summary
    plot_stage_curves(
        axes[0, 0], module_metrics, "energy_distance_squared", "Squared energy distance",
        "Module-state distribution distance", log_scale=True, reference_table=reference_for_plot,
        pooled_reference_band=args.refined_display, explicit_stage_labels=labeled_mode,
    )
    plot_stage_curves(
        axes[0, 1], module_metrics, "observed_state_mass_coverage", "Observed state mass covered",
        "Module-state coverage", ylim=(0, 1.05), reference_table=reference_for_plot,
        pooled_reference_band=args.refined_display, explicit_stage_labels=labeled_mode,
    )
    plot_stage_curves(
        axes[0, 2], module_metrics, "state_js_similarity", "1 − Jensen–Shannon divergence",
        "Module-state proportional fidelity", ylim=(0, 1.05), reference_table=reference_for_plot,
        pooled_reference_band=args.refined_display, explicit_stage_labels=labeled_mode,
    )
    for ax, label in zip(axes[0], "ABC"):
        add_panel_label(ax, label)

    cci_plot_metrics = cci_metrics.copy()
    if args.refined_display:
        no_signal = cci_plot_metrics["no_detectable_signal"].astype(bool)
        cci_plot_metrics.loc[no_signal, "cci_state_js_similarity"] = 0.0
    if clean_mode:
        plot_stage_curves(
            axes[1, 0], cci_plot_metrics, "observed_mass_on_predicted_support",
            "Observed CCI-state mass covered", "CCI-state coverage", ylim=(0, 1.05),
            explicit_stage_labels=labeled_mode,
        )
    else:
        plot_stage_curves(
            axes[1, 0], cci_plot_metrics, "cci_state_js_similarity", "1 − Jensen–Shannon divergence",
            "CCI-state proportional fidelity", ylim=(0, 1.05), explicit_stage_labels=labeled_mode,
        )
    plot_stage_curves(
        axes[1, 1], cci_metrics, "core_80_mass_recovery", "Observed core mass recovered",
        "Core CCI-state recovery", ylim=(0, 1.05), explicit_stage_labels=labeled_mode,
    )
    plot_stage_curves(
        axes[1, 2], cci_metrics, "effective_diversity_ratio", "Effective diversity / observed",
        "CCI-state diversity recovery", ylim=(0, 1.05), explicit_stage_labels=labeled_mode,
    )
    axes[1, 2].axhline(1.0, color="#111827", linewidth=1.1, linestyle=":", alpha=0.75)
    no_signal_text = None
    if primary_no_signal_cases:
        for item in primary_no_signal_cases:
            for ax in axes[1]:
                ax.scatter(
                    [item["K"]], [0], marker="x", s=55, linewidths=1.8,
                    color=STAGE_COLORS[item["stage"]], zorder=5,
                )
        case_text = ", ".join(
            f"K={item['K']} {item['stage'].lower()}" for item in primary_no_signal_cases
        )
        no_signal_text = (
            f"{case_text}: no detectable CCI signal"
            + ("; shown as zero recovery" if args.refined_display else "")
        )
    if no_signal_text is not None:
        axes[1, 0].annotate(
            no_signal_text,
            xy=(primary_no_signal_cases[0]["K"], 0),
            xytext=(27, 0.16 if args.refined_display else 0.18),
            fontsize=8.2, color="#374151", arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 0.8},
        )
    for ax, label in zip(axes[1], "DEF"):
        add_panel_label(ax, label)

    if labeled_mode:
        panel_a_offsets = None
        if publication_mode:
            panel_a_offsets = {
                ("Early", 20): (-7, -17), ("Late", 20): (7, 9),
                ("Early", 50): (-8, -18), ("Late", 50): (8, 9),
                ("Early", 100): (-15, 9), ("Late", 100): (13, -13),
            }
        add_value_labels(
            axes[0, 0], module_metrics, "energy_distance_squared", decimals=2, log_scale=True,
            offset_overrides=panel_a_offsets,
        )
        add_value_labels(axes[0, 1], module_metrics, "observed_state_mass_coverage", decimals=3)
        add_value_labels(axes[0, 2], module_metrics, "state_js_similarity", decimals=3)
        add_value_labels(axes[1, 0], cci_plot_metrics, "observed_mass_on_predicted_support", decimals=3)
        add_value_labels(axes[1, 1], cci_metrics, "core_80_mass_recovery", decimals=3)
        add_value_labels(axes[1, 2], cci_metrics, "effective_diversity_ratio", decimals=3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.940 if publication_mode else 0.925),
        ncol=len(labels), frameon=False, fontsize=9.5,
    )
    if args.paper_title_display:
        figure_title = "GSE267904 State-Recovery Benchmark across Leca-VC Agent Granularities"
    elif args.publication_display:
        figure_title = "More Leca-VC Agents Improve State Recovery in GSE267904"
    else:
        figure_title = "GSE267904 heterogeneity recovery across increasing Leca-VC Agent granularity"
    fig.suptitle(
        figure_title, fontsize=16.0 if publication_mode else 16.5,
        fontweight="bold", y=0.985 if publication_mode else 0.975,
    )
    fig.text(0.012, 0.635, "MODULE-EXPRESSION\nSTATE SPACE", rotation=90, ha="center", va="center", fontsize=9.3, fontweight="bold", color="#4B5563")
    fig.text(0.012, 0.245, "COMMUNICATION\nSTATE SPACE", rotation=90, ha="center", va="center", fontsize=9.3, fontweight="bold", color="#4B5563")
    figure_outputs = save_figure(
        fig, figure_dir, refined_display=args.refined_display, clean_display=clean_mode,
        labeled_display=labeled_mode, publication_display=args.publication_display,
        paper_title_display=args.paper_title_display,
    )
    plt.close(fig)

    if clean_mode:
        reference_caption = (
            "Equal-size observed-reference subsampling controls were retained in the accompanying source data "
            "but omitted from the figure for visual clarity. "
        )
    elif args.refined_display:
        reference_caption = (
            "The grey bands and dotted centre lines in A-C show the pooled stage-specific 95% ranges and "
            "medians from 1,000 equally sized observed-reference subsamples. "
        )
    else:
        reference_caption = (
            "Dotted curves and shaded intervals in A-C show the median and 95% range from 1,000 equally "
            "sized observed-reference subsamples. "
        )
    if primary_no_signal_cases:
        case_text = ", ".join(
            f"K={item['K']} at the {item['stage'].lower()} stage"
            for item in primary_no_signal_cases
        )
        no_signal_caption = (
            f"No CCI signal was detected for {case_text}; Jensen-Shannon similarity is undefined "
            "for those cases in the source data"
            + (
                " and is displayed as zero recovery only to preserve visual continuity. "
                if args.refined_display else ". "
            )
        )
    else:
        no_signal_caption = "Detectable CCI signal was present for every evaluated K and stage. "
    cci_panel_d_caption = (
        "(D) Observed CCI-state probability mass represented by states with positive reconstructed weight. "
        if clean_mode else
        "(D) Jensen-Shannon similarity of the complete CCI-state distribution, where a state was defined by "
        "one of five pathway families (TGF-beta, CCL, CXCL, PDGF, or VEGF), a sender cell type, and a receiver "
        "cell type. "
    )
    count_text = ", ".join(
        f"{stage.lower()} K={int(row.K)}: {int(row.actual_agent_count)}"
        for stage in STAGES
        for row in module_metrics[module_metrics.stage.eq(stage)].sort_values("K").itertuples()
    )
    caption = (
        "Heterogeneity recovery across increasing Leca-VC Agent granularity in GSE267904. "
        "Early and late denote the d7 bleomycin and d21 bleomycin stages, respectively. "
        "(A) Squared energy distance between the observed and virtual three-dimensional module-expression "
        "state distributions (TGF-beta response, inflammation, and fibrosis/ECM); lower values indicate "
        "closer distributional recovery. (B) Observed probability mass represented by the module-expression "
        "states detected among virtual Agents. The eight states were defined by the high/low combinations "
        "of the three modules using stage-specific observed medians. (C) Jensen-Shannon similarity between "
        "the observed and virtual eight-state proportions. " + reference_caption + "Nominal K values "
        "used the evaluated live Agent counts (" + count_text + "). "
        + cci_panel_d_caption + "CCI states were defined by one of five pathway families (TGF-beta, CCL, CXCL, "
        "PDGF, or VEGF), a sender cell type, and a receiver cell type. "
        "(E) Recovery of the observed CCI states accounting for 80% of communication mass. "
        "(F) Ratio of predicted to observed Shannon effective CCI-state diversity; one denotes matched "
        "diversity. The primary CCI analysis included same-type and cross-type communication across the nine "
        "manuscript cell types; cross-type-only results are provided as a sensitivity analysis. "
        + no_signal_caption + "The results quantify recovery of predefined module-expression "
        "and communication states and do not demonstrate discovery of new cell types or native single-cell "
        "heterogeneity."
    )
    (figure_dir / "FIGURE_CAPTION.txt").write_text(caption + "\n", encoding="utf-8")

    input_manifest = [{"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in input_paths]
    output_files = sorted(path for path in output.rglob("*") if path.is_file())
    output_manifest = [
        {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in output_files
    ]
    if args.paper_title_display:
        status = "PASS_GSE267904_AGENT_GRANULARITY_HETEROGENEITY_V1_5_PAPER_TITLE"
    elif args.publication_display:
        status = "PASS_GSE267904_AGENT_GRANULARITY_HETEROGENEITY_V1_4_PUBLICATION_DISPLAY"
    elif args.labeled_display:
        status = "PASS_GSE267904_AGENT_GRANULARITY_HETEROGENEITY_V1_3_LABELED_DISPLAY"
    elif args.clean_display:
        status = "PASS_GSE267904_AGENT_GRANULARITY_HETEROGENEITY_V1_2_CLEAN_DISPLAY"
    elif args.refined_display:
        status = "PASS_GSE267904_AGENT_GRANULARITY_HETEROGENEITY_V1_1_REFINED_DISPLAY"
    else:
        status = "PASS_GSE267904_AGENT_GRANULARITY_HETEROGENEITY_V1"
    audit = {
        "status": status,
        "analysis_type": "locked post-hoc analysis of frozen K-specific outputs",
        "random_seed": args.random_seed,
        "capacity_matched_reference_draws": args.reference_draws,
        "nominal_K_values": list(K_VALUES),
        "module_definitions": MODULES,
        "module_state_definition": "three stage-specific observed-median high/low indicators; eight states",
        "module_standardization": "stage-specific Observed mean and sample standard deviation",
        "cci_families": list(CCI_FAMILIES),
        "cci_cell_types": cell_types,
        "cci_primary_state_space": "5 families x 9 senders x 9 receivers = 405; same-type included",
        "cci_sensitivity_state_space": "5 families x 9 senders x 8 non-self receivers = 360",
        "display_style": (
            (
                (
                    "paper-title display with noun-phrase title, raised legend, explicit d7/d21 labels, and collision-adjusted values"
                    if args.paper_title_display else
                    "publication display with concise conclusion-led title, raised legend, explicit d7/d21 labels, and collision-adjusted values"
                )
                if publication_mode else
                "clean display without matched-Real overlays; controls retained in source data"
            )
            if clean_mode else
            ("pooled grey equal-size Real reference band" if args.refined_display else "stage-specific matched-Real curves")
        ),
        "no_signal_policy": (
            "JS similarity remains NA in source data; refined figure displays zero recovery only for visual continuity; core recovery and effective diversity are zero"
            if args.refined_display else
            (
                "Panel D uses observed support-mass coverage, which is defined as zero for no signal; CCI JS remains NA in source data"
                if clean_mode else
                "JS similarity stored as NA; core recovery and effective diversity are zero; figure explicitly labels no detectable signal"
            )
        ),
        "upstream_reruns": {"LLM": False, "PhysiCell": False, "BioFVM": False, "COMMOT": False},
        "historical_outputs_modified": False,
        "gse267904_fibrosis_application_v3_resumed": False,
        "module_source_audit": module_source_audit,
        "figure_outputs": figure_outputs,
        "input_manifest": input_manifest,
        "validation": validations,
    }
    write_json(audit_dir / "analysis_audit.json", audit)
    write_json(audit_dir / "output_manifest.json", {"files": output_manifest})

    results_cn = (
        f"# GSE267904 Agent数量与异质性恢复 "
        f"{'V1.5 paper title' if args.paper_title_display else ('V1.4 publication' if args.publication_display else ('V1.3 labeled' if args.labeled_display else ('V1.2 clean' if args.clean_display else ('V1.1 refined' if args.refined_display else 'V1'))))}\n\n"
        "本分析只比较同一Leca-VC框架在K=20、50和100下的冻结结果；没有重跑LLM、PhysiCell、"
        "BioFVM或COMMOT。上排衡量三模块表达状态的分布距离、真实状态覆盖和比例准确度；"
        "下排衡量CCI状态比例、核心通信状态恢复和有效通信多样性。"
        + (
            "无信号组合为："
            + "、".join(
                f"K={item['K']} {item['stage']}" for item in primary_no_signal_cases
            )
            + "。\n\n"
            if primary_no_signal_cases
            else "所有K与阶段均检测到CCI信号。\n\n"
        )
        + "允许的结论：增加Agent granularity总体改善了已定义模块表达状态和通信状态的覆盖与恢复。"
        "不得将这些module-expression states解释为新细胞类型，也不得声称恢复了原生逐细胞异质性。\n"
    )
    (output / "RESULTS_SUMMARY_CN.md").write_text(results_cn, encoding="utf-8")

    # Completion is written last and includes hashes of every preceding result.
    final_files = sorted(path for path in output.rglob("*") if path.is_file())
    completion = {
        "status": status,
        "completed_utc": "2026-09-06T00:00:00Z",
        "files": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in final_files
        ],
    }
    write_json(output / "COMPLETE.json", completion)
    print(json.dumps({
        "status": completion["status"],
        "output_dir": str(output),
        "figure": figure_outputs["highres_png"],
        "module_metrics": module_metrics.to_dict(orient="records"),
        "cci_metrics": cci_metrics.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
