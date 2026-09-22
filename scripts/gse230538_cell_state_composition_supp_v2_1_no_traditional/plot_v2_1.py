#!/usr/bin/env python3
"""GSE230538 multimethod state-composition supplementary figure.

Observed and Leca-VC retain their frozen native state semantics. scGen, WOT,
and CellRank are explicitly post-hoc composition
proxies: their frozen cell-line mean expression targets are deterministically
expanded over matched untreated cells, then scored with the original frozen
marker/module state rule. No external model or simulation is rerun.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gse230538-state-multimethod-v2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
V7 = ROOT / "outputs/GSE230538_v7_dynamic_agent_benchmark"
EXPRESSION = ROOT / "outputs/GSE230538_multimethod_expression_benchmark_v1_v7_frozen"
NPZ = EXPRESSION / "01_harmonized/harmonized_7576_expression.npz"
RAW = ROOT / "data/GSE230538/raw"
OUT = ROOT / "outputs/GSE230538_cell_state_composition_supp_v2_multimethod_proxy"

STATES = (
    "proliferative_sensitive",
    "early_drug_response",
    "P2RX7_ion_signal_response",
    "senescent_like",
    "immune_like_resistant",
    "intrinsic_resistant",
    "stress_adaptive",
    "apoptotic_or_dying",
)
STATE_LABELS = (
    "Proliferative\nsensitive",
    "Early drug\nresponse",
    "P2RX7/ion\nresponse",
    "Senescent-like",
    "Immune-like\nresistant",
    "Intrinsic\nresistant",
    "Stress\nadaptive",
    "Apoptotic/\ndying",
)
FULL_MODULES = {
    "proliferation": ("MKI67", "TOP2A", "PCNA", "CCNB1", "CDK1"),
    "early_stress_MAPK": ("DUSP1", "DUSP5", "JUN", "FOS", "ATF3"),
    "P2RX7_ion_response": ("P2RX7", "ATP2B1", "ITPR1", "CALM1"),
    "senescence": ("CDKN1A", "CDKN2A", "SERPINE1", "IL6", "CXCL8"),
    "immune_IFN_like": ("ISG15", "IFIT1", "IFIT3", "MX1", "HLA-A", "HLA-B"),
    "invasive_resistant": ("AXL", "VIM", "ZEB1", "NGFR", "FN1"),
    "melanocytic_sensitive": ("MITF", "MLANA", "PMEL", "TYR"),
    "apoptosis_dying": ("BAX", "CASP3", "CASP7", "BBC3"),
}
FROZEN_MISSING_MARKERS = {"DUSP1", "DUSP5", "CDKN2A", "SERPINE1", "IL6", "CXCL8", "AXL", "ZEB1", "CASP7", "BBC3"}
MODULES = {
    module: tuple(gene for gene in genes if gene not in FROZEN_MISSING_MARKERS)
    for module, genes in FULL_MODULES.items()
}
STATE_TO_MODULE = {
    "proliferative_sensitive": ("proliferation", "melanocytic_sensitive"),
    "early_drug_response": ("early_stress_MAPK",),
    "P2RX7_ion_signal_response": ("P2RX7_ion_response",),
    "senescent_like": ("senescence",),
    "immune_like_resistant": ("immune_IFN_like",),
    "intrinsic_resistant": ("invasive_resistant",),
    "stress_adaptive": ("early_stress_MAPK", "invasive_resistant"),
    "apoptotic_or_dying": ("apoptosis_dying",),
}
MARKER_GENES = tuple(sorted({gene for genes in MODULES.values() for gene in genes}))

CELL_LINES = ("IPC-298", "M20", "MEL-JUSO", "SK-MEL-30")
DAYS = (4, 33)
SAMPLE_MAP = {
    "GSM7226481": ("MEL-JUSO", 0),
    "GSM7226482": ("MEL-JUSO", 1),
    "GSM7226483": ("MEL-JUSO", 4),
    "GSM7226484": ("MEL-JUSO", 33),
    "GSM7226485": ("IPC-298", 0),
    "GSM7226486": ("IPC-298", 1),
    "GSM7226487": ("IPC-298", 4),
    "GSM7226488": ("IPC-298", 33),
    "GSM7226489": ("M20", 0),
    "GSM7226490": ("M20", 1),
    "GSM7226491": ("M20", 4),
    "GSM7226492": ("M20", 33),
    "GSM7226493": ("SK-MEL-30", 0),
    "GSM7226494": ("SK-MEL-30", 1),
    "GSM7226495": ("SK-MEL-30", 4),
    "GSM7226496": ("SK-MEL-30", 33),
}

METHODS = ("Observed", "scGen", "WOT", "CellRank", "Leca-VC")
EXTERNAL_METHODS = ("scGen", "WOT", "CellRank")
ARRAY_KEYS = {"scGen": "scgen", "WOT": "wot", "CellRank": "cellrank"}
METHOD_LABELS = {
    "Observed": "Observed",
    "scGen": "scGen\N{DOUBLE DAGGER}",
    "WOT": "WOT*\N{DOUBLE DAGGER}",
    "CellRank": "CellRank\N{DAGGER}\N{DOUBLE DAGGER}",
    "Leca-VC": "Leca-VC",
}
COLORS = {
    "Observed": "#4C78A8",
    "scGen": "#F58518",
    "WOT": "#54A24B",
    "CellRank": "#4C9ED9",
    "Leca-VC": "#B279A2",
}
MODE_LABELS = {"v7_cell_like_agent_dynamic": "Leca-VC"}

STEM = "GSE230538_cell_state_composition_multimethod_proxy"
PREVIEW = OUT / f"{STEM}_preview.png"
PNG = OUT / f"{STEM}_600dpi.png"
PDF = OUT / f"{STEM}_vector.pdf"
SVG = OUT / f"{STEM}.svg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_files() -> list[Path]:
    paths = sorted(RAW.glob("GSM*_DGE.txt.gz"))
    if len(paths) != 16:
        raise RuntimeError(f"Expected 16 raw DGE files, found {len(paths)}")
    return paths


def sample_info(path: Path) -> tuple[str, str, int]:
    gsm = path.name.split("_", 1)[0]
    if gsm not in SAMPLE_MAP:
        raise RuntimeError(f"Unregistered GSE230538 sample: {gsm}")
    cell_line, day = SAMPLE_MAP[gsm]
    return gsm, cell_line, day


def read_marker_logcp10k(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return source marker log1p(CP10K) and original raw library sizes."""
    gene_to_index = {gene: index for index, gene in enumerate(MARKER_GENES)}
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        n_cells = len(header) - 1
        counts = np.zeros((n_cells, len(MARKER_GENES)), dtype=np.float64)
        library_sizes = np.zeros(n_cells, dtype=np.float64)
        seen: set[str] = set()
        for line in handle:
            gene, values_text = line.rstrip("\r\n").split("\t", 1)
            values = np.fromstring(values_text, sep="\t", dtype=np.float64)
            if len(values) != n_cells:
                raise RuntimeError(f"Malformed row {gene} in {path}")
            library_sizes += values
            index = gene_to_index.get(gene)
            if index is not None:
                counts[:, index] = values
                seen.add(gene)
    if len(seen) != len(MARKER_GENES):
        missing = sorted(set(MARKER_GENES) - seen)
        raise RuntimeError(f"Missing marker genes in {path}: {missing}")
    scale = np.zeros(n_cells, dtype=np.float64)
    positive = library_sizes > 0
    scale[positive] = 10000.0 / library_sizes[positive]
    logcp10k = np.log1p(counts * scale[:, None])
    if not np.isfinite(logcp10k).all() or np.any(logcp10k < 0):
        raise RuntimeError(f"Invalid marker normalization in {path}")
    return logcp10k, library_sizes


def simplex_projection_columns(matrix: np.ndarray, target_means: np.ndarray) -> np.ndarray:
    """Project columns onto nonnegative simplexes with exact target means."""
    source = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(target_means, dtype=np.float64)
    if source.ndim != 2 or targets.shape != (source.shape[1],):
        raise RuntimeError("Simplex shape mismatch")
    if not np.isfinite(source).all() or not np.isfinite(targets).all() or np.any(source < 0) or np.any(targets < 0):
        raise RuntimeError("Invalid simplex inputs")
    row_count = source.shape[0]
    target_sums = targets * row_count
    ordered = np.sort(source, axis=0)[::-1]
    cssv = np.cumsum(ordered, axis=0) - target_sums[None, :]
    indexes = np.arange(1, row_count + 1, dtype=np.float64)[:, None]
    active = ordered - cssv / indexes > 0
    rho = np.maximum(active.sum(axis=0) - 1, 0)
    theta = cssv[rho, np.arange(source.shape[1])] / (rho + 1.0)
    projected = np.maximum(source - theta[None, :], 0.0)
    residual = target_sums - projected.sum(axis=0)
    positive = projected > 0
    counts = positive.sum(axis=0)
    for column in np.where(np.abs(residual) > 1e-12)[0]:
        if counts[column] > 0:
            projected[positive[:, column], column] += residual[column] / counts[column]
        elif target_sums[column] > 0:
            projected[:, column] = target_sums[column] / row_count
    projected = np.maximum(projected, 0.0)
    error = float(np.max(np.abs(projected.mean(axis=0) - targets)))
    if error > 1e-10:
        raise RuntimeError(f"Simplex target error {error}")
    return projected


def infer_states_from_proxy_logcp10k(proxy: np.ndarray, library_sizes: np.ndarray) -> np.ndarray:
    """Apply the original raw-log marker rule using source library-size anchoring."""
    cp10k = np.expm1(proxy)
    raw_equivalent = cp10k * (library_sizes[:, None] / 10000.0)
    log_raw = np.log1p(raw_equivalent)
    gene_index = {gene: index for index, gene in enumerate(MARKER_GENES)}
    module_scores: dict[str, np.ndarray] = {}
    for module, genes in MODULES.items():
        indices = [gene_index[gene] for gene in genes]
        module_scores[module] = log_raw[:, indices].mean(axis=1)
    state_scores = np.empty((len(proxy), len(STATES)), dtype=np.float64)
    for state_index, state in enumerate(STATES):
        state_scores[:, state_index] = np.vstack(
            [module_scores[module] for module in STATE_TO_MODULE[state]]
        ).mean(axis=0)
    return np.asarray(STATES, dtype=object)[np.argmax(state_scores, axis=1)]


def composition(labels: np.ndarray) -> np.ndarray:
    return np.asarray([(labels == state).mean() for state in STATES], dtype=np.float64)


def load_observed_units() -> pd.DataFrame:
    path = V7 / "real_benchmark/metadata_checked.csv"
    metadata = pd.read_csv(path, usecols=["gsm_id", "aligned_real_day", "inferred_state"])
    metadata["canonical_cell_line"] = metadata["gsm_id"].map(
        {gsm: cell_line for gsm, (cell_line, _) in SAMPLE_MAP.items()}
    )
    if metadata["canonical_cell_line"].isna().any():
        raise RuntimeError("Observed metadata contains an unmapped GSM")
    rows: list[dict[str, object]] = []
    for day in DAYS:
        for cell_line in CELL_LINES:
            selected = metadata[
                metadata["aligned_real_day"].eq(day)
                & metadata["canonical_cell_line"].eq(cell_line)
            ]
            if len(selected) != 3000:
                raise RuntimeError(f"Expected 3000 observed cells for {cell_line}/Day{day}")
            vector = composition(selected["inferred_state"].astype(str).to_numpy())
            rows.extend(
                {
                    "day": day,
                    "method": "Observed",
                    "unit": cell_line,
                    "unit_type": "observed cell line",
                    "state": state,
                    "proportion": float(value),
                    "native_or_proxy": "native observed marker/module state",
                }
                for state, value in zip(STATES, vector)
            )
    return pd.DataFrame(rows)


def load_external_proxy_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    arrays = np.load(NPZ, allow_pickle=False)
    genes = list(map(str, arrays["genes"]))
    marker_indices = pd.Index(genes).get_indexer(MARKER_GENES)
    if np.any(marker_indices < 0):
        raise RuntimeError("Frozen expression panel is missing marker genes")
    npz_lines = list(map(str, arrays["cell_lines"]))
    npz_days = list(map(int, arrays["days"]))
    if set(npz_lines) != set(CELL_LINES) or tuple(npz_days) != DAYS:
        raise RuntimeError("Frozen expression array scope changed")

    day0_paths = {
        cell_line: path
        for path in raw_files()
        for _, cell_line, day in [sample_info(path)]
        if day == 0
    }
    unit_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for cell_line in CELL_LINES:
        source, library_sizes = read_marker_logcp10k(day0_paths[cell_line])
        line_index = npz_lines.index(cell_line)
        for day in DAYS:
            day_index = npz_days.index(day)
            for method in EXTERNAL_METHODS:
                target = np.asarray(arrays[ARRAY_KEYS[method]][day_index, line_index, marker_indices], dtype=np.float64)
                proxy = simplex_projection_columns(source, target)
                labels = infer_states_from_proxy_logcp10k(proxy, library_sizes)
                vector = composition(labels)
                target_error = float(np.max(np.abs(proxy.mean(axis=0) - target)))
                unit_rows.extend(
                    {
                        "day": day,
                        "method": method,
                        "unit": cell_line,
                        "unit_type": "cell-line expression proxy",
                        "state": state,
                        "proportion": float(value),
                        "native_or_proxy": "deterministic cell-expanded expression-derived composition proxy",
                    }
                    for state, value in zip(STATES, vector)
                )
                audit_rows.append(
                    {
                        "day": day,
                        "method": method,
                        "cell_line": cell_line,
                        "source_cells": len(source),
                        "marker_gene_count": len(MARKER_GENES),
                        "maximum_marker_target_mean_error": target_error,
                        "composition_sum": float(vector.sum()),
                        "native_single_cell_prediction_claimed": False,
                        "state_rule": "frozen-common-panel available-marker raw-log module argmax after source-library-size anchoring",
                    }
                )
    return pd.DataFrame(unit_rows), pd.DataFrame(audit_rows)


def load_simulation_units() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, method in MODE_LABELS.items():
        for seed in range(3):
            path = V7 / f"runs/{mode}/seed_{seed}/simulation_summary.csv"
            frame = pd.read_csv(path)
            for day in DAYS:
                selected = frame[frame["aligned_real_day"].astype(int).eq(day)]
                if len(selected) != 1:
                    raise RuntimeError(f"Missing simulation row {mode}/seed{seed}/Day{day}")
                vector = selected.iloc[0][list(STATES)].to_numpy(dtype=np.float64)
                if not np.isclose(vector.sum(), 1.0, atol=1e-10):
                    raise RuntimeError("Simulation composition does not sum to one")
                rows.extend(
                    {
                        "day": day,
                        "method": method,
                        "unit": f"seed_{seed}",
                        "unit_type": "simulation seed",
                        "state": state,
                        "proportion": float(value),
                        "native_or_proxy": "native frozen simulation state composition",
                    }
                    for state, value in zip(STATES, vector)
                )
    return pd.DataFrame(rows)


def summarize(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        units.groupby(["day", "method", "state"], sort=False)["proportion"]
        .agg(mean_proportion="mean", unit_sd=lambda values: values.to_numpy(float).std(ddof=0), n_units="size")
        .reset_index()
    )
    observed = summary[summary["method"].eq("Observed")].set_index(["day", "state"])["mean_proportion"]
    rmse_rows: list[dict[str, object]] = []
    for day in DAYS:
        truth = observed.loc[[(day, state) for state in STATES]].to_numpy(dtype=float)
        for method in METHODS[1:]:
            predicted = (
                summary[summary["day"].eq(day) & summary["method"].eq(method)]
                .set_index("state")
                .loc[list(STATES), "mean_proportion"]
                .to_numpy(dtype=float)
            )
            rmse_rows.append(
                {
                    "day": day,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "bar_mean_RMSE": float(np.sqrt(np.mean((predicted - truth) ** 2))),
                }
            )
    return summary, pd.DataFrame(rmse_rows)


def render(summary: pd.DataFrame, rmse: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.7), sharey=True)
    x = np.arange(len(STATES), dtype=float)
    width = 0.15
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    for ax, day, panel in zip(axes, DAYS, ("A", "B")):
        for method, offset in zip(METHODS, offsets):
            selected = (
                summary[summary["day"].eq(day) & summary["method"].eq(method)]
                .set_index("state")
                .loc[list(STATES)]
            )
            means = selected["mean_proportion"].to_numpy(dtype=float)
            errors = selected["unit_sd"].to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                means,
                width=width,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.4,
                yerr=errors,
                error_kw={"ecolor": "#4A4A4A", "elinewidth": 0.7, "capsize": 1.5, "capthick": 0.7},
                zorder=3,
            )
            for bar, value, error in zip(bars, means, errors):
                if value >= 0.015:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(value + error + 0.011, 1.03),
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.2,
                        color="#333333",
                        rotation=90,
                        clip_on=False,
                    )

        day_rmse = rmse[rmse["day"].eq(day)].set_index("method")["bar_mean_RMSE"]
        ax.set_title(f"{panel}. Day {day} held-out", loc="left", fontsize=12.4, y=1.17, pad=0)
        ax.text(
            0.5,
            1.095,
            "   ".join(f"{METHOD_LABELS[m]} {day_rmse[m]:.3f}" for m in ("scGen", "WOT", "CellRank")),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#444444",
        )
        ax.text(
            0.5,
            1.045,
            f"Bar-mean RMSE: {METHOD_LABELS['Leca-VC']} {day_rmse['Leca-VC']:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#444444",
        )
        ax.set_xticks(x, STATE_LABELS)
        ax.tick_params(axis="x", rotation=27, pad=5, labelsize=7.2)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.set_xlim(-0.58, len(STATES) - 0.42)
        ax.set_ylim(0, 1.06)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.62, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Cell-state proportion")
    handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=5, frameon=False)
    fig.suptitle("GSE230538 cell-state composition benchmark", fontsize=17, fontweight="bold", y=0.986)
    fig.text(
        0.5,
        0.052,
        "Bars show means; error bars are \N{PLUS-MINUS SIGN}1 population SD across four cell lines (Observed and external proxies) or three frozen seeds (simulations).",
        ha="center",
        fontsize=8.0,
        color="#4A4A4A",
    )
    fig.text(
        0.5,
        0.025,
        "\N{DOUBLE DAGGER} Expression-derived deterministic composition proxy, not a native single-cell prediction. "
        "* WOT is target-derived; \N{DAGGER} CellRank is an untreated-only, day-invariant terminal-expression proxy.",
        ha="center",
        fontsize=8.0,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.058, right=0.992, top=0.73, bottom=0.245, wspace=0.12)
    fig.savefig(PREVIEW, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    outputs = (
        PREVIEW,
        PNG,
        PDF,
        SVG,
        OUT / "unit_level_compositions.csv",
        OUT / "plotting_source.csv",
        OUT / "bar_mean_rmse.csv",
        OUT / "external_proxy_audit.csv",
        OUT / "caption.txt",
        OUT / "figure_audit.json",
    )
    for path in outputs:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing output: {path}")

    observed = load_observed_units()
    external, proxy_audit = load_external_proxy_units()
    simulations = load_simulation_units()
    units = pd.concat([observed, external, simulations], ignore_index=True)
    expected_units = {
        "Observed": 4,
        "scGen": 4,
        "WOT": 4,
        "CellRank": 4,
        "Leca-VC": 3,
    }
    for day in DAYS:
        for method, count in expected_units.items():
            selected = units[units["day"].eq(day) & units["method"].eq(method)]
            if len(selected) != count * len(STATES) or selected["unit"].nunique() != count:
                raise RuntimeError(f"Incomplete unit table: {method}/Day{day}")
            sums = selected.groupby("unit")["proportion"].sum().to_numpy(float)
            if not np.allclose(sums, 1.0, atol=1e-10):
                raise RuntimeError(f"Composition sum failure: {method}/Day{day}")

    summary, rmse = summarize(units)
    units.to_csv(OUT / "unit_level_compositions.csv", index=False)
    summary.to_csv(OUT / "plotting_source.csv", index=False)
    rmse.to_csv(OUT / "bar_mean_rmse.csv", index=False)
    proxy_audit.to_csv(OUT / "external_proxy_audit.csv", index=False)
    render(summary, rmse)

    caption = (
        "GSE230538 held-out cell-state composition benchmark. Observed Day 4 and Day 33 state proportions "
        "are compared with scGen, target-derived WOT, an untreated-only CellRank terminal-expression proxy, "
        "and Leca-VC. scGen, WOT, and CellRank compositions are post-hoc, "
        "expression-derived deterministic proxies obtained by expanding each frozen cell-line mean expression "
        "target over its matched untreated cells and applying the original marker/module state rule; they are "
        "not native single-cell composition outputs. Bars show means, and error bars show plus or minus one "
        "population SD across four cell lines for Observed/external proxies or three frozen simulation seeds "
        "for Leca-VC. Day 4 and Day 33 were held out from Leca-VC model generation."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    audit = {
        "status": "PASS_GSE230538_MULTIMETHOD_STATE_COMPOSITION_PROXY_FIGURE_NO_TRADITIONAL",
        "new_external_model_runs": 0,
        "new_simulation_runs": 0,
        "historical_outputs_overwritten": False,
        "days": list(DAYS),
        "states": list(STATES),
        "methods": list(METHODS),
        "external_proxy_methods": list(EXTERNAL_METHODS),
        "external_native_single_cell_composition_claimed": False,
        "external_proxy_formula": (
            "simplex-expand frozen cell-line mean log1p(CP10K) available-marker targets over matched Day0 cells; "
            "anchor to source library sizes; apply available-common-panel raw-log marker/module argmax"
        ),
        "marker_coverage": {
            module: {
                "requested": list(FULL_MODULES[module]),
                "available": list(MODULES[module]),
                "missing_from_frozen_7576_panel": [
                    gene for gene in FULL_MODULES[module] if gene in FROZEN_MISSING_MARKERS
                ],
            }
            for module in FULL_MODULES
        },
        "wot_target_derived": True,
        "cellrank_day_invariant_untreated_only_proxy": True,
        "error_bar_semantics": {
            "Observed_scGen_WOT_CellRank": "population SD across four cell lines",
            "Leca-VC": "population SD across three frozen simulation seeds",
        },
        "rmse_semantics": "RMSE between displayed method-mean and displayed observed-mean state vectors",
        "maximum_external_marker_target_mean_error": float(proxy_audit["maximum_marker_target_mean_error"].max()),
        "source_hashes": {
            "harmonized_expression_npz": sha256(NPZ),
            "observed_metadata": sha256(V7 / "real_benchmark/metadata_checked.csv"),
            **{
                f"simulation_{mode}_seed_{seed}": sha256(V7 / f"runs/{mode}/seed_{seed}/simulation_summary.csv")
                for mode in MODE_LABELS
                for seed in range(3)
            },
        },
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    (OUT / "figure_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "pdf": str(PDF), "rmse": rmse.to_dict("records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
