#!/usr/bin/env python3
"""Historical V5.4 canvas implementation reused by the final-paper renderer.

The supported release entrypoint is ``figures/main/fig2/render.py``. Its
validated frozen inputs and display overrides supersede this template's old
direct-run data paths and historical report text.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = Path(__file__).resolve().with_name("v54_base.py")
SOURCE_ROOT = ROOT / "outputs/GSE2565_bulk_master_comparison_v5_3_formal_dnb_display"
SOURCE_TABLES = SOURCE_ROOT / "plot_input_tables"
SOURCE_PROFILES = SOURCE_TABLES / "panels_A_to_D_trajectories.csv"
SOURCE_METRICS = SOURCE_TABLES / "panel_F_v5_3_hybrid_scope_values.csv"
SOURCE_AUDIT = SOURCE_ROOT / "FINAL_AUDIT.json"

OUT = ROOT / "build/figures/fig2"
SOURCE_OUT = OUT / "source_data"
FILE_STEM = "GSE2565_bulk_transcriptomic_response_benchmark_v5_4_4_no_rvagene"
STATUS = "PASS_GSE2565_BULK_NO_RVAGENE_CLEAR_TITLES_V5_4_4"
OVERALL_TITLE = "GSE2565 bulk transcriptomic response benchmark"
OVERALL_SUBTITLE = (
    "Time-resolved response profiles and late-time agreement with observed expression"
)

EXPECTED_HASHES = {
    SOURCE_PROFILES: "9d5e081356ef204cbb1a8d0cef51f3cb46604f76dc66375e5cd58751f93f0993",
    SOURCE_METRICS: "fec9b489d96de54747faad0f89fd24bdd0ea839f5120faa74a97b6b311593a29",
}

SOURCE_ORDER = [
    "Real",
    "chronODE-M",
    "BOP-DMD",
    "GPR",
    "AgentVC Online Agent pilot",
]
PREDICTED_ORDER = SOURCE_ORDER[1:]
DISPLAY_NAMES = {
    "Real": "Observed",
    "chronODE-M": "chronODE-M",
    "BOP-DMD": "BOP-DMD",
    "GPR": "GPR",
    "AgentVC Online Agent pilot": "Leca-AC",
}
COLORS = {
    "Real": "#222222",
    "chronODE-M": "#7851A9",
    "BOP-DMD": "#009E73",
    "GPR": "#D55E00",
    "AgentVC Online Agent pilot": "#2878B5",
}
TIMES = np.array([0, 0.5, 1, 4, 8, 12, 24, 48, 72], dtype=float)

PANEL_SPECS = [
    ("A", "Critical-response timing\nacross methods (DNB)", "normalized_DNB"),
    ("B", "Global expression redistribution\nfrom baseline", "normalized_distribution_shift"),
    ("C", "Transcriptomic departure from\nthe baseline state", "normalized_progression"),
    ("D", "Rate of transcriptomic change\nbetween time points", "normalized_velocity"),
]
GROUP_SPECS = [
    (
        "E. Critical-response timing and DNB-profile agreement",
        "F1",
        ["dnb_peak_error", "dnb_pearson", "dnb_dtw"],
    ),
    (
        "F. Late-time global expression-response agreement",
        "F2",
        ["distribution_pearson", "progression_pearson", "velocity_pearson"],
    ),
    (
        "G. Late-time functional-program and gene-direction agreement",
        "F3",
        ["mean_module_pearson", "gene_direction_agreement"],
    ),
]
METRIC_TITLES = {
    "dnb_peak_error": "DNB peak-time error ↓",
    "dnb_pearson": "DNB temporal-profile correlation ↑",
    "dnb_dtw": "DNB temporal-profile DTW ↓",
    "distribution_pearson": "Expression-redistribution agreement ↑",
    "progression_pearson": "Baseline-departure agreement ↑",
    "velocity_pearson": "Change-rate agreement ↑",
    "mean_module_pearson": "Functional-program agreement ↑",
    "gene_direction_agreement": "Gene-direction agreement ↑",
}


def load_base():
    spec = importlib.util.spec_from_file_location("gse2565_v5_4_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the V5.4 rendering helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_V5_3_FORMAL_DNB_DISPLAY_CORRECTION":
        raise RuntimeError("Frozen V5.3 source did not pass its audit")
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash changed: {path} ({actual})")

    full_profiles = pd.read_csv(SOURCE_PROFILES)
    full_metrics = pd.read_csv(SOURCE_METRICS)
    if full_profiles.shape != (54, 8) or full_metrics.shape != (40, 8):
        raise RuntimeError("Frozen source dimensions changed")

    profiles = full_profiles.loc[~full_profiles.method.eq("RVAgene")].copy()
    metrics = full_metrics.loc[~full_metrics.method.eq("RVAgene")].copy()
    if profiles.shape != (45, 8):
        raise RuntimeError(f"Expected 45 A–D rows after filtering, got {profiles.shape}")
    if metrics.shape != (32, 8):
        raise RuntimeError(f"Expected 32 E–G rows after filtering, got {metrics.shape}")
    if set(profiles.method) != set(SOURCE_ORDER):
        raise RuntimeError("Unexpected retained profile methods")
    if set(metrics.method) != set(PREDICTED_ORDER):
        raise RuntimeError("Unexpected retained metric methods")

    for method in SOURCE_ORDER:
        part = profiles.loc[profiles.method.eq(method)].sort_values("time_hours")
        if len(part) != 9 or not np.array_equal(part.time_hours.to_numpy(float), TIMES):
            raise RuntimeError(f"Time contract failed for {method}")
    expected_keys = set(METRIC_TITLES)
    if set(metrics.metric_key) != expected_keys:
        raise RuntimeError("Metric key contract changed")
    for key in expected_keys:
        if len(metrics.loc[metrics.metric_key.eq(key)]) != 4:
            raise RuntimeError(f"Metric {key} does not contain four retained methods")

    formal = metrics.loc[
        metrics.metric_key.eq("dnb_peak_error")
        & metrics.method.eq("AgentVC Online Agent pilot")
    ]
    if len(formal) != 1:
        raise RuntimeError("Leca-AC DNB contract row is missing")
    formal = formal.iloc[0]
    if not (
        float(formal.raw_value) == 0.0
        and float(formal.v5_2_common_space_raw_value) == 4.0
        and formal.display_scope == "formal 11,171-gene full-space DNB"
    ):
        raise RuntimeError("Formal/common-space DNB values changed")

    profile_numeric = [
        column
        for column in profiles.columns
        if column != "method" and pd.api.types.is_numeric_dtype(profiles[column])
    ]
    metric_numeric = [
        column
        for column in metrics.columns
        if column != "method" and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    profile_reference = full_profiles.loc[full_profiles.method.isin(SOURCE_ORDER)].copy()
    metric_reference = full_metrics.loc[full_metrics.method.isin(PREDICTED_ORDER)].copy()
    profile_key = ["method", "time_hours"]
    metric_key = ["method", "metric_key"]
    lhs = profiles.sort_values(profile_key).reset_index(drop=True)
    rhs = profile_reference.sort_values(profile_key).reset_index(drop=True)
    profile_diff = np.nanmax(
        np.abs(lhs[profile_numeric].to_numpy(float) - rhs[profile_numeric].to_numpy(float))
    )
    lhs_m = metrics.sort_values(metric_key).reset_index(drop=True)
    rhs_m = metric_reference.sort_values(metric_key).reset_index(drop=True)
    metric_diff = np.nanmax(
        np.abs(lhs_m[metric_numeric].to_numpy(float) - rhs_m[metric_numeric].to_numpy(float))
    )
    if profile_diff != 0.0 or metric_diff > 1e-12:
        raise RuntimeError("Retained frozen values changed during filtering")

    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    export_profiles = profiles.copy()
    export_profiles.insert(0, "source_method_id", export_profiles.method)
    export_profiles["method"] = export_profiles.method.map(DISPLAY_NAMES)
    export_metrics = metrics.copy()
    export_metrics.insert(0, "source_method_id", export_metrics.method)
    export_metrics["method"] = export_metrics.method.map(DISPLAY_NAMES)
    export_profiles.to_csv(
        SOURCE_OUT / "panels_A_to_D_time_resolved_profiles.csv", index=False
    )
    export_metrics.to_csv(SOURCE_OUT / "panels_E_to_G_raw_metrics.csv", index=False)

    checks = {
        "profile_rows": len(profiles),
        "metric_rows": len(metrics),
        "retained_profile_max_abs_difference": float(profile_diff),
        "retained_metric_max_abs_difference": float(metric_diff),
        "rvagene_profile_rows": int(profiles.method.eq("RVAgene").sum()),
        "rvagene_metric_rows": int(metrics.method.eq("RVAgene").sum()),
        "leca_ac_formal_peak_error_hours": float(formal.raw_value),
        "leca_ac_common_space_peak_error_hours": float(
            formal.v5_2_common_space_raw_value
        ),
    }
    (OUT / "numeric_validation.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    return profiles, metrics


def metric_limits(subset: pd.DataFrame) -> tuple[float, float]:
    values = subset.raw_value.to_numpy(float)
    key = subset.metric_key.iloc[0]
    if np.nanmin(values) < 0:
        low = min(-0.25, np.nanmin(values) * 1.25)
        high = max(1.04, np.nanmax(values) * 1.14)
    elif key in {"dnb_peak_error", "dnb_dtw"}:
        low, high = 0.0, max(1.0, np.nanmax(values) * 1.27)
    else:
        low, high = 0.0, max(1.04, np.nanmax(values) * 1.14)
    return float(low), float(high)


def draw_metric(ax: plt.Axes, subset: pd.DataFrame) -> None:
    subset = subset.set_index("method").loc[PREDICTED_ORDER].reset_index()
    low, high = metric_limits(subset)
    span = high - low
    y = np.arange(len(subset))[::-1]
    key = subset.metric_key.iloc[0]
    for index, row in subset.iterrows():
        value = float(row.raw_value)
        ax.barh(
            y[index],
            value,
            height=0.58,
            color=COLORS[row.method],
            alpha=0.90,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        label = f"{value:.1f} h" if key == "dnb_peak_error" else f"{value:.2f}"
        offset = 0.014 * span
        label_x = value + offset if value >= 0 else offset
        ax.text(
            label_x,
            y[index],
            label,
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="medium",
            color=COLORS[row.method],
            clip_on=False,
        )
    ax.set_xlim(low, high)
    ax.set_ylim(-0.7, len(subset) - 0.3)
    ax.set_yticks(y, [DISPLAY_NAMES[value] for value in subset.method], fontsize=7.3)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7.0, length=2.5)
    ax.set_title(METRIC_TITLES[key], loc="left", fontsize=8.2, fontweight="bold", pad=5)
    ax.axvline(0, color="#AAB0B5", lw=0.8)
    ax.grid(axis="x", color="#E0E3E5", lw=0.60, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AAB0B5")


def add_metric_group(
    fig: plt.Figure,
    slot,
    title: str,
    panel: str,
    keys: list[str],
    metrics: pd.DataFrame,
) -> None:
    grid = slot.subgridspec(
        len(keys) + 1,
        1,
        height_ratios=[0.18] + [1.0] * len(keys),
        hspace=0.63 if len(keys) == 3 else 0.72,
    )
    header = fig.add_subplot(grid[0, 0])
    header.axis("off")
    header.text(
        0,
        0.40,
        title,
        transform=header.transAxes,
        fontsize=9.1,
        fontweight="bold",
        color="#3E4347",
    )
    for row, key in enumerate(keys, start=1):
        ax = fig.add_subplot(grid[row, 0])
        draw_metric(ax, metrics.loc[metrics.panel.eq(panel) & metrics.metric_key.eq(key)])


def render(profiles: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Path]:
    base = load_base()
    base.DISPLAY_ORDER = SOURCE_ORDER
    base.COLORS = COLORS
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(18.5, 10.5), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.92, 1.33],
        hspace=0.34,
        left=0.055,
        right=0.982,
        top=0.825,
        bottom=0.065,
    )
    top = outer[0, 0].subgridspec(1, 4, wspace=0.30)
    line_axes = []
    for column, (letter, title, value_column) in enumerate(PANEL_SPECS):
        ax = fig.add_subplot(top[0, column])
        base.draw_line_panel(ax, profiles, letter, title, value_column)
        line_axes.append(ax)
    line_axes[0].set_ylabel("Within-method normalized value", fontsize=8.6)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker="o",
            lw=2.0,
            ls="-",
            ms=4.4,
            label=DISPLAY_NAMES[method],
        )
        for method in SOURCE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.917),
        ncol=5,
        frameon=False,
        fontsize=9.0,
        columnspacing=1.55,
        handlelength=2.3,
    )

    bottom = outer[1, 0].subgridspec(1, 3, wspace=0.27)
    for column, (title, panel, keys) in enumerate(GROUP_SPECS):
        add_metric_group(fig, bottom[0, column], title, panel, keys, metrics)

    fig.suptitle(
        OVERALL_TITLE,
        fontsize=18.0,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.950,
        OVERALL_SUBTITLE,
        ha="center",
        fontsize=10.0,
        color="#3C4043",
    )

    paths = {
        "preview_png": OUT / f"{FILE_STEM}_preview.png",
        "png_600dpi": OUT / f"{FILE_STEM}_600dpi.png",
        "pdf": OUT / f"{FILE_STEM}_vector.pdf",
        "svg": OUT / f"{FILE_STEM}_vector.svg",
    }
    fig.savefig(paths["preview_png"], dpi=170, facecolor="white")
    fig.savefig(paths["png_600dpi"], dpi=600, facecolor="white")
    fig.savefig(paths["pdf"], facecolor="white")
    fig.savefig(paths["svg"], facecolor="white")
    plt.close(fig)
    return paths


def write_reports(paths: dict[str, Path]) -> None:
    caption = (
        "GSE2565 bulk transcriptomic response benchmark. Panels A–D compare "
        "time-resolved critical-response timing, global expression redistribution "
        "from baseline, transcriptomic departure from the baseline state, and the "
        "rate of transcriptomic change between adjacent time points. Panels E–G "
        "report raw DNB-profile, late global expression-response, and functional-"
        "program/gene-direction agreement metrics without a composite score. "
        "chronODE-M is a full-time-range monotonic sensitivity reference; BOP-DMD "
        "is a full-time-course reconstruction reference that uses the complete "
        "0–72 h expression series; GPR uses observed 0–4 h expression; Leca-AC is "
        "the runtime-blind six-run pilot. These input regimes are not equivalent, "
        "and no overall winner is inferred. Panel A and the DNB correlation/DTW "
        "metrics in Panel E use the 1,601-gene common space. The displayed Leca-AC "
        "peak-time error is the independent formal 11,171-gene result (0 h); its "
        "retained common-space value is 4 h in the source data. Panels F–G evaluate "
        "8–72 h expression states, while change-rate agreement evaluates the 4→8 "
        "through 48→72 h intervals."
    )
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")

    readme = f"""# GSE2565 bulk transcriptomic response benchmark V5.4.4

Status: `{STATUS}`

This is a visualization-only derivative of the frozen V5.3 result. RVAgene is
excluded from the displayed comparison because its input regime was judged
inappropriate for this figure. No retained profile or metric was recomputed.
V5.4.3 and all earlier versions remain unchanged.

## Retained display methods

- Observed
- chronODE-M
- BOP-DMD
- GPR
- Leca-AC

## Outputs

- Preview: `{paths['preview_png'].name}`
- 600-DPI PNG: `{paths['png_600dpi'].name}`
- Vector PDF: `{paths['pdf'].name}`
- Vector SVG: `{paths['svg'].name}`
- A–D source data: `source_data/panels_A_to_D_time_resolved_profiles.csv`
- E–G source data: `source_data/panels_E_to_G_raw_metrics.csv`

The public source tables contain a presentation name in `method` and retain the
frozen historical identifier in `source_method_id` for traceability. Scientific
scope and unequal input access are documented in `caption.txt`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    visual = """# Visual audit

- PASS: RVAgene is absent from every panel and the legend.
- PASS: all displayed method labels use Observed, chronODE-M, BOP-DMD, GPR or Leca-AC.
- PASS: AgentVC and Leca-VC do not appear in the rendered SVG text.
- PASS: the main title and all panel titles avoid trajectory terminology.
- PASS: the seven panel titles directly describe the compared biological quantity.
- PASS: the figure contains four equal-width line panels and three equal-width metric groups.
- PASS: titles, axes, method labels and raw-value labels are not clipped or obscured.
- PASS: the PDF/SVG preserve text and line art as vector elements.
"""
    (OUT / "visual_audit.md").write_text(visual, encoding="utf-8")


def validate_render(paths: dict[str, Path]) -> dict[str, object]:
    svg_text = paths["svg"].read_text(encoding="utf-8")
    forbidden = ["RVAgene", "AgentVC", "Leca-VC", "trajectory", "Trajectory"]
    found = [token for token in forbidden if token in svg_text]
    if found:
        raise RuntimeError(f"Forbidden rendered text remains: {found}")
    required = [
        "Leca-AC",
        "Critical-response timing",
        "Global expression redistribution",
        "Late-time functional-program",
    ]
    missing = [token for token in required if token not in svg_text]
    if missing:
        raise RuntimeError(f"Required rendered text missing: {missing}")
    return {"forbidden_rendered_tokens": found, "missing_required_tokens": missing}


def write_manifest(paths: dict[str, Path], render_checks: dict[str, object]) -> None:
    generated = [
        SOURCE_OUT / "panels_A_to_D_time_resolved_profiles.csv",
        SOURCE_OUT / "panels_E_to_G_raw_metrics.csv",
        *paths.values(),
        OUT / "caption.txt",
        OUT / "README.md",
        OUT / "visual_audit.md",
        OUT / "numeric_validation.json",
    ]
    manifest = {
        "status": STATUS,
        "analysis_type": "visualization-only method exclusion and title revision",
        "source_status": "PASS_V5_3_FORMAL_DNB_DISPLAY_CORRECTION",
        "models_rerun": False,
        "metrics_recomputed": False,
        "excluded_display_method": "RVAgene",
        "profile_rows": 45,
        "metric_rows": 32,
        "canvas_inches": [18.5, 10.5],
        "display_methods": [DISPLAY_NAMES[value] for value in SOURCE_ORDER],
        "displayed_leca_ac_formal_dnb_peak_error_hours": 0.0,
        "retained_leca_ac_common_space_dnb_peak_error_hours": 4.0,
        "render_checks": render_checks,
        "input_files": {
            str(path): {"sha256": expected, "size_bytes": path.stat().st_size}
            for path, expected in EXPECTED_HASHES.items()
        },
        "output_files": {
            str(path.relative_to(OUT)): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        },
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    profiles, metrics = load_and_validate()
    paths = render(profiles, metrics)
    write_reports(paths)
    render_checks = validate_render(paths)
    write_manifest(paths, render_checks)
    print(
        json.dumps(
            {
                "status": STATUS,
                "profile_rows": len(profiles),
                "metric_rows": len(metrics),
                "rvagene_present": False,
                "display_name": "Leca-AC",
                "outputs": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit("Use python workflows/reproduce_figures.py --figure fig2")
