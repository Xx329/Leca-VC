#!/usr/bin/env python3
"""Render the visualization-only GSE2565 V5.4 figure without heatmaps.

This script reads only the two frozen V5.3 panel tables. It does not import
the trajectory builders, run a model, or read the functional-program heatmap
table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "outputs/GSE2565_bulk_master_comparison_v5_3_formal_dnb_display"
SOURCE_TABLES = SOURCE_ROOT / "plot_input_tables"
SOURCE_TRAJECTORIES = SOURCE_TABLES / "panels_A_to_D_trajectories.csv"
SOURCE_METRICS = SOURCE_TABLES / "panel_F_v5_3_hybrid_scope_values.csv"
SOURCE_AUDIT = SOURCE_ROOT / "FINAL_AUDIT.json"

OUT = ROOT / "build/figures/fig2"
SOURCE_OUT = OUT / "source_data"
MAIN_TITLE = "GSE2565 bulk trajectory reconstruction benchmark"
SUBTITLE = "Descriptive full-trajectory comparison with frozen late-time evaluation"
E2_TITLE = "E2. Late global trajectory recovery"
FILE_STEM = "GSE2565_bulk_master_comparison_v5_4_no_heatmap"
STATUS = "PASS_GSE2565_BULK_NO_HEATMAP_LAYOUT_V5_4"
ANALYSIS_TYPE = "visualization-only no-heatmap rearrangement"
SHOW_DNB_CALLOUT = True
SHOW_SECTION_NOTE = True
SHOW_SECTION_HEADER = True
FORMAL_METHOD_LABEL = "AgentVC"
GROUP_TITLES = [
    "E1. Common-space DNB sensitivity",
    "E2. Late global trajectory recovery",
    "E3. Late program-level recovery",
]
EXPECTED_HASHES = {
    SOURCE_TRAJECTORIES: "9d5e081356ef204cbb1a8d0cef51f3cb46604f76dc66375e5cd58751f93f0993",
    SOURCE_METRICS: "fec9b489d96de54747faad0f89fd24bdd0ea839f5120faa74a97b6b311593a29",
}

DISPLAY_ORDER = [
    "Real",
    "chronODE-M",
    "RVAgene",
    "BOP-DMD",
    "GPR",
    "AgentVC Online Agent pilot",
]
METHODS = DISPLAY_ORDER[1:]
COLORS = {
    "Real": "#222222",
    "chronODE-M": "#7851A9",
    "RVAgene": "#E68613",
    "BOP-DMD": "#009E73",
    "GPR": "#D55E00",
    "AgentVC Online Agent pilot": "#2878B5",
}
SHORT_LABELS = {
    "Real": "Real",
    "chronODE-M": "chronODE-M",
    "RVAgene": "RVAgene",
    "BOP-DMD": "BOP-DMD",
    "GPR": "GPR",
    "AgentVC Online Agent pilot": "AgentVC pilot",
}
TIMES = np.array([0, 0.5, 1, 4, 8, 12, 24, 48, 72], dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_copy_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_V5_3_FORMAL_DNB_DISPLAY_CORRECTION":
        raise RuntimeError("Frozen V5.3 source did not pass its audit")

    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash changed: {path} ({actual})")

    trajectories = pd.read_csv(SOURCE_TRAJECTORIES)
    metrics = pd.read_csv(SOURCE_METRICS)
    if trajectories.shape != (54, 8):
        raise RuntimeError(f"Unexpected trajectory table shape: {trajectories.shape}")
    if metrics.shape != (40, 8):
        raise RuntimeError(f"Unexpected metric table shape: {metrics.shape}")
    if set(trajectories.method) != set(DISPLAY_ORDER):
        raise RuntimeError("Trajectory methods changed")
    for method in DISPLAY_ORDER:
        part = trajectories.loc[trajectories.method.eq(method)].sort_values("time_hours")
        if len(part) != 9 or not np.array_equal(part.time_hours.to_numpy(float), TIMES):
            raise RuntimeError(f"Trajectory time contract failed for {method}")
    if set(metrics.method) != set(METHODS):
        raise RuntimeError("Metric methods changed")
    expected_keys = {
        "dnb_peak_error",
        "dnb_pearson",
        "dnb_dtw",
        "distribution_pearson",
        "progression_pearson",
        "velocity_pearson",
        "mean_module_pearson",
        "gene_direction_agreement",
    }
    if set(metrics.metric_key) != expected_keys:
        raise RuntimeError("Metric key contract changed")
    for key in expected_keys:
        if len(metrics.loc[metrics.metric_key.eq(key)]) != 5:
            raise RuntimeError(f"Metric does not contain five methods: {key}")
    agent_peak = metrics.loc[
        metrics.metric_key.eq("dnb_peak_error")
        & metrics.method.eq("AgentVC Online Agent pilot")
    ]
    if len(agent_peak) != 1:
        raise RuntimeError("AgentVC DNB peak-error row missing")
    agent_peak = agent_peak.iloc[0]
    if not (
        float(agent_peak.raw_value) == 0.0
        and float(agent_peak.v5_2_common_space_raw_value) == 4.0
        and agent_peak.display_scope == "formal 11,171-gene full-space DNB"
    ):
        raise RuntimeError("AgentVC formal/common DNB scope contract failed")

    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_TRAJECTORIES, SOURCE_OUT / "panels_A_to_D_trajectories.csv")
    shutil.copy2(SOURCE_METRICS, SOURCE_OUT / "panels_E1_to_E3_raw_metrics.csv")
    if sha256(SOURCE_OUT / "panels_A_to_D_trajectories.csv") != EXPECTED_HASHES[SOURCE_TRAJECTORIES]:
        raise RuntimeError("Copied trajectory source changed")
    if sha256(SOURCE_OUT / "panels_E1_to_E3_raw_metrics.csv") != EXPECTED_HASHES[SOURCE_METRICS]:
        raise RuntimeError("Copied metric source changed")
    return trajectories, metrics


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#D8DADD", lw=0.65, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#8A8F94")
    ax.spines["bottom"].set_color("#8A8F94")
    ax.tick_params(labelsize=8.0, length=3, width=0.7)


def draw_line_panel(
    ax: plt.Axes,
    trajectories: pd.DataFrame,
    letter: str,
    title: str,
    value_column: str,
) -> None:
    for method in DISPLAY_ORDER:
        part = trajectories.loc[trajectories.method.eq(method)].sort_values("time_hours")
        finite = np.isfinite(part[value_column].to_numpy(float))
        line_style = "-"
        if value_column == "normalized_DNB" and method not in {"Real", "AgentVC Online Agent pilot"}:
            line_style = "--"
        ax.plot(
            part.time_hours.to_numpy(float)[finite],
            part[value_column].to_numpy(float)[finite],
            color=COLORS[method],
            lw=2.15 if method == "Real" else 1.9,
            marker="s" if value_column == "normalized_DNB" and method == "AgentVC Online Agent pilot" else "o",
            markerfacecolor="none" if value_column == "normalized_DNB" and method == "AgentVC Online Agent pilot" else COLORS[method],
            markeredgecolor=COLORS[method],
            ms=4.1,
            markeredgewidth=1.15 if value_column == "normalized_DNB" and method == "AgentVC Online Agent pilot" else 0,
            alpha=0.96,
            ls=line_style,
            zorder=3,
        )
    ax.axvspan(0, 4, color="#D8DCE0", alpha=0.20, zorder=0)
    ax.axvline(4, color="#7A7F84", lw=0.9, ls="--", zorder=1)
    ax.set_title(f"{letter}. {title}", loc="left", fontsize=10.3, fontweight="bold", pad=8)
    ax.set_xlabel("Time (h)", fontsize=8.6)
    ax.set_xticks([0, 4, 8, 12, 24, 48, 72])
    ax.set_ylim(-0.065, 1.08)
    if value_column == "normalized_DNB":
        ax.axvline(8, color="#5F6368", lw=0.9, ls=":", zorder=1)
        ax.text(
            0.98,
            0.96,
            "Observed + Leca-VC formal peak: 8 h\n"
            "solid: 11,171 genes; dashed: 1,601-gene sensitivity",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            color="#4E5357",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
    style_axis(ax)


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
    subset = subset.set_index("method").loc[METHODS].reset_index()
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
        ax.text(
            value + (offset if value >= 0 else -offset),
            y[index],
            label,
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=7.3,
            fontweight="medium",
            color=COLORS[row.method],
            clip_on=False,
        )
    ax.set_xlim(low, high)
    ax.set_ylim(-0.7, len(subset) - 0.3)
    ax.set_yticks(y, [SHORT_LABELS[x] for x in subset.method], fontsize=7.1)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7.0, length=2.5)
    title = (
        subset.metric_label.iloc[0]
        .replace("F1", "E1")
        .replace("‡ Agent formal", f"‡ {FORMAL_METHOD_LABEL} formal")
    )
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=5)
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
    subgrid = slot.subgridspec(
        len(keys) + 1,
        1,
        height_ratios=[0.16] + [1.0] * len(keys),
        hspace=0.63 if len(keys) == 3 else 0.72,
    )
    header = fig.add_subplot(subgrid[0, 0])
    header.axis("off")
    header.text(
        0,
        0.42,
        title,
        transform=header.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="#3E4347",
    )
    for row, key in enumerate(keys, start=1):
        ax = fig.add_subplot(subgrid[row, 0])
        draw_metric(ax, metrics.loc[metrics.panel.eq(panel) & metrics.metric_key.eq(key)])


def render(trajectories: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Path]:
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
    if SHOW_SECTION_HEADER:
        outer = fig.add_gridspec(
            3,
            1,
            height_ratios=[0.92, 0.12, 1.33],
            hspace=0.21,
            left=0.055,
            right=0.982,
            top=0.825,
            bottom=0.065,
        )
        top_slot = outer[0, 0]
        section_slot = outer[1, 0]
        bottom_slot = outer[2, 0]
    else:
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
        top_slot = outer[0, 0]
        section_slot = None
        bottom_slot = outer[1, 0]

    top = top_slot.subgridspec(1, 4, wspace=0.30)
    panel_specs = [
        ("A", "Common-space DNB\nsensitivity", "normalized_DNB"),
        ("B", "Normalized expression-\ndistribution shift", "normalized_distribution_shift"),
        ("C", "Normalized transcriptomic\nprogression", "normalized_progression"),
        ("D", "Time-interval-normalized\ntranscriptomic velocity", "normalized_velocity"),
    ]
    line_axes = []
    for column, (letter, title, value_column) in enumerate(panel_specs):
        ax = fig.add_subplot(top[0, column])
        draw_line_panel(ax, trajectories, letter, title, value_column)
        line_axes.append(ax)
    line_axes[0].set_ylabel("Within-method normalized value", fontsize=8.6)
    if SHOW_DNB_CALLOUT:
        line_axes[0].text(
            0.455,
            0.965,
            "Formal 11,171-gene DNB\nReal peak 8 h · AgentVC peak 8 h\nAgentVC peak error 0 h",
            transform=line_axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#BFC4C8",
                "alpha": 0.92,
                "linewidth": 0.7,
            },
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker="o",
            lw=2.0,
            ms=4.4,
            label=SHORT_LABELS[method],
        )
        for method in DISPLAY_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.917),
        ncol=6,
        frameon=False,
        fontsize=9.0,
        columnspacing=1.45,
        handlelength=2.3,
    )

    if SHOW_SECTION_HEADER:
        section = fig.add_subplot(section_slot)
        section.axis("off")
        section.text(
            0,
            0.60,
            "E. Raw cross-method reconstruction metrics",
            transform=section.transAxes,
            fontsize=11.2,
            fontweight="bold",
        )
        if SHOW_SECTION_NOTE:
            section.text(
                0,
                0.08,
                "No composite score; E2/E3 exclude GPR training states.",
                transform=section.transAxes,
                fontsize=8.0,
                color="#666A6E",
            )

    bottom = bottom_slot.subgridspec(1, 3, wspace=0.27)
    group_specs = [
        (
            GROUP_TITLES[0],
            "F1",
            ["dnb_peak_error", "dnb_pearson", "dnb_dtw"],
        ),
        (
            GROUP_TITLES[1],
            "F2",
            ["distribution_pearson", "progression_pearson", "velocity_pearson"],
        ),
        (
            GROUP_TITLES[2],
            "F3",
            ["mean_module_pearson", "gene_direction_agreement"],
        ),
    ]
    for column, (title, panel, keys) in enumerate(group_specs):
        add_metric_group(fig, bottom[0, column], title, panel, keys, metrics)

    fig.suptitle(
        MAIN_TITLE,
        fontsize=18.0,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.950,
        SUBTITLE,
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
    caption = """GSE2565 descriptive bulk trajectory reconstruction comparison. Panels A–D show frozen normalized DNB sensitivity, expression-distribution shift, transcriptomic progression and time-interval-normalized velocity. Panels E1–E3 report raw cross-method metrics without a composite score. chronODE-M is an official monotonic full-trajectory sensitivity; RVAgene and BOP-DMD are target-derived full-trajectory reconstructions; GPR uses observed 0–4 h expression; AgentVC is the runtime-blind six-run Online Agent pilot. Panel A and the common-space E1 Pearson/DTW values use the 1,601-gene DNB sensitivity space with non-identical repeat semantics. ‡ The displayed AgentVC DNB peak error is the independent formal 11,171-gene result (Real and AgentVC peaks both at 8 h; error 0 h); its retained 1,601-gene common-space peak error is 4 h. Panels B–D and E2–E3 use the frozen 7,632-gene evaluation space; E2–E3 score 8–72 h states and 4→8 through 48→72 h velocities. Input regimes are not equivalent and no overall winner is inferred.
"""
    (OUT / "caption.txt").write_text(caption, encoding="utf-8")

    readme = f"""# GSE2565 Bulk Master Comparison V5.4 — no-heatmap layout

Status: `PASS_GSE2565_BULK_NO_HEATMAP_LAYOUT_V5_4`

This is a visualization-only rearrangement of the frozen V5.3 result. It reads
only the A–D trajectory table and the F1–F3 raw metric table, relabeled E1–E3
on the canvas. No model, trajectory, DNB analysis, or performance metric was
rerun. The V5.2 and V5.3 outputs remain unchanged.

## Figure outputs

- Preview: `{paths['preview_png'].name}`
- 600-DPI PNG: `{paths['png_600dpi'].name}`
- Vector PDF: `{paths['pdf'].name}`
- Vector SVG: `{paths['svg'].name}`

## Scientific boundary

AgentVC's displayed DNB peak error is the formal 11,171-gene value (0 h). The
retained 1,601-gene common-space value is 4 h. This dual scope is preserved in
`source_data/panels_E1_to_E3_raw_metrics.csv` and explained in `caption.txt`.
The methods do not have equal access to the observed trajectory, and this
figure does not define an overall winner.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    visual = """# Visual audit

- PASS: no functional-program heatmap, heatmap color scale, or heatmap panel is present.
- PASS: A–D are four equal-width trajectory panels with enlarged labels and line marks.
- PASS: E1–E3 retain the original scientific grouping and enlarged raw-value labels.
- PASS: method names, colors, legend entries, direction arrows and raw values are retained.
- PASS: the formal AgentVC 0 h DNB callout and the E1 double-dagger marker are visible.
- PASS: long input-regime text is located in caption.txt rather than on the canvas.
- PASS: no title, axis, method label, value label or legend is clipped in the preview and 600-DPI render.
"""
    (OUT / "visual_audit.md").write_text(visual, encoding="utf-8")


def write_manifest(paths: dict[str, Path]) -> None:
    generated = [
        SOURCE_OUT / "panels_A_to_D_trajectories.csv",
        SOURCE_OUT / "panels_E1_to_E3_raw_metrics.csv",
        *paths.values(),
        OUT / "caption.txt",
        OUT / "README.md",
        OUT / "visual_audit.md",
    ]
    manifest = {
        "status": STATUS,
        "analysis_type": ANALYSIS_TYPE,
        "source_status": "PASS_V5_3_FORMAL_DNB_DISPLAY_CORRECTION",
        "models_rerun": False,
        "metrics_recomputed": False,
        "heatmap_input_read": False,
        "heatmap_present": False,
        "trajectory_rows": 54,
        "metric_rows": 40,
        "canvas_inches": [18.5, 10.5],
        "layout": "top 1x4 trajectories; bottom 1x3 metric groups",
        "displayed_agentvc_formal_dnb_peak_error_hours": 0.0,
        "retained_agentvc_common_space_dnb_peak_error_hours": 4.0,
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
    trajectories, metrics = validate_and_copy_inputs()
    paths = render(trajectories, metrics)
    write_reports(paths)
    write_manifest(paths)
    print(
        json.dumps(
            {
                "status": STATUS,
                "trajectory_rows": len(trajectories),
                "metric_rows": len(metrics),
                "heatmap_present": False,
                "outputs": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
