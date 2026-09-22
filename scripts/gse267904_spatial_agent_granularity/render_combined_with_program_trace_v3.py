"""Render the frozen GSE267904 main figure with an auditable fibroblast program trace.

The A--F heterogeneity metrics and spatial-map pixels are reused without
recomputation.  Panel H reads the frozen, de-identified K=100 decision and
PhysiCell writeback tables and visualizes one fully auditable agent trajectory.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gse267904-program-trace-v3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    os.environ.get(
        "GSE267904_HETEROGENEITY_DATA",
        str(
            ROOT
            / "outputs/GSE267904_deidentified_rerun_v1/postprocess/agent_granularity_heterogeneity/data"
        ),
    )
).resolve()
SOURCE = Path(
    os.environ.get(
        "GSE267904_MODULE_MAP_SOURCE",
        str(
            ROOT
            / "outputs/GSE267904_deidentified_rerun_v1/figures/agent_granularity_spatial_combined"
            / "GSE267904_balanced_metrics_and_spatial_maps_600dpi.png"
        ),
    )
).resolve()
RERUN = Path(
    os.environ.get(
        "GSE267904_DEIDENTIFIED_RERUN",
        str(ROOT / "outputs/GSE267904_deidentified_rerun_v1"),
    )
).resolve()
OUT = Path(
    os.environ.get(
        "GSE267904_PROGRAM_TRACE_FIGURE_OUT",
        str(ROOT / "outputs/GSE267904_agent_granularity_spatial_combined_v3_program_trace"),
    )
).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def load_trace():
    decision_paths = [
        RERUN / f"K_100/micro_agents/llm_decision_interval_{i}.csv" for i in (0, 1)
    ]
    applied_paths = [
        RERUN / f"K_100/physicell/applied_decision_{i}.csv" for i in (0, 1)
    ]
    decision_rows = []
    applied_rows = []
    for interval, path in enumerate(decision_paths):
        table = pd.read_csv(path)
        row = table.loc[table.agent_id.eq("micro_agent_032")].copy()
        if len(row) != 1:
            raise RuntimeError(f"Expected one micro_agent_032 row in {path}; found {len(row)}")
        row["interval"] = interval
        decision_rows.append(row)
    for interval, path in enumerate(applied_paths):
        table = pd.read_csv(path)
        row = table.loc[table.agent_id.eq("micro_agent_032")].copy()
        if len(row) != 1:
            raise RuntimeError(f"Expected one micro_agent_032 row in {path}; found {len(row)}")
        row["interval"] = interval
        applied_rows.append(row)
    decision = pd.concat(decision_rows, ignore_index=True)
    applied = pd.concat(applied_rows, ignore_index=True)
    if not decision.dominant_program.eq("matrix_remodeling").all():
        raise RuntimeError("Frozen trace no longer has the expected matrix_remodeling program")
    checks = {
        "tgfb_writeback_exact": bool(
            np.allclose(decision.TGFb_like_signal, applied.TGFb_secretion, atol=0, rtol=0)
        ),
        "remodeling_writeback_exact": bool(
            np.allclose(decision.fibrosis_signal, applied.fibrosis_secretion, atol=0, rtol=0)
        ),
        "real_physicell_used": bool(applied.REAL_PHYSICELL_USED.all()),
        "llm_agent_decision_used": bool(applied.LLM_AGENT_DECISION_USED.all()),
        "hardcoded_transition_rules_disabled": bool(
            applied.HARDCODED_TRANSITION_RULES_DISABLED.all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Trace validation failed: {checks}")
    source = pd.DataFrame(
        {
            "decision": ["Decision 1", "Decision 2"],
            "dominant_program": decision.dominant_program,
            "agent_TGFb_like_output": decision.TGFb_like_signal,
            "agent_matrix_remodeling_output": decision.fibrosis_signal,
            "PhysiCell_TGFb_writeback": applied.TGFb_secretion,
            "PhysiCell_matrix_remodeling_writeback": applied.fibrosis_secretion,
        }
    )
    return source, checks, decision_paths + applied_paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    module = pd.read_csv(DATA / "module_heterogeneity_metrics.csv")
    cci = pd.read_csv(DATA / "cci_heterogeneity_metrics_primary.csv")
    trace, trace_checks, trace_paths = load_trace()

    # Use the original normalized map source, rather than a screenshot of the
    # already combined figure, whenever it is available.
    raw_map_source = (
        ROOT
        / "outputs/GSE267904_spatial_agent_granularity/figures/module_expression_K20_K50_K100_old_style_horizontal"
        / "GSE267904_module_expression_K20_K50_K100_old_style_panel_normalized.png"
    )
    map_source = raw_map_source if raw_map_source.exists() else SOURCE
    pixels = np.asarray(Image.open(map_source).convert("RGB"))
    chromatic = np.ptp(pixels.astype(np.int16), axis=2) > 30
    columns = [r for r in runs(chromatic.sum(axis=0) > 400) if r[1] - r[0] > 300]
    rows = [r for r in runs(chromatic.sum(axis=1) > 400) if r[1] - r[0] > 300]
    if len(columns) != 4 or len(rows) != 6:
        raise RuntimeError(f"Unable to recover 6x4 map grid from {map_source}: {columns}, {rows}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(22, 13.2))
    left = fig.add_gridspec(
        2, 3, left=0.035, right=0.495, bottom=0.315, top=0.79, wspace=0.38, hspace=0.52
    )
    right = fig.add_gridspec(
        6, 4, left=0.600, right=0.985, bottom=0.315, top=0.79, wspace=0.06, hspace=0.18
    )
    specs = [
        (module, "energy_distance_squared", "Module-state distribution distance", "Squared energy distance"),
        (module, "observed_state_mass_coverage", "Module-state coverage", "Observed state mass covered"),
        (module, "state_js_similarity", "Module-state proportional fidelity", "1 − Jensen–Shannon divergence"),
        (cci, "observed_mass_on_predicted_support", "CCI-state coverage", "Observed CCI-state mass covered"),
        (cci, "core_80_mass_recovery", "Core CCI-state recovery", "Observed core mass recovered"),
        (cci, "effective_diversity_ratio", "CCI-state diversity recovery", "Effective diversity / observed"),
    ]
    for i, (table, key, title, ylabel) in enumerate(specs):
        ax = fig.add_subplot(left[i // 3, i % 3])
        for stage, color, marker, style in [
            ("Early", "#D55E00", "o", "-"),
            ("Late", "#0072B2", "s", "--"),
        ]:
            subset = table[table.stage.eq(stage)].sort_values("K")
            ax.plot(
                subset.K,
                subset[key],
                color=color,
                marker=marker,
                ls=style,
                lw=2,
                ms=6,
                label="d7 (early)" if stage == "Early" else "d21 (late)",
            )
            for k, value in zip(subset.K, subset[key]):
                dy = 11 if stage == "Early" else -17
                if i == 0:
                    dy = 12 if stage == "Early" else -17
                    if k == 50:
                        dy = -20 if stage == "Early" else 14
                if value == 0:
                    dy = 8
                ax.annotate(
                    f"{value:.2f}" if i == 0 else f"{value:.3f}",
                    (k, value),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=color,
                    weight="bold",
                )
        ax.set_title(title, fontsize=10.5, weight="bold", pad=13)
        ax.text(-0.20, 1.06, "ABCDEF"[i], transform=ax.transAxes, fontsize=13, weight="bold")
        ax.set_xticks([20, 50, 100])
        ax.set_xlim(13, 107)
        ax.set_xlabel("Nominal Agent granularity (K)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e5eb")
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_yscale("log")
            ax.set_ylim(0.8, 11)
        else:
            ax.set_ylim(-0.045, 1.065)
        if i == 5:
            ax.axhline(1, color="#7c8390", ls=":", lw=1)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.265, 0.865),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    row_names = [
        "TGF-β response\nd7 (early)",
        "TGF-β response\nd21 (late)",
        "Inflammation\nd7 (early)",
        "Inflammation\nd21 (late)",
        "Fibrosis / ECM\nd7 (early)",
        "Fibrosis / ECM\nd21 (late)",
    ]
    boxes = []
    for r, (y0, y1) in enumerate(rows):
        for c, (x0, x1) in enumerate(columns):
            ax = fig.add_subplot(right[r, c])
            size = max(y1 - y0, x1 - x0)
            xc = (x0 + x1) // 2
            yc = (y0 + y1) // 2
            box = (xc - size // 2, yc - size // 2, xc - size // 2 + size, yc - size // 2 + size)
            crop = pixels[box[1] : box[3], box[0] : box[2]]
            ax.imshow(crop, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if r == 0:
                ax.set_title(["Real", "K=20", "K=50", "K=100"][c], fontsize=13, weight="bold", pad=10)
            if c == 0:
                pos = ax.get_position()
                fig.text(0.585, (pos.y0 + pos.y1) / 2, row_names[r], fontsize=11, ha="right", va="center", weight="bold")
            boxes.append([int(v) for v in box])

    fig.text(0.265, 0.965, "GSE267904 State-Recovery Benchmark\nacross Leca-VC Agent Granularities", ha="center", va="top", fontsize=16, weight="bold")
    fig.text(0.755, 0.965, "GSE267904 Module-Expression Recovery\nacross Agent Granularities", ha="center", va="top", fontsize=16, weight="bold")
    fig.text(0.535, 0.795, "G", fontsize=13, weight="bold")

    # Panel H: one frozen agent is traced from its structured program to the
    # exact values written to the real PhysiCell executor.
    bottom = fig.add_gridspec(1, 3, left=0.075, right=0.965, bottom=0.055, top=0.205, wspace=0.34)
    colors = {"TGFβ-like": "#CC79A7", "Matrix remodeling": "#009E73"}
    x = np.array([0, 1])
    labels = trace.decision.tolist()

    ax_tgfb = fig.add_subplot(bottom[0, 0])
    ax_matrix = fig.add_subplot(bottom[0, 1])
    ax_write = fig.add_subplot(bottom[0, 2])
    for ax, key, title, color in [
        (ax_tgfb, "agent_TGFb_like_output", "TGFβ-like program output", colors["TGFβ-like"]),
        (ax_matrix, "agent_matrix_remodeling_output", "Matrix-remodeling output", colors["Matrix remodeling"]),
    ]:
        values = trace[key].to_numpy(float)
        ax.plot(x, values, color=color, marker="o", ms=8, lw=2.6)
        for xx, value in zip(x, values):
            ax.annotate(f"{value:.2f}", (xx, value), xytext=(0, 9), textcoords="offset points", ha="center", color=color, weight="bold")
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0, 0.5)
        ax.set_ylabel("Validated output strength")
        ax.set_title(title, fontsize=11.5, weight="bold", pad=9)
        ax.grid(axis="y", color="#e1e5eb")
        ax.spines[["top", "right"]].set_visible(False)

    for interval, marker in [(0, "o"), (1, "s")]:
        row = trace.iloc[interval]
        for name, out_key, app_key in [
            ("TGFβ-like", "agent_TGFb_like_output", "PhysiCell_TGFb_writeback"),
            ("Matrix remodeling", "agent_matrix_remodeling_output", "PhysiCell_matrix_remodeling_writeback"),
        ]:
            ax_write.scatter(
                row[out_key], row[app_key], s=74, marker=marker, color=colors[name],
                edgecolor="white", linewidth=0.8,
                label=f"{name}, {labels[interval]}", zorder=3,
            )
    ax_write.plot([0, 0.5], [0, 0.5], color="#555555", ls="--", lw=1.4, label="Exact writeback")
    ax_write.set_xlim(0, 0.5)
    ax_write.set_ylim(0, 0.5)
    ax_write.set_aspect("equal", adjustable="box")
    ax_write.set_xlabel("Agent program output")
    ax_write.set_ylabel("PhysiCell-applied value")
    ax_write.set_title("Program-to-executor trace", fontsize=11.5, weight="bold", pad=9)
    ax_write.grid(color="#e1e5eb")
    ax_write.spines[["top", "right"]].set_visible(False)
    ax_write.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.02))

    fig.text(0.035, 0.267, "H", fontsize=13, weight="bold")
    fig.text(0.075, 0.270, "Traceable fibroblast matrix-remodeling program (K=100; agent 032)", fontsize=13.5, weight="bold", va="center")
    fig.text(0.075, 0.247, "Dominant program: matrix_remodeling at both decisions", fontsize=10.5, color="#4b5563", va="top")

    trace.to_csv(OUT / "fibroblast_program_trace_source_data.csv", index=False)
    stem = OUT / "GSE267904_metrics_spatial_maps_and_fibroblast_program_trace"
    fig.savefig(str(stem) + "_preview.png", dpi=140, bbox_inches="tight")
    fig.savefig(str(stem) + "_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)

    audit = {
        "status": "PASS_GSE267904_COMBINED_FIGURE_WITH_FIBROBLAST_PROGRAM_TRACE_V3",
        "scientific_metrics_recomputed": False,
        "llm_or_simulation_rerun": False,
        "trace_agent": "micro_agent_032",
        "trace_cell_type": "fibroblast_myofibroblast",
        "trace_selection": "frozen K=100 auditable example",
        "trace_checks": trace_checks,
        "metric_data": str(DATA),
        "map_source": str(map_source),
        "map_source_sha256": sha256(map_source),
        "trace_inputs": [{"path": str(p), "sha256": sha256(p)} for p in trace_paths],
        "map_crop_boxes": boxes,
        "source_outputs_modified": False,
        "interpretation_limit": "The trace shows strengthening and exact execution of an existing matrix_remodeling program; it does not establish a discrete cell-identity transition.",
    }
    (OUT / "COMPLETE.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
