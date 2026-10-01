"""Create a Supplementary Figure of auditable single-Agent program updates.

This analysis is visualization-only. It reads frozen, de-identified GSE267904
Agent decisions and the corresponding real PhysiCell writeback tables. It does
not call an LLM or rerun PhysiCell/BioFVM.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gse267904-agent-trace-supp-v1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "outputs/GSE267904_deidentified_rerun_v1"
OUT = ROOT / "outputs/GSE267904_agent_program_trace_supp_v1"
DATA_OUT = OUT / "data"
FIG_OUT = OUT / "figure"
AUDIT_OUT = OUT / "audit"

COMMUNICATION_MAP = {
    "damage_signal": "damage_secretion",
    "inflammatory_signal": "inflammatory_secretion",
    "fibrosis_signal": "fibrosis_secretion",
    "resolution_signal": "resolution_secretion",
    "TGFb_like_signal": "TGFb_secretion",
    "SPP1_like_signal": "SPP1_secretion",
    "chemokine_signal": "chemokine_secretion",
}

TRACE_SPECS = [
    {
        "panel": "A",
        "agent_id": "micro_agent_032",
        "cell_label": "Fibroblast/myofibroblast",
        "title": "TGFβ-associated matrix-remodeling activation",
        "series": [
            ("TGFβ-like output", "TGFb_like_signal", "#CC79A7"),
            ("Matrix-remodeling output", "fibrosis_signal", "#009E73"),
        ],
        "note": "Functional program intensification; annotated cell identity unchanged",
    },
    {
        "panel": "B",
        "agent_id": "micro_agent_014",
        "cell_label": "Macrophage",
        "title": "Inflammatory-to-resolution transition",
        "series": [
            ("Inflammatory output", "inflammatory_signal", "#D55E00"),
            ("Resolution output", "resolution_signal", "#0072B2"),
        ],
        "note": "Discrete dominant-program update",
    },
    {
        "panel": "C",
        "agent_id": "micro_agent_007",
        "cell_label": "Alveolar epithelial cell",
        "title": "Homeostatic-to-stress-adaptive repair transition",
        "series": [
            ("Damage-associated output", "damage_signal", "#E69F00"),
            ("Resolution output", "resolution_signal", "#0072B2"),
        ],
        "note": "Discrete dominant-program update",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(k: int):
    decision = [FROZEN / f"K_{k}/micro_agents/llm_decision_interval_{i}.csv" for i in (0, 1)]
    applied = [FROZEN / f"K_{k}/physicell/applied_decision_{i}.csv" for i in (0, 1)]
    return decision, applied


def load_k(k: int):
    decision_paths, applied_paths = source_paths(k)
    decision_frames = []
    applied_frames = []
    for interval, path in enumerate(decision_paths):
        table = pd.read_csv(path)
        table.insert(0, "interval", interval)
        decision_frames.append(table)
    for interval, path in enumerate(applied_paths):
        table = pd.read_csv(path)
        if "interval" not in table.columns:
            table.insert(0, "interval", interval)
        applied_frames.append(table)
    return pd.concat(decision_frames, ignore_index=True), pd.concat(applied_frames, ignore_index=True)


def collect_all_program_changes():
    rows = []
    for k in (20, 50, 100):
        decision, _ = load_k(k)
        before = decision.loc[decision.interval.eq(0)].copy()
        after = decision.loc[decision.interval.eq(1)].copy()
        merged = before.merge(after, on=["agent_id", "cell_type"], suffixes=("_decision1", "_decision2"))
        merged.insert(0, "K", k)
        merged["program_changed"] = merged.dominant_program_decision1.ne(merged.dominant_program_decision2)
        rows.append(merged)
    all_matched = pd.concat(rows, ignore_index=True)
    changes = all_matched.loc[all_matched.program_changed].copy()
    return all_matched, changes


def collect_writeback_pairs(decision: pd.DataFrame, applied: pd.DataFrame):
    merged = decision.merge(
        applied,
        on=["interval", "agent_id", "cell_type"],
        suffixes=("_agent", "_executor"),
        validate="one_to_one",
    )
    rows = []
    for agent_col, executor_col in COMMUNICATION_MAP.items():
        for _, row in merged.iterrows():
            rows.append(
                {
                    "interval": int(row.interval),
                    "decision": f"Decision {int(row.interval) + 1}",
                    "agent_id": row.agent_id,
                    "cell_type": row.cell_type,
                    "field": agent_col,
                    "agent_output": float(row[agent_col]),
                    "physicell_writeback": float(row[executor_col]),
                    "absolute_difference": abs(float(row[agent_col]) - float(row[executor_col])),
                }
            )
    return pd.DataFrame(rows), merged


def trace_table(decision: pd.DataFrame, applied: pd.DataFrame):
    rows = []
    for spec in TRACE_SPECS:
        for interval in (0, 1):
            selected = decision.loc[
                decision.interval.eq(interval) & decision.agent_id.eq(spec["agent_id"])
            ]
            executed = applied.loc[
                applied.interval.eq(interval) & applied.agent_id.eq(spec["agent_id"])
            ]
            if len(selected) != 1 or len(executed) != 1:
                raise RuntimeError(
                    f"Expected one frozen decision and writeback for {spec['agent_id']} interval {interval}"
                )
            d = selected.iloc[0]
            a = executed.iloc[0]
            for label, field, _ in spec["series"]:
                rows.append(
                    {
                        "panel": spec["panel"],
                        "agent_id": spec["agent_id"],
                        "cell_type": d.cell_type,
                        "decision": f"Decision {interval + 1}",
                        "interval": interval,
                        "dominant_program": d.dominant_program,
                        "displayed_output": label,
                        "agent_output_field": field,
                        "agent_output": float(d[field]),
                        "physicell_field": COMMUNICATION_MAP[field],
                        "physicell_writeback": float(a[COMMUNICATION_MAP[field]]),
                    }
                )
    return pd.DataFrame(rows)


def program_label(text: str) -> str:
    return text.replace("_", " ")


def draw_trace_panel(ax, decision: pd.DataFrame, spec: dict):
    selected = decision.loc[decision.agent_id.eq(spec["agent_id"])].sort_values("interval")
    if len(selected) != 2:
        raise RuntimeError(f"Expected two decisions for {spec['agent_id']}; found {len(selected)}")
    x = np.array([0.0, 1.0])
    plotted = []
    for label, field, color in spec["series"]:
        values = selected[field].to_numpy(float)
        ax.plot(x, values, marker="o", ms=8, lw=2.5, color=color, label=label, zorder=3)
        plotted.append((label, color, values))
    for series_index, (_, color, values) in enumerate(plotted):
        for point_index, (xx, value) in enumerate(zip(x, values)):
            offset = 9
            other = plotted[1 - series_index][2][point_index]
            if abs(value - other) < 0.035:
                offset = -18 if value < other else 9
            ax.annotate(
                f"{value:.2f}",
                (xx, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=10,
                weight="bold",
            )
    ax.set_xticks(x, ["Decision 1", "Decision 2"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Validated program output")
    ax.grid(axis="y", color="#DFE4EA", lw=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 1.30, spec["title"], transform=ax.transAxes, fontsize=12.2, weight="bold", ha="center", va="center")
    ax.text(-0.12, 1.30, spec["panel"], transform=ax.transAxes, fontsize=15, weight="bold", va="center")
    ax.text(0.0, 1.17, spec["cell_label"], transform=ax.transAxes, fontsize=10.5, weight="bold", va="center")
    p0 = program_label(selected.iloc[0].dominant_program)
    p1 = program_label(selected.iloc[1].dominant_program)
    box = dict(boxstyle="round,pad=0.35", facecolor="#F4F6F8", edgecolor="#9AA4B2", lw=0.9)
    ax.text(0.02, 1.04, p0, transform=ax.transAxes, fontsize=9.2, ha="left", va="center", bbox=box)
    ax.text(0.98, 1.04, p1, transform=ax.transAxes, fontsize=9.2, ha="right", va="center", bbox=box)
    ax.annotate(
        "",
        xy=(0.67, 1.04),
        xytext=(0.33, 1.04),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#4B5563", lw=1.5),
    )
    ax.text(0.5, -0.18, spec["note"], transform=ax.transAxes, ha="center", va="top", fontsize=9.2, color="#5A6472")
    ax.legend(frameon=False, fontsize=9.2, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2)


def render(decision: pd.DataFrame, writeback: pd.DataFrame):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(15.8, 10.0))
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.965, bottom=0.085, top=0.77, wspace=0.28, hspace=0.64)
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    for ax, spec in zip(axes[:3], TRACE_SPECS):
        draw_trace_panel(ax, decision, spec)

    ax = axes[3]
    colors = {"Decision 1": "#D55E00", "Decision 2": "#0072B2"}
    for label in ("Decision 1", "Decision 2"):
        subset = writeback.loc[writeback.decision.eq(label)]
        ax.scatter(
            subset.agent_output,
            subset.physicell_writeback,
            s=23,
            alpha=0.34,
            color=colors[label],
            edgecolor="none",
            label=f"{label} ({len(subset)} values)",
            zorder=2,
        )
    ax.plot([0, 0.5], [0, 0.5], ls="--", lw=1.7, color="#343A40", label="Identity line", zorder=3)
    ax.set_xlim(-0.015, 0.515)
    ax.set_ylim(-0.015, 0.515)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Validated Agent output")
    ax.set_ylabel("PhysiCell-applied value")
    ax.set_title("Program-to-executor fidelity", fontsize=12.2, weight="bold", pad=18)
    ax.text(-0.12, 1.12, "D", transform=ax.transAxes, fontsize=15, weight="bold")
    ax.grid(color="#DFE4EA", lw=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    max_delta = writeback.absolute_difference.max()
    ax.text(
        0.04,
        0.94,
        f"{len(writeback)}/{len(writeback)} exact writebacks\nmax |difference| = {max_delta:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#AAB2BD", alpha=0.94),
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("Auditable Leca-VC Agent-Program Updates in GSE267904", fontsize=19, weight="bold", y=0.972)
    fig.text(
        0.5,
        0.928,
        "Frozen de-identified K=100 simulation; each trace follows the same Agent across two decision intervals",
        ha="center",
        va="center",
        fontsize=10.8,
        color="#4B5563",
    )
    stem = FIG_OUT / "GSE267904_auditable_single_agent_program_updates"
    fig.savefig(str(stem) + "_preview.png", dpi=150)
    fig.savefig(str(stem) + "_600dpi.png", dpi=600)
    fig.savefig(str(stem) + ".pdf", metadata={"CreationDate": None})
    fig.savefig(str(stem) + ".svg", metadata={"Date": None})
    plt.close(fig)
    return stem


def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)

    decision, applied = load_k(100)
    # The frozen run contains 50 new decisions at the first boundary and 45 at
    # the second. Five Agents retained their prior program in the executor and
    # are not counted as newly emitted LLM decisions.
    if len(decision) != 95 or len(applied) != 100:
        raise RuntimeError(f"Expected 95 new K=100 decisions and 100 executor rows; found {len(decision)} and {len(applied)}")
    trace = trace_table(decision, applied)
    writeback, merged = collect_writeback_pairs(decision, applied)
    all_matched, changes = collect_all_program_changes()

    if writeback.absolute_difference.max() != 0:
        raise RuntimeError("At least one displayed communication output differs from its PhysiCell writeback")
    flags = [
        "REAL_PHYSICELL_USED",
        "PHYSICELL_USED_AS_SPATIAL_EXECUTOR",
        "HARDCODED_TRANSITION_RULES_DISABLED",
        "LLM_AGENT_DECISION_USED",
    ]
    if not all(bool(merged[col].all()) for col in flags):
        raise RuntimeError("Frozen execution-integrity flags are not all true")

    trace.to_csv(DATA_OUT / "selected_agent_traces.csv", index=False)
    writeback.to_csv(DATA_OUT / "all_K100_communication_writebacks.csv", index=False)
    all_matched.to_csv(DATA_OUT / "all_matched_agent_programs_K20_K50_K100.csv", index=False)
    changes.to_csv(DATA_OUT / "all_observed_program_changes_K20_K50_K100.csv", index=False)
    summary = (
        all_matched.groupby("K", as_index=False)
        .agg(matched_agents=("agent_id", "size"), changed_programs=("program_changed", "sum"))
    )
    summary["changed_fraction"] = summary.changed_programs / summary.matched_agents
    summary.to_csv(DATA_OUT / "program_change_summary.csv", index=False)

    stem = render(decision, writeback)
    caption = (
        "Auditable single-Agent program updates during the frozen de-identified GSE267904 spatial simulation. "
        "(A) A fibroblast/myofibroblast Agent retained a matrix-remodeling dominant program while its validated "
        "TGF-beta-like and matrix-remodeling outputs increased from 0.12 to 0.32 and from 0.18 to 0.38, "
        "respectively. This panel represents continuous activation of a TGF-beta-associated matrix-remodeling "
        "program rather than a change in annotated cell identity. (B) A macrophage switched from inflammatory "
        "recruitment to a resolution-transition program, with inflammatory output decreasing from 0.42 to 0.12 "
        "and resolution output increasing from 0.08 to 0.42. (C) An alveolar epithelial Agent switched from "
        "homeostatic maintenance to stress-adaptive repair, accompanied by increased damage-associated and "
        "resolution outputs. (D) Across all 665 communication parameters emitted in 95 K=100 Agent-interval "
        "decisions, validated Agent outputs were written unchanged to the real PhysiCell executor. The examples "
        "illustrate auditable program updates and should not be interpreted as independently validated cell-fate conversions."
    )
    (OUT / "caption.txt").write_text(caption + "\n")

    input_paths = []
    for k in (20, 50, 100):
        dpaths, apaths = source_paths(k)
        input_paths.extend(dpaths + apaths)
    audit = {
        "status": "PASS_GSE267904_AGENT_PROGRAM_TRACE_SUPP_V1",
        "analysis_type": "visualization-only from frozen outputs",
        "llm_calls": 0,
        "physicell_reruns": 0,
        "biofvm_reruns": 0,
        "selected_examples": [
            {"panel": s["panel"], "agent_id": s["agent_id"], "cell_label": s["cell_label"]}
            for s in TRACE_SPECS
        ],
        "new_agent_decisions": int(len(decision)),
        "executor_rows_including_retained_programs": int(len(applied)),
        "writeback_pairs": int(len(writeback)),
        "max_absolute_writeback_difference": float(writeback.absolute_difference.max()),
        "execution_integrity_flags": {col: bool(merged[col].all()) for col in flags},
        "program_change_summary": summary.to_dict(orient="records"),
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
        "outputs": {
            "preview_png": str(stem) + "_preview.png",
            "png_600dpi": str(stem) + "_600dpi.png",
            "pdf": str(stem) + ".pdf",
            "svg": str(stem) + ".svg",
        },
        "interpretation_limit": (
            "Panel A is continuous functional-program activation, not a discrete cell-identity transition; "
            "Panels B and C are dominant-program label changes, not independently validated lineage conversions."
        ),
    }
    (AUDIT_OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": audit["status"],
                "audit": str(AUDIT_OUT / "analysis_audit.json"),
                "figure_pdf": str(stem) + ".pdf",
                "figure_png_600dpi": str(stem) + "_600dpi.png",
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT)


if __name__ == "__main__":
    main()
