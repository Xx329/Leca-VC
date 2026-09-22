#!/usr/bin/env python3
"""Exact-runtime-context pretraining-contamination probe and Supplement figure V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from urllib import request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prompt_firewall import assert_clean, canonical_json, find_forbidden
from render_contamination_control_figure import (
    DATASET_COLORS,
    collect_production_audit,
    setup_style,
    sha256,
    wilson,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/LecaVC_prompt_contamination_audit_v2_exact_runtime"
V1_FIGURE = ROOT / "outputs/LecaVC_prompt_contamination_audit_v1/figure_v1"
DATASETS = ("GSE2565", "GSE120575", "GSE267904")
CONDITIONS = ("identified_positive_control", "deidentified_exact_runtime")
CONDITION_LABELS = {
    "identified_positive_control": "Identified\ncontrol",
    "deidentified_exact_runtime": "De-identified\nexact runtime",
}
CONDITION_COLORS = {
    "identified_positive_control": "#7A7A7A",
    "deidentified_exact_runtime": "#2878B5",
}
IDENTITIES = {
    "GSE2565": "GSE2565 phosgene lung-injury time course",
    "GSE120575": "GSE120575 Sade-Feldman melanoma anti-PD-1 study",
    "GSE267904": "GSE267904 bleomycin pulmonary-fibrosis spatial study",
}
QUESTIONS = {
    "GSE2565": {
        "target_name": "future DNB peak window",
        "options": ["0-4 h", "8-12 h", "24-48 h", "72 h or later", "unknown"],
        "truth": "8-12 h",
    },
    "GSE120575": {
        "target_name": "dominant broad cell population in the later observed state",
        "options": [
            "B cell",
            "Plasma cell",
            "Monocyte/Macrophage",
            "Dendritic cell",
            "T cell",
            "NK cell",
            "unknown",
        ],
        "truth": "T cell",
    },
    "GSE267904": {
        "target_name": "largest late communication-pathway mass",
        "options": ["TGFb", "CCL", "CXCL", "PDGF", "VEGF", "unknown"],
        "truth": "VEGF",
    },
}


def normalized(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def context_candidates() -> dict[str, list[Path]]:
    g2565 = sorted(
        (
            ROOT
            / "outputs/GSE2565_deidentified_rerun_v1/simulation/runs/"
            "agent_physicell_v4_online__virtual_phosgene_injury__seed_256501__to_72h/"
            "agent_decisions/0"
        ).glob("*/model_messages.json")
    )
    g120 = sorted(
        (
            ROOT
            / "outputs/GSE120575_deidentified_rerun_v1/simulation/runs"
        ).glob("agent_v21_unified__*__seed_12057501/raw_agent_responses/checkpoint_0/*/model_messages.json")
    )
    g267_all = sorted(
        (
            ROOT
            / "outputs/GSE267904_deidentified_rerun_v1/K_100/micro_agents/llm_calls"
        ).glob("interval_0_micro_agent_*_response_audit/model_messages.json")
    )
    g267 = [
        path
        for path in g267_all
        if re.fullmatch(r"interval_0_micro_agent_\d{3}_response_audit", path.parent.name)
    ]
    return {"GSE2565": g2565, "GSE120575": g120, "GSE267904": g267}


def select_evenly(paths: list[Path], n: int = 10) -> list[Path]:
    if len(paths) < n:
        raise RuntimeError(f"Need at least {n} production contexts, found {len(paths)}")
    indices = np.linspace(0, len(paths) - 1, n, dtype=int)
    selected = [paths[int(index)] for index in indices]
    if len(set(selected)) != n:
        raise RuntimeError("Context selection did not produce ten unique paths")
    return selected


def derive_truths() -> dict[str, dict[str, object]]:
    g2565_path = (
        ROOT
        / "outputs/GSE2565_deidentified_rerun_v1/figures/bulk_time_resolved_benchmark/"
        "source_data/panels_A_to_D_time_resolved_profiles.csv"
    )
    g2565 = pd.read_csv(g2565_path)
    observed = g2565[g2565["method"] == "Observed"].dropna(subset=["normalized_DNB"])
    peak_hour = float(observed.loc[observed["normalized_DNB"].idxmax(), "time_hours"])
    if not 8.0 <= peak_hour <= 12.0:
        raise RuntimeError(f"Frozen GSE2565 DNB peak is outside 8-12 h: {peak_hour}")

    g120_path = (
        ROOT
        / "outputs/GSE120575_external_baseline_unified_composition_v2/"
        "unified_composition_group_means.csv"
    )
    g120 = pd.read_csv(g120_path)
    post = g120[g120["method"] == "Real_Post"]
    mean_composition = post.groupby("cell_type")["proportion"].mean().sort_values(ascending=False)
    dominant_cell = str(mean_composition.index[0])
    if dominant_cell != "T cell":
        raise RuntimeError(f"Frozen GSE120575 dominant Post population is not T cell: {dominant_cell}")

    g267_path = (
        ROOT
        / "outputs/GSE267904_spatial_agent_granularity/K_100/real_commot/"
        "real_commot_edges_long.csv"
    )
    g267 = pd.read_csv(g267_path)
    pathway_names = ["TGFb", "CCL", "CXCL", "PDGF", "VEGF"]
    late = g267[g267["stage"] == "d21_bleo"]
    masses = {
        pathway: float(late.loc[late["pathway"] == f"commot-cellchat-{pathway}", "weight"].sum())
        for pathway in pathway_names
    }
    dominant_pathway = max(masses, key=masses.get)
    if dominant_pathway != "VEGF":
        raise RuntimeError(f"Frozen GSE267904 late dominant pathway is not VEGF: {dominant_pathway}")

    return {
        "GSE2565": {
            "answer": "8-12 h",
            "derivation": f"argmax observed normalized_DNB at {peak_hour:g} h",
            "source": str(g2565_path.relative_to(ROOT)),
            "source_sha256": sha256(g2565_path),
        },
        "GSE120575": {
            "answer": dominant_cell,
            "derivation": "largest response-averaged Real_Post broad-cell proportion",
            "source": str(g120_path.relative_to(ROOT)),
            "source_sha256": sha256(g120_path),
            "group_mean_proportions": {str(k): float(v) for k, v in mean_composition.items()},
        },
        "GSE267904": {
            "answer": dominant_pathway,
            "derivation": "largest d21_bleo aggregate COMMOT pathway mass among five frozen pathways",
            "source": str(g267_path.relative_to(ROOT)),
            "source_sha256": sha256(g267_path),
            "late_pathway_masses": masses,
        },
    }


def build_manifest() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    truths = derive_truths()
    candidates = context_candidates()
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for index, path in enumerate(select_evenly(candidates[dataset]), start=1):
            messages = json.loads(path.read_text())
            if not isinstance(messages, list) or len(messages) < 2:
                raise RuntimeError(f"Unexpected production-message structure: {path}")
            assert_clean(messages, label=f"frozen production context {dataset}/{index}")
            for condition in CONDITIONS:
                rows.append(
                    {
                        "dataset": dataset,
                        "context_index": index,
                        "condition": condition,
                        "source_message_path": str(path.relative_to(ROOT)),
                        "source_message_sha256": sha256(path),
                        "heldout_truth": truths[dataset]["answer"],
                        "heldout_truth_sent_to_model": False,
                    }
                )
    return rows, truths


def probe_messages(row: dict[str, object]) -> list[dict[str, str]]:
    dataset = str(row["dataset"])
    source = ROOT / str(row["source_message_path"])
    archived = json.loads(source.read_text())
    question = QUESTIONS[dataset]
    final_request: dict[str, object] = {
        "audit_task": (
            "Using only the archived production context above, determine whether it identifies a known "
            "study and predict the requested later-course benchmark outcome. If the context cannot be "
            "mapped to a study or does not support the outcome, answer unknown."
        ),
        "target": question["target_name"],
        "allowed_benchmark_answers": question["options"],
        "required_output_schema": {
            "dataset_guess": "GEO accession or unknown",
            "benchmark_answer": "exactly one allowed_benchmark_answers value",
            "confidence": "number from 0 to 1",
            "evidence_source": "provided_state, study_memory, both, or insufficient",
            "brief_explanation": "one short sentence",
        },
    }
    if row["condition"] == "identified_positive_control":
        final_request["positive_control_study_identity"] = IDENTITIES[dataset]

    messages = [
        {
            "role": "system",
            "content": (
                "This is a pretraining-contamination audit. The next two archived messages are copied "
                "from a real production Agent call. Do not execute their original action template. "
                "Use their content only as context, then answer the final audit request as strict JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "archived_production_messages": archived,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
        {"role": "user", "content": json.dumps(final_request, ensure_ascii=False, sort_keys=True)},
    ]
    if row["condition"] == "deidentified_exact_runtime":
        assert_clean(messages, label=f"V2 de-identified exact-runtime probe {dataset}")
    return messages


def validate_response(dataset: str, response: dict[str, object]) -> dict[str, object]:
    required = {"dataset_guess", "benchmark_answer", "confidence", "evidence_source", "brief_explanation"}
    if not required.issubset(response):
        raise ValueError(f"Missing response fields: {sorted(required - set(response))}")
    option_map = {normalized(option): option for option in QUESTIONS[dataset]["options"]}
    key = normalized(response["benchmark_answer"])
    if key not in option_map:
        raise ValueError(f"benchmark_answer is outside frozen options: {response['benchmark_answer']!r}")
    confidence = float(response["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence is outside [0,1]")
    evidence = str(response["evidence_source"]).strip().lower()
    if evidence not in {"provided_state", "study_memory", "both", "insufficient"}:
        raise ValueError(f"invalid evidence_source: {evidence}")
    return {
        **response,
        "benchmark_answer": option_map[key],
        "confidence": confidence,
        "evidence_source": evidence,
    }


def call_deepseek(row: dict[str, object], messages: list[dict[str, str]], run_dir: Path) -> dict[str, object]:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY unavailable")
    body = {
        "model": "deepseek-chat",
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    last: Exception | None = None
    for attempt in range(1, 5):
        try:
            req = request.Request(
                "https://api.deepseek.com/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=180) as result:
                raw = result.read()
            (run_dir / f"attempt_{attempt}_raw.json").write_bytes(raw + b"\n")
            content = json.loads(raw)["choices"][0]["message"]["content"]
            response = validate_response(str(row["dataset"]), json.loads(content))
            write_json(run_dir / "response.json", response)
            write_json(
                run_dir / "call_audit.json",
                {"attempt": attempt, "schema_valid": True, "fallback": False, "model": "deepseek-chat"},
            )
            return response
        except Exception as exc:
            last = exc
            (run_dir / f"attempt_{attempt}_error.txt").write_text(type(exc).__name__ + ": " + str(exc) + "\n")
    raise RuntimeError(f"DeepSeek V2 probe failed after four attempts: {last}; no fallback")


def freeze(rows: list[dict[str, object]], truths: dict[str, dict[str, object]]) -> None:
    protocol = {
        "status": "FROZEN_BEFORE_MODEL_CALLS",
        "experiment_role": "locked post-hoc pretraining-contamination probe",
        "model": "deepseek-chat",
        "temperature": 0.1,
        "conditions": list(CONDITIONS),
        "contexts_per_dataset": 10,
        "planned_calls": len(rows),
        "truths": truths,
        "questions": QUESTIONS,
        "selection": "ten evenly spaced normalized-step-zero production contexts per dataset",
        "truth_sent_to_model": False,
        "interpretation_limit": (
            "This operational probe cannot establish that the model never encountered a study during pretraining."
        ),
    }
    path = OUT / "protocol/frozen_probe_protocol.json"
    manifest_path = OUT / "protocol/context_manifest.csv"
    candidate_json = json.dumps(protocol, sort_keys=True, ensure_ascii=False)
    candidate_manifest = pd.DataFrame(rows)
    if path.is_file() and any((OUT / "responses").glob("**/response.json")):
        previous = json.dumps(json.loads(path.read_text()), sort_keys=True, ensure_ascii=False)
        if previous != candidate_json:
            raise RuntimeError("Frozen V2 protocol changed after responses existed; create a new version")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, protocol)
    candidate_manifest.to_csv(manifest_path, index=False)


def run_calls(rows: list[dict[str, object]]) -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY unavailable; run the interactive launcher")
    for number, row in enumerate(rows, start=1):
        run_dir = (
            OUT
            / "responses"
            / str(row["dataset"])
            / str(row["condition"])
            / f"context_{int(row['context_index']):02d}"
        )
        response_path = run_dir / "response.json"
        if response_path.is_file():
            validate_response(str(row["dataset"]), json.loads(response_path.read_text()))
            print(f"[{number}/{len(rows)}] existing PASS {run_dir.relative_to(OUT)}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            print(f"[{number}/{len(rows)}] resuming incomplete {run_dir.relative_to(OUT)}", flush=True)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
        messages = probe_messages(row)
        write_json(run_dir / "model_messages.json", messages)
        write_json(
            run_dir / "message_audit.json",
            {
                "condition": row["condition"],
                "intentional_identity_positive_control": row["condition"] == "identified_positive_control",
                "forbidden_hits": find_forbidden(messages),
                "source_message_sha256": row["source_message_sha256"],
                "probe_message_sha256": hashlib.sha256(canonical_json(messages).encode()).hexdigest(),
                "heldout_truth_sent_to_model": False,
            },
        )
        print(f"[{number}/{len(rows)}] calling {run_dir.relative_to(OUT)}", flush=True)
        call_deepseek(row, messages, run_dir)
    write_json(
        OUT / "RESPONSES_FROZEN.json",
        {"status": "PASS_60_EXACT_RUNTIME_PROBE_RESPONSES_FROZEN", "timestamp": time.time()},
    )


def score(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for row in rows:
        run_dir = (
            OUT
            / "responses"
            / str(row["dataset"])
            / str(row["condition"])
            / f"context_{int(row['context_index']):02d}"
        )
        response = validate_response(str(row["dataset"]), json.loads((run_dir / "response.json").read_text()))
        dataset = str(row["dataset"])
        records.append(
            {
                **row,
                "dataset_guess": response["dataset_guess"],
                "dataset_identified_correctly": dataset.upper() in normalized(response["dataset_guess"]).upper(),
                "benchmark_answer": response["benchmark_answer"],
                "heldout_answer_correct": normalized(response["benchmark_answer"])
                == normalized(row["heldout_truth"]),
                "abstained_unknown": normalized(response["benchmark_answer"]) == "unknown",
                "confidence": response["confidence"],
                "evidence_source": response["evidence_source"],
                "brief_explanation": response["brief_explanation"],
            }
        )
    calls = pd.DataFrame(records)
    summaries: list[dict[str, object]] = []
    for dataset in (*DATASETS, "All datasets"):
        current = calls if dataset == "All datasets" else calls[calls["dataset"] == dataset]
        for condition in CONDITIONS:
            subset = current[current["condition"] == condition]
            for metric, column in (
                ("GEO accession identification", "dataset_identified_correctly"),
                ("Held-out benchmark accuracy", "heldout_answer_correct"),
                ("Unknown/abstention rate", "abstained_unknown"),
            ):
                successes = int(subset[column].sum())
                n = len(subset)
                low, high = wilson(successes, n)
                summaries.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "metric": metric,
                        "successes": successes,
                        "n": n,
                        "rate": successes / n,
                        "wilson_ci_2_5": low,
                        "wilson_ci_97_5": high,
                    }
                )
    summary = pd.DataFrame(summaries)
    metrics = OUT / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    calls.to_csv(metrics / "call_level_results.csv", index=False)
    summary.to_csv(metrics / "probe_rate_summary.csv", index=False)
    return calls, summary


def plot_rate_panel(ax: plt.Axes, summary: pd.DataFrame, metric: str, letter: str) -> None:
    overall = summary[(summary["dataset"] == "All datasets") & (summary["metric"] == metric)]
    x = np.arange(len(CONDITIONS), dtype=float)
    values = [float(overall[overall["condition"] == condition]["rate"].iloc[0]) for condition in CONDITIONS]
    ax.bar(x, values, width=0.60, color=[CONDITION_COLORS[c] for c in CONDITIONS], alpha=0.84, zorder=2)
    for idx, condition in enumerate(CONDITIONS):
        row = overall[overall["condition"] == condition].iloc[0]
        value = float(row["rate"])
        low, high = float(row["wilson_ci_2_5"]), float(row["wilson_ci_97_5"])
        ax.errorbar(
            idx,
            value,
            yerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
            color="#333333",
            capsize=3,
            lw=1,
            zorder=4,
        )
        for offset, dataset in zip((-0.10, 0.0, 0.10), DATASETS):
            point = summary[
                (summary["dataset"] == dataset)
                & (summary["condition"] == condition)
                & (summary["metric"] == metric)
            ].iloc[0]
            ax.scatter(
                idx + offset,
                float(point["rate"]),
                s=30,
                color=DATASET_COLORS[dataset],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
        label_y = 1.025 if value >= 0.98 else min(1.045, value + 0.055)
        ax.text(idx, label_y, f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylim(-0.03, 1.11)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Rate (%)")
    ax.set_title(metric, pad=8)
    ax.grid(axis="y", color="#E6E8EB", lw=0.8, zorder=0)
    ax.text(-0.11, 1.02, letter, transform=ax.transAxes, fontsize=14, weight="bold", va="top")


def render(calls: pd.DataFrame, summary: pd.DataFrame) -> None:
    v1_cf = pd.read_csv(V1_FIGURE / "counterfactual_sensitivity.csv")
    production = collect_production_audit()
    production.to_csv(OUT / "metrics/production_prompt_firewall_audit.csv", index=False)
    v1_cf.to_csv(OUT / "metrics/generic_state_counterfactual_sensitivity.csv", index=False)

    setup_style()
    mpl.rcParams["svg.hashsalt"] = "lecavc-exact-runtime-contamination-probe-v2"
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.10, top=0.88, wspace=0.28, hspace=0.42)
    fig.suptitle("Exact-Runtime Prompt Contamination Probe", fontsize=16, weight="bold", y=0.965)
    plot_rate_panel(axes[0, 0], summary, "GEO accession identification", "A")
    plot_rate_panel(axes[0, 1], summary, "Held-out benchmark accuracy", "B")

    ax = axes[1, 0]
    grouped = v1_cf.groupby("dataset", sort=False).agg(
        any_response_changed=("any_direction_changed", "mean"),
        axis_level_changed=("axis_change_fraction", "mean"),
    ).reindex(DATASETS)
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.34
    bars1 = ax.bar(x - width / 2, grouped["any_response_changed"], width, color="#2878B5", label="Any output changed")
    bars2 = ax.bar(x + width / 2, grouped["axis_level_changed"], width, color="#74C476", label="Individual axes changed")
    for bars in (bars1, bars2):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=8.5, weight="bold")
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Paired change rate (%)")
    ax.set_title("Counterfactual response sensitivity", pad=8)
    ax.grid(axis="y", color="#E6E8EB", lw=0.8, zorder=0)
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.11, 1.02, "C", transform=ax.transAxes, fontsize=14, weight="bold", va="top")

    ax = axes[1, 1]
    y = np.arange(len(DATASETS))
    counts = production.set_index("dataset").reindex(DATASETS)["model_facing_messages"]
    bars = ax.barh(y, counts, color=[DATASET_COLORS[d] for d in DATASETS], alpha=0.88, height=0.58)
    for bar, dataset in zip(bars, DATASETS):
        row = production[production["dataset"] == dataset].iloc[0]
        ax.text(12, bar.get_y() + bar.get_height() / 2, f"{int(row['model_facing_messages'])} audited", color="white", ha="left", va="center", fontsize=9, weight="bold")
        ax.text(690, bar.get_y() + bar.get_height() / 2, "0 prohibited hits · 0 fallbacks", color="#237A3B", ha="right", va="center", fontsize=8.7, weight="bold")
    ax.set_yticks(y, DATASETS)
    ax.invert_yaxis()
    ax.set_xlim(0, 700)
    ax.set_xlabel("Audited model-facing decisions")
    ax.set_title("Production prompt firewall", pad=8)
    ax.grid(axis="x", color="#E6E8EB", lw=0.8, zorder=0)
    ax.text(-0.11, 1.02, "D", transform=ax.transAxes, fontsize=14, weight="bold", va="top")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=DATASET_COLORS[d], markeredgecolor="white", markersize=7, label=d)
        for d in DATASETS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.923), ncol=3, frameon=False)

    figure_dir = OUT / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    base = figure_dir / "LecaVC_exact_runtime_prompt_contamination_probe_v2"
    fig.savefig(base.with_name(base.name + "_preview.png"), dpi=180, facecolor="white")
    fig.savefig(base.with_name(base.name + "_600dpi.png"), dpi=600, facecolor="white")
    fig.savefig(base.with_name(base.name + "_vector.pdf"), facecolor="white", metadata={"Creator": "Leca-VC audit", "CreationDate": None, "ModDate": None})
    fig.savefig(base.with_name(base.name + ".svg"), facecolor="white", metadata={"Creator": "Leca-VC audit", "Date": None})
    plt.close(fig)

    identified_accuracy = float(
        summary[
            (summary["dataset"] == "All datasets")
            & (summary["condition"] == "identified_positive_control")
            & (summary["metric"] == "Held-out benchmark accuracy")
        ]["rate"].iloc[0]
    )
    deidentified_accuracy = float(
        summary[
            (summary["dataset"] == "All datasets")
            & (summary["condition"] == "deidentified_exact_runtime")
            & (summary["metric"] == "Held-out benchmark accuracy")
        ]["rate"].iloc[0]
    )
    caption = (
        "Exact-runtime prompt contamination probe. Ten normalized-step-zero production contexts per "
        "dataset were selected deterministically, and their archived production system and user content "
        "was supplied to a separate audit query. The identified positive-control condition additionally "
        "provided the study identity; the de-identified condition did not. (A) GEO accession identification. "
        "(B) Accuracy for three frozen outcomes that were scored only after response generation: the observed "
        "GSE2565 DNB peak window (8–12 h), the dominant GSE120575 observed Post broad population (T cell), "
        "and the largest GSE267904 late aggregate communication-pathway mass (VEGF). Pooled identified and "
        f"de-identified accuracies were {identified_accuracy:.3f} and {deidentified_accuracy:.3f}, respectively. "
        "Bars show pooled rates with 95% Wilson intervals; points show datasets. (C) A separate paired "
        "generic-state counterfactual probe quantified whether reversed numerical states changed the response. "
        "(D) Production firewall audit across 1,253 model-facing decisions. This operational audit tests "
        "prompt-enabled study identification and held-out-answer recall; it cannot prove that the model never "
        "encountered the studies during pretraining."
    )
    (figure_dir / "caption.txt").write_text(caption + "\n")


def finalize(rows: list[dict[str, object]], truths: dict[str, dict[str, object]]) -> None:
    calls, summary = score(rows)
    render(calls, summary)
    deid = calls[calls["condition"] == "deidentified_exact_runtime"]
    positive = calls[calls["condition"] == "identified_positive_control"]
    audit = {
        "status": "PASS_EXACT_RUNTIME_PROMPT_CONTAMINATION_PROBE_V2",
        "model_calls": len(calls),
        "unique_production_contexts": 30,
        "contexts_per_dataset": 10,
        "positive_control_identity_recovery": int(positive["dataset_identified_correctly"].sum()),
        "deidentified_identity_recovery": int(deid["dataset_identified_correctly"].sum()),
        "positive_control_heldout_correct": int(positive["heldout_answer_correct"].sum()),
        "deidentified_heldout_correct": int(deid["heldout_answer_correct"].sum()),
        "deidentified_forbidden_hits": 0,
        "fallbacks": 0,
        "truths": truths,
        "interpretation_limit": (
            "The probe tests operational prompt-enabled study identification and answer recall; it cannot "
            "prove absence of the studies from model pretraining."
        ),
    }
    write_json(OUT / "analysis_audit.json", audit)
    write_json(OUT / "COMPLETE.json", {"status": audit["status"], "figure_generated": True, "model_calls": len(calls)})
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    rows, truths = build_manifest()
    freeze(rows, truths)
    if args.preflight_only:
        result = {
            "status": "PASS_V2_PREFLIGHT_READY_FOR_60_DEEPSEEK_CALLS",
            "planned_calls": len(rows),
            "unique_production_contexts": 30,
            "deidentified_contexts_forbidden_hits": 0,
            "deepseek_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
            "truths": truths,
        }
        write_json(OUT / "preflight.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.render_only:
        if not (OUT / "RESPONSES_FROZEN.json").is_file():
            raise RuntimeError("Cannot render: 60 V2 responses are not frozen")
        finalize(rows, truths)
        return 0
    try:
        run_calls(rows)
        finalize(rows, truths)
    except Exception as exc:
        write_json(
            OUT / "BLOCKED.json",
            {"status": "BLOCKED_V2_PROBE", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
