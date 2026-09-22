#!/usr/bin/env python3
"""Prospective DeepSeek prompt-identification and counterfactual probe (90 calls)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib import request

from prompt_firewall import PROTOCOL_ID, assert_clean, canonical_json, find_forbidden


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/LecaVC_prompt_contamination_audit_v1/probe"
DATASETS = ("GSE2565", "GSE120575", "GSE267904")
CONDITIONS = ("identified_positive_control", "deidentified_runtime", "counterfactual")
REPLICATES = range(1, 11)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def payload(dataset: str, condition: str) -> dict[str, object]:
    generic = {
        "protocol": PROTOCOL_ID,
        "normalized_step": 0.5,
        "current_state": {
            "damage": 0.72,
            "inflammation": 0.64,
            "repair": 0.18,
            "population_shift": 0.31,
        },
        "task": (
            "Return JSON with dataset_guess, accession_confidence, next_change_direction, "
            "peak_window_bin, and a one-sentence evidence_source. Guess unknown when identity "
            "cannot be inferred from supplied values."
        ),
    }
    if condition == "counterfactual":
        generic["current_state"] = {
            "damage": 0.12,
            "inflammation": 0.08,
            "repair": 0.79,
            "population_shift": -0.26,
        }
        generic["counterfactual_instruction"] = "Follow these reversed numerical trends only."
    if condition == "identified_positive_control":
        identified = {
            "GSE2565": "GSE2565 phosgene lung injury bulk time course",
            "GSE120575": "GSE120575 Sade-Feldman melanoma anti-PD-1 responder study",
            "GSE267904": "GSE267904 bleomycin pulmonary fibrosis d7 to d21 spatial study",
        }
        generic["study_identity"] = identified[dataset]
    return generic


def manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for replicate in REPLICATES:
                item = payload(dataset, condition)
                if condition != "identified_positive_control":
                    assert_clean(item, label=f"probe {dataset}/{condition}")
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "replicate": replicate,
                        "request_sha256": hashlib.sha256(canonical_json(item).encode()).hexdigest(),
                        "intentional_forbidden_hits": find_forbidden(item),
                    }
                )
    return rows


def call_deepseek(messages: list[dict[str, str]], path: Path) -> dict[str, object]:
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
            with request.urlopen(req, timeout=120) as response:
                raw = response.read()
            (path.parent / f"attempt_{attempt}_raw.json").write_bytes(raw + b"\n")
            content = json.loads(raw)["choices"][0]["message"]["content"]
            result = json.loads(content)
            required = {"dataset_guess", "accession_confidence", "next_change_direction", "peak_window_bin", "evidence_source"}
            if not required.issubset(result):
                raise ValueError(f"missing fields: {sorted(required - set(result))}")
            write_json(path, result)
            write_json(path.parent / "call_audit.json", {"attempt": attempt, "fallback": False, "schema_valid": True})
            return result
        except Exception as exc:
            last = exc
            (path.parent / f"attempt_{attempt}_error.txt").write_text(type(exc).__name__ + ": " + str(exc) + "\n")
    raise RuntimeError(f"DeepSeek probe failed after four attempts: {last}; no fallback")


def run() -> None:
    frozen = manifest()
    write_json(OUT / "frozen_probe_manifest.json", {"status": "FROZEN", "n_calls": len(frozen), "rows": frozen})
    for row in frozen:
        run_dir = OUT / str(row["dataset"]) / str(row["condition"]) / f"replicate_{int(row['replicate']):02d}"
        response_path = run_dir / "response.json"
        if response_path.is_file():
            continue
        item = payload(str(row["dataset"]), str(row["condition"]))
        messages = [
            {
                "role": "system",
                "content": (
                    "This is a prompt-information audit, not a production biological decision. "
                    "Use only the user message and return strict JSON."
                ),
            },
            {"role": "user", "content": json.dumps(item, ensure_ascii=False, sort_keys=True)},
        ]
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "model_messages.json", messages)
        write_json(
            run_dir / "message_audit.json",
            {
                "intentional_positive_control": row["condition"] == "identified_positive_control",
                "forbidden_hits": find_forbidden(messages),
                "message_sha256": hashlib.sha256(canonical_json(messages).encode()).hexdigest(),
            },
        )
        call_deepseek(messages, response_path)
    write_json(OUT / "RESPONSES_FROZEN.json", {"status": "PASS_90_RESPONSES_FROZEN", "timestamp": time.time()})
    score()


def score() -> None:
    if not (OUT / "RESPONSES_FROZEN.json").is_file():
        raise RuntimeError("responses must be frozen before offline scoring")
    rows = []
    for item in manifest():
        response = json.loads(
            (OUT / str(item["dataset"]) / str(item["condition"]) / f"replicate_{int(item['replicate']):02d}/response.json").read_text()
        )
        guess = str(response["dataset_guess"]).replace("-", "").replace("_", "").upper()
        rows.append(
            {
                **item,
                "identified_correctly": str(item["dataset"]).upper() in guess,
                "next_change_direction": response["next_change_direction"],
                "peak_window_bin": response["peak_window_bin"],
            }
        )
    summary = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            subset = [row for row in rows if row["dataset"] == dataset and row["condition"] == condition]
            summary[f"{dataset}|{condition}"] = {
                "n": len(subset),
                "geo_identification_rate": sum(bool(row["identified_correctly"]) for row in subset) / len(subset),
                "next_change_directions": [row["next_change_direction"] for row in subset],
                "peak_window_bins": [row["peak_window_bin"] for row in subset],
            }
    counterfactual_consistency = {}
    for dataset in DATASETS:
        base = {
            int(row["replicate"]): str(row["next_change_direction"]).strip().lower()
            for row in rows
            if row["dataset"] == dataset and row["condition"] == "deidentified_runtime"
        }
        counter = {
            int(row["replicate"]): str(row["next_change_direction"]).strip().lower()
            for row in rows
            if row["dataset"] == dataset and row["condition"] == "counterfactual"
        }
        paired = sorted(set(base) & set(counter))
        counterfactual_consistency[dataset] = {
            "n_paired": len(paired),
            "direction_changed_rate": sum(base[i] != counter[i] for i in paired) / len(paired),
            "paired_directions": [{"replicate": i, "runtime": base[i], "counterfactual": counter[i]} for i in paired],
        }
    write_json(
        OUT / "offline_probe_scores.json",
        {
            "status": "PASS_OFFLINE_SCORING_COMPLETE",
            "summary": summary,
            "counterfactual_consistency": counterfactual_consistency,
            "interpretation_limit": "This probes prompt-enabled recognition/recall; it cannot prove absence from model pretraining.",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    rows = manifest()
    if args.preflight_only:
        result = {
            "status": "PASS_PROBE_PREFLIGHT_READY_FOR_DEEPSEEK",
            "planned_calls": len(rows),
            "identified_positive_controls": 30,
            "clean_deidentified_calls": 60,
            "deepseek_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
            "production_started": False,
        }
        write_json(OUT / "preflight.json", result)
        print(json.dumps(result, indent=2))
        return 0
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
