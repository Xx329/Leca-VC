#!/usr/bin/env python3
"""Generate one first-person DeepSeek policy for each d7 cell-agent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, FIELDS, dump_json, ensure_dirs, load_config, resolve_out, sha256


POLICY_FIELDS = [
    "injury_response", "inflammatory_persistence", "macrophage_activation", "tgfb_response",
    "ecm_deposition", "homeostasis_repair", "motility", "survival", "proliferation", "memory_gain",
]


def first_person_payload(row: pd.Series) -> dict:
    return {
        "agent_id": str(row["agent_id"]),
        "identity": str(row["cell_type"]),
        "instruction": (
            f"You are one {row['cell_type']} representative virtual lung cell in a PhysiCell/BioFVM tissue. "
            "Decide only your own cell response to your local microenvironment. You are not a controller, "
            "you do not manage other cells, and you must not make a global tissue decision."
        ),
        "local_substrates": {field: float(row[field]) for field in FIELDS},
        "cell_state": {
            "fibrosis_memory": float(row["fibrosis_memory"]),
            "represented_abundance": float(row["represented_abundance"]),
        },
        "time": "t=0 initialized from d7 only; abstract PhysiCell progression time",
        "heldout_d21_visible": False,
        "output_schema": {field: "number from 0.0 to 1.0" for field in POLICY_FIELDS},
        "constraints": [
            "Return exactly one JSON object for this agent_id.",
            "Use the biological capabilities of your stated cell identity.",
            "Do not output secretion rates or physical units; output bounded response propensities only.",
            "Do not mention or infer real day21 values or any virtual intervention.",
        ],
    }


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[A-Za-z0-9_-]*", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_policy(obj: dict, agent_id: str, cell_type: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("LLM policy is not a JSON object")
    policy = obj.get("policy", obj)
    returned_agent = policy.get("agent_id", obj.get("agent_id"))
    if returned_agent is not None and str(returned_agent) != agent_id:
        raise ValueError(f"LLM policy agent_id mismatch: expected {agent_id}, got {returned_agent}")
    missing = [name for name in POLICY_FIELDS if name not in policy]
    if missing:
        raise ValueError(f"LLM policy for {agent_id} missing fields {missing}")
    extra = sorted(set(policy) - set(POLICY_FIELDS) - {"agent_id"})
    if extra:
        raise ValueError(f"LLM policy for {agent_id} has unexpected fields {extra}")
    row = {"agent_id": agent_id, "cell_type": cell_type}
    for name in POLICY_FIELDS:
        value = float(policy[name])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{agent_id} {name}={value} is outside [0,1]")
        row[name] = value
    return row


def deepseek_policy(payload: dict, raw_path: Path, retries: int, prompt_sha256: str) -> tuple[dict, dict]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required; no fake LLM policy fallback is allowed")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "qwen3.7-plus")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Connection": "close"}
    body = {
        "model": model, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are exactly one virtual cell. Return strict JSON only and decide only your own cell behavior from d7 local fields."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    error_path = raw_path.with_suffix(".errors.log")
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(base + "/chat/completions", json=body, headers=headers, timeout=(20, 240))
            if response.status_code >= 400 and "response_format" in response.text and "response_format" in body:
                body.pop("response_format", None)
                raise RuntimeError(f"HTTP {response.status_code} rejected response_format")
            response.raise_for_status()
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps({"prompt_sha256": prompt_sha256, "envelope": envelope, "content": content}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return extract_json(content), {"model": model, "base_url": base, "temperature": 0}
        except Exception as exc:
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(f"attempt {attempt}/{retries}: {type(exc).__name__}: {exc}\n")
            if attempt == retries:
                raise RuntimeError(f"DeepSeek failed for {payload['agent_id']} after {retries} attempts; see {error_path}") from exc
            time.sleep(min(30, 2 * attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sample-id", help="Optional single d7 section for incremental execution")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    ensure_dirs(out)
    config = load_config()
    registries = sorted((out / "agents").glob("GSM*_cell_agents.csv"))
    if args.sample_id:
        registries = [p for p in registries if p.name.startswith(args.sample_id + "_")]
    if not registries:
        raise FileNotFoundError("No d7 cell-agent registries found; run prepare-d7 first")
    audit_rows = []
    provider = None
    for registry in registries:
        agents = pd.read_csv(registry)
        if len(agents) != 50:
            raise RuntimeError(f"{registry} has {len(agents)} agents, expected 50")
        sample_id = str(agents["sample_id"].iloc[0])
        print(f"[{sample_id}] validating/generating 50 first-person cell policies", flush=True)
        policies = []
        prompt_hashes = []
        cache_hits = 0
        for agent_index, (_, agent) in enumerate(agents.iterrows(), start=1):
            agent_id = str(agent["agent_id"])
            payload = first_person_payload(agent)
            prompt_path = out / "llm/raw" / sample_id / f"{agent_id}_prompt.json"
            raw_path = out / "llm/raw" / sample_id / f"{agent_id}_response.json"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            prompt_sha256 = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()
            prompt_path.write_text(json.dumps({"prompt_sha256": prompt_sha256, "payload": payload}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            cached = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else None
            if cached and cached.get("prompt_sha256") == prompt_sha256:
                obj = extract_json(cached["content"])
                provider = {"model": cached.get("envelope", {}).get("model", "cached_deepseek"), "temperature": 0}
                cache_hits += 1
                source = "validated cache"
            else:
                obj, provider = deepseek_policy(payload, raw_path, args.retries, prompt_sha256)
                source = "new DeepSeek call"
            policies.append(validate_policy(obj, agent_id, str(agent["cell_type"])))
            prompt_hashes.append(prompt_sha256)
            print(f"[{sample_id}] {agent_index:02d}/50 {agent_id}: {source}", flush=True)
        policy_path = out / "llm/policies" / f"{sample_id}_cell_agent_policies.csv"
        pd.DataFrame(policies).to_csv(policy_path, index=False)
        audit_rows.append({
            "sample_id": sample_id, "n_cell_agent_policies": len(policies),
            "n_new_api_calls": len(policies) - cache_hits, "n_validated_cache_hits": cache_hits,
            "policy_sha256": sha256(policy_path),
            "prompt_set_sha256": hashlib.sha256("".join(sorted(prompt_hashes)).encode("ascii")).hexdigest(),
            "first_person_cell_semantics": True, "global_controller_semantics": False,
        })
    dump_json(out / "audit/llm_cell_agent_policy_audit.json", {
        "provider": provider, "temperature": 0, "heldout_d21_visible": False,
        "intervention_visible_to_llm": False, "samples": audit_rows,
        "policy_fields": POLICY_FIELDS,
    })
    print(f"Generated strict first-person cell policies for {len(registries)} d7 sections")


if __name__ == "__main__":
    main()
