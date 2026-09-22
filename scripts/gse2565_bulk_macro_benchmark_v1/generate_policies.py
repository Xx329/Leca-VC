#!/usr/bin/env python3
"""Generate frozen-schema rule and real-LLM Agent policy schedules."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
from urllib import request

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v1"
POLICY = OUT / "policies"
FIELDS = ["oxidative_stress", "epithelial_injury", "inflammation", "edema_proxy", "death_signal", "repair_signal"]
OUTPUT_KEYS = [*FIELDS, "toxicant_uptake", "motility", "injury_sensitivity", "repair_response"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def rule_values(agent: str, condition: str, hour: float) -> dict[str, float]:
    injury = condition == "virtual_phosgene_injury"
    z = {k: 0.02 for k in OUTPUT_KEYS}
    z.update({"toxicant_uptake": 0.05, "motility": 0.08, "injury_sensitivity": 0.25, "repair_response": 0.25})
    if agent in {"AT1", "AT2", "injured_Krt8_epithelial", "airway_epithelial"}:
        z.update({"oxidative_stress": .35, "epithelial_injury": .38, "death_signal": .16, "repair_signal": .18, "toxicant_uptake": .28, "injury_sensitivity": .72, "repair_response": .55})
    if agent in {"resident_macrophage", "recruited_monocyte_macrophage", "neutrophil"}:
        z.update({"inflammation": .42, "oxidative_stress": .16, "death_signal": .08, "motility": .45, "injury_sensitivity": .48, "repair_signal": .10})
    if agent == "endothelial":
        z.update({"edema_proxy": .40, "epithelial_injury": .12, "toxicant_uptake": .18, "injury_sensitivity": .62})
    if agent == "fibroblast_stromal":
        z.update({"repair_signal": .34, "inflammation": .10, "motility": .18, "repair_response": .68})
    if agent == "lymphoid_immune":
        z.update({"inflammation": .14, "motility": .34})
    if not injury:
        for k in FIELDS[:-1]:
            z[k] *= .12
        z["repair_signal"] *= .35
    if hour >= 8:
        z["repair_signal"] *= 1.35
        z["inflammation"] *= .82
    return {k: min(1.0, max(0.0, float(v))) for k, v in z.items()}


def prompt_for(agent: str, condition: str, hour: float) -> dict:
    return {
        "experiment": "GSE2565 bulk macro dynamics benchmark V1",
        "role": "cell-type/state Agent conditional program, not a tissue coordinator",
        "agent_type": agent,
        "condition": condition,
        "decision_checkpoint_hour": hour,
        "exposure_note": "normalized toxicant proxy; not measured tissue phosgene concentration",
        "available_evidence": [
            "independent healthy identity from GSE141259 WholeLung PBS",
            "general acute phosgene injury mechanisms: early oxidative epithelial and endothelial injury, inflammation, edema, death, then repair",
            "GSE2565 0.5-1 h may constrain direction only; no 4-72 h expression value or target trajectory is provided",
        ],
        "local_inputs": ["toxicant_proxy", "oxidative_stress", "epithelial_injury", "inflammation", "edema_proxy", "death_signal", "repair_signal"],
        "required_output": {k: "number in [0,1]" for k in OUTPUT_KEYS},
        "semantic_note": "field outputs are capacities gated by each worker cell local BioFVM values; multiple workers share this Agent program",
        "forbidden": ["late real expression", "8 h peak targeting", "global tissue coordination", "invented physical phosgene concentration"],
    }


def validate(obj: object) -> dict[str, float]:
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    out = {}
    for key in OUTPUT_KEYS:
        if key not in obj or isinstance(obj[key], bool):
            raise ValueError(f"missing/non-numeric {key}")
        value = float(obj[key])
        if not 0 <= value <= 1:
            raise ValueError(f"{key} outside [0,1]")
        out[key] = value
    return out


def one_call(agent: str, condition: str, hour: float) -> dict:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; no fallback is allowed")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    prompt = prompt_for(agent, condition, hour)
    prompt_text = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    prompt_hash = sha(prompt_text)
    call_dir = POLICY / "llm_calls" / condition / f"hour_{hour:g}" / agent
    call_dir.mkdir(parents=True, exist_ok=True)
    validated_path = call_dir / "validated.json"
    if validated_path.exists():
        prior = json.loads(validated_path.read_text())
        if prior.get("prompt_hash") == prompt_hash:
            return {"agent_id": agent, "condition": condition, "checkpoint_hour": hour, "prompt_hash": prompt_hash, **prior["policy"]}
    (call_dir / "prompt.json").write_text(json.dumps(prompt, indent=2, ensure_ascii=False) + "\n")
    body = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only strict JSON matching every requested numeric field. You are a biological cell-type/state prototype Agent, never a tissue coordinator."},
            {"role": "user", "content": prompt_text},
        ],
    }
    req = request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    last = None
    for attempt in range(3):
        try:
            with request.urlopen(req, timeout=120) as response:
                raw_text = response.read().decode()
            (call_dir / "raw_response.json").write_text(raw_text + "\n")
            raw = json.loads(raw_text)
            content = raw["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            policy = validate(parsed)
            (call_dir / "parsed_response.json").write_text(json.dumps(parsed, indent=2) + "\n")
            audit = {"prompt_hash": prompt_hash, "schema_valid": True, "model": model, "policy": policy, "rule_fallback_used": False}
            validated_path.write_text(json.dumps(audit, indent=2) + "\n")
            return {"agent_id": agent, "condition": condition, "checkpoint_hour": hour, "prompt_hash": prompt_hash, **policy}
        except Exception as exc:
            last = exc
            (call_dir / f"attempt_{attempt+1}_error.txt").write_text(type(exc).__name__ + ": " + str(exc)[:2000] + "\n")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DeepSeek failed/schema invalid for {agent}/{condition}/{hour}: {last}; no fallback used")


def main() -> None:
    config = yaml.safe_load((HERE / "experiment.yaml").read_text())
    POLICY.mkdir(parents=True, exist_ok=True)
    rule_rows = []
    tasks = []
    for condition in config["conditions"]:
        for hour in config["time_mapping"]["policy_checkpoint_hours"]:
            for agent in config["agent_types"]:
                rule_rows.append({"agent_id": agent, "condition": condition, "checkpoint_hour": hour, "prompt_hash": "not_applicable_rule_control", **rule_values(agent, condition, hour)})
                tasks.append((agent, condition, float(hour)))
    pd.DataFrame(rule_rows).to_csv(POLICY / "traditional_rule_policy_schedule.csv", index=False)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one_call, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    agent = pd.DataFrame(rows).sort_values(["condition", "checkpoint_hour", "agent_id"])
    agent.to_csv(POLICY / "agent_policy_schedule.csv", index=False)
    audit = {
        "expected_calls": len(tasks), "validated_calls": len(agent), "all_schema_valid": len(agent) == len(tasks),
        "unique_prompt_hashes": int(agent.prompt_hash.nunique()), "rule_fallback_used": False,
        "late_real_expression_in_prompt": False, "global_tissue_coordinator": False,
    }
    (OUT / "audit/agent_policy_generation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
