#!/usr/bin/env python3
"""De-identified model-facing bridge for the frozen GSE120575 V2.1 executor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from urllib import request


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "scripts/gse120575_full_online_celltype_v21/online_inference_v21.py"
CONTROLLERS_PATH = ROOT / "scripts/gse120575_full_online_celltype/controllers.py"

import sys

sys.path.insert(0, str(ROOT / "scripts/leca_vc_prompt_isolation_v1"))
sys.path.insert(0, str(ROOT / "scripts/gse120575_full_online_celltype"))

from prompt_firewall import (  # noqa: E402
    PROTOCOL_ID,
    assert_clean,
    sanitize_gse120575_payload,
    validate_response_text,
    write_message_audit,
)
from controllers import template, validate_action  # noqa: E402


def load_base():
    spec = importlib.util.spec_from_file_location("gse120575_v21_deid_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def call_llm_deidentified(
    prompt_payload: dict[str, object],
    cell_type: str,
    raw_dir: Path,
    cache_path: Path,
    cache_identity: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; no fallback")
    assert_clean(prompt_payload, label="GSE120575 production payload")
    model = os.getenv("LECAVC_LLM_MODEL", "deepseek-chat")
    endpoint = os.getenv("LECAVC_CHAT_COMPLETIONS_URL", "https://api.deepseek.com/chat/completions")
    temperature = 0.1
    system = (
        "You are the decision layer for one biological cell-type Agent in a generic "
        "multicellular perturbation simulation. Use only the supplied current and recurrent "
        "state. Return one strict JSON object matching the template exactly. Program strengths "
        "must be in [0,1] and phenotype multipliers in [0.5,1.5]. Do not propose cross-lineage "
        "transitions or include hidden reasoning. Template: "
        + json.dumps(template(cell_type), ensure_ascii=False, sort_keys=True)
    )
    prompt = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    prompt_hash = hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()
    expected_cache = {
        **cache_identity,
        "prompt_hash": prompt_hash,
        "model": model,
        "prompt_protocol": PROTOCOL_ID,
    }
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if all(metadata.get(key) == value for key, value in expected_cache.items()):
            action = validate_action(json.loads(cache_path.read_text()), cell_type)
            validate_response_text(action, dataset_role="GSE120575")
            return action, {**metadata, "fresh_call": False, "cache_reused": True}

    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "model_visible_payload.json").write_text(
        json.dumps(prompt_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    last_error: Exception | None = None
    for attempt in range(1, 5):
        content: str | None = None
        try:
            write_message_audit(
                raw_dir,
                messages,
                dataset_role="GSE120575",
                trusted_envelope=cache_identity,
            )
            body = {
                "model": model,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
            req = request.Request(
                endpoint,
                data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=120) as response:
                raw = response.read()
            (raw_dir / f"attempt_{attempt}_raw.json").write_bytes(raw + b"\n")
            content = json.loads(raw)["choices"][0]["message"]["content"]
            (raw_dir / f"attempt_{attempt}_content.json").write_text(content + "\n")
            action = validate_action(json.loads(content), cell_type)
            validate_response_text(action, dataset_role="GSE120575")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(action, ensure_ascii=False, indent=2) + "\n")
            metadata = {
                **expected_cache,
                "fresh_call": True,
                "cache_reused": False,
                "fallback_used": False,
                "attempt": attempt,
                "timestamp": time.time(),
                "temperature": temperature,
                "response_hash": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "exact_messages_sha256": hashlib.sha256(canonical(messages).encode()).hexdigest(),
            }
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            return action, metadata
        except Exception as exc:
            last_error = exc
            (raw_dir / f"attempt_{attempt}_error.txt").write_text(
                type(exc).__name__ + ": " + str(exc) + "\n"
            )
            if content is not None:
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "The response failed schema, range, or de-identification validation. "
                                "Return one complete corrected JSON object using only the supplied state."
                            ),
                        },
                    ]
                )
    raise RuntimeError(
        f"LLM validation failed after four attempts for {cell_type}: {last_error}; no fallback"
    )


def main() -> int:
    base = load_base()
    original_atomic = base.atomic_json

    def sanitized_atomic(path: Path, value: object) -> None:
        path = Path(path)
        if "prompts" in path.parts and isinstance(value, dict):
            sanitized = sanitize_gse120575_payload(value)
            value.clear()
            value.update(sanitized)
        original_atomic(path, value)

    base.atomic_json = sanitized_atomic
    base.call_llm = call_llm_deidentified
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())

