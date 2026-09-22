"""Minimal DeepSeek client with mandatory firewall, validation, and no fallback."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import append_jsonl, stable_hash
from .prompt_firewall import enforce_messages


class MissingAPIKeyError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


def require_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError(
            "DEEPSEEK_API_KEY is not set; online execution stopped. No fallback is permitted."
        )
    return key


def chat_completion(
    messages: Sequence[Mapping[str, Any]],
    *,
    audit_path: Path,
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    max_attempts: int = 3,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    enforce_messages(messages)
    key = require_api_key()
    payload = {"model": model, "messages": list(messages), "temperature": temperature, "response_format": {"type": "json_object"}}
    prompt_hash = stable_hash(payload["messages"])
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            append_jsonl(audit_path, {"prompt_hash": prompt_hash, "model": model, "temperature": temperature, "attempt": attempt, "status": "accepted_http_response", "messages": list(messages), "response": result})
            return result
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            append_jsonl(audit_path, {"prompt_hash": prompt_hash, "model": model, "temperature": temperature, "attempt": attempt, "status": "retryable_error", "error_type": type(error).__name__})
            if attempt == max_attempts:
                raise LLMRequestError(f"DeepSeek request failed after {max_attempts} attempts; no fallback used") from error
            time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")

