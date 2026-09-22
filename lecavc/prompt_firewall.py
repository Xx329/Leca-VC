"""Allowlist-oriented checks for model-facing Leca-VC messages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


PROHIBITED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("geo_accession", re.compile(r"\b(?:GSE|GSM)\s*[-_]?\s*\d+\b", re.I)),
    ("response_label", re.compile(r"\b(?:non[- ]?responder|responder)\b", re.I)),
    ("endpoint_hint", re.compile(r"\b(?:held[- ]?out|endpoint|late[- ]?stage|terminal distribution)\b", re.I)),
    ("publication_context", re.compile(r"\b(?:Sade[- ]?Feldman|et\s+al\.|doi\s*:|publication|paper conclusion)\b", re.I)),
    ("treatment_identity", re.compile(r"\b(?:anti[- ]?PD[- ]?1|phosgene|bleomycin|lung injury)\b", re.I)),
    ("real_time_label", re.compile(r"\b(?:d7|d21|\d+(?:\.\d+)?\s*(?:hours?|hrs?|days?))\b", re.I)),
    ("sample_identifier", re.compile(r"\b(?:Pre|Post)[-_]?[A-Za-z]*\d+\b", re.I)),
)


@dataclass(frozen=True)
class FirewallHit:
    rule: str
    match: str
    start: int
    end: int


class PromptFirewallError(ValueError):
    """Raised before a prohibited model-facing message can be sent."""


def serialize_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    """Return a stable representation used by the scanner and audit hash."""

    return json.dumps(list(messages), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scan_text(text: str) -> list[FirewallHit]:
    hits: list[FirewallHit] = []
    for name, pattern in PROHIBITED_PATTERNS:
        hits.extend(FirewallHit(name, match.group(0), match.start(), match.end()) for match in pattern.finditer(text))
    return sorted(hits, key=lambda item: (item.start, item.rule))


def scan_messages(messages: Iterable[Mapping[str, Any]]) -> list[FirewallHit]:
    return scan_text(serialize_messages(messages))


def enforce_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    """Validate and return serialized messages, or stop before network access."""

    serialized = serialize_messages(messages)
    hits = scan_text(serialized)
    if hits:
        summary = ", ".join(f"{hit.rule}={hit.match!r}" for hit in hits)
        raise PromptFirewallError(f"Model-facing prompt rejected: {summary}")
    return serialized

