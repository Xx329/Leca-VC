"""Strict schema, type, and numeric-range validation for agent programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class NumericField:
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ProgramSchema:
    numeric_fields: Mapping[str, NumericField]
    string_fields: tuple[str, ...] = ("dominant_program",)
    allow_extra: bool = False


def _flatten(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            result.update(_flatten(name, item))
        else:
            result[name] = item
    return result


def validate_program(program: Mapping[str, Any], schema: ProgramSchema) -> dict[str, Any]:
    """Validate without coercion; accepted values are returned unchanged."""

    flat = _flatten("", program)
    expected = set(schema.string_fields) | set(schema.numeric_fields)
    missing = expected - set(flat)
    extras = set(flat) - expected
    if missing:
        raise SchemaValidationError(f"Missing fields: {sorted(missing)}")
    if extras and not schema.allow_extra:
        raise SchemaValidationError(f"Unexpected fields: {sorted(extras)}")
    for name in schema.string_fields:
        value = flat[name]
        if not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(f"{name} must be a non-empty string")
    for name, bounds in schema.numeric_fields.items():
        value = flat[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaValidationError(f"{name} must be numeric")
        if not bounds.minimum <= float(value) <= bounds.maximum:
            raise SchemaValidationError(
                f"{name}={value} outside [{bounds.minimum}, {bounds.maximum}]"
            )
    return dict(program)

