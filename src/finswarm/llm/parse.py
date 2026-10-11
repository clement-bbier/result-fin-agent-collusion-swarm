"""Parse LLM output into validated pydantic objects, with one repair round-trip.

Pipeline: extract (fenced JSON, whole text, or the last decodable top-level object; or a
section parser such as Fish's text template) -> validate against the output schema ->
on failure, one repair call that quotes the bad answer and the validation error -> on
failure again, the caller's default (``parse_ok=False``).

Unknown keys are dropped before validation (the schemas forbid extras, but a stray
``"reasoning"`` key is not worth a repair call).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)```", re.S)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

REPAIR_TEMPLATE = (
    "Your previous reply could not be used.\n"
    "Previous reply:\n<<<\n{raw}\n>>>\n"
    "Problem: {error}\n"
    "Reply again following exactly the required output format{schema_hint}. "
    "Output nothing else."
)

SectionsParser = Callable[[str], dict[str, Any] | None]
RepairFn = Callable[[str, str], Awaitable[str]]


@dataclass
class ParseResult[T: BaseModel]:
    value: T | None
    parse_ok: bool
    repair_used: bool
    raw: str  # text of the last answer that was parsed
    error: str | None
    attempts: int  # answers parsed (1 + repairs)


def _loads_dict(s: str) -> dict[str, Any] | None:
    for candidate in (s, _TRAILING_COMMA.sub(r"\1", s.translate(_SMART_QUOTES))):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the JSON object in ``text`` (fenced block, whole text, or last object)."""
    if not text:
        return None
    fenced = [d for m in _FENCE.finditer(text) if (d := _loads_dict(m.group(1).strip()))]
    if fenced:
        return fenced[-1]
    s = text.strip()
    whole = _loads_dict(s)
    if whole is not None:
        return whole
    raw_objs = _scan_objects(s)
    cleaned = _TRAILING_COMMA.sub(r"\1", s.translate(_SMART_QUOTES))
    clean_objs = _scan_objects(cleaned) if cleaned != s else raw_objs
    # The cleaned scan recovers objects with trailing commas / smart quotes; prefer it
    # only when it finds more top-level objects than the raw scan.
    objs = clean_objs if len(clean_objs) > len(raw_objs) else raw_objs
    return objs[-1] if objs else None


def _scan_objects(src: str) -> list[dict[str, Any]]:
    """Top-level JSON objects embedded in ``src``, in order of appearance."""
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    i = src.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(src, i)
        except json.JSONDecodeError:
            i = src.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = src.find("{", end)
    return found


def format_validation_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {e.get('msg', 'invalid')}")
    return "; ".join(parts)


def validate[T: BaseModel](data: dict[str, Any], model_cls: type[T]) -> tuple[T | None, str | None]:
    known = {k: v for k, v in data.items() if k in model_cls.model_fields}
    try:
        return model_cls.model_validate(known), None
    except ValidationError as e:
        return None, format_validation_error(e)


def parse_text[T: BaseModel](
    text: str, model_cls: type[T], sections_parser: SectionsParser | None = None
) -> tuple[T | None, str | None]:
    data = sections_parser(text) if sections_parser is not None else extract_json(text)
    if data is None and sections_parser is not None:
        data = extract_json(text)  # a model may answer in JSON even when asked for text
    if data is None:
        return None, "no JSON object found" if sections_parser is None else "template not found"
    return validate(data, model_cls)


def repair_message(raw: str, error: str, schema: dict[str, Any] | None = None) -> str:
    hint = ""
    if schema is not None:
        hint = " (a single JSON object matching this schema: " + json.dumps(schema) + ")"
    return REPAIR_TEMPLATE.format(raw=raw[:4000], error=error, schema_hint=hint)


async def parse_with_repair[T: BaseModel](
    text: str,
    model_cls: type[T],
    *,
    repair: RepairFn | None = None,
    max_repairs: int = 1,
    sections_parser: SectionsParser | None = None,
    default: T | Callable[[], T] | None = None,
) -> ParseResult[T]:
    """Parse ``text``; on failure call ``repair(raw, error)`` up to ``max_repairs`` times.

    When every attempt fails the result carries ``default`` (or ``None``) with
    ``parse_ok=False``.
    """
    value, err = parse_text(text, model_cls, sections_parser)
    if value is not None:
        return ParseResult(value, True, False, text, None, 1)
    raw = text
    repairs = 0
    if repair is not None:
        for _ in range(max(0, max_repairs)):
            repairs += 1
            raw = await repair(raw, err or "invalid output")
            value, err = parse_text(raw, model_cls, sections_parser)
            if value is not None:
                return ParseResult(value, True, True, raw, None, 1 + repairs)
    fallback = default() if callable(default) else default
    return ParseResult(fallback, False, repairs > 0, raw, err, 1 + repairs)
