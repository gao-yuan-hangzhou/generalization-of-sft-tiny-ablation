from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .data import normalize_json_text


_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_LEADING_ASSISTANT_RE = re.compile(r"^(?:assistant|Assistant)\s*[:\n\r]+\s*")


@dataclass(frozen=True)
class FormatCheckResult:
    ok: bool
    reason: str | None = None


def strip_leading_assistant_marker(text: str) -> str:
    """
    Some chat models occasionally emit a redundant role header like `assistant\\n`
    at the start of the completion. This helper removes *only* that initial marker.
    """

    stripped = text.lstrip()
    stripped2 = _LEADING_ASSISTANT_RE.sub("", stripped, count=1)
    return stripped2 if stripped2 != stripped else text


def extract_first_json_object(text: str) -> str | None:
    """
    Return the first balanced JSON object substring (from the first '{' to its matching '}'),
    or None if no balanced object is found.
    """
    s = text
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def extract_first_json_object_span(text: str) -> tuple[int, int] | None:
    """
    Return (start, end) for the first balanced JSON object substring, or None.
    """
    s = text
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def count_json_objects(text: str, *, max_objects: int = 3) -> int:
    """
    Count balanced JSON object substrings in `text` (up to `max_objects`).
    """
    count = 0
    offset = 0
    while count < max_objects:
        span = extract_first_json_object_span(text[offset:])
        if span is None:
            break
        start, end = span
        offset += end
        count += 1
    return count


def _strict_json_loads(text: str) -> Any:
    text = text.strip()
    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(text)
    rest = text[end:].strip()
    if rest:
        raise ValueError("extra_text_after_json")
    return obj


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "date":
        return isinstance(value, str) and bool(_DATE_RE.match(value))
    return True


def check_strict_json_schema(output_text: str, schema: dict[str, str]) -> FormatCheckResult:
    text = normalize_json_text(output_text)
    if not text.startswith("{") or not text.endswith("}"):
        return FormatCheckResult(False, "not_braced_json_object")

    try:
        obj = _strict_json_loads(text)
    except Exception as exc:  # noqa: BLE001
        return FormatCheckResult(False, f"json_parse_error:{exc}")

    if not isinstance(obj, dict):
        return FormatCheckResult(False, "not_json_object")

    required_keys = set(schema.keys())
    found_keys = set(obj.keys())

    if found_keys != required_keys:
        missing = sorted(required_keys - found_keys)
        extra = sorted(found_keys - required_keys)
        return FormatCheckResult(False, f"keys_mismatch:missing={missing},extra={extra}")

    for key, type_name in schema.items():
        if not _type_ok(obj.get(key), type_name):
            return FormatCheckResult(False, f"type_mismatch:{key}:{type_name}")

    return FormatCheckResult(True, None)


def check_extractable_json_schema(output_text: str, schema: dict[str, str]) -> FormatCheckResult:
    extracted = extract_first_json_object(output_text)
    if extracted is None:
        return FormatCheckResult(False, "no_json_object_found")
    return check_strict_json_schema(extracted, schema)
