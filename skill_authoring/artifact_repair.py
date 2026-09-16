"""One host-addressed code-body edit, never free-text regeneration or execution."""
import re

from .artifact_checks import draft_spans
from .compiler import obj
from network_runtime.l0.structured_schema import validate_data


def regions(text, checks):
    failures = {r["location"] for r in checks["inspection"]["checks"] if r["status"] == "fail"}
    result = {}
    for index, span in enumerate(draft_spans(text)):
        key = f"draft-{index}"
        lines = span.splitlines(keepends=True)
        if key not in failures or len(lines) < 3:
            continue
        opening = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\s]*)\s*", lines[0].rstrip("\r\n"))
        if not opening:
            continue
        marker, language = opening.groups()
        if not re.fullmatch(rf" {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*", lines[-1].rstrip("\r\n")):
            continue
        result[key] = {"language": language, "code": "".join(lines[1:-1]),
                       "checks": [r for r in checks["inspection"]["checks"] if r["location"] == key]}
    return result


def response_schema(slots):
    return obj({"replacements": {"type": "array", "minItems": 1, "maxItems": len(slots),
        "items": obj({"location": {"type": "string", "enum": list(slots)}, "code": {"type": "string", "minLength": 1, "maxLength": 20000}})}})


def apply(text, checks, response):
    slots = regions(text, checks)
    validate_data(response_schema(slots), response)
    edits = {}
    for item in response["replacements"]:
        key, code = item["location"], item["code"]
        if key in edits or not code.strip() or re.search(r"(?m)^ {0,3}(?:`{3,}|~{3,})", code):
            raise ValueError("duplicate, blank or nested-fence repair")
        edits[key] = code.rstrip("\r\n") + "\n"
    parts = draft_spans(text)
    for index, span in enumerate(parts):
        if f"draft-{index}" in edits:
            lines = span.splitlines(keepends=True)
            parts[index] = lines[0] + edits[f"draft-{index}"] + lines[-1]
    # No check is bypassed. The caller re-inspects the complete rendered answer.
    return {"answer": "".join(parts)}
