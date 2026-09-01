"""Fail-closed binding between an authoritative request and a Tool call.

The Runtime can validate only the arguments it receives. A Harness/LLM may
silently drop, add, or change fields before invoking a Tool, so schema
validation at the Tool boundary alone is insufficient. This module compares
the confirmed structured request with the actual invocation without retaining
raw values in the resulting evidence.

Raw natural-language text is not an authoritative argument source. Callers
must first obtain a structured request from an API, form, or explicit user /
approval confirmation before using this binding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .contracts import sha256_json


ARGUMENT_BINDING_SCHEMA = "netopyu.io/exact-argument-binding/v1"


@dataclass(frozen=True)
class ExactArgumentBinding:
    """Privacy-minimized result of an identity argument mapping check."""

    ok: bool
    code: str
    source_digest: str
    invocation_digest: str
    source_keys: tuple[str, ...]
    invocation_keys: tuple[str, ...]
    unknown_source_fields: tuple[str, ...]
    unknown_invocation_fields: tuple[str, ...]
    missing_source_fields: tuple[str, ...]
    dropped_fields: tuple[str, ...]
    injected_fields: tuple[str, ...]
    changed_fields: tuple[str, ...]
    schema: str = ARGUMENT_BINDING_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name, item in tuple(value.items()):
            if isinstance(item, tuple):
                value[name] = list(item)
        return value


def validate_exact_argument_binding(
    *,
    authoritative: Mapping[str, Any],
    invocation: Mapping[str, Any],
    declared_fields: Iterable[str],
    required_fields: Iterable[str],
) -> ExactArgumentBinding:
    """Prove an exact identity mapping or return field-level rejection evidence.

    Defaults, aliases, coercions, and derived values require a separately
    reviewed mapping contract; an LLM must not silently introduce them.
    """

    source = dict(authoritative)
    called = dict(invocation)
    declared = set(declared_fields)
    required = set(required_fields)
    source_keys = set(source)
    invocation_keys = set(called)
    unknown_source = sorted(source_keys - declared)
    unknown_invocation = sorted(invocation_keys - declared)
    missing_source = sorted(
        name for name in required
        if name not in source or source[name] in (None, "")
    )
    dropped = sorted((source_keys & declared) - invocation_keys)
    injected = sorted(invocation_keys - source_keys)
    changed = sorted(
        name for name in source_keys & invocation_keys & declared
        if source[name] != called[name]
    )
    failures = (
        unknown_source, unknown_invocation, missing_source,
        dropped, injected, changed,
    )
    ok = not any(failures)
    return ExactArgumentBinding(
        ok=ok,
        code="argument_binding_verified" if ok else "argument_binding_rejected",
        source_digest=sha256_json(source),
        invocation_digest=sha256_json(called),
        source_keys=tuple(sorted(source_keys)),
        invocation_keys=tuple(sorted(invocation_keys)),
        unknown_source_fields=tuple(unknown_source),
        unknown_invocation_fields=tuple(unknown_invocation),
        missing_source_fields=tuple(missing_source),
        dropped_fields=tuple(dropped),
        injected_fields=tuple(injected),
        changed_fields=tuple(changed),
    )


__all__ = [
    "ARGUMENT_BINDING_SCHEMA",
    "ExactArgumentBinding",
    "validate_exact_argument_binding",
]
