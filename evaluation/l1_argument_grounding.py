"""Versioned, Oracle-independent grounding for model-proposed L1 arguments."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from network_runtime.contracts import sha256_json


GROUNDING_POLICY_SCHEMA = "netopyu.io/l1-argument-grounding-policy/v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains(prompt: str, evidence: str) -> bool:
    needle = _normalized(evidence)
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9]+", needle):
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", prompt) is not None
    return needle in prompt


@dataclass(frozen=True)
class GroundingResult:
    arguments: dict[str, Any]
    dropped_fields: tuple[str, ...]
    normalized_fields: tuple[str, ...]


class L1ArgumentGroundingPolicy:
    """Accept only request-grounded values under reviewed normalization rules."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("apiVersion") != GROUNDING_POLICY_SCHEMA:
            raise ValueError("C3 argument grounding policy schema is invalid")
        if set(payload) != {
            "apiVersion", "policyId", "maxPromptCharacters", "genericIdentifierValues",
            "reasonEvidencePatterns", "casefoldFields", "aliases",
        }:
            raise ValueError("C3 argument grounding policy fields are not exact")
        if not isinstance(payload["policyId"], str) or not payload["policyId"].strip():
            raise ValueError("C3 argument grounding policy id is invalid")
        self.max_prompt_characters = int(payload["maxPromptCharacters"])
        if not 1 <= self.max_prompt_characters <= 10_000:
            raise ValueError("C3 argument grounding prompt bound is invalid")
        casefold_fields = payload["casefoldFields"]
        if not isinstance(casefold_fields, list) or not all(
            isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in casefold_fields
        ):
            raise ValueError("C3 casefold field contract is invalid")
        self.casefold_fields = frozenset(casefold_fields)
        generic = payload["genericIdentifierValues"]
        if not isinstance(generic, dict):
            raise ValueError("C3 generic identifier contract is invalid")
        self.generic_identifier_values: dict[str, frozenset[str]] = {}
        for field, values in generic.items():
            if (
                not isinstance(field, str) or not _IDENTIFIER.fullmatch(field)
                or not isinstance(values, list)
                or not all(isinstance(item, str) and item.strip() for item in values)
            ):
                raise ValueError("C3 generic identifier values are invalid")
            self.generic_identifier_values[field] = frozenset(_normalized(item) for item in values)
        patterns = payload["reasonEvidencePatterns"]
        if not isinstance(patterns, list) or not patterns or not all(
            isinstance(item, str) and item for item in patterns
        ):
            raise ValueError("C3 reason evidence patterns are invalid")
        self.reason_evidence_patterns = tuple(re.compile(item, re.IGNORECASE) for item in patterns)
        aliases = payload["aliases"]
        if not isinstance(aliases, dict):
            raise ValueError("C3 alias contract is invalid")
        normalized_aliases: dict[str, dict[str, tuple[str, ...]]] = {}
        for field, values in aliases.items():
            if not isinstance(field, str) or not _IDENTIFIER.fullmatch(field) or not isinstance(values, dict):
                raise ValueError("C3 alias field is invalid")
            normalized_aliases[field] = {}
            for canonical, evidence in values.items():
                if not isinstance(canonical, str) or not isinstance(evidence, list) or not evidence:
                    raise ValueError("C3 alias value is invalid")
                if not all(isinstance(item, str) and item.strip() for item in evidence):
                    raise ValueError("C3 alias evidence is invalid")
                normalized_aliases[field][_normalized(canonical)] = tuple(evidence)
        self.aliases = normalized_aliases
        self.digest = sha256_json(payload)

    def _ground_scalar(
        self,
        prompt: str,
        field: str,
        value: Any,
    ) -> tuple[bool, Any, bool]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return False, value, False
            normalized_value = _normalized(stripped)
            if normalized_value in self.generic_identifier_values.get(field, frozenset()):
                return False, value, False
            if field == "reason" and not any(
                pattern.search(prompt) for pattern in self.reason_evidence_patterns
            ):
                return False, value, False
            alias_values = self.aliases.get(field, {}).get(normalized_value, ())
            grounded = _contains(prompt, stripped) or any(
                _contains(prompt, evidence) for evidence in alias_values
            )
            if not grounded:
                return False, value, False
            result = normalized_value if field in self.casefold_fields else stripped
            alias_canonical = next((
                canonical for canonical, evidence in self.aliases.get(field, {}).items()
                if normalized_value == canonical or any(
                    _normalized(stripped) == _normalized(item) for item in evidence
                )
            ), None)
            if alias_canonical is not None and any(
                _contains(prompt, item) for item in self.aliases[field][alias_canonical]
            ):
                result = alias_canonical
            return True, result, result != value
        if value is None:
            return False, value, False
        if isinstance(value, bool):
            tokens = ("true", "yes", "是", "启用") if value else ("false", "no", "否", "禁用")
            return any(_contains(prompt, item) for item in tokens), value, False
        if isinstance(value, (int, float)):
            return _contains(prompt, str(value)), value, False
        return False, value, False

    def _ground_value(
        self,
        prompt: str,
        field: str,
        value: Any,
    ) -> tuple[bool, Any, bool]:
        if isinstance(value, list):
            normalized: list[Any] = []
            changed = False
            for item in value:
                valid, current, item_changed = self._ground_value(prompt, field, item)
                if not valid:
                    return False, value, False
                normalized.append(current)
                changed = changed or item_changed
            return True, normalized, changed
        if isinstance(value, dict):
            normalized_object: dict[str, Any] = {}
            changed = False
            for key, item in value.items():
                valid, current, item_changed = self._ground_value(prompt, field, item)
                if not valid:
                    return False, value, False
                normalized_object[key] = current
                changed = changed or item_changed
            return True, normalized_object, changed
        return self._ground_scalar(prompt, field, value)

    def apply(
        self,
        prompt: str,
        supplied: dict[str, Any],
        allowed_fields: set[str],
    ) -> GroundingResult:
        if len(prompt) > self.max_prompt_characters:
            raise ValueError("C3 prompt exceeds the grounding policy bound")
        if not set(supplied) <= allowed_fields:
            raise ValueError("C3 supplied arguments escape the candidate contract")
        normalized_prompt = _normalized(prompt)
        accepted: dict[str, Any] = {}
        dropped: list[str] = []
        changed: list[str] = []
        for field, value in supplied.items():
            valid, current, was_changed = self._ground_value(normalized_prompt, field, value)
            if not valid:
                dropped.append(field)
                continue
            accepted[field] = current
            if was_changed:
                changed.append(field)
        return GroundingResult(
            arguments=accepted,
            dropped_fields=tuple(sorted(dropped)),
            normalized_fields=tuple(sorted(changed)),
        )


__all__ = [
    "GROUNDING_POLICY_SCHEMA",
    "GroundingResult",
    "L1ArgumentGroundingPolicy",
]
