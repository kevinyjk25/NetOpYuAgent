"""Reviewed deterministic policies for production L1 proposals."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from network_runtime.contracts import sha256_json


GUARD_POLICY_SCHEMA = "netopyu.io/l1-guard-policy/v1"
GROUNDING_POLICY_SCHEMA = "netopyu.io/l1-argument-grounding-policy/v1"
GuardAction = Literal["allow", "refuse", "out_of_scope"]
_HEX_CODEPOINT = re.compile(r"[0-9a-f]{4,6}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains(prompt: str, evidence: str) -> bool:
    needle = _normalized(evidence)
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9]+", needle):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", prompt,
        ) is not None
    return needle in prompt


@dataclass(frozen=True)
class GuardVerdict:
    action: GuardAction
    reason_code: str
    matched_rule: str | None = None
    evidence: tuple[str, ...] = ()


class GuardPolicy:
    """May close unsafe/unrelated requests; can never select a capability."""

    def __init__(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("apiVersion") != GUARD_POLICY_SCHEMA:
            raise ValueError("production L1 guard policy Schema is invalid")
        if set(raw) != {
            "apiVersion", "policyId", "normalization", "selectionConfidenceFloor",
            "safetyRules", "domainEvidence",
        }:
            raise ValueError("production L1 guard policy fields are not exact")
        normalization = raw["normalization"]
        if not isinstance(normalization, dict) or set(normalization) != {
            "form", "removeCodepoints", "maxCharacters",
        } or normalization.get("form") != "NFKC":
            raise ValueError("production L1 guard normalization is invalid")
        maximum = normalization["maxCharacters"]
        remove = normalization["removeCodepoints"]
        if not isinstance(maximum, int) or not 1 <= maximum <= 16_000:
            raise ValueError("production L1 guard request bound is invalid")
        if not isinstance(remove, list) or any(
            not isinstance(item, str) or not _HEX_CODEPOINT.fullmatch(item)
            for item in remove
        ):
            raise ValueError("production L1 guard codepoint policy is invalid")
        floor = raw["selectionConfidenceFloor"]
        if not isinstance(floor, (int, float)) or not 0 <= float(floor) <= 1:
            raise ValueError("production L1 confidence floor is invalid")
        rules: list[tuple[str, tuple[re.Pattern[str], ...]]] = []
        seen: set[str] = set()
        for raw_rule in raw["safetyRules"]:
            if not isinstance(raw_rule, dict) or set(raw_rule) != {"id", "patterns"}:
                raise ValueError("production L1 safety rule shape is invalid")
            rule_id = raw_rule["id"]
            patterns = raw_rule["patterns"]
            if (
                not isinstance(rule_id, str)
                or not re.fullmatch(r"[a-z0-9_]{1,64}", rule_id)
                or rule_id in seen
                or not isinstance(patterns, list)
                or not patterns
                or any(not isinstance(item, str) or len(item) > 512 for item in patterns)
            ):
                raise ValueError("production L1 safety rule is invalid")
            seen.add(rule_id)
            rules.append((rule_id, tuple(re.compile(item, re.I) for item in patterns)))
        domain = raw["domainEvidence"]
        if not isinstance(domain, dict) or set(domain) != {
            "anchors", "entityPatterns", "excludedDomains",
        }:
            raise ValueError("production L1 domain evidence policy is invalid")
        anchors = domain["anchors"]
        entity_patterns = domain["entityPatterns"]
        excluded = domain["excludedDomains"]
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("production L1 domain anchors are absent")
        self.policy_id = str(raw["policyId"])
        self.max_characters = maximum
        self.selection_confidence_floor = float(floor)
        self.remove_codepoints = frozenset(chr(int(item, 16)) for item in remove)
        self._rules = tuple(rules)
        self._anchors = tuple(str(item).casefold() for item in anchors)
        self._entity_patterns = tuple(re.compile(str(item), re.I) for item in entity_patterns)
        self._excluded_patterns = tuple(re.compile(str(item), re.I) for item in excluded)
        self.digest = sha256_json(raw)

    def normalize(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("production L1 request must be non-empty text")
        if len(prompt) > self.max_characters:
            raise ValueError("production L1 request exceeds policy limit")
        normalized = unicodedata.normalize("NFKC", prompt)
        return "".join(
            character for character in normalized
            if character not in self.remove_codepoints
        ).casefold()

    def classify(self, prompt: str) -> GuardVerdict:
        normalized = self.normalize(prompt)
        for rule_id, patterns in self._rules:
            if any(pattern.search(normalized) for pattern in patterns):
                return GuardVerdict("refuse", f"guard_{rule_id}", matched_rule=rule_id)
        domain_evidence = tuple(dict.fromkeys(
            anchor for anchor in self._anchors if anchor in normalized
        ))
        entity_evidence = tuple(
            match.group(0)
            for pattern in self._entity_patterns
            if (match := pattern.search(normalized)) is not None
        )
        excluded = tuple(
            pattern.pattern
            for pattern in self._excluded_patterns
            if pattern.search(normalized)
        )
        if excluded and not domain_evidence and not entity_evidence:
            return GuardVerdict(
                "out_of_scope", "guard_excluded_domain", evidence=excluded,
            )
        if not domain_evidence and not entity_evidence:
            return GuardVerdict("out_of_scope", "guard_no_domain_evidence")
        return GuardVerdict(
            "allow", "guard_domain_evidence",
            evidence=(*domain_evidence[:8], *entity_evidence[:4]),
        )


@dataclass(frozen=True)
class GroundingResult:
    arguments: dict[str, Any]
    dropped_fields: tuple[str, ...]
    normalized_fields: tuple[str, ...]


class GroundingPolicy:
    """Accept only values that have explicit evidence in the direct user text."""

    def __init__(self, path: Path) -> None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("apiVersion") != GROUNDING_POLICY_SCHEMA:
            raise ValueError("production L1 grounding policy Schema is invalid")
        if set(payload) != {
            "apiVersion", "policyId", "maxPromptCharacters", "genericIdentifierValues",
            "reasonEvidencePatterns", "casefoldFields", "aliases",
        }:
            raise ValueError("production L1 grounding policy fields are not exact")
        self.max_prompt_characters = int(payload["maxPromptCharacters"])
        if not 1 <= self.max_prompt_characters <= 10_000:
            raise ValueError("production L1 grounding request bound is invalid")
        fields = payload["casefoldFields"]
        if not isinstance(fields, list) or not all(
            isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in fields
        ):
            raise ValueError("production L1 casefold fields are invalid")
        self.casefold_fields = frozenset(fields)
        generic = payload["genericIdentifierValues"]
        if not isinstance(generic, dict):
            raise ValueError("production L1 generic identifier policy is invalid")
        self.generic_identifier_values = {
            str(field): frozenset(_normalized(str(item)) for item in values)
            for field, values in generic.items()
        }
        patterns = payload["reasonEvidencePatterns"]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError("production L1 reason evidence patterns are absent")
        self.reason_evidence_patterns = tuple(re.compile(str(item), re.I) for item in patterns)
        raw_aliases = payload["aliases"]
        if not isinstance(raw_aliases, dict):
            raise ValueError("production L1 grounding aliases are invalid")
        self.aliases: dict[str, dict[str, tuple[str, ...]]] = {}
        for field, canonical_values in raw_aliases.items():
            if not isinstance(canonical_values, dict):
                raise ValueError("production L1 alias values are invalid")
            self.aliases[str(field)] = {
                _normalized(str(canonical)): tuple(str(item) for item in evidence)
                for canonical, evidence in canonical_values.items()
            }
        self.digest = sha256_json(payload)

    def _ground_scalar(
        self, prompt: str, field: str, value: Any,
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
            if not (_contains(prompt, stripped) or any(
                _contains(prompt, evidence) for evidence in alias_values
            )):
                return False, value, False
            result: Any = normalized_value if field in self.casefold_fields else stripped
            alias_canonical = next((
                canonical
                for canonical, evidence in self.aliases.get(field, {}).items()
                if normalized_value == canonical or any(
                    _normalized(stripped) == _normalized(item) for item in evidence
                )
            ), None)
            if alias_canonical is not None and any(
                _contains(prompt, item)
                for item in self.aliases[field][alias_canonical]
            ):
                result = alias_canonical
            return True, result, result != value
        if value is None:
            return False, value, False
        if isinstance(value, bool):
            tokens = ("true", "yes", "是", "启用") if value else (
                "false", "no", "否", "禁用",
            )
            return any(_contains(prompt, item) for item in tokens), value, False
        if isinstance(value, (int, float)):
            return _contains(prompt, str(value)), value, False
        return False, value, False

    def _ground_value(
        self, prompt: str, field: str, value: Any,
    ) -> tuple[bool, Any, bool]:
        if isinstance(value, list):
            output: list[Any] = []
            changed = False
            for item in value:
                valid, normalized, item_changed = self._ground_value(prompt, field, item)
                if not valid:
                    return False, value, False
                output.append(normalized)
                changed = changed or item_changed
            return True, output, changed
        if isinstance(value, dict):
            output_object: dict[str, Any] = {}
            changed = False
            for key, item in value.items():
                valid, normalized, item_changed = self._ground_value(prompt, field, item)
                if not valid:
                    return False, value, False
                output_object[str(key)] = normalized
                changed = changed or item_changed
            return True, output_object, changed
        return self._ground_scalar(prompt, field, value)

    def apply(
        self, prompt: str, supplied: dict[str, Any], allowed_fields: set[str],
    ) -> GroundingResult:
        if len(prompt) > self.max_prompt_characters:
            raise ValueError("production L1 request exceeds grounding bound")
        if not set(supplied) <= allowed_fields:
            raise ValueError("production L1 arguments escape candidate Schema")
        normalized_prompt = _normalized(prompt)
        accepted: dict[str, Any] = {}
        dropped: list[str] = []
        changed: list[str] = []
        for field, value in supplied.items():
            valid, normalized, was_changed = self._ground_value(
                normalized_prompt, field, value,
            )
            if not valid:
                dropped.append(field)
                continue
            accepted[field] = normalized
            if was_changed:
                changed.append(field)
        return GroundingResult(
            arguments=accepted,
            dropped_fields=tuple(sorted(dropped)),
            normalized_fields=tuple(sorted(changed)),
        )


__all__ = [
    "GroundingPolicy",
    "GroundingResult",
    "GuardPolicy",
    "GuardVerdict",
]
