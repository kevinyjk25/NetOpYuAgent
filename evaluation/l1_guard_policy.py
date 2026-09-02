"""Deterministic P1.8-C2 request guard.

The guard is deliberately narrow: it may require refusal, classify an
unrelated request, or abstain on low-confidence selection.  It cannot select a
capability, add arguments, call a Tool, or authorize an effect.  The policy is
reviewed data and never imports evaluation scenarios or their expected labels.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from network_runtime.contracts import sha256_json

from .l1_contract import L1Action, L1Decision, L1_DECISION_SCHEMA


GUARD_POLICY_SCHEMA = "netopyu.io/l1-guard-policy/v1"
GuardAction = Literal["allow", "refuse", "out_of_scope"]
_HEX_CODEPOINT = re.compile(r"[0-9a-f]{4,6}")


@dataclass(frozen=True)
class GuardVerdict:
    action: GuardAction
    reason_code: str
    matched_rule: str | None = None
    evidence: tuple[str, ...] = ()


class L1GuardPolicy:
    """Validated, digest-bound safety and domain boundary policy."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("apiVersion") != GUARD_POLICY_SCHEMA:
            raise ValueError("invalid P1.8-C2 guard policy schema")
        expected = {
            "apiVersion", "policyId", "normalization", "selectionConfidenceFloor",
            "safetyRules", "domainEvidence",
        }
        if set(raw) != expected:
            raise ValueError("P1.8-C2 guard policy fields are not exact")
        policy_id = raw.get("policyId")
        if not isinstance(policy_id, str) or not re.fullmatch(r"[a-z0-9_.-]{1,80}", policy_id):
            raise ValueError("P1.8-C2 policy id is invalid")
        normalization = raw.get("normalization")
        if not isinstance(normalization, dict) or set(normalization) != {
            "form", "removeCodepoints", "maxCharacters",
        }:
            raise ValueError("P1.8-C2 normalization contract is invalid")
        if normalization.get("form") != "NFKC":
            raise ValueError("P1.8-C2 requires NFKC normalization")
        maximum = normalization.get("maxCharacters")
        if not isinstance(maximum, int) or not 1 <= maximum <= 16_000:
            raise ValueError("P1.8-C2 maximum request length is invalid")
        remove = normalization.get("removeCodepoints")
        if not isinstance(remove, list) or any(
            not isinstance(item, str) or not _HEX_CODEPOINT.fullmatch(item)
            for item in remove
        ):
            raise ValueError("P1.8-C2 removable codepoints are invalid")
        floor = raw.get("selectionConfidenceFloor")
        if not isinstance(floor, (int, float)) or not 0 <= float(floor) <= 1:
            raise ValueError("P1.8-C2 confidence floor is invalid")

        rules = raw.get("safetyRules")
        if not isinstance(rules, list) or not rules:
            raise ValueError("P1.8-C2 safety rules are absent")
        compiled_rules: list[tuple[str, tuple[re.Pattern[str], ...]]] = []
        seen: set[str] = set()
        for rule in rules:
            if not isinstance(rule, dict) or set(rule) != {"id", "patterns"}:
                raise ValueError("P1.8-C2 safety rule shape is invalid")
            rule_id = rule.get("id")
            patterns = rule.get("patterns")
            if (
                not isinstance(rule_id, str)
                or not re.fullmatch(r"[a-z0-9_]{1,64}", rule_id)
                or rule_id in seen
                or not isinstance(patterns, list)
                or not patterns
                or any(not isinstance(item, str) or len(item) > 512 for item in patterns)
            ):
                raise ValueError("P1.8-C2 safety rule is invalid")
            seen.add(rule_id)
            compiled_rules.append((rule_id, tuple(re.compile(item, re.I) for item in patterns)))

        domain = raw.get("domainEvidence")
        if not isinstance(domain, dict) or set(domain) != {
            "anchors", "entityPatterns", "excludedDomains",
        }:
            raise ValueError("P1.8-C2 domain evidence contract is invalid")
        anchors = domain.get("anchors")
        entity_patterns = domain.get("entityPatterns")
        excluded = domain.get("excludedDomains")
        if not isinstance(anchors, list) or not anchors or any(
            not isinstance(item, str) or not item.strip() or len(item) > 80 for item in anchors
        ):
            raise ValueError("P1.8-C2 domain anchors are invalid")
        for values, label in ((entity_patterns, "entity"), (excluded, "excluded")):
            if not isinstance(values, list) or any(
                not isinstance(item, str) or len(item) > 512 for item in values
            ):
                raise ValueError(f"P1.8-C2 {label} patterns are invalid")

        self.policy_id = policy_id
        self.max_characters = maximum
        self.remove_codepoints = frozenset(chr(int(item, 16)) for item in remove)
        self.selection_confidence_floor = float(floor)
        self._rules = tuple(compiled_rules)
        self._anchors = tuple(item.casefold() for item in anchors)
        self._entity_patterns = tuple(re.compile(item, re.I) for item in entity_patterns)
        self._excluded_patterns = tuple(re.compile(item, re.I) for item in excluded)
        self.raw = raw
        self.digest = sha256_json(raw)

    def normalize(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("P1.8-C2 request must be non-empty text")
        if len(prompt) > self.max_characters:
            raise ValueError("P1.8-C2 request exceeds the policy limit")
        normalized = unicodedata.normalize("NFKC", prompt)
        return "".join(ch for ch in normalized if ch not in self.remove_codepoints).casefold()

    def classify(self, prompt: str) -> GuardVerdict:
        normalized = self.normalize(prompt)
        for rule_id, patterns in self._rules:
            if any(pattern.search(normalized) for pattern in patterns):
                return GuardVerdict(
                    "refuse", f"guard_{rule_id}", matched_rule=rule_id,
                )
        domain_evidence = tuple(dict.fromkeys(
            anchor for anchor in self._anchors if anchor in normalized
        ))
        entity_evidence = tuple(
            match.group(0) for pattern in self._entity_patterns
            if (match := pattern.search(normalized)) is not None
        )
        excluded = tuple(
            pattern.pattern for pattern in self._excluded_patterns if pattern.search(normalized)
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

    def enforce(self, decision: L1Decision | None, verdict: GuardVerdict) -> tuple[L1Decision | None, str]:
        if verdict.action in {"refuse", "out_of_scope"}:
            action = L1Action.REFUSE if verdict.action == "refuse" else L1Action.OUT_OF_SCOPE
            return L1Decision(
                apiVersion=L1_DECISION_SCHEMA,
                action=action,
                confidence=1.0,
                reason_code=verdict.reason_code,
            ), "policy_override"
        if decision is None:
            return None, "protocol_failure"
        if (
            decision.action in {L1Action.SELECT_SKILL, L1Action.SELECT_TOOL}
            and decision.confidence < self.selection_confidence_floor
        ):
            return None, "low_confidence_abstention"
        return decision, "unchanged"


__all__ = [
    "GUARD_POLICY_SCHEMA",
    "GuardAction",
    "GuardVerdict",
    "L1GuardPolicy",
]
