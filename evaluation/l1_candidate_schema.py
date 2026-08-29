"""Versioned C3 candidate-Schema overlays without mutating frozen C1/C2 cards."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import yaml

from network_runtime.contracts import sha256_json

from .l1_catalog import L1CatalogEntry


CANDIDATE_SCHEMA_POLICY = "netopyu.io/l1-candidate-schema-policy/v1"
_IDENTITY = re.compile(r"(?:skill|tool):[A-Za-z0-9_.:-]{1,128}\Z")
_FIELD = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class L1CandidateSchemaPolicy:
    """Apply reviewed parameter contracts only to exact candidate identities."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "apiVersion", "policyId", "dominance", "overrides",
        } or payload.get("apiVersion") != CANDIDATE_SCHEMA_POLICY:
            raise ValueError("C3 candidate Schema policy contract is invalid")
        if not isinstance(payload["policyId"], str) or not payload["policyId"].strip():
            raise ValueError("C3 candidate Schema policy id is invalid")
        raw_overrides = payload["overrides"]
        if not isinstance(raw_overrides, dict):
            raise ValueError("C3 candidate Schema overrides must be an object")
        overrides: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {}
        for identity, raw in raw_overrides.items():
            if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
                raise ValueError("C3 candidate Schema identity is invalid")
            if not isinstance(raw, dict) or set(raw) != {"parameters", "requiredParameters"}:
                raise ValueError("C3 candidate Schema override fields are invalid")
            parameters = raw["parameters"]
            required = raw["requiredParameters"]
            if not isinstance(parameters, dict) or not parameters:
                raise ValueError("C3 candidate Schema parameters are invalid")
            normalized_parameters: dict[str, str] = {}
            for field, description in parameters.items():
                if (
                    not isinstance(field, str) or not _FIELD.fullmatch(field)
                    or not isinstance(description, str) or not description.strip()
                ):
                    raise ValueError("C3 candidate Schema parameter is invalid")
                normalized_parameters[field] = description
            if not isinstance(required, list) or not all(
                isinstance(field, str) and field in normalized_parameters for field in required
            ) or len(required) != len(set(required)):
                raise ValueError("C3 candidate Schema required fields are invalid")
            overrides[identity] = (normalized_parameters, tuple(required))
        self.overrides = overrides
        raw_dominance = payload["dominance"]
        if not isinstance(raw_dominance, dict):
            raise ValueError("C3 candidate dominance must be an object")
        dominance: dict[str, tuple[str, ...]] = {}
        for dominant, suppressed in raw_dominance.items():
            if (
                not isinstance(dominant, str) or not _IDENTITY.fullmatch(dominant)
                or not isinstance(suppressed, list) or not suppressed
                or not all(isinstance(item, str) and _IDENTITY.fullmatch(item) for item in suppressed)
                or len(suppressed) != len(set(suppressed))
                or dominant in suppressed
            ):
                raise ValueError("C3 candidate dominance relation is invalid")
            dominance[dominant] = tuple(suppressed)
        self.dominance = dominance
        self.digest = sha256_json(payload)

    def apply(self, candidates: tuple[L1CatalogEntry, ...]) -> tuple[L1CatalogEntry, ...]:
        present = {f"{item.kind}:{item.target}" for item in candidates}
        suppressed = {
            identity
            for dominant, identities in self.dominance.items()
            if dominant in present
            for identity in identities
        }
        refined: list[L1CatalogEntry] = []
        for candidate in candidates:
            if f"{candidate.kind}:{candidate.target}" in suppressed:
                continue
            override = self.overrides.get(f"{candidate.kind}:{candidate.target}")
            if override is None:
                refined.append(candidate)
                continue
            parameters, required = override
            if candidate.parameters and candidate.parameters != parameters:
                raise ValueError("C3 candidate Schema override conflicts with a native contract")
            refined.append(replace(
                candidate,
                parameters=dict(parameters),
                required_parameters=required,
            ))
        return tuple(refined)


__all__ = ["CANDIDATE_SCHEMA_POLICY", "L1CandidateSchemaPolicy"]
