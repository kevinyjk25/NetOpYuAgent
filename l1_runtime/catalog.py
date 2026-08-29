"""Production capability catalog and bounded candidate retrieval."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from dsh_adapter.backend import resolve_backend_mode
from dsh_adapter.skills import build_skill_manifest
from network_runtime.contracts import sha256_json
from retrieval.bm25 import BM25Retriever


CATALOG_POLICY_SCHEMA = "netopyu.io/l1-catalog-policy/v1"
_PARAMETER_LINE = re.compile(r"^- `([^`]+)`: (.+)$", re.MULTILINE)
_IDENTITY = re.compile(r"(?:skill|tool):[A-Za-z0-9_.:-]{1,128}\Z")


@dataclass(frozen=True)
class CapabilityCard:
    target: str
    kind: str
    profile: str
    description: str
    parameter_schemas: dict[str, dict[str, Any]]
    required_parameters: tuple[str, ...]
    workflow_hint: tuple[str, ...]
    risk_level: str
    requires_approval: bool
    searchable_text: str

    @property
    def identity(self) -> str:
        return f"{self.kind}:{self.target}"

    def public_card(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("searchable_text")
        return value


def _skill_parameters(content: str) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    parameters: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for name, description in _PARAMETER_LINE.findall(content):
        parameters[name] = {"type": "string", "description": description}
        if "optional" not in description.casefold() and "可选" not in description:
            required.append(name)
    return parameters, tuple(required)


class CatalogPolicy:
    def __init__(self, path: Path) -> None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "apiVersion", "policyId", "aliases", "requiredParameters",
            "dominance", "skillOverrides",
        } or payload.get("apiVersion") != CATALOG_POLICY_SCHEMA:
            raise ValueError("production L1 catalog policy is invalid")
        aliases = payload["aliases"]
        required = payload["requiredParameters"]
        dominance = payload["dominance"]
        overrides = payload["skillOverrides"]
        if not all(isinstance(value, dict) for value in (
            aliases, required, dominance, overrides,
        )):
            raise ValueError("production L1 catalog policy maps are invalid")
        self.aliases: dict[str, tuple[str, ...]] = {}
        for identity, values in aliases.items():
            if not _IDENTITY.fullmatch(str(identity)) or not isinstance(values, list):
                raise ValueError("production L1 catalog alias is invalid")
            self.aliases[str(identity)] = tuple(str(item) for item in values)
        self.required_parameters: dict[str, tuple[str, ...]] = {}
        for identity, values in required.items():
            if not _IDENTITY.fullmatch(str(identity)) or not isinstance(values, list):
                raise ValueError("production L1 required parameter policy is invalid")
            self.required_parameters[str(identity)] = tuple(str(item) for item in values)
        self.dominance: dict[str, tuple[str, ...]] = {}
        for dominant, suppressed in dominance.items():
            if (
                not _IDENTITY.fullmatch(str(dominant))
                or not isinstance(suppressed, list)
                or not all(_IDENTITY.fullmatch(str(item)) for item in suppressed)
            ):
                raise ValueError("production L1 dominance policy is invalid")
            self.dominance[str(dominant)] = tuple(str(item) for item in suppressed)
        self.skill_overrides: dict[str, tuple[dict[str, dict[str, Any]], tuple[str, ...]]] = {}
        for identity, raw in overrides.items():
            if not _IDENTITY.fullmatch(str(identity)) or not isinstance(raw, dict):
                raise ValueError("production L1 skill override is invalid")
            if set(raw) != {"parameters", "requiredParameters"}:
                raise ValueError("production L1 skill override fields are invalid")
            parameters = {
                str(name): {"type": "string", "description": str(description)}
                for name, description in dict(raw["parameters"]).items()
            }
            required_fields = tuple(str(item) for item in raw["requiredParameters"])
            if not set(required_fields) <= set(parameters):
                raise ValueError("production L1 skill required fields escape Schema")
            self.skill_overrides[str(identity)] = (parameters, required_fields)
        self.digest = sha256_json(payload)

    def refine(self, candidates: tuple[CapabilityCard, ...]) -> tuple[CapabilityCard, ...]:
        present = {item.identity for item in candidates}
        suppressed = {
            identity
            for dominant, identities in self.dominance.items()
            if dominant in present
            for identity in identities
        }
        output: list[CapabilityCard] = []
        for candidate in candidates:
            if candidate.identity in suppressed:
                continue
            override = self.skill_overrides.get(candidate.identity)
            if override is not None:
                schemas, required = override
                if candidate.parameter_schemas and candidate.parameter_schemas != schemas:
                    raise ValueError("production L1 skill override conflicts with native Schema")
                candidate = replace(
                    candidate,
                    parameter_schemas=dict(schemas),
                    required_parameters=required,
                )
            output.append(candidate)
        return tuple(output)


def build_catalog(
    profile: str,
    tool_declarations: list[dict[str, Any]],
    policy: CatalogPolicy,
) -> tuple[CapabilityCard, ...]:
    if profile not in {"lan", "dc", "wan"}:
        raise ValueError("production L1 profile must be lan, dc, or wan")
    entries: list[CapabilityCard] = []
    for raw in sorted(tool_declarations, key=lambda value: str(value.get("name", ""))):
        if not isinstance(raw, dict):
            raise TypeError("production L1 tool declaration must be an object")
        name = str(raw.get("name") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", name):
            raise ValueError("production L1 tool name is invalid")
        raw_parameters = raw.get("parameters") or {}
        if not isinstance(raw_parameters, dict):
            raise ValueError("production L1 tool parameters must be an object")
        schemas: dict[str, dict[str, Any]] = {}
        declared_required: list[str] = []
        for field, raw_schema in raw_parameters.items():
            if not isinstance(raw_schema, dict):
                raise ValueError("production L1 tool parameter Schema must be an object")
            schema = {key: value for key, value in raw_schema.items() if key != "required"}
            schemas[str(field)] = schema
            if raw_schema.get("required") is True:
                declared_required.append(str(field))
        identity = f"tool:{name}"
        required = policy.required_parameters.get(identity, tuple(declared_required))
        if not set(required) <= set(schemas):
            raise ValueError(f"production L1 required fields escape {identity} Schema")
        description = str(raw.get("description") or name)
        tags = tuple(str(item) for item in raw.get("tags") or ())
        aliases = policy.aliases.get(identity, ())
        entries.append(CapabilityCard(
            target=name,
            kind="tool",
            profile=profile,
            description=description,
            parameter_schemas=schemas,
            required_parameters=required,
            workflow_hint=(),
            risk_level=str(raw.get("action_type") or "read_only"),
            requires_approval=bool(raw.get("requires_approval")),
            searchable_text=" ".join((
                name, name.replace("_", " "), description,
                " ".join(schemas), " ".join(tags), " ".join(aliases),
            )),
        ))
    skill_manifest = build_skill_manifest(profile, resolve_backend_mode())
    for raw in skill_manifest["skills"]:
        name = str(raw["name"])
        identity = f"skill:{name}"
        schemas, declared_required = _skill_parameters(str(raw.get("content") or ""))
        required = policy.required_parameters.get(identity, declared_required)
        metadata = dict(raw.get("metadata") or {})
        network_workflow = raw.get("network_workflow")
        workflow = tuple(
            str(item) for item in (
                network_workflow.get("allowed_tools", ())
                if isinstance(network_workflow, dict) else ()
            )
        )
        description = str(raw.get("description") or name)
        aliases = policy.aliases.get(identity, ())
        entries.append(CapabilityCard(
            target=name,
            kind="skill",
            profile=profile,
            description=description,
            parameter_schemas=schemas,
            required_parameters=required,
            workflow_hint=workflow,
            risk_level=str(metadata.get("risk_level") or "low"),
            requires_approval=str(metadata.get("requires_hitl") or "false").casefold() == "true",
            searchable_text=" ".join((
                name, name.replace("-", " "), description,
                " ".join(schemas), str(metadata.get("tags") or ""),
                str(metadata.get("tool_deps") or ""), " ".join(aliases),
            )),
        ))
    identities = [item.identity for item in entries]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate production L1 capability in {profile}")
    return tuple(entries)


class CandidateRetriever:
    def __init__(self, catalog: tuple[CapabilityCard, ...], policy: CatalogPolicy) -> None:
        self.catalog = catalog
        self.policy = policy
        self._by_identity = {item.identity: item for item in catalog}
        self._retriever = BM25Retriever()
        self._retriever.index([
            {
                "id": item.identity,
                "text": item.searchable_text,
                "kind": item.kind,
                "target": item.target,
            }
            for item in catalog
        ])

    def retrieve(self, prompt: str, *, top_k: int = 12) -> tuple[CapabilityCard, ...]:
        if not 1 <= top_k <= 12:
            raise ValueError("production L1 candidate bound must be 1..12")
        result = self._retriever.retrieve(
            prompt[:4_000], top_k=min(len(self.catalog), top_k * 2), min_score=0.0,
        )
        raw = tuple(self._by_identity[item.id] for item in result.matches)
        return self.policy.refine(raw)[:top_k]


def catalog_digest(catalog: tuple[CapabilityCard, ...]) -> str:
    return sha256_json([item.public_card() for item in catalog])


def candidate_digest(candidates: tuple[CapabilityCard, ...]) -> str:
    return sha256_json([item.public_card() for item in candidates])


__all__ = [
    "CapabilityCard",
    "CandidateRetriever",
    "CatalogPolicy",
    "build_catalog",
    "candidate_digest",
    "catalog_digest",
]
