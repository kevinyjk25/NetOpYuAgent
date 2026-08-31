"""Run one immutable local model through the forward L1 -> L0.5 -> L0 path.

The model may emit only a strict, non-executable semantic proposal.  Trusted
catalog lookup, L0.5/L0 materialization, schema validation, Promotion checks,
and observation recording remain deterministic Runtime responsibilities.
Public reverse-bootstrap cases exercise this path but never qualify a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx
import yaml
from pydantic import ValidationError

from network_runtime.contracts import canonical_json, sha256_json, utc_now
from skills.skill_format import parse_skill_md

from .forward_qualification import (
    MODEL_DECISION_SCHEMA,
    ForwardCase,
    ForwardLabel,
    ForwardModelDecision,
    ForwardObservation,
    SemanticContract,
    TRAJECTORY_ROOT,
    STUDY_MANIFEST_SCHEMA,
    adjudicate_forward_labels,
    build_public_calibration,
    evaluator_fingerprint,
    qualify_forward_files,
    record_forward_observation,
    seal_forward_cases,
    _score_one,
)
from .models import API_VERSION, AtomicEffectManifest
from .promotion import (
    StructuredNaturalLanguageSkill,
    assess_promotion,
    build_l05_spec,
    l05_yaml,
    load_capability_catalog,
)


MODEL_RUN_SCHEMA = "netopyu.io/promotion-forward-model-run/v1"
MODEL_RUN_STATE_SCHEMA = "netopyu.io/promotion-forward-model-run-state/v1"
MODEL_RUN_CHECKPOINT_SCHEMA = "netopyu.io/promotion-forward-model-checkpoint/v1"
RUNTIME_REASSESSMENT_SCHEMA = "netopyu.io/promotion-runtime-reassessment/v1"
PROTOCOL_VERSION = "netopyu-forward-authoring-9b/v8"
CATALOG_AUTHORING_GUIDE_VERSION = "netopyu-catalog-authoring-guide/v2"
CATALOG_DECISION_VALIDATOR_VERSION = "netopyu-catalog-decision-validator/v3"
PROMPT_PACKET_VERSION = "netopyu-forward-prompt-packet/v1"
PROMPT_SERIALIZATION = "json-compact-sort-keys/v1"
NORMALIZER_VERSION = "netopyu-enum-value-wrapper-normalizer/v1"
NORMALIZER_RULES = {
    "allowed_path": "semantic_contract.parameters.*.enum[*]",
    "accepted_wrapper": {"value": "one non-null JSON primitive"},
    "rejected": "all other objects, arrays and paths",
    "authority": "syntax-only; cannot activate, execute, add or delete semantics",
}
DEFAULT_OUTPUT_ROOT = Path("artifacts/promotion-forward-model/qwen3.5-9b")

_SYSTEM_PROMPT = f"""You translate one natural-language L1 Skill into a bounded semantic proposal.

Security boundary:
- You never execute a tool, activate a contract, or claim that a write succeeded.
- Use only capability ids present in the supplied trusted Capability Catalog.
- Preserve every documented parameter constraint, intent field, profile, risk,
  approval, preflight, independent verification, compensation, and unknown-outcome rule.
- This is authoring-time compilation of a reusable Skill, not invocation-time execution.
  Concrete values for declared parameters (for example user_id or vlan_id) are neither
  expected nor missing. Choose clarify only when a parameter definition, constraint, or
  safety semantic needed to author the reusable contract is absent. Never invent those
  definitions; reject unsafe or contradictory authoring requests.
- disposition=proposal whenever the supplied L1 and trusted catalog let you construct a
  complete semantic_contract. Never attach a semantic_contract to clarify or reject. Do
  not label a complete compatible contract reject merely because L0 strengthens provider
  optionality or safety. Clarify/reject are fail-closed outcomes, not schema shortcuts.
- For proposal, catalog_id must equal the supplied catalog id. Explicitly identify
  the preflight, success-verification, compensation, and compensation-verification
  capabilities. Every explicit observation must also appear in observation_capabilities.
- Parameter keys must exactly preserve the L1 parameter set. Parameter values must
  contain type and required plus every stated enum/range/length/pattern/resolver/fixed/
  sensitive constraint. Do not add parser defaults such as empty descriptions or nulls.
  enum is always a JSON array of primitive values matching the parameter type, for
  example ["prod", "staging", "dev"], never [{{"value": "prod"}}].
- Catalog input constraints are the provider envelope, not the final L0 policy.
  In particular catalog required=false means the provider accepts omission; it does not
  conflict with an L1/L0 rule that makes the input required. L0 may strengthen required,
  narrow ranges/enums, and strengthen approval, but may not widen type/range/profile or
  weaken a documented L1 safety constraint.
- Predicates and intent must use the exact L0 field/operator/value representation implied
  by the Skill and catalog outputs. Writes always require approval. A provider write
  response is never independent verification.
- Observation selection must follow the catalog's exact observationPhases declaration:
  preflight uses a capability declaring preflight and snapshots its state output
  (for an existence check, `field` is the exact catalog output key, `operator` is
  `exists`, and `expected` is null; never put `exists` in `field`); success verification uses the dedicated verifier and
  requires both success_verification scope and its `passed` output to equal true;
  compensation verification requires compensation_verification scope and the rollback
  verifier and requires its `restored` output to equal true. Every predicate list is
  mandatory and must reference an output actually declared by that selected capability.
- If the L1 explicitly states there is no safe automatic inverse, set compensation and
  compensation-verification capabilities to null, their predicate list to empty,
  requires_compensation=false, and verificationFailed=manual_intervention. This changes
  compensation only: it never removes required preflight or independent success
  verification. Otherwise select both compensation capabilities and use
  verificationFailed=compensate.

Return only JSON conforming to {MODEL_DECISION_SCHEMA}. This output is an untrusted
proposal and grants no execution authority."""


@dataclass(frozen=True)
class ModelReply:
    decision: ForwardModelDecision | None
    raw_content: str
    raw_digest: str
    latency_ms: float
    model_calls: int
    repair_attempts: int
    input_tokens: int
    output_tokens: int
    raw_protocol_valid: bool = True
    syntax_normalization_paths: tuple[str, ...] = ()
    normalized_digest: str | None = None
    validation_errors: tuple[str, ...] = ()
    error: str | None = None
    error_stage: str | None = None


class ModelRunPausedError(ValueError):
    """The evidence run paused after a bounded model-service fault streak."""


def normalize_model_decision_json(
    content: str,
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    """Apply one lossless, path-bounded compatibility normalization.

    Some small models serialize enum primitives as ``{"value": primitive}``.
    Only that exact one-key wrapper is unboxed, only inside declared parameter
    enum arrays.  Any other object remains untouched and therefore fails the
    strict Pydantic contract.  This boundary never changes L0 core schemas.
    """

    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("model decision must be one JSON object")
    contract = payload.get("semantic_contract")
    parameters = contract.get("parameters") if isinstance(contract, dict) else None
    normalized_paths: list[str] = []
    if isinstance(parameters, dict):
        for name, parameter in parameters.items():
            if not isinstance(parameter, dict) or not isinstance(parameter.get("enum"), list):
                continue
            values = parameter["enum"]
            for index, value in enumerate(values):
                if (
                    isinstance(value, dict)
                    and set(value) == {"value"}
                    and value["value"] is not None
                    and not isinstance(value["value"], (dict, list))
                ):
                    values[index] = value["value"]
                    normalized_paths.append(
                        f"semantic_contract.parameters.{name}.enum[{index}]"
                    )
    return payload, tuple(normalized_paths), sha256_json(payload)


class OllamaForwardAdapter:
    """Small native Ollama adapter with schema-constrained output and one repair."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 180.0,
        repair_limit: int = 1,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.repair_limit = repair_limit

    def service_preflight(self) -> dict[str, Any]:
        """Check API/model registry reachability without claiming inference health."""

        started = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                "Ollama registry preflight failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError("Ollama registry preflight returned a non-object response")
        models = payload.get("models") or []
        if not isinstance(models, list):
            raise ValueError("Ollama registry preflight returned an invalid models list")
        match = next((
            item for item in models
            if isinstance(item, dict) and item.get("name") == self.model
        ), None)
        if match is None:
            raise ValueError(f"Ollama model is not installed: {self.model}")
        digest = str(match.get("digest") or "")
        if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
            artifact_digest = f"sha256:{digest}"
        else:
            artifact_digest = sha256_json({"model": self.model, "metadata": match})
        return {
            "schema": "netopyu.io/model-service-preflight/v1",
            "checkedAt": utc_now(),
            "provider": "ollama",
            "base_url": str(httpx.URL(self.base_url).copy_with(
                username=None, password=None,
            )),
            "model": self.model,
            "model_artifact_digest": artifact_digest,
            "registry_reachable": True,
            "model_registered": True,
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
            "claimBoundary": (
                "Registry reachability does not prove inference-engine stability."
            ),
        }

    def artifact_digest(self) -> str:
        return str(self.service_preflight()["model_artifact_digest"])

    def decide(
        self,
        *,
        user_prompt: str,
        decision_validator: Callable[[ForwardModelDecision], tuple[str, ...]] | None = None,
    ) -> ModelReply:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        started = time.monotonic()
        calls = 0
        input_tokens = 0
        output_tokens = 0
        last_content = ""
        last_error = "model returned no response"
        last_normalization_paths: tuple[str, ...] = ()
        last_normalized_digest: str | None = None
        validation_errors: list[str] = []
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for repair in range(self.repair_limit + 1):
                calls += 1
                try:
                    response = client.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model,
                            "stream": False,
                            "think": False,
                            "format": ForwardModelDecision.model_json_schema(by_alias=True),
                            "messages": messages,
                            "options": {
                                "temperature": 0,
                                "num_ctx": 16384,
                                "num_predict": 4096,
                                "seed": 20260830,
                            },
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise TypeError("Ollama chat response must be one JSON object")
                except (httpx.HTTPError, json.JSONDecodeError, TypeError) as error:
                    transport_error = f"{type(error).__name__}: {error}"
                    return ModelReply(
                        decision=None,
                        raw_content=last_content,
                        raw_digest=sha256_json({"content": last_content}),
                        latency_ms=(time.monotonic() - started) * 1000,
                        model_calls=calls,
                        repair_attempts=repair,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        raw_protocol_valid=False,
                        syntax_normalization_paths=last_normalization_paths,
                        normalized_digest=last_normalized_digest,
                        validation_errors=tuple(validation_errors),
                        error=transport_error[:4000],
                        error_stage="model_transport",
                    )
                input_tokens += int(payload.get("prompt_eval_count") or 0)
                output_tokens += int(payload.get("eval_count") or 0)
                last_content = str((payload.get("message") or {}).get("content") or "")
                try:
                    try:
                        decision = ForwardModelDecision.model_validate_json(last_content)
                        raw_protocol_valid = True
                        normalization_paths: tuple[str, ...] = ()
                        normalized_digest = sha256_json(json.loads(last_content))
                    except (ValidationError, ValueError, json.JSONDecodeError) as raw_error:
                        normalized, normalization_paths, normalized_digest = (
                            normalize_model_decision_json(last_content)
                        )
                        last_normalization_paths = normalization_paths
                        last_normalized_digest = normalized_digest
                        if not normalization_paths:
                            raise raw_error
                        decision = ForwardModelDecision.model_validate(normalized)
                        raw_protocol_valid = False
                        validation_errors.append(
                            "raw_protocol: " + f"{type(raw_error).__name__}: {raw_error}"[:3900]
                        )
                    if decision_validator is not None:
                        semantic_errors = decision_validator(decision)
                        if semantic_errors:
                            raise ValueError(
                                "trusted_catalog: " + "; ".join(semantic_errors)
                            )
                    return ModelReply(
                        decision=decision,
                        raw_content=last_content,
                        raw_digest=sha256_json({"content": last_content}),
                        latency_ms=(time.monotonic() - started) * 1000,
                        model_calls=calls,
                        repair_attempts=repair,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        raw_protocol_valid=raw_protocol_valid,
                        syntax_normalization_paths=normalization_paths,
                        normalized_digest=normalized_digest,
                        validation_errors=tuple(validation_errors),
                    )
                except (ValidationError, ValueError) as error:
                    last_error = f"{type(error).__name__}: {error}"
                    validation_errors.append(last_error[:4000])
                    if repair >= self.repair_limit:
                        break
                    messages.extend((
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": (
                                "Your JSON failed strict validation. Correct only the JSON; "
                                "do not add prose and do not change disposition merely to "
                                "evade validation. Re-read the original L1 and trusted Catalog. "
                                "When they define a complete contract, construct or preserve "
                                "semantic_contract and use disposition=proposal. Catalog "
                                "required=false is provider optionality and may be strengthened "
                                "to L1 required=true; a generic object output is still a valid "
                                "declared output. Clarify only for an authoring semantic truly "
                                "absent from both inputs and list that semantic in missing_fields. "
                                "A contract without an automatic compensator still requires "
                                "preflight and independent success verification when the L1 "
                                "requires them. Predicate field must be an exact output key "
                                "from the selected phase capability; for existence use "
                                "operator=exists and expected=null. "
                                "Validation summary: " + last_error[:2000]
                            ),
                        },
                    ))
        return ModelReply(
            decision=None,
            raw_content=last_content,
            raw_digest=sha256_json({"content": last_content}),
            latency_ms=(time.monotonic() - started) * 1000,
            model_calls=calls,
            repair_attempts=max(0, calls - 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_protocol_valid=False,
            syntax_normalization_paths=last_normalization_paths,
            normalized_digest=last_normalized_digest,
            validation_errors=tuple(validation_errors),
            error=last_error[:4000],
        )


def authoring_protocol_digest() -> str:
    return sha256_json({
        "protocol": PROTOCOL_VERSION,
        "system_prompt": _SYSTEM_PROMPT,
        "decision_schema": ForwardModelDecision.model_json_schema(by_alias=True),
        "normalizer_version": NORMALIZER_VERSION,
        "normalizer_rules": NORMALIZER_RULES,
        "catalog_authoring_guide_version": CATALOG_AUTHORING_GUIDE_VERSION,
        "catalog_decision_validator_version": CATALOG_DECISION_VALIDATOR_VERSION,
        "prompt_packet_version": PROMPT_PACKET_VERSION,
        "prompt_serialization": PROMPT_SERIALIZATION,
    })


def _parameter_view(raw: dict[str, Any]) -> dict[str, Any]:
    useful = {
        "type", "required", "enum", "minimum", "maximum", "minLength",
        "maxLength", "pattern", "resolver", "fixed", "sensitive",
    }
    return {
        key: value for key, value in raw.items()
        if key in useful and value not in (None, "", [], ())
        and not (key == "sensitive" and value is False)
    }


def _catalog_prompt(path: Path, *, catalog_id: str) -> tuple[str, str]:
    catalog, digest, _ = load_capability_catalog(path)
    capabilities: list[dict[str, Any]] = []
    phase_options: dict[str, list[dict[str, Any]]] = {
        "preflight": [],
        "success_verification": [],
        "compensation_verification": [],
    }
    for item in catalog.capabilities:
        output_keys = list(item.outputs)
        capabilities.append({
            "id": item.id,
            "role": item.role,
            "observationPhases": list(item.observation_phases),
            "tool": item.tool,
            "profiles": list(item.profiles),
            "inputs": {
                name: _parameter_view(value.model_dump(by_alias=True, mode="json"))
                for name, value in item.inputs.items()
            },
            "outputs": {
                name: _parameter_view(value.model_dump(by_alias=True, mode="json"))
                for name, value in item.outputs.items()
            },
        })
        for phase in item.observation_phases:
            required_predicates = [
                predicate.model_dump(by_alias=True, mode="json")
                for predicate in item.phase_predicates.get(phase, ())
            ]
            phase_options[phase].append({
                "capability": item.id,
                "outputKeys": output_keys,
                "requiredPredicates": required_predicates,
                "predicateSyntax": {
                    "existence": {
                        "field": output_keys[0] if output_keys else "<declared-output-key>",
                        "operator": "exists",
                        "expected": None,
                    },
                    "rule": (
                        "field is an exact outputKeys entry; operator carries the test. "
                        "For Catalog v3 copy requiredPredicates exactly; never weaken or "
                        "replace them. For older catalogs choose expected from L1 semantics, "
                        "never from provider write response."
                    ),
                },
            })
    packet = {
        "catalog_id": catalog_id,
        "catalog_sha256": digest,
        "provider": catalog.provider,
        "version": catalog.version,
        "authoringGuide": {
            "providerEnvelopeRule": (
                "Catalog input required=false means the provider accepts omission; "
                "L1/L0 may safely strengthen it to required=true. This is compatible."
            ),
            "phaseOptions": phase_options,
            "phaseAvailability": {
                phase: bool(options) for phase, options in phase_options.items()
            },
            "nonCompensableRule": (
                "No safe automatic inverse removes compensation only. It does not "
                "remove preflight or independent success verification."
            ),
        },
        "capabilities": capabilities,
    }
    return json.dumps(
        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ), digest


def _case_prompt(case: ForwardCase) -> tuple[str, str]:
    trajectory = TRAJECTORY_ROOT / case.family
    catalog_text, catalog_digest = _catalog_prompt(
        trajectory / "00-capability-catalog.yaml", catalog_id=case.family,
    )
    packet = {
        "apiVersion": PROMPT_PACKET_VERSION,
        "case": {
            "case_id": case.case_id,
            "family": case.family,
            "profile": case.profile,
            "language": case.language,
            "challenge": case.challenge,
        },
        "trustedCapabilityCatalog": json.loads(catalog_text),
        "untrustedL1SkillRequest": case.prompt,
    }
    return json.dumps(
        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ), catalog_digest


def _prompt_corpus_metrics(cases: Iterable[ForwardCase]) -> dict[str, Any]:
    case_list = list(cases)
    prompts = [_case_prompt(case)[0] for case in case_list]
    sizes = [len(prompt.encode("utf-8")) for prompt in prompts]
    legacy_sizes: list[int] = []
    for case in case_list:
        trajectory = TRAJECTORY_ROOT / case.family
        catalog_text, _ = _catalog_prompt(
            trajectory / "00-capability-catalog.yaml", catalog_id=case.family,
        )
        pretty_catalog = json.dumps(
            json.loads(catalog_text), ensure_ascii=False, indent=2, sort_keys=True,
        )
        legacy_prompt = (
            "FORWARD CASE\n"
            f"case_id: {case.case_id}\n"
            f"family: {case.family}\n"
            f"profile: {case.profile}\n"
            f"language: {case.language}\n"
            f"challenge: {case.challenge}\n\n"
            "TRUSTED CAPABILITY CATALOG\n"
            f"{pretty_catalog}\n\n"
            "UNTRUSTED L1 SKILL REQUEST\n"
            f"{case.prompt}"
        )
        legacy_sizes.append(len(legacy_prompt.encode("utf-8")))
    total = sum(sizes)
    legacy_total = sum(legacy_sizes)
    return {
        "packet_version": PROMPT_PACKET_VERSION,
        "serialization": PROMPT_SERIALIZATION,
        "case_count": len(prompts),
        "system_prompt_bytes": len(_SYSTEM_PROMPT.encode("utf-8")),
        "total_user_prompt_bytes": total,
        "mean_user_prompt_bytes": round(total / len(sizes), 3) if sizes else 0,
        "max_user_prompt_bytes": max(sizes, default=0),
        "legacy_v7_equivalent_user_prompt_bytes": legacy_total,
        "byte_reduction_vs_v7_equivalent": (
            round((legacy_total - total) / legacy_total, 6) if legacy_total else 0
        ),
        "prompt_corpus_digest": sha256_json(prompts),
    }


def _argument_bindings(capability: Any, semantic: SemanticContract) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for name, parameter in capability.inputs.items():
        if parameter.fixed is not None:
            bindings[name] = parameter.fixed
        elif name in semantic.parameters:
            bindings[name] = f"${{arguments.{name}}}"
    return bindings


def _selected_capability(
    capabilities: dict[str, Any], capability_id: str | None, *, role: str,
    observation_phase: str | None = None,
) -> Any:
    if capability_id is None or capability_id not in capabilities:
        raise ValueError(f"unknown {role} capability: {capability_id}")
    capability = capabilities[capability_id]
    if capability.role != role:
        raise ValueError(
            f"capability {capability_id} has role {capability.role}, expected {role}"
        )
    if observation_phase is not None and not capability.supports_observation_phase(
        observation_phase
    ):
        raise ValueError(
            f"capability {capability_id} does not support observation phase "
            f"{observation_phase}; declared={list(capability.observation_phases)}"
        )
    return capability


def _candidate_from_semantic(
    semantic: SemanticContract, *, catalog_path: Path, family: str,
) -> AtomicEffectManifest:
    if semantic.catalog_id != family:
        raise ValueError("semantic catalog_id does not match the trusted case catalog")
    catalog, _, _ = load_capability_catalog(catalog_path)
    capabilities = catalog.by_id()
    effect = _selected_capability(
        capabilities, semantic.effect_capability, role="effect",
    )
    preflight = _selected_capability(
        capabilities, semantic.preflight_capability, role="observation",
        observation_phase="preflight",
    )
    verification = _selected_capability(
        capabilities, semantic.verification_capability, role="observation",
        observation_phase="success_verification",
    )
    compensation = None
    rollback_verifier = None
    if semantic.requires_compensation:
        compensation = _selected_capability(
            capabilities, semantic.compensation_capability, role="compensation",
        )
        rollback_verifier = _selected_capability(
            capabilities,
            semantic.compensation_verification_capability,
            role="observation",
            observation_phase="compensation_verification",
        )
    selected = tuple(item for item in (
        effect, preflight, verification, compensation, rollback_verifier,
    ) if item is not None)
    for item in selected:
        if not set(semantic.profiles).issubset(set(item.profiles)):
            raise ValueError(f"profile is outside capability scope: {item.id}")
    raw = {
        "apiVersion": API_VERSION,
        "kind": "AtomicEffect",
        "metadata": {
            "id": f"{family}.model-candidate",
            "version": "0.1.0",
            "owner": "netopyu-forward-evaluator",
            "description": "Untrusted model proposal; no registration authority",
            "labels": {"authority": "proposal-only"},
        },
        "spec": {
            "template": "netopyu-forward-evaluator-v1",
            "profiles": list(semantic.profiles),
            "effect": {
                "capability": effect.id,
                "tool": effect.tool,
                "request": _argument_bindings(effect, semantic),
            },
            "intent": semantic.intent.model_dump(by_alias=True, mode="json"),
            "parameters": {
                name: value.model_dump(by_alias=True, mode="json")
                for name, value in semantic.parameters.items()
            },
            "preflight": [{
                "id": "approved-state",
                "capability": preflight.id,
                "arguments": _argument_bindings(preflight, semantic),
                "snapshotFields": list(preflight.outputs),
                "predicates": [
                    item.model_dump(by_alias=True, mode="json")
                    for item in semantic.preflight_predicates
                ],
            }],
            "verification": {
                "capability": verification.id,
                "arguments": _argument_bindings(verification, semantic),
                "predicates": [
                    item.model_dump(by_alias=True, mode="json")
                    for item in semantic.verification_predicates
                ],
            },
            "approval": {
                "required": semantic.approval_required,
                "risk": semantic.risk,
                "mode": semantic.approval_mode,
            },
            "failurePolicy": semantic.failure_policy.model_dump(
                by_alias=True, mode="json",
            ),
        },
    }
    if compensation is not None and rollback_verifier is not None:
        raw["spec"]["compensation"] = {
            "capability": compensation.id,
            "tool": compensation.tool,
            "arguments": _argument_bindings(compensation, semantic),
            "verification": {
                "capability": rollback_verifier.id,
                "arguments": _argument_bindings(rollback_verifier, semantic),
                "predicates": [
                    item.model_dump(by_alias=True, mode="json")
                    for item in semantic.compensation_verification_predicates
                ],
            },
        }
    return AtomicEffectManifest.model_validate(raw)


def validate_forward_model_decision_against_catalog(
    *, case: ForwardCase, decision: ForwardModelDecision,
) -> tuple[str, ...]:
    """Check one untrusted decision against the trusted Catalog before accepting it.

    This is an authoring-time repair boundary, not an execution or activation path.
    It deliberately reports only deterministic Catalog/phase/output facts that the
    model can correct without gaining authority. Promotion remains the final gate.
    """

    if decision.disposition != "proposal":
        return ()
    semantic = decision.semantic_contract
    if semantic is None:
        return ("PROPOSAL_CONTRACT_MISSING",)
    trajectory = TRAJECTORY_ROOT / case.family
    catalog_path = trajectory / "00-capability-catalog.yaml"
    errors: list[str] = []
    try:
        _candidate_from_semantic(
            semantic, catalog_path=catalog_path, family=case.family,
        )
    except (ValidationError, ValueError) as error:
        errors.append(f"CATALOG_CONTRACT_INVALID {str(error)[:1000]}")
        return tuple(errors)

    catalog, _, _ = load_capability_catalog(catalog_path)
    capabilities = catalog.by_id()
    phase_bindings = (
        (
            "preflight", semantic.preflight_capability,
            semantic.preflight_predicates,
        ),
        (
            "success_verification", semantic.verification_capability,
            semantic.verification_predicates,
        ),
        (
            "compensation_verification",
            semantic.compensation_verification_capability,
            semantic.compensation_verification_predicates,
        ),
    )
    for phase, capability_id, predicates in phase_bindings:
        if capability_id is None:
            continue
        capability = capabilities[capability_id]
        allowed = sorted(capability.outputs)
        for index, predicate in enumerate(predicates):
            root_field = predicate.field.split(".", 1)[0]
            if root_field not in capability.outputs:
                errors.append(
                    "CATALOG_OUTPUT_FIELD_UNKNOWN "
                    f"phase={phase} capability={capability_id} "
                    f"predicate[{index}].field={predicate.field!r} "
                    f"allowed_outputs={allowed}; field must be one allowed output key"
                )
        required_predicates = capability.phase_predicates.get(phase, ())
        if required_predicates:
            required = {
                canonical_json(item.model_dump(by_alias=True, mode="json"))
                for item in required_predicates
            }
            actual = {
                canonical_json(item.model_dump(by_alias=True, mode="json"))
                for item in predicates
            }
            if not required.issubset(actual):
                errors.append(
                    "CATALOG_PHASE_PREDICATE_MISMATCH "
                    f"phase={phase} capability={capability_id} "
                    "candidate predicates must include every requiredPredicates entry="
                    + json.dumps(
                        [
                            item.model_dump(by_alias=True, mode="json")
                            for item in required_predicates
                        ],
                        ensure_ascii=False, sort_keys=True,
                    )
                )
    return tuple(errors)


def _l05_from_semantic(
    base: StructuredNaturalLanguageSkill, semantic: SemanticContract,
) -> StructuredNaturalLanguageSkill:
    raw = base.model_dump(by_alias=True, mode="json")
    raw["profiles"] = list(semantic.profiles)
    raw["parameters"] = {
        name: json.dumps(
            value.model_dump(by_alias=True, mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        for name, value in semantic.parameters.items()
    }
    raw["semanticIntents"] = [{
        "effectCapability": semantic.effect_capability,
        **semantic.intent.model_dump(by_alias=True, mode="json"),
    }]
    raw["capabilities"] = {
        "effects": [semantic.effect_capability],
        "observations": list(semantic.observation_capabilities),
        "compensations": (
            [semantic.compensation_capability]
            if semantic.compensation_capability else []
        ),
        "preflightObservations": [semantic.preflight_capability],
        "successVerificationObservations": [semantic.verification_capability],
        "compensationVerificationObservations": (
            [semantic.compensation_verification_capability]
            if semantic.compensation_verification_capability else []
        ),
    }
    phase_options = {
        "preflight": [semantic.preflight_capability],
        "effect": [semantic.effect_capability],
        "verification": [semantic.verification_capability],
        "compensation": (
            [semantic.compensation_capability]
            if semantic.compensation_capability else []
        ),
        "compensation_verification": (
            [semantic.compensation_verification_capability]
            if semantic.compensation_verification_capability else []
        ),
    }
    for step in raw["workflow"]:
        values = phase_options.get(step["phase"])
        if values is not None:
            step["capabilityOptions"] = [item for item in values if item]
    raw["safety"]["risk"] = semantic.risk
    raw["safety"]["approvalRequired"] = semantic.approval_required
    raw["unresolvedQuestions"] = []
    return StructuredNaturalLanguageSkill.model_validate(raw)


def materialize_and_assess(
    *, case: ForwardCase, semantic: SemanticContract, destination: Path,
) -> tuple[Path, dict[str, Any]]:
    """Compile an untrusted semantic proposal through real Promotion checks."""

    trajectory = TRAJECTORY_ROOT / case.family
    source_text = (trajectory / "01-L1-SKILL.md").read_text(encoding="utf-8")
    parsed = parse_skill_md(source_text)
    source_root = destination / "source" / parsed.name
    source_root.mkdir(parents=True, exist_ok=True)
    skill_path = source_root / "SKILL.md"
    skill_path.write_text(source_text, encoding="utf-8")
    catalog_path = trajectory / "00-capability-catalog.yaml"
    candidate = _candidate_from_semantic(
        semantic, catalog_path=catalog_path, family=case.family,
    )
    base_l05 = build_l05_spec(
        skill_path=skill_path, capability_catalog_path=catalog_path,
    )
    l05 = _l05_from_semantic(base_l05, semantic)
    proposal = destination / "proposal"
    proposal.mkdir(parents=True, exist_ok=True)
    l05_path = proposal / "02-L0.5.yaml"
    candidate_path = proposal / "03-L0-authoring.yaml"
    l05_path.write_text(l05_yaml(l05), encoding="utf-8")
    candidate_path.write_text(yaml.safe_dump(
        candidate.model_dump(by_alias=True, mode="json"),
        sort_keys=False,
        allow_unicode=True,
    ), encoding="utf-8")
    assessment = assess_promotion(
        skill_path=skill_path,
        l05_path=l05_path,
        candidate_path=candidate_path,
        capability_catalog_path=catalog_path,
    )
    (proposal / "report.json").write_text(
        json.dumps(assessment.report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return proposal, assessment.report


def _balanced_cases(cases: Iterable[ForwardCase], limit: int | None) -> list[ForwardCase]:
    grouped: dict[str, list[ForwardCase]] = defaultdict(list)
    for case in cases:
        grouped[case.family].append(case)
    ordered: list[ForwardCase] = []
    depth = max(len(values) for values in grouped.values())
    for index in range(depth):
        for family in sorted(grouped):
            if index < len(grouped[family]):
                ordered.append(grouped[family][index])
    return ordered if limit is None else ordered[:limit]


def _write_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.write_text("".join(
        json.dumps(item.model_dump(by_alias=True, mode="json"),
                   ensure_ascii=False, sort_keys=True) + "\n"
        for item in values
    ), encoding="utf-8")


def _load_private_cases(path: str | Path) -> list[ForwardCase]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("private forward cases are missing or exceed 32 MiB")
    cases = [
        ForwardCase.model_validate_json(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [item.case_id for item in cases]
    if not cases or len(identifiers) != len(set(identifiers)):
        raise ValueError("private forward cases must be non-empty with unique ids")
    return cases


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Durably replace one JSON object without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _model_run_fingerprint(configuration: dict[str, Any]) -> str:
    """Bind checkpoints to the immutable evidence-producing configuration."""

    return sha256_json({
        "schema": MODEL_RUN_STATE_SCHEMA,
        "configuration": configuration,
    })


def _checkpoint_name(*, position: int, repetition: int) -> str:
    return f"r{repetition:02d}-c{position:04d}.json"


def _write_case_checkpoint(
    checkpoint_dir: Path,
    *,
    run_id: str,
    run_fingerprint: str,
    position: int,
    observation: ForwardObservation,
    failure: dict[str, str] | None,
) -> Path:
    body: dict[str, Any] = {
        "schema": MODEL_RUN_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "case_id": observation.case_id,
        "repetition": observation.repetition,
        "position": position,
        "observation": observation.model_dump(by_alias=True, mode="json"),
        "failure": failure,
    }
    body["checkpoint_digest"] = sha256_json(body)
    path = checkpoint_dir / _checkpoint_name(
        position=position, repetition=observation.repetition,
    )
    _write_json_atomic(path, body)
    return path


def _load_case_checkpoints(
    checkpoint_dir: Path,
    *,
    run_id: str,
    run_fingerprint: str,
    expected_positions: dict[tuple[str, int], int],
) -> tuple[
    dict[tuple[str, int], ForwardObservation],
    dict[tuple[str, int], dict[str, str]],
]:
    observations: dict[tuple[str, int], ForwardObservation] = {}
    failures: dict[tuple[str, int], dict[str, str]] = {}
    if not checkpoint_dir.is_dir():
        return observations, failures
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid model checkpoint {path.name}: {error}") from error
        if not isinstance(body, dict):
            raise ValueError(f"invalid model checkpoint object: {path.name}")
        digest = body.pop("checkpoint_digest", None)
        if digest != sha256_json(body):
            raise ValueError(f"model checkpoint digest mismatch: {path.name}")
        if body.get("schema") != MODEL_RUN_CHECKPOINT_SCHEMA:
            raise ValueError(f"model checkpoint schema mismatch: {path.name}")
        if body.get("run_id") != run_id or body.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"model checkpoint run identity mismatch: {path.name}")
        try:
            observation = ForwardObservation.model_validate(body.get("observation"))
        except ValidationError as error:
            raise ValueError(f"invalid checkpoint observation: {path.name}") from error
        key = (observation.case_id, observation.repetition)
        position = expected_positions.get(key)
        if position is None or body.get("case_id") != key[0]:
            raise ValueError(f"unexpected checkpoint case/repetition: {path.name}")
        if body.get("repetition") != key[1] or body.get("position") != position:
            raise ValueError(f"checkpoint index mismatch: {path.name}")
        if path.name != _checkpoint_name(position=position, repetition=key[1]):
            raise ValueError(f"checkpoint filename mismatch: {path.name}")
        if key in observations:
            raise ValueError(f"duplicate model checkpoint: {key[0]} repetition {key[1]}")
        observations[key] = observation
        failure = body.get("failure")
        if failure is not None:
            if not isinstance(failure, dict) or any(
                not isinstance(failure.get(name), str)
                for name in ("case_id", "repetition", "stage", "error")
            ):
                raise ValueError(f"invalid checkpoint failure: {path.name}")
            if failure["case_id"] != key[0] or failure["repetition"] != str(key[1]):
                raise ValueError(f"checkpoint failure identity mismatch: {path.name}")
            failures[key] = failure
    return observations, failures


def _validate_resume_inputs(
    state: dict[str, Any], *, expected_configuration: dict[str, Any], root: Path,
) -> None:
    if state.get("schema") != MODEL_RUN_STATE_SCHEMA:
        raise ValueError("active model run state schema is invalid")
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("active model run id is invalid")
    fingerprint = _model_run_fingerprint(expected_configuration)
    if state.get("run_fingerprint") != fingerprint:
        raise ValueError(
            "resume configuration mismatch: model artifact, protocol, data set, "
            "catalog snapshot, repetitions, repair policy, or transport-failure "
            "policy changed"
        )
    if state.get("configuration") != expected_configuration:
        raise ValueError("resume configuration payload mismatch")
    artifacts = state.get("input_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("active model run input artifact binding is missing")
    filenames = {
        "cases": "cases.jsonl",
        "reviewer_one": "reverse-reviewer-a.jsonl",
        "reviewer_two": "reverse-reviewer-b.jsonl",
        "manifest": "manifest.json",
        "study_plan": "study-plan.json",
        "resolutions": "resolutions.jsonl",
    }
    required = {"cases", "reviewer_one", "reviewer_two", "manifest"}
    if not required.issubset(artifacts):
        raise ValueError("active model run input artifact binding is incomplete")
    for name, item in artifacts.items():
        filename = filenames.get(name)
        if filename is None:
            raise ValueError(f"active model run artifact binding is unknown: {name}")
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"active model run artifact binding is invalid: {name}")
        path = root / filename
        if not path.is_file() or _file_digest(path) != item["sha256"]:
            raise ValueError(f"resume input artifact changed or is missing: {name}")


def _catalog_snapshot_digest(cases: Iterable[ForwardCase]) -> str:
    snapshots = {}
    for family in sorted({item.family for item in cases}):
        path = TRAJECTORY_ROOT / family / "00-capability-catalog.yaml"
        snapshots[family] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return sha256_json(snapshots)


def inspect_private_study_inputs(
    cases_path: str | Path,
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Resolve immutable run digests without inference or access to labels."""

    cases = _load_private_cases(cases_path)
    if {item.split for item in cases} != {"private_holdout"}:
        raise ValueError("private study inspection accepts private_holdout cases only")
    adapter = OllamaForwardAdapter(
        model=model, base_url=base_url, timeout_seconds=timeout_seconds,
        repair_limit=0,
    )
    return {
        "schema": "netopyu.io/promotion-forward-study-inputs/v1",
        "model": model,
        "model_artifact_digest": adapter.artifact_digest(),
        "authoring_protocol_digest": authoring_protocol_digest(),
        "catalog_snapshot_digest": _catalog_snapshot_digest(cases),
        "evaluator_fingerprint": evaluator_fingerprint(),
        "case_count": len(cases),
        "family_count": len({item.family for item in cases}),
        "cases_digest": sha256_json([
            item.model_dump(by_alias=True, mode="json")
            for item in sorted(cases, key=lambda value: value.case_id)
        ]),
        "model_calls": 0,
        "contains_prompts": False,
    }


def _error_observation(
    *, case: ForwardCase, repetition: int, adapter: OllamaForwardAdapter,
    model_digest: str, protocol_digest: str, catalog_digest: str,
    reply: ModelReply, semantic: SemanticContract | None, error: Exception | str,
) -> ForwardObservation:
    disposition = "proposal" if semantic is not None else "protocol_error"
    return ForwardObservation(
        case_id=case.case_id,
        repetition=repetition,
        model=adapter.model,
        model_artifact_digest=model_digest,
        authoring_protocol_digest=protocol_digest,
        catalog_snapshot_digest=catalog_digest,
        raw_protocol_valid=reply.raw_protocol_valid,
        valid_protocol=semantic is not None,
        disposition=disposition,
        semantic_contract=semantic,
        promotion_status="blocked" if semantic is not None else "protocol_error",
        blocking_requirements=1 if semantic is not None else 0,
        latency_ms=reply.latency_ms,
        model_calls=reply.model_calls,
        repair_attempts=reply.repair_attempts,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        syntax_normalization_count=len(reply.syntax_normalization_paths),
        syntax_normalization_paths=reply.syntax_normalization_paths,
        normalized_output_digest=reply.normalized_digest,
        output_digest=sha256_json({
            "raw": reply.raw_digest,
            "error_type": type(error).__name__ if isinstance(error, Exception) else "error",
        }),
    )


def run_public_model_evaluation(
    *,
    model: str = "qwen3.5:9b",
    base_url: str = "http://127.0.0.1:11434",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    limit: int | None = None,
    families: tuple[str, ...] = (),
    case_ids: tuple[str, ...] = (),
    repetitions: int = 1,
    timeout_seconds: float = 180.0,
    repair_limit: int = 1,
    transport_failure_limit: int = 2,
    resume: bool = False,
    private_cases_path: str | Path | None = None,
    private_manifest_path: str | Path | None = None,
    private_study_plan_path: str | Path | None = None,
    private_reviewer_one_path: str | Path | None = None,
    private_reviewer_two_path: str | Path | None = None,
    private_resolutions_path: str | Path | None = None,
) -> dict[str, Any]:
    private_values = (
        private_cases_path, private_manifest_path, private_study_plan_path,
        private_reviewer_one_path, private_reviewer_two_path,
    )
    private_mode = any(value is not None for value in private_values)
    if private_mode and any(value is None for value in private_values):
        raise ValueError(
            "private run requires cases, manifest, study plan and both reviewer files"
        )
    if private_mode and limit is not None:
        raise ValueError("private qualification cannot use a partial --limit")
    if private_mode and (families or case_ids):
        raise ValueError("private qualification cannot filter families or case ids")
    if not private_mode and limit is not None and not 1 <= limit <= 210:
        raise ValueError("limit must be between 1 and 210")
    if not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be between 1 and 10")
    if not 0 <= transport_failure_limit <= 100:
        raise ValueError("transport_failure_limit must be between 0 and 100")
    private_sources: dict[str, Path] = {}
    if private_mode:
        assert private_cases_path is not None
        assert private_manifest_path is not None
        assert private_study_plan_path is not None
        assert private_reviewer_one_path is not None
        assert private_reviewer_two_path is not None
        private_sources = {
            "cases": Path(private_cases_path).expanduser().resolve(),
            "manifest": Path(private_manifest_path).expanduser().resolve(),
            "study_plan": Path(private_study_plan_path).expanduser().resolve(),
            "reviewer_one": Path(private_reviewer_one_path).expanduser().resolve(),
            "reviewer_two": Path(private_reviewer_two_path).expanduser().resolve(),
        }
        if private_resolutions_path is not None:
            private_sources["resolutions"] = Path(
                private_resolutions_path
            ).expanduser().resolve()
        if any(not path.is_file() for path in private_sources.values()):
            raise ValueError("one or more private qualification inputs are missing")
        cases = _load_private_cases(private_sources["cases"])
        private_manifest = json.loads(
            private_sources["manifest"].read_text(encoding="utf-8")
        )
        if private_manifest.get("apiVersion") != STUDY_MANIFEST_SCHEMA:
            raise ValueError("private model run requires a pre-registered v2 manifest")
        adjudication = adjudicate_forward_labels(
            private_sources["cases"], private_sources["manifest"],
            private_sources["reviewer_one"], private_sources["reviewer_two"],
            study_plan_path=private_sources["study_plan"],
            resolutions_path=private_sources.get("resolutions"),
        )
        if not adjudication["qualification_eligible"]:
            raise ValueError("private reviewer/adjudication evidence is not qualification-ready")
        if private_manifest.get("planned_model") != model:
            raise ValueError("requested model differs from the pre-registered study")
        if private_manifest.get("repetitions") != repetitions:
            raise ValueError("requested repetitions differ from the pre-registered study")
        reviewer_one: list[ForwardLabel] = []
        reviewer_two: list[ForwardLabel] = []
    else:
        all_cases, all_labels = build_public_calibration()
        available_families = {item.family for item in all_cases}
        unknown_families = sorted(set(families) - available_families)
        available_case_ids = {item.case_id for item in all_cases}
        unknown_case_ids = sorted(set(case_ids) - available_case_ids)
        if unknown_families:
            raise ValueError(f"unknown public families: {unknown_families}")
        if unknown_case_ids:
            raise ValueError(f"unknown public case ids: {unknown_case_ids}")
        selected_cases = [
            item for item in all_cases
            if (not families or item.family in families)
            and (not case_ids or item.case_id in case_ids)
        ]
        if not selected_cases:
            raise ValueError("public case filters selected no cases")
        cases = _balanced_cases(selected_cases, limit)
        labels_by_id = {item.case_id: item for item in all_labels}
        labels = [labels_by_id[item.case_id] for item in cases]
        reviewer_one = [
            item.model_copy(update={"reviewer_id": "public-reviewer-a"})
            for item in labels
        ]
        reviewer_two = [
            item.model_copy(update={"reviewer_id": "public-reviewer-b"})
            for item in labels
        ]
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases_path = root / "cases.jsonl"
    first_path = root / "reverse-reviewer-a.jsonl"
    second_path = root / "reverse-reviewer-b.jsonl"
    observations_path = root / "observations.jsonl"
    manifest_path = root / "manifest.json"
    study_plan_path = root / "study-plan.json"
    resolutions_path = root / "resolutions.jsonl"
    evaluator_path = root / "evaluator-report.json"
    active_state_path = root / "active-run.json"
    adapter = OllamaForwardAdapter(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        repair_limit=repair_limit,
    )
    service_preflight = adapter.service_preflight()
    model_digest = str(service_preflight["model_artifact_digest"])
    protocol_digest = authoring_protocol_digest()
    catalog_digest = _catalog_snapshot_digest(cases)
    prompt_metrics = _prompt_corpus_metrics(cases)
    if private_mode:
        for key, actual in (
            ("model_artifact_digest", model_digest),
            ("authoring_protocol_digest", protocol_digest),
            ("catalog_snapshot_digest", catalog_digest),
        ):
            if private_manifest.get(key) != actual:
                raise ValueError(f"private study runtime binding drift: {key}")
    configuration: dict[str, Any] = {
        "model": model,
        "model_artifact_digest": model_digest,
        "authoring_protocol_digest": protocol_digest,
        "catalog_snapshot_digest": catalog_digest,
        "case_payload_digest": sha256_json([
            item.model_dump(by_alias=True, mode="json") for item in cases
        ]),
        "reviewer_one_payload_digest": (
            _file_digest(private_sources["reviewer_one"])
            if private_mode else sha256_json([
                item.model_dump(by_alias=True, mode="json") for item in reviewer_one
            ])
        ),
        "reviewer_two_payload_digest": (
            _file_digest(private_sources["reviewer_two"])
            if private_mode else sha256_json([
                item.model_dump(by_alias=True, mode="json") for item in reviewer_two
            ])
        ),
        "study_plan_digest": (
            _file_digest(private_sources["study_plan"]) if private_mode else None
        ),
        "resolutions_digest": (
            _file_digest(private_sources["resolutions"])
            if "resolutions" in private_sources else None
        ),
        "private_study": private_mode,
        "limit": limit,
        "families": sorted(set(families)),
        "case_ids": sorted(set(case_ids)),
        "repetitions": repetitions,
        "repair_limit": repair_limit,
        "transport_failure_limit": transport_failure_limit,
        "prompt_packet_version": PROMPT_PACKET_VERSION,
        "prompt_serialization": PROMPT_SERIALIZATION,
        "case_count": len(cases),
    }
    run_fingerprint = _model_run_fingerprint(configuration)
    expected_positions = {
        (case.case_id, repetition): position
        for repetition in range(1, repetitions + 1)
        for position, case in enumerate(cases, start=1)
    }
    if resume:
        if not active_state_path.is_file():
            raise ValueError("no active model run exists for --resume")
        try:
            state = json.loads(active_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("active model run state is unreadable") from error
        if not isinstance(state, dict):
            raise ValueError("active model run state is invalid")
        _validate_resume_inputs(
            state, expected_configuration=configuration, root=root,
        )
        run_id = str(state["run_id"])
        checkpoint_dir = root / "checkpoints" / run_id
        observations_by_key, failures_by_key = _load_case_checkpoints(
            checkpoint_dir,
            run_id=run_id,
            run_fingerprint=run_fingerprint,
            expected_positions=expected_positions,
        )
        resumed_from_checkpoints = len(observations_by_key)
        state["status"] = "running"
        state.pop("pause", None)
        state["resume_count"] = int(state.get("resume_count") or 0) + 1
        state.setdefault("service_preflights", []).append(service_preflight)
        state["completed_observations"] = resumed_from_checkpoints
        state["updatedAt"] = utc_now()
        _write_json_atomic(active_state_path, state)
        print(json.dumps({
            "state": "resumed",
            "run_id": run_id,
            "completed": resumed_from_checkpoints,
            "total": len(expected_positions),
        }, ensure_ascii=False), flush=True)
    else:
        run_id = secrets.token_hex(16)
        checkpoint_dir = root / "checkpoints" / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        if private_mode:
            copies = {
                "cases": cases_path,
                "manifest": manifest_path,
                "study_plan": study_plan_path,
                "reviewer_one": first_path,
                "reviewer_two": second_path,
            }
            if "resolutions" in private_sources:
                copies["resolutions"] = resolutions_path
            for name, destination in copies.items():
                destination.write_bytes(private_sources[name].read_bytes())
                destination.chmod(0o600)
        else:
            _write_jsonl(cases_path, cases)
            _write_jsonl(first_path, reviewer_one)
            _write_jsonl(second_path, reviewer_two)
            manifest = seal_forward_cases(
                cases_path,
                dataset_id="public-model-forward-calibration",
                version="v1",
                provenance="reverse_bootstrap_calibration",
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        observations_by_key = {}
        failures_by_key = {}
        resumed_from_checkpoints = 0
        state = {
            "schema": MODEL_RUN_STATE_SCHEMA,
            "run_id": run_id,
            "run_fingerprint": run_fingerprint,
            "status": "running",
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "expected_observations": len(expected_positions),
            "completed_observations": 0,
            "resume_count": 0,
            "service_preflights": [service_preflight],
            "pause_history": [],
            "configuration": configuration,
            "input_artifacts": {
                "cases": {"path": str(cases_path), "sha256": _file_digest(cases_path)},
                "reviewer_one": {"path": str(first_path), "sha256": _file_digest(first_path)},
                "reviewer_two": {"path": str(second_path), "sha256": _file_digest(second_path)},
                "manifest": {"path": str(manifest_path), "sha256": _file_digest(manifest_path)},
                **({
                    "study_plan": {
                        "path": str(study_plan_path),
                        "sha256": _file_digest(study_plan_path),
                    },
                } if private_mode else {}),
                **({
                    "resolutions": {
                        "path": str(resolutions_path),
                        "sha256": _file_digest(resolutions_path),
                    },
                } if private_mode and resolutions_path.is_file() else {}),
            },
            "checkpoint_dir": str(checkpoint_dir),
        }
        _write_json_atomic(active_state_path, state)
    proposal_root = root / "proposals" / run_id

    def persist_checkpoint(
        *, position: int, observation: ForwardObservation,
        failure: dict[str, str] | None,
    ) -> None:
        key = (observation.case_id, observation.repetition)
        _write_case_checkpoint(
            checkpoint_dir,
            run_id=run_id,
            run_fingerprint=run_fingerprint,
            position=position,
            observation=observation,
            failure=failure,
        )
        observations_by_key[key] = observation
        if failure is None:
            failures_by_key.pop(key, None)
        else:
            failures_by_key[key] = failure
        state["completed_observations"] = len(observations_by_key)
        state["updatedAt"] = utc_now()
        _write_json_atomic(active_state_path, state)

    consecutive_transport_failures = 0
    for repetition in range(1, repetitions + 1):
        for position, case in enumerate(cases, start=1):
            key = (case.case_id, repetition)
            if key in observations_by_key:
                continue
            prompt, _ = _case_prompt(case)
            reply = adapter.decide(
                user_prompt=prompt,
                decision_validator=lambda decision, selected_case=case: (
                    validate_forward_model_decision_against_catalog(
                        case=selected_case, decision=decision,
                    )
                ),
            )
            if reply.error_stage == "model_transport":
                consecutive_transport_failures += 1
            else:
                consecutive_transport_failures = 0
            destination = proposal_root / case.case_id / f"r{repetition}"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "untrusted-model-output.json").write_text(
                json.dumps({
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "raw_response": reply.raw_content,
                    "raw_digest": reply.raw_digest,
                    "raw_protocol_valid": reply.raw_protocol_valid,
                    "syntax_normalization": {
                        "version": NORMALIZER_VERSION,
                        "paths": list(reply.syntax_normalization_paths),
                        "normalized_digest": reply.normalized_digest,
                    },
                    "validation_errors": list(reply.validation_errors),
                    "parsed_decision": (
                        reply.decision.model_dump(by_alias=True, mode="json")
                        if reply.decision is not None else None
                    ),
                    "authority": "untrusted proposal; never executable",
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if reply.decision is None:
                error = reply.error or "protocol validation failed"
                observation = _error_observation(
                    case=case,
                    repetition=repetition,
                    adapter=adapter,
                    model_digest=model_digest,
                    protocol_digest=protocol_digest,
                    catalog_digest=catalog_digest,
                    reply=reply,
                    semantic=None,
                    error=error,
                )
                failure = {
                    "case_id": case.case_id,
                    "repetition": str(repetition),
                    "stage": reply.error_stage or "model_protocol",
                    "error": error[:500],
                }
                persist_checkpoint(
                    position=position, observation=observation, failure=failure,
                )
                completed = (repetition - 1) * len(cases) + position
                print(json.dumps({
                    "progress": completed,
                    "total": len(cases) * repetitions,
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "state": reply.error_stage or "protocol_error",
                }, ensure_ascii=False), flush=True)
                if (
                    reply.error_stage == "model_transport"
                    and transport_failure_limit
                    and consecutive_transport_failures >= transport_failure_limit
                ):
                    pause = {
                        "code": "MODEL_TRANSPORT_CIRCUIT_OPEN",
                        "pausedAt": utc_now(),
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "consecutive_failures": consecutive_transport_failures,
                        "failure_limit": transport_failure_limit,
                        "completed_observations": len(observations_by_key),
                        "expected_observations": len(expected_positions),
                        "resume_command": "rerun the same command with --resume",
                        "evidencePolicy": (
                            "The triggering failure is checkpointed and will not be "
                            "silently retried or erased on resume."
                        ),
                    }
                    state["status"] = "paused_model_transport"
                    state["pause"] = pause
                    state.setdefault("pause_history", []).append(pause)
                    state["updatedAt"] = utc_now()
                    _write_json_atomic(active_state_path, state)
                    raise ModelRunPausedError(
                        "model run paused after "
                        f"{consecutive_transport_failures} consecutive transport failures; "
                        f"checkpointed {len(observations_by_key)}/{len(expected_positions)} "
                        f"observations in {active_state_path}; restore the model service "
                        "and rerun the same command with --resume"
                    )
                continue
            decision = reply.decision
            if decision.disposition != "proposal":
                observation = record_forward_observation(
                    case_id=case.case_id,
                    repetition=repetition,
                    model=model,
                    model_artifact_digest=model_digest,
                    authoring_protocol_digest=protocol_digest,
                    catalog_snapshot_digest=catalog_digest,
                    disposition=decision.disposition,
                    missing_fields=decision.missing_fields,
                    latency_ms=reply.latency_ms,
                    model_calls=reply.model_calls,
                    repair_attempts=reply.repair_attempts,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    raw_protocol_valid=reply.raw_protocol_valid,
                    syntax_normalization_paths=reply.syntax_normalization_paths,
                    normalized_output_digest=reply.normalized_digest,
                )
                persist_checkpoint(
                    position=position, observation=observation, failure=None,
                )
                completed = (repetition - 1) * len(cases) + position
                print(json.dumps({
                    "progress": completed,
                    "total": len(cases) * repetitions,
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "state": decision.disposition,
                }, ensure_ascii=False), flush=True)
                continue
            assert decision.semantic_contract is not None
            failure = None
            try:
                proposal, assessment_report = materialize_and_assess(
                    case=case,
                    semantic=decision.semantic_contract,
                    destination=destination,
                )
                observation = record_forward_observation(
                    case_id=case.case_id,
                    repetition=repetition,
                    model=model,
                    model_artifact_digest=model_digest,
                    authoring_protocol_digest=protocol_digest,
                    catalog_snapshot_digest=catalog_digest,
                    disposition="proposal",
                    proposal_path=proposal,
                    catalog_id=case.family,
                    latency_ms=reply.latency_ms,
                    model_calls=reply.model_calls,
                    repair_attempts=reply.repair_attempts,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    raw_protocol_valid=reply.raw_protocol_valid,
                    syntax_normalization_paths=reply.syntax_normalization_paths,
                    normalized_output_digest=reply.normalized_digest,
                )
                if assessment_report["status"] == "blocked":
                    finding_codes = sorted({
                        str(item.get("code") or "UNKNOWN")
                        for item in assessment_report.get("findings", [])
                    })
                    failure = {
                        "case_id": case.case_id,
                        "repetition": str(repetition),
                        "stage": "promotion_assessment",
                        "error": "blocking findings: " + ", ".join(finding_codes),
                    }
            except Exception as error:  # deterministic fail-closed evidence
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "error.json").write_text(json.dumps({
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "stage": "runtime_materialization",
                    "error_type": type(error).__name__,
                    "message": str(error)[:2000],
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                observation = _error_observation(
                    case=case,
                    repetition=repetition,
                    adapter=adapter,
                    model_digest=model_digest,
                    protocol_digest=protocol_digest,
                    catalog_digest=catalog_digest,
                    reply=reply,
                    semantic=decision.semantic_contract,
                    error=error,
                )
                failure = {
                    "case_id": case.case_id,
                    "repetition": str(repetition),
                    "stage": "runtime_materialization",
                    "error": f"{type(error).__name__}: {str(error)[:500]}",
                }
            persist_checkpoint(
                position=position, observation=observation, failure=failure,
            )
            completed = (repetition - 1) * len(cases) + position
            print(json.dumps({
                "progress": completed,
                "total": len(cases) * repetitions,
                "case_id": case.case_id,
                "repetition": repetition,
                "state": observation.promotion_status,
            }, ensure_ascii=False), flush=True)
    ordered_keys = [
        (case.case_id, repetition)
        for repetition in range(1, repetitions + 1)
        for case in cases
    ]
    observations = [observations_by_key[key] for key in ordered_keys]
    failures = [failures_by_key[key] for key in ordered_keys if key in failures_by_key]
    _write_jsonl(observations_path, observations)
    evaluator = qualify_forward_files(
        cases_path, manifest_path, first_path, second_path, observations_path,
        study_plan_path=study_plan_path if private_mode else None,
        resolutions_path=(
            resolutions_path if private_mode and resolutions_path.is_file() else None
        ),
    )
    evaluator_path.write_text(
        json.dumps(evaluator, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema": MODEL_RUN_SCHEMA,
        "generatedAt": utc_now(),
        "status": (
            evaluator["status"] if private_mode
            else "public_calibration_only_not_qualified"
        ),
        "qualified": bool(evaluator["qualified"] if private_mode else False),
        "model": model,
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "resumed_from_checkpoints": resumed_from_checkpoints,
        "checkpoint_count": len(observations_by_key),
        "model_service_preflights": state.get("service_preflights", []),
        "model_service_pause_history": state.get("pause_history", []),
        "transport_failure_limit": transport_failure_limit,
        "prompt_packet": prompt_metrics,
        "model_artifact_digest": model_digest,
        "authoring_protocol_digest": protocol_digest,
        "catalog_snapshot_digest": catalog_digest,
        "dataset": evaluator["dataset"],
        "metrics": evaluator["metrics"],
        "slices": evaluator["slices"],
        "gate_checks": evaluator["gate_checks"],
        "qualification_requirements": evaluator["qualification_requirements"],
        "latency": evaluator["latency"],
        "efficiency": evaluator["efficiency"],
        "failed_case_digests": evaluator["failed_case_digests"],
        "failure_counts": dict(sorted(Counter(
            item["stage"] for item in failures
        ).items())),
        "model_protocol_failures": sum(
            item["stage"] == "model_protocol" for item in failures
        ),
        "model_transport_failures": sum(
            item["stage"] == "model_transport" for item in failures
        ),
        "runtime_materialization_failures": sum(
            item["stage"] == "runtime_materialization" for item in failures
        ),
        "failure_details": failures,
        "evaluator_report_digest": evaluator["reportDigest"],
        "artifacts": {
            "cases": str(cases_path),
            "observations": str(observations_path),
            "manifest": str(manifest_path),
            **({"study_plan": str(study_plan_path)} if private_mode else {}),
            **({"resolutions": str(resolutions_path)}
               if private_mode and resolutions_path.is_file() else {}),
            "evaluator_report": str(evaluator_path),
            "proposals": str(proposal_root),
            "active_run": str(active_state_path),
            "checkpoints": str(checkpoint_dir),
        },
        "claimBoundary": (
            (
                "This report qualifies only the sealed private data set, exact model "
                "artifact, authoring protocol, Catalog, evaluator and Runtime versions. "
                "It is not a production success probability."
            ) if private_mode else (
                "This is a real qwen model run over a public reverse-bootstrap calibration "
                "matrix. It measures this fixed artifact/protocol path but is not independent "
                "model qualification or a production success probability."
            )
        ),
    }
    report["reportDigest"] = sha256_json(report)
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state["status"] = "complete"
    state["completed_observations"] = len(observations_by_key)
    state["report_digest"] = report["reportDigest"]
    state["updatedAt"] = utc_now()
    _write_json_atomic(active_state_path, state)
    return report


def rescore_public_model_evaluation(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Re-run only deterministic scoring over an existing immutable model run.

    This updates evaluator-derived fields after evaluator fixes without calling
    the model again.  Model outputs, model latency, protocol/artifact digests,
    failure evidence and original run time remain unchanged.
    """

    root = Path(output_root).expanduser().resolve()
    paths = {
        "cases": root / "cases.jsonl",
        "manifest": root / "manifest.json",
        "reviewer_one": root / "reverse-reviewer-a.jsonl",
        "reviewer_two": root / "reverse-reviewer-b.jsonl",
        "observations": root / "observations.jsonl",
        "evaluator": root / "evaluator-report.json",
        "report": root / "report.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("model rescore artifacts are incomplete: " + ", ".join(missing))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    private_mode = manifest.get("apiVersion") == STUDY_MANIFEST_SCHEMA
    study_plan = root / "study-plan.json"
    resolutions = root / "resolutions.jsonl"
    if private_mode and not study_plan.is_file():
        raise ValueError("private model rescore is missing study-plan.json")
    evaluator = qualify_forward_files(
        paths["cases"], paths["manifest"], paths["reviewer_one"],
        paths["reviewer_two"], paths["observations"],
        study_plan_path=study_plan if private_mode else None,
        resolutions_path=(resolutions if private_mode and resolutions.is_file() else None),
    )
    paths["evaluator"].write_text(
        json.dumps(evaluator, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if report.get("schema") != MODEL_RUN_SCHEMA:
        raise ValueError("model run report schema is invalid")
    for key in (
        "metrics", "slices", "gate_checks", "latency", "efficiency",
        "failed_case_digests", "qualification_requirements", "dataset",
    ):
        report[key] = evaluator[key]
    report["status"] = (
        evaluator["status"] if private_mode
        else "public_calibration_only_not_qualified"
    )
    report["qualified"] = bool(evaluator["qualified"] if private_mode else False)
    report["evaluator_report_digest"] = evaluator["reportDigest"]
    report["rescoredAt"] = utc_now()
    report["rescoreMode"] = (
        "deterministic evaluator only; no model call and no new model evidence"
    )
    report.pop("reportDigest", None)
    report["reportDigest"] = sha256_json(report)
    paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    active_state_path = root / "active-run.json"
    if active_state_path.is_file():
        try:
            state = json.loads(active_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("active model run state is unreadable during rescore") from error
        if not isinstance(state, dict) or state.get("schema") != MODEL_RUN_STATE_SCHEMA:
            raise ValueError("active model run state is invalid during rescore")
        if state.get("run_id") != report.get("run_id"):
            raise ValueError("active model run identity differs from report during rescore")
        state["report_digest"] = report["reportDigest"]
        state["evaluator_report_digest"] = evaluator["reportDigest"]
        state["updatedAt"] = utc_now()
        _write_json_atomic(active_state_path, state)
    return report


def reassess_public_model_evaluation(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Replay stored semantic proposals through the current deterministic Runtime.

    This is a counterfactual gate assessment, not new model evidence: the original
    model output, labels, latency, tokens and historical observations remain intact.
    """

    root = Path(output_root).expanduser().resolve()
    if (root / "study-plan.json").is_file():
        raise ValueError(
            "public Runtime reassessment cannot consume private study material; "
            "use deterministic private rescore with its bound adjudication"
        )
    paths = {
        "cases": root / "cases.jsonl",
        "labels": root / "reverse-reviewer-a.jsonl",
        "observations": root / "observations.jsonl",
        "model_report": root / "report.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(
            "model reassessment artifacts are incomplete: " + ", ".join(missing)
        )

    def load_jsonl(path: Path, model: Any) -> list[Any]:
        values: list[Any] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            if not line.strip():
                continue
            try:
                values.append(model.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(
                    f"invalid reassessment input {path.name}:{line_number}"
                ) from error
        return values

    cases = load_jsonl(paths["cases"], ForwardCase)
    labels = {
        item.case_id: item for item in load_jsonl(paths["labels"], ForwardLabel)
    }
    observations = load_jsonl(paths["observations"], ForwardObservation)
    case_by_id = {item.case_id: item for item in cases}
    model_report = json.loads(paths["model_report"].read_text(encoding="utf-8"))
    run_id = str(model_report.get("run_id") or "")
    if not run_id:
        raise ValueError("model run report does not declare run_id")

    current_catalog_digest = _catalog_snapshot_digest(cases)
    runtime_fingerprint = sha256_json({
        "schema": RUNTIME_REASSESSMENT_SCHEMA,
        "promotion": _file_digest(Path(__file__).with_name("promotion.py")),
        "materializer": _file_digest(Path(__file__)),
        "catalog_snapshot_digest": current_catalog_digest,
        "authoring_protocol_digest": authoring_protocol_digest(),
    })
    destination_root = root / "current-runtime-reassessment"
    details: list[dict[str, Any]] = []
    counts = Counter()
    for observation in observations:
        case = case_by_id.get(observation.case_id)
        label = labels.get(observation.case_id)
        if case is None or label is None:
            raise ValueError(f"reassessment identity is unknown: {observation.case_id}")
        historical_ready = (
            observation.promotion_status == "ready_for_review"
            and observation.blocking_requirements == 0
        )
        score = _score_one(label, observation)
        semantic_exact = bool(score["semantic"])
        current_status = "not_applicable"
        finding_codes: list[str] = []
        error_message: str | None = None
        if observation.semantic_contract is not None:
            counts["proposal_observations"] += 1
            try:
                _, assessment = materialize_and_assess(
                    case=case,
                    semantic=observation.semantic_contract,
                    destination=(
                        destination_root / observation.case_id
                        / f"r{observation.repetition}"
                    ),
                )
                current_status = str(assessment["status"])
                finding_codes = sorted({
                    str(item.get("code") or "UNKNOWN")
                    for item in assessment.get("findings", [])
                    if item.get("severity") == "error"
                })
            except Exception as error:  # deterministic fail-closed replay
                current_status = "materialization_error"
                error_message = f"{type(error).__name__}: {str(error)[:500]}"
            current_ready = current_status == "ready_for_review"
            counts[f"current_{current_status}"] += 1
            if current_ready:
                counts["current_ready"] += 1
            else:
                counts["current_fail_closed"] += 1
            if historical_ready and not current_ready:
                counts["newly_fail_closed"] += 1
            if not historical_ready and current_ready:
                counts["newly_ready"] += 1
            if semantic_exact and historical_ready:
                counts["historical_exact_ready"] += 1
                if current_ready:
                    counts["exact_ready_preserved"] += 1
                else:
                    counts["exact_ready_regressed"] += 1
            if not semantic_exact and historical_ready and not current_ready:
                counts["false_ready_closed"] += 1
            if historical_ready != current_ready or not current_ready:
                details.append({
                    "case_id": observation.case_id,
                    "repetition": observation.repetition,
                    "semantic_exact": semantic_exact,
                    "historical_status": observation.promotion_status,
                    "current_status": current_status,
                    "finding_codes": finding_codes,
                    "error": error_message,
                })
        else:
            counts["not_applicable"] += 1

    report: dict[str, Any] = {
        "schema": RUNTIME_REASSESSMENT_SCHEMA,
        "generatedAt": utc_now(),
        "mode": "stored-semantic-proposal-current-runtime-replay",
        "modelCalls": 0,
        "sourceRun": {
            "runId": run_id,
            "reportDigest": model_report.get("reportDigest"),
            "modelArtifactDigest": model_report.get("model_artifact_digest"),
            "historicalCatalogSnapshotDigest": model_report.get(
                "catalog_snapshot_digest"
            ),
            "historicalAuthoringProtocolDigest": model_report.get(
                "authoring_protocol_digest"
            ),
        },
        "currentRuntime": {
            "fingerprint": runtime_fingerprint,
            "catalogSnapshotDigest": current_catalog_digest,
            "authoringProtocolDigest": authoring_protocol_digest(),
        },
        "observations": len(observations),
        "counts": dict(sorted(counts.items())),
        "changedOrBlockedCases": details,
        "gateConclusion": (
            "regressed"
            if counts["exact_ready_regressed"]
            else "phase_gate_closed_false_ready_without_exact_regression"
            if counts["false_ready_closed"]
            else "stable"
        ),
        "claimBoundary": (
            "No model was called. Stored normalized semantic proposals were replayed "
            "against the current phase-typed Catalog and Runtime. This measures the "
            "deterministic gate delta only; it does not alter historical evidence, "
            "qualify the model, or estimate production success probability."
        ),
    }
    report["reportDigest"] = sha256_json(report)
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
