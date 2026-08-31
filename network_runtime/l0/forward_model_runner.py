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
from typing import Any, Iterable

import httpx
import yaml
from pydantic import ValidationError

from network_runtime.contracts import sha256_json, utc_now
from skills.skill_format import parse_skill_md

from .forward_qualification import (
    MODEL_DECISION_SCHEMA,
    ForwardCase,
    ForwardLabel,
    ForwardModelDecision,
    ForwardObservation,
    SemanticContract,
    TRAJECTORY_ROOT,
    build_public_calibration,
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
PROTOCOL_VERSION = "netopyu-forward-authoring-9b/v3"
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
  (normally an `exists` predicate whose expected value is null); success verification uses the dedicated verifier and
  requires both success_verification scope and its `passed` output to equal true;
  compensation verification requires compensation_verification scope and the rollback
  verifier and requires its `restored` output to equal true. Every predicate list is
  mandatory and must reference an output actually declared by that selected capability.
- If the L1 explicitly states there is no safe automatic inverse, set compensation and
  compensation-verification capabilities to null, their predicate list to empty,
  requires_compensation=false, and verificationFailed=manual_intervention. Otherwise
  select both compensation capabilities and use verificationFailed=compensate.

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

    def artifact_digest(self) -> str:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models") or []
        match = next((item for item in models if item.get("name") == self.model), None)
        if match is None:
            raise ValueError(f"Ollama model is not installed: {self.model}")
        digest = str(match.get("digest") or "")
        if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
            return f"sha256:{digest}"
        return sha256_json({"model": self.model, "metadata": match})

    def decide(self, *, user_prompt: str) -> ModelReply:
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
                calls += 1
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
                                "do not add prose. If you already constructed a complete "
                                "semantic_contract, preserve it and use disposition=proposal; "
                                "do not delete it merely to make reject/clarify validate. "
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
    for item in catalog.capabilities:
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
    packet = {
        "catalog_id": catalog_id,
        "catalog_sha256": digest,
        "provider": catalog.provider,
        "version": catalog.version,
        "capabilities": capabilities,
    }
    return json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True), digest


def _case_prompt(case: ForwardCase) -> tuple[str, str]:
    trajectory = TRAJECTORY_ROOT / case.family
    catalog_text, catalog_digest = _catalog_prompt(
        trajectory / "00-capability-catalog.yaml", catalog_id=case.family,
    )
    return (
        "FORWARD CASE\n"
        f"case_id: {case.case_id}\n"
        f"family: {case.family}\n"
        f"profile: {case.profile}\n"
        f"language: {case.language}\n"
        f"challenge: {case.challenge}\n\n"
        "TRUSTED CAPABILITY CATALOG\n"
        f"{catalog_text}\n\n"
        "UNTRUSTED L1 SKILL REQUEST\n"
        f"{case.prompt}",
        catalog_digest,
    )


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
            "catalog snapshot, repetitions, or repair policy changed"
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
    }
    for name, filename in filenames.items():
        item = artifacts.get(name)
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
    repetitions: int = 1,
    timeout_seconds: float = 180.0,
    repair_limit: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    if limit is not None and not 1 <= limit <= 210:
        raise ValueError("limit must be between 1 and 210")
    if not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be between 1 and 10")
    all_cases, all_labels = build_public_calibration()
    cases = _balanced_cases(all_cases, limit)
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
    evaluator_path = root / "evaluator-report.json"
    active_state_path = root / "active-run.json"
    adapter = OllamaForwardAdapter(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        repair_limit=repair_limit,
    )
    model_digest = adapter.artifact_digest()
    protocol_digest = authoring_protocol_digest()
    catalog_digest = _catalog_snapshot_digest(cases)
    configuration: dict[str, Any] = {
        "model": model,
        "model_artifact_digest": model_digest,
        "authoring_protocol_digest": protocol_digest,
        "catalog_snapshot_digest": catalog_digest,
        "case_payload_digest": sha256_json([
            item.model_dump(by_alias=True, mode="json") for item in cases
        ]),
        "reviewer_one_payload_digest": sha256_json([
            item.model_dump(by_alias=True, mode="json") for item in reviewer_one
        ]),
        "reviewer_two_payload_digest": sha256_json([
            item.model_dump(by_alias=True, mode="json") for item in reviewer_two
        ]),
        "limit": limit,
        "repetitions": repetitions,
        "repair_limit": repair_limit,
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
        state["resume_count"] = int(state.get("resume_count") or 0) + 1
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
            "configuration": configuration,
            "input_artifacts": {
                "cases": {"path": str(cases_path), "sha256": _file_digest(cases_path)},
                "reviewer_one": {"path": str(first_path), "sha256": _file_digest(first_path)},
                "reviewer_two": {"path": str(second_path), "sha256": _file_digest(second_path)},
                "manifest": {"path": str(manifest_path), "sha256": _file_digest(manifest_path)},
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

    for repetition in range(1, repetitions + 1):
        for position, case in enumerate(cases, start=1):
            key = (case.case_id, repetition)
            if key in observations_by_key:
                continue
            prompt, _ = _case_prompt(case)
            reply = adapter.decide(user_prompt=prompt)
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
                    "stage": "model_protocol",
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
                    "state": "protocol_error",
                }, ensure_ascii=False), flush=True)
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
    )
    evaluator_path.write_text(
        json.dumps(evaluator, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema": MODEL_RUN_SCHEMA,
        "generatedAt": utc_now(),
        "status": "public_calibration_only_not_qualified",
        "qualified": False,
        "model": model,
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "resumed_from_checkpoints": resumed_from_checkpoints,
        "checkpoint_count": len(observations_by_key),
        "model_artifact_digest": model_digest,
        "authoring_protocol_digest": protocol_digest,
        "catalog_snapshot_digest": catalog_digest,
        "dataset": evaluator["dataset"],
        "metrics": evaluator["metrics"],
        "slices": evaluator["slices"],
        "gate_checks": evaluator["gate_checks"],
        "latency": evaluator["latency"],
        "efficiency": evaluator["efficiency"],
        "failed_case_digests": evaluator["failed_case_digests"],
        "failure_counts": dict(sorted(Counter(
            item["stage"] for item in failures
        ).items())),
        "model_protocol_failures": sum(
            item["stage"] == "model_protocol" for item in failures
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
            "evaluator_report": str(evaluator_path),
            "proposals": str(proposal_root),
            "active_run": str(active_state_path),
            "checkpoints": str(checkpoint_dir),
        },
        "claimBoundary": (
            "This is a real qwen model run over a public reverse-bootstrap calibration "
            "matrix. It measures this fixed artifact/protocol path but is not independent "
            "model qualification or a production success probability."
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
    evaluator = qualify_forward_files(
        paths["cases"], paths["manifest"], paths["reviewer_one"],
        paths["reviewer_two"], paths["observations"],
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
        "failed_case_digests",
    ):
        report[key] = evaluator[key]
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
