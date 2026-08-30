"""Run one immutable local model through the forward L1 -> L0.5 -> L0 path.

The model may emit only a strict, non-executable semantic proposal.  Trusted
catalog lookup, L0.5/L0 materialization, schema validation, Promotion checks,
and observation recording remain deterministic Runtime responsibilities.
Public reverse-bootstrap cases exercise this path but never qualify a model.
"""

from __future__ import annotations

import hashlib
import json
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
    ForwardModelDecision,
    ForwardObservation,
    SemanticContract,
    TRAJECTORY_ROOT,
    build_public_calibration,
    qualify_forward_files,
    record_forward_observation,
    seal_forward_cases,
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
PROTOCOL_VERSION = "netopyu-forward-authoring-9b/v2"
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
- Observation selection is phase-specific even though catalog role=observation is shared:
  preflight uses the current-state/read/get capability and snapshots its state output
  (normally an `exists` predicate whose expected value is null); success verification uses the dedicated verifier and
  requires its `passed` output to equal true; compensation verification uses the rollback
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
) -> Any:
    if capability_id is None or capability_id not in capabilities:
        raise ValueError(f"unknown {role} capability: {capability_id}")
    capability = capabilities[capability_id]
    if capability.role != role:
        raise ValueError(
            f"capability {capability_id} has role {capability.role}, expected {role}"
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
    )
    verification = _selected_capability(
        capabilities, semantic.verification_capability, role="observation",
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
    }
    phase_options = {
        "preflight": [semantic.preflight_capability],
        "effect": [semantic.effect_capability],
        "verification": [semantic.verification_capability],
        "compensation": (
            [semantic.compensation_capability]
            if semantic.compensation_capability else []
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
) -> dict[str, Any]:
    if limit is not None and not 1 <= limit <= 210:
        raise ValueError("limit must be between 1 and 210")
    if not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be between 1 and 10")
    all_cases, all_labels = build_public_calibration()
    cases = _balanced_cases(all_cases, limit)
    labels_by_id = {item.case_id: item for item in all_labels}
    labels = [labels_by_id[item.case_id] for item in cases]
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases_path = root / "cases.jsonl"
    first_path = root / "reverse-reviewer-a.jsonl"
    second_path = root / "reverse-reviewer-b.jsonl"
    observations_path = root / "observations.jsonl"
    manifest_path = root / "manifest.json"
    evaluator_path = root / "evaluator-report.json"
    _write_jsonl(cases_path, cases)
    _write_jsonl(first_path, [
        item.model_copy(update={"reviewer_id": "public-reviewer-a"})
        for item in labels
    ])
    _write_jsonl(second_path, [
        item.model_copy(update={"reviewer_id": "public-reviewer-b"})
        for item in labels
    ])
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
    adapter = OllamaForwardAdapter(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        repair_limit=repair_limit,
    )
    model_digest = adapter.artifact_digest()
    protocol_digest = authoring_protocol_digest()
    catalog_digest = _catalog_snapshot_digest(cases)
    observations: list[ForwardObservation] = []
    failures: list[dict[str, str]] = []
    proposal_root = root / "proposals"
    for repetition in range(1, repetitions + 1):
        for position, case in enumerate(cases, start=1):
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
                observations.append(_error_observation(
                    case=case,
                    repetition=repetition,
                    adapter=adapter,
                    model_digest=model_digest,
                    protocol_digest=protocol_digest,
                    catalog_digest=catalog_digest,
                    reply=reply,
                    semantic=None,
                    error=error,
                ))
                failures.append({
                    "case_id": case.case_id,
                    "repetition": str(repetition),
                    "stage": "model_protocol",
                    "error": error[:500],
                })
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
                observations.append(record_forward_observation(
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
                ))
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
            try:
                proposal, assessment_report = materialize_and_assess(
                    case=case,
                    semantic=decision.semantic_contract,
                    destination=destination,
                )
                observations.append(record_forward_observation(
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
                ))
                if assessment_report["status"] == "blocked":
                    finding_codes = sorted({
                        str(item.get("code") or "UNKNOWN")
                        for item in assessment_report.get("findings", [])
                    })
                    failures.append({
                        "case_id": case.case_id,
                        "repetition": str(repetition),
                        "stage": "promotion_assessment",
                        "error": "blocking findings: " + ", ".join(finding_codes),
                    })
            except Exception as error:  # deterministic fail-closed evidence
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "error.json").write_text(json.dumps({
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "stage": "runtime_materialization",
                    "error_type": type(error).__name__,
                    "message": str(error)[:2000],
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                observations.append(_error_observation(
                    case=case,
                    repetition=repetition,
                    adapter=adapter,
                    model_digest=model_digest,
                    protocol_digest=protocol_digest,
                    catalog_digest=catalog_digest,
                    reply=reply,
                    semantic=decision.semantic_contract,
                    error=error,
                ))
                failures.append({
                    "case_id": case.case_id,
                    "repetition": str(repetition),
                    "stage": "runtime_materialization",
                    "error": f"{type(error).__name__}: {str(error)[:500]}",
                })
            completed = (repetition - 1) * len(cases) + position
            print(json.dumps({
                "progress": completed,
                "total": len(cases) * repetitions,
                "case_id": case.case_id,
                "repetition": repetition,
                "state": observations[-1].promotion_status,
            }, ensure_ascii=False), flush=True)
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
        "model_artifact_digest": model_digest,
        "authoring_protocol_digest": protocol_digest,
        "catalog_snapshot_digest": catalog_digest,
        "dataset": evaluator["dataset"],
        "metrics": evaluator["metrics"],
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
        "metrics", "gate_checks", "latency", "efficiency", "failed_case_digests",
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
    return report
