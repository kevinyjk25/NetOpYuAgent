"""Proposal-only L1 -> L0.5 -> L0 authoring tools for real harness agents.

The LLM supplies semantic translation fields.  This module owns the trusted
catalog lookup, strict parsing, compilation, provenance hashes, and durable
proposal trajectory.  A successful proposal is still not registered and has
no execution authority.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

import yaml

from network_runtime.l0.models import AtomicEffectManifest
from network_runtime.l0.promotion import (
    PromotionError,
    StructuredNaturalLanguageSkill,
    assess_promotion,
    build_l05_spec,
    l05_yaml,
    load_capability_catalog,
    package_promotion,
)
from network_runtime.l0.workbench import export_workbench_html
from skills.skill_format import parse_skill_md


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOGS = {
    "lan-user-access": PROJECT_ROOT
    / "network_runtime/l0/production_trajectories/network.lan.user-access.grant/00-capability-catalog.yaml",
}
_RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


L1_TEMPLATE = """---
name: restore-employee-lan-access
description: Restore one active employee's LAN admission after checking the current state and policy.
allowed-tools: grant_user_access
metadata:
  skill_id: restore_employee_lan_access
  display_name: Restore employee LAN access
  purpose: Restore an active employee's LAN admission safely and verify the result.
  risk_level: high
  requires_hitl: 'true'
  profiles: lan
  tags: lan,access,remediation
  tool_deps: grant_user_access
  returns: Independently verified LAN admission or an explicit recovery state.
---

# Restore employee LAN access

Check the exact user first. Never infer a user id. Preserve the current access
state, require approval for the exact change, grant access once, and verify the
new state independently. If verification fails, restore the previous state and
verify restoration. Never retry an uncertain write blindly.

## Exact Semantic Intent

This marked block is the review anchor that Runtime preserves field-for-field;
do not omit it or ask the model to infer a replacement from prose.

<!-- netopyu:semantic-intents/v1 -->
```yaml
- effectCapability: network.lan.user-access.grant
  kind: grant_network_access
  targetFields:
    - user_id
  desiredState:
    admitted: true
```

## Parameters
- `user_id`: Exact enterprise user identifier.
- `reason`: Required audit reason supplied by the operator.

## Constraints
- Human approval is mandatory.
- Stop when the identity is inactive or unknown.
- A write response is not proof of success.
"""


def _proposal_root() -> Path:
    configured = os.environ.get("NETOPYU_L0_PROPOSALS_DIR")
    root = Path(configured).expanduser() if configured else PROJECT_ROOT / "data/l0-proposals"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root.resolve()


def _existing_artifact_paths(attempt: Path) -> dict[str, str]:
    """Return only authoritative, currently existing proposal artifacts."""

    proposal = attempt / "proposal"
    candidates = {
        "attempt_directory": attempt,
        "captured_l1_skill": next(iter((attempt / "source").glob("*/SKILL.md")), None),
        "working_l05": attempt / "02-L0.5.yaml",
        "working_l0_authoring": attempt / "03-L0-authoring.yaml",
        "agent_trace": attempt / "agent-trace.json",
        "proposal_directory": proposal,
        "proposal_capability_catalog": proposal / "00-capability-catalog.yaml",
        "proposal_l1_skill": proposal / "01-L1-SKILL.md",
        "proposal_l05": proposal / "02-L0.5.yaml",
        "proposal_l0_authoring": proposal / "03-L0-authoring.yaml",
        "proposal_l0_compiled": proposal / "04-L0-compiled.json",
        "validation_report": (
            proposal / "report.json"
            if (proposal / "report.json").is_file()
            else attempt / "report.json"
        ),
        "promotion_trajectory": proposal / "trajectory.json",
        "semantic_review_workbench": attempt / "semantic-review.html",
    }
    return {
        name: str(path.resolve())
        for name, path in candidates.items()
        if path is not None and path.exists()
    }


def _catalog(catalog_id: str) -> tuple[Any, str, Path]:
    try:
        path = CATALOGS[catalog_id]
    except KeyError as error:
        raise PromotionError(f"unsupported authoring catalog {catalog_id!r}") from error
    return load_capability_catalog(path)


def authoring_template() -> dict[str, Any]:
    catalog, digest, path = _catalog("lan-user-access")
    argument = lambda name: "${arguments." + name + "}"
    translation_example = {
        "l0_id": "network.lan.user-access.grant.agent-proposal",
        "profiles": ["lan"],
        "parameters": {
            "user_id": {"type": "string", "required": True, "maxLength": 128},
            "reason": {
                "type": "string", "required": True, "minLength": 1,
                "maxLength": 512,
            },
        },
        "effect_capability": "network.lan.user-access.grant",
        "observation_capability": "network.access.user.get",
        "verification_capability": "netopyu.verifier.lan-access-granted",
        "compensation_capability": "network.lan.user-access.revoke",
        "compensation_verification_capability": "netopyu.rollback-verifier.inverse-tool-v1",
        "effect_request": {"user_id": argument("user_id"), "reason": argument("reason")},
        "intent": {
            "kind": "grant_network_access", "target_fields": ["user_id"],
            "desired_state": {"admitted": True},
        },
        "preflight": {
            "arguments": {"user_id": argument("user_id")},
            "snapshot_fields": ["facts"],
            "predicates": [
                {"field": "facts", "operator": "exists"},
                {"field": "facts.status", "operator": "equals", "expected": "active"},
            ],
        },
        "verification_arguments": {"user_id": argument("user_id")},
        "verification_predicates": [
            {"field": "passed", "operator": "equals", "expected": True},
        ],
        "compensation_arguments": {"user_id": argument("user_id")},
        "compensation_verification_arguments": {"user_id": argument("user_id")},
        "compensation_verification_predicates": [
            {"field": "restored", "operator": "equals", "expected": True},
        ],
        "risk": "high",
        "approval_mode": "single",
        "translation_logic": [
            "user_id is the immutable target; reason is mandatory audit context",
            "use independent observations before and after the effect",
            "verification failure restores and independently verifies the prior state",
        ],
    }
    return {
        "schema": "netopyu.io/agentized-l0-authoring-template/v1",
        "catalog_id": "lan-user-access",
        "catalog_sha256": digest,
        "catalog_path": str(path),
        "l1_skill_template": L1_TEMPLATE,
        "trusted_capabilities": [
            {
                "id": item.id,
                "role": item.role,
                "observationPhases": list(item.observation_phases),
                "tool": item.tool,
                "profiles": list(item.profiles),
                "inputs": sorted(item.inputs),
                "outputs": sorted(item.outputs),
            }
            for item in catalog.capabilities
        ],
        "model_must_translate": [
            "parameters and their strict types/limits",
            "one trusted effect and its exact argument bindings",
            "independent preflight and verification evidence",
            "desired state and machine-checkable predicates",
            "compensation and independent restoration verification",
            "risk, approval mode, and concise translation logic",
            "every safety-critical L1 precondition as an explicit observation predicate",
        ],
        "translation_example": translation_example,
        "runtime_owns": [
            "catalog allowlisting and schema validation",
            "L1/L0.5/L0 scope and risk monotonicity checks",
            "compilation, provenance hashes, and trajectory persistence",
            "manual review gate; no automatic registration or execution",
            "requirement-level L1/L0.5/L0 semantic coverage and weakening gate",
        ],
        "activation": {"automatic": False, "execution_authority": False},
    }


def _valid_attempt_id(attempt_id: str) -> bool:
    suffix = attempt_id.removeprefix("proposal-")
    return (
        attempt_id.startswith("proposal-")
        and len(suffix) == 16
        and all(char in "0123456789abcdef" for char in suffix)
    )


def capture_authoring(arguments: dict[str, Any]) -> dict[str, Any]:
    skill_markdown = str(arguments.get("skill_markdown") or "").strip()
    if not skill_markdown:
        raise PromotionError("skill_markdown is required")
    if len(skill_markdown.encode("utf-8")) > 128_000:
        raise PromotionError("skill_markdown exceeds 128 KiB")
    parsed = parse_skill_md(skill_markdown)
    attempt_id = "proposal-" + secrets.token_hex(8)
    source_dir = _proposal_root() / attempt_id / "source" / parsed.name
    source_dir.mkdir(parents=True, mode=0o700)
    source_path = source_dir / "SKILL.md"
    source_path.write_text(skill_markdown + "\n", encoding="utf-8")
    return {
        "ok": True,
        "status": "l1_captured",
        "draft_id": attempt_id,
        "source_name": parsed.name,
        "source_path": str(source_path),
        "next_tool": "netopyu_l0_authoring_submit",
        "authority": "Source capture only; no proposal, activation, or execution authority.",
    }


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromotionError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise PromotionError(f"{name} must be an array")
    return value


def _build_candidate(
    *, skill_name: str, translation: dict[str, Any], capabilities: dict[str, Any],
) -> AtomicEffectManifest:
    effect_id = str(translation.get("effect_capability") or "")
    observation_id = str(translation.get("observation_capability") or "")
    verification_id = str(translation.get("verification_capability") or "")
    compensation_id = str(translation.get("compensation_capability") or "")
    compensation_verifier_id = str(
        translation.get("compensation_verification_capability") or ""
    )
    required = {
        "effect_capability": effect_id,
        "observation_capability": observation_id,
        "verification_capability": verification_id,
        "compensation_capability": compensation_id,
        "compensation_verification_capability": compensation_verifier_id,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise PromotionError("translation misses capability selections: " + ", ".join(missing))
    for name, capability_id in required.items():
        if capability_id not in capabilities:
            raise PromotionError(f"{name} invents capability {capability_id!r}")
    if capabilities[effect_id].role != "effect":
        raise PromotionError("effect_capability is not an effect")
    if capabilities[compensation_id].role != "compensation":
        raise PromotionError("compensation_capability is not a compensation")
    for capability_id in (observation_id, verification_id, compensation_verifier_id):
        if capabilities[capability_id].role != "observation":
            raise PromotionError(f"{capability_id!r} is not an observation")
    phase_selections = {
        "preflight": observation_id,
        "success_verification": verification_id,
        "compensation_verification": compensation_verifier_id,
    }
    for phase, capability_id in phase_selections.items():
        if not capabilities[capability_id].supports_observation_phase(phase):
            raise PromotionError(
                f"{capability_id!r} is not trusted for observation phase {phase}"
            )

    parameters = _require_mapping(translation.get("parameters"), "translation.parameters")
    intent = _require_mapping(translation.get("intent"), "translation.intent")
    preflight = _require_mapping(translation.get("preflight"), "translation.preflight")
    risk = str(translation.get("risk") or "")
    if risk not in _RISK_RANK:
        raise PromotionError("translation.risk must be low, medium, high, or critical")
    mode = str(translation.get("approval_mode") or "")
    if mode not in {"single", "dual"}:
        raise PromotionError("translation.approval_mode must be single or dual")
    candidate_id = str(translation.get("l0_id") or skill_name.replace("-", "."))
    raw = {
        "apiVersion": "netopyu.io/l0-effect/v2",
        "kind": "AtomicEffect",
        "metadata": {
            "id": candidate_id,
            "version": "0.1.0",
            "owner": "agent-proposal",
            "description": f"Untrusted agent-authored candidate for {skill_name}",
            "labels": {"authoring-entrypoint": "dsh-agent", "activation": "proposal-only"},
        },
        "spec": {
            "template": "netopyu-runtime-v2",
            "profiles": [str(item) for item in _require_list(translation.get("profiles"), "translation.profiles")],
            "effect": {
                "capability": effect_id,
                "tool": capabilities[effect_id].tool,
                "request": _require_mapping(translation.get("effect_request"), "translation.effect_request"),
            },
            "intent": {
                "kind": str(intent.get("kind") or ""),
                "targetFields": _require_list(intent.get("target_fields"), "translation.intent.target_fields"),
                "desiredState": _require_mapping(intent.get("desired_state"), "translation.intent.desired_state"),
            },
            "parameters": parameters,
            "preflight": [{
                "id": "approved-state",
                "capability": observation_id,
                "arguments": _require_mapping(preflight.get("arguments"), "translation.preflight.arguments"),
                "snapshotFields": _require_list(preflight.get("snapshot_fields"), "translation.preflight.snapshot_fields"),
                "predicates": _require_list(preflight.get("predicates"), "translation.preflight.predicates"),
            }],
            "verification": {
                "capability": verification_id,
                "arguments": _require_mapping(translation.get("verification_arguments"), "translation.verification_arguments"),
                "predicates": _require_list(translation.get("verification_predicates"), "translation.verification_predicates"),
            },
            "compensation": {
                "capability": compensation_id,
                "tool": capabilities[compensation_id].tool,
                "arguments": _require_mapping(translation.get("compensation_arguments"), "translation.compensation_arguments"),
                "verification": {
                    "capability": compensation_verifier_id,
                    "arguments": _require_mapping(
                        translation.get("compensation_verification_arguments"),
                        "translation.compensation_verification_arguments",
                    ),
                    "predicates": _require_list(
                        translation.get("compensation_verification_predicates"),
                        "translation.compensation_verification_predicates",
                    ),
                },
            },
            "approval": {"required": True, "risk": risk, "mode": mode},
            "failurePolicy": {
                "beforeSend": "abort",
                "afterSendUnknown": "reconcile_read_only",
                "verificationFailed": "compensate",
                "compensationFailed": "manual_intervention",
            },
        },
    }
    return AtomicEffectManifest.model_validate(raw)


def submit_authoring(arguments: dict[str, Any]) -> dict[str, Any]:
    draft_id = str(arguments.get("draft_id") or "").strip()
    skill_markdown = str(arguments.get("skill_markdown") or "").strip()
    if draft_id:
        if not _valid_attempt_id(draft_id):
            raise PromotionError("invalid draft_id")
        matches = list((_proposal_root() / draft_id / "source").glob("*/SKILL.md"))
        if len(matches) != 1:
            raise PromotionError(f"captured L1 source not found: {draft_id}")
        source_path = matches[0]
        skill_markdown = source_path.read_text(encoding="utf-8").strip()
        attempt_id = draft_id
        attempt = _proposal_root() / attempt_id
    else:
        if not skill_markdown:
            raise PromotionError("draft_id is required; capture the L1 source first")
        if len(skill_markdown.encode("utf-8")) > 128_000:
            raise PromotionError("skill_markdown exceeds 128 KiB")
        attempt_id = "proposal-" + secrets.token_hex(8)
        attempt = _proposal_root() / attempt_id
    supplied_translation = arguments.get("translation")
    if isinstance(supplied_translation, dict):
        translation = supplied_translation
    else:
        # Small local models are materially more reliable with a flat Tool
        # schema. Keep accepting the nested protocol for API compatibility.
        translation = {
            key: value for key, value in arguments.items()
            if key not in {"draft_id", "skill_markdown", "catalog_id"}
        }
        if not translation:
            raise PromotionError("translation fields are required")
    logic = _require_list(translation.get("translation_logic"), "translation.translation_logic")
    if not logic or any(not isinstance(item, str) or not item.strip() for item in logic):
        raise PromotionError("translation_logic must contain concise non-empty strings")

    parsed = parse_skill_md(skill_markdown)
    catalog_id = str(arguments.get("catalog_id") or "lan-user-access")
    catalog, _, catalog_path = _catalog(catalog_id)
    capabilities = catalog.by_id()
    candidate = _build_candidate(
        skill_name=parsed.name, translation=translation, capabilities=capabilities,
    )
    if not draft_id:
        source_dir = attempt / "source" / parsed.name
        source_dir.mkdir(parents=True, mode=0o700)
        source_path = source_dir / "SKILL.md"
        source_path.write_text(skill_markdown + "\n", encoding="utf-8")

    base_l05 = build_l05_spec(
        skill_path=source_path, capability_catalog_path=catalog_path,
    )
    l05_raw = base_l05.model_dump(by_alias=True, mode="json")
    chosen_observations = [
        str(translation["observation_capability"]),
        str(translation["verification_capability"]),
        str(translation["compensation_verification_capability"]),
    ]
    l05_raw["capabilities"] = {
        "effects": [str(translation["effect_capability"])],
        "observations": list(dict.fromkeys(chosen_observations)),
        "compensations": [str(translation["compensation_capability"])],
        "preflightObservations": [
            str(translation["observation_capability"]),
        ],
        "successVerificationObservations": [
            str(translation["verification_capability"]),
        ],
        "compensationVerificationObservations": [
            str(translation["compensation_verification_capability"]),
        ],
    }
    for step in l05_raw["workflow"]:
        if step["phase"] == "effect":
            step["capabilityOptions"] = l05_raw["capabilities"]["effects"]
        elif step["phase"] == "preflight":
            step["capabilityOptions"] = l05_raw["capabilities"][
                "preflightObservations"
            ]
        elif step["phase"] == "verification":
            step["capabilityOptions"] = l05_raw["capabilities"][
                "successVerificationObservations"
            ]
        elif step["phase"] == "compensation":
            step["capabilityOptions"] = l05_raw["capabilities"]["compensations"]
        elif step["phase"] == "compensation_verification":
            step["capabilityOptions"] = l05_raw["capabilities"][
                "compensationVerificationObservations"
            ]
    l05_raw["safety"]["risk"] = str(translation["risk"])
    l05_raw["safety"]["approvalRequired"] = True
    l05_raw["unresolvedQuestions"] = []
    l05 = StructuredNaturalLanguageSkill.model_validate(l05_raw)
    l05_path = attempt / "02-L0.5.yaml"
    candidate_path = attempt / "03-L0-authoring.yaml"
    l05_path.write_text(l05_yaml(l05), encoding="utf-8")
    candidate_path.write_text(yaml.safe_dump(
        candidate.model_dump(by_alias=True, mode="json"),
        sort_keys=False, allow_unicode=True,
    ), encoding="utf-8")

    assessment = assess_promotion(
        skill_path=source_path,
        candidate_path=candidate_path,
        capability_catalog_path=catalog_path,
        l05_path=l05_path,
    )
    trace_path = attempt / "agent-trace.json"
    artifact_paths = {
        "attempt_directory": str(attempt),
        "captured_l1_skill": str(source_path),
        "working_l05": str(l05_path),
        "working_l0_authoring": str(candidate_path),
        "agent_trace": str(trace_path),
    }
    trace = {
        "schema": "netopyu.io/agentized-l0-authoring-trace/v1",
        "attempt_id": attempt_id,
        "model_stage": {
            "provided_fields": sorted(translation),
            "translation_logic": logic,
            "candidate_is_untrusted": True,
        },
        "runtime_stage": {
            "catalog_id": catalog_id,
            "catalog_provider": catalog.provider,
            "validation_status": assessment.report["status"],
            "findings": assessment.report["findings"],
            "semantic_coverage": assessment.report["semanticCoverage"],
            "compiled_hash": assessment.report["candidate"]["compiledHash"],
        },
        "activation": {"automatic": False, "execution_authority": False},
        "artifact_paths": artifact_paths,
    }
    if assessment.report["status"] != "ready_for_review":
        blocked_report_path = attempt / "report.json"
        blocked_report_path.write_text(
            json.dumps(assessment.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_paths["validation_report"] = str(blocked_report_path)
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "ok": False,
            "status": "blocked",
            "attempt_id": attempt_id,
            "proposal_directory": str(attempt),
            "artifact_paths": artifact_paths,
            "trajectory": trace,
            "semantic_coverage": assessment.report["semanticCoverage"],
            "errors": [
                item for item in assessment.report["findings"]
                if item["severity"] == "error"
            ],
            "auto_activated": False,
        }

    packaged = package_promotion(
        skill_path=source_path,
        candidate_path=candidate_path,
        capability_catalog_path=catalog_path,
        output_directory=attempt / "proposal",
        l05_path=l05_path,
    )
    proposal_directory = attempt / "proposal"
    semantic_review = attempt / "semantic-review.html"
    export_workbench_html(proposal_directory, semantic_review)
    artifact_paths.update({
        "proposal_directory": str(proposal_directory),
        "proposal_capability_catalog": str(proposal_directory / "00-capability-catalog.yaml"),
        "proposal_l1_skill": str(proposal_directory / "01-L1-SKILL.md"),
        "proposal_l05": str(proposal_directory / "02-L0.5.yaml"),
        "proposal_l0_authoring": str(proposal_directory / "03-L0-authoring.yaml"),
        "proposal_l0_compiled": str(proposal_directory / "04-L0-compiled.json"),
        "validation_report": str(proposal_directory / "report.json"),
        "promotion_trajectory": str(proposal_directory / "trajectory.json"),
        "semantic_review_workbench": str(semantic_review),
    })
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **packaged,
        "attempt_id": attempt_id,
        "proposal_directory": str(proposal_directory),
        "artifact_paths": artifact_paths,
        "trajectory": trace,
        "semantic_coverage": assessment.report["semanticCoverage"],
        "next_action": "Manual review only; this proposal is not registered or executable.",
    }


def authoring_trace(attempt_id: str) -> dict[str, Any]:
    if not _valid_attempt_id(attempt_id):
        raise PromotionError("invalid attempt_id")
    path = _proposal_root() / attempt_id / "agent-trace.json"
    if not path.is_file():
        raise PromotionError(f"authoring attempt not found: {attempt_id}")
    trace = json.loads(path.read_text(encoding="utf-8"))
    # Old proposals remain readable after the structured path contract was
    # introduced. Derivation is Runtime-owned and existence-checked; the LLM
    # never guesses filenames from a directory.
    trace["artifact_paths"] = _existing_artifact_paths(path.parent)
    trace["path_reporting_policy"] = {
        "artifact_paths_authoritative": True,
        "inferred_paths_forbidden": True,
    }
    return trace
