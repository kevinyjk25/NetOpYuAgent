"""Read-only P2.0 review projection and offline L0.5 workbench.

The workbench validates an immutable Promotion package and renders a local,
self-contained HTML review artifact.  The browser may export an edited L0.5
draft, but neither Python nor JavaScript can approve, activate, register, or
execute a contract.  Every exported draft remains untrusted input to the
existing deterministic Promotion assessment.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from network_runtime.contracts import sha256_json
from skills.skill_format import parse_skill_md, to_flat_dict

from .models import CompiledAtomicEffect, CompiledCompositeEffect
from .promotion import (
    CAPABILITY_API_VERSION,
    PROMOTION_SCHEMA,
    TRAJECTORY_SCHEMA,
    CapabilityCatalogManifest,
    PromotionError,
    StructuredNaturalLanguageSkill,
)


WORKBENCH_SCHEMA = "netopyu.io/l0-promotion-workbench/v1"
_MAX_DOCUMENT_BYTES = 2_000_000
_PACKAGE_FILES = {
    "00-capability-catalog.yaml",
    "01-L1-SKILL.md",
    "02-L0.5.yaml",
    "03-L0-authoring.yaml",
    "04-L0-compiled.json",
    "trajectory.json",
}
_STAGE_FILES = (
    ("L1", "01-L1-SKILL.md"),
    ("L0.5", "02-L0.5.yaml"),
    ("L0-authoring", "03-L0-authoring.yaml"),
    ("L0-compiled", "04-L0-compiled.json"),
)
_REVIEW_FIELDS = {
    "schema", "proposalHash", "decision", "reviewer", "reason", "reviewedAt",
    "activatesRuntime", "grantsExecutionAuthority", "reviewHash",
}


def _bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"workbench package file is missing or unsafe: {path.name}")
    raw = path.read_bytes()
    if len(raw) > _MAX_DOCUMENT_BYTES:
        raise PromotionError(f"workbench package file exceeds 2 MB: {path.name}")
    return raw


def _file_digest(path: Path) -> str:
    return f"sha256:{sha256(_bytes(path)).hexdigest()}"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"workbench JSON is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"workbench JSON must be an object: {path.name}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(_bytes(path))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PromotionError(f"workbench YAML is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"workbench YAML must be an object: {path.name}")
    return value


def _validate_report(root: Path) -> dict[str, Any]:
    report = _json(root / "report.json")
    if report.get("schema") != PROMOTION_SCHEMA or report.get("status") != "ready_for_review":
        raise PromotionError("workbench accepts only a ready_for_review Promotion v2 package")
    if report.get("executionEligible") is not False or report.get("autoActivated") is not False:
        raise PromotionError("workbench package attempts to carry execution authority")
    stored_hash = report.get("proposalHash")
    if stored_hash != sha256_json({
        key: value for key, value in report.items() if key != "proposalHash"
    }):
        raise PromotionError("workbench proposal report integrity check failed")
    package_files = report.get("packageFiles")
    if not isinstance(package_files, dict) or set(package_files) != _PACKAGE_FILES:
        raise PromotionError("workbench package file manifest is incomplete")
    for name, expected in package_files.items():
        if not isinstance(expected, str) or _file_digest(root / name) != expected:
            raise PromotionError(f"workbench package integrity check failed: {name}")
    return report


def _validate_trajectory(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    trajectory = _json(root / "trajectory.json")
    if (
        trajectory.get("schema") != TRAJECTORY_SCHEMA
        or trajectory.get("executionEligible") is not False
        or trajectory.get("autoActivated") is not False
    ):
        raise PromotionError("workbench trajectory authority/schema is invalid")
    stored_hash = trajectory.get("trajectoryHash")
    if (
        stored_hash != report.get("trajectoryHash")
        or stored_hash != sha256_json({
            key: value for key, value in trajectory.items() if key != "trajectoryHash"
        })
    ):
        raise PromotionError("workbench trajectory digest is invalid")
    stages = trajectory.get("stages")
    if not isinstance(stages, list) or len(stages) != len(_STAGE_FILES):
        raise PromotionError("workbench trajectory stage coverage is invalid")
    previous: str | None = None
    for stage, (expected_stage, expected_file) in zip(stages, _STAGE_FILES):
        if not isinstance(stage, dict):
            raise PromotionError("workbench trajectory stage must be an object")
        if (
            stage.get("stage") != expected_stage
            or stage.get("file") != expected_file
            or stage.get("previousSha256") != previous
            or stage.get("sha256") != _file_digest(root / expected_file)
        ):
            raise PromotionError(f"workbench trajectory stage is invalid: {expected_stage}")
        previous = stage["sha256"]
    capability = trajectory.get("capabilityCatalog")
    if not isinstance(capability, dict) or capability != {
        "file": "00-capability-catalog.yaml",
        "sha256": _file_digest(root / "00-capability-catalog.yaml"),
    }:
        raise PromotionError("workbench capability Catalog binding is invalid")
    return trajectory


def _review(root: Path, proposal_hash: str) -> tuple[str, dict[str, Any]]:
    path = root / "review.json"
    if not path.exists():
        return "ready_for_review", {
            "present": False,
            "decision": None,
            "review_hash": None,
            "reviewer_digest": None,
            "reason_digest": None,
        }
    review = _json(path)
    if set(review) != _REVIEW_FIELDS:
        raise PromotionError("workbench review contract fields are invalid")
    if (
        review.get("schema") != "netopyu.io/l0-promotion-review/v2"
        or review.get("proposalHash") != proposal_hash
        or review.get("decision") not in {"approve", "reject"}
        or review.get("activatesRuntime") is not False
        or review.get("grantsExecutionAuthority") is not False
        or review.get("reviewHash") != sha256_json({
            key: value for key, value in review.items() if key != "reviewHash"
        })
    ):
        raise PromotionError("workbench review integrity/authority check failed")
    reviewer = str(review.get("reviewer") or "")
    reason = str(review.get("reason") or "")
    if not reviewer:
        raise PromotionError("workbench review is missing reviewer identity")
    decision = str(review["decision"])
    return ("approved_not_active" if decision == "approve" else "rejected_not_active"), {
        "present": True,
        "decision": decision,
        "review_hash": review["reviewHash"],
        "reviewer_digest": sha256_json({"reviewer": reviewer}),
        "reason_digest": sha256_json({"reason": reason}),
    }


def _compiled(root: Path, report: dict[str, Any]) -> CompiledAtomicEffect | CompiledCompositeEffect:
    raw = _json(root / "04-L0-compiled.json")
    model = {
        "CompiledAtomicEffect": CompiledAtomicEffect,
        "CompiledCompositeEffect": CompiledCompositeEffect,
    }.get(raw.get("kind"))
    if model is None:
        raise PromotionError("workbench compiled contract kind is unsupported")
    try:
        contract = model.model_validate(raw)
    except ValidationError as error:
        raise PromotionError("workbench compiled contract is invalid") from error
    hash_field = "contractHash" if isinstance(contract, CompiledAtomicEffect) else "definitionHash"
    calculated = sha256_json({key: value for key, value in raw.items() if key != hash_field})
    stored = raw[hash_field]
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise PromotionError("workbench candidate report is invalid")
    if (
        stored != calculated
        or stored != candidate.get("compiledHash")
        or contract.metadata.id != candidate.get("id")
        or contract.metadata.version != candidate.get("version")
        or contract.kind != f"Compiled{candidate.get('kind')}"
    ):
        raise PromotionError("workbench compiled contract digest is invalid")
    return contract


def _source(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        text = _bytes(root / "01-L1-SKILL.md").decode("utf-8")
        parsed = parse_skill_md(text)
        skill_id, definition = to_flat_dict(parsed, skill_id_hint=parsed.name.replace("-", "_"))
    except (UnicodeDecodeError, TypeError, ValueError) as error:
        raise PromotionError("workbench L1 Skill is invalid") from error
    return definition, {
        "name": parsed.name,
        "skill_id": skill_id,
        "description": parsed.frontmatter["description"],
        "sha256": _file_digest(root / "01-L1-SKILL.md"),
        "profiles": list(definition.get("profiles") or ("default",)),
        "parameters": dict(definition.get("parameters") or {}),
        "declared_tools": sorted({
            *(str(item) for item in definition.get("allowed_tools", ())),
            *(str(item) for item in definition.get("tool_deps", ())),
        }),
        "risk": str(definition.get("risk_level") or "low"),
        "approval_required": bool(definition.get("requires_hitl")),
    }


def _l05(root: Path, source_digest: str, catalog_digest: str) -> StructuredNaturalLanguageSkill:
    try:
        value = StructuredNaturalLanguageSkill.model_validate(_yaml(root / "02-L0.5.yaml"))
    except ValidationError as error:
        raise PromotionError("workbench L0.5 document is invalid") from error
    if (
        value.source_skill_sha256 != source_digest
        or value.previous_stage_sha256 != source_digest
        or value.capability_catalog_sha256 != catalog_digest
    ):
        raise PromotionError("workbench L0.5 stage binding is invalid")
    return value


def _catalog(root: Path) -> CapabilityCatalogManifest:
    try:
        value = CapabilityCatalogManifest.model_validate(_yaml(root / "00-capability-catalog.yaml"))
    except ValidationError as error:
        raise PromotionError("workbench Capability Catalog is invalid") from error
    if value.api_version != CAPABILITY_API_VERSION:
        raise PromotionError("workbench Capability Catalog version is unsupported")
    return value


def _contract_projection(
    contract: CompiledAtomicEffect | CompiledCompositeEffect,
) -> dict[str, Any]:
    if isinstance(contract, CompiledAtomicEffect):
        steps = [
            "validate", "preflight", "approve", "revalidate", "execute",
            "verify", *(["compensate"] if contract.spec.compensation else []), "audit",
        ]
        observations = {
            *(item.capability for item in contract.spec.preflight),
            contract.spec.verification.capability,
        }
        if contract.spec.compensation is not None:
            observations.add(contract.spec.compensation.verification.capability)
        return {
            "id": contract.metadata.id,
            "version": contract.metadata.version,
            "kind": contract.kind,
            "hash": contract.contract_hash,
            "profiles": list(contract.spec.profiles),
            "parameters": sorted(contract.spec.parameters),
            "effect_capabilities": [contract.spec.effect.capability],
            "observation_capabilities": sorted(observations),
            "compensation_capabilities": (
                [contract.spec.compensation.capability]
                if contract.spec.compensation else []
            ),
            "approval": contract.spec.approval.model_dump(mode="json"),
            "failure_policy": contract.spec.failure_policy.model_dump(
                by_alias=True, mode="json",
            ),
            "steps": steps,
            "step_edges": [
                {"from": left, "to": right}
                for left, right in zip(steps, steps[1:])
            ],
        }
    step_ids = [item.id for item in contract.steps]
    return {
        "id": contract.metadata.id,
        "version": contract.metadata.version,
        "kind": contract.kind,
        "hash": contract.definition_hash,
        "profiles": [],
        "parameters": sorted(contract.inputs),
        "effect_capabilities": sorted({item.capability for item in contract.steps}),
        "observation_capabilities": sorted({
            item.capability
            for checkpoint in contract.checkpoints
            for item in checkpoint.observations
        }),
        "compensation_capabilities": sorted({
            item.compensation_capability
            for item in contract.steps if item.compensation_capability
        }),
        "approval": contract.approval.model_dump(mode="json"),
        "failure_policy": {"compensationOrder": contract.compensation_order},
        "steps": step_ids,
        "step_edges": [
            {"from": dependency, "to": item.id}
            for item in contract.steps
            for dependency in item.depends_on
        ],
    }


def _semantic_diff(
    source: dict[str, Any],
    l05: StructuredNaturalLanguageSkill,
    contract: dict[str, Any],
) -> dict[str, Any]:
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    source_parameters = set(source["parameters"])
    l05_parameters = set(l05.parameters)
    l0_parameters = set(contract["parameters"])
    source_profiles = set(source["profiles"])
    l05_profiles = set(l05.profiles)
    l0_profiles = set(contract["profiles"])
    capabilities = {
        "effects": {
            "declared": list(l05.capabilities.effects),
            "compiled": list(contract["effect_capabilities"]),
        },
        "observations": {
            "declared": list(l05.capabilities.observations),
            "compiled": list(contract["observation_capabilities"]),
        },
        "compensations": {
            "declared": list(l05.capabilities.compensations),
            "compiled": list(contract["compensation_capabilities"]),
        },
    }
    return {
        "L1_to_L0.5": {
            "parameter_names_exact": source_parameters == l05_parameters,
            "profiles_not_widened": l05_profiles.issubset(source_profiles),
            "risk_not_weakened": rank[l05.safety.risk] >= rank[source["risk"]],
            "approval_not_weakened": (
                not source["approval_required"] or l05.safety.approval_required
            ),
        },
        "L0.5_to_L0": {
            "required_parameter_names_preserved": l0_parameters.issubset(l05_parameters),
            "profiles_not_widened": not l0_profiles or l0_profiles.issubset(l05_profiles),
            "approval_required": bool(contract["approval"].get("required")),
            "capabilities": capabilities,
        },
        "parameter_sets": {
            "L1": sorted(source_parameters),
            "L0.5": sorted(l05_parameters),
            "L0": sorted(l0_parameters),
        },
    }


def inspect_workbench(proposal_directory: str | Path) -> dict[str, Any]:
    """Validate and project one package without mutating it."""
    supplied_root = Path(proposal_directory).expanduser()
    if supplied_root.is_symlink():
        raise PromotionError("workbench proposal directory is missing or unsafe")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise PromotionError("workbench proposal directory is missing or unsafe")
    report = _validate_report(root)
    trajectory = _validate_trajectory(root, report)
    state, review = _review(root, str(report["proposalHash"]))
    source_definition, source = _source(root)
    catalog = _catalog(root)
    l05 = _l05(
        root,
        source_digest=_file_digest(root / "01-L1-SKILL.md"),
        catalog_digest=_file_digest(root / "00-capability-catalog.yaml"),
    )
    contract = _contract_projection(_compiled(root, report))
    source_report = report.get("sourceSkill")
    l05_report = report.get("structuredSkill")
    catalog_report = report.get("capabilityCatalog")
    if (
        not isinstance(source_report, dict)
        or source_report.get("sha256") != source["sha256"]
        or source_report.get("skillId") != source["skill_id"]
        or not isinstance(l05_report, dict)
        or l05_report.get("sha256") != _file_digest(root / "02-L0.5.yaml")
        or l05_report.get("skillId") != l05.skill_id
        or not isinstance(catalog_report, dict)
        or catalog_report.get("sha256") != _file_digest(root / "00-capability-catalog.yaml")
        or catalog_report.get("provider") != catalog.provider
        or catalog_report.get("version") != catalog.version
    ):
        raise PromotionError("workbench report-to-stage binding is invalid")
    stages = [
        {
            "id": item["stage"],
            "format": item["format"],
            "sha256": item["sha256"],
            "previous_sha256": item["previousSha256"],
        }
        for item in trajectory["stages"]
    ]
    body: dict[str, Any] = {
        "status": state,
        "proposal": {
            "proposal_hash": report["proposalHash"],
            "trajectory_hash": trajectory["trajectoryHash"],
            "integrity_valid": True,
            "execution_eligible": False,
            "auto_activated": False,
            "activation_available": False,
        },
        "review": review,
        "source_skill": source,
        "structured_skill": {
            "document": l05.model_dump(by_alias=True, mode="json"),
            "sha256": _file_digest(root / "02-L0.5.yaml"),
        },
        "compiled_contract": contract,
        "capability_catalog": {
            "provider": catalog.provider,
            "version": catalog.version,
            "sha256": _file_digest(root / "00-capability-catalog.yaml"),
            "capabilities": [
                {"id": item.id, "role": item.role, "profiles": list(item.profiles)}
                for item in catalog.capabilities
            ],
        },
        "findings": report.get("findings", []),
        "manual_certification_required": report.get("manualCertificationRequired", []),
        "trajectory": {
            "nodes": stages,
            "edges": [
                {"from": left["id"], "to": right["id"]}
                for left, right in zip(stages, stages[1:])
            ],
        },
        "contract_graph": {
            "nodes": contract["steps"],
            "edges": contract["step_edges"],
        },
        "semantic_diff": _semantic_diff(source, l05, contract),
        "semantic_coverage": report.get("semanticCoverage") or {
            "schema": "netopyu.io/l0-semantic-coverage/legacy-unavailable",
            "gate": "not_evaluated",
            "claimBoundary": (
                "This package predates requirement-level semantic coverage. "
                "Regenerate and reassess it before relying on semantic preservation."
            ),
            "summary": {
                "totalRequirements": 0,
                "preserved": 0,
                "strengthened": 0,
                "weakened": 0,
                "missing": 0,
                "ambiguous": 0,
                "non_machine_verifiable": 0,
                "blockingRequirements": 0,
                "attentionRequirements": 0,
                "lowConfidenceRequirements": 0,
                "languageLossRequirements": 0,
                "averageMappingConfidence": 0.0,
                "averageL1ToL05Confidence": 0.0,
                "averageL05ToL0Confidence": 0.0,
                "semanticCoveragePercent": 0.0,
                "machineEnforcedPercent": 0.0,
                "extraEffects": 0,
            },
            "requirements": [],
            "extraEffects": [],
        },
        "controls": {
            "editor_output": "untrusted_L0.5_draft_only",
            "human_review_channel": "independent_CLI_only",
            "same_session_approval": False,
            "runtime_registration": False,
            "execution_authority": False,
        },
        "claim_boundary": (
            "The workbench validates and visualizes an immutable local proposal. Editing exports "
            "an untrusted L0.5 draft that must pass promote-assess/package and independent review; "
            "the workbench cannot activate or execute a contract."
        ),
    }
    return {
        "apiVersion": WORKBENCH_SCHEMA,
        **body,
        "view_digest": sha256_json(body),
    }


def list_workbench(proposal_root: str | Path, *, limit: int = 100) -> dict[str, Any]:
    """List bounded direct-child proposal packages; invalid entries are digest-only."""
    if not 1 <= limit <= 500:
        raise PromotionError("workbench list limit must be 1..500")
    supplied_root = Path(proposal_root).expanduser()
    if supplied_root.is_symlink():
        raise PromotionError("workbench proposal root is missing or unsafe")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise PromotionError("workbench proposal root is missing or unsafe")
    proposals: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name)[:limit]:
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            view = inspect_workbench(child)
            proposals.append({
                "proposal_id": sha256_json({"directory_name": child.name}),
                "status": view["status"],
                "proposal_hash": view["proposal"]["proposal_hash"],
                "contract": {
                    key: view["compiled_contract"][key]
                    for key in ("id", "version", "kind", "hash")
                },
                "integrity_valid": True,
                "activation_available": False,
            })
        except (OSError, PromotionError, TypeError, ValueError):
            proposals.append({
                "proposal_id": sha256_json({"directory_name": child.name}),
                "status": "invalid",
                "proposal_hash": None,
                "contract": None,
                "integrity_valid": False,
                "activation_available": False,
            })
    body = {
        "count": len(proposals),
        "proposals": proposals,
        "activation_available": False,
    }
    return {"apiVersion": WORKBENCH_SCHEMA, **body, "view_digest": sha256_json(body)}


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
<title>NetOpYu Semantic Mapping Workbench</title>
<style>
:root{color-scheme:dark;--bg:#07101d;--panel:#111d30;--panel2:#0b1728;--line:#29415f;--text:#e7eef8;--muted:#9bb0c9;--ok:#4ade80;--warn:#fbbf24;--bad:#fb7185;--info:#60a5fa;--l1:#c084fc;--l05:#22d3ee;--l0:#4ade80}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#17345a 0,#091525 38%,#050a12 100%);color:var(--text);font:14px/1.55 ui-sans-serif,system-ui;padding:28px}.wrap{max-width:1540px;margin:auto}.hero,.panel{background:rgba(17,29,48,.95);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px;box-shadow:0 15px 45px #0005}h1,h2,h3{margin:0 0 10px}h1{font-size:27px}h2{font-size:18px}h3{font-size:14px}.muted{color:var(--muted)}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{border:1px solid var(--line);padding:4px 9px;border-radius:99px;background:#081321}.ok,.preserved,.strengthened{color:var(--ok)}.warn,.ambiguous,.non_machine_verifiable{color:var(--warn)}.bad,.missing,.weakened{color:var(--bad)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.flow{display:flex;align-items:center;gap:8px;overflow:auto;padding:12px 0}.node{white-space:nowrap;border:1px solid var(--info);border-radius:10px;padding:10px;background:var(--panel2)}.arrow{color:var(--muted);font-size:20px}pre,textarea{width:100%;background:var(--bg);color:#dbeafe;border:1px solid var(--line);border-radius:9px;padding:12px;overflow:auto}textarea{min-height:430px;resize:vertical;font:12px/1.45 ui-monospace,monospace}button{font:inherit}button.action{background:#1d4ed8;color:#fff;border:0;border-radius:8px;padding:9px 13px;cursor:pointer;margin-right:8px}button.secondary{background:#334155}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}.path{font:12px/1.4 ui-monospace,monospace;color:#bfdbfe}.notice{border-left:4px solid var(--warn);padding:10px 14px;background:#2a210c}code{color:#bfdbfe}.semantic-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:14px 0}.semantic-toolbar input{min-width:280px;flex:1;background:#07101d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px 12px}.score-note{border-left:3px solid var(--info);padding:8px 12px;background:#0b1728}.alert-stack{display:grid;gap:8px;margin:12px 0}.alert-card{width:100%;display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;text-align:left;color:var(--text);background:#231b0d;border:1px solid #7c5b13;border-radius:10px;padding:10px 12px;cursor:pointer}.alert-card.critical{background:#2c111a;border-color:#9f2942}.alert-card:hover,.alert-card:focus-visible{outline:2px solid var(--warn);outline-offset:2px}.alert-icon{font-size:18px}.alert-score{font:700 13px ui-monospace,monospace}.semantic-scroll{overflow:visible;border:1px solid var(--line);border-radius:12px;background:#07101d;padding:10px}.semantic-map{min-width:0}.semantic-header{display:none}.semantic-case{border:1px solid transparent;border-radius:11px;margin:6px 0;background:#091525}.semantic-case.attention{border-color:#745719}.semantic-case.critical{border-color:#8f2940}.case-summary{width:100%;display:grid;grid-template-columns:minmax(150px,.8fr) minmax(240px,2fr) auto auto auto auto;gap:12px;align-items:center;text-align:left;color:var(--text);background:#0d1b2f;border:0;border-radius:10px;padding:11px 13px;cursor:pointer}.case-summary:hover,.case-summary:focus-visible{outline:2px solid #60a5fa88;outline-offset:1px}.case-summary .case-id{font:700 11px/1.4 ui-monospace,monospace;color:#bfdbfe}.case-summary .case-source{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.case-score{font:700 11px/1.35 ui-monospace,monospace;border:1px solid var(--line);border-radius:99px;padding:4px 8px;white-space:nowrap}.case-score.low{color:var(--bad);border-color:#8f2940}.case-score.medium{color:var(--warn);border-color:#745719}.case-score.high{color:var(--ok)}.case-verdict{font:700 11px/1.35 ui-monospace,monospace;white-space:nowrap}.case-toggle{color:var(--muted);white-space:nowrap}.semantic-case.expanded .case-summary{border-bottom-left-radius:0;border-bottom-right-radius:0;background:#10223a}.semantic-case:not(.expanded) .semantic-row{display:none}.semantic-row{display:grid;grid-template-columns:minmax(210px,1fr) 104px minmax(210px,1fr) 104px minmax(210px,1fr);gap:10px;position:relative;padding:14px 12px;transition:.15s}.semantic-row.attention::after{content:'!';position:absolute;left:-8px;top:50%;transform:translateY(-50%);display:grid;place-items:center;width:22px;height:22px;border-radius:50%;font-weight:900;background:var(--warn);color:#161008}.semantic-row.critical::after{background:var(--bad);color:#fff}.semantic-node{position:relative;z-index:1;min-height:132px;width:100%;display:block;text-align:left;color:var(--text);background:#0d1b2f;border:1px solid var(--line);border-top:3px solid;border-radius:11px;padding:12px;cursor:pointer;transition:transform .15s,box-shadow .15s,border-color .15s,opacity .15s}.semantic-node.l1{border-top-color:var(--l1)}.semantic-node.l05{border-top-color:var(--l05)}.semantic-node.l0{border-top-color:var(--l0)}.semantic-link{z-index:2;align-self:center;width:100%;display:grid;place-items:center;gap:2px;text-align:center;color:var(--text);background:#10223a;border:1px dashed #4e7199;border-radius:10px;padding:9px 5px;cursor:pointer;transition:.15s}.semantic-link .link-label{font:700 10px/1.3 ui-monospace,monospace;color:#bfdbfe}.semantic-link .link-score{font:800 19px/1 ui-monospace,monospace}.semantic-link .link-loss{font:10px/1.3 ui-monospace,monospace;color:var(--muted)}.semantic-link.low{border-color:var(--bad);background:#28121a}.semantic-link.medium{border-color:var(--warn);background:#231b0d}.semantic-node:hover,.semantic-node:focus-visible,.semantic-node.active,.semantic-link:hover,.semantic-link:focus-visible,.semantic-link.active{outline:none;transform:translateY(-2px);border-color:#fff8;box-shadow:0 0 0 2px #60a5fa88,0 10px 28px #0008}.semantic-node.dimmed,.semantic-link.dimmed{opacity:.26}.semantic-node.attention{background:#211b10}.semantic-node.critical{background:#28121a}.node-meta{display:flex;justify-content:space-between;gap:8px;margin-bottom:8px;color:var(--muted);font:11px/1.3 ui-monospace,monospace}.node-body{white-space:pre-wrap;overflow-wrap:anywhere}.evidence-item{margin-top:7px;padding-top:7px;border-top:1px solid #29415f}.empty-evidence{color:var(--bad);font-weight:700}.semantic-detail{display:grid;grid-template-columns:minmax(240px,.7fr) minmax(320px,1.3fr);gap:18px;margin-top:14px;border:1px solid var(--line);border-radius:12px;padding:16px;background:#0b1728}.detail-score{display:flex;align-items:center;gap:12px}.score-number{font:800 34px/1 ui-monospace,monospace}.meter{height:8px;background:#26364d;border-radius:99px;overflow:hidden;margin-top:10px}.meter>span{display:block;height:100%;background:var(--ok)}.meter.loss>span{background:var(--warn)}.detail-grid{display:grid;grid-template-columns:auto 1fr;gap:6px 12px}.detail-grid dt{color:var(--muted)}.detail-grid dd{margin:0;overflow-wrap:anywhere}.critical-text{color:var(--bad)}.warning-text{color:var(--warn)}@media(max-width:1180px){.case-summary{grid-template-columns:minmax(140px,.8fr) minmax(180px,1.6fr) auto auto auto}.case-verdict{display:none}.semantic-row{grid-template-columns:1fr;gap:8px}.semantic-link{width:min(360px,92%);margin:auto;min-height:58px}.semantic-link::before{content:'↓';font-size:18px;color:var(--info)}.semantic-node{min-height:auto}.semantic-row.attention::after{top:18px;transform:none}}@media(max-width:760px){body{padding:14px}.semantic-detail{grid-template-columns:1fr}.hero,.panel{padding:15px}.semantic-toolbar input{min-width:100%}.case-summary{grid-template-columns:1fr auto}.case-source{grid-column:1/-1}.case-score{display:none}}
</style></head><body><main class="wrap">
<section class="hero"><h1>L1 → L0.5 → L0 语义映射工作台</h1><p class="muted">Semantic Mapping Workbench · 同源证据联动、语言丢失定位、可复算置信度</p><div class="badges" id="badges"></div></section>
<section class="panel notice"><strong>无激活能力 / No activation capability.</strong> 本页只审查不可变 proposal；编辑器仅导出不可信 L0.5 草稿，不能批准、注册或执行合同。</section>
<section class="grid"><article class="panel"><h2>Proposal</h2><table id="summary"></table></article><article class="panel"><h2>Review controls</h2><pre id="controls"></pre></article></section>
<section class="panel"><h2>L1 → L0.5 → L0 trajectory</h2><div class="flow" id="trajectory"></div></section>
<section class="panel"><h2>Runtime contract flow</h2><div class="flow" id="contract-flow"></div></section>
<section class="panel" id="semantic-explorer"><h2>Semantic coverage gate / 语义映射与覆盖门禁</h2><div class="badges" id="coverage-summary"></div><p class="muted" id="coverage-boundary"></p><p class="score-note"><strong>渐进披露：</strong>默认仅显示 requirement 摘要和两段转换分数；展开关注项后才加载完整证据链。置信度由确定性规则计算，不是 LLM 自报概率或生产成功率。</p><div id="semantic-alerts" class="alert-stack" aria-live="polite"></div><div class="semantic-toolbar"><input id="semantic-search" type="search" placeholder="搜索 requirement、路径、原文或 verdict…" aria-label="搜索语义映射"><div><button class="action" id="alert-toggle" type="button" aria-pressed="false">只看告警</button><button class="action secondary" id="expand-risk" type="button">展开风险项</button><button class="action secondary" id="collapse-all" type="button">收起全部</button></div></div><div class="semantic-scroll"><div class="semantic-map" id="semantic-map"><div class="semantic-header"><div class="lane-title lane-l1">L1 · 自然语言意图与约束</div><div class="transition-head">① L1 → L0.5</div><div class="lane-title lane-l05">L0.5 · 结构化自然语言证据</div><div class="transition-head">② L0.5 → L0</div><div class="lane-title lane-l0">L0 · 可执行确定性约束</div></div><div id="semantic-rows"></div></div></div><div class="semantic-detail" id="semantic-detail" aria-live="polite"><p class="muted">选择一个 requirement 后显示转换解释和精确修复位置。</p></div></section>
<section class="grid"><article class="panel"><h2>Semantic diff</h2><pre id="diff"></pre></article><article class="panel"><h2>Findings / certification</h2><pre id="findings"></pre></article></section>
<section class="panel"><h2>L0.5 draft editor</h2><p class="muted">编辑 JSON（YAML 兼容）并下载草稿。下载不会修改 proposal。随后以 <code>--l05</code> 重新运行 promote-assess。</p><textarea id="editor" spellcheck="false"></textarea><p><button class="action" id="download">Download untrusted L0.5 draft</button><button class="action secondary" id="reset">Reset</button></p></section>
<section class="panel"><h2>Independent review command</h2><p>在终端中由独立 reviewer 执行；页面刻意不提供批准按钮：</p><pre>scripts/netopyu-l0 promote-review --proposal &lt;proposal-directory&gt; --reviewer &lt;identity&gt; --decision approve|reject --reason &lt;text&gt;</pre></section>
</main><script type="application/json" id="workbench-data">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('workbench-data').textContent);
const el=id=>document.getElementById(id);
const text=(tag,value,cls)=>{const n=document.createElement(tag);n.textContent=String(value);if(cls)n.className=cls;return n};
for(const value of [data.status,data.proposal.integrity_valid?'integrity valid':'invalid',data.proposal.activation_available?'activation available':'activation unavailable'])el('badges').append(text('span',value,'badge '+(String(value).includes('invalid')?'warn':'ok')));
const proposalRows=[['Contract',`${data.compiled_contract.id}@${data.compiled_contract.version}`],['Kind',data.compiled_contract.kind],['Proposal hash',data.proposal.proposal_hash],['Contract hash',data.compiled_contract.hash],['Review',data.review.decision??'pending'],['Execution eligible',data.proposal.execution_eligible]];
for(const [key,value] of proposalRows){const row=document.createElement('tr');row.append(text('th',key));row.append(text('td',value));el('summary').append(row)}
el('controls').textContent=JSON.stringify(data.controls,null,2);el('diff').textContent=JSON.stringify(data.semantic_diff,null,2);el('findings').textContent=JSON.stringify({findings:data.findings,manualCertificationRequired:data.manual_certification_required},null,2);
const flow=(id,values)=>{values.forEach((value,index)=>{if(index)el(id).append(text('span','→','arrow'));el(id).append(text('span',typeof value==='string'?value:value.id,'node'))})};flow('trajectory',data.trajectory.nodes);flow('contract-flow',data.contract_graph.nodes);
const coverage=data.semantic_coverage;const cs=coverage.summary;
for(const [label,value,cls] of [['gate',coverage.gate,coverage.gate==='passed'?'ok':'bad'],['coverage',`${cs.semanticCoveragePercent}%`,'ok'],['L1→L0.5',`${cs.averageL1ToL05Confidence??0}%`,(cs.averageL1ToL05Confidence??0)>=85?'ok':'warn'],['L0.5→L0',`${cs.averageL05ToL0Confidence??0}%`,(cs.averageL05ToL0Confidence??0)>=85?'ok':'warn'],['machine enforced',`${cs.machineEnforcedPercent}%`,'ok'],['blocking',cs.blockingRequirements,cs.blockingRequirements?'bad':'ok'],['需关注',cs.attentionRequirements??0,(cs.attentionRequirements??0)?'warn':'ok']])el('coverage-summary').append(text('span',`${label}: ${value}`,'badge '+cls));
el('coverage-boundary').textContent=coverage.claimBoundary;
const confidence=item=>item.mappingConfidence||{score:0,band:'low',basis:['Legacy package has no confidence evidence.'],claimBoundary:'Regenerate the proposal.'};
const loss=item=>item.languageLoss||{type:'unknown',riskPercent:100,explanation:'Legacy package has no language-loss assessment.'};
const transition=(item,key)=>item.transitionAssessments?.[key]||{score:0,band:'low',verdict:'not_evaluated',lossRiskPercent:100,explanation:'Regenerate this legacy proposal to evaluate this transition.'};
const compact=value=>{const raw=typeof value==='string'?value:JSON.stringify(value);return raw.length>210?raw.slice(0,207)+'…':raw};
const evidenceText=items=>(items||[]).map(item=>`${item.path}: ${compact(item.value)}`);
const requirements=new Map(coverage.requirements.map(item=>[item.id,item]));const caseElements=new Map();let pinnedId=null;let alertsOnly=false;
function buildNode(item,stage,items){const node=document.createElement('button');node.type='button';node.className=`semantic-node ${stage} ${item.alertLevel==='critical'?'critical':item.attentionRequired?'attention':''}`;node.dataset.requirementId=item.id;node.setAttribute('aria-label',`${stage} ${item.id}`);const meta=document.createElement('div');meta.className='node-meta';meta.append(text('span',stage.toUpperCase()));meta.append(text('span',item.id));node.append(meta);const body=document.createElement('div');body.className='node-body';if(stage==='l1'){body.append(text('div',item.source.path,'path'));body.append(text('div',item.source.text))}else{const lines=evidenceText(items);if(!lines.length)body.append(text('div','未找到显式证据 / no explicit evidence','empty-evidence'));for(const line of lines)body.append(text('div',line,'evidence-item path'))}node.append(body);node.addEventListener('pointerenter',()=>activate(item.id,false));node.addEventListener('pointerleave',()=>pinnedId?activate(pinnedId,false):clearHighlight());node.addEventListener('focus',()=>activate(item.id,false));node.addEventListener('blur',()=>pinnedId?activate(pinnedId,false):clearHighlight());node.addEventListener('click',()=>{pinnedId=pinnedId===item.id?null:item.id;if(pinnedId)activate(pinnedId,true);else clearHighlight()});return node}
function buildTransition(item,key,label){const edge=transition(item,key);const node=document.createElement('button');node.type='button';node.className=`semantic-link ${edge.band}`;node.dataset.requirementId=item.id;node.setAttribute('aria-label',`${label} ${item.id}`);node.append(text('span',label,'link-label'));node.append(text('span',`${edge.score}/100`,'link-score'));node.append(text('span',`${edge.verdict} · loss ${edge.lossRiskPercent}%`,'link-loss'));node.addEventListener('pointerenter',()=>activate(item.id,false));node.addEventListener('pointerleave',()=>pinnedId?activate(pinnedId,false):clearHighlight());node.addEventListener('focus',()=>activate(item.id,false));node.addEventListener('blur',()=>pinnedId?activate(pinnedId,false):clearHighlight());node.addEventListener('click',()=>{pinnedId=pinnedId===item.id?null:item.id;if(pinnedId)activate(pinnedId,true);else clearHighlight()});return node}
function setCaseExpanded(id,expanded){const entry=caseElements.get(id);if(!entry)return;entry.container.classList.toggle('expanded',expanded);entry.summary.setAttribute('aria-expanded',String(expanded));entry.toggle.textContent=expanded?'收起':'展开'}
function expandCase(id,scroll=false){setCaseExpanded(id,true);pinnedId=id;activate(id,true);const entry=caseElements.get(id);if(scroll&&entry)entry.container.scrollIntoView({behavior:'smooth',block:'center'})}
function resetDetail(){const detail=el('semantic-detail');detail.replaceChildren(text('p','选择一个 requirement 后显示转换解释和精确修复位置。','muted'))}
function buildCaseSummary(item){const first=transition(item,'l1ToL05'),second=transition(item,'l05ToL0'),summary=document.createElement('button');summary.type='button';summary.className='case-summary';summary.setAttribute('aria-expanded','false');summary.dataset.requirementId=item.id;const identity=document.createElement('span');identity.append(text('span',item.id,'case-id'));identity.append(text('span',` · ${item.category}`,'muted'));summary.append(identity);summary.append(text('span',item.source.text,'case-source'));summary.append(text('span',`① ${first.score} · loss ${first.lossRiskPercent}%`,`case-score ${first.band}`));summary.append(text('span',`② ${second.score} · loss ${second.lossRiskPercent}%`,`case-score ${second.band}`));summary.append(text('span',item.verdict,`case-verdict ${item.verdict}`));const toggle=text('span','展开','case-toggle');summary.append(toggle);summary.addEventListener('click',()=>{const entry=caseElements.get(item.id),expanded=!entry.container.classList.contains('expanded');setCaseExpanded(item.id,expanded);if(expanded){pinnedId=item.id;activate(item.id,true)}else if(pinnedId===item.id){pinnedId=null;clearHighlight();resetDetail()}});return {summary,toggle}}
function renderRows(){for(const item of coverage.requirements){const container=document.createElement('article');container.className=`semantic-case ${item.alertLevel==='critical'?'critical':item.attentionRequired?'attention':''}`;container.dataset.requirementId=item.id;container.dataset.search=`${item.id} ${item.category} ${item.verdict} ${item.source.path} ${item.source.text}`.toLowerCase();const built=buildCaseSummary(item);const row=document.createElement('div');row.className=`semantic-row ${item.alertLevel==='critical'?'critical':item.attentionRequired?'attention':''}`;row.dataset.requirementId=item.id;row.append(buildNode(item,'l1',[]),buildTransition(item,'l1ToL05','① L1 → L0.5'),buildNode(item,'l05',item.l05Evidence),buildTransition(item,'l05ToL0','② L0.5 → L0'),buildNode(item,'l0',item.l0Evidence));container.append(built.summary,row);caseElements.set(item.id,{container,summary:built.summary,toggle:built.toggle});el('semantic-rows').append(container)}}
function clearHighlight(){for(const node of document.querySelectorAll('.semantic-node,.semantic-link')){node.classList.remove('active','dimmed');node.setAttribute('aria-pressed','false')}}
function activate(id,pinned){const item=requirements.get(id);if(!item)return;for(const node of document.querySelectorAll('.semantic-node,.semantic-link')){const match=node.dataset.requirementId===id;node.classList.toggle('active',match);node.classList.toggle('dimmed',!match);node.setAttribute('aria-pressed',String(Boolean(pinned&&match)))}renderDetail(item)}
function detailPair(list,label,value,cls){list.append(text('dt',label));list.append(text('dd',value,cls))}
function renderDetail(item){const conf=confidence(item),langLoss=loss(item),first=transition(item,'l1ToL05'),second=transition(item,'l05ToL0'),detail=el('semantic-detail');detail.replaceChildren();const score=document.createElement('section');score.append(text('h3',`${item.id} · ${item.category}`));const scoreLine=document.createElement('div');scoreLine.className='detail-score';scoreLine.append(text('span',conf.score,'score-number '+(conf.band==='low'?'critical-text':'')));scoreLine.append(text('span',`/ 100 全链路映射置信度\n${conf.band.toUpperCase()} · ${item.verdict}`));score.append(scoreLine);const meter=document.createElement('div');meter.className='meter';const fill=document.createElement('span');fill.style.width=`${Math.max(0,Math.min(100,conf.score))}%`;meter.append(fill);score.append(meter);score.append(text('p',conf.claimBoundary,'muted'));const facts=document.createElement('section');const list=document.createElement('dl');list.className='detail-grid';detailPair(list,'① L1 → L0.5',`${first.score}/100 · ${first.verdict} · loss ${first.lossRiskPercent}%`,first.band==='low'?'critical-text':first.band==='medium'?'warning-text':'ok');detailPair(list,'① 转换解释',first.explanation);detailPair(list,'② L0.5 → L0',`${second.score}/100 · ${second.verdict} · loss ${second.lossRiskPercent}%`,second.band==='low'?'critical-text':second.band==='medium'?'warning-text':'ok');detailPair(list,'② 转换解释',second.explanation);detailPair(list,'全链路语言丢失',`${langLoss.type} · 风险 ${langLoss.riskPercent}%`,langLoss.riskPercent?'warning-text':'ok');detailPair(list,'置信度分量',JSON.stringify(conf.components||{}),'path');detailPair(list,'判定解释',item.reason);detailPair(list,'证据依据',conf.basis.join(' '));detailPair(list,'修复阶段',item.fix.stage);detailPair(list,'修改文件',item.fix.file||'—','path');detailPair(list,'精确路径',item.fix.path||'—','path');detailPair(list,'建议动作',item.fix.hint);facts.append(list);detail.append(score,facts)}
function renderAlerts(){const items=coverage.requirements.filter(item=>item.attentionRequired||item.blocksPromotion).sort((left,right)=>(right.blocksPromotion-left.blocksPromotion)||(confidence(left).score-confidence(right).score));if(!items.length){el('semantic-alerts').append(text('p','✓ 没有检测到低置信度或语言丢失告警。','ok'));return}for(const item of items){const button=document.createElement('button');button.type='button';button.className=`alert-card ${item.alertLevel}`;button.dataset.requirementId=item.id;button.append(text('span',item.blocksPromotion?'⛔':'⚠','alert-icon'));const message=document.createElement('span');message.append(text('strong',`${item.id} · ${item.verdict}`));message.append(text('div',`${loss(item).explanation} 修复：${item.fix.file||'—'} → ${item.fix.path||'—'}`,'muted'));button.append(message);button.append(text('span',`${confidence(item).score}/100`,'alert-score'));button.addEventListener('pointerenter',()=>activate(item.id,false));button.addEventListener('pointerleave',()=>pinnedId?activate(pinnedId,false):clearHighlight());button.addEventListener('click',()=>expandCase(item.id,true));el('semantic-alerts').append(button)}}
function applyFilters(){const query=el('semantic-search').value.trim().toLowerCase();for(const entry of caseElements.values()){const item=requirements.get(entry.container.dataset.requirementId);entry.container.hidden=(alertsOnly&&!item.attentionRequired&&!item.blocksPromotion)||(query&&!entry.container.dataset.search.includes(query))}}
renderRows();renderAlerts();
el('semantic-search').addEventListener('input',applyFilters);el('alert-toggle').addEventListener('click',event=>{alertsOnly=!alertsOnly;event.currentTarget.setAttribute('aria-pressed',String(alertsOnly));event.currentTarget.textContent=alertsOnly?'显示全部':'只看告警';applyFilters()});el('expand-risk').addEventListener('click',()=>{for(const item of coverage.requirements)if(item.attentionRequired||item.blocksPromotion)setCaseExpanded(item.id,true)});el('collapse-all').addEventListener('click',()=>{for(const id of caseElements.keys())setCaseExpanded(id,false);pinnedId=null;clearHighlight();resetDetail()});document.addEventListener('keydown',event=>{if(event.key==='Escape'){pinnedId=null;clearHighlight()}});
const original=JSON.stringify(data.structured_skill.document,null,2);el('editor').value=original;el('reset').addEventListener('click',()=>{el('editor').value=original});el('download').addEventListener('click',()=>{let parsed;try{parsed=JSON.parse(el('editor').value)}catch(error){alert('Invalid JSON: '+error.message);return}const blob=new Blob([JSON.stringify(parsed,null,2)+'\\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='L0.5-draft.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)});
</script></body></html>"""


def render_workbench_html(view: dict[str, Any]) -> str:
    if view.get("apiVersion") != WORKBENCH_SCHEMA:
        raise PromotionError("workbench HTML requires a v1 workbench view")
    encoded = json.dumps(
        view, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return _HTML_TEMPLATE.replace("__DATA__", encoded)


def export_workbench_html(proposal_directory: str | Path, output: str | Path) -> dict[str, Any]:
    view = inspect_workbench(proposal_directory)
    supplied_destination = Path(output).expanduser()
    if supplied_destination.is_symlink():
        raise PromotionError("workbench output target is unsafe")
    destination = supplied_destination.resolve()
    if destination.exists() and not destination.is_file():
        raise PromotionError("workbench output target is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_workbench_html(view)
    destination.write_text(rendered, encoding="utf-8")
    return {
        "ok": True,
        "status": view["status"],
        "output": str(destination),
        "view_digest": view["view_digest"],
        "html_sha256": f"sha256:{sha256(rendered.encode('utf-8')).hexdigest()}",
        "activation_available": False,
    }


__all__ = [
    "WORKBENCH_SCHEMA",
    "export_workbench_html",
    "inspect_workbench",
    "list_workbench",
    "render_workbench_html",
]
