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
<title>NetOpYu L0 Promotion Workbench</title>
<style>
:root{color-scheme:dark;--bg:#09111f;--panel:#111d30;--line:#29415f;--text:#e7eef8;--muted:#9bb0c9;--ok:#4ade80;--warn:#fbbf24;--accent:#60a5fa}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#07101d,#10233d);color:var(--text);font:14px/1.55 ui-sans-serif,system-ui;padding:28px}.wrap{max-width:1180px;margin:auto}.hero,.panel{background:rgba(17,29,48,.94);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px;box-shadow:0 15px 45px #0005}h1,h2{margin:0 0 10px}h1{font-size:25px}h2{font-size:17px}.muted{color:var(--muted)}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{border:1px solid var(--line);padding:4px 9px;border-radius:99px}.ok{color:var(--ok)}.warn{color:var(--warn)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.flow{display:flex;align-items:center;gap:8px;overflow:auto;padding:12px 0}.node{white-space:nowrap;border:1px solid var(--accent);border-radius:10px;padding:10px;background:#0b1728}.arrow{color:var(--muted);font-size:20px}pre,textarea{width:100%;background:#07101d;color:#dbeafe;border:1px solid var(--line);border-radius:9px;padding:12px;overflow:auto}textarea{min-height:430px;resize:vertical;font:12px/1.45 ui-monospace,monospace}button{background:#1d4ed8;color:white;border:0;border-radius:8px;padding:9px 13px;cursor:pointer;margin-right:8px}button.secondary{background:#334155}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}.notice{border-left:4px solid var(--warn);padding:10px 14px;background:#2a210c}code{color:#bfdbfe}details{margin-top:10px}a{color:#93c5fd}
</style></head><body><main class="wrap">
<section class="hero"><h1>NetOpYu L0 Promotion Workbench</h1><p class="muted">P2.0 本地审查工作台 / local review workbench</p><div class="badges" id="badges"></div></section>
<section class="panel notice"><strong>无激活能力 / No activation capability.</strong> 编辑器只导出不可信 L0.5 草稿；必须重新经过确定性 assess/package 和独立人工 review。此页面没有批准、注册、Runtime 或 Provider API。</section>
<section class="grid"><article class="panel"><h2>Proposal</h2><table id="summary"></table></article><article class="panel"><h2>Review controls</h2><pre id="controls"></pre></article></section>
<section class="panel"><h2>L1 → L0.5 → L0 trajectory</h2><div class="flow" id="trajectory"></div></section>
<section class="panel"><h2>Runtime contract flow</h2><div class="flow" id="contract-flow"></div></section>
<section class="grid"><article class="panel"><h2>Semantic diff</h2><pre id="diff"></pre></article><article class="panel"><h2>Findings / certification</h2><pre id="findings"></pre></article></section>
<section class="panel"><h2>L0.5 draft editor</h2><p class="muted">编辑 JSON（YAML 兼容）并下载草稿。下载不会修改 proposal。随后以 <code>--l05</code> 重新运行 promote-assess。</p><textarea id="editor" spellcheck="false"></textarea><p><button id="download">Download untrusted L0.5 draft</button><button class="secondary" id="reset">Reset</button></p></section>
<section class="panel"><h2>Independent review command</h2><p>在终端中由独立 reviewer 执行；页面刻意不提供批准按钮：</p><pre>scripts/netopyu-l0 promote-review --proposal &lt;proposal-directory&gt; --reviewer &lt;identity&gt; --decision approve|reject --reason &lt;text&gt;</pre></section>
</main><script type="application/json" id="workbench-data">__DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('workbench-data').textContent);const el=id=>document.getElementById(id);const text=(tag,value,cls)=>{const n=document.createElement(tag);n.textContent=String(value);if(cls)n.className=cls;return n};
for(const value of [data.status,data.proposal.integrity_valid?'integrity valid':'invalid',data.proposal.activation_available?'activation available':'activation unavailable'])el('badges').append(text('span',value,'badge '+(String(value).includes('invalid')?'warn':'ok')));
const rows=[['Contract',`${data.compiled_contract.id}@${data.compiled_contract.version}`],['Kind',data.compiled_contract.kind],['Proposal hash',data.proposal.proposal_hash],['Contract hash',data.compiled_contract.hash],['Review',data.review.decision??'pending'],['Execution eligible',data.proposal.execution_eligible]];for(const [k,v] of rows){const tr=document.createElement('tr');tr.append(text('th',k));tr.append(text('td',v));el('summary').append(tr)}
el('controls').textContent=JSON.stringify(data.controls,null,2);el('diff').textContent=JSON.stringify(data.semantic_diff,null,2);el('findings').textContent=JSON.stringify({findings:data.findings,manualCertificationRequired:data.manual_certification_required},null,2);
const flow=(id,values)=>{values.forEach((value,index)=>{if(index)el(id).append(text('span','→','arrow'));el(id).append(text('span',typeof value==='string'?value:value.id,'node'))})};flow('trajectory',data.trajectory.nodes);flow('contract-flow',data.contract_graph.nodes);
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
