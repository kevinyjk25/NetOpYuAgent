"""Versioned 24-tool / 60-Skill development corpus for P2.6-B.

This is a transparent reverse-bootstrap development set.  It proves fixture
shape and supports repeatable local A/B experiments; it is deliberately not an
external hidden qualification set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from effect_runtime.mcp_lab import DEFAULT_ENTITIES, DOMAINS, TOOLS
from network_runtime.contracts import sha256_json


DATASET_SCHEMA = "effect-runtime.io/general-effect-development-set/v1"
FEATURE_FAMILIES = (
    "references", "approvals", "conditional_branching",
    "multi_step", "scripts", "composition",
)
SCENARIO_PATTERNS = (
    "success", "missing_required", "unknown_parameter", "approval_denied",
    "revision_conflict", "verification_mismatch", "after_send_unknown",
    "provider_error_before_send", "compensation_failure", "success_alternate",
)


@dataclass(frozen=True)
class GeneralEffectCase:
    case_id: str
    skill_id: str
    feature_family: str
    domain: str
    language: str
    scenario_pattern: str
    user_input: str
    tool_name: str
    l0_skill_id: str
    arguments: dict[str, Any]
    approved: bool
    fault: str


def build_cases() -> tuple[GeneralEffectCase, ...]:
    values: list[GeneralEffectCase] = []
    desired = {
        "network": "vlan-120", "iam": "operator", "cloud": "medium",
        "service_desk": "p1", "data": "restricted", "platform": "replicas-4",
    }
    for family_index, family in enumerate(FEATURE_FAMILIES):
        short = {
            "references": "ref", "approvals": "approval",
            "conditional_branching": "branch", "multi_step": "steps",
            "scripts": "script", "composition": "compose",
        }[family]
        for index, pattern in enumerate(SCENARIO_PATTERNS):
            domain = DOMAINS[(family_index + index) % len(DOMAINS)]
            case_id = f"{short}-{index + 1:02d}"
            entity = DEFAULT_ENTITIES[domain]
            arguments: dict[str, Any] = {
                "entity_id": entity,
                "desired_value": desired[domain],
                "expected_revision": 1,
                "change_id": f"chg-{case_id}",
                "reason": f"P2.6-B controlled case {case_id}",
            }
            approved = pattern != "approval_denied"
            fault = {
                "verification_mismatch": "verification_mismatch",
                "after_send_unknown": "after_send_unknown",
                "provider_error_before_send": "provider_error_before_send",
                "compensation_failure": "verification_mismatch+compensation_failure",
            }.get(pattern, "none")
            if pattern == "missing_required":
                arguments.pop("reason")
            elif pattern == "unknown_parameter":
                arguments["invented_scope"] = "all"
            elif pattern == "revision_conflict":
                arguments["expected_revision"] = 99
            language = "zh" if index % 2 == 0 else "en"
            if language == "zh":
                prompt = (
                    f"按变更 {arguments.get('change_id')} 将 {domain} 对象 {entity} 的状态调整为"
                    f" {desired[domain]}，原因是 {arguments.get('reason', '未提供')}。"
                )
            else:
                prompt = (
                    f"Under change {arguments.get('change_id')}, set {domain} entity {entity} "
                    f"to {desired[domain]}; reason: {arguments.get('reason', 'not supplied')}."
                )
            values.append(GeneralEffectCase(
                case_id=case_id,
                skill_id=f"effect-{short}-{index + 1:02d}",
                feature_family=family,
                domain=domain,
                language=language,
                scenario_pattern=pattern,
                user_input=prompt,
                tool_name=f"{domain}_apply_change",
                l0_skill_id=f"effect.{domain}.state.apply",
                arguments=arguments,
                approved=approved,
                fault=fault,
            ))
    return tuple(values)


def _skill_body(case: GeneralEffectCase) -> tuple[str, dict[str, str]]:
    domain = case.domain
    common = (
        "## Contract\n\n"
        f"Operate only on the explicit `{domain}` entity and use change/revision values from the request.\n"
        "Never infer a missing target, reason, revision, or approval.\n\n"
    )
    resources: dict[str, str] = {}
    if case.feature_family == "references":
        resources["references/policy.md"] = (
            f"# {domain} policy\n\nOnly one explicit entity may be changed. "
            "An independent read must verify the exact desired value.\n"
        )
        body = common + (
            "Read [the domain policy](references/policy.md) before proposing the operation.\n\n"
            "## Steps\n\n1. Resolve the entity.\n2. Read and retain its revision.\n"
            "3. Request approval.\n4. Apply once.\n5. Verify or restore the snapshot.\n"
        )
    elif case.feature_family == "approvals":
        body = common + (
            "## Approval boundary\n\nCreate an immutable proposal first. Execute only after an approver "
            "accepts that exact proposal; rejection is terminal and sends no write.\n"
        )
    elif case.feature_family == "conditional_branching":
        body = common + (
            "## Branches\n\n- If any required value is absent, ask one precise question.\n"
            "- If the revision changed after approval, stop without writing.\n"
            "- If verification fails, restore the approved snapshot.\n"
            "- Otherwise return verified success with evidence.\n"
        )
    elif case.feature_family == "multi_step":
        body = common + (
            "## Ordered steps\n\n1. Validate exact parameters.\n2. Read the pre-change state.\n"
            "3. Bind approval to the plan.\n4. Revalidate the snapshot.\n5. Apply exactly once.\n"
            "6. Independently verify.\n7. Compensate on mismatch.\n8. Seal the terminal audit.\n"
        )
    elif case.feature_family == "scripts":
        resources.update({
            "scripts/preflight.py": "# read-only preflight adapter contract; never execute during package inspection\n",
            "scripts/apply.py": "# provider adapter contract; execution authority comes only from bound Capability\n",
            "scripts/verify.py": "# independent verification adapter contract\n",
            "scripts/rollback.py": "# compensation adapter contract restoring the approved snapshot\n",
        })
        body = common + (
            "## Bundled implementation references\n\nUse `scripts/preflight.py`, `scripts/apply.py`, "
            "`scripts/verify.py`, and `scripts/rollback.py` only through their reviewed Capability "
            "bindings. Package inspection must never execute them.\n"
        )
    else:
        resources["references/composition.md"] = (
            "# Composition\n\nL1 may order active L0 contracts. L0 must never invoke L1. "
            "Every L0 reference requires an exact version and artifact digest.\n"
        )
        body = common + (
            "Read [the composition rules](references/composition.md). This L1 Skill may orchestrate "
            f"the active `{case.l0_skill_id}` contract, but may not directly invoke a write tool.\n"
        )
    return body, resources


def _skill_text(case: GeneralEffectCase) -> tuple[str, dict[str, str]]:
    body, resources = _skill_body(case)
    tool_names = " ".join((
        f"{case.domain}_get_state", f"{case.domain}_validate_change",
        f"{case.domain}_apply_change", f"{case.domain}_restore_state",
    ))
    metadata = [
        f"  skill_id: '{case.skill_id.replace('-', '_')}'",
        f"  domain: '{case.domain}'",
        "  risk_level: 'medium'",
        "  requires_hitl: 'true'",
        f"  feature_family: '{case.feature_family}'",
    ]
    if case.feature_family == "scripts":
        metadata.append(
            "  effect-runtime-script-roles: 'scripts/preflight.py=preflight,"
            "scripts/apply.py=provider_adapter,scripts/verify.py=verifier,"
            "scripts/rollback.py=compensator'"
        )
    text = (
        "---\n"
        f"name: {case.skill_id}\n"
        f"description: Safely apply one reviewed {case.domain} state transition and verify the result.\n"
        f"allowed-tools: {tool_names}\n"
        "metadata:\n" + "\n".join(metadata) + "\n"
        "---\n"
        f"# {case.skill_id}\n\n" + body
    )
    return text, resources


def materialize_dataset(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser()
    skills_root = root / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    for case in cases:
        package = skills_root / case.skill_id
        package.mkdir(parents=True, exist_ok=True)
        skill_text, resources = _skill_text(case)
        (package / "SKILL.md").write_text(skill_text, encoding="utf-8")
        for relative, content in resources.items():
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    case_values = [asdict(item) for item in cases]
    tool_values = [asdict(item) for item in TOOLS]
    manifest = {
        "schema": DATASET_SCHEMA,
        "toolCount": len(tool_values),
        "skillCount": len(case_values),
        "domains": list(DOMAINS),
        "featureFamilies": {
            family: sum(item.feature_family == family for item in cases)
            for family in FEATURE_FAMILIES
        },
        "scenarioPatterns": {
            pattern: sum(item.scenario_pattern == pattern for item in cases)
            for pattern in SCENARIO_PATTERNS
        },
        "tools": tool_values,
        "cases": case_values,
        "claimBoundary": (
            "Transparent reverse-bootstrap development corpus; not an external hidden "
            "qualification set and not a production success-rate estimate."
        ),
    }
    manifest["datasetDigest"] = sha256_json({
        "schema": DATASET_SCHEMA, "tools": tool_values, "cases": case_values,
    })
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "DATASET_SCHEMA", "FEATURE_FAMILIES", "GeneralEffectCase",
    "SCENARIO_PATTERNS", "build_cases", "materialize_dataset",
]
