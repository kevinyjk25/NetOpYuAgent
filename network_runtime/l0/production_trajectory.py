"""Readable, hash-linked archives for every authoritative production L0.

The archive is a reverse bootstrap from already reviewed L0 contracts. It
proves structural L1/L0.5/L0 parity and exact compiler round trips; it does not
claim that a model independently rediscovered the production semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from network_runtime.contracts import sha256_json
from network_runtime.policies import reviewed_contracts
from skills.skill_format import parse_skill_md

from .compiler import compile_documents, parse_document
from .models import CompiledAtomicEffect, ParameterSpec
from .production import (
    CATALOG,
    PRODUCTION_DEFINITIONS,
    ProductionDefinition,
    authoring_document,
)
from .promotion import L05_API_VERSION, assess_promotion, build_l05_spec, l05_yaml


ARCHIVE_SCHEMA = "netopyu.io/l0-production-trajectory/v1"
TRAJECTORY_SCHEMA = "netopyu.io/l0-production-stage-chain/v1"
DEFAULT_ARCHIVE_ROOT = Path(__file__).with_name("production_trajectories")
_DIRECT_ARGUMENT = re.compile(r"^\$\{\s*arguments\.([A-Za-z_][A-Za-z0-9_]*)\s*\}$")
_PROMOTION_LABELS = {
    "source-skill",
    "source-sha256",
    "capability-catalog-sha256",
    "l0.5-sha256",
    "promotion-state",
}
_STAGE_FILES = (
    "01-L1-SKILL.md",
    "02-L0.5.yaml",
    "03-L0-authoring.yaml",
    "04-L0-compiled.json",
)
_ARCHIVE_FILES = (
    "00-capability-catalog.yaml",
    *_STAGE_FILES,
    "trajectory.json",
    "report.json",
)


class ProductionTrajectoryError(ValueError):
    pass


def _digest_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _risk(contract: CompiledAtomicEffect) -> str:
    return contract.spec.approval.risk


def _parameter_description(name: str, parameter: ParameterSpec) -> str:
    requirement = "必填 / Required" if parameter.required else "可选 / Optional"
    parts = [f"{requirement} {parameter.type}"]
    if parameter.enum:
        parts.append("允许值 / allowed values: " + ", ".join(map(str, parameter.enum)))
    if parameter.minimum is not None:
        parts.append(f"最小值 / minimum: {parameter.minimum:g}")
    if parameter.maximum is not None:
        parts.append(f"最大值 / maximum: {parameter.maximum:g}")
    if parameter.min_length is not None:
        parts.append(f"最短长度 / minimum length: {parameter.min_length}")
    if parameter.max_length is not None:
        parts.append(f"最长长度 / maximum length: {parameter.max_length}")
    if parameter.pattern:
        parts.append(f"格式 / pattern: `{parameter.pattern}`")
    return f"{name}；" + "；".join(parts) + "。"


def _skill_name(definition: ProductionDefinition) -> str:
    return definition.skill_id.replace(".", "-")


def _l1_text(
    definition: ProductionDefinition,
    contract: CompiledAtomicEffect,
) -> str:
    name = _skill_name(definition)
    has_compensation = contract.spec.compensation is not None
    purpose = (
        f"以精确参数安全执行 {definition.skill_id}，并通过独立证据验证结果。 / "
        f"Safely execute {definition.skill_id} with exact inputs and independent evidence."
    )
    returns = (
        "独立验证的目标状态或独立验证的精确补偿状态。 / Independently verified "
        "desired state or independently verified exact compensation."
        if has_compensation else
        "独立验证的目标状态，或明确的人工介入终态。 / Independently verified "
        "desired state or an explicit manual-intervention terminal state."
    )
    frontmatter = {
        "name": name,
        "description": (
            f"Readable bootstrap projection of production L0 {definition.skill_id}; "
            "collect exact intent, require approval, and trust only independent verification."
        ),
        "allowed-tools": definition.tool_name,
        "metadata": {
            "skill_id": name.replace("-", "_"),
            "display_name": definition.skill_id,
            "purpose": purpose,
            "risk_level": _risk(contract),
            "requires_hitl": "true",
            "profiles": ",".join(definition.profiles),
            "tags": "production,l0-v2,readability,bootstrap",
            "tool_deps": definition.tool_name,
            "returns": returns,
            "origin": "bootstrap-from-reviewed-production-l0-v2",
        },
    }
    parameter_lines = "\n".join(
        f"- `{name}`: {_parameter_description(name, parameter)}"
        for name, parameter in contract.spec.parameters.items()
    )
    compensation_line = (
        "- 验证失败时只能使用合同声明的补偿并独立验证恢复。 / On verification "
        "failure, use only contractual compensation and independently verify restoration."
        if has_compensation else
        "- Non-compensable justification: 该受审合同没有安全自动逆操作；验证失败必须进入"
        "人工介入，禁止盲目重试。 / The reviewed contract has no safe automatic inverse; "
        "verification failure requires manual intervention and never a blind retry."
    )
    semantic_intents = yaml.safe_dump([
        {
            "effectCapability": contract.spec.effect.capability,
            **contract.spec.intent.model_dump(by_alias=True, mode="json"),
        }
    ], sort_keys=False, allow_unicode=True).strip()
    body = f"""# {definition.skill_id}

> 本文件是从已受审生产 L0 反向生成的可读基线，用于解释与 round-trip 验证；它不是新的执行授权。  
> This file is a readable baseline reverse-bootstrapped from a reviewed production L0 for explanation and round-trip validation; it grants no execution authority.

## 目标 / Purpose

{purpose}

## 精确语义意图 / Exact Semantic Intent

以下小型结构块是 L1、L0.5 与 L0 之间的可审计语义锚点；Runtime 必须逐字段保真，
不得从周边自然语言猜测或补全。 / This small structured block is the auditable
semantic anchor across L1, L0.5 and L0; Runtime must preserve every field and may
not guess or complete it from surrounding prose.

<!-- netopyu:semantic-intents/v1 -->
```yaml
{semantic_intents}
```

## Parameters

{parameter_lines}

## Steps

1. 收集全部必填参数且不得推断关键值。 / Collect every required input without inferring critical values.
2. 通过合同 Observation 读取并保存审批前状态。 / Read and preserve pre-approval state through contractual observation.
3. 展示并绑定不可变计划，等待明确的一次性人工审批。 / Bind the immutable plan and wait for explicit one-shot human approval.
4. 执行前重读状态；漂移时停止，不发送 Effect。 / Re-read before execution and stop on drift without sending the effect.
5. 仅通过 `{definition.tool_name}` 发送一次受审 Effect。 / Send the reviewed effect exactly once through `{definition.tool_name}`.
6. 使用独立 verifier 判断结果；写响应本身不是成功。 / Use the independent verifier; the write response alone is not success.
7. 按合同补偿或进入人工介入终态。 / Compensate contractually or enter manual intervention.

## Constraints

- 人工审批强制且只绑定当前合同、参数、目标和前态。 / Human approval is mandatory and binds only this contract, inputs, target, and preflight.
- 只允许 profile：{', '.join(definition.profiles)}。 / Allowed profiles only: {', '.join(definition.profiles)}.
- Provider 返回不能替代独立验证。 / A Provider response cannot replace independent verification.
- 写结果不确定时先只读对账，禁止盲目重试。 / Reconcile read-only after an indeterminate result; never retry blindly.
{compensation_line}
"""
    return (
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
        + "\n---\n\n"
        + body.strip()
        + "\n"
    )


def _input_spec(
    value: Any,
    parameters: dict[str, ParameterSpec],
) -> dict[str, Any]:
    match = _DIRECT_ARGUMENT.fullmatch(value) if isinstance(value, str) else None
    if match and match.group(1) in parameters:
        raw = parameters[match.group(1)].model_dump(by_alias=True, mode="json")
        raw["required"] = False
        return raw
    value_type = (
        "boolean" if isinstance(value, bool)
        else "integer" if isinstance(value, int)
        else "number" if isinstance(value, float)
        else "array" if isinstance(value, list)
        else "object" if isinstance(value, dict)
        else "string"
    )
    return {"type": value_type, "required": False}


def _output_spec(field: str, expected: Any) -> dict[str, Any]:
    if field in {"passed", "restored"} or isinstance(expected, bool):
        return {"type": "boolean"}
    if isinstance(expected, int) and not isinstance(expected, bool):
        return {"type": "integer"}
    if isinstance(expected, float):
        return {"type": "number"}
    if isinstance(expected, list):
        return {"type": "array"}
    if isinstance(expected, str):
        return {"type": "string"}
    return {"type": "object"}


def _capability_catalog(
    definition: ProductionDefinition,
    contract: CompiledAtomicEffect,
) -> dict[str, Any]:
    capabilities: dict[str, dict[str, Any]] = {}

    def add(
        capability_id: str,
        role: str,
        arguments: dict[str, Any],
        fields: dict[str, Any],
        *,
        tool: str | None = None,
    ) -> None:
        entry = capabilities.setdefault(capability_id, {
            "id": capability_id,
            "role": role,
            "profiles": list(definition.profiles),
            "inputs": {},
            "outputs": {},
        })
        if entry["role"] != role:
            raise ProductionTrajectoryError(
                f"capability {capability_id} has conflicting roles"
            )
        if tool:
            entry["tool"] = tool
        for name, value in arguments.items():
            entry["inputs"][name] = _input_spec(value, contract.spec.parameters)
        for field, expected in fields.items():
            entry["outputs"].setdefault(
                field.split(".", 1)[0],
                _output_spec(field, expected),
            )

    spec = contract.spec
    add(
        spec.effect.capability,
        "effect",
        spec.effect.request,
        {"accepted": True},
        tool=spec.effect.tool,
    )
    for observation in spec.preflight:
        fields = {name: None for name in observation.snapshot_fields}
        fields.update({item.field: item.expected for item in observation.predicates})
        add(observation.capability, "observation", observation.arguments, fields)
    add(
        spec.verification.capability,
        "observation",
        spec.verification.arguments,
        {item.field: item.expected for item in spec.verification.predicates},
    )
    if spec.compensation is not None:
        add(
            spec.compensation.capability,
            "compensation",
            spec.compensation.arguments,
            {"restored": True},
            tool=spec.compensation.tool,
        )
        add(
            spec.compensation.verification.capability,
            "observation",
            spec.compensation.verification.arguments,
            {
                item.field: item.expected
                for item in spec.compensation.verification.predicates
            },
        )
    return {
        "apiVersion": "netopyu.io/capability-catalog/v1",
        "provider": f"netopyu.production.{definition.tool_name}",
        "version": definition.version,
        "capabilities": list(capabilities.values()),
    }


def _semantic_payload(contract: CompiledAtomicEffect) -> dict[str, Any]:
    raw = contract.model_dump(by_alias=True, mode="json")
    raw.pop("contractHash", None)
    labels = raw.get("metadata", {}).get("labels", {})
    raw["metadata"]["labels"] = {
        key: value for key, value in labels.items() if key not in _PROMOTION_LABELS
    }
    return raw


def _refine_l05_workflow(l05: Any, contract: CompiledAtomicEffect) -> Any:
    """Narrow generic capability choices to the exact production phase."""

    raw = l05.model_dump(by_alias=True, mode="json")
    spec = contract.spec
    raw["semanticIntents"] = [{
        "effectCapability": spec.effect.capability,
        **spec.intent.model_dump(by_alias=True, mode="json"),
    }]
    choices = {
        "preflight": [item.capability for item in spec.preflight],
        "effect": [spec.effect.capability],
        "verification": [spec.verification.capability],
        "compensation": (
            [spec.compensation.capability]
            if spec.compensation is not None else []
        ),
    }
    for step in raw["workflow"]:
        if step["phase"] in choices:
            step["capabilityOptions"] = choices[step["phase"]]
        if step["phase"] == "compensation" and spec.compensation is None:
            step["instruction"] = (
                "该合同没有安全自动补偿；验证失败后进入人工介入并禁止盲目重试。 / "
                "This contract has no safe automatic compensation; verification "
                "failure enters manual intervention and forbids blind retry."
            )
    if spec.compensation is None:
        raw["outcomes"]["rollback"] = (
            "无自动回滚声明；必须由人工介入恢复并另行验证。 / No automatic rollback "
            "is declared; manual recovery and separate verification are required."
        )
    return type(l05).model_validate(raw)


def _trajectory_text(
    capability_text: str,
    l1_text: str,
    l05_text: str,
    authoring_text: str,
    compiled_text: str,
    contract_hash: str,
) -> str:
    stage_values = (
        ("L1", "anthropic-skill-markdown", _STAGE_FILES[0], l1_text),
        ("L0.5", L05_API_VERSION, _STAGE_FILES[1], l05_text),
        ("L0-authoring", "netopyu.io/l0-effect/v2", _STAGE_FILES[2], authoring_text),
        ("L0-compiled", "netopyu.io/l0-effect-compiled/v2", _STAGE_FILES[3], compiled_text),
    )
    previous: str | None = None
    stages: list[dict[str, Any]] = []
    for stage, format_id, filename, content in stage_values:
        digest = _digest_text(content)
        item: dict[str, Any] = {
            "stage": stage,
            "format": format_id,
            "file": filename,
            "sha256": digest,
            "previousSha256": previous,
        }
        if stage == "L0-compiled":
            item["contractHash"] = contract_hash
        stages.append(item)
        previous = digest
    trajectory: dict[str, Any] = {
        "schema": TRAJECTORY_SCHEMA,
        "bootstrapDirection": "reviewed-L0-to-readable-L1/L0.5-baseline",
        "executionEligible": False,
        "autoActivated": False,
        "capabilityCatalog": {
            "file": "00-capability-catalog.yaml",
            "sha256": _digest_text(capability_text),
        },
        "stages": stages,
    }
    trajectory["trajectoryHash"] = sha256_json(trajectory)
    return json.dumps(trajectory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _validate_stage_chain(root: Path, trajectory: dict[str, Any]) -> None:
    stored_hash = trajectory.get("trajectoryHash")
    calculated_hash = sha256_json({
        key: value for key, value in trajectory.items() if key != "trajectoryHash"
    })
    if stored_hash != calculated_hash:
        raise ProductionTrajectoryError(f"trajectory hash mismatch: {root.name}")
    capability = trajectory.get("capabilityCatalog") or {}
    capability_path = root / str(capability.get("file", ""))
    if (
        not capability_path.is_file()
        or _digest_file(capability_path) != capability.get("sha256")
    ):
        raise ProductionTrajectoryError(
            f"trajectory capability catalog mismatch: {root.name}"
        )
    expected = ("L1", "L0.5", "L0-authoring", "L0-compiled")
    stages = trajectory.get("stages")
    if not isinstance(stages, list) or tuple(x.get("stage") for x in stages) != expected:
        raise ProductionTrajectoryError(f"trajectory stage order mismatch: {root.name}")
    previous: str | None = None
    for stage in stages:
        path = root / str(stage.get("file", ""))
        if (
            stage.get("previousSha256") != previous
            or not path.is_file()
            or _digest_file(path) != stage.get("sha256")
        ):
            raise ProductionTrajectoryError(
                f"trajectory stage integrity mismatch: {root.name}/{stage.get('stage')}"
            )
        previous = str(stage["sha256"])


def _assess_texts(
    *,
    l1_text: str,
    capability_text: str,
    l05_text: str,
    authoring_text: str,
) -> Any:
    name = parse_skill_md(l1_text).name
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        skill_root = root / name
        skill_root.mkdir()
        skill = skill_root / "SKILL.md"
        capability = root / "capabilities.yaml"
        l05 = root / "L0.5.yaml"
        authoring = root / "authoring.yaml"
        skill.write_text(l1_text, encoding="utf-8")
        capability.write_text(capability_text, encoding="utf-8")
        l05.write_text(l05_text, encoding="utf-8")
        authoring.write_text(authoring_text, encoding="utf-8")
        return assess_promotion(
            skill_path=skill,
            l05_path=l05,
            candidate_path=authoring,
            capability_catalog_path=capability,
        )


def _build_one(definition: ProductionDefinition) -> dict[str, str]:
    reviewed = reviewed_contracts()[definition.tool_name]
    production_contract = CATALOG.require(definition.skill_id, definition.version)
    if not isinstance(production_contract, CompiledAtomicEffect):
        raise ProductionTrajectoryError(f"production L0 is not atomic: {definition.skill_id}")
    authoring = authoring_document(definition, reviewed)
    authoring_text = yaml.safe_dump(authoring, sort_keys=False, allow_unicode=True)
    rebuilt = compile_documents([
        parse_document(authoring, source=f"trajectory:{definition.skill_id}")
    ])[0]
    if not isinstance(rebuilt, CompiledAtomicEffect):
        raise ProductionTrajectoryError(f"round trip is not atomic: {definition.skill_id}")
    if rebuilt.contract_hash != production_contract.contract_hash:
        raise ProductionTrajectoryError(f"round-trip hash mismatch: {definition.skill_id}")
    compiled_text = json.dumps(
        production_contract.model_dump(by_alias=True, mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    l1_text = _l1_text(definition, production_contract)
    capability_text = yaml.safe_dump(
        _capability_catalog(definition, production_contract),
        sort_keys=False,
        allow_unicode=True,
    )

    name = _skill_name(definition)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / name
        root.mkdir()
        skill = root / "SKILL.md"
        capability = root / "capabilities.yaml"
        skill.write_text(l1_text, encoding="utf-8")
        capability.write_text(capability_text, encoding="utf-8")
        l05 = build_l05_spec(
            skill_path=skill,
            capability_catalog_path=capability,
        )
        l05_text = l05_yaml(_refine_l05_workflow(l05, production_contract))

    assessment = _assess_texts(
        l1_text=l1_text,
        capability_text=capability_text,
        l05_text=l05_text,
        authoring_text=authoring_text,
    )
    if assessment.report["status"] != "ready_for_review":
        raise ProductionTrajectoryError(
            f"promotion assessment failed: {definition.skill_id}: "
            f"{assessment.report['findings']}"
        )
    assessed = assessment.compiled_contract
    if not isinstance(assessed, CompiledAtomicEffect):
        raise ProductionTrajectoryError(
            f"promotion did not compile atomically: {definition.skill_id}"
        )
    production_semantic_hash = sha256_json(_semantic_payload(production_contract))
    promotion_semantic_hash = sha256_json(_semantic_payload(assessed))
    if production_semantic_hash != promotion_semantic_hash:
        raise ProductionTrajectoryError(
            f"promotion semantic parity mismatch: {definition.skill_id}"
        )
    trajectory_text = _trajectory_text(
        capability_text,
        l1_text,
        l05_text,
        authoring_text,
        compiled_text,
        production_contract.contract_hash,
    )
    report: dict[str, Any] = {
        "schema": ARCHIVE_SCHEMA,
        "skillId": definition.skill_id,
        "version": definition.version,
        "tool": definition.tool_name,
        "bootstrap": {
            "source": "reviewed-production-l0-v2",
            "direction": "L0-to-readable-L1/L0.5-baseline",
            "claim": (
                "structural semantic parity and exact compiler round trip; "
                "not independent natural-language inference"
            ),
        },
        "promotionAssessment": {
            "status": assessment.report["status"],
            "summary": assessment.report["summary"],
            "findings": assessment.report["findings"],
            "executionEligible": False,
            "autoActivated": False,
        },
        "roundTrip": {
            "productionContractHash": production_contract.contract_hash,
            "recompiledContractHash": rebuilt.contract_hash,
            "productionSemanticHash": production_semantic_hash,
            "promotionSemanticHash": promotion_semantic_hash,
            "exactContractHash": production_contract.contract_hash == rebuilt.contract_hash,
            "semanticParity": production_semantic_hash == promotion_semantic_hash,
        },
        "trajectoryHash": json.loads(trajectory_text)["trajectoryHash"],
    }
    values = {
        "00-capability-catalog.yaml": capability_text,
        "01-L1-SKILL.md": l1_text,
        "02-L0.5.yaml": l05_text,
        "03-L0-authoring.yaml": authoring_text,
        "04-L0-compiled.json": compiled_text,
        "trajectory.json": trajectory_text,
    }
    report["archiveFiles"] = {
        name: _digest_text(content) for name, content in values.items()
    }
    report["reportHash"] = sha256_json(report)
    values["report.json"] = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    return values


def _index_text(items: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"| `{item['skill_id']}` | `{item['tool']}` | "
        f"`{item['contract_hash']}` | [{item['directory']}]({item['directory']}/) |"
        for item in items
    )
    return f"""# Production L0 trajectories / 生产 L0 轨迹

## 中文

本目录保存全部 {len(items)} 个存量生产 L0 的可读、可审计基线。每个目录包含 Capability Catalog、L1 自然语言 Skill、L0.5 结构化自然语言 Skill、L0 authoring、L0 compiled、逐级 hash trajectory 和验证报告。

这些 L1/L0.5 是从已经受审的生产 L0 **反向 bootstrap**，用于建立解释基线并验证 Promotion 结构约束和编译 round trip；它不证明模型可以从任意自然语言独立恢复相同合同。后续人工修改必须重新通过 Promotion、Provider/故障认证和显式发布。

| L0 | Tool | Contract hash | 目录 |
|---|---|---|---|
{rows}

## English

This directory preserves readable, auditable baselines for all {len(items)} existing production L0 contracts. Each directory contains the Capability Catalog, natural-language L1 Skill, structured-natural-language L0.5 Skill, L0 authoring and compiled artifacts, a predecessor-linked trajectory, and a validation report.

The L1/L0.5 files are reverse-bootstrapped from already reviewed production L0 contracts. They establish an explanation baseline and validate Promotion structure plus exact compiler round trips; they do not prove that a model can independently recover the same contract from arbitrary prose. Any later human change still requires Promotion, Provider/fault qualification, and explicit publication.
"""


def build_production_trajectories(
    output_root: str | Path = DEFAULT_ARCHIVE_ROOT,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    index_items: list[dict[str, Any]] = []
    for definition in PRODUCTION_DEFINITIONS:
        destination = root / definition.skill_id
        destination.mkdir(parents=True, exist_ok=True)
        values = _build_one(definition)
        for name, content in values.items():
            (destination / name).write_text(content, encoding="utf-8")
        contract = CATALOG.require(definition.skill_id, definition.version)
        index_items.append({
            "skill_id": definition.skill_id,
            "tool": definition.tool_name,
            "contract_hash": contract.contract_hash,
            "directory": definition.skill_id,
        })
    (root / "INDEX.md").write_text(_index_text(index_items), encoding="utf-8")
    return {
        "ok": True,
        "contracts": len(index_items),
        "output": str(root),
        "mode": "reverse-bootstrap-with-round-trip-parity",
    }


def _validate_one(root: Path, definition: ProductionDefinition) -> dict[str, Any]:
    directory = root / definition.skill_id
    actual_files = {item.name for item in directory.iterdir() if item.is_file()}
    if actual_files != set(_ARCHIVE_FILES):
        raise ProductionTrajectoryError(
            f"trajectory file coverage mismatch for {definition.skill_id}: "
            f"missing={sorted(set(_ARCHIVE_FILES) - actual_files)}, "
            f"extra={sorted(actual_files - set(_ARCHIVE_FILES))}"
        )
    missing = [name for name in _ARCHIVE_FILES if not (directory / name).is_file()]
    if missing:
        raise ProductionTrajectoryError(
            f"missing trajectory files for {definition.skill_id}: {missing}"
        )
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    stored_report_hash = report.get("reportHash")
    calculated_report_hash = sha256_json({
        key: value for key, value in report.items() if key != "reportHash"
    })
    if stored_report_hash != calculated_report_hash:
        raise ProductionTrajectoryError(f"report hash mismatch: {definition.skill_id}")
    for name, expected in report.get("archiveFiles", {}).items():
        if _digest_file(directory / name) != expected:
            raise ProductionTrajectoryError(
                f"archive file hash mismatch: {definition.skill_id}/{name}"
            )
    trajectory = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
    _validate_stage_chain(directory, trajectory)
    if report.get("trajectoryHash") != trajectory.get("trajectoryHash"):
        raise ProductionTrajectoryError(
            f"report/trajectory hash mismatch: {definition.skill_id}"
        )

    authoring_text = (directory / "03-L0-authoring.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(authoring_text)
    rebuilt = compile_documents([
        parse_document(raw, source=f"archive:{definition.skill_id}")
    ])[0]
    production = CATALOG.require(definition.skill_id, definition.version)
    stored_compiled = json.loads(
        (directory / "04-L0-compiled.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(rebuilt, CompiledAtomicEffect)
        or rebuilt.contract_hash != production.contract_hash
        or stored_compiled != production.model_dump(by_alias=True, mode="json")
    ):
        raise ProductionTrajectoryError(
            f"compiled production parity mismatch: {definition.skill_id}"
        )
    round_trip = report.get("roundTrip") or {}
    if (
        round_trip.get("productionContractHash") != production.contract_hash
        or round_trip.get("recompiledContractHash") != production.contract_hash
        or round_trip.get("exactContractHash") is not True
        or round_trip.get("semanticParity") is not True
    ):
        raise ProductionTrajectoryError(
            f"round-trip report mismatch: {definition.skill_id}"
        )

    assessment = _assess_texts(
        l1_text=(directory / "01-L1-SKILL.md").read_text(encoding="utf-8"),
        capability_text=(directory / "00-capability-catalog.yaml").read_text(
            encoding="utf-8"
        ),
        l05_text=(directory / "02-L0.5.yaml").read_text(encoding="utf-8"),
        authoring_text=authoring_text,
    )
    assessed = assessment.compiled_contract
    if (
        assessment.report["status"] != "ready_for_review"
        or not isinstance(assessed, CompiledAtomicEffect)
        or sha256_json(_semantic_payload(assessed))
        != sha256_json(_semantic_payload(production))
    ):
        raise ProductionTrajectoryError(
            f"L1/L0.5 promotion parity mismatch: {definition.skill_id}"
        )
    return {
        "skill_id": definition.skill_id,
        "tool": definition.tool_name,
        "contract_hash": production.contract_hash,
        "promotion": "ready_for_review",
        "round_trip": "exact",
    }


def validate_production_trajectories(
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
) -> dict[str, Any]:
    root = Path(archive_root).expanduser().resolve()
    if not root.is_dir():
        raise ProductionTrajectoryError(f"production trajectory archive is missing: {root}")
    expected = {item.skill_id for item in PRODUCTION_DEFINITIONS}
    actual = {item.name for item in root.iterdir() if item.is_dir()}
    if actual != expected:
        raise ProductionTrajectoryError(
            f"production trajectory directory coverage mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    root_files = {item.name for item in root.iterdir() if item.is_file()}
    if root_files != {"INDEX.md"}:
        raise ProductionTrajectoryError(
            f"production trajectory root file mismatch: {sorted(root_files)}"
        )
    index_items = []
    for definition in PRODUCTION_DEFINITIONS:
        contract = CATALOG.require(definition.skill_id, definition.version)
        index_items.append({
            "skill_id": definition.skill_id,
            "tool": definition.tool_name,
            "contract_hash": contract.contract_hash,
            "directory": definition.skill_id,
        })
    if (root / "INDEX.md").read_text(encoding="utf-8") != _index_text(index_items):
        raise ProductionTrajectoryError("production trajectory INDEX.md is stale")
    items = [_validate_one(root, definition) for definition in PRODUCTION_DEFINITIONS]
    return {
        "ok": True,
        "contracts": len(items),
        "promotion_ready": sum(x["promotion"] == "ready_for_review" for x in items),
        "exact_round_trips": sum(x["round_trip"] == "exact" for x in items),
        "archive": str(root),
        "items": items,
    }
