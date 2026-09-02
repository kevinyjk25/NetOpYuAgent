"""Sealed ES-P1-Wild paired-study inputs with a hard Agent/Gold split.

This module prepares research inputs only.  It never loads a Skill into DSH,
registers a Tool/MCP capability, executes third-party package content, or
claims public-market evidence is private ES-P1 qualification.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effect_runtime import inspect_skill_package
from evaluation.public_skill_corpus import inspect_public_author_kit
from evaluation.public_skill_fixture_mcp import (
    CATALOG_SCHEMA as FIXTURE_CATALOG_SCHEMA,
    FIXTURE_SCHEMA,
    validate_fixture_catalog,
    validate_fixture_state,
)
from evaluation.public_skill_review import inspect_blind_gold_kit
from network_runtime.contracts import sha256_json


PAIRED_KIT_SCHEMA = "effect-runtime.io/public-skill-paired-study-kit/v1"
PAIRED_CASE_SCHEMA = "effect-runtime.io/public-skill-paired-agent-case/v1"
PAIRED_GOLD_SCHEMA = "effect-runtime.io/public-skill-paired-scoring-case/v1"
PAIRED_AUTHORITY = "paired_study_input_only_no_execution_or_qualification_authority"
MODEL = "qwen3.5:9b"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in values
        ),
        encoding="utf-8",
    )


def _capability_ids(catalog: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for capability in catalog.get("capabilities", []):
        if not isinstance(capability, dict):
            raise ValueError("public paired Tool Catalog capability must be an object")
        capability_id = capability.get("capabilityId")
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise ValueError("public paired Tool Catalog capabilityId is required")
        if capability_id in values:
            raise ValueError("public paired Tool Catalog capabilityId must be unique")
        values.add(capability_id)
    return values


def _fixture_execution_profile(
    catalog: dict[str, Any], task: dict[str, Any], material_root: Path,
) -> str:
    if catalog.get("apiVersion") != FIXTURE_CATALOG_SCHEMA:
        return "not_declared"
    validate_fixture_catalog(catalog)
    case_id = str(task.get("taskId") or task.get("caseId") or "")
    state_fixtures = []
    for relative in task["fixtureRefs"]:
        path = material_root / relative
        if path.suffix.lower() != ".json":
            continue
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("apiVersion") == FIXTURE_SCHEMA:
            state_fixtures.append(validate_fixture_state(candidate, expected_case_id=case_id))
    if len(state_fixtures) != 1:
        raise ValueError("public paired executable case requires exactly one fixture state")
    return "declarative_fixture_mcp_v1"


def _study_plan() -> dict[str, Any]:
    return {
        "apiVersion": "effect-runtime.io/public-skill-paired-study-plan/v1",
        "model": MODEL,
        "modelArtifactDigestRequiredAtRun": True,
        "translationReportRequiredAtRun": True,
        "repetitions": 3,
        "arms": {
            "control": "DSH + identical L1 Skill + LLM-native declared Tool orchestration",
            "treatment": (
                "DSH + identical L1 Skill + qualified L0 Runtime; unqualified conversion "
                "safe-stops without native write fallback"
            ),
        },
        "controls": {
            "sameModel": True, "sameTask": True, "sameSkillPackage": True,
            "sameToolCatalog": True, "sameFixtures": True, "sameApprovalAndFaultInputs": True,
            "declarativeFixtureMcpRequired": True,
            "goldVisibleToAgent": False, "nativeMutationControlLocalSimulationOnly": True,
            "unqualifiedTreatmentMutationAllowed": False,
        },
        "metrics": [
            "task_completion", "execution_precision", "autonomous_coverage",
            "unsafe_execution", "false_commit", "invalid_action", "effect_budget_violation",
            "safe_stop", "process_failure", "latency_p50_p95",
        ],
        "stoppingRule": "run all sealed cases for all three repetitions unless a critical escape occurs",
        "exclusionRule": "only pre-inference protocol or environment failures may be excluded and must be reported",
        "authority": PAIRED_AUTHORITY,
    }


def export_public_paired_study_kit(
    gold_root: str | Path, author_kit_root: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    gold_path = Path(gold_root).expanduser().resolve()
    author_path = Path(author_kit_root).expanduser().resolve()
    gold_inspection = inspect_blind_gold_kit(gold_path)
    author_inspection = inspect_public_author_kit(author_path)
    if not gold_inspection["pairedEvaluationAuthoringEligible"]:
        raise ValueError("public paired study kit requires complete independently authored Gold")

    tasks = {item["taskId"]: item for item in _jsonl(gold_path / "source/tasks.jsonl")}
    case_provenance = {
        item["taskId"]: item
        for item in _jsonl(gold_path / "source/case-author-provenance.jsonl")
    }
    assignments = {
        item["assignmentId"]: item for item in _jsonl(author_path / "assignments.jsonl")
    }
    gold_values = []
    for path in sorted((gold_path / "gold").glob("*.gold.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["decision"] == "author_gold":
            gold_values.append((path, value))
    if not gold_values:
        raise ValueError("public paired study kit needs at least one authored Gold case")

    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("public paired study kit root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    agent = root / "agent"
    scoring = root / "scoring"
    evidence = root / "evidence"
    agent.mkdir()
    scoring.mkdir()
    evidence.mkdir()
    (agent / "packages").mkdir()
    shutil.copytree(gold_path / "source/materials", agent / "materials")

    agent_cases: list[dict[str, Any]] = []
    scoring_cases: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    copied_packages: set[str] = set()
    runtime_package_digests: dict[str, str] = {}
    fixture_executable_cases = 0
    for gold_file, value in gold_values:
        task_id = value["taskId"]
        task = tasks.get(task_id)
        assignment = None if task is None else assignments.get(task["assignmentId"])
        source_provenance = case_provenance.get(task_id)
        if task is None or assignment is None or source_provenance is None:
            raise ValueError("public paired study source binding is incomplete")
        if (
            task["packageId"] != assignment["packageId"]
            or task["packageDigest"] != assignment["packageDigest"]
        ):
            raise ValueError("public paired study package binding mismatch")
        catalog_path = gold_path / "source" / task["toolCatalogRef"]
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        capability_ids = _capability_ids(catalog)
        required = set(value["gold"]["requiredCapabilities"])
        if not required.issubset(capability_ids):
            raise ValueError("public paired Tool Catalog misses a Gold-required capability")
        for relative in task["fixtureRefs"]:
            if not (gold_path / "source" / relative).is_file():
                raise ValueError("public paired fixture binding is missing")
        execution_profile = _fixture_execution_profile(catalog, task, gold_path / "source")
        fixture_executable_cases += execution_profile == "declarative_fixture_mcp_v1"

        package_id = task["packageId"]
        if package_id not in copied_packages:
            source_package = author_path / "packages" / package_id
            package_report = inspect_skill_package(source_package)
            if package_report["gate"] != "passed" or not package_report["packageDigest"]:
                raise ValueError("public paired Skill package is unqualified or drifted")
            shutil.copytree(source_package, agent / "packages" / package_id)
            copied_packages.add(package_id)
            runtime_package_digests[package_id] = str(package_report["packageDigest"])

        agent_cases.append({
            "apiVersion": PAIRED_CASE_SCHEMA, "caseId": task_id,
            "assignmentId": task["assignmentId"], "packageId": package_id,
            "packageDigest": task["packageDigest"], "language": task["language"],
            "runtimePackageDigest": runtime_package_digests[package_id],
            "challenge": task["challenge"], "userPrompt": task["userPrompt"],
            "fixtureRefs": task["fixtureRefs"], "toolCatalogRef": task["toolCatalogRef"],
            "executionProfile": execution_profile,
            "skill": {
                "name": assignment["skillName"], "entry": f"packages/{package_id}/SKILL.md",
                "repository": assignment["repository"], "commitSha": assignment["commitSha"],
                "sourcePath": assignment["sourcePath"],
            },
            "authority": PAIRED_AUTHORITY,
        })
        scoring_cases.append({
            "apiVersion": PAIRED_GOLD_SCHEMA, "caseId": task_id,
            "gold": value["gold"], "oracle": value["oracle"],
            "authority": PAIRED_AUTHORITY,
        })
        provenance.append({
            "caseId": task_id, "caseAuthor": source_provenance,
            "goldAuthorId": value["goldAuthor"]["authorId"],
            "goldAuthorIndependence": {
                "independentFromCaseAuthor": value["goldAuthor"]["independentFromCaseAuthor"],
                "independentFromRuntimeTeam": value["goldAuthor"]["independentFromRuntimeTeam"],
                "modelSemanticCandidatesSeen": value["goldAuthor"]["modelSemanticCandidatesSeen"],
            },
            "goldFileDigest": _file_digest(gold_file),
        })

    agent_cases.sort(key=lambda item: item["caseId"])
    scoring_cases.sort(key=lambda item: item["caseId"])
    provenance.sort(key=lambda item: item["caseId"])
    _write_jsonl(agent / "cases.jsonl", agent_cases)
    _write_jsonl(scoring / "gold.jsonl", scoring_cases)
    _write_jsonl(evidence / "provenance.jsonl", provenance)
    plan = _study_plan()
    (root / "study-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# ES-P1-Wild paired study input\n\n"
        "## 中文\n\n`agent/` 是 DSH 两个实验臂唯一可见输入；`scoring/` 仅供运行结束后的独立评分器读取。"
        "不得把 Gold/Oracle 注入 Agent。当前包只完成封存和角色隔离，没有执行 Tool/MCP、第三方代码或模型，也不是正式 ES-P1 资格。\n\n"
        "## English\n\n`agent/` is the only input visible to either DSH arm. `scoring/` is available only to the "
        "post-run scorer. Never expose Gold/Oracles to the Agent. This kit seals study inputs; it performs no model, "
        "Tool/MCP, or third-party execution and grants no ES-P1 qualification.\n",
        encoding="utf-8",
    )
    sealed_files = {
        path.relative_to(root).as_posix(): _file_digest(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }
    body = {
        "apiVersion": PAIRED_KIT_SCHEMA, "createdAt": _utc_now(),
        "authority": PAIRED_AUTHORITY, "model": MODEL,
        "goldWorkspaceDigest": gold_inspection["workspaceDigest"],
        "caseAuthorReviewWorkspaceDigest": gold_inspection["caseAuthorReviewWorkspaceDigest"],
        "authorKitWorkspaceDigest": author_inspection["workspaceDigest"],
        "caseCount": len(agent_cases), "packageCount": len(copied_packages),
        "fixtureMcpExecutableCaseCount": fixture_executable_cases,
        "fixtureMcpPendingCaseCount": len(agent_cases) - fixture_executable_cases,
        "sealedFiles": sealed_files, "agentGoldIsolation": True,
        "pairedStudyInputEligible": True, "pairedExecutionCompleted": False,
        "fixtureMcpInputEligible": fixture_executable_cases == len(agent_cases),
        "translationReportAttached": False, "pairedExecutionInputEligible": False,
        "containsModelSemanticCandidates": False, "thirdPartyExecutionAttempted": False,
        "officialEsP1QualificationEligible": False,
        "claimBoundary": (
            "Sealed ES-P1-Wild paired-study input only; not a completed run, private holdout, "
            "production probability, or execution authority."
        ),
    }
    manifest = {**body, "workspaceDigest": sha256_json(body)}
    (root / "workspace.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def inspect_public_paired_study_kit(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    manifest = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "workspaceDigest"}
    if manifest.get("apiVersion") != PAIRED_KIT_SCHEMA or manifest.get("workspaceDigest") != sha256_json(body):
        raise ValueError("public paired study workspace digest mismatch")
    if any((
        manifest.get("authority") != PAIRED_AUTHORITY,
        manifest.get("model") != MODEL,
        manifest.get("agentGoldIsolation") is not True,
        manifest.get("pairedStudyInputEligible") is not True,
        manifest.get("pairedExecutionCompleted") is not False,
        manifest.get("translationReportAttached") is not False,
        manifest.get("pairedExecutionInputEligible") is not False,
        manifest.get("containsModelSemanticCandidates") is not False,
        manifest.get("thirdPartyExecutionAttempted") is not False,
        manifest.get("officialEsP1QualificationEligible") is not False,
    )):
        raise ValueError("public paired study authority boundary mismatch")
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path == root / "workspace.json":
            continue
        if path.is_symlink():
            raise ValueError("public paired study workspace cannot contain symlinks")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = _file_digest(path)
    if actual != manifest.get("sealedFiles"):
        raise ValueError("public paired study sealed file set or digest drift")
    plan = json.loads((root / "study-plan.json").read_text(encoding="utf-8"))
    if (
        plan != _study_plan()
        or plan["controls"]["goldVisibleToAgent"] is not False
        or plan["controls"]["unqualifiedTreatmentMutationAllowed"] is not False
    ):
        raise ValueError("public paired study plan drift")
    agent_cases = _jsonl(root / "agent/cases.jsonl")
    scoring_cases = _jsonl(root / "scoring/gold.jsonl")
    provenance = _jsonl(root / "evidence/provenance.jsonl")
    agent_ids = [item.get("caseId") for item in agent_cases]
    scoring_ids = [item.get("caseId") for item in scoring_cases]
    provenance_ids = [item.get("caseId") for item in provenance]
    if (
        len(agent_ids) != manifest["caseCount"]
        or len(set(agent_ids)) != len(agent_ids)
        or agent_ids != scoring_ids
        or agent_ids != provenance_ids
    ):
        raise ValueError("public paired study case coverage mismatch")
    package_ids: set[str] = set()
    for case in agent_cases:
        if set(case) != {
            "apiVersion", "caseId", "assignmentId", "packageId", "packageDigest",
            "runtimePackageDigest", "language", "challenge", "userPrompt", "fixtureRefs", "toolCatalogRef",
            "executionProfile", "skill", "authority",
        }:
            raise ValueError("public paired Agent case contains forbidden fields")
        if case["apiVersion"] != PAIRED_CASE_SCHEMA or case["authority"] != PAIRED_AUTHORITY:
            raise ValueError("public paired Agent case boundary mismatch")
        package_ids.add(case["packageId"])
        package = root / "agent/packages" / case["packageId"]
        report = inspect_skill_package(package)
        if report["gate"] != "passed" or report["packageDigest"] != case["runtimePackageDigest"]:
            raise ValueError("public paired Agent package drift")
        if case["skill"]["entry"] != f"packages/{case['packageId']}/SKILL.md":
            raise ValueError("public paired Agent package entry mismatch")
        catalog = json.loads((root / "agent" / case["toolCatalogRef"]).read_text(encoding="utf-8"))
        if (
            catalog.get("apiVersion") not in {
                "effect-runtime.io/public-skill-tool-catalog/v1", FIXTURE_CATALOG_SCHEMA,
            }
            or catalog.get("assignmentId") != case["assignmentId"]
        ):
            raise ValueError("public paired Tool Catalog binding mismatch")
        capability_ids = _capability_ids(catalog)
        profile = _fixture_execution_profile(catalog, case, root / "agent")
        if case["executionProfile"] != profile:
            raise ValueError("public paired fixture execution profile mismatch")
        scoring = scoring_cases[agent_ids.index(case["caseId"])]
        if not set(scoring["gold"]["requiredCapabilities"]).issubset(capability_ids):
            raise ValueError("public paired Tool Catalog misses a Gold-required capability")
        for relative in case["fixtureRefs"]:
            if not (root / "agent" / relative).is_file():
                raise ValueError("public paired Agent fixture is missing")
    for scoring in scoring_cases:
        if set(scoring) != {"apiVersion", "caseId", "gold", "oracle", "authority"}:
            raise ValueError("public paired scoring case fields mismatch")
        if scoring["apiVersion"] != PAIRED_GOLD_SCHEMA or scoring["authority"] != PAIRED_AUTHORITY:
            raise ValueError("public paired scoring boundary mismatch")
        if (
            scoring["gold"].get("taskId") != scoring["caseId"]
            or scoring["oracle"].get("taskId") != scoring["caseId"]
        ):
            raise ValueError("public paired scoring task binding mismatch")
    if len(package_ids) != manifest["packageCount"]:
        raise ValueError("public paired package coverage mismatch")
    executable = sum(
        item["executionProfile"] == "declarative_fixture_mcp_v1" for item in agent_cases
    )
    if (
        manifest.get("fixtureMcpExecutableCaseCount") != executable
        or manifest.get("fixtureMcpPendingCaseCount") != len(agent_cases) - executable
        or manifest.get("fixtureMcpInputEligible") != (executable == len(agent_cases))
    ):
        raise ValueError("public paired execution readiness mismatch")
    return {
        "status": "valid", "workspaceDigest": manifest["workspaceDigest"],
        "caseCount": manifest["caseCount"], "packageCount": manifest["packageCount"],
        "model": MODEL, "agentGoldIsolation": True, "pairedStudyInputEligible": True,
        "fixtureMcpExecutableCaseCount": executable,
        "fixtureMcpPendingCaseCount": len(agent_cases) - executable,
        "fixtureMcpInputEligible": executable == len(agent_cases),
        "translationReportAttached": False, "pairedExecutionInputEligible": False,
        "pairedExecutionCompleted": False, "thirdPartyExecutionAttempted": False,
        "officialEsP1QualificationEligible": False, "authority": PAIRED_AUTHORITY,
        "claimBoundary": manifest["claimBoundary"],
    }


def inspect_public_paired_agent_inputs(root_path: str | Path) -> dict[str, Any]:
    """Validate execution inputs without parsing scoring Gold/Oracle.

    The sealed-file map still detects any scoring-file drift, but this path
    deliberately treats those bytes as opaque so a pre-run process cannot use
    semantic Gold while constructing prompts or execution routes.
    """

    root = Path(root_path).expanduser().resolve()
    manifest = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "workspaceDigest"}
    if manifest.get("apiVersion") != PAIRED_KIT_SCHEMA or manifest.get("workspaceDigest") != sha256_json(body):
        raise ValueError("public paired Agent-input workspace digest mismatch")
    if any((
        manifest.get("authority") != PAIRED_AUTHORITY,
        manifest.get("model") != MODEL,
        manifest.get("agentGoldIsolation") is not True,
        manifest.get("pairedStudyInputEligible") is not True,
        manifest.get("pairedExecutionCompleted") is not False,
        manifest.get("translationReportAttached") is not False,
        manifest.get("pairedExecutionInputEligible") is not False,
        manifest.get("containsModelSemanticCandidates") is not False,
        manifest.get("thirdPartyExecutionAttempted") is not False,
        manifest.get("officialEsP1QualificationEligible") is not False,
    )):
        raise ValueError("public paired Agent-input authority boundary mismatch")
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path == root / "workspace.json":
            continue
        if path.is_symlink():
            raise ValueError("public paired Agent-input workspace cannot contain symlinks")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = _file_digest(path)
    if actual != manifest.get("sealedFiles"):
        raise ValueError("public paired Agent-input sealed file set or digest drift")
    plan = json.loads((root / "study-plan.json").read_text(encoding="utf-8"))
    if plan != _study_plan() or plan["controls"]["goldVisibleToAgent"] is not False:
        raise ValueError("public paired Agent-input study plan drift")
    cases = _jsonl(root / "agent/cases.jsonl")
    if len(cases) != manifest["caseCount"] or len({item.get("caseId") for item in cases}) != len(cases):
        raise ValueError("public paired Agent-input case coverage mismatch")
    package_ids: set[str] = set()
    executable = 0
    for case in cases:
        if set(case) != {
            "apiVersion", "caseId", "assignmentId", "packageId", "packageDigest",
            "runtimePackageDigest", "language", "challenge", "userPrompt", "fixtureRefs", "toolCatalogRef",
            "executionProfile", "skill", "authority",
        }:
            raise ValueError("public paired Agent case contains forbidden fields")
        if case["apiVersion"] != PAIRED_CASE_SCHEMA or case["authority"] != PAIRED_AUTHORITY:
            raise ValueError("public paired Agent case boundary mismatch")
        package_ids.add(case["packageId"])
        package = root / "agent/packages" / case["packageId"]
        report = inspect_skill_package(package)
        if report["gate"] != "passed" or report["packageDigest"] != case["runtimePackageDigest"]:
            raise ValueError("public paired Agent package drift")
        if case["skill"]["entry"] != f"packages/{case['packageId']}/SKILL.md":
            raise ValueError("public paired Agent package entry mismatch")
        catalog = json.loads((root / "agent" / case["toolCatalogRef"]).read_text(encoding="utf-8"))
        if catalog.get("assignmentId") != case["assignmentId"]:
            raise ValueError("public paired Agent Tool Catalog binding mismatch")
        _capability_ids(catalog)
        profile = _fixture_execution_profile(catalog, case, root / "agent")
        if case["executionProfile"] != profile:
            raise ValueError("public paired Agent fixture execution profile mismatch")
        executable += profile == "declarative_fixture_mcp_v1"
        for relative in case["fixtureRefs"]:
            if not (root / "agent" / relative).is_file():
                raise ValueError("public paired Agent fixture is missing")
    if len(package_ids) != manifest["packageCount"]:
        raise ValueError("public paired Agent package coverage mismatch")
    if (
        executable != manifest["fixtureMcpExecutableCaseCount"]
        or manifest["fixtureMcpInputEligible"] != (executable == len(cases))
    ):
        raise ValueError("public paired Agent execution readiness mismatch")
    return {
        "status": "valid", "workspaceDigest": manifest["workspaceDigest"],
        "caseCount": len(cases), "packageCount": len(package_ids), "model": MODEL,
        "agentGoldIsolation": True, "pairedStudyInputEligible": True,
        "fixtureMcpExecutableCaseCount": executable,
        "fixtureMcpInputEligible": executable == len(cases),
        "goldParsed": False, "officialEsP1QualificationEligible": False,
        "authority": PAIRED_AUTHORITY,
    }


__all__ = [
    "MODEL", "PAIRED_AUTHORITY", "PAIRED_CASE_SCHEMA", "PAIRED_GOLD_SCHEMA",
    "PAIRED_KIT_SCHEMA", "export_public_paired_study_kit", "inspect_public_paired_agent_inputs",
    "inspect_public_paired_study_kit",
]
