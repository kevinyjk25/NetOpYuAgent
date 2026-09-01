"""Role-separated external workspace for forward qualification studies.

This delivery layer deliberately stays outside the evaluator fingerprint. It
orchestrates existing strict qualification contracts without changing scoring
semantics or manufacturing private truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from network_runtime.contracts import sha256_json, utc_now

from .forward_qualification import (
    REPORT_SCHEMA,
    ForwardCase,
    ForwardLabel,
    ForwardStudyPlan,
    _load_jsonl,
    _load_study_plan,
    _validate_manifest,
    _validate_study_binding,
    adjudicate_forward_labels,
    evaluator_fingerprint,
    forward_qualification_schemas,
    seal_forward_cases,
)


QUALIFICATION_KIT_SCHEMA = "netopyu.io/promotion-forward-qualification-kit/v1"


def _qualification_kit_file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_forward_qualification_kit(output_root: str | Path) -> dict[str, Any]:
    """Create a role-separated external study workspace without private truth.

    The kit intentionally contains only schemas, marked examples, empty working
    files, and a runbook.  It never creates cases, reviewer truth, enterprise
    identities, or a qualification result on the user's behalf.
    """

    root = Path(output_root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("qualification kit destination must be a directory")
    if root.exists() and any(root.iterdir()):
        raise ValueError("qualification kit destination must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("author", "reviewer-a", "reviewer-b", "adjudicator", "run",
                     "schemas", "templates"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    schemas = forward_qualification_schemas()
    schema_files = {
        "schemas/case.schema.json": schemas["case"],
        "schemas/label.schema.json": schemas["label"],
        "schemas/study-plan.schema.json": schemas["study_plan"],
        "schemas/resolution.schema.json": schemas["resolution"],
        "schemas/observation.schema.json": schemas["observation"],
        "schemas/model-decision.schema.json": schemas["model_decision"],
        "schemas/thresholds.json": schemas["thresholds"],
    }
    for relative, value in schema_files.items():
        (root / relative).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    example_case = ForwardCase(
        case_id="example-do-not-submit",
        family="replace-with-private-family",
        profile="lan",
        language="zh",
        challenge="replace-with-private-challenge",
        split="private_holdout",
        prompt="REPLACE_WITH_INDEPENDENT_PRIVATE_PROMPT",
    )
    example_label = ForwardLabel(
        case_id=example_case.case_id,
        reviewer_id="replace-with-assigned-reviewer",
        disposition="clarify",
        missing_fields=("replace-with-missing-field",),
        semantic_contract=None,
    )
    (root / "templates/case.example.json").write_text(
        json.dumps(
            example_case.model_dump(by_alias=True, mode="json"),
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (root / "templates/label.example.json").write_text(
        json.dumps(
            example_label.model_dump(by_alias=True, mode="json"),
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    readme = """# 外部正向资格工作区 / External Forward Qualification Workspace

## 中文

该目录用于仓库外、独立人员参与的 L1 → L0.5 → L0 正向资格。工具只提供
Schema、角色隔离、密封、盲审、仲裁、重复运行和聚合评分；不会替你生成独立
用例、Reviewer 真值、企业身份或资格结论。

1. Case author 依据真实业务经验独立编写 `author/cases.jsonl`，不得复制公开反向
   校准集或 `templates/` 中的占位内容。至少 200 条、10 个能力族、5 类挑战，
   LAN/DC/WAN 均覆盖，中英文各至少 20 条，split 必须为 `private_holdout`。
2. 在任何模型运行或 Reviewer 看见对方结果前执行 `forward-eval-study-inputs`、
   `forward-eval-study-init` 和 `forward-eval-study-seal`，生成根目录下的
   `study-plan.json` 与 `manifest.json`。
3. 分别为两个预注册 Reviewer 生成盲审包；他们只能写各自的
   `reviewer-a/labels.jsonl`、`reviewer-b/labels.jsonl`。
4. 有分歧时生成 adjudicator 包，并把绑定两份原标签摘要的决定写入
   `adjudicator/resolutions.jsonl`。不得改写 Reviewer 原文件。
5. 对同一不可变模型制品运行至少三次，把结果写入 `run/`；最后运行 Doctor。

每一步后执行：

```bash
scripts/netopyu-l0 forward-eval-study-doctor --root /ABSOLUTE/PATH/TO/THIS-KIT
```

Doctor 只报告计数、摘要、门禁和下一步，不输出 Prompt、Label 或参数值。

## English

This workspace is for a repository-external, independently authored forward study.
The tooling supplies schemas, role separation, sealing, blinded review, adjudication,
repeated execution, and aggregate scoring. It does not manufacture cases, reviewer
truth, enterprise identity, or a qualification outcome. Run the read-only Doctor
after every step; it reports only counts, digests, gates, and the next action.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    for relative in (
        "author/cases.jsonl", "reviewer-a/labels.jsonl",
        "reviewer-b/labels.jsonl", "adjudicator/resolutions.jsonl",
    ):
        (root / relative).write_text("", encoding="utf-8")

    managed = sorted([
        "README.md", *schema_files,
        "templates/case.example.json", "templates/label.example.json",
    ])
    body = {
        "apiVersion": QUALIFICATION_KIT_SCHEMA,
        "createdAt": utc_now(),
        "managedFiles": [
            {"path": relative, "digest": _qualification_kit_file_digest(root / relative)}
            for relative in managed
        ],
        "workingFiles": [
            "author/cases.jsonl", "study-plan.json", "manifest.json",
            "reviewer-a/labels.jsonl", "reviewer-b/labels.jsonl",
            "adjudicator/resolutions.jsonl", "run/report.json",
        ],
        "claimBoundary": (
            "Workspace readiness is not independent truth, model qualification, "
            "enterprise identity proof, or a production success probability."
        ),
    }
    manifest = {**body, "kitDigest": sha256_json(body)}
    (root / "kit-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "root": str(root),
        "kitDigest": manifest["kitDigest"],
        "managedFiles": len(managed),
        "phase": "authoring",
        "nextAction": "Independently author author/cases.jsonl, then run the Doctor.",
        "claimBoundary": body["claimBoundary"],
    }


def inspect_forward_qualification_study(root_path: str | Path) -> dict[str, Any]:
    """Read-only staged Doctor for an external qualification workspace."""

    root = Path(root_path).expanduser().resolve()
    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, status: str, detail: str, **evidence: Any) -> None:
        gates[name] = {"status": status, "detail": detail, **evidence}

    kit_path = root / "kit-manifest.json"
    try:
        raw_kit = json.loads(kit_path.read_text(encoding="utf-8"))
        body = {key: value for key, value in raw_kit.items() if key != "kitDigest"}
        if raw_kit.get("apiVersion") != QUALIFICATION_KIT_SCHEMA:
            raise ValueError("qualification kit Schema is invalid")
        if raw_kit.get("kitDigest") != sha256_json(body):
            raise ValueError("qualification kit manifest digest is invalid")
        managed = raw_kit.get("managedFiles")
        if not isinstance(managed, list) or not managed:
            raise ValueError("qualification kit managed file list is invalid")
        for item in managed:
            relative = str(item["path"])
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError("qualification kit managed file is missing")
            if item.get("digest") != _qualification_kit_file_digest(target):
                raise ValueError("qualification kit managed file digest drift")
        gate("kit_integrity", "passed", "managed schemas and guidance are intact",
             managedFiles=len(managed), kitDigest=raw_kit["kitDigest"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        gate("kit_integrity", "failed", str(error))

    cases_path = root / "author/cases.jsonl"
    cases: list[ForwardCase] | None = None
    preview: dict[str, Any] | None = None
    if not cases_path.is_file() or cases_path.stat().st_size == 0:
        gate("cases", "pending", "author/cases.jsonl is empty")
    else:
        try:
            raw_cases = cases_path.read_text(encoding="utf-8")
            if "REPLACE_WITH" in raw_cases or "example-do-not-submit" in raw_cases:
                raise ValueError("template markers are forbidden in private cases")
            cases = _load_jsonl(cases_path, ForwardCase)
            if any(
                item.case_id.startswith("example-")
                or item.family.startswith("replace-with-")
                or "REPLACE_WITH" in item.prompt
                for item in cases
            ):
                raise ValueError("template markers are forbidden in private cases")
            preview = seal_forward_cases(
                cases_path, dataset_id="doctor-preview", version="v1",
                provenance="independent_forward",
            )
            status = "passed" if preview["qualification_eligible"] else "failed"
            gate(
                "cases", status,
                "independent-forward structural coverage preview",
                caseCount=preview["case_count"],
                familyCount=len(preview["family_counts"]),
                coverageRequirements=preview["coverage_requirements"],
                casesDigest=preview["cases_digest"],
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            gate("cases", "failed", str(error))

    plan_path = root / "study-plan.json"
    plan: ForwardStudyPlan | None = None
    raw_plan: dict[str, Any] | None = None
    if not plan_path.is_file():
        gate("preregistration", "pending", "study-plan.json is not present")
    else:
        try:
            plan, raw_plan = _load_study_plan(plan_path)
            if plan.evaluator_fingerprint != evaluator_fingerprint():
                raise ValueError("pre-registered evaluator fingerprint has drifted")
            gate(
                "preregistration", "passed", "study inputs and separated roles are frozen",
                planDigest=raw_plan["planDigest"], repetitions=plan.repetitions,
                reviewerCount=len(plan.reviewer_ids),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            gate("preregistration", "failed", str(error))

    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] | None = None
    if not manifest_path.is_file():
        gate("sealed_dataset", "pending", "manifest.json is not present")
    elif cases is None or plan is None or raw_plan is None:
        gate("sealed_dataset", "pending", "valid cases and study plan are required first")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_manifest(cases, manifest)
            _validate_study_binding(manifest, plan, raw_plan, cases)
            if not manifest["qualification_eligible"]:
                raise ValueError("sealed dataset does not meet the pre-registered coverage")
            gate(
                "sealed_dataset", "passed", "cases are digest-bound to the study plan",
                manifestDigest=manifest["manifest_digest"],
                caseCount=manifest["case_count"],
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            gate("sealed_dataset", "failed", str(error))

    first_path = root / "reviewer-a/labels.jsonl"
    second_path = root / "reviewer-b/labels.jsonl"
    resolutions_path = root / "adjudicator/resolutions.jsonl"
    labels_ready = all(path.is_file() and path.stat().st_size > 0
                       for path in (first_path, second_path))
    adjudication: dict[str, Any] | None = None
    if not labels_ready:
        gate("independent_review", "pending", "both reviewer label files are required")
    elif manifest is None or plan is None:
        gate("independent_review", "pending", "sealed study is required first")
    else:
        try:
            adjudication = adjudicate_forward_labels(
                cases_path, manifest_path, first_path, second_path,
                study_plan_path=plan_path,
                resolutions_path=(
                    resolutions_path
                    if resolutions_path.is_file() and resolutions_path.stat().st_size > 0
                    else None
                ),
            )
            ready = bool(adjudication["ready_for_holdout_run"])
            gate(
                "independent_review", "passed" if ready else "pending",
                "review consensus is complete" if ready else "review disagreements need resolution",
                consensusCount=adjudication["consensus_count"],
                disagreementCount=adjudication["disagreement_count"],
                resolutionCount=adjudication["resolution_count"],
                consensusLabelsDigest=adjudication["consensus_labels_digest"],
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            gate("independent_review", "failed", str(error))

    report_path = root / "run/report.json"
    report: dict[str, Any] | None = None
    if not report_path.is_file():
        gate("model_run", "pending", "run/report.json is not present")
    elif (
        manifest is None or plan is None or adjudication is None
        or not adjudication.get("ready_for_holdout_run")
    ):
        gate("model_run", "pending", "valid sealed consensus is required first")
    else:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema") != REPORT_SCHEMA:
                raise ValueError("qualification report Schema is invalid")
            if not isinstance(report.get("qualified"), bool):
                raise ValueError("qualification report outcome is invalid")
            report_body = {
                key: value for key, value in report.items() if key != "reportDigest"
            }
            if report.get("reportDigest") != sha256_json(report_body):
                raise ValueError("qualification report integrity check failed")
            dataset = report.get("dataset") or {}
            if int(dataset.get("repetitions") or 0) < 3:
                raise ValueError("qualification report has fewer than three repetitions")
            bindings = {
                "model": plan.model,
                "model_artifact_digest": plan.model_artifact_digest,
                "authoring_protocol_digest": plan.authoring_protocol_digest,
                "catalog_snapshot_digest": plan.catalog_snapshot_digest,
                "evaluator_fingerprint": plan.evaluator_fingerprint,
            }
            if any(report.get(key) != value for key, value in bindings.items()):
                raise ValueError("qualification report has pre-registration binding drift")
            if (
                dataset.get("manifest_digest") != manifest["manifest_digest"]
                or dataset.get("consensus_labels_digest")
                != adjudication["consensus_labels_digest"]
                or dataset.get("case_count") != manifest["case_count"]
                or dataset.get("repetitions") != plan.repetitions
            ):
                raise ValueError("qualification report has dataset or consensus binding drift")
            gate(
                "model_run", "passed",
                "aggregate qualification report is valid",
                qualified=report["qualified"],
                caseCount=dataset.get("case_count"),
                repetitions=dataset.get("repetitions"),
                reportDigest=report.get("reportDigest"),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            gate("model_run", "failed", str(error))

    failed = [name for name, item in gates.items() if item["status"] == "failed"]
    if failed:
        phase = "blocked"
        next_action = "Repair failed gates: " + ", ".join(failed)
    elif gates["cases"]["status"] != "passed":
        phase = "authoring"
        next_action = "Independently author and coverage-check author/cases.jsonl."
    elif gates["preregistration"]["status"] != "passed":
        phase = "preregistration"
        next_action = "Freeze model/protocol/Catalog/evaluator and separated roles."
    elif gates["sealed_dataset"]["status"] != "passed":
        phase = "sealing"
        next_action = "Seal cases against study-plan.json into manifest.json."
    elif gates["independent_review"]["status"] != "passed":
        phase = "blind_review_or_adjudication"
        next_action = "Complete both blind reviews and resolve only disagreements."
    elif gates["model_run"]["status"] == "pending":
        phase = "ready_for_model_run"
        next_action = "Run the immutable model for the pre-registered repetitions."
    else:
        phase = "qualification_complete" if report and report["qualified"] else "qualification_failed"
        next_action = "Review the aggregate report and preserve the immutable evidence bundle."

    return {
        "apiVersion": "netopyu.io/promotion-forward-study-doctor/v1",
        "root": str(root),
        "phase": phase,
        "ok": not failed,
        "readyForModelRun": phase == "ready_for_model_run",
        "qualificationComplete": phase in {"qualification_complete", "qualification_failed"},
        "gates": gates,
        "nextAction": next_action,
        "privacy": "No prompts, labels, semantic contracts, or argument values are emitted.",
        "claimBoundary": (
            "Doctor status is workflow evidence, not independent truth, enterprise "
            "identity proof, model qualification, or a production success probability."
        ),
    }



