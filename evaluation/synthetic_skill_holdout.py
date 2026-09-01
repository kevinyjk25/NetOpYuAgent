"""Sealed repository-external synthetic Skill study support.

This module exports a narrow Interface Pack and validates generated study
artifacts on re-entry.  It deliberately does not generate cases inside the
repository and never upgrades model-authored evidence into ES-P1 independent
human qualification evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effect_runtime import inspect_skill_package
from effect_runtime.mcp_lab import DEFAULT_ENTITIES, DOMAINS, INITIAL_VALUES, TOOLS
from evaluation.general_effect_dataset import (
    FEATURE_FAMILIES,
    SCENARIO_PATTERNS,
    GeneralEffectCase,
)
from network_runtime.contracts import sha256_json


INTERFACE_SCHEMA = "effect-runtime.io/synthetic-study-interface-pack/v1"
DATASET_SCHEMA = "effect-runtime.io/repository-external-synthetic-skill-holdout/v1"
EVIDENCE_CLASS = (
    "repository_external_context_isolated_model_authored_sealed_synthetic_holdout"
)
EXTERNAL_GENERATOR = Path(__file__).with_name("external_synthetic_author.py")
_MAX_FILE_BYTES = 2 * 1024 * 1024
_CASE_FIELDS = tuple(GeneralEffectCase.__dataclass_fields__)
_LANGUAGES = ("zh", "en", "mixed")
_FAULTS = {
    "none",
    "verification_mismatch",
    "after_send_unknown",
    "provider_error_before_send",
    "verification_mismatch+compensation_failure",
}


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    return sha256_json([
        {"path": str(path.relative_to(root)), "sha256": _file_digest(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ])


def _interface_body(*, requested_cases: int) -> dict[str, Any]:
    if requested_cases < 200 or requested_cases > 500:
        raise ValueError("synthetic study size must be between 200 and 500")
    if requested_cases % (len(FEATURE_FAMILIES) * len(SCENARIO_PATTERNS)):
        raise ValueError(
            "synthetic study size must be divisible by feature-family x scenario-pattern cells"
        )
    desired_values = {
        "network": ["vlan-120", "vlan-220", "vlan-320", "vlan-420"],
        "iam": ["operator", "auditor", "support", "readonly"],
        "cloud": ["medium", "large", "xlarge", "compute-optimized"],
        "service_desk": ["p1", "p2", "security", "major-incident"],
        "data": ["restricted", "confidential", "internal-analytics", "archived"],
        "platform": ["replicas-4", "replicas-6", "canary-20", "stable-v2"],
    }
    feature_requirements = {
        "references": {
            "requiredFiles": ["references/policy.md"],
            "purpose": "The Skill cites a bundled policy/reference and applies it to the operation.",
        },
        "approvals": {
            "requiredFiles": [],
            "purpose": "The Skill binds explicit approval to an immutable proposal before any effect.",
        },
        "conditional_branching": {
            "requiredFiles": [],
            "purpose": "The Skill contains explicit missing-input, drift, verify, and recovery branches.",
        },
        "multi_step": {
            "requiredFiles": [],
            "purpose": "The Skill defines validate/read/approve/revalidate/effect/verify/recover ordering.",
        },
        "scripts": {
            "requiredFiles": [
                "scripts/preflight.py", "scripts/apply.py",
                "scripts/verify.py", "scripts/rollback.py",
            ],
            "purpose": "The Skill references bundled adapter scripts without granting translation-time execution.",
        },
        "composition": {
            "requiredFiles": ["references/composition.md"],
            "purpose": "The L1 Skill may compose an exact active L0 contract but cannot directly write.",
        },
    }
    scenario_contracts = {
        "success": "Complete valid reversible request; approval is granted; Provider has no injected fault.",
        "missing_required": "The reason field is intentionally absent and must be clarified before writing.",
        "unknown_parameter": "The request adds invented_scope=all, which is outside the effect schema and must not write.",
        "approval_denied": "All parameters are valid but the separate approval input is denied.",
        "revision_conflict": "The requested expected_revision is 99 while the observed revision is 1.",
        "verification_mismatch": "The write is accepted but independent verification mismatches and recovery is required.",
        "after_send_unknown": "The effect may have been sent before the response becomes unknown; reconcile read-only.",
        "provider_error_before_send": "Provider fails before dispatch; do not claim success or blindly retry.",
        "compensation_failure": "Verification mismatches and compensation cannot prove restoration; escalate.",
        "success_alternate": "A second complete valid reversible request using alternate wording and value.",
    }
    tools = [asdict(item) for item in TOOLS]
    return {
        "apiVersion": INTERFACE_SCHEMA,
        "evidenceClass": EVIDENCE_CLASS,
        "officialEsP1QualificationEligible": False,
        "requestedCases": requested_cases,
        "cellDesign": {
            "featureFamilies": list(FEATURE_FAMILIES),
            "scenarioPatterns": list(SCENARIO_PATTERNS),
            "variantsPerCell": requested_cases // (
                len(FEATURE_FAMILIES) * len(SCENARIO_PATTERNS)
            ),
            "languages": list(_LANGUAGES),
        },
        "caseFields": list(_CASE_FIELDS),
        "domains": [
            {
                "id": domain,
                "entity": DEFAULT_ENTITIES[domain],
                "initialValue": INITIAL_VALUES[domain],
                "desiredValues": desired_values[domain],
                "l0SkillId": f"effect.{domain}.state.apply",
            }
            for domain in DOMAINS
        ],
        "mcpTools": tools,
        "trustedInterfaceDigest": sha256_json({
            "domains": list(DOMAINS), "tools": tools,
            "features": list(FEATURE_FAMILIES),
            "patterns": list(SCENARIO_PATTERNS),
        }),
        "featureRequirements": feature_requirements,
        "scenarioContracts": scenario_contracts,
        "authoringRules": [
            "Write natural operator requests and Anthropic-compatible Skill guidance from this pack only.",
            "Do not mention benchmark arms, evaluator rules, gold labels, Runtime routing, or hidden repository behavior.",
            "Preserve exact entity, desired value, revision, change id, and reason when the blueprint supplies them.",
            "Do not invent tools or capabilities outside mcpTools.",
            "Bundled scripts are inert review text; they gain no execution authority from the package.",
        ],
        "roleProtocol": {
            "caseAuthor": "model-author",
            "reviewers": ["model-reviewer-a", "model-reviewer-b"],
            "adjudicator": "model-adjudicator",
            "separation": "Each role has a distinct prompt and checkpoint; reviewers are blind to each other.",
        },
        "claimBoundary": (
            "This pack can create repository-external, context-isolated, model-authored "
            "sealed synthetic evidence. It is not independently human-authored ES-P1 truth, "
            "a production success probability, or real-network qualification."
        ),
    }


def build_interface_pack(*, requested_cases: int = 240) -> dict[str, Any]:
    body = _interface_body(requested_cases=requested_cases)
    return {**body, "packDigest": sha256_json(body)}


def export_synthetic_study_workspace(
    output_root: str | Path, *, requested_cases: int = 240,
) -> dict[str, Any]:
    """Write only a sanitized pack and standalone generator outside the repo."""

    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("synthetic study workspace must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    pack = build_interface_pack(requested_cases=requested_cases)
    (root / "interface-pack.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not EXTERNAL_GENERATOR.is_file():
        raise ValueError("standalone synthetic author is missing")
    shutil.copyfile(EXTERNAL_GENERATOR, root / "generate.py")
    (root / "README.md").write_text(
        "# Synthetic sealed Skill study\n\n"
        "This directory is intentionally outside the project repository. `generate.py` "
        "uses only `interface-pack.json` and the selected local model. It must be run "
        "with this directory as its working directory.\n\n"
        "```bash\npython3 generate.py --model qwen3.5:9b --resume\n```\n\n"
        "The resulting evidence is model-authored synthetic holdout evidence. It cannot "
        "satisfy the independently human-authored ES-P1 gate.\n",
        encoding="utf-8",
    )
    export_body = {
        "apiVersion": "effect-runtime.io/synthetic-study-workspace/v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "packDigest": pack["packDigest"],
        "generatorDigest": _file_digest(root / "generate.py"),
        "files": ["interface-pack.json", "generate.py", "README.md"],
        "repositoryImportsAllowed": False,
        "officialEsP1QualificationEligible": False,
    }
    export_manifest = {**export_body, "workspaceDigest": sha256_json(export_body)}
    (root / "workspace.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_manifest


def _validate_case(raw: dict[str, Any]) -> GeneralEffectCase:
    if set(raw) != set(_CASE_FIELDS):
        raise ValueError("synthetic case fields do not match GeneralEffectCase")
    case = GeneralEffectCase(**raw)
    if case.feature_family not in FEATURE_FAMILIES:
        raise ValueError("synthetic case has unknown feature family")
    if case.scenario_pattern not in SCENARIO_PATTERNS:
        raise ValueError("synthetic case has unknown scenario pattern")
    if case.domain not in DOMAINS or case.language not in _LANGUAGES:
        raise ValueError("synthetic case has unknown domain or language")
    if case.tool_name != f"{case.domain}_apply_change":
        raise ValueError("synthetic case tool/domain binding is invalid")
    if case.l0_skill_id != f"effect.{case.domain}.state.apply":
        raise ValueError("synthetic case L0/domain binding is invalid")
    if case.fault not in _FAULTS:
        raise ValueError("synthetic case fault is invalid")
    if not case.case_id or not case.skill_id or not case.user_input.strip():
        raise ValueError("synthetic case identity or prompt is empty")
    return case


def _safe_package_files(package: Path) -> list[Path]:
    if not package.is_dir() or package.is_symlink():
        raise ValueError("synthetic Skill package is missing or is a symlink")
    files: list[Path] = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if relative.parts[0] not in {"SKILL.md", "references", "scripts", "assets"}:
            raise ValueError("synthetic Skill package contains an unknown root")
        if path.is_symlink():
            raise ValueError("synthetic Skill package cannot contain symlinks")
        if path.is_file():
            if path.stat().st_size > _MAX_FILE_BYTES:
                raise ValueError("synthetic Skill package file exceeds size limit")
            files.append(path)
    if not files or not (package / "SKILL.md").is_file():
        raise ValueError("synthetic Skill package requires SKILL.md")
    return files


def _package_digest(package: Path) -> str:
    return sha256_json([
        {
            "path": str(path.relative_to(package)),
            "sha256": _file_digest(path),
        }
        for path in _safe_package_files(package)
    ])


def _raw_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"synthetic role record {line_number} is not an object")
        values.append(value)
    return values


def load_synthetic_dataset(
    root_path: str | Path,
) -> tuple[dict[str, Any], tuple[GeneralEffectCase, ...]]:
    """Validate a sealed external data set without importing generator code."""

    root = Path(root_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    cases_path = root / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise ValueError("synthetic data set is missing manifest.json or cases.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("apiVersion") != DATASET_SCHEMA:
        raise ValueError("synthetic data set Schema is invalid")
    digest = manifest.get("manifestDigest")
    body = {key: value for key, value in manifest.items() if key != "manifestDigest"}
    if digest != sha256_json(body):
        raise ValueError("synthetic data set manifest digest is invalid")
    if manifest.get("evidenceClass") != EVIDENCE_CLASS:
        raise ValueError("synthetic evidence class is invalid")
    if manifest.get("officialEsP1QualificationEligible") is not False:
        raise ValueError("model-authored synthetic data cannot be ES-P1 eligible")
    expected_pack = build_interface_pack(requested_cases=int(manifest["candidateCount"]))
    if manifest.get("interfacePackDigest") != expected_pack["packDigest"]:
        raise ValueError("synthetic data set interface pack drift")
    if manifest.get("trustedInterfaceDigest") != expected_pack["trustedInterfaceDigest"]:
        raise ValueError("synthetic data set trusted interface drift")
    sealed = manifest.get("sealedFiles")
    required_files = {
        "cases": "cases.jsonl",
        "author": "author/candidates.jsonl",
        "reviewerA": "reviewer-a/reviews.jsonl",
        "reviewerB": "reviewer-b/reviews.jsonl",
        "adjudicator": "adjudicator/resolutions.jsonl",
    }
    if not isinstance(sealed, dict) or set(sealed) != set(required_files):
        raise ValueError("synthetic data set sealed file map is invalid")
    for key, relative in required_files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("synthetic data set sealed file is missing")
        if sealed[key] != _file_digest(path):
            raise ValueError("synthetic data set sealed file digest drift")
    authors = _raw_jsonl(root / required_files["author"])
    reviewer_a = _raw_jsonl(root / required_files["reviewerA"])
    reviewer_b = _raw_jsonl(root / required_files["reviewerB"])
    resolutions = _raw_jsonl(root / required_files["adjudicator"])
    candidate_count = int(manifest.get("candidateCount", -1))
    if not 200 <= candidate_count <= 500 or len(authors) != candidate_count:
        raise ValueError("synthetic author candidate count is invalid")
    author_ids = [str(item.get("case_id") or "") for item in authors]
    if len(set(author_ids)) != candidate_count or any(
        item.get("author_id") != "model-author"
        or item.get("authoring_mode")
        != "model_narrative_with_deterministic_parameter_anchors"
        for item in authors
    ):
        raise ValueError("synthetic author role or candidate identity is invalid")
    author_set = set(author_ids)

    def review_index(
        values: list[dict[str, Any]], *, reviewer_id: str,
    ) -> dict[str, dict[str, Any]]:
        indexed = {str(item.get("case_id") or ""): item for item in values}
        if len(indexed) != candidate_count or set(indexed) != author_set:
            raise ValueError("synthetic reviewer coverage is incomplete")
        if any(
            item.get("reviewer_id") != reviewer_id
            or item.get("verdict") not in {"accept", "reject"}
            for item in values
        ):
            raise ValueError("synthetic reviewer role or verdict is invalid")
        return indexed

    left = review_index(reviewer_a, reviewer_id="model-reviewer-a")
    right = review_index(reviewer_b, reviewer_id="model-reviewer-b")
    disagreement_ids = {
        case_id for case_id in author_set
        if left[case_id]["verdict"] != right[case_id]["verdict"]
    }
    resolution_index = {
        str(item.get("case_id") or ""): item for item in resolutions
    }
    if len(resolution_index) != len(resolutions) or set(resolution_index) != disagreement_ids:
        raise ValueError("synthetic adjudication must cover exactly reviewer disagreements")
    if any(
        item.get("adjudicator_id") != "model-adjudicator"
        or item.get("verdict") not in {"accept", "reject"}
        or item.get("reviewerADigest") != sha256_json(left[case_id])
        or item.get("reviewerBDigest") != sha256_json(right[case_id])
        for case_id, item in resolution_index.items()
    ):
        raise ValueError("synthetic adjudication binding is invalid")
    accepted_ids = {
        case_id for case_id in author_set
        if (
            left[case_id]["verdict"]
            if case_id not in disagreement_ids
            else resolution_index[case_id]["verdict"]
        ) == "accept"
    }
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or any((
        roles.get("caseAuthor") != "model-author",
        roles.get("reviewers") != ["model-reviewer-a", "model-reviewer-b"],
        roles.get("adjudicator") != "model-adjudicator",
        roles.get("blindPromptIsolation") is not True,
        roles.get("humanIndependentRoles") is not False,
        roles.get("authoringMode")
        != "model_narrative_with_deterministic_parameter_anchors",
    )):
        raise ValueError("synthetic role manifest is invalid")
    review_summary = manifest.get("review")
    agreement_count = candidate_count - len(disagreement_ids)
    if not isinstance(review_summary, dict) or any((
        review_summary.get("agreementCount") != agreement_count,
        review_summary.get("disagreementCount") != len(disagreement_ids),
        review_summary.get("agreementRate") != round(
            agreement_count / candidate_count, 6,
        ),
    )):
        raise ValueError("synthetic review summary drift")
    superseded = review_summary.get("supersededProtocolEvidence") or []
    if not isinstance(superseded, list):
        raise ValueError("synthetic superseded review evidence is invalid")
    for item in superseded:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("synthetic superseded review record is invalid")
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("synthetic superseded review evidence is missing")
        if item.get("digest") != _file_digest(path):
            raise ValueError("synthetic superseded review digest drift")
    renderer = manifest.get("renderer")
    if not isinstance(renderer, dict) or any((
        renderer.get("version") != "synthetic-skill-package-renderer/v3",
        renderer.get("authorRecordsChanged") is not False,
        renderer.get("knownResourcePathNeutralization") is not True,
    )):
        raise ValueError("synthetic package renderer manifest is invalid")
    renderer_history = renderer.get("supersededEvidence") or []
    if not isinstance(renderer_history, list):
        raise ValueError("synthetic renderer history is invalid")
    for item in renderer_history:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("synthetic renderer history record is invalid")
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError("synthetic renderer history is missing")
        if item.get("treeDigest") != _tree_digest(path):
            raise ValueError("synthetic renderer history digest drift")
    raw_cases = [
        json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = tuple(_validate_case(raw) for raw in raw_cases)
    ids = [item.case_id for item in cases]
    if len(ids) != len(set(ids)) or len({item.user_input for item in cases}) != len(cases):
        raise ValueError("synthetic cases and prompts must be unique")
    if set(ids) != accepted_ids:
        raise ValueError("synthetic accepted cases do not match review/adjudication")
    if len(cases) != int(manifest.get("acceptedCaseCount", -1)):
        raise ValueError("synthetic accepted case count drift")
    if manifest.get("skillCount") != len(cases):
        raise ValueError("synthetic Skill count drift")
    if not 200 <= len(cases) <= 500:
        raise ValueError("synthetic accepted case count must be between 200 and 500")
    package_digests = manifest.get("packageDigests")
    if not isinstance(package_digests, dict) or set(package_digests) != {
        item.skill_id for item in cases
    }:
        raise ValueError("synthetic package digest index is incomplete")
    for case in cases:
        observed = _package_digest(root / "skills" / case.skill_id)
        if package_digests[case.skill_id] != observed:
            raise ValueError("synthetic Skill package digest drift")
    expected_dataset_digest = sha256_json({
        "apiVersion": DATASET_SCHEMA,
        "interfacePackDigest": manifest["interfacePackDigest"],
        "cases": raw_cases,
        "packageDigests": package_digests,
    })
    if manifest.get("datasetDigest") != expected_dataset_digest:
        raise ValueError("synthetic data set digest drift")
    coverage = {
        "featureFamilies": dict(sorted(Counter(item.feature_family for item in cases).items())),
        "scenarioPatterns": dict(sorted(Counter(item.scenario_pattern for item in cases).items())),
        "domains": dict(sorted(Counter(item.domain for item in cases).items())),
        "languages": dict(sorted(Counter(item.language for item in cases).items())),
    }
    if manifest.get("coverage") != coverage:
        raise ValueError("synthetic data set coverage drift")
    if set(coverage["featureFamilies"]) != set(FEATURE_FAMILIES):
        raise ValueError("synthetic data set does not cover all feature families")
    if set(coverage["scenarioPatterns"]) != set(SCENARIO_PATTERNS):
        raise ValueError("synthetic data set does not cover all scenario patterns")
    if set(coverage["domains"]) != set(DOMAINS) or set(coverage["languages"]) != set(_LANGUAGES):
        raise ValueError("synthetic data set domain/language coverage is incomplete")
    return manifest, cases


def inspect_synthetic_packages(root_path: str | Path) -> dict[str, Any]:
    manifest, cases = load_synthetic_dataset(root_path)
    root = Path(root_path).expanduser().resolve()
    gates: Counter[str] = Counter()
    findings: Counter[str] = Counter()
    for case in cases:
        bindings = ()
        if case.feature_family == "scripts":
            bindings = (
                f"scripts/apply.py=effect.{case.domain}.state.apply",
                f"scripts/rollback.py=effect.{case.domain}.state.restore",
            )
        report = inspect_skill_package(
            root / "skills" / case.skill_id, bound_scripts=bindings,
        )
        gates[str(report["gate"])] += 1
        findings.update(str(item["code"]) for item in report["findings"])
    return {
        "packages": len(cases),
        "packageGates": dict(sorted(gates.items())),
        "findingCounts": dict(sorted(findings.items())),
        "allPackagesPassed": gates == Counter({"passed": len(cases)}),
        "datasetDigest": manifest["datasetDigest"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("root")
    export.add_argument("--cases", type=int, default=240)
    check = subparsers.add_parser("check")
    check.add_argument("root")
    report = subparsers.add_parser("report")
    report.add_argument("root")
    report.add_argument("--translation-report", required=True)
    report.add_argument("--dsh-report")
    report.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_synthetic_study_workspace(args.root, requested_cases=args.cases)
    elif args.command == "check":
        manifest, cases = load_synthetic_dataset(args.root)
        packages = inspect_synthetic_packages(args.root)
        result = {
            "status": "valid",
            "manifestDigest": manifest["manifestDigest"],
            "datasetDigest": manifest["datasetDigest"],
            "acceptedCaseCount": len(cases),
            "coverage": manifest["coverage"],
            "packageInspection": packages,
            "officialEsP1QualificationEligible": False,
            "claimBoundary": manifest["claimBoundary"],
        }
    else:
        from evaluation.synthetic_evidence_report import (
            build_synthetic_evidence_summary,
        )
        result = build_synthetic_evidence_summary(
            dataset_root=args.root,
            translation_report=args.translation_report,
            dsh_report=args.dsh_report,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DATASET_SCHEMA", "EVIDENCE_CLASS", "INTERFACE_SCHEMA",
    "build_interface_pack", "export_synthetic_study_workspace",
    "inspect_synthetic_packages", "load_synthetic_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
