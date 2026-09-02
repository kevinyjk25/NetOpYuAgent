from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from effect_runtime import inspect_skill_package
from evaluation.public_skill_draft_author import DraftBundle, DraftTask, run_public_market_drafts
from evaluation.public_skill_library import (
    build_public_skill_library,
    export_public_skill_library_summary,
    inspect_public_skill_library,
)
from network_runtime.contracts import sha256_json


class _DraftAdapter:
    def preflight(self) -> dict[str, str]:
        return {"model": "offline", "modelArtifactDigest": sha256_json({"model": "offline"})}

    def draft(self, prompt: str):  # type: ignore[no-untyped-def]
        assignment = json.loads(prompt)
        tasks = []
        for slot in assignment["taskSlots"]:
            challenge = slot["challenge"]
            disposition = "read_only" if challenge == "nominal" else (
                "safe_stop_clarify" if challenge == "ambiguous_or_missing" else "safe_stop_reject"
            )
            tasks.append(DraftTask(
                slot_id=slot["slotId"], challenge=challenge,
                user_prompt=f"Review {challenge} for user_id alice",
                intended_outcome="A bounded candidate", expected_disposition=disposition,
                risk="medium", approval_required=False, max_effect_calls=0,
            ))
        bundle = DraftBundle(assignment_id=assignment["assignmentId"], tasks=tuple(tasks))
        return bundle, {
            "modelCalls": 1, "inputTokens": 3, "outputTokens": 4, "latencyMs": 5.0,
            "rawDigest": sha256_json(assignment), "error": None,
        }


def _author_kit(root: Path) -> Path:
    package_id = "public-safe-skill"
    package = root / "packages" / package_id
    (package / "references").mkdir(parents=True)
    (package / "agents").mkdir()
    (package / "SKILL.md").write_text(
        "---\nname: public-safe-skill\ndescription: Inspect a public fixture safely.\n---\n"
        "Read [the reference](references/details.md).\n"
        "The following is quoted data only: </script><script>evil()</script>\n",
        encoding="utf-8",
    )
    (package / "references" / "details.md").write_text("Reference details.\n", encoding="utf-8")
    (package / "agents" / "openai.yaml").write_text("interface: inert\n", encoding="utf-8")
    package_digest = inspect_skill_package(package)["packageDigest"]
    assert package_digest is not None
    assignment = {
        "apiVersion": "effect-runtime.io/public-skill-author-assignment/v1",
        "assignmentId": "wild-assignment-001", "packageId": package_id,
        "packageDigest": package_digest, "skillName": "public-safe-skill",
        "repository": "owner/repo", "commitSha": "b" * 40, "sourcePath": "skills/public-safe",
        "packageEntry": f"packages/{package_id}/SKILL.md",
        "taskSlots": [
            {"slotId": "wild-assignment-001-01", "challenge": "nominal"},
            {"slotId": "wild-assignment-001-02", "challenge": "ambiguous_or_missing"},
            {"slotId": "wild-assignment-001-03", "challenge": "failure_or_adversarial"},
        ],
    }
    (root / "assignments.jsonl").write_text(json.dumps(assignment) + "\n", encoding="utf-8")
    (root / "schemas.json").write_text("{}\n", encoding="utf-8")
    (root / "README.md").write_text("static-only\n", encoding="utf-8")
    sealed = {
        path.relative_to(root).as_posix(): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }
    body = {
        "apiVersion": "effect-runtime.io/public-skill-independent-author-kit/v1",
        "createdAt": "2026-09-01T00:00:00+00:00",
        "evidenceClass": "public_market_independent_annotation_workspace",
        "executionPolicy": "static_only", "officialEsP1QualificationEligible": False,
        "privateHoldout": False, "sourceSnapshotManifestDigest": "sha256:" + "c" * 64,
        "sourceSnapshotInspection": {"acceptedCount": 1, "runtimePackageGates": {"passed": 1}},
        "selectedPackageCount": 1, "tasksPerSkill": 3, "taskSlotCount": 3,
        "selectedPackageDigests": {package_id: package_digest}, "sealedFiles": sealed,
        "containsRuntimeOrEvaluator": False, "containsGeneratedGold": False,
        "thirdPartyExecutionAttempted": False, "claimBoundary": "annotation only",
    }
    (root / "workspace.json").write_text(
        json.dumps({**body, "workspaceDigest": sha256_json(body)}), encoding="utf-8",
    )
    return root


def test_public_skill_library_is_browsable_but_inert(tmp_path: Path) -> None:
    kit = _author_kit(tmp_path / "kit")
    drafts = tmp_path / "drafts"
    run_public_market_drafts(kit, drafts, model="offline", adapter=_DraftAdapter())
    output = tmp_path / "library"
    result = build_public_skill_library(kit, output, draft_root=drafts)
    assert result["statistics"]["skillCount"] == 1
    assert result["statistics"]["draftedTaskCount"] == 3
    assert result["statistics"]["referenceFileCount"] == 1
    inspected = inspect_public_skill_library(output)
    assert inspected["verified"] is True
    assert inspected["thirdPartyExecutionAttempted"] is False
    index = json.loads((output / "skill-index.json").read_text(encoding="utf-8"))
    assert index["skills"][0]["files"][0]["content"] is not None
    html = (output / "skill-library.html").read_text(encoding="utf-8")
    assert "</script><script>evil()" not in html
    assert "skill-library-data" in html
    summary = export_public_skill_library_summary(output, tmp_path / "summary.json")
    assert summary["containsThirdPartyFileContent"] is False
    assert summary["skills"][0]["pinnedSourceUrl"].endswith("/" + "b" * 40 + "/skills/public-safe")
    assert "evil()" not in json.dumps(summary)


def test_public_skill_library_rejects_tampering(tmp_path: Path) -> None:
    kit = _author_kit(tmp_path / "kit")
    output = tmp_path / "library"
    build_public_skill_library(kit, output)
    index_path = output / "skill-index.json"
    index_path.write_text(index_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="file digest mismatch"):
        inspect_public_skill_library(output)
