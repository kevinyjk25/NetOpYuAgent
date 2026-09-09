from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.translation_corpus import (
    build_translation_corpus,
    inspect_translation_corpus,
)


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _package(
    root: Path, package_id: str, repository: str, skill: bytes,
) -> dict[str, object]:
    package = root / "packages" / package_id
    package.mkdir(parents=True)
    (package / "SKILL.md").write_bytes(skill)
    files = [{"path": "SKILL.md", "bytes": len(skill), "sha256": _digest(skill)}]
    return {
        "status": "accepted",
        "candidateId": package_id,
        "packageId": package_id,
        "name": package_id,
        "repository": repository,
        "sourcePath": f"skills/{package_id}",
        "commitSha": "a" * 40,
        "packageDigest": "sha256:source-package",
        "licenseSpdx": "MIT",
        "language": "en",
        "instructionRiskCodes": [],
        "materializedExecutableFiles": False,
        "files": files,
    }


def test_translation_corpus_separates_primary_runtime_and_robustness_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    standard = b"---\nname: standard\ndescription: Standard Skill.\n---\nRead records.\n"
    partial = (
        b"---\nname: partial\ndescription: Partial Skill.\n---\n"
        b"Follow [the policy](references/missing.md).\n"
    )
    invalid = b"No frontmatter. This remains robustness input only.\n"
    rows = [
        _package(snapshot, "standard", "owner/a", standard),
        _package(snapshot, "partial", "owner/b", partial),
        _package(snapshot, "variant", "owner/c", invalid),
    ]
    (snapshot / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evaluation.translation_corpus.inspect_public_snapshot",
        lambda _: {"manifestDigest": "sha256:snapshot"},
    )
    output = tmp_path / "corpus"
    build_translation_corpus(snapshot, output, batch_size=1)
    report = inspect_translation_corpus(output)
    assert report["skillCount"] == 3
    assert report["primaryEligibleCount"] == 2
    assert report["runtimeReadyCount"] == 1
    assert report["robustnessOnlyCount"] == 1
    assert report["proofCohortEligible"] is False
    assert report["runtimeAuthorityGranted"] is False
    assert report["thirdPartyExecutionAttempted"] is False
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    classifications = {row["packageId"]: row["classification"] for row in index["skills"]}
    assert classifications == {
        "partial": "translation_only_partial_context",
        "standard": "runtime_ready",
        "variant": "format_variant_robustness_only",
    }
    assert (output / "skill-library.html").is_file()


def test_translation_corpus_detects_index_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    row = _package(
        snapshot,
        "standard",
        "owner/a",
        b"---\nname: standard\ndescription: Standard Skill.\n---\nRead.\n",
    )
    (snapshot / "records.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "evaluation.translation_corpus.inspect_public_snapshot",
        lambda _: {"manifestDigest": "sha256:snapshot"},
    )
    output = tmp_path / "corpus"
    build_translation_corpus(snapshot, output)
    (output / "index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed file drift"):
        inspect_translation_corpus(output)


@pytest.mark.parametrize("mode,valid,primary,classification", [
    ("100755", True, True, "translation_only_partial_context"),
    ("100755", False, False, "format_variant_robustness_only"),
    ("120000", True, False, "excluded_from_translation"),
])
def test_quarantined_markdown_qualification_never_restores_runtime_resources(
    tmp_path, monkeypatch, mode, valid, primary, classification,
):
    import evaluation.public_skill_corpus as corpus

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    row = _package(snapshot, "entry", "owner/repo", b"temporary")
    (snapshot / "packages/entry/SKILL.md").unlink()
    row["files"] = []
    data = b"---\nname: entry\ndescription: Static data.\n---\nRead.\n" if valid else b"bad format"
    evidence = corpus._quarantine_text(snapshot, "entry", {"relative": "SKILL.md", "mode": mode}, data)
    row.update(quarantinedFiles=[evidence], withheldFiles=[{"path": "SKILL.md", "reason": "executable-mode"}])
    (snapshot / "records.jsonl").write_text(json.dumps(row) + "\n")
    (snapshot / "manifest.json").write_text("{}")
    monkeypatch.setattr("evaluation.translation_corpus.inspect_public_snapshot", lambda _: {"manifestDigest": "test"})
    build_translation_corpus(snapshot, tmp_path / "library")
    skill = json.loads((tmp_path / "library/index.json").read_text())["skills"][0]
    assert skill["primaryTranslationEligible"] is primary
    assert skill["classification"] == classification
    assert not skill["runtimeReady"]
    assert not (snapshot / "packages/entry/SKILL.md").exists()
