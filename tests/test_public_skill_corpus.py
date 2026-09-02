from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.public_skill_corpus as corpus
from network_runtime.contracts import sha256_json


def _discovery(path: Path, candidates: list[dict[str, object]]) -> None:
    body = {
        "apiVersion": corpus.DISCOVERY_SCHEMA,
        "createdAt": "2026-09-01T00:00:00+00:00",
        "source": "SkillsMP", "queries": ["test"], "sortBy": "recent",
        "language": None, "requestedLimit": len(candidates), "maxPerRepository": 5,
        "candidateCount": len(candidates), "candidates": candidates,
        "claimBoundary": "Discovery metadata is not a quality, safety, license, or ES-P1 qualification label.",
    }
    path.write_text(json.dumps({**body, "discoveryDigest": sha256_json(body)}), encoding="utf-8")


def _candidate(identifier: str, path: str) -> dict[str, object]:
    return {
        "id": identifier, "name": identifier, "author": "author",
        "description": "test", "language": "en",
        "githubUrl": f"https://github.com/owner/repo/tree/main/{path}",
        "skillUrl": "https://skillsmp.com/test", "stars": 1,
        "updatedAt": 1, "discoveryQuery": "test",
    }


def test_github_source_parser_rejects_non_github_and_traversal() -> None:
    with pytest.raises(ValueError):
        corpus._parse_github_source("https://example.com/a/b/tree/main/skill", default_branch="main")
    with pytest.raises(ValueError):
        corpus._parse_github_source("https://github.com/a/b/tree/main/../skill", default_branch="main")


def test_static_snapshot_never_materializes_executable_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = tmp_path / "discovery.json"
    _discovery(discovery, [_candidate("safe", "skills/safe"), _candidate("scripted", "skills/scripted")])
    sha = "a" * 40
    safe_skill = b"---\nname: safe\ndescription: Static safe test Skill.\n---\nSafe text.\n"
    tree = [
        {"path": "skills/safe/SKILL.md", "type": "blob", "mode": "100644", "size": len(safe_skill)},
        {"path": "skills/safe/references/policy.md", "type": "blob", "mode": "100644", "size": 12},
        {"path": "skills/scripted/SKILL.md", "type": "blob", "mode": "100644", "size": 31},
        {"path": "skills/scripted/scripts/run.py", "type": "blob", "mode": "100755", "size": 10},
    ]

    def fake_json(url: str, *, token=None):  # type: ignore[no-untyped-def]
        if url == "https://api.github.com/repos/owner/repo":
            return {"default_branch": "main", "license": {"spdx_id": "MIT"}}
        if "/commits/" in url:
            return {"sha": sha}
        if "/git/trees/" in url:
            return {"truncated": False, "tree": tree}
        raise AssertionError(url)

    blobs = {
        "skills/safe/SKILL.md": safe_skill,
        "skills/safe/references/policy.md": b"Policy text\n",
    }

    def fake_bytes(url: str, *, token=None, max_bytes=0):  # type: ignore[no-untyped-def]
        source_path = url.split(f"/{sha}/", 1)[1]
        return blobs[source_path]

    monkeypatch.setattr(corpus, "_json_get", fake_json)
    monkeypatch.setattr(corpus, "_bounded_get", fake_bytes)
    output = tmp_path / "snapshot"
    manifest = corpus.snapshot_public_skills(discovery, output, limit=2)
    assert manifest["acceptedCount"] == 1
    assert manifest["complete"] is False
    assert manifest["officialEsP1QualificationEligible"] is False
    assert not list(output.rglob("*.py"))
    inspected = corpus.inspect_public_snapshot(output)
    assert inspected["status"] == "valid"
    assert inspected["executionPolicy"] == "static_only"
    assert inspected["runtimePackageInspection"]["executionAttempted"] is False
    report = corpus.build_public_pilot_report(output, tmp_path / "report", discovery_path=discovery)
    assert report["status"] == "static_import_pilot_complete_runtime_eval_not_started"
    assert report["quarantine"]["executionAttempted"] is False
    assert (tmp_path / "report/public-skill-pilot-report.md").is_file()
    kit = corpus.export_public_author_kit(output, tmp_path / "author-kit", tasks_per_skill=3)
    assert kit["selectedPackageCount"] == 1
    assert kit["taskSlotCount"] == 3
    assert kit["containsGeneratedGold"] is False
    checked_kit = corpus.inspect_public_author_kit(tmp_path / "author-kit")
    assert checked_kit["status"] == "valid"
    assert checked_kit["thirdPartyExecutionAttempted"] is False


def test_snapshot_tamper_and_authority_drift_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = tmp_path / "discovery.json"
    _discovery(discovery, [_candidate("safe", "skills/safe")])
    sha = "b" * 40
    data = b"---\nname: safe\n---\nStatic only.\n"

    def fake_json(url: str, *, token=None):  # type: ignore[no-untyped-def]
        if url == "https://api.github.com/repos/owner/repo":
            return {"default_branch": "main", "license": {"spdx_id": "Apache-2.0"}}
        if "/commits/" in url:
            return {"sha": sha}
        return {"truncated": False, "tree": [
            {"path": "skills/safe/SKILL.md", "type": "blob", "mode": "100644", "size": len(data)},
        ]}

    monkeypatch.setattr(corpus, "_json_get", fake_json)
    monkeypatch.setattr(corpus, "_bounded_get", lambda *args, **kwargs: data)
    output = tmp_path / "snapshot"
    corpus.snapshot_public_skills(discovery, output, limit=1)
    skill = next((output / "packages").rglob("SKILL.md"))
    skill.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="digest drift"):
        corpus.inspect_public_snapshot(output)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["officialEsP1QualificationEligible"] = True
    body = {key: value for key, value in manifest.items() if key != "manifestDigest"}
    manifest["manifestDigest"] = sha256_json(body)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="authority boundary"):
        corpus.inspect_public_snapshot(output)


def test_snapshot_rejects_unsealed_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = tmp_path / "discovery.json"
    _discovery(discovery, [_candidate("safe", "skills/safe")])
    sha = "c" * 40
    data = b"---\nname: safe\n---\nNo execution.\n"

    def fake_json(url: str, *, token=None):  # type: ignore[no-untyped-def]
        if url == "https://api.github.com/repos/owner/repo":
            return {"default_branch": "main", "license": {"spdx_id": "MIT"}}
        if "/commits/" in url:
            return {"sha": sha}
        return {"truncated": False, "tree": [
            {"path": "skills/safe/SKILL.md", "type": "blob", "mode": "100644", "size": len(data)},
        ]}

    monkeypatch.setattr(corpus, "_json_get", fake_json)
    monkeypatch.setattr(corpus, "_bounded_get", lambda *args, **kwargs: data)
    output = tmp_path / "snapshot"
    corpus.snapshot_public_skills(discovery, output, limit=1)
    package = next(path for path in (output / "packages").iterdir() if path.is_dir())
    (package / "extra.md").write_text("not sealed", encoding="utf-8")
    with pytest.raises(ValueError, match="unsealed files"):
        corpus.inspect_public_snapshot(output)


def test_snapshot_can_extend_a_validated_seed_without_redownloading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = tmp_path / "discovery.json"
    _discovery(discovery, [
        _candidate("first", "skills/first"),
        _candidate("second", "skills/second"),
    ])
    sha = "d" * 40
    first = b"---\nname: first\ndescription: First static Skill.\n---\nRead only.\n"
    second = b"---\nname: second\ndescription: Second static Skill.\n---\nPlan only.\n"
    tree = [
        {"path": "skills/first/SKILL.md", "type": "blob", "mode": "100644", "size": len(first)},
        {"path": "skills/second/SKILL.md", "type": "blob", "mode": "100644", "size": len(second)},
    ]

    def fake_json(url: str, *, token=None):  # type: ignore[no-untyped-def]
        if url == "https://api.github.com/repos/owner/repo":
            return {"default_branch": "main", "license": {"spdx_id": "MIT"}}
        if "/commits/" in url:
            return {"sha": sha}
        return {"truncated": False, "tree": tree}

    calls: list[str] = []

    def fake_bytes(url: str, *, token=None, max_bytes=0):  # type: ignore[no-untyped-def]
        source = url.split(f"/{sha}/", 1)[1]
        calls.append(source)
        return {"skills/first/SKILL.md": first, "skills/second/SKILL.md": second}[source]

    monkeypatch.setattr(corpus, "_json_get", fake_json)
    monkeypatch.setattr(corpus, "_bounded_get", fake_bytes)
    seed = tmp_path / "seed"
    seed_manifest = corpus.snapshot_public_skills(discovery, seed, limit=1)
    assert calls == ["skills/first/SKILL.md"]
    expanded = tmp_path / "expanded"
    manifest = corpus.snapshot_public_skills(
        discovery, expanded, limit=2, seed_snapshot_root=seed,
    )
    assert calls == ["skills/first/SKILL.md", "skills/second/SKILL.md"]
    assert manifest["acceptedCount"] == 2
    assert manifest["seedSnapshotDigest"] == seed_manifest["manifestDigest"]
    assert corpus.inspect_public_snapshot(expanded)["complete"] is True
