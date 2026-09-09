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


def test_merge_preserves_query_provenance_without_requery_or_replacement(tmp_path, monkeypatch):
    row = _candidate("shared", "shared")
    other = {**row, "discoveryQuery": "second"}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _discovery(a, [row])
    _discovery(b, [other, _candidate("other", "other")])
    monkeypatch.setattr(corpus, "_json_get", lambda *args, **kw: pytest.fail("no network during merge"))
    result = corpus.merge_public_discoveries([a, b], tmp_path / "merged.json")
    assert result["candidateCount"] == 2
    assert result["duplicateSourceCount"] == 1
    assert result["candidates"][0]["discoveredInQueries"] == ["test", "second"]
    assert result["parentDiscoveryDigests"] == [corpus.load_discovery(a)["discoveryDigest"], corpus.load_discovery(b)["discoveryDigest"]]
    assert corpus.load_discovery(tmp_path / "merged.json") == result
    with pytest.raises(FileExistsError):
        corpus.merge_public_discoveries([a, b], tmp_path / "merged.json")
    _discovery(b, [{**row, "githubUrl": "https://github.com/owner/repo/tree/main/conflict"}])
    with pytest.raises(ValueError, match="conflicting"):
        corpus.merge_public_discoveries([a, b], tmp_path / "conflict.json")


def test_source_dedup_preserves_case_sensitive_branches_and_paths(tmp_path):
    rows = [_candidate("one", "skill"), _candidate("two", "Skill")]
    rows.append({**rows[0], "id": "three", "githubUrl": rows[0]["githubUrl"].replace("/main/", "/Main/")})
    rows.append({**rows[0], "githubUrl": rows[0]["githubUrl"].replace("/owner/repo/", "/OWNER/REPO/")})
    _discovery(tmp_path / "d.json", rows)
    result = corpus.merge_public_discoveries([tmp_path / "d.json"], tmp_path / "out.json")
    assert result["candidateCount"] == 3
    assert result["duplicateSourceCount"] == 1


@pytest.fixture
def inert_snapshot(tmp_path, monkeypatch):
    discovery = tmp_path / "discovery.json"
    _discovery(discovery, [_candidate("scripted", "skills/scripted")])
    blobs = {
        "SKILL.md": b"---\nname: scripted\ndescription: Read and check.\n---\nReview attached scripts without execution.\n<script>alert('untrusted')</script>\n",
        "scripts/check.py": b"raise RuntimeError('NEVER EXECUTE THIRD PARTY CONTENT')\n",
        "bin/check": b"#!/bin/sh\nexit 99\n",
        "assets/payload.so": b"\x00\xff\x01",
        "references/link.md": b"/etc/passwd",
    }
    tree = [
        {"path": "skills/scripted/" + name, "type": "blob", "size": len(data),
         "mode": "120000" if "link" in name else "100755" if name == "bin/check" else "100644"}
        for name, data in blobs.items()
    ]
    tree.append({"path": "skills/scripted/vendor", "type": "commit", "mode": "160000", "size": 0})

    def fake_json(url, **kwargs):
        if url == "https://api.github.com/repos/owner/repo":
            return {"default_branch": "main", "license": {"spdx_id": "MIT"}}
        return {"sha": "a" * 40} if "/commits/" in url else {"tree": tree, "truncated": False}

    monkeypatch.setattr(corpus, "_json_get", fake_json)
    monkeypatch.setattr(corpus, "_bounded_get", lambda url, **kw: blobs[url.split("skills/scripted/")[1]])
    output = tmp_path / "snapshot"
    corpus.snapshot_public_skills(discovery, output, limit=1, script_policy="inert-text")
    return discovery, output, blobs


@pytest.fixture
def sampled_sources(inert_snapshot, tmp_path):
    from evaluation.translation_corpus import build_translation_corpus

    discovery, snapshot, _ = inert_snapshot
    sampling = tmp_path / "sampling"
    sampling.mkdir()
    rows = [_candidate("scripted", "skills/scripted"), _candidate("failed", "skills/failed")]
    _discovery(sampling / "discovery.json", rows)
    sampled = corpus.load_discovery(sampling / "discovery.json")
    plan = {
        "sampledDiscoveryDigest": sampled["discoveryDigest"], "selectedCount": 2,
        "excludedRepositories": [],
        "batches": [{"batchId": "public-01", "candidateIds": ["scripted", "failed"]}],
    }
    plan["samplingDigest"] = sha256_json(plan)
    (sampling / "sampling.json").write_text(json.dumps(plan))
    batch = tmp_path / "batches/public-01"
    batch.mkdir(parents=True)
    batch_discovery = {k: v for k, v in sampled.items() if k != "discoveryDigest"}
    batch_discovery["parentSamplingDigest"] = plan["samplingDigest"]
    batch_discovery["discoveryDigest"] = sha256_json(batch_discovery)
    (batch / "discovery.json").write_text(json.dumps(batch_discovery))
    snapshot.rename(batch / "snapshot")
    snapshot = batch / "snapshot"
    records_path = snapshot / "records.jsonl"
    failed = {
        "status": "excluded", "reason": "license_not_declared", "candidateId": "failed",
        "githubUrl": rows[1]["githubUrl"], "repository": "owner/repo",
    }
    records_path.write_text(records_path.read_text() + json.dumps(failed) + "\n")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest.pop("manifestDigest")
    manifest.update(discoveryDigest=batch_discovery["discoveryDigest"], complete=False,
                    requestedAccepted=2, excludedCount=1, recordsDigest=corpus._file_digest(records_path))
    manifest["manifestDigest"] = sha256_json(manifest)
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    build_translation_corpus(snapshot, batch / "library", discovery_path=batch / "discovery.json")
    return sampling, batch.parent


def test_sampling_report_preserves_failures_and_never_equates_import_with_translation(sampled_sources, tmp_path):
    from evaluation.translation_corpus import build_public_sampling_report

    sampling, batches = sampled_sources
    result = build_public_sampling_report(sampling, batches, tmp_path / "report")
    assert result["statistics"]["sampledCandidates"] == 2
    assert result["statistics"]["acceptedSkills"] == 1
    assert result["statistics"]["excludedCandidates"] == 1
    assert result["allCandidatesProcessed"] is True
    assert result["batches"][0]["importerAcceptedTargetMet"] is False
    assert result["translationMetrics"] is None
    assert result["statistics"]["verifiedDomains"] is None
    assert result["runtimeAuthorityGranted"] is False
    assert result["candidates"][1]["reason"] == "license_not_declared"
    page = (tmp_path / "report/skill-library.html").read_text()
    assert "license_not_declared" in page and "NEVER EXECUTE THIRD PARTY CONTENT" in page
    assert "&lt;script&gt;alert" in page
    assert "<script" not in page and "<details open" not in page
    replay = build_public_sampling_report(sampling, batches, tmp_path / "replay")
    assert replay == result
    assert (tmp_path / "replay/skill-library.html").read_bytes() == (tmp_path / "report/skill-library.html").read_bytes()
    with pytest.raises(FileExistsError):
        build_public_sampling_report(sampling, batches, tmp_path / "report")


@pytest.mark.parametrize("mutation", ["digest", "duplicate", "unselected", "path", "exposure", "library"])
def test_sampling_report_detects_binding_and_coverage_drift(sampled_sources, tmp_path, mutation):
    from evaluation.translation_corpus import build_public_sampling_report

    sampling, batches = sampled_sources
    path = sampling / "sampling.json"
    plan = json.loads(path.read_text())
    if mutation == "digest":
        plan["selectedCount"] = 99
    elif mutation == "duplicate":
        plan["batches"][0]["candidateIds"] = ["scripted", "scripted"]
    elif mutation == "unselected":
        plan["batches"][0]["candidateIds"] = ["scripted", "outside"]
    elif mutation == "path":
        plan["batches"][0]["batchId"] = "../public-01"
    elif mutation == "exposure":
        plan["excludedRepositories"] = ["owner/repo"]
    else:
        (batches / "public-01/library/index.json").write_text("{}")
    if mutation != "digest":
        plan["samplingDigest"] = sha256_json({k: v for k, v in plan.items() if k != "samplingDigest"})
        batch_path = batches / "public-01/discovery.json"
        batch = json.loads(batch_path.read_text())
        batch["parentSamplingDigest"] = plan["samplingDigest"]
        batch["discoveryDigest"] = sha256_json({k: v for k, v in batch.items() if k != "discoveryDigest"})
        batch_path.write_text(json.dumps(batch))
        manifest_path = batches / "public-01/snapshot/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["discoveryDigest"] = batch["discoveryDigest"]
        manifest["manifestDigest"] = sha256_json({k: v for k, v in manifest.items() if k != "manifestDigest"})
        manifest_path.write_text(json.dumps(manifest))
        if mutation == "exposure":
            from evaluation.translation_corpus import build_translation_corpus
            library = batches / "public-01/library"
            library.rename(tmp_path / "original-library")
            build_translation_corpus(batches / "public-01/snapshot", library, discovery_path=batch_path)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="repository exposure" if mutation == "exposure" else None):
        build_public_sampling_report(sampling, batches, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_inert_script_evidence_is_readable_but_never_a_runtime_resource(inert_snapshot, tmp_path):
    from evaluation.translation_corpus import build_translation_corpus

    discovery, output, blobs = inert_snapshot
    inspected = corpus.inspect_public_snapshot(output)
    assert inspected["acceptedCount"] == 1
    assert inspected["runtimePackageInspection"]["gates"] == {"blocked": 1}
    record = json.loads((output / "records.jsonl").read_text())
    assert [x["path"] for x in record["files"]] == ["SKILL.md"]
    assert len(record["withheldFiles"]) == 5
    assert len(record["quarantinedFiles"]) == 4
    assert not list(output.rglob("*.py"))
    assert not any(x.is_symlink() for x in output.rglob("*"))
    for item in record["quarantinedFiles"]:
        if item["evidencePath"]:
            file = output / item["evidencePath"]
            assert file.read_bytes() == blobs[item["sourcePath"]]
            assert not file.stat().st_mode & 0o111
    # Even an unreferenced withheld file prevents accidental Runtime-kit eligibility.
    with pytest.raises(ValueError, match="Runtime gate"):
        corpus.export_public_author_kit(output, tmp_path / "kit")
    build_translation_corpus(output, tmp_path / "library", discovery_path=discovery)
    index = json.loads((tmp_path / "library/index.json").read_text())
    skill = index["skills"][0]
    assert skill["primaryTranslationEligible"] and not skill["runtimeReady"]
    assert not skill["contextComplete"]
    assert len(skill["files"]) == 5
    assert any(x["path"] == "[quarantine] scripts/check.py" for x in skill["files"])
    expanded = tmp_path / "extended"
    corpus.snapshot_public_skills(discovery, expanded, limit=1, script_policy="inert-text", seed_snapshot_root=output)
    assert corpus.inspect_public_snapshot(expanded)["status"] == "valid"


@pytest.mark.parametrize("tamper", ["content", "extra", "mode", "symlink"])
def test_quarantined_evidence_tampering_is_rejected(inert_snapshot, tamper):
    _, output, _ = inert_snapshot
    file = next((output / "quarantine").rglob("*.txt"))
    if tamper == "content":
        file.write_text("drift")
    elif tamper == "extra":
        (file.parent / "extra.txt").write_text("extra")
    elif tamper == "mode":
        file.chmod(0o700)
    else:
        file.unlink()
        file.symlink_to(output / "records.jsonl")
    with pytest.raises(ValueError, match="quarantine"):
        corpus.inspect_public_snapshot(output)


def test_extensionless_executable_mode_is_a_surface():
    from pathlib import PurePosixPath

    assert corpus._executable_surface(PurePosixPath("check"), "100755", "blob") == "executable-mode"


def test_metadata_sampling_is_balanced_disjoint_and_keeps_repository_groups(inert_snapshot, tmp_path):
    _, prior, _ = inert_snapshot
    rows = [_candidate("already-seen", "old")]
    for domain in ("a", "b", "c"):
        for number in range(4):
            for skill in range(3):
                row = _candidate(f"{domain}-{number}-{skill}", f"skills/{skill}")
                row.update(githubUrl=f"https://github.com/{domain}/repo{number}/tree/main/skills/{skill}",
                           discoveryQuery=domain)
                rows.append(row)
    source = tmp_path / "fresh.json"
    _discovery(source, rows)
    plan = corpus.sample_public_skills(source, tmp_path / "sample", prior_snapshots=[prior],
                                      seed="fixed", limit=12, batch_size=4)
    assert plan["queryStrataCounts"] == {"a": 4, "b": 4, "c": 4}
    assert plan["excludedRepositories"] == ["owner/repo"]
    assert {"candidateId": "already-seen", "reason": "prior_repository"} in plan["rejected"]
    selected = corpus.load_discovery(tmp_path / "sample/discovery.json")["candidates"]
    repository_batches = {}
    by_id = {row["id"]: row for row in selected}
    for batch in plan["batches"]:
        for key in batch["candidateIds"]:
            repository = corpus._repo_key(by_id[key]["githubUrl"])
            assert repository_batches.setdefault(repository, batch["batchId"]) == batch["batchId"]
    assert max(sum(corpus._repo_key(x["githubUrl"]) == repo for x in selected)
               for repo in repository_batches) <= 2
    assert not plan["proofCohortEligible"] and not plan["independentGold"]
    # Content, gold, runtime output and selected-result feedback are not inputs.
    _discovery(source, list(reversed(rows)))
    again = corpus.sample_public_skills(source, tmp_path / "sample2", prior_snapshots=[prior],
                                       seed="fixed", limit=12, batch_size=4)
    assert again["batches"] == plan["batches"]
    assert len(selected) + len(plan["rejected"]) + len(plan["unselectedCandidateIds"]) == len(rows)


def test_sampling_reports_shortfall_without_reusing_known_sources(inert_snapshot, tmp_path):
    discovery, prior, _ = inert_snapshot
    plan = corpus.sample_public_skills(discovery, tmp_path / "empty", prior_snapshots=[prior],
                                      seed="fixed", limit=60)
    assert not plan["complete"] and plan["selectedCount"] == 0
    assert plan["batches"] == []
    with pytest.raises(FileExistsError):
        corpus.sample_public_skills(discovery, tmp_path / "empty", prior_snapshots=[prior], seed="fixed")
    with pytest.raises(ValueError, match="exposure"):
        corpus.sample_public_skills(discovery, tmp_path / "no-prior", prior_snapshots=[], seed="fixed")
    additional = corpus.sample_public_skills(discovery, tmp_path / "manual", prior_snapshots=[prior],
        seed="fixed", prior_repositories=["Other/Seen", "other/seen"])
    assert additional["explicitlyExposedRepositories"] == ["other/seen"]
    assert "other/seen" in additional["excludedRepositories"]
    with pytest.raises(ValueError, match="exposed repository"):
        corpus.sample_public_skills(discovery, tmp_path / "unsafe", prior_snapshots=[prior],
            seed="fixed", prior_repositories=["../oops/repo"])


def test_sampling_pins_protocol_and_rejects_parameter_drift(inert_snapshot, tmp_path):
    discovery, prior, _ = inert_snapshot
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"sampleTarget": 60, "maxSkillsPerRepository": 2,
                                    "batchSize": 20, "samplingSeed": "fixed"}))
    plan = corpus.sample_public_skills(discovery, tmp_path / "pinned", prior_snapshots=[prior],
                                      seed="fixed", protocol_path=protocol)
    assert plan["protocol"]["sha256"] == corpus._file_digest(protocol)
    assert "effect_runtime/skill_package.py" in plan["implementation"]
    assert "evaluation/flow_condition_expression.py" in plan["implementation"]
    with pytest.raises(ValueError, match="frozen protocol"):
        corpus.sample_public_skills(discovery, tmp_path / "drift", prior_snapshots=[prior],
                                   seed="changed", protocol_path=protocol)


def test_recover_scripts_keeps_all_original_exclusions_as_development(inert_snapshot, tmp_path):
    discovery, _, _ = inert_snapshot
    excluded = tmp_path / "excluded"
    corpus.snapshot_public_skills(discovery, excluded, limit=1, script_policy="exclude")
    target = tmp_path / "recovery.json"
    result = corpus.recover_script_discovery(discovery, excluded, target)
    assert corpus.load_discovery(target)["candidateCount"] == 1
    assert result["candidates"] == corpus.load_discovery(discovery)["candidates"]
    assert result["evidenceRole"] == "known_metadata_script_recovery_development_not_holdout"
    with pytest.raises(FileExistsError):
        corpus.recover_script_discovery(discovery, excluded, target)


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
