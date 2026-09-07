"""Intake integrity checks, not translation accuracy or semantic Gold."""

import copy
import hashlib

import pytest

from evaluation.translation_boundary_pilot import BoundaryPilot, build_boundary_report
from network_runtime.contracts import sha256_json


def _seal(index):
    index["indexDigest"] = sha256_json({k: v for k, v in index.items() if k != "indexDigest"})
    return index


def _fixture():
    content = "Read the host tool contract before choosing an operation."
    index = _seal({"skills": [
        {"packageId": f"skill-{i}", "name": f"Skill {i}", "repository": f"repo-{i // 2}",
         "domain": f"domain-{i // 4}", "commitSha": "fixture-commit", "packageDigest": "fixture-package",
         "primaryTranslationEligible": True,
         "files": [{"path": "SKILL.md", "content": content,
                    "sha256": "sha256:" + hashlib.sha256(content.encode()).hexdigest()}]}
        for i in range(8)
    ]})
    pilot = {
        "corpus_index_digest": index["indexDigest"], "reviewer_kind": "test_fixture",
        "reviewer_id": "protocol-only-not-independent", "evidence_role": "known_development_intake_only",
        "entries": [{"package_id": f"skill-{i}", "structures": ["references"],
                     "required_environment": ["Host tool schemas and trusted implementation"],
                     "single_read_gap": "A fixture for intake only; no translation has been attempted.",
                     "citations": [{"path": "SKILL.md", "exact_quote": "host tool contract"}]}
                    for i in range(8)],
    }
    return index, pilot


def test_intake_counts_are_not_accuracy_or_execution():
    index, raw = _fixture()
    report = build_boundary_report(index, BoundaryPilot.model_validate(raw))
    assert (report["skillCount"], report["repositoryCount"], report["domainCount"]) == (8, 4, 2)
    assert report["structureCounts"] == {"references": 8}
    assert report["translationRuns"] == report["runtimeRuns"] == report["modelCalls"] == 0
    assert not report["proofCohortEligible"] and not report["thirdPartyExecutionAttempted"]
    assert "accuracy" not in report and "NOT zero translation accuracy" in report["boundary"]
    for row in report["rows"]:
        assert row["translationOutcome"] == "not_run" and not row["runtimeAuthorityGranted"]
        citation = row["citations"][0]
        source = index["skills"][0]["files"][0]["content"]
        assert source[citation["start"]:citation["end"]] == citation["exactQuote"]
    assert report["reportDigest"] == sha256_json({k: v for k, v in report.items() if k != "reportDigest"})


@pytest.mark.parametrize("mutation", ["index", "pilot_digest", "duplicate_entry", "duplicate_skill",
                                      "unknown", "ineligible", "duplicate_file", "file_digest",
                                      "quote", "ambiguous_quote", "path", "binary"])
def test_stale_or_invalid_intake_is_rejected(mutation):
    index, raw = _fixture()
    entry = raw["entries"][0]
    skill = index["skills"][0]
    if mutation == "index":
        skill["domain"] = "altered"
    elif mutation == "pilot_digest":
        raw["corpus_index_digest"] = "wrong"
    elif mutation == "duplicate_entry":
        raw["entries"][1] = copy.deepcopy(entry)
    elif mutation == "duplicate_skill":
        index["skills"].append(copy.deepcopy(skill))
    elif mutation == "unknown":
        entry["package_id"] = "unknown"
    elif mutation == "ineligible":
        skill["primaryTranslationEligible"] = False
    elif mutation == "duplicate_file":
        skill["files"].append(copy.deepcopy(skill["files"][0]))
    elif mutation == "file_digest":
        skill["files"][0]["sha256"] = "wrong"
    elif mutation == "quote":
        entry["citations"][0]["exact_quote"] = "Invented contract"
    elif mutation == "ambiguous_quote":
        entry["citations"][0]["exact_quote"] = "o"
    elif mutation == "path":
        entry["citations"][0]["path"] = "missing.md"
    else:
        skill["files"][0]["content"] = None
    if mutation not in {"index", "pilot_digest"}:
        raw["corpus_index_digest"] = _seal(index)["indexDigest"]
    with pytest.raises(ValueError):
        build_boundary_report(index, BoundaryPilot.model_validate(raw))


@pytest.mark.parametrize("mutation", ["too_small", "too_large", "human", "proof", "extra"])
def test_pilot_cannot_claim_another_evidence_role(mutation):
    _, raw = _fixture()
    if mutation == "too_small":
        raw["entries"].pop()
    elif mutation == "too_large":
        raw["entries"] *= 2
    elif mutation == "human":
        raw["reviewer_kind"] = "independent_human"
    elif mutation == "proof":
        raw["evidence_role"] = "unseen_proof"
    else:
        raw["runtimeAuthorityGranted"] = True
    with pytest.raises(ValueError):
        BoundaryPilot.model_validate(raw)
