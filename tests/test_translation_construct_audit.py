from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.translation_case_authoring import run_anchored_case_authoring
from evaluation.translation_construct_audit import audit_constructs
from network_runtime.contracts import sha256_json
from tests.test_translation_case_authoring import FakeAdapter, _corpus


def test_audit_preserves_sealed_sources_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(tmp_path, monkeypatch)
    authoring = tmp_path / "authoring"
    run_anchored_case_authoring(
        corpus, authoring, batch_id="development-01", adapter=FakeAdapter(),
    )
    snapshot = {path: path.read_bytes() for path in authoring.rglob("*") if path.is_file()}
    output = tmp_path / "audit/report.json"
    report = audit_constructs(corpus, [authoring], output)
    assert report["uniqueSkillCount"] == 1
    assert report["currentMechanicalPassCount"] == 1
    assert report["semanticAlignmentProven"] is False
    assert report["goldAuthored"] is False
    assert report["reportDigest"] == sha256_json({
        key: value for key, value in report.items() if key != "reportDigest"
    })
    assert json.loads(output.read_text()) == report
    assert {path: path.read_bytes() for path in authoring.rglob("*") if path.is_file()} == snapshot
    with pytest.raises(ValueError, match="already exists"):
        audit_constructs(corpus, [authoring], output)
    with pytest.raises(ValueError, match="outside sealed inputs"):
        audit_constructs(corpus, [authoring], authoring / "audit.json")


def test_audit_cannot_duplicate_denominator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique nonempty"):
        audit_constructs(tmp_path, [tmp_path, tmp_path], tmp_path / "report.json")
