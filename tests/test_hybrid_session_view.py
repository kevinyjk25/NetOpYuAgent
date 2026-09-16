import json

import pytest

from evaluation.hybrid_session_view import build
from skill_authoring.artifacts import write_artifacts
from skill_authoring.contracts import seal


def fixture(tmp_path):
    root = tmp_path / "run"
    write_artifacts(root, {"freeze.json": seal({"knownCases": ["sample"],
        "inputsAndExpectations": {"sample": [{}, {"task": "Explain the evidence <script>alert(1)</script>"}]}})})
    folder = root / "sample/sessions/one"
    write_artifacts(folder, {"request.json": seal({"task": "retained"})})
    write_artifacts(folder / "delivery-contract", {"report.json": seal({"requirements": [
        {"id": "d0", "kind": "analysis", "quote": "Unverified interpretation"}]})})
    evidence = seal({"noExecution": True})
    write_artifacts(folder / "draft", {"evidence.json": evidence})
    report = seal({"hostResult": {"state": "rejected", "message": "Rejected by host", "deliveryDigest": None},
                   "evidenceDigest": evidence["reportDigest"], "task": {"delivery": None}})
    write_artifacts(folder / "draft/result", {"report.json": report})
    (root / "sample/dsh-stdout.txt").write_text("Completed! <img src=x onerror=alert(1)>")
    return root, folder


def test_view_keeps_native_claim_separate_and_escapes_all_source_text(tmp_path):
    root, _ = fixture(tmp_path)
    result = build(root, tmp_path / "view")
    html = (tmp_path / "view/host-result.html").read_text()
    assert "Rejected by host" in html and "Completed!" in html
    assert "&lt;script&gt;" in html and "<script>" not in html and "<img " not in html
    assert "Unverified interpretation" in html and "default-src 'none'" in html
    assert result["semanticAssessment"] == "not_performed" and not result["sourceScriptsExecuted"]
    assert "sample/dsh-stdout.txt" in result["sourceDigests"]


def test_view_never_writes_inside_frozen_run_or_replaces_existing_output(tmp_path):
    root, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        build(root, root / "view")
    build(root, tmp_path / "view")
    with pytest.raises(ValueError):
        build(root, tmp_path / "view")


def test_view_rejects_host_receipt_drift(tmp_path):
    root, folder = fixture(tmp_path)
    path = folder / "draft/result/report.json"
    report = json.loads(path.read_text())
    report["hostResult"]["state"] = "success"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="drift"):
        build(root, tmp_path / "view")
