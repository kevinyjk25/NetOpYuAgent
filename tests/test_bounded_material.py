"""Zero-model checks of the standalone material and explicit review boundary."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from evaluation.bounded_acceptance import load_material
from evaluation.bounded_material import finalize, verify


ROOT = Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "data/bounded-pilot/r0-development-20260917"


def read(directory, name):
    return json.loads((directory / name).read_text())


@pytest.fixture
def material(tmp_path):
    target = tmp_path / "standalone"
    shutil.copytree(MATERIAL, target)
    # Tests support a package already finalized by the real root: restore its
    # immutable pending snapshot only in this isolated test copy.
    if (target / "review-manifest.pending.json").exists():
        shutil.copyfile(target / "review-manifest.pending.json", target / "review-manifest.json")
        (target / "review-manifest.pending.json").unlink()
        (target / "root-review.json").unlink()
    return target


def approval(directory):
    manifest = read(directory, "review-manifest.json")
    return {"reviewer": "root", "decision": "approved_for_bounded_development",
            "pending_manifest_sha256": "sha256:" + hashlib.sha256((directory / "review-manifest.json").read_bytes()).hexdigest(),
            "adjudication_sha256": manifest["extra_files"]["adjudication.json"],
            "references_sha256": manifest["files"]["references.json"],
            "note": "Explicit synthetic approval fixture used only in this temporary test copy."}


def test_standalone_structure_and_pending_gate(material):
    assert verify(material)["cases"] == 12
    assert verify(material)["actualModelCalls"] == 0
    with pytest.raises(ValueError, match="drafts cannot run"):
        load_material(material)
    cases = read(material, "cases.json")
    assert sum(c["kind"] == "positive" for c in cases) == 8
    assert len({c["repository_family"] for c in cases}) == 6
    assert len({c["domain"] for c in cases}) >= 3


def test_verification_does_not_import_artifact_constructors(material):
    code = ("import sys; sys.modules['evaluation.bounded_cases']=None; "
            "sys.modules['evaluation.bounded_network_policy']=None; "
            "from evaluation.bounded_material import verify; "
            "assert verify(sys.argv[1])['cases']==12")
    result = subprocess.run([sys.executable, "-B", "-c", code, str(material)], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_finalization_is_explicit_hash_bound_and_once_only(material):
    review = approval(material)
    for field in ("reviewer", "decision", "pending_manifest_sha256", "adjudication_sha256", "references_sha256", "note"):
        bad = copy.deepcopy(review)
        bad[field] = ""
        with pytest.raises(ValueError, match="explicit root review"):
            finalize(material, bad)
    result = finalize(material, review)
    assert result["status"] == "frozen_for_bounded_development"
    loaded, frozen = load_material(material)
    assert len(loaded["cases.json"]) == 12
    assert "root-review.json" in frozen["extra_files"]
    assert read(material, "metadata.json")["status"] == "pending_root_adjudication"
    with pytest.raises(ValueError, match="once"):
        finalize(material, review)


@pytest.mark.parametrize("name", ["cases.json", "references.json", "dialogues.json", "support.json",
                                   "annotation-a.json", "annotation-b.json", "sources.json", "metadata.json",
                                   "adjudication.json", "annotation-amendment-v4.json"])
def test_every_material_is_bound_to_actual_bytes(material, name):
    path = material / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(material)


def test_full_original_texts_and_item_coverage_preserved(material):
    for role in ("a", "b"):
        wrapper = read(material, f"annotation-{role}.json")
        for part in ("original", "addendum"):
            raw = wrapper[part]["text"].encode()
            assert wrapper[part]["sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
            assert json.loads(raw)
    decisions = read(material, "adjudication.json")["cases"]
    refs = {r["case_id"]: r for r in read(material, "references.json")}
    for decision in decisions:
        ref = refs[decision["case_id"]]
        assert ref["criteria"] == decision["final_criteria"]
        assert ref["duties"] == decision["final_duties"]
        ids = {r["id"] for r in ref["criteria"] + ref["duties"]} | {"conditional_not_applicable"}
        for role in ("a", "b"):
            rows = decision["reviewer_item_dispositions"][role]
            assert [r["reviewer_duty"] for r in rows] == decision["reviewer_duties_considered"][role]
            assert all(set(r["final_requirement_ids"]) <= ids for r in rows)


def test_strict_scope_and_incident_unknowns_not_erased(material):
    refs = {r["case_id"]: r for r in read(material, "references.json")}
    for name in ("irql-draft", "irql-missing-enricher"):
        duties = {d["id"]: d for d in refs[name]["duties"]}
        assert duties["time-window"]["strict_eligible"]
        assert not duties["composition"]["strict_eligible"]
        assert not duties["inventory-branch"]["strict_eligible"]
    duties = {d["id"]: d for d in refs["network-policy-least-privilege-audit"]["duties"]}
    assert not duties["least-privilege"]["strict_eligible"]
    assert duties["segmentation"]["strict_eligible"]
    new_ticket = refs["incident-new-ticket-without-go-ahead"]
    proposal = next(d for d in new_ticket["duties"] if d["id"] == "full-proposal")
    assert "root cause and verified fix are unknown" in proposal["statement"]
    assert not proposal["strict_eligible"]
    assert "write-policy" not in {d["id"] for d in new_ticket["duties"]}
    assert next(c for c in new_ticket["calls"] if "approval" in c["tool"])["min_calls"] == 0


def test_reviewed_inputs_and_raw_state_unchanged(material):
    cases = read(material, "cases.json")
    amendment = read(material, "annotation-amendment-v4.json")
    assert amendment["cases"] == [{"case_id": c["case_id"], "skill_id": c["skill_id"],
        "repository": c["repository_id"], "commit": c["source_revision"], "agent_input": c["agent_input"]} for c in cases]
    assert amendment["pre_run_raw_providers"] == [{"case_id": c["case_id"], "provider_fixture": c["provider_fixture"]} for c in cases]
