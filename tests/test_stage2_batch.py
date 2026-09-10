"""Mechanical batch/representation tests; no model or public scripts execute."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_ledger as ledger, stage2_batch, stage2_cases
from evaluation.stage1_cases import packet


def test_predeclared_selection_is_unique_diverse_and_retains_scripts_and_l1():
    selection = json.loads((stage2_batch.ROOT / "data/stage2_public_selection.json").read_text())
    cases = stage2_batch.validate_selection(selection)
    assert len({r["domain"] for r in cases}) == 9
    assert "quarantined_scripts" in {f for r in cases for f in r["features"]}
    assert "simple-english" in {r["id"] for r in cases}
    selection["cases"][-1] = selection["cases"][0]
    with pytest.raises(ValueError):
        stage2_batch.validate_selection(selection)


@pytest.mark.parametrize("case", list(stage2_cases.EXPECTATIONS))
def test_author_input_has_no_review_answers_or_execution(case):
    bundle = packet("reference")["bundle"]
    before = copy.deepcopy(bundle)
    author = stage2_cases.author_packet(bundle, case)
    if case == "simple-english":
        assert author is None
    else:
        assert set(author) == {"bundle", "task", "taskOrigin", "inputSchema", "catalog", "reads"}
        assert author["bundle"] == bundle
        assert "review-requirements" not in json.dumps(author)
        assert all(t["annotations"]["readOnlyHint"] for t in author["catalog"]["tools"])
        assert len(author["reads"]) == len(author["catalog"]["tools"])
    assert bundle == before
    review = stage2_cases.review_requirements(bundle, case)
    assert review["semanticVerdict"] is None and not review["wholeSkillAccepted"]


def test_shared_source_scan_keeps_exact_validation_language():
    value = packet("reference")
    wire, _ = ledger.make_request(value, ledger.initial_state(value, "plan_first", semantic_plan=True))
    full = wire["format"]
    scan = next(v for v in full["oneOf"] if v["properties"]["mode"]["const"] == "operation_plan")["properties"]["source_scan"]
    fragment = full["$defs"]["SourceScanFragment"]
    shared = {"$defs": full["$defs"], **scan}
    expanded = {**scan, "properties": {key: copy.deepcopy(fragment) for key in scan["properties"]}}
    assert scan["required"] == list(scan["properties"])
    assert all(v == {"$ref": "#/$defs/SourceScanFragment"} for v in scan["properties"].values())
    good = {key: {"roles": ["observation"], "meaning": "Original source observation."} for key in scan["properties"]}
    key = next(iter(good))
    invalid = []
    missing = copy.deepcopy(good)
    missing.pop(key)
    invalid.append(missing)
    for field, value in (("roles", []), ("roles", ["new_authority"]), ("roles", ["observation"] * 5),
                         ("meaning", "short"), ("meaning", "x" * 401)):
        changed = copy.deepcopy(good)
        changed[key][field] = value
        invalid.append(changed)
    extra = copy.deepcopy(good)
    extra[key]["approved"] = True
    invalid.append(extra)
    for candidate in [good, *invalid]:
        assert Draft202012Validator(shared).is_valid(candidate) == Draft202012Validator(expanded).is_valid(candidate)
    assert Draft202012Validator(shared).is_valid(good)
    assert all(not Draft202012Validator(shared).is_valid(v) for v in invalid)
    assert len(json.dumps(shared)) < len(json.dumps({"$defs": full["$defs"], **expanded}))


def test_utf8_paging_retains_all_original_characters_and_offsets(monkeypatch):
    text = ("原文：keep ALL constraints / café / 🧪.\n" * 600)
    base = {"path": "source/SKILL.md", "sourceDigest": "original", "text": text, "start": 0, "end": len(text), "startLine": 1}
    monkeypatch.setattr(ledger.prior, "pages_for", lambda _: {"page": base})
    pages = list(ledger.pages_for({"bundle": {"references": []}}).values())
    assert len(pages) > 6  # unread source may remain; never claim all was supplied.
    assert "".join(p["text"] for p in pages) == text
    assert pages[0]["start"] == 0 and pages[-1]["end"] == len(text)
    assert all(a["end"] == b["start"] for a, b in zip(pages, pages[1:]))
    assert all(p["text"] == text[p["start"]:p["end"]] and len(p["text"].encode()) <= 2048 for p in pages)


def test_preparation_never_overwrites(tmp_path):
    with pytest.raises(FileExistsError):
        stage2_batch.prepare(Path("does-not-need-to-exist"), tmp_path)


def test_zero_budget_and_output_preservation(tmp_path, monkeypatch):
    m = {"reportDigest": "prep", "skillCount": 1, "repositoryCount": 1, "domainCount": 1,
         "cases": [{"id": "notion", "status": "ready_for_first_construction"}]}
    monkeypatch.setattr(stage2_batch, "verify_preparation", lambda _: m)
    monkeypatch.setattr(ledger, "freeze", lambda *a, **k: pytest.fail("zero budget must not contact model"))
    out = tmp_path / "out"
    report = stage2_batch.run(tmp_path / "input", out)
    assert report["rows"][0]["status"] == "not_run_call_budget"
    with pytest.raises(FileExistsError):
        stage2_batch.run(tmp_path / "input", out)
