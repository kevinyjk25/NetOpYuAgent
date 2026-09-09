"""Source intake tests are mechanical regressions, not translated Skill scores."""

import copy
import json

import pytest

from evaluation import translation_intake as intake
from network_runtime.contracts import sha256_json


def bundle(text="Read the current state.\n", *, extra=()):
    docs = [intake._document("skills/test/SKILL.md", text.encode(), mode="100644", origin="fixture")]
    docs += [intake._document(path, data, mode=mode, origin="fixture") for path, data, mode in extra]
    return intake._bundle({
        "apiVersion": intake.PROTOCOL, "repository": "owner/repo", "commitSha": "a" * 40,
        "snapshotDigest": "fixture", "candidateId": "fixture", "entryPath": "skills/test/SKILL.md",
        "documents": docs, "supplementAttempts": [], "parentBundleDigest": None,
    })


def test_pages_reconstruct_long_unicode_sources_without_truncation_or_normalization():
    text = "中文🙂\r\n" * 14000 + "a" * 15000
    b = bundle(text, extra=[("skills/test/references/context.md", b"Details.\n", "100644")])
    pages = intake.review_pages(b, max_characters=1000)
    for doc in b["documents"]:
        selected = [p for p in pages["pages"] if p["path"] == doc["path"]]
        assert "".join(p["text"] for p in selected) == doc["content"]
        assert all(p["text"] == doc["content"][p["start"]:p["end"]] and len(p["text"]) <= 1000 for p in selected)
    assert not pages["semanticCoverageProven"]
    assert not pages["runtimeAuthorityGranted"]


def test_template_examples_remain_visible_and_code_dependencies_are_not_suppressed():
    source = "Follow [policy](references/policy.md).\n```markdown\n[![Badge](link)](#)\n```\n```sh\nrun scripts/check.sh\n```\n"
    refs = intake.reference_occurrences("skills/demo/SKILL.md", source)
    by_target = {r["rawTarget"]: r for r in refs}
    assert by_target["link"]["contextRole"] == "template_example_candidate"
    assert by_target["scripts/check.sh"]["contextRole"] == "code_reference_candidate"
    assert by_target["references/policy.md"]["contextRole"] == "prose_reference_candidate"
    assert by_target["scripts/check.sh"]["targetPath"] is None
    assert by_target["scripts/check.sh"]["candidatePaths"] == ["skills/demo/scripts/check.sh", "scripts/check.sh"]
    assert all(source[r["start"]:r["end"]] == r["rawTarget"] for r in refs)


@pytest.mark.parametrize("target,code", [
    ("../../../../secrets", "outside_repository"), ("/etc/passwd", "unsafe_path"),
    ("%2fetc/passwd", "unsafe_path"), ("https://evil.invalid/data", "external_url_not_fetched"),
    ("//evil.invalid/path", "external_url_not_fetched"), ("http://[invalid", "malformed_reference"),
])
def test_unsafe_and_external_references_are_evidence_not_fetch_instructions(target, code):
    refs = intake.reference_occurrences("skills/test/SKILL.md", f"Read [details]({target}).")
    assert refs[0]["resolution"] == code
    assert refs[0]["targetPath"] is None


def test_fence_lengths_and_directory_references():
    b = bundle("````markdown\n```sh\n[example](missing.md)\n```\n````\nRead [all](references/).\n",
               extra=[("skills/test/references/a.md", b"Read.\n", "100644")])
    refs = b["references"]
    assert next(r for r in refs if r["rawTarget"] == "missing.md")["contextRole"] == "template_example_candidate"
    assert next(r for r in refs if r["rawTarget"] == "references/")["availability"] == "directory_listing_present"
    assert b["closureStatus"] == "pending_semantic_reference_review"


def test_script_and_symlink_content_never_executes_or_resolves_as_a_link(tmp_path):
    sentinel = tmp_path / "executed"
    code = f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n".encode()
    b = bundle(extra=[("skills/test/scripts/action.py", code, "100755"),
                      ("skills/test/references/link.md", b"/etc/passwd", "120000"),
                      ("skills/test/assets/binary", b"\x00\xff", "100644")])
    intake.validate_bundle(b)
    pages = intake.review_pages(b)
    assert len(pages["unavailableTextPaths"]) == 2
    assert not sentinel.exists()
    assert any("Path(" in p["text"] for p in pages["pages"])


@pytest.mark.parametrize("mutation", ["text", "reference", "digest", "authority"])
def test_bundle_drift_is_rejected(mutation):
    b = bundle("Read [details](refs.md).")
    if mutation == "text":
        b["documents"][0]["content"] += "changed"
    elif mutation == "reference":
        b["references"] = []
    elif mutation == "authority":
        b["runtimeAuthorityGranted"] = True
    else:
        b["bundleDigest"] = "bad"
    if mutation != "digest":
        b["bundleDigest"] = sha256_json({k: v for k, v in b.items() if k != "bundleDigest"})
    with pytest.raises(ValueError):
        intake.validate_bundle(b)


def test_explicit_supplement_keeps_parent_failures_commit_and_raw_bytes():
    parent = bundle("Follow [sibling](../sibling/SKILL.md).")
    before = copy.deepcopy(parent)
    calls = []

    def fetch(path):
        calls.append(path)
        if path.endswith("missing.md"):
            raise OSError("missing at pinned commit")
        return {"repository": parent["repository"], "commitSha": parent["commitSha"], "path": path,
                "mode": "100755", "data": b"Never run me.\r\n"}

    child = intake.supplement_bundle(parent, ["skills/sibling/SKILL.md", "missing.md"], fetch)
    intake.validate_bundle(child)
    assert parent == before
    assert calls == ["skills/sibling/SKILL.md", "missing.md"]
    assert child["parentBundleDigest"] == parent["bundleDigest"]
    assert [r["status"] for r in child["supplementAttempts"]] == ["saved", "unavailable"]
    assert child["documents"][0]["content"] == "Never run me.\r\n"
    assert child["references"][0]["availability"] == "inert_text_present"
    with pytest.raises(ValueError, match="replace"):
        intake.supplement_bundle(child, ["skills/sibling/SKILL.md"], fetch)


@pytest.mark.parametrize("field", ["repository", "commitSha", "path"])
def test_wrong_supplement_source_is_recorded_not_admitted(field):
    b = bundle()
    def fetch(path):
        value = {"repository": b["repository"], "commitSha": b["commitSha"], "path": path,
                 "mode": "100644", "data": b"Untrusted content."}
        value[field] = "wrong"
        return value
    child = intake.supplement_bundle(b, ["sibling.md"], fetch)
    assert len(child["documents"]) == 1
    assert child["supplementAttempts"][0]["status"] == "unavailable"


@pytest.mark.parametrize("paths", [["../secret"], ["/etc/passwd"], ["a", "a"]])
def test_supplement_rejects_unsafe_selection_before_network(paths):
    def fetch(_):
        pytest.fail("must validate before network")
    with pytest.raises(ValueError):
        intake.supplement_bundle(bundle(), paths, fetch)


def test_host_schemas_are_lossless_and_constraints_are_not_silently_dropped():
    schema = {"type": "object", "properties": {"selections": {"type": "object", "properties": {
        "TRAP_SEVERITY": {"type": "array", "items": {"type": "string", "enum": ["crit"]}}}}},
        "required": ["selections"], "additionalProperties": False}
    catalog = {"tools": [{"name": "query", "inputSchema": schema, "annotations": {"readOnlyHint": True}}]}
    original = copy.deepcopy(catalog)
    report = intake.host_catalog_diagnostic(catalog)
    assert report["catalog"] == original and catalog == original
    assert not report["currentReadSchemaCompatible"] and not report["runtimeAuthorityGranted"]
    assert any(d["pointer"] == "/tools/0/inputSchema/properties/selections/type" for d in report["diagnostics"])
    assert {d["code"] for d in report["diagnostics"]} == {"missing_host_schema", "unsupported_current_read_contract"}
    catalog["tools"][0]["name"] = "mutated"
    assert report["catalog"] == original


def test_valid_scalar_schema_is_compatible_not_authorized():
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"], "additionalProperties": False}
    report = intake.host_catalog_diagnostic({"tools": [{"name": "query", "inputSchema": schema, "outputSchema": schema}]})
    assert report["currentReadSchemaCompatible"]
    assert not report["semanticAlignmentProven"] and not report["runtimeAuthorityGranted"]


def test_host_remote_refs_never_trigger_retrieval(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("no remote refs"))
    r = intake.host_catalog_diagnostic({"tools": [{"name": "x", "inputSchema": {"$ref": "https://invalid.test/schema"}}]})
    assert not r["currentReadSchemaCompatible"]


@pytest.mark.parametrize("catalog", [[], {}, {"tools": []}, {"tools": [{"name": " "}]}, {"tools": [{"name": "x"}, {"name": "x"}]}])
def test_missing_empty_or_duplicate_host_catalog_is_not_vacuously_compatible(catalog):
    with pytest.raises(ValueError):
        intake.host_catalog_diagnostic(catalog)


def test_hash_only_source_cannot_bypass_file_size_budget():
    b = bundle(extra=[("skills/test/blob.bin", b"\x00", "100644")])
    next(d for d in b["documents"] if d["content"] is None)["bytes"] = -100
    b["bundleDigest"] = sha256_json({k: v for k, v in b.items() if k != "bundleDigest"})
    with pytest.raises(ValueError, match="bounds"):
        intake.validate_bundle(b)


@pytest.mark.parametrize("mode,data", [("120000", b"SKILL.md"), ("100644", b"\x00\xff")])
def test_resealed_link_or_binary_cannot_claim_available_text(mode, data):
    b = bundle(extra=[("skills/test/source", data, mode)])
    next(d for d in b["documents"] if d["path"].endswith("/source"))["representation"] = "inert_utf8_text"
    b["bundleDigest"] = sha256_json({k: v for k, v in b.items() if k != "bundleDigest"})
    with pytest.raises(ValueError, match="representation"):
        intake.validate_bundle(b)


def test_intake_outputs_exclusive_and_reproducible(tmp_path):
    b = bundle("[![Badge](link)](#)\n")
    report = intake.write_intake(b, tmp_path / "one")
    assert report["translationMetrics"] is None and report["hostStatus"] == "missing_host_catalog"
    assert intake.write_intake(b, tmp_path / "two") == report
    assert (tmp_path / "one/bundle.json").read_bytes() == (tmp_path / "two/bundle.json").read_bytes()
    assert json.loads((tmp_path / "one/review-pages.json").read_text())["bundleDigest"] == b["bundleDigest"]
    with pytest.raises(FileExistsError):
        intake.write_intake(b, tmp_path / "one")


def test_pinned_git_supplement_handles_padded_sizes_and_never_checks_out(monkeypatch):
    commands = []
    def git(args, **kwargs):
        commands.append(args)
        if "rev-parse" in args:
            return ("a" * 40 + "\n").encode()
        if "ls-tree" in args:
            return ("100644 blob " + "b" * 40 + "      5\trefs/a file.md\0").encode()
        if "cat-file" in args:
            return b"hello"
        return b""
    monkeypatch.setattr(intake, "_git_run", git)
    result = intake.supplement_from_git(bundle(), ["refs/a file.md"])
    assert result["supplementAttempts"][0]["status"] == "saved"
    assert result["documents"][0]["path"] == "refs/a file.md"
    assert not any("checkout" in args for args in commands)
    fetch = next(args for args in commands if "fetch" in args)
    assert fetch[-1] == "a" * 40 and fetch[-2] == "https://github.com/owner/repo.git"
    assert "--literal-pathspecs" in next(args for args in commands if "ls-tree" in args)


def test_pinned_git_commit_drift_blocks_before_blob_read(monkeypatch):
    def git(args, **kwargs):
        assert "cat-file" not in args and "ls-tree" not in args
        return ("b" * 40).encode() if "rev-parse" in args else b""
    monkeypatch.setattr(intake, "_git_run", git)
    with pytest.raises(ValueError, match="commit differs"):
        intake.supplement_from_git(bundle(), ["refs/a.md"])


def test_existing_corpus_cli_exposes_intake_without_model_or_network(tmp_path, monkeypatch, capsys):
    from evaluation.public_skill_corpus import main
    monkeypatch.setattr(intake, "bundle_from_snapshot", lambda *args: bundle())
    assert main(["translation-intake", "snapshot", "candidate", "--output-root", str(tmp_path / "new")]) == 0
    assert json.loads(capsys.readouterr().out)["hostStatus"] == "missing_host_catalog"
    assert (tmp_path / "new/review-pages.json").exists()


def test_ambiguous_host_json_rejected_before_snapshot_or_fetch(tmp_path, monkeypatch):
    path = tmp_path / "catalog.json"
    path.write_text('{"tools":[],"tools":[{"name":"different"}]}')
    monkeypatch.setattr(intake, "bundle_from_snapshot", lambda *a: pytest.fail("ambiguous catalog must fail first"))
    with pytest.raises(ValueError, match="duplicate"):
        intake.prepare_intake("unused", "unused", tmp_path / "out", host_catalog=path)
