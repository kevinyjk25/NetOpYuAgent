"""Lossless, inert Skill/host intake before translation, never Runtime authority.

Keep this research input layer separate from the strict executable-package gate.
Reference roles are lexical hints, not entailment or a permission to fetch URLs.
Existing snapshots and the frozen FlowSources/ReadObjectSchema stay unchanged.
"""

from __future__ import annotations

from skill_authoring.source import (
    _digest as _digest, _seal, _repo_path, _resolve as _resolve,
    reference_occurrences as reference_occurrences, _document, _bundle, validate_bundle, review_pages,
)

import argparse
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator, SchemaError
from pydantic import ValidationError

from evaluation.public_skill_corpus import _git_run, _parse_git_tree, _validate_git_source, inspect_public_snapshot
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import ReadObjectSchema
from network_runtime.l0.read_contracts import _source_object

PROTOCOL = "effect-runtime.io/translation-intake/v1"
MAX_FILES = 256
MAX_FILE_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 8 * MAX_FILE_BYTES
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_RESOURCE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*(?:scripts|references|assets)/[^\s`'\"<>()\[\]]+)")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


















def bundle_from_snapshot(snapshot_root: str | Path, candidate_id: str) -> dict:
    root = Path(snapshot_root).resolve()
    checked = inspect_public_snapshot(root)
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    selected = [row for row in records if row["candidateId"] == candidate_id]
    if len(selected) != 1 or selected[0]["status"] != "accepted":
        raise ValueError("candidate does not have one accepted source snapshot")
    row = selected[0]
    prefix = row["sourcePath"].strip("/")
    docs = []
    for item in row["files"]:
        docs.append(_document(f"{prefix}/{item['path']}" if prefix else item["path"],
                    (root / "packages" / row["packageId"] / item["path"]).read_bytes(),
                    mode="100644", origin="snapshot_package"))
    for item in row.get("quarantinedFiles", []):
        path = f"{prefix}/{item['sourcePath']}" if prefix else item["sourcePath"]
        if item["evidencePath"] is not None:
            docs.append(_document(path, (root / item["evidencePath"]).read_bytes(),
                        mode=item["sourceMode"], origin="snapshot_quarantine"))
        else:
            docs.append({"path": path, "bytes": item["sourceBytes"], "sha256": item["sourceDigest"],
                         "content": None, "mode": item["sourceMode"], "origin": "snapshot_quarantine",
                         "executionAllowed": False, "representation": "symlink_target_text_not_followed"
                         if item["sourceMode"] == "120000" else "binary_hash_only"})
    return _bundle({
        "apiVersion": PROTOCOL, "candidateId": candidate_id, "repository": row["repository"],
        "commitSha": row["commitSha"], "snapshotDigest": checked["manifestDigest"],
        "entryPath": f"{prefix}/SKILL.md" if prefix else "SKILL.md", "documents": docs,
        "unavailableSnapshotResources": row.get("withheldFiles", []),
        "supplementAttempts": [], "parentBundleDigest": None,
    })


def supplement_bundle(bundle: dict, paths: list[str], fetch: Callable[[str], dict]) -> dict:
    """Fetch only an explicit caller-selected path list from a pinned provider.

    No recursive crawling, URL fetch, symlink following, source execution or
    candidate replacement. The provider must bind repository/commit separately.
    Failed attempts remain in a new child bundle, never overwrite its parent.
    """
    validate_bundle(bundle)
    selected = [_repo_path(path) for path in paths]
    if len(selected) != len(set(selected)) or len(selected) > MAX_FILES:
        raise ValueError("supplement paths must be unique and bounded")
    docs = list(bundle["documents"])
    existing = {doc["path"] for doc in docs}
    if existing.intersection(selected):
        raise ValueError("supplement cannot replace an existing source")
    attempts = []
    for path in selected:
        try:
            fetched = fetch(path)
            if (fetched.get("repository") != bundle["repository"] or fetched.get("commitSha") != bundle["commitSha"]
                    or fetched.get("path") != path):
                raise ValueError("supplement source repository/commit/path mismatch")
            doc = _document(path, fetched["data"], mode=fetched["mode"], origin="explicit_pinned_repository_supplement")
            if len(docs) + 1 > MAX_FILES or sum(d["bytes"] for d in docs) + doc["bytes"] > MAX_BUNDLE_BYTES:
                raise ValueError("source bundle limit exceeded")
            docs.append(doc)
            attempts.append({"path": path, "status": "saved", "sha256": doc["sha256"]})
        except (ValueError, OSError) as error:
            attempts.append({"path": path, "status": "unavailable", "reason": str(error)[:500]})
    body = {k: v for k, v in bundle.items() if k not in {
        "bundleDigest", "references", "sourceBytes", "sourceCharacters", "closureStatus",
        "runtimeAuthorityGranted", "thirdPartyExecutionAttempted",
    }}
    return _bundle({**body, "documents": docs, "parentBundleDigest": bundle["bundleDigest"],
                    "supplementAttempts": bundle["supplementAttempts"] + attempts})


def supplement_from_git(bundle: dict, paths: list[str]) -> dict:
    """Retrieve explicit repository files at the exact recorded commit, no checkout."""
    validate_bundle(bundle)
    for path in paths:
        _repo_path(path)
    if not paths or len(paths) > MAX_FILES or len(paths) != len(set(paths)):
        raise ValueError("explicit unique bounded supplement paths required")
    owner, repo = bundle["repository"].split("/")
    commit = bundle["commitSha"]
    _validate_git_source(owner, repo, commit)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("supplement requires an exact commit, not a moving branch")
    with tempfile.TemporaryDirectory(prefix="ensuredskill-source-") as scratch:
        bare = str(Path(scratch) / "source.git")
        _git_run(["init", "--bare", bare])
        _git_run(["--git-dir", bare, "fetch", "--depth", "1", "--filter=blob:none",
                  f"https://github.com/{owner}/{repo}.git", commit])
        observed = _git_run(["--git-dir", bare, "rev-parse", "FETCH_HEAD^{commit}"]).decode().strip()
        if observed != commit:
            raise ValueError("fetched supplement commit differs from source snapshot")

        def fetch(path):
            entries = _parse_git_tree(_git_run([
                "--literal-pathspecs", "--git-dir", bare, "ls-tree", "-z", "-l", commit, "--", path,
            ]))
            if len(entries) != 1 or entries[0]["path"] != path or entries[0]["type"] != "blob":
                raise ValueError("explicit supplement path is not one file blob")
            entry = entries[0]
            if entry["size"] > MAX_FILE_BYTES:
                raise ValueError("supplement blob exceeds byte limit")
            data = _git_run(["--git-dir", bare, "cat-file", "blob", entry["sha"]], max_bytes=MAX_FILE_BYTES)
            return {"repository": bundle["repository"], "commitSha": commit, "path": path,
                    "mode": entry["mode"], "data": data}

        return supplement_bundle(bundle, paths, fetch)




def host_catalog_diagnostic(catalog: dict) -> dict:
    """Retain raw MCP-like schemas and identify exact current compiler gaps.

    Never drop enum/constraints, rename keys, coerce objects to strings, invent
    an output schema, resolve remote $refs or derive permission from hints.
    """
    wire = json.dumps(catalog, ensure_ascii=False, allow_nan=False)
    if len(wire.encode()) > MAX_BUNDLE_BYTES or not isinstance(catalog, dict):
        raise ValueError("host catalog must be a bounded JSON object")
    raw = json.loads(wire)
    tools = raw.get("tools")
    if not isinstance(tools, list) or not 1 <= len(tools) <= 128:
        raise ValueError("host catalog requires a bounded tools list")
    names, diagnostics = set(), []
    for index, tool in enumerate(tools):
        if (not isinstance(tool, dict) or not isinstance(tool.get("name"), str)
                or not tool["name"].strip() or tool["name"] in names):
            raise ValueError("host catalog tool names must be nonblank and unique")
        names.add(tool["name"])
        for role in ("inputSchema", "outputSchema"):
            pointer = f"/tools/{index}/{role}"
            if role not in tool:
                diagnostics.append({"pointer": pointer, "code": "missing_host_schema"})
                continue
            schema = tool[role]
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as error:
                diagnostics.append({"pointer": pointer, "code": "invalid_json_schema", "detail": error.message})
                continue
            try:
                ReadObjectSchema.model_validate(schema)
            except ValidationError as error:
                for finding in error.errors(include_url=False, include_input=False):
                    path = "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in finding["loc"])
                    diagnostics.append({"pointer": pointer + ("/" + path if path else ""),
                                        "code": "unsupported_current_read_contract", "detail": finding["msg"]})
    return _seal({"apiVersion": PROTOCOL, "catalog": raw, "catalogDigest": sha256_json(raw),
                  "diagnostics": diagnostics, "currentReadSchemaCompatible": not diagnostics,
                  "semanticAlignmentProven": False, "runtimeAuthorityGranted": False,
                  "claimBoundary": "Declared schemas are not authenticated tools, read-only proof or execution permission."},
                 "diagnosticDigest")


def write_intake(bundle: dict, output: str | Path, *, catalog: dict | None = None) -> dict:
    validate_bundle(bundle)
    pages = review_pages(bundle)
    host = host_catalog_diagnostic(catalog) if catalog is not None else None
    report = _seal({
        "apiVersion": PROTOCOL, "bundleDigest": bundle["bundleDigest"], "pagesDigest": pages["pagesDigest"],
        "hostDiagnosticDigest": host["diagnosticDigest"] if host else None,
        "fileCount": len(bundle["documents"]), "sourceCharacters": bundle["sourceCharacters"],
        "pageCount": len(pages["pages"]), "referenceCount": len(bundle["references"]),
        "referenceContexts": dict(Counter(r["contextRole"] for r in bundle["references"])),
        "missingNonTemplateReferences": [r for r in bundle["references"] if
            r["contextRole"] != "template_example_candidate" and r["availability"] != "inert_text_present"],
        "hostStatus": "diagnosed_not_authorized" if host else "missing_host_catalog",
        "status": "pending_source_task_host_alignment", "translationMetrics": None,
        "runtimeAuthorityGranted": False, "thirdPartyExecutionAttempted": False,
    }, "reportDigest")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    artifacts = {"bundle.json": bundle, "review-pages.json": pages, "report.json": report}
    if host is not None:
        artifacts["host-diagnostic.json"] = host
    for name, data in artifacts.items():
        (root / name).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def prepare_intake(snapshot: str | Path, candidate_id: str, output: str | Path, *,
                   host_catalog: str | Path | None = None, supplement_paths: list[str] | None = None) -> dict:
    """Shared corpus CLI entry: validate output/catalog before any optional fetch."""
    if Path(output).exists():
        raise ValueError("output must not exist; do not overwrite source evidence")
    host = None
    if host_catalog is not None:
        path = Path(host_catalog)
        if path.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError("host catalog byte limit exceeded")
        host = _source_object(path.read_text(encoding="utf-8"))
        host_catalog_diagnostic(host)
    bundle = bundle_from_snapshot(snapshot, candidate_id)
    if supplement_paths:
        bundle = supplement_from_git(bundle, supplement_paths)
    return write_intake(bundle, output, catalog=host)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("candidate_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-catalog")
    parser.add_argument("--supplement-path", action="append", default=[],
                        help="Explicit same-repository, same-commit file to fetch as inert evidence; repeatable")
    args = parser.parse_args()
    print(json.dumps(prepare_intake(args.snapshot, args.candidate_id, args.output,
                                   host_catalog=args.host_catalog, supplement_paths=args.supplement_path)))


if __name__ == "__main__":
    main()
