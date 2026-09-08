"""Lossless, inert Skill/host intake before translation, never Runtime authority.

Keep this research input layer separate from the strict executable-package gate.
Reference roles are lexical hints, not entailment or a permission to fetch URLs.
Existing snapshots and the frozen FlowSources/ReadObjectSchema stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import unquote, urlsplit

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


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _seal(value: dict, field: str) -> dict:
    return {**value, field: sha256_json(value)}


def _repo_path(path: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("invalid repository path")
    raw = PurePosixPath(path)
    if raw.is_absolute() or any(part in {".", ".."} for part in path.split("/")):
        raise ValueError("repository path must be normalized and confined")
    if raw.as_posix() != path:
        raise ValueError("repository path must be canonical")
    return path


def _resolve(source: str, target: str) -> tuple[str | None, str]:
    raw = unquote(target.strip().strip("<>"))
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None, "malformed_reference"
    if parsed.scheme or parsed.netloc:
        return None, "external_url_not_fetched"
    if not parsed.path:
        return source, "same_document_anchor"
    if parsed.path.startswith("/") or "\\" in parsed.path or "\x00" in parsed.path:
        return None, "unsafe_path"
    parts = list(PurePosixPath(source).parent.parts)
    for part in parsed.path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None, "outside_repository"
            parts.pop()
        else:
            parts.append(part)
    return ("/".join(parts), "repository_relative") if parts else (None, "unsafe_path")


def reference_occurrences(path: str, text: str) -> list[dict]:
    """Locate candidates with offsets and fence context; never infer obligations.

    Markdown inside an output-template fence remains visible as an example,
    while shell/python resource mentions remain code-context candidates. We do
    not suppress all fenced references or relax the executable package gate.
    This deliberately does not claim to be a complete Markdown/import parser.
    """
    _repo_path(path)
    result, fence, language, offset = [], None, "", 0
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        marker = _FENCE.match(line)
        if marker:
            token, tail = marker.groups()
            if fence is None:
                fence, language = token, tail.strip().split(" ", 1)[0].lower()
            elif token[0] == fence[0] and len(token) >= len(fence) and not tail.strip():
                fence, language = None, ""
        found: set[tuple[int, int]] = set()
        for pattern, kind in ((_LINK, "markdown_link"), (_RESOURCE, "resource_mention")):
            for match in pattern.finditer(line):
                start, end = match.span(1)
                if any(a <= start and end <= b for a, b in found):
                    continue
                found.add((start, end))
                target, resolution = _resolve(path, match.group(1))
                candidates = [target] if target else []
                if kind == "resource_mention" and resolution == "repository_relative":
                    # Code/prose path mentions may be repo-root-relative rather
                    # than document-relative. Keep both; never guess from names.
                    raw = unquote(match.group(1))
                    try:
                        rooted = _repo_path(raw)
                    except ValueError:
                        rooted = None
                    if rooted and rooted != target:
                        candidates.append(rooted)
                        target, resolution = None, "ambiguous_reference_base"
                role = ("template_example_candidate" if fence and language in {"md", "markdown"}
                        else "code_reference_candidate" if fence else "prose_reference_candidate")
                result.append({
                    "sourcePath": path, "line": number, "start": offset + start, "end": offset + end,
                    "rawTarget": match.group(1), "targetPath": target, "resolution": resolution,
                    "candidatePaths": candidates,
                    "syntax": kind, "contextRole": role,
                    "referenceId": _digest(f"{path}:{offset + start}:{offset + end}".encode()),
                })
        offset += len(line)
    return result


def _document(path: str, data: bytes, *, mode: str, origin: str) -> dict:
    _repo_path(path)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("source file byte limit exceeded")
    if mode not in {"100644", "100755", "120000"}:
        raise ValueError("source must be a file blob; submodules are not imported")
    try:
        text = data.decode("utf-8", errors="strict") if b"\x00" not in data else None
    except UnicodeDecodeError:
        text = None
    return {
        "path": path, "bytes": len(data), "sha256": _digest(data), "content": text,
        "mode": mode, "origin": origin, "executionAllowed": False,
        "representation": "symlink_target_text_not_followed" if mode == "120000" else
        "inert_utf8_text" if text is not None else "binary_hash_only",
    }


def _bundle(body: dict) -> dict:
    docs = sorted(body["documents"], key=lambda doc: doc["path"])
    if len(docs) > MAX_FILES or sum(d["bytes"] for d in docs) > MAX_BUNDLE_BYTES:
        raise ValueError("source bundle limit exceeded; never truncate")
    by_path = {d["path"]: d for d in docs}
    if len(by_path) != len(docs):
        raise ValueError("duplicate source paths")
    refs = []
    for doc in docs:
        if doc["content"] is None or doc["mode"] == "120000":
            continue
        for ref in reference_occurrences(doc["path"], doc["content"]):
            target = by_path.get(ref["targetPath"])
            present = [p for p in ref["candidatePaths"] if p in by_path and by_path[p]["representation"] == "inert_utf8_text"]
            ref["presentCandidatePaths"] = present
            ref["availability"] = ("inert_text_present" if target and target["representation"] == "inert_utf8_text"
                                   else "non_text_or_link" if target
                                   else "ambiguous_base_candidate_present" if present
                                   else "directory_listing_present" if ref["rawTarget"].endswith("/") and ref["targetPath"]
                                   and any(p.startswith(ref["targetPath"] + "/") for p in by_path)
                                   else "not_in_bundle")
            ref["sourceDigest"] = doc["sha256"]
            refs.append(ref)
    stable = {**body, "documents": sorted(docs, key=lambda d: d["path"]), "references": refs,
              "sourceBytes": sum(d["bytes"] for d in docs),
              "sourceCharacters": sum(len(d["content"] or "") for d in docs),
              "closureStatus": "pending_semantic_reference_review",
              "runtimeAuthorityGranted": False, "thirdPartyExecutionAttempted": False}
    return _seal(stable, "bundleDigest")


def validate_bundle(bundle: dict) -> None:
    if bundle.get("apiVersion") != PROTOCOL:
        raise ValueError("unknown intake protocol")
    body = {k: v for k, v in bundle.items() if k != "bundleDigest"}
    if bundle.get("bundleDigest") != sha256_json(body):
        raise ValueError("source bundle digest drift")
    if bundle.get("runtimeAuthorityGranted") is not False or bundle.get("thirdPartyExecutionAttempted") is not False:
        raise ValueError("source bundle has no execution authority")
    paths = set()
    for doc in bundle["documents"]:
        _repo_path(doc["path"])
        if (type(doc["bytes"]) is not int or not 0 <= doc["bytes"] <= MAX_FILE_BYTES
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", doc["sha256"])
                or doc["mode"] not in {"100644", "100755", "120000"}):
            raise ValueError("invalid source document bounds or metadata")
        if doc["path"] in paths or doc["executionAllowed"] is not False:
            raise ValueError("duplicate or executable source document")
        paths.add(doc["path"])
        expected = ("symlink_target_text_not_followed" if doc["mode"] == "120000" else
                    "inert_utf8_text" if doc["content"] is not None else "binary_hash_only")
        if doc["representation"] != expected:
            raise ValueError("source representation contradicts mode or content")
        if doc["content"] is not None:
            data = doc["content"].encode("utf-8")
            if len(data) != doc["bytes"] or _digest(data) != doc["sha256"]:
                raise ValueError("source document digest drift")
    if _bundle({k: v for k, v in body.items() if k not in {
        "references", "sourceBytes", "sourceCharacters", "closureStatus",
        "runtimeAuthorityGranted", "thirdPartyExecutionAttempted",
    }}) != bundle:
        raise ValueError("source reference derivation drift")


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


def review_pages(bundle: dict, *, max_characters: int = 12000) -> dict:
    """Lossless per-file pages, with exact offsets even for a single long line.

    This removes the need to concatenate/truncate an entire Skill for review.
    Pagination is not complete workflow compilation or semantic coverage.
    """
    validate_bundle(bundle)
    if type(max_characters) is not int or not 256 <= max_characters <= 64000:
        raise ValueError("invalid review page character budget")
    pages, unavailable = [], []
    for doc in bundle["documents"]:
        text = doc["content"]
        if text is None or doc["mode"] == "120000":
            unavailable.append(doc["path"])
            continue
        for start in range(0, len(text), max_characters):
            end = min(start + max_characters, len(text))
            pages.append({
                "pageId": _digest(f"{doc['path']}:{doc['sha256']}:{start}:{end}".encode()),
                "path": doc["path"], "sourceDigest": doc["sha256"],
                "start": start, "end": end, "text": text[start:end],
                "startLine": text.count("\n", 0, start) + 1,
                "endLine": text.count("\n", 0, max(start, end - 1)) + 1,
                "referenceIds": [r["referenceId"] for r in bundle["references"]
                                 if r["sourcePath"] == doc["path"] and r["start"] < end and r["end"] > start],
            })
    return _seal({"bundleDigest": bundle["bundleDigest"], "maxCharacters": max_characters,
                  "pages": pages, "unavailableTextPaths": unavailable,
                  "semanticCoverageProven": False, "runtimeAuthorityGranted": False}, "pagesDigest")


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
