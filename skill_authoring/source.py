from __future__ import annotations
import hashlib
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit
from network_runtime.contracts import sha256_json
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
