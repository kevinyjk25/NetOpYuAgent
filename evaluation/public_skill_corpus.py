"""Read-only public Skill discovery, commit pinning, and static quarantine.

Marketplace packages are untrusted data.  This module never imports, installs,
or executes their contents.  Public-market evidence is ecological-validity
evidence and can never self-declare the independent private ES-P1 gate passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from effect_runtime import inspect_skill_package
from network_runtime.contracts import sha256_json


DISCOVERY_SCHEMA = "effect-runtime.io/public-skill-discovery/v1"
SNAPSHOT_SCHEMA = "effect-runtime.io/public-skill-static-snapshot/v1"
AUTHOR_KIT_SCHEMA = "effect-runtime.io/public-skill-independent-author-kit/v1"
EVIDENCE_CLASS = "public_market_repository_external_static_only_ecological_validity"
EXECUTION_POLICY = "static_only"
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024
_MAX_PACKAGE_BYTES = 4 * 1024 * 1024
_MAX_PACKAGE_FILES = 128
_EXECUTABLE_SUFFIXES = {
    ".bat", ".cjs", ".class", ".cmd", ".com", ".dll", ".dylib", ".exe",
    ".jar", ".js", ".mjs", ".pl", ".ps1", ".py", ".rb", ".sh", ".so",
    ".ts", ".wasm", ".zsh",
}
_EXECUTABLE_DIRS = {".git", ".github", "bin", "hooks", "scripts"}
_EXECUTABLE_NAMES = {
    ".mcp.json", "dockerfile", "hooks.json", "makefile", "package.json",
    "pyproject.toml", "requirements.txt", "setup.cfg", "setup.py",
}
_INSTRUCTION_PATTERNS = {
    "destructive-command": re.compile(r"\b(?:rm\s+-rf|format\s+|mkfs\b)", re.I),
    "external-download": re.compile(r"\b(?:curl|wget)\b", re.I),
    "privilege-escalation": re.compile(r"\b(?:sudo|doas)\b", re.I),
    "shell-or-process": re.compile(r"\b(?:subprocess|os\.system|child_process)\b", re.I),
    "dynamic-evaluation": re.compile(r"\b(?:eval|exec)\s*\(", re.I),
    "package-install": re.compile(r"\b(?:pip|npm|pnpm|yarn|brew|apt)\s+install\b", re.I),
}


def _tls_context() -> ssl.SSLContext:
    """Use a verified CA bundle; never fall back to an unverified context."""

    try:
        import certifi  # type: ignore[import-not-found]
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bounded_get(url: str, *, token: str | None = None, max_bytes: int = _MAX_RESPONSE_BYTES) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "api.github.com", "raw.githubusercontent.com", "skillsmp.com",
    }:
        raise ValueError("public Skill fetch host is not allowlisted")
    headers = {
        "Accept": "application/vnd.github+json" if parsed.hostname == "api.github.com" else "application/json",
        "User-Agent": "NetOpYuAgent-ES-P1-Wild/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(  # noqa: S310 - strict host allowlist above
        request, timeout=30, context=_tls_context(),
    ) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname not in {
            "api.github.com", "raw.githubusercontent.com", "skillsmp.com",
        }:
            raise ValueError("public Skill fetch redirected outside the allowlist")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("public Skill response exceeds size limit")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("public Skill response exceeds size limit")
    return data


def _json_get(url: str, *, token: str | None = None) -> Any:
    return json.loads(_bounded_get(url, token=token).decode("utf-8"))


def _repo_key(github_url: str) -> str:
    parsed = urllib.parse.urlparse(github_url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme != "https" or parsed.hostname != "github.com" or len(parts) < 4:
        raise ValueError("Skill source must be a GitHub tree/blob URL")
    if parts[2] not in {"tree", "blob"}:
        raise ValueError("Skill source must point to a GitHub tree/blob")
    return f"{parts[0].lower()}/{parts[1].lower()}"


def _parse_github_source(github_url: str, *, default_branch: str) -> dict[str, str]:
    parsed = urllib.parse.urlparse(github_url)
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if parsed.scheme != "https" or parsed.hostname != "github.com" or len(parts) < 4:
        raise ValueError("Skill source must be a GitHub tree/blob URL")
    owner, repo, kind = parts[:3]
    if kind not in {"tree", "blob"}:
        raise ValueError("Skill source must point to a GitHub tree/blob")
    tail = parts[3:]
    branch_parts = [part for part in default_branch.split("/") if part]
    if tail[:len(branch_parts)] == branch_parts:
        ref = default_branch
        source_parts = tail[len(branch_parts):]
    elif tail[0] in {"main", "master"}:
        ref = tail[0]
        source_parts = tail[1:]
    else:
        ref = default_branch
        source_parts = tail
    if kind == "blob" and source_parts:
        source_parts = source_parts[:-1]
    if any(part in {"", ".", ".."} for part in source_parts):
        raise ValueError("Skill source package path is invalid")
    return {
        "owner": owner, "repo": repo, "ref": ref,
        "path": "/".join(source_parts),
    }


def _market_token() -> str | None:
    return os.environ.get("SKILLSMP_API_KEY") or None


def _github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None


def discover_public_skills(
    *, queries: Iterable[str], limit: int = 80, per_query: int = 20,
    max_per_repo: int = 5, language: str | None = None,
    sort_by: str = "recent",
) -> dict[str, Any]:
    if not 1 <= limit <= 500 or not 1 <= per_query <= 50 or max_per_repo < 1:
        raise ValueError("invalid discovery limits")
    normalized_queries = tuple(dict.fromkeys(value.strip() for value in queries if value.strip()))
    if not normalized_queries:
        raise ValueError("at least one non-empty discovery query is required")
    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    repo_counts: Counter[str] = Counter()
    for query in normalized_queries:
        params = {"q": query, "page": 1, "limit": per_query, "sortBy": sort_by}
        if language:
            params["language"] = language
        url = "https://skillsmp.com/api/v1/skills/search?" + urllib.parse.urlencode(params)
        payload = _json_get(url, token=_market_token())
        values = ((payload or {}).get("data") or {}).get("skills") or []
        for item in values:
            if not isinstance(item, dict):
                continue
            source = str(item.get("githubUrl") or "").strip()
            try:
                repo_key = _repo_key(source)
            except ValueError:
                continue
            normalized_url = source.rstrip("/")
            if normalized_url.lower() in seen_urls or repo_counts[repo_key] >= max_per_repo:
                continue
            candidate = {
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "author": str(item.get("author") or ""),
                "description": str(item.get("description") or ""),
                "language": str(item.get("contentLanguage") or "und"),
                "githubUrl": normalized_url,
                "skillUrl": str(item.get("skillUrl") or ""),
                "stars": int(item.get("stars") or 0),
                "updatedAt": item.get("updatedAt"),
                "discoveryQuery": query,
            }
            if not candidate["id"] or not candidate["name"]:
                continue
            selected.append(candidate)
            seen_urls.add(normalized_url.lower())
            repo_counts[repo_key] += 1
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    body = {
        "apiVersion": DISCOVERY_SCHEMA,
        "createdAt": _utc_now(),
        "source": "SkillsMP",
        "queries": list(normalized_queries),
        "sortBy": sort_by,
        "language": language,
        "requestedLimit": limit,
        "maxPerRepository": max_per_repo,
        "candidateCount": len(selected),
        "candidates": selected,
        "claimBoundary": "Discovery metadata is not a quality, safety, license, or ES-P1 qualification label.",
    }
    return {**body, "discoveryDigest": sha256_json(body)}


def write_discovery(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    result = discover_public_skills(**kwargs)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def load_discovery(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("apiVersion") != DISCOVERY_SCHEMA:
        raise ValueError("public Skill discovery Schema is invalid")
    digest = value.get("discoveryDigest")
    body = {key: item for key, item in value.items() if key != "discoveryDigest"}
    if digest != sha256_json(body) or len(value.get("candidates") or []) != value.get("candidateCount"):
        raise ValueError("public Skill discovery digest or count drift")
    return value


def _safe_relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("public Skill tree contains an unsafe path")
    return relative


def _executable_surface(path: PurePosixPath, mode: str, object_type: str) -> str | None:
    if mode == "120000":
        return "symlink"
    if mode == "160000" or object_type != "blob":
        return "special-or-submodule"
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & _EXECUTABLE_DIRS:
        return "executable-directory"
    if path.suffix.lower() in _EXECUTABLE_SUFFIXES or path.name.lower() in _EXECUTABLE_NAMES:
        return "executable-file"
    return None


def _package_id(candidate: dict[str, Any]) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", str(candidate["id"]).lower()).strip("-")[:72] or "skill"
    return f"{stem}-{hashlib.sha256(str(candidate['githubUrl']).encode()).hexdigest()[:10]}"


def _raw_url(owner: str, repo: str, commit: str, path: str) -> str:
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return f"https://raw.githubusercontent.com/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/{commit}/{encoded}"


def _git_run(args: list[str], *, max_bytes: int = _MAX_RESPONSE_BYTES) -> bytes:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    completed = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never", *args],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=90, env=env,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise ValueError(f"Git metadata operation failed: {detail}")
    if len(completed.stdout) > max_bytes:
        raise ValueError("Git metadata response exceeds size limit")
    return completed.stdout


def _validate_git_source(owner: str, repo: str, ref: str) -> None:
    atom = re.compile(r"[A-Za-z0-9_.-]+")
    if not atom.fullmatch(owner) or not atom.fullmatch(repo):
        raise ValueError("GitHub owner/repository contains unsupported characters")
    if (
        not ref or ref.startswith("-") or ".." in ref
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", ref)
    ):
        raise ValueError("GitHub ref contains unsupported characters")


def _parse_git_tree(raw: bytes) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, path = record.split(b"\t", 1)
        header_parts = header.decode("ascii").split(" ")
        if len(header_parts) == 3:
            mode, object_type, object_sha = header_parts
            size = "-"
        elif len(header_parts) == 4:
            mode, object_type, object_sha, size = header_parts
        else:
            raise ValueError("Git tree record has an invalid shape")
        values.append({
            "path": path.decode("utf-8", errors="strict"), "mode": mode,
            "type": object_type, "sha": object_sha,
            "size": 0 if size == "-" else int(size),
        })
    return values


def _git_repository(
    owner: str, repo: str, ref: str, scratch: Path,
) -> tuple[str, list[dict[str, Any]], Path, str]:
    _validate_git_source(owner, repo, ref)
    bare = scratch / hashlib.sha256(f"{owner}/{repo}@{ref}".encode()).hexdigest()[:20]
    _git_run([
        "clone", "--bare", "--filter=blob:none", "--depth", "1", "--single-branch",
        "--branch", ref, f"https://github.com/{owner}/{repo}.git", str(bare),
    ])
    commit = _git_run(["--git-dir", str(bare), "rev-parse", "HEAD"]).decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Git source commit could not be pinned")
    tree = _parse_git_tree(_git_run(["--git-dir", str(bare), "ls-tree", "-r", "-z", commit]))
    root_tree = _parse_git_tree(_git_run(["--git-dir", str(bare), "ls-tree", "-z", commit]))
    license_spdx = "NOASSERTION"
    for item in root_tree:
        name = PurePosixPath(item["path"]).name.lower()
        if item["type"] != "blob" or not (name.startswith("license") or name.startswith("copying")):
            continue
        license_text = _git_run(
            ["--git-dir", str(bare), "show", f"{commit}:{item['path']}"],
            max_bytes=_MAX_FILE_BYTES,
        ).decode("utf-8", errors="replace").lower()
        if "permission is hereby granted, free of charge" in license_text:
            license_spdx = "MIT"
        elif "apache license" in license_text and "version 2.0" in license_text:
            license_spdx = "Apache-2.0"
        elif "mozilla public license" in license_text:
            license_spdx = "MPL-2.0"
        elif "gnu general public license" in license_text:
            license_spdx = "GPL-LicenseRef"
        elif "redistribution and use in source and binary forms" in license_text:
            license_spdx = "BSD-LicenseRef"
        elif "permission to use, copy, modify, and/or distribute this software" in license_text:
            license_spdx = "ISC"
        if license_spdx != "NOASSERTION":
            break
    return commit, tree, bare, license_spdx


def _git_blob(bare: Path, commit: str, path: str) -> bytes:
    return _git_run(
        ["--git-dir", str(bare), "show", f"{commit}:{path}"], max_bytes=_MAX_FILE_BYTES,
    )


def _repository_tree(owner: str, repo: str, ref: str, *, token: str | None) -> tuple[str, list[dict[str, Any]]]:
    base = f"https://api.github.com/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}"
    commit = _json_get(f"{base}/commits/{urllib.parse.quote(ref, safe='')}", token=token)
    sha = str((commit or {}).get("sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GitHub source commit could not be pinned")
    tree = _json_get(f"{base}/git/trees/{sha}?recursive=1", token=token)
    if not isinstance(tree, dict) or tree.get("truncated") is True or not isinstance(tree.get("tree"), list):
        raise ValueError("GitHub recursive tree is missing or truncated")
    return sha, tree["tree"]


def _package_entries(tree: list[dict[str, Any]], package_path: str) -> list[dict[str, Any]]:
    prefix = package_path.rstrip("/") + "/" if package_path else ""
    values = [item for item in tree if isinstance(item, dict) and str(item.get("path") or "").startswith(prefix)]
    normalized: list[dict[str, Any]] = []
    total = 0
    for item in values:
        source_path = str(item.get("path") or "")
        relative = _safe_relative(source_path[len(prefix):])
        kind = str(item.get("type") or "")
        if kind == "tree":
            continue
        size = int(item.get("size") or 0)
        if size < 0 or size > _MAX_FILE_BYTES:
            raise ValueError("public Skill file exceeds size limit")
        total += size
        normalized.append({**item, "relative": relative.as_posix(), "size": size})
    if len(normalized) > _MAX_PACKAGE_FILES or total > _MAX_PACKAGE_BYTES:
        raise ValueError("public Skill package exceeds static quarantine limits")
    if not any(item["relative"] == "SKILL.md" for item in normalized):
        raise ValueError("public Skill package does not contain root SKILL.md")
    return sorted(normalized, key=lambda item: item["relative"])


def snapshot_public_skills(
    discovery_path: str | Path, output_root: str | Path, *, limit: int = 20,
    script_policy: str = "exclude", license_policy: str = "known",
    source_backend: str = "api",
) -> dict[str, Any]:
    if (
        script_policy not in {"exclude", "metadata-only"}
        or license_policy not in {"known", "record-only"}
        or source_backend not in {"api", "git"}
        or limit < 1
    ):
        raise ValueError("invalid public Skill snapshot policy")
    discovery = load_discovery(discovery_path)
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("public Skill snapshot root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    package_root = root / "packages"
    package_root.mkdir()
    github_token = _github_token()
    repo_meta_cache: dict[str, dict[str, Any]] = {}
    tree_cache: dict[tuple[str, str, str], tuple[str, list[dict[str, Any]]]] = {}
    git_cache: dict[tuple[str, str, str], tuple[str, list[dict[str, Any]], Path, str]] = {}
    scratch_handle = tempfile.TemporaryDirectory(prefix="netopyu-market-static-")
    scratch = Path(scratch_handle.name)
    records: list[dict[str, Any]] = []
    accepted = 0
    for candidate in discovery["candidates"]:
        if accepted >= limit:
            break
        record: dict[str, Any] = {
            "candidateId": candidate["id"], "name": candidate["name"],
            "language": candidate["language"], "githubUrl": candidate["githubUrl"],
            "status": "excluded", "executionPolicy": EXECUTION_POLICY,
        }
        target_root: Path | None = None
        try:
            repo_key = _repo_key(candidate["githubUrl"])
            owner, repo = repo_key.split("/", 1)
            url_parts = [part for part in urllib.parse.urlparse(candidate["githubUrl"]).path.split("/") if part]
            url_ref = urllib.parse.unquote(url_parts[3])
            if source_backend == "api":
                if repo_key not in repo_meta_cache:
                    repo_meta_cache[repo_key] = _json_get(
                        f"https://api.github.com/repos/{owner}/{repo}", token=github_token,
                    )
                meta = repo_meta_cache[repo_key]
                license_spdx = str(((meta.get("license") or {}).get("spdx_id") or "NOASSERTION"))
                default_branch = str(meta.get("default_branch") or "")
                if not default_branch:
                    raise ValueError("GitHub repository has no default branch")
            else:
                default_branch = url_ref
            source = _parse_github_source(candidate["githubUrl"], default_branch=default_branch)
            cache_key = (owner, repo, source["ref"])
            bare: Path | None = None
            if source_backend == "api":
                if cache_key not in tree_cache:
                    tree_cache[cache_key] = _repository_tree(owner, repo, source["ref"], token=github_token)
                commit, tree = tree_cache[cache_key]
            else:
                if cache_key not in git_cache:
                    git_cache[cache_key] = _git_repository(owner, repo, source["ref"], scratch)
                commit, tree, bare, license_spdx = git_cache[cache_key]
            record.update({
                "repository": repo_key, "licenseSpdx": license_spdx,
                "licenseSource": "github-api" if source_backend == "api" else "static-license-text-detection",
            })
            if license_policy == "known" and license_spdx in {"", "NOASSERTION", "OTHER"}:
                record["reason"] = "license_not_declared"
                records.append(record)
                continue
            entries = _package_entries(tree, source["path"])
            git_blobs: dict[str, bytes] = {}
            if source_backend == "git":
                total_bytes = 0
                for entry in entries:
                    source_path = "/".join(
                        part for part in (source["path"], entry["relative"]) if part
                    )
                    blob = _git_blob(bare, commit, source_path)  # type: ignore[arg-type]
                    entry["size"] = len(blob)
                    total_bytes += len(blob)
                    if total_bytes > _MAX_PACKAGE_BYTES:
                        raise ValueError("public Skill package exceeds static quarantine limits")
                    git_blobs[entry["relative"]] = blob
            surfaces = sorted({
                risk for item in entries
                if (risk := _executable_surface(
                    PurePosixPath(item["relative"]), str(item.get("mode") or ""), str(item.get("type") or ""),
                ))
            })
            record.update({
                "repository": repo_key, "sourceRef": source["ref"], "sourcePath": source["path"],
                "commitSha": commit,
                "licenseSpdx": license_spdx,
                "fileCount": len(entries), "byteCount": sum(item["size"] for item in entries),
                "executableSurface": surfaces,
            })
            if surfaces and script_policy == "exclude":
                record["reason"] = "executable_surface_excluded"
                records.append(record)
                continue
            package_id = _package_id(candidate)
            target_root = package_root / package_id
            target_root.mkdir()
            file_records: list[dict[str, Any]] = []
            instruction_findings: set[str] = set()
            for entry in entries:
                if _executable_surface(
                    PurePosixPath(entry["relative"]), str(entry.get("mode") or ""), str(entry.get("type") or ""),
                ) and script_policy == "metadata-only":
                    continue
                source_path = "/".join(part for part in (source["path"], entry["relative"]) if part)
                data = (
                    _bounded_get(_raw_url(owner, repo, commit, source_path), max_bytes=_MAX_FILE_BYTES)
                    if source_backend == "api"
                    else git_blobs[entry["relative"]]
                )
                if len(data) != entry["size"]:
                    raise ValueError("public Skill blob size drift")
                target = (target_root / entry["relative"]).resolve()
                if not target.is_relative_to(target_root.resolve()):
                    raise ValueError("public Skill target path escapes quarantine")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if b"\x00" not in data:
                    text = data.decode("utf-8", errors="replace")
                    instruction_findings.update(
                        code for code, pattern in _INSTRUCTION_PATTERNS.items() if pattern.search(text)
                    )
                file_records.append({
                    "path": entry["relative"], "bytes": len(data), "sha256": _file_digest(target),
                })
            package_digest = sha256_json(file_records)
            record.update({
                "status": "accepted", "packageId": package_id,
                "packageDigest": package_digest, "files": file_records,
                "instructionRiskCodes": sorted(instruction_findings),
                "materializedExecutableFiles": False,
            })
            accepted += 1
        except Exception as exc:  # each remote candidate is an independently excluded record
            if target_root is not None and target_root.exists():
                shutil.rmtree(target_root)
            record["reason"] = f"snapshot_error:{type(exc).__name__}:{exc}"
        records.append(record)
    scratch_handle.cleanup()
    records_path = root / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )
    accepted_records = [item for item in records if item["status"] == "accepted"]
    body = {
        "apiVersion": SNAPSHOT_SCHEMA,
        "createdAt": _utc_now(),
        "evidenceClass": EVIDENCE_CLASS,
        "executionPolicy": EXECUTION_POLICY,
        "scriptPolicy": script_policy,
        "licensePolicy": license_policy,
        "sourceBackend": source_backend,
        "officialEsP1QualificationEligible": False,
        "privateHoldout": False,
        "discoveryDigest": discovery["discoveryDigest"],
        "requestedAccepted": limit,
        "acceptedCount": len(accepted_records),
        "excludedCount": len(records) - len(accepted_records),
        "sourceRepositoryCount": len({item["repository"] for item in accepted_records}),
        "coverage": {
            "languages": dict(sorted(Counter(item["language"] for item in accepted_records).items())),
            "instructionRiskCodes": dict(sorted(Counter(code for item in accepted_records for code in item["instructionRiskCodes"]).items())),
        },
        "recordsDigest": _file_digest(records_path),
        "packageDigests": {item["packageId"]: item["packageDigest"] for item in accepted_records},
        "complete": len(accepted_records) == limit,
        "claimBoundary": "Public static-only ecological-validity evidence; not private ES-P1 qualification or production probability.",
    }
    manifest = {**body, "manifestDigest": sha256_json(body)}
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def inspect_public_snapshot(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("apiVersion") != SNAPSHOT_SCHEMA:
        raise ValueError("public Skill snapshot Schema is invalid")
    body = {key: value for key, value in manifest.items() if key != "manifestDigest"}
    if manifest.get("manifestDigest") != sha256_json(body):
        raise ValueError("public Skill snapshot manifest digest drift")
    if any((
        manifest.get("evidenceClass") != EVIDENCE_CLASS,
        manifest.get("executionPolicy") != EXECUTION_POLICY,
        manifest.get("officialEsP1QualificationEligible") is not False,
        manifest.get("privateHoldout") is not False,
    )):
        raise ValueError("public Skill snapshot authority boundary is invalid")
    records_path = root / "records.jsonl"
    if manifest.get("recordsDigest") != _file_digest(records_path):
        raise ValueError("public Skill snapshot record digest drift")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    accepted = [item for item in records if item.get("status") == "accepted"]
    if len(accepted) != manifest.get("acceptedCount"):
        raise ValueError("public Skill snapshot accepted count drift")
    package_gates: Counter[str] = Counter()
    package_findings: Counter[str] = Counter()
    for record in accepted:
        if record.get("materializedExecutableFiles") is not False:
            raise ValueError("public Skill snapshot executable materialization is forbidden")
        package = (root / "packages" / record["packageId"]).resolve()
        if not package.is_relative_to(root) or not package.is_dir() or package.is_symlink():
            raise ValueError("public Skill snapshot package path is invalid")
        expected_paths = {item["path"] for item in record["files"]}
        actual_paths: set[str] = set()
        for path in package.rglob("*"):
            if path.is_symlink():
                raise ValueError("public Skill snapshot cannot contain symlinks")
            if path.is_file():
                actual_paths.add(path.relative_to(package).as_posix())
        if actual_paths != expected_paths:
            raise ValueError("public Skill snapshot contains unsealed files")
        observed: list[dict[str, Any]] = []
        for expected in record["files"]:
            path = (package / expected["path"]).resolve()
            if not path.is_relative_to(package) or not path.is_file() or path.is_symlink():
                raise ValueError("public Skill snapshot file is missing or unsafe")
            current = {"path": expected["path"], "bytes": path.stat().st_size, "sha256": _file_digest(path)}
            if current != expected:
                raise ValueError("public Skill snapshot file digest drift")
            observed.append(current)
        if sha256_json(observed) != record["packageDigest"]:
            raise ValueError("public Skill snapshot package digest drift")
        package_report = inspect_skill_package(package)
        package_gates[str(package_report["gate"])] += 1
        package_findings.update(str(item["code"]) for item in package_report["findings"])
    if manifest.get("packageDigests") != {item["packageId"]: item["packageDigest"] for item in accepted}:
        raise ValueError("public Skill snapshot package index drift")
    return {
        "status": "valid", "acceptedCount": len(accepted),
        "excludedCount": len(records) - len(accepted),
        "complete": manifest["complete"], "manifestDigest": manifest["manifestDigest"],
        "executionPolicy": EXECUTION_POLICY,
        "officialEsP1QualificationEligible": False,
        "coverage": manifest["coverage"],
        "runtimePackageInspection": {
            "gates": dict(sorted(package_gates.items())),
            "findingCounts": dict(sorted(package_findings.items())),
            "executionAttempted": False,
        },
        "claimBoundary": manifest["claimBoundary"],
    }


def build_public_pilot_report(
    root_path: str | Path, output_root: str | Path, *, discovery_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    inspection = inspect_public_snapshot(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    accepted = [item for item in records if item["status"] == "accepted"]
    blocked: list[dict[str, Any]] = []
    for record in accepted:
        package_report = inspect_skill_package(root / "packages" / record["packageId"])
        if package_report["gate"] != "passed":
            blocked.append({
                "candidateId": record["candidateId"], "name": record["name"],
                "repository": record["repository"], "gate": package_report["gate"],
                "findingCodes": sorted({item["code"] for item in package_report["findings"]}),
            })
    discovery_count: int | None = None
    if discovery_path is not None:
        discovery = load_discovery(discovery_path)
        if discovery["discoveryDigest"] != manifest["discoveryDigest"]:
            raise ValueError("public Skill report discovery digest mismatch")
        discovery_count = int(discovery["candidateCount"])
    exclusions = Counter(
        str(item.get("reason") or "unknown").split(":", 1)[0]
        for item in records if item["status"] != "accepted"
    )
    body = {
        "apiVersion": "effect-runtime.io/public-skill-static-pilot-report/v1",
        "generatedAt": _utc_now(), "status": "static_import_pilot_complete_runtime_eval_not_started",
        "evidenceClass": EVIDENCE_CLASS, "officialEsP1QualificationEligible": False,
        "source": {
            "manifestDigest": manifest["manifestDigest"],
            "discoveryDigest": manifest["discoveryDigest"],
            "discoveryCandidateCount": discovery_count,
            "processedCandidateCount": len(records),
            "sourceBackend": manifest.get("sourceBackend"),
        },
        "quarantine": {
            "executionPolicy": EXECUTION_POLICY, "scriptPolicy": manifest["scriptPolicy"],
            "licensePolicy": manifest["licensePolicy"], "executionAttempted": False,
            "materializedExecutableFiles": 0,
        },
        "corpus": {
            "accepted": len(accepted), "excluded": len(records) - len(accepted),
            "repositories": len({item["repository"] for item in accepted}),
            "languages": manifest["coverage"]["languages"],
            "instructionRiskCodes": manifest["coverage"]["instructionRiskCodes"],
            "exclusionReasons": dict(sorted(exclusions.items())),
        },
        "runtimePackageInspection": inspection["runtimePackageInspection"],
        "blockedPackages": blocked,
        "nextGate": (
            "Independent authors must add user tasks, fixtures, Gold semantics, Tool/MCP catalogs, "
            "risk/effect budgets, and expected dispositions before paired DSH/Runtime evaluation."
        ),
        "claimBoundary": manifest["claimBoundary"],
    }
    report = {**body, "reportDigest": sha256_json(body)}
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "public-skill-pilot-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    gates = report["runtimePackageInspection"]["gates"]
    markdown = f"""# 公开 Skill 静态导入 Pilot / Public Skill Static Import Pilot

## 中文

- 状态：`{report['status']}`
- 候选：{discovery_count if discovery_count is not None else '未提供'}；已处理：{len(records)}
- 接纳：{len(accepted)}；拒绝：{len(records) - len(accepted)}；来源仓库：{report['corpus']['repositories']}
- Runtime 包门禁：passed {gates.get('passed', 0)}；blocked {gates.get('blocked', 0)}
- 可执行文件物化：0；第三方代码执行：否
- Manifest：`{manifest['manifestDigest']}`

本报告只证明公开 Skill 的静态发现、固定 commit、许可证门禁、零执行隔离、摘要封存和 Runtime 包格式兼容性。它尚未包含独立任务、Gold/Oracle 或 DSH 配对，因此不是 ES-P1 资格结果。

## English

The pilot accepted {len(accepted)} static-only packages from {report['corpus']['repositories']} repositories after processing {len(records)} candidates. Runtime package inspection passed {gates.get('passed', 0)} and blocked {gates.get('blocked', 0)} packages. No executable file was materialized and no third-party code was executed. This is a static ecosystem-compatibility pilot, not private ES-P1 qualification or paired Runtime evidence.
"""
    (output / "public-skill-pilot-report.md").write_text(markdown, encoding="utf-8")
    return report


def _annotation_schemas() -> dict[str, Any]:
    common_id = {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{1,127}$"}
    digest = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
    return {
        "task": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ES-P1-Wild public Skill task",
            "type": "object", "additionalProperties": False,
            "required": [
                "apiVersion", "taskId", "assignmentId", "packageId", "packageDigest",
                "language", "challenge", "userPrompt", "fixtureRefs", "toolCatalogRef",
                "authorId", "authoredAt",
            ],
            "properties": {
                "apiVersion": {"const": "effect-runtime.io/public-skill-task/v1"},
                "taskId": common_id, "assignmentId": common_id,
                "packageId": common_id, "packageDigest": digest,
                "language": {"enum": ["zh", "en", "mixed"]},
                "challenge": {"enum": ["nominal", "ambiguous_or_missing", "failure_or_adversarial"]},
                "userPrompt": {"type": "string", "minLength": 1, "maxLength": 64000},
                "fixtureRefs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "toolCatalogRef": {"type": "string", "minLength": 1},
                "authorId": common_id, "authoredAt": {"type": "string", "minLength": 1},
            },
        },
        "gold": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ES-P1-Wild public Skill Gold/Oracle",
            "type": "object", "additionalProperties": False,
            "required": [
                "apiVersion", "taskId", "expectedDisposition", "intendedOutcome",
                "requiredCapabilities", "forbiddenCapabilities", "parameters", "risk",
                "approvalRequired", "maxEffectCalls", "preconditions", "verification",
                "recovery", "authorId",
            ],
            "properties": {
                "apiVersion": {"const": "effect-runtime.io/public-skill-gold/v1"},
                "taskId": common_id,
                "expectedDisposition": {
                    "enum": ["proposal", "read_only", "safe_stop_clarify", "safe_stop_reject"],
                },
                "intendedOutcome": {"type": "string", "minLength": 1},
                "requiredCapabilities": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "forbiddenCapabilities": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "parameters": {"type": "object"},
                "risk": {"enum": ["low", "medium", "high", "critical"]},
                "approvalRequired": {"type": "boolean"},
                "maxEffectCalls": {"type": "integer", "minimum": 0, "maximum": 20},
                "preconditions": {"type": "array", "items": {"type": "object"}},
                "verification": {"type": "array", "items": {"type": "object"}},
                "recovery": {"type": "array", "items": {"type": "object"}},
                "authorId": common_id,
            },
        },
        "toolCatalog": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ES-P1-Wild declared Tool/MCP catalog",
            "type": "object", "additionalProperties": False,
            "required": ["apiVersion", "assignmentId", "capabilities"],
            "properties": {
                "apiVersion": {"const": "effect-runtime.io/public-skill-tool-catalog/v1"},
                "assignmentId": common_id,
                "capabilities": {"type": "array", "items": {"type": "object"}},
            },
        },
        "authority": (
            "These schemas collect evaluation evidence only. They grant no Tool, MCP, "
            "Runtime registration, activation, or execution authority."
        ),
    }


def export_public_author_kit(
    snapshot_root: str | Path, output_root: str | Path, *, tasks_per_skill: int = 3,
) -> dict[str, Any]:
    if not 1 <= tasks_per_skill <= 10:
        raise ValueError("tasks per public Skill must be between 1 and 10")
    source_root = Path(snapshot_root).expanduser().resolve()
    snapshot_inspection = inspect_public_snapshot(source_root)
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    source_records = [
        json.loads(line) for line in (source_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected: list[dict[str, Any]] = []
    for record in source_records:
        if record.get("status") != "accepted":
            continue
        report = inspect_skill_package(source_root / "packages" / record["packageId"])
        if report["gate"] == "passed":
            selected.append(record)
    if not selected:
        raise ValueError("public author kit requires at least one package that passes the Runtime gate")
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("public author kit root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    packages = root / "packages"
    packages.mkdir()
    assignments: list[dict[str, Any]] = []
    challenges = ("nominal", "ambiguous_or_missing", "failure_or_adversarial")
    for index, record in enumerate(selected, start=1):
        source_package = source_root / "packages" / record["packageId"]
        shutil.copytree(source_package, packages / record["packageId"], symlinks=False)
        assignment_id = f"wild-assignment-{index:03d}"
        slots = [
            {
                "slotId": f"{assignment_id}-{slot + 1:02d}",
                "challenge": challenges[slot % len(challenges)],
            }
            for slot in range(tasks_per_skill)
        ]
        assignments.append({
            "apiVersion": "effect-runtime.io/public-skill-author-assignment/v1",
            "assignmentId": assignment_id,
            "packageId": record["packageId"], "packageDigest": record["packageDigest"],
            "skillName": record["name"], "repository": record["repository"],
            "commitSha": record["commitSha"], "sourcePath": record["sourcePath"],
            "packageEntry": f"packages/{record['packageId']}/SKILL.md",
            "taskSlots": slots,
        })
    (root / "assignments.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in assignments),
        encoding="utf-8",
    )
    (root / "schemas.json").write_text(
        json.dumps(_annotation_schemas(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# ES-P1-Wild independent annotation kit\n\n"
        "This workspace contains only pinned, static-only public Skill packages that passed "
        "the Runtime package gate, blank task slots, and evidence schemas. It contains no "
        "Runtime implementation, evaluator, model output, generated Gold, credentials, or "
        "execution authority.\n\n"
        "Independent authors create `tasks.jsonl`, `gold.jsonl`, Tool/MCP catalog files, and "
        "fixtures from the assignments. Do not execute package content. Keep Gold private from "
        "the Agent and from reviewers until the preregistered scoring stage. Public-market cases "
        "support ecological validity and never become a private holdout.\n",
        encoding="utf-8",
    )
    sealed_files = {
        path.relative_to(root).as_posix(): _file_digest(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }
    body = {
        "apiVersion": AUTHOR_KIT_SCHEMA, "createdAt": _utc_now(),
        "evidenceClass": "public_market_independent_annotation_workspace",
        "executionPolicy": EXECUTION_POLICY,
        "officialEsP1QualificationEligible": False, "privateHoldout": False,
        "sourceSnapshotManifestDigest": source_manifest["manifestDigest"],
        "sourceSnapshotInspection": {
            "acceptedCount": snapshot_inspection["acceptedCount"],
            "runtimePackageGates": snapshot_inspection["runtimePackageInspection"]["gates"],
        },
        "selectedPackageCount": len(selected),
        "tasksPerSkill": tasks_per_skill, "taskSlotCount": len(selected) * tasks_per_skill,
        "selectedPackageDigests": {item["packageId"]: item["packageDigest"] for item in selected},
        "sealedFiles": sealed_files,
        "containsRuntimeOrEvaluator": False, "containsGeneratedGold": False,
        "thirdPartyExecutionAttempted": False,
        "claimBoundary": (
            "A role-separated public annotation workspace, not completed cases, private ES-P1 "
            "qualification, or production probability."
        ),
    }
    manifest = {**body, "workspaceDigest": sha256_json(body)}
    (root / "workspace.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def inspect_public_author_kit(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    manifest = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("apiVersion") != AUTHOR_KIT_SCHEMA:
        raise ValueError("public author kit Schema is invalid")
    body = {key: value for key, value in manifest.items() if key != "workspaceDigest"}
    if manifest.get("workspaceDigest") != sha256_json(body):
        raise ValueError("public author kit workspace digest drift")
    if any((
        manifest.get("executionPolicy") != EXECUTION_POLICY,
        manifest.get("officialEsP1QualificationEligible") is not False,
        manifest.get("privateHoldout") is not False,
        manifest.get("containsRuntimeOrEvaluator") is not False,
        manifest.get("containsGeneratedGold") is not False,
        manifest.get("thirdPartyExecutionAttempted") is not False,
    )):
        raise ValueError("public author kit authority boundary is invalid")
    expected = manifest.get("sealedFiles")
    if not isinstance(expected, dict):
        raise ValueError("public author kit sealed file map is invalid")
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path == root / "workspace.json":
            continue
        if path.is_symlink():
            raise ValueError("public author kit cannot contain symlinks")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = _file_digest(path)
    if actual != expected:
        raise ValueError("public author kit sealed file set or digest drift")
    assignments = [
        json.loads(line) for line in (root / "assignments.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        len(assignments) != manifest["selectedPackageCount"]
        or sum(len(item["taskSlots"]) for item in assignments) != manifest["taskSlotCount"]
    ):
        raise ValueError("public author kit assignment coverage drift")
    for assignment in assignments:
        package = root / "packages" / assignment["packageId"]
        report = inspect_skill_package(package)
        if report["gate"] != "passed" or assignment["packageDigest"] != manifest["selectedPackageDigests"][assignment["packageId"]]:
            raise ValueError("public author kit contains an unqualified or unbound package")
    return {
        "status": "valid", "selectedPackageCount": len(assignments),
        "taskSlotCount": manifest["taskSlotCount"], "executionPolicy": EXECUTION_POLICY,
        "thirdPartyExecutionAttempted": False, "containsGeneratedGold": False,
        "officialEsP1QualificationEligible": False,
        "workspaceDigest": manifest["workspaceDigest"],
        "claimBoundary": manifest["claimBoundary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover")
    discover.add_argument("--query", action="append", required=True)
    discover.add_argument("--limit", type=int, default=80)
    discover.add_argument("--per-query", type=int, default=20)
    discover.add_argument("--max-per-repo", type=int, default=5)
    discover.add_argument("--language")
    discover.add_argument("--sort-by", choices=("recent", "stars"), default="recent")
    discover.add_argument("--output", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("discovery")
    snapshot.add_argument("--output-root", required=True)
    snapshot.add_argument("--limit", type=int, default=20)
    snapshot.add_argument("--script-policy", choices=("exclude", "metadata-only"), default="exclude")
    snapshot.add_argument("--license-policy", choices=("known", "record-only"), default="known")
    snapshot.add_argument("--source-backend", choices=("api", "git"), default="api")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("root")
    report = commands.add_parser("report")
    report.add_argument("root")
    report.add_argument("--discovery")
    report.add_argument("--output-root", required=True)
    author_kit = commands.add_parser("author-kit")
    author_kit.add_argument("snapshot_root")
    author_kit.add_argument("--output-root", required=True)
    author_kit.add_argument("--tasks-per-skill", type=int, default=3)
    inspect_author_kit = commands.add_parser("author-kit-inspect")
    inspect_author_kit.add_argument("root")
    draft_author = commands.add_parser("draft-author")
    draft_author.add_argument("author_kit_root")
    draft_author.add_argument("--output-root", required=True)
    draft_author.add_argument("--model", default="qwen3.5:9b")
    draft_author.add_argument("--no-resume", action="store_true")
    inspect_drafts = commands.add_parser("draft-inspect")
    inspect_drafts.add_argument("draft_root")
    inspect_drafts.add_argument("author_kit_root")
    library = commands.add_parser("library")
    library.add_argument("author_kit_root")
    library.add_argument("--output-root", required=True)
    library.add_argument("--draft-root")
    library.add_argument("--snapshot-root")
    inspect_library = commands.add_parser("library-inspect")
    inspect_library.add_argument("root")
    library_summary = commands.add_parser("library-summary")
    library_summary.add_argument("root")
    library_summary.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "discover":
        result = write_discovery(
            args.output, queries=args.query, limit=args.limit, per_query=args.per_query,
            max_per_repo=args.max_per_repo, language=args.language, sort_by=args.sort_by,
        )
    elif args.command == "snapshot":
        result = snapshot_public_skills(
            args.discovery, args.output_root, limit=args.limit, script_policy=args.script_policy,
            license_policy=args.license_policy, source_backend=args.source_backend,
        )
    elif args.command == "inspect":
        result = inspect_public_snapshot(args.root)
    elif args.command == "report":
        result = build_public_pilot_report(
            args.root, args.output_root, discovery_path=args.discovery,
        )
    elif args.command == "author-kit":
        result = export_public_author_kit(
            args.snapshot_root, args.output_root, tasks_per_skill=args.tasks_per_skill,
        )
    elif args.command == "author-kit-inspect":
        result = inspect_public_author_kit(args.root)
    elif args.command == "draft-author":
        from evaluation.public_skill_draft_author import run_public_market_drafts
        result = run_public_market_drafts(
            args.author_kit_root, args.output_root, model=args.model,
            resume=not args.no_resume,
        )
    elif args.command == "draft-inspect":
        from evaluation.public_skill_draft_author import inspect_public_market_drafts
        result = inspect_public_market_drafts(args.draft_root, args.author_kit_root)
    elif args.command == "library":
        from evaluation.public_skill_library import build_public_skill_library
        result = build_public_skill_library(
            args.author_kit_root, args.output_root, draft_root=args.draft_root,
            snapshot_root=args.snapshot_root,
        )
    elif args.command == "library-inspect":
        from evaluation.public_skill_library import inspect_public_skill_library
        result = inspect_public_skill_library(args.root)
    else:
        from evaluation.public_skill_library import export_public_skill_library_summary
        result = export_public_skill_library_summary(args.root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUTHOR_KIT_SCHEMA", "DISCOVERY_SCHEMA", "EVIDENCE_CLASS", "EXECUTION_POLICY", "SNAPSHOT_SCHEMA",
    "discover_public_skills", "inspect_public_snapshot", "load_discovery",
    "snapshot_public_skills", "write_discovery", "build_public_pilot_report",
    "export_public_author_kit", "inspect_public_author_kit",
]


if __name__ == "__main__":
    raise SystemExit(main())
