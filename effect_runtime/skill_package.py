"""Trust-boundary inspection for Anthropic-compatible Skill packages.

The inspector is deliberately read-only: bundled scripts are hashed and
classified, never imported or executed.  Its report can therefore be used as
one of the deterministic gates before an L1 Skill is translated into an L0
Effect Contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

from skills.skill_format import SkillFormatError, parse_skill_md


_MANAGED_ROOTS = frozenset({"scripts", "references", "assets"})
_SCRIPT_SUFFIXES = frozenset({".py", ".sh", ".bash", ".js", ".ts", ".ps1"})
_SCRIPT_ROLES = frozenset(
    {"transformer", "preflight", "verifier", "compensator", "provider_adapter"}
)
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_INLINE_RESOURCE_RE = re.compile(
    r"(?<![A-Za-z0-9_])((?:scripts|references|assets)/[^\s`'\"<>()\[\]]+)"
)
_SIDE_EFFECT_RE = re.compile(
    r"(?:\bos\.system\b|\bsubprocess\b|\bparamiko\b|\bnetmiko\b|\bncclient\b|"
    r"\brequests\.(?:post|put|patch|delete)\b|\b(?:unlink|rmtree|remove)\s*\(|"
    r"\bopen\s*\([^\n]{0,120}['\"](?:w|a|x|\+)['\"]|"
    r"\bcurl\b[^\n]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--request\s*)|"
    r"\brm\s+-|\bssh\b|\bsocket\.)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PackageFinding:
    severity: str
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class PackageResource:
    path: str
    kind: str
    size: int
    sha256: str
    referenced_by: tuple[str, ...]
    script_role: str | None = None
    capability_bound: bool | None = None
    capability_binding: str | None = None
    side_effect_signals: bool | None = None


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_digest(resources: Iterable[PackageResource]) -> str:
    payload = [
        {"path": item.path, "sha256": item.sha256, "size": item.size}
        for item in sorted(resources, key=lambda value: value.path)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _kind(path: PurePosixPath) -> str:
    if path.as_posix() == "SKILL.md":
        return "skill"
    return path.parts[0] if path.parts and path.parts[0] in _MANAGED_ROOTS else "other"


def _script_role(path: str, explicit: dict[str, str]) -> str:
    if path in explicit:
        return explicit[path]
    name = PurePosixPath(path).stem.lower()
    if any(word in name for word in ("verify", "assert", "check_result", "readback")):
        return "verifier"
    if any(word in name for word in ("preflight", "inspect", "snapshot", "check_input")):
        return "preflight"
    if any(word in name for word in ("rollback", "restore", "compensat", "undo")):
        return "compensator"
    if any(word in name for word in ("transform", "normalize", "render", "parse", "convert")):
        return "transformer"
    if any(word in name for word in ("apply", "write", "update", "delete", "deploy", "adapter")):
        return "provider_adapter"
    return "unknown"


def _explicit_script_roles(metadata: dict[str, Any]) -> tuple[dict[str, str], list[PackageFinding]]:
    """Parse the optional standard-metadata extension.

    Example::

        metadata:
          effect-runtime-script-roles: scripts/check.py=verifier,scripts/apply.py=provider_adapter
    """
    findings: list[PackageFinding] = []
    raw = str(metadata.get("effect-runtime-script-roles", "")).strip()
    result: dict[str, str] = {}
    if not raw:
        return result, findings
    for item in raw.split(","):
        if "=" not in item:
            findings.append(PackageFinding(
                "error", "SCRIPT_ROLE_DECLARATION_INVALID", "SKILL.md",
                f"Invalid script-role declaration {item.strip()!r}; expected path=role.",
            ))
            continue
        path, role = (part.strip() for part in item.split("=", 1))
        if role not in _SCRIPT_ROLES or not path.startswith("scripts/"):
            findings.append(PackageFinding(
                "error", "SCRIPT_ROLE_DECLARATION_INVALID", "SKILL.md",
                f"Unsupported script binding {path!r}={role!r}.",
            ))
            continue
        result[path] = role
    return result, findings


def _script_bindings(values: Iterable[str]) -> tuple[dict[str, str], list[PackageFinding]]:
    result: dict[str, str] = {}
    findings: list[PackageFinding] = []
    for raw in values:
        if "=" not in raw:
            findings.append(PackageFinding(
                "error", "SCRIPT_BINDING_INVALID", str(raw),
                "Script binding must use path=capability-id; a path alone is not authority.",
            ))
            continue
        path, capability = (item.strip() for item in raw.split("=", 1))
        normalized = PurePosixPath(path).as_posix()
        if (
            not normalized.startswith("scripts/")
            or not capability
            or any(character.isspace() for character in capability)
        ):
            findings.append(PackageFinding(
                "error", "SCRIPT_BINDING_INVALID", path,
                "Script binding requires a scripts/ path and a non-empty Capability id.",
            ))
            continue
        if normalized in result and result[normalized] != capability:
            findings.append(PackageFinding(
                "error", "SCRIPT_BINDING_CONFLICT", normalized,
                "One script cannot bind to multiple Capability ids in one package decision.",
            ))
            continue
        result[normalized] = capability
    return result, findings


def _extract_refs(text: str) -> set[str]:
    candidates = set(_LINK_RE.findall(text)) | set(_INLINE_RESOURCE_RE.findall(text))
    result: set[str] = set()
    for candidate in candidates:
        raw = unquote(candidate.strip().strip("`<>\"'"))
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or raw.startswith("#"):
            continue
        clean = parsed.path.rstrip(".,;:")
        if clean:
            result.add(clean)
    return result


def _safe_reference(source: str, reference: str) -> tuple[str | None, str | None]:
    raw = PurePosixPath(reference)
    if raw.is_absolute():
        return None, "absolute path"
    base = PurePosixPath(source).parent
    combined = base / raw
    parts: list[str] = []
    for part in combined.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None, "path traversal"
            parts.pop()
        else:
            parts.append(part)
    normalized = PurePosixPath(*parts).as_posix()
    if not normalized or normalized.startswith("../"):
        return None, "path traversal"
    return normalized, None


def inspect_skill_package(
    skill: str | Path,
    *,
    bound_scripts: Iterable[str] = (),
    max_files: int = 128,
    max_file_bytes: int = 512 * 1024,
    max_total_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    """Inspect a Skill folder and return a deterministic JSON-ready report."""
    input_path = Path(skill).expanduser()
    skill_path = input_path / "SKILL.md" if input_path.is_dir() else input_path
    root = skill_path.parent
    findings: list[PackageFinding] = []
    if input_path.is_symlink() or skill_path.is_symlink():
        findings.append(PackageFinding(
            "error", "SYMLINK_FORBIDDEN", "SKILL.md",
            "The Skill entry point must not be a symbolic link.",
        ))
    if not skill_path.is_file():
        return {
            "schema": "effect-runtime.io/skill-package-report/v1",
            "gate": "blocked",
            "executionEligible": False,
            "packageDigest": None,
            "skill": None,
            "summary": {"files": 0, "bytes": 0, "referenceCoveragePercent": 0.0},
            "resources": [],
            "referenceGraph": [],
            "findings": [asdict(PackageFinding(
                "error", "SKILL_MD_MISSING", "SKILL.md", "SKILL.md does not exist.",
            ))],
            "claimBoundary": "Package integrity is not proof of semantic correctness or production success.",
        }

    files: dict[str, tuple[Path, bytes]] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if _kind(PurePosixPath(relative)) == "other":
            # The supported Anthropic package surface is deliberately narrow.
            # Conversion outputs and local notes beside the package are not
            # source inputs; a reference to them is reported as missing.
            continue
        if path.is_symlink():
            findings.append(PackageFinding(
                "error", "SYMLINK_FORBIDDEN", relative,
                "Bundled symbolic links are outside the immutable package boundary.",
            ))
            continue
        if not path.is_file():
            continue
        if len(files) >= max_files:
            findings.append(PackageFinding(
                "error", "PACKAGE_FILE_LIMIT", relative,
                f"Package exceeds the {max_files}-file inspection limit.",
            ))
            break
        size = path.stat().st_size
        if size > max_file_bytes:
            findings.append(PackageFinding(
                "error", "PACKAGE_FILE_TOO_LARGE", relative,
                f"File exceeds the {max_file_bytes}-byte limit.",
            ))
            continue
        data = path.read_bytes()
        total_bytes += len(data)
        files[relative] = (path, data)
    if total_bytes > max_total_bytes:
        findings.append(PackageFinding(
            "error", "PACKAGE_TOTAL_SIZE_LIMIT", ".",
            f"Package exceeds the {max_total_bytes}-byte inspection limit.",
        ))

    parsed = None
    try:
        parsed = parse_skill_md(files.get("SKILL.md", (skill_path, b""))[1].decode("utf-8"))
    except (UnicodeDecodeError, SkillFormatError) as exc:
        findings.append(PackageFinding(
            "error", "SKILL_FORMAT_INVALID", "SKILL.md", str(exc),
        ))
    explicit_roles: dict[str, str] = {}
    if parsed is not None:
        explicit_roles, role_findings = _explicit_script_roles(parsed.metadata)
        findings.extend(role_findings)

    graph: list[dict[str, str]] = []
    referenced_by: dict[str, set[str]] = {path: set() for path in files}
    readable_suffixes = {".md", ".txt", ".yaml", ".yml", ".json"}
    for source, (_, data) in files.items():
        if source != "SKILL.md" and PurePosixPath(source).suffix.lower() not in readable_suffixes:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if source == "SKILL.md" and parsed is not None:
            # Frontmatter metadata may legitimately contain path=role
            # declarations; only the Markdown body defines disclosure edges.
            text = parsed.body
        for raw_reference in sorted(_extract_refs(text)):
            target, error = _safe_reference(source, raw_reference)
            if error:
                findings.append(PackageFinding(
                    "error", "RESOURCE_REFERENCE_UNSAFE", source,
                    f"Reference {raw_reference!r} uses {error}.",
                ))
                continue
            assert target is not None
            if target not in files:
                findings.append(PackageFinding(
                    "error", "RESOURCE_REFERENCE_MISSING", source,
                    f"Referenced package resource {target!r} does not exist.",
                ))
                continue
            referenced_by[target].add(source)
            graph.append({"from": source, "to": target})

    bindings, binding_findings = _script_bindings(bound_scripts)
    findings.extend(binding_findings)
    resources: list[PackageResource] = []
    for relative, (_, data) in sorted(files.items()):
        path = PurePosixPath(relative)
        kind = _kind(path)
        role: str | None = None
        capability_bound: bool | None = None
        side_effect_signals: bool | None = None
        is_script = kind == "scripts" or path.suffix.lower() in _SCRIPT_SUFFIXES
        if is_script:
            if kind != "scripts":
                findings.append(PackageFinding(
                    "error", "SCRIPT_OUTSIDE_SCRIPT_DIRECTORY", relative,
                    "Executable-looking resources must be placed under scripts/.",
                ))
            role = _script_role(relative, explicit_roles)
            capability_bound = relative in bindings
            text = data.decode("utf-8", errors="ignore")
            side_effect_signals = bool(_SIDE_EFFECT_RE.search(text))
            if role in {"provider_adapter", "compensator"} or side_effect_signals:
                if not capability_bound:
                    findings.append(PackageFinding(
                        "error", "SCRIPT_EFFECT_UNBOUND", relative,
                        "Potential side effect is not bound to a reviewed Capability; Runtime write access is denied.",
                    ))
            elif role == "unknown":
                findings.append(PackageFinding(
                    "warning", "SCRIPT_ROLE_UNKNOWN", relative,
                    "Script role is unknown; declare it in metadata before promotion.",
                ))
        resources.append(PackageResource(
            path=relative,
            kind=kind,
            size=len(data),
            sha256=_sha256_bytes(data),
            referenced_by=tuple(sorted(referenced_by.get(relative, set()))),
            script_role=role,
            capability_bound=capability_bound,
            capability_binding=bindings.get(relative),
            side_effect_signals=side_effect_signals,
        ))

    managed = [item for item in resources if item.kind in _MANAGED_ROOTS]
    referenced = [item for item in managed if item.referenced_by]
    for item in managed:
        if not item.referenced_by:
            findings.append(PackageFinding(
                "warning", "RESOURCE_UNREFERENCED", item.path,
                "Bundled resource is not reachable from SKILL.md or another referenced document.",
            ))
    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    coverage = round(100.0 * len(referenced) / len(managed), 2) if managed else 100.0
    gate = "passed" if errors == 0 else "blocked"
    return {
        "schema": "effect-runtime.io/skill-package-report/v1",
        "gate": gate,
        "executionEligible": gate == "passed",
        "packageDigest": _canonical_digest(resources),
        "skill": None if parsed is None else {
            "name": parsed.name,
            "description": parsed.frontmatter["description"],
        },
        "summary": {
            "files": len(resources),
            "packageFiles": len(resources),
            "bytes": total_bytes,
            "managedResources": len(managed),
            "referencedManagedResources": len(referenced),
            "referenceCoveragePercent": coverage,
            "scripts": sum(item.kind == "scripts" for item in resources),
            "boundScripts": sum(item.capability_bound is True for item in resources),
            "errors": errors,
            "warnings": warnings,
        },
        "resources": [asdict(item) for item in resources],
        "referenceGraph": graph,
        "findings": [asdict(item) for item in findings],
        "claimBoundary": "Package integrity and traceability are deterministic gates, not proof of semantic correctness or production success.",
    }


def build_skill_disclosure_packet(
    skill: str | Path,
    *,
    bound_scripts: Iterable[str] = (),
    max_disclosed_bytes: int = 512 * 1024,
) -> dict[str, Any]:
    """Return only resources reachable through progressive-disclosure edges.

    Text is transported as untrusted authoring evidence. Binary assets remain
    digest-only, and no script is imported or executed.
    """
    report = inspect_skill_package(skill, bound_scripts=bound_scripts)
    input_path = Path(skill).expanduser()
    skill_path = input_path / "SKILL.md" if input_path.is_dir() else input_path
    root = skill_path.parent
    adjacency: dict[str, set[str]] = {}
    for edge in report.get("referenceGraph") or []:
        adjacency.setdefault(str(edge["from"]), set()).add(str(edge["to"]))
    reachable: set[str] = set()
    pending = list(adjacency.get("SKILL.md", set()))
    while pending:
        item = pending.pop()
        if item in reachable:
            continue
        reachable.add(item)
        pending.extend(adjacency.get(item, set()))

    index = {item["path"]: item for item in report.get("resources") or []}
    disclosed: list[dict[str, Any]] = []
    total = 0
    for relative in sorted(reachable):
        resource = index.get(relative)
        if resource is None or resource.get("kind") not in _MANAGED_ROOTS:
            continue
        path = root / relative
        data = path.read_bytes()
        total += len(data)
        if total > max_disclosed_bytes:
            raise ValueError(
                f"progressive disclosure exceeds {max_disclosed_bytes} bytes"
            )
        try:
            content: str | None = data.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = None
            encoding = "binary-digest-only"
        disclosed.append({
            "path": relative,
            "kind": resource["kind"],
            "sha256": resource["sha256"],
            "script_role": resource.get("script_role"),
            "capability_binding": resource.get("capability_binding"),
            "encoding": encoding,
            "content": content,
            "trust": "untrusted_authoring_evidence",
            "executable": False,
        })
    return {
        "schema": "effect-runtime.io/skill-disclosure-packet/v1",
        "gate": report["gate"],
        "packageDigest": report["packageDigest"],
        "summary": report["summary"],
        "resources": disclosed,
        "findings": report["findings"],
        "executionBoundary": "Bundled scripts are disclosed as text and are never executed during authoring.",
    }


__all__ = [
    "PackageFinding", "PackageResource", "build_skill_disclosure_packet",
    "inspect_skill_package",
]
