"""
skills/skill_format.py
──────────────────────
Single source of truth for the Anthropic-standard SKILL.md format.

Background
----------
Anthropic's open Agent Skills standard (github.com/anthropics/skills) defines
a skill as a *folder* whose entry point is a ``SKILL.md`` file:

    my-skill/
      SKILL.md          # YAML frontmatter + markdown body
      scripts/          # optional
      references/       # optional

The frontmatter allows ONLY these top-level keys (enforced by the upstream
``quick_validate.py``)::

    {name, description, license, allowed-tools, metadata, compatibility}

  - ``name``         kebab-case, ^[a-z0-9]+(-[a-z0-9]+)*$, ≤64 chars (required)
  - ``description``  no angle brackets, ≤1024 chars (required)
  - ``metadata``     str→str map — the ONLY legal extension slot

Our internal catalog/retriever/golden-set machinery, however, is keyed on a
snake_case ``skill_id`` and consumes a *flat dict* with fields like
``purpose``, ``risk_level``, ``requires_hitl``, ``tags``, ``tool_deps`` …

This module bridges the two representations and nothing else imports the YAML
parser directly. It exposes:

    parse_skill_md(text)        -> ParsedSkill          (frontmatter+body→model)
    to_flat_dict(parsed)        -> (skill_id, dict)      (model→internal dict)
    flat_dict_to_skill_md(...)  -> str                   (internal dict→SKILL.md)
    validate_frontmatter(fm)    -> None | raises         (quick_validate parity)
    load_skill_md(text)         -> (skill_id, dict)      (parse+flatten one-shot)

Design rule: the snake_case ``skill_id`` and the kebab-case ``name`` are kept
in lock-step — ``name == skill_id.replace('_','-')`` and the round trip is
loss-free for that mapping. ``metadata.skill_id`` is the authoritative anchor;
when absent we derive it from ``name``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import yaml
except Exception as _exc:  # pragma: no cover - yaml is a hard dependency
    raise ImportError(
        "skills.skill_format requires PyYAML (pip install pyyaml)"
    ) from _exc


# Frontmatter keys allowed by the Anthropic standard. Anything else is a
# validation failure — mirrors quick_validate.py ALLOWED_PROPERTIES.
ALLOWED_FRONTMATTER_KEYS = frozenset(
    {"name", "description", "license", "allowed-tools", "metadata", "compatibility",
     # cross-agent skill fields (2026-06): declare peer deps + offline boundary
     "delegates-to", "degraded-capability"}
)

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class SkillFormatError(ValueError):
    """Raised when a SKILL.md does not conform to the standard."""


@dataclass
class ParsedSkill:
    """In-memory representation of a parsed SKILL.md.

    ``frontmatter`` is the raw (validated) YAML mapping; ``body`` is the
    markdown after the closing ``---``.
    """
    frontmatter: dict[str, Any]
    body:        str

    @property
    def name(self) -> str:
        return self.frontmatter.get("name", "")

    @property
    def metadata(self) -> dict[str, str]:
        return self.frontmatter.get("metadata", {}) or {}


# ──────────────────────────────────────────────────────────────────────────
# Parsing + validation
# ──────────────────────────────────────────────────────────────────────────

def validate_frontmatter(fm: dict[str, Any]) -> None:
    """Enforce the Anthropic frontmatter rules. Raises SkillFormatError.

    Parity with quick_validate.py: allowed-keys, name kebab-case + length,
    description angle-brackets + length, compatibility length.
    """
    if not isinstance(fm, dict):
        raise SkillFormatError("Frontmatter must be a YAML mapping")

    bad = set(fm) - ALLOWED_FRONTMATTER_KEYS
    if bad:
        raise SkillFormatError(
            f"Unexpected key(s) in SKILL.md frontmatter: {sorted(bad)}. "
            f"Allowed: {sorted(ALLOWED_FRONTMATTER_KEYS)}"
        )

    if "name" not in fm:
        raise SkillFormatError("Missing 'name' in frontmatter")
    if "description" not in fm:
        raise SkillFormatError("Missing 'description' in frontmatter")

    name = fm["name"]
    if not isinstance(name, str):
        raise SkillFormatError(f"Name must be a string, got {type(name).__name__}")
    if not _NAME_RE.match(name):
        raise SkillFormatError(
            f"Name {name!r} should be kebab-case (lowercase letters, digits, "
            f"single hyphens; no leading/trailing/consecutive hyphens)"
        )
    if len(name) > 64:
        raise SkillFormatError(
            f"Name is too long ({len(name)} characters). Maximum is 64."
        )

    desc = fm["description"]
    if not isinstance(desc, str):
        raise SkillFormatError(
            f"Description must be a string, got {type(desc).__name__}"
        )
    if "<" in desc or ">" in desc:
        raise SkillFormatError("Description cannot contain angle brackets (< or >)")
    if len(desc) > 1024:
        raise SkillFormatError(
            f"Description is too long ({len(desc)} characters). Maximum is 1024."
        )

    compat = fm.get("compatibility")
    if compat is not None:
        if not isinstance(compat, str):
            raise SkillFormatError(
                f"Compatibility must be a string, got {type(compat).__name__}"
            )
        if len(compat) > 500:
            raise SkillFormatError(
                f"Compatibility is too long ({len(compat)} characters). Maximum is 500."
            )

    # metadata, when present, must be a str→str map (the standard allows only
    # string values). We coerce ints/bools at flatten time but reject nested
    # structures here so authors don't silently lose data.
    meta = fm.get("metadata")
    if meta is not None:
        if not isinstance(meta, dict):
            raise SkillFormatError("metadata must be a mapping")
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                raise SkillFormatError(
                    f"metadata[{k!r}] must be a scalar (str/int/bool), not "
                    f"{type(v).__name__} — the standard restricts metadata to "
                    f"string values. Use a bundled file for structured data."
                )


def parse_skill_md(text: str) -> ParsedSkill:
    """Split a SKILL.md string into validated frontmatter + body."""
    m = _FRONTMATTER_RE.match(text.lstrip("\ufeff"))
    if not m:
        raise SkillFormatError("No YAML frontmatter found (file must start with '---')")
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillFormatError(f"Invalid YAML in frontmatter: {exc}") from exc
    validate_frontmatter(fm)
    return ParsedSkill(frontmatter=fm, body=m.group(2).strip())


def strip_frontmatter(text: str) -> str:
    """Return only the markdown body, dropping any leading YAML frontmatter.

    Tolerant: if there is no frontmatter, returns the text unchanged. Used by
    legacy parsers that expect a bare body.
    """
    m = _FRONTMATTER_RE.match(text.lstrip("\ufeff"))
    return m.group(2).strip() if m else text.strip()


def has_frontmatter(text: str) -> bool:
    return bool(_FRONTMATTER_RE.match(text.lstrip("\ufeff")))


# ──────────────────────────────────────────────────────────────────────────
# id ↔ name mapping
# ──────────────────────────────────────────────────────────────────────────

def name_to_skill_id(name: str) -> str:
    """kebab-case name → snake_case skill_id."""
    return name.replace("-", "_")


def skill_id_to_name(skill_id: str) -> str:
    """snake_case skill_id → kebab-case name (standard-compliant)."""
    # Lowercase, replace illegal chars with hyphen, collapse repeats.
    s = re.sub(r"[^a-z0-9]+", "-", skill_id.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "skill"


def _coerce_str(value: Any) -> str:
    """metadata values must be strings on disk; coerce bools/ints sanely."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# ──────────────────────────────────────────────────────────────────────────
# body parsing — markdown sections → structured fields
# ──────────────────────────────────────────────────────────────────────────

def _parse_body_sections(body: str) -> dict[str, Any]:
    """Extract Parameters / Examples / Constraints / Steps from the markdown
    body. Tolerant of the historical "agentskills.io" body style used by the
    evolver (``**Purpose:**``, ``## Steps``…) as well as plain prose.
    """
    out: dict[str, Any] = {
        "parameters":  {},
        "examples":    [],
        "constraints": [],
        "steps":       [],
        "body_purpose": "",
        "body_tags":   [],
        "body_risk":   None,
        "body_hitl":   None,
    }
    current = ""
    for raw in body.splitlines():
        line = raw.strip()

        # Inline metadata lines the evolver historically emitted in the body.
        if "**Purpose:**" in line or line.startswith("Purpose:"):
            out["body_purpose"] = re.sub(r"\*?\*?Purpose:\*?\*?\s*", "", line).strip()
            continue
        if "**Tags:**" in line or line.startswith("Tags:"):
            tags_str = re.sub(r"\*?\*?Tags:\*?\*?\s*", "", line)
            out["body_tags"] = [
                t.strip().strip("[]") for t in tags_str.split(",") if t.strip()
            ]
            continue
        if "**Risk:**" in line or line.startswith("Risk:"):
            risk = re.sub(r"\*?\*?Risk:\*?\*?\s*", "", line).strip().lower()
            if risk in ("low", "medium", "high", "critical"):
                out["body_risk"] = risk
            continue
        if "**HITL:**" in line or line.startswith("HITL:"):
            out["body_hitl"] = "yes" in line.lower() or "true" in line.lower()
            continue

        if line.startswith("## "):
            current = line[3:].strip().lower()
            continue

        if current == "parameters" and line.startswith("-"):
            m = re.match(r"-\s+`?(\w+)`?\s*(?:\(([^)]*)\))?:?\s*(.*)", line)
            if m:
                out["parameters"][m.group(1)] = m.group(3) or m.group(1)
            continue

        if current == "steps" and re.match(r"\d+\.", line):
            out["steps"].append(re.sub(r"^\d+\.\s*", "", line))
            continue

        if current == "constraints" and line.startswith("-"):
            out["constraints"].append(line[1:].strip())
            continue

        if current == "examples" and line.startswith("-"):
            out["examples"].append({"note": line[1:].strip()})
            continue

    return out


# ──────────────────────────────────────────────────────────────────────────
# ParsedSkill → internal flat dict (what register_all / retriever consume)
# ──────────────────────────────────────────────────────────────────────────

def to_flat_dict(
    parsed: ParsedSkill,
    *,
    skill_id_hint: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Convert a ParsedSkill into the internal ``(skill_id, defn)`` shape.

    ``skill_id`` resolution order:
      1. metadata.skill_id (authoritative)
      2. skill_id_hint (e.g. derived from the folder/file name)
      3. name → snake_case
    """
    fm = parsed.frontmatter
    meta = parsed.metadata
    sections = _parse_body_sections(parsed.body)

    skill_id = (
        meta.get("skill_id")
        or skill_id_hint
        or name_to_skill_id(fm["name"])
    )

    def _meta_bool(key: str, default: bool = False) -> bool:
        v = meta.get(key)
        if v is None:
            return default
        return str(v).strip().lower() in ("true", "yes", "1")

    def _meta_list(key: str) -> list[str]:
        v = meta.get(key, "")
        return [x.strip() for x in str(v).split(",") if x.strip()]

    # Display name: prefer an explicit metadata.display_name, else Title-Case
    # the skill_id (keeps parity with the historical flat dicts which used
    # "Syslog Search" rather than the kebab name).
    display_name = meta.get("display_name") or skill_id.replace("_", " ").title()

    purpose = (
        meta.get("purpose")
        or sections["body_purpose"]
        or fm["description"]
    )

    tags = _meta_list("tags") or sections["body_tags"]
    tool_deps = _meta_list("tool_deps")

    risk_level = meta.get("risk_level") or sections["body_risk"] or "low"
    requires_hitl = (
        _meta_bool("requires_hitl")
        if "requires_hitl" in meta
        else bool(sections["body_hitl"])
    )

    # allowed-tools (Anthropic standard frontmatter key): the set of tools this
    # skill is permitted to invoke. Carried through so runtime can enforce a
    # per-skill tool whitelist when the skill is active. Accepts a YAML list or
    # a comma-separated string.
    _at = fm.get("allowed-tools")
    if isinstance(_at, str):
        allowed_tools = [x.strip() for x in _at.split(",") if x.strip()]
    elif isinstance(_at, (list, tuple)):
        allowed_tools = [str(x).strip() for x in _at if str(x).strip()]
    else:
        allowed_tools = []

    # delegates-to (cross-agent skills): the peer agent id(s) or *capability
    # tokens this skill hands subtasks to via [DELEGATE:]. Declared SEPARATELY
    # from allowed-tools so we don't conflate local tool deps with cross-domain
    # delegation (the old allowed_tools ambiguity). Accepts YAML list or CSV.
    _dt = fm.get("delegates-to")
    if isinstance(_dt, str):
        delegates_to = [x.strip() for x in _dt.split(",") if x.strip()]
    elif isinstance(_dt, (list, tuple)):
        delegates_to = [str(x).strip() for x in _dt if str(x).strip()]
    else:
        delegates_to = []

    # degraded-capability (cross-agent skills): free-text boundary contract —
    # what the skill can STILL deliver when a delegated peer is offline. Since
    # peer availability is treated as the NORMAL case (agent environments are
    # uncontrolled), this is injected into the prompt whenever a declared peer
    # is unreachable so the LLM knows its boundary up front instead of
    # improvising after a failed delegation. Free text (may be multiline).
    _dc = fm.get("degraded-capability")
    degraded_capability = (str(_dc).strip() if _dc is not None else "")

    # The description fed to the retriever/prompt is the full standard
    # description plus the body so retrieval keeps the rich SOP text the LAN
    # skills rely on. Steps (if any) are appended for prompt fidelity.
    description = fm["description"]
    if parsed.body:
        description = f"{description}\n\n{parsed.body}"
    elif sections["steps"]:
        description = description + "\n\nSteps:\n" + "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(sections["steps"])
        )

    defn: dict[str, Any] = {
        "name":           display_name,
        "purpose":        purpose,
        "risk_level":     risk_level,
        "requires_hitl":  requires_hitl,
        "tags":           tags,
        "description":    description,
        "parameters":     sections["parameters"],
        "returns":        meta.get("returns", "string"),
        "tool_deps":      tool_deps,
        "examples":       sections["examples"],
        "constraints":    sections["constraints"],
        "estimated_size": meta.get("estimated_size", "small"),
        "returns_large":  _meta_bool("returns_large"),
        "allowed_tools":  allowed_tools,
        "delegates_to":       delegates_to,
        "degraded_capability": degraded_capability,
        # Carry the standard fields through so callers that want the raw
        # standard view (webui, AgentCard) don't have to re-parse.
        "_std_name":        fm["name"],
        "_std_description": fm["description"],
    }
    return skill_id, defn


def load_skill_md(
    text: str, *, skill_id_hint: Optional[str] = None
) -> tuple[str, dict[str, Any]]:
    """One-shot: parse + validate + flatten a SKILL.md string."""
    return to_flat_dict(parse_skill_md(text), skill_id_hint=skill_id_hint)


# ──────────────────────────────────────────────────────────────────────────
# internal flat dict → SKILL.md (serialization for migration + persistence)
# ──────────────────────────────────────────────────────────────────────────

def flat_dict_to_skill_md(skill_id: str, defn: dict[str, Any]) -> str:
    """Serialize an internal flat-dict skill into a standard SKILL.md string.

    Used by (a) the migration script that converts the legacy registries and
    (b) the evolver/upload paths so generated + uploaded skills are written
    in standard form.
    """
    name = skill_id_to_name(skill_id)

    purpose = (defn.get("purpose") or "").strip()
    # description frontmatter field: standard says "what it does + when to use".
    raw_desc = (defn.get("_std_description") or purpose or defn.get("name", name)).strip()
    # Standard forbids angle brackets and caps at 1024.
    description = raw_desc.replace("<", "").replace(">", "")[:1024] or name

    metadata: dict[str, str] = {"skill_id": skill_id}
    display = defn.get("name")
    if display:
        metadata["display_name"] = _coerce_str(display)
    if purpose:
        metadata["purpose"] = _coerce_str(purpose)[:500]
    metadata["risk_level"] = _coerce_str(defn.get("risk_level", "low"))
    metadata["requires_hitl"] = _coerce_str(bool(defn.get("requires_hitl", False)))
    if defn.get("tags"):
        metadata["tags"] = ",".join(_coerce_str(t) for t in defn["tags"])
    if defn.get("tool_deps"):
        metadata["tool_deps"] = ",".join(_coerce_str(t) for t in defn["tool_deps"])
    if defn.get("returns") and defn["returns"] != "string":
        metadata["returns"] = _coerce_str(defn["returns"])
    if defn.get("estimated_size") and defn["estimated_size"] != "small":
        metadata["estimated_size"] = _coerce_str(defn["estimated_size"])
    if defn.get("returns_large"):
        metadata["returns_large"] = "true"

    frontmatter = {
        "name": name,
        "description": description,
        "metadata": metadata,
    }
    fm_yaml = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()

    # Body: prefer the raw markdown body if the defn carries one (e.g. from a
    # SKILL.md round-trip); otherwise synthesise a body from structured fields.
    body = _synthesise_body(skill_id, defn)

    return f"---\n{fm_yaml}\n---\n\n{body}\n"


def _synthesise_body(skill_id: str, defn: dict[str, Any]) -> str:
    """Build a readable markdown body from structured fields.

    If the definition carries a verbatim ``_raw_body`` (e.g. an evolver LLM
    generation), prefer it so the author's prose/notes survive intact — only
    stripping any accidental frontmatter.
    """
    raw = defn.get("_raw_body")
    if raw:
        return strip_frontmatter(raw) if has_frontmatter(raw) else raw.strip()

    title = defn.get("name") or skill_id.replace("_", " ").title()
    lines = [f"# {title}", ""]

    # Use the long description if it differs from the one-line purpose.
    desc = (defn.get("description") or "").strip()
    purpose = (defn.get("purpose") or "").strip()
    # If description already embeds the body (from a round-trip), avoid double
    # frontmatter — strip any leading frontmatter just in case.
    desc = strip_frontmatter(desc) if has_frontmatter(desc) else desc
    if desc and desc != purpose:
        lines += [desc, ""]
    elif purpose:
        lines += [purpose, ""]

    params = defn.get("parameters") or {}
    if params:
        lines.append("## Parameters")
        for pname, pdesc in params.items():
            lines.append(f"- `{pname}`: {pdesc}")
        lines.append("")

    constraints = defn.get("constraints") or []
    if constraints:
        lines.append("## Constraints")
        for c in constraints:
            lines.append(f"- {c}")
        lines.append("")

    examples = defn.get("examples") or []
    if examples:
        lines.append("## Examples")
        for ex in examples:
            if isinstance(ex, dict):
                args = ex.get("args")
                note = ex.get("note", "")
                if args:
                    lines.append(f"- args: {args} — {note}")
                elif note:
                    lines.append(f"- {note}")
            else:
                lines.append(f"- {ex}")
        lines.append("")

    return "\n".join(lines).strip()
