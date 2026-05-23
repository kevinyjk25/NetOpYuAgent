#!/usr/bin/env python3
"""
scripts/audit_wiring.py
-----------------------
Static audit: every key written to `services[...]` MUST have at least one
reader outside its writer's file. Catches the "ghost service" anti-pattern:
a cross-module component is constructed, registered in `services`, and
never invoked — so the config flag turns on a feature that does nothing.

Real cases this would have caught before they shipped:
  - FactConflictDetector registered, never invoked (until we wired it
    into MemoryAdapter.add_fact)
  - MemoryConsolidator instantiated, no caller fired consolidate_session

This is complementary to audit_imports.py (import-resolution) and
audit_module_independence.py (cross-module imports). This script
checks RUNTIME usage of the services dict.

Run:
    python scripts/audit_wiring.py

Exits 0 if every registered service has at least one external reader,
1 otherwise.

How it works:
  WRITES — collect all string-literal keys K where the source has
           `services[K] = ...` or assignment via subscript.
  READS  — collect all string-literal keys K read via
           `services.get(K[, ...])` or `services[K]` on the RHS,
           outside the writer's file.

Limitations:
  - Dynamic keys (f-strings, variables) are skipped — they're rare and
    a runtime audit would catch them.
  - A service that's only read inside the SAME file as it's written is
    flagged as a ghost; in practice cross-module services should always
    have external callers, so this is the intended signal.
  - The script does NOT check that the value type matches the consumer's
    expectations (that's what unit tests are for).
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────

# Variable names that we track as "the services dict". `services` covers
# main.py + webui/backend.py + any module that takes a services arg.
SERVICES_VARS = {"services", "_services"}

IGNORE_PATH_FRAGMENTS = ["scripts/audit_wiring.py"]
# Note: .venv / site-packages / __pycache__ / tests / examples are
# already filtered by _audit_common.iter_repo_python_files — see that
# module for the canonical exclusion list.

# Keys deliberately registered for introspection/observability and used
# via paths the audit can't see — singletons resolved by getter functions
# (`get_X_registry()`), background tasks owned by lifecycle handlers,
# loaders that consumer modules instantiate directly. Whitelisting tells
# the audit "yes, this is a real registration with no `services.get` —
# but I've checked and it's intentional".
#
# Each whitelist entry must be JUSTIFIED in the comment to its right —
# otherwise it'll silently grow to mask real bugs.
KEY_WHITELIST: set[str] = {
    # Introspection-only: wired via MemoryAdapter.set_conflict_detector,
    # entry kept so /system/wiring can report status.
    "fact_conflict_detector",
    # Same pattern: wired via consumer's own wiring; entry exposes status.
    "journal_to_facts_adapter",
    # Module-level singletons accessed via get_X_registry() / global state.
    "meta_tool_registry",
    "schema_registry",
    # Background lifecycle tasks — started/stopped by app lifespan hooks,
    # not consumed by reading services[...].
    "skill_journal_consumer",
    # Loaders re-instantiated where needed (webui/backend.py imports
    # SkillLoader directly and builds its own). Registration is for
    # /system/wiring visibility.
    "skill_loader",
    # Tool retriever is hot-swapped into LLMEngine via patch_retriever;
    # the services entry is for diagnostics.
    "tool_retriever",
}


# ── Setters that must be called somewhere (Phase 2B, 2026-05) ─────────────
#
# Some wiring isn't a `services[K] = X` registration but a method call on an
# object — typically `engine.attach_foo(...)`. If the setter is defined but
# never called, the engine has the slot but no data flows in. Symptom is
# always "the feature appears wired in /system/wiring but does nothing at
# runtime".
#
# The previous example was `llm_engine.attach_peer_registry` — defined and
# `_build_peers_section` consumed it inside the prompt, but main.py forgot
# to call attach. Result: LLM prompt described the [DELEGATE:] syntax but
# never listed any actual peer, so the LLM didn't delegate (2026-05).
#
# This list pairs each method with a one-line reason. Adding here means:
# "if this method has zero call sites in the repo, fail the audit."
REQUIRED_METHOD_CALLS: list[tuple[str, str]] = [
    (
        "attach_peer_registry",
        "wires AgentRegistry into LLMEngine so the system prompt lists "
        "available peers + capabilities for [DELEGATE:] decisions",
    ),
    (
        "set_task_store",
        "wires task_system.store into HitlExecutor so inbound A2A "
        "delegations (peer→this agent) get recorded — without this, "
        "the Delegations tab shows nothing when this agent is processing "
        "a [DELEGATE:] from a peer (Phase 2B+, 2026-05)",
    ),
]


def _find_method_call_sites(method_name: str) -> list[Path]:
    """Return all repo files containing `.method_name(`.

    Simple substring match — good enough; method names are distinctive.
    We don't need AST here because both definition and call use the
    same name token. The audit's job is just "is there at least one
    call site outside the definition file".
    """
    from _audit_common import iter_repo_python_files
    repo_root = Path(__file__).resolve().parent.parent
    hits: list[Path] = []
    needle = f".{method_name}("
    for f in iter_repo_python_files(repo_root):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if needle in text:
            hits.append(f)
    return hits


def _find_method_def_files(method_name: str) -> list[Path]:
    """Return files where `def method_name(...)` is defined (excludes calls)."""
    from _audit_common import iter_repo_python_files
    repo_root = Path(__file__).resolve().parent.parent
    hits: list[Path] = []
    needle_def = f"def {method_name}("
    for f in iter_repo_python_files(repo_root):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if needle_def in text:
            hits.append(f)
    return hits


def _str_const(node: ast.AST) -> str | None:
    """Return string literal if node is `Constant(str)` or `Str` (Py3.7+)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_services_subscript(node: ast.Subscript) -> str | None:
    """If `node` is `services[<str>]` (or _services[...]), return the key."""
    val = node.value
    if isinstance(val, ast.Name) and val.id in SERVICES_VARS:
        # ast.Index wrapper went away in 3.9; handle both
        slc = node.slice.value if isinstance(getattr(node, "slice", None), ast.Index) else node.slice
        return _str_const(slc)
    return None


def _collect_writes(tree: ast.AST) -> set[str]:
    """services[K] = ... and services.update({K: ...})  → {K, ...}"""
    keys: set[str] = set()
    for n in ast.walk(tree):
        # services["foo"] = X
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Subscript):
                    k = _is_services_subscript(tgt)
                    if k is not None:
                        keys.add(k)
        # services["foo"] += X (rare but valid)
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript):
            k = _is_services_subscript(n.target)
            if k is not None:
                keys.add(k)
        # services.update({"foo": ..., "bar": ...})
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "update"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in SERVICES_VARS):
            for arg in n.args:
                if isinstance(arg, ast.Dict):
                    for k_node in arg.keys:
                        s = _str_const(k_node) if k_node is not None else None
                        if s is not None:
                            keys.add(s)
    return keys


def _collect_reads(tree: ast.AST) -> set[str]:
    """services.get("K") / services["K"] on RHS → {K, ...}"""
    keys: set[str] = set()
    for n in ast.walk(tree):
        # services.get("foo") / services.get("foo", default)
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in SERVICES_VARS
                and n.args):
            s = _str_const(n.args[0])
            if s is not None:
                keys.add(s)
        # services["foo"] on RHS — any subscript whose context is Load
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load):
            k = _is_services_subscript(n)
            if k is not None:
                keys.add(k)
        # K in services / K not in services — also a "read" intent
        if isinstance(n, ast.Compare):
            for cmp_node in n.comparators:
                if isinstance(cmp_node, ast.Name) and cmp_node.id in SERVICES_VARS:
                    s = _str_const(n.left)
                    if s is not None:
                        keys.add(s)
    return keys


def _scan(repo: Path) -> tuple[
    dict[str, list[Path]],   # writes: key → files that write it
    dict[str, list[Path]],   # reads:  key → files that read it
]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _audit_common import iter_repo_python_files

    writes: dict[str, list[Path]] = defaultdict(list)
    reads:  dict[str, list[Path]] = defaultdict(list)
    for f in iter_repo_python_files(repo):
        if any(frag in str(f) for frag in IGNORE_PATH_FRAGMENTS):
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for k in _collect_writes(tree):
            writes[k].append(f.relative_to(repo))
        for k in _collect_reads(tree):
            reads[k].append(f.relative_to(repo))
    return writes, reads


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    writes, reads = _scan(repo)

    if not writes:
        print("audit_wiring: no services[...] writes found — nothing to audit.")
        return 0

    ghosts: list[tuple[str, list[Path]]] = []
    internal_only: list[tuple[str, list[Path], list[Path]]] = []
    for key, write_files in sorted(writes.items()):
        if key in KEY_WHITELIST:
            continue
        read_files = reads.get(key, [])
        # External readers = readers in files OTHER than the writer(s)
        write_set = set(write_files)
        external_readers = [f for f in read_files if f not in write_set]
        if not read_files:
            ghosts.append((key, write_files))
        elif not external_readers:
            internal_only.append((key, write_files, read_files))

    failed = False

    if ghosts:
        failed = True
        print("=" * 70)
        print("FAIL — services keys with ZERO readers (ghost services):")
        print("=" * 70)
        for key, write_files in ghosts:
            files_str = ", ".join(str(f) for f in write_files[:3])
            if len(write_files) > 3:
                files_str += f" (+{len(write_files) - 3} more)"
            print(f"  services[{key!r}]")
            print(f"    written in: {files_str}")
            print(f"    read in:    NONE — feature is registered but unused")
            print()

    if internal_only:
        # Soft warning, not fail — same-file read/write may be a legit
        # initialisation pattern (e.g. services["x"] = ...; services["x"].init()
        # within build_services). Surface for review but don't break CI.
        print("=" * 70)
        print("WARN — services keys read ONLY in their writer file:")
        print("(may be intentional self-init; review and whitelist if so)")
        print("=" * 70)
        for key, w_files, r_files in internal_only:
            print(f"  services[{key!r}]")
            print(f"    written in: {', '.join(str(f) for f in w_files[:2])}")
            print(f"    read only in writer file(s)")
        print()

    if failed:
        print(f"audit_wiring: FAIL — {len(ghosts)} ghost service(s).")
        return 1

    # ── Second pass: required-method-call audit ───────────────────────────
    # Some wiring isn't service-dict registration but `engine.attach_X(...)`.
    # The earlier ghost-services check can't see these — a setter can be
    # defined and a getter consumed downstream while nobody actually calls
    # attach. List which setters MUST be called in REQUIRED_METHOD_CALLS;
    # if any has zero call sites outside its definition file, we fail.
    missing_calls: list[tuple[str, str, list[Path]]] = []
    for method_name, reason in REQUIRED_METHOD_CALLS:
        all_hits      = _find_method_call_sites(method_name)
        def_files     = _find_method_def_files(method_name)
        # External callers = files containing `.method_name(` that are NOT
        # the definition site. (A def file may also self-test internally,
        # but we want at least one external caller wiring the slot.)
        def_set       = set(def_files)
        external_call = [f for f in all_hits if f not in def_set]
        if not external_call:
            missing_calls.append((method_name, reason, def_files))

    if missing_calls:
        print("=" * 70)
        print("FAIL — required setters with NO external call site:")
        print("=" * 70)
        for method_name, reason, def_files in missing_calls:
            print(f"  {method_name}(...)")
            print(f"    defined in: {', '.join(str(f) for f in def_files) or 'NONE'}")
            print(f"    called in:  NONE — wiring forgotten")
            print(f"    why required: {reason}")
            print()
        print(f"audit_wiring: FAIL — {len(missing_calls)} unwired setter(s).")
        return 1

    total_keys = len(writes)
    print(
        f"audit_wiring: ✓ PASS — {total_keys} service keys, "
        f"all have external readers; "
        f"{len(REQUIRED_METHOD_CALLS)} required setter(s) all called."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
