"""Shared one-attempt evidence plumbing; no cases, oracles or semantic policy.

New fingerprints reject pre-consolidation runs. Replay historical evidence
with its recorded Git snapshot, never a hash bypass.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Callable

from evaluation.flow_model_transport import send
from evaluation.flow_translation import _write
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, verify_receipt
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter

# Shared source/compiler/transport dependencies, not an inheritance chain of
# experimental pilots. Runtime files are conservatively pinned as a whole.
CORE_FILES = (
    "flow_checkpoint.py", "flow_model_transport.py", "flow_behavior.py",
    "flow_contract_authoring.py", "flow_guard_binding.py",
    "flow_guard_counterfactual.py", "flow_guard_necessity.py",
    "flow_grounded_translation.py", "flow_source_selection.py",
    "flow_translation.py", "flow_tree.py", "flow_tree_authoring.py",
    "flow_tree_capabilities.py", "read_l05_review.py",
    "translation_case_authoring.py", "translation_source_alignment.py",
)


def environment() -> dict:
    return {name: version(name) for name in ("pydantic", "jsonschema", "httpx")}


def implementation(*extra_paths: str) -> dict:
    paths = {"evaluation/" + name for name in CORE_FILES} | set(extra_paths)
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "network_runtime").rglob("*.py"))
    return {path: digest_file(ROOT / path) for path in sorted(paths)}


Derive = Callable[[dict], tuple[dict, dict]]


def replay(folder: Path, request: dict, derive: Derive, *, label: str) -> tuple[dict, dict]:
    """Check recorded bytes and exact re-derivation without contacting a model."""
    verify_receipt(folder)
    if json.loads((folder / "request.json").read_text()) != request:
        raise ValueError(f"{label} input/implementation drift")
    files, result = derive(json.loads((folder / "response.json").read_text()))
    if (set(receipt(folder)) != {"request.json", "response.json", "result.json", *files}
            or json.loads((folder / "result.json").read_text()) != result
            or any(json.loads((folder / name).read_text()) != data for name, data in files.items())):
        raise ValueError(f"{label} checkpoint derivation drift")
    return files, result


def author_once(root: Path, inputs: dict, derive: Derive, *, max_new_calls: int, label: str) -> dict:
    """One explicit call, or offline replay; partial/failed runs never retry."""
    if root.exists():
        verify_receipt(root)
        stored = json.loads((root / "request.json").read_text())
        files, result = replay(root, {**inputs, "model": stored.get("model")}, derive, label=label)
        return dict(result=result, **files)
    if type(max_new_calls) is not int or max_new_calls < 1:
        raise ValueError("explicit one-call budget required")
    model = OllamaAnchoredAuthorAdapter().preflight()
    root.mkdir(parents=True)
    _write(root / "request.json", {**inputs, "model": model})
    envelope = send("ollama", inputs["wireRequest"])
    _write(root / "response.json", envelope)
    files, result = derive(envelope)
    for name, data in files.items():
        _write(root / name, data)
    _write(root / "result.json", result)
    _write(root / "receipt.json", receipt(root))
    return dict(result=result, **files)
