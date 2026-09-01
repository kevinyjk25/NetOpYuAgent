"""Immutable ES-P1 research freeze manifest.

The forward qualification protocol already seals cases, labels and model
observations.  This module closes the other side of the experiment: it binds
the exact Runtime kernel, L0 contracts, evaluator, authoring protocol, harness
boundary, model artifact and Python environment.  A dirty worktree may be
inspected as a preview, but it can never be represented as a completed freeze.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from network_runtime.contracts import sha256_json, utc_now


RESEARCH_FREEZE_SCHEMA = "netopyu.io/ensured-skill-research-freeze/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")

# Only implementation and immutable research inputs are selected.  Private
# prompts, labels, model outputs and generated artifacts are intentionally not
# part of this list.
_SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "runtime_kernel": (
        "effect_runtime/",
        "network_runtime/argument_binding.py",
        "network_runtime/capabilities.py",
        "network_runtime/compensators.py",
        "network_runtime/contracts.py",
        "network_runtime/engine.py",
        "network_runtime/evidence.py",
        "network_runtime/journal.py",
        "network_runtime/policies.py",
        "network_runtime/proposal_binding.py",
        "network_runtime/provider_contracts.py",
        "network_runtime/validation.py",
        "network_runtime/verifiers.py",
        "network_runtime/workflows.py",
        "network_runtime/l0/catalog.py",
        "network_runtime/l0/compiler.py",
        "network_runtime/l0/expressions.py",
        "network_runtime/l0/models.py",
        "network_runtime/l0/production.py",
        "network_runtime/l0/runtime_loader.py",
    ),
    "authoring_and_evaluator": (
        "network_runtime/l0/forward_model_runner.py",
        "network_runtime/l0/forward_qualification.py",
        "network_runtime/l0/forward_study_workspace.py",
        "network_runtime/l0/promotion.py",
        "network_runtime/l0/research_freeze.py",
        "evaluation/ensured_skill_ablation.py",
        "evaluation/dsh_adapter_parity.py",
        "evaluation/ensured_skill_evidence_report.py",
        "evaluation/ensured_skill_protocol.py",
        "evaluation/ensured_skill_runner.py",
    ),
    "harness_boundary": (
        "dsh_adapter/",
        "dsh-plugin-netopyu/",
        "dsh-plugin-effect-harness/",
    ),
    "contract_trajectories": (
        "network_runtime/l0/production_trajectory.py",
        "network_runtime/l0/production_trajectories/",
    ),
    "environment_inputs": (
        "requirements.txt",
        "requirements-core.txt",
        "requirements-dev.txt",
        "requirements-observability.txt",
        "requirements-pragmatic.txt",
    ),
    "es_p0_baseline": (
        "docs/benchmarks/es-p0-evidence-summary.json",
    ),
}


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[:500]
        raise ValueError(f"research freeze requires a Git checkout: {detail}")
    return completed.stdout


def _tracked_files() -> tuple[str, ...]:
    return tuple(
        line for line in _git("ls-files").splitlines()
        if line and (PROJECT_ROOT / line).is_file()
    )


def _selected_files(
    tracked: Iterable[str], selectors: Iterable[str],
) -> tuple[str, ...]:
    values = []
    for relative in tracked:
        if any(
            relative.startswith(selector) if selector.endswith("/")
            else relative == selector
            for selector in selectors
        ):
            values.append(relative)
    return tuple(sorted(values))


def _file_digest(relative: str) -> str:
    return "sha256:" + hashlib.sha256(
        (PROJECT_ROOT / relative).read_bytes()
    ).hexdigest()


def _source_group(tracked: tuple[str, ...], selectors: tuple[str, ...]) -> dict[str, Any]:
    paths = _selected_files(tracked, selectors)
    if not paths:
        raise ValueError(f"research freeze source group is empty: {selectors!r}")
    members = {path: _file_digest(path) for path in paths}
    return {
        "fileCount": len(paths),
        "digest": sha256_json(members),
        "members": members,
    }


def _environment_binding(
    source_groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    installed = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    body = {
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "platformSystem": platform.system(),
        "platformMachine": platform.machine(),
        "platformRelease": platform.release(),
        "dependencyInputsDigest": source_groups["environment_inputs"]["digest"],
        "installedPackageCount": len(installed),
        "installedPackagesDigest": sha256_json(installed),
    }
    return {**body, "digest": sha256_json(body)}


def _es_p0_binding(source_groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = PROJECT_ROOT / "docs/benchmarks/es-p0-evidence-summary.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "sourceDigest": source_groups["es_p0_baseline"]["digest"],
        "aggregateArtifactDigest": raw.get("aggregateArtifactDigest"),
        "claimBoundary": raw.get("claimBoundary"),
    }


def _binding_snapshot(
    *, label: str, model: str, model_artifact_digest: str,
    provider_lab_digest: str,
) -> dict[str, Any]:
    if not _LABEL.fullmatch(label):
        raise ValueError("research freeze label is invalid")
    if not model.strip():
        raise ValueError("research freeze model cannot be empty")
    if not _DIGEST.fullmatch(model_artifact_digest):
        raise ValueError("model artifact digest must be sha256")
    if not _DIGEST.fullmatch(provider_lab_digest):
        raise ValueError("provider/lab fingerprint digest must be sha256")

    # Lazy imports keep ordinary L0 CLI startup small and avoid widening the
    # authority of this read-only research utility.
    from network_runtime.l0.forward_model_runner import authoring_protocol_digest
    from network_runtime.l0.forward_qualification import evaluator_fingerprint
    from network_runtime.l0.production import CATALOG
    from network_runtime.l0.production_trajectory import (
        validate_production_trajectories,
    )

    tracked = _tracked_files()
    source_groups = {
        name: _source_group(tracked, selectors)
        for name, selectors in _SOURCE_GROUPS.items()
    }
    dirty_lines = tuple(
        line for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line
    )
    trajectories = validate_production_trajectories()
    catalog_json = json.loads(CATALOG.to_json())
    source_state = {
        "commit": _git("rev-parse", "HEAD").strip(),
        "branch": _git("branch", "--show-current").strip() or "detached",
        "clean": not dirty_lines,
        "dirtyEntryCount": len(dirty_lines),
        "dirtyStateDigest": sha256_json(sorted(dirty_lines)),
    }
    return {
        "label": label,
        "sourceState": source_state,
        "sourceGroups": source_groups,
        "sourceGroupsDigest": sha256_json({
            name: value["digest"] for name, value in source_groups.items()
        }),
        "runtimeKernelDigest": source_groups["runtime_kernel"]["digest"],
        "harnessBoundaryDigest": source_groups["harness_boundary"]["digest"],
        "contractCatalog": {
            "digest": sha256_json(catalog_json),
            "contractCount": len(catalog_json),
            "trajectoryContractCount": trajectories["contracts"],
            "promotionReadyCount": trajectories["promotion_ready"],
            "exactRoundTripCount": trajectories["exact_round_trips"],
        },
        "evaluatorFingerprint": evaluator_fingerprint(),
        "authoringProtocolDigest": authoring_protocol_digest(),
        "model": {
            "id": model,
            "artifactDigest": model_artifact_digest,
        },
        "providerLabFingerprintDigest": provider_lab_digest,
        "environment": _environment_binding(source_groups),
        "esP0Baseline": _es_p0_binding(source_groups),
    }


def create_research_freeze_manifest(
    output_path: str | Path,
    *,
    label: str,
    model: str,
    model_artifact_digest: str,
    provider_lab_digest: str,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Write one canonical freeze, or an explicitly non-frozen dirty preview."""

    bindings = _binding_snapshot(
        label=label, model=model, model_artifact_digest=model_artifact_digest,
        provider_lab_digest=provider_lab_digest,
    )
    clean = bool(bindings["sourceState"]["clean"])
    if not clean and not allow_dirty:
        raise ValueError(
            "research freeze requires a clean worktree; use --allow-dirty only "
            "for a non-frozen local preview"
        )
    frozen = clean
    body = {
        "apiVersion": RESEARCH_FREEZE_SCHEMA,
        "createdAt": utc_now(),
        "status": "frozen" if frozen else "preview_dirty_not_frozen",
        "frozen": frozen,
        "bindings": bindings,
        "invariants": [
            "private prompts, labels and observations are excluded",
            "a dirty worktree cannot produce a completed freeze",
            "model, protocol, evaluator, contracts and runtime are jointly bound",
            "the manifest grants no execution or activation authority",
        ],
        "claimBoundary": (
            "This manifest proves artifact identity and local consistency only; "
            "it does not prove model accuracy or production safety."
        ),
    }
    manifest = {**body, "freezeDigest": sha256_json(body)}
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_research_freeze_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Recompute all public bindings and report exact drift without private data."""

    source = Path(manifest_path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("apiVersion") != RESEARCH_FREEZE_SCHEMA:
        raise ValueError("research freeze manifest Schema is invalid")
    body = {key: value for key, value in raw.items() if key != "freezeDigest"}
    integrity_valid = raw.get("freezeDigest") == sha256_json(body)
    stored = raw.get("bindings")
    if not isinstance(stored, dict):
        raise ValueError("research freeze bindings are missing")
    model = stored.get("model")
    if not isinstance(model, dict):
        raise ValueError("research freeze model binding is missing")
    current = _binding_snapshot(
        label=str(stored.get("label", "")),
        model=str(model.get("id", "")),
        model_artifact_digest=str(model.get("artifactDigest", "")),
        provider_lab_digest=str(stored.get("providerLabFingerprintDigest", "")),
    )
    drift = [
        key for key in sorted(set(stored) | set(current))
        if stored.get(key) != current.get(key)
    ]
    frozen = bool(
        raw.get("frozen") is True
        and raw.get("status") == "frozen"
        and stored.get("sourceState", {}).get("clean") is True
        and current.get("sourceState", {}).get("clean") is True
    )
    bindings_valid = not drift
    return {
        "apiVersion": "netopyu.io/ensured-skill-research-freeze-check/v1",
        "ok": bool(integrity_valid and bindings_valid and frozen),
        "frozen": frozen,
        "integrityValid": integrity_valid,
        "bindingsValid": bindings_valid,
        "bindingDrift": drift,
        "freezeDigest": raw.get("freezeDigest"),
        "claimBoundary": raw.get("claimBoundary"),
    }
