from __future__ import annotations

import json
import subprocess
from pathlib import Path

from network_runtime.contracts import sha256_json
from network_runtime.l0.research_freeze import (
    RESEARCH_FREEZE_SCHEMA,
    create_research_freeze_manifest,
    verify_research_freeze_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIGEST = "sha256:" + "a" * 64
LAB_DIGEST = "sha256:" + "c" * 64


def _preview(tmp_path: Path) -> Path:
    path = tmp_path / "research-freeze.json"
    manifest = create_research_freeze_manifest(
        path,
        label="es-p1-unit-test",
        model="qwen3.5:9b",
        model_artifact_digest=MODEL_DIGEST,
        provider_lab_digest=LAB_DIGEST,
        allow_dirty=True,
    )
    assert manifest["apiVersion"] == RESEARCH_FREEZE_SCHEMA
    assert manifest["bindings"]["contractCatalog"]["contractCount"] == 21
    assert manifest["bindings"]["contractCatalog"]["promotionReadyCount"] == 21
    assert manifest["bindings"]["contractCatalog"]["exactRoundTripCount"] == 21
    assert manifest["bindings"]["model"]["artifactDigest"] == MODEL_DIGEST
    return path


def test_research_freeze_preview_is_integrity_checked_and_never_overclaims(
    tmp_path: Path,
) -> None:
    path = _preview(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = verify_research_freeze_manifest(path)
    assert result["integrityValid"] is True
    assert result["bindingsValid"] is True
    assert result["bindingDrift"] == []
    assert result["ok"] is manifest["frozen"]
    if not manifest["bindings"]["sourceState"]["clean"]:
        assert manifest["status"] == "preview_dirty_not_frozen"
        assert result["ok"] is False


def test_research_freeze_detects_integrity_and_semantic_binding_tampering(
    tmp_path: Path,
) -> None:
    path = _preview(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["bindings"]["evaluatorFingerprint"] = "sha256:" + "b" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    result = verify_research_freeze_manifest(path)
    assert result["integrityValid"] is False
    assert result["bindingsValid"] is False
    assert "evaluatorFingerprint" in result["bindingDrift"]

    raw["freezeDigest"] = sha256_json({
        key: value for key, value in raw.items() if key != "freezeDigest"
    })
    path.write_text(json.dumps(raw), encoding="utf-8")
    rebound = verify_research_freeze_manifest(path)
    assert rebound["integrityValid"] is True
    assert rebound["bindingsValid"] is False
    assert "evaluatorFingerprint" in rebound["bindingDrift"]


def test_research_freeze_cli_writes_a_compact_preview(tmp_path: Path) -> None:
    path = tmp_path / "cli-freeze.json"
    completed = subprocess.run(
        [
            str(ROOT / "scripts/netopyu-l0"), "research-freeze-create",
            "--label", "es-p1-cli-test",
            "--model", "qwen3.5:9b",
            "--model-artifact-digest", MODEL_DIGEST,
            "--provider-lab-digest", LAB_DIGEST,
            "--output", str(path), "--allow-dirty",
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    printed = json.loads(completed.stdout)
    assert printed["contractCount"] == 21
    assert printed["output"] == str(path)
    assert "bindings" not in printed
    assert path.is_file()
