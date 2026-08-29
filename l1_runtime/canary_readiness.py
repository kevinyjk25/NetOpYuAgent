"""Evidence-only P1.9-C1 canary readiness gate.

This module never changes adapter configuration and never authorizes traffic.
Its strongest result is ``ready_for_review``: a privacy-minimized statement
that four independently produced evidence documents are internally consistent.
Enterprise signatures, deployment authorization, and the actual activation
change remain external controls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from .adapter_qualification import ADAPTER_QUALIFICATION_SCHEMA
from .contracts import StrictModel
from .qualification import QUALIFICATION_SCHEMA


CANARY_PRODUCT_EVIDENCE_SCHEMA = "netopyu.io/l1-canary-product-evidence/v1"
CANARY_OPS_EVIDENCE_SCHEMA = "netopyu.io/l1-canary-ops-evidence/v1"
CANARY_READINESS_SCHEMA = "netopyu.io/l1-canary-readiness/v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_DOCUMENT_BYTES = 4_000_000
_WORKER_REQUIREMENTS = {
    "sealed_consensus_ready", "catalog_baseline_clean", "immutable_model_artifact",
    "label_coverage", "at_least_two_repetitions", "input_contract_parity",
    "decision_semantic_parity", "dsh_repeatability", "hermes_repeatability",
    "dsh_protocol", "hermes_protocol", "dsh_full_oracle", "hermes_full_oracle",
    "no_unsafe_escape", "all_expected_targets_retrieved",
}
_ADAPTER_REQUIREMENTS = {
    "sealed_consensus_ready", "catalog_baseline_clean", "immutable_model_artifact",
    "exact_case_coverage", "prompt_binding", "input_contract_parity",
    "decision_digest_parity", "protocol_success", "repeatability_when_requested",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _require_digest(value: str, label: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a canonical sha256 digest")
    return value


def _require_window(
    issued_at: datetime,
    expires_at: datetime,
    *,
    maximum: timedelta,
) -> None:
    if issued_at.tzinfo is None or expires_at.tzinfo is None:
        raise ValueError("evidence timestamps must include a timezone")
    if expires_at <= issued_at or expires_at - issued_at > maximum:
        raise ValueError("evidence validity window exceeds its reviewed bound")


class EvidenceBindings(StrictModel):
    model_artifact_digest: str
    sealed_manifest_digest: str
    consensus_labels_digest: str
    catalog_snapshot_digest: str

    @model_validator(mode="after")
    def validate_digests(self) -> "EvidenceBindings":
        for name in (
            "model_artifact_digest", "sealed_manifest_digest",
            "consensus_labels_digest", "catalog_snapshot_digest",
        ):
            _require_digest(getattr(self, name), name)
        return self


class ProductSurfaceEvidence(StrictModel):
    harness: Literal["dsh", "hermes"]
    entrypoint: Literal["web_ui", "cli"]
    real_process_exercised: Literal[True]
    decision_receipt_verified: Literal[True]
    interaction_slo_met: Literal[True]
    distribution_digest: str
    deployment_identity_digest: str
    test_receipt_digest: str

    @model_validator(mode="after")
    def validate_surface(self) -> "ProductSurfaceEvidence":
        expected = {"dsh": "web_ui", "hermes": "cli"}
        if self.entrypoint != expected[self.harness]:
            raise ValueError("Harness product entrypoint does not match the reviewed surface")
        for name in (
            "distribution_digest", "deployment_identity_digest", "test_receipt_digest",
        ):
            _require_digest(getattr(self, name), name)
        return self


class CanaryProductEvidence(StrictModel):
    api_version: Literal[CANARY_PRODUCT_EVIDENCE_SCHEMA] = Field(alias="apiVersion")
    issued_at: datetime
    expires_at: datetime
    bindings: EvidenceBindings
    worker_report_digest: str
    adapter_report_digest: str
    surfaces: tuple[ProductSurfaceEvidence, ProductSurfaceEvidence]
    reviewer_ids: tuple[str, str]
    evidence_digest: str

    @model_validator(mode="after")
    def validate_product_evidence(self) -> "CanaryProductEvidence":
        _require_window(self.issued_at, self.expires_at, maximum=timedelta(days=30))
        _require_digest(self.worker_report_digest, "worker report digest")
        _require_digest(self.adapter_report_digest, "adapter report digest")
        if {item.harness for item in self.surfaces} != {"dsh", "hermes"}:
            raise ValueError("product evidence requires exactly DSH and Hermes surfaces")
        if len(set(self.reviewer_ids)) != 2 or any(
            not 1 <= len(item.strip()) <= 128 for item in self.reviewer_ids
        ):
            raise ValueError("product evidence requires two distinct bounded reviewers")
        body = self.model_dump(by_alias=True, mode="json", exclude={"evidence_digest"})
        if self.evidence_digest != _digest(body):
            raise ValueError("product evidence digest does not bind the document")
        return self


class ExercisedControl(StrictModel):
    configured: Literal[True]
    tested: Literal[True]
    receipt_digest: str

    @model_validator(mode="after")
    def validate_receipt(self) -> "ExercisedControl":
        _require_digest(self.receipt_digest, "control receipt digest")
        return self


class CanaryLimits(StrictModel):
    max_traffic_percent: float = Field(gt=0.0, le=5.0)
    max_duration_minutes: int = Field(ge=1, le=120)
    automatic_approval_enabled: Literal[False]
    runtime_bypass_enabled: Literal[False]
    provider_bypass_enabled: Literal[False]


class CoreControlEvidence(StrictModel):
    passed: int = Field(ge=0)
    total: int = Field(ge=1)
    report_digest: str

    @model_validator(mode="after")
    def validate_receipt(self) -> "CoreControlEvidence":
        _require_digest(self.report_digest, "Core control report digest")
        return self


class RuntimeTrendEvidence(StrictModel):
    status: Literal["improved", "stable"]
    distinct_implementation_versions: int = Field(ge=3)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p50_within_threshold: Literal[True]
    p95_within_threshold: Literal[True]
    report_digest: str

    @model_validator(mode="after")
    def validate_trend(self) -> "RuntimeTrendEvidence":
        if self.p95_ms < self.p50_ms:
            raise ValueError("Runtime trend p95 cannot be lower than p50")
        _require_digest(self.report_digest, "Runtime trend report digest")
        return self


class PlanBindingEvidence(StrictModel):
    passed: int = Field(ge=0)
    total: int = Field(ge=1)
    decision_replay_count: int = Field(ge=0)
    authority_escape_count: int = Field(ge=0)
    report_digest: str

    @model_validator(mode="after")
    def validate_receipt(self) -> "PlanBindingEvidence":
        _require_digest(self.report_digest, "Decision-plan binding report digest")
        return self


class CanaryOpsEvidence(StrictModel):
    api_version: Literal[CANARY_OPS_EVIDENCE_SCHEMA] = Field(alias="apiVersion")
    issued_at: datetime
    expires_at: datetime
    bindings: EvidenceBindings
    product_evidence_digest: str
    limits: CanaryLimits
    kill_switch_to_shadow: ExercisedControl
    rollback_to_shadow: ExercisedControl
    alert_delivery: ExercisedControl
    no_effect_replay: ExercisedControl
    core_controls: CoreControlEvidence
    runtime_trend: RuntimeTrendEvidence
    decision_plan_binding: PlanBindingEvidence
    owner_ids: tuple[str, str]
    evidence_digest: str

    @model_validator(mode="after")
    def validate_ops_evidence(self) -> "CanaryOpsEvidence":
        _require_window(self.issued_at, self.expires_at, maximum=timedelta(days=7))
        _require_digest(self.product_evidence_digest, "product evidence digest")
        if len(set(self.owner_ids)) != 2 or any(
            not 1 <= len(item.strip()) <= 128 for item in self.owner_ids
        ):
            raise ValueError("operations evidence requires two distinct bounded owners")
        if self.core_controls.passed != 64 or self.core_controls.total != 64:
            raise ValueError("Runtime Core-72 fault controls must remain 64/64")
        control_receipts = {
            self.kill_switch_to_shadow.receipt_digest,
            self.rollback_to_shadow.receipt_digest,
            self.alert_delivery.receipt_digest,
            self.no_effect_replay.receipt_digest,
        }
        if len(control_receipts) != 4:
            raise ValueError("operations drills require distinct receipts")
        binding = self.decision_plan_binding
        if (
            binding.passed != binding.total
            or binding.decision_replay_count != 0
            or binding.authority_escape_count != 0
        ):
            raise ValueError("Decision-plan binding evidence is incomplete or escaped")
        body = self.model_dump(by_alias=True, mode="json", exclude={"evidence_digest"})
        if self.evidence_digest != _digest(body):
            raise ValueError("operations evidence digest does not bind the document")
        return self


def _report_integrity(payload: dict[str, Any], schema: str) -> bool:
    try:
        if payload.get("apiVersion") != schema or not _DIGEST.fullmatch(
            str(payload.get("report_digest") or "")
        ):
            return False
        body = dict(payload)
        report_digest = body.pop("report_digest")
        body.pop("apiVersion", None)
        return report_digest == _digest(body)
    except (TypeError, ValueError):
        return False


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _requirements_pass(value: Any, expected: set[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == expected
        and all(item is True for item in value.values())
    )


def _binding_tuple(payload: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        payload.get("model_artifact_digest"),
        payload.get("sealed_manifest_digest"),
        payload.get("consensus_labels_digest"),
        payload.get("catalog_snapshot_digest"),
    )


def _binding_model_tuple(bindings: EvidenceBindings) -> tuple[str, str, str, str]:
    return (
        bindings.model_artifact_digest,
        bindings.sealed_manifest_digest,
        bindings.consensus_labels_digest,
        bindings.catalog_snapshot_digest,
    )


def evaluate_canary_readiness(
    worker_report: dict[str, Any] | None,
    adapter_report: dict[str, Any] | None,
    product_payload: dict[str, Any] | None,
    ops_payload: dict[str, Any] | None,
    *,
    checked_at: datetime | None = None,
    initial_reasons: tuple[str, ...] = (),
    source_digests: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Return ``ready_for_review`` only when every independent gate agrees."""
    now = checked_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("readiness check time must include a timezone")
    reasons = set(initial_reasons)
    checks: dict[str, bool] = {}

    worker_valid = isinstance(worker_report, dict) and _report_integrity(
        worker_report, QUALIFICATION_SCHEMA,
    )
    checks["worker_report_integrity"] = worker_valid
    worker_scope = _mapping(worker_report.get("scope")) if worker_valid else {}
    worker_privacy = _mapping(worker_report.get("privacy")) if worker_valid else {}
    worker_harnesses = worker_scope.get("harnesses")
    worker_passed = bool(
        worker_valid
        and worker_report.get("status") == "qualified"
        and worker_report.get("qualified") is True
        and _requirements_pass(worker_report.get("requirements"), _WORKER_REQUIREMENTS)
        and worker_scope.get("level") == "shared_worker_decision_contract"
        and worker_scope.get("full_harness_adapter_loop") is False
        and isinstance(worker_harnesses, (list, tuple))
        and set(worker_harnesses) == {"dsh", "hermes"}
        and isinstance(worker_report.get("case_count"), int)
        and worker_report.get("case_count") >= 20
        and isinstance(worker_report.get("repetitions"), int)
        and worker_report.get("repetitions") >= 2
        and worker_report.get("execution_count")
        == worker_report.get("case_count") * worker_report.get("repetitions") * 2
        and worker_privacy.get("raw_prompts_emitted") is False
        and worker_privacy.get("raw_labels_emitted") is False
        and worker_privacy.get("argument_values_emitted") is False
    )
    checks["worker_qualification_passed"] = worker_passed
    if not worker_valid:
        reasons.add("worker_report_invalid")
    elif not worker_passed:
        reasons.add("worker_qualification_failed")

    adapter_valid = isinstance(adapter_report, dict) and _report_integrity(
        adapter_report, ADAPTER_QUALIFICATION_SCHEMA,
    )
    checks["adapter_report_integrity"] = adapter_valid
    adapter_scope = _mapping(adapter_report.get("scope")) if adapter_valid else {}
    adapter_privacy = _mapping(adapter_report.get("privacy")) if adapter_valid else {}
    adapter_passed = bool(
        adapter_valid
        and adapter_report.get("status") == "adapter_parity_passed"
        and adapter_report.get("passed") is True
        and _requirements_pass(adapter_report.get("requirements"), _ADAPTER_REQUIREMENTS)
        and adapter_scope.get("level") == "adapter_hook_to_worker"
        and adapter_scope.get("dsh_javascript_agent_pre_step") is True
        and adapter_scope.get("hermes_python_pre_llm_call") is True
        and adapter_scope.get("persistent_owner_only_worker") is True
        and isinstance(adapter_report.get("case_count"), int)
        and adapter_report.get("case_count") >= 20
        and isinstance(adapter_report.get("repetitions"), int)
        and adapter_report.get("repetitions") >= 1
        and adapter_privacy.get("raw_prompts_emitted") is False
        and adapter_privacy.get("raw_labels_emitted") is False
        and adapter_privacy.get("argument_values_emitted") is False
    )
    checks["adapter_qualification_passed"] = adapter_passed
    if not adapter_valid:
        reasons.add("adapter_report_invalid")
    elif not adapter_passed:
        reasons.add("adapter_qualification_failed")

    product: CanaryProductEvidence | None = None
    try:
        if not isinstance(product_payload, dict):
            raise ValueError("missing product evidence")
        product = CanaryProductEvidence.model_validate(product_payload)
    except (TypeError, ValueError):
        reasons.add("product_evidence_invalid")
    checks["product_evidence_valid"] = product is not None

    ops: CanaryOpsEvidence | None = None
    try:
        if not isinstance(ops_payload, dict):
            raise ValueError("missing operations evidence")
        ops = CanaryOpsEvidence.model_validate(ops_payload)
    except (TypeError, ValueError):
        reasons.add("operations_evidence_invalid")
    checks["operations_evidence_valid"] = ops is not None

    product_fresh = bool(
        product
        and product.issued_at <= now + timedelta(minutes=5)
        and product.expires_at > now
    )
    ops_fresh = bool(
        ops and ops.issued_at <= now + timedelta(minutes=5) and ops.expires_at > now
    )
    checks["product_evidence_fresh"] = product_fresh
    checks["operations_evidence_fresh"] = ops_fresh
    if product is not None and not product_fresh:
        reasons.add("product_evidence_stale")
    if ops is not None and not ops_fresh:
        reasons.add("operations_evidence_stale")

    cross_binding = False
    product_refs = False
    ops_refs = False
    if worker_valid and adapter_valid and product is not None:
        worker_binding = _binding_tuple(worker_report)
        adapter_binding = _binding_tuple(adapter_report)
        product_binding = _binding_model_tuple(product.bindings)
        cross_binding = (
            worker_binding == adapter_binding == product_binding
            and all(isinstance(item, str) and _DIGEST.fullmatch(item) for item in worker_binding)
            and worker_report.get("model") == adapter_report.get("model")
            and worker_report.get("case_count") == adapter_report.get("case_count")
        )
        product_refs = (
            product.worker_report_digest == worker_report.get("report_digest")
            and product.adapter_report_digest == adapter_report.get("report_digest")
        )
        if ops is not None:
            cross_binding = cross_binding and _binding_model_tuple(ops.bindings) == product_binding
            ops_refs = ops.product_evidence_digest == product.evidence_digest
    checks["artifact_cross_binding"] = cross_binding
    checks["product_report_references"] = product_refs
    checks["operations_product_reference"] = ops_refs
    role_separation = bool(
        product
        and ops
        and set(product.reviewer_ids).isdisjoint(ops.owner_ids)
    )
    checks["reviewer_owner_separation"] = role_separation
    if not cross_binding:
        reasons.add("artifact_binding_mismatch")
    if not product_refs:
        reasons.add("product_report_reference_mismatch")
    if not ops_refs:
        reasons.add("operations_product_reference_mismatch")
    if not role_separation:
        reasons.add("reviewer_owner_role_overlap")

    ready = all(checks.values()) and not reasons
    status = "ready_for_review" if ready else "not_ready"
    sources = source_digests or {}
    body = {
        "status": status,
        "checked_at": now.astimezone(timezone.utc).isoformat(),
        "checks": checks,
        "reason_codes": sorted(reasons),
        "source_document_digests": {
            name: sources.get(name)
            for name in ("worker_report", "adapter_report", "product_evidence", "ops_evidence")
        },
        "bound_artifacts": {
            "model_artifact_digest": (
                product.bindings.model_artifact_digest if product else None
            ),
            "sealed_manifest_digest": (
                product.bindings.sealed_manifest_digest if product else None
            ),
            "consensus_labels_digest": (
                product.bindings.consensus_labels_digest if product else None
            ),
            "catalog_snapshot_digest": (
                product.bindings.catalog_snapshot_digest if product else None
            ),
        },
        "activation_authorized": False,
        "configuration_changed": False,
        "traffic_changed": False,
        "claim_boundary": (
            "ready_for_review confirms local evidence consistency only; it does not activate "
            "canary, authorize production traffic, prove enterprise identity/non-repudiation, "
            "or state a production success probability."
        ),
        "privacy": {
            "raw_prompts_emitted": False,
            "raw_labels_emitted": False,
            "argument_values_emitted": False,
            "reviewer_or_owner_ids_emitted": False,
        },
    }
    return {
        "apiVersion": CANARY_READINESS_SCHEMA,
        **body,
        "report_digest": _digest(body),
    }


def _read_document(path: Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        if path.is_symlink() or not path.is_file():
            return None, None, "missing_or_unsafe_evidence_file"
        raw = path.read_bytes()
        if len(raw) > _MAX_DOCUMENT_BYTES:
            return None, None, "evidence_file_too_large"
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None, f"sha256:{sha256(raw).hexdigest()}", "evidence_document_not_object"
        return payload, f"sha256:{sha256(raw).hexdigest()}", None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None, "evidence_file_unreadable"


def evaluate_canary_readiness_files(
    worker_path: Path,
    adapter_path: Path,
    product_path: Path,
    ops_path: Path,
    *,
    checked_at: datetime | None = None,
) -> dict[str, Any]:
    documents: dict[str, dict[str, Any] | None] = {}
    digests: dict[str, str | None] = {}
    reasons: list[str] = []
    for name, path in (
        ("worker_report", worker_path),
        ("adapter_report", adapter_path),
        ("product_evidence", product_path),
        ("ops_evidence", ops_path),
    ):
        payload, digest, error = _read_document(path)
        documents[name] = payload
        digests[name] = digest
        if error:
            reasons.append(f"{name}_{error}")
    return evaluate_canary_readiness(
        documents["worker_report"],
        documents["adapter_report"],
        documents["product_evidence"],
        documents["ops_evidence"],
        checked_at=checked_at,
        initial_reasons=tuple(reasons),
        source_digests=digests,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check P1.9 canary evidence without changing configuration or traffic.",
    )
    parser.add_argument("worker_report", type=Path)
    parser.add_argument("adapter_report", type=Path)
    parser.add_argument("product_evidence", type=Path)
    parser.add_argument("ops_evidence", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    arguments = parser.parse_args(argv)
    report = evaluate_canary_readiness_files(
        arguments.worker_report,
        arguments.adapter_report,
        arguments.product_evidence,
        arguments.ops_evidence,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANARY_OPS_EVIDENCE_SCHEMA",
    "CANARY_PRODUCT_EVIDENCE_SCHEMA",
    "CANARY_READINESS_SCHEMA",
    "CanaryOpsEvidence",
    "CanaryProductEvidence",
    "evaluate_canary_readiness",
    "evaluate_canary_readiness_files",
]
