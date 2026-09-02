"""Independent failure-semantics qualification runner for effect Providers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from .capabilities import CapabilityContract
from .contracts import sha256_json
from .provider_release import (
    QUALIFICATION_SCHEMA,
    QUALIFICATION_SUITE_VERSION,
    ProviderManifest,
    QualificationReport,
)


class ProviderQualificationTarget(Protocol):
    """Test-only control surface implemented by a Provider qualification adapter."""

    def describe_capability(self, tool_name: str) -> CapabilityContract: ...

    async def reset(self) -> str: ...

    async def snapshot_digest(self) -> str: ...

    async def apply(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        operation_id: str,
        sequence: int,
        fault: str | None = None,
    ) -> dict[str, Any]: ...

    async def reconcile(self, operation_id: str) -> dict[str, Any]: ...

    async def compensate(
        self,
        operation_id: str,
        *,
        fault: str | None = None,
    ) -> dict[str, Any]: ...

    async def restart(self) -> None: ...

    async def escalation_state(self, operation_id: str) -> str: ...


async def _raises(awaitable: Any) -> tuple[bool, str]:
    try:
        await awaitable
    except Exception as error:
        return True, type(error).__name__
    return False, "none"


def _released_contract(manifest: ProviderManifest, tool_name: str) -> dict[str, Any]:
    capability = next(
        (item for item in manifest.capabilities if item.tool_name == tool_name),
        None,
    )
    if capability is None:
        raise ValueError(f"tool {tool_name!r} is absent from the Provider manifest")
    if capability.kind != "effect" or capability.effect_semantics != "reversible":
        raise ValueError("qualification runner requires one reversible effect capability")
    return {
        key: value for key, value in capability.model_dump(mode="json").items()
        if key not in {"result_contract", "l0_contract_hashes"}
    }


async def run_provider_qualification(
    target: ProviderQualificationTarget,
    manifest: ProviderManifest,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    environment: str,
    now: datetime | None = None,
) -> QualificationReport:
    """Run the fixed failure suite and return a signable report only on 9/9 pass."""
    expected = _released_contract(manifest, tool_name)
    checks: dict[str, bool] = {}
    evidence: dict[str, dict[str, Any]] = {}

    discovered = target.describe_capability(tool_name).to_dict()
    checks["identity_and_schema_binding"] = discovered == expected
    evidence["identity_and_schema_binding"] = {
        "expected": sha256_json(expected),
        "discovered": sha256_json(discovered),
    }

    baseline = await target.reset()
    raised, error_type = await _raises(target.apply(
        tool_name, arguments,
        operation_id="qualification-before-send", sequence=1,
        fault="timeout_before_send",
    ))
    after = await target.snapshot_digest()
    checks["timeout_before_send_is_safe"] = raised and after == baseline
    evidence["timeout_before_send_is_safe"] = {
        "raised": raised, "error_type": error_type,
        "baseline": baseline, "after": after,
    }

    baseline = await target.reset()
    first = await target.apply(
        tool_name, arguments, operation_id="qualification-duplicate", sequence=1,
    )
    first_snapshot = await target.snapshot_digest()
    second = await target.apply(
        tool_name, arguments, operation_id="qualification-duplicate", sequence=1,
    )
    second_snapshot = await target.snapshot_digest()
    checks["duplicate_operation_is_idempotent"] = (
        first.get("operation_id") == second.get("operation_id")
        and first.get("state") == second.get("state") == "applied"
        and first_snapshot == second_snapshot
        and first_snapshot != baseline
    )
    evidence["duplicate_operation_is_idempotent"] = {
        "first": sha256_json(first), "second": sha256_json(second),
        "baseline": baseline, "first_snapshot": first_snapshot,
        "second_snapshot": second_snapshot,
    }

    baseline = await target.reset()
    raised, error_type = await _raises(target.apply(
        tool_name, arguments,
        operation_id="qualification-out-of-order", sequence=2,
    ))
    after = await target.snapshot_digest()
    checks["out_of_order_operation_is_rejected"] = raised and after == baseline
    evidence["out_of_order_operation_is_rejected"] = {
        "raised": raised, "error_type": error_type,
        "baseline": baseline, "after": after,
    }

    baseline = await target.reset()
    raised, error_type = await _raises(target.apply(
        tool_name, arguments,
        operation_id="qualification-partial", sequence=1,
        fault="after_commit_before_response",
    ))
    reconciled = await target.reconcile("qualification-partial")
    after = await target.snapshot_digest()
    checks["partial_success_is_reconciled"] = (
        raised and reconciled.get("state") == "applied" and after != baseline
    )
    evidence["partial_success_is_reconciled"] = {
        "raised": raised, "error_type": error_type,
        "reconciled": sha256_json(reconciled), "baseline": baseline, "after": after,
    }

    await target.reset()
    raised, error_type = await _raises(target.apply(
        tool_name, arguments,
        operation_id="qualification-unknown", sequence=1,
        fault="unknown_terminal",
    ))
    unknown = await target.reconcile("qualification-unknown")
    checks["unknown_terminal_is_not_blindly_retried"] = (
        raised
        and unknown.get("state") == "unknown"
        and unknown.get("apply_attempts") == 1
    )
    evidence["unknown_terminal_is_not_blindly_retried"] = {
        "raised": raised, "error_type": error_type,
        "reconciled": sha256_json(unknown),
        "apply_attempts": unknown.get("apply_attempts"),
    }

    baseline = await target.reset()
    applied = await target.apply(
        tool_name, arguments,
        operation_id="qualification-compensate", sequence=1,
    )
    compensated = await target.compensate("qualification-compensate")
    restored = await target.snapshot_digest()
    checks["compensation_restores_baseline"] = (
        applied.get("state") == "applied"
        and compensated.get("state") == "compensated"
        and restored == baseline
    )
    evidence["compensation_restores_baseline"] = {
        "applied": sha256_json(applied),
        "compensated": sha256_json(compensated),
        "baseline": baseline, "restored": restored,
    }

    await target.reset()
    await target.apply(
        tool_name, arguments,
        operation_id="qualification-compensation-failure", sequence=1,
    )
    raised, error_type = await _raises(target.compensate(
        "qualification-compensation-failure", fault="compensation_failure",
    ))
    escalation = await target.escalation_state("qualification-compensation-failure")
    checks["compensation_failure_escalates"] = (
        raised and escalation == "manual_intervention_required"
    )
    evidence["compensation_failure_escalates"] = {
        "raised": raised, "error_type": error_type, "escalation": escalation,
    }

    baseline = await target.reset()
    raised, error_type = await _raises(target.apply(
        tool_name, arguments,
        operation_id="qualification-restart", sequence=1,
        fault="after_commit_before_response",
    ))
    await target.restart()
    recovered = await target.reconcile("qualification-restart")
    after = await target.snapshot_digest()
    checks["restart_recovery_preserves_operation_state"] = (
        raised and recovered.get("state") == "applied" and after != baseline
    )
    evidence["restart_recovery_preserves_operation_state"] = {
        "raised": raised, "error_type": error_type,
        "recovered": sha256_json(recovered), "baseline": baseline, "after": after,
    }

    return QualificationReport(
        apiVersion=QUALIFICATION_SCHEMA,
        provider_id=manifest.provider_id,
        manifest_digest=manifest.digest,
        suite_version=QUALIFICATION_SUITE_VERSION,
        environment=environment,
        executed_at=(now or datetime.now(timezone.utc)).isoformat(),
        checks=checks,
        evidence_digests={name: sha256_json(value) for name, value in evidence.items()},
    )


__all__ = ["ProviderQualificationTarget", "run_provider_qualification"]
