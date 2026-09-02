"""Out-of-process Tool backend for the public-Skill DSH paired study.

Control calls the declarative fixture Provider directly.  Treatment keeps the
same visible Tool surface, but a qualified mutation is consumed by a sealed
L0 plan and executed as one deterministic transaction.  Unqualified writes
fail closed; reads remain available in every mode.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from network_runtime.contracts import OutcomeIndeterminateError, sha256_json

from evaluation.public_skill_fixture_mcp import (
    FixtureCapability,
    FixtureToolEngine,
    load_fixture_inputs,
)
from evaluation.public_skill_translation import L0_PLAN_SCHEMA


RESULT_SCHEMA = "effect-runtime.io/public-skill-harness-tool-result/v1"
Mode = Literal["l1_native", "l0_runtime", "safe_stop"]


@dataclass(frozen=True)
class PublicHarnessContext:
    case_id: str
    mode: Mode
    session_id: str
    catalog_path: Path
    fixture_path: Path
    catalog_digest: str
    fixture_digest: str
    l0_path: Path | None
    l0_digest: str | None

    @classmethod
    def load(cls, path: str | Path) -> "PublicHarnessContext":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("public Harness context must be an object")
        mode = str(value.get("mode") or "")
        if mode not in {"l1_native", "l0_runtime", "safe_stop"}:
            raise ValueError("public Harness mode is invalid")
        l0_value = value.get("l0Path")
        context = cls(
            case_id=str(value.get("caseId") or ""), mode=mode,  # type: ignore[arg-type]
            session_id=str(value.get("sessionId") or ""),
            catalog_path=Path(str(value.get("catalogPath") or "")).resolve(),
            fixture_path=Path(str(value.get("fixturePath") or "")).resolve(),
            catalog_digest=str(value.get("catalogDigest") or ""),
            fixture_digest=str(value.get("fixtureDigest") or ""),
            l0_path=None if not l0_value else Path(str(l0_value)).resolve(),
            l0_digest=None if value.get("l0Digest") is None else str(value["l0Digest"]),
        )
        if not context.case_id or not context.session_id:
            raise ValueError("public Harness caseId and sessionId are required")
        catalog, _, fixture = load_fixture_inputs(
            context.catalog_path, context.fixture_path,
            expected_case_id=context.case_id,
        )
        if sha256_json(catalog) != context.catalog_digest or sha256_json(fixture) != context.fixture_digest:
            raise ValueError("public Harness catalog or fixture digest drift")
        if context.mode == "l0_runtime":
            if context.l0_path is None or not context.l0_path.is_file():
                raise ValueError("public Harness Runtime mode requires a sealed L0 plan")
            plan = json.loads(context.l0_path.read_text(encoding="utf-8"))
            plan_body = {key: item for key, item in plan.items() if key != "planDigest"}
            if (
                plan.get("apiVersion") != L0_PLAN_SCHEMA
                or plan.get("planDigest") != context.l0_digest
                or plan.get("planDigest") != sha256_json(plan_body)
                or plan.get("caseId") != context.case_id
                or plan.get("toolCatalogDigest") != context.catalog_digest
            ):
                raise ValueError("public Harness L0 plan binding mismatch")
        elif context.l0_path is not None or context.l0_digest is not None:
            raise ValueError("public Harness non-Runtime mode cannot load L0")
        return context


def _result(ok: bool, code: str, **values: Any) -> dict[str, Any]:
    return {"apiVersion": RESULT_SCHEMA, "ok": ok, "code": code, **values}


def _arguments_for(
    capability: FixtureCapability, effect_arguments: dict[str, Any],
    *, snapshot: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    values: dict[str, Any] = {}
    snapshot_name = capability.operation.get("snapshotArgument")
    for name in capability.input_schema["required"]:
        if name == snapshot_name and snapshot is not None:
            values[name] = snapshot
        elif name in effect_arguments:
            values[name] = effect_arguments[name]
        else:
            return None
    for name in capability.input_schema["properties"]:
        if name in effect_arguments:
            values[name] = effect_arguments[name]
    return values


def _snapshot(value: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("snapshot", "record", "current"):
        candidate = value.get(key)
        if isinstance(candidate, dict):
            return candidate
    return None


def _desired_verified(
    effect: FixtureCapability, effect_arguments: dict[str, Any],
    observation: dict[str, Any], before: dict[str, Any] | None,
) -> bool:
    kind = effect.operation["kind"]
    if kind == "delete_record":
        return observation.get("code") == "record_not_found"
    record = observation.get("record")
    if not observation.get("ok") or not isinstance(record, dict):
        return False
    desired = {
        field: effect_arguments[argument]
        for field, argument in effect.operation.get("valueArguments", {}).items()
    }
    if any(record.get(field) != value for field, value in desired.items()):
        return False
    return before is None or int(record.get("revision") or 0) > int(before.get("revision") or 0)


def _snapshot_restored(observation: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    return observation.get("ok") is True and observation.get("record") == snapshot


def _runtime_transaction(
    *, context: PublicHarnessContext, engine: FixtureToolEngine,
    tool_name: str, arguments: dict[str, Any],
) -> dict[str, Any]:
    plan = json.loads(context.l0_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    transaction = plan["transaction"]
    by_capability = {item.capability_id: item for item in engine.capabilities}
    effect = by_capability.get(str(transaction.get("effectCapability") or ""))
    called = engine.by_name.get(tool_name)
    binding = {
        "calledTool": tool_name,
        "expectedTool": None if effect is None else effect.tool_name,
        "exact": effect is not None and called is not None and called.capability_id == effect.capability_id,
    }
    if effect is None or not binding["exact"]:
        return _result(
            False, "l0_effect_binding_rejected", terminal="rejected",
            execution="l0_runtime", argumentBinding=binding,
        )
    if (
        set(arguments) != set(transaction["parameterNames"])
        or arguments != transaction["parameterValues"]
    ):
        return _result(
            False, "l0_parameter_binding_rejected", terminal="rejected",
            execution="l0_runtime", argumentBinding={
                **binding, "parametersExact": False,
                "expectedDigest": sha256_json(transaction["parameterValues"]),
                "observedDigest": sha256_json(arguments),
            },
        )
    if not engine.approval["approved"]:
        return _result(
            False, "approval_denied", terminal="rejected", execution="l0_runtime",
            argumentBinding={**binding, "parametersExact": True}, effectCalls=0,
        )

    evidence: list[dict[str, Any]] = []
    snapshot: dict[str, Any] | None = None
    for capability_id in transaction["preflightCapabilities"]:
        capability = by_capability[capability_id]
        values = _arguments_for(capability, arguments)
        if values is None:
            return _result(
                False, "preflight_parameter_mapping_failed", terminal="rejected",
                execution="l0_runtime", effectCalls=0,
            )
        observed = engine.invoke(capability.tool_name, values)
        evidence.append({"phase": "preflight", "capability": capability_id, "result": observed})
        if not observed.get("ok"):
            return _result(
                False, "preflight_failed", terminal="rejected", execution="l0_runtime",
                evidence=evidence, effectCalls=0,
            )
        snapshot = snapshot or _snapshot(observed)
    if snapshot is None:
        return _result(
            False, "preflight_snapshot_missing", terminal="rejected",
            execution="l0_runtime", evidence=evidence, effectCalls=0,
        )

    effect_sent = False
    effect_unknown = False
    try:
        effect_sent = True
        effect_result = engine.invoke(effect.tool_name, arguments, runtime_authorized=True)
        evidence.append({"phase": "effect", "capability": effect.capability_id, "result": effect_result})
        if not effect_result.get("ok"):
            return _result(
                False, "effect_rejected", terminal="rejected", execution="l0_runtime",
                evidence=evidence, effectCalls=1,
            )
    except OutcomeIndeterminateError as error:
        effect_unknown = True
        evidence.append({"phase": "effect", "capability": effect.capability_id, "outcome": "unknown", "error": str(error)})
    except Exception as error:
        if engine.fixture["fault"] == "provider_error_before_send":
            return _result(
                False, "provider_error_before_send", terminal="rejected",
                execution="l0_runtime", evidence=evidence, effectCalls=0,
            )
        effect_unknown = True
        evidence.append({"phase": "effect", "capability": effect.capability_id, "outcome": "unknown", "error": str(error)})

    verification = by_capability[str(transaction["verificationCapability"])]
    verification_arguments = _arguments_for(verification, arguments)
    if verification_arguments is None:
        return _result(
            False, "verification_parameter_mapping_failed",
            terminal="manual_intervention_required" if effect_sent else "rejected",
            execution="l0_runtime", evidence=evidence, effectCalls=int(effect_sent),
        )
    observed = engine.invoke(verification.tool_name, verification_arguments)
    verified = _desired_verified(effect, arguments, observed, snapshot)
    evidence.append({
        "phase": "reconciliation" if effect_unknown else "verification",
        "capability": verification.capability_id, "passed": verified, "result": observed,
    })
    if verified:
        return _result(
            True, "verified_success", terminal="verified_success", execution="l0_runtime",
            evidence=evidence, effectCalls=1, reconciled=effect_unknown,
            argumentBinding={**binding, "parametersExact": True},
        )

    compensation_id = transaction.get("compensationCapability")
    if not compensation_id:
        return _result(
            False, "verification_failed", terminal="manual_intervention_required",
            execution="l0_runtime", evidence=evidence, effectCalls=1,
        )
    compensation = by_capability[str(compensation_id)]
    compensation_arguments = _arguments_for(compensation, arguments, snapshot=snapshot)
    if compensation_arguments is None:
        return _result(
            False, "compensation_parameter_mapping_failed", terminal="manual_intervention_required",
            execution="l0_runtime", evidence=evidence, effectCalls=1,
        )
    try:
        restored = engine.invoke(
            compensation.tool_name, compensation_arguments, runtime_authorized=True,
        )
        evidence.append({"phase": "compensation", "capability": compensation.capability_id, "result": restored})
    except Exception as error:
        evidence.append({"phase": "compensation", "capability": compensation.capability_id, "error": str(error)})
        return _result(
            False, "compensation_failed", terminal="manual_intervention_required",
            execution="l0_runtime", evidence=evidence, effectCalls=1,
        )
    rollback_observation = engine.invoke(verification.tool_name, verification_arguments)
    rollback_verified = _snapshot_restored(rollback_observation, snapshot)
    evidence.append({
        "phase": "rollback_verification", "capability": verification.capability_id,
        "passed": rollback_verified, "result": rollback_observation,
    })
    return _result(
        rollback_verified, "rollback_verified" if rollback_verified else "rollback_verification_failed",
        terminal="rollback_verified" if rollback_verified else "manual_intervention_required",
        execution="l0_runtime", evidence=evidence, effectCalls=1,
    )


def execute_public_harness_tool(
    *, context: PublicHarnessContext, store_path: str | Path,
    trace_path: str | Path, tool_name: str, arguments: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    catalog, _, fixture = load_fixture_inputs(
        context.catalog_path, context.fixture_path, expected_case_id=context.case_id,
    )
    engine = FixtureToolEngine(catalog, fixture, store_path, mode=context.mode)
    capability = engine.by_name.get(tool_name)
    if capability is None:
        response = _result(False, "tool_not_found", execution=context.mode)
    elif context.mode == "l0_runtime" and capability.action_type != "read_only":
        response = _runtime_transaction(
            context=context, engine=engine, tool_name=tool_name, arguments=arguments,
        )
    elif context.mode == "safe_stop" and capability.action_type != "read_only":
        response = _result(
            False, "translation_unqualified_safe_stop", terminal="rejected",
            execution="safe_stop", effectCalls=0,
        )
    else:
        try:
            provider = engine.invoke(tool_name, arguments)
            response = _result(
                bool(provider.get("ok")), str(provider.get("code") or "provider_result"),
                execution=context.mode, provider=provider,
            )
        except OutcomeIndeterminateError as error:
            response = _result(False, "outcome_unknown", execution=context.mode, error=str(error))
        except Exception as error:
            response = _result(False, "provider_error", execution=context.mode, error=str(error))
    trace = {
        "caseId": context.case_id, "sessionId": context.session_id,
        "mode": context.mode, "tool": tool_name, "argumentsDigest": sha256_json(arguments),
        "result": response, "latencyMs": round((time.perf_counter() - started) * 1000, 3),
        "audit": engine.store.audit(),
    }
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--arguments", required=True)
    args = parser.parse_args(argv)
    context = PublicHarnessContext.load(args.context)
    arguments = json.loads(args.arguments)
    if not isinstance(arguments, dict):
        raise ValueError("public Harness Tool arguments must be an object")
    result = execute_public_harness_tool(
        context=context, store_path=args.store, trace_path=args.trace,
        tool_name=args.tool, arguments=arguments,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "PublicHarnessContext", "RESULT_SCHEMA", "execute_public_harness_tool",
]


if __name__ == "__main__":
    raise SystemExit(main())
