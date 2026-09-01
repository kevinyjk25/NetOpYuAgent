"""Tool backend used by the real DSH Skill execution comparison.

The Harness sees the same four Provider tools in both experiment arms.  This
module is invoked out-of-process by the Harness plugin so Provider state
survives individual tool calls without giving the Agent filesystem access.

``l1_native`` maps every call directly to the simulated Provider.  In
``l0_runtime`` only the effect call is wrapped by the production Runtime; read
tools remain ordinary Provider observations. ``safe_stop`` permits reads but
never a Provider mutation when translation is unqualified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from effect_runtime.mcp_lab import (
    DOMAINS,
    TOOLS_BY_NAME,
    EffectLabBackendFactory,
    EffectLabStore,
    effect_lab_runtime_registration,
)
from network_runtime import (
    NetworkRuntime, PlanState, validate_exact_argument_binding,
)
from network_runtime.contracts import OutcomeIndeterminateError, sha256_json


Mode = Literal["l1_native", "l0_runtime", "safe_stop"]
TOOL_RESULT_SCHEMA = "effect-runtime.io/harness-tool-result/v1"


@dataclass(frozen=True)
class HarnessToolContext:
    case_id: str
    domain: str
    mode: Mode
    approved: bool
    fault: str
    l0_skill_id: str
    session_id: str
    harness: str
    request_arguments: dict[str, Any]
    request_arguments_digest: str

    @classmethod
    def load(cls, path: str | Path) -> "HarnessToolContext":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Harness tool context must be a JSON object")
        context = cls(
            case_id=str(value["caseId"]),
            domain=str(value["domain"]),
            mode=str(value["mode"]),  # type: ignore[arg-type]
            approved=value["approved"],
            fault=str(value.get("fault") or "none"),
            l0_skill_id=str(value["l0SkillId"]),
            session_id=str(value["sessionId"]),
            harness=str(value["harness"]),
            request_arguments=dict(value["requestArguments"]),
            request_arguments_digest=str(value["requestArgumentsDigest"]),
        )
        if context.domain not in DOMAINS:
            raise ValueError(f"Unknown Effect Lab domain: {context.domain}")
        if context.mode not in {"l1_native", "l0_runtime", "safe_stop"}:
            raise ValueError(f"Unknown execution mode: {context.mode}")
        if not isinstance(context.approved, bool):
            raise ValueError("approved must be a boolean")
        if not context.case_id or not context.session_id:
            raise ValueError("caseId and sessionId are required")
        if sha256_json(context.request_arguments) != context.request_arguments_digest:
            raise ValueError("request argument digest does not match context")
        return context


def _result(*, ok: bool, code: str, **values: Any) -> dict[str, Any]:
    return {"schema": TOOL_RESULT_SCHEMA, "ok": ok, "code": code, **values}


def _provider_result(
    value: dict[str, Any], *, code: str, execution: str,
) -> dict[str, Any]:
    payload = {key: item for key, item in value.items() if key not in {"ok", "code"}}
    return _result(
        ok=bool(value.get("ok")),
        code=str(value.get("code") or code),
        execution=execution,
        **payload,
    )


def _validate_arguments(tool_name: str, arguments: dict[str, Any]) -> str | None:
    spec = TOOLS_BY_NAME.get(tool_name)
    if spec is None:
        return "tool_not_found"
    unknown = sorted(set(arguments) - set(spec.parameters))
    if unknown:
        return "unknown_parameters:" + ",".join(unknown)
    missing = sorted(
        name for name in spec.required
        if name not in arguments or arguments[name] in (None, "")
    )
    if missing:
        return "missing_parameters:" + ",".join(missing)
    expected_types = {
        "string": str,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for name, value in arguments.items():
        declared = str(spec.parameters[name].get("type") or "")
        expected = expected_types.get(declared)
        if expected is not None and (
            not isinstance(value, expected)
            or (declared == "integer" and isinstance(value, bool))
        ):
            return f"invalid_type:{name}"
    return None


def _append_trace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


async def _runtime_effect(
    *, context: HarnessToolContext, store: EffectLabStore,
    journal_path: Path, arguments: dict[str, Any],
) -> dict[str, Any]:
    spec = TOOLS_BY_NAME[f"{context.domain}_apply_change"]
    binding = validate_exact_argument_binding(
        authoritative=context.request_arguments,
        invocation=arguments,
        declared_fields=spec.parameters,
        required_fields=spec.required,
    )
    if not binding.ok:
        return _result(
            ok=False,
            code="request_binding_rejected",
            terminal="rejected",
            argumentBinding=binding.to_dict(),
            execution="l0_runtime",
        )
    runtime = NetworkRuntime(
        journal_path, backend_factory=EffectLabBackendFactory(store),
    )
    prepared = await runtime.prepare(
        f"effect-{context.domain}",
        f"{context.domain}_apply_change",
        arguments,
        l0_skill_id=context.l0_skill_id,
        session_id=context.session_id,
        harness=context.harness,
    )
    if prepared.get("status") != "plan_ready":
        return _result(
            ok=False,
            code=str(prepared.get("status") or "runtime_prepare_rejected"),
            terminal=str(prepared.get("status") or "rejected"),
            errors=list(prepared.get("errors") or ()),
            argumentBinding=binding.to_dict(),
            execution="l0_runtime",
        )
    plan = prepared["plan"]
    if not context.approved:
        rejected = runtime.reject(
            plan_id=plan["plan_id"],
            plan_hash=plan["plan_hash"],
            reason="controlled external approval denied",
        )
        audit = runtime.audit(plan["plan_id"])
        return _result(
            ok=False,
            code="approval_denied",
            terminal=str(rejected["state"]),
            planId=plan["plan_id"],
            auditValid=bool(audit["ok"]),
            argumentBinding=binding.to_dict(),
            execution="l0_runtime",
        )
    outcome = await runtime.execute(
        plan_id=plan["plan_id"],
        plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        allow_destructive=True,
        approval_request_id=f"approval-{context.case_id}",
        approval_actor="controlled-evaluation-operator",
    )
    audit = runtime.audit(plan["plan_id"])
    terminal = outcome.state.value
    return _result(
        ok=terminal == PlanState.VERIFIED_SUCCESS.value,
        code=terminal,
        terminal=terminal,
        planId=plan["plan_id"],
        auditValid=bool(audit["ok"]),
        argumentBinding=binding.to_dict(),
        evidence=[asdict(item) for item in outcome.evidence],
        error=outcome.error,
        execution="l0_runtime",
    )


async def execute_harness_tool(
    *, context: HarnessToolContext, store_path: str | Path,
    journal_path: str | Path, trace_path: str | Path,
    tool_name: str, arguments: dict[str, Any],
) -> dict[str, Any]:
    """Execute one Harness-visible tool call and persist a tamper-evident trace."""

    started = time.perf_counter()
    spec = TOOLS_BY_NAME.get(tool_name)
    error = _validate_arguments(tool_name, arguments)
    if spec is None or spec.domain != context.domain:
        error = "tool_not_available_for_case"
    elif context.mode == "safe_stop" and spec.role == "effect":
        # A non-admitted translation may not turn a model invention into an
        # authoritative request.  Validate the original user arguments before
        # considering the Agent invocation, so a filled-in required field is
        # still surfaced as a clarification rather than silently accepted.
        authoritative_error = _validate_arguments(
            tool_name, context.request_arguments,
        )
        if authoritative_error:
            error = authoritative_error
    store = EffectLabStore(store_path)
    store.fault = context.fault
    if error:
        if context.mode == "safe_stop":
            clarification = error.startswith("missing_parameters:")
            response = _result(
                ok=False,
                code="clarification_required" if clarification else "rejected",
                terminal="clarification_required" if clarification else "rejected",
                reason=error,
                execution="safe_stop",
            )
        else:
            response = _result(ok=False, code=error, execution=context.mode)
    elif spec.role == "observation":
        response = _provider_result(
            store.state(context.domain, str(arguments["entity_id"])),
            code="observation", execution=context.mode,
        )
    elif spec.role == "preflight":
        value = store.validate(
            context.domain,
            str(arguments["entity_id"]),
            str(arguments["desired_value"]),
        )
        response = _provider_result(
            value,
            code="preflight_valid" if value.get("valid") else "preflight_rejected",
            execution=context.mode,
        )
    elif spec.role == "restore":
        if context.mode in {"l0_runtime", "safe_stop"}:
            # Compensation belongs to the Runtime transaction.  Keeping the
            # Tool visible preserves both-arm schema parity, but the Agent may
            # not perform an extra Provider write after a pre-write rejection
            # or after Runtime has already reached a terminal outcome.
            response = _result(
                ok=False,
                code=(
                    "translation_unqualified_safe_stop"
                    if context.mode == "safe_stop" else
                    "runtime_managed_compensation"
                ),
                terminal="rejected" if context.mode == "safe_stop" else None,
                execution=context.mode,
            )
        elif not context.approved:
            response = _result(
                ok=False, code="approval_denied", execution=context.mode,
            )
        else:
            try:
                value = store.restore(
                    context.domain,
                    str(arguments["entity_id"]),
                    dict(arguments["approved_preflight"]),
                )
                response = _provider_result(
                    value,
                    code="restored" if value.get("ok") else "restore_failed",
                    execution=context.mode,
                )
            except Exception as failure:  # provider failure becomes Agent-visible evidence
                response = _result(
                    ok=False, code="compensation_failed",
                    error=f"{type(failure).__name__}: {failure}",
                    execution=context.mode,
                )
    elif context.mode == "safe_stop":
        response = _result(
            ok=False,
            code="translation_unqualified_safe_stop",
            terminal="rejected",
            execution="safe_stop",
        )
    elif context.mode == "l0_runtime":
        with effect_lab_runtime_registration():
            response = await _runtime_effect(
                context=context,
                store=store,
                journal_path=Path(journal_path),
                arguments=arguments,
            )
    elif not context.approved:
        response = _result(
            ok=False, code="approval_denied", execution="l1_native",
        )
    else:
        try:
            value = store.apply(
                context.domain,
                str(arguments["entity_id"]),
                str(arguments["desired_value"]),
                int(arguments["expected_revision"]),
                str(arguments["change_id"]),
                str(arguments["reason"]),
            )
            response = _provider_result(
                value,
                code="effect_accepted" if value.get("ok") else "effect_rejected",
                execution="l1_native",
            )
        except OutcomeIndeterminateError as failure:
            response = _result(
                ok=False, code="outcome_unknown", error=str(failure),
                execution="l1_native",
            )
        except Exception as failure:
            # Do not reveal whether the provider failed before or after send.
            response = _result(
                ok=False, code="outcome_unknown",
                error=f"Provider transport failed: {type(failure).__name__}",
                execution="l1_native",
            )
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    response["adapterLatencyMs"] = elapsed
    _append_trace(Path(trace_path), {
        "schema": "effect-runtime.io/harness-tool-trace/v1",
        "caseId": context.case_id,
        "sessionId": context.session_id,
        "harness": context.harness,
        "mode": context.mode,
        "tool": tool_name,
        "arguments": arguments,
        "argumentDigest": sha256_json(arguments),
        "result": response,
        "latencyMs": elapsed,
    })
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--arguments", required=True)
    args = parser.parse_args(argv)
    try:
        context = HarnessToolContext.load(args.context)
        arguments = json.loads(args.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        result = asyncio.run(execute_harness_tool(
            context=context,
            store_path=args.store,
            journal_path=args.journal,
            trace_path=args.trace,
            tool_name=args.tool,
            arguments=arguments,
        ))
    except Exception as failure:
        result = _result(
            ok=False,
            code="adapter_failure",
            error=f"{type(failure).__name__}: {failure}",
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "HarnessToolContext", "Mode", "TOOL_RESULT_SCHEMA",
    "execute_harness_tool",
]
