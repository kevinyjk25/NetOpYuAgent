"""Production L1 decision service with shadow-first rollout semantics."""

from __future__ import annotations

import os
import re
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from network_runtime.contracts import sha256_json

from .catalog import (
    CandidateRetriever,
    CatalogPolicy,
    CapabilityCard,
    build_catalog,
    candidate_digest,
    catalog_digest,
)
from .client import OpenAISelectionClient, SelectionClient, SelectionProtocolError
from .contracts import (
    L1Decision,
    L1DecisionAction,
    L1DecisionEnvelope,
    L1DecisionEvidence,
)
from .policies import GroundingPolicy, GroundingResult, GuardPolicy, GuardVerdict
from .store import DecisionStore


_CANDIDATE_TOOL = re.compile(r"select_candidate_(\d{2})\Z")
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
_POLICY_ROOT = _PACKAGE_ROOT / "policies"
_TERMINAL_TOOLS = {
    "refuse_l1_request": L1DecisionAction.REFUSE,
    "reject_l1_out_of_scope": L1DecisionAction.OUT_OF_SCOPE,
}


def _combined_policy_digest(
    catalog_policy: CatalogPolicy,
    guard_policy: GuardPolicy,
    grounding_policy: GroundingPolicy,
) -> str:
    return sha256_json({
        "catalog": catalog_policy.digest,
        "guard": guard_policy.digest,
        "grounding": grounding_policy.digest,
    })


class L1DecisionPlane:
    """Narrow natural language to a verified proposal without executing it."""

    def __init__(
        self,
        *,
        selection_client: SelectionClient,
        store: DecisionStore,
        catalog_policy: CatalogPolicy | None = None,
        guard_policy: GuardPolicy | None = None,
        grounding_policy: GroundingPolicy | None = None,
        repair_limit: int = 1,
    ) -> None:
        if repair_limit not in {0, 1}:
            raise ValueError("production L1 repair limit must be zero or one")
        self.selection_client = selection_client
        self.store = store
        self.catalog_policy = catalog_policy or CatalogPolicy(
            _POLICY_ROOT / "catalog.yaml",
        )
        self.guard_policy = guard_policy or GuardPolicy(_POLICY_ROOT / "guard.yaml")
        self.grounding_policy = grounding_policy or GroundingPolicy(
            _POLICY_ROOT / "grounding.yaml",
        )
        self.repair_limit = repair_limit

    def _compile_attempt(
        self,
        tool_name: str,
        supplied: dict[str, Any],
        candidates: tuple[CapabilityCard, ...],
        prompt: str,
    ) -> tuple[L1Decision, int | None, GroundingResult]:
        terminal_action = _TERMINAL_TOOLS.get(tool_name)
        if terminal_action is not None:
            if supplied:
                raise ValueError("terminal L1 proposal cannot carry arguments")
            return (
                L1Decision(
                    action=terminal_action,
                    confidence=0.55,
                    reason_code=(
                        "model_refusal"
                        if terminal_action == L1DecisionAction.REFUSE
                        else "model_out_of_scope"
                    ),
                ),
                None,
                GroundingResult({}, (), ()),
            )
        match = _CANDIDATE_TOOL.fullmatch(tool_name)
        if match is None:
            raise ValueError("model Tool escapes candidate contract")
        index = int(match.group(1))
        if index >= len(candidates):
            raise ValueError("model candidate index escapes candidate contract")
        selected = candidates[index]
        if not set(supplied) <= set(selected.parameter_schemas):
            raise ValueError("model arguments escape candidate Schema")
        grounding = self.grounding_policy.apply(
            prompt, supplied, set(selected.parameter_schemas),
        )
        missing = tuple(
            field
            for field in selected.required_parameters
            if field not in grounding.arguments
        )
        if missing:
            decision = L1Decision(
                action=L1DecisionAction.CLARIFY,
                target=selected.target,
                arguments=grounding.arguments,
                missing_fields=missing,
                confidence=0.55,
                reason_code=f"candidate_schema_{index:02d}_missing",
            )
        else:
            decision = L1Decision(
                action=(
                    L1DecisionAction.SELECT_SKILL
                    if selected.kind == "skill"
                    else L1DecisionAction.SELECT_TOOL
                ),
                target=selected.target,
                arguments=grounding.arguments,
                workflow=selected.workflow_hint,
                confidence=0.55,
                reason_code=f"candidate_schema_{index:02d}",
            )
        return decision, index, grounding

    async def decide(
        self,
        *,
        profile: str,
        session_id: str,
        harness: str,
        prompt: str,
        tool_declarations: list[dict[str, Any]],
        mode: str = "shadow",
    ) -> L1DecisionEnvelope:
        if mode not in {"shadow", "canary", "enforced"}:
            raise ValueError("production L1 mode must be shadow, canary, or enforced")
        started = time.perf_counter()
        catalog = build_catalog(profile, tool_declarations, self.catalog_policy)
        candidates = CandidateRetriever(catalog, self.catalog_policy).retrieve(prompt)
        if not candidates:
            raise RuntimeError("production L1 candidate catalog is empty")
        verdict = self.guard_policy.classify(prompt)
        decision: L1Decision | None = None
        selected_index: int | None = None
        grounding = GroundingResult({}, (), ())
        attempts = 0
        input_tokens = 0
        output_tokens = 0
        token_usage_complete = True
        protocol_valid = True
        status = "decided"
        repair_reason: str | None = None
        attempt_error_types: list[str] = []
        if verdict.action in {"refuse", "out_of_scope"}:
            decision = L1Decision(
                action=(
                    L1DecisionAction.REFUSE
                    if verdict.action == "refuse"
                    else L1DecisionAction.OUT_OF_SCOPE
                ),
                confidence=1.0,
                reason_code=verdict.reason_code,
            )
            status = "policy_terminal"
        else:
            for attempt_index in range(self.repair_limit + 1):
                attempts += 1
                try:
                    proposal = await self.selection_client.select(
                        prompt,
                        candidates,
                        candidate_digest(candidates),
                        repair_reason=repair_reason,
                    )
                    input_tokens += proposal.input_tokens
                    output_tokens += proposal.output_tokens
                    decision, selected_index, grounding = self._compile_attempt(
                        proposal.tool_name, proposal.arguments, candidates, prompt,
                    )
                    break
                except SelectionProtocolError as error:
                    input_tokens += error.input_tokens
                    output_tokens += error.output_tokens
                    token_usage_complete = (
                        token_usage_complete and error.usage_complete
                    )
                    attempt_error_types.append(type(error).__name__)
                    repair_reason = str(error)[:160]
                    if attempt_index >= self.repair_limit:
                        protocol_valid = False
                except (httpx.HTTPError, TypeError, ValueError) as error:
                    token_usage_complete = False
                    attempt_error_types.append(type(error).__name__)
                    repair_reason = type(error).__name__ + ": " + str(error)[:160]
                    if attempt_index >= self.repair_limit:
                        protocol_valid = False
            if decision is None:
                status = "protocol_failure"
        duration_ms = round((time.perf_counter() - started) * 1_000, 3)
        evidence = L1DecisionEvidence(
            prompt_digest=sha256_json({"direct_user_text": prompt}),
            catalog_digest=catalog_digest(catalog),
            candidate_digest=candidate_digest(candidates),
            policy_digest=_combined_policy_digest(
                self.catalog_policy, self.guard_policy, self.grounding_policy,
            ),
            model=self.selection_client.model if attempts else None,
            model_attempts=attempts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_usage_complete=token_usage_complete,
            selected_candidate_index=selected_index,
            candidate_ids=tuple(item.identity for item in candidates),
            dropped_argument_fields=grounding.dropped_fields,
            normalized_argument_fields=grounding.normalized_fields,
            attempt_error_types=tuple(attempt_error_types),
            guard_action=verdict.action,
            guard_reason=verdict.reason_code,
            protocol_valid=protocol_valid,
            duration_ms=duration_ms,
        )
        evidence_payload = evidence.model_dump(by_alias=True, mode="json")
        envelope = L1DecisionEnvelope(
            decision_id=f"l1-{uuid4()}",
            mode=mode,
            profile=profile,
            session_id=session_id,
            harness=harness,
            status=status,
            decision=decision,
            evidence=evidence,
            decision_digest=decision.digest if decision is not None else None,
            evidence_digest=sha256_json(evidence_payload),
        )
        self.store.record(envelope)
        return envelope


def _decision_store_path() -> Path:
    raw = os.getenv("NETOPYU_L1_DECISION_STORE")
    return Path(raw) if raw else _PROJECT_ROOT / "data" / "l1_decisions.sqlite"


def _selection_client(model: str) -> OpenAISelectionClient:
    base_url = os.getenv("NETOPYU_L1_DECISION_BASE_URL", "http://127.0.0.1:11434/v1")
    parsed = urlparse(base_url)
    allow_remote = os.getenv(
        "NETOPYU_L1_DECISION_ALLOW_REMOTE", "0",
    ).strip().casefold() in {"1", "true", "yes", "on"}
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or (
            parsed.hostname.casefold() not in {"127.0.0.1", "localhost", "::1"}
            and not allow_remote
        )
    ):
        raise ValueError(
            "production L1 model endpoint must be loopback unless "
            "NETOPYU_L1_DECISION_ALLOW_REMOTE=1 is explicitly configured"
        )
    api_key = os.getenv("NETOPYU_L1_DECISION_API_KEY")
    timeout_seconds = float(os.getenv("NETOPYU_L1_DECISION_TIMEOUT_SECONDS", "120"))
    return OpenAISelectionClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


async def decide_shadow(
    *,
    profile: str,
    session_id: str,
    harness: str,
    prompt: str,
    tool_declarations: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    repair_limit = int(os.getenv("NETOPYU_L1_DECISION_REPAIR_LIMIT", "1"))
    plane = L1DecisionPlane(
        selection_client=_selection_client(model),
        store=DecisionStore(_decision_store_path()),
        repair_limit=repair_limit,
    )
    envelope = await plane.decide(
        profile=profile,
        session_id=session_id,
        harness=harness,
        prompt=prompt,
        tool_declarations=tool_declarations,
        mode="shadow",
    )
    return envelope.model_dump(by_alias=True, mode="json")


def recent_decisions(
    *, limit: int = 20, session_id: str | None = None,
) -> dict[str, Any]:
    rows = DecisionStore(_decision_store_path()).recent(
        limit=limit, session_id=session_id,
    )
    return {
        "apiVersion": "netopyu.io/l1-decision-history/v1",
        "privacy": "prompt_digest_and_argument_keys_only",
        "count": len(rows),
        "decisions": rows,
    }


def observe_decision(
    *,
    decision_id: str,
    session_id: str,
    observed_kind: str,
    observed_target: str,
    observed_arguments: dict[str, Any],
) -> dict[str, Any]:
    return DecisionStore(_decision_store_path()).observe(
        decision_id=decision_id,
        session_id=session_id,
        observed_kind=observed_kind,
        observed_target=observed_target,
        observed_arguments=observed_arguments,
    )


def close_decision(
    *, decision_id: str, session_id: str, reason: str,
) -> dict[str, Any]:
    return DecisionStore(_decision_store_path()).close(
        decision_id=decision_id,
        session_id=session_id,
        reason=reason,
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def decision_metrics(*, limit: int = 500) -> dict[str, Any]:
    rows = DecisionStore(_decision_store_path()).recent(limit=limit)
    envelopes = [row["envelope"] for row in rows]
    observations = [row["observation"] for row in rows if row["observation"]]
    statuses: dict[str, int] = {}
    lifecycle_statuses: dict[str, int] = {}
    lifecycle_reasons: dict[str, int] = {}
    for envelope in envelopes:
        status = str(envelope["status"])
        statuses[status] = statuses.get(status, 0) + 1
    for row in rows:
        lifecycle = str(row["lifecycle_status"])
        lifecycle_statuses[lifecycle] = lifecycle_statuses.get(lifecycle, 0) + 1
        reason = row["lifecycle_reason"]
        if reason is not None:
            lifecycle_reasons[str(reason)] = lifecycle_reasons.get(str(reason), 0) + 1
    durations = [float(item["evidence"]["duration_ms"]) for item in envelopes]
    model_attempts = [int(item["evidence"]["model_attempts"]) for item in envelopes]
    input_tokens = [int(item["evidence"]["input_tokens"]) for item in envelopes]
    output_tokens = [int(item["evidence"]["output_tokens"]) for item in envelopes]
    usage_complete = [bool(item["evidence"]["token_usage_complete"]) for item in envelopes]
    target_matches = sum(bool(item["target_match"]) for item in observations)
    argument_observations = [
        item for item in observations if item["arguments_exact"] is not None
    ]
    argument_matches = sum(
        bool(item["arguments_exact"]) for item in argument_observations
    )
    safety_escapes = sum(bool(item["safety_escape"]) for item in observations)
    total = len(envelopes)
    observed_total = len(observations)
    return {
        "apiVersion": "netopyu.io/l1-decision-metrics/v1",
        "scope": "local_shadow_observations",
        "warning": "These are bounded local observations, not production success probabilities.",
        "decisions": total,
        "observed_routes": observed_total,
        "status_counts": statuses,
        "lifecycle_status_counts": lifecycle_statuses,
        "lifecycle_reason_counts": lifecycle_reasons,
        "unobserved_decisions": total - observed_total,
        "protocol_success_rate": (
            round((total - statuses.get("protocol_failure", 0)) / total, 4)
            if total else None
        ),
        "routing_agreement_rate": (
            round(target_matches / observed_total, 4) if observed_total else None
        ),
        "direct_tool_argument_exact_rate": (
            round(argument_matches / len(argument_observations), 4)
            if argument_observations else None
        ),
        "safety_escape_count": safety_escapes,
        "repair_rate": (
            round(sum(value > 1 for value in model_attempts) / total, 4)
            if total else None
        ),
        "average_model_attempts": (
            round(sum(model_attempts) / total, 4) if total else None
        ),
        "reported_tokens": {
            "input": sum(input_tokens),
            "output": sum(output_tokens),
            "usage_complete_rate": (
                round(sum(usage_complete) / total, 4) if total else None
            ),
        },
        "grounding_dropped_field_count": sum(
            len(item["evidence"]["dropped_argument_fields"]) for item in envelopes
        ),
        "decision_latency_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
        },
    }


__all__ = [
    "L1DecisionPlane",
    "close_decision",
    "decide_shadow",
    "decision_metrics",
    "observe_decision",
    "recent_decisions",
]
