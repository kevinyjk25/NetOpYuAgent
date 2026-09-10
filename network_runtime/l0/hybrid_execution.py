"""Bounded mixed scheduling; strict regions always use the original L0 engine.

Callbacks are trusted host registrations, never loaded from a Skill or model.
Late callbacks may still finish externally; their results cannot revive a run.
No Effect invocation, activation API, provider discovery or script execution.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from network_runtime.access import ObservationAccessContext, ObservationPolicy
from network_runtime.contracts import sha256_json

from .flow import HostFlowConsent, parse_flow, run_read_flow
from .hybrid import CandidateAdmission, GovernedHybridFlow, ReasoningTask, StrictRegion, qualify_hybrid
from .read_execution import HostReadBinding
from .structured_bindings import materialize_binding
from .structured_schema import DataBindingError, snapshot_json, validate_data


@dataclass(frozen=True)
class ReasoningReply:
    candidate: Any
    model: str
    configuration_digest: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class HostReasoningBinding:
    model: str
    configuration_digest: str
    invoke: Callable[[dict], ReasoningReply]
    required_scopes: frozenset[str] = frozenset({"reasoning:invoke"})


@dataclass(frozen=True)
class HostCandidateGate:
    policy_digest: str
    validate: Callable[[Any, Any], bool]
    allowed_region_digests: frozenset[str]
    required_scopes: frozenset[str] = frozenset({"candidate:admit"})


@dataclass(frozen=True)
class HostHybridConsent:
    graph_digest: str
    arguments_digest: str
    context_digest: str


def context_digest(context):
    return sha256_json({"subject": context.subject_id, "roles": sorted(context.roles), "scopes": sorted(context.scopes),
        "purpose": context.purpose, "clearance": context.clearance.value, "authenticated": context.authenticated,
        "session": context.session_id, "implicit": context.implicit_local_context})


def run_hybrid(proposal: GovernedHybridFlow, arguments: dict, *, reads: Mapping, read_bindings: Mapping,
               reasoners: Mapping[str, HostReasoningBinding], gates: Mapping[str, HostCandidateGate],
               context: ObservationAccessContext, consent: HostHybridConsent) -> dict:
    packet = qualify_hybrid(proposal, reads)
    proposal = GovernedHybridFlow.model_validate(packet["proposal"])
    arguments = validate_data(proposal.input_schema, arguments)
    if (not isinstance(context, ObservationAccessContext) or context.authenticated is not True or context.implicit_local_context is not False
            or not context.subject_id.strip() or not context.purpose.strip()):
        raise PermissionError("mixed flow requires an explicit authenticated host context")
    context = replace(context, roles=frozenset(context.roles), scopes=frozenset(context.scopes))
    if consent != HostHybridConsent(packet["graphDigest"], sha256_json(arguments), context_digest(context)):
        raise PermissionError("host consent must bind graph, arguments and context")
    nodes = {n.id: n for n in proposal.nodes}
    reads, read_bindings, reasoners, gates = dict(reads), dict(read_bindings), dict(reasoners), dict(gates)

    # Verify every declared host/model/gate before the first callback; a graph
    # cannot run a prefix before discovering that a later binding is unapproved.
    for key in packet["order"]:
        node = nodes[key]
        if isinstance(node, ReasoningTask):
            binding = reasoners.get(node.binding_id)
            if (not isinstance(binding, HostReasoningBinding) or binding.model != node.model
                    or binding.configuration_digest != node.configuration_digest
                    or not binding.required_scopes <= context.scopes):
                raise PermissionError("reasoning model/configuration/scope is not host-bound")
        elif isinstance(node, CandidateAdmission):
            gate = gates.get(node.gate_id)
            if (not isinstance(gate, HostCandidateGate) or gate.policy_digest != node.policy_digest
                    or not gate.required_scopes <= context.scopes):
                raise PermissionError("candidate policy/scope is not host-bound")
        elif isinstance(node, StrictRegion):
            for contract in packet["regions"][key]["readContracts"]:
                registered = read_bindings.get(contract)
                if (not isinstance(registered, HostReadBinding) or registered.approved_contract_hash != contract
                        or registered.required_scopes != frozenset(reads[contract].spec.access.required_scopes)
                        or not registered.required_scopes <= context.scopes
                        or "system" in context.roles or "*" in context.scopes
                        or not ObservationPolicy().authorize(registered.capability, {}, context).allowed):
                    raise PermissionError("all strict-region read contracts need host bindings")
            for source in packet["inputBindings"][key]["requiredSources"]:
                if packet["roles"][source] == "admitted_candidate":
                    gate = gates[nodes[source].gate_id]
                    if packet["regions"][key]["flowDigest"] not in gate.allowed_region_digests:
                        raise PermissionError("candidate gate does not authorize arguments for this exact region")

    values, trace, completed, running = {"input": arguments}, [], set(), {}
    observed_started, model_calls, current_node = {}, 0, None
    closed = threading.Event()
    began = time.monotonic()
    deadline = began + proposal.timeout_seconds
    executor = ThreadPoolExecutor(max_workers=proposal.max_parallel, thread_name_prefix="ensuredskill-mixed")

    def fresh(key):
        now = time.monotonic()
        for parent in packet["ancestors"][key]:
            if parent not in observed_started:
                continue
            region = nodes[parent]
            limits = [region.flow.max_read_age_seconds]
            limits.extend(read_bindings[c].capability.freshness_limit_seconds for c in packet["regions"][parent]["readContracts"])
            if not 0 <= now - observed_started[parent] <= min(limits):
                raise DataBindingError("hybrid_evidence_expired", "/nodes/" + key, "source/control read snapshot expired")

    def guarded_read(binding):
        def invoke(args):
            if closed.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("mixed run no longer admits read calls")
            return binding.observe(args)
        return replace(binding, observe=invoke)

    def invoke(key, supplied, candidate, ages):
        node = nodes[key]
        if closed.is_set():
            raise RuntimeError("closed mixed run")
        if isinstance(node, StrictRegion):
            region = packet["regions"][key]
            report = run_read_flow(parse_flow(region["proposal"]), supplied, reads=reads, effects={},
                bindings={c: guarded_read(read_bindings[c]) for c in region["readContracts"]}, context=context,
                consent=HostFlowConsent(region["flowDigest"], sha256_json(supplied)))
            if report["status"] not in {"read_path_completed", "needs_l1"}:
                return None, {"kind": node.kind, "status": "region_blocked", "regionReport": report}
            value = {"observations": {r["node"]: r["receipt"]["payload"] for r in report["trace"] if r["kind"] == "read"},
                     "outcome": report["status"]}
            return value, {"kind": node.kind, "status": "succeeded", "regionReport": report}
        if isinstance(node, ReasoningTask):
            request = {"graphDigest": packet["graphDigest"], "nodeId": key, "taskDigest": proposal.task_digest,
                "instructions": node.instructions, "inputs": supplied, "outputSchema": node.output_schema,
                "model": node.model, "configurationDigest": node.configuration_digest,
                "maxOutputTokens": node.max_output_tokens, "tools": [],
                "evidencePolicy": "historical_snapshot_analysis_not_current_action_evidence", "observationAgesAtStartMs": ages,
                "outputRole": "model_candidate", "runtimeAuthorityGranted": False}
            if len(json.dumps(request, ensure_ascii=False).encode()) > node.max_input_bytes:
                raise DataBindingError("reasoning_input_budget", "/nodes/" + key, "projected request exceeds frozen budget")
            reply = reasoners[node.binding_id].invoke(snapshot_json(request))
            if (not isinstance(reply, ReasoningReply) or reply.model != node.model
                    or reply.configuration_digest != node.configuration_digest):
                raise ValueError("reasoning reply model/configuration drift")
            for count in (reply.input_tokens, reply.output_tokens):
                if count is not None and (type(count) is not int or count < 0):
                    raise ValueError("invalid token receipt")
            if reply.output_tokens is not None and reply.output_tokens > node.max_output_tokens:
                raise ValueError("reasoning output token budget exceeded")
            value = validate_data(node.output_schema, reply.candidate)
            if len(json.dumps(value, ensure_ascii=False).encode()) > node.max_output_bytes:
                raise ValueError("reasoning output byte budget exceeded")
            return value, {"kind": node.kind, "status": "succeeded", "role": "model_candidate",
                "requestDigest": sha256_json(request), "model": reply.model,
                "inputTokens": reply.input_tokens, "outputTokens": reply.output_tokens,
                "semanticCorrectnessProven": False}
        if isinstance(node, CandidateAdmission):
            if gates[node.gate_id].validate(snapshot_json(candidate), snapshot_json(supplied)) is not True:
                raise PermissionError("independent candidate policy rejected")
            return candidate, {"kind": node.kind, "status": "succeeded", "role": "admitted_candidate",
                "policyDigest": node.policy_digest, "candidateDigest": sha256_json(candidate),
                "evidenceDigest": sha256_json(supplied), "observedFact": False, "effectAuthorized": False}
        return supplied, {"kind": node.kind, "status": "succeeded", "joinRule": "all_succeeded"}

    def finish(status, blocked=None):
        closed.set()
        for future in running:
            future.cancel()
        body = {"status": status, "graphDigest": packet["graphDigest"], "argumentsDigest": sha256_json(arguments),
            "trace": trace, "completedNodes": sorted(completed), "modelCallsReserved": model_calls,
            "pendingResultsIgnored": sorted(info[0] for info in running.values()),
            "providerCancellationProven": False, "effectExecuted": False, "runtimeAuthorityGranted": False,
            "wholeSkillCorrectnessProven": False, "modelOutputsAreFacts": False}
        if blocked is not None:
            body["blocked"] = blocked
        if status == "governed_graph_completed":
            body["outputs"] = {k: {"role": packet["roles"][k], "value": values[k]} for k in proposal.outputs}
            body["claimBoundary"] = "Graph conformance only; model outputs remain candidates, not verified business success."
        return {**body, "reportDigest": sha256_json(body)}

    try:
        while len(completed) < len(nodes):
            now = time.monotonic()
            if now < began or now >= deadline:
                return finish("blocked", {"code": "graph_deadline"})
            active = {info[0] for info in running.values()}
            for key in packet["order"]:
                if len(running) >= proposal.max_parallel:
                    break
                node = nodes[key]
                if key in completed or key in active or not set(node.depends_on) <= completed:
                    continue
                current_node = key
                # Analysis and joins may describe old snapshots, never upgrade
                # them to fresh action evidence. Strict operations/admission
                # still check every source/control observation ancestor.
                if node.kind not in {"reason", "join"}:
                    fresh(key)
                if isinstance(node, StrictRegion):
                    pending = [p for p in packet["ancestors"][key] if isinstance(nodes[p], StrictRegion)
                        and values[p]["outcome"] == "needs_l1"]
                    for parent in pending:
                        bridged = any(isinstance(nodes[g], CandidateAdmission) and parent in packet["ancestors"][nodes[g].candidate]
                            and packet["regions"][key]["flowDigest"] in gates[nodes[g].gate_id].allowed_region_digests
                            for g in packet["ancestors"][key])
                        if not bridged:
                            return finish("blocked", {"code": "unresolved_region_handoff", "node": key, "sourceRegion": parent})
                plan = packet["inputBindings"][key]
                supplied = materialize_binding(plan, {s: values[s] for s in plan["requiredSources"]})["arguments"]
                if isinstance(node, ReasoningTask):
                    model_calls += 1
                    if model_calls > proposal.max_model_calls:
                        return finish("blocked", {"code": "model_call_budget"})
                started = time.monotonic()
                until = min(deadline, started + node.timeout_seconds) if isinstance(node, ReasoningTask) else deadline
                candidate = copy.deepcopy(values[node.candidate]) if isinstance(node, CandidateAdmission) else None
                ages = {p: (started - observed_started[p]) * 1000 for p in packet["ancestors"][key] if p in observed_started}
                future = executor.submit(invoke, key, snapshot_json(supplied), candidate, ages)
                running[future] = (key, started, until)
                trace.append({"node": key, "event": "started", "inputsDigest": sha256_json(supplied)})
            if not running:
                return finish("blocked", {"code": "no_runnable_node"})
            wait(running, timeout=max(0, min(info[2] for info in running.values()) - time.monotonic()), return_when=FIRST_COMPLETED)
            now = time.monotonic()
            if any(now < started or now >= until for _, started, until in running.values()):
                return finish("blocked", {"code": "node_timeout_late_results_ignored"})
            for future, (key, started, _) in list(running.items()):
                if not future.done():
                    continue
                current_node = key
                del running[future]
                value, event = future.result()
                trace.append({"node": key, "event": "finished", "elapsedMs": (time.monotonic() - started) * 1000, **event})
                if event["status"] != "succeeded":
                    return finish("blocked", {"code": "strict_region_blocked", "node": key})
                if isinstance(nodes[key], CandidateAdmission):
                    fresh(key)  # do not accept a slow verifier's stale evidence
                values[key] = validate_data(packet["outputSchemas"][key], value)
                if isinstance(nodes[key], StrictRegion):
                    observed_started[key] = started
                completed.add(key)
        return finish("governed_graph_completed")
    except Exception as error:
        code = error.code if isinstance(error, DataBindingError) else type(error).__name__
        return finish("blocked", {"code": code, "node": current_node,
            "pointer": error.pointer if isinstance(error, DataBindingError) else None,
            "detail": "Host/model/binding check failed; provider details redacted."})
    finally:
        closed.set()
        executor.shutdown(wait=False, cancel_futures=True)
