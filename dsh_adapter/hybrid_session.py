"""One host-configured read-prefix -> original Runtime -> bounded L1 session.

Prototype compiler/transport are reused, not a new reviewer or executor. Only
the operator's local profile supplies tools/resources. Agent inputs cannot set
permissions, resource policy, model, output paths, approval or task scores.
"""
from __future__ import annotations

import os
import re
import uuid
import hashlib
import fcntl
import json
from pathlib import Path
from contextlib import contextmanager

from skill_authoring import compiler as author
from skill_authoring import delivery
from skill_authoring import artifact_repair
from skill_authoring import isolated_compiler
from skill_authoring.contracts import budget, seal
from skill_authoring.artifacts import read_json, write_artifacts
from skill_authoring.artifact_checks import inspect_candidate
from skill_authoring.local_execution import run_prefix, draft_from_evidence, bindings_for, local_context
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import DataBindingError, snapshot_json, validate_data
from network_runtime.l0.read_execution import execute_host_read
from network_runtime.l0.structured_reads import parse_read_contract

PROFILE = "netopyu.io/local-hybrid-host/v1"
COMPACT_HOST_PROFILE = "netopyu.io/local-hybrid-host/v2"
CHOICE_HOST_PROFILE = "netopyu.io/local-hybrid-host/v3"
TASK_HOST_PROFILE = "netopyu.io/local-hybrid-host/v4"
ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    paths = [Path(__file__)]
    for name in ("skill_authoring", "network_runtime"):
        paths.extend((ROOT / name).rglob("*.py"))
    return {str(p.relative_to(ROOT)): "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def _host():
    path = os.environ.get("NETOPYU_HYBRID_HOST_PROFILE")
    if not path:
        raise PermissionError("operator must configure NETOPYU_HYBRID_HOST_PROFILE; disabled by default")
    value = read_json(Path(path))
    required = {"apiVersion", "enabled", "packet", "resources"}
    if (not required <= set(value) or set(value) - required - {"requiredReads", "compilerMode", "artifactRepair"}
            or ("artifactRepair" in value and (value.get("apiVersion") != TASK_HOST_PROFILE or value["artifactRepair"] is not True))
            or ("requiredReads" in value and value.get("apiVersion") != TASK_HOST_PROFILE)
            or ("compilerMode" in value and (value.get("apiVersion") != TASK_HOST_PROFILE or value["compilerMode"] != "isolated"))
            or value["apiVersion"] not in {PROFILE, COMPACT_HOST_PROFILE, CHOICE_HOST_PROFILE, TASK_HOST_PROFILE} or value["enabled"] is not True):
        raise PermissionError("explicit enabled local read-only host profile required")
    packet = author.validate_packet(value["packet"])
    resources = snapshot_json(value["resources"])
    if not isinstance(resources, dict) or set(resources) - packet["reads"].keys():
        raise ValueError("resources must belong to the host read catalog")
    # These are inert synthetic payloads, never executable provider definitions.
    for name, resource in resources.items():
        entries = resource["resources"] if isinstance(resource, dict) and set(resource) == {"resources"} else [resource]
        contract = packet["reads"][name]["spec"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("nonempty explicit resource inventory required")
        seen = set()
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("resource entry must be [exact arguments, observed payload]")
            validate_data(contract["inputSchema"], entry[0])
            validate_data(contract["outputSchema"], entry[1])
            key = sha256_json(entry[0])
            if key in seen:
                raise ValueError("ambiguous resource arguments")
            seen.add(key)
    # Operator-declared prerequisites only, never inferred from prose, a model
    # confidence score or every available resource. They confer no new access.
    requirements = value.get("requiredReads", [])
    if not isinstance(requirements, list) or len(requirements) > 8:
        raise ValueError("at most eight explicit host read prerequisites")
    seen = set()
    for item in requirements:
        if (not isinstance(item, dict) or set(item) != {"tool", "arguments"}
                or not isinstance(item["tool"], str) or item["tool"] not in packet["reads"]):
            raise ValueError("host prerequisite must name a declared read and concrete arguments")
        validate_data(packet["reads"][item["tool"]]["spec"]["inputSchema"], item["arguments"])
        resource = resources.get(item["tool"])
        entries = resource["resources"] if isinstance(resource, dict) else [resource]
        if not any(entry and sha256_json(entry[0]) == sha256_json(item["arguments"]) for entry in entries):
            raise PermissionError("host prerequisite outside existing exact resource ACL")
        key = sha256_json(item)
        if key in seen:
            raise ValueError("duplicate host prerequisite")
        seen.add(key)
    return value


def _root():
    root = Path(os.environ.get("NETOPYU_HYBRID_SESSIONS_DIR", str(ROOT / "data/hybrid-sessions")))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def _read_session(session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise ValueError("invalid session id")
    folder = _root() / session_id
    frozen = read_json(folder / "request.json")
    if frozen != seal({k: v for k, v in frozen.items() if k != "reportDigest"}):
        raise ValueError("session request drift")
    return folder, frozen


@contextmanager
def _locked(folder):
    # Read collection and draft freezing cannot race across CLI/worker processes.
    with (folder / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _sealed(path):
    value = read_json(path)
    if value != seal({k: v for k, v in value.items() if k != "reportDigest"}):
        raise ValueError("session evidence seal drift")
    return value


def _delivery_origins(frozen, *, task_only=False):
    pages = author.pages_for(frozen["packet"])
    return {"task": frozen["packet"]["task"], **({} if task_only else {
        key: pages[key]["text"] for key in frozen["visible"]})}


def _compile_delivery(frozen, proposal, *, task_only=False):
    if frozen.get("deliveryProtocol") == delivery.TASK_PROFILE:
        return delivery.compile_task(proposal, _delivery_origins(frozen, task_only=task_only))
    if frozen.get("deliveryProtocol") == delivery.CHOICE_PROFILE:
        return delivery.compile_selection(proposal, _delivery_origins(frozen, task_only=task_only), single_choice=True)
    compile_fn = delivery.compile_selection if frozen.get("deliveryProtocol") == delivery.COMPACT_PROFILE else delivery.compile_contract
    return compile_fn(proposal, _delivery_origins(frozen, task_only=task_only))


def _with_delivery(folder, report):
    if not (folder / "delivery-contract/report.json").is_file():
        return report
    contract = _delivery_contract(folder)
    action = _delivery_action(folder, report)
    return seal({**{k: v for k, v in report.items() if k != "reportDigest"},
        "deliveryContract": contract, "deliveryResponseSchema": delivery.response_schema(contract),
        "deliveryGuidance": delivery.generation(contract), "deliveryAction": action,
        "evidenceState": _evidence_state(folder),
        **({"next_tool": "netopyu_hybrid_read", "guidance": action["guidance"]}
           if report["route"] == "l1_fallback" and report["runtime"]["status"] == "not_executed" else {})})


def _delivery_action(folder, report):
    if report["route"] == "collecting_evidence":
        action = {"tool": "netopyu_hybrid_draft", "arguments": {"session_id": folder.name},
                "guidance": "For normal generation after evidence collection call draft with ONLY session_id. No response field. Native deliver is not available on this admitted Runtime path."}
    elif report["route"] == "l1_fallback" and report["runtime"]["status"] == "not_executed":
        action = {"tool": "netopyu_hybrid_deliver", "session_id": folder.name,
                "requiredKeys": ["session_id", "response_json"],
                "guidance": "Delivery requirements are frozen. Gather permitted evidence, compose the typed response natively, then call netopyu_hybrid_deliver(session_id,response_json) to validate/render it. This invokes no Runtime model and executes no artifact. Runtime generation through draft is unavailable on this fallback path."}
    else:
        return {"tool": "netopyu_hybrid_inspect", "arguments": {"session_id": folder.name},
                "guidance": "Stopped or unknown execution; no generation or replay."}
    if _sealed(folder / "request.json").get("deliveryProtocol") == delivery.TASK_PROFILE:
        action["incompleteExit"] = {"tool": "netopyu_hybrid_draft",
            "arguments": {"session_id": folder.name, "close_incomplete": True}}
        action["guidance"] += " To explicitly stop as NOT completed instead, call draft(session_id,close_incomplete=true). This admits no answer and invokes no model."
    return action


def _evidence_state(folder, *, closed=False):
    """Host receipt state, not model opinions or proof of sufficient evidence."""
    records = []
    prefix_path = folder / "prefix/summary/report.json"
    if prefix_path.is_file():
        prefix = _sealed(prefix_path)
        if _sealed(folder / "result/report.json")["executionReportDigest"] != prefix["reportDigest"]:
            raise ValueError("prefix evidence state binding drift")
        completed = prefix["execution"]["status"] == "governed_graph_completed"
        for i, call in enumerate(prefix["providerCalls"]):
            records.append({"id": f"prefix-{i}", **call, "recordDigest": prefix["reportDigest"],
                "state": "read_completed" if completed else "call_recorded_outcome_unverified"})
    for i, attempt in enumerate(sorted((folder / "followup").glob("attempt-*"))):
        action = _sealed(attempt / "request.json")
        result = _sealed(attempt / "result/report.json") if (attempt / "result/report.json").is_file() else None
        if result and result["requestDigest"] != action["reportDigest"]:
            raise ValueError("follow-up evidence state binding drift")
        records.append({"id": f"followup-{i}", "tool": action["tool"], "arguments": action["arguments"],
            "recordDigest": result["reportDigest"] if result else action["reportDigest"],
            "state": result["status"] if result else "outcome_unknown"})
    frozen = _sealed(folder / "request.json")
    requirements = []
    for i, item in enumerate(frozen.get("requiredReads", [])):
        matches = [r for r in records if r["state"] == "read_completed" and r["tool"] == item["tool"]
                   and sha256_json(r["arguments"]) == sha256_json(item["arguments"])]
        requirements.append({"id": f"r{i}", **item, "observed": bool(matches),
                             "recordDigests": [r["recordDigest"] for r in matches]})
    snapshot = (_observation_model(frozen["hostDigest"])
                if frozen.get("deliveryProtocol") == delivery.TASK_PROFILE else None)
    return seal({"profile": "host-recorded-read-state/v1", "records": records,
        **({"observationModel": snapshot} if snapshot else {}),
        **({"requiredReads": requirements,
            "requiredReadsComplete": all(r["observed"] for r in requirements) if requirements else None,
            "requirementsMeaning": "Operator-declared exact reads, not inferred task completeness. Undeclared means unknown, not complete."}
           if snapshot else {}),
        "collectionClosed": closed, "completedReads": sum(r["state"] == "read_completed" for r in records),
        "contentSufficiency": "not_assessed", "authorityGranted": False,
        "meaning": "Actual recorded operations only. Reading an index does not prove required data is present; absent entries are not proof data does not exist. Model uncertainties cannot change these states."})


def _observation_model(host_digest):
    """Semantics of THIS host's inert resources, not of arbitrary read tools."""
    return {"kind": "immutable_session_snapshot", "snapshotId": host_digest,
        "identity": "tool_and_exact_validated_arguments", "liveRefreshSupported": False,
        "sameIdentityRead": "no_new_observation_repeat_not_executed",
        "differentIdentityRead": "subject_to_original_access_schema_resource_and_attempt_limits",
        "meaning": "These operator-bound payloads are frozen for this session. Re-reading the same identity cannot reveal later time windows, recovery or changed facts. A different export may contain different evidence; its existence/content is not inferred. New live evidence requires an external acquisition capability not supplied by this snapshot host."}


def _recorded_snapshot_read(folder, request):
    # Only finalized, bound records qualify; rejected/unknown attempts and
    # mentions of paths in source text never create observation identity.
    state = _evidence_state(folder)
    for record in state["records"]:
        if (record["state"] == "read_completed" and record["tool"] == request["tool"]
                and sha256_json(record["arguments"]) == sha256_json(request["arguments"])):
            return {"recordId": record["id"], "recordDigest": record["recordDigest"],
                    "snapshotId": state["observationModel"]["snapshotId"]}
    return None


def _install_delivery(folder, frozen, contract):
    write_artifacts(folder / "delivery-contract", {"report.json": contract,
        "binding.json": seal({"sessionRequestDigest": frozen["reportDigest"], "contractDigest": contract["reportDigest"]})})


def _delivery_contract(folder):
    if not (folder / "delivery-contract/report.json").is_file():
        raise PermissionError("bind delivery requirements with submit before reads or drafting")
    contract = _sealed(folder / "delivery-contract/report.json")
    binding = _sealed(folder / "delivery-contract/binding.json")
    if binding != seal({"sessionRequestDigest": _sealed(folder / "request.json")["reportDigest"],
                        "contractDigest": contract["reportDigest"]}):
        raise ValueError("delivery contract/session binding drift")
    return contract


def prepare(request):
    """Capture exact user task/arguments, return source and translation schema."""
    if set(request) != {"task", "arguments"}:
        raise ValueError("only task and arguments may be submitted by the agent")
    host = _host()
    compact = host["apiVersion"] in {COMPACT_HOST_PROFILE, CHOICE_HOST_PROFILE}
    task_bound = host["apiVersion"] == TASK_HOST_PROFILE
    packet = author.validate_packet({**host["packet"], "task": request["task"]})
    arguments = validate_data(packet["inputSchema"], request["arguments"])
    visible = list(author.pages_for(packet))
    if host.get("compilerMode") == "isolated":
        return _prepare_isolated(host, packet, arguments, visible)
    wire = author.make_request(packet, visible)
    origins = {"task": packet["task"], **{k: v["text"] for k, v in author.pages_for(packet).items() if k in visible}}
    delivery_schema = {"type": "null"} if task_bound else delivery.selection_schema(origins) if compact else delivery.proposal_schema(origins)
    delivery_instruction = delivery.TASK_INSTRUCTION if task_bound else delivery.instruction(compact)
    payload = json.loads(wire["messages"][-1]["content"])
    if task_bound:
        payload["observationModel"] = _observation_model(sha256_json(host))
        payload["requiredReads"] = host.get("requiredReads", [])
        payload["callProtocol"] = "Only plan.reads uses caller/literal bindings. netopyu_hybrid_read uses concrete values matching its tool schema, never a compiler binding expression."
    if compact:
        # Replace full page strings with addressed lossless segments, not an
        # extra duplicate copy or a model-authored summary of the source.
        payload.pop("sourcePages")
        payload["deliverySourceReferences"] = {ref: {"origin": r["origin"], "text": r["text"]}
            for ref, r in delivery.source_references(origins).items()}
        payload["sourceReferenceMeaning"] = "Concatenate each origin's spans in order to recover its original page. Plan evidence still uses task/p000-style origin IDs; delivery requirements use source_ref IDs. No span implies a requirement or authority."
        wire["messages"][0]["content"] = wire["messages"][0]["content"].replace(
            "sourcePages are already available TO YOU NOW;",
            "Original source text is available TO YOU NOW in lossless deliverySourceReferences spans;")
    payload["requiredOutputSchema"] = author.obj({"plan": payload["requiredOutputSchema"], "delivery": delivery_schema})
    wire["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False)
    wire["messages"][0]["content"] += "\n" + delivery_instruction
    session_id = uuid.uuid4().hex
    frozen = seal({"hostDigest": sha256_json(host), "implementation": fingerprint(), "packet": packet, "arguments": arguments,
                   **({"artifactRepair": True} if host.get("artifactRepair") else {}),
                   **({"requiredReads": host.get("requiredReads", [])} if task_bound else {}),
                   "visible": visible, "authorProtocol": author.PROTOCOL,
                   "deliveryProtocol": delivery.TASK_PROFILE if task_bound else delivery.CHOICE_PROFILE if host["apiVersion"] == CHOICE_HOST_PROFILE
                        else delivery.COMPACT_PROFILE if compact else delivery.PROFILE})
    folder = _root() / session_id
    write_artifacts(folder, {"request.json": frozen})
    if not budget(wire)["accepted"]:
        # Source overflow is a translation boundary, not loss of the bounded
        # native read channel. Keep the full original on disk, never truncate it.
        result = _fallback(folder, frozen, "complete_source_exceeds_budget")
        return {**result, "session_id": session_id, "sourceSuppliedToModel": False,
                **({"observationModel": _observation_model(frozen["hostDigest"]), "requiredReads": frozen["requiredReads"]} if task_bound else {}),
                "sourceRetainedOnHost": True, "limitation": "Only scoped diagnosis is available; whole Skill not supplied.",
                "next_tool": "netopyu_hybrid_submit", "deliverySchema": {"type": "null"} if task_bound else delivery.selection_schema({"task": packet["task"]}) if compact else delivery.proposal_schema(["task"]),
                **({"deliverySourceReferences": {ref: {"origin": r["origin"], "text": r["text"]}
                    for ref, r in delivery.source_references({"task": packet["task"]}).items()}} if compact else {}),
                "guidance": delivery_instruction + "\nBind task-only delivery with submit(plan=null, delivery=null for task-bound mode; otherwise use the supplied schema). Then read, compose the response natively and call netopyu_hybrid_deliver(session_id,response_json), not draft. No Runtime model call or whole-Skill compilation is claimed."}
    return {"session_id": session_id, "route": "awaiting_agent_translation",
            "authoring_instruction": wire["messages"][0]["content"], "authoring": json.loads(wire["messages"][-1]["content"]),
            "contract": "Submit read_prefix only. Then collect missing observations with read before calling draft once. Submission does not generate a candidate. Full source/task retained; compilation does not prove meaning.",
            "writeAuthority": False, "taskSuccess": None}


def _prepare_isolated(host, packet, arguments, visible):
    """Compiler owns AST; execution Agent receives source and observations only."""
    folder = _root() / uuid.uuid4().hex
    frozen = seal({"hostDigest": sha256_json(host), "implementation": fingerprint(),
        **({"artifactRepair": True} if host.get("artifactRepair") else {}),
        "packet": packet, "arguments": arguments, "requiredReads": host.get("requiredReads", []),
        "visible": visible, "authorProtocol": author.PROTOCOL, "deliveryProtocol": delivery.TASK_PROFILE,
        "compilerMode": "isolated"})
    write_artifacts(folder, {"request.json": frozen})
    with _locked(folder):
        source_supplied = budget(isolated_compiler.make_request(packet, visible))["accepted"]
        if not source_supplied:
            report = _fallback(folder, frozen, "complete_source_exceeds_budget")
        else:
            # Claim once before the physical call. Crash/timeout stays unknown;
            # public submit cannot inject a replacement AST in this mode.
            write_artifacts(folder / "compiler", {"claim.json": seal({
                "sessionRequestDigest": frozen["reportDigest"], "attemptsAllowed": 1,
                "invocationValuesSupplied": False, "observationsSupplied": False, "toolsAvailable": False})})
            try:
                proposal = isolated_compiler.invoke(packet, visible, folder / "compiler")
            except isolated_compiler.ResponseRejected:
                if sha256_json(_host()) != frozen["hostDigest"] or fingerprint() != frozen["implementation"]:
                    raise PermissionError("host or implementation changed during isolated compilation")
                report = _fallback(folder, frozen, "isolated_compiler_response_rejected")
            except Exception as error:
                write_artifacts(folder / "compiler/unknown", {"report.json": seal({
                    "errorType": type(error).__name__, "retryAllowed": False, "writeAuthority": False})})
                return {"session_id": folder.name, "route": "pending_or_unknown_no_retry", "phase": "compiler",
                    "next_tool": "netopyu_hybrid_inspect", "writeAuthority": False, "taskSuccess": None,
                    "guidance": "Compiler did not establish a completed response. No execution or replacement session; inspect or hand off."}
            else:
                report = _submit({"session_id": folder.name, "plan": proposal, "delivery": None}, folder, frozen)
                if report["route"] == "translation_needs_correction":
                    # Single author call, no autonomous repair/self-review loop.
                    report = _fallback(folder, frozen, "isolated_compiler_proposal_rejected")
        if not (folder / "delivery-contract").exists():
            _install_delivery(folder, frozen, _compile_delivery(frozen, None, task_only=not source_supplied))
        result = _with_delivery(folder, report)
        return {**result, "session_id": folder.name,
            "executionContext": {"originalTask": packet["task"], "arguments": arguments,
                "sourcePages": {k: v["text"] for k, v in author.pages_for(packet).items()} if source_supplied else {},
                "sourceSuppliedToExecutionAgent": source_supplied, "sourceRetainedOnHost": True,
                "meaning": "Inert original Skill and caller request, not observations or authority. Author AST is isolated from this context."}}


def _fallback(folder, frozen, reason, *, compilation=None, execution=None):
    stopped = execution is not None
    report = seal({"route": "l1_fallback", "reason": reason,
        "translation": {"compiled": compilation is not None, "semanticFidelity": "not_assessed"},
        "runtime": {"status": execution["execution"]["status"] if execution else "not_executed",
                    "providerCalls": execution["providerCalls"] if execution else [], "effectCalls": 0},
        "task": {"status": "not_completed", "success": None},
        "handoff": {"originalTask": frozen["packet"]["task"], "arguments": frozen["arguments"],
                    "allowed": ["clarify", "proposal", "human_handoff"] if stopped else ["read_only_diagnosis", "clarify", "proposal", "human_handoff"],
                    "writeAuthority": False},
        "next_tool": "netopyu_hybrid_inspect" if stopped else "netopyu_hybrid_submit", "host_tools": frozen["packet"]["catalog"]["tools"],
        "guidance": ("Prefix execution stopped; no further host reads or replay. Inspect or hand off. Native textual reasoning grants no execution authority."
                     if stopped else "Bind delivery requirements with submit(plan=null, delivery=...) before reading. Then use permitted reads, compose the required typed response in native L1, and submit netopyu_hybrid_deliver(session_id,response_json) for checks/rendering, not draft. No Runtime model or artifact execution occurs; do not create a retry session or resubmit a read plan."),
        "executionReportDigest": execution["reportDigest"] if execution else None})
    write_artifacts(folder / "result", {"report.json": report})
    return report


def submit(request):
    """Two bounded compile proposals, at most ONE admitted execution."""
    if set(request) != {"session_id", "plan", "delivery"}:
        raise ValueError("only session_id, untrusted plan and source-anchored delivery may be submitted")
    folder, frozen = _read_session(request["session_id"])
    if frozen.get("compilerMode") == "isolated":
        raise PermissionError("isolated compiler owns proposals; execution Agent cannot submit or replace AST")
    with _locked(folder):
        return _submit(request, folder, frozen)


def _submit(request, folder, frozen):
    host = _host()
    if (sha256_json(host) != frozen["hostDigest"] or frozen["authorProtocol"] != author.PROTOCOL
            or fingerprint() != frozen["implementation"]):
        raise PermissionError("host/source/contract changed; prepare a new session")
    if (folder / "result/report.json").exists():
        existing = _sealed(folder / "result/report.json")
        if (existing["route"] == "l1_fallback" and existing["runtime"]["status"] == "not_executed"
                and not (folder / "delivery-contract").exists()):
            if request["plan"] is not None:
                raise ValueError("fallback contract binding requires plan=null; no prefix will execute")
            if (folder / "followup").exists():
                raise PermissionError("delivery requirements must be fixed before observations")
            task_only = existing["reason"] == "complete_source_exceeds_budget"
            attempts = sorted((folder / "contract-proposals").glob("attempt-*"))
            if len(attempts) >= 2 or any(not (p / "result/report.json").is_file() for p in attempts):
                raise PermissionError("delivery proposal budget exhausted or unknown")
            target = folder / "contract-proposals" / f"attempt-{len(attempts)}"
            write_artifacts(target, {"proposal.json": snapshot_json(request["delivery"])})
            try:
                contract = _compile_delivery(frozen, request["delivery"], task_only=task_only)
            except ValueError as error:
                rejected = seal({"route": "delivery_needs_correction", "reason": str(error),
                    "remainingProposalAttempts": 1 - len(attempts), "executed": False, "writeAuthority": False})
                write_artifacts(target / "result", {"report.json": rejected})
                return rejected
            _install_delivery(folder, frozen, contract)
            write_artifacts(target / "result", {"report.json": seal({"contractDigest": contract["reportDigest"], "executed": False})})
            return _with_delivery(folder, existing)
        return {"route": "already_submitted", "executionReplayed": False,
                "existingSessionRoute": existing["route"], "submissionIgnored": True,
                "guidance": "This submission changed no state. A null plan cannot switch an admitted Runtime session into fallback. Inspect the existing session and follow its deliveryAction.",
                "existing": inspect({"session_id": request["session_id"]}), "next_tool": "netopyu_hybrid_inspect"}
    attempts = sorted((folder / "proposals").glob("attempt-*"))
    if any(not (p / "rejection/report.json").is_file() for p in attempts):
        return {"route": "pending_or_unknown_no_retry", "executionReplayed": False}
    if len(attempts) >= 2:
        raise PermissionError("compile proposal budget exhausted")
    if attempts:
        _sealed(attempts[-1] / "rejection/report.json")
    attempt = folder / "proposals" / f"attempt-{len(attempts)}"
    # Claim BEFORE compilation. Only a persisted deterministic rejection permits
    # one correction; a missing rejection means unknown, never implicit replay.
    write_artifacts(attempt, {"plan.json": snapshot_json(request["plan"]), "delivery.json": snapshot_json(request["delivery"])})
    packet = frozen["packet"]
    try:
        contract = _compile_delivery(frozen, request["delivery"])
        if not isinstance(request["plan"], dict) or request["plan"].get("mode") != "read_prefix":
            raise ValueError("this host accepts only retained-task read_prefix; no arbitrary graph activation")
        compilation = author.compile_proposal(packet, frozen["visible"], request["plan"])
    except (ValueError, KeyError, TypeError) as error:
        reason = "translation_rejected: " + str(error)[:500]
        rejection = seal({"route": "translation_needs_correction", "reason": reason,
            "planDigest": sha256_json(request["plan"]), "providerCalls": [], "modelCalls": [],
            "executed": False, "remainingProposalAttempts": 1 - len(attempts),
            "writeAuthority": False, "taskSuccess": None,
            "guidance": "No operation executed. Correct only this rejected proposal using the prepared schema, then resubmit in this session once. Invocation arguments are caller bindings (input#/field), not original source literals. Do not invent quotes or relax contracts. If unsupported, use an honest boundary."})
        write_artifacts(attempt / "rejection", {"report.json": rejection})
        if not attempts:
            return rejection
        return _fallback(folder, frozen, reason)
    _install_delivery(folder, frozen, contract)
    # Typed compilation is not semantic approval. The host grants only bounded
    # read authority; exact resource checks still run BEFORE each provider call.
    review = {"case": request["session_id"], "compilationDigest": compilation["reportDigest"],
        "decision": "admit_local_read_reason_only", "reviewKind": "host_readonly_policy_not_semantic_approval",
        "rationale": "Operator-enabled isolated read catalog; independently guarded exact resources; no Effects.",
        "hostDigest": frozen["hostDigest"], "semanticApproval": False}
    files = {"packet.json": packet, "compilation.json": compilation, "admission.json": review,
             "fixture.json": {"arguments": frozen["arguments"], "resources": host["resources"]}}
    write_artifacts(folder / "prepared", files)
    execution = run_prefix(packet, compilation, frozen["arguments"], host["resources"], folder / "prefix")
    if execution["execution"]["status"] not in {"governed_graph_completed", "no_prefix_reads"}:
        return _fallback(folder, frozen, "runtime_stopped", compilation=compilation, execution=execution)
    report = seal({"route": "collecting_evidence",
        "translation": {"compiled": True, "originalTaskRetained": compilation["retainedOriginalTask"],
                        "sourceDigest": packet["bundle"]["bundleDigest"], "semanticFidelity": "not_assessed",
                        "strictReadNodes": len(request["plan"]["reads"]), "l0WholeSkill": False},
        "runtime": {"status": execution["execution"]["status"], "providerCalls": execution["providerCalls"],
                    "effectCalls": execution["effectCalls"], "admission": review["reviewKind"]},
        "task": {"status": "not_drafted", "success": None},
        "observations": execution["execution"]["outputs"],
        "next_tools": ["netopyu_hybrid_read", "netopyu_hybrid_draft"],
        "host_tools": packet["catalog"]["tools"],
        "guidance": "Inspect actual observations first. If an index points to required evidence, request it with read before draft. Draft freezes evidence and makes one bounded model call; it does not prove completeness. Do not deliver an index as a finished analysis.",
        "writeAuthority": False, "executionReportDigest": execution["reportDigest"],
        "runtimeWallLatencyMs": execution["runtimeWallLatencyMs"], "modelCalls": execution["modelCalls"],
        "artifactDirectory": str(folder)})
    write_artifacts(folder / "result", {"report.json": report})
    return _with_delivery(folder, report)


def inspect(request):
    if set(request) != {"session_id"}:
        raise ValueError("only session_id is accepted")
    folder, _ = _read_session(request["session_id"])
    if (folder / "revision/result/report.json").is_file():
        return _sealed(folder / "revision/result/report.json")
    if (folder / "revision").exists():
        return {"route": "pending_or_unknown_no_retry", "phase": "revision", "writeAuthority": False, "taskSuccess": None}
    if (folder / "draft/result/report.json").is_file():
        return _sealed(folder / "draft/result/report.json")
    if (folder / "draft").exists():
        return {"route": "pending_or_unknown_no_retry", "phase": "draft", "writeAuthority": False, "taskSuccess": None}
    if (folder / "compiler").exists() and not (folder / "result/report.json").exists():
        return {"route": "pending_or_unknown_no_retry", "phase": "compiler", "writeAuthority": False, "taskSuccess": None}
    if (folder / "result/report.json").is_file():
        report = read_json(folder / "result/report.json")
        if report != seal({k: v for k, v in report.items() if k != "reportDigest"}):
            raise ValueError("session result drift")
        return _with_delivery(folder, report)
    attempts = sorted((folder / "proposals").glob("attempt-*"))
    if attempts and all((p / "rejection/report.json").is_file() for p in attempts):
        return _sealed(attempts[-1] / "rejection/report.json")
    return {"route": "pending_or_unknown_no_retry" if attempts or (folder / "attempt").exists() else "awaiting_agent_translation",
            "writeAuthority": False, "taskSuccess": None, "artifactDirectory": str(folder)}


def read(request):
    """L1 fallback can request two further reads, through the SAME read gateway.

    No graph mutation, LLM invocation, approval or new source execution. Attempts
    (including denied/failed requests) consume the bound; never retry unknown work.
    """
    if set(request) != {"session_id", "tool", "arguments"}:
        raise ValueError("only session_id, tool and arguments are accepted")
    folder, frozen = _read_session(request["session_id"])
    host = _host()
    if sha256_json(host) != frozen["hostDigest"] or fingerprint() != frozen["implementation"]:
        raise PermissionError("host or implementation changed")
    if not (folder / "result/report.json").is_file():
        raise PermissionError("no follow-up reads during pending or unknown submission")
    with _locked(folder):
        if (folder / "draft").exists():
            raise PermissionError("evidence frozen for draft; no late reads or automatic retry")
        prior = _sealed(folder / "result/report.json")
        if prior["route"] == "l1_fallback" and prior["runtime"]["status"] != "not_executed":
            raise PermissionError("prefix execution stopped; no follow-up replay of possibly uncertain reads")
        _delivery_contract(folder)
        followup = folder / "followup"
        followup.mkdir(exist_ok=True, mode=0o700)
        if any(not (p / "result/report.json").is_file() for p in followup.glob("attempt-*")):
            raise PermissionError("unknown previous read completion; inspect or hand off")
        index = len(list(followup.glob("attempt-*")))
        if index >= 2:
            raise PermissionError("two follow-up attempts exhausted; clarify or hand off")
        target = followup / f"attempt-{index}"
        action = seal(snapshot_json(request))
        write_artifacts(target, {"request.json": action})
        calls = []
        try:
            if request["tool"] not in frozen["packet"]["reads"]:
                raise PermissionError("tool is not in operator read catalog")
            contract = parse_read_contract(frozen["packet"]["reads"][request["tool"]])
            validate_data(contract.spec.input_schema, request["arguments"])
            existing = (_recorded_snapshot_read(folder, request)
                        if frozen.get("deliveryProtocol") == delivery.TASK_PROFILE else None)
            if existing:
                result = {"status": "read_not_reexecuted", "reason": "same_immutable_snapshot_already_observed",
                    "existingObservation": existing, "newObservation": False,
                    "guidance": "Use the already returned observation. This snapshot cannot refresh by repeating the call. A different authorized resource may be read within remaining limits; otherwise explain the precise missing external evidence. This attempt still consumes budget."}
            else:
                _, bindings = bindings_for(frozen["packet"], host["resources"], calls)
                receipt = execute_host_read(contract, request["arguments"], local_context(), bindings[contract.contract_hash])
                result = {"status": "read_completed", "receipt": receipt}
        except (ValueError, TypeError, PermissionError) as error:
            result = {"status": "read_rejected", "errorType": type(error).__name__}
            if frozen.get("deliveryProtocol") == delivery.TASK_PROFILE and isinstance(error, DataBindingError):
                result["diagnostic"] = {**error.as_dict(), "pointer": "/arguments" + error.pointer,
                    "argumentProtocol": "concrete_values_not_compiler_bindings",
                    "guidance": "Use the real tool input schema. Caller/literal source expressions belong only in plan.reads. No coercion or default was applied; rejection still consumes this read attempt."}
                result["inputSchema"] = frozen["packet"]["reads"][request["tool"]]["spec"]["inputSchema"]
        report = seal({**result, "requestDigest": action["reportDigest"], "providerCalls": calls, "remainingReadAttempts": 1 - index,
                       "deliveryAction": _delivery_action(folder, prior), "writeAuthority": False, "taskSuccess": None})
        write_artifacts(target / "result", {"report.json": report})
        return seal({**{k: v for k, v in report.items() if k != "reportDigest"},
                     "evidenceState": _evidence_state(folder)})


_FINISHED_DELIVERY_GUIDANCE = (
    "This delivery attempt cannot be retried. Do not call draft/deliver again or create a replacement session. "
    "If task.delivery exists, return its rendered text faithfully with its limitations; it is not semantic approval. "
    "If validation failed, state that no validated delivery exists and label any native prose as unvalidated. "
    "An unknown outcome permits inspect only, not replay. A retained candidate is not proof it passed validation."
)


def _host_result(delivered, checks):
    """Authoritative mechanical outcome, never a model's completion claim."""
    if delivered is None:
        state, message = "rejected", "No validated delivery exists. Retained candidate text was NOT admitted."
    elif not delivered["shapeComplete"] or checks["status"] == "failed_checks":
        state, message = "needs_revision", "Text retained, but required content or static checks remain unresolved."
    else:
        state, message = "candidate_unverified", "Representation accepted. Business meaning and task completion have NOT been verified."
    return {"state": state, "message": message, "semanticApproval": False, "taskSuccess": None,
            "deliveryDigest": delivered["reportDigest"] if delivered is not None else None,
            "agentProseAuthority": "cannot_override_this_host_result"}


def _existing_draft(folder):
    return {"route": "already_drafted_or_unknown", "executionReplayed": False, "retryAllowed": False,
            "guidance": _FINISHED_DELIVERY_GUIDANCE, "existing": inspect({"session_id": folder.name})}


def _offer_artifact_revision(report, frozen):
    """Operator-enabled, check-triggered ONE code-body correction, not a retry."""
    if not frozen.get("artifactRepair") or report["hostResult"]["state"] != "needs_revision":
        return report
    slots = artifact_repair.regions(report["task"]["delivery"]["rendered"], report["artifactChecks"])
    if not slots:
        return report
    return seal({**{k: v for k, v in report.items() if k != "reportDigest"},
        "revisionAllowed": True, "revisionAttemptsAllowed": 1,
        "revisionSchema": artifact_repair.response_schema(slots), "revisionRegions": slots,
        "guidance": "Host static checks found a concrete code-artifact defect. Exactly ONE native text revision is permitted via netopyu_hybrid_deliver. "
                    "response_json must encode revisionSchema (replacements with location and code), NOT answer. Edit code bodies only; "
                    "the host preserves fences and all other text. Inspect failed AND unverified checks against the original task/observations; "
                    "do not delete the requested artifact or invent missing function behavior. No reads, model regeneration, graph changes or effects. "
                    "Rechecks cannot prove whole-task correctness. If unable to repair, draft(close_incomplete=true) closes without completion."})


def _revise_delivery(folder, frozen, request, *, close=False):
    """Caller holds the session lock. Claim before parsing; never a second chance."""
    if (folder / "revision").exists():
        return _existing_draft(folder)
    initial = _sealed(folder / "draft/result/report.json") if (folder / "draft/result/report.json").is_file() else None
    if not frozen.get("artifactRepair") or not initial or initial.get("revisionAllowed") is not True:
        return _existing_draft(folder)
    evidence = _sealed(folder / "draft/evidence.json")
    if evidence["reportDigest"] != initial["evidenceDigest"]:
        raise ValueError("revision evidence drift")
    followups, attempts = _followup_evidence(folder)
    if followups != evidence["followups"] or attempts != evidence["readAttemptDigests"]:
        raise ValueError("revision cannot use altered observations")
    observations = [row["result"] for row in followups]
    if evidence["prefixDigest"] is not None:
        prefix = _sealed(folder / "prefix/summary/report.json")
        if prefix["reportDigest"] != evidence["prefixDigest"]:
            raise ValueError("revision prefix drift")
        observations[:0] = [o["value"] for o in prefix["execution"]["outputs"].values()]
    write_artifacts(folder / "revision", {"request.json": seal({"initialReportDigest": initial["reportDigest"],
        "evidenceDigest": evidence["reportDigest"], "wire": snapshot_json(request), "closeIncomplete": close})})
    delivered, checks, candidate, diagnostic = None, None, None, None
    if not close:
        try:
            patch = delivery.decode_response(request["response_json"])
            candidate = artifact_repair.apply(initial["task"]["delivery"]["rendered"], initial["artifactChecks"], patch)
            delivered, checks = _render_delivery(_delivery_contract(folder), candidate, frozen["packet"]["task"],
                observations, evidence_state=evidence["evidenceState"])
        except (ValueError, TypeError) as error:
            diagnostic = error.as_dict() if isinstance(error, DataBindingError) else {"code": type(error).__name__, "detail": str(error)[:300]}
    report = seal({"sessionId": folder.name, "route": "artifact_revision_closed" if close else "artifact_revision_checked",
        "hostResult": _host_result(delivered, checks), "revisionAllowed": False, "retryAllowed": False,
        "initialReportDigest": initial["reportDigest"], "guidance": _FINISHED_DELIVERY_GUIDANCE,
        "translation": initial["translation"], "runtime": initial["runtime"],
        "task": {"status": "not_completed" if delivered is None else "candidate_needs_revision"
                 if checks["status"] == "failed_checks" else "candidate_unverified", "success": None,
                 "candidate": candidate, "delivery": delivered},
        "artifactChecks": checks, "diagnostic": diagnostic, "evidenceFrozen": True,
        "evidenceDigest": evidence["reportDigest"], "writeAuthority": False,
        "modelCalls": [], "providerCalls": [], "candidateOrigin": "native_harness_code_edit_not_runtime_regeneration",
        "artifactDirectory": str(folder)})
    write_artifacts(folder / "revision/result", {"report.json": report})
    return report


def _collection_gate(folder, frozen, prior, contract, followups, attempts, *, close=False, native_request=None):
    """Gate declared read obligations, not arbitrary natural-language meaning.

    Must run under the existing session lock. No graph, read quota or permission
    is changed; incomplete closure produces no model call or admitted text.
    """
    if frozen.get("deliveryProtocol") != delivery.TASK_PROFILE:
        return None
    state = _evidence_state(folder)
    missing = [r for r in state["requiredReads"] if not r["observed"]]
    if not missing and not close:
        return None
    remaining = 2 - len(attempts)
    if missing and remaining > 0 and not close:
        return seal({"route": "needs_required_evidence", "session_id": folder.name,
            "evidenceState": state, "remainingReadAttempts": remaining,
            "nextReads": [{"session_id": folder.name, "tool": r["tool"], "arguments": r["arguments"]} for r in missing],
            "incompleteExit": {"tool": "netopyu_hybrid_draft", "arguments": {"session_id": folder.name, "close_incomplete": True}},
            "guidance": "Normal delivery is blocked before generation. Collect the declared evidence within the original read budget, or close_incomplete for a terminal non-completion. No automatic read, coercion, retry or approval.",
            "providerCalls": [], "modelCalls": [], "evidenceFrozen": False, "writeAuthority": False, "taskSuccess": None})
    state = _evidence_state(folder, closed=True)
    evidence = seal({"prefixDigest": prior.get("executionReportDigest"), "followups": followups,
        "readAttemptDigests": attempts, "deliveryContractDigest": contract["reportDigest"], "evidenceState": state,
        "historicalSnapshotOnly": True, "collectionClosed": True, "semanticCompletenessProven": False})
    files = {"evidence.json": evidence}
    if native_request is not None:
        files["native-wire.json"] = snapshot_json(native_request)
    write_artifacts(folder / "draft", files)
    host_result = {**_host_result(None, None),
        "message": "未完成 / Incomplete: required observations were not collected or the Agent requested incomplete closure. No generated or native candidate was admitted."}
    report = seal({"sessionId": folder.name, "hostResult": host_result, "route": "closed_incomplete",
        "reason": "agent_requested_incomplete" if close else "required_evidence_budget_exhausted",
        "retryAllowed": False, "guidance": _FINISHED_DELIVERY_GUIDANCE,
        "translation": prior["translation"], "runtime": {**prior["runtime"],
            "draftStatus": "not_generated_explicit_incomplete" if close else "not_generated_missing_evidence"},
        "task": {"status": "not_completed", "success": None, "delivery": None, "candidate": None},
        "artifactChecks": None, "diagnostic": {"missingRequiredReadIds": [r["id"] for r in missing]},
        "evidenceFrozen": True, "evidenceDigest": evidence["reportDigest"], "writeAuthority": False,
        "modelCalls": [], "executionReportDigest": None, "artifactDirectory": str(folder)})
    write_artifacts(folder / "draft/result", {"report.json": report})
    return report


def draft(request):
    """Freeze actual evidence, then run ONE retained-task reason node. No self-judge."""
    if "session_id" not in request or set(request) - {"session_id", "close_incomplete"}:
        raise ValueError("only session_id plus optional v4 close_incomplete=true is accepted by draft; omit response. Native fallback text uses the separate netopyu_hybrid_deliver tool. No evidence or approval accepted.")
    folder, frozen = _read_session(request["session_id"])
    if "close_incomplete" in request and (request["close_incomplete"] is not True or frozen.get("deliveryProtocol") != delivery.TASK_PROFILE):
        raise ValueError("close_incomplete is an explicit true-only v4 safe stop, never a generation override")
    host = _host()
    if sha256_json(host) != frozen["hostDigest"] or fingerprint() != frozen["implementation"]:
        raise PermissionError("host or implementation changed")
    with _locked(folder):
        if (folder / "draft").exists():
            if request.get("close_incomplete"):
                return _revise_delivery(folder, frozen, request, close=True)
            return _existing_draft(folder)
        prefix_report = _sealed(folder / "result/report.json")
        if request.get("close_incomplete"):
            if prefix_report["route"] != "collecting_evidence" and not (
                    prefix_report["route"] == "l1_fallback" and prefix_report["runtime"]["status"] == "not_executed"):
                raise PermissionError("unknown or stopped execution cannot be closed as known")
            contract = _delivery_contract(folder)
            followups, attempts = _followup_evidence(folder)
            return _collection_gate(folder, frozen, prefix_report, contract, followups, attempts, close=True)
        if prefix_report["route"] != "collecting_evidence":
            raise PermissionError("no admitted prefix; native L1 fallback must use netopyu_hybrid_deliver(session_id,response_json), not draft")
        compilation = _sealed(folder / "prepared/compilation.json")
        prefix = _sealed(folder / "prefix/summary/report.json")
        if prefix_report["executionReportDigest"] != prefix["reportDigest"]:
            raise ValueError("prefix report binding drift")
        contract = _delivery_contract(folder)
        followups, attempts = _followup_evidence(folder)
        blocked = _collection_gate(folder, frozen, prefix_report, contract, followups, attempts)
        if blocked is not None:
            return blocked
        evidence_state = _evidence_state(folder, closed=True)
        # Claim before the model call. A crash/timeout never permits regeneration.
        write_artifacts(folder / "draft", {"evidence.json": seal({"prefixDigest": prefix["reportDigest"],
            "followups": followups, "readAttemptDigests": attempts, "historicalSnapshotOnly": True,
            "deliveryContractDigest": contract["reportDigest"],
            "evidenceState": evidence_state,
            "collectionClosed": True, "semanticCompletenessProven": False})})
    execution = draft_from_evidence(frozen["packet"], compilation, frozen["arguments"],
        prefix, followups, folder / "draft/execution", delivery_contract=contract, evidence_state=evidence_state)
    completed = execution["execution"]["status"] == "governed_graph_completed"
    checks, delivered, diagnostic = None, None, None
    if completed:
        candidate = execution["execution"]["outputs"]["n7"]["value"]
        observations = [o["value"] for o in prefix["execution"]["outputs"].values()]
        observations.extend(row["result"] for row in followups)
        try:
            delivered, checks = _render_delivery(contract, candidate, frozen["packet"]["task"], observations, evidence_state=evidence_state)
        except DataBindingError as error:
            diagnostic = error.as_dict()
    status = "candidate_needs_revision" if checks and (checks["status"] == "failed_checks" or not delivered["shapeComplete"]) else "candidate_unverified"
    report = seal({"sessionId": folder.name, "hostResult": _host_result(delivered, checks),
        "route": ("governed_hybrid_candidate" if delivered is not None else "draft_delivery_rejected") if completed else "draft_stopped_no_retry",
        "retryAllowed": False, "guidance": _FINISHED_DELIVERY_GUIDANCE,
        "translation": prefix_report["translation"],
        "runtime": {**prefix_report["runtime"], "draftStatus": execution["execution"]["status"]},
        "task": {"status": status if completed and delivered is not None else "not_completed", "success": None,
                 "candidate": execution["execution"].get("outputs", {}), "delivery": delivered},
        "artifactChecks": checks,
        "diagnostic": diagnostic,
        "evidenceFrozen": True, "evidenceDigest": _sealed(folder / "draft/evidence.json")["reportDigest"],
        "writeAuthority": False, "modelCalls": execution["modelCalls"],
        "executionReportDigest": execution["reportDigest"], "artifactDirectory": str(folder)})
    report = _offer_artifact_revision(report, frozen)
    write_artifacts(folder / "draft/result", {"report.json": report})
    return report


def _followup_evidence(folder):
    followups, attempts = [], []
    for attempt in sorted((folder / "followup").glob("attempt-*")):
        # An unknown/partial result cannot disappear during evidence freezing.
        report = _sealed(attempt / "result/report.json")
        action = _sealed(attempt / "request.json")
        if report["requestDigest"] != action["reportDigest"]:
            raise ValueError("read action binding drift")
        attempts.append(report["reportDigest"])
        if report["status"] == "read_completed":
            receipt = report["receipt"]
            if receipt["receiptDigest"] != sha256_json({k: v for k, v in receipt.items() if k != "receiptDigest"}):
                raise ValueError("read receipt drift")
            tool = action["tool"]
            if report["providerCalls"] != [{"tool": tool, "arguments": action["arguments"]}]:
                raise ValueError("read arguments/receipt association drift")
            followups.append({"tool": tool, "arguments": {tool: action["arguments"]},
                "result": {tool: receipt["payload"]}, "receiptDigest": receipt["receiptDigest"]})
    return followups, attempts


def _render_delivery(contract, response, task, observations, *, evidence_state=None):
    delivered = delivery.render(contract, response, evidence_state=evidence_state)
    checks = inspect_candidate({"draft": delivered["rendered"]}, task, observations)
    return delivered, checks


def deliver(request):
    """Native fallback submits typed text; same contract/renderer, ZERO model calls."""
    if set(request) != {"session_id", "response_json"}:
        raise ValueError("only session_id and response_json (one JSON text envelope) are accepted; no implicit response type coercion")
    folder, frozen = _read_session(request["session_id"])
    host = _host()
    if sha256_json(host) != frozen["hostDigest"] or fingerprint() != frozen["implementation"]:
        raise PermissionError("host or implementation changed")
    with _locked(folder):
        if (folder / "draft").exists():
            return _revise_delivery(folder, frozen, request)
        prior = _sealed(folder / "result/report.json")
        if prior["route"] != "l1_fallback" or prior["runtime"]["status"] != "not_executed":
            raise PermissionError("native response is only accepted in pre-execution L1 fallback. This session is not fallback: inspect its state; an admitted prefix uses netopyu_hybrid_draft with only session_id, no response. Do not resubmit a null plan to switch routes.")
        contract = _delivery_contract(folder)
        followups, attempts = _followup_evidence(folder)
        blocked = _collection_gate(folder, frozen, prior, contract, followups, attempts, native_request=request)
        if blocked is not None:
            return blocked
        evidence_state = _evidence_state(folder, closed=True)
        write_artifacts(folder / "draft", {"evidence.json": seal({"prefixDigest": None, "followups": followups,
            "readAttemptDigests": attempts, "deliveryContractDigest": contract["reportDigest"],
            "evidenceState": evidence_state,
            "historicalSnapshotOnly": True, "collectionClosed": True, "semanticCompletenessProven": False}),
            "native-wire.json": snapshot_json(request)})
    candidate = None
    try:
        candidate = delivery.decode_response(request["response_json"])
        delivered, checks = _render_delivery(contract, candidate, frozen["packet"]["task"],
            [r["result"] for r in followups], evidence_state=evidence_state)
        status = "candidate_unverified" if delivered["shapeComplete"] and checks["status"] != "failed_checks" else "candidate_needs_revision"
        problem, diagnostic = None, None
    except (ValueError, TypeError) as error:
        delivered, checks, status, problem = None, None, "not_completed", type(error).__name__
        diagnostic = error.as_dict() if isinstance(error, DataBindingError) else {
            "code": problem, "pointer": "", "detail": "candidate validation failed; inspect retained input and response schema"}
    report = seal({"sessionId": folder.name, "hostResult": _host_result(delivered, checks),
        "route": "native_l1_delivery_candidate" if delivered is not None else "native_l1_delivery_rejected",
        "candidateOrigin": "native_harness_not_runtime_model",
        "retryAllowed": False, "guidance": _FINISHED_DELIVERY_GUIDANCE,
        "translation": prior["translation"], "runtime": {**prior["runtime"], "draftStatus": "native_text_checked_no_execution"},
        "task": {"status": status, "success": None, "delivery": delivered, "candidate": candidate},
        "wireDigest": sha256_json(request), "transport": "strict_json_text_object/v1",
        "artifactChecks": checks, "errorType": problem, "diagnostic": diagnostic, "evidenceFrozen": True,
        "evidenceDigest": _sealed(folder / "draft/evidence.json")["reportDigest"],
        "writeAuthority": False, "modelCalls": [], "executionReportDigest": None, "artifactDirectory": str(folder)})
    report = _offer_artifact_revision(report, frozen)
    write_artifacts(folder / "draft/result", {"report.json": report})
    return report


def describe():
    """Host-generated first-class tool schema, never an agent-selected policy."""
    host = _host()
    packet = host["packet"]
    if host.get("compilerMode") == "isolated":
        return {"inputSchema": packet["inputSchema"], "compilerMode": "isolated",
                "artifactRepair": host.get("artifactRepair", False),
                "readRequestSchema": _read_request_schema(packet), "canCloseIncomplete": True}
    return {"inputSchema": packet["inputSchema"], "artifactRepair": host.get("artifactRepair", False),
            **({"readRequestSchema": _read_request_schema(packet), "canCloseIncomplete": True}
               if host["apiVersion"] == TASK_HOST_PROFILE else {}),
            "planSchema": {"anyOf": [author.author_response_schema(packet, list(author.pages_for(packet))), {"type": "null"}]},
            "deliverySchema": {"type": "null"} if host["apiVersion"] == TASK_HOST_PROFILE else delivery.selection_schema()
                if host["apiVersion"] in {COMPACT_HOST_PROFILE, CHOICE_HOST_PROFILE} else delivery.proposal_schema(["task", *author.pages_for(packet)])}


def _read_request_schema(packet):
    """Expose operator read contracts as concrete tool arguments, not bindings."""
    def embedded(schema, prefix):
        value = snapshot_json(schema)
        def rebase(node):
            if "$ref" in node:
                node["$ref"] = prefix + node["$ref"][1:]
            for key in ("properties", "$defs"):
                for child in node.get(key, {}).values():
                    rebase(child)
            for key in ("items", "additionalProperties"):
                if isinstance(node.get(key), dict):
                    rebase(node[key])
        rebase(value)
        return value
    reads = list(packet["reads"].items())
    base = author.obj({"session_id": {"type": "string"}, "tool": {"type": "string"}, "arguments": {"type": "object"}})
    if not reads:
        return {**base, "not": {}}
    if len(reads) == 1:
        name, contract = reads[0]
        base["properties"]["tool"] = {"const": name}
        base["properties"]["arguments"] = embedded(contract["spec"]["inputSchema"], "#/properties/arguments")
    else:
        base["properties"]["tool"]["enum"] = [name for name, _ in reads]
        base["anyOf"] = [{"properties": {"tool": {"const": name}, "arguments": embedded(
            contract["spec"]["inputSchema"], f"#/anyOf/{i}/properties/arguments")}} for i, (name, contract) in enumerate(reads)]
    return base
