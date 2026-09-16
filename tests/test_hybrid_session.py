"""Canonical agent session invariants; mocked replies are NOT semantic evidence."""
import copy
import json
import sys

import pytest

from dsh_adapter import hybrid_session as session
from evaluation.semantic_closure_transfer import packet_for
from evaluation.structured_flow_demo import fixture
from skill_authoring import compiler, local_execution
from skill_authoring.artifacts import write_artifacts
from network_runtime.l0.hybrid_execution import ReasoningReply

DELIVERY = {"requirements": [{"kind": "analysis", "language": "", "origin": "task",
    "quote": "explain its current limitations"}], "unrepresented": []}


@pytest.fixture
def host(tmp_path, monkeypatch):
    bundle, _, _, _ = fixture()
    schema = compiler.obj({"path": {"type": "string"}})
    tool = {"name": "read_export", "inputSchema": schema,
            "outputSchema": compiler.obj({"text": {"type": "string"}}), "annotations": {"readOnlyHint": True}}
    packet = packet_for(bundle, {"task": "Read the supplied snapshot and explain its current limitations, without changing anything.",
                                "inputSchema": schema, "tools": [tool]})
    profile = {"apiVersion": session.PROFILE, "enabled": True, "packet": packet,
               "resources": {"read_export": [{"path": "/sandbox/snapshot"}, {"text": "Pia has not approved."}]}}
    write_artifacts(tmp_path / "host", {"profile.json": profile})
    monkeypatch.setenv("NETOPYU_HYBRID_HOST_PROFILE", str(tmp_path / "host/profile.json"))
    monkeypatch.setenv("NETOPYU_HYBRID_SESSIONS_DIR", str(tmp_path / "sessions"))
    request = {"task": packet["task"], "arguments": {"path": "/sandbox/snapshot"}}
    prepared = session.prepare(request)
    choice = {"mode": "read_prefix", "intent_summary": "Read the exact caller snapshot under host authority and retain the original task for bounded open reasoning.",
              "reads": [{"id": "n0", "after": [], "evidence": ["task"], "tool": "read_export",
                         "arguments": {"path": {"caller": "input#/path"}}}], "boundaries": []}
    observed = []
    def invoke(req, folder, costs):
        observed.append(copy.deepcopy(req))
        costs.append({"inputTokens": 10, "outputTokens": 10, "latencyMs": 1})
        return ReasoningReply({"delivery": {"d0": {"state": "provided", "content": {
            "text": "Pia has not approved; review status is unknown."}, "gap": ""}}, "uncertainties": []},
                              compiler.MODEL, compiler.CONFIG_DIGEST, 10, 10)
    monkeypatch.setattr(local_execution, "invoke_local", invoke)
    return tmp_path, profile, request, prepared, choice, observed


def test_shared_compiler_is_exact_same_object_as_evaluator():
    from evaluation import hybrid_authoring
    assert hybrid_authoring is compiler


@pytest.fixture
def isolated_host(host, monkeypatch):
    root, profile, request, _, choice, observed = host
    profile.update(apiVersion=session.TASK_HOST_PROFILE, compilerMode="isolated")
    (root / "host/profile.json").write_text(json.dumps(profile))
    calls = []
    def propose(packet, visible, folder):
        calls.append(copy.deepcopy(packet))
        assert "arguments" not in packet and "resources" not in packet
        assert visible == list(compiler.pages_for(packet))
        return copy.deepcopy(choice)
    monkeypatch.setattr(session.isolated_compiler, "invoke", propose)
    return root, profile, request, choice, calls, observed


def test_isolated_compiler_owns_ast_and_agent_gets_only_execution_context(isolated_host):
    root, profile, request, _, calls, observed = isolated_host
    schema = session.describe()
    assert schema["compilerMode"] == "isolated" and "planSchema" not in schema and "deliverySchema" not in schema
    prepared = session.prepare(request)
    assert prepared["route"] == "collecting_evidence" and len(calls) == 1 and not observed
    assert prepared["runtime"]["providerCalls"] == [{"tool": "read_export", "arguments": request["arguments"]}]
    assert prepared["executionContext"]["originalTask"] == request["task"]
    assert prepared["executionContext"]["arguments"] == request["arguments"]
    assert prepared["executionContext"]["sourcePages"] == {k: v["text"] for k, v in compiler.pages_for(profile["packet"]).items()}
    assert not ({"authoring", "authoring_instruction", "planSchema", "requiredOutputSchema", "plan"} & set(prepared))
    assert session.inspect({"session_id": prepared["session_id"]})["route"] == "collecting_evidence"
    with pytest.raises(PermissionError, match="owns proposals"):
        session.submit({"session_id": prepared["session_id"], "plan": None, "delivery": None})
    assert len(calls) == 1
    assert session._sealed(root / "sessions" / prepared["session_id"] / "compiler/claim.json")["toolsAvailable"] is False


@pytest.mark.parametrize("fault", ["concrete_arguments", "write_tool", "response_rejected"])
def test_isolated_rejection_keeps_native_readonly_fallback_not_a_second_author_call(isolated_host, monkeypatch, fault):
    _, _, request, choice, calls, observed = isolated_host
    def propose(*_):
        calls.append(fault)
        if fault == "response_rejected":
            raise session.isolated_compiler.ResponseRejected("malformed")
        bad = copy.deepcopy(choice)
        if fault == "concrete_arguments":
            bad["reads"][0]["arguments"] = {"path": "/sandbox/snapshot"}
        else:
            bad["reads"][0]["tool"] = "write_config"
        return bad
    monkeypatch.setattr(session.isolated_compiler, "invoke", propose)
    result = session.prepare(request)
    assert result["route"] == "l1_fallback" and result["runtime"]["providerCalls"] == []
    assert result["deliveryAction"]["tool"] == "netopyu_hybrid_deliver"
    sid = result["session_id"]
    assert session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})["status"] == "read_completed"
    answer = session.deliver({"session_id": sid, "response_json": '{"answer":"Local candidate, not proven."}'})
    assert answer["hostResult"]["state"] == "candidate_unverified" and not observed and len(calls) == 1


def test_unknown_isolated_author_call_never_falls_back_or_replays(isolated_host, monkeypatch):
    _, _, request, _, calls, _ = isolated_host
    def uncertain(*_):
        calls.append("timeout")
        raise TimeoutError("outcome unknown")
    monkeypatch.setattr(session.isolated_compiler, "invoke", uncertain)
    result = session.prepare(request)
    sid = result["session_id"]
    assert result["route"] == "pending_or_unknown_no_retry"
    for _ in range(2):
        assert session.inspect({"session_id": sid})["phase"] == "compiler"
    with pytest.raises(PermissionError):
        session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})
    with pytest.raises(PermissionError):
        session.submit({"session_id": sid, "plan": None, "delivery": None})
    assert calls == ["timeout"]


def test_isolated_source_overflow_binds_fallback_without_author_call(isolated_host, monkeypatch):
    _, _, request, _, calls, _ = isolated_host
    monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    result = session.prepare(request)
    assert not calls and result["route"] == "l1_fallback"
    assert result["executionContext"]["sourcePages"] == {}
    assert not result["executionContext"]["sourceSuppliedToExecutionAgent"]
    assert result["deliveryAction"]["tool"] == "netopyu_hybrid_deliver"


def test_isolated_prefix_still_honors_required_reads_and_one_runtime_draft(isolated_host, monkeypatch):
    root, profile, request, _, calls, observed = isolated_host
    profile["resources"]["read_export"] = {"resources": [profile["resources"]["read_export"],
        [{"path": "/sandbox/next"}, {"text": "New evidence, not an approval."}]]}
    profile["requiredReads"] = [{"tool": "read_export", "arguments": {"path": "/sandbox/next"}}]
    (root / "host/profile.json").write_text(json.dumps(profile))
    def draft(req, *_):
        observed.append(req)
        return ReasoningReply({"answer": "Both snapshots observed; task meaning remains unverified."}, compiler.MODEL, compiler.CONFIG_DIGEST, 10, 5)
    monkeypatch.setattr(local_execution, "invoke_local", draft)
    started = session.prepare(request)
    sid = started["session_id"]
    blocked = session.draft({"session_id": sid})
    assert blocked["route"] == "needs_required_evidence" and not observed
    session.read(blocked["nextReads"][0])
    result = session.draft({"session_id": sid})
    assert result["hostResult"]["state"] == "candidate_unverified" and not result["writeAuthority"]
    assert len(calls) == len(observed) == 1
    assert not session.draft({"session_id": sid})["executionReplayed"]
    assert len(observed) == 1


def test_isolated_rejected_response_does_not_bypass_host_drift(isolated_host, monkeypatch):
    root, profile, request, _, _, _ = isolated_host
    def reject_after_host_change(*_):
        profile["resources"]["read_export"][1]["text"] = "Changed host snapshot."
        (root / "host/profile.json").write_text(json.dumps(profile))
        raise session.isolated_compiler.ResponseRejected("invalid response")
    monkeypatch.setattr(session.isolated_compiler, "invoke", reject_after_host_change)
    with pytest.raises(PermissionError, match="changed"):
        session.prepare(request)
    assert not list((root / "sessions").glob("*/prefix"))


@pytest.mark.parametrize("value", [None, "agent", True, {}])
def test_compiler_mode_is_explicit_operator_configuration_only(host, value):
    root, profile, request, _, _, _ = host
    profile.update(apiVersion=session.TASK_HOST_PROFILE, compilerMode=value)
    (root / "host/profile.json").write_text(json.dumps(profile))
    with pytest.raises(PermissionError):
        session.prepare(request)


@pytest.fixture
def evidence_host(host, monkeypatch):
    root, profile, request, _, choice, observed = host
    profile["apiVersion"] = session.TASK_HOST_PROFILE
    profile["resources"]["read_export"] = {"resources": [profile["resources"]["read_export"],
        [{"path": "/sandbox/required"}, {"text": "Host-required evidence."}],
        [{"path": "/sandbox/optional"}, {"text": "Available but not required."}]]}
    profile["requiredReads"] = [{"tool": "read_export", "arguments": {"path": "/sandbox/required"}}]
    (root / "host/profile.json").write_text(json.dumps(profile))
    def invoke(req, folder, costs):
        observed.append(copy.deepcopy(req))
        return ReasoningReply({"answer": "Required observation obtained; meaning is not approved."},
                              compiler.MODEL, compiler.CONFIG_DIGEST, 10, 10)
    monkeypatch.setattr(local_execution, "invoke_local", invoke)
    return root, profile, request, choice, observed


@pytest.mark.parametrize("fallback", [False, True])
def test_missing_host_evidence_blocks_both_routes_before_model_or_text_admission(evidence_host, monkeypatch, fallback):
    root, _, request, choice, observed = evidence_host
    if fallback:
        monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": None if fallback else choice, "delivery": None})
    call = ({"session_id": sid, "response_json": '{"answer":"Untrusted completion claim"}'} if fallback else {"session_id": sid})
    fn = session.deliver if fallback else session.draft
    blocked = fn(call)
    assert blocked["route"] == "needs_required_evidence" and not blocked["modelCalls"]
    assert not (root / "sessions" / sid / "draft").exists() and not observed
    assert blocked["remainingReadAttempts"] == 2 and not blocked["evidenceFrozen"]
    assert len(blocked["nextReads"]) == 1  # Optional resource is not silently mandatory.
    read = session.read(blocked["nextReads"][0])
    assert read["evidenceState"]["requiredReadsComplete"] is True
    delivered = fn(call)
    assert delivered["hostResult"]["state"] == "candidate_unverified"
    assert delivered["task"]["success"] is None
    assert len(observed) == (0 if fallback else 1)
    assert not session.draft({"session_id": sid})["executionReplayed"]


def test_exhausted_budget_closes_without_generation_and_preserves_failed_reads(evidence_host):
    root, _, request, choice, observed = evidence_host
    sid = session.prepare(request)["session_id"]
    session.submit({"session_id": sid, "plan": choice, "delivery": None})
    bad = {"session_id": sid, "tool": "read_export", "arguments": {"path": {"literal": "private-value", "caller": "input#/path"}}}
    first = session.read(bad)
    assert first["diagnostic"]["pointer"] == "/arguments/path"
    assert "private-value" not in json.dumps(first["diagnostic"])
    assert first["inputSchema"]["properties"]["path"] == {"type": "string"}
    assert first["remainingReadAttempts"] == 1 and not first["providerCalls"]
    second = session.read({**bad, "arguments": {"path": "/sandbox/optional"}})
    assert second["remainingReadAttempts"] == 0
    closed = session.draft({"session_id": sid})
    assert closed["route"] == "closed_incomplete" and closed["hostResult"]["state"] == "rejected"
    assert closed["reason"] == "required_evidence_budget_exhausted"
    assert closed["task"]["delivery"] is None and not closed["modelCalls"] and not observed
    evidence = session._sealed(root / "sessions" / sid / "draft/evidence.json")
    assert len(evidence["readAttemptDigests"]) == 2 and not evidence["evidenceState"]["requiredReadsComplete"]
    with pytest.raises(PermissionError, match="frozen"):
        session.read(closed.get("nextReads", [bad])[0])
    assert not session.draft({"session_id": sid})["executionReplayed"]


@pytest.mark.parametrize("fallback", [False, True])
def test_explicit_incomplete_exit_is_safe_on_both_routes_without_spending_remaining_reads(evidence_host, monkeypatch, fallback):
    root, _, request, choice, observed = evidence_host
    if fallback:
        monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    sid = session.prepare(request)["session_id"]
    submitted = session.submit({"session_id": sid, "plan": None if fallback else choice, "delivery": None})
    action = submitted["deliveryAction"]
    assert action["incompleteExit"]["arguments"] == {"session_id": sid, "close_incomplete": True}
    assert "close_incomplete=true" in action["guidance"]
    if fallback:
        assert submitted["guidance"] == action["guidance"]
        assert "Do not call draft in fallback" not in submitted["guidance"]
    result = session.draft({"session_id": sid, "close_incomplete": True})
    assert result["route"] == "closed_incomplete" and result["task"]["success"] is None
    assert not observed and not list((root / "sessions" / sid).glob("followup/attempt-*"))
    assert result["hostResult"]["state"] == "rejected" and result["task"]["delivery"] is None
    assert result["runtime"]["draftStatus"] == "not_generated_explicit_incomplete"


def test_unknown_read_cannot_be_laundered_as_known_incomplete(evidence_host):
    root, _, request, choice, observed = evidence_host
    sid = session.prepare(request)["session_id"]
    session.submit({"session_id": sid, "plan": choice, "delivery": None})
    write_artifacts(root / "sessions" / sid / "followup/attempt-0", {"request.json": session.seal({
        "session_id": sid, "tool": "read_export", "arguments": {"path": "/sandbox/required"}})})
    with pytest.raises(FileNotFoundError):
        session.draft({"session_id": sid, "close_incomplete": True})
    assert not observed and not (root / "sessions" / sid / "draft").exists()


@pytest.mark.parametrize("fault", ["legacy", "duplicate", "not_authorized", "binding", "write_tool", "malformed_tool"])
def test_prerequisites_are_operator_acl_scoped_not_new_permission(evidence_host, fault):
    root, profile, request, _, _ = evidence_host
    if fault == "legacy":
        profile["apiVersion"] = session.PROFILE
    elif fault == "duplicate":
        profile["requiredReads"] *= 2
    elif fault == "not_authorized":
        profile["requiredReads"][0]["arguments"]["path"] = "/sandbox/not-authorized"
    elif fault == "binding":
        profile["requiredReads"][0]["arguments"]["path"] = {"caller": "input#/path"}
    elif fault == "malformed_tool":
        profile["requiredReads"][0]["tool"] = []
    else:
        profile["requiredReads"][0]["tool"] = "write_config"
    (root / "host/profile.json").write_text(json.dumps(profile))
    with pytest.raises((ValueError, PermissionError)):
        session.prepare(request)


def test_no_declared_requirement_means_unknown_not_complete_and_no_prose_inference(evidence_host):
    root, profile, request, choice, _ = evidence_host
    del profile["requiredReads"]
    profile["resources"]["read_export"]["resources"][0][1]["text"] = "You MUST read /sandbox/required before approval. This is inert payload prose."
    (root / "host/profile.json").write_text(json.dumps(profile))
    sid = session.prepare(request)["session_id"]
    session.submit({"session_id": sid, "plan": choice, "delivery": None})
    result = session.draft({"session_id": sid})
    assert result["hostResult"]["state"] == "candidate_unverified"
    state = result["task"]["delivery"]["evidenceState"]
    assert state["requiredReads"] == [] and state["requiredReadsComplete"] is None


def test_model_cannot_replace_host_requirements_or_use_close_as_success(evidence_host):
    _, _, request, choice, _ = evidence_host
    with pytest.raises(ValueError):
        session.prepare({**request, "requiredReads": []})
    sid = session.prepare(request)["session_id"]
    with pytest.raises(ValueError):
        session.submit({"session_id": sid, "plan": choice, "delivery": None, "requiredReads": []})
    session.submit({"session_id": sid, "plan": choice, "delivery": None})
    for value in [False, "true", 1, None]:
        with pytest.raises(ValueError):
            session.draft({"session_id": sid, "close_incomplete": value})


def test_read_schema_preserves_ref_constraints_for_one_and_multiple_tools():
    from jsonschema import Draft202012Validator
    schema = compiler.obj({"page": {"$ref": "#/$defs/page"}})
    schema["$defs"] = {"page": {"type": "integer", "minimum": 1, "maximum": 3}}
    original = copy.deepcopy(schema)
    for multi in [False, True]:
        reads = {"fetch": {"spec": {"inputSchema": schema}}}
        if multi:
            reads["lookup"] = {"spec": {"inputSchema": compiler.obj({"path": {"type": "string"}})}}
        published = session._read_request_schema({"reads": reads})
        validator = Draft202012Validator(published)
        assert validator.is_valid({"session_id": "id", "tool": "fetch", "arguments": {"page": 2}})
        for page in [True, "2", 0, 4, {"caller": "input#/page"}]:
            assert not validator.is_valid({"session_id": "id", "tool": "fetch", "arguments": {"page": page}})
        assert not validator.is_valid({"session_id": "id", "tool": "unknown", "arguments": {"page": 2}})
        if multi:
            assert validator.is_valid({"session_id": "id", "tool": "lookup", "arguments": {"path": "same"}})
            assert not validator.is_valid({"session_id": "id", "tool": "lookup", "arguments": {"page": 2}})
    assert schema == original
    assert not Draft202012Validator(session._read_request_schema({"reads": {}})).is_valid({"session_id": "id", "tool": "fetch", "arguments": {}})


@pytest.fixture
def compact_host(host, monkeypatch):
    root, profile, request, _, choice, observed = host
    profile["apiVersion"] = session.COMPACT_HOST_PROFILE
    (root / "host/profile.json").write_text(json.dumps(profile))
    def invoke(req, folder, costs):
        observed.append(copy.deepcopy(req))
        costs.append({"inputTokens": 10, "outputTokens": 10, "latencyMs": 1})
        return ReasoningReply({"delivery": {"d0": {"text": "Pia has not approved; review status unknown."}},
            "unresolved": {}, "uncertainties": []}, compiler.MODEL, compiler.CONFIG_DIGEST, 10, 10)
    monkeypatch.setattr(local_execution, "invoke_local", invoke)
    prepared = session.prepare(request)
    selected = {"requirements": [{"kind": "analysis", "language": "", "source_ref": "task:0"}], "unrepresented": []}
    return root, profile, request, prepared, choice, observed, selected


def test_compact_prepare_replaces_source_copy_with_lossless_addressed_spans(compact_host):
    _, profile, _, prepared, _, _, _ = compact_host
    payload = prepared["authoring"]
    assert "sourcePages" not in payload
    refs = payload["deliverySourceReferences"]
    for key, page in compiler.pages_for(profile["packet"]).items():
        assert "".join(r["text"] for r in refs.values() if r["origin"] == key) == page["text"]
    assert "source_ref" in session.describe()["deliverySchema"]["properties"]["requirements"]["items"]["required"]
    assert "sourcePages are already available" not in prepared["authoring_instruction"]
    assert "deliverySourceReferences" in prepared["authoring_instruction"]


def test_compact_tool_header_does_not_load_the_whole_reference_catalog(compact_host, monkeypatch):
    def forbidden(_):
        raise AssertionError("source IDs belong in prepare, not static tool metadata")
    monkeypatch.setattr(session.delivery, "source_references", forbidden)
    schema = session.describe()["deliverySchema"]
    assert len(json.dumps(schema, ensure_ascii=False).encode()) < 2048
    field = schema["properties"]["requirements"]["items"]["properties"]["source_ref"]
    assert field["type"] == "string" and "enum" not in field


def test_compact_contract_and_runtime_share_schema_evidence_and_no_replay(compact_host):
    _, _, request, prepared, choice, observed, selected = compact_host
    sid = prepared["session_id"]
    report = session.submit({"session_id": sid, "plan": choice, "delivery": selected})
    assert report["deliveryContract"]["profile"] == session.delivery.COMPACT_PROFILE
    assert report["deliveryContract"]["requirements"][0]["quote"] == request["task"]
    result = session.draft({"session_id": sid})
    assert result["task"]["status"] == "candidate_unverified" and result["diagnostic"] is None
    assert result["task"]["delivery"]["shapeComplete"] and result["task"]["success"] is None
    assert len(observed) == 1 and observed[0]["tools"] == []
    assert "host-owned internal representation" in observed[0]["instructions"]
    assert json.loads(observed[0]["inputs"]["host_read_state"])["completedReads"] == 1
    assert not session.draft({"session_id": sid})["executionReplayed"]
    assert len(observed) == 1


def test_compact_fallback_references_only_supplied_task_and_accepts_content(compact_host, monkeypatch):
    _, _, request, _, _, observed, selected = compact_host
    monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    assert not prepared["sourceSuppliedToModel"]
    assert all(r["origin"] == "task" for r in prepared["deliverySourceReferences"].values())
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": None, "delivery": selected})
    session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})
    result = session.deliver({"session_id": sid, "response_json": json.dumps({
        "delivery": {"d0": {"text": "Not approved."}}, "unresolved": {}, "uncertainties": []})})
    assert result["task"]["delivery"]["shapeComplete"] and not result["modelCalls"] and not observed
    assert result["task"]["delivery"]["evidenceState"]["completedReads"] == 1
    assert not result["writeAuthority"]


def test_compact_runtime_null_without_gap_persists_failure_without_replay(compact_host, monkeypatch):
    _, _, _, prepared, choice, observed, selected = compact_host
    def invalid(req, folder, costs):
        observed.append(req)
        return ReasoningReply({"delivery": {"d0": None}, "unresolved": {}, "uncertainties": []},
            compiler.MODEL, compiler.CONFIG_DIGEST, 10, 10)
    monkeypatch.setattr(local_execution, "invoke_local", invalid)
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": choice, "delivery": selected})
    result = session.draft({"session_id": sid})
    assert result["task"]["status"] == "not_completed" and result["task"]["delivery"] is None
    assert result["route"] == "draft_delivery_rejected" and result["hostResult"]["state"] == "rejected"
    assert result["diagnostic"]["pointer"] == "/unresolved/d0"
    assert session.inspect({"session_id": sid}) == result
    assert not session.draft({"session_id": sid})["executionReplayed"] and len(observed) == 1


@pytest.mark.parametrize("fallback", [False, True])
def test_choice_protocol_uses_one_schema_on_both_routes(compact_host, monkeypatch, fallback):
    root, profile, request, _, choice, observed, selected = compact_host
    profile["apiVersion"] = session.CHOICE_HOST_PROFILE
    (root / "host/profile.json").write_text(json.dumps(profile))
    candidate = {"delivery": {"d0": {"text": "Not approved; evidence incomplete."}}, "uncertainties": []}
    def invoke(req, folder, costs):
        observed.append(copy.deepcopy(req))
        return ReasoningReply(candidate, compiler.MODEL, compiler.CONFIG_DIGEST, 10, 10)
    monkeypatch.setattr(local_execution, "invoke_local", invoke)
    if fallback:
        monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    sid = prepared["session_id"]
    report = session.submit({"session_id": sid, "plan": None if fallback else choice, "delivery": selected})
    assert report["deliveryContract"]["profile"] == session.delivery.CHOICE_PROFILE
    assert set(report["deliveryResponseSchema"]["properties"]) == {"delivery", "uncertainties"}
    if fallback:
        session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})
        result = session.deliver({"session_id": sid, "response_json": json.dumps(candidate)})
    else:
        result = session.draft({"session_id": sid})
        assert observed[0]["outputSchema"] == report["deliveryResponseSchema"]
    assert len(observed) == (0 if fallback else 1)
    assert result["hostResult"]["state"] == "candidate_unverified" and result["task"]["success"] is None
    assert result["task"]["delivery"]["evidenceState"]["completedReads"] == 1
    assert not result["writeAuthority"] and not session.draft({"session_id": sid})["executionReplayed"]


@pytest.mark.parametrize("fallback", [False, True])
def test_task_bound_session_removes_semantic_selection_not_runtime_controls(host, monkeypatch, fallback):
    root, profile, request, _, choice, observed = host
    profile["apiVersion"] = session.TASK_HOST_PROFILE
    (root / "host/profile.json").write_text(json.dumps(profile))
    answer = "## Status\nNot approved; review unknown.\n\n## Next\nRequest missing evidence."
    def invoke(req, folder, costs):
        observed.append(copy.deepcopy(req))
        return ReasoningReply({"answer": answer}, compiler.MODEL, compiler.CONFIG_DIGEST, 10, 10)
    monkeypatch.setattr(local_execution, "invoke_local", invoke)
    if fallback:
        monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    assert session.describe()["deliverySchema"] == {"type": "null"}
    if not fallback:
        assert prepared["authoring"]["requiredOutputSchema"]["properties"]["delivery"] == {"type": "null"}
    sid = prepared["session_id"]
    report = session.submit({"session_id": sid, "plan": None if fallback else choice, "delivery": None})
    assert report["deliveryContract"]["originalTask"] == request["task"]
    assert report["deliveryContract"]["requirements"] == []
    if fallback:
        session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})
        result = session.deliver({"session_id": sid, "response_json": json.dumps({"answer": answer})})
    else:
        result = session.draft({"session_id": sid})
        assert observed[0]["inputs"]["original_task"] == request["task"]
        assert observed[0]["tools"] == []
        assert observed[0]["outputSchema"] == report["deliveryResponseSchema"]
    assert len(observed) == (0 if fallback else 1)
    assert result["task"]["delivery"]["rendered"] == answer
    assert result["task"]["delivery"]["declaredCoverageComplete"] is None
    assert result["task"]["delivery"]["evidenceState"]["completedReads"] == 1
    assert not result["writeAuthority"] and result["task"]["success"] is None
    assert not session.draft({"session_id": sid})["executionReplayed"]
    with pytest.raises(PermissionError, match="frozen"):
        session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})


@pytest.mark.parametrize("fallback", [False, True])
def test_task_snapshot_duplicate_followup_has_no_new_receipt_or_provider_call(host, monkeypatch, fallback):
    root, profile, request, _, choice, observed = host
    profile["apiVersion"] = session.TASK_HOST_PROFILE
    (root / "host/profile.json").write_text(json.dumps(profile))
    if fallback:
        monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    metadata = prepared.get("observationModel", prepared.get("authoring", {}).get("observationModel"))
    assert metadata["kind"] == "immutable_session_snapshot" and not metadata["liveRefreshSupported"]
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": None if fallback else choice, "delivery": None})
    read = {"session_id": sid, "tool": "read_export", "arguments": request["arguments"]}
    if fallback:
        assert session.read(read)["status"] == "read_completed"
    report = session.read(read)
    assert report["status"] == "read_not_reexecuted" and not report["providerCalls"]
    assert not report["newObservation"] and "receipt" not in report
    assert report["evidenceState"]["completedReads"] == 1
    assert report["existingObservation"]["recordId"] == ("followup-0" if fallback else "prefix-0")
    assert not observed
    followups, attempts = session._followup_evidence(root / "sessions" / sid)
    assert len(followups) == int(fallback) and len(attempts) == 1 + int(fallback)
    if not fallback:
        assert session.read(read)["status"] == "read_not_reexecuted"
    with pytest.raises(PermissionError, match="exhausted"):
        session.read(read)


def test_task_snapshot_different_resource_is_not_a_duplicate(host):
    root, profile, request, _, choice, _ = host
    profile["apiVersion"] = session.TASK_HOST_PROFILE
    profile["resources"]["read_export"] = {"resources": [profile["resources"]["read_export"],
        [{"path": "/sandbox/other"}, {"text": "A distinct frozen payload."}]]}
    (root / "host/profile.json").write_text(json.dumps(profile))
    prepared = session.prepare(request)
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": choice, "delivery": None})
    fresh = session.read({"session_id": sid, "tool": "read_export", "arguments": {"path": "/sandbox/other"}})
    assert fresh["status"] == "read_completed" and len(fresh["providerCalls"]) == 1
    assert fresh["evidenceState"]["completedReads"] == 2
    denied = session.read({"session_id": sid, "tool": "read_export", "arguments": {"path": "/sandbox/forbidden"}})
    assert denied["status"] == "read_rejected" and not denied["providerCalls"]


def test_task_snapshot_unknown_attempt_blocks_before_any_deduplication(host):
    root, profile, request, _, choice, _ = host
    profile["apiVersion"] = session.TASK_HOST_PROFILE
    (root / "host/profile.json").write_text(json.dumps(profile))
    prepared = session.prepare(request)
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": choice, "delivery": None})
    write_artifacts(root / "sessions" / sid / "followup/attempt-0", {"request.json": {"unknown": True}})
    with pytest.raises(PermissionError, match="unknown previous read"):
        session.read({"session_id": sid, "tool": "read_export", "arguments": request["arguments"]})


@pytest.mark.parametrize("changed", [{"path": "/sandbox/snapshot", "refresh": True}, {"path": 1}])
def test_task_snapshot_identity_does_not_bypass_argument_schema(host, changed):
    root, profile, request, _, choice, _ = host
    profile["apiVersion"] = session.TASK_HOST_PROFILE
    (root / "host/profile.json").write_text(json.dumps(profile))
    prepared = session.prepare(request)
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": choice, "delivery": None})
    result = session.read({"session_id": sid, "tool": "read_export", "arguments": changed})
    assert result["status"] == "read_rejected" and not result["providerCalls"]


def test_snapshot_identity_is_exact_json_not_path_or_python_equality(monkeypatch):
    monkeypatch.setattr(session, "_evidence_state", lambda _: {"observationModel": {"snapshotId": "bound"},
        "records": [{"state": "read_completed", "tool": "lookup", "arguments": {"scope": "a", "page": 1},
                     "id": "bound-id", "recordDigest": "bound-report"}]})
    base = {"tool": "lookup", "arguments": {"page": 1, "scope": "a"}}
    assert session._recorded_snapshot_read(None, base)["recordId"] == "bound-id"
    for req in [{"tool": "other", "arguments": base["arguments"]},
                {"tool": "lookup", "arguments": {"scope": "a", "page": True}},
                {"tool": "lookup", "arguments": {"scope": "a", "page": 2}},
                {"tool": "lookup", "arguments": {"scope": "b", "page": 1}}]:
        assert session._recorded_snapshot_read(None, req) is None


def test_session_runs_original_read_engine_and_retains_unverified_l1(host):
    root, _, request, prepared, choice, observed = host
    result = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    assert result["route"] == "collecting_evidence" and result["task"]["status"] == "not_drafted"
    assert not observed and result["modelCalls"] == []
    result = session.draft({"session_id": prepared["session_id"]})
    assert result["route"] == "governed_hybrid_candidate"
    assert result["runtime"]["providerCalls"] == [{"tool": "read_export", "arguments": request["arguments"]}]
    assert result["translation"]["compiled"] and result["translation"]["originalTaskRetained"]
    assert result["translation"]["semanticFidelity"] == "not_assessed"
    assert result["task"]["success"] is None and result["task"]["status"] == "candidate_unverified"
    assert observed[0]["inputs"]["original_task"] == request["task"]
    assert observed[0]["tools"] == [] and observed[0]["runtimeAuthorityGranted"] is False
    assert not result["writeAuthority"] and result["runtime"]["effectCalls"] == 0
    assert session.inspect({"session_id": prepared["session_id"]}) == result
    assert result["evidenceFrozen"] and result["artifactChecks"]["semanticApproval"] is False
    state = result["task"]["delivery"]["evidenceState"]
    assert state["completedReads"] == 1 and state["collectionClosed"]
    assert state["contentSufficiency"] == "not_assessed" and not state["authorityGranted"]
    assert json.loads(observed[0]["inputs"]["host_read_state"]) == state
    assert (root / "sessions" / prepared["session_id"] / "prepared/compilation.json").is_file()


@pytest.mark.parametrize("key,value", [("confidence", 1), ("approval", True), ("policy", {})])
def test_agent_cannot_add_authority_fields(host, key, value):
    _, _, _, prepared, choice, observed = host
    with pytest.raises(ValueError, match="only session_id"):
        session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice, key: value})
    assert not observed


def test_outside_resources_stop_before_provider_and_model(host):
    root, _, _, prepared, choice, observed = host
    # Valid schema and source literal do not grant resource access.
    frozen = json.loads((root / "sessions" / prepared["session_id"] / "request.json").read_text())
    new = session.prepare({"task": frozen["packet"]["task"], "arguments": {"path": "/private/secret"}})
    result = session.submit({"delivery": DELIVERY, "session_id": new["session_id"], "plan": choice})
    assert result["route"] == "l1_fallback" and result["runtime"]["providerCalls"] == []
    assert not observed and result["handoff"]["writeAuthority"] is False
    with pytest.raises(PermissionError, match="prefix execution stopped"):
        session.read({"session_id": new["session_id"], "tool": "read_export", "arguments": {"path": "/sandbox/snapshot"}})


@pytest.mark.parametrize("plan", [{"mode": "proposal", "confidence": 1}, {"mode": "read_prefix", "reads": []}, "shell rm", {}])
def test_invalid_translation_hands_back_original_task_without_authority(host, plan):
    _, _, request, prepared, _, observed = host
    result = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": plan})
    assert result["route"] == "translation_needs_correction" and result["executed"] is False
    assert result["remainingProposalAttempts"] == 1 and not observed
    result = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": plan})
    assert result["route"] == "l1_fallback" and result["runtime"]["status"] == "not_executed"
    assert result["handoff"]["originalTask"] == request["task"] and not observed


def test_submission_is_one_shot_and_inspection_never_reexecutes(host):
    _, _, _, prepared, choice, observed = host
    assert session.inspect({"session_id": prepared["session_id"]})["route"] == "awaiting_agent_translation"
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    duplicate = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    assert duplicate["executionReplayed"] is False
    session.inspect({"session_id": prepared["session_id"]})
    assert not observed
    session.draft({"session_id": prepared["session_id"]})
    duplicate = session.draft({"session_id": prepared["session_id"]})
    assert not duplicate["executionReplayed"] and len(observed) == 1


def test_host_disabled_or_drift_never_executes(host, monkeypatch):
    root, profile, _, prepared, choice, observed = host
    profile["resources"]["read_export"][1]["text"] = "Changed snapshot"
    (root / "host/profile.json").write_text(json.dumps(profile))
    with pytest.raises(PermissionError, match="changed"):
        session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    monkeypatch.delenv("NETOPYU_HYBRID_HOST_PROFILE")
    with pytest.raises(PermissionError, match="disabled"):
        session.prepare({"task": "Explain this snapshot please", "arguments": {"path": "/x"}})
    assert not observed


def test_model_error_never_becomes_success_or_retries(host, monkeypatch):
    _, _, _, prepared, choice, _ = host
    calls = []
    def fail(*args):
        calls.append(1)
        raise TimeoutError("unknown local reply")
    monkeypatch.setattr(local_execution, "invoke_local", fail)
    result = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    assert result["route"] == "collecting_evidence" and not calls
    result = session.draft({"session_id": prepared["session_id"]})
    assert result["route"] == "draft_stopped_no_retry" and result["task"]["success"] is None
    assert len(result["runtime"]["providerCalls"]) == 1 and len(calls) == 1
    assert not session.draft({"session_id": prepared["session_id"]})["executionReplayed"]
    assert len(calls) == 1


def test_product_imports_no_evaluator_in_fresh_process():
    import subprocess
    p = subprocess.run([sys.executable, "-c", "import dsh_adapter.hybrid_session, sys; assert not any(n == 'evaluation' or n.startswith('evaluation.') for n in sys.modules)"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_native_l1_followup_uses_same_gateway_and_budget(host):
    _, _, request, prepared, choice, observed = host
    with pytest.raises(PermissionError, match="pending"):
        session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    first = session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    assert first["status"] == "read_completed" and first["receipt"]["payload"]["text"] == "Pia has not approved."
    assert first["deliveryAction"]["tool"] == "netopyu_hybrid_draft"
    assert first["evidenceState"]["completedReads"] == 2 and not first["evidenceState"]["collectionClosed"]
    assert first["remainingReadAttempts"] == 1 and not observed
    denied = session.read({"session_id": prepared["session_id"], "tool": "shell", "arguments": {}})
    assert denied["status"] == "read_rejected" and not denied["providerCalls"]
    assert denied["evidenceState"]["completedReads"] == 2
    assert denied["evidenceState"]["records"][-1]["state"] == "read_rejected"
    with pytest.raises(PermissionError, match="exhausted"):
        session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    session.draft({"session_id": prepared["session_id"]})
    assert observed[0]["inputs"]["priorReadResults"][0]["result"]["read_export"]["text"] == "Pia has not approved."
    assert len(observed) == 1
    with pytest.raises(PermissionError, match="frozen"):
        session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})


def test_async_worker_routes_same_session_functions(host):
    import asyncio
    from dsh_adapter.worker import dispatch
    _, _, request, _, _, _ = host
    result = asyncio.run(dispatch({"command": "hybrid-prepare", "args": request, "profile": "lan"}))
    assert result["route"] == "awaiting_agent_translation"


def test_source_overflow_preserves_real_fallback_read_channel(host, monkeypatch):
    _, _, request, _, _, observed = host
    monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    assert prepared["route"] == "l1_fallback" and prepared["sourceRetainedOnHost"]
    assert prepared["sourceSuppliedToModel"] is False
    session.submit({"session_id": prepared["session_id"], "plan": None, "delivery": DELIVERY})
    result = session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    assert result["status"] == "read_completed" and not observed


def test_first_class_tool_schema_requires_bindings_not_plain_path(host):
    from jsonschema import Draft202012Validator, ValidationError
    _, _, _, _, choice, _ = host
    schema = session.describe()["planSchema"]
    Draft202012Validator(schema).validate(choice)
    choice["reads"][0]["arguments"]["path"] = "/sandbox/snapshot"
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(choice)


def test_empty_prefix_is_not_a_fake_read_or_a_completed_task(host):
    _, _, _, prepared, choice, observed = host
    choice["reads"] = []
    result = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    assert result["runtime"]["status"] == "no_prefix_reads"
    assert result["observations"] == {} and not observed
    result = session.draft({"session_id": prepared["session_id"]})
    assert len(observed) == 1 and result["task"]["success"] is None
    assert not result["runtime"]["providerCalls"]


def test_draft_cannot_accept_agent_evidence_or_approval(host):
    _, _, _, prepared, _, observed = host
    with pytest.raises(ValueError, match="only session_id"):
        session.draft({"session_id": prepared["session_id"], "evidence": "invented", "approval": True})
    assert not observed


def test_unknown_read_blocks_further_reads_and_drafting(host):
    root, _, request, prepared, choice, observed = host
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    folder = root / "sessions" / prepared["session_id"]
    write_artifacts(folder / "followup/attempt-0", {"request.json": session.seal(request)})
    with pytest.raises(PermissionError, match="unknown previous read"):
        session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    with pytest.raises(FileNotFoundError):
        session.draft({"session_id": prepared["session_id"]})
    assert not observed


def test_changed_evidence_cannot_reach_reasoner(host):
    root, _, _, prepared, choice, observed = host
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    report_path = root / "sessions" / prepared["session_id"] / "prefix/summary/report.json"
    report = json.loads(report_path.read_text())
    report["execution"]["outputs"]["n0"]["value"]["observations"]["read"]["text"] = "Invented approval"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="drift"):
        session.draft({"session_id": prepared["session_id"]})
    assert not observed


def test_freeze_claim_blocks_late_reads_even_if_execution_is_unknown(host):
    root, _, request, prepared, choice, observed = host
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    (root / "sessions" / prepared["session_id"] / "draft").mkdir()
    assert session.inspect({"session_id": prepared["session_id"]})["route"] == "pending_or_unknown_no_retry"
    assert not session.draft({"session_id": prepared["session_id"]})["executionReplayed"]
    with pytest.raises(PermissionError, match="frozen"):
        session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    assert not observed


def test_read_and_freeze_share_one_cross_process_lock(host):
    root, _, request, prepared, choice, observed = host
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    with session._locked(root / "sessions" / prepared["session_id"]):
        with pytest.raises(BlockingIOError):
            session.draft({"session_id": prepared["session_id"]})
        with pytest.raises(BlockingIOError):
            session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    assert not observed


def test_fallback_does_not_gain_draft_authority(host):
    _, _, _, prepared, _, observed = host
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": {}})
    session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": {}})
    with pytest.raises(PermissionError, match="fallback"):
        session.draft({"session_id": prepared["session_id"]})
    assert not observed


def test_compile_correction_does_not_replay_reads_or_relax_contract(host):
    root, _, _, prepared, choice, observed = host
    bad = copy.deepcopy(choice)
    bad["reads"][0]["arguments"]["path"] = {"literal": "/sandbox/snapshot", "origin": "task", "quote": "/sandbox/snapshot"}
    rejected = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": bad})
    assert rejected["route"] == "translation_needs_correction" and rejected["providerCalls"] == []
    assert session.inspect({"session_id": prepared["session_id"]}) == rejected
    accepted = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    assert len(accepted["runtime"]["providerCalls"]) == 1 and not observed
    assert not session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})["executionReplayed"]
    session.draft({"session_id": prepared["session_id"]})
    assert len(observed) == 1
    assert len(list((root / "sessions" / prepared["session_id"] / "proposals").glob("attempt-*"))) == 2


def test_unknown_proposal_never_allows_correction(host):
    root, _, _, prepared, choice, observed = host
    write_artifacts(root / "sessions" / prepared["session_id"] / "proposals/attempt-0", {"plan.json": choice})
    result = session.submit({"delivery": DELIVERY, "session_id": prepared["session_id"], "plan": choice})
    assert result["route"] == "pending_or_unknown_no_retry" and not result["executionReplayed"]
    assert not observed


def test_native_fallback_uses_same_delivery_contract_without_runtime_model(host, monkeypatch):
    _, _, request, _, _, observed = host
    monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    with pytest.raises(PermissionError, match="delivery"):
        session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    bound = session.submit({"session_id": prepared["session_id"], "plan": None, "delivery": DELIVERY})
    assert not bound["translation"]["compiled"] and bound["deliveryContract"]["quoteMembershipChecked"]
    assert bound["deliveryAction"]["tool"] == "netopyu_hybrid_deliver"
    session.read({"session_id": prepared["session_id"], "tool": "read_export", "arguments": request["arguments"]})
    candidate = {"delivery": {"d0": {"state": "provided", "content": {"text": "Pia has not approved."}, "gap": ""}}, "uncertainties": []}
    result = session.deliver({"session_id": prepared["session_id"], "response_json": json.dumps(candidate)})
    assert result["route"] == "native_l1_delivery_candidate" and not observed and not result["modelCalls"]
    assert "Pia has not approved" in result["task"]["delivery"]["rendered"]
    assert result["task"]["success"] is None and result["evidenceFrozen"]
    assert result["transport"] == "strict_json_text_object/v1" and result["diagnostic"] is None
    assert result["task"]["delivery"]["evidenceState"]["completedReads"] == 1
    assert not session.deliver({"session_id": prepared["session_id"], "response_json": json.dumps(candidate)})["executionReplayed"]


def test_native_candidate_cannot_override_admitted_runtime_graph(host):
    _, _, _, prepared, choice, observed = host
    session.submit({"session_id": prepared["session_id"], "plan": choice, "delivery": DELIVERY})
    with pytest.raises(PermissionError, match="only accepted"):
        session.deliver({"session_id": prepared["session_id"], "response_json": "{}"})
    assert not observed


def test_runtime_draft_never_interprets_response_as_a_route_choice(host):
    _, _, _, prepared, choice, observed = host
    sid = prepared["session_id"]
    admitted = session.submit({"session_id": sid, "plan": choice, "delivery": DELIVERY})
    assert admitted["deliveryAction"]["arguments"] == {"session_id": sid}
    with pytest.raises(ValueError, match="omit response"):
        session.draft({"session_id": sid, "response": {}})
    duplicate = session.submit({"session_id": sid, "plan": None, "delivery": DELIVERY})
    assert duplicate["submissionIgnored"] and duplicate["existingSessionRoute"] == "collecting_evidence"
    assert not observed
    assert session.draft({"session_id": sid})["route"] == "governed_hybrid_candidate"
    assert len(observed) == 1


def test_worker_has_distinct_native_delivery_dispatch(host, monkeypatch):
    import asyncio
    from dsh_adapter.worker import dispatch
    received = []
    monkeypatch.setattr(session, "deliver", lambda request: received.append(request) or {"route": "native_text_checked"})
    request = {"session_id": "test", "response_json": "{}"}
    result = asyncio.run(dispatch({"command": "hybrid-deliver", "args": request, "profile": "lan"}))
    assert received == [request] and result["route"] == "native_text_checked"


@pytest.mark.parametrize("wire,pointer", [('{"x":1,"x":2}', "/response_json"),
    (json.dumps({"delivery": {"d0": {"state": "provided", "content": {"text": "Observed"}}}, "uncertainties": []}), "/delivery/d0/gap"),
    (json.dumps({"delivery": {"d0": {"state": "provided", "content": {"text": 3}, "gap": ""}}, "uncertainties": []}), "/delivery/d0/content/text")])
def test_native_failure_has_precise_diagnostics_and_never_retries(host, monkeypatch, wire, pointer):
    _, _, request, _, _, observed = host
    monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    sid = prepared["session_id"]
    session.submit({"session_id": sid, "plan": None, "delivery": DELIVERY})
    result = session.deliver({"session_id": sid, "response_json": wire})
    assert result["task"]["status"] == "not_completed" and result["task"]["delivery"] is None
    assert result["diagnostic"]["pointer"] == pointer and not result["modelCalls"] and not observed
    assert not result["retryAllowed"] and "no validated delivery" in result["guidance"]
    repeated = session.deliver({"session_id": sid, "response_json": "{}"})
    assert not repeated["executionReplayed"] and not repeated["retryAllowed"]
    assert repeated["existing"] == result and "not proof it passed" in repeated["guidance"]


def test_contract_binding_detects_resealed_contract_swap(host):
    root, _, _, prepared, choice, observed = host
    session.submit({"session_id": prepared["session_id"], "plan": choice, "delivery": DELIVERY})
    path = root / "sessions" / prepared["session_id"] / "delivery-contract/report.json"
    contract = json.loads(path.read_text())
    contract["requirements"][0]["kind"] = "artifact"
    path.write_text(json.dumps(session.seal({k: v for k, v in contract.items() if k != "reportDigest"})))
    with pytest.raises(ValueError, match="binding drift"):
        session.draft({"session_id": prepared["session_id"]})
    assert not observed


def test_fallback_cannot_quote_source_pages_not_supplied_to_model(host, monkeypatch):
    _, _, request, _, _, observed = host
    monkeypatch.setattr(session, "budget", lambda _: {"accepted": False})
    prepared = session.prepare(request)
    bad = copy.deepcopy(DELIVERY)
    bad["requirements"][0]["origin"] = "p000"
    for remaining in (1, 0):
        rejected = session.submit({"session_id": prepared["session_id"], "plan": None, "delivery": bad})
        assert rejected["remainingProposalAttempts"] == remaining and not rejected["executed"]
    with pytest.raises(PermissionError, match="budget"):
        session.submit({"session_id": prepared["session_id"], "plan": None, "delivery": DELIVERY})
    assert not observed
