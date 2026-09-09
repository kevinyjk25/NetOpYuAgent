"""Plan/fill isolation and closure, not an oracle for model source semantics."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_ledger as ledger, source_plan
from evaluation.source_blocks import citation_blocks
from evaluation.structured_flow_demo import fixture
from tests.test_source_catalog import literal
from tests.test_source_ledger import envelope


@pytest.fixture
def packet():
    bundle, tree, reads, _ = fixture()
    return {"bundle": bundle, "task": "Read lab-edge-42, retain all original prerequisites and unconverted duties.",
        "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
        "catalog": {"tools": [json.loads(next(s.text for s in c.spec.sources if s.role == "tool")) for c in reads.values()]},
        "reads": {name: c.model_dump(by_alias=True, mode="json") for name, c in reads.items()}}


def request(packet):
    state = ledger.initial_state(packet, "plan_first")
    return state, ledger.make_request(packet, state)[0]


def choice(wire):
    content = json.loads(wire["messages"][1]["content"])
    mark = {"block_id": next(b["id"] for b in content["sourceBlocks"] if "先按" in b["text"])}
    return {"mode": "operation_plan", "purpose": "A bounded read region; remaining logic needs review.",
        "region": {"steps": [{"kind": "read", "source": mark, "tool": "get_interfaces",
                              "whyNeeded": "The source requires observing interfaces before further decisions."}],
                   "exit": end(mark)}, "issues": [], "remaining": []}


def end(mark):
    return {"kind": "end", "source": copy.deepcopy(mark), "outcome": "needs_l1",
            "explanation": "Unconverted reasoning remains; not full-Skill success."}


def prepared(packet):
    state, wire = request(packet)
    value = choice(wire)
    blocks = citation_blocks(ledger.frame(packet, state))
    plan = source_plan.prepare(value, packet, blocks)
    return state, wire, plan


def arguments(wire):
    context = json.loads(wire["messages"][1]["content"])["frozenPlan"]
    slot = context["currentRead"]
    return {"mode": "planned_arguments", "planDigest": context["planDigest"], "readPointer": slot["treePointer"],
            "arguments": {"kind": "object", "fields": {"device": {"kind": "reference", "source": "input", "pointer": "/device"}}}}


def test_plan_schema_excludes_arguments_requires_exit_and_exact_tools(packet):
    _, wire = request(packet)
    value = choice(wire)
    validator = Draft202012Validator(wire["format"])
    assert validator.is_valid(value)
    for mutate in (lambda v: v["region"].pop("exit"),
                   lambda v: v["region"]["steps"][0].update(arguments={}),
                   lambda v: v["region"]["steps"][0].update(tool="invented")):
        bad = copy.deepcopy(value)
        mutate(bad)
        assert not validator.is_valid(bad)


def test_current_task_is_distinct_from_inert_source_and_host_prose(packet):
    _, wire = request(packet)
    content = json.loads(wire["messages"][1]["content"])
    assert list(content)[-2:] == ["currentTask", "requiredOutputSchema"]
    assert content["requiredOutputSchema"] == wire["format"]
    assert content["currentTask"]["text"] == packet["task"]
    assert "task" not in content
    assert "FUTURE external observation" in wire["messages"][0]["content"]


def scenario(packet):
    return {"apiVersion": "netopyu.io/authoring-scenario/v1", "taskDigest": ledger.prior.sha256_json(packet["task"]),
        "futureTask": "Observe the interfaces of lab-edge-42, subject to the original access and source requirements.",
        "reviewKind": "caller_supplied_scenario_not_independent_gold"}


def test_scenario_is_task_bound_separate_and_cannot_enable_execution(packet):
    supplied = scenario(packet)
    state = ledger.initial_state(packet, "plan_first", scenario=supplied)
    supplied["futureTask"] = "mutated by caller"
    wire, _ = ledger.make_request(packet, state)
    content = json.loads(wire["messages"][1]["content"])
    assert content["futureScenario"]["futureTask"] == scenario(packet)["futureTask"]
    assert content["currentTask"]["text"] == packet["task"]
    assert not content["futureScenario"]["taskEquivalenceProven"]
    assert not content["futureScenario"]["runtimeAuthorityGranted"]


@pytest.mark.parametrize("field,value", [("taskDigest", "sha256:wrong"), ("futureTask", ""),
    ("reviewKind", "independent_gold"), ("runtimeAuthorityGranted", True)])
def test_invalid_scenario_rejected(packet, field, value):
    supplied = scenario(packet)
    supplied[field] = value
    with pytest.raises(ValueError, match="task-bound"):
        ledger.initial_state(packet, "plan_first", scenario=supplied)


def test_scenario_cannot_silently_change_default_direct_profile(packet):
    with pytest.raises(ValueError, match="requires plan_first"):
        ledger.initial_state(packet, scenario=scenario(packet))


def test_reasoning_is_explicit_and_budgeted_without_changing_plan_input(packet):
    off = ledger.initial_state(packet, "plan_first", scenario=scenario(packet))
    on = ledger.initial_state(packet, "plan_first", scenario=scenario(packet), reasoning=True)
    first, before = ledger.make_request(packet, off)
    second, after = ledger.make_request(packet, on)
    assert not first["think"] and second["think"]
    assert first["messages"] == second["messages"] and first["format"] == second["format"]
    assert first["options"]["num_predict"] == 4096 and second["options"]["num_predict"] == 8192
    assert before["proxyLimit"] - after["proxyLimit"] == 4096
    assert after["outputTokenReserve"] == 8192


@pytest.mark.parametrize("profile,reasoning", [("direct", True), ("plan_first", "true"), ("plan_first", 1)])
def test_reasoning_cannot_change_other_profiles_or_accept_truthy_values(packet, profile, reasoning):
    with pytest.raises(ValueError, match="reasoning"):
        ledger.initial_state(packet, profile, reasoning=reasoning)


@pytest.mark.parametrize("present", [True, False])
def test_declared_operation_requires_visible_occurrence_not_unrelated_block(packet, present):
    state, wire = request(packet)
    value = choice(wire)
    content = json.loads(wire["messages"][1]["content"])
    blocks = citation_blocks(ledger.frame(packet, state))
    mark = value["region"]["steps"][0]["source"]
    operation = "unavailable_operation" if not present else blocks[mark["block_id"]]["text"][:8]
    schema = source_plan.planning_schema(blocks, packet["catalog"], [], wire["format"]["oneOf"][0],
        wire["format"]["oneOf"][1], [{"hostTool": "get_interfaces", "sourceOperation": operation}])
    validator = Draft202012Validator(schema)
    Draft202012Validator.check_schema(schema)
    value["region"]["steps"][0]["sourceOperation"] = operation
    assert validator.is_valid(value) is present
    if present:
        plan = source_plan.prepare(value, packet, blocks)
        assert plan["reads"][0]["sourceOperation"] == operation
        assert "sourceOperation" not in plan["tree"]["steps"][0]
        assert not plan["semanticEntailmentProven"]
    value["region"]["steps"] = []
    assert validator.is_valid(value)  # explicit abstention remains available
    assert content["currentTask"]["text"] == packet["task"]


def test_future_execution_gates_do_not_become_compiler_added_issues(packet):
    _, _, plan = prepared(packet)
    assert not plan["tree"]["unresolved"]
    assert plan["structurallyClosed"] and not plan["runtimeAuthorityGranted"]
    assert not plan["semanticEntailmentProven"]
    assert len(plan["reads"]) == 1


def test_explicit_issues_preserved_and_no_argument_call_allowed(packet):
    state, wire = request(packet)
    value = choice(wire)
    value["issues"] = [{"source": value["region"]["steps"][0]["source"], "explanation": "This necessary source condition has no available observation."}]
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "operation_plan_has_unresolved_issues"
    assert files["prepared-plan.json"]["tree"]["unresolved"]
    assert "next-state.json" not in files


@pytest.mark.parametrize("exit_kind", ["continue", "already_closed"])
def test_open_root_cannot_be_closed_by_claim(packet, exit_kind):
    state, wire = request(packet)
    value = choice(wire)
    value["region"]["exit"] = {"kind": exit_kind}
    with pytest.raises(ValueError):
        source_plan.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))


def branch_plan(wire, *, closed):
    value = choice(wire)
    mark = value["region"]["steps"][0]["source"]
    value["region"]["steps"].append({"kind": "if_equal", "source": mark,
        "left": {"kind": "reference", "source": "/steps/0", "pointer": "/interfaces/0/adminUp"}, "equals": True,
        "when_equal": {"steps": [], "exit": end(mark) if closed else {"kind": "continue"}},
        "otherwise": {"steps": [], "exit": end(mark)}})
    if closed:
        value["region"]["exit"] = {"kind": "already_closed"}
    return value


@pytest.mark.parametrize("closed", [True, False])
def test_branch_closure_preserves_explicit_control_flow(packet, closed):
    state, wire = request(packet)
    value = branch_plan(wire, closed=closed)
    assert Draft202012Validator(wire["format"]).is_valid(value)
    plan = source_plan.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))
    steps = plan["tree"]["steps"]
    assert len(steps) == (2 if closed else 3)
    assert steps[1]["otherwise"][0]["kind"] == "end"
    assert len(steps[1]["when_equal"]) == int(closed)


def test_unreachable_terminal_normalizes_without_inserting_reachable_success(packet):
    state, wire = request(packet)
    value = branch_plan(wire, closed=True)
    value["region"]["exit"] = end(value["region"]["steps"][0]["source"])
    before = copy.deepcopy(value)
    plan = source_plan.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))
    assert value == before == plan["choice"]
    assert len(plan["tree"]["steps"]) == 2
    assert plan["tree"]["steps"][-1]["kind"] == "if_equal"
    assert plan["normalizations"][0]["original"] == value["region"]["exit"]
    assert plan["normalizations"][0]["kind"] == "unreachable_redundant_terminal"
    canonical = branch_plan(wire, closed=True)
    expected = source_plan.prepare(canonical, packet, citation_blocks(ledger.frame(packet, state)))
    assert plan["tree"] == expected["tree"]


def test_unreachable_read_is_still_rejected_not_deduplicated(packet):
    state, wire = request(packet)
    value = branch_plan(wire, closed=True)
    value["region"]["steps"].append(copy.deepcopy(value["region"]["steps"][0]))
    with pytest.raises(ValueError, match="unreachable"):
        source_plan.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))


def test_operation_modes_cannot_change_between_plan_and_compile(packet):
    _, _, plan = prepared(packet)
    with pytest.raises(ValueError, match="drift"):
        source_plan.compile_plan(packet, plan, [], [{"changed": "declaration"}])


def test_declared_mode_and_dynamic_leaves_survive_full_plan_pipeline(packet):
    from evaluation import source_modes
    tool = packet["catalog"]["tools"][0]
    mode = {"hostTool": tool["name"], "catalogDigest": ledger.prior.sha256_json(packet["catalog"]),
        "contractHash": packet["reads"][tool["name"]]["contractHash"],
        "inputSchemaDigest": ledger.prior.sha256_json(tool["inputSchema"]),
        "outputSchemaDigest": ledger.prior.sha256_json(tool["outputSchema"]),
        "reviewKind": source_modes.REVIEW_KIND, "declarationEvidence": "Mechanical fixture mode keeps explicit device id construction.",
        "modes": [{"id": "explicit_device", "description": "Requires the device id key, no semantic or access assurance.",
                   "objects": [{"objectPointer": "/device", "requiredKeys": ["id"], "allowedKeys": ["id"]}]}]}
    state = ledger.initial_state(packet, "plan_first", [mode])
    wire, _ = ledger.make_request(packet, state)
    value = choice(wire)
    value["region"]["steps"][0]["operationMode"] = "explicit_device"
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "operation_plan_recorded"
    state = files["next-state.json"]
    wire, _ = ledger.make_request(packet, state)
    value = arguments(wire)
    value["arguments"]["fields"]["device"] = {"kind": "object", "fields": {
        "id": {"kind": "reference", "source": "input", "pointer": "/device/id"}}}
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "compiled_region_requires_semantic_review", result
    assert not files["argument-origin-check.json"]["issues"]
    assert files["plan-to-tree.json"]["runtimeAuthorityGranted"] is False


@pytest.mark.parametrize("source,pointer", [("/steps/2", "/interfaces/0/adminUp"), ("/steps/1", "/interfaces/0/adminUp"),
    ("/steps/1/otherwise/0", "/interfaces/0/adminUp"), ("/steps/0", "/interfaces")])
def test_planned_branch_rejects_non_dominating_or_non_scalar_reference(packet, source, pointer):
    state, wire = request(packet)
    value = branch_plan(wire, closed=True)
    value["region"]["steps"][1]["left"].update(source=source, pointer=pointer)
    with pytest.raises(ValueError):
        source_plan.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))


def test_duplicate_planned_reads_not_silently_removed(packet):
    state, wire = request(packet)
    value = choice(wire)
    value["region"]["steps"] *= 2
    plan = source_plan.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))
    assert len(plan["reads"]) == 2
    assert plan["reads"][1]["dominatingReadPointers"] == ["/steps/0"]


def test_empty_plan_is_abstention_not_success(packet):
    state, wire = request(packet)
    value = choice(wire)
    value["region"]["steps"] = []
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "operation_plan_defers_without_reads"
    assert "compilation.json" not in files and "next-state.json" not in files


def test_plan_then_one_slot_compiles_through_original_engine(packet):
    state, wire = request(packet)
    first, outcome = ledger.derive(packet, state, wire, envelope(choice(wire)))
    assert outcome["candidateStatus"] == "operation_plan_recorded"
    next_state = first["next-state.json"]
    arg_wire, _ = ledger.make_request(packet, next_state)
    value = arguments(arg_wire)
    assert Draft202012Validator(arg_wire["format"]).is_valid(value)
    files, result = ledger.derive(packet, next_state, arg_wire, envelope(value))
    assert result["candidateStatus"] == "compiled_region_requires_semantic_review", result
    assert files["tree.json"]["steps"][-1]["outcome"] == "needs_l1"
    assert files["plan-to-tree.json"]["planDigest"] == first["prepared-plan.json"]["reportDigest"]
    assert not files["compilation.json"]["runtimeAuthorityGranted"]


@pytest.mark.parametrize("change", ["plan", "slot", "extra_read", "future", "whole_object_origin"])
def test_argument_phase_cannot_rewrite_plan_or_skip_provenance(packet, change):
    state, wire = request(packet)
    files, _ = ledger.derive(packet, state, wire, envelope(choice(wire)))
    arg_wire, _ = ledger.make_request(packet, files["next-state.json"])
    value = arguments(arg_wire)
    if change == "plan":
        value["planDigest"] = "different"
    elif change == "slot":
        value["readPointer"] = "/steps/1"
    elif change == "extra_read":
        value["tool"] = "another_api"
    elif change == "future":
        value["arguments"]["fields"]["device"]["source"] = "/steps/0"
    else:
        value["arguments"] = literal({"device": {"id": "lab-edge-42"}})
    assert not Draft202012Validator(arg_wire["format"]).is_valid(value)


def test_binding_navigation_contains_only_lexically_available_schema_paths(packet):
    _, _, plan = prepared(packet)
    view = source_plan.binding_sources(packet, plan, plan["reads"][0])
    assert not view["runtimeValuesProvided"] and not view["targetMappingInferred"]
    refs = [r["reference"] for r in view["paths"]]
    assert {"kind": "reference", "source": "input", "pointer": "/device/id"} in refs
    assert {r["source"] for r in refs} == {"input"}


def test_invented_literal_stops_before_the_next_parameter_call(packet):
    state, wire = request(packet)
    plan = choice(wire)
    plan["region"]["steps"] *= 2
    first, _ = ledger.derive(packet, state, wire, envelope(plan))
    state = first["next-state.json"]
    wire, _ = ledger.make_request(packet, state)
    value = arguments(wire)
    mark = json.loads(wire["messages"][1]["content"])["sourceBlocks"][0]
    value["arguments"]["fields"]["device"] = {"kind": "object", "fields": {"id": literal("invented-device",
        {"kind": "source", "block_id": mark["id"], "quote": mark["text"]})}}
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "argument_slot_origin_rejected"
    assert "next-state.json" not in files and "compilation.json" not in files
    assert files["argument-origin-check.json"]["issues"][0]["code"] == "literal_origin"
    assert files["planned-arguments.json"]["arguments"] == value["arguments"]


def test_all_slots_bind_before_compilation_and_replay_is_zero_call(packet, tmp_path, monkeypatch):
    from evaluation import flow_checkpoint
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "fixture"})
    calls = []
    def send(arm, wire):
        calls.append(wire)
        phase = json.loads(wire["messages"][1]["content"])["authoringPhase"]
        if phase == "operation_plan_without_arguments":
            value = choice(wire)
            value["region"]["steps"] *= 2
        else:
            value = arguments(wire)
        return envelope(value)
    monkeypatch.setattr(flow_checkpoint, "send", send)
    root = tmp_path / "plan-run"
    ledger.freeze(packet, root, profile="plan_first")
    partial = ledger.run(root, max_new_calls=2)
    assert not partial["compiled"] and partial["planProduced"] and partial["argumentSlotsCompleted"] == 1
    report = ledger.run(root, max_new_calls=1)
    assert report["compiled"] and report["compiledReadNodes"] == 2
    assert report["argumentSlotsCompleted"] == 2 and report["modelCallsRecorded"] == 3
    assert ledger.run(root, max_new_calls=0) == report and len(calls) == 3


def test_assembly_preserves_old_and_new_source_windows_in_separate_namespaces(packet):
    _, _, plan = prepared(packet)
    blocks = copy.deepcopy(plan["blocks"])
    key = next(iter(blocks))
    # Same source ID in another response refers to a different original span.
    other = next(k for k in blocks if k != key)
    blocks[key] = blocks[other]
    row = {"planDigest": plan["reportDigest"], "readPointer": "/steps/0", "blocks": blocks,
           "arguments": {"kind": "object", "fields": {"device": {"kind": "reference", "source": "input", "pointer": "/device"}}}}
    tree, combined = source_plan.assemble(plan, [row])
    assert combined["plan_" + key] == plan["blocks"][key]
    assert combined["argument0_" + key] == blocks[other]
    assert tree["steps"][0]["source"]["block_id"].startswith("plan_")


@pytest.mark.parametrize("change", ["digest", "slot", "count"])
def test_assembly_rejects_drift_and_incomplete_arguments(packet, change):
    _, _, plan = prepared(packet)
    rows = [{"planDigest": plan["reportDigest"], "readPointer": "/steps/0", "blocks": plan["blocks"],
             "arguments": {"kind": "reference", "source": "input", "pointer": ""}}]
    if change == "digest":
        plan["tree"]["purpose"] = "changed"
    elif change == "slot":
        rows[0]["readPointer"] = "/steps/9"
    else:
        rows = []
    with pytest.raises(ValueError):
        source_plan.assemble(plan, rows)


@pytest.mark.parametrize("field", ["task", "catalog", "bundle"])
def test_compilation_binds_source_task_and_host(packet, field):
    _, _, plan = prepared(packet)
    if field == "task":
        packet["task"] += " changed"
    elif field == "catalog":
        packet["catalog"]["description"] = "changed"
    else:
        packet["bundle"]["bundleDigest"] = "different"
    with pytest.raises(ValueError, match="drift"):
        source_plan.compile_plan(packet, plan, [], [])


def test_host_constant_literal_origin_is_constructor_bound_not_model_quotation():
    from evaluation.source_candidate_schema import tighten
    from evaluation.source_catalog import constrain
    from evaluation.structured_authoring import response_schema
    bundle, tree, _, _ = fixture()
    schema = response_schema({"bundle": bundle, "inputSchema": tree.input_schema}, {}, {})
    tighten(schema, ["fixed_api"])
    schema["$defs"]["SourceSpan"] = {"type": "object"}
    constrain(schema, {"tools": [{"name": "fixed_api", "inputSchema": {"type": "string", "const": "fixed"}}]}, host_constants=True)
    schema = {"$defs": schema["$defs"], "$ref": "#/$defs/StructuredTreeRead"}
    value = {"kind": "read", "source": {}, "tool": "fixed_api", "arguments": literal("fixed", {"kind": "host_schema", "pointer": "/const"})}
    assert Draft202012Validator(schema).is_valid(value)
    value["arguments"]["origin"] = {"kind": "source", "block_id": "invented", "quote": "fixed"}
    assert not Draft202012Validator(schema).is_valid(value)


@pytest.mark.parametrize("nested", [False, True])
def test_equal_host_constants_keep_their_distinct_original_schema_locations(nested):
    from evaluation.source_candidate_schema import tighten
    from evaluation.source_catalog import constrain
    from evaluation.structured_authoring import response_schema
    bundle, tree, _, _ = fixture()
    data = {"type": "string", "const": "same"}
    if nested:
        data = {"type": "object", "properties": {"value": data}, "required": ["value"], "additionalProperties": False}
    target = {"type": "object", "properties": {"a": data, "b": copy.deepcopy(data)},
              "required": ["a", "b"], "additionalProperties": False}
    schema = response_schema({"bundle": bundle, "inputSchema": tree.input_schema}, {}, {})
    tighten(schema, ["fixed_api"])
    schema["$defs"]["SourceSpan"] = {"type": "object"}
    constrain(schema, {"tools": [{"name": "fixed_api", "inputSchema": target}]}, host_constants=True)
    validator = Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/StructuredTreeRead"})
    fields = {}
    for key in ("a", "b"):
        path = f"/properties/{key}" + ("/properties/value" if nested else "") + "/const"
        expr = literal("same", {"kind": "host_schema", "pointer": path})
        fields[key] = {"kind": "object", "fields": {"value": expr}} if nested else expr
    value = {"kind": "read", "tool": "fixed_api", "source": {}, "arguments": {"kind": "object", "fields": fields}}
    assert validator.is_valid(value)
    b = fields["b"]["fields"]["value"] if nested else fields["b"]
    b["origin"]["pointer"] = b["origin"]["pointer"].replace("/b/", "/a/")
    assert not validator.is_valid(value)
