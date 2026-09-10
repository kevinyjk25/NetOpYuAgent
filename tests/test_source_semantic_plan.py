"""Typed semantic frontend regression; hand-authored mechanics, not model scores."""
import ast
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_ledger as ledger, source_plan, source_program_lines
from evaluation.source_blocks import citation_blocks
from network_runtime.l0.structured_bindings import compile_binding, materialize_binding, verify_binding
from network_runtime.contracts import sha256_json
from tests.test_source_obligations import packet as packet_fixture
from tests.test_source_plan import choice
from tests.test_source_ledger import envelope as model_envelope

packet = packet_fixture


def wire_choice(value):
    """Encode manually authored test statements in the current wire grammar.

    This is test construction only, never a source translator or model prompt.
    Invalid suffixes remain present for negative schema tests, not discarded.
    """
    def node(rows):
        if not rows:
            return None
        row = copy.deepcopy(rows[0])
        if row["op"] in {"complete", "handoff"}:
            row.pop("explanation", None)  # current wire has fixed control-status labels
        if row["op"] in source_program_lines.BRANCHES:
            for key in ("when_equal", "otherwise"):
                row[key] = node(row[key])
        if len(rows) > 1 or row["op"] == "read":
            row["next"] = node(rows[1:])
        return row
    value = copy.deepcopy(value)
    if value.get("mode") == "operation_plan" and isinstance(value.get("program"), list):
        value["program"] = node(value["program"])
    return value


def envelope(value):
    return model_envelope(wire_choice(value))


def typed_program(text):
    """Test author convenience only; never translates a Skill or enters model input."""
    def reference(node):
        if isinstance(node, ast.Name):
            return {"kind": "alias", "name": node.id}
        return {"kind": node.func.id, "source": node.args[0].id, "pointer": ast.literal_eval(node.args[1])}
    def block(body):
        rows = []
        for node in body:
            base = {}
            if isinstance(node, ast.Assign):
                call = node.value
                if call.func.id == "read":
                    rows.append({**base, "op": "read", "name": node.targets[0].id, "tool": ast.literal_eval(call.args[0])})
                else:
                    rows.append({**base, "op": "define", "name": node.targets[0].id, "value": reference(call)})
            elif isinstance(node, ast.If):
                ref = reference(node.test.left)
                branch = {"op": "if_equal", "value": ref}
                rows.append({**base, **branch, "equals": ast.literal_eval(node.test.comparators[0]),
                             "when_equal": block(node.body), "otherwise": block(node.orelse)})
            else:
                args = node.value.args
                duties = ast.literal_eval(args[2]) if len(args) > 2 else []
                outcome = ast.literal_eval(args[0])
                if outcome == "read_path_completed":
                    rows.append({**base, "op": "complete", "explanation": ast.literal_eval(args[1])})
                else:
                    rows.append({**base, "op": "handoff", "outcome": outcome, "explanation": ast.literal_eval(args[1]),
                                 "duties": [{"when": when, "requirement": duty} for when, duty in duties], "restrictions": []})
        return rows
    return block(ast.parse(text).body)


@pytest.mark.parametrize("value", [[], [1], [1, 2, 3]])
def test_array_length_is_a_typed_total_operation_on_arrays(value):
    schemas = {"input": {"type": "array", "items": {"type": "integer"}}}
    plan = compile_binding(schemas, {"type": "integer"}, {"kind": "array_length", "source": "input", "pointer": ""})
    assert plan["requiredSources"] == ["input"]
    assert materialize_binding(plan, {"input": value})["arguments"] == len(value)
    assert not plan["runtimeAuthorityGranted"]


@pytest.mark.parametrize("kind", ["string", "object", ["array", "null"]])
def test_length_never_coerces_other_types(kind):
    schema = {"type": kind, "items": {"type": "integer"}} if isinstance(kind, list) else {"type": kind}
    with pytest.raises(ValueError, match="array_length"):
        compile_binding({"input": schema}, {"type": "integer"}, {"kind": "array_length", "source": "input", "pointer": ""})


@pytest.mark.parametrize("bad", [None, "abc", {"0": 1}, [True]])
def test_array_length_rechecks_actual_source(bad):
    plan = compile_binding({"input": {"type": "array", "items": {"type": "integer"}}}, {"type": "integer"},
                           {"kind": "array_length", "source": "input", "pointer": ""})
    with pytest.raises(ValueError):
        materialize_binding(plan, {"input": bad})


def mark_program(rows, source_id):
    for row in rows:
        row["source_id"] = source_id
        for key in ("duties", "restrictions"):
            for duty in row.get(key, []):
                duty["source_id"] = source_id
        if row["op"] in source_program_lines.BRANCHES:
            for key in ("when_equal", "otherwise"):
                mark_program(row[key], source_id)
    return rows


def unmarked(rows):
    import copy
    rows = copy.deepcopy(rows)
    for row in rows:
        row.pop("source_id", None)
        for key in ("duties", "restrictions"):
            for duty in row.get(key, []):
                duty.pop("source_id", None)
        if row["op"] in source_program_lines.BRANCHES:
            for key in ("when_equal", "otherwise"):
                row[key] = unmarked(row[key])
    return rows


def semantic_choice(wire, inline=True):
    value = choice(wire)
    payload = json.loads(wire["messages"][1]["content"])
    value["source_scan"] = {row["id"]: {"roles": ["background"], "meaning": "Synthetic plumbing-only review, not semantic evidence."}
                            for row in payload["sourceScanFragments"]}
    value["business_gaps"] = value.pop("issues")
    value["outside_task_duties"] = value.pop("remaining")
    mark = value.pop("region")["steps"][0]["source"]
    value["procedure"] = [{"source": mark, "statement": "Read the supplied device's interfaces; stop if empty, otherwise hand off the remaining checks."}]
    value["execution_requirements"] = [{"source": mark, "requirement": "Host consent, resource access and evidence age remain mandatory at execution."}]
    value["program"] = typed_program('''obs0 = read("get_interfaces")
if length(obs0, "/interfaces") == 0:
    end("read_path_completed", "The list is empty; inspection finished")
else:
    end("needs_l1", "Further checks remain for L1", [("When the list is nonempty", "Preserve conditional counter inspection and proposal-only constraints")])
''')
    if inline:
        mark_program(value["program"], mark["block_id"])
    return value


def prepared(packet, inline=True):
    state = ledger.initial_state(packet, "plan_first", semantic_plan=True)
    wire, _ = ledger.make_request(packet, state)
    value = semantic_choice(wire, inline=inline)
    if inline:
        assert Draft202012Validator(wire["format"]).is_valid(wire_choice(value))
    return state, wire, value


def anchor_reply(wire):
    content = json.loads(wire["messages"][1]["content"])
    frozen = content["frozenProgram"]
    block = next(b for b in content["sourceBlocks"] if "先按" in b["text"])
    return {"mode": "program_sources", "draftDigest": frozen["draftDigest"],
            "sources": {slot["id"]: {"basis": "Fixture mapping, not a semantic proof.", "evidence_id": block["id"]}
                        for slot in frozen["sourceSlots"]}}


def anchored(packet, state, files):
    if files["next-state.json"]["planAuthoring"]["phase"] == "binding":
        assert files["program-source-bindings.json"]["additionalSourceBindingModelCalls"] == 0
        return files
    state = files["next-state.json"]
    wire, _ = ledger.make_request(packet, state)
    files, result = ledger.derive(packet, state, wire, envelope(anchor_reply(wire)))
    assert result["candidateStatus"] == "operation_plan_recorded", result
    return files


def slot_arguments(wire):
    """Synthetic fixture reply for the first read; never used by a model author."""
    payload = json.loads(wire["messages"][1]["content"])
    return {"mode": "slot_arguments", "planDigest": payload["frozenPlan"]["planDigest"],
            "readPointer": payload["frozenPlan"]["currentRead"]["treePointer"],
            "slotPacketDigest": payload["parameterSlots"]["slotPacketDigest"],
            "bindings": {"/device/id": "input#/device/id"}}


def test_compilation_retains_path_duties_and_future_gates(packet):
    state, wire, value = prepared(packet)
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "operation_plan_recorded"
    assert files["program-draft.json"]["choice"] == wire_choice(value)
    files = anchored(packet, state, files)
    assert files["prepared-plan.json"]["observationPaths"] == {"input": "input", "obs0": "/steps/0"}
    assert not files["program-source-bindings.json"]["programStructureChanged"]
    state = files["next-state.json"]
    wire, _ = ledger.make_request(packet, state)
    payload = json.loads(wire["messages"][1]["content"])
    assert payload["frozenPlan"]["proposal"]  # v44's target-only prompt lost useful procedure context
    assert payload["sourceBlocks"]  # full original source remains, not a model-only summary
    assert payload["parameterSlots"]
    files, result = ledger.derive(packet, state, wire, envelope(slot_arguments(wire)))
    assert result["candidateStatus"] == "compiled_region_requires_semantic_review"
    assert files["compilation.json"]["qualification"]["runtimeAuthorityGranted"] is False
    assert files["remaining.json"]["originalPlanRemaining"] == []
    assert files["remaining.json"]["duties"][0]["terminalPointer"] == "/steps/1/otherwise/0"
    assert not files["execution-requirements.json"]["satisfied"]
    assert files["tree.json"]["steps"][1]["left"]["kind"] == "array_length"
    assert len(files["handoffs.json"]["boundaries"]) == 2


@pytest.mark.parametrize("minimum,guaranteed", [(0, False), (1, True), (2, True)])
def test_planning_navigation_preserves_schema_presence_not_item_field_requiredness(minimum, guaranteed):
    schema = {"type": "object", "properties": {"items": {"type": "array", "minItems": minimum,
        "items": {"type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]}}},
        "required": ["items"]}
    paths = source_plan.planning_sources({"tools": []}, schema)
    array = next(p for p in paths if p["pointer"] == "/items")
    field = next(p for p in paths if p["pointer"] == "/items/0/ready")
    assert array["sourcePathGuaranteedPresent"] is True
    assert array["arrayMinItems"] == minimum and array["arrayMaxItems"] is None
    assert field["sourcePathGuaranteedPresent"] is guaranteed


def test_optional_array_parent_never_claims_guaranteed_element_presence():
    schema = {"type": "object", "properties": {"items": {"type": "array", "minItems": 1, "items": {"type": "string"}}}}
    assert all(not p["sourcePathGuaranteedPresent"] for p in source_plan.planning_sources({"tools": []}, schema))


def test_default_language_unchanged_and_mixed_inspections_rejected(packet):
    state, _, value = prepared(packet)
    old, _ = ledger.make_request(packet, ledger.initial_state(packet, "plan_first"))
    assert not Draft202012Validator(old["format"]).is_valid(value)
    with pytest.raises(ValueError, match="replaces"):
        ledger.initial_state(packet, "plan_first", account_duties=True, semantic_plan=True)
    with pytest.raises(ValueError, match="syntax"):
        source_plan.prepare(value, packet, citation_blocks(ledger.frame(packet, state)))


@pytest.mark.parametrize("original,replacement", [
    ('length(obs0', 'length(obs7'),
    ('"/interfaces"', '"/interfaces/0/adminUp"'),
    ('"/interfaces"', '"interfaces[0]"'),
    ('obs0 =', 'input ='),
    ('obs0 =', 'obs7 ='),
    ('"get_interfaces"', '"nonexistent_tool"'),
])
def test_actual_symbols_paths_types_and_tools_still_checked(packet, original, replacement):
    state, wire, value = prepared(packet)
    source_id = value["program"][0]["source_id"]
    value["program"] = mark_program(typed_program(source_program_lines.render(unmarked(value["program"]), packet["catalog"]).replace(original, replacement)), source_id)
    _, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "candidate_invalid_or_unresolved"


def test_names_cannot_shadow_a_previous_observation(packet):
    state, wire, value = prepared(packet)
    value["program"].insert(0, value["program"][0])
    _, result = ledger.derive(packet, state, wire, envelope(value))
    assert "unique" in result["diagnostic"]


def test_sibling_observation_cannot_leak(packet):
    state, wire, value = prepared(packet)
    source_id = value["program"][0]["source_id"]
    value["program"] = mark_program(typed_program('''obs0 = read("get_interfaces")
if length(obs0, "/interfaces") == 0:
    obs1 = read("get_interfaces")
    end("read_path_completed", "First branch inspection ends")
else:
    if length(obs1, "/interfaces") == 0:
        end("read_path_completed", "Nested branch inspection ends")
    else:
        end("read_path_completed", "Other branch inspection ends")
'''), source_id)
    _, result = ledger.derive(packet, state, wire, envelope(value))
    assert "dominate" in result["diagnostic"]


def test_semantic_prompt_and_navigation_are_generic(packet):
    _, wire, _ = prepared(packet)
    system = wire["messages"][0]["content"]
    assert "get_interfaces" not in system and "adminUp" not in system
    assert "requiredOutputSchema" in json.loads(wire["messages"][1]["content"])
    paths = source_plan.planning_sources({"tools": [{"name": "input", "outputSchema": {"type": "boolean"}}]}, {"type": "string"})
    assert len(paths) == 2
    assert {p["originKind"] for p in paths} == {"caller_input", "tool_output"}


def test_compact_model_view_keeps_original_source_and_only_opt_in_decoding(packet):
    _, semantic, _ = prepared(packet)
    legacy, _ = ledger.make_request(packet, ledger.initial_state(packet, "plan_first"))
    small = json.loads(semantic["messages"][1]["content"])
    original = json.loads(legacy["messages"][1]["content"])
    for field in ("sourceBlocks", "hostCatalog", "inputSchema", "currentTask", "sourceIndex"):
        assert small[field] == original[field]
    assert "authoringBoundary" not in small
    assert "planningValuePaths" in small and "programLanguage" in small
    assert semantic["options"]["presence_penalty"] == 0
    assert semantic["options"]["repeat_penalty"] == 1
    assert "presence_penalty" not in legacy["options"]
    assert "repeat_penalty" not in legacy["options"]


def test_reasoning_profile_records_explicit_loop_control_and_budget(packet):
    state = ledger.initial_state(packet, "plan_first", semantic_plan=True, reasoning=True)
    wire, _ = ledger.make_request(packet, state)
    assert wire["think"] is True
    assert wire["options"]["presence_penalty"] == 1.5
    assert wire["options"]["num_predict"] == 8192


def test_business_gaps_still_block_before_argument_construction(packet):
    state, wire, value = prepared(packet)
    value["business_gaps"] = [{"source": value["procedure"][0]["source"],
                                "explanation": "The source does not define a required business condition."}]
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "operation_plan_has_unresolved_issues"
    assert "next-state.json" not in files and not files["program-draft.json"]["runtimeAuthorityGranted"]


def test_model_subset_excludes_redundant_scalar_alias_namespace(packet):
    state, wire, value = prepared(packet)
    expression = 'length(obs0, "/interfaces")'
    source_id = value["program"][0]["source_id"]
    value["program"] = mark_program(typed_program(source_program_lines.render(unmarked(value["program"]), packet["catalog"]).replace("if " + expression, "COUNT = " + expression + "\nif COUNT")), source_id)
    files, result = ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "candidate_invalid_or_unresolved"


def test_length_binding_is_versioned_and_cannot_be_downgraded():
    plan = compile_binding({"input": {"type": "array", "items": {"type": "string"}}}, {"type": "integer"},
                           {"kind": "array_length", "source": "input", "pointer": ""})
    assert plan["apiVersion"].endswith("/v2")
    assert verify_binding(plan) == plan
    bad = {**plan, "apiVersion": "netopyu.io/l0-data-binding/v1"}
    bad["planDigest"] = sha256_json({k: v for k, v in bad.items() if k != "planDigest"})
    with pytest.raises(ValueError):
        verify_binding(bad)


def test_named_argument_sources_lower_exactly_without_template_interpretation():
    plan = {"semanticPlanProfile": "test", "reportDigest": "plan", "observationPaths": {
        "input": "input", "early": "/steps/0", "nested": "/steps/1/otherwise/0", "sibling": "/steps/1/when_equal/0"}}
    slot = {"treePointer": "/steps/1/otherwise/1", "dominatingReadPointers": ["/steps/0", "/steps/1/otherwise/0"]}
    assert list(source_plan.binding_aliases(plan, slot)) == ["input", "early", "nested"]
    value = {"kind": "object", "fields": {"id": {"kind": "reference", "source": "nested", "pointer": "/health/id"}}}
    lowered, audit = source_plan.lower_argument_names(plan, slot, value)
    assert value["fields"]["id"]["source"] == "nested"
    assert lowered["fields"]["id"]["source"] == "/steps/1/otherwise/0"
    assert not audit["literalTemplatesInterpreted"] and not audit["runtimeAuthorityGranted"]
    value["fields"]["id"]["source"] = "sibling"
    with pytest.raises(ValueError, match="dominating"):
        source_plan.lower_argument_names(plan, slot, value)
    literal = {"kind": "literal", "value": "{{nested.id}}"}
    assert source_plan.lower_argument_names(plan, slot, literal)[0] == literal  # origin check still rejects unsupported literals


def test_argument_reasoning_is_explicit_frozen_and_phase_local(packet):
    state = ledger.initial_state(packet, "plan_first", semantic_plan=True, argument_reasoning=True)
    wire, _ = ledger.make_request(packet, state)
    assert wire["think"] is False and wire["options"]["num_predict"] == 4096
    files, _ = ledger.derive(packet, state, wire, envelope(semantic_choice(wire)))
    # Inline provenance has no second source-mapping model request.
    assert files["next-state.json"]["planAuthoring"]["phase"] == "binding"
    files = anchored(packet, state, files)
    binding_wire, _ = ledger.make_request(packet, files["next-state.json"])
    assert binding_wire["think"] is True and binding_wire["options"]["num_predict"] == 8192
    assert binding_wire["options"]["presence_penalty"] == 1.5
    assert files["next-state.json"]["modelArgumentReasoning"] is True
    with pytest.raises(ValueError, match="argument reasoning"):
        ledger.initial_state(packet, "plan_first", argument_reasoning=True)
    with pytest.raises(ValueError, match="argument reasoning"):
        ledger.initial_state(packet, "plan_first", semantic_plan=True, argument_reasoning=True, reasoning=True)
