"""Host-directed construction mechanics, not model accuracy or semantic Gold."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_catalog as catalog, source_ledger as ledger
from evaluation.structured_flow_demo import fixture
from tests.test_source_ledger import envelope


@pytest.fixture
def packet():
    bundle, tree, reads, _ = fixture()
    return {"bundle": bundle, "task": "Read lab-edge-42 with explicit source prerequisites; retain unknown duties.",
        "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
        "catalog": {"tools": [json.loads(next(s.text for s in c.spec.sources if s.role == "tool")) for c in reads.values()]},
        "reads": {n: c.model_dump(by_alias=True, mode="json") for n, c in reads.items()}}


def request(packet):
    state = ledger.initial_state(packet, "catalog_bound")
    return state, ledger.make_request(packet, state)[0]


def candidate(packet, wire):
    _, tree, _, _ = fixture()
    raw = tree.model_dump(mode="json")
    blocks = json.loads(wire["messages"][1]["content"])["sourceBlocks"]
    mark = {"block_id": next(b["id"] for b in blocks if raw["steps"][0]["source"]["quote"] in b["text"])}
    read = raw["steps"][0]
    read["source"] = mark
    read.pop("bind")
    raw["steps"] = [read, {"kind": "end", "source": mark, "outcome": "needs_l1",
        "explanation": "All untranslated procedures and conditions remain for review."}]
    return {"mode": "candidate", "tree": raw, "remaining": []}


def literal(value, origin=None):
    return {"kind": "literal", "value": value, "origin": origin or {"kind": "task", "quote": str(value)}}


def custom_schema(target):
    from evaluation.structured_authoring import response_schema
    from evaluation.source_candidate_schema import tighten, omit_schema_titles
    bundle, tree, _, _ = fixture()
    schema = response_schema({"bundle": bundle, "inputSchema": tree.input_schema}, {}, {})
    tighten(schema, ["unrelated_api"])
    schema["$defs"]["SourceSpan"] = {"type": "object", "properties": {"block_id": {"const": "b0000"}},
                                    "required": ["block_id"], "additionalProperties": False}
    catalog.constrain(schema, {"tools": [{"name": "unrelated_api", "inputSchema": target}]})
    schema = omit_schema_titles(schema)
    schema = {"$defs": schema["$defs"], "$ref": "#/$defs/StructuredTreeRead"}
    return catalog.prune_definitions(schema)


def valid_expression(expression, target):
    schema = custom_schema(target)
    Draft202012Validator.check_schema(schema)
    value = {"kind": "read", "source": {"block_id": "b0000"}, "tool": "unrelated_api", "arguments": expression}
    return Draft202012Validator(schema).is_valid(value)


@pytest.mark.parametrize("value", ["edge", 4, True, None, {"id": "edge"}, ["edge"]])
def test_literal_shape_preserves_host_schema_without_coercion(value):
    target = {"type": "string", "minLength": 1, "maxLength": 12}
    assert valid_expression(literal(value, {"kind": "task", "quote": "edge"}), target) == isinstance(value, str)


@pytest.mark.parametrize("fields,valid", [({}, False), ({"x": literal("a")}, True),
    ({"x": literal("a"), "invented": literal("b")}, False), ({"x": literal(4)}, False)])
def test_required_fields_extra_fields_and_types_are_in_generation_schema(fields, valid):
    target = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": False}
    assert valid_expression({"kind": "object", "fields": fields}, target) == valid


@pytest.mark.parametrize("items,valid", [([], False), ([literal("x")], True), ([literal("x"), literal("y")], False)])
def test_array_bounds_are_materialized_item_bounds(items, valid):
    target = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 1}
    assert valid_expression({"kind": "array", "items": items}, target) == valid


def test_const_aggregate_cannot_escape_constraints_through_object_expression():
    target = {"type": "object", "properties": {"id": {"type": "integer"}}, "const": {"id": 4}}
    assert not valid_expression({"kind": "object", "fields": {"id": literal(5)}}, target)
    assert not valid_expression(literal({"id": 5}), target)
    assert valid_expression(literal({"id": 4}), target)


def test_nullable_refs_and_schema_literal_ref_strings_remain_data():
    target = {"$defs": {"thing": {"type": ["string", "null"], "enum": ["small", None]}},
              "type": "object", "properties": {"x": {"$ref": "#/$defs/thing"}}, "required": ["x"], "additionalProperties": False}
    for x, valid in [("small", True), (None, True), ("large", False), (7, False)]:
        assert valid_expression({"kind": "object", "fields": {"x": literal(x)}}, target) == valid
    data = {"$ref": "#/a", "title": "actual value", "properties": {"$ref": "literal"}}
    schema = {"type": "object", "const": data, "additionalProperties": True}
    assert valid_expression(literal(data), schema)
    assert schema["const"] == data


def test_typed_and_untyped_additional_properties_keep_original_flexibility():
    for extra, value, good in [(True, 2, True), ({"type": "integer"}, 2, True), ({"type": "integer"}, "bad", False)]:
        target = {"type": "object", "additionalProperties": extra}
        assert valid_expression({"kind": "object", "fields": {"arbitrary": literal(value)}}, target) == good


def test_raw_aliases_and_wrong_tool_are_rejected_before_lowering(packet):
    _, wire = request(packet)
    choice = candidate(packet, wire)
    validator = Draft202012Validator(wire["format"])
    assert validator.is_valid(choice)
    choice["tree"]["steps"][0]["bind"] = "repeated_host_binding_id"
    assert not validator.is_valid(choice)
    choice["tree"]["steps"][0].pop("bind")
    choice["tree"]["steps"][0]["tool"] = "invented"
    assert not validator.is_valid(choice)


def test_candidate_lowering_compiles_with_existing_executor_contracts_only(packet):
    state, wire = request(packet)
    choice = candidate(packet, wire)
    before = copy.deepcopy(choice)
    files, outcome = ledger.derive(packet, state, wire, envelope(choice))
    assert outcome["candidateStatus"] == "compiled_region_requires_semantic_review"
    assert files["tree.json"]["steps"][0]["bind"] == "read_000"
    assert choice == before
    report = files["catalog-lowering.json"]
    assert not report["issues"] and not report["runtimeAuthorityGranted"]
    assert not report["semanticEntailmentProven"] and report["providerCalls"] == 0


def test_task_literal_has_preserved_provenance_not_authority(packet):
    state, wire = request(packet)
    choice = candidate(packet, wire)
    choice["tree"]["steps"][0]["arguments"] = {"kind": "object", "fields": {"device": {
        "kind": "object", "fields": {"id": literal("lab-edge-42")}}}}
    files, outcome = ledger.derive(packet, state, wire, envelope(choice))
    assert outcome["candidateStatus"] == "compiled_region_requires_semantic_review"
    row = files["catalog-lowering.json"]["valueOrigins"][0]
    assert packet["task"][row["start"]:row["end"]] == "lab-edge-42"
    assert not row["semanticEntailmentProven"]
    assert "origin" not in files["tree.json"]["steps"][0]["arguments"]["fields"]["device"]["fields"]["id"]


@pytest.mark.parametrize("origin,value", [({"kind": "task", "quote": "absent"}, "edge"),
    ({"kind": "task", "quote": "lab-edge-42"}, "other-device"),
    ({"kind": "host_schema", "pointer": "/properties/device"}, "edge")])
def test_missing_or_unrelated_literal_origins_block_without_repair(packet, origin, value):
    state, wire = request(packet)
    choice = candidate(packet, wire)
    choice["tree"]["unresolved"] = ["The regional procedure remains unresolved."]
    choice["tree"]["steps"][0]["arguments"] = {"kind": "object", "fields": {"device": {
        "kind": "object", "fields": {"id": literal(value, origin)}}}}
    files, outcome = ledger.derive(packet, state, wire, envelope(choice))
    assert outcome["candidateStatus"] == "catalog_candidate_blocked"
    assert {i["code"] for i in files["catalog-lowering.json"]["issues"]} >= {"unresolved_region", "literal_origin"}
    assert "compilation.json" not in files and files["catalog-tree.json"]["unresolved"]


def test_source_occurrence_does_not_convert_number_substrings_or_infer_duration():
    assert not catalog._occurs(1, "10")
    assert not catalog._occurs(1, "-1")
    assert not catalog._occurs(1, "+1")
    assert not catalog._occurs(-86400, "past 24 hours")
    assert catalog._occurs(-86400, '"after":-86400')
    assert catalog._occurs(True, '"enabled":true')
    assert catalog._occurs("", '"value":""')
    assert not catalog._occurs({"enabled": 1}, '{"enabled":true}')
    assert not catalog._occurs([1], '[true]')


@pytest.mark.parametrize("reference", ["/steps/0", "/steps/2", "/steps/1/otherwise/0"])
def test_self_future_and_sibling_references_are_not_rebound(packet, reference):
    state, wire = request(packet)
    choice = candidate(packet, wire)
    choice["tree"]["steps"][0]["arguments"]["fields"]["device"]["source"] = reference
    files, outcome = ledger.derive(packet, state, wire, envelope(choice))
    assert outcome["candidateStatus"] == "catalog_candidate_blocked"
    assert any(i["code"] == "reference_scope" for i in files["catalog-lowering.json"]["issues"])


def test_duplicate_calls_have_unique_aliases_but_are_not_deduplicated(packet):
    state, wire = request(packet)
    choice = candidate(packet, wire)
    choice["tree"]["steps"].insert(1, copy.deepcopy(choice["tree"]["steps"][0]))
    files, outcome = ledger.derive(packet, state, wire, envelope(choice))
    assert outcome["candidateStatus"] == "compiled_region_requires_semantic_review"
    assert [s["bind"] for s in files["tree.json"]["steps"] if s["kind"] == "read"] == ["read_000", "read_001"]
    assert not files["catalog-lowering.json"]["semanticEntailmentProven"]
    assert files["catalog-lowering.json"]["sameArgumentGroups"][0]["treePointers"] == ["/steps/0", "/steps/1"]
    assert not files["catalog-lowering.json"]["sameArgumentGroups"][0]["semanticRedundancyProven"]


@pytest.mark.parametrize("change,code", [("missing", "unclosed_region"), ("unreachable", "unreachable_statement")])
def test_terminal_errors_are_reported_alongside_unresolved_not_silently_fixed(packet, change, code):
    state, wire = request(packet)
    choice = candidate(packet, wire)
    choice["tree"]["unresolved"] = ["An independent unresolved obligation."]
    if change == "missing":
        choice["tree"]["steps"].pop()
    else:
        choice["tree"]["steps"].append(copy.deepcopy(choice["tree"]["steps"][0]))
    files, outcome = ledger.derive(packet, state, wire, envelope(choice))
    assert outcome["candidateStatus"] == "catalog_candidate_blocked"
    assert {i["code"] for i in files["catalog-lowering.json"]["issues"]} >= {code, "unresolved_region"}
    assert "compilation.json" not in files


def test_profile_manifest_replay_is_zero_call_and_fixed(packet, tmp_path, monkeypatch):
    from evaluation import flow_checkpoint
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": ledger.QWEN_MODEL, "digest": "fixture"})
    calls = []
    def send(arm, wire):
        calls.append(wire)
        return envelope(candidate(packet, wire))
    monkeypatch.setattr(flow_checkpoint, "send", send)
    root = tmp_path / "run"
    manifest = ledger.freeze(packet, root, profile="catalog_bound")
    report = ledger.run(root, max_new_calls=1)
    assert report["compiled"] and report["profile"] == "catalog_bound"
    assert report["providerCalls"] == 0 and len(calls) == 1
    assert ledger.run(root) == report and len(calls) == 1
    assert manifest["initialState"]["catalogAuthoring"] == catalog.PROFILE


def test_pruning_does_not_remove_literal_dollar_refs_or_active_rules():
    schema = {"$defs": {"used": {"type": "string", "minLength": 2}, "unused": {"type": "integer"}},
              "type": "object", "properties": {"x": {"$ref": "#/$defs/used"}, "payload": {"const": {"$ref": "#/$defs/unused"}}},
              "required": ["x", "payload"]}
    compact = catalog.prune_definitions(schema)
    assert set(compact["$defs"]) == {"used"}
    for x in ["", "ab", 1]:
        value = {"x": x, "payload": {"$ref": "#/$defs/unused"}}
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(compact).is_valid(value)


def test_nested_dominating_reads_keep_branch_scope_and_compiler_owned_names(packet):
    state, wire = request(packet)
    _, original, _, _ = fixture()
    raw = original.model_dump(mode="json")
    blocks = json.loads(wire["messages"][1]["content"])["sourceBlocks"]
    refs = {"interfaces": "/steps/0", "counters": "/steps/1/when_equal/0"}
    def expressions(expr):
        if expr["kind"] in {"reference", "column_rows"}:
            expr["source"] = refs.get(expr["source"], expr["source"])
        elif expr["kind"] == "object":
            for child in expr["fields"].values():
                expressions(child)
        elif expr["kind"] == "array":
            for child in expr["items"]:
                expressions(child)
    def statements(steps):
        for step in steps:
            step["source"] = {"block_id": next(b["id"] for b in blocks if step["source"]["quote"] in b["text"])}
            if step["kind"] == "read":
                step.pop("bind")
                expressions(step["arguments"])
            elif step["kind"] == "if_equal":
                expressions(step["left"])
                statements(step["when_equal"])
                statements(step["otherwise"])
            elif step["kind"] == "effect_candidate":
                source = step["source"]
                step.clear()
                step.update(kind="end", source=source, outcome="needs_l1",
                            explanation="Effects remain unsupported by the read-only host catalog.")
    statements(raw["steps"])
    files, outcome = ledger.derive(packet, state, wire, envelope({"mode": "candidate", "tree": raw, "remaining": []}))
    assert outcome["candidateStatus"] == "compiled_region_requires_semantic_review"
    assert [r["treePointer"] for r in files["catalog-lowering.json"]["aliases"]] == ["/steps/0", "/steps/1/when_equal/0"]
    inner = files["tree.json"]["steps"][1]["when_equal"][1]
    assert inner["left"]["source"] == "read_001"

    # A sibling branch cannot consume the true branch's local result.
    raw["steps"][1]["otherwise"] = [copy.deepcopy(raw["steps"][1]["when_equal"][1])]
    files, outcome = ledger.derive(packet, state, wire, envelope({"mode": "candidate", "tree": raw, "remaining": []}))
    assert outcome["candidateStatus"] == "catalog_candidate_blocked"
    assert any(r["code"] == "reference_scope" for r in files["catalog-lowering.json"]["issues"])


def test_source_literal_trace_keeps_file_coordinates_and_does_not_prove_meaning(packet):
    state, wire = request(packet)
    blocks = ledger.citation_blocks(ledger.frame(packet, state))
    block_id = next(k for k, b in blocks.items() if "Host-bound contract" in b["text"])
    choice = candidate(packet, wire)
    value = literal("Host-bound contract", {"kind": "source", "block_id": block_id, "quote": "Host-bound contract"})
    choice["tree"]["steps"][0]["arguments"] = {"kind": "object", "fields": {"device": {"kind": "object", "fields": {"id": value}}}}
    files, outcome = ledger.derive(packet, state, wire, envelope(choice))
    assert outcome["candidateStatus"] == "compiled_region_requires_semantic_review"
    origin = files["catalog-lowering.json"]["valueOrigins"][0]
    text = next(d["content"] for d in packet["bundle"]["documents"] if d["path"] == origin["path"])
    assert text[origin["start"]:origin["end"]] == origin["quote"]
    # A phrase can occur and fit a string schema yet be an inappropriate device ID.
    assert not origin["semanticEntailmentProven"]


@pytest.mark.parametrize("origin,blocked", [
    ({"kind": "host_schema", "pointer": "/const"}, False),
    ({"kind": "host_schema", "pointer": "/default"}, True),
    ({"kind": "host_schema", "pointer": "/examples/0"}, True),
])
def test_host_constant_origins_do_not_treat_defaults_or_examples_as_value_authority(packet, origin, blocked):
    _, wire = request(packet)
    raw = candidate(packet, wire)["tree"]
    name = raw["steps"][0]["tool"]
    packet["catalog"] = {"tools": [{"name": name, "inputSchema": {"type": "string", "const": "BLUE",
        "default": "BLUE", "examples": ["BLUE"]}, "outputSchema": {"type": "object"}}]}
    raw["steps"][0]["arguments"] = literal("BLUE", origin)
    _, diagnostic = catalog.lower(raw, packet, {})
    assert bool(diagnostic["issues"]) == blocked
    assert not diagnostic["semanticEntailmentProven"]


def test_static_aggregate_constraints_are_checked_after_origin_wrappers_removed(packet):
    _, wire = request(packet)
    raw = candidate(packet, wire)["tree"]
    name = raw["steps"][0]["tool"]
    packet["task"] += " Requested value BLUE."
    packet["catalog"] = {"tools": [{"name": name, "inputSchema": {"type": "array",
        "items": {"type": "string"}, "uniqueItems": True}, "outputSchema": {"type": "object"}}]}
    raw["steps"][0]["arguments"] = {"kind": "array", "items": [literal("BLUE"), literal("BLUE")]}
    before = copy.deepcopy(raw)
    _, diagnostic = catalog.lower(raw, packet, {})
    assert any(i["code"] == "constant_host_constraint" for i in diagnostic["issues"])
    assert raw == before and not diagnostic["runtimeAuthorityGranted"]


def test_bad_origin_checkpoint_is_retained_not_automatically_retried(packet, tmp_path, monkeypatch):
    from evaluation import flow_checkpoint
    monkeypatch.setattr(ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": ledger.QWEN_MODEL, "digest": "fixture"})
    calls = []
    def send(arm, wire):
        calls.append(wire)
        choice = candidate(packet, wire)
        choice["tree"]["steps"][0]["arguments"] = {"kind": "object", "fields": {
            "device": {"kind": "literal", "value": {"id": "invented"},
                       "origin": {"kind": "task", "quote": "absent"}}}}
        return envelope(choice)
    monkeypatch.setattr(flow_checkpoint, "send", send)
    root = tmp_path / "run"
    ledger.freeze(packet, root, profile="catalog_bound")
    report = ledger.run(root, max_new_calls=6)
    assert report["status"] == "catalog_candidate_blocked" and report["candidateProduced"]
    assert not report["compiled"] and (root / "round-000/response.json").exists()
    assert ledger.run(root, max_new_calls=6) == report and len(calls) == 1
