"""Mechanics and safety counterexamples; not semantic accuracy measurements."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_catalog, source_ledger, source_modes as modes
from evaluation.netdata_fixture import IsolatedNetdataHost, QUERY_KEYS, TOOL
from evaluation.structured_authoring import sha256_json
from evaluation.structured_flow_demo import fixture
from tests.test_source_catalog import literal
from tests.test_source_ledger import envelope


def obj(fields, required=()):
    return {"type": "object", "properties": fields, "required": list(required), "additionalProperties": False}


def mode(name, pointer, required, allowed=None):
    return {"id": name, "description": "Explicit host parameter combination, not an instruction to execute.",
            "objects": [{"objectPointer": pointer, "requiredKeys": required,
                         "allowedKeys": required if allowed is None else allowed}]}


@pytest.fixture
def packet():
    bundle, tree, _, _ = fixture()
    host = IsolatedNetdataHost(node="lab-edge-42", listener="local", device_ip="10.0.0.8")
    contract, _ = host.contract_and_binding(bundle["documents"][0]["content"])
    return {"bundle": bundle, "task": "Read lab-edge-42, retaining all necessary original prerequisites.",
            "taskOrigin": "developer_authored_evaluation_request", "inputSchema": tree.input_schema,
            "catalog": host.catalog, "reads": {TOOL: contract.model_dump(by_alias=True, mode="json")}}


@pytest.fixture
def declarations(packet):
    tool = packet["catalog"]["tools"][0]
    return [{"hostTool": TOOL, "catalogDigest": sha256_json(packet["catalog"]),
             "contractHash": packet["reads"][TOOL]["contractHash"],
             "inputSchemaDigest": sha256_json(tool["inputSchema"]),
             "outputSchemaDigest": sha256_json(tool["outputSchema"]), "reviewKind": modes.REVIEW_KIND,
             "declarationEvidence": "observe permits exactly info, or exactly all query keys; never their mixture.",
             "modes": [mode("inspect", "/body", ["info"]), mode("query", "/body", sorted(QUERY_KEYS))]}]


def choice(packet, wire):
    _, tree, _, _ = fixture()
    raw = tree.model_dump(mode="json")
    blocks = json.loads(wire["messages"][1]["content"])["sourceBlocks"]
    b = blocks[0]
    # A deliberate developer-authored candidate for mechanical tests only.
    mark = {"block_id": b["id"]}
    args = {"kind": "object", "fields": {
        "node": literal("lab-edge-42"),
        "function": literal("snmp:traps", {"kind": "host_schema", "pointer": "/properties/function/const"}),
        "body": {"kind": "object", "fields": {
            "info": literal(True, {"kind": "host_schema", "pointer": "/properties/body/properties/info/const"})}}}}
    raw["steps"] = [{"kind": "read", "source": mark, "operationMode": "inspect",
                     "tool": TOOL, "arguments": args},
                    {"kind": "end", "source": mark, "outcome": "needs_l1", "explanation": "All necessary unconverted duties remain for review."}]
    return {"mode": "candidate", "tree": raw, "remaining": []}


@pytest.mark.parametrize("names", [("probe", "filter"), ("preview", "payload"), ("describe", "volume")])
def test_mode_refinement_is_generic_and_intersects_original_schema(names):
    a, b = names
    schema = obj({"payload": obj({a: {"type": "boolean", "const": True}, b: {"type": "integer", "minimum": 1}})}, ["payload"])
    before = copy.deepcopy(schema)
    restricted = modes.refine(schema, mode("first", "/payload", [a]))
    assert schema == before
    for body, valid in [({a: True}, True), ({a: False}, False), ({}, False), ({a: True, b: 1}, False), ({b: 1}, False)]:
        assert Draft202012Validator(restricted).is_valid({"payload": body}) == valid
        if valid:
            assert Draft202012Validator(schema).is_valid({"payload": body})


def test_original_host_gap_is_closed_without_changing_host(packet, declarations):
    original = copy.deepcopy(packet)
    assert modes.validate(packet, declarations) == declarations
    schema = packet["catalog"]["tools"][0]["inputSchema"]
    mixed = {"node": "lab-edge-42", "function": "snmp:traps", "body": {"info": True, "before": 0}}
    assert Draft202012Validator(schema).is_valid(mixed)
    assert all(not Draft202012Validator(modes.refine(schema, m)).is_valid(mixed) for m in declarations[0]["modes"])
    assert packet == original


@pytest.mark.parametrize("field", ["catalogDigest", "contractHash", "inputSchemaDigest", "outputSchemaDigest"])
def test_digest_drift_rejected(packet, declarations, field):
    declarations[0][field] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="drift"):
        modes.validate(packet, declarations)


@pytest.mark.parametrize("change", ["tool", "duplicate_tool", "duplicate_mode", "overlap", "duplicate_rule", "unknown_field", "bad_pointer", "extra_permission"])
def test_invalid_or_ambiguous_declaration_is_not_repaired(packet, declarations, change):
    d = declarations[0]
    if change == "tool":
        d["hostTool"] = "invented"
    elif change == "duplicate_tool":
        declarations.append(copy.deepcopy(d))
    elif change == "duplicate_mode":
        d["modes"].append(copy.deepcopy(d["modes"][0]))
    elif change == "overlap":
        d["modes"][1]["objects"] = copy.deepcopy(d["modes"][0]["objects"])
    elif change == "duplicate_rule":
        d["modes"][0]["objects"] *= 2
    elif change == "unknown_field":
        d["modes"][0]["objects"][0]["allowedKeys"].append("invented")
    elif change == "bad_pointer":
        d["modes"][0]["objects"][0]["objectPointer"] = "/body~2"
    else:
        d["permissionGranted"] = True
    with pytest.raises(ValueError):
        modes.validate(packet, declarations)


def test_shared_ref_refinement_does_not_change_other_property():
    body = obj({"left": {"type": "integer"}, "right": {"type": "integer"}})
    schema = {**obj({"a": {"$ref": "#/$defs/body"}, "b": {"$ref": "#/$defs/body"}}), "$defs": {"body": body}}
    restricted = modes.refine(schema, mode("left", "/a", ["left"]))
    assert restricted["$defs"]["body"] == body
    assert restricted["properties"]["b"] == {"$ref": "#/$defs/body"}
    assert Draft202012Validator(restricted).is_valid({"a": {"left": 1}, "b": {"right": 2}})
    assert not Draft202012Validator(restricted).is_valid({"b": {"right": 2}})


@pytest.mark.parametrize("schema", [obj({"x": {"type": "integer"}}, ["x"]),
    {**obj({"x": {"type": "integer"}}), "const": {"x": 1}},
    {**obj({"x": {"type": "integer"}}), "minProperties": 1},
    {**obj({"x": {"type": "integer"}}), "type": ["object", "null"]}])
def test_conflicts_and_unsupported_refinements_fail_closed(schema):
    with pytest.raises(ValueError):
        modes.refine(schema, mode("empty", "", []))


def test_nested_rules_cannot_resurrect_forbidden_parent():
    schema = obj({"a": obj({"x": {"type": "integer"}})})
    m = mode("nested", "", [])
    m["objects"].append(mode("child", "/a", ["x"])["objects"][0])
    with pytest.raises(ValueError):
        modes.refine(schema, m)


@pytest.mark.parametrize("expr,okay", [
    ({"kind": "reference", "source": "input", "pointer": ""}, False),
    ({"kind": "object", "fields": {"body": {"kind": "reference", "source": "input", "pointer": "/body"}}}, False),
    ({"kind": "object", "fields": {"body": {"kind": "object", "fields": {"info": {"kind": "reference", "source": "input", "pointer": "/info"}}}}}, True),
    ({"kind": "literal", "value": {"body": {"info": True}}}, True),
    ({"kind": "literal", "value": {"body": {"info": True, "query": "x"}}}, False),
    ({"kind": "literal", "value": {}}, False)])
def test_dynamic_shape_not_type_overlap_must_establish_mode(expr, okay):
    assert (not modes.shape_errors(expr, mode("inspect", "/body", ["info"]))) == okay


def test_mode_profile_requires_explicit_opt_in(packet, declarations):
    assert "operationModes" not in source_ledger.initial_state(packet)
    with pytest.raises(ValueError):
        source_ledger.initial_state(packet, "mode_bound")
    with pytest.raises(ValueError):
        source_ledger.initial_state(packet, "catalog_bound", declarations)


def test_generation_binds_mode_to_shape_and_uses_leaf_origins(packet, declarations):
    state = source_ledger.initial_state(packet, "mode_bound", declarations)
    wire, _ = source_ledger.make_request(packet, state)
    value = choice(packet, wire)
    validator = Draft202012Validator(wire["format"])
    assert validator.is_valid(value)
    value["tree"]["steps"][0]["operationMode"] = "query"
    assert not validator.is_valid(value)
    value["tree"]["steps"][0]["operationMode"] = "inspect"
    value["tree"]["steps"][0]["arguments"] = literal({"node": "lab-edge-42", "function": "snmp:traps", "body": {"info": True}})
    assert not validator.is_valid(value)


def test_original_compiler_receives_no_mode_or_quote_fields(packet, declarations):
    state = source_ledger.initial_state(packet, "mode_bound", declarations)
    wire, _ = source_ledger.make_request(packet, state)
    value = choice(packet, wire)
    before = copy.deepcopy(value)
    files, result = source_ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "compiled_region_requires_semantic_review", result
    assert value == before
    step = files["tree.json"]["steps"][0]
    assert "operationMode" not in step and "actionQuote" not in step
    plan = files["operation-plan.json"]
    assert plan["operations"][0]["modeShapeProven"]
    assert plan["operations"][0]["actionWitness"]["blockResolved"]
    witness = plan["operations"][0]["actionWitness"]
    source = next(d["content"] for d in packet["bundle"]["documents"] if d["path"] == witness["sourcePath"])
    assert source[witness["start"]:witness["end"]] == witness["quote"]
    assert not plan["sourceActionEntailmentProven"] and not plan["runtimeAuthorityGranted"]
    assert plan["providerCalls"] == 0


def test_dynamic_shape_diagnostic_blocks_original_compilation(packet, declarations):
    state = source_ledger.initial_state(packet, "mode_bound", declarations)
    wire, _ = source_ledger.make_request(packet, state)
    value = choice(packet, wire)
    step = value["tree"]["steps"][0]
    step["arguments"] = {"kind": "reference", "source": "input", "pointer": ""}
    files, result = source_ledger.derive(packet, state, wire, envelope(value))
    assert result["candidateStatus"] == "catalog_candidate_blocked"
    assert {e["code"] for e in files["catalog-lowering.json"]["issues"]} >= {"operation_mode_shape"}
    assert "compilation.json" not in files


def test_unknown_action_block_is_not_located_by_fuzzy_matching(packet, declarations):
    from evaluation.source_blocks import citation_blocks
    state = source_ledger.initial_state(packet, "mode_bound", declarations)
    wire, _ = source_ledger.make_request(packet, state)
    value = choice(packet, wire)
    value["tree"]["steps"][0]["source"]["block_id"] = "missing"
    _, report = source_catalog.lower(value["tree"], packet, citation_blocks(source_ledger.frame(packet, state)), declarations)
    assert any(e["code"] == "operation_source" for e in report["issues"])


def test_model_cannot_replace_action_witness_with_retyped_text(packet, declarations):
    state = source_ledger.initial_state(packet, "mode_bound", declarations)
    wire, _ = source_ledger.make_request(packet, state)
    value = choice(packet, wire)
    value["tree"]["steps"][0]["actionQuote"] = "a flattened command is not original evidence"
    assert not Draft202012Validator(wire["format"]).is_valid(value)


def test_duplicate_operations_retained_not_automatically_deduplicated(packet, declarations):
    state = source_ledger.initial_state(packet, "mode_bound", declarations)
    wire, _ = source_ledger.make_request(packet, state)
    value = choice(packet, wire)
    value["tree"]["steps"].insert(1, copy.deepcopy(value["tree"]["steps"][0]))
    files, _ = source_ledger.derive(packet, state, wire, envelope(value))
    assert len(files["operation-plan.json"]["operations"]) == 2
    assert files["catalog-lowering.json"]["sameArgumentGroups"]


def test_packet_version_and_shape():
    assert modes.packet_declarations({"apiVersion": modes.API_VERSION, "declarations": []}) == []
    with pytest.raises(ValueError):
        modes.packet_declarations({"apiVersion": "unversioned", "declarations": []})


def test_compact_definition_ids_preserve_references_and_literal_data():
    data = {"$ref": "#/$defs/LongName", "$defs": {"LongName": {"const": "unchanged"}}}
    schema = {"$defs": {"LongName": {"type": "object", "const": data},
                        "IndirectName": {"$ref": "#/$defs/LongName"}}, "$ref": "#/$defs/IndirectName",
              "discriminator": {"propertyName": "kind", "mapping": {"object": "#/$defs/LongName"}}}
    compact = source_catalog.compact_definition_ids(schema)
    assert compact["$defs"]["d0"]["const"] == data
    assert compact["discriminator"]["mapping"]["object"] == "#/$defs/d0"
    for value in (data, {}, {"$ref": "#/$defs/d0"}):
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(compact).is_valid(value)


def test_authoring_view_retains_all_mode_constraints_but_not_repeated_binding_prose(packet, declarations):
    view = modes.authoring_view(declarations)
    assert view["tools"][0]["modes"] == declarations[0]["modes"]
    assert view["declarationsDigest"] == sha256_json(declarations)


def test_mode_manifest_is_digest_bound_and_replay_does_not_retry(packet, declarations, tmp_path, monkeypatch):
    from evaluation import flow_checkpoint
    monkeypatch.setattr(source_ledger.prior.OllamaAnchoredAuthorAdapter, "preflight", lambda self: {"model": "fixture"})
    calls = []
    def send(arm, wire):
        calls.append(wire)
        return envelope(choice(packet, wire))
    monkeypatch.setattr(flow_checkpoint, "send", send)
    root = tmp_path / "mode-run"
    source_ledger.freeze(packet, root, profile="mode_bound", operations=declarations)
    report = source_ledger.run(root, max_new_calls=1)
    assert report["compiled"]
    assert source_ledger.run(root, max_new_calls=0) == report and len(calls) == 1
    with pytest.raises(FileExistsError):
        source_ledger.freeze(packet, root, profile="mode_bound", operations=declarations)


def test_lowering_does_not_trust_an_undeclared_mode(packet, declarations):
    state = source_ledger.initial_state(packet, "mode_bound", declarations)
    wire, _ = source_ledger.make_request(packet, state)
    raw = choice(packet, wire)["tree"]
    raw["steps"][0]["operationMode"] = "invented"
    from evaluation.source_blocks import citation_blocks
    _, report = source_catalog.lower(raw, packet, citation_blocks(source_ledger.frame(packet, state)), declarations)
    assert any(e["code"] == "operation_mode_missing" for e in report["issues"])
