"""Column projection and existing compiler/executor integration, not model scores."""

import copy
from itertools import permutations

import pytest

from evaluation.netdata_fixture import IsolatedNetdataHost, array, obj, query_arguments
from evaluation.netdata_task_demo import context
from evaluation.structured_flow_tree import StructuredFlowTree, compile_structured_tree
from evaluation.translation_intake import _bundle, _document
from network_runtime.contracts import sha256_json
from network_runtime.l0.column_rows import decode_column_rows
from network_runtime.l0.flow import (
    HostFlowConsent,
    StructuredEffectTarget,
    parse_flow,
    qualify_flow,
    run_read_flow,
)
from network_runtime.l0.structured_bindings import compile_binding, materialize_binding
from network_runtime.l0.structured_schema import DataBindingError


def decode(value, fields=None, **limits):
    return decode_column_rows(
        value,
        ["value"] if fields is None else fields,
        **({"max_rows": 4, "max_columns": 8} | limits),
    )


def payload():
    return {
        "columns": {"private": {"index": 0}, "value": {"index": 1}},
        "data": [["secret", 42], ["other", 43]],
    }


def test_column_permutations_preserve_original_keys_and_all_rows():
    names = ["a/b", "~raw", "nested", "null"]
    values = {"a/b": "text", "~raw": 9, "nested": {"inner": True}, "null": None}
    for order in permutations(names):
        response = {
            "columns": {f: {"index": i} for i, f in enumerate(order)},
            "data": [[values[f] for f in order]] * 3,
        }
        result = decode(response, names)
        assert result == [values] * 3
        result[0]["nested"]["inner"] = False
        assert response["data"][0][order.index("nested")]["inner"] is True


@pytest.mark.parametrize(
    "case,code",
    [
        ("duplicate", "column_index_duplicate"),
        ("negative", "column_index"),
        ("bool", "column_index"),
        ("string", "column_index"),
        ("out_of_budget", "column_index"),
        ("missing", "column_missing"),
        ("short", "column_row_width"),
        ("extra", "column_row_width"),
        ("non_array", "column_row_width"),
        ("too_many_rows", "column_row_budget"),
        ("empty_metadata", "column_metadata_budget"),
    ],
)
def test_bad_metadata_or_rows_never_truncate_guess_or_skip(case, code):
    value = payload()
    if case == "duplicate":
        value["columns"]["private"]["index"] = 1
    elif case in {"negative", "bool", "string", "out_of_budget"}:
        value["columns"]["value"]["index"] = {
            "negative": -1,
            "bool": True,
            "string": "1",
            "out_of_budget": 8,
        }[case]
    elif case == "missing":
        del value["columns"]["value"]
    elif case == "short":
        value["data"][-1] = ["secret"]
    elif case == "extra":
        value["data"][-1].append("unmapped")
    elif case == "non_array":
        value["data"][-1] = {"value": 3}
    elif case == "too_many_rows":
        value["data"] *= 3
    else:
        value["columns"] = {}
    with pytest.raises(DataBindingError) as error:
        decode(value)
    assert error.value.code == code
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "field_list", [[], ["value", "value"], [None], [["value"]], [""], ["v" * 129]]
)
def test_field_selection_is_explicit_and_bounded(field_list):
    with pytest.raises(DataBindingError, match="column_fields"):
        decode(payload(), field_list)


@pytest.mark.parametrize(
    "limits",
    [{"max_rows": True}, {"max_rows": 0}, {"max_rows": 257}, {"max_columns": 129}],
)
def test_explicit_budgets_cannot_expand_profile(limits):
    with pytest.raises(DataBindingError):
        decode(payload(), **limits)


def test_empty_rows_remain_empty_and_bad_metadata_still_fails():
    p = payload()
    p["data"] = []
    assert decode(p) == []
    p["columns"]["private"]["index"] = 1
    with pytest.raises(DataBindingError, match="column_index_duplicate"):
        decode(p)


def binding():
    source = obj(
        {
            "columns": {
                "type": "object",
                "additionalProperties": obj({"index": {"type": "integer"}}),
            },
            "data": array(array({"type": ["integer", "string"]})),
        }
    )
    target = array(obj({"value": {"type": "integer"}}))
    expr = {
        "kind": "column_rows",
        "source": "read",
        "pointer": "",
        "fields": ["value"],
        "max_rows": 4,
        "max_columns": 8,
    }
    return source, target, expr


def test_binding_rechecks_every_cell_and_retains_dynamic_mapping():
    source, target, expr = binding()
    plan = compile_binding({"read": source}, target, expr)
    assert plan["requiredSources"] == ["read"]
    assert plan["mappings"][0]["indexResolution"] == "per_response_columns_metadata"
    assert materialize_binding(plan, {"read": payload()})["arguments"] == [
        {"value": 42},
        {"value": 43},
    ]
    p = payload()
    p["data"][-1][-1] = "must-not-coerce"
    with pytest.raises(DataBindingError) as error:
        materialize_binding(plan, {"read": p})
    assert error.value.pointer == "/target/1/value"
    p["data"][-1].pop()
    with pytest.raises(DataBindingError) as error:
        materialize_binding(plan, {"read": p})
    assert error.value.pointer == "/sources/read/data/1"
    tampered = copy.deepcopy(plan)
    tampered["expression"]["max_rows"] = 5
    with pytest.raises(DataBindingError, match="binding_plan_drift"):
        materialize_binding(tampered, {"read": payload()})


@pytest.mark.parametrize(
    "change", ["source", "pointer", "target", "field", "schema_index", "code"]
)
def test_projection_compiler_rejects_unbound_or_unsupported_forms(change):
    source, target, expr = binding()
    if change == "source":
        expr["source"] = "missing"
    elif change == "pointer":
        expr["pointer"] = None
    elif change == "target":
        target = {"type": "string"}
    elif change == "field":
        expr["fields"] = ["invented"]
    elif change == "schema_index":
        source["properties"]["columns"]["additionalProperties"]["properties"]["index"][
            "type"
        ] = "string"
    else:
        expr["script"] = "eval(...)"
    with pytest.raises(DataBindingError):
        compile_binding({"read": source}, target, expr)


def test_shared_flow_uses_projection_and_keeps_alias_dominance_and_candidate_boundary(
    monkeypatch,
):
    text = (
        "Read synthetic tabular data and propose an inactive export; do not execute it."
    )
    b = _bundle(
        {
            "apiVersion": "effect-runtime.io/translation-intake/v1",
            "candidateId": "column-wiring-fixture",
            "repository": "local/fixture",
            "commitSha": "0" * 40,
            "snapshotDigest": sha256_json(text),
            "entryPath": "SKILL.md",
            "documents": [
                _document(
                    "SKILL.md", text.encode(), mode="100644", origin="developer_fixture"
                )
            ],
            "supplementAttempts": [],
            "parentBundleDigest": None,
        }
    )
    host = IsolatedNetdataHost(
        node="fixture-node", listener="local", device_ip="10.0.0.8"
    )
    contract, bound = host.contract_and_binding(text)
    args = query_arguments(
        host._info, node=host.node, listener=host.listener, device_ip=host.device_ip
    )
    span = {"path": "SKILL.md", "start": 0, "end": len(text), "quote": text}
    effects = {
        "export": StructuredEffectTarget(
            profile="fixture",
            tool="unexecuted_export",
            skill_id="fixture.export",
            contract_hash="sha256:" + "e" * 64,
            input_schema=obj(
                {"rows": array(obj({"TRAP_SEVERITY": {"type": "string"}}))}
            ),
        )
    }
    expr = {
        "kind": "column_rows",
        "source": "page",
        "pointer": "",
        "fields": ["TRAP_SEVERITY"],
        "max_rows": 200,
        "max_columns": 64,
    }
    raw = {
        "api_version": "netopyu.io/structured-flow-tree/v1",
        "source_digest": b["bundleDigest"],
        "purpose": "mechanical column binding to inactive candidate, not full Netdata task",
        "input_schema": obj({}),
        "max_read_age_seconds": 5,
        "steps": [
            {
                "kind": "read",
                "source": span,
                "tool": contract.spec.tool,
                "bind": "page",
                "arguments": {"kind": "literal", "value": args},
            },
            {
                "kind": "effect_candidate",
                "source": span,
                "binding_id": "export",
                "arguments": {"kind": "object", "fields": {"rows": expr}},
            },
        ],
    }
    compiled = compile_structured_tree(
        b,
        StructuredFlowTree.model_validate(raw),
        {contract.spec.tool: contract},
        effects,
    )
    packet = compiled["qualification"]
    result = run_read_flow(
        parse_flow(compiled["flow"]),
        {},
        reads={contract.contract_hash: contract},
        effects=effects,
        bindings={contract.contract_hash: bound},
        context=context(host),
        consent=HostFlowConsent(packet["flowDigest"], sha256_json({})),
    )
    assert (
        result["status"] == "awaiting_effect_admission"
        and result["effectExecuted"] is False
    )
    assert len(host.calls) == 1
    assert len(result["candidate"]["arguments"]["rows"]) == 3
    changed = copy.deepcopy(compiled["flow"])
    changed["nodes"][1]["arguments"]["fields"]["rows"]["source"] = "node-1"
    with pytest.raises(ValueError, match="every incoming path"):
        qualify_flow(parse_flow(changed), {contract.contract_hash: contract}, effects)
    ticks = iter([0.0, 6.0])
    monkeypatch.setattr("network_runtime.l0.flow.time.monotonic", lambda: next(ticks))
    expired = run_read_flow(
        parse_flow(compiled["flow"]),
        {},
        reads={contract.contract_hash: contract},
        effects=effects,
        bindings={contract.contract_hash: bound},
        context=context(host),
        consent=HostFlowConsent(packet["flowDigest"], sha256_json({})),
    )
    assert expired["status"] == "blocked" and "candidate" not in expired
    assert expired["trace"][-1]["diagnostic"]["code"] == "read_evidence_expired"
    expr["source"] = "unknown"
    with pytest.raises(ValueError, match="lexical scope"):
        compile_structured_tree(
            b,
            StructuredFlowTree.model_validate(raw),
            {contract.spec.tool: contract},
            effects,
        )
