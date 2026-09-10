"""Generic representation/security regressions; not public-Skill accuracy Gold."""
import copy
import json
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from evaluation import source_closed_program, source_ledger as ledger, source_plan, source_program
from evaluation.source_gap_search import search_gaps
from network_runtime.l0 import flow as engine
from tests.test_source_semantic_plan import prepared, envelope, slot_arguments
from tests.test_source_obligations import packet as packet_fixture
from tests.test_structured_flow import setup, execute, host_bindings

packet = packet_fixture


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


@pytest.mark.parametrize("left,right,valid", [
    (("obs0", "/text"), ("input", "/request"), True),
    (("input", "/request"), ("obs0", "/text"), True),
    (("obs0", "/request"), ("input", "/request"), False),
    (("obs0", "/text"), ("input", "/text"), False),
    (("input", "/count"), ("obs0", "/text"), False),
    (("input", "/count"), ("obs0", "/count"), True),
])
def test_both_decoder_operands_couple_input_and_output_paths(left, right, valid):
    catalog = {"tools": [{"name": "fetch", "outputSchema": object_schema({
        "text": {"type": "string"}, "count": {"type": "integer"}})}]}
    inputs = object_schema({"request": {"type": "string"}, "count": {"type": "integer"}})
    schema = source_closed_program.schema(catalog, observation_names=["obs0"], evidence_ids=["witness"],
                                         value_paths=source_plan.planning_sources(catalog, inputs))
    def ref(item):
        return {"kind": "field", "source": item[0], "pointer": item[1]}
    end = {"op": "complete", "source_id": "witness"}
    node = {"op": "if_equal", "source_id": "witness", "value": ref(left), "equals": ref(right),
            "when_equal": end, "otherwise": end}
    assert Draft202012Validator(schema).is_valid(node) is valid


def test_reference_comparison_survives_every_authoring_lowering(packet):
    state, wire, choice = prepared(packet)
    branch = choice["program"][1]
    branch["value"] = {"kind": "field", "source": "obs0", "pointer": "/interfaces/0/name"}
    branch["equals"] = {"kind": "field", "source": "input", "pointer": "/device/id"}
    files, result = ledger.derive(packet, state, wire, envelope(choice))
    assert result["candidateStatus"] == "operation_plan_recorded", result
    assert files["prepared-plan.json"]["tree"]["steps"][1]["equals"]["source"] == "input"
    state = files["next-state.json"]
    wire, _ = ledger.make_request(packet, state)
    files, result = ledger.derive(packet, state, wire, envelope(slot_arguments(wire)))
    assert result["candidateStatus"] == "compiled_region_requires_semantic_review", result
    branch = files["compilation.json"]["flow"]["nodes"][1]
    assert branch["equals"] == {"kind": "reference", "source": "input", "pointer": "/device/id"}
    assert not files["compilation.json"]["runtimeAuthorityGranted"]


@pytest.mark.parametrize("rhs", ['field(later, "/value", "witness")',
                                 'field(obs0, "/request", "witness")',
                                 '__import__("os")', 'input',
                                 'field(input, "/request", "unrelated")'])
def test_rhs_rejects_future_wrong_schema_arbitrary_code_and_wrong_witness(rhs):
    program = ('obs0 = read("fetch", "witness", "Original observed evidence")\n'
               'if field(obs0, "/value", "witness") == ' + rhs + ':\n'
               '    end("read_path_completed", "witness", "Closed read procedure")\n'
               'else:\n    end("read_path_completed", "witness", "Closed read procedure")\n')
    with pytest.raises(ValueError):
        source_program.parse(program, input_schema=object_schema({"request": {"type": "string"}}),
            catalog={"tools": [{"name": "fetch", "outputSchema": object_schema({"value": {"type": "string"}})}]})


@pytest.mark.parametrize("a,b,matched", [("x", "x", True), ("x", "y", False),
    (None, None, True), (False, 0, False), (1, True, False), (1, 1.0, True)])
def test_runtime_reference_equality_is_typed_not_python_coercion(a, b, matched):
    scalar = {"type": ["string", "number", "boolean", "null"]}
    proposal = engine.parse_flow({"api_version": "netopyu.io/l0-flow-proposal/v2",
        "source_digest": "sha256:" + "a" * 64, "purpose": "Synthetic representation test only",
        "input_schema": object_schema({"a": scalar, "b": scalar}), "entry": "compare", "max_read_age_seconds": 5,
        "nodes": [{"kind": "branch", "id": "compare",
            "left": {"kind": "reference", "source": "input", "pointer": "/a"},
            "equals": {"kind": "reference", "source": "input", "pointer": "/b"}, "on_true": "yes", "on_false": "no"},
            {"kind": "end", "id": "yes", "outcome": "read_path_completed", "explanation": "Matched"},
            {"kind": "end", "id": "no", "outcome": "needs_l1", "explanation": "Not matched"}]})
    result = execute(proposal, {}, {}, {}, arguments={"a": a, "b": b})
    assert result["trace"][0]["matched"] is matched
    assert result["effectExecuted"] is False


def test_rhs_only_control_evidence_expires_before_downstream_effect_candidate(monkeypatch):
    _, _, reads, effects, _, proposal = setup()
    raw = proposal.model_dump(mode="json")
    raw["nodes"][1]["left"] = {"kind": "reference", "source": "input", "pointer": "/device/id"}
    raw["nodes"][1]["equals"] = {"kind": "reference", "source": "node-0", "pointer": "/interfaces/0/name"}
    branch = raw["nodes"][1]
    branch["on_true"], branch["on_false"] = branch["on_false"], branch["on_true"]
    proposal = engine.parse_flow(raw)
    now, calls = [0.0], []
    monkeypatch.setattr(engine.time, "monotonic", lambda: now[0])
    bindings = host_bindings(reads, calls, interfaces=[{"name": "eth0", "adminUp": False}], errors=7)
    key = reads["get_interface_counters"].contract_hash
    observe = bindings[key].observe
    def delayed(args):
        result = observe(args)
        now[0] = 6.0
        return result
    bindings[key] = replace(bindings[key], observe=delayed)
    result = execute(proposal, reads, effects, bindings)
    assert result["status"] == "blocked" and len(calls) == 2
    assert result["trace"][-1]["diagnostic"]["code"] == "read_evidence_expired"
    assert "candidate" not in result and not result["effectExecuted"]


def test_lossless_pages_rejoin_upstream_midword_cuts_and_preserve_lines(monkeypatch):
    text = "".join(f"第 {i} 行 original complete sentence without invented meaning.\n" for i in range(200))
    split = 2191
    base = {"path": "SKILL.md", "sourceDigest": "unchanged"}
    upstream = {str(i): {**base, "start": a, "end": b, "text": text[a:b], "startLine": 1 + text[:a].count("\n")}
                for i, (a, b) in enumerate(((0, split), (split, len(text))))}
    before = copy.deepcopy(upstream)
    monkeypatch.setattr(ledger.prior, "pages_for", lambda _: upstream)
    pages = list(ledger.pages_for({"bundle": {"references": []}}).values())
    assert "".join(p["text"] for p in pages) == text and upstream == before
    assert all(p["text"].endswith("\n") and len(p["text"].encode()) <= 2048 for p in pages)
    assert all(p["text"] == text[p["start"]:p["end"]] for p in pages)


def test_oversized_unbroken_utf8_line_is_bounded_and_not_lost(monkeypatch):
    text = "密🧪" * 2000
    monkeypatch.setattr(ledger.prior, "pages_for", lambda _: {"p": {"path": "long.md", "sourceDigest": "x",
        "start": 0, "end": len(text), "text": text, "startLine": 1}})
    pages = list(ledger.pages_for({"bundle": {"references": []}}).values())
    assert "".join(p["text"] for p in pages) == text
    assert all(0 < len(p["text"].encode()) <= 2048 for p in pages)


def test_business_gap_retrieves_inert_reference_before_admitting_broken_program(packet, monkeypatch):
    state, wire, choice = prepared(packet)
    pages = ledger.pages_for(packet)
    pages["p999"] = {**next(iter(pages.values())), "path": "source/references/details.md", "text": "Actual retained details."}
    monkeypatch.setattr(ledger, "pages_for", lambda _: pages)
    source = {"block_id": json.loads(wire["messages"][1]["content"])["sourceBlocks"][0]["id"]}
    choice["business_gaps"] = [{"source": source, "explanation": "Cannot inspect `references/details.md` yet."}]
    choice["program"][1]["value"]["source"] = "obs7"  # decoder-valid, undefined: must never admit it
    files, result = ledger.derive(packet, state, wire, envelope(choice))
    assert result["candidateStatus"] == "source_window_requested", result
    assert files["next-state.json"]["window"] == ["p999"]
    assert not files["provisional-plan-gaps.json"]["draftAdmitted"]
    assert "prepared-plan.json" not in files
    assert not files["gap-source-search.json"]["semanticResolutionProven"]


def test_literal_path_navigation_never_executes_scripts_or_revisits_delivered_pages():
    pages = {"p0": {"path": "original/scripts/check.py", "text": "dangerous inert code", "start": 0}}
    gaps = [{"missing": "Need `scripts/check.py`."}]
    result = search_gaps(gaps, pages, {"submitted": [], "window": []})
    assert result["selectedPages"] == ["p0"]
    assert result["networkCalls"] == result["sourceScriptCalls"] == 0
    assert not result["semanticResolutionProven"]
    assert not search_gaps(gaps, pages, {"submitted": ["p0"], "window": []})["selectedPages"]
