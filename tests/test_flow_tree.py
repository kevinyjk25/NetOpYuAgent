"""Offline compiler semantics; hand-authored trees are not LLM translation Gold."""

import json
import random

import pytest

from evaluation.flow_tree import FlowTree, assess_tree, compile_report, compile_tree, tree_review_input
from evaluation.flow_translation import local_sources, lower
from evaluation.flow_source_selection import expand
from evaluation.flow_grounded_translation import project
from evaluation.read_local_demo import host_binding
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_source_alignment import ClaimAssessment, SourceAssessment
from network_provider.local_inventory import LocalInventoryReader
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import DataSensitivity
from network_runtime.contracts import sha256_json
from network_runtime.l0.flow import HostFlowConsent, run_read_flow


def ref(source="input", field="device_id"):
    return {"kind": "reference", "source": source, "field": field}


def const(value):
    return {"kind": "constant", "value": value}


def read(alias, argument=None):
    return {"kind": "read", "source_id": "s0001", "tool": "read_inventory_device", "bind": alias,
            "arguments": {"device_id": argument or ref()}}


def end(outcome="read_path_completed"):
    return {"kind": "end", "source_id": "s0001", "outcome": outcome}


def branch(yes=(), no=(), *, left=None, equals="campus"):
    return {"kind": "if_equal", "source_id": "s0001", "left": left or ref("first", "site"),
        "equals": const(equals), "true_source_id": "s0001", "false_source_id": "s0001",
        "when_equal": list(yes), "otherwise": list(no)}


def tree(steps):
    return FlowTree.model_validate({"business_source_ids": ["s0001"], "steps": steps, "issues": []})


def sources():
    return local_sources().model_copy(update={"source_text":
        "Hand-authored compiler fixture only; citations are not semantic evidence or live device health.\n"})


def run_existing(tmp_path, ast, device, *, scopes=None):
    source = sources()
    selected, _ = compile_tree(source, ast)
    flow, packet = lower(source, project(source, expand(source, selected)))
    dataset = tmp_path / "inventory.json"
    payload = {"campus-sw1": {"site": "campus", "status": "planned-lab"},
               "idc-sw1": {"site": "idc", "status": "planned-lab"}}
    dataset.write_text(json.dumps(payload))
    before = dataset.read_bytes()
    calls = []

    class RecordedReader(LocalInventoryReader):
        def observe(self, arguments):
            calls.append(dict(arguments))
            return super().observe(arguments)

    reads = {contract.contract_hash: contract for contract in source.reads.values()}
    bindings = {key: host_binding(contract, RecordedReader(dataset)) for key, contract in reads.items()}
    context = ObservationAccessContext(subject_id="offline-compiler-test", roles=frozenset({"network-reader"}),
        scopes=frozenset(scopes if scopes is not None else {"inventory:read", "device_id:campus-sw1", "device_id:idc-sw1"}),
        purpose="Explicit local compiler regression", clearance=DataSensitivity.INTERNAL)
    args = {"device_id": device}
    actual = run_read_flow(flow, args, reads=reads, effects={}, bindings=bindings, context=context,
                          consent=HostFlowConsent(packet["flowDigest"], sha256_json(args)))
    assert dataset.read_bytes() == before
    return actual["status"], calls


def interpret_tree_for_test(ast, device):
    """Independent test oracle, never exported or usable for production execution."""
    calls = []

    def value(v, env):
        return v["value"] if v["kind"] == "constant" else env[v["source"]][v["field"]]

    def block(statements, parent):
        env = dict(parent)
        for item in statements:
            if item["kind"] == "read":
                args = {key: value(v, env) for key, v in item["arguments"].items()}
                calls.append(args)
                identifier = args["device_id"]
                env[item["bind"]] = {"device_id": identifier,
                    "site": "campus" if identifier == "campus-sw1" else "idc", "status": "planned-lab"}
            elif item["kind"] == "end":
                return item["outcome"]
            else:
                selected = item["when_equal"] if value(item["left"], env) == value(item["equals"], env) else item["otherwise"]
                stopped = block(selected, env)
                if stopped:
                    return stopped
        return None
    result = block(ast.model_dump(mode="json")["steps"], {"input": {"device_id": device}})
    return result, calls


PROGRAMS = [
    [read("first"), end()],
    [read("first"), read("second", ref("first")), end()],
    [read("first"), branch([end("needs_l1")], [read("second", ref("first")), end()])],
    [read("first"), branch([read("campus", const("campus-sw1"))], [read("dc", const("idc-sw1"))]), read("common"), end()],
    [read("first"), branch([branch([end()], [end("unsupported")], left=ref("first", "status"), equals="planned-lab")], [end("needs_l1")])],
    [read("first"), branch([], [end("unsupported")]), read("after"), end()],
    [read("first"), branch([], []), end()],
]


@pytest.mark.parametrize("device", ["campus-sw1", "idc-sw1"])
@pytest.mark.parametrize("program", PROGRAMS)
def test_actual_read_runtime_matches_independent_tree_semantics(tmp_path, program, device):
    ast = tree(program)
    assert run_existing(tmp_path, ast, device) == interpret_tree_for_test(ast, device)


def test_common_continuation_emitted_once_and_all_edges_forward():
    selected, origins = compile_tree(sources(), tree(PROGRAMS[3]))
    nodes = selected.model_dump(mode="json")["steps"]
    assert len(nodes) == 6
    assert nodes[2]["next"] == nodes[3]["next"] == 4
    assert nodes[4]["requires"] == [0, 1]  # Neither branch-local read dominates the join.
    assert origins[2]["treePointer"] == "/steps/1/when_equal/0"
    for index, node in enumerate(nodes):
        assert all(node[edge] > index for edge in ("next", "on_true", "on_false") if edge in node)


@pytest.mark.parametrize("program", [
    [read("first")],
    [end(), read("later")],
    [read("first"), branch([end()], [end()]), read("later"), end()],
    [read("first"), branch([], [end()])],
    [read("input"), end()],
    [read("same"), read("same"), end()],
    [read("self", ref("self")), end()],
    [read("first"), branch([read("local")], []), read("after", ref("local")), end()],
    [read("first"), branch([read("local")], [read("after", ref("local"))]), end()],
    [read("first"), branch([read("same")], [read("same")]), end()],
])
def test_unreachable_incomplete_or_ambiguous_trees_rejected(program):
    with pytest.raises(ValueError):
        compile_tree(sources(), tree(program))


@pytest.mark.parametrize("field,value", [("next", 9), ("id", "foo"), ("requires", [0]), ("entry", 0)])
def test_model_cannot_supply_graph_wiring(field, value):
    program = [read("first"), end()]
    program[0][field] = value
    with pytest.raises(ValueError):
        tree(program)


def test_compiler_preserves_wrong_but_well_typed_business_polarity():
    wrong = tree([read("first"), branch([end()], [end("needs_l1")])])
    selected, _ = compile_tree(sources(), wrong)
    assert selected.steps[selected.steps[1].on_true].outcome == "read_path_completed"
    assert selected.steps[selected.steps[1].on_false].outcome == "needs_l1"
    # Only source review can establish whether this was the requested business polarity.
    assert compile_report(sources(), wrong)["runtimeAuthorityGranted"] is False


def test_global_node_and_depth_budgets():
    with pytest.raises(ValueError, match="64"):
        compile_tree(sources(), tree([read("first"), branch([read(f"a{i}") for i in range(33)], [read(f"b{i}") for i in range(33)]), end()]))
    nested = end()
    for _ in range(17):
        nested = branch([nested], [end()], left=ref(), equals="campus-sw1")
    with pytest.raises(ValueError, match="16"):
        compile_tree(sources(), tree([nested]))


def test_compiled_branch_does_not_turn_access_error_into_false(tmp_path):
    ast = tree(PROGRAMS[2])
    assert run_existing(tmp_path, ast, "idc-sw1", scopes={"inventory:read", "device_id:campus-sw1"}) == ("blocked", [])


@pytest.mark.parametrize("mutation", ["unknown_tool", "unknown_field", "wrong_type", "unknown_source_id"])
def test_existing_contract_and_source_checks_remain_mandatory(mutation):
    program = [read("first"), end()]
    if mutation == "unknown_tool":
        program[0]["tool"] = "undeclared_tool"
    elif mutation == "unknown_field":
        program[0]["arguments"]["device_id"] = ref(field="unavailable")
    elif mutation == "wrong_type":
        program[0]["arguments"]["device_id"] = const(True)
    else:
        program[0]["source_id"] = "s9999"
    with pytest.raises(ValueError):
        compile_tree(sources(), tree(program))


def test_effect_is_terminal_candidate_not_execution():
    from network_runtime.l0.flow import EffectTarget
    source = sources()
    target = EffectTarget(profile="test", tool="test_change", skill_id="test.change", contract_hash=sha256_json("test-only"), input_schema=source.input_schema)
    source = source.model_copy(update={"effects": {"host_change": target}})
    effect = {"kind": "effect_candidate", "source_id": "s0001", "binding_id": "host_change", "arguments": {"device_id": ref("first")}}
    result = compile_report(source, tree([read("first"), effect]))
    assert result["flow"]["nodes"][-1]["kind"] == "effect_candidate"
    assert not result["runtimeAuthorityGranted"]
    with pytest.raises(ValueError, match="unreachable"):
        compile_tree(source, tree([read("first"), effect, end()]))


def test_issues_still_block_despite_structural_qualification():
    source, ast = sources(), tree(PROGRAMS[0])
    raw = ast.model_dump(mode="json")
    raw["issues"] = [{"kind": "source_ambiguity", "source_id": "s0001", "question": "A genuine missing source fact in a protocol test?"}]
    ast = FlowTree.model_validate(raw)
    assert assess_tree(source, ast, review_for(tree_review_input(source, ast)))["status"] == "blocked"


def review_for(packet):
    return ReadL05Review(reviewer_id="mechanical-test-only", reviewer_kind="test_fixture",
        assessment=SourceAssessment(input_digest=packet["inputDigest"], scope_note="Mechanical test of review binding, not semantic evidence.",
            claims=tuple(ClaimAssessment(claim_id=c["claimId"], verdict="supported",
                source_span_ids=tuple(c.get("requiredCitationId", "skill-0001") if kind == "skill" else "host-0001"
                                     for kind in c["requiredEvidenceKinds"]),
                rationale="Mechanical fixture, not proof of source meaning.", suggested_revision="Inspect original source.")
                for c in packet["claims"])))


def test_alias_rename_keeps_graph_but_invalidates_ast_review():
    source, ast = sources(), tree(PROGRAMS[1])
    packet = tree_review_input(source, ast)
    review = review_for(packet)
    assert not assess_tree(source, ast, review)["runtimeAuthorityGranted"]
    raw = ast.model_dump(mode="json")
    raw["steps"][0]["bind"] = "renamed"
    raw["steps"][1]["arguments"]["device_id"]["source"] = "renamed"
    changed = FlowTree.model_validate(raw)
    assert compile_tree(source, ast)[0] == compile_tree(source, changed)[0]
    with pytest.raises(ValueError, match="digest"):
        assess_tree(source, changed, review)


def test_example_is_replayable_compilation_not_execution():
    from evaluation.flow_tree import example_report
    result = example_report()
    assert result["modelCalls"] == result["runtimeExecutions"] == 0
    assert result["reportDigest"] == sha256_json({k: v for k, v in result.items() if k != "reportDigest"})
    assert len(result["flow"]["nodes"]) == len(result["origins"]) == 5
    assert result["flow"]["nodes"][1]["on_true"] == "node-2"
    assert result["flow"]["nodes"][1]["on_false"] == "node-4"


@pytest.mark.parametrize("seed", range(20))
def test_generated_known_trees_match_reference_on_both_sites(tmp_path, seed):
    rng, counter = random.Random(seed), 0

    def make(depth):
        nonlocal counter
        counter += 1
        name = f"r{counter}"
        statements = [read(name, const(rng.choice(["campus-sw1", "idc-sw1"])))]
        if depth:
            statements.append(branch(make(depth - 1), make(depth - 1), left=ref(name, "site"), equals=rng.choice(["campus", "idc"])))
        return statements

    ast = tree([*make(2), end()])
    for device in ("campus-sw1", "idc-sw1"):
        assert run_existing(tmp_path, ast, device) == interpret_tree_for_test(ast, device)
