"""Structural duty accounting tests, not natural-language accuracy evidence."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation import flow_checkpoint, source_duty_accounting as accounting, source_ledger as ledger, source_plan
from evaluation.source_blocks import citation_blocks
from tests.test_source_plan import arguments, choice, end
from tests.test_source_obligations import packet as packet_fixture, transport as transport_fixture
from tests.test_source_ledger import envelope

packet = packet_fixture
transport = transport_fixture


def inspection(wire):
    content = json.loads(wire["messages"][1]["content"])
    mark = {"block_id": next(b["id"] for b in content["sourceBlocks"] if "先按" in b["text"])}
    def row(requirement, category, handling):
        return {"source": mark, "requirement": requirement, "category": category,
            "phases": ["execution"], "handling": handling, "hostGate": "none",
            "reason": "A developer test declaration of a source duty, not independent Gold."}
    return {"mode": "inspect_obligations", "obligations": [
        row("Observe interfaces for the explicitly supplied device.", "operation", "flow_proposal"),
        row("If the first interface is disabled, read its counters; otherwise finish.", "control_flow", "flow_proposal"),
        row("If errors are zero finish; otherwise propose only a change without writing.", "control_flow", "l1")]}


def inspected(packet):
    state = ledger.initial_state(packet, "plan_first", account_duties=True)
    wire, _ = ledger.make_request(packet, state)
    files, status = ledger.derive(packet, state, wire, envelope(inspection(wire)))
    assert status["candidateStatus"] == "source_obligations_reviewed_not_verified"
    return files["next-state.json"]


def branch(wire):
    value = choice(wire)
    mark = value["region"]["steps"][0]["source"]
    completed = {**end(mark), "outcome": "read_path_completed"}
    value["region"]["steps"].append({"kind": "if_equal", "source": mark,
        "left": {"kind": "reference", "source": "/steps/0", "pointer": "/interfaces/0/adminUp"}, "equals": False,
        "when_equal": {"steps": [{"kind": "read", "source": mark, "tool": "get_interface_counters",
            "whyNeeded": "Observe counters for the exact previously discovered interface."}], "exit": end(mark)},
        "otherwise": {"steps": [], "exit": completed}})
    value["region"]["exit"] = {"kind": "already_closed"}
    return value


def setup(packet):
    state = inspected(packet)
    wire, _ = ledger.make_request(packet, state)
    plan = source_plan.prepare(branch(wire), packet, citation_blocks(ledger.frame(packet, state)))
    inv = accounting.inventory(packet, state)
    return inv, plan


def audit(inv, plan):
    return {"mode": "account_source_duties", "inventoryDigest": inv["reportDigest"], "planDigest": plan["reportDigest"], "duties": {
        "o000": {"disposition": "represented", "nodePointers": ["/steps/0"], "requiredBefore": [],
                 "explanation": "This read observes the interfaces as required by the supplied declaration."},
        "o001": {"disposition": "represented", "nodePointers": ["/steps/1"], "requiredBefore": [{
            "readPointer": "/steps/1/when_equal/0", "guardPointer": "/steps/1", "branch": "when_equal"}],
            "explanation": "Only the disabled-interface branch reaches the dependent counter read."},
        "o002": {"disposition": "handoff", "terminalPointers": ["/steps/1/when_equal/1"], "requiredBefore": [],
            "explanation": "After counters are read, L1 must retain the zero/nonzero decision and proposal-only limit."}}}


def test_exact_inventory_and_frozen_nodes_generate_valid_schema(packet):
    inv, plan = setup(packet)
    schema = accounting.schema(packet, inv, plan)
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(audit(inv, plan))
    result = accounting.check(packet, inv, plan, audit(inv, plan))
    assert result["structuralAccountingPassed"] and result["inventoryRowsAccounted"] == 3
    assert result["handoffs"][0]["duties"][0]["dutyId"] == "o002"
    assert "zero" in result["handoffs"][0]["duties"][0]["requirement"]
    assert not result["sourceCoverageProven"] and not result["guardMeaningAndPolarityVerified"]
    assert not result["runtimeAuthorityGranted"] and result["unlistedSourceDutiesMayExist"]


@pytest.mark.parametrize("change", ["omit", "invent", "location", "digest", "unknown_field"])
def test_omission_or_invented_location_never_passes(packet, change):
    inv, plan = setup(packet)
    proposal = audit(inv, plan)
    if change == "omit":
        del proposal["duties"]["o002"]
    elif change == "invent":
        proposal["duties"]["new"] = proposal["duties"]["o002"]
    elif change == "location":
        proposal["duties"]["o000"]["nodePointers"] = ["/steps/99"]
    elif change == "digest":
        proposal["planDigest"] = "sha256:wrong"
    else:
        proposal["runtimeAuthorityGranted"] = True
    with pytest.raises(ValueError, match="exactly"):
        accounting.check(packet, inv, plan, proposal)


@pytest.mark.parametrize("mutation,code", [("observation", "observation_is_not_a_decision"),
    ("polarity", "required_branch_does_not_dominate_read"), ("late", "required_branch_does_not_dominate_read"),
    ("handoff", "included_read_prerequisite_not_represented"),
    ("scope", "included_read_prerequisite_not_represented"), ("unresolved", "unresolved_source_duty")])
def test_specific_duty_and_read_diagnostics(packet, mutation, code):
    inv, plan = setup(packet)
    proposal = audit(inv, plan)
    row = proposal["duties"]["o001"]
    if mutation == "observation":
        row["nodePointers"] = ["/steps/0"]
        row["requiredBefore"] = []
    elif mutation == "polarity":
        row["requiredBefore"][0]["branch"] = "otherwise"
    elif mutation == "late":
        row["requiredBefore"][0]["readPointer"] = "/steps/0"
    elif mutation == "handoff":
        row["disposition"] = "handoff"
        row.pop("nodePointers")
        row["terminalPointers"] = ["/steps/1/when_equal/1"]
    else:
        row["disposition"] = "outside_task" if mutation == "scope" else "unresolved"
        row.pop("nodePointers")
    result = accounting.check(packet, inv, plan, proposal)
    assert not result["structuralAccountingPassed"]
    assert code in {i["code"] for i in result["issues"]}
    assert all(i["dutyId"] == "o001" for i in result["issues"])


def test_preceding_check_is_not_enough_when_both_branches_rejoin(packet):
    inv, plan = setup(packet)
    plan = copy.deepcopy(plan)
    conditional = plan["tree"]["steps"][1]
    counter, terminal = conditional["when_equal"]
    conditional["when_equal"] = []
    conditional["otherwise"] = []
    plan["tree"]["steps"].extend([counter, terminal])
    plan = ledger.prior.seal({k: v for k, v in plan.items() if k != "reportDigest"})
    proposal = audit(inv, plan)
    proposal["duties"]["o001"]["requiredBefore"][0]["readPointer"] = "/steps/2"
    proposal["duties"]["o002"]["terminalPointers"] = ["/steps/3"]
    result = accounting.check(packet, inv, plan, proposal)
    assert result["issues"][0]["code"] == "required_branch_does_not_dominate_read"


@pytest.mark.parametrize("category", ["control_flow", "operation"])
def test_explicit_dependency_guard_needs_no_duplicate_node_entry(packet, category):
    inv, plan = setup(packet)
    inv["obligations"][1]["category"] = category
    inv = ledger.prior.seal({k: v for k, v in inv.items() if k != "reportDigest"})
    proposal = audit(inv, plan)
    proposal["duties"]["o001"]["nodePointers"] = ["/steps/1/when_equal/0"]
    result = accounting.check(packet, inv, plan, proposal)
    assert result["structuralAccountingPassed"]
    proposal["duties"]["o001"]["requiredBefore"][0]["branch"] = "otherwise"
    assert not accounting.check(packet, inv, plan, proposal)["structuralAccountingPassed"]


def test_vague_handoff_cannot_replace_source_duties(packet):
    inv, plan = setup(packet)
    proposal = audit(inv, plan)
    proposal["duties"]["o002"] = {"disposition": "outside_task", "requiredBefore": [], "explanation": "A claimed scope exclusion cannot account for a needs-L1 terminal."}
    result = accounting.check(packet, inv, plan, proposal)
    assert result["issues"][0]["code"] == "handoff_without_source_duties"
    assert result["issues"][0]["planPointer"] == "/steps/1/when_equal/1"


def test_generic_host_gate_cannot_implement_domain_condition(packet):
    inv, plan = setup(packet)
    proposal = audit(inv, plan)
    proposal["duties"]["o001"] = {"disposition": "future_host_gate", "requiredBefore": [],
        "gateIds": [next(iter(ledger.obligations.host_gates(packet)))], "explanation": "This intentionally invalid mapping pretends a host gate checks the domain condition."}
    result = accounting.check(packet, inv, plan, proposal)
    assert result["issues"][0]["code"] == "host_gate_cannot_discharge_domain_duty"
    assert not result["accountedDuties"][1]["hostGates"][0]["satisfied"]


@pytest.mark.parametrize("what", ["plan", "inventory", "task", "catalog"])
def test_digest_and_context_drift_rejected(packet, what):
    inv, plan = setup(packet)
    proposal = audit(inv, plan)
    if what == "plan":
        plan["tree"]["purpose"] += " changed"
    elif what == "inventory":
        inv["obligations"][0]["requirement"] += " changed"
    elif what == "task":
        packet["task"] += " changed"
    else:
        packet["catalog"]["tools"][0]["description"] = "changed"
    with pytest.raises(ValueError, match="drift"):
        accounting.check(packet, inv, plan, proposal)


def test_inventory_preserves_exact_source_and_cannot_invent_it(packet):
    state = inspected(packet)
    state["obligationReview"]["obligations"][0]["source"]["quote"] += " changed"
    with pytest.raises(ValueError, match="location drift"):
        accounting.inventory(packet, state)


def test_trace_budget_is_explicit_and_does_not_crop(packet, monkeypatch):
    _, plan = setup(packet)
    monkeypatch.setattr(accounting, "MAX_TRACES", 1)
    with pytest.raises(ValueError, match="budget"):
        accounting.plan_paths(plan)


def test_opt_in_inspection_remains_task_and_plan_isolated(packet):
    state = ledger.initial_state(packet, "plan_first", account_duties=True)
    wire, budget = ledger.make_request(packet, state)
    content = json.loads(wire["messages"][1]["content"])
    assert budget["accepted"]
    assert wire["messages"][0]["content"] == ledger.obligations.SYSTEM
    assert "currentTask" not in content and "hostCatalog" not in content and "frozenPlan" not in content
    assert content["requiredOutputSchema"] == wire["format"]
    plain, _ = ledger.make_request(packet, ledger.initial_state(packet, "plan_first"))
    assert plain["messages"][0]["content"] == source_plan.SYSTEM


@pytest.mark.parametrize("profile,option", [("direct", True), ("plan_first", 1), ("plan_first", "true")])
def test_accounting_requires_explicit_opt_in(packet, profile, option):
    with pytest.raises(ValueError, match="explicit"):
        ledger.initial_state(packet, profile, account_duties=option)


def one_read_audit(wire):
    context = json.loads(wire["messages"][1]["content"])
    return {"mode": "account_source_duties", "inventoryDigest": context["sourceInventory"]["reportDigest"],
        "planDigest": context["frozenPlan"]["planDigest"], "duties": {
            "o000": {"disposition": "represented", "nodePointers": ["/steps/0"], "requiredBefore": [],
                "explanation": "The included read observes the explicit device's interface list."},
            **{ident: {"disposition": "handoff", "terminalPointers": ["/steps/1"], "requiredBefore": [],
                "explanation": "The conditional inspection and proposal-only work remain L1 responsibilities after this read."}
                for ident in ("o001", "o002")}}}


def test_complete_opt_in_pipeline_retains_handoffs_and_replays_offline(packet, transport, tmp_path, monkeypatch):
    queue, calls = transport
    queue.extend([inspection, choice, one_read_audit, arguments])
    root = tmp_path / "accounted"
    ledger.freeze(packet, root, profile="plan_first", account_duties=True)
    partial = ledger.run(root, max_new_calls=3)
    assert partial["status"] == "new_call_budget_exhausted" and len(calls) == 3
    result = ledger.run(root, max_new_calls=1)
    assert result["compiled"] and result["modelCallsRecorded"] == len(calls) == 4
    more = json.loads((root / "round-003/remaining.json").read_text())
    assert more["originalPlanRemaining"] == [] and len(more["duties"]) == 2
    handoffs = json.loads((root / "round-003/handoffs.json").read_text())
    assert handoffs["boundaries"][0]["terminalPointer"] == "/steps/1"
    assert not handoffs["runtimeAuthorityGranted"] and not more["resolved"]
    monkeypatch.setattr(flow_checkpoint, "send", lambda *a: pytest.fail("offline replay must not call model"))
    assert ledger.run(root) == result


def test_blocked_accounting_stops_before_argument_calls(packet, transport, tmp_path):
    queue, calls = transport
    def bad(wire):
        proposal = one_read_audit(wire)
        proposal["duties"]["o001"] = {"disposition": "represented", "nodePointers": ["/steps/0"],
            "requiredBefore": [], "explanation": "Incorrectly treating a read as the missing conditional inspection."}
        return proposal
    queue.extend([inspection, choice, bad])
    root = tmp_path / "blocked"
    ledger.freeze(packet, root, profile="plan_first", account_duties=True)
    result = ledger.run(root, max_new_calls=6)
    assert result["status"] == "source_duty_accounting_blocked" and len(calls) == 3
    assert result["argumentSlotsCompleted"] == 0 and not result["compiled"]
    assert result["sourceDutyAccounting"]["issues"][0]["code"] == "observation_is_not_a_decision"


def test_compilation_attachment_rechecks_edited_report(packet):
    inv, plan = setup(packet)
    report = accounting.check(packet, inv, plan, audit(inv, plan))
    report["handoffs"] = []
    report = ledger.prior.seal({k: v for k, v in report.items() if k != "reportDigest"})
    with pytest.raises(ValueError, match="drift"):
        accounting.attach(packet, plan, report, {})
