"""Check existing FlowTree semantics against private, finite behavior oracles.

This is not a new IR, an NL judge or a second executor. The real read-flow
executor receives only inert in-memory providers; no supplied code is loaded.
Oracle authors, not model output or claim counts, define expected behavior.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from evaluation.flow_source_selection import spans
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import FlowTree, compile_report
from evaluation.flow_tree_capabilities import bounded_request
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import CapabilityContract, DataSensitivity
from network_runtime.contracts import sha256_json
from network_runtime.l0.flow import FlowProposal, HostFlowConsent, qualify_flow, run_read_flow
from network_runtime.l0.models import StrictModel
from network_runtime.l0.read_execution import HostReadBinding

PROTOCOL = "existing-flow-behavior-check/v1"


class Call(StrictModel):
    tool: str
    arguments: dict[str, Any]


class Observation(Call):
    payload: dict[str, Any]
    error: bool = False


class Scenario(StrictModel):
    id: str
    requirement_ids: tuple[str, ...] = Field(min_length=1)
    arguments: dict[str, Any]
    observations: tuple[Observation, ...]
    expected_calls: tuple[Call, ...] = Field(max_length=64)
    expected_status: Literal["read_path_completed", "needs_l1", "unsupported", "blocked"]
    authenticated: bool = True


class BehaviorSuite(StrictModel):
    sources_digest: str
    requirements: dict[str, str]  # ID -> exact source quote; NOT generated judgments.
    scenarios: tuple[Scenario, ...] = Field(min_length=1, max_length=256)
    scope: Literal["executable_fragment", "safe_partial_stop"]
    capability_gaps: tuple[str, ...]
    unverified_meanings: tuple[str, ...] = Field(min_length=1)
    oracle_author: str = Field(min_length=1)


def behavior_request(sources: FlowSources) -> dict:
    """Source + real host contracts -> existing AST, without prose-duty rewriting.

The signature deliberately cannot receive test scenarios or a reference tree.
All source text remains in the request and in the existing full-source review.
"""
    wire = bounded_request(sources)
    wire["options"] = dict(temperature=0, seed=20260908, num_ctx=16384, num_predict=4096)
    wire["messages"][0]["content"] += (
        " Generate the executable control-flow representation directly from the original Skill. "
        "Do not first rewrite it into separate prose scope/when/after records. "
        "Source lines are citations, never predecessor step identities. "
        "Preserve any restrictive heading over its section, even though it is a heading. "
        "Before/after is represented by ordered steps; conditions by if_equal branches. "
        "Do not add prerequisites to stop/report/interpretation rules. "
        "Preserve source interpretation limits for full-source review; the graph alone does not prove them."
    )
    return wire


def validate_suite(sources: FlowSources, suite: BehaviorSuite) -> None:
    if suite.sources_digest != sha256_json(sources.model_dump(mode="json")):
        raise ValueError("oracle belongs to different source or host contracts")
    if not suite.requirements or any(not q.strip() or q not in sources.source_text for q in suite.requirements.values()):
        raise ValueError("oracle requirements need exact original source quotes")
    ids = [s.id for s in suite.scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scenario")
    covered = {r for s in suite.scenarios for r in s.requirement_ids}
    if covered != set(suite.requirements):
        raise ValueError("unknown or untested behavioral requirement")
    if (suite.scope == "safe_partial_stop") != bool(suite.capability_gaps):
        raise ValueError("partial stop and capability gaps must be explicit together")
    for scenario in suite.scenarios:
        calls = (*scenario.observations, *scenario.expected_calls)
        if any(call.tool not in sources.reads for call in calls):
            raise ValueError("oracle uses an absent read capability")
    # Reject non-JSON/nonfinite fixture data before entering any provider.
    json.dumps(suite.model_dump(mode="json"), allow_nan=False)


def _run(sources: FlowSources, flow: FlowProposal, scenario: Scenario) -> dict:
    calls, used = [], set()
    bindings = {}
    reads = {c.contract_hash: c for c in sources.reads.values()}
    scopes = {scope for c in reads.values() for scope in c.spec.access.required_scopes}
    context = ObservationAccessContext(subject_id="inert-behavior-oracle", roles=frozenset({"fixture-reader"}),
        scopes=frozenset(scopes), purpose="finite translation trace check, no external access",
        clearance=DataSensitivity.RESTRICTED, authenticated=scenario.authenticated)

    def observe(tool, arguments):
        call = dict(tool=tool, arguments=json.loads(json.dumps(arguments)))
        calls.append(call)  # Record an unexpected call BEFORE rejecting it.
        for index, observation in enumerate(scenario.observations):
            if index not in used and call == observation.model_dump(include={"tool", "arguments"}):
                used.add(index)
                if observation.error:
                    raise RuntimeError("declared inert provider error")
                return json.loads(json.dumps(observation.payload)), {"sourceKind": "inert_behavior_fixture"}
        raise RuntimeError("no inert response for this call; never access an external provider")

    for tool, contract in sources.reads.items():
        capability = CapabilityContract.from_metadata(tool, dict(capability_id=contract.spec.capability,
            action_type="read_only", domain="behavior-fixture", required_roles=["fixture-reader"],
            scope_fields=[], sensitivity=contract.spec.access.data_classification,
            input_schema_digest=sha256_json(contract.spec.input_schema.model_dump(by_alias=True, mode="json")),
            output_schema_digest=sha256_json(contract.spec.output_schema.model_dump(by_alias=True, mode="json"))),
            source="inert-behavior-fixture")
        bindings[contract.contract_hash] = HostReadBinding(contract.contract_hash, capability,
            frozenset(contract.spec.access.required_scopes), lambda args, name=tool: observe(name, args))
    packet = qualify_flow(flow, reads, sources.effects)
    result = run_read_flow(flow, scenario.arguments, reads=reads, effects=sources.effects,
        bindings=bindings, context=context,
        consent=HostFlowConsent(packet["flowDigest"], sha256_json(scenario.arguments)))
    return dict(calls=calls, outcome=result)


def check_behavior(sources: FlowSources, tree: FlowTree, suite: BehaviorSuite) -> dict:
    """Return counterexamples, never infer semantic acceptance from a safe stop."""
    validate_suite(sources, suite)
    body = dict(protocol=PROTOCOL, sourcesDigest=suite.sources_digest,
        suiteDigest=sha256_json(suite.model_dump(mode="json")), treeDigest=sha256_json(tree.model_dump(mode="json")),
        scope=suite.scope, capabilityGaps=list(suite.capability_gaps), oracleAuthor=suite.oracle_author,
        unverifiedMeanings=list(suite.unverified_meanings), fullSourceReview="required_not_run",
        annotations="not_used_as_behavior_oracle", wholeSkillTranslations=0,
        runtimeAuthorityGranted=False, externalProviderCalls=0, sourceScriptsExecuted=0)
    try:
        compiled = compile_report(sources, tree)
    except (ValueError, KeyError) as error:
        return _seal({**body, "representation": "invalid_candidate", "structuralError": str(error),
            "behavior": "not_run", "scenarios": [], "passed": 0, "total": len(suite.scenarios)})
    flow = FlowProposal.model_validate(compiled["flow"])
    origins = {row["l0Pointer"].split("/")[-1]: row for row in compiled["origins"]}
    node_paths = {node.id: origins[str(i)] for i, node in enumerate(flow.nodes)}
    rows = []
    for scenario in suite.scenarios:
        actual = _run(sources, flow, scenario)
        expected = [c.model_dump(mode="json") for c in scenario.expected_calls]
        differences = []
        if actual["calls"] != expected:
            at = next(i for i in range(max(len(expected), len(actual["calls"])))
                if expected[i:i + 1] != actual["calls"][i:i + 1])
            differences.append(dict(facet="operation_order_arguments_or_count", callIndex=at,
                expected=expected[at:at + 1], actual=actual["calls"][at:at + 1]))
        if actual["outcome"]["status"] != scenario.expected_status:
            differences.append(dict(facet="outcome", expected=scenario.expected_status, actual=actual["outcome"]["status"]))
        rows.append(dict(id=scenario.id, passed=not differences,
            requirements={r: suite.requirements[r] for r in scenario.requirement_ids},
            input=scenario.arguments, observations=[o.model_dump(mode="json") for o in scenario.observations],
            expectedCalls=expected, expectedStatus=scenario.expected_status, actualCalls=actual["calls"],
            actualStatus=actual["outcome"]["status"], differences=differences,
            trace=[{**step, **node_paths[step["node"]]} for step in actual["outcome"]["trace"]]))
    passed = sum(r["passed"] for r in rows)
    return _seal({**body, "representation": "compiled", "flowDigest": compiled["flowDigest"],
        "flow": compiled["flow"], "sourceArchive": sources.source_text,
        "sourceSpans": spans(sources), "sourceReviewInputDigest": compiled["reviewInput"]["inputDigest"],
        "behavior": "matched_finite_oracle" if passed == len(rows) else "counterexample_found",
        "passed": passed, "total": len(rows), "scenarios": rows,
        "inMemoryRuntimeRuns": len(rows), "semanticAccuracy": None})


def _seal(body):
    return {**body, "reportDigest": sha256_json(body)}
