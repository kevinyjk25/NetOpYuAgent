"""Local reference host, reusing the original Runtime; no semantic reviewer.

Synthetic observations and real localhost model calls are explicitly separate.
No scripts, shell, arbitrary URLs, business writes or model-supplied policies.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import httpx

from . import compiler as author, reasoning_transport, delivery
from .artifacts import write_artifacts
from .contracts import budget, seal
from .model_endpoint import resolve_model_endpoint
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import CapabilityContract
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import GovernedHybridFlow, qualify_hybrid
from network_runtime.l0.hybrid_execution import (
    HostHybridConsent, HostReasoningBinding, ReasoningReply, context_digest, run_hybrid,
)
from network_runtime.l0.read_execution import HostReadBinding
from network_runtime.l0.structured_reads import parse_read_contract
from network_runtime.l0.structured_schema import checked_schema, validate_data

ENDPOINT = "http://127.0.0.1:11434"


def local_context():
    return ObservationAccessContext("operator-local-reference", frozenset({"reviewer"}),
        frozenset({"stage2:read", "reasoning:invoke"}), "Operator-enabled isolated read-only session")


def bindings_for(packet, resources, calls):
    reads, bindings = {}, {}
    for name, raw in packet["reads"].items():
        contract = parse_read_contract(raw)
        capability = CapabilityContract.from_metadata(name, {
            "capability_id": contract.spec.capability, "action_type": "read_only", "domain": "local_reference",
            "input_schema_digest": sha256_json(contract.spec.input_schema),
            "output_schema_digest": sha256_json(contract.spec.output_schema),
            "required_roles": ["reviewer"], "sensitivity": contract.spec.access.data_classification,
            "scope_fields": list(contract.spec.resource_scopes), "freshness_limit_seconds": 10,
        }, source="operator_bound_inert_resources/v1")

        def observe(args, tool=name):
            resource = resources.get(tool)
            entries = resource["resources"] if isinstance(resource, dict) and set(resource) == {"resources"} else [resource]
            matches = [entry[1] for entry in entries if entry is not None and entry[0] == args]
            if len(matches) != 1:
                raise PermissionError("outside operator-declared exact local resources")
            calls.append({"tool": tool, "arguments": copy.deepcopy(args)})
            return copy.deepcopy(matches[0]), {"source": "operator_bound_inert_resources/v1", "synthetic": True,
                                              "realDevice": False, "networkCalls": 0}

        reads[contract.contract_hash] = contract
        bindings[contract.contract_hash] = HostReadBinding(contract.contract_hash, capability,
            frozenset(contract.spec.access.required_scopes), observe)
    return reads, bindings


def make_request(request):
    """Pure wire construction shared by invocation and offline metering."""
    return {"model": author.MODEL, "stream": False, "think": False, "format": checked_schema(request["outputSchema"]),
        "options": {k: v for k, v in author.MODEL_CONFIG.items() if k != "think"},
        "messages": reasoning_transport.messages(request)}


def invoke_local(request, folder, costs):
    """One physical local request; exceptions/unknown usage never auto-retry."""
    wire = make_request(request)
    if not budget(wire)["accepted"]:
        raise ValueError("complete reasoning request exceeds context budget; no truncation")
    endpoint = resolve_model_endpoint("runtime", model=author.MODEL, default_endpoint=ENDPOINT)
    # Only the operator's localhost model endpoint, never a URL from an L1 Skill.
    with httpx.Client(timeout=360, trust_env=False) as client:
        tags = client.get(endpoint.base_url + "/api/tags")
        tags.raise_for_status()
        models = [m for m in tags.json()["models"] if m["name"] == author.MODEL]
        if len(models) != 1 or not models[0].get("digest"):
            raise ValueError("exact local 9B artifact unavailable")
        endpoint.check_model_digest(models[0]["digest"])
        write_artifacts(folder, {"request.json": {"wireRequest": wire, "governedRequest": request,
                                                   "modelArtifact": models[0]["digest"]}})
        began = time.monotonic()
        cost = {"node": request["nodeId"], "inputTokens": None, "outputTokens": None, "status": "unknown",
                "decoderConstraint": "host_output_schema_independent_validation_still_required"}
        try:
            response = client.post(endpoint.base_url + "/api/chat", json=wire)
            response.raise_for_status()
            envelope = response.json()
            write_artifacts(folder / "response", {"envelope.json": envelope})
            cost.update(inputTokens=envelope.get("prompt_eval_count"), outputTokens=envelope.get("eval_count"))
            if envelope.get("model") != author.MODEL or envelope.get("done") is not True or envelope.get("done_reason") != "stop":
                raise ValueError("wrong model identity or incomplete model response")
            candidate = validate_data(request["outputSchema"], json.loads(envelope["message"]["content"]))
            write_artifacts(folder / "candidate", {"candidate.json": candidate})
            cost["status"] = "schema_valid_candidate_not_semantic_proof"
            return ReasoningReply(candidate, author.MODEL, author.CONFIG_DIGEST, cost["inputTokens"], cost["outputTokens"])
        except Exception as error:
            cost["errorType"] = type(error).__name__
            raise
        finally:
            cost["latencyMs"] = (time.monotonic() - began) * 1000
            costs.append(copy.deepcopy(cost))
            write_artifacts(folder / "cost", {"report.json": seal(cost)})


def run_local(packet, compilation, arguments, resources, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("never replay an uncertain local session")
    if compilation != author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"]):
        raise ValueError("source/compiled proposal drift")
    flow = GovernedHybridFlow.model_validate(compilation["flow"])
    if (compilation["plan"].get("mode") != "read_prefix"
            or sum(n.kind == "reason" for n in flow.nodes) != 1):
        raise ValueError("host profile admits only a read prefix and one retained-task reason node")
    calls, costs = [], []
    reads, bindings = bindings_for(packet, resources, calls)
    qualification = qualify_hybrid(flow, reads)
    ctx = local_context()
    write_artifacts(output, {"inputs.json": seal({"packet": packet, "compilation": compilation,
        "arguments": arguments, "resources": resources, "syntheticProvider": True, "writesAuthorized": False})})
    began = time.monotonic()
    outcome = run_hybrid(flow, arguments, reads=reads, read_bindings=bindings,
        reasoners={"local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST,
            lambda req: invoke_local(req, output / "model" / req["nodeId"], costs))}, gates={}, context=ctx,
        consent=HostHybridConsent(qualification["graphDigest"], sha256_json(arguments), context_digest(ctx)))
    report = seal({"execution": outcome, "providerCalls": calls, "modelCalls": costs,
                   "runtimeWallLatencyMs": (time.monotonic() - began) * 1000,
                   "effectCalls": 0, "sourceScriptCalls": 0, "taskSuccess": None})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def run_prefix(packet, compilation, arguments, resources, output):
    """Execute only compiled strict reads. Never call a model to draft from an index."""
    if compilation != author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"]):
        raise ValueError("source/compiled proposal drift")
    if compilation["plan"].get("mode") != "read_prefix":
        raise ValueError("only read_prefix may be split")
    output = Path(output)
    write_artifacts(output, {"inputs.json": seal({"compilationDigest": compilation["reportDigest"],
                                                "arguments": arguments, "readOnly": True})})
    raw = copy.deepcopy(compilation["flow"])
    raw["nodes"] = [n for n in raw["nodes"] if n["kind"] == "strict_region"]
    raw["outputs"] = [n["id"] for n in raw["nodes"]]
    raw["max_model_calls"] = 0
    calls = []
    reads, bindings = bindings_for(packet, resources, calls)
    began = time.monotonic()
    if raw["nodes"]:
        flow = GovernedHybridFlow.model_validate(raw)
        qualification = qualify_hybrid(flow, reads)
        ctx = local_context()
        outcome = run_hybrid(flow, arguments, reads=reads, read_bindings=bindings,
            reasoners={}, gates={}, context=ctx,
            consent=HostHybridConsent(qualification["graphDigest"], sha256_json(arguments), context_digest(ctx)))
    else:
        outcome = {"status": "no_prefix_reads", "outputs": {}, "trace": []}
    report = seal({"execution": outcome, "providerCalls": calls, "modelCalls": [],
        "runtimeWallLatencyMs": (time.monotonic() - began) * 1000,
        "effectCalls": 0, "sourceScriptCalls": 0, "taskSuccess": None,
        "compilationDigest": compilation["reportDigest"], "prefixOnly": True})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def draft_from_evidence(packet, compilation, arguments, prefix, followups, output, *, delivery_contract=None, evidence_state=None):
    """Host-only projection of sealed historical observations into the original L1 node.

    This is a new reason-only graph, not a resumed read grant. No reads are
    replayed and historical receipts do not gain freshness or action authority.
    The session layer freezes/verifies these inputs before this function runs.
    """
    if compilation != author.compile_proposal(packet, compilation["suppliedPages"], compilation["plan"]):
        raise ValueError("source/compiled proposal drift")
    raw = copy.deepcopy(compilation["flow"])
    reasons = [n for n in raw["nodes"] if n["kind"] == "reason"]
    if len(reasons) != 1 or compilation["plan"].get("mode") != "read_prefix":
        raise ValueError("exactly one retained-task reasoning node required")
    node = reasons[0]
    observed = prefix["execution"]["outputs"]
    if set(node["depends_on"]) != set(observed):
        raise ValueError("all declared strict predecessors must have completed")
    for key in node["depends_on"]:
        if observed[key]["role"] != "observed_region":
            raise ValueError("a model candidate is not read evidence")
        node["inputs"]["fields"][key] = author.literal(observed[key]["value"])
    # Exact snapshots are represented by a const schema, not permissive Any.
    node["inputs"]["fields"]["priorReadResults"] = author.literal(followups)
    contracts = {name: parse_read_contract(raw) for name, raw in packet["reads"].items()}
    args_schema = author.obj({name: c.spec.input_schema for name, c in contracts.items()})
    results_schema = author.obj({name: c.spec.output_schema for name, c in contracts.items()})
    args_schema["required"], results_schema["required"] = [], []
    node["input_schema"]["properties"]["priorReadResults"] = {"type": "array", "const": followups,
        "items": author.obj({"tool": {"type": "string"}, "arguments": args_schema,
                             "result": results_schema, "receiptDigest": {"type": "string"}})}
    node["input_schema"]["required"].append("priorReadResults")
    if delivery_contract is not None:
        node["inputs"]["fields"]["delivery_contract"] = author.literal(json.dumps(delivery_contract, ensure_ascii=False))
        node["input_schema"]["properties"]["delivery_contract"] = {"type": "string"}
        node["input_schema"]["required"].append("delivery_contract")
        node["output_schema"] = delivery.response_schema(delivery_contract)
        node["instructions"] = node["instructions"].replace(
            "Return only a candidate draft, uncertainties and remaining_actions. ", "", 1) + "\n" + delivery.generation(delivery_contract)
    if evidence_state is not None:
        state_text = json.dumps(evidence_state, ensure_ascii=False)
        node["inputs"]["fields"]["host_read_state"] = author.literal(state_text)
        node["input_schema"]["properties"]["host_read_state"] = {"type": "string", "const": state_text}
        node["input_schema"]["required"].append("host_read_state")
    node["depends_on"] = []
    raw["nodes"], raw["outputs"], raw["max_model_calls"] = [node], [node["id"]], 1
    flow = GovernedHybridFlow.model_validate(raw)
    qualification = qualify_hybrid(flow, {})
    ctx, costs = local_context(), []
    output = Path(output)
    write_artifacts(output, {"inputs.json": seal({"flow": raw, "arguments": arguments,
        "compilationDigest": compilation["reportDigest"], "prefixDigest": prefix["reportDigest"],
        "followups": followups, "historicalSnapshotOnly": True})})
    began = time.monotonic()
    outcome = run_hybrid(flow, arguments, reads={}, read_bindings={},
        reasoners={"local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST,
            lambda req: invoke_local(req, output / "model" / req["nodeId"], costs))},
        gates={}, context=ctx, consent=HostHybridConsent(qualification["graphDigest"], sha256_json(arguments), context_digest(ctx)))
    report = seal({"execution": outcome, "providerCalls": [], "modelCalls": costs,
        "runtimeWallLatencyMs": (time.monotonic() - began) * 1000,
        "effectCalls": 0, "sourceScriptCalls": 0, "taskSuccess": None})
    write_artifacts(output / "summary", {"report.json": report})
    return report
