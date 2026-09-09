"""Visible development oracles and manual feasibility witnesses, not model input.

These six sources are already known. Reference trees test representability;
they are NEVER handed to the translator or counted as generated translations.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from evaluation.flow_behavior import BehaviorSuite
from evaluation.flow_translation import FlowSources
from evaluation.flow_translation_batch import development_sources
from evaluation.flow_tree import FlowTree
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import AtomicReadManifest, ReadObjectSchema
from network_runtime.l0.read_contracts import compile_read

ROOT = Path(__file__).resolve().parents[1]


def ref(source, field):
    return dict(kind="reference", source=source, field=field)


def constant(value):
    return dict(kind="constant", value=value)


def end(outcome="read_path_completed", source_id="s0003"):
    return dict(kind="end", source_id=source_id, outcome=outcome)


def read(tool, bind, arguments, source_id="s0003"):
    return dict(kind="read", source_id=source_id, tool=tool, bind=bind, arguments=arguments)


def branch(left, equals, yes, no, source_id="s0003", no_source_id=None):
    return dict(kind="if_equal", source_id=source_id, left=left, equals=constant(equals),
        true_source_id=source_id, false_source_id=no_source_id or source_id, when_equal=yes, otherwise=no)


def call(tool, **arguments):
    return dict(tool=tool, arguments=arguments)


def observation(tool, arguments, payload, error=False):
    return dict(tool=tool, arguments=arguments, payload=payload, error=error)


def _case(identifier, sources, steps, scenarios, gaps=(), issue_source="s0003"):
    issues = [dict(kind="missing_host_capability", source_id=issue_source, question=g) for g in gaps]
    tree = FlowTree(business_source_ids=[issue_source], steps=steps, issues=issues)
    suite = BehaviorSuite(sources_digest=sha256_json(sources.model_dump(mode="json")),
        requirements={"behavior": sources.source_text.splitlines()[int(issue_source[1:]) - 1]},
        scenarios=[dict(requirement_ids=["behavior"], **s) for s in scenarios],
        scope="safe_partial_stop" if gaps else "executable_fragment", capability_gaps=gaps,
        unverified_meanings=["Full-source interpretation, explanation fidelity and behavior outside the finite scenarios remain unproven."],
        oracle_author="same-development-assistant-manual-reference-not-independent-gold")
    return dict(id=identifier, sources=sources.model_dump(mode="json"), reference=tree.model_dump(mode="json"),
        suite=suite.model_dump(mode="json"))


def _inventory_cases():
    chosen = {"direct-read", "inverted-branch", "missing-approval-write", "unavailable-script-prerequisite"}
    rows = []
    tool = "read_inventory_device"
    for info, sources in development_sources():
        identifier = info["id"]
        if identifier not in chosen:
            continue
        first = read(tool, "snapshot", dict(device_id=ref("input", "device_id")))
        steps, gaps = [first, end()], ()
        if identifier == "inverted-branch":
            steps = [first, branch(ref("snapshot", "site"), "campus", [end("needs_l1")],
                [read(tool, "again", dict(device_id=ref("snapshot", "device_id"))), end()])]
        elif identifier == "missing-approval-write":
            steps = [first, end("unsupported")]
            gaps = ("Approval and VLAN write contracts are absent; the preceding read is not approval or configuration.",)
        elif identifier == "unavailable-script-prerequisite":
            steps = [end("unsupported")]
            gaps = ("The required script and runner contract are absent; do not read past the missing prerequisite.",)
        scenarios = []
        for index, site in enumerate(("campus", "idc", "remote")):
            arguments = dict(device_id=f"requested-{index}")
            # Deliberately distinct returned ID catches input-vs-result binding errors.
            returned = dict(device_id=f"canonical-{index}", site=site, status="planned-lab")
            observations = [observation(tool, arguments, returned), observation(tool, dict(device_id=returned["device_id"]), returned)]
            calls = [call(tool, **arguments)]
            status = "read_path_completed"
            if identifier == "inverted-branch":
                if site == "campus":
                    status = "needs_l1"
                else:
                    calls.append(call(tool, device_id=returned["device_id"]))
            elif gaps:
                status = "unsupported"
                if identifier == "unavailable-script-prerequisite":
                    calls = []
            scenarios.append(dict(id=site, arguments=arguments, observations=observations,
                expected_calls=calls, expected_status=status))
        if identifier != "unavailable-script-prerequisite":
            for error in ("provider-error", "malformed-result", "access-denied"):
                arguments = dict(device_id="error-case")
                scenarios.append(dict(id=error, arguments=arguments,
                    observations=[observation(tool, arguments, {}, error == "provider-error")],
                    expected_calls=[] if error == "access-denied" else [call(tool, **arguments)],
                    expected_status="blocked", authenticated=error != "access-denied"))
        rows.append(_case(identifier, sources, steps, scenarios, gaps))
    return rows


def _schema(properties):
    return dict(type="object", properties={k: dict(type=v) for k, v in properties.items()},
        required=list(properties), additionalProperties=False)


def _contract(name, inputs, outputs, description):
    tool = dict(name=name, description=description, annotations=dict(readOnlyHint=True),
        inputSchema=_schema(inputs), outputSchema=_schema(outputs))
    access = dict(requiredScopes=["fixture:read"], dataClassification="internal")
    adapter = dict(capability="fixture." + name, tool=name, effect="read_only", access=access)
    texts = dict(skill=description, tool=json.dumps(tool), adapter=json.dumps(adapter))
    return compile_read(AtomicReadManifest.model_validate(dict(apiVersion="netopyu.io/l0-effect/v2", kind="AtomicRead",
        metadata=dict(id="fixture." + name, version="1.0.0", owner="local-behavior-fixture", description=description),
        spec=dict(**adapter, inputSchema=tool["inputSchema"], outputSchema=tool["outputSchema"],
            sources=[dict(role=k, origin="inert-fixture://" + name + "/" + k, text=t,
                sha256="sha256:" + hashlib.sha256(t.encode()).hexdigest()) for k, t in texts.items()]))))


def _original_source(filename, identifier):
    rows = json.loads((ROOT / "examples/source-duty-development" / filename).read_text())
    item = next(r for r in rows if r["id"] == identifier)
    return item["bundle"]["documents"][0]


def _source(document, contracts):
    return FlowSources(source_text=document["text"], source_path=document["path"],
        input_schema=ReadObjectSchema.model_validate(_schema({})),
        reads={c.spec.tool: c for c in contracts}, effects={}, max_read_age_seconds=30)


def cases():
    rows = _inventory_cases()
    backup = _original_source("cases.json", "backup-prerequisite")
    sources = _source(backup, [_contract("read_backup_metadata", {}, dict(status="string"),
        "Read backup metadata only. Does not run prerequisites, scripts, or prove verification.")])
    rows.append(_case("backup-prerequisite", sources, [end("unsupported", "s0002")],
        [dict(id="missing-reference-and-runner", arguments={}, observations=[], expected_calls=[], expected_status="unsupported")],
        ("references/precheck.md and the required runner contract are absent; no prerequisite execution is available.",), "s0002"))

    guide = _original_source("guard-extra-cases.json", "access-guide")
    decision = _contract("read_owner_decision", dict(request_id="string"),
        dict(available="boolean", current="boolean", granted="boolean"),
        "Read an existing owner decision for request_id. available states whether it exists; current states whether it is current; "
        "granted states whether it grants protected-attribute reading. Does not create approval or authorize owner changes.")
    protected = _contract("read_protected_attributes", dict(request_id="string"), dict(value="string"),
        "Read protected attributes for request_id. Does not change owners or independently check owner decisions.")
    sources = _source(guide, [decision, protected])
    arguments = dict(request_id=constant("Q6"))
    stop = [end("unsupported", "s0004")]
    continuation = [read(protected.spec.tool, "attributes", arguments), end()]
    for field in ("granted", "current", "available"):
        continuation = [branch(ref("decision", field), True, continuation, stop, no_source_id="s0004")]
    steps = [read(decision.spec.tool, "decision", arguments), *continuation]
    scenarios = []
    for flags in itertools.product((False, True), repeat=3):
        state = dict(zip(("available", "current", "granted"), flags, strict=True))
        calls = [call(decision.spec.tool, request_id="Q6")]
        if all(flags):
            calls.append(call(protected.spec.tool, request_id="Q6"))
        scenarios.append(dict(id="-".join(str(int(v)) for v in flags), arguments={},
            observations=[observation(decision.spec.tool, dict(request_id="Q6"), state),
                observation(protected.spec.tool, dict(request_id="Q6"), dict(value="inert-test-value"))],
            expected_calls=calls, expected_status="read_path_completed" if all(flags) else "unsupported"))
    for name, payload, error in (("missing-result-field", dict(available=True, granted=True), False),
            ("provider-error", {}, True), ("wrong-result-type", dict(available=True, current=True, granted="true"), False)):
        scenarios.append(dict(id=name, arguments={},
            observations=[observation(decision.spec.tool, dict(request_id="Q6"), payload, error)],
            expected_calls=[call(decision.spec.tool, request_id="Q6")], expected_status="blocked"))
    rows.append(_case("access-guide", sources, steps, scenarios))
    return rows
