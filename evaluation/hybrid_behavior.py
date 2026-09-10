"""Disclosed in-process hosts for reviewed public mixed candidates, not devices.

Data is synthetic. No filesystem, shell, third-party scripts, business network,
or model-generated query execution. Fixtures never implement semantic selection.
"""
from __future__ import annotations

import copy

from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import CapabilityContract
from network_runtime.contracts import sha256_json
from network_runtime.l0.read_execution import HostReadBinding
from network_runtime.l0.structured_reads import parse_read_contract

FIXTURE_VERSION = "stage2-public-synthetic-hosts/v1"


def scenario(case):
    if case == "handoff":
        return {"notesPath": "/sandbox/shift.md"}, {
            "read_text_file": [{"path": "/sandbox/shift.md"}, {"text": "Outgoing: Alice; incoming: Bob, not yet acknowledged. "
                "Active: INC-42 API p95 900ms since 08:10 UTC; owner Alice; mitigation lowered batch size, not resolved. "
                "Investigation: packet loss on uplink B, owner Chen, next inspect optical counters at 10:30 UTC. "
                "Recent change: release 1.4 at 07:50 UTC. Known issue: dashboard stale; use raw metrics. "
                "Upcoming maintenance 11:00 UTC, owner Dana. Alerts have NOT been verified by Bob. "
                "Escalation: primary oncall, then incident commander after 15 minutes. Dashboard: https://example.invalid/inc42."}]}
    if case == "notion":
        return {"pageId": "page-local-42"}, {"notion_pages_get": [{"page_id": "page-local-42"},
            {"markdown": "---\ntitle: Local review page\nstatus: draft\n---\n# Maintenance\nPlanned at 11:00 UTC; not yet approved.\n"}]}
    if case == "phoenix":
        return {"applicationPath": "/sandbox/app.py"}, {"read_text_file": [{"path": "/sandbox/app.py"},
            {"text": "from openai import OpenAI\nclient = OpenAI()\ndef answer(question: str):\n    response = client.chat.completions.create(model='example-model', messages=[{'role': 'user', 'content': question}])\n    return response.choices[0].message.content\n"}]}
    if case == "documentation":
        return {"projectPath": "/sandbox/project"}, {"list_directory": [{"path": "/sandbox/project"},
            {"entries": [{"name": "pyproject.toml", "kind": "file"}, {"name": "src", "kind": "directory"}]}],
            "read_text_file": [{"path": "/sandbox/project/pyproject.toml"},
                {"text": "[project]\nname='example-project'\nversion='0.1.0'\nrequires-python='>=3.11'\n"}]}
    if case == "warehouse":
        return {"ticketId": "ticket-local-42"}, {"conversations-tickets-retrieve": [{"ticket_id": "ticket-local-42"},
            {"id": "ticket-local-42", "person": {"is_identified": True, "properties": {"email": "alice@example.invalid"}},
             "email_from": "alice@example.invalid", "text": "Claimed EU team 123 orders sync stopped. The ticket does not contain authenticated membership or database evidence."}]}
    if case == "incident":
        return {"monitorId": "monitor-42", "monitorTitle": "API latency", "environment": "prod", "evidence": "p95 latency 900ms; no database lock evidence."}, {
            "linear_list_issues": [{"label": "incident-alert"}, {"issues": [
                {"id": "issue-1", "title": "[staging] API latency", "description": "Monitor ID monitor-99. Recognize it: database locks."},
                {"id": "issue-2", "title": "[prod] API latency", "description": "Monitor ID monitor-42. Recognize it: p95 > 800ms AND database locks. Fix: inspect lock owner, not yet justified by current evidence."}]}],
            "linear_get_issue": [{"id": "issue-2"}, {"id": "issue-2", "title": "[prod] API latency", "description": "Monitor ID monitor-42. Recognize it: p95 > 800ms AND database locks."}],
            "linear_list_comments": [{"issue_id": "issue-2"}, {"comments": [{"body": "Previous locks observed; current incident not confirmed."}]}]}
    if case == "tetragon":
        return {}, {"kubernetes_get_pods": [{"namespace": "kube-system", "label_selector": "app.kubernetes.io/name=tetragon"},
            {"items": [{"metadata": {"name": "tetragon-local"}, "status": {"phase": "Running", "containerStatuses": [{"ready": False}]}}]}],
            "tetragon_status": [{}, {"stdout": "Synthetic status: agent responds; health has not been verified", "exitCode": 0}]}
    if case == "simple-english":
        return {"paragraph": "We shall utilize the diagnostic subsystem in order to ascertain whether latency is elevated. "
            "Keep `timeout=30` unchanged. The exact error is \"Connection refused\". Do not claim the cause is known."}, {}
    raise ValueError("no admitted executable local scenario for this case; never synthesize query results")


def bindings_for(packet, fixtures, calls, *, fault=None):
    result, reads = {}, {name: parse_read_contract(raw) for name, raw in packet["reads"].items()}
    for name, contract in reads.items():
        capability = CapabilityContract.from_metadata(name, {"capability_id": contract.spec.capability,
            "action_type": "read_only", "domain": "evaluation", "input_schema_digest": sha256_json(contract.spec.input_schema),
            "output_schema_digest": sha256_json(contract.spec.output_schema), "required_roles": ["reviewer"],
            "sensitivity": contract.spec.access.data_classification, "scope_fields": list(contract.spec.resource_scopes),
            "freshness_limit_seconds": 10}, source=FIXTURE_VERSION)
        def observe(arguments, name=name):
            if name not in fixtures:
                raise PermissionError("no declared isolated resource")
            expected, payload = fixtures[name]
            # Explicit sandbox resource/data predicate, NOT a hidden business
            # classifier or a task result disguised as a low-level observation.
            if arguments != expected:
                raise PermissionError("outside the predeclared isolated fixture request")
            calls.append({"tool": name, "arguments": copy.deepcopy(arguments)})
            if fault == "provider_error":
                raise RuntimeError("synthetic provider failure")
            return ({} if fault == "invalid_output" else copy.deepcopy(payload)), {
                "source": FIXTURE_VERSION, "synthetic": True, "realDevice": False, "networkCalls": 0}
        result[contract.contract_hash] = HostReadBinding(contract.contract_hash, capability,
            frozenset(contract.spec.access.required_scopes), observe)
    return {c.contract_hash: c for c in reads.values()}, result


def context():
    return ObservationAccessContext("stage2-local-reviewer", frozenset({"reviewer"}),
        frozenset({"stage2:read", "reasoning:invoke", "candidate:admit"}), "Reviewed synthetic public Skill test")
