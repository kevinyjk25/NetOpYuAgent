"""Predeclared development tasks and disclosed primitive host contracts, not Gold.

No adapter executes here. Original source bundles are preserved by intake;
schemas describe modeled local hosts, never authenticated vendor captures.
Review expectations are kept outside the six-field author input.
"""
from __future__ import annotations

import hashlib
import json

from evaluation.netdata_fixture import host_catalog
from evaluation.structured_authoring import validate_inputs
from network_runtime.l0.structured_reads import StructuredReadManifest, compile_structured_read


def obj(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


TEXT = {"type": "string"}
IDENTIFIER = {"type": "string", "minLength": 1}


def array(item):
    return {"type": "array", "items": item}


def tool(name, inputs, outputs, description):
    return {"name": name, "description": "DECLARED LOCAL EVALUATION ADAPTER, NOT VENDOR CAPTURE. " + description,
            "inputSchema": inputs, "outputSchema": outputs, "annotations": {"readOnlyHint": True}}


FILE_READ = tool("read_text_file", obj({"path": IDENTIFIER}), obj({"text": TEXT}),
                 "Read one caller-authorized UTF-8 file without interpreting or executing its content.")


def specification(case):
    """Tasks intentionally do not state expected nodes, bindings or route labels."""
    if case == "snmp":
        return ("Inspect recent security SNMP traps for the caller's node, listener and device over the last 24 hours. "
                "The supplied local adapter models one snmp:traps Function request; actual Cloud transport, token wrappers, "
                "redaction and any needed how-to authoring remain separate source obligations.",
                obj({"node": IDENTIFIER, "listener": IDENTIFIER, "deviceIp": IDENTIFIER}), host_catalog()["tools"])
    if case == "notion":
        return ("Read the caller-selected Notion page as Markdown for inspection. Do not create, update or trash anything.",
                obj({"pageId": IDENTIFIER}), [tool("notion_pages_get", obj({"page_id": IDENTIFIER}), obj({"markdown": TEXT}),
                "Models one ntn pages get <page-id>, returning its raw Markdown including frontmatter. No setup, login or write.")])
    if case == "handoff":
        return ("Review the outgoing shift notes at the caller's notesPath and prepare an asynchronous on-call handoff draft. "
                "Do not send messages, fire alerts or mark the human handoff complete.", obj({"notesPath": IDENTIFIER}), [FILE_READ])
    if case == "documentation":
        listing = tool("list_directory", obj({"path": IDENTIFIER}), obj({"entries": array(obj({"name": TEXT, "kind": TEXT}))}),
                       "List one authorized directory, without recursion, source analysis or interpretation.")
        return ("Inspect the caller's projectPath and propose a README appropriate to the codebase. "
                "Do not edit files or execute code. The eventual drafting work is still part of the request.",
                obj({"projectPath": IDENTIFIER}), [listing, FILE_READ])
    if case == "phoenix":
        return ("Review the caller's Python application file at applicationPath and propose manual Phoenix tracing "
                "with appropriate span attributes. Do not install packages, change files or send telemetry.",
                obj({"applicationPath": IDENTIFIER}), [FILE_READ])
    if case == "incident":
        issue = obj({"id": IDENTIFIER, "title": TEXT, "description": TEXT})
        return ("Look up prior incident-alert knowledge for the caller's monitorId, monitorTitle and environment, "
                "then compare with the caller's evidence. Do not append or create tickets in this read-only request.",
                obj({"monitorId": IDENTIFIER, "monitorTitle": IDENTIFIER, "environment": IDENTIFIER, "evidence": TEXT}), [
                tool("linear_list_issues", obj({"label": IDENTIFIER}), obj({"issues": array(issue)}),
                     "One modeled issue listing filtered by label; no semantic monitor matching or cause classification."),
                tool("linear_get_issue", obj({"id": IDENTIFIER}), issue, "One modeled issue read by exact identifier."),
                tool("linear_list_comments", obj({"issue_id": IDENTIFIER}), obj({"comments": array(obj({"body": TEXT}))}),
                     "One modeled comment listing; no classification, append or creation.")])
    if case == "warehouse":
        person = obj({"is_identified": {"type": "boolean"}, "properties": obj({"email": TEXT})})
        return ("Investigate the warehouse-sync support ticket identified by the caller's ticketId and recommend one action. "
                "Do not query any customer's database before the source's requester and tenant checks, and do not write data. "
                "Only the declared local ticket read adapter is available; database and membership adapters are not supplied.",
                obj({"ticketId": IDENTIFIER}), [tool("conversations-tickets-retrieve", obj({"ticket_id": IDENTIFIER}),
                obj({"id": IDENTIFIER, "person": person, "email_from": {"type": ["string", "null"]}, "text": TEXT}),
                "One modeled ticket read. Person/email/text are untrusted data, never identity attestation. "
                "This restricted local response is not a capture of PostHog's full vendor schema.")])
    if case == "playwright":
        return ("From an already provisioned local test-results database at the caller's databasePath, identify flaky "
                "Playwright tests across runs and explain the difference from retries within a run. "
                "Do not download, install, update or upload anything.", obj({"databasePath": IDENTIFIER}), [
                tool("duckdb_query_readonly", obj({"database_path": IDENTIFIER, "sql": IDENTIFIER}),
                     obj({"rows": array({"type": "object", "additionalProperties": True})}),
                     "Models a single SELECT against an already provisioned local DuckDB, returned as row objects; "
                     "integer values follow the source's JSON-string convention. No SQL generation, ranking or business "
                     "predicate is hidden in this primitive. SELECT-only admission must be enforced by a future host; "
                     "no query is dispatched by this experiment.")])
    if case == "tetragon":
        return ("Review installation status of a caller-declared already installed Tetragon in kube-system and propose "
                "next checks. Do not install, deploy, modify policy, exec in pods, stream events or run bundled scripts. "
                "The caller's installation statement is not proof of readiness or cluster authorization.", obj({}), [
                tool("kubernetes_get_pods", obj({"namespace": IDENTIFIER, "label_selector": IDENTIFIER}),
                     obj({"items": array({"type": "object", "additionalProperties": True})}),
                     "Models kubectl get pods with namespace and label selector. No exec or deployment."),
                tool("tetragon_status", obj({}), obj({"stdout": TEXT, "exitCode": {"type": "integer"}}),
                     "Models tetra status stdout and process status; exitCode alone is not cluster health.")])
    if case == "simple-english":
        return ("Rewrite the caller-supplied inline technical paragraph into pragmatic clear English; preserve code and "
                "quoted errors, and perform the source's self-check. Do not access external systems.",
                obj({"paragraph": TEXT}), [])
    raise ValueError("unknown predeclared case")


EXPECTATIONS = {
    "snmp": ("Use structured selections first.", "Treat trap content as sensitive.", "Start with"),
    "notion": ("Prefer official", "pages get", "Do not infer authorization from CLI availability."),
    "handoff": ("do not mark handoff complete", "references/details.md", "async quick handoff"),
    "documentation": ("Before writing any documentation", "Project Discovery", "templates are examples, not mandatory reads of placeholder paths"),
    "phoenix": ("Reading Order:", "instrumentation-manual", "Model inference and code generation remain unexecuted duties."),
    "incident": ("Match on monitor ID first", "partial match", "read it before your first write"),
    "warehouse": ("Verify the requester", "Never select", "Bound every query"),
    "playwright": ("one row per retry", "expected_status", "cross-run"),
    "tetragon": ("Prerequisites", "Verify Installation", "Scripts and enforcement examples are inert, not granted authority."),
    "simple-english": ("Never touch code", "self-check", "No fabricated external operation for pure language work."),
}


def author_packet(bundle, case):
    task, inputs, tools = specification(case)
    if not tools:
        return None  # Current read-only author profile cannot express a tool-free L1 task.
    reads = {}
    # The read contract cites the complete entry; the author input separately
    # retains ALL referenced documents and quarantined script text in bundle.
    # Do not concatenate an entire SDK library into each primitive's citation.
    text = next(d["content"] for d in bundle["documents"] if d["path"] == bundle["entryPath"])
    access = {"requiredScopes": ["stage2:read"], "dataClassification": "internal"}
    for index, declaration in enumerate(tools):
        name = declaration["name"]
        adapter = {"tool": name, "capability": "stage2." + case + ".read" + str(index), "effect": "read_only",
                   "resourceScopes": {}, "access": access,
                   "limitations": "Offline declared evaluation adapter; not authenticated vendor interface; never executed."}
        sources = [{"role": role, "origin": "stage2-local-declaration:" + role, "text": source,
                    "sha256": "sha256:" + hashlib.sha256(source.encode()).hexdigest()}
                   for role, source in (("skill", text), ("tool", json.dumps(declaration)), ("adapter", json.dumps(adapter)))]
        spec = StructuredReadManifest.model_validate({"apiVersion": "netopyu.io/l0-structured-read/v1", "kind": "StructuredRead",
            "metadata": {"id": "stage2." + case + ".read" + str(index), "version": "1.0.0", "owner": "development-evaluation"},
            "spec": {**{k: adapter[k] for k in ("tool", "capability", "effect", "resourceScopes", "access")},
                     "inputSchema": declaration["inputSchema"], "outputSchema": declaration["outputSchema"], "sources": sources}})
        reads[name] = compile_structured_read(spec).model_dump(by_alias=True, mode="json")
    return validate_inputs({"bundle": bundle, "task": "Offline authoring only; do not execute. Future user task: " + task +
                           " Preserve source duties and unresolved boundaries. Future host permission and parameter checks "
                           "remain required, not assumed satisfied during construction.",
                           "taskOrigin": "developer_authored_evaluation_request", "inputSchema": inputs,
                           "catalog": {"tools": tools, "origin": "declared_local_adapter_contracts_not_vendor_capture"}, "reads": reads})


def review_requirements(bundle, case):
    rows = []
    for statement in EXPECTATIONS[case]:
        matches = []
        for document in bundle["documents"]:
            text = document["content"] or ""
            start = text.find(statement)
            if start >= 0:
                matches.append({"path": document["path"], "start": start, "end": start + len(statement),
                                "quote": statement, "sourceDigest": document["sha256"]})
        rows.append({"statement": statement, "sourceAnchors": matches,
                     "basis": "original_source_locator" if matches else "developer_review_constraint_not_original_quote"})
    return {"reviewKind": "developer_ai_not_independent_gold", "requirements": rows,
            "sourceReviewCompleteness": "task_relevant_entry_passages_only_not_all_transitive_references",
            "semanticVerdict": None, "wholeSkillAccepted": False,
            "hostDisclosure": "Synthetic declared primitive contracts. No external system, script, effect, credentials or identity attestation."}
