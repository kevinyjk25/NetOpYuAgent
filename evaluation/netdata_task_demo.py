"""Offline Netdata-shaped host → shared read gateway → bounded data projection.

Developer-wired partial task, not a model-generated Skill or live Netdata test.
Raw receipts stay in memory; only enum counts and non-payload trace are exported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from evaluation.netdata_fixture import FUNCTION, HOST_VERSION, SEVERITIES, IsolatedNetdataHost, array, obj, query_arguments
from evaluation.structured_binding_probe import read_json, write_artifacts
from evaluation.translation_intake import validate_bundle
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import DataSensitivity
from network_runtime.contracts import sha256_json
from network_runtime.l0.read_execution import execute_host_read
from network_runtime.l0.structured_bindings import compile_binding, materialize_binding
from network_runtime.l0.structured_schema import DataBindingError, validate_data

ROOT_SOURCE = "docs/netdata-ai/skills/query-snmp-traps/SKILL.md"
LOG_SOURCE = "docs/netdata-ai/skills/query-netdata-cloud/query-logs.md"
SOURCE_BUNDLE_DIGEST = "sha256:ef838bdff84763a8ea493133d282a2eb1db039f645ae28cfd2624ae6bd5675a6"


def projection(contract):
    fields = {"timestamp": {"type": "integer", "minimum": 1},
              "TRAP_SEVERITY": {"type": "string", "enum": SEVERITIES},
              **{key: {"type": "string", "minLength": 1, "maxLength": 128} for key in
                 ("TRAP_JOB", "TRAP_CATEGORY", "TRAP_REPORT_TYPE", "TRAP_SOURCE_IP")}}
    expression = {"kind": "column_rows", "source": "query", "pointer": "",
                  "fields": list(fields), "max_rows": 200, "max_columns": 64}
    return compile_binding({"query": contract.spec.output_schema}, array(obj(fields), 200), expression)


def summarize_page(contract, response, arguments, *, now_seconds):
    """Check the requested slice; never infer full-window coverage or sanitization."""
    response = validate_data(contract.spec.output_schema, response)
    arguments = validate_data(contract.spec.input_schema, arguments)
    if response["status"] != 200:
        raise DataBindingError("netdata_query_error", "/status", "Function did not return success; error text is not exported")
    if response["v"] != 3 or response["type"] != "logs":
        raise DataBindingError("netdata_query_version", "", "response does not match the reviewed isolated host profile")
    if response.get("partial") is not False:
        raise DataBindingError("netdata_partial_or_unknown", "/partial", "partial or unknown scan must not yield an apparently complete summary")
    body = arguments["body"]
    if set(body) != {"after", "before", "last", "direction", "selections"}:
        raise DataBindingError("netdata_query_shape", "/body", "exact reviewed query form required")
    plan = projection(contract)
    draft = materialize_binding(plan, {"query": response})
    rows = draft["arguments"]
    if len(rows) > body["last"]:
        raise DataBindingError("netdata_page_bound", "/data", "received more rows than the requested page limit")
    if type(now_seconds) is not int or not 1 <= now_seconds <= 4102444800:
        raise DataBindingError("netdata_clock_context", "", "explicit seconds clock required from the isolated host")
    expected = {"TRAP_JOB": body["selections"]["__logs_sources"][0],
                **{k: body["selections"][k][0] for k in ("TRAP_SOURCE_IP", "TRAP_CATEGORY", "TRAP_REPORT_TYPE")}}
    lower, upper = (now_seconds + body["after"]) * 1000000, now_seconds * 1000000
    for i, row in enumerate(rows):
        if any(row[key] != value for key, value in expected.items()):
            raise DataBindingError("netdata_filter_mismatch", f"/data/{i}", "a returned row contradicts the requested selections")
        if not lower <= row["timestamp"] <= upper:
            raise DataBindingError("netdata_time_unit_or_window", f"/data/{i}", "timestamp must be microseconds within the seconds-based query window")
    counts = Counter(row["TRAP_SEVERITY"] for row in rows)
    return {"status": "page_summary_ready_window_coverage_unproven", "returnedPageRows": len(rows),
            "severityCounts": {s: counts[s] for s in SEVERITIES if counts[s]},
            "wholeWindowCount": None, "wholeWindowCoverageProven": False, "emptyPageProvesNoTraps": False,
            "paginationFollowed": False, "rawRowsExported": False, "sensitiveFieldValuesExported": False,
            "publicationPolicy": "local-evaluation-enum-counts-only/v1", "productionPrivacyProven": False,
            "bindingDigest": plan["bindingDigest"], "limitations": [
                "Only this received page is counted; no fleet or complete-window claim.",
                "No hostname, message, source IP, row timestamp, or raw trap value is published.",
                "Counts and host clock are synthetic; no production privacy, identity or timestamp attestation."]}


def context(host):
    return ObservationAccessContext(subject_id="isolated-netdata-reviewer", roles=frozenset({"netops"}),
        scopes=frozenset({"netdata:logs:read", "node_id:" + host.node, "function_id:" + FUNCTION}),
        purpose="developer-wired isolated Netdata page evaluation", clearance=DataSensitivity.CONFIDENTIAL)


def execute_task(host, source_text, *, access_context=None, window_seconds=86400, last=200):
    contract, binding = host.contract_and_binding(source_text)
    access_context = context(host) if access_context is None else access_context
    discovery = execute_host_read(contract, {"node": host.node, "function": FUNCTION, "body": {"info": True}}, access_context, binding)
    args = query_arguments(discovery["payload"], node=host.node, listener=host.listener, device_ip=host.device_ip,
                           window_seconds=window_seconds, last=last)
    receipt = execute_host_read(contract, args, access_context, binding)
    summary = summarize_page(contract, receipt["payload"], args, now_seconds=host.now_seconds)
    # No receipt payloads, provider errors or raw rows escape this entry point.
    return {"summary": summary, "trace": {"hostVersion": HOST_VERSION, "steps": [
        {"operation": "info", "receiptDigest": discovery["receiptDigest"], "accessDecision": discovery["accessDecision"]},
        {"operation": "query", "receiptDigest": receipt["receiptDigest"], "accessDecision": receipt["accessDecision"]},
        {"operation": "column_rows", "bindingDigest": summary["bindingDigest"]},
        {"operation": "local_enum_counts_publication", "rawPayloadExported": False}],
        "queryTimeUnit": "seconds", "rowTimeUnit": "microseconds", "hostCalls": list(host.calls),
        "networkCalls": 0, "effectCalls": 0}, "catalog": host.catalog,
        "projection": projection(contract), "contractHash": contract.contract_hash}


def run_demo(bundle_path, output):
    if Path(output).exists():
        raise FileExistsError("output must not exist; retain previous evidence")
    bundle = read_json(bundle_path)
    validate_bundle(bundle)
    if bundle["bundleDigest"] != SOURCE_BUNDLE_DIGEST:
        raise ValueError("this development demo binds the reviewed C3o source; new sources need explicit alignment")
    docs = {d["path"]: d for d in bundle["documents"]}
    source_map = []
    for path, start_line, end_line, purpose in (
        (ROOT_SOURCE, 44, 59, "structured selections, privacy and one-node scope"),
        (ROOT_SOURCE, 139, 151, "metadata-driven row decoding; do not execute jq"),
        (ROOT_SOURCE, 154, 171, "discover available listener before query"),
        (LOG_SOURCE, 74, 94, "seconds query bounds, microsecond cursor, separate discovery"),
        (LOG_SOURCE, 187, 207, "envelope, row metadata, pagination and partial scans"),
    ):
        doc = docs[path]
        lines = doc["content"].splitlines(keepends=True)
        start, end = len("".join(lines[:start_line-1])), len("".join(lines[:end_line]))
        source_map.append({"path": path, "sourceDigest": doc["sha256"], "start": start, "end": end,
                           "quote": doc["content"][start:end], "purpose": purpose, "semanticEntailmentProven": False})
    host = IsolatedNetdataHost(node="00000000-0000-4000-8000-000000000001", listener="local", device_ip="10.0.0.8")
    result = execute_task(host, docs[ROOT_SOURCE]["content"])
    files = {"summary.json": result["summary"], "trace.json": result["trace"], "host-catalog.json": result["catalog"],
             "projection.json": result["projection"], "source-map.json": {"bundleDigest": bundle["bundleDigest"], "anchors": source_map,
                "decisions": ["Transport is explicitly replaced by a synthetic in-process adapter; no token-safe wrapper is executed.",
                              "Explicit task forbids raw durable output; recipe file writes are not performed.",
                              "Enum counts replace free-form publication; recipe output fidelity is not claimed.",
                              "No pagination/whole-window completion; this is a reviewed partial task, not a compiled whole Skill."]}}
    report = {"apiVersion": "netopyu.io/netdata-isolated-demo/v1", "status": result["summary"]["status"],
              "evidenceRole": "developer_wired_synthetic_host_not_model_translation", "sourceBundleDigest": bundle["bundleDigest"],
              "hostVersion": HOST_VERSION, "readGatewayCalls": len(host.calls), "networkCalls": 0, "modelCalls": 0,
              "sourceScriptCalls": 0, "effectCalls": 0, "contractActivated": False, "wholeSkillTranslationProven": False,
              "taskCompleted": False, "translationMetrics": None, "runtimeLatencyMetrics": None,
              "artifactDigests": {name: sha256_json(data) for name, data in sorted(files.items())}}
    report["reportDigest"] = sha256_json(report)
    write_artifacts(output, {**files, "report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_demo(args.bundle, args.output)
    print(report["status"], "gateway reads:", report["readGatewayCalls"], "whole Skill:", report["wholeSkillTranslationProven"])


if __name__ == "__main__":
    main()
