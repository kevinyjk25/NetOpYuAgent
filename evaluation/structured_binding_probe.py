"""Offline structured-binding CLI/demo: synthetic data, no LLM or provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from network_runtime.contracts import sha256_json
from network_runtime.l0.models import ReadObjectSchema
from network_runtime.l0.read_contracts import _source_object
from network_runtime.l0.structured_bindings import (
    compile_binding, compile_tool_binding, materialize_binding, materialize_tool_binding,
)
from network_runtime.l0.structured_schema import MAX_BYTES, DataBindingError, checked_schema


def read_json(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("input JSON exceeds byte budget")
    return _source_object(path.read_text(encoding="utf-8"))


def write_artifacts(output: str | Path, files: dict) -> None:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        (root / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def demo(catalog: dict, output: str | Path) -> dict:
    """A deterministic wiring exercise, explicitly not a public-Skill benchmark."""
    tool = next(t for t in catalog["tools"] if t["name"] == "get_device_status")
    old_count, new_count = 0, 0
    for role in ("inputSchema", "outputSchema"):
        try:
            ReadObjectSchema.model_validate(tool[role])
            old_count += 1
        except ValidationError:
            pass
        checked_schema(tool[role])
        new_count += 1
    source = {"type": "object", "properties": {
        "device": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False},
        "states": {"type": "array", "items": {"type": "string"}},
    }, "required": ["device", "states"], "additionalProperties": False}
    expression = {"kind": "object", "fields": {
        "deviceId": {"kind": "reference", "source": "input", "pointer": "/device/id"},
        "selections": {"kind": "object", "fields": {
            "states": {"kind": "reference", "source": "input", "pointer": "/states"},
        }},
    }}
    host = compile_tool_binding(catalog, tool["name"], {"input": source}, expression)
    good = {"input": {"device": {"id": "lab-router-1"}, "states": ["down"]}}
    bad = {"input": {"device": {"id": "lab-router-1"}, "states": ["BROKEN"]}}
    materialized = materialize_tool_binding(host, catalog, good)
    checks = []
    try:
        materialize_tool_binding(host, catalog, bad)
    except DataBindingError as error:
        checks.append({"case": "out_of_catalog_enum", "blocked": True, "diagnostic": error.as_dict()})
    port = tool["outputSchema"]["properties"]["interfaces"]["items"]
    projection = compile_binding({"status": tool["outputSchema"]}, port,
                                 {"kind": "reference", "source": "status", "pointer": "/interfaces/0"})
    fixture = {"status": {"interfaces": [{"ifName": "eth0", "state": "down"}]}}
    projected = materialize_binding(projection, fixture)
    try:
        materialize_binding(projection, {"status": {"interfaces": []}})
    except DataBindingError as error:
        checks.append({"case": "empty_array_no_implicit_selection", "blocked": True, "diagnostic": error.as_dict()})
    changed = json.loads(json.dumps(catalog))
    changed["tools"][0]["description"] += " Changed declaration."
    try:
        materialize_tool_binding(host, changed, good)
    except DataBindingError as error:
        checks.append({"case": "catalog_drift", "blocked": True, "diagnostic": error.as_dict()})
    if len(checks) != 3:
        raise AssertionError("a negative demo path was not blocked")
    files = {"catalog.json": catalog, "host-binding.json": host, "request-source.json": good,
             "invalid-request-source.json": bad, "arguments.json": materialized,
             "output-projection.json": projection, "output-fixture.json": fixture, "projected-output.json": projected}
    report = {
        "apiVersion": "netopyu.io/structured-binding-demo/v1", "evidenceKind": "synthetic_mechanical_binding_demo",
        "sourceSchemaCount": 2, "legacyFlatSchemasAccepted": old_count, "newStructuredSchemasAccepted": new_count,
        "checks": checks, "artifactDigests": {name: sha256_json(value) for name, value in sorted(files.items())},
        "newModelCalls": 0, "providerCalls": 0, "thirdPartyExecutionAttempted": False,
        "runtimeAuthorityGranted": False, "wholeSkillTranslationProven": False,
        "translationMetrics": None, "runtimeLatencyMetrics": None,
        "claimBoundary": "Declared synthetic schemas and fixture values only, not public-Skill semantic generalization or authenticated host execution.",
    }
    report["reportDigest"] = sha256_json(report)
    write_artifacts(output, {**files, "report.json": report})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    compile_command = subs.add_parser("compile", help="compile a JSON packet with catalog, tool, sourceSchemas, expression")
    compile_command.add_argument("packet")
    materialize = subs.add_parser("materialize", help="validate exact catalog/source values into an arguments draft")
    materialize.add_argument("host_binding")
    materialize.add_argument("catalog")
    materialize.add_argument("values")
    demo_command = subs.add_parser("demo", help="run the supplied synthetic structured interface example offline")
    demo_command.add_argument("--catalog", default=str(Path(__file__).resolve().parents[1] / "examples/translation-intake/mcp-catalog.json"))
    for command in (compile_command, materialize, demo_command):
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("output must not exist; preserve previous evidence")
    if args.command == "compile":
        packet = read_json(args.packet)
        if set(packet) != {"catalog", "tool", "sourceSchemas", "expression"}:
            parser.error("compile packet requires exactly catalog, tool, sourceSchemas and expression")
        result = compile_tool_binding(packet["catalog"], packet["tool"], packet["sourceSchemas"], packet["expression"])
        write_artifacts(args.output, {"host-binding.json": result})
    elif args.command == "materialize":
        result = materialize_tool_binding(read_json(args.host_binding), read_json(args.catalog), read_json(args.values))
        write_artifacts(args.output, {"arguments.json": result})
    else:
        result = demo(read_json(args.catalog), args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
