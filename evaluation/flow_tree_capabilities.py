"""Host-derived authoring vocabulary, not answer-derived flow repair."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from jsonschema import Draft202012Validator

from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree import FlowTree, compile_report
from evaluation.flow_tree_authoring import tree_request

PROTOCOL = "host-bounded-tree/v1"


def host_schema(sources: FlowSources) -> dict:
    sources = FlowSources.model_validate(sources.model_dump())
    schema = tree_request(sources)["format"]
    removed = set()
    for name, available, field in (("TreeRead", sources.reads, "tool"), ("TreeEffect", sources.effects, "binding_id")):
        if available:
            schema["$defs"][name]["properties"][field]["enum"] = sorted(available)
        else:
            removed.add("#/$defs/" + name)
            del schema["$defs"][name]

    def prune(value):
        if isinstance(value, dict):
            result = {key: prune(item) for key, item in value.items()}
            for key in ("oneOf", "anyOf"):
                if key in result:
                    result[key] = [item for item in result[key] if item.get("$ref") not in removed]
            if "discriminator" in result:
                discriminator = result["discriminator"]
                discriminator["mapping"] = {key: ref for key, ref in discriminator.get("mapping", {}).items() if ref not in removed}
            return result
        return [prune(item) for item in value] if isinstance(value, list) else value

    result = prune(schema)
    Draft202012Validator.check_schema(result)
    return result


def bounded_request(sources: FlowSources) -> dict:
    wire = tree_request(sources)
    schema = host_schema(sources)
    payload = json.loads(wire["messages"][1]["content"])
    payload["outputSchema"] = schema
    payload["constructorBoundary"] = {
        "readTools": sorted(sources.reads), "effectTargetIds": sorted(sources.effects),
        "end": "An outcome, not an Effect. End never uses a binding_id.",
        "missingCapability": "Retain unsupported stop and issue; never invent a tool or replace a missing prerequisite.",
    }
    wire["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    wire["format"] = schema
    return wire


def validate_bounded(sources: FlowSources, raw: dict) -> dict:
    """Decoder support is not trusted; enforce host vocabulary again offline."""
    Draft202012Validator(host_schema(sources)).validate(raw)
    return compile_report(sources, FlowTree.model_validate(raw))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schema", "request", "validate"))
    parser.add_argument("sources", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = FlowSources.model_validate_json(args.sources.read_text())
    if args.command == "schema":
        result = host_schema(source)
    elif args.command == "request":
        result = bounded_request(source)
    elif args.tree:
        result = validate_bounded(source, json.loads(args.tree.read_text()))
    else:
        parser.error("validate requires --tree")
    _write(args.output, result)


if __name__ == "__main__":
    main()
