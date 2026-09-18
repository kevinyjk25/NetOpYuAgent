"""Finite six-request offline preflight check, never a six-Skill evaluation.

Requires host-built and content-pinned renderer/tokenizer assets. Twelve
preparations use the actual local vocabulary but no model forward/decode.
Generation, installed-backend parity and pilot qualification remain disabled.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from evaluation.bounded_dsh_probe import packet
from evaluation.bounded_pilot import seal, write_new
from evaluation.bounded_preflight import Assets, prepare, file_digest
from evaluation.bounded_transport import MODEL
from network_runtime.contracts import sha256_json
from skill_authoring import compiler, isolated_compiler, local_execution


def cases():
    base = {"model": MODEL, "stream": False, "think": False,
            "messages": [{"role": "user", "content": "Describe this local observation without taking action."}],
            "options": {"num_predict": 128, "temperature": 0}}
    unicode = copy.deepcopy(base)
    unicode["messages"][0]["content"] = "设备状态：未知。🌐\nLiteral <|im_start|> and embedded \u0000 remain input data."
    tool = {"type": "function", "function": {"name": "read_export", "description": "Read scoped local data.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}}}
    tools = copy.deepcopy(base)
    tools["tools"] = [tool]
    multi = copy.deepcopy(tools)
    multi["messages"] += [{"role": "assistant", "content": "", "tool_calls": [{"id": "read-1",
        "function": {"name": "read_export", "arguments": {"path": "/sandbox/snapshot"}}}]},
        {"role": "tool", "tool_name": "read_export", "content": '{"state":"unknown","approved":false}'},
        {"role": "user", "content": "Explain only what this observation supports."}]
    p = packet()
    author = isolated_compiler.make_request(p, list(compiler.pages_for(p)))
    governed = {"nodeId": "fixture", "instructions": "Report uncertainty; never grant approval.",
        "inputs": {"task": "Explain the supplied state.", "state": "unknown"},
        "outputSchema": compiler.obj({"text": {"type": "string"}}),
        "evidencePolicy": {}, "observationAgesAtStartMs": {}, "maxOutputTokens": 2048}
    reason = local_execution.make_request(governed)
    return [("plain", base), ("unicode_special_literal", unicode), ("tools", tools),
            ("tool_history", multi), ("actual_compiler_wire", author), ("actual_runtime_wire", reason)]


def _sources():
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in (
        "evaluation/bounded_preflight.py", "evaluation/bounded_preflight_probe.py",
        "evaluation/bounded_transport.py", "evaluation/chat_codec.py", "evaluation/bounded_dsh_probe.py",
        "evaluation/local_read_fixture.py", "skill_authoring/source.py", "skill_authoring/compiler.py",
        "skill_authoring/reasoning_transport.py", "skill_authoring/local_execution.py",
        "skill_authoring/isolated_compiler.py")]
    paths.extend(path for path in (root / "evaluation/preflight").rglob("*") if path.is_file())
    return {str(path.relative_to(root)): file_digest(path) for path in sorted(paths)}


def run(assets_file, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = Assets.load(assets_file)
    frozen = _sources()
    write_new(output / "assets.json", assets.inspect())
    write_new(output / "source-freeze.json", seal({"files": frozen, "completeDependencySnapshot": False}))
    rows = []
    for name, request in cases():
        directory = output / name
        directory.mkdir()
        write_new(directory / "source-request.json", request)
        row = {"case": name, "passed": False, "preparations": [], "actualModelCalls": 0}
        try:
            for repeat in range(2):
                began = time.monotonic()
                prepared = prepare(assets, "native", request, timeout=120)
                value = prepared.inspect()
                write_new(directory / f"prepared-{repeat + 1}.json", value)
                binding = prepared.check_binding(assets, "native", request)
                row["preparations"].append({"digest": value["digest"], "inputTokens": value["input_tokens"],
                    "promptDigest": value["rendered_prompt_digest"], "tokensDigest": value["token_ids_digest"],
                    "milliseconds": (time.monotonic() - began) * 1000, "binding": binding})
            first, second = row["preparations"]
            row["passed"] = all(first[k] == second[k] for k in ("digest", "inputTokens", "promptDigest", "tokensDigest"))
        except (ValueError, OSError, TimeoutError) as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        write_new(directory / "observation.json", seal(row))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not row["passed"]:
            break
    unchanged = frozen == _sources()
    report = seal({"schema": "ensuredskill.io/offline-preflight-probe/v1", "cases": rows,
        "allPassed": len(rows) == 6 and all(row["passed"] for row in rows) and unchanged,
        "implementationUnchanged": unchanged, "sourceDigest": sha256_json(frozen),
        "assetsDigest": assets.inspect()["digest"], "actualModelCalls": 0,
        "evidenceRole": "six_request_shapes_not_six_skills_or_semantic_accuracy",
        "generationEnabled": False, "liveAdapterReady": False, "pilotQualified": False,
        "installedBackendParity": "not_established", "generationTokenParity": "not_tested",
        "completeDependencySnapshot": False})
    write_new(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.assets, args.output)["allPassed"] else 1)
