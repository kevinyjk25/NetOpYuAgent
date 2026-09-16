"""Real installed DSH + Python host, SCRIPTED model transport, no semantic claim.

Checks normal and rejected native delivery terminate without a further request.
Never calls Ollama or an external model. Preserves every failed probe directory.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading

from dsh_adapter.hybrid_session import CHOICE_HOST_PROFILE, TASK_HOST_PROFILE, fingerprint
from dsh_adapter.settings import sync_settings
from evaluation.dsh_shadow import _default_dsh_binary, _node_path, parse_dumped_config, REQUIRED_DISABLED_IDS
from evaluation.dsh_shadow_tool import _read_transcript
from evaluation.ollama_no_think_proxy import _openai_response
from evaluation.semantic_closure_transfer import packet_for
from evaluation.stage2_batch import digest
from evaluation.structured_flow_demo import fixture, object_schema
from skill_authoring.artifacts import read_json, write_artifacts
from skill_authoring.contracts import seal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "scripted-terminal-probe-not-an-llm"
TASK = "Read the supplied snapshot and explain its current limitations, without changing anything."


class ProbeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, rejected, task_bound=False, required_evidence=False):
        super().__init__(("127.0.0.1", 0), ProbeHandler)
        self.rejected = rejected
        self.task_bound = task_bound
        self.required_evidence = required_evidence
        self.requests = []


class ProbeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        if not self.path.endswith("/chat/completions") or not 0 < size <= 4 * 1024 * 1024:
            self.send_error(400)
            return
        request = json.loads(self.rfile.read(size))
        index = len(self.server.requests)
        self.server.requests.append(request)
        limit = (7 if self.server.rejected else 8) if self.server.required_evidence else 6
        if index >= limit:
            self.send_error(409, "Unexpected model request after terminal delivery")
            return
        sid = None
        for item in request["messages"]:
            if item.get("role") == "tool":
                try:
                    sid = json.loads(item["content"]).get("session_id", sid)
                except (ValueError, TypeError):
                    pass
        selected = {"requirements": [{"kind": "analysis", "language": "", "source_ref": "task:0"}], "unrepresented": []}
        if self.server.task_bound:
            selected = None
        if index == 0:
            action, args = "prepare", {"task": TASK, "arguments": {"path": "/sandbox/snapshot"}}
        elif index in {1, 2, 3}:
            # Two deterministic unsupported-plan rejections, then normal native
            # fallback binding. No handcrafted executable L0 is used or scored.
            action, args = "submit", {"session_id": sid, "plan": None, "delivery": selected}
        elif index == 4:
            action, args = "read", {"session_id": sid, "tool": "read_export", "arguments": {"path": "/sandbox/snapshot"}}
        elif self.server.required_evidence and index == 6:
            if self.server.rejected:
                action, args = "draft", {"session_id": sid, "close_incomplete": True}
            else:
                action, args = "read", {"session_id": sid, "tool": "read_export", "arguments": {"path": "/sandbox/required"}}
        else:
            answer = "Scripted probe candidate; not a real LLM assessment."
            candidate = {"answer": answer} if self.server.task_bound else {
                "delivery": {"d0": {"text": answer}}, "uncertainties": []}
            response = "invalid JSON" if self.server.rejected and not self.server.required_evidence else json.dumps(candidate)
            action, args = "deliver", {"session_id": sid, "response_json": response}
        native = {"model": MODEL, "message": {"content": "", "tool_calls": [{"id": f"probe-{index}",
            "function": {"name": "netopyu_hybrid_" + action, "arguments": args}}]},
            "prompt_eval_count": 0, "eval_count": 0}
        stream = request.get("stream", False)
        body = json.dumps(_openai_response(native, stream=stream))
        data = ("data: " + body + "\n\ndata: [DONE]\n\n" if stream else body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run(output, *, task_bound=False, required_evidence=False):
    if required_evidence and not task_bound:
        raise ValueError("required evidence probe needs the explicit task-bound host")
    output = Path(output).resolve()
    binary = _default_dsh_binary()
    modules = binary.parent.parent
    headless = list((modules / ".pnpm").glob("@deepseek-ai+dsh-headless@*/node_modules/@deepseek-ai/dsh-headless/lib/index.js"))
    if len(headless) != 1:
        raise ValueError("one installed DSH headless package is required; no guessed package selection")
    bundle, *_ = fixture()
    schema = object_schema({"path": {"type": "string"}})
    tool = {"name": "read_export", "inputSchema": schema,
            "outputSchema": object_schema({"text": {"type": "string"}}), "annotations": {"readOnlyHint": True}}
    packet = packet_for(bundle, {"task": TASK, "inputSchema": schema, "tools": [tool]})
    impl = fingerprint()
    for path in ["evaluation/hybrid_terminal_probe.py", "dsh-plugin-netopyu/src/index.js",
                 "dsh-plugin-netopyu/src/bridge.js", "dsh-plugin-netopyu/src/hybrid-local.js",
                 "dsh-plugin-netopyu/src/hybrid-terminal.js", "dsh-plugin-netopyu/src/hybrid-headless.js"]:
        impl[path] = digest(ROOT / path)
    write_artifacts(output, {"freeze.json": seal({"implementation": impl, "fixturePacket": packet,
        "model": MODEL, "realModelCalls": 0, "realDSH": True, "maxSecondsPerCase": 90,
        "expectedRequests": {"candidate": 8, "rejected": 7} if required_evidence else 6,
        "expectedReads": {"candidate": 2, "rejected": 1} if required_evidence else 1,
        "semanticSuccessClaimed": False, "taskBound": task_bound, "requiredEvidence": required_evidence,
        "installedHeadless": {"path": str(headless[0]), "digest": digest(headless[0])}})})
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for path in impl:
            archive.add(ROOT / path, arcname=path, recursive=False)
    rows = []
    for rejected in [False, True]:
        case = "rejected" if rejected else "candidate"
        folder = output / case
        profile = {"apiVersion": TASK_HOST_PROFILE if task_bound else CHOICE_HOST_PROFILE, "enabled": True,
            "packet": packet, "resources": {"read_export": [{"path": "/sandbox/snapshot"},
                {"text": "Probe snapshot; no operation is approved."}]}}
        if required_evidence:
            profile["resources"]["read_export"] = {"resources": [profile["resources"]["read_export"],
                [{"path": "/sandbox/required"}, {"text": "Operator-declared prerequisite, no expected answer."}]]}
            profile["requiredReads"] = [{"tool": "read_export", "arguments": {"path": "/sandbox/required"}}]
        write_artifacts(folder, {"host.json": profile})
        patch = (ROOT / "evaluation/dsh_shadow.patch.yml").read_text()
        patch += "\n- id: headless-runner\n  disabled: true\n"
        patch += "\n- insert:\n    - id: netopyu-hybrid-headless\n      name: " + json.dumps(str(ROOT / "dsh-plugin-netopyu/src/hybrid-headless.js"))
        patch += "\n      inject:\n        - headlessStartup\n      config:\n        task: !!js ctx.headlessStartup.task\n"
        patch += "\n- insert:\n    - id: netopyu-hybrid-local\n      name: " + json.dumps(str(ROOT / "dsh-plugin-netopyu/src/hybrid-local.js")) + "\n"
        (folder / "hybrid.patch.yml").write_text(patch)
        server = ProbeServer(rejected, task_bound, required_evidence)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            home = folder / "dsh-home"
            sync_settings(home / "settings.yaml", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                          primary_model=MODEL, fast_model=MODEL, default_model=MODEL)
            env = {"PATH": _node_path(), "HOME": str(Path.home()), "DSH_HOME": str(home),
                "LANG": os.environ.get("LANG", "C.UTF-8"), "DSH_PERMISSION_MODE": "read-only",
                "DSH_TELEMETRY_MODE": "DISABLED", "DSH_TOOLS_MODE": "native", "NETOPYU_OLLAMA_API_KEY": "local-fixture",
                "NETOPYU_L1_SHADOW_SYSTEM_PROMPT": "Scripted transport probe; no semantic evaluation.",
                "NETOPYU_ROOT": str(ROOT), "NETOPYU_PYTHON": sys.executable,
                "NETOPYU_HYBRID_HOST_PROFILE": str(folder / "host.json"), "NETOPYU_HYBRID_SESSIONS_DIR": str(folder / "sessions"),
                "NETOPYU_HYBRID_TERMINAL_DELIVERY": "1", "NETOPYU_DSH_HEADLESS_ENTRY": str(headless[0])}
            argv = [str(binary), "--profile", "headless", "--patch", str(folder / "hybrid.patch.yml")]
            config = subprocess.run([*argv, "--dump-config"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=45)
            (folder / "composed-config.txt").write_text(config.stdout)
            entries = {e.entry_id: e for e in parse_dumped_config(config.stdout)}
            if (config.returncode or any(key in entries and not entries[key].disabled for key in REQUIRED_DISABLED_IDS)
                    or "netopyu-hybrid-headless" not in entries or entries["netopyu-hybrid-headless"].disabled
                    or "headless-runner" not in entries or not entries["headless-runner"].disabled):
                raise ValueError("DSH configuration/isolation failed")
            result = subprocess.run([*argv, TASK], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            (folder / "dsh-stdout.txt").write_text(result.stdout)
            (folder / "dsh-stderr.txt").write_text(result.stderr)
        finally:
            server.shutdown()
            worker.join(timeout=5)
            server.server_close()
            write_artifacts(folder / "transport", {"requests.json": {"scriptedNotLLM": True, "requests": server.requests}})
        events = []
        for path in (home).rglob("session.jsonl.zstd"):
            transcript, _ = _read_transcript(path)
            events.extend(transcript)
        finals = list((folder / "sessions").glob("*/draft/result/report.json"))
        final = read_json(finals[0]) if len(finals) == 1 else None
        reads = [read_json(path) for path in (folder / "sessions").glob("*/followup/*/result/report.json")]
        expected_requests = (7 if rejected else 8) if required_evidence else 6
        expected_reads = (1 if rejected else 2) if required_evidence else 1
        passed = bool(result.returncode == 0 and len(server.requests) == expected_requests and len(reads) == expected_reads
            and all(len(r["providerCalls"]) == 1 for r in reads) and final and final["task"]["success"] is None
            and final["hostResult"]["state"] == ("rejected" if rejected else "candidate_unverified")
            and "Host delivery" in result.stdout
            and ("No admitted delivery text" if rejected else "Scripted probe candidate") in result.stdout)
        gate_seen = any('"route": "needs_required_evidence"' in str(msg.get("content", ""))
            for req in server.requests for msg in req["messages"] if msg.get("role") == "tool")
        if required_evidence:
            passed = passed and gate_seen and final["modelCalls"] == []
        rows.append({"case": case, "passed": passed, "processExit": result.returncode, "missingEvidenceGateSeen": gate_seen,
            "scriptedRequests": len(server.requests), "realModelCalls": 0,
            "reads": sum(len(r["providerCalls"]) for r in reads), "stdoutDigest": digest(folder / "dsh-stdout.txt"),
            "terminalHostReportDigest": final["reportDigest"] if final else None})
        write_artifacts(folder / "observation", {"report.json": seal(rows[-1]), "events.json": {"events": events}})
        print(json.dumps(rows[-1]), flush=True)
    report = seal({"cases": rows, "allPassed": all(r["passed"] for r in rows), "realModelCalls": 0,
        "semanticSuccessClaimed": False, "formalStageExit": False,
        "artifactDigests": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}})
    write_artifacts(output / "summary", {"report.json": report})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--task-bound", action="store_true")
    parser.add_argument("--required-evidence", action="store_true")
    args = parser.parse_args()
    run(args.output, task_bound=args.task_bound, required_evidence=args.required_evidence)
