from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class TestPersistentDshWorker(unittest.TestCase):
    def test_unix_socket_protocol_and_destructive_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            socket_path = root / "bridge.sock"
            environment = {
                **os.environ,
                "NETOPYU_DSH_BACKEND": "mock",
                "NETOPYU_DSH_TOOL_RESULT_STORE": str(root / "results.sqlite"),
                # Exercise the optional DSH Worker OTel bootstrap without
                # requiring a collector or emitting sampled console spans.
                "NETOPYU_DSH_OTEL_ENABLED": "true",
                "OTEL_SAMPLE_RATIO": "0",
            }
            process = subprocess.Popen(
                [sys.executable, "-m", "dsh_adapter.worker", "--socket", str(socket_path)],
                cwd=Path(__file__).parents[1],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for _ in range(100):
                    if socket_path.is_socket():
                        break
                    if process.poll() is not None:
                        self.fail(process.stderr.read())
                    time.sleep(0.02)
                else:
                    self.fail("persistent worker socket did not become ready")

                self.assertEqual(stat.S_IMODE(socket_path.stat().st_mode), 0o600)
                ping = self._request(socket_path, {"id": "ping", "command": "ping"})
                self.assertTrue(ping["ok"])
                self.assertEqual(ping["payload"]["worker_pid"], process.pid)

                manifest = self._request(socket_path, {
                    "id": "manifest", "command": "manifest", "profile": "lan",
                    "include_destructive": False,
                })
                self.assertTrue(manifest["ok"])
                self.assertIn("list_devices", {tool["name"] for tool in manifest["payload"]["tools"]})
                self.assertNotIn("restart_service", {tool["name"] for tool in manifest["payload"]["tools"]})

                denied = self._request(socket_path, {
                    "id": "denied", "command": "invoke", "profile": "lan",
                    "tool": "restart_service",
                    "args": {"service": "crm", "environment": "staging"},
                    "allow_destructive": False,
                })
                self.assertFalse(denied["ok"])
                self.assertIn("PermissionError", denied["error"])

                allowed = self._request(socket_path, {
                    "id": "allowed", "command": "invoke", "profile": "lan",
                    "tool": "restart_service",
                    "args": {"service": "crm", "environment": "staging"},
                    "allow_destructive": True,
                    "correlation_id": "dsh-call-42",
                })
                self.assertTrue(allowed["ok"])
                self.assertIn("restart", allowed["payload"]["result"].lower())
            finally:
                process.terminate()
                process.wait(timeout=5)
            log_rows = [json.loads(line) for line in process.stdout.read().splitlines()]
            correlated = next(row for row in log_rows if row["request_id"] == "allowed")
            self.assertEqual(correlated["correlation_id"], "dsh-call-42")
            self.assertEqual(correlated["tool"], "restart_service")
            self.assertTrue(correlated["ok"])
            self.assertNotIn("args", correlated)
            self.assertIsInstance(correlated["duration_ms"], float)

    @staticmethod
    def _request(path: Path, payload: dict) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(str(path))
            client.sendall((json.dumps(payload) + "\n").encode())
            chunks = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        return json.loads(b"".join(chunks))
