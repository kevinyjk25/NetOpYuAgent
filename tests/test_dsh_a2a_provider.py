from __future__ import annotations

import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dsh_adapter.a2a_provider import delegate_a2a, discover_peers


class _PeerHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path.endswith("agent-card.json"):
            body = json.dumps({
                "agent_id": "dc-agent",
                "name": "DC Agent",
                "url": self.server.base_url,
                "skills": [{"id": "app_access", "name": "Application access", "tags": ["rbac", "dc"]}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        self.server.last_request = json.loads(self.rfile.read(size))
        prompt = self.server.last_request["params"]["message"]["parts"][0]["text"]
        metadata = self.server.last_request["params"]["metadata"]
        if prompt == "require hitl" and metadata.get("operator_decision") != "approve":
            event = {"kind": "taskStatusUpdate", "status": {"state": "input-required", "message": "peer-interrupt-1"}}
        elif metadata.get("resume_interrupt_id") == "peer-interrupt-1":
            event = {"kind": "message", "message": {"parts": [{"kind": "text", "text": "DC approved result"}]}}
        else:
            event = {"kind": "message", "message": {"parts": [{"kind": "text", "text": "DC result"}]}}
        body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestDshA2AProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _PeerHandler)
        cls.server.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_discovery_and_capability_delegation(self):
        peers = asyncio.run(discover_peers(peer_urls=[self.server.base_url]))
        self.assertEqual(peers["peers"][0]["agent_id"], "dc-agent")
        result = asyncio.run(delegate_a2a(
            prompt="check alice", capability="rbac", session_id="dsh-session",
            own_agent_id="lan-agent", peer_urls=[self.server.base_url], timeout_seconds=5,
        ))
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["text"], "DC result")
        metadata = self.server.last_request["params"]["metadata"]
        self.assertEqual(metadata["delegation_chain"], ["lan-agent"])
        self.assertEqual(metadata["dsh_provider"], "netopyu-a2a")

    def test_no_peer_and_loop_fail_closed(self):
        unavailable = asyncio.run(delegate_a2a(
            prompt="x", target="dc-agent", session_id="s", own_agent_id="lan-agent", peer_urls=[],
        ))
        self.assertEqual(unavailable["status"], "unavailable")
        refused = asyncio.run(delegate_a2a(
            prompt="x", target="dc-agent", session_id="s", own_agent_id="lan-agent",
            delegation_chain=["lan-agent"], peer_urls=[self.server.base_url],
        ))
        self.assertEqual(refused["status"], "refused")

    def test_remote_hitl_is_not_reported_as_success(self):
        result = asyncio.run(delegate_a2a(
            prompt="require hitl", target="dc-agent", session_id="s-hitl",
            own_agent_id="lan-agent", peer_urls=[self.server.base_url], timeout_seconds=5,
        ))
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "input-required")
        self.assertEqual(result["interrupt_id"], "peer-interrupt-1")

        resumed = asyncio.run(delegate_a2a(
            prompt="require hitl", target="dc-agent", session_id="s-hitl",
            own_agent_id="lan-agent", peer_urls=[self.server.base_url], timeout_seconds=5,
            resume_interrupt_id=result["interrupt_id"], operator_decision="approve",
        ))
        self.assertTrue(resumed["ok"])
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["text"], "DC approved result")
        metadata = self.server.last_request["params"]["metadata"]
        self.assertEqual(metadata["resume_interrupt_id"], "peer-interrupt-1")
        self.assertEqual(metadata["operator_decision"], "approve")

    def test_remote_hitl_resume_fields_fail_closed(self):
        result = asyncio.run(delegate_a2a(
            prompt="require hitl", target="dc-agent", session_id="s-hitl-bad",
            own_agent_id="lan-agent", peer_urls=[self.server.base_url],
            resume_interrupt_id="peer-interrupt-1",
        ))
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "refused")
