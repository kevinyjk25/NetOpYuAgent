from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from dsh_adapter.local_dc_peer import LocalDcPeer
from network_runtime.engine import NetworkRuntime


def _run(value):
    return asyncio.run(value)


def _message_json(event):
    return json.loads(event["message"]["parts"][0]["text"])


class TestLocalDcPeer(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="netopyu-local-dc-peer-")
        root = Path(self.directory.name)
        self.runtime_path = root / "runtime.sqlite"
        self.state_path = root / "peer.sqlite"
        self.previous_backend = os.environ.get("NETOPYU_DSH_BACKEND")
        os.environ["NETOPYU_DSH_BACKEND"] = "mock"
        self.peer = LocalDcPeer(runtime_path=self.runtime_path, state_path=self.state_path)

    def tearDown(self):
        if self.previous_backend is None:
            os.environ.pop("NETOPYU_DSH_BACKEND", None)
        else:
            os.environ["NETOPYU_DSH_BACKEND"] = self.previous_backend
        self.directory.cleanup()

    def test_app_plan_requires_resume_then_verifies_and_enables_path(self):
        prompt = (
            "Invoke dc-app-access-diagnose for user_id=peeruser, app_id=crm. "
            "Check current application access and ACL; if denied, grant the reviewed base role."
        )
        metadata = {"source_session_id": "ui-session", "session_id": "ui-session"}
        pending = _run(self.peer.handle(prompt, metadata))
        self.assertEqual(pending["status"]["state"], "input-required")
        detail = pending["status"]["message"]
        self.assertEqual(detail["approval"]["l0_skill_id"], "network.dc.app-access.grant")
        self.assertEqual(detail["approval"]["arguments"]["user_id"], "peeruser")

        duplicate = _run(self.peer.handle(prompt, metadata))
        self.assertEqual(duplicate["status"]["message"]["interrupt_id"], detail["interrupt_id"])

        completed = _run(self.peer.handle(prompt, {
            **metadata,
            "resume_interrupt_id": detail["interrupt_id"],
            "operator_decision": "approve",
        }))
        result = _message_json(completed)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["verified"])
        inspected = NetworkRuntime(self.runtime_path).inspect(result["plan_id"])
        self.assertEqual(inspected["record"]["state"], "verified_success")
        self.assertTrue(inspected["audit"]["ok"])

        path = _message_json(_run(self.peer.handle(
            "Invoke dc-path-troubleshoot for user_id=peeruser, app_id=crm; verify end-to-end path.",
            metadata,
        )))
        self.assertTrue(path["application_access_verified"])
        self.assertTrue(path["path_verified"])

    def test_rejection_closes_remote_plan_without_write(self):
        prompt = "Invoke dc-app-access-diagnose for user_id=rejecteduser, app_id=crm."
        metadata = {"source_session_id": "reject-session"}
        pending = _run(self.peer.handle(prompt, metadata))
        detail = pending["status"]["message"]
        rejected = _message_json(_run(self.peer.handle(prompt, {
            **metadata,
            "resume_interrupt_id": detail["interrupt_id"],
            "operator_decision": "reject",
        })))
        self.assertEqual(rejected["status"], "rejected")
        inspected = NetworkRuntime(self.runtime_path).inspect(detail["approval"]["plan_id"])
        self.assertEqual(inspected["record"]["state"], "rejected")

    def test_peer_refuses_pragmatic_mode(self):
        os.environ["NETOPYU_DSH_BACKEND"] = "pragmatic"
        with self.assertRaisesRegex(RuntimeError, "mock-only"):
            LocalDcPeer(runtime_path=self.runtime_path, state_path=self.state_path)


if __name__ == "__main__":
    unittest.main()
