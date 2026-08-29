from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from network_runtime import NetworkRuntime, PlanState
from network_runtime.contracts import ApprovalError
from network_runtime.identity import (
    ApprovalControlPlane,
    ENFORCED_MODE,
    SubjectIdentity,
    local_subject_context,
)
from network_runtime.journal import NetworkJournal
from network_runtime.l0_skills import REGISTRY as L0_SKILLS


def run(coro):
    return asyncio.run(coro)


class EnterpriseTestVerifier:
    """Test double for the future OIDC/enterprise credential adapter."""

    def verify(self, context, *, expected_harness, expected_session_id):
        if context.get("credential") != "verified-test-credential":
            raise ApprovalError("enterprise credential rejected")
        if context.get("harness") != expected_harness:
            raise ApprovalError("enterprise Harness binding mismatch")
        if context.get("session_id") != expected_session_id:
            raise ApprovalError("enterprise session binding mismatch")
        now = datetime.now(timezone.utc)
        return SubjectIdentity(
            subject_id=str(context["subject_id"]),
            issuer="https://idp.example.test",
            harness=expected_harness,
            session_id=expected_session_id,
            roles=tuple(context["roles"]),
            scopes=tuple(context["scopes"]),
            purpose="qualified-change",
            assurance_level=3,
            auth_method="oidc-mfa-test-double",
            authenticated_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=5)).isoformat(),
            credential_id=str(context["credential_id"]),
            local_simulation=False,
        )


class IdentityControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.journal = self.root / "runtime.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runtime(self, *, control_plane=None) -> NetworkRuntime:
        return NetworkRuntime(self.journal, approval_control_plane=control_plane)

    def prepare(
        self,
        runtime: NetworkRuntime,
        *,
        tool: str = "grant_user_access",
        arguments=None,
        subject_context=None,
        session_id: str = "session-1",
        harness: str = "dsh",
    ):
        contract = L0_SKILLS.for_tool("lan", tool)
        return run(runtime.prepare(
            "lan",
            tool,
            arguments or {"user_id": "erin", "reason": "identity test"},
            session_id=session_id,
            l0_skill_id=contract.skill_id,
            subject_context=subject_context,
            harness=harness,
        ))

    def test_local_signed_proof_binds_requester_approver_policy_and_journal(self) -> None:
        runtime = self.runtime()
        requester = local_subject_context(
            "alice", harness="dsh", session_id="session-1",
            roles=("network-operator",),
        )
        prepared = self.prepare(runtime, subject_context=requester)
        plan = prepared["plan"]
        self.assertEqual(plan["requester_identity"]["subject_id"], "alice")
        self.assertTrue(plan["requester_identity"]["local_simulation"])

        issued = runtime.approve(
            plan_id=plan["plan_id"],
            plan_hash=plan["plan_hash"],
            approval_request_id="dsh-hitl-1",
            approver_contexts=[local_subject_context(
                "bob", harness="dsh", session_id="session-1",
                roles=("network-approver",),
            )],
        )
        self.assertTrue(issued["approval_proof"])
        self.assertEqual(issued["evidence"]["requester_subject_id"], "alice")
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"],
            plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_proof=issued["approval_proof"],
            allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        with NetworkJournal(self.journal) as journal:
            record = journal.record(plan["plan_id"])
        self.assertEqual(record["approval_actor"], "bob")
        self.assertEqual(record["approval_proof_id"], issued["evidence"]["proof_id"])
        self.assertEqual(
            record["approval_evidence"]["requester_digest"], plan["requester_digest"],
        )
        with sqlite3.connect(self.journal) as database:
            database.execute(
                "UPDATE plans SET approval_evidence_json=replace(approval_evidence_json, 'bob', 'mallory') "
                "WHERE plan_id=?",
                (plan["plan_id"],),
            )
        audit = runtime.audit(plan["plan_id"])
        self.assertFalse(audit["ok"])
        self.assertIn(
            "approval_evidence_hash_mismatch",
            {item["error"] for item in audit["errors"]},
        )

    def test_tampered_or_cross_plan_proof_fails_before_nonce_consumption(self) -> None:
        runtime = self.runtime()
        requester = local_subject_context("alice", harness="dsh", session_id="session-1")
        first = self.prepare(runtime, subject_context=requester)
        second = self.prepare(
            runtime,
            tool="restart_service",
            arguments={"service": "crm", "environment": "staging"},
            subject_context=requester,
        )
        first_plan = first["plan"]
        issued = runtime.approve(
            plan_id=first_plan["plan_id"],
            plan_hash=first_plan["plan_hash"],
            approval_request_id="dsh-hitl-2",
            approver_contexts=[local_subject_context(
                "bob", harness="dsh", session_id="session-1",
                roles=("network-approver",),
            )],
        )
        token = issued["approval_proof"]
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
        with self.assertRaisesRegex(ApprovalError, "signature"):
            run(runtime.execute(
                plan_id=first_plan["plan_id"], plan_hash=first_plan["plan_hash"],
                execution_nonce=first["execution_nonce"], approval_proof=tampered,
                allow_destructive=True,
            ))
        with self.assertRaisesRegex(ApprovalError, "different plan"):
            run(runtime.execute(
                plan_id=second["plan"]["plan_id"], plan_hash=second["plan"]["plan_hash"],
                execution_nonce=second["execution_nonce"], approval_proof=token,
                allow_destructive=True,
            ))
        with NetworkJournal(self.journal) as journal:
            self.assertFalse(journal.record(first_plan["plan_id"])["nonce_consumed"])

    def test_enforced_mode_rejects_raw_local_identity_without_enterprise_verifier(self) -> None:
        control = ApprovalControlPlane(
            key_path=self.root / "approval.key",
            mode=ENFORCED_MODE,
            signing_key=b"x" * 32,
        )
        runtime = self.runtime(control_plane=control)
        with self.assertRaisesRegex(ApprovalError, "enterprise credential verifier"):
            self.prepare(
                runtime,
                subject_context=local_subject_context(
                    "alice", harness="dsh", session_id="session-1",
                ),
            )

    def test_enforced_critical_change_requires_ticket_and_separation_of_duties(self) -> None:
        control = ApprovalControlPlane(
            key_path=self.root / "approval.key",
            mode=ENFORCED_MODE,
            verifier=EnterpriseTestVerifier(),
            require_external_controls=False,
            signing_key=b"y" * 32,
        )
        runtime = self.runtime(control_plane=control)
        requester = {
            "credential": "verified-test-credential",
            "credential_id": "oidc-requester-1",
            "subject_id": "alice",
            "harness": "dsh",
            "session_id": "session-1",
            "roles": ["network-operator"],
            "scopes": ["*"],
        }
        prepared = self.prepare(
            runtime,
            tool="delete_resource",
            arguments={"resource_id": "cache-1", "force": True},
            subject_context=requester,
        )
        plan = prepared["plan"]
        self.assertEqual(plan["risk_level"], "critical")
        self_approver = {
            **requester,
            "credential_id": "oidc-approver-self",
            "roles": ["network-approver"],
        }
        with self.assertRaisesRegex(ApprovalError, "separation of duties"):
            runtime.approve(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                approval_request_id="change-1", approver_contexts=[self_approver],
                change_context={"ticket_id": "CHG-100"},
            )
        other_approver = {
            **requester,
            "subject_id": "bob",
            "credential_id": "oidc-approver-2",
            "roles": ["network-approver"],
        }
        with self.assertRaisesRegex(ApprovalError, "change ticket"):
            runtime.approve(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                approval_request_id="change-2", approver_contexts=[other_approver],
            )
        issued = runtime.approve(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            approval_request_id="change-3", approver_contexts=[other_approver],
            change_context={"ticket_id": "CHG-100"},
        )
        outcome = run(runtime.execute(
            plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_proof=issued["approval_proof"], allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)

    def test_enforced_mode_disables_legacy_actor_string(self) -> None:
        control = ApprovalControlPlane(
            key_path=self.root / "approval.key",
            mode=ENFORCED_MODE,
            verifier=EnterpriseTestVerifier(),
            require_external_controls=False,
            signing_key=b"z" * 32,
        )
        runtime = self.runtime(control_plane=control)
        prepared = self.prepare(runtime, subject_context={
            "credential": "verified-test-credential",
            "credential_id": "oidc-requester-2",
            "subject_id": "alice",
            "harness": "dsh",
            "session_id": "session-1",
            "roles": ["network-operator"],
            "scopes": ["*"],
        })
        plan = prepared["plan"]
        with self.assertRaisesRegex(ApprovalError, "legacy approval actor"):
            run(runtime.execute(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                execution_nonce=prepared["execution_nonce"],
                approval_request_id="legacy", approval_actor="alice",
                allow_destructive=True,
            ))


if __name__ == "__main__":
    unittest.main()
