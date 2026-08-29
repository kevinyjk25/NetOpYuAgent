from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from network_runtime import NetworkRuntime, PlanState
from network_runtime.contracts import ApprovalError
from network_runtime.enterprise import (
    ControlPlaneTransportConfig,
    HttpChangeAuthority,
    HttpGatewayAttestationMinter,
    HttpPolicyDecisionPoint,
    JwksJwtDecoder,
    JwtValidationConfig,
    OidcJwksSubjectVerifier,
)
from network_runtime.enterprise_conformance import (
    configuration_report,
    live_contract_report,
)
from network_runtime.identity import ApprovalControlPlane, ENFORCED_MODE
from network_runtime.l0_skills import REGISTRY as L0_SKILLS


def run(coro):
    return asyncio.run(coro)


class EnterpriseControlPlaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.gateway_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.subject_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(
            cls.subject_key.public_key(), as_dict=True,
        )
        cls.subject_jwk.update({"kid": "subject-1", "alg": "RS256", "use": "sig"})
        cls.gateway_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(
            cls.gateway_key.public_key(), as_dict=True,
        )
        cls.gateway_jwk.update({"kid": "gateway-1", "alg": "RS256", "use": "sig"})
        cls.requests: list[dict] = []

        parent = cls

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                del format, args

            def _send(self, value, status=200):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/subject/jwks":
                    self._send({"keys": [parent.subject_jwk]})
                elif self.path == "/gateway/jwks":
                    self._send({"keys": [parent.gateway_jwk]})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                parent.requests.append({"path": self.path, "request": request})
                now = datetime.now(timezone.utc)
                if self.path == "/pdp":
                    subjects = request.get("subjects") or []
                    denied = any(item.get("subject_id") == "blocked" for item in subjects)
                    self._send({
                        "schema": "netopyu.pdp-decision/v1",
                        "allow": not denied,
                        "decision_id": f"decision-{len(parent.requests)}",
                        "policy_id": "enterprise-network-change",
                        "policy_version": "2026-08-28",
                        "evaluated_at": now.isoformat(),
                        "reason": "blocked by test policy" if denied else "allowed",
                        "obligations": (
                            {
                                "require_change_ticket": True,
                                "separation_of_duties": True,
                                "required_approvers": 1,
                            }
                            if request.get("action") == "effect.approve" else {}
                        ),
                    })
                elif self.path == "/gateway/mint":
                    try:
                        subject = jwt.decode(
                            request.get("subject_token", ""),
                            parent.subject_key.public_key(),
                            algorithms=["RS256"],
                            audience="netopyu-runtime",
                            issuer="https://idp.example.test",
                        )
                        gateway_token = parent._gateway_token(
                            str(subject["sub"]),
                            subject_jti=str(subject["jti"]),
                            session_id=str(request.get("session_id") or ""),
                        )
                        self._send({
                            "schema": "netopyu.gateway-mint-response/v1",
                            "gateway_token": gateway_token,
                        })
                    except Exception:
                        self._send({"error": "invalid subject"}, 401)
                elif self.path == "/change":
                    ticket = str(request.get("ticket_id") or "")
                    self._send({
                        "schema": "netopyu.change-record/v1",
                        "ticket_id": ticket,
                        "record_id": f"record-{ticket}",
                        "revision": "7",
                        "status": "denied" if ticket == "CHG-DENIED" else "approved",
                        "approved_by": ["cab@example.test"],
                        "window_start": (now - timedelta(minutes=2)).isoformat(),
                        "window_end": (now + timedelta(minutes=10)).isoformat(),
                        "allowed_profiles": ["dc"] if ticket == "CHG-WRONG-SCOPE" else ["*"],
                        "allowed_capabilities": ["*"],
                        "allowed_targets": ["*"],
                        "risk_ceiling": "critical",
                        "authority": "local-enterprise-qualification-server",
                    })
                else:
                    self._send({"error": "not found"}, 404)

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.control = ApprovalControlPlane(
            key_path=self.root / "approval.key",
            mode=ENFORCED_MODE,
            verifier=OidcJwksSubjectVerifier(
                JwksJwtDecoder(JwtValidationConfig(
                    issuer="https://idp.example.test",
                    audience="netopyu-runtime",
                    jwks_url=f"{self.base_url}/subject/jwks",
                    allow_loopback_http=True,
                )),
                JwksJwtDecoder(JwtValidationConfig(
                    issuer="https://gateway.example.test",
                    audience="netopyu-gateway",
                    jwks_url=f"{self.base_url}/gateway/jwks",
                    allow_loopback_http=True,
                )),
            ),
            pdp=HttpPolicyDecisionPoint(
                f"{self.base_url}/pdp", allow_loopback_http=True,
            ),
            change_authority=HttpChangeAuthority(
                f"{self.base_url}/change", allow_loopback_http=True,
            ),
            signing_key=b"enterprise-approval-signing-key!!",
        )
        self.runtime = NetworkRuntime(
            self.root / "runtime.sqlite", approval_control_plane=self.control,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @classmethod
    def _subject_token(
        cls,
        subject_id: str,
        *,
        session_id: str = "enterprise-session",
        roles: list[str] | None = None,
        scopes: str = "*",
        kid: str = "subject-1",
    ) -> str:
        now = datetime.now(timezone.utc)
        return jwt.encode({
            "iss": "https://idp.example.test",
            "aud": "netopyu-runtime",
            "sub": subject_id,
            "exp": now + timedelta(minutes=5),
            "iat": now,
            "nbf": now - timedelta(seconds=1),
            "auth_time": int(now.timestamp()),
            "jti": f"subject-{subject_id}-{now.timestamp()}",
            "sid": session_id,
            "harness": "dsh",
            "roles": roles or ["network-operator"],
            "scope": scopes,
            "purpose": "qualified-network-change",
            "clearance": "restricted",
            "aal": 2,
            "token_use": "access",
        }, cls.subject_key, algorithm="RS256", headers={"kid": kid})

    @classmethod
    def _gateway_token(
        cls,
        subject_id: str,
        *,
        subject_jti: str,
        session_id: str = "enterprise-session",
    ) -> str:
        now = datetime.now(timezone.utc)
        return jwt.encode({
            "iss": "https://gateway.example.test",
            "aud": "netopyu-gateway",
            "sub": "dsh-gateway",
            "client_id": "dsh-production-adapter",
            "act_sub": subject_id,
            "subject_jti": subject_jti,
            "exp": now + timedelta(minutes=2),
            "iat": now,
            "nbf": now - timedelta(seconds=1),
            "jti": f"gateway-{subject_id}-{now.timestamp()}",
            "sid": session_id,
            "harness": "dsh",
            "token_use": "gateway_attestation",
        }, cls.gateway_key, algorithm="RS256", headers={"kid": "gateway-1"})

    @classmethod
    def _context(
        cls,
        subject_id: str,
        *,
        roles: list[str] | None = None,
        gateway_subject: str | None = None,
    ) -> dict[str, str]:
        subject_token = cls._subject_token(subject_id, roles=roles)
        subject_jti = str(jwt.decode(
            subject_token, options={"verify_signature": False},
        )["jti"])
        return {
            "subject_token": subject_token,
            "gateway_token": cls._gateway_token(
                gateway_subject or subject_id, subject_jti=subject_jti,
            ),
        }

    def _prepare(self, context: dict[str, str]):
        contract = L0_SKILLS.for_tool("lan", "grant_user_access")
        return run(self.runtime.prepare(
            "lan",
            "grant_user_access",
            {"user_id": "erin", "reason": "enterprise qualification"},
            session_id="enterprise-session",
            l0_skill_id=contract.skill_id,
            subject_context=context,
            harness="dsh",
        ))

    def test_http_oidc_gateway_pdp_change_and_signed_proof_end_to_end(self) -> None:
        requester_context = self._context("alice")
        observation = run(self.runtime.invoke_read(
            "lan", "list_devices", {},
            access_context=self._context(
                "alice", roles=["operations-reader", "network-operator"],
            ),
            session_id="enterprise-session",
            harness="dsh",
        ))
        self.assertIn("Device inventory", observation)
        prepared = self._prepare(requester_context)
        self.assertEqual(prepared["status"], "plan_ready")
        plan = prepared["plan"]
        requester = plan["requester_identity"]
        self.assertEqual(requester["subject_id"], "alice")
        self.assertFalse(requester["local_simulation"])
        self.assertEqual(requester["gateway_identity"]["subject_id"], "dsh-gateway")
        self.assertEqual(
            requester["authorization_evidence"]["policy_id"],
            "enterprise-network-change",
        )
        issued = self.runtime.approve(
            plan_id=plan["plan_id"],
            plan_hash=plan["plan_hash"],
            approval_request_id="enterprise-approval-1",
            approver_contexts=[self._context(
                "bob", roles=["network-approver", "change-approver"],
            )],
            change_context={"ticket_id": "CHG-100"},
        )
        decision = issued["evidence"]["policy_decision"]
        self.assertEqual(decision["change_record"]["revision"], "7")
        self.assertEqual(decision["pdp"]["policy_id"], "enterprise-network-change")
        public_record = json.dumps({
            "plan": plan,
            "approval_evidence": issued["evidence"],
        }, sort_keys=True)
        self.assertNotIn(requester_context["subject_token"], public_record)
        self.assertNotIn(requester_context["gateway_token"], public_record)
        outcome = run(self.runtime.execute(
            plan_id=plan["plan_id"],
            plan_hash=plan["plan_hash"],
            execution_nonce=prepared["execution_nonce"],
            approval_proof=issued["approval_proof"],
            allow_destructive=True,
        ))
        self.assertEqual(outcome.state, PlanState.VERIFIED_SUCCESS)
        self.assertTrue(self.runtime.audit(plan["plan_id"])["ok"])

    def test_gateway_subject_substitution_and_unsigned_role_injection_fail(self) -> None:
        with self.assertRaisesRegex(ApprovalError, "different subject"):
            self._prepare(self._context("alice", gateway_subject="mallory"))
        context = self._context("alice")
        context["gateway_token"] = self._gateway_token(
            "alice", subject_jti="different-access-token",
        )
        with self.assertRaisesRegex(ApprovalError, "different access token"):
            self._prepare(context)
        context = self._context("alice", roles=["operations-reader"])
        context["roles"] = ["network-operator"]  # Untrusted raw assertion is ignored.
        with self.assertRaisesRegex(ApprovalError, "effect-operation role"):
            self._prepare(context)

    def test_pdp_denial_and_unknown_jwks_kid_fail_closed(self) -> None:
        with self.assertRaisesRegex(ApprovalError, "blocked by test policy"):
            self._prepare(self._context("blocked"))
        context = self._context("alice")
        context["subject_token"] = self._subject_token("alice", kid="retired-key")
        with self.assertRaisesRegex(ApprovalError, "key id"):
            self._prepare(context)

    def test_change_status_and_scope_are_authoritative(self) -> None:
        prepared = self._prepare(self._context("alice"))
        plan = prepared["plan"]
        approver = self._context(
            "bob", roles=["network-approver", "change-approver"],
        )
        with self.assertRaisesRegex(ApprovalError, "not approved"):
            self.runtime.approve(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                approval_request_id="enterprise-denied-ticket",
                approver_contexts=[approver],
                change_context={"ticket_id": "CHG-DENIED"},
            )
        with self.assertRaisesRegex(ApprovalError, "profile"):
            self.runtime.approve(
                plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                approval_request_id="enterprise-wrong-scope",
                approver_contexts=[approver],
                change_context={"ticket_id": "CHG-WRONG-SCOPE"},
            )

    def test_default_runtime_wires_complete_enterprise_environment(self) -> None:
        environment = {
            "NETOPYU_IDENTITY_MODE": "enforced",
            "NETOPYU_OIDC_ISSUER": "https://idp.example.test",
            "NETOPYU_OIDC_AUDIENCE": "netopyu-runtime",
            "NETOPYU_OIDC_JWKS_URL": f"{self.base_url}/subject/jwks",
            "NETOPYU_GATEWAY_ISSUER": "https://gateway.example.test",
            "NETOPYU_GATEWAY_AUDIENCE": "netopyu-gateway",
            "NETOPYU_GATEWAY_JWKS_URL": f"{self.base_url}/gateway/jwks",
            "NETOPYU_PDP_URL": f"{self.base_url}/pdp",
            "NETOPYU_CHANGE_AUTHORITY_URL": f"{self.base_url}/change",
            "NETOPYU_ENTERPRISE_ALLOW_LOOPBACK_HTTP": "1",
        }
        with patch.dict(os.environ, environment, clear=False):
            runtime = NetworkRuntime(self.root / "environment-runtime.sqlite")
            contract = L0_SKILLS.for_tool("lan", "grant_user_access")
            prepared = run(runtime.prepare(
                "lan", "grant_user_access",
                {"user_id": "erin", "reason": "environment wiring"},
                session_id="enterprise-session",
                l0_skill_id=contract.skill_id,
                subject_context=self._context("alice"),
                harness="dsh",
            ))
            self.assertEqual(prepared["status"], "plan_ready")
            self.assertEqual(
                prepared["plan"]["requester_identity"]["auth_method"],
                "oidc-jwks+gateway-attestation",
            )

    def test_dynamic_gateway_minting_binds_each_runtime_session(self) -> None:
        dynamic = ApprovalControlPlane(
            key_path=self.root / "dynamic.key",
            mode=ENFORCED_MODE,
            verifier=OidcJwksSubjectVerifier(
                JwksJwtDecoder(JwtValidationConfig(
                    issuer="https://idp.example.test",
                    audience="netopyu-runtime",
                    jwks_url=f"{self.base_url}/subject/jwks",
                    allow_loopback_http=True,
                )),
                JwksJwtDecoder(JwtValidationConfig(
                    issuer="https://gateway.example.test",
                    audience="netopyu-gateway",
                    jwks_url=f"{self.base_url}/gateway/jwks",
                    allow_loopback_http=True,
                )),
                gateway_minter=HttpGatewayAttestationMinter(
                    f"{self.base_url}/gateway/mint", allow_loopback_http=True,
                ),
            ),
            pdp=HttpPolicyDecisionPoint(
                f"{self.base_url}/pdp", allow_loopback_http=True,
            ),
            change_authority=HttpChangeAuthority(
                f"{self.base_url}/change", allow_loopback_http=True,
            ),
            signing_key=b"enterprise-approval-signing-key!!",
        )
        runtime = NetworkRuntime(
            self.root / "dynamic-runtime.sqlite", approval_control_plane=dynamic,
        )
        session_id = "dynamic-session-42"
        contract = L0_SKILLS.for_tool("lan", "grant_user_access")
        prepared = run(runtime.prepare(
            "lan",
            "grant_user_access",
            {"user_id": "erin", "reason": "dynamic gateway qualification"},
            session_id=session_id,
            l0_skill_id=contract.skill_id,
            subject_context={
                "subject_token": self._subject_token("alice", session_id=session_id),
            },
            harness="dsh",
        ))
        identity = prepared["plan"]["requester_identity"]
        self.assertEqual(identity["gateway_identity"]["session_id"], session_id)
        self.assertTrue(any(
            item["path"] == "/gateway/mint"
            and item["request"]["session_id"] == session_id
            for item in self.requests
        ))

    def test_mtls_transport_requires_owner_only_key_and_loads_certificate(self) -> None:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "netopyu-client")])
        now = datetime.now(timezone.utc)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .sign(key, hashes.SHA256())
        )
        certificate_path = self.root / "client.pem"
        key_path = self.root / "client.key"
        certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        key_path.chmod(0o600)
        transport = ControlPlaneTransportConfig(
            ca_bundle=str(certificate_path),
            client_certificate=str(certificate_path),
            client_key=str(key_path),
        )
        self.assertIsNotNone(transport.ssl_context())
        self.assertTrue(transport.binding["mutual_tls"])
        self.assertNotIn(str(key_path), json.dumps(transport.binding))
        key_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "owner-only"):
            ControlPlaneTransportConfig(
                client_certificate=str(certificate_path), client_key=str(key_path),
            )

    def test_secret_safe_doctor_and_live_contract_check(self) -> None:
        self.assertFalse(configuration_report({})["ok"])
        environment = {
            "NETOPYU_IDENTITY_MODE": "enforced",
            "NETOPYU_OIDC_ISSUER": "https://idp.example.test",
            "NETOPYU_OIDC_AUDIENCE": "netopyu-runtime",
            "NETOPYU_OIDC_JWKS_URL": f"{self.base_url}/subject/jwks",
            "NETOPYU_GATEWAY_ISSUER": "https://gateway.example.test",
            "NETOPYU_GATEWAY_AUDIENCE": "netopyu-gateway",
            "NETOPYU_GATEWAY_JWKS_URL": f"{self.base_url}/gateway/jwks",
            "NETOPYU_GATEWAY_MINT_URL": f"{self.base_url}/gateway/mint",
            "NETOPYU_PDP_URL": f"{self.base_url}/pdp",
            "NETOPYU_CHANGE_AUTHORITY_URL": f"{self.base_url}/change",
            "NETOPYU_ENTERPRISE_ALLOW_LOOPBACK_HTTP": "1",
            "NETOPYU_OIDC_TOKEN": self._subject_token("alice"),
            "NETOPYU_APPROVER_OIDC_TOKEN": self._subject_token(
                "bob", roles=["network-approver", "change-approver"],
            ),
            "NETOPYU_CONTROL_PLANE_BEARER_TOKEN": "doctor-secret-value",
        }
        report = configuration_report(environment)
        self.assertTrue(report["ok"])
        public_report = json.dumps(report)
        self.assertNotIn(environment["NETOPYU_CONTROL_PLANE_BEARER_TOKEN"], public_report)
        self.assertNotIn(self.base_url, public_report)
        with patch.dict(os.environ, environment, clear=False):
            live = live_contract_report(
                harness="dsh",
                session_id="enterprise-session",
                profile="lan",
                capability_id="grant_user_access",
                target="user:erin",
                risk_level="low",
                ticket_id="CHG-CONFORMANCE",
            )
        self.assertTrue(live["ok"], live)
        self.assertTrue(live["no_network_effects"])
        live_public = json.dumps(live)
        self.assertNotIn(environment["NETOPYU_OIDC_TOKEN"], live_public)
        self.assertNotIn(environment["NETOPYU_APPROVER_OIDC_TOKEN"], live_public)


if __name__ == "__main__":
    unittest.main()
