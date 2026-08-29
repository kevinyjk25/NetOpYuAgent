from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from dsh_adapter.backend import BackendSession
from network_runtime.capabilities import (
    CapabilityContract,
    CapabilityKind,
    DataSensitivity,
    EffectSemantics,
)
from network_runtime.contracts import sha256_json
from network_runtime.provider_qualification import run_provider_qualification
from network_runtime.provider_release import (
    BUNDLE_SCHEMA,
    DEPLOYMENT_SCHEMA,
    MANIFEST_SCHEMA,
    TRUST_SCHEMA,
    ProviderAdmissionGate,
    ProviderDeploymentAttestation,
    ProviderManifest,
    ProviderReleaseBundle,
    ProviderReleaseError,
    ProviderReleaseRegistry,
    ProviderTrustStore,
    QualificationReport,
    ReleasedCapability,
    SIGNED_DEPLOYMENT_SCHEMA,
    SignedProviderDeployment,
    TrustedKey,
    compatibility_report,
    provider_admission_from_environment,
    sign_digest,
)
from network_runtime.provider_release_cli import main as provider_cli


def run(awaitable):
    return asyncio.run(awaitable)


class FakeQualificationTarget:
    def __init__(self, contract: CapabilityContract) -> None:
        self.contract = contract
        self.current = "baseline"
        self.operations: dict[str, dict] = {}
        self.escalations: dict[str, str] = {}

    def describe_capability(self, tool_name: str) -> CapabilityContract:
        if tool_name != self.contract.tool_name:
            raise KeyError(tool_name)
        return self.contract

    async def reset(self) -> str:
        self.current = "baseline"
        self.operations = {}
        self.escalations = {}
        return self.current

    async def snapshot_digest(self) -> str:
        return self.current

    async def apply(
        self, tool_name, arguments, *, operation_id, sequence, fault=None,
    ):
        if tool_name != self.contract.tool_name or sequence != 1:
            raise RuntimeError("out of order")
        if fault == "timeout_before_send":
            raise TimeoutError("before send")
        existing = self.operations.get(operation_id)
        if existing is not None:
            return dict(existing)
        if fault == "unknown_terminal":
            value = {"operation_id": operation_id, "state": "unknown", "apply_attempts": 1}
            self.operations[operation_id] = value
            raise TimeoutError("terminal unknown")
        self.current = sha256_json(arguments)
        value = {"operation_id": operation_id, "state": "applied", "apply_attempts": 1}
        self.operations[operation_id] = value
        if fault == "after_commit_before_response":
            raise TimeoutError("response lost")
        return dict(value)

    async def reconcile(self, operation_id):
        return dict(self.operations.get(operation_id) or {
            "operation_id": operation_id, "state": "unknown", "apply_attempts": 0,
        })

    async def compensate(self, operation_id, *, fault=None):
        if operation_id not in self.operations:
            raise RuntimeError("unknown operation")
        if fault == "compensation_failure":
            self.escalations[operation_id] = "manual_intervention_required"
            raise RuntimeError("compensation failed")
        self.current = "baseline"
        self.operations[operation_id] = {
            **self.operations[operation_id], "state": "compensated",
        }
        return dict(self.operations[operation_id])

    async def restart(self):
        return None

    async def escalation_state(self, operation_id):
        return self.escalations.get(operation_id, "none")


class ProviderReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime.now(timezone.utc)
        self.publisher = Ed25519PrivateKey.generate()
        self.qualifier = Ed25519PrivateKey.generate()
        self.deployer = Ed25519PrivateKey.generate()
        self.contract = CapabilityContract(
            tool_name="vendor_set_access_vlan",
            capability_id="vendor.fabric.access-vlan.set",
            capability_version="1.0.0",
            domain="network",
            kind=CapabilityKind.EFFECT,
            action_type="reversible",
            effect_semantics=EffectSemantics.REVERSIBLE,
            provider_role="actor",
            provider_identity="mcp:vendor:vendor-network-actor@1.0.0",
            provider_kind="vendor-network-actor-mcp",
            input_schema_digest=sha256_json({"port": "string", "vlan": "integer"}),
            output_schema_digest=sha256_json({"state": "string"}),
            sensitivity=DataSensitivity.RESTRICTED,
            required_roles=("network-operator",),
            scope_fields=("port",),
            freshness_limit_seconds=60,
        )
        self.trust = ProviderTrustStore(
            apiVersion=TRUST_SCHEMA,
            keys=(
                self._trusted("publisher-1", "publisher", self.publisher),
                self._trusted("qualifier-1", "qualifier", self.qualifier),
            ),
        )
        self.deployment_trust = ProviderTrustStore(
            apiVersion=TRUST_SCHEMA,
            required_artifacts=("oci-image", "sbom", "provenance"),
            require_deployment_attestation=True,
            keys=(
                self._trusted("publisher-1", "publisher", self.publisher),
                self._trusted("qualifier-1", "qualifier", self.qualifier),
                self._trusted("deployer-1", "deployer", self.deployer),
            ),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _trusted(self, key_id, role, private_key):
        public = private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        return TrustedKey(
            key_id=key_id,
            role=role,
            public_key_pem=public,
            providers=("vendor.fabric",),
            not_before=(self.now - timedelta(minutes=1)).isoformat(),
            not_after=(self.now + timedelta(days=60)).isoformat(),
        )

    def _manifest(
        self,
        version="1.0.0",
        *,
        contract=None,
        compatibility="compatible",
        supersedes=None,
    ):
        capability = ReleasedCapability.from_runtime(
            contract or self.contract,
            result_contract="network-evidence-envelope-v1",
            l0_contract_hashes=(sha256_json({"l0": "vendor-access-vlan"}),),
        )
        return ProviderManifest(
            apiVersion=MANIFEST_SCHEMA,
            provider_id="vendor.fabric",
            provider_version=version,
            provider_identity=(contract or self.contract).provider_identity,
            released_at=self.now.isoformat(),
            compatibility=compatibility,
            supersedes=supersedes,
            artifacts={
                "oci-image": sha256_json({"image": version}),
                "sbom": sha256_json({"spdx": version}),
                "provenance": sha256_json({"slsa": version}),
            },
            capabilities=(capability,),
        )

    def _bundle(self, manifest):
        report = run(run_provider_qualification(
            FakeQualificationTarget(self.contract),
            manifest,
            tool_name=self.contract.tool_name,
            arguments={"port": "Ethernet1", "vlan": 20},
            environment="isolated-local-qualification",
            now=self.now,
        ))
        return ProviderReleaseBundle(
            apiVersion=BUNDLE_SCHEMA,
            manifest=manifest,
            manifest_signature=sign_digest(
                manifest.digest,
                private_key=self.publisher,
                key_id="publisher-1",
                role="publisher",
                now=self.now,
                ttl_seconds=2_592_000,
            ),
            qualification=report,
            qualification_signature=sign_digest(
                report.digest,
                private_key=self.qualifier,
                key_id="qualifier-1",
                role="qualifier",
                now=self.now,
                ttl_seconds=2_592_000,
            ),
        )

    def _deployment(
        self,
        bundle,
        *,
        environment="production",
        deployment_id="deploy-1",
        now=None,
        artifacts=None,
    ):
        observed = now or self.now
        attestation = ProviderDeploymentAttestation(
            apiVersion=DEPLOYMENT_SCHEMA,
            provider_id=bundle.manifest.provider_id,
            provider_version=bundle.manifest.provider_version,
            provider_identity=bundle.manifest.provider_identity,
            release_digest=bundle.digest,
            manifest_digest=bundle.manifest.digest,
            environment=environment,
            deployment_id=deployment_id,
            controller_identity="spiffe://deployment-controller/test",
            artifact_digests=artifacts or bundle.manifest.artifacts,
            deployed_at=observed.isoformat(),
            expires_at=(observed + timedelta(days=7)).isoformat(),
        )
        return SignedProviderDeployment(
            apiVersion=SIGNED_DEPLOYMENT_SCHEMA,
            attestation=attestation,
            signature=sign_digest(
                attestation.digest,
                private_key=self.deployer,
                key_id="deployer-1",
                role="deployer",
                now=observed,
                ttl_seconds=7 * 24 * 60 * 60,
            ),
        )

    def test_fixed_failure_suite_and_independent_dual_signature(self) -> None:
        bundle = self._bundle(self._manifest())
        evidence = self.trust.verify_bundle(bundle, now=self.now)
        self.assertTrue(evidence["ok"])
        self.assertNotEqual(
            bundle.manifest_signature.key_id,
            bundle.qualification_signature.key_id,
        )
        self.assertEqual(len(bundle.qualification.checks), 9)
        self.assertTrue(all(bundle.qualification.checks.values()))

        changed = bundle.manifest_signature.model_copy(update={
            "expires_at": (self.now + timedelta(days=29)).isoformat(),
        })
        tampered = bundle.model_copy(update={"manifest_signature": changed})
        with self.assertRaisesRegex(ProviderReleaseError, "verification failed"):
            self.trust.verify_bundle(tampered, now=self.now)

        with self.assertRaisesRegex(ValidationError, "unsupported checks"):
            QualificationReport.model_validate({
                **bundle.qualification.model_dump(by_alias=True, mode="json"),
                "checks": {**bundle.qualification.checks, "unreviewed_check": True},
                "evidence_digests": {
                    **bundle.qualification.evidence_digests,
                    "unreviewed_check": sha256_json({"unreviewed": True}),
                },
            })

    def test_same_key_cannot_publish_and_qualify(self) -> None:
        with self.assertRaisesRegex(ValidationError, "independent key material"):
            ProviderTrustStore(
                apiVersion=TRUST_SCHEMA,
                keys=(
                    self._trusted("publisher-same", "publisher", self.publisher),
                    self._trusted("qualifier-same", "qualifier", self.publisher),
                ),
            )

    def test_required_artifacts_and_deployment_attestation_fail_closed(self) -> None:
        bundle = self._bundle(self._manifest())
        registry = ProviderReleaseRegistry(
            self.root / "deployment-releases.sqlite", self.deployment_trust,
        )
        registry.stage(bundle)
        registry.publish(bundle.digest)

        with self.assertRaisesRegex(ProviderReleaseError, "deployment attestation"):
            registry.promote(bundle.digest, environment="production")

        incomplete = bundle.model_copy(update={
            "manifest": bundle.manifest.model_copy(update={
                "artifacts": {"oci-image": bundle.manifest.artifacts["oci-image"]},
            }),
        })
        with self.assertRaisesRegex(ProviderReleaseError, "required artifacts"):
            self.deployment_trust.verify_bundle(incomplete, now=self.now)

        wrong_artifacts = dict(bundle.manifest.artifacts)
        wrong_artifacts["sbom"] = sha256_json({"tampered": True})
        with self.assertRaisesRegex(ProviderReleaseError, "differs from the signed release"):
            self.deployment_trust.verify_deployment(
                bundle,
                self._deployment(bundle, artifacts=wrong_artifacts),
                environment="production",
                now=self.now,
            )

        deployment = self._deployment(bundle)
        promoted = registry.promote(
            bundle.digest, environment="production", deployment=deployment,
        )
        self.assertEqual(
            promoted["deployment"]["deployment_digest"], deployment.digest,
        )
        admitted = ProviderAdmissionGate(
            registry, environment="production",
        ).admit(
            self.contract,
            provider_id="vendor.fabric",
            result_contract="network-evidence-envelope-v1",
        )
        self.assertEqual(admitted.deployment_digest, deployment.digest)
        self.assertEqual(
            registry.status()["activations"][0]["deployment_digest"], deployment.digest,
        )

        renewed = self._deployment(
            bundle,
            deployment_id="deploy-1-renewed",
        )
        renewal = registry.promote(
            bundle.digest, environment="production", deployment=renewed,
        )
        self.assertFalse(renewal["idempotent"])
        self.assertEqual(
            registry.status()["activations"][0]["deployment_digest"], renewed.digest,
        )

        second = self._bundle(self._manifest("1.1.0"))
        registry.stage(second)
        registry.publish(second.digest)
        second_deployment = self._deployment(
            second, deployment_id="deploy-2",
        )
        registry.promote(
            second.digest,
            environment="production",
            deployment=second_deployment,
        )
        with self.assertRaisesRegex(ProviderReleaseError, "deployment attestation"):
            registry.rollback(
                provider_id="vendor.fabric",
                environment="production",
                approval_reference="CHG-ROLLBACK-MISSING-DEPLOYMENT",
            )
        rollback_deployment = self._deployment(
            bundle, deployment_id="deploy-1-rollback",
        )
        rolled_back = registry.rollback(
            provider_id="vendor.fabric",
            environment="production",
            approval_reference="CHG-ROLLBACK-DEPLOYMENT",
            deployment=rollback_deployment,
        )
        self.assertEqual(rolled_back["release_digest"], bundle.digest)
        self.assertEqual(
            rolled_back["deployment"]["deployment_digest"], rollback_deployment.digest,
        )

        with self.assertRaisesRegex(ProviderReleaseError, "expired"):
            self.deployment_trust.verify_deployment(
                bundle,
                renewed,
                environment="production",
                now=self.now + timedelta(days=8),
            )

    def test_deployment_cli_sign_bundle_and_verify(self) -> None:
        bundle = self._bundle(self._manifest())
        deployment = self._deployment(bundle)
        attestation_path = self.root / "deployment-attestation.json"
        signature_path = self.root / "deployment-signature.json"
        deployment_path = self.root / "signed-deployment.json"
        bundle_path = self.root / "release-bundle.json"
        trust_path = self.root / "provider-trust.json"
        key_path = self.root / "deployer.pem"
        attestation_path.write_text(
            deployment.attestation.model_dump_json(by_alias=True), encoding="utf-8",
        )
        bundle_path.write_text(bundle.model_dump_json(by_alias=True), encoding="utf-8")
        trust_path.write_text(
            self.deployment_trust.model_dump_json(by_alias=True), encoding="utf-8",
        )
        key_path.write_bytes(self.deployer.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        key_path.chmod(0o600)

        def call(arguments):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = provider_cli(arguments)
            result = json.loads(output.getvalue())
            self.assertEqual(code, 0, result)
            return result

        signed = call([
            "sign-deployment",
            "--deployment-attestation", str(attestation_path),
            "--private-key", str(key_path),
            "--key-id", "deployer-1",
            "--ttl-seconds", str(6 * 24 * 60 * 60),
            "--output", str(signature_path),
        ])
        self.assertEqual(signed["role"], "deployer")
        combined = call([
            "deployment-bundle",
            "--deployment-attestation", str(attestation_path),
            "--deployment-signature", str(signature_path),
            "--output", str(deployment_path),
        ])
        verified = call([
            "verify-deployment",
            "--bundle", str(bundle_path),
            "--deployment", str(deployment_path),
            "--trust-store", str(trust_path),
            "--environment", "production",
        ])
        self.assertEqual(
            verified["deployment"]["deployment_digest"],
            combined["deployment_digest"],
        )

    def test_publish_promote_admit_rollback_deprecate_and_audit(self) -> None:
        registry = ProviderReleaseRegistry(self.root / "releases.sqlite", self.trust)
        first = self._bundle(self._manifest())
        staged = registry.stage(first)
        self.assertEqual(staged["state"], "staged")
        registry.publish(first.digest)
        registry.promote(first.digest, environment="test")
        admission = ProviderAdmissionGate(registry, environment="test").admit(
            self.contract,
            provider_id="vendor.fabric",
            result_contract="network-evidence-envelope-v1",
        )
        self.assertEqual(admission.release_digest, first.digest)

        trust_path = self.root / "provider-trust.json"
        trust_path.write_text(
            self.trust.model_dump_json(by_alias=True), encoding="utf-8",
        )
        with patch.dict(os.environ, {
            "NETOPYU_PROVIDER_ADMISSION": "enforced",
            "NETOPYU_PROVIDER_TRUST_STORE": str(trust_path),
            "NETOPYU_PROVIDER_RELEASE_DB": str(registry.path),
            "NETOPYU_PROVIDER_ENVIRONMENT": "test",
        }):
            configured = provider_admission_from_environment()
            self.assertIsNotNone(configured)
            self.assertEqual(configured.admit(
                self.contract,
                provider_id="vendor.fabric",
                result_contract="network-evidence-envelope-v1",
            ).release_digest, first.digest)
        with patch.dict(os.environ, {
            "NETOPYU_PROVIDER_ADMISSION": "enforced",
            "NETOPYU_PROVIDER_TRUST_STORE": "",
            "NETOPYU_PROVIDER_RELEASE_DB": "",
            "NETOPYU_PROVIDER_ENVIRONMENT": "",
        }):
            with self.assertRaisesRegex(ProviderReleaseError, "requires trust store"):
                provider_admission_from_environment()

        second = self._bundle(self._manifest("1.1.0"))
        self.assertTrue(compatibility_report(first.manifest, second.manifest)["compatible"])
        regressed_capability = second.manifest.capabilities[0].model_copy(update={
            "capability_version": "0.9.0",
        })
        regressed = second.manifest.model_copy(update={
            "capabilities": (regressed_capability,),
        })
        self.assertIn(
            "capability version decreased",
            " ".join(compatibility_report(first.manifest, regressed)["breaking_reasons"]),
        )
        registry.stage(second)
        registry.publish(second.digest)
        promoted = registry.promote(second.digest, environment="test")
        self.assertEqual(promoted["previous_release_digest"], first.digest)
        with self.assertRaisesRegex(ProviderReleaseError, "active release"):
            registry.deprecate(second.digest, reason="still active")
        rolled_back = registry.rollback(
            provider_id="vendor.fabric",
            environment="test",
            approval_reference="CHG-ROLLBACK-1",
        )
        self.assertEqual(rolled_back["release_digest"], first.digest)
        self.assertEqual(registry.deprecate(
            second.digest, reason="superseded after rollback",
        )["state"], "deprecated")
        self.assertTrue(registry.audit()["ok"])

        metadata = {
            "action_type": self.contract.action_type,
            "hitl": True,
            "capability_id": self.contract.capability_id,
            "capability_version": self.contract.capability_version,
            "domain": self.contract.domain,
            "provider_role": self.contract.provider_role,
            "provider_identity": self.contract.provider_identity,
            "provider_kind": self.contract.provider_kind,
            "input_schema_digest": self.contract.input_schema_digest,
            "output_schema_digest": self.contract.output_schema_digest,
            "sensitivity": self.contract.sensitivity.value,
            "required_roles": list(self.contract.required_roles),
            "scope_fields": list(self.contract.scope_fields),
            "freshness_limit_seconds": self.contract.freshness_limit_seconds,
            "result_contract": "network-evidence-envelope-v1",
            "release_provider_id": "vendor.fabric",
        }
        session = BackendSession(
            mode="pragmatic",
            profile_id="lan",
            callables={self.contract.tool_name: lambda _: None},
            metadata={self.contract.tool_name: metadata},
            sources={self.contract.tool_name: "mcp:vendor"},
            report={"ready": True},
            _provider_admission_gate=ProviderAdmissionGate(registry, environment="test"),
        )
        self.assertEqual(session.describe_capability(
            self.contract.tool_name,
        ), self.contract)
        self.assertEqual(metadata["provider_release_digest"], first.digest)
        self.assertEqual(
            metadata["provider_qualification_digest"], first.qualification.digest,
        )
        self.assertEqual(metadata["provider_deployment_digest"], "not-required")
        self.assertEqual(
            metadata["provider_l0_contract_hashes"],
            list(first.manifest.capabilities[0].l0_contract_hashes),
        )

    def test_breaking_release_requires_supersedes_and_approval(self) -> None:
        registry = ProviderReleaseRegistry(self.root / "breaking.sqlite", self.trust)
        first = self._bundle(self._manifest())
        registry.stage(first)
        registry.publish(first.digest)
        registry.promote(first.digest, environment="production")
        changed_contract = CapabilityContract(
            **{
                **self.contract.__dict__,
                "input_schema_digest": sha256_json({"port": "string", "vlan": "string"}),
            }
        )
        manifest = self._manifest(
            "2.0.0",
            contract=changed_contract,
            compatibility="breaking",
            supersedes=first.manifest.digest,
        )
        # The qualification adapter must describe exactly the candidate contract.
        report = run(run_provider_qualification(
            FakeQualificationTarget(changed_contract), manifest,
            tool_name=changed_contract.tool_name,
            arguments={"port": "Ethernet1", "vlan": "20"},
            environment="isolated-local-qualification", now=self.now,
        ))
        bundle = ProviderReleaseBundle(
            apiVersion=BUNDLE_SCHEMA,
            manifest=manifest,
            manifest_signature=sign_digest(
                manifest.digest, private_key=self.publisher, key_id="publisher-1",
                role="publisher", now=self.now, ttl_seconds=2_592_000,
            ),
            qualification=report,
            qualification_signature=sign_digest(
                report.digest, private_key=self.qualifier, key_id="qualifier-1",
                role="qualifier", now=self.now, ttl_seconds=2_592_000,
            ),
        )
        registry.stage(bundle)
        registry.publish(bundle.digest)
        with self.assertRaisesRegex(ProviderReleaseError, "approval_reference"):
            registry.promote(bundle.digest, environment="production")
        promoted = registry.promote(
            bundle.digest,
            environment="production",
            approval_reference="CHG-BREAKING-2",
        )
        self.assertFalse(promoted["compatibility"]["compatible"])


if __name__ == "__main__":
    unittest.main()
