from __future__ import annotations

import asyncio
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from network_runtime.capabilities import (
    CapabilityContract,
    CapabilityKind,
    DataSensitivity,
    EffectSemantics,
)
from network_runtime.contracts import sha256_json
from network_runtime.provider_external import (
    EXTERNAL_QUALIFICATION_SCHEMA,
    ExternalProviderProtocolError,
    ExternalQualificationConfig,
    ExternalQualificationTarget,
    qualify_external_provider,
)
from network_runtime.provider_release import (
    BUNDLE_SCHEMA,
    DEPLOYMENT_SCHEMA,
    MANIFEST_SCHEMA,
    TRUST_SCHEMA,
    ProviderAdmissionGate,
    ProviderDeploymentAttestation,
    ProviderManifest,
    ProviderReleaseBundle,
    ProviderReleaseRegistry,
    ProviderTrustStore,
    ReleasedCapability,
    SIGNED_DEPLOYMENT_SCHEMA,
    SignedProviderDeployment,
    TrustedKey,
    sign_digest,
)
from network_runtime.provider_release_cli import main as provider_cli


def run(awaitable):
    return asyncio.run(awaitable)


class ExternalProviderQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="external-provider-")
        self.root = Path(self.temp.name).resolve()
        source = Path(__file__).parent / "fixtures" / "external_qualification_provider.py"
        self.provider_script = self.root / "external-provider" / "provider.py"
        self.provider_script.parent.mkdir()
        shutil.copyfile(source, self.provider_script)
        self.state_path = self.root / "external-provider" / "state.json"
        self.contract_path = self.root / "external-provider" / "contract.json"
        self.contract = CapabilityContract(
            tool_name="external_set_access_vlan",
            capability_id="external.fabric.access-vlan.set",
            capability_version="1.0.0",
            domain="network",
            kind=CapabilityKind.EFFECT,
            action_type="reversible",
            effect_semantics=EffectSemantics.REVERSIBLE,
            provider_role="actor",
            provider_identity="mcp:external:external-network-actor@1.0.0",
            provider_kind="external-network-actor-mcp",
            input_schema_digest=sha256_json({"port": "string", "vlan": "integer"}),
            output_schema_digest=sha256_json({"state": "string"}),
            sensitivity=DataSensitivity.RESTRICTED,
            required_roles=("network-operator",),
            scope_fields=("port",),
            freshness_limit_seconds=60,
        )
        self.contract_path.write_text(
            json.dumps(self.contract.to_dict()), encoding="utf-8",
        )
        capability = ReleasedCapability.from_runtime(
            self.contract,
            result_contract="network-evidence-envelope-v1",
            l0_contract_hashes=(sha256_json({"l0": "external-access-vlan"}),),
        )
        self.manifest = ProviderManifest(
            apiVersion=MANIFEST_SCHEMA,
            provider_id="external.fabric",
            provider_version="1.0.0",
            provider_identity=self.contract.provider_identity,
            released_at=datetime.now(timezone.utc).isoformat(),
            artifacts={
                "oci-image": sha256_json({"image": "external-provider"}),
                "sbom": sha256_json({"spdx": "external-provider"}),
                "provenance": sha256_json({"slsa": "external-provider"}),
            },
            capabilities=(capability,),
        )
        self.config = ExternalQualificationConfig(
            apiVersion=EXTERNAL_QUALIFICATION_SCHEMA,
            command=(
                sys.executable,
                str(self.provider_script),
                str(self.state_path),
                str(self.contract_path),
            ),
            cwd=str(self.provider_script.parent),
            timeout_seconds=2,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_repository_external_process_passes_fixed_suite_and_real_restart(self) -> None:
        report = run(qualify_external_provider(
            self.config,
            self.manifest,
            tool_name=self.contract.tool_name,
            arguments={"port": "Ethernet1", "vlan": 20},
            environment="isolated-external-process",
        ))
        self.assertEqual(len(report.checks), 9)
        self.assertTrue(all(report.checks.values()))
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(
            state["operations"]["qualification-restart"]["state"], "applied",
        )

        config_path = self.root / "external-target.json"
        manifest_path = self.root / "manifest.json"
        arguments_path = self.root / "arguments.json"
        output_path = self.root / "qualification.json"
        config_path.write_text(
            self.config.model_dump_json(by_alias=True), encoding="utf-8",
        )
        manifest_path.write_text(
            self.manifest.model_dump_json(by_alias=True), encoding="utf-8",
        )
        arguments_path.write_text(
            json.dumps({"port": "Ethernet1", "vlan": 20}), encoding="utf-8",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = provider_cli([
                "qualify-external",
                "--config", str(config_path),
                "--manifest", str(manifest_path),
                "--tool-name", self.contract.tool_name,
                "--arguments", str(arguments_path),
                "--environment", "isolated-external-cli",
                "--output", str(output_path),
            ])
        self.assertEqual(exit_code, 0, output.getvalue())
        self.assertEqual(
            json.loads(output.getvalue())["checks"], 9,
        )
        self.assertTrue(output_path.is_file())

    def test_protocol_and_command_configuration_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "absolute file"):
            ExternalQualificationConfig(
                apiVersion=EXTERNAL_QUALIFICATION_SCHEMA,
                command=("python", "provider.py"),
                cwd=str(self.root),
            )

        async def undiscovered():
            async with ExternalQualificationTarget(self.config) as target:
                target.describe_capability(self.contract.tool_name)

        with self.assertRaisesRegex(
            ExternalProviderProtocolError, "discover_capability",
        ):
            run(undiscovered())

    def test_external_provider_release_deployment_and_admission_chain(self) -> None:
        now = datetime.now(timezone.utc)
        publisher = Ed25519PrivateKey.generate()
        qualifier = Ed25519PrivateKey.generate()
        deployer = Ed25519PrivateKey.generate()

        def trusted(key_id, role, private_key):
            return TrustedKey(
                key_id=key_id,
                role=role,
                public_key_pem=private_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode(),
                providers=("external.fabric",),
                not_before=(now - timedelta(minutes=1)).isoformat(),
                not_after=(now + timedelta(days=30)).isoformat(),
            )

        trust = ProviderTrustStore(
            apiVersion=TRUST_SCHEMA,
            required_artifacts=("oci-image", "sbom", "provenance"),
            require_deployment_attestation=True,
            keys=(
                trusted("external-publisher", "publisher", publisher),
                trusted("external-qualifier", "qualifier", qualifier),
                trusted("external-deployer", "deployer", deployer),
            ),
        )
        report = run(qualify_external_provider(
            self.config,
            self.manifest,
            tool_name=self.contract.tool_name,
            arguments={"port": "Ethernet1", "vlan": 20},
            environment="isolated-external-release",
            now=now,
        ))
        bundle = ProviderReleaseBundle(
            apiVersion=BUNDLE_SCHEMA,
            manifest=self.manifest,
            manifest_signature=sign_digest(
                self.manifest.digest,
                private_key=publisher,
                key_id="external-publisher",
                role="publisher",
                now=now,
                ttl_seconds=14 * 24 * 60 * 60,
            ),
            qualification=report,
            qualification_signature=sign_digest(
                report.digest,
                private_key=qualifier,
                key_id="external-qualifier",
                role="qualifier",
                now=now,
                ttl_seconds=14 * 24 * 60 * 60,
            ),
        )
        attestation = ProviderDeploymentAttestation(
            apiVersion=DEPLOYMENT_SCHEMA,
            provider_id=self.manifest.provider_id,
            provider_version=self.manifest.provider_version,
            provider_identity=self.manifest.provider_identity,
            release_digest=bundle.digest,
            manifest_digest=self.manifest.digest,
            environment="production-simulation",
            deployment_id="external-deploy-1",
            controller_identity="spiffe://deployment-controller/external-test",
            artifact_digests=self.manifest.artifacts,
            deployed_at=now.isoformat(),
            expires_at=(now + timedelta(days=7)).isoformat(),
        )
        deployment = SignedProviderDeployment(
            apiVersion=SIGNED_DEPLOYMENT_SCHEMA,
            attestation=attestation,
            signature=sign_digest(
                attestation.digest,
                private_key=deployer,
                key_id="external-deployer",
                role="deployer",
                now=now,
                ttl_seconds=7 * 24 * 60 * 60,
            ),
        )
        registry = ProviderReleaseRegistry(self.root / "release.sqlite", trust)
        registry.stage(bundle)
        registry.publish(bundle.digest)
        registry.promote(
            bundle.digest,
            environment="production-simulation",
            deployment=deployment,
        )
        evidence = ProviderAdmissionGate(
            registry, environment="production-simulation",
        ).admit(
            self.contract,
            provider_id="external.fabric",
            result_contract="network-evidence-envelope-v1",
        )
        self.assertEqual(evidence.release_digest, bundle.digest)
        self.assertEqual(evidence.deployment_digest, deployment.digest)
        self.assertTrue(registry.audit()["ok"])


if __name__ == "__main__":
    unittest.main()
