"""CLI for signed Provider release verification and lifecycle management."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .provider_external import (
    ExternalQualificationConfig,
    qualify_external_provider,
)

from .provider_release import (
    BUNDLE_SCHEMA,
    DetachedSignature,
    ProviderDeploymentAttestation,
    ProviderManifest,
    ProviderReleaseBundle,
    ProviderReleaseError,
    ProviderReleaseRegistry,
    ProviderTrustStore,
    QualificationReport,
    SIGNED_DEPLOYMENT_SCHEMA,
    SignedProviderDeployment,
    compatibility_report,
    load_bundle,
    load_deployment,
    load_private_key,
    sign_digest,
)


def _write_new(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _model(path: str | Path, model: Any) -> Any:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _registry(arguments: argparse.Namespace) -> ProviderReleaseRegistry:
    registry_path = str(arguments.registry or "").strip()
    if not registry_path:
        raise ProviderReleaseError(
            "release lifecycle commands require --registry or "
            "NETOPYU_PROVIDER_RELEASE_DB"
        )
    trust = ProviderTrustStore.from_path(arguments.trust_store)
    return ProviderReleaseRegistry(registry_path, trust)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    schema = commands.add_parser("schema")
    schema.add_argument(
        "--kind",
        choices=[
            "manifest", "qualification", "bundle", "trust", "external-target",
            "deployment-attestation", "signed-deployment",
        ],
        required=True,
    )

    qualify = commands.add_parser("qualify-external")
    qualify.add_argument("--config", required=True)
    qualify.add_argument("--manifest", required=True)
    qualify.add_argument("--tool-name", required=True)
    qualify.add_argument("--arguments", required=True)
    qualify.add_argument("--environment", required=True)
    qualify.add_argument("--output", required=True)

    compatibility = commands.add_parser("compatibility")
    compatibility.add_argument("--previous", required=True)
    compatibility.add_argument("--candidate", required=True)

    for name, role, subject in (
        ("sign-manifest", "publisher", "manifest"),
        ("sign-qualification", "qualifier", "qualification"),
        ("sign-deployment", "deployer", "deployment-attestation"),
    ):
        command = commands.add_parser(name)
        command.add_argument(f"--{subject}", required=True)
        command.add_argument("--private-key", required=True)
        command.add_argument("--key-id", required=True)
        command.add_argument("--ttl-seconds", type=int, default=2_592_000)
        command.add_argument("--output", required=True)
        command.set_defaults(signature_role=role, signature_subject=subject)

    bundle = commands.add_parser("bundle")
    bundle.add_argument("--manifest", required=True)
    bundle.add_argument("--manifest-signature", required=True)
    bundle.add_argument("--qualification", required=True)
    bundle.add_argument("--qualification-signature", required=True)
    bundle.add_argument("--output", required=True)

    deployment = commands.add_parser("deployment-bundle")
    deployment.add_argument("--deployment-attestation", required=True)
    deployment.add_argument("--deployment-signature", required=True)
    deployment.add_argument("--output", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--trust-store", required=True)

    verify_deployment = commands.add_parser("verify-deployment")
    verify_deployment.add_argument("--bundle", required=True)
    verify_deployment.add_argument("--deployment", required=True)
    verify_deployment.add_argument("--trust-store", required=True)
    verify_deployment.add_argument("--environment", required=True)

    for name in ("stage", "publish", "promote", "rollback", "deprecate", "status", "audit"):
        command = commands.add_parser(name)
        command.add_argument(
            "--registry", default=os.environ.get("NETOPYU_PROVIDER_RELEASE_DB", ""),
        )
        command.add_argument("--trust-store", required=True)
        if name == "stage":
            command.add_argument("--bundle", required=True)
        elif name in {"publish", "promote", "deprecate"}:
            command.add_argument("--release-digest", required=True)
        if name == "promote":
            command.add_argument("--environment", required=True)
            command.add_argument("--approval-reference", default="")
            command.add_argument("--deployment")
        elif name == "rollback":
            command.add_argument("--provider-id", required=True)
            command.add_argument("--environment", required=True)
            command.add_argument("--approval-reference", required=True)
            command.add_argument("--target-release-digest")
            command.add_argument("--deployment")
        elif name == "deprecate":
            command.add_argument("--reason", required=True)
    return value


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "schema":
        models = {
            "manifest": ProviderManifest,
            "qualification": QualificationReport,
            "bundle": ProviderReleaseBundle,
            "trust": ProviderTrustStore,
            "external-target": ExternalQualificationConfig,
            "deployment-attestation": ProviderDeploymentAttestation,
            "signed-deployment": SignedProviderDeployment,
        }
        return {
            "ok": True,
            "kind": arguments.kind,
            "json_schema": models[arguments.kind].model_json_schema(by_alias=True),
        }
    if arguments.command == "qualify-external":
        raw_arguments = json.loads(Path(arguments.arguments).read_text(encoding="utf-8"))
        if not isinstance(raw_arguments, dict):
            raise ValueError("external qualification arguments must be a JSON object")
        report = asyncio.run(qualify_external_provider(
            ExternalQualificationConfig.from_path(arguments.config),
            _model(arguments.manifest, ProviderManifest),
            tool_name=arguments.tool_name,
            arguments=raw_arguments,
            environment=arguments.environment,
        ))
        _write_new(arguments.output, report.model_dump(by_alias=True, mode="json"))
        return {
            "ok": True,
            "output": str(Path(arguments.output).expanduser().resolve()),
            "qualification_digest": report.digest,
            "checks": len(report.checks),
        }
    if arguments.command == "compatibility":
        return {
            "ok": True,
            **compatibility_report(
                _model(arguments.previous, ProviderManifest),
                _model(arguments.candidate, ProviderManifest),
            ),
        }
    if arguments.command in {"sign-manifest", "sign-qualification", "sign-deployment"}:
        if arguments.signature_subject == "manifest":
            subject = _model(arguments.manifest, ProviderManifest)
        elif arguments.signature_subject == "qualification":
            subject = _model(arguments.qualification, QualificationReport)
        else:
            subject = _model(
                arguments.deployment_attestation, ProviderDeploymentAttestation,
            )
        signature = sign_digest(
            subject.digest,
            private_key=load_private_key(arguments.private_key),
            key_id=arguments.key_id,
            role=arguments.signature_role,
            ttl_seconds=arguments.ttl_seconds,
        )
        _write_new(arguments.output, signature.model_dump(by_alias=True, mode="json"))
        return {
            "ok": True,
            "output": str(Path(arguments.output).expanduser().resolve()),
            "subject_digest": subject.digest,
            "key_id": arguments.key_id,
            "role": arguments.signature_role,
        }
    if arguments.command == "bundle":
        release = ProviderReleaseBundle(
            apiVersion=BUNDLE_SCHEMA,
            manifest=_model(arguments.manifest, ProviderManifest),
            manifest_signature=_model(arguments.manifest_signature, DetachedSignature),
            qualification=_model(arguments.qualification, QualificationReport),
            qualification_signature=_model(
                arguments.qualification_signature, DetachedSignature,
            ),
        )
        _write_new(arguments.output, release.model_dump(by_alias=True, mode="json"))
        return {
            "ok": True,
            "output": str(Path(arguments.output).expanduser().resolve()),
            "release_digest": release.digest,
        }
    if arguments.command == "deployment-bundle":
        deployment = SignedProviderDeployment(
            apiVersion=SIGNED_DEPLOYMENT_SCHEMA,
            attestation=_model(
                arguments.deployment_attestation, ProviderDeploymentAttestation,
            ),
            signature=_model(arguments.deployment_signature, DetachedSignature),
        )
        _write_new(arguments.output, deployment.model_dump(by_alias=True, mode="json"))
        return {
            "ok": True,
            "output": str(Path(arguments.output).expanduser().resolve()),
            "deployment_digest": deployment.digest,
        }
    if arguments.command == "verify":
        return ProviderTrustStore.from_path(arguments.trust_store).verify_bundle(
            load_bundle(arguments.bundle),
        )
    if arguments.command == "verify-deployment":
        bundle = load_bundle(arguments.bundle)
        trust = ProviderTrustStore.from_path(arguments.trust_store)
        return {
            **trust.verify_bundle(bundle),
            "deployment": trust.verify_deployment(
                bundle,
                load_deployment(arguments.deployment),
                environment=arguments.environment,
            ),
        }

    registry = _registry(arguments)
    if arguments.command == "stage":
        return registry.stage(load_bundle(arguments.bundle))
    if arguments.command == "publish":
        return registry.publish(arguments.release_digest)
    if arguments.command == "promote":
        return registry.promote(
            arguments.release_digest,
            environment=arguments.environment,
            approval_reference=arguments.approval_reference,
            deployment=(load_deployment(arguments.deployment) if arguments.deployment else None),
        )
    if arguments.command == "rollback":
        return registry.rollback(
            provider_id=arguments.provider_id,
            environment=arguments.environment,
            approval_reference=arguments.approval_reference,
            target_release_digest=arguments.target_release_digest,
            deployment=(load_deployment(arguments.deployment) if arguments.deployment else None),
        )
    if arguments.command == "deprecate":
        return registry.deprecate(
            arguments.release_digest, reason=arguments.reason,
        )
    if arguments.command == "status":
        return registry.status()
    return registry.audit()


def main(argv: list[str] | None = None) -> int:
    try:
        report = _run(parser().parse_args(argv))
    except (OSError, ValueError, ProviderReleaseError) as error:
        report = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
