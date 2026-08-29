"""Secret-safe readiness and live contract checks for enterprise controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import ApprovalError, RiskLevel, sha256_json
from .enterprise import (
    ControlPlaneTransportConfig,
    JwtValidationConfig,
    control_plane_from_environment,
    validate_control_plane_url,
)
from .identity import ENFORCED_MODE


SCHEMA = "netopyu.enterprise-conformance/v1"
_REQUIRED = (
    "NETOPYU_OIDC_ISSUER",
    "NETOPYU_OIDC_AUDIENCE",
    "NETOPYU_OIDC_JWKS_URL",
    "NETOPYU_GATEWAY_ISSUER",
    "NETOPYU_GATEWAY_AUDIENCE",
    "NETOPYU_GATEWAY_JWKS_URL",
    "NETOPYU_PDP_URL",
    "NETOPYU_CHANGE_AUTHORITY_URL",
)
_URLS = (
    "NETOPYU_OIDC_JWKS_URL",
    "NETOPYU_GATEWAY_JWKS_URL",
    "NETOPYU_PDP_URL",
    "NETOPYU_CHANGE_AUTHORITY_URL",
    "NETOPYU_GATEWAY_MINT_URL",
)
_CREDENTIALS = (
    "NETOPYU_OIDC_TOKEN",
    "NETOPYU_GATEWAY_TOKEN",
    "NETOPYU_APPROVER_OIDC_TOKEN",
    "NETOPYU_APPROVER_GATEWAY_TOKEN",
    "NETOPYU_CONTROL_PLANE_BEARER_TOKEN",
    "NETOPYU_GATEWAY_MINT_BEARER_TOKEN",
)


def _truth(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _error_text(error: Exception, environment: Mapping[str, str]) -> str:
    value = str(error)
    for name in _CREDENTIALS:
        secret = environment.get(name, "")
        if secret:
            value = value.replace(secret, "<redacted>")
    return f"{type(error).__name__}: {value}"


def configuration_report(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate deployment shape without network calls or secret disclosure."""
    env = os.environ if environment is None else environment
    errors: list[str] = []
    warnings: list[str] = []
    mode = str(env.get("NETOPYU_IDENTITY_MODE", "local-simulation"))
    missing = [name for name in _REQUIRED if not str(env.get(name, "")).strip()]
    if mode != ENFORCED_MODE:
        errors.append("NETOPYU_IDENTITY_MODE must be enforced for enterprise qualification")
    if missing:
        errors.append("required enterprise settings are incomplete")

    transport_binding: dict[str, Any] = {
        "kind": "explicit-tls",
        "custom_ca": False,
        "mutual_tls": False,
        "trust_environment": False,
    }
    try:
        transport = ControlPlaneTransportConfig(
            ca_bundle=env.get("NETOPYU_CONTROL_PLANE_CA_BUNDLE"),
            client_certificate=env.get("NETOPYU_CONTROL_PLANE_CLIENT_CERT"),
            client_key=env.get("NETOPYU_CONTROL_PLANE_CLIENT_KEY"),
            trust_environment=_truth(env.get("NETOPYU_CONTROL_PLANE_TRUST_ENV")),
        )
        transport_binding = transport.binding
    except (OSError, ValueError) as error:
        transport = None
        errors.append(_error_text(error, env))

    allow_loopback = _truth(env.get("NETOPYU_ENTERPRISE_ALLOW_LOOPBACK_HTTP"))
    algorithms = tuple(
        item.strip()
        for item in str(env.get("NETOPYU_OIDC_ALGORITHMS", "RS256")).split(",")
        if item.strip()
    )
    if not missing and transport is not None:
        try:
            JwtValidationConfig(
                issuer=str(env["NETOPYU_OIDC_ISSUER"]),
                audience=str(env["NETOPYU_OIDC_AUDIENCE"]),
                jwks_url=str(env["NETOPYU_OIDC_JWKS_URL"]),
                allowed_algorithms=algorithms,
                allow_loopback_http=allow_loopback,
                transport=transport,
            )
            JwtValidationConfig(
                issuer=str(env["NETOPYU_GATEWAY_ISSUER"]),
                audience=str(env["NETOPYU_GATEWAY_AUDIENCE"]),
                jwks_url=str(env["NETOPYU_GATEWAY_JWKS_URL"]),
                allowed_algorithms=algorithms,
                allow_loopback_http=allow_loopback,
                transport=transport,
            )
            for name in _URLS[2:]:
                endpoint = str(env.get(name, "")).strip()
                if endpoint:
                    validate_control_plane_url(
                        endpoint, allow_loopback_http=allow_loopback,
                    )
            minimum_aal = int(env.get("NETOPYU_OIDC_MIN_AAL", "2"))
            if not 1 <= minimum_aal <= 4:
                raise ValueError("minimum assurance level must be between 1 and 4")
        except (TypeError, ValueError) as error:
            errors.append(_error_text(error, env))

    dynamic_minting = bool(str(env.get("NETOPYU_GATEWAY_MINT_URL", "")).strip())
    if not dynamic_minting:
        warnings.append(
            "dynamic Gateway attestation minting is not configured; callers must supply a session-bound attestation"
        )
    if mode == ENFORCED_MODE and not transport_binding.get("mutual_tls"):
        warnings.append("mTLS is not configured for enterprise control-plane APIs")

    endpoints = {
        name.removeprefix("NETOPYU_").lower(): (
            _digest(str(env[name])) if str(env.get(name, "")).strip() else None
        )
        for name in _URLS
    }
    credentials = {
        "requester_access_token": bool(env.get("NETOPYU_OIDC_TOKEN")),
        "requester_gateway_attestation": bool(env.get("NETOPYU_GATEWAY_TOKEN")),
        "approver_access_token": bool(env.get("NETOPYU_APPROVER_OIDC_TOKEN")),
        "approver_gateway_attestation": bool(env.get("NETOPYU_APPROVER_GATEWAY_TOKEN")),
        "dynamic_gateway_minting": dynamic_minting,
        "control_api_authentication": bool(
            env.get("NETOPYU_CONTROL_PLANE_BEARER_TOKEN")
            or transport_binding.get("mutual_tls")
        ),
    }
    return {
        "schema": SCHEMA,
        "check": "configuration",
        "ok": not errors,
        "mode": mode,
        "missing_settings": missing,
        "algorithms": list(algorithms),
        "transport": transport_binding,
        "endpoints": endpoints,
        "credentials_present": credentials,
        "warnings": warnings,
        "errors": errors,
        "secret_safe": True,
    }


def live_contract_report(
    *,
    harness: str,
    session_id: str,
    profile: str,
    capability_id: str,
    target: str,
    risk_level: str,
    ticket_id: str,
) -> dict[str, Any]:
    """Exercise identity, JWKS, minting, PDP, and change contracts without effects."""
    config = configuration_report()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "check": "live-contract",
        "ok": False,
        "configuration": config,
        "checks": {},
        "errors": [],
        "secret_safe": True,
        "no_network_effects": True,
    }
    if not config["ok"]:
        result["errors"].append("enterprise configuration is not ready")
        return result
    requester_token = os.environ.get("NETOPYU_OIDC_TOKEN", "")
    approver_token = os.environ.get("NETOPYU_APPROVER_OIDC_TOKEN", "")
    dynamic_minting = bool(os.environ.get("NETOPYU_GATEWAY_MINT_URL"))
    if not requester_token or not approver_token:
        result["errors"].append("requester and approver access tokens are required")
        return result
    if not dynamic_minting and (
        not os.environ.get("NETOPYU_GATEWAY_TOKEN")
        or not os.environ.get("NETOPYU_APPROVER_GATEWAY_TOKEN")
    ):
        result["errors"].append(
            "Gateway attestations are required when dynamic minting is disabled"
        )
        return result
    if not ticket_id.strip():
        result["errors"].append("a qualification change ticket is required")
        return result
    try:
        risk = RiskLevel(risk_level)
    except ValueError:
        result["errors"].append("risk_level is invalid")
        return result

    requester_context = {"subject_token": requester_token}
    approver_context = {"subject_token": approver_token}
    if os.environ.get("NETOPYU_GATEWAY_TOKEN"):
        requester_context["gateway_token"] = os.environ["NETOPYU_GATEWAY_TOKEN"]
    if os.environ.get("NETOPYU_APPROVER_GATEWAY_TOKEN"):
        approver_context["gateway_token"] = os.environ["NETOPYU_APPROVER_GATEWAY_TOKEN"]
    try:
        with tempfile.TemporaryDirectory(prefix="netopyu-enterprise-") as directory:
            control = control_plane_from_environment(
                key_path=Path(directory) / "unused-qualification-proof.key",
            )
            requester = control.bind_requester(
                requester_context,
                harness=harness,
                session_id=session_id,
                profile=profile,
                capability_id=capability_id,
            )
            observer = control.bind_observer(
                requester_context,
                harness=harness,
                session_id=session_id,
                profile=profile,
                capability_id=capability_id,
                arguments={"qualification_target": target},
            )
            approver = control.verify_subject_context(
                approver_context, harness=harness, session_id=session_id,
            )
            control.policy.authorize_approvers(
                requester,
                (approver,),
                profile=profile,
                capability_id=capability_id,
                risk_level=risk,
                approval_mode="single",
                enforced=True,
                change_context={"ticket_id": ticket_id},
            )
            approval = control.authorize_external(
                action="effect.approve",
                subjects=(approver,),
                resource={
                    "profile": profile,
                    "capability_id": capability_id,
                    "targets": [target],
                    "risk_level": risk.value,
                    "requester_digest": requester.digest,
                    "qualification": True,
                },
                context={"ticket_id": ticket_id, "qualification": True},
            )
            if control.change_authority is None:
                raise ApprovalError("change authority is not configured")
            change = control.change_authority.qualify({
                "ticket_id": ticket_id,
                "requester": requester.to_dict(),
                "approvers": [approver.to_dict()],
                "profile": profile,
                "capability_id": capability_id,
                "targets": [target],
                "risk_level": risk.value,
                "plan_id": "enterprise-contract-qualification",
            })
        result["checks"] = {
            "requester": {
                "ok": True,
                "subject_digest": sha256_json({"subject_id": requester.subject_id}),
                "credential_id": requester.credential_id,
                "gateway_credential_id": requester.gateway_identity.get("credential_id"),
                "pdp_decision_id": requester.authorization_evidence.get("decision_id"),
            },
            "observation": {
                "ok": True,
                "pdp_decision_id": observer.authorization_evidence.get("decision_id"),
            },
            "approver": {
                "ok": True,
                "subject_digest": sha256_json({"subject_id": approver.subject_id}),
                "credential_id": approver.credential_id,
                "gateway_credential_id": approver.gateway_identity.get("credential_id"),
                "pdp_decision_id": approval.get("decision_id"),
            },
            "change": {
                "ok": True,
                "record_id": change.get("record_id"),
                "revision": change.get("revision"),
                "scope_hash": change.get("scope_hash"),
            },
        }
        result["ok"] = True
    except Exception as error:
        result["errors"].append(_error_text(error, os.environ))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate configuration without network calls")
    contract = subparsers.add_parser(
        "contract-test", help="call enterprise authorities without network effects",
    )
    contract.add_argument("--harness", default="dsh")
    contract.add_argument("--session-id", required=True)
    contract.add_argument("--profile", default="lan")
    contract.add_argument("--capability-id", default="grant_user_access")
    contract.add_argument("--target", default="qualification-target")
    contract.add_argument(
        "--risk-level", choices=[item.value for item in RiskLevel], default="low",
    )
    contract.add_argument(
        "--ticket-id", default=os.environ.get("NETOPYU_CONFORMANCE_CHANGE_TICKET", ""),
    )
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "doctor":
        report = configuration_report()
    else:
        report = live_contract_report(
            harness=arguments.harness,
            session_id=arguments.session_id,
            profile=arguments.profile,
            capability_id=arguments.capability_id,
            target=arguments.target,
            risk_level=arguments.risk_level,
            ticket_id=arguments.ticket_id,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
