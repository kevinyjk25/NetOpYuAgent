"""Enterprise identity, policy and change-system adapters.

All endpoints are deployment configuration, never model input.  Production
mode accepts only asymmetric JWTs, pins issuer/audience/algorithm, cross-binds
the human token to a separately signed Gateway attestation, and fails closed
on every network, schema, cache or authorization error.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import ssl
import stat
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import jwt

from .contracts import ApprovalError, RiskLevel, sha256_json
from .identity import (
    ApprovalControlPlane,
    ENFORCED_MODE,
    SubjectIdentity,
)


_ASYMMETRIC_ALGORITHMS = {"RS256", "RS384", "RS512", "ES256", "ES384"}
_RISK_ORDER = {
    RiskLevel.LOW.value: 0,
    RiskLevel.MEDIUM.value: 1,
    RiskLevel.HIGH.value: 2,
    RiskLevel.CRITICAL.value: 3,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any, *, field: str) -> str:
    try:
        timestamp = datetime.fromtimestamp(float(value), timezone.utc)
    except (TypeError, ValueError, OSError) as error:
        raise ApprovalError(f"JWT claim {field!r} is not a valid NumericDate") from error
    return timestamp.isoformat()


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split()
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise ApprovalError(f"JWT claim {field!r} must be a string or string array")
    normalized = tuple(sorted({item.strip() for item in values if item.strip()}))
    if not normalized:
        raise ApprovalError(f"JWT claim {field!r} cannot be empty")
    return normalized


def _trusted_url(value: str, *, allow_loopback_http: bool) -> str:
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("control-plane URL cannot contain credentials or fragments")
    if parsed.scheme == "https" and parsed.hostname:
        return value
    if parsed.scheme != "http" or not parsed.hostname or not allow_loopback_http:
        raise ValueError("control-plane URL must use HTTPS")
    host = parsed.hostname
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise ValueError("plain HTTP is allowed only for an explicit loopback qualification lab")
    return value


def validate_control_plane_url(
    value: str,
    *,
    allow_loopback_http: bool = False,
) -> str:
    """Validate one deployment-owned authority URL and return it unchanged."""
    return _trusted_url(value, allow_loopback_http=allow_loopback_http)


def _public_endpoint_binding(kind: str, endpoint: str, **values: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "endpoint_hash": "sha256:" + hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        **values,
    }


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ControlPlaneTransportConfig:
    """Explicit TLS trust and optional client certificate for control APIs."""

    ca_bundle: str | None = None
    client_certificate: str | None = None
    client_key: str | None = None
    trust_environment: bool = False

    def __post_init__(self) -> None:
        if bool(self.client_certificate) != bool(self.client_key):
            raise ValueError("mTLS requires both client certificate and private key")
        for label, value in (
            ("CA bundle", self.ca_bundle),
            ("client certificate", self.client_certificate),
            ("client private key", self.client_key),
        ):
            if value and not Path(value).expanduser().is_file():
                raise ValueError(f"{label} is not a readable file")
        if self.client_key:
            key_path = Path(self.client_key).expanduser()
            if stat.S_IMODE(key_path.stat().st_mode) & 0o077:
                raise ValueError("mTLS client private key must be owner-only")

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "kind": "explicit-tls",
            "custom_ca": bool(self.ca_bundle),
            "mutual_tls": bool(self.client_certificate),
            "ca_digest": (
                _file_digest(Path(self.ca_bundle).expanduser())
                if self.ca_bundle else None
            ),
            "client_certificate_digest": (
                _file_digest(Path(self.client_certificate).expanduser())
                if self.client_certificate else None
            ),
            "trust_environment": self.trust_environment,
        }

    def ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(
            cafile=str(Path(self.ca_bundle).expanduser()) if self.ca_bundle else None,
        )
        if self.client_certificate and self.client_key:
            context.load_cert_chain(
                certfile=str(Path(self.client_certificate).expanduser()),
                keyfile=str(Path(self.client_key).expanduser()),
            )
        return context

    def client(self, *, timeout_seconds: float) -> httpx.Client:
        return httpx.Client(
            verify=self.ssl_context(),
            trust_env=self.trust_environment,
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )


@dataclass(frozen=True)
class JwtValidationConfig:
    issuer: str
    audience: str
    jwks_url: str
    allowed_algorithms: tuple[str, ...] = ("RS256",)
    cache_ttl_seconds: int = 300
    timeout_seconds: float = 3.0
    clock_skew_seconds: int = 15
    max_token_lifetime_seconds: int = 900
    allow_loopback_http: bool = False
    transport: ControlPlaneTransportConfig = field(
        default_factory=ControlPlaneTransportConfig,
    )

    def __post_init__(self) -> None:
        if not self.issuer.strip() or not self.audience.strip():
            raise ValueError("JWT issuer and audience are required")
        _trusted_url(self.jwks_url, allow_loopback_http=self.allow_loopback_http)
        algorithms = set(self.allowed_algorithms)
        if not algorithms or not algorithms.issubset(_ASYMMETRIC_ALGORITHMS):
            raise ValueError("only explicitly pinned asymmetric JWT algorithms are allowed")
        if not 15 <= self.cache_ttl_seconds <= 3600:
            raise ValueError("JWKS cache TTL must be between 15 and 3600 seconds")
        if not 0 <= self.clock_skew_seconds <= 60:
            raise ValueError("JWT clock skew must be between 0 and 60 seconds")
        if not 60 <= self.max_token_lifetime_seconds <= 3600:
            raise ValueError("JWT lifetime bound must be between 60 and 3600 seconds")


class JwksJwtDecoder:
    """Small pinned JWKS cache with unknown-kid refresh and strict JWT checks."""

    def __init__(self, config: JwtValidationConfig) -> None:
        self.config = config
        self._keys: dict[str, dict[str, Any]] = {}
        self._refresh_at = 0.0
        self._lock = threading.Lock()

    @property
    def binding(self) -> dict[str, Any]:
        return _public_endpoint_binding(
            "oidc-jwks",
            self.config.jwks_url,
            issuer=self.config.issuer,
            audience=self.config.audience,
            algorithms=list(self.config.allowed_algorithms),
            transport=self.config.transport.binding,
        )

    def _refresh(self) -> None:
        try:
            with self.config.transport.client(
                timeout_seconds=self.config.timeout_seconds,
            ) as client:
                response = client.get(self.config.jwks_url)
                response.raise_for_status()
        except Exception as error:
            raise ApprovalError("JWKS authority is unavailable") from error
        if len(response.content) > 1_048_576:
            raise ApprovalError("JWKS response exceeds 1 MiB")
        try:
            document = response.json()
            keys = document["keys"]
        except (ValueError, KeyError, TypeError) as error:
            raise ApprovalError("JWKS response is malformed") from error
        if not isinstance(keys, list) or not 1 <= len(keys) <= 100:
            raise ApprovalError("JWKS must contain between 1 and 100 public keys")
        accepted: dict[str, dict[str, Any]] = {}
        for item in keys:
            if not isinstance(item, dict):
                raise ApprovalError("JWKS contains a non-object key")
            kid = str(item.get("kid") or "")
            algorithm = str(item.get("alg") or "")
            if not kid or algorithm not in self.config.allowed_algorithms:
                continue
            if item.get("use") not in (None, "sig"):
                continue
            key_ops = item.get("key_ops")
            if key_ops is not None and "verify" not in key_ops:
                continue
            if any(name in item for name in ("d", "p", "q", "dp", "dq", "qi")):
                raise ApprovalError("JWKS must not expose private key material")
            if kid in accepted:
                raise ApprovalError("JWKS contains duplicate key ids")
            accepted[kid] = dict(item)
        if not accepted:
            raise ApprovalError("JWKS contains no usable pinned signing keys")
        self._keys = accepted
        self._refresh_at = time.monotonic() + self.config.cache_ttl_seconds

    def _jwk(self, kid: str) -> dict[str, Any]:
        with self._lock:
            if time.monotonic() >= self._refresh_at:
                self._refresh()
            value = self._keys.get(kid)
            if value is None:
                self._refresh()
                value = self._keys.get(kid)
            if value is None:
                raise ApprovalError("JWT signing key id is not trusted")
            return dict(value)

    def decode(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or not token or len(token) > 16_384:
            raise ApprovalError("JWT credential is missing or too large")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as error:
            raise ApprovalError("JWT header is malformed") from error
        kid = str(header.get("kid") or "")
        algorithm = str(header.get("alg") or "")
        if not kid or algorithm not in self.config.allowed_algorithms:
            raise ApprovalError("JWT algorithm or signing key id is not allowed")
        jwk = self._jwk(kid)
        if jwk.get("alg") not in (None, algorithm):
            raise ApprovalError("JWT header algorithm does not match its JWK")
        try:
            key = jwt.PyJWK.from_dict(jwk, algorithm=algorithm).key
            claims = jwt.decode(
                token,
                key=key,
                algorithms=[algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=self.config.clock_skew_seconds,
                options={"require": ["iss", "aud", "sub", "exp", "iat", "nbf", "jti", "sid"]},
            )
        except jwt.PyJWTError as error:
            raise ApprovalError("JWT signature or registered claims are invalid") from error
        if not isinstance(claims, dict):
            raise ApprovalError("JWT claims must be an object")
        try:
            lifetime = float(claims["exp"]) - float(claims["iat"])
        except (KeyError, TypeError, ValueError) as error:
            raise ApprovalError("JWT lifetime claims are invalid") from error
        if lifetime <= 0 or lifetime > self.config.max_token_lifetime_seconds:
            raise ApprovalError("JWT lifetime exceeds the configured maximum")
        return claims


class OidcJwksSubjectVerifier:
    """Verify a human access token plus a separate Gateway sender assertion."""

    def __init__(
        self,
        subject_decoder: JwksJwtDecoder,
        gateway_decoder: JwksJwtDecoder,
        *,
        gateway_minter: "HttpGatewayAttestationMinter | None" = None,
        minimum_assurance_level: int = 2,
    ) -> None:
        if not 1 <= minimum_assurance_level <= 4:
            raise ValueError("minimum assurance level must be between 1 and 4")
        self.subject_decoder = subject_decoder
        self.gateway_decoder = gateway_decoder
        self.gateway_minter = gateway_minter
        self.minimum_assurance_level = minimum_assurance_level

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "kind": "oidc+jwks+gateway-attestation",
            "subject": self.subject_decoder.binding,
            "gateway": self.gateway_decoder.binding,
            "gateway_minter": (
                self.gateway_minter.binding if self.gateway_minter else None
            ),
            "minimum_assurance_level": self.minimum_assurance_level,
        }

    def verify(
        self,
        context: dict[str, Any],
        *,
        expected_harness: str,
        expected_session_id: str,
    ) -> SubjectIdentity:
        subject_token = str(context.get("subject_token") or "")
        subject = self.subject_decoder.decode(subject_token)
        gateway_token = str(context.get("gateway_token") or "")
        if not gateway_token and self.gateway_minter is not None:
            gateway_token = self.gateway_minter.mint(
                subject_token=subject_token,
                harness=expected_harness,
                session_id=expected_session_id,
                purpose=str(subject.get("purpose") or "enterprise-effect-operation"),
            )
        gateway = self.gateway_decoder.decode(gateway_token)
        subject_id = str(subject.get("sub") or "")
        gateway_session_id = str(gateway.get("sid") or "")
        harness = str(gateway.get("harness") or "")
        if gateway_session_id != expected_session_id:
            raise ApprovalError("Gateway attestation session binding mismatch")
        if harness != expected_harness:
            raise ApprovalError("Gateway attestation Harness binding mismatch")
        if str(gateway.get("act_sub") or "") != subject_id:
            raise ApprovalError("Gateway attestation is bound to a different subject")
        if str(gateway.get("subject_jti") or "") != str(subject.get("jti") or ""):
            raise ApprovalError("Gateway attestation is bound to a different access token")
        authorized_party = str(subject.get("azp") or "")
        gateway_client = str(gateway.get("client_id") or "")
        if authorized_party and authorized_party != gateway_client:
            raise ApprovalError("OIDC authorized party does not match the Gateway client")
        if gateway.get("token_use") != "gateway_attestation":
            raise ApprovalError("Gateway JWT is not a sender attestation")
        if subject.get("token_use") not in ("access", "access_token"):
            raise ApprovalError("OIDC credential is not an access token")
        try:
            assurance = int(subject.get("aal"))
        except (TypeError, ValueError) as error:
            raise ApprovalError("OIDC access token has no numeric assurance level") from error
        if assurance < self.minimum_assurance_level:
            raise ApprovalError("OIDC assurance level is below policy")
        roles = _string_list(subject.get("roles"), field="roles")
        scopes = _string_list(subject.get("scope"), field="scope")
        clearance = str(subject.get("clearance") or "")
        if clearance not in {"public", "internal", "confidential", "restricted"}:
            raise ApprovalError("OIDC access token has no valid data clearance")
        subject_expires = datetime.fromtimestamp(float(subject["exp"]), timezone.utc)
        gateway_expires = datetime.fromtimestamp(float(gateway["exp"]), timezone.utc)
        credential_material = {
            "subject_issuer": subject["iss"],
            "subject_jti": subject["jti"],
            "gateway_issuer": gateway["iss"],
            "gateway_jti": gateway["jti"],
        }
        return SubjectIdentity(
            subject_id=subject_id,
            issuer=str(subject["iss"]),
            harness=harness,
            session_id=gateway_session_id,
            roles=roles,
            scopes=scopes,
            purpose=str(subject.get("purpose") or "enterprise-effect-operation"),
            assurance_level=assurance,
            auth_method="oidc-jwks+gateway-attestation",
            authenticated_at=_iso(subject.get("auth_time", subject["iat"]), field="auth_time"),
            expires_at=min(subject_expires, gateway_expires).isoformat(),
            credential_id=sha256_json(credential_material),
            local_simulation=False,
            gateway_identity={
                "subject_id": str(gateway.get("sub") or ""),
                "issuer": str(gateway["iss"]),
                "client_id": str(gateway.get("client_id") or ""),
                "credential_id": sha256_json({
                    "issuer": gateway["iss"], "jti": gateway["jti"],
                }),
                "harness": harness,
                "session_id": gateway_session_id,
                "subject_token_id": sha256_json({
                    "issuer": subject["iss"], "jti": subject["jti"],
                }),
            },
            subject_attributes={"clearance": clearance},
        )


class _HttpAuthority:
    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 3.0,
        allow_loopback_http: bool = False,
        transport: ControlPlaneTransportConfig | None = None,
    ) -> None:
        self.endpoint = _trusted_url(endpoint, allow_loopback_http=allow_loopback_http)
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self.transport = transport or ControlPlaneTransportConfig()

    def _post(self, payload: dict[str, Any], *, unavailable: str) -> dict[str, Any]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        try:
            with self.transport.client(timeout_seconds=self.timeout_seconds) as client:
                response = client.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
        except Exception as error:
            raise ApprovalError(unavailable) from error
        if len(response.content) > 1_048_576:
            raise ApprovalError("control-plane response exceeds 1 MiB")
        try:
            value = response.json()
        except ValueError as error:
            raise ApprovalError("control-plane response is malformed") from error
        if not isinstance(value, dict):
            raise ApprovalError("control-plane response must be an object")
        return value


class HttpGatewayAttestationMinter(_HttpAuthority):
    """Exchange a human token plus Harness session for a short-lived sender JWT."""

    @property
    def binding(self) -> dict[str, Any]:
        return _public_endpoint_binding(
            "http-gateway-attestation-minter",
            self.endpoint,
            protocol="netopyu.gateway-mint/v1",
            transport=self.transport.binding,
        )

    def mint(
        self,
        *,
        subject_token: str,
        harness: str,
        session_id: str,
        purpose: str,
    ) -> str:
        if not subject_token or len(subject_token) > 16_384:
            raise ApprovalError("Gateway mint requires one bounded subject token")
        value = self._post({
            "schema": "netopyu.gateway-mint-request/v1",
            "subject_token": subject_token,
            "harness": harness,
            "session_id": session_id,
            "purpose": purpose,
        }, unavailable="Gateway attestation minter is unavailable")
        if value.get("schema") != "netopyu.gateway-mint-response/v1":
            raise ApprovalError("Gateway minter returned an invalid response schema")
        token = value.get("gateway_token")
        if not isinstance(token, str) or not token or len(token) > 16_384:
            raise ApprovalError("Gateway minter returned an invalid attestation")
        return token


class HttpPolicyDecisionPoint(_HttpAuthority):
    @property
    def binding(self) -> dict[str, Any]:
        return _public_endpoint_binding(
            "http-pdp", self.endpoint,
            protocol="netopyu.pdp/v1", transport=self.transport.binding,
        )

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        value = self._post(
            {"schema": "netopyu.pdp-request/v1", **request},
            unavailable="enterprise PDP is unavailable",
        )
        if value.get("schema") != "netopyu.pdp-decision/v1" or not isinstance(
            value.get("allow"), bool
        ):
            raise ApprovalError("enterprise PDP returned an invalid decision schema")
        obligations = value.get("obligations") or {}
        if not isinstance(obligations, dict):
            raise ApprovalError("enterprise PDP obligations must be an object")
        return {
            "allow": value["allow"],
            "decision_id": str(value.get("decision_id") or ""),
            "policy_id": str(value.get("policy_id") or ""),
            "policy_version": str(value.get("policy_version") or ""),
            "evaluated_at": str(value.get("evaluated_at") or ""),
            "reason": str(value.get("reason") or ""),
            "obligations": obligations,
        }


class HttpChangeAuthority(_HttpAuthority):
    @property
    def binding(self) -> dict[str, Any]:
        return _public_endpoint_binding(
            "http-change-authority", self.endpoint,
            protocol="netopyu.change/v1", transport=self.transport.binding,
        )

    def qualify(self, request: dict[str, Any]) -> dict[str, Any]:
        value = self._post(
            {"schema": "netopyu.change-query/v1", **request},
            unavailable="change authority is unavailable",
        )
        if value.get("schema") != "netopyu.change-record/v1":
            raise ApprovalError("change authority returned an invalid record schema")
        ticket_id = str(value.get("ticket_id") or "")
        if ticket_id != str(request.get("ticket_id") or ""):
            raise ApprovalError("change authority returned a different ticket")
        if str(value.get("status") or "").lower() != "approved":
            raise ApprovalError("change ticket is not approved")
        revision = str(value.get("revision") or "")
        approved_by = value.get("approved_by")
        if not revision or not isinstance(approved_by, list) or not approved_by:
            raise ApprovalError("change record lacks revision or approver evidence")
        start_raw = str(value.get("window_start") or "")
        end_raw = str(value.get("window_end") or "")
        try:
            start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
            if start.tzinfo is None or end.tzinfo is None:
                raise ValueError("timezone required")
            now = _utc_now()
            if start >= end or not start.astimezone(timezone.utc) <= now < end.astimezone(timezone.utc):
                raise ApprovalError("change ticket is outside its approved window")
        except ApprovalError:
            raise
        except (TypeError, ValueError) as error:
            raise ApprovalError("change ticket has an invalid maintenance window") from error
        profiles = _string_list(value.get("allowed_profiles"), field="allowed_profiles")
        capabilities = _string_list(
            value.get("allowed_capabilities"), field="allowed_capabilities",
        )
        targets = _string_list(value.get("allowed_targets"), field="allowed_targets")
        if "*" not in profiles and str(request.get("profile")) not in profiles:
            raise ApprovalError("change ticket does not authorize this profile")
        if "*" not in capabilities and str(request.get("capability_id")) not in capabilities:
            raise ApprovalError("change ticket does not authorize this capability")
        requested_targets = {str(item) for item in request.get("targets") or []}
        if "*" not in targets and not requested_targets.issubset(set(targets)):
            raise ApprovalError("change ticket does not authorize all targets")
        ceiling = str(value.get("risk_ceiling") or "")
        if ceiling not in _RISK_ORDER or str(request.get("risk_level")) not in _RISK_ORDER:
            raise ApprovalError("change ticket has an invalid risk ceiling")
        if _RISK_ORDER[str(request["risk_level"])] > _RISK_ORDER[ceiling]:
            raise ApprovalError("change ticket risk ceiling is too low")
        return {
            "ticket_id": ticket_id,
            "record_id": str(value.get("record_id") or ticket_id),
            "revision": revision,
            "status": "approved",
            "approved_by": [str(item) for item in approved_by],
            "window_start": start.astimezone(timezone.utc).isoformat(),
            "window_end": end.astimezone(timezone.utc).isoformat(),
            "scope_hash": sha256_json({
                "profiles": list(profiles),
                "capabilities": list(capabilities),
                "targets": list(targets),
                "risk_ceiling": ceiling,
            }),
            "authority": str(value.get("authority") or "enterprise-change-system"),
        }


def control_plane_from_environment(*, key_path: str | os.PathLike[str]) -> ApprovalControlPlane:
    """Create the default local or fully wired enforced control plane."""
    mode = os.environ.get("NETOPYU_IDENTITY_MODE", "local-simulation")
    if mode != ENFORCED_MODE:
        return ApprovalControlPlane(key_path=key_path, mode=mode)
    required = {
        "NETOPYU_OIDC_ISSUER": os.environ.get("NETOPYU_OIDC_ISSUER"),
        "NETOPYU_OIDC_AUDIENCE": os.environ.get("NETOPYU_OIDC_AUDIENCE"),
        "NETOPYU_OIDC_JWKS_URL": os.environ.get("NETOPYU_OIDC_JWKS_URL"),
        "NETOPYU_GATEWAY_ISSUER": os.environ.get("NETOPYU_GATEWAY_ISSUER"),
        "NETOPYU_GATEWAY_AUDIENCE": os.environ.get("NETOPYU_GATEWAY_AUDIENCE"),
        "NETOPYU_GATEWAY_JWKS_URL": os.environ.get("NETOPYU_GATEWAY_JWKS_URL"),
        "NETOPYU_PDP_URL": os.environ.get("NETOPYU_PDP_URL"),
        "NETOPYU_CHANGE_AUTHORITY_URL": os.environ.get("NETOPYU_CHANGE_AUTHORITY_URL"),
    }
    if any(not value for value in required.values()):
        return ApprovalControlPlane(key_path=key_path, mode=ENFORCED_MODE)
    allow_loopback = os.environ.get(
        "NETOPYU_ENTERPRISE_ALLOW_LOOPBACK_HTTP", "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    algorithms = tuple(
        item.strip() for item in os.environ.get("NETOPYU_OIDC_ALGORITHMS", "RS256").split(",")
        if item.strip()
    )
    transport = ControlPlaneTransportConfig(
        ca_bundle=os.environ.get("NETOPYU_CONTROL_PLANE_CA_BUNDLE"),
        client_certificate=os.environ.get("NETOPYU_CONTROL_PLANE_CLIENT_CERT"),
        client_key=os.environ.get("NETOPYU_CONTROL_PLANE_CLIENT_KEY"),
        trust_environment=os.environ.get(
            "NETOPYU_CONTROL_PLANE_TRUST_ENV", "0",
        ).strip().lower() in {"1", "true", "yes", "on"},
    )
    subject_decoder = JwksJwtDecoder(JwtValidationConfig(
        issuer=str(required["NETOPYU_OIDC_ISSUER"]),
        audience=str(required["NETOPYU_OIDC_AUDIENCE"]),
        jwks_url=str(required["NETOPYU_OIDC_JWKS_URL"]),
        allowed_algorithms=algorithms,
        allow_loopback_http=allow_loopback,
        transport=transport,
    ))
    gateway_decoder = JwksJwtDecoder(JwtValidationConfig(
        issuer=str(required["NETOPYU_GATEWAY_ISSUER"]),
        audience=str(required["NETOPYU_GATEWAY_AUDIENCE"]),
        jwks_url=str(required["NETOPYU_GATEWAY_JWKS_URL"]),
        allowed_algorithms=algorithms,
        allow_loopback_http=allow_loopback,
        transport=transport,
    ))
    bearer = os.environ.get("NETOPYU_CONTROL_PLANE_BEARER_TOKEN")
    gateway_mint_url = os.environ.get("NETOPYU_GATEWAY_MINT_URL")
    gateway_minter = (
        HttpGatewayAttestationMinter(
            gateway_mint_url,
            bearer_token=(
                os.environ.get("NETOPYU_GATEWAY_MINT_BEARER_TOKEN") or bearer
            ),
            allow_loopback_http=allow_loopback,
            transport=transport,
        )
        if gateway_mint_url else None
    )
    return ApprovalControlPlane(
        key_path=key_path,
        mode=ENFORCED_MODE,
        verifier=OidcJwksSubjectVerifier(
            subject_decoder,
            gateway_decoder,
            gateway_minter=gateway_minter,
            minimum_assurance_level=int(os.environ.get("NETOPYU_OIDC_MIN_AAL", "2")),
        ),
        pdp=HttpPolicyDecisionPoint(
            str(required["NETOPYU_PDP_URL"]),
            bearer_token=bearer,
            allow_loopback_http=allow_loopback,
            transport=transport,
        ),
        change_authority=HttpChangeAuthority(
            str(required["NETOPYU_CHANGE_AUTHORITY_URL"]),
            bearer_token=bearer,
            allow_loopback_http=allow_loopback,
            transport=transport,
        ),
        require_external_controls=True,
    )


__all__ = [
    "ControlPlaneTransportConfig",
    "HttpChangeAuthority",
    "HttpGatewayAttestationMinter",
    "HttpPolicyDecisionPoint",
    "JwksJwtDecoder",
    "JwtValidationConfig",
    "OidcJwksSubjectVerifier",
    "control_plane_from_environment",
    "validate_control_plane_url",
]
