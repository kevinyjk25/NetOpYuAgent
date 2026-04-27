"""
auth_core.py
────────────
Dependency-free authentication primitives — JWT (HS256) issue/verify and
Identity dataclass. No FastAPI imports here; this lets the security logic
be unit-tested in environments without web framework dependencies.

The FastAPI integration (Depends/Header/HTTPException) lives in auth.py
which imports from this module.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass


# ── Configuration (read at import time) ───────────────────────────────────────

JWT_SECRET    = os.getenv("NETOPYU_JWT_SECRET", "")
JWT_ISSUER    = os.getenv("NETOPYU_JWT_ISSUER", "netopyu-auth")
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "0") == "1"


@dataclass
class Identity:
    """Verified identity. Attached to every authenticated request."""
    operator_id: str
    roles:       list[str]
    auth_method: str    # "jwt" | "api_key" | "disabled"

    def has_role(self, role: str) -> bool:
        return role in self.roles or "admin" in self.roles


class AuthError(Exception):
    """Raised by JWT verification on any failure (expired, tampered, malformed).
    The FastAPI layer catches this and converts it to HTTP 401."""
    pass


# ── Base64url helpers (RFC 7515 §2) ───────────────────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


# ── JWT issue / verify ───────────────────────────────────────────────────────

def issue_jwt(operator_id: str, roles: list[str], ttl_seconds: int = 3600,
              secret: str | None = None) -> str:
    """Issue an HS256 JWT. Raises RuntimeError if no secret configured."""
    sec = secret if secret is not None else JWT_SECRET
    if not sec:
        raise RuntimeError("Set NETOPYU_JWT_SECRET (or pass secret=...) to issue tokens")
    now = int(time.time())
    header  = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub":   operator_id,
        "roles": roles,
        "iat":   now,
        "exp":   now + ttl_seconds,
        "iss":   JWT_ISSUER,
    }
    h = _b64url(json.dumps(header,  separators=(",", ":")).encode())
    p = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(sec.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url(sig)}"


def verify_jwt(token: str, secret: str | None = None) -> dict:
    """Verify HS256 JWT and return claims. Raises AuthError on any failure."""
    sec = secret if secret is not None else JWT_SECRET
    if not sec:
        raise AuthError("JWT_SECRET not configured")

    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("Malformed token (expected 3 parts)")
    h_b64, p_b64, s_b64 = parts

    try:
        signing_input = f"{h_b64}.{p_b64}".encode()
        expected = hmac.new(sec.encode(), signing_input, hashlib.sha256).digest()
        actual   = _b64url_decode(s_b64)
    except Exception:
        raise AuthError("Malformed signature encoding")

    if not hmac.compare_digest(expected, actual):
        raise AuthError("Invalid signature")

    try:
        claims = json.loads(_b64url_decode(p_b64))
    except Exception:
        raise AuthError("Malformed payload")

    now = int(time.time())
    if claims.get("exp", 0) < now:
        raise AuthError("Token expired")
    if claims.get("iss") != JWT_ISSUER:
        raise AuthError("Invalid issuer")
    if not claims.get("sub"):
        raise AuthError("Missing subject claim")

    return claims


# ── API-key registry (configured via env var) ────────────────────────────────

def parse_api_keys(env_value: str) -> dict[str, tuple[str, list[str]]]:
    """
    Parse NETOPYU_API_KEYS="key1:operator1:role1,key2:operator2:role2"
    into a dict: { key: (operator_id, [roles]) }.
    """
    out: dict[str, tuple[str, list[str]]] = {}
    for entry in (env_value or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            key, opid, role = entry.split(":", 2)
            out[key] = (opid, [role])
        except ValueError:
            continue   # skip malformed
    return out


_API_KEYS = parse_api_keys(os.getenv("NETOPYU_API_KEYS", ""))


def lookup_api_key(key: str) -> Identity | None:
    """Return Identity if key is registered; None otherwise."""
    entry = _API_KEYS.get(key)
    if entry is None:
        return None
    opid, roles = entry
    return Identity(operator_id=opid, roles=roles, auth_method="api_key")
