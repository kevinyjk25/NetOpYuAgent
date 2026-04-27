"""
auth.py
───────
FastAPI layer over auth_core.

When config `auth.enabled` is false, ALL auth dependencies become no-ops
that return a single dev identity. No Authorization header, no JWT check,
no role check — every request is treated as the dev_operator.

When config `auth.enabled` is true, full JWT/API-key verification runs.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth_core import (
    Identity, AuthError,
    verify_jwt, lookup_api_key, issue_jwt,
)
from config import cfg

logger = logging.getLogger(__name__)

# Read once at import time. Changing config requires a restart.
_AUTH_ENABLED = cfg.auth.enabled
_DEV_OPERATOR = cfg.auth.dev_operator

if not _AUTH_ENABLED:
    logger.warning(
        "=" * 60
        + "\nauth.enabled=false — authentication is OFF (DEV ONLY)\n"
        + "Set auth.enabled: true in config.yaml for production.\n"
        + "=" * 60
    )

_security = HTTPBearer(auto_error=False)


# ── verify_identity ──────────────────────────────────────────────────────────
# Two branches at module import time so FastAPI sees a parameter-less function
# when auth is disabled. This eliminates ALL Header/HTTPBearer parameter
# inference complications (which was the root cause of the persistent 422).

if _AUTH_ENABLED:
    async def verify_identity(
        creds: Optional[HTTPAuthorizationCredentials] = Depends(_security),
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ) -> Identity:
        if x_api_key:
            ident = lookup_api_key(x_api_key)
            if ident is not None:
                return ident

        if creds is not None and creds.scheme.lower() == "bearer":
            try:
                claims = verify_jwt(creds.credentials)
            except AuthError as exc:
                raise HTTPException(status_code=401, detail=str(exc))
            return Identity(
                operator_id = claims["sub"],
                roles       = claims.get("roles", ["operator"]),
                auth_method = "jwt",
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (Authorization: Bearer <jwt> or X-API-Key)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def require_role(role: str):
        async def _dep(identity: Identity = Depends(verify_identity)) -> Identity:
            if not identity.has_role(role):
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{role}' required (have: {identity.roles})",
                )
            return identity
        return _dep

else:
    # Auth disabled — verify_identity takes NO parameters and just returns dev identity.
    # No HTTPBearer, no Header — nothing for FastAPI to body-coerce.
    async def verify_identity() -> Identity:
        return Identity(
            operator_id = _DEV_OPERATOR,
            roles       = ["admin"],
            auth_method = "disabled",
        )

    def require_role(role: str):
        # Same no-arg signature when disabled. Roles ignored.
        async def _dep() -> Identity:
            return Identity(
                operator_id = _DEV_OPERATOR,
                roles       = ["admin"],
                auth_method = "disabled",
            )
        return _dep


def issue_dev_token(operator_id: str, roles: list[str], ttl_seconds: int = 3600) -> str:
    return issue_jwt(operator_id, roles, ttl_seconds)


# Legacy compat: some callers reference AUTH_DISABLED directly
AUTH_DISABLED = not _AUTH_ENABLED
