"""Explicit host-owned local model routing; not a metering or identity oracle.

Without configuration the caller's legacy endpoint is unchanged. A configured
route is a capability supplied by the host, never by model output or a Skill.
The gateway must enforce budgets itself; a matching advertised digest alone
does not prove model weights, token accounting, or semantic correctness.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

SCHEMA = "netopyu.local-model-routes/v1"
ROUTES_ENV = "NETOPYU_LOCAL_MODEL_ROUTES"
ARM_ENV = "NETOPYU_LOCAL_MODEL_ARM_ID"
_MAX_CONFIG_BYTES = 65536
_ROLES = frozenset({"compile", "runtime"})
_BOUND_ROUTES = ContextVar("netopyu_host_model_routes", default=None)


@dataclass(frozen=True)
class ModelEndpoint:
    base_url: str
    role: str
    model: str
    model_digest: str | None = None
    arm_id: str | None = None

    def check_model_digest(self, advertised_digest):
        """Compare the configured pin before POST; do not attest real weights."""
        if self.model_digest is not None and advertised_digest != self.model_digest:
            raise ValueError("local model route digest mismatch")


def _object(value, keys, description):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"invalid local model {description}")
    return value


def _text(value):
    return isinstance(value, str) and bool(value) and value == value.strip() and not any(ord(c) < 32 for c in value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate local model route configuration key")
        result[key] = value
    return result


def _read_config(filename):
    if not _text(filename) or not Path(filename).is_absolute():
        raise ValueError("local model routes require an absolute owner-only file")
    # Do not follow a final-component symlink, or block on a FIFO/device.
    descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
                or not 0 < info.st_size <= _MAX_CONFIG_BYTES):
            raise ValueError("local model routes require a bounded owner-only regular file")
        chunks, total = [], 0
        while total <= _MAX_CONFIG_BYTES:
            chunk = os.read(descriptor, min(8192, _MAX_CONFIG_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_CONFIG_BYTES:
            raise ValueError("local model routes file exceeds its bound")
        return json.loads(b"".join(chunks), object_pairs_hook=_unique_object)
    finally:
        os.close(descriptor)


def _base_url(value):
    if not _text(value):
        raise ValueError("local model route URL required")
    try:
        parsed = urlsplit(value)
        address = ipaddress.ip_address(parsed.hostname or "")
        valid = (parsed.scheme == "http" and address.is_loopback
                 and parsed.port is not None and 0 < parsed.port <= 65535
                 and parsed.username is None and parsed.password is None
                 and "?" not in value and "#" not in value
                 and re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*/?", parsed.path) is not None)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("local model route requires an unambiguous numeric loopback HTTP URL")
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"http://{host}:{parsed.port}{parsed.path.rstrip('/')}"


def _load_routes(path, arm_id):
    config = _object(_read_config(os.fspath(path)),
                     {"schema", "model", "model_digest", "arm_id", "routes"}, "route configuration")
    if (config["schema"] != SCHEMA or not _text(config["model"])
            or not _text(config["model_digest"]) or not _text(config["arm_id"])
            or config["arm_id"] != arm_id):
        raise ValueError("local model route schema/model/arm binding mismatch")
    routes = _object(config["routes"], _ROLES, "role set")
    urls = {name: _base_url(_object(route, {"base_url"}, "role route")["base_url"])
            for name, route in routes.items()}
    if len(set(urls.values())) != len(_ROLES):
        raise ValueError("compile and runtime require distinct local model routes")
    return tuple(ModelEndpoint(urls[role], role, config["model"], config["model_digest"], config["arm_id"])
                 for role in sorted(_ROLES))


@contextmanager
def bind_model_routes(path, arm_id):
    """Freeze validated routes in a host-owned thread/task scope, without env writes.

    Enter this scope INSIDE the tool handler that invokes compilation/reasoning;
    ContextVars do not automatically propagate to newly spawned threads. Keep
    the scope around the entire operation, including any late completion. A
    closed broker then fails closed: environment changes cannot select another
    arm or restore the legacy endpoint. Never accept path/arm_id from a Skill
    or model request. Nested scopes restore the prior binding even on errors.
    """
    endpoints = _load_routes(path, arm_id)
    token = _BOUND_ROUTES.set(endpoints)
    try:
        yield
    finally:
        _BOUND_ROUTES.reset(token)


def resolve_model_endpoint(role, *, model, default_endpoint):
    """Resolve a fixed call-site role; explicit invalid configuration never falls back.

    A frozen host context wins over environment configuration. Otherwise,
    NETOPYU_LOCAL_MODEL_ROUTES names a 0400/0600 JSON file with exactly schema,
    model, model_digest, arm_id, and routes. Both compile/runtime routes contain
    only base_url and must differ. NETOPYU_LOCAL_MODEL_ARM_ID binds this process
    to that file's arm. All configured URLs use numeric loopback addresses.
    """
    if role not in _ROLES:
        raise ValueError("host model role must be compile or runtime")
    endpoints = _BOUND_ROUTES.get()
    if endpoints is None:
        if ROUTES_ENV not in os.environ:
            return ModelEndpoint(default_endpoint, role, model)
        endpoints = _load_routes(os.environ[ROUTES_ENV], os.environ.get(ARM_ENV))
    endpoint = next(item for item in endpoints if item.role == role)
    if endpoint.model != model:
        raise ValueError("local model route model binding mismatch")
    return endpoint
