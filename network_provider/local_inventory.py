"""Auditable local experiment provider: one fixed file, read-only, no CLI/network."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


TOOL = {
    "name": "read_inventory_device",
    "description": "Read one device from a local inventory snapshot, not live network telemetry.",
    "annotations": {"readOnlyHint": True},
    "inputSchema": {"type": "object", "properties": {"device_id": {"type": "string"}},
                    "required": ["device_id"], "additionalProperties": False},
    "outputSchema": {"type": "object", "properties": {
        "device_id": {"type": "string"}, "site": {"type": "string"}, "status": {"type": "string"},
    }, "required": ["device_id", "site", "status"], "additionalProperties": False},
}
ACCESS = {"requiredScopes": ["inventory:read"], "dataClassification": "internal"}
CAPABILITY = "local.inventory.device.read"


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate inventory JSON key")
        result[key] = value
    return result


class LocalInventoryReader:
    """The host fixes the dataset path; user arguments never affect file selection."""

    def __init__(self, path: Path):
        self.path = Path(path).absolute()

    def observe(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if set(arguments) != {"device_id"} or not isinstance(arguments["device_id"], str):
            raise ValueError("inventory requires only a string device_id")
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("inventory must be a regular file")
            raw = source.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("inventory exceeds 1 MiB")
        rows = json.loads(raw, object_pairs_hook=_unique)
        json.dumps(rows, allow_nan=False)
        if not isinstance(rows, dict):
            raise ValueError("inventory must be an object keyed by device ID")
        device_id = arguments["device_id"]
        row = rows.get(device_id)
        if not isinstance(row, dict):
            raise ValueError("inventory device not found")
        # Only the declared fields can leave the provider, even if the file has secrets.
        payload = {"device_id": device_id, "site": row["site"], "status": row["status"]}
        if any(not isinstance(value, str) for value in payload.values()):
            raise ValueError("inventory output fields must be strings")
        return payload, {
            "sourceSha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "sourceKind": "local_inventory_snapshot_not_live_telemetry",
        }


def adapter_declaration() -> dict[str, Any]:
    implementation = Path(__file__).read_text(encoding="utf-8")
    return {
        "capability": CAPABILITY, "tool": TOOL["name"], "effect": "read_only", "access": ACCESS,
        "implementation": {
            "path": "network_provider/local_inventory.py", "text": implementation,
            "sha256": "sha256:" + hashlib.sha256(implementation.encode()).hexdigest(),
        },
        "boundary": "Host-owned local file; no client-selected paths; snapshot contents are not live device state.",
    }
