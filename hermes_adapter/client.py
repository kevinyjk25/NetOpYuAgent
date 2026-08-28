"""Small synchronous client for the owner-only NetOpYu harness Worker."""

from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path
from typing import Any


MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class WorkerProtocolError(RuntimeError):
    """Raised when the Worker is unavailable or violates its wire contract."""


class HermesWorkerClient:
    """Use one short-lived Unix-socket connection per Hermes tool call.

    The Worker owns all Python dependencies and Network Runtime state.  The
    Hermes process only sees JSON-safe declarations and results.
    """

    def __init__(self, socket_path: str | Path, *, timeout_seconds: float = 120.0) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive")

    def request(
        self,
        command: str,
        *,
        profile: str = "lan",
        tool: str = "",
        args: dict[str, Any] | None = None,
        **fields: Any,
    ) -> Any:
        request_id = str(uuid.uuid4())
        request = {
            "id": request_id,
            "command": command,
            "profile": profile,
            "args": args or {},
            **({"tool": tool} if tool else {}),
            **fields,
        }
        payload = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.connect(str(self.socket_path))
                connection.sendall(payload)
                chunks: list[bytes] = []
                size = 0
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    newline = chunk.find(b"\n")
                    if newline >= 0:
                        chunk = chunk[:newline]
                        chunks.append(chunk)
                        size += len(chunk)
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise WorkerProtocolError("Worker response exceeds 16 MiB")
        except (FileNotFoundError, ConnectionError, OSError, TimeoutError) as error:
            raise WorkerProtocolError(
                f"NetOpYu Worker unavailable at {self.socket_path}: {type(error).__name__}"
            ) from error
        if not chunks:
            raise WorkerProtocolError("NetOpYu Worker closed without a response")
        try:
            response = json.loads(b"".join(chunks))
        except json.JSONDecodeError as error:
            raise WorkerProtocolError("NetOpYu Worker returned invalid JSON") from error
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise WorkerProtocolError("NetOpYu Worker response id mismatch")
        if response.get("ok") is not True:
            raise WorkerProtocolError(str(response.get("error") or "NetOpYu Worker request failed"))
        return response.get("payload")

    def ping(self) -> dict[str, Any]:
        payload = self.request("ping")
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise WorkerProtocolError("NetOpYu Worker ping failed")
        return payload
