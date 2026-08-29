"""Isolated JSON-lines client for repository-external Provider qualification.

The command is deployment/CI configuration, never data from a Provider bundle.
It is executed without a shell and receives a minimal environment.  This module
adapts the external wire protocol to ``ProviderQualificationTarget``; it does
not grant Runtime execution authority or dynamically import Provider code.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .capabilities import (
    CapabilityContract,
    CapabilityKind,
    DataSensitivity,
    EffectSemantics,
)


EXTERNAL_QUALIFICATION_SCHEMA = "netopyu.io/provider-qualification-target/v1"
EXTERNAL_WIRE_SCHEMA = "netopyu.io/provider-qualification-wire/v1"
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class ExternalProviderProtocolError(RuntimeError):
    """The external Provider violated the qualification wire contract."""


class ExternalProviderOperationError(RuntimeError):
    """The external Provider returned a structured operation failure."""


class ExternalQualificationConfig(BaseModel):
    """Deployment-owned command configuration for an isolated Provider target."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    api_version: str = Field(
        default=EXTERNAL_QUALIFICATION_SCHEMA,
        alias="apiVersion",
    )
    command: tuple[str, ...]
    cwd: str
    pass_environment: tuple[str, ...] = ()
    timeout_seconds: float = 10.0
    max_response_bytes: int = 1_048_576

    @model_validator(mode="after")
    def validate_target(self) -> "ExternalQualificationConfig":
        if self.api_version != EXTERNAL_QUALIFICATION_SCHEMA:
            raise ValueError("unsupported external qualification target schema")
        if not self.command or any(not item or "\x00" in item for item in self.command):
            raise ValueError("external qualification command must contain non-empty argv")
        executable = Path(self.command[0]).expanduser()
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("external qualification executable must be an absolute file")
        working_directory = Path(self.cwd).expanduser()
        if not working_directory.is_absolute() or not working_directory.is_dir():
            raise ValueError("external qualification cwd must be an absolute directory")
        if not 0.1 <= self.timeout_seconds <= 120:
            raise ValueError("external qualification timeout must be between 0.1 and 120 seconds")
        if not 1_024 <= self.max_response_bytes <= 8_388_608:
            raise ValueError("external response limit must be between 1 KiB and 8 MiB")
        if len(set(self.pass_environment)) != len(self.pass_environment):
            raise ValueError("pass_environment contains duplicate names")
        if any(not _ENV_NAME.fullmatch(name) for name in self.pass_environment):
            raise ValueError("pass_environment contains an invalid variable name")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> "ExternalQualificationConfig":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _capability(value: Any) -> CapabilityContract:
    if not isinstance(value, dict):
        raise ExternalProviderProtocolError("describe result must be an object")
    try:
        return CapabilityContract(
            tool_name=str(value["tool_name"]),
            capability_id=str(value["capability_id"]),
            capability_version=str(value["capability_version"]),
            domain=str(value["domain"]),
            kind=CapabilityKind(str(value["kind"])),
            action_type=str(value["action_type"]),
            effect_semantics=EffectSemantics(str(value["effect_semantics"])),
            provider_role=str(value["provider_role"]),
            provider_identity=str(value["provider_identity"]),
            provider_kind=str(value["provider_kind"]),
            input_schema_digest=str(value["input_schema_digest"]),
            output_schema_digest=str(value["output_schema_digest"]),
            sensitivity=DataSensitivity(str(value["sensitivity"])),
            required_roles=tuple(str(item) for item in value["required_roles"]),
            scope_fields=tuple(str(item) for item in value["scope_fields"]),
            freshness_limit_seconds=int(value["freshness_limit_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ExternalProviderProtocolError(
            "external Provider returned an invalid CapabilityContract"
        ) from error


class ExternalQualificationTarget:
    """Persistent, bounded JSONL adapter implementing ProviderQualificationTarget."""

    def __init__(self, config: ExternalQualificationConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = bytearray()
        self._capabilities: dict[str, CapabilityContract] = {}

    async def __aenter__(self) -> "ExternalQualificationTarget":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _environment(self) -> dict[str, str]:
        environment = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        for name in self.config.pass_environment:
            if name not in os.environ:
                raise ExternalProviderProtocolError(
                    f"required external Provider environment variable {name!r} is missing"
                )
            environment[name] = os.environ[name]
        return environment

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        self._stderr_tail.clear()
        self._process = await asyncio.create_subprocess_exec(
            *self.config.command,
            cwd=self.config.cwd,
            env=self._environment(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.config.max_response_bytes + 1,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(4_096)
            if not chunk:
                return
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > 16_384:
                del self._stderr_tail[:-16_384]

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self._stderr_task is not None:
            try:
                await self._stderr_task
            except asyncio.CancelledError:  # pragma: no cover - defensive cleanup
                pass
            self._stderr_task = None

    async def _request(self, action: str, payload: dict[str, Any]) -> Any:
        async with self._lock:
            await self.start()
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise ExternalProviderProtocolError("external Provider process has no JSONL pipes")
            request_id = str(uuid.uuid4())
            encoded = json.dumps({
                "apiVersion": EXTERNAL_WIRE_SCHEMA,
                "requestId": request_id,
                "action": action,
                "payload": payload,
            }, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            if len(encoded) > self.config.max_response_bytes:
                raise ExternalProviderProtocolError("external Provider request exceeds size limit")
            try:
                process.stdin.write(encoded)
                await asyncio.wait_for(
                    process.stdin.drain(), timeout=self.config.timeout_seconds,
                )
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=self.config.timeout_seconds,
                )
            except (BrokenPipeError, ConnectionError, asyncio.TimeoutError) as error:
                detail = self._stderr_tail.decode("utf-8", errors="replace")[-1_024:]
                raise ExternalProviderProtocolError(
                    f"external Provider transport failed for {action}: {type(error).__name__}; "
                    f"stderr_tail={detail!r}"
                ) from error
            if not line:
                raise ExternalProviderProtocolError(
                    f"external Provider exited during {action} with code {process.returncode}"
                )
            if len(line) > self.config.max_response_bytes:
                raise ExternalProviderProtocolError("external Provider response exceeds size limit")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExternalProviderProtocolError(
                    "external Provider returned non-JSON output"
                ) from error
            if not isinstance(response, dict):
                raise ExternalProviderProtocolError("external Provider response must be an object")
            if response.get("apiVersion") != EXTERNAL_WIRE_SCHEMA:
                raise ExternalProviderProtocolError("external Provider response schema mismatch")
            if response.get("requestId") != request_id:
                raise ExternalProviderProtocolError("external Provider response request id mismatch")
            if response.get("ok") is not True:
                error = response.get("error")
                if not isinstance(error, dict):
                    raise ExternalProviderProtocolError(
                        "external Provider failure has no structured error"
                    )
                raise ExternalProviderOperationError(
                    f"{error.get('code') or 'provider_error'}: "
                    f"{error.get('message') or 'external Provider operation failed'}"
                )
            if "result" not in response:
                raise ExternalProviderProtocolError("external Provider success has no result")
            return response["result"]

    async def discover_capability(self, tool_name: str) -> CapabilityContract:
        contract = _capability(await self._request("describe", {"tool_name": tool_name}))
        if contract.tool_name != tool_name:
            raise ExternalProviderProtocolError("external Provider described another tool")
        self._capabilities[tool_name] = contract
        return contract

    def describe_capability(self, tool_name: str) -> CapabilityContract:
        try:
            return self._capabilities[tool_name]
        except KeyError as error:
            raise ExternalProviderProtocolError(
                "discover_capability must run before fixed qualification"
            ) from error

    async def reset(self) -> str:
        value = await self._request("reset", {})
        if not isinstance(value, str) or not value:
            raise ExternalProviderProtocolError("reset must return a non-empty snapshot digest")
        return value

    async def snapshot_digest(self) -> str:
        value = await self._request("snapshot", {})
        if not isinstance(value, str) or not value:
            raise ExternalProviderProtocolError("snapshot must return a non-empty digest")
        return value

    async def apply(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        operation_id: str,
        sequence: int,
        fault: str | None = None,
    ) -> dict[str, Any]:
        value = await self._request("apply", {
            "tool_name": tool_name,
            "arguments": arguments,
            "operation_id": operation_id,
            "sequence": sequence,
            "fault": fault,
        })
        if not isinstance(value, dict):
            raise ExternalProviderProtocolError("apply result must be an object")
        return value

    async def reconcile(self, operation_id: str) -> dict[str, Any]:
        value = await self._request("reconcile", {"operation_id": operation_id})
        if not isinstance(value, dict):
            raise ExternalProviderProtocolError("reconcile result must be an object")
        return value

    async def compensate(
        self,
        operation_id: str,
        *,
        fault: str | None = None,
    ) -> dict[str, Any]:
        value = await self._request("compensate", {
            "operation_id": operation_id,
            "fault": fault,
        })
        if not isinstance(value, dict):
            raise ExternalProviderProtocolError("compensate result must be an object")
        return value

    async def restart(self) -> None:
        await self.close()
        await self.start()

    async def escalation_state(self, operation_id: str) -> str:
        value = await self._request("escalation", {"operation_id": operation_id})
        if not isinstance(value, str):
            raise ExternalProviderProtocolError("escalation result must be a string")
        return value


async def qualify_external_provider(
    config: ExternalQualificationConfig,
    manifest: Any,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    environment: str,
    now: Any = None,
) -> Any:
    """Discover then run the standard fixed suite against an external process."""
    from .provider_qualification import run_provider_qualification

    async with ExternalQualificationTarget(config) as target:
        await target.discover_capability(tool_name)
        return await run_provider_qualification(
            target,
            manifest,
            tool_name=tool_name,
            arguments=arguments,
            environment=environment,
            now=now,
        )


__all__ = [
    "EXTERNAL_QUALIFICATION_SCHEMA",
    "EXTERNAL_WIRE_SCHEMA",
    "ExternalProviderOperationError",
    "ExternalProviderProtocolError",
    "ExternalQualificationConfig",
    "ExternalQualificationTarget",
    "qualify_external_provider",
]
