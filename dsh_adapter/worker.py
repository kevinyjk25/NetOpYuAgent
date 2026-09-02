"""Persistent, owner-only Unix-socket worker shared by harness adapters."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any

from runtime.tracing import configure as configure_tracing
from runtime.tracing import start_span

from .a2a_provider import delegate_a2a, discover_peers
from .agentized_authoring import (
    authoring_template, authoring_trace, capture_authoring, submit_authoring,
)
from .backend import resolve_backend_mode
from .bridge import (
    _build_manifest,
    approve_network_plan,
    audit_network_plan,
    backend_report,
    execute_network_plan,
    inspect_network_plan,
    invoke_tool,
    observe_network_workflow,
    prepare_network_plan,
    reject_network_plan,
    start_network_workflow,
)
from .scoped_services import recall_memory, search_capabilities
from .skills import build_skill_manifest
from l1_runtime.service import (
    close_decision,
    decide_shadow,
    decision_metrics,
    observe_decision,
    recent_decisions,
)


MAX_REQUEST_BYTES = 1_048_576
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _configure_tracing() -> bool:
    """Enable the existing optional OTel adapter for the shared Worker."""
    enabled_value = os.getenv(
        "NETOPYU_DSH_OTEL_ENABLED",
        os.getenv("OTEL_TRACING_ENABLED", "false"),
    )
    try:
        sample_ratio = float(os.getenv("OTEL_SAMPLE_RATIO", "1.0"))
    except ValueError:
        sample_ratio = 1.0
    return configure_tracing(
        enabled=enabled_value.strip().lower() in _TRUE_VALUES,
        service_name=os.getenv("OTEL_SERVICE_NAME", "netopyu-harness-worker"),
        service_version=os.getenv("OTEL_SERVICE_VERSION", "0.1.0"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
        sample_ratio=sample_ratio,
    )


async def dispatch(request: dict[str, Any]) -> Any:
    command = str(request.get("command", ""))
    profile = str(request.get("profile", "lan"))
    arguments = request.get("args") or {}
    if not isinstance(arguments, dict):
        raise TypeError("bridge arguments must be a JSON object")
    if command == "ping":
        return {"ok": True, "worker_pid": os.getpid()}
    if command == "manifest":
        return await _build_manifest(
            profile,
            include_destructive=bool(request.get("include_destructive")),
        )
    if command == "invoke":
        result = await invoke_tool(
            profile,
            str(request.get("tool", "")),
            arguments,
            allow_destructive=bool(request.get("allow_destructive")),
            access_context=(
                request.get("access_context")
                if isinstance(request.get("access_context"), dict) else None
            ),
            session_id=(str(request.get("session_id")) if request.get("session_id") else None),
            harness=str(request.get("harness") or "local"),
        )
        return {"ok": True, "result": result}
    if command == "runtime-prepare":
        return await prepare_network_plan(
            profile, str(request.get("tool", "")), arguments,
            session_id=(str(request.get("session_id")) if request.get("session_id") else None),
            l0_skill_id=(
                str(request.get("l0_skill_id")) if request.get("l0_skill_id") else None
            ),
            subject_context=(
                request.get("subject_context")
                if isinstance(request.get("subject_context"), dict) else None
            ),
            harness=str(request.get("harness") or "local"),
            l1_decision_envelope=(
                request.get("l1_decision_envelope")
                if isinstance(request.get("l1_decision_envelope"), dict) else None
            ),
            l1_route_context=(
                request.get("l1_route_context")
                if isinstance(request.get("l1_route_context"), dict) else None
            ),
        )
    if command == "runtime-approve":
        return approve_network_plan(arguments)
    if command == "runtime-execute":
        return await execute_network_plan(
            arguments, allow_destructive=bool(request.get("allow_destructive")),
        )
    if command == "runtime-inspect":
        return inspect_network_plan(str(arguments.get("plan_id", "")))
    if command == "runtime-audit":
        return audit_network_plan(str(arguments.get("plan_id", "")))
    if command == "runtime-reject":
        return reject_network_plan(arguments)
    if command == "workflow-start":
        return start_network_workflow(profile, arguments)
    if command == "workflow-observe":
        return await observe_network_workflow(profile, arguments)
    if command == "backend":
        return await backend_report(profile)
    if command == "memory-recall":
        return await recall_memory(**arguments)
    if command == "capability-search":
        return await search_capabilities(profile_id=profile, **arguments)
    if command == "a2a-peers":
        return await discover_peers(**arguments)
    if command == "a2a-delegate":
        return await delegate_a2a(**arguments)
    if command == "skill-manifest":
        return build_skill_manifest(profile, resolve_backend_mode())
    if command == "agent-authoring-template":
        return authoring_template()
    if command == "agent-authoring-capture":
        return capture_authoring(arguments)
    if command == "agent-authoring-submit":
        return submit_authoring(arguments)
    if command == "agent-authoring-trace":
        return authoring_trace(str(arguments.get("attempt_id") or ""))
    if command == "l1-decision-shadow":
        tool_declarations = arguments.get("tool_declarations")
        if not isinstance(tool_declarations, list):
            raise TypeError("L1 decision tool_declarations must be an array")
        return await decide_shadow(
            profile=profile,
            session_id=str(arguments.get("session_id") or ""),
            harness=str(arguments.get("harness") or "dsh"),
            prompt=str(arguments.get("user_request") or ""),
            tool_declarations=tool_declarations,
            model=str(arguments.get("model") or ""),
        )
    if command == "l1-decision-recent":
        return recent_decisions(
            limit=int(arguments.get("limit", 20)),
            session_id=(
                str(arguments["session_id"])
                if arguments.get("session_id") is not None else None
            ),
        )
    if command == "l1-decision-observe":
        observed_arguments = arguments.get("observed_arguments")
        if not isinstance(observed_arguments, dict):
            raise TypeError("L1 observed_arguments must be an object")
        return observe_decision(
            decision_id=str(arguments.get("decision_id") or ""),
            session_id=str(arguments.get("session_id") or ""),
            observed_kind=str(arguments.get("observed_kind") or ""),
            observed_target=str(arguments.get("observed_target") or ""),
            observed_arguments=observed_arguments,
        )
    if command == "l1-decision-close":
        return close_decision(
            decision_id=str(arguments.get("decision_id") or ""),
            session_id=str(arguments.get("session_id") or ""),
            reason=str(arguments.get("reason") or ""),
        )
    if command == "l1-decision-metrics":
        return decision_metrics(limit=int(arguments.get("limit", 500)))
    raise ValueError(f"unsupported persistent bridge command {command!r}")


async def _handle(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    semaphore: asyncio.Semaphore,
) -> None:
    started = time.perf_counter()
    request: dict[str, Any] = {}
    request_id: Any = None
    error_type: str | None = None
    try:
        raw = await reader.readline()
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("bridge request exceeds 1 MiB")
        request = json.loads(raw or b"{}")
        if not isinstance(request, dict):
            raise TypeError("bridge request must be a JSON object")
        request_id = request.get("id")
        async with semaphore:
            with start_span("netopyu.dsh.bridge", **{
                "bridge.command": str(request.get("command", "")),
                "bridge.profile": str(request.get("profile", "lan")),
                "bridge.tool": str(request.get("tool") or ""),
                "bridge.correlation_id": str(request.get("correlation_id") or request_id or ""),
            }):
                payload = await dispatch(request)
        response = {"id": request_id, "ok": True, "payload": payload}
    except Exception as error:
        error_type = type(error).__name__
        response = {
            "id": request_id,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }
    print(json.dumps({
        "event": "netopyu_bridge_request",
        "request_id": request_id,
        "correlation_id": request.get("correlation_id") or request_id,
        "command": request.get("command"),
        "profile": request.get("profile", "lan"),
        "tool": request.get("tool"),
        "network_plan_id": (
            (request.get("args") or {}).get("plan_id")
            if request.get("command") in {"runtime-execute", "runtime-inspect"}
            and isinstance(request.get("args") or {}, dict)
            else None
        ),
        "ok": response["ok"],
        "error_type": error_type,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }, ensure_ascii=False), flush=True)
    try:
        writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
        await writer.drain()
    except (BrokenPipeError, ConnectionError):
        # A Web process may be cancelled or restarted while a comparatively
        # slow backend request is still completing.  The result is no longer
        # observable by that client, but it must not surface as an unhandled
        # asyncio callback exception or destabilize the shared Worker.
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionError):
            pass


def _remove_stale_socket(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_socket():
        # Never unlink a socket owned by a live Worker.  Without this probe a
        # manually started second Worker could silently orphan the first one,
        # producing split-brain manifests and sparse/corrupted shared logs.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.25)
            try:
                client.connect(str(path))
            except (ConnectionRefusedError, FileNotFoundError, TimeoutError):
                pass
            else:
                raise RuntimeError(f"bridge worker socket is already active: {path}")
        path.unlink()
        return
    raise RuntimeError(f"refusing to replace non-socket worker path: {path}")


def _claim_worker_lock(path: Path) -> int:
    """Hold an owner-only process lock for one Worker socket namespace."""

    lock_path = path.with_name(path.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(
            f"another bridge worker already owns lock: {lock_path}"
        ) from error
    return descriptor


async def serve(path: Path) -> None:
    _configure_tracing()
    from network_runtime.engine import NetworkRuntime, default_journal_path
    from network_runtime.journal import NetworkJournal

    # This is the only safe point to reconcile plans interrupted by a prior
    # worker process. New request connections must never perform recovery.
    with NetworkJournal(default_journal_path(), recover_crashed=True):
        pass
    # Reconcile uncertain device outcomes using verifier reads only. No write
    # command is replayed after a crash.
    await NetworkRuntime().recover_inflight()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_descriptor = _claim_worker_lock(path)
    try:
        _remove_stale_socket(path)
    except Exception:
        os.close(lock_descriptor)
        raise
    try:
        concurrency = int(
            os.getenv("NETOPYU_WORKER_CONCURRENCY")
            or os.getenv("NETOPYU_DSH_WORKER_CONCURRENCY", "8")
        )
    except ValueError as error:
        raise ValueError("NETOPYU_WORKER_CONCURRENCY must be an integer") from error
    if not 1 <= concurrency <= 64:
        raise ValueError("NETOPYU_WORKER_CONCURRENCY must be between 1 and 64")
    semaphore = asyncio.Semaphore(concurrency)
    # Apply owner-only permissions at bind time. A post-bind chmod alone has a
    # race in which another local user can observe/connect to a 0755 socket.
    previous_umask = os.umask(0o177)
    try:
        server = await asyncio.start_unix_server(
            lambda reader, writer: _handle(reader, writer, semaphore),
            path=str(path),
        )
    finally:
        os.umask(previous_umask)
    os.chmod(path, 0o600)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        async with server:
            await stop.wait()
    finally:
        server.close()
        await server.wait_closed()
        if path.is_socket():
            path.unlink()
        os.close(lock_descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(serve(args.socket))


if __name__ == "__main__":
    main()
