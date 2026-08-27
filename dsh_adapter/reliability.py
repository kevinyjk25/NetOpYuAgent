"""Deterministic local-only reliability and DSH retirement checks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _request(socket_path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload) + "\n").encode())
        response = bytearray()
        while chunk := client.recv(65536):
            response.extend(chunk)
    return json.loads(response), (time.perf_counter() - started) * 1000


def _start_worker(python: str, root: Path, socket_path: Path, result_store: Path) -> subprocess.Popen[str]:
    environment = {
        **os.environ,
        "NETOPYU_DSH_BACKEND": "mock",
        "NETOPYU_DSH_TOOL_RESULT_STORE": str(result_store),
        # Prove that the explicit per-request False gate overrides ambient env.
        "NETOPYU_DSH_ALLOW_DESTRUCTIVE": "1",
        "NETOPYU_DSH_OTEL_ENABLED": "false",
    }
    process = subprocess.Popen(
        [python, "-m", "dsh_adapter.worker", "--socket", str(socket_path)],
        cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    for _ in range(100):
        if socket_path.is_socket():
            return process
        if process.poll() is not None:
            raise RuntimeError(process.stderr.read())
        time.sleep(0.02)
    process.terminate()
    process.wait(timeout=5)
    raise TimeoutError("local reliability worker did not become ready")


def _stop_worker(process: subprocess.Popen[str], socket_path: Path) -> None:
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)
    for _ in range(50):
        if not socket_path.exists():
            return
        time.sleep(0.02)
    raise RuntimeError("worker did not remove its Unix socket during shutdown")


def run_local_reliability(
    *, project_root: str, python_executable: str, request_count: int = 24,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Exercise load, malformed input, restart, policy gating, and hard retirement."""
    if not 4 <= request_count <= 500:
        raise ValueError("request_count must be between 4 and 500")
    if not 1 <= concurrency <= 32:
        raise ValueError("concurrency must be between 1 and 32")
    root = Path(project_root).expanduser().resolve()
    # Preserve virtualenv symlinks: resolving them would silently switch the
    # child to the base interpreter and lose the project's installed packages.
    python = os.path.abspath(os.path.expanduser(python_executable))
    with TemporaryDirectory(prefix="netopyu-reliability-") as directory:
        temp = Path(directory)
        socket_path = temp / "bridge.sock"
        result_store = temp / "results.sqlite"
        worker = _start_worker(python, root, socket_path, result_store)
        try:
            def invoke(index: int) -> tuple[dict[str, Any], float]:
                return _request(socket_path, {
                    "id": f"load-{index}", "correlation_id": f"retirement-{index}",
                    "command": "invoke", "profile": "lan", "tool": "list_devices",
                    "args": {}, "allow_destructive": False,
                })
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                load_results = list(executor.map(invoke, range(request_count)))
            latencies = sorted(duration for _, duration in load_results)
            load_ok = all(response.get("ok") is True for response, _ in load_results)

            denied, _ = _request(socket_path, {
                "id": "policy-denied", "command": "invoke", "profile": "lan",
                "tool": "restart_service", "args": {"service": "crm", "environment": "staging"},
                "allow_destructive": False,
            })
            allowed, _ = _request(socket_path, {
                "id": "policy-allowed", "command": "invoke", "profile": "lan",
                "tool": "restart_service", "args": {"service": "crm", "environment": "staging"},
                "allow_destructive": True,
            })

            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(5)
                client.connect(str(socket_path))
                client.sendall(b"not-json\n")
                malformed = json.loads(client.recv(65536))
            healthy_after_chaos, _ = _request(socket_path, {"id": "after-chaos", "command": "ping"})
        finally:
            _stop_worker(worker, socket_path)

        restarted = _start_worker(python, root, socket_path, result_store)
        try:
            healthy_after_restart, _ = _request(socket_path, {"id": "after-restart", "command": "ping"})
        finally:
            _stop_worker(restarted, socket_path)

    p95_index = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))
    retired_paths = [root / "main.py", root / "webui", root / "scripts" / "netopyu-legacy"]
    checks = {
        "load": load_ok,
        "load_p95_under_1s": latencies[p95_index] < 1000,
        "ambient_destructive_env_does_not_bypass_request_gate": denied.get("ok") is False,
        "explicit_local_simulation_authorization": allowed.get("ok") is True,
        "malformed_request_isolated": malformed.get("ok") is False and healthy_after_chaos.get("ok") is True,
        "restart_recovery": healthy_after_restart.get("ok") is True,
        "legacy_surfaces_removed": all(not path.exists() for path in retired_paths),
        "dsh_launcher_present": (root / "scripts" / "netopyu-dsh").is_file(),
    }
    return {
        "ok": all(checks.values()), "scope": "local-mock-only", "checks": checks,
        "load": {
            "requests": request_count, "concurrency": concurrency,
            "p95_ms": round(latencies[p95_index], 2), "max_ms": round(max(latencies), 2),
        },
        "real_network_actions": 0,
        "temporary_state_removed": True,
    }
