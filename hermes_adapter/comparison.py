"""Local DSH-vs-Hermes adapter parity check for Network Runtime invariants."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from dsh_adapter.worker import dispatch
from network_runtime import NetworkRuntime
from network_runtime.l0_skills import REGISTRY as L0_SKILLS

from .plugin import HermesAdapterConfig, NetOpYuHermesAdapter


class InProcessWorkerClient:
    """Exercise the exact Worker dispatcher without depending on Hermes itself."""

    def request(
        self,
        command: str,
        *,
        profile: str = "lan",
        tool: str = "",
        args: dict[str, Any] | None = None,
        **fields: Any,
    ) -> Any:
        request = {
            "command": command,
            "profile": profile,
            "args": args or {},
            **({"tool": tool} if tool else {}),
            **fields,
        }
        return asyncio.run(dispatch(request))

    def ping(self) -> dict[str, Any]:
        return self.request("ping")


class FakeHermesContext:
    """Minimal official PluginContext surface used for compatibility tests."""

    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.commands: dict[str, Any] = {}
        self.skills: dict[str, str] = {}
        self.hooks: dict[str, list[Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[str(definition["name"])] = definition

    def register_command(self, name: str, handler: Any, description: str = "") -> None:
        self.commands[name] = {"handler": handler, "description": description}

    def register_skill(self, name: str, path: str) -> None:
        self.skills[name] = path

    def register_hook(self, name: str, handler: Any) -> None:
        self.hooks.setdefault(name, []).append(handler)


def _stable_plan(plan: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "profile", "tool_name", "tool_version", "action_type", "arguments",
        "argument_provenance", "targets", "risk_level", "risk_reasons",
        "verification_contract", "rollback_contract", "l0_skill_id",
        "l0_skill_version", "l0_contract_hash", "intent_spec", "intent_hash",
        "step_contract", "workflow_template_hash",
        "approval_mode", "approval_policy_id", "approval_policy_version",
        "approval_policy_hash",
    )
    return {key: plan.get(key) for key in keys}


def _runtime_execute(runtime: NetworkRuntime, prepared: dict[str, Any], actor: str) -> dict[str, Any]:
    plan = prepared["plan"]
    return asyncio.run(runtime.execute(
        plan_id=plan["plan_id"],
        plan_hash=plan["plan_hash"],
        execution_nonce=prepared["execution_nonce"],
        approval_request_id=f"comparison:{actor}",
        approval_actor=actor,
        allow_destructive=True,
    )).to_dict()


def run_comparison() -> dict[str, Any]:
    previous = {
        key: os.environ.get(key)
        for key in ("NETOPYU_BACKEND", "NETOPYU_NETWORK_RUNTIME_STORE", "NETOPYU_TOOL_RESULT_STORE")
    }
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ["NETOPYU_BACKEND"] = "mock"
            os.environ["NETOPYU_TOOL_RESULT_STORE"] = str(root / "tool-results.sqlite")
            arguments = {"service": "crm", "environment": "staging"}
            l0 = L0_SKILLS.for_tool("lan", "restart_service")
            assert l0 is not None

            os.environ["NETOPYU_NETWORK_RUNTIME_STORE"] = str(root / "dsh.sqlite")
            dsh_runtime = NetworkRuntime()
            started = time.perf_counter()
            dsh_prepared = asyncio.run(dsh_runtime.prepare(
                "lan", "restart_service", arguments,
                session_id="comparison-dsh", l0_skill_id=l0.skill_id,
            ))
            dsh_outcome = _runtime_execute(dsh_runtime, dsh_prepared, "dsh-comparison")
            dsh_ms = round((time.perf_counter() - started) * 1000, 2)

            os.environ["NETOPYU_NETWORK_RUNTIME_STORE"] = str(root / "hermes.sqlite")
            config = HermesAdapterConfig(
                profile="lan",
                socket_path=root / "unused.sock",
                include_destructive=True,
                operator_id="hermes-comparison",
                own_agent_id="hermes-lan",
                peer_urls=(),
                timeout_seconds=120,
            )
            adapter = NetOpYuHermesAdapter(InProcessWorkerClient(), config)
            context = FakeHermesContext()
            adapter.register(context)
            started = time.perf_counter()
            prepared_text = context.tools["restart_service"]["handler"](
                arguments, task_id="comparison-hermes",
            )
            hermes_prepared = json.loads(prepared_text)
            command = hermes_prepared["approval"]["command"]
            command_name, raw = command[1:].split(" ", 1)
            hermes_outcome = json.loads(context.commands[command_name]["handler"](raw))
            hermes_ms = round((time.perf_counter() - started) * 1000, 2)

            dsh_plan = dsh_prepared["plan"]
            hermes_plan = hermes_prepared["plan"]
            dsh_audit = dsh_runtime.audit(dsh_plan["plan_id"])
            hermes_audit = NetworkRuntime().audit(hermes_plan["plan_id"])
            duplicate = json.loads(context.commands[command_name]["handler"](raw))
            invariant_match = _stable_plan(dsh_plan) == _stable_plan(hermes_plan)
            return {
                "ok": bool(
                    invariant_match
                    and dsh_outcome.get("state") == "verified_success"
                    and hermes_outcome.get("state") == "verified_success"
                    and dsh_audit.get("ok") is True
                    and hermes_audit.get("ok") is True
                    and duplicate.get("ok") is False
                ),
                "runtime_invariants_equal": invariant_match,
                "dsh": {
                    "state": dsh_outcome.get("state"),
                    "audit_valid": dsh_audit.get("ok"),
                    "elapsed_ms": dsh_ms,
                    "approval_actor": "dsh-comparison",
                },
                "hermes": {
                    "state": hermes_outcome.get("state"),
                    "audit_valid": hermes_audit.get("ok"),
                    "elapsed_ms": hermes_ms,
                    "approval_actor": "hermes-comparison",
                    "duplicate_blocked": duplicate.get("ok") is False,
                    "nonce_exposed_to_model": "execution_nonce" in hermes_prepared,
                },
                "expected_adapter_differences": [
                    "plan ids, timestamps, requester identities/digests, plan hashes, approval ids and approval actors",
                    "Hermes uses an explicit user-only slash command instead of a DSH approval card",
                    "Hermes pending authorization is process-local and is lost safely on restart",
                ],
            }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = run_comparison()
    print(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
