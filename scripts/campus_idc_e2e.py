#!/usr/bin/env python3
"""Exercise the reviewed LAN L1 + LAN/DC L0 flow on the real local lab."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / "labs" / "p075-a-campus-idc" / "lab.yaml"
DEFAULT_CONFIG = ROOT / "config.campus-idc-lab.yaml"


def _message(event: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(event["message"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "A2A peer did not return a completed message: "
            + json.dumps(event, ensure_ascii=False)
        ) from error


async def reset_baseline(
    *, manifest_path: Path = DEFAULT_MANIFEST, config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    os.environ["NETOPYU_DSH_BACKEND"] = "pragmatic"
    os.environ["NETOPYU_CONFIG_PATH"] = str(config_path)
    from network_lab import ContainerlabProvider, load_manifest

    provider = ContainerlabProvider(load_manifest(manifest_path))
    status = await provider.topology_status()
    if not status["ok"]:
        raise RuntimeError("campus/IDC topology is not fully running")
    await provider.set_user_admission("bob", admitted=True)
    await provider.set_application_access("bob", "crm", allowed=True)
    await provider.set_user_admission("erin", admitted=False)
    await provider.set_application_access("erin", "crm", allowed=False)
    return {
        "ok": True,
        "topology": provider.manifest.name,
        "lan_admitted": await provider.user_admitted("erin"),
        "crm_blocked": await provider.application_access_blocked("erin", "crm"),
        "erin_http_ok": (await provider.application_probe("erin", "crm"))["ok"],
        "bob_http_ok": (await provider.application_probe("bob", "crm"))["ok"],
    }


async def run(
    *, leave_provisioned: bool, runtime_path: Path, peer_state: Path,
    peer_url: str | None = None, exercise_rollback: bool = False,
    manifest_path: Path = DEFAULT_MANIFEST, config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if leave_provisioned and exercise_rollback:
        raise ValueError("--leave-provisioned and --exercise-rollback are mutually exclusive")
    os.environ["NETOPYU_DSH_BACKEND"] = "pragmatic"
    os.environ["NETOPYU_CONFIG_PATH"] = str(config_path)
    os.environ["NETOPYU_DSH_NETWORK_RUNTIME_STORE"] = str(runtime_path)
    os.environ["NETOPYU_TOOL_RESULT_STORE"] = str(runtime_path.with_name("tool-results.sqlite"))

    from dsh_adapter.local_dc_peer import LocalDcPeer
    from network_lab import ContainerlabProvider, load_manifest
    from network_runtime import NetworkRuntime
    from network_runtime.l0_skills import REGISTRY as L0_SKILLS
    from network_runtime.workflows import WorkflowRuntime

    provider = ContainerlabProvider(load_manifest(manifest_path))
    status = await provider.topology_status()
    if not status["ok"]:
        raise RuntimeError("campus/IDC topology is not fully running")

    # Reviewed deterministic baseline: control user works; Erin is denied by
    # both the LAN enforcement point and the CRM application enforcement point.
    await provider.set_user_admission("bob", admitted=True)
    await provider.set_application_access("bob", "crm", allowed=True)
    await provider.set_user_admission("erin", admitted=False)
    await provider.set_application_access("erin", "crm", allowed=False)
    bob_baseline = await provider.application_probe("bob", "crm")
    erin_baseline = await provider.application_probe("erin", "crm")
    if not bob_baseline["ok"] or erin_baseline["ok"]:
        raise RuntimeError("baseline controls did not prove expected allow/deny behavior")

    runtime = NetworkRuntime(runtime_path)
    session = "campus-idc-e2e:erin:crm"
    with WorkflowRuntime(runtime_path) as workflows:
        workflows.start(
            session_id=session, profile="lan", mode="pragmatic",
            skill_name="lan-new-employee-onboarding-access",
        )

    user_access = await runtime.invoke_read("lan", "get_user_access", {"user_id": "erin"})
    nac_policy = await runtime.invoke_read("lan", "check_nac_policy", {"user_id": "erin"})
    with WorkflowRuntime(runtime_path) as workflows:
        workflows.observe(
            session_id=session, tool_name="get_user_access",
            arguments={"user_id": "erin"}, result=user_access,
            success=True, mutating=False,
        )
        workflows.observe(
            session_id=session, tool_name="check_nac_policy",
            arguments={"user_id": "erin"}, result=nac_policy,
            success=True, mutating=False,
        )

    lan_arguments = {"user_id": "erin", "reason": "approved local onboarding exercise"}
    lan_l0 = L0_SKILLS.for_tool("lan", "grant_user_access")
    if lan_l0 is None:
        raise RuntimeError("LAN L0 Skill is missing")
    lan_prepared = await runtime.prepare(
        "lan", "grant_user_access", lan_arguments,
        session_id=session, l0_skill_id=lan_l0.skill_id,
    )
    if lan_prepared.get("status") != "plan_ready":
        raise RuntimeError("LAN plan was not prepared: " + json.dumps(lan_prepared, ensure_ascii=False))
    lan_plan = lan_prepared["plan"]
    lan_outcome = await runtime.execute(
        plan_id=lan_plan["plan_id"], plan_hash=lan_plan["plan_hash"],
        execution_nonce=lan_prepared["execution_nonce"],
        approval_request_id="local-e2e-lan-approval",
        approval_actor="local-e2e-operator", allow_destructive=True,
    )
    if not lan_outcome.ok:
        raise RuntimeError("LAN L0 execution did not verify: " + json.dumps(lan_outcome.to_dict()))

    dc_prompt = (
        "Invoke dc-app-access-diagnose for user_id=erin, app_id=crm. "
        "Check current application access and ACL; if denied, grant the reviewed base role "
        "with reason=approved local onboarding exercise."
    )
    metadata = {"source_session_id": session, "session_id": session}
    if peer_url:
        from dsh_adapter.a2a_provider import delegate_a2a

        pending = await delegate_a2a(
            prompt=dc_prompt, target="dc-agent", session_id=session,
            own_agent_id="lan-agent", peer_urls=[peer_url],
        )
        if pending.get("status") != "input-required":
            raise RuntimeError("DC HTTP A2A peer did not return its exact approval continuation")
        interrupt_id = str(pending["interrupt_id"])
        dc_completed = await delegate_a2a(
            prompt=dc_prompt, target="dc-agent", session_id=session,
            own_agent_id="lan-agent", peer_urls=[peer_url],
            resume_interrupt_id=interrupt_id, operator_decision="approve",
        )
        if dc_completed.get("status") != "completed":
            raise RuntimeError("DC HTTP A2A resume failed: " + json.dumps(dc_completed))
        dc_result = json.loads(str(dc_completed["text"]))
    else:
        peer = LocalDcPeer(runtime_path=runtime_path, state_path=peer_state)
        pending = await peer.handle(dc_prompt, metadata)
        if pending.get("status", {}).get("state") != "input-required":
            raise RuntimeError("DC A2A peer did not return its exact approval continuation")
        interrupt_id = pending["status"]["message"]["interrupt_id"]
        dc_completed = await peer.handle(dc_prompt, {
            **metadata,
            "resume_interrupt_id": interrupt_id,
            "operator_decision": "approve",
        })
        dc_result = _message(dc_completed)
    if dc_result.get("status") != "completed" or dc_result.get("verified") is not True:
        raise RuntimeError("DC L0 execution did not independently verify")

    path_prompt = "Invoke dc-path-troubleshoot for user_id=erin, app_id=crm; verify end-to-end path."
    if peer_url:
        path_event = await delegate_a2a(
            prompt=path_prompt, target="dc-agent", session_id=session,
            own_agent_id="lan-agent", peer_urls=[peer_url],
        )
        if path_event.get("status") != "completed":
            raise RuntimeError("DC HTTP A2A path verification failed: " + json.dumps(path_event))
        path_result = json.loads(str(path_event["text"]))
    else:
        path_event = await peer.handle(path_prompt, metadata)
        path_result = _message(path_event)
    final_probe = await provider.application_probe("erin", "crm")
    if path_result.get("path_verified") is not True or not final_probe["ok"]:
        raise RuntimeError("final manifest-bound HTTP path was not verified")

    result = {
        "ok": True,
        "topology": provider.manifest.name,
        "a2a_transport": "http" if peer_url else "in-process",
        "baseline": {"bob_to_crm": bob_baseline, "erin_to_crm": erin_baseline},
        "lan_l0": {
            "plan_id": lan_plan["plan_id"], "state": lan_outcome.state.value,
            "audit_ok": runtime.audit(lan_plan["plan_id"])["ok"],
        },
        "dc_l0": {
            "plan_id": dc_result["plan_id"], "state": dc_result["terminal_state"],
            "audit_ok": runtime.audit(dc_result["plan_id"])["ok"],
            "interrupt_id": interrupt_id,
        },
        "final_path": path_result,
        "final_probe": final_probe,
        "left_provisioned": leave_provisioned,
    }
    if exercise_rollback:
        # Force the independent HTTP postcondition to fail after the exact DC
        # write. The Runtime must invoke dc_revoke_app_access and prove that the
        # typed denied preflight was restored before the fault is cleared.
        await provider.set_application_access("erin", "crm", allowed=False)
        rollback_session = session + ":forced-rollback"
        with WorkflowRuntime(runtime_path) as workflows:
            workflows.start(
                session_id=rollback_session, profile="dc", mode="pragmatic",
                skill_name="dc-app-access-diagnose",
            )
        denied = await runtime.invoke_read(
            "dc", "dc_check_user_app_access", {"user_id": "erin", "app_id": "crm"},
        )
        acl = await runtime.invoke_read("dc", "dc_get_app_acl", {"app_id": "crm"})
        with WorkflowRuntime(runtime_path) as workflows:
            workflows.observe(
                session_id=rollback_session, tool_name="dc_check_user_app_access",
                arguments={"user_id": "erin", "app_id": "crm"},
                result=denied, success=True, mutating=False,
            )
            workflows.observe(
                session_id=rollback_session, tool_name="dc_get_app_acl",
                arguments={"app_id": "crm"}, result=acl,
                success=True, mutating=False,
            )
        rollback_arguments = {
            "user_id": "erin", "app_id": "crm", "role": "sales-rep",
            "reason": "forced local verification-failure exercise",
        }
        dc_l0 = L0_SKILLS.for_tool("dc", "dc_grant_app_access")
        if dc_l0 is None:
            raise RuntimeError("DC L0 Skill is missing")
        rollback_prepared = await runtime.prepare(
            "dc", "dc_grant_app_access", rollback_arguments,
            session_id=rollback_session, l0_skill_id=dc_l0.skill_id,
        )
        if rollback_prepared.get("status") != "plan_ready":
            raise RuntimeError("forced rollback plan was not prepared")
        rollback_plan = rollback_prepared["plan"]
        await provider.set_fault("crm-app-egress", kind="loss_pct", value=100)
        try:
            rollback_outcome = await runtime.execute(
                plan_id=rollback_plan["plan_id"], plan_hash=rollback_plan["plan_hash"],
                execution_nonce=rollback_prepared["execution_nonce"],
                approval_request_id="local-e2e-forced-rollback-approval",
                approval_actor="local-e2e-operator", allow_destructive=True,
            )
        finally:
            await provider.set_fault("crm-app-egress", kind="clear_netem")
        restored_block = await provider.application_access_blocked("erin", "crm")
        restored_probe = await provider.application_probe("erin", "crm")
        if (
            rollback_outcome.state.value != "rollback_verified"
            or not runtime.audit(rollback_plan["plan_id"])["ok"]
            or not restored_block or restored_probe["ok"]
        ):
            raise RuntimeError("forced verification failure did not prove inverse rollback")
        result["forced_failure_rollback"] = {
            "plan_id": rollback_plan["plan_id"],
            "state": rollback_outcome.state.value,
            "audit_ok": True,
            "application_block_restored": restored_block,
            "post_rollback_http_ok": restored_probe["ok"],
        }
    if not leave_provisioned:
        await provider.set_application_access("erin", "crm", allowed=False)
        await provider.set_user_admission("erin", admitted=False)
        restored = {
            "lan_admitted": await provider.user_admitted("erin"),
            "crm_blocked": await provider.application_access_blocked("erin", "crm"),
            "http_ok": (await provider.application_probe("erin", "crm"))["ok"],
        }
        if restored != {"lan_admitted": False, "crm_blocked": True, "http_ok": False}:
            raise RuntimeError(f"baseline restoration failed: {restored}")
        result["baseline_restored"] = restored
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leave-provisioned", action="store_true",
        help="keep Erin admitted and CRM-authorized after the verified run",
    )
    parser.add_argument("--runtime-store", type=Path)
    parser.add_argument("--peer-state", type=Path)
    parser.add_argument(
        "--peer-url",
        help="exercise the running loopback A2A peer over HTTP instead of in-process",
    )
    parser.add_argument(
        "--exercise-rollback", action="store_true",
        help="force one DC postcondition failure and prove inverse rollback",
    )
    parser.add_argument(
        "--reset-only", action="store_true",
        help="restore the reviewed Bob-allowed/Erin-denied baseline without running L0",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="reviewed lab manifest (defaults to the campus/IDC lab)",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="pragmatic DSH configuration bound to the selected manifest",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if args.reset_only:
        print(json.dumps(asyncio.run(reset_baseline(
            manifest_path=manifest_path, config_path=config_path,
        )), ensure_ascii=False, indent=2, sort_keys=True))
        return
    runtime_store = args.runtime_store
    if args.peer_url and runtime_store is None:
        from network_runtime.engine import default_journal_path

        runtime_store = Path(
            os.environ.get("NETOPYU_DSH_NETWORK_RUNTIME_STORE") or default_journal_path()
        )
    if runtime_store and (args.peer_state or args.peer_url):
        result = asyncio.run(run(
            leave_provisioned=args.leave_provisioned,
            runtime_path=runtime_store.expanduser().resolve(),
            peer_state=(
                args.peer_state or runtime_store.with_name("unused-http-peer.sqlite")
            ).expanduser().resolve(),
            peer_url=args.peer_url,
            exercise_rollback=args.exercise_rollback,
            manifest_path=manifest_path,
            config_path=config_path,
        ))
    else:
        with tempfile.TemporaryDirectory(prefix="netopyu-campus-idc-e2e-") as directory:
            root = Path(directory)
            result = asyncio.run(run(
                leave_provisioned=args.leave_provisioned,
                runtime_path=(args.runtime_store or root / "runtime.sqlite").expanduser().resolve(),
                peer_state=(args.peer_state or root / "dc-peer.sqlite").expanduser().resolve(),
                peer_url=args.peer_url,
                exercise_rollback=args.exercise_rollback,
                manifest_path=manifest_path,
                config_path=config_path,
            ))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
