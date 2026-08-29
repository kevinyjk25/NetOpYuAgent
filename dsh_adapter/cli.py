"""Line-oriented JSON CLI for the DSH Node.js plugin."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from .bridge import (
    audit_network_plan,
    approve_network_plan,
    backend_report,
    build_l0_skill_catalog,
    build_manifest,
    execute_network_plan,
    inspect_network_plan,
    invoke_tool,
    observe_network_workflow,
    prepare_network_plan,
    reject_network_plan,
    recent_network_plans,
    start_network_workflow,
)
from .a2a_provider import delegate_a2a, discover_peers
from .scoped_services import recall_memory, search_capabilities
from .evaluation import parity_report
from .learning import mine_candidates, review_candidate
from .reliability import run_local_reliability
from .skills import build_skill_manifest
from .backend import resolve_backend_mode


def _read_arguments() -> dict[str, Any]:
    raw = sys.stdin.read()
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise TypeError("tool arguments must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dsh_adapter.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--profile", default="lan")
    manifest.add_argument("--include-destructive", action="store_true")

    invoke = subparsers.add_parser("invoke")
    invoke.add_argument("--profile", default="lan")
    invoke.add_argument("--tool", required=True)
    invoke.add_argument("--session")

    runtime_prepare = subparsers.add_parser("runtime-prepare")
    runtime_prepare.add_argument("--profile", default="lan")
    runtime_prepare.add_argument("--tool", required=True)
    runtime_prepare.add_argument("--session")
    runtime_prepare.add_argument("--l0-skill")

    runtime_execute = subparsers.add_parser("runtime-execute")
    runtime_execute.add_argument("--profile", default="lan")
    runtime_execute.add_argument("--tool")

    runtime_approve = subparsers.add_parser("runtime-approve")
    runtime_approve.add_argument("--profile", default="lan")
    runtime_approve.add_argument("--tool")

    runtime_inspect = subparsers.add_parser("runtime-inspect")
    runtime_inspect.add_argument("--plan", required=True)

    runtime_audit = subparsers.add_parser("runtime-audit")
    runtime_audit.add_argument("--plan", required=True)

    runtime_list = subparsers.add_parser("runtime-list")
    runtime_list.add_argument("--limit", type=int, default=20)

    runtime_reject = subparsers.add_parser("runtime-reject")
    runtime_reject.add_argument("--profile", default="lan")

    l0_skills = subparsers.add_parser("l0-skills")
    l0_skills.add_argument("--profile", default="lan")

    workflow_start = subparsers.add_parser("workflow-start")
    workflow_start.add_argument("--profile", default="lan")

    workflow_observe = subparsers.add_parser("workflow-observe")
    workflow_observe.add_argument("--profile", default="lan")

    backend = subparsers.add_parser("backend")
    backend.add_argument("--profile", default="lan")

    memory_recall = subparsers.add_parser("memory-recall")
    memory_recall.add_argument("--profile", default="lan")

    capability_search = subparsers.add_parser("capability-search")
    capability_search.add_argument("--profile", default="lan")

    a2a_peers = subparsers.add_parser("a2a-peers")
    a2a_peers.add_argument("--profile", default="lan")

    a2a_delegate = subparsers.add_parser("a2a-delegate")
    a2a_delegate.add_argument("--profile", default="lan")

    parity = subparsers.add_parser("parity")
    parity.add_argument("--profile", default="lan")
    parity.add_argument("--golden", required=True)
    parity.add_argument("--include-destructive", action="store_true")

    trajectory_mine = subparsers.add_parser("trajectory-mine")
    trajectory_mine.add_argument("--source", required=True)
    trajectory_mine.add_argument("--database", required=True)
    trajectory_mine.add_argument("--min-occurrences", type=int, default=2)
    trajectory_mine.add_argument("--apply", action="store_true")

    trajectory_review = subparsers.add_parser("trajectory-review")
    trajectory_review.add_argument("--database", required=True)
    trajectory_review.add_argument("--candidate", required=True)
    trajectory_review.add_argument("--decision", choices=("approve", "reject"), required=True)
    trajectory_review.add_argument("--reviewer", required=True)
    trajectory_review.add_argument("--reason", default="")
    trajectory_review.add_argument("--proposal-directory")

    reliability = subparsers.add_parser("reliability")
    reliability.add_argument("--project-root", required=True)
    reliability.add_argument("--python", required=True)
    reliability.add_argument("--requests", type=int, default=24)
    reliability.add_argument("--concurrency", type=int, default=8)

    skill_manifest = subparsers.add_parser("skill-manifest")
    skill_manifest.add_argument("--profile", default="lan")

    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            payload = build_manifest(args.profile, include_destructive=args.include_destructive)
        elif args.command == "invoke":
            raw_access = os.environ.get("NETOPYU_DSH_ACCESS_CONTEXT", "")
            access_context = json.loads(raw_access) if raw_access else None
            if access_context is not None and not isinstance(access_context, dict):
                raise TypeError("NETOPYU_DSH_ACCESS_CONTEXT must be a JSON object")
            result = asyncio.run(invoke_tool(
                args.profile, args.tool, _read_arguments(), access_context=access_context,
                session_id=args.session,
                harness=os.environ.get("NETOPYU_HARNESS", "local"),
            ))
            payload = {"ok": True, "result": result}
        elif args.command == "runtime-prepare":
            raw_subject = os.environ.get("NETOPYU_DSH_SUBJECT_CONTEXT", "")
            subject_context = json.loads(raw_subject) if raw_subject else None
            if subject_context is not None and not isinstance(subject_context, dict):
                raise TypeError("NETOPYU_DSH_SUBJECT_CONTEXT must be a JSON object")
            payload = asyncio.run(prepare_network_plan(
                args.profile, args.tool, _read_arguments(),
                session_id=args.session, l0_skill_id=args.l0_skill,
                subject_context=subject_context,
                harness=os.environ.get("NETOPYU_HARNESS", "local"),
            ))
        elif args.command == "runtime-approve":
            payload = approve_network_plan(_read_arguments())
        elif args.command == "runtime-execute":
            payload = asyncio.run(execute_network_plan(
                _read_arguments(),
                allow_destructive=os.environ.get("NETOPYU_DSH_ALLOW_DESTRUCTIVE") == "1",
            ))
        elif args.command == "runtime-inspect":
            payload = inspect_network_plan(args.plan)
        elif args.command == "runtime-audit":
            payload = audit_network_plan(args.plan)
        elif args.command == "runtime-list":
            payload = recent_network_plans(args.limit)
        elif args.command == "runtime-reject":
            payload = reject_network_plan(_read_arguments())
        elif args.command == "l0-skills":
            payload = build_l0_skill_catalog(args.profile)
        elif args.command == "workflow-start":
            payload = start_network_workflow(args.profile, _read_arguments())
        elif args.command == "workflow-observe":
            payload = asyncio.run(observe_network_workflow(args.profile, _read_arguments()))
        elif args.command == "backend":
            payload = asyncio.run(backend_report(args.profile))
        elif args.command == "memory-recall":
            payload = asyncio.run(recall_memory(**_read_arguments()))
        elif args.command == "capability-search":
            request = _read_arguments()
            payload = asyncio.run(search_capabilities(profile_id=args.profile, **request))
        elif args.command == "a2a-peers":
            payload = asyncio.run(discover_peers(**_read_arguments()))
        elif args.command == "a2a-delegate":
            payload = asyncio.run(delegate_a2a(**_read_arguments()))
        elif args.command == "parity":
            payload = asyncio.run(parity_report(
                profile_id=args.profile, golden_path=args.golden,
                include_destructive=args.include_destructive,
            ))
        elif args.command == "trajectory-mine":
            payload = mine_candidates(
                source_database=args.source, review_database=args.database,
                min_occurrences=args.min_occurrences, apply_changes=args.apply,
            )
        elif args.command == "trajectory-review":
            payload = review_candidate(
                review_database=args.database, candidate_id=args.candidate,
                decision=args.decision, reviewer=args.reviewer, reason=args.reason,
                proposal_directory=args.proposal_directory,
            )
        elif args.command == "reliability":
            payload = run_local_reliability(
                project_root=args.project_root, python_executable=args.python,
                request_count=args.requests, concurrency=args.concurrency,
            )
        else:
            payload = build_skill_manifest(args.profile, resolve_backend_mode())
    except Exception as error:
        payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(payload, ensure_ascii=False))
        return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
