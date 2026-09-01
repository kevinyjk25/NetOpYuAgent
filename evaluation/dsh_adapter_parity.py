"""Offline DSH capability-retrieval retirement gate.

This module intentionally lives in the Evaluation Plane.  Product DSH adapter
processes must not import golden sets or evaluator code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any
from unittest.mock import patch

from dsh_adapter.bridge import _build_manifest
from dsh_adapter.scoped_services import search_capabilities
from evaluation import load_golden_set


RETIREMENT_RECALL_AT_3 = 0.95
RETIREMENT_MRR = 0.90


async def parity_report(
    *, profile_id: str, golden_path: str, include_destructive: bool = False,
) -> dict[str, Any]:
    # Capability discovery attaches paging tools even though this evaluator
    # never invokes them.  Force those stores into memory so an offline gate
    # cannot read, prune, or otherwise mutate a live DSH product database.
    with patch.dict(os.environ, {
        "NETOPYU_TOOL_RESULT_STORE": ":memory:",
        "NETOPYU_DSH_TOOL_RESULT_STORE": ":memory:",
    }):
        return await _parity_report(
            profile_id=profile_id,
            golden_path=golden_path,
            include_destructive=include_destructive,
        )


async def _parity_report(
    *, profile_id: str, golden_path: str, include_destructive: bool,
) -> dict[str, Any]:
    manifest = await _build_manifest(
        profile_id, include_destructive=include_destructive,
    )
    cases = load_golden_set(golden_path)
    results = []
    hits_1 = hits_3 = 0
    reciprocal = 0.0
    for case in cases:
        response = await search_capabilities(
            profile_id=profile_id, query=case.query, top_k=5, kinds=[case.kind],
            allowed_tool_names=[tool["name"] for tool in manifest["tools"]],
        )
        ids = [match["id"] for match in response["matches"]]
        rank = next(
            (
                index for index, item in enumerate(ids, 1)
                if item in case.expected_ids
            ),
            None,
        )
        hits_1 += int(rank == 1)
        hits_3 += int(rank is not None and rank <= 3)
        reciprocal += 1 / rank if rank else 0
        results.append({
            "query": case.query,
            "kind": case.kind,
            "expected": case.expected_ids,
            "retrieved": ids,
            "rank": rank,
        })
    total = len(results)
    metrics = {
        "cases": total,
        "recall_at_1": hits_1 / total if total else 0.0,
        "recall_at_3": hits_3 / total if total else 0.0,
        "mrr": reciprocal / total if total else 0.0,
    }
    gate_passed = bool(
        total > 0
        and metrics["recall_at_3"] >= RETIREMENT_RECALL_AT_3
        and metrics["mrr"] >= RETIREMENT_MRR
    )
    return {
        "ok": gate_passed,
        "profile": profile_id,
        "exposed_tools": len(manifest["tools"]),
        "approval_gated_tools": sum(
            tool["requires_approval"] for tool in manifest["tools"]
        ),
        "metrics": metrics,
        "failures": [
            result for result in results
            if result["rank"] is None or result["rank"] > 3
        ],
        "thresholds": {
            "recall_at_3": RETIREMENT_RECALL_AT_3,
            "mrr": RETIREMENT_MRR,
        },
        "retirement_gate": "pass" if gate_passed else "fail",
        "scope": "offline_evaluation_only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline DSH capability retrieval retirement gate",
    )
    parser.add_argument("--profile", default="lan")
    parser.add_argument("--golden", required=True)
    parser.add_argument("--include-destructive", action="store_true")
    args = parser.parse_args(argv)
    report = asyncio.run(parity_report(
        profile_id=args.profile,
        golden_path=args.golden,
        include_destructive=args.include_destructive,
    ))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
