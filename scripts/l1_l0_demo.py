#!/usr/bin/env python3
"""Run the isolated Network L1 Skill + L0 Skill access demonstration."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_runtime.demo import run_l1_l0_access_demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-local-simulation",
        action="store_true",
        help="explicitly authorize the two temporary mock writes",
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(run_l1_l0_access_demo(
            approve_local_simulation=args.approve_local_simulation,
        ))
    except Exception as error:
        print(json.dumps({
            "ok": False, "error": f"{type(error).__name__}: {error}",
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
