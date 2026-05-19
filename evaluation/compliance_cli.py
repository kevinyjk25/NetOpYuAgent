"""
evaluation/compliance_cli.py
----------------------------
CLI runner for tool-compliance bench. Separate from cli.py (which runs
retrieval bench) because the inputs/outputs are different enough that
flag overloading on one entry point gets confusing.

Usage:
    # text protocol (current default behaviour):
    python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl

    # native tools (Ollama ≥ 0.4 + supported model):
    python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --native

    # gate for CI:
    python -m evaluation.compliance_cli ... --fail-below-compliance 0.7

Requires a running Ollama; the bench calls the real model. For CI,
use a mocked engine (compliance is a quality metric, not a unit test —
nightly cadence works fine).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional


def _load_cases(path: Path) -> list:
    from evaluation.tool_compliance_types import ToolCallCase
    cases = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  warn: line {line_no} skipped (JSON error: {e})",
                      file=sys.stderr)
                continue
            try:
                cases.append(ToolCallCase(**raw))
            except (TypeError, ValueError) as e:
                print(f"  warn: line {line_no} skipped (case error: {e})",
                      file=sys.stderr)
                continue
    return cases


def _build_engine(model: str, native: bool):
    """Build a real OllamaEngine with the right capabilities flag."""
    from integrations.clients.llm_engine import OllamaEngine
    from config import LLMCapabilities

    # Minimal capabilities: just toggle native tools. Everything else uses
    # the engine's own defaults so we don't accidentally change behaviour
    # that affects compliance scoring (e.g. think-tag stripping).
    cap = LLMCapabilities(
        thinking_tag="think" if "qwen3" in model.lower() or "deepseek" in model.lower() else "",
        supports_native_tools=native,
    )
    engine = OllamaEngine(
        model=model,
        temperature=0.0,    # Compliance bench wants deterministic emission.
        max_tokens=512,     # Tool calls are short; don't waste tokens.
        think=False,
        capabilities=cap,
    )
    return engine


def _render_report(report, verbose: bool) -> None:
    print()
    print("=" * 60)
    print(f"Tool-Compliance Report — {report.backend_name}")
    print("=" * 60)
    print(f"  total cases:        {report.total}")
    print(f"  parse_ok:           {report.parse_ok_count:3d} ({report.parse_rate:.0%})")
    print(f"  name_ok:            {report.name_ok_count:3d} ({report.name_rate:.0%})")
    print(f"  args_ok:            {report.args_ok_count:3d} ({report.args_rate:.0%})")
    print(f"  fully compliant:    {report.fully_compliant_count:3d} ({report.compliance:.0%})")
    print(f"  errored:            {report.errored_count:3d} ({report.error_rate:.0%})")
    print(f"  avg latency:        {report.avg_elapsed_ms:.0f} ms")
    print()

    if verbose:
        print("Per-case (failures only):")
        for r in report.cases:
            if r.fully_compliant:
                continue
            tags = "/".join(r.case.tags) or "-"
            badge = ("E" if r.error else
                     "P" if not r.parse_ok else
                     "N" if not r.name_ok else
                     "A")  # args
            print(f"  [{badge}] q={r.case.query[:60]!r} [{tags}]")
            if r.error:
                print(f"        error: {r.error}")
                continue
            print(f"        expected: {r.case.expected_tool}({r.case.required_arg_names})")
            print(f"        got:      {r.parsed_name}({list(r.parsed_args.keys())})")
            if r.missing_args:
                print(f"        missing args:  {r.missing_args}")
            if r.wrong_value_args:
                print(f"        wrong values:  {r.wrong_value_args}")
            if r.forbidden_present:
                print(f"        forbidden present: {r.forbidden_present}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tool-compliance bench runner.")
    parser.add_argument("--golden",  required=True, help="Path to tool_compliance_set.jsonl")
    parser.add_argument("--model",   default="qwen2.5:7b",
                        help="Ollama model tag — must be reachable on the configured base URL")
    parser.add_argument("--native",  action="store_true",
                        help="Enable native tools API (Ollama ≥ 0.4). "
                             "Default is text protocol [TOOL:name] {...}.")
    parser.add_argument("--out",     default="",
                        help="Optional JSONL output path for per-case results")
    parser.add_argument("--verbose", action="store_true", help="Print per-case failures")
    parser.add_argument("--fail-below-compliance", type=float, default=0.0,
                        help="Exit 1 if fully-compliant rate is below this fraction")
    parser.add_argument("--fail-below-args", type=float, default=0.0,
                        help="Exit 1 if args_ok rate is below this fraction")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args(argv)

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    golden = Path(args.golden)
    if not golden.exists():
        print(f"error: golden set not found: {golden}", file=sys.stderr)
        return 2

    cases = _load_cases(golden)
    if not cases:
        print("error: no cases loaded", file=sys.stderr)
        return 2
    print(f"loaded {len(cases)} cases from {golden}")

    engine = _build_engine(args.model, args.native)
    mode_tag = "native" if args.native else "text"

    from evaluation.tool_compliance_bench import ToolComplianceBench
    bench = ToolComplianceBench(engine=engine, name=f"{args.model}/{mode_tag}")
    report = asyncio.run(bench.run(cases))

    _render_report(report, verbose=args.verbose)

    # Optional dump
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in report.cases:
                row = {
                    "query":         r.case.query,
                    "expected_tool": r.case.expected_tool,
                    "parsed_name":   r.parsed_name,
                    "parsed_args":   r.parsed_args,
                    "parse_ok":      r.parse_ok,
                    "name_ok":       r.name_ok,
                    "args_ok":       r.args_ok,
                    "missing":       r.missing_args,
                    "wrong_value":   r.wrong_value_args,
                    "forbidden":     r.forbidden_present,
                    "elapsed_ms":    r.elapsed_ms,
                    "error":         r.error,
                    "language":      r.case.language,
                    "tags":          r.case.tags,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote per-case dump to {args.out}")

    # Threshold gates — collect ALL failures before exiting so CI surfaces both.
    failures: list[str] = []
    if args.fail_below_compliance > 0 and report.compliance < args.fail_below_compliance:
        failures.append(
            f"compliance {report.compliance:.3f} < threshold {args.fail_below_compliance:.3f}"
        )
    if args.fail_below_args > 0 and report.args_rate < args.fail_below_args:
        failures.append(
            f"args_ok {report.args_rate:.3f} < threshold {args.fail_below_args:.3f}"
        )
    if failures:
        print(file=sys.stderr)
        for f in failures:
            print(f"  ✗ {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
