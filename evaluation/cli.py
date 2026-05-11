"""
evaluation/cli.py
-----------------
CLI runner — independent of the FastAPI app. Useful for:
  - CI checks: `python -m evaluation.cli --golden data/golden_set.jsonl`
  - Pre-merge gates: `--fail-below-mrr 0.5`
  - Comparing backends: `--backend bm25` then `--backend hybrid`

Doesn't import main.py or webui — only depends on the retrieval framework
and the eval module.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a retrieval bench on a golden set.")
    parser.add_argument("--golden",     required=True, help="Path to golden_set.jsonl")
    parser.add_argument("--backend",    default="hybrid",
                        help="Retriever backend: bm25 | embedding | hybrid | keyword")
    parser.add_argument("--top-k",      type=int, default=5, help="Top-K to retrieve")
    parser.add_argument("--kind",       default="skill",
                        help="Which corpus to test: 'skill' (default) or 'tool'")
    parser.add_argument("--out",        default="", help="Optional JSONL output path")
    parser.add_argument("--verbose",    action="store_true", help="Per-case detail in text report")
    parser.add_argument("--fail-below-mrr", type=float, default=0.0,
                        help="Exit 1 if MRR is below this threshold")
    parser.add_argument("--quiet",      action="store_true", help="Reduce log noise")
    args = parser.parse_args(argv)

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    import config as _config
    # Override backend per CLI arg
    _config.cfg.retrieval.backend = args.backend
    _config.cfg.retrieval.cache.enabled = False  # disable cache for clean measurement

    from evaluation import (
        load_golden_set, validate_golden_set,
        RetrievalBench, format_text_report, format_jsonl_report,
    )
    from tools.loader import ToolLoader

    cases = load_golden_set(args.golden)
    if not cases:
        print(f"ERROR: no cases loaded from {args.golden!r}", file=sys.stderr)
        return 2

    # Build the right retriever for this benchmark
    if args.kind == "skill":
        from retrieval import build_skill_retriever
        defs = ToolLoader(_config.cfg.mode).skill_definitions()
        retriever = build_skill_retriever(_config.cfg, embedder=None,
                                          skill_definitions=defs)
        available_ids = set(defs.keys())
    elif args.kind == "tool":
        from retrieval import build_tool_retriever
        meta = ToolLoader(_config.cfg.mode).build_metadata()
        retriever = build_tool_retriever(_config.cfg, embedder=None, tool_metadata=meta)
        available_ids = set(meta.keys())
    else:
        print(f"ERROR: --kind must be 'skill' or 'tool' (got {args.kind!r})", file=sys.stderr)
        return 2

    warnings = validate_golden_set(cases, available_ids)
    if warnings:
        print("\n[golden set warnings]")
        for w in warnings:
            print(f"  ⚠ {w}")

    bench = RetrievalBench(retriever, cases, top_k=args.top_k)
    report = bench.run()
    print()
    print(format_text_report(report, verbose=args.verbose))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(format_jsonl_report(report))
        print(f"\nJSONL written to {args.out}")

    if args.fail_below_mrr > 0 and report.mrr < args.fail_below_mrr:
        print(f"\n✗ MRR {report.mrr:.3f} below threshold {args.fail_below_mrr:.3f}",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
