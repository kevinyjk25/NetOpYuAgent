"""
evaluation/
-----------
Independent evaluation module — golden-set + benchmark harness for the
framework's retrieval and skill-selection paths.

Framework principle:
  This module reads from the framework but does NOT write to runtime state.
  Evals are run on demand (CLI or webui endpoint), never in the hot path.

  Outputs are pure data structures + optional JSONL files. No global state.

Usage:
  golden = load_golden_set("data/golden_set.jsonl")
  bench  = RetrievalBench(retriever=skill_retriever, golden_set=golden)
  result = bench.run()
  print(result.summary())   # recall@k, precision@k, MRR, per-case verdicts

Designed for:
  - Tracking retrieval quality across config changes (algorithm A vs B)
  - CI gate (PR can't merge if MRR drops > X%)
  - WebUI "Eval" tab to see which queries are mis-routed today
"""
from .types         import EvalCase, EvalCaseResult, BenchReport
from .golden_set    import load_golden_set, save_golden_set, validate_golden_set
from .retrieval_bench import RetrievalBench
from .reporters     import format_text_report, format_jsonl_report

__all__ = [
    "EvalCase", "EvalCaseResult", "BenchReport",
    "load_golden_set", "save_golden_set", "validate_golden_set",
    "RetrievalBench",
    "format_text_report", "format_jsonl_report",
]
