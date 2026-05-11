"""
evaluation/reporters.py
-----------------------
Format BenchReport for humans (text) and machines (JSONL).
"""
from __future__ import annotations

import json
from typing import TextIO, Optional

from .types import BenchReport


def format_text_report(report: BenchReport, *, verbose: bool = False) -> str:
    """Pretty multi-line text. Verbose mode includes per-case details."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"  Retrieval Bench — backend={report.backend_name!r}")
    lines.append("=" * 72)
    lines.append(f"  n_cases:        {report.total}")
    lines.append(f"  recall@1:       {report.recall_at_1:.3f}")
    lines.append(f"  recall@3:       {report.recall_at_3:.3f}")
    lines.append(f"  recall@5:       {report.recall_at_5:.3f}")
    lines.append(f"  MRR:            {report.mrr:.3f}")
    lines.append(f"  avg latency:    {report.avg_elapsed_ms:.1f} ms")
    lines.append("")

    for dim, groups in report.breakdown_by.items():
        if not groups:
            continue
        lines.append(f"  Breakdown by {dim}:")
        lines.append(f"    {'group':<20} {'n':>4} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'MRR':>6}")
        for grp, m in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            lines.append(
                f"    {grp[:20]:<20} {m['n']:>4} "
                f"{m['recall_1']:>6.3f} {m['recall_3']:>6.3f} "
                f"{m['recall_5']:>6.3f} {m['mrr']:>6.3f}"
            )
        lines.append("")

    if verbose:
        lines.append("-" * 72)
        lines.append("  Per-case results")
        lines.append("-" * 72)
        for r in report.cases:
            status = f"hit@{r.rank}" if r.hit else "MISS"
            qpre = r.case.query[:50].replace("\n", " ")
            lines.append(
                f"  [{status:>6}]  {qpre:<50}  expected={r.case.expected_ids[:3]}"
            )
            if not r.hit:
                lines.append(f"           got: {r.retrieved_ids[:5]}")

    lines.append("=" * 72)
    return "\n".join(lines)


def format_jsonl_report(report: BenchReport, out: Optional[TextIO] = None) -> str:
    """Emit one summary line + one line per case. Returns the full string;
    writes to `out` if provided.
    """
    out_lines = []
    summary = {
        "kind":          "bench_summary",
        "backend_name":  report.backend_name,
        "n_cases":       report.total,
        "recall_1":      report.recall_at_1,
        "recall_3":      report.recall_at_3,
        "recall_5":      report.recall_at_5,
        "mrr":           report.mrr,
        "avg_elapsed_ms": report.avg_elapsed_ms,
        "breakdown":     report.breakdown_by,
    }
    out_lines.append(json.dumps(summary, ensure_ascii=False))

    for r in report.cases:
        out_lines.append(json.dumps({
            "kind":            "bench_case",
            "query":           r.case.query,
            "expected_ids":    r.case.expected_ids,
            "language":        r.case.language,
            "tags":            r.case.tags,
            "kind_label":      r.case.kind,
            "retrieved_ids":   r.retrieved_ids,
            "scores":          r.scores,
            "rank":            r.rank,
            "reciprocal_rank": r.reciprocal_rank,
            "elapsed_ms":      r.elapsed_ms,
        }, ensure_ascii=False))

    text = "\n".join(out_lines) + "\n"
    if out is not None:
        out.write(text)
    return text
