"""
evaluation/retrieval_bench.py
-----------------------------
RetrievalBench — runs a list of EvalCases against any Retriever
(or duck-typed equivalent with .retrieve(query, top_k)).

Computes recall@k, MRR, latency. Splits by language / tag / kind for breakdowns.

Independence:
  Accepts any object with a .retrieve(query: str, top_k: int) → result
  where result has .matches[].id. This is the framework's Retriever protocol;
  external systems can adapt easily.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Protocol

from .types import BenchReport, EvalCase, EvalCaseResult

logger = logging.getLogger(__name__)


class _RetrieverProtocol(Protocol):
    name: str
    def retrieve(self, query: str, top_k: int = 5) -> Any: ...


class RetrievalBench:
    """Run a set of EvalCases against a retriever and produce a BenchReport.

    Usage:
        bench = RetrievalBench(retriever=my_retriever, golden_set=cases, top_k=5)
        report = bench.run()
        print(report.summary())

    Multi-backend comparison:
        for r in [r_bm25, r_hybrid, r_llm_judge]:
            print(RetrievalBench(r, cases).run().summary())
    """

    def __init__(
        self,
        retriever:   _RetrieverProtocol,
        golden_set:  list[EvalCase],
        *,
        top_k:       int = 5,
        backend_name: Optional[str] = None,
        skip_failing_queries: bool = True,
    ):
        self._retriever = retriever
        self._cases     = list(golden_set)
        self._top_k     = max(1, int(top_k))
        self._name      = backend_name or getattr(retriever, "name", "unknown")
        self._skip_failing = bool(skip_failing_queries)

    def run(self) -> BenchReport:
        results:    list[EvalCaseResult] = []
        hits_1     = 0
        hits_3     = 0
        hits_5     = 0
        mrr_sum    = 0.0
        elapsed_sum = 0.0
        n_useful   = 0

        for case in self._cases:
            t0 = time.monotonic()
            try:
                ret = self._retriever.retrieve(case.query, top_k=self._top_k)
            except Exception as exc:
                if self._skip_failing:
                    logger.warning("Bench: retriever error on %r: %s", case.query, exc)
                    continue
                raise
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            elapsed_sum += elapsed_ms
            n_useful += 1

            matches = getattr(ret, "matches", None) or []
            retrieved_ids = [m.id for m in matches]
            scores        = [float(getattr(m, "score", 0.0)) for m in matches]

            # Find rank of first expected id
            rank = None
            for i, rid in enumerate(retrieved_ids, start=1):
                if rid in case.expected_ids:
                    rank = i
                    break

            rr = 1.0 / rank if rank else 0.0
            mrr_sum += rr

            if rank and rank <= 1: hits_1 += 1
            if rank and rank <= 3: hits_3 += 1
            if rank and rank <= 5: hits_5 += 1

            results.append(EvalCaseResult(
                case=case, retrieved_ids=retrieved_ids, scores=scores,
                rank=rank, reciprocal_rank=rr, elapsed_ms=elapsed_ms,
            ))

        total = max(1, n_useful)
        report = BenchReport(
            backend_name=self._name,
            total=n_useful,
            hits_at_1=hits_1,
            hits_at_3=hits_3,
            hits_at_5=hits_5,
            mrr=mrr_sum / total,
            avg_elapsed_ms=elapsed_sum / total,
            cases=results,
        )

        # Breakdowns
        report.breakdown_by["language"] = self._breakdown(results, key=lambda r: r.case.language)
        report.breakdown_by["kind"]     = self._breakdown(results, key=lambda r: r.case.kind)
        report.breakdown_by["tags"]     = self._breakdown_tags(results)

        return report

    @staticmethod
    def _breakdown(
        results: list[EvalCaseResult],
        key:     "Callable[[EvalCaseResult], str]",
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[EvalCaseResult]] = {}
        for r in results:
            groups.setdefault(key(r), []).append(r)
        out = {}
        for grp, items in groups.items():
            n = len(items)
            out[grp] = {
                "n":         n,
                "recall_1":  sum(1 for r in items if r.hit_at(1)) / n,
                "recall_3":  sum(1 for r in items if r.hit_at(3)) / n,
                "recall_5":  sum(1 for r in items if r.hit_at(5)) / n,
                "mrr":       sum(r.reciprocal_rank for r in items) / n,
            }
        return out

    @staticmethod
    def _breakdown_tags(results: list[EvalCaseResult]) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[EvalCaseResult]] = {}
        for r in results:
            for t in r.case.tags or ["<untagged>"]:
                groups.setdefault(t, []).append(r)
        out = {}
        for tag, items in groups.items():
            n = len(items)
            out[tag] = {
                "n":         n,
                "recall_1":  sum(1 for r in items if r.hit_at(1)) / n,
                "recall_3":  sum(1 for r in items if r.hit_at(3)) / n,
                "recall_5":  sum(1 for r in items if r.hit_at(5)) / n,
                "mrr":       sum(r.reciprocal_rank for r in items) / n,
            }
        return out
