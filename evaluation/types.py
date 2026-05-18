"""
evaluation/types.py
-------------------
Eval primitives — pure dataclasses, no behaviour.

EvalCase
  One labeled (query, expected_ids) example. Stable schema for serialisation.

EvalCaseResult
  Outcome of running one case against a retriever (or any ranking system).

BenchReport
  Aggregate metrics + per-case results. The bench runner produces one of these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Any, Optional


@dataclass
class EvalCase:
    """One labeled retrieval example.

    Fields:
        query:           the user query (or paraphrase) to test
        expected_ids:    list of CORRECT item ids (a query may have multiple
                         valid answers; if any are in the top-k, the case passes)
        kind:            "skill" | "tool" | other — for grouping in reports
        notes:           free-form annotation (why this case exists, edge case
                         description, etc.) Not used by the bench itself.
        language:        "en" | "zh" | "mixed" | "other" — for split metrics
        tags:            optional category tags (e.g. ["paraphrase", "negation"])
                         so reports can filter by case type
    """
    query:        str
    expected_ids: list[str]
    kind:         str          = "skill"      # "skill" | "tool"
    notes:        str          = ""
    language:     str          = "en"
    tags:         list[str]    = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.expected_ids, list) or not self.expected_ids:
            raise ValueError(f"EvalCase: expected_ids must be a non-empty list (got {self.expected_ids!r})")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("EvalCase: query must be a non-empty string")


@dataclass
class EvalCaseResult:
    """Outcome of running one case.

    Metrics computed:
      hit@k         — True iff any expected id appears in top-k retrieved
      rank          — 1-based position of the FIRST expected id (or None)
      reciprocal_rank — 1/rank if hit else 0 — contributes to MRR
      retrieved_ids — what the system actually returned (ordered, top-N kept)
      scores        — retrieval scores for retrieved_ids
    """
    case:          EvalCase
    retrieved_ids: list[str]
    scores:        list[float]    = field(default_factory=list)
    rank:          Optional[int]  = None
    reciprocal_rank: float        = 0.0
    elapsed_ms:    float          = 0.0

    @property
    def hit(self) -> bool:
        return self.rank is not None

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


@dataclass
class BenchReport:
    """Aggregate benchmark report — produced by RetrievalBench.run()."""
    backend_name:  str
    total:         int
    hits_at_1:     int
    hits_at_3:     int
    hits_at_5:     int
    mrr:           float
    avg_elapsed_ms: float
    cases:         list[EvalCaseResult] = field(default_factory=list)
    breakdown_by:  dict[str, dict[str, Any]] = field(default_factory=dict)  # by language / tag / kind

    @property
    def recall_at_1(self) -> float: return self.hits_at_1 / self.total if self.total else 0.0
    @property
    def recall_at_3(self) -> float: return self.hits_at_3 / self.total if self.total else 0.0
    @property
    def recall_at_5(self) -> float: return self.hits_at_5 / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"Bench[{self.backend_name}] n={self.total}  "
            f"r@1={self.recall_at_1:.2f}  r@3={self.recall_at_3:.2f}  "
            f"r@5={self.recall_at_5:.2f}  MRR={self.mrr:.3f}  "
            f"avg_latency={self.avg_elapsed_ms:.1f}ms"
        )
