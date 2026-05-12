"""
integrations.adapters — cross-module bridges (opt-in via config).

Adapters connect two otherwise-independent framework modules. None of these
should be referenced from runtime/loop.py — they're wired by main.py based
on config, then injected into services dict.

  - memory_facts_adapter   : SkillJournal observations → MemoryFacts
  - fact_conflict_detector : conflict-aware MemoryFact writes
  - hitl_executor          : A2A protocol ↔ HITL gate

Adding a new adapter:
  1. Drop a new file here.
  2. Define a Protocol for each dependency (don't import concrete classes).
  3. Add a cfg.cross_module.<feature> entry in config.py.
  4. Wire in main.py only when cfg.cross_module.<feature>.enabled is true.
"""
from .memory_facts_adapter   import JournalToFactsAdapter  # noqa: F401
from .fact_conflict_detector import (  # noqa: F401
    FactConflictDetector, ReconcileResult,
    VERDICT_EQUIVALENT, VERDICT_REFINEMENT,
    VERDICT_CONTRADICTION, VERDICT_UNRELATED,
)
