"""
agent_memory v5 — Production AI Agent Memory Module
=====================================================
Five memory dimensions:
  Short-term   (ShortTermStore)    P0 tool cache, byte-offset paging
  Mid-term     (MidTermStore)      Hermes facts — dedup, TTL, confidence decay
  Long-term    (LongTermStore)     Claw chunks — recency+importance+batch
  Cross-session                    Unified via user_id namespace
  Procedural   (SkillStore)        ← NEW v5: reusable task skills
  Behavioral   (UserModelEngine)   Dialectic user profiling
  Hot-track    (SessionState)      Dual-track: confirmed_facts, working_set
  Budget       (ContextBudgetManager) Priority token budget
  Consolidation(MemoryConsolidator) ← NEW v5: long-session compression
  Reflection   (ReflectionEngine)  ← NEW v5: lessons from success/failure
  Embedding    (EmbeddingIndex)    ← NEW v5: semantic vector retrieval

Zero external dependencies (TF-IDF default). Pluggable:
  embedding_fn / embedding_backend for sentence-transformers / OpenAI
"""
from agent_memory.schemas import MemoryChunk, MemoryFact, ToolResultEntry, RetrievalResult
from agent_memory.memory_manager import MemoryManager
from agent_memory.retrieval.fact_extractor import FactExtractor
from agent_memory.retrieval.vector_store import TFIDFIndex
from agent_memory.retrieval.embedding_store import (
    EmbeddingIndex, EmbeddingBackend,
    TFIDFBackend, CallableBackend,
    SentenceTransformerBackend, OpenAIBackend,
    cosine_similarity,
)
from agent_memory.stores.long_term_store import LongTermStore
from agent_memory.stores.mid_term_store import MidTermStore
from agent_memory.stores.short_term_store import ShortTermStore
from agent_memory.stores.skill_store import SkillStore, Skill
from agent_memory.user_model import (
    UserModelEngine, UserProfile, InferredTrait,
    TechnicalLevel, CommunicationStyle,
)
from agent_memory.session_state import (
    SessionState, SessionStateRegistry,
    ConfirmedFact, WorkingSetEntry, RecentToolResult,
    mmr_rerank,
)
from agent_memory.context_budget import (
    ContextBudgetManager, BudgetReport, BudgetSection,
    Priority, estimate_tokens,
)
from agent_memory.consolidation import MemoryConsolidator, ReflectionEngine

__version__ = "5.0.0"

__all__ = [
    "MemoryManager",
    "MemoryChunk", "MemoryFact", "ToolResultEntry", "RetrievalResult",
    "FactExtractor", "TFIDFIndex",
    "EmbeddingIndex", "EmbeddingBackend",
    "TFIDFBackend", "CallableBackend",
    "SentenceTransformerBackend", "OpenAIBackend",
    "cosine_similarity",
    "LongTermStore", "MidTermStore", "ShortTermStore",
    "SkillStore", "Skill",
    "UserModelEngine", "UserProfile", "InferredTrait",
    "TechnicalLevel", "CommunicationStyle",
    "SessionState", "SessionStateRegistry",
    "ConfirmedFact", "WorkingSetEntry", "RecentToolResult",
    "mmr_rerank",
    "ContextBudgetManager", "BudgetReport", "BudgetSection",
    "Priority", "estimate_tokens",
    "MemoryConsolidator", "ReflectionEngine",
]
