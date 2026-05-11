"""
config.py  [v2 — mode-aware: mock | pragmatic]
-----------------------------------------------
Both modes use real LLM, real embeddings, real Redis.
Mode controls only whether tools/MCP are simulated or real.

New sections vs v1:
  - mode: "mock" | "pragmatic"
  - embeddings: backend/model/dim (used by both modes)
  - pragmatic: device_inventory, mcp_servers, napalm_getters
"""
from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
        with p.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("config: PyYAML not installed — using env vars only.")
        return {}
    except Exception as exc:
        logger.warning("config: failed to load %s: %s", path, exc)
        return {}


def _env_bool(name: str, yaml_val) -> bool:
    v = os.getenv(name)
    if v is not None:
        return v.lower() in ("true", "1", "yes")
    return bool(yaml_val)

def _env_str(name: str, yaml_val, default: str = "") -> str:
    v = os.getenv(name)
    if v is not None:
        return v
    return str(yaml_val) if yaml_val is not None else default

def _env_int(name: str, yaml_val, default: int = 0) -> int:
    v = os.getenv(name)
    if v is not None:
        try:
            return int(v)
        except ValueError:
            pass
    return int(yaml_val) if yaml_val is not None else default

def _env_float(name: str, yaml_val, default: float = 0.0) -> float:
    v = os.getenv(name)
    if v is not None:
        try:
            return float(v)
        except ValueError:
            pass
    return float(yaml_val) if yaml_val is not None else default

def _resolve_env(value: str) -> str:
    """Substitute ${ENV_VAR} in a string from environment."""
    import re
    def _sub(m):
        return os.getenv(m.group(1), m.group(0))
    return re.sub(r'\$\{(\w+)\}', _sub, value)


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class ServerConfig:
    host: str; port: int; reload: bool; a2a_base_url: str

@dataclass
class LLMConfig:
    backend: str; model: str; base_url: str
    temperature: float; max_tokens: int; log_detail: str

@dataclass
class MCPConfig:
    use_mock: bool; config_json: str

@dataclass
class OpenAPIConfig:
    use_mock: bool; spec_url: str; base_url: str
    auth_type: str; token_env: str

@dataclass
class ToolsConfig:
    mcp: MCPConfig; openapi: OpenAPIConfig; hitl_tool_names: list[str]
    schema_validation_enabled: bool = True   # validate args via schema/ before tool dispatch

@dataclass
class HITLSLAConfig:
    critical: int; high: int; medium: int; low: int

@dataclass
class HITLConfig:
    confidence_threshold: float; max_auto_host_count: int
    skill_ambiguity: bool; slack_webhook_url: Optional[str]
    pagerduty_routing_key: Optional[str]
    sla: HITLSLAConfig; destructive_action_types: list[str]

@dataclass
class DTMConfig:
    compaction_turns: int; nudge_turns: int
    track_b_weight: float; temporal_half_life_days: float

@dataclass
class MemoryConfig:
    data_dir: str; redis_url: Optional[str]; postgres_dsn: Optional[str]
    chroma_path: str; dtm: DTMConfig
    embedding_model: str = "nomic-embed-text"
    embedding_dim:   int = 768

@dataclass
class SkillsConfig:
    top_k: int; ambiguity_threshold: float

@dataclass
class AuthConfig:
    enabled:        bool = False         # true = enforce auth; false = skip ALL auth
    dev_operator:   str  = "dev-user"    # operator_id used when auth is disabled
    jwt_secret_env: str  = "NETOPYU_JWT_SECRET"

@dataclass
class StopConfig:
    max_turns: int; max_tool_calls: int
    token_budget: int; max_no_progress_turns: int

@dataclass
class RuntimeConfig:
    simple_confidence_floor: float; simple_max_tool_calls: int
    tool_result_inline_limit: int; stop: StopConfig
    pre_verification: bool; post_verification: bool; model_tiering: bool
    # PERF-1: cache cadence for per-turn recall + skill selection
    recall_refresh_every_n_turns:        int  = 3   # refresh memory_results every N turns
    skill_select_refresh_every_n_turns:  int  = 5   # refresh skill selection every N turns
    recall_refresh_facts_growth:         int  = 3   # also refresh when N new facts added
    emit_matched_skills_only_on_change:  bool = True   # PERF-4: dedup SSE skill events

@dataclass
class RegistryConfig:
    agent_urls: list[str]; lb_strategy: str; health_check_interval: int

@dataclass
class LoggingConfig:
    mode: str

# ── NEW: Embeddings ───────────────────────────────────────────────────────────

@dataclass
class EmbeddingsConfig:
    backend:  str    # ollama | openai | none
    model:    str
    base_url: str
    dim:      int

# ── NEW: Pragmatic device entry ───────────────────────────────────────────────

@dataclass
class PragmaticDevice:
    id:          str
    device_type: str          # netmiko device_type string
    host:        str
    username:    str
    password:    str
    secret:      str  = ""
    port:        int  = 22
    timeout:     int  = 30
    label:       str  = ""
    tags:        list[str] = field(default_factory=list)

@dataclass
class PragmaticMCPServer:
    name:      str
    transport: str
    url:       str = ""
    command:   list[str] = field(default_factory=list)
    auth:      dict = field(default_factory=dict)

@dataclass
class PragmaticConfig:
    device_inventory: list[PragmaticDevice]
    mcp_servers:      list[PragmaticMCPServer]
    napalm_getters:   list[str]

# ── Top-level AppConfig ───────────────────────────────────────────────────────


@dataclass
class HermesConfig:
    """Configuration for the Hermes post-turn learning pipeline."""
    skill_evolver_llm_timeout_seconds: float = 30.0   # asyncio.wait_for timeout per LLM call
    skill_evolver_enabled: bool = True
    skill_min_complexity_score: float = 0.6           # eligibility threshold
    skill_max_similar_distance: float = 0.3           # merge-vs-create threshold
    reflection_enabled: bool = True
    consolidation_enabled: bool = True

@dataclass
class PostVerifyConfig:
    """
    Configuration for post-action health verification.
    Maps tool name patterns (regex) to health keywords that must appear in the result.
    Patterns are tried in order; first match wins.
    Set to empty list to disable all post-verification.
    """
    # Each entry: {"pattern": "<regex>", "require_any": ["kw1","kw2"], "require_none": ["err"]}
    rules: list = None  # populated from config.yaml
    # If no rule matches the tool name, default behaviour:
    # True  = pass without inspection (permissive)
    # False = fail unless result is non-empty (strict)
    default_pass: bool = True

    def __post_init__(self):
        if self.rules is None:
            self.rules = []

@dataclass 
class SessionStoreConfig:
    """Configuration for per-session in-memory stores (clarification counter, etc.)."""
    clarification_session_ttl_seconds: int = 3600    # evict entries older than this
    clarification_max_sessions: int = 10_000          # max sessions tracked at once

@dataclass
class ConcurrencyConfig:
    """Async concurrency tuning knobs."""
    hitl_pipeline_poll_interval_ms: int = 50          # BUG-03: _run_steps poll interval
    registry_rr_lock_enabled: bool = True             # DESIGN-05: guard _rr_cursors with asyncio.Lock

@dataclass
class ClassifierFallbackConfig:
    """
    Keyword fallback lists for AgentRuntimeLoop.classify() when the
    PolicyEngine LLM is unavailable. Default English + Chinese pairs
    cover the common destructive-action vocabulary; operators can extend
    or replace these in config.yaml without touching runtime code.
    """
    destructive_keywords: list = None
    p0p1_keywords:        list = None
    fast_model_keywords:  list = None

    def __post_init__(self):
        if self.destructive_keywords is None:
            self.destructive_keywords = []
        if self.p0p1_keywords is None:
            self.p0p1_keywords = []
        if self.fast_model_keywords is None:
            self.fast_model_keywords = []


@dataclass
class WebuiConfig:
    """Frontend timing knobs surfaced via /webui/system/wiring."""
    hitl_poll_interval_ms:    int = 3000      # how often UI polls /hitl/pending
    stats_poll_interval_ms:   int = 20000     # how often UI polls system status/wiring
    hitl_pending_log_at_info: bool = False    # PERF-3: suppress INFO log when count==0



@dataclass
class RetrievalCacheConfig:
    """LRU + TTL cache wrapping the active Retriever.

    Hit rates of 60-90% are typical because users often repeat or paraphrase
    queries within a session, and intra-turn iterations always reuse the
    same query.
    """
    enabled:     bool  = True
    max_entries: int   = 1024
    ttl_seconds: float = 600.0



@dataclass
class LLMJudgeConfig:
    """Knobs for the two-stage LLM-judge retriever (cfg.retrieval.backend=llm_judge).

    The retriever uses any base retriever (typically Hybrid) to fetch
    first_stage_top_k candidates, then asks an LLM to rerank them.
    """
    first_stage_top_k:  int   = 15      # candidate pool size for reranking
    timeout_seconds:    float = 10.0    # LLM call timeout (falls back to base)
    fusion_alpha:       float = 0.3     # weight on base score vs LLM score
                                          # (final = alpha * base + (1-alpha) * llm)
    max_text_chars:     int   = 200     # truncate item text shown to LLM


@dataclass
class HybridFusionConfig:
    bm25_weight:  float = 0.5
    embed_weight: float = 0.5
    fusion:       str   = "weighted_sum"   # weighted_sum | rrf
    rrf_k:        int   = 60
    oversample:   int   = 4


@dataclass
class LLMJudgeConfig:
    """Two-stage retrieve-then-rerank: first stage feeds top-K candidates to
    an LLM that judges relevance. Best for cross-lingual / paraphrase-heavy
    queries where BM25+embedding alone miss.
    """
    first_stage_top_k:  int   = 15
    timeout_seconds:    float = 10.0
    fusion_alpha:       float = 0.3   # 0.0=pure judge, 1.0=pure first stage
    max_text_chars:     int   = 200   # candidate description truncation


@dataclass
class RetrievalCacheConfig:
    """LRU+TTL cache around any Retriever. Composition-based: when enabled,
    the factory wraps the chosen backend with CachedRetriever.

    Hit-rate target in normal usage: 70-90% (same query repeats across
    multi-turn conversations and across operators in the same shift)."""
    enabled:     bool  = True
    max_entries: int   = 1024
    ttl_seconds: float = 600.0   # 10 minutes — typical turn duration band


@dataclass
class RetrievalConfig:
    """Per-retriever knobs for tool/skill top-K selection."""
    backend:                          str   = "hybrid"   # hybrid | bm25 | embedding | keyword | llm_judge
    tool_top_k:                       int   = 5
    skill_top_k:                      int   = 3
    always_inject_extra_tools:        list  = None
    shorten_tool_system_after_turn:   int   = 1
    embed_index_concurrency:          int   = 8     # max in-flight embed() during indexing
    hybrid:    HybridFusionConfig   = field(default_factory=HybridFusionConfig)
    cache:     RetrievalCacheConfig = field(default_factory=RetrievalCacheConfig)
    llm_judge: LLMJudgeConfig       = field(default_factory=LLMJudgeConfig)
    cache:     RetrievalCacheConfig = field(default_factory=RetrievalCacheConfig)
    llm_judge: LLMJudgeConfig        = field(default_factory=LLMJudgeConfig)
    embed_index_concurrency: int = 8   # max in-flight embed() calls during async indexing

    def __post_init__(self):
        if self.always_inject_extra_tools is None:
            self.always_inject_extra_tools = []


@dataclass
class MetaToolsBuiltinConfig:
    list_tools:   bool = True
    list_skills:  bool = True
    tool_details: bool = True


@dataclass
class MetaToolsConfig:
    builtin: MetaToolsBuiltinConfig = field(default_factory=MetaToolsBuiltinConfig)



@dataclass
class PaginationConfig:
    """Behaviour controls for read_stored_result pagination across many turns.

    Without these guards, an LLM may keep calling read_stored_result without
    writing findings — and since older pages get dropped from context to
    save tokens, the final answer ends up based only on the last page.
    """
    findings_nudge_enabled:        bool  = True
    findings_silent_threshold:     int   = 2     # nudge after N tool-only paged reads
    findings_nudge_min_chars:      int   = 40    # response shorter than this = "empty findings"



@dataclass
class SkillScoringConfig:
    """Weights for the LEGACY multi-field keyword scorer.

    Applied only when no Retriever is attached (production path uses retriever).
    Weights should sum to ~1.0 but normalisation is automatic if they don't.

    Tuning guidance:
      - Increase purpose_weight when skill descriptions are tightly written
      - Increase tags_weight when your skill taxonomy is rich
      - Increase params_weight if param names are domain-specific keywords
    """
    purpose_weight:     float = 0.40
    description_weight: float = 0.20
    tags_weight:        float = 0.20
    params_weight:      float = 0.10
    name_id_weight:     float = 0.10

@dataclass
class SkillOrchestrationConfig:
    """Skill selection, ambiguity HITL, and observability (Journal).

    Three feature groups live here:
      1. Ambiguity-triggered HITL — automatically ask the operator when
         multiple skills score similarly above a floor.
      2. Scoring thresholds — control which top-K skills count as "real
         matches" vs filler.
      3. SkillJournal — passive observability, no control-flow effect.
    """
    # ── Ambiguity HITL ──
    # When True (default), runtime stream yields a stop_hitl chunk with
    # user_choice kind when ambiguity_gap_threshold + ambiguity_floor are met.
    hitl_on_ambiguity:        bool  = True
    # Top-1 score must be >= this for ambiguity to even be considered.
    # Below the floor, no skill is a real match — let the LLM improvise.
    ambiguity_floor:          float = 0.40
    # Top-2 score gap must be < this for ambiguity to fire.
    # Lower = stricter (only very-close-tie triggers HITL).
    # Higher = looser (more situations trigger operator pick).
    ambiguity_gap_threshold:  float = 0.08
    # Maximum number of choices to surface to the operator.
    ambiguity_max_choices:    int   = 5

    # ── Scoring weights (LEGACY scorer fallback path) ──
    # Production uses the Retriever-driven path (cfg.retrieval.backend).
    # These weights only kick in when no retriever is attached, e.g. during
    # cold start, tests, or when retrieval is intentionally disabled.
    scoring: "SkillScoringConfig" = field(default_factory=lambda: SkillScoringConfig())

    # ── SkillJournal ──
    # Recording is essentially free, so on by default. Disable per-deployment
    # if you have strict cardinality limits on logs / storage.
    journal_enabled:          bool  = True
    journal_max_entries:      int   = 200          # in-memory ring buffer cap
    journal_persist_path:     str   = ""           # empty = no disk persistence
    # Expose stats via /webui/skill_journal/* endpoints when True.
    journal_api_enabled:      bool  = True

    # ── SkillEvolver feedback from journal ──
    # When True, a background task periodically scans recent journal entries
    # and feeds dormant/failed skills back to SkillEvolver.apply_feedback().
    # Drives the "self-improving skill" Hermes feature with real usage data.
    evolver_feedback_enabled:    bool   = True
    evolver_feedback_interval_s: int    = 300   # how often to scan (seconds)
    evolver_feedback_min_uses:   int    = 3     # min observations before feedback
    evolver_dormant_threshold:   float  = 0.6   # dormant_count/use_count above this → feedback

@dataclass
class StreamingConfig:
    """Server-Sent Event stream timeouts and queue limits."""
    sse_stall_timeout_seconds:    float = 180.0    # break stream after N seconds idle
    exec_task_drain_timeout_seconds: float = 5.0   # graceful task drain on stream end
    chunk_queue_maxsize:          int   = 1000     # back-pressure on token producers


@dataclass
class TruncationConfig:
    """
    Length caps for prose snippets passed to LLMs and stored as context.
    Separated from token budgets so prompt-engineering can tune one
    independently of the other.
    """
    recall_context_chars:        int = 1500   # _fts_context truncation in clarification gate
    confirmed_facts_preview:     int = 10     # max facts shown in pre_verify summary
    skill_detail_chars:          int = 2000   # current_detail in feedback patch
    operator_feedback_chars:     int = 500
    operator_prefs_chars:        int = 200
    rationale_chars:             int = 100    # diff_summary in skill creation
    response_preview_chars:      int = 200    # llm_trace previews
    tool_debug_chars:            int = 2000   # TOOL RESULT debug log
    final_response_summary:      int = 500    # PREV_ANALYSIS summary line
    log_redaction_preview:       int = 120    # warning log truncation


@dataclass
class ContextBudgetDisplayConfig:
    """
    Per-tool-output display caps used inside ContextBudgetManager.
    These control how much of each tool's result is shown in the prompt;
    the full result lives in the ToolResultStore and is fetched lazily.
    """
    paged_result_limit:    int = 1200   # read_stored_result page display cap
    normal_result_limit:   int = 600    # other tools' display cap
    latest_result_bonus:   int = 600    # extra chars for the most recent tool result
    stored_lines_preview:  int = 3      # number of [STORED:]/Preview lines to show
    fallback_preview:      int = 200    # bytes shown when [STORED:] preview parse fails
    page_default_size:     int = 2000   # default offset advance per read_stored_result page
    working_set_show:      int = 10     # max device refs displayed in working_set section



# ─────────────────────────────────────────────────────────────────────────
# Cross-module adapters — framework principle: independent modules,
# explicit opt-in联动 via config.
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class JournalToFactsConfig:
    """SkillJournal → MemoryFacts bridge (Tier 1 #1).

    OFF by default; turn on once you've validated the produced facts look
    sensible in /webui/skill_journal stats.
    """
    enabled:               bool   = False
    interval_s:            int    = 600       # scan period
    min_observations:      int    = 3         # min journal entries before promoting
    dormant_threshold:     float  = 0.6       # dormant ratio → emit "lesson" fact
    success_threshold:     float  = 0.9       # success ratio → emit positive fact
    fact_ttl_days:         float  = 14.0      # auto-facts expire sooner than user-authored
    max_facts_per_scan:    int    = 10
    target_user_id:        str    = "_system"
    target_session_id:     str    = "_cross_session"


@dataclass
class FactConflictDetectionConfig:
    """Inserts go through FactConflictDetector when this is on (Tier 1 #2)."""
    enabled:                bool   = False
    similarity_threshold:   float  = 0.70     # min similarity to consider conflict
    equivalence_threshold:  float  = 0.85     # above this, treat as equivalent
    llm_reconcile_enabled:  bool   = False    # ask LLM when heuristic unsure
    llm_timeout_s:          float  = 8.0
    top_k_candidates:       int    = 5
    confidence_boost:       float  = 0.05     # raise existing on equivalent re-insert
    contradiction_demote:   float  = 0.4      # multiplier (1-x) applied to loser


@dataclass
class CrossModuleConfig:
    """Container for inter-module联动 features.

    Every feature in this section MUST be safe to disable by toggling
    `enabled: false`. No functional regression when off.
    """
    journal_to_facts:        JournalToFactsConfig        = field(default_factory=JournalToFactsConfig)
    fact_conflict_detection: FactConflictDetectionConfig = field(default_factory=FactConflictDetectionConfig)


# ─────────────────────────────────────────────────────────────────────────
# Context budget — pluggable strategy (Tier 2 #3)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ContextBudgetConfig:
    """Selects the context-budget strategy at runtime.

    strategy:
      "legacy"   — runtime/context_budget.py compress_paged_outputs (default,
                   preserves all existing behaviour)
      "priority" — runtime/context_budget_v2.TokenBudget priority-based
                   trimming. New code must use the v2 API explicitly; legacy
                   call sites are unchanged.

    The other fields apply only when strategy="priority".
    """
    strategy:                 str   = "legacy"      # "legacy" | "priority"
    total_chars:              int   = 64000          # ~16k tokens at 4 chars/token
    # Per-section size + priority. Each section is a dict so YAML can extend.
    section_system_core:      int   = 4000
    section_user_profile:     int   = 500
    section_recent_turns:     int   = 20000
    section_tool_results:     int   = 30000
    section_retrieved_memory: int   = 10000
    section_skills:           int   = 5000
    section_older_summary:    int   = 5000


# ─────────────────────────────────────────────────────────────────────────
# Evaluation harness (Tier 2 #4)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationConfig:
    """Optional retrieval-quality evaluation harness.

    Runs golden-set tests against the current retrievers. Can run:
      - On startup (bench_on_startup=true), to gate broken changes
      - On demand via CLI: `python -m evaluation.retrieval_bench ...`
      - On demand via WebUI: /webui/eval/run
    """
    golden_set_path:    str   = ""           # path to golden_set.jsonl
    bench_on_startup:   bool  = False
    bench_top_k:        int   = 5
    report_path:        str   = ""           # optional JSONL output
    fail_below_mrr:     float = 0.0          # if >0, startup raises if MRR < this


@dataclass
class AppConfig:
    mode:       str   # "mock" | "pragmatic"
    server:     ServerConfig
    llm:        LLMConfig
    tools:      ToolsConfig
    hitl:       HITLConfig
    memory:     MemoryConfig
    skills:     SkillsConfig
    runtime:    RuntimeConfig
    registry:   RegistryConfig
    logging:    LoggingConfig
    embeddings: EmbeddingsConfig
    pragmatic:  PragmaticConfig
    auth:       AuthConfig = field(default_factory=AuthConfig)
    policies:   list = field(default_factory=list)
    hermes:     HermesConfig = field(default_factory=HermesConfig)
    post_verify: PostVerifyConfig = field(default_factory=PostVerifyConfig)
    session_store: SessionStoreConfig = field(default_factory=SessionStoreConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    streaming:   StreamingConfig = field(default_factory=StreamingConfig)
    truncation:  TruncationConfig = field(default_factory=TruncationConfig)
    context_budget_display: ContextBudgetDisplayConfig = field(default_factory=ContextBudgetDisplayConfig)
    classifier_fallback: ClassifierFallbackConfig = field(default_factory=ClassifierFallbackConfig)
    webui:               WebuiConfig = field(default_factory=WebuiConfig)
    retrieval:           RetrievalConfig = field(default_factory=RetrievalConfig)
    meta_tools:          MetaToolsConfig = field(default_factory=MetaToolsConfig)
    pagination:          PaginationConfig = field(default_factory=PaginationConfig)
    skill_orchestration: SkillOrchestrationConfig = field(default_factory=SkillOrchestrationConfig)
    cross_module:        CrossModuleConfig    = field(default_factory=CrossModuleConfig)
    context_budget:      ContextBudgetConfig  = field(default_factory=ContextBudgetConfig)
    evaluation:          EvaluationConfig     = field(default_factory=EvaluationConfig)  # prompt-based policies from config.yaml
    def is_mock(self) -> bool:
        return self.mode == "mock"

    @property
    def is_pragmatic(self) -> bool:
        return self.mode == "pragmatic"

    def dump_summary(self) -> str:
        mode_tag = "🔧 PRAGMATIC" if self.is_pragmatic else "🎭 MOCK"
        n_dev = len(self.pragmatic.device_inventory)
        n_mcp = len(self.pragmatic.mcp_servers)
        return (
            f"━━ Configuration ━━\n"
            f"  Mode     : {mode_tag}\n"
            f"  LLM      : {self.llm.backend}/{self.llm.model}\n"
            f"  Embed    : {self.embeddings.backend}/{self.embeddings.model} dim={self.embeddings.dim}\n"
            f"  Tools    : {'mock MCP + mock_tools' if self.is_mock else f'{n_dev} real device(s), {n_mcp} MCP server(s)'}\n"
            f"  Memory   : {self.memory.data_dir}  Redis={'yes' if self.memory.redis_url else 'stub'}\n"
            f"  Server   : {self.server.host}:{self.server.port}"
        )


# ── Builder ───────────────────────────────────────────────────────────────────

def _load_hermes_config(h: dict) -> "HermesConfig":
    return HermesConfig(
        skill_evolver_llm_timeout_seconds = _env_float("HERMES_SKILL_EVOLVER_TIMEOUT",    h.get("skill_evolver_llm_timeout_seconds", 30.0)),
        skill_evolver_enabled             = _env_bool ("HERMES_SKILL_EVOLVER_ENABLED",     h.get("skill_evolver_enabled",             True)),
        skill_min_complexity_score        = _env_float("HERMES_SKILL_MIN_COMPLEXITY",      h.get("skill_min_complexity_score",         0.6)),
        skill_max_similar_distance        = _env_float("HERMES_SKILL_MAX_SIMILAR_DIST",    h.get("skill_max_similar_distance",         0.3)),
        reflection_enabled                = _env_bool ("HERMES_REFLECTION_ENABLED",         h.get("reflection_enabled",                True)),
        consolidation_enabled             = _env_bool ("HERMES_CONSOLIDATION_ENABLED",      h.get("consolidation_enabled",             True)),
    )

def _load_post_verify_config(pv: dict) -> "PostVerifyConfig":
    rules_raw = pv.get("rules", [])
    # Default rules if not configured
    if not rules_raw:
        rules_raw = [
            {"pattern": r"restart.*service|service.*restart", "require_any": ["healthy", "running", "active", "started", "ok"], "require_none": ["error", "failed", "crash"]},
            {"pattern": r"push.*config|config.*push|edit.*config|config.*edit", "require_any": [], "require_none": ["syntax error", "invalid", "rejected"]},
            {"pattern": r"rollback|failover|drain", "require_any": [], "require_none": ["error", "failed", "timeout"]},
        ]
    return PostVerifyConfig(
        rules=rules_raw,
        default_pass=bool(pv.get("default_pass", True)),
    )

def _load_session_store_config(ss: dict) -> "SessionStoreConfig":
    return SessionStoreConfig(
        clarification_session_ttl_seconds = _env_int("SESSION_CLARIF_TTL_SECONDS", ss.get("clarification_session_ttl_seconds", 3600)),
        clarification_max_sessions        = _env_int("SESSION_CLARIF_MAX",         ss.get("clarification_max_sessions",        10_000)),
    )

def _load_concurrency_config(cc: dict) -> "ConcurrencyConfig":
    return ConcurrencyConfig(
        hitl_pipeline_poll_interval_ms = _env_int ("HITL_PIPELINE_POLL_INTERVAL_MS", cc.get("hitl_pipeline_poll_interval_ms", 50)),
        registry_rr_lock_enabled       = _env_bool("REGISTRY_RR_LOCK_ENABLED",        cc.get("registry_rr_lock_enabled",       True)),
    )


def _load_streaming_config(s: dict) -> "StreamingConfig":
    return StreamingConfig(
        sse_stall_timeout_seconds       = _env_float("SSE_STALL_TIMEOUT_SECONDS",       s.get("sse_stall_timeout_seconds",       180.0)),
        exec_task_drain_timeout_seconds = _env_float("SSE_EXEC_DRAIN_TIMEOUT_SECONDS",  s.get("exec_task_drain_timeout_seconds",   5.0)),
        chunk_queue_maxsize             = _env_int  ("SSE_CHUNK_QUEUE_MAX",             s.get("chunk_queue_maxsize",             1000)),
    )


def _load_truncation_config(t: dict) -> "TruncationConfig":
    return TruncationConfig(
        recall_context_chars    = _env_int("TRUNC_RECALL_CONTEXT_CHARS",      t.get("recall_context_chars",     1500)),
        confirmed_facts_preview = _env_int("TRUNC_CONFIRMED_FACTS_PREVIEW",   t.get("confirmed_facts_preview",  10)),
        skill_detail_chars      = _env_int("TRUNC_SKILL_DETAIL_CHARS",        t.get("skill_detail_chars",       2000)),
        operator_feedback_chars = _env_int("TRUNC_OPERATOR_FEEDBACK_CHARS",   t.get("operator_feedback_chars",   500)),
        operator_prefs_chars    = _env_int("TRUNC_OPERATOR_PREFS_CHARS",      t.get("operator_prefs_chars",      200)),
        rationale_chars         = _env_int("TRUNC_RATIONALE_CHARS",           t.get("rationale_chars",           100)),
        response_preview_chars  = _env_int("TRUNC_RESPONSE_PREVIEW_CHARS",    t.get("response_preview_chars",    200)),
        tool_debug_chars        = _env_int("TRUNC_TOOL_DEBUG_CHARS",          t.get("tool_debug_chars",         2000)),
        final_response_summary  = _env_int("TRUNC_FINAL_RESPONSE_SUMMARY",    t.get("final_response_summary",    500)),
        log_redaction_preview   = _env_int("TRUNC_LOG_REDACTION_PREVIEW",     t.get("log_redaction_preview",     120)),
    )


def _load_cb_display_config(c: dict) -> "ContextBudgetDisplayConfig":
    return ContextBudgetDisplayConfig(
        paged_result_limit   = _env_int("CTX_PAGED_RESULT_LIMIT",   c.get("paged_result_limit",   1200)),
        normal_result_limit  = _env_int("CTX_NORMAL_RESULT_LIMIT",  c.get("normal_result_limit",   600)),
        latest_result_bonus  = _env_int("CTX_LATEST_RESULT_BONUS",  c.get("latest_result_bonus",   600)),
        stored_lines_preview = _env_int("CTX_STORED_LINES_PREVIEW", c.get("stored_lines_preview",    3)),
        fallback_preview     = _env_int("CTX_FALLBACK_PREVIEW",     c.get("fallback_preview",       200)),
        page_default_size    = _env_int("CTX_PAGE_DEFAULT_SIZE",    c.get("page_default_size",     2000)),
        working_set_show     = _env_int("CTX_WORKING_SET_SHOW",     c.get("working_set_show",        10)),
    )


def _load_classifier_fallback_config(cf: dict) -> "ClassifierFallbackConfig":
    """Load the keyword fallback lists; defaults preserve current hard-coded behaviour."""
    DEFAULT_DESTRUCTIVE = [
        "restart", "rollback", "delete", "drain", "failover", "flush",
        "reboot", "terminate", "shutdown", "wipe", "reset",
        "重启", "回滚", "删除", "终止", "关机", "重置", "下发配置", "推送配置",
    ]
    DEFAULT_P0P1 = [
        "p0", "p1", "critical", "outage", "down", "emergency",
        "sev0", "sev1", "major incident",
    ]
    DEFAULT_FAST = [
        "dns", "ping", "status", "check", "what is", "show me", "list",
    ]
    return ClassifierFallbackConfig(
        destructive_keywords = cf.get("destructive_keywords") or DEFAULT_DESTRUCTIVE,
        p0p1_keywords        = cf.get("p0p1_keywords")        or DEFAULT_P0P1,
        fast_model_keywords  = cf.get("fast_model_keywords")  or DEFAULT_FAST,
    )


def _load_webui_config(w: dict) -> "WebuiConfig":
    return WebuiConfig(
        hitl_poll_interval_ms     = _env_int ("WEBUI_HITL_POLL_INTERVAL_MS",   w.get("hitl_poll_interval_ms",   3000)),
        stats_poll_interval_ms    = _env_int ("WEBUI_STATS_POLL_INTERVAL_MS",  w.get("stats_poll_interval_ms", 20000)),
        hitl_pending_log_at_info  = _env_bool("WEBUI_HITL_PENDING_LOG_INFO",   w.get("hitl_pending_log_at_info",  False)),
    )


def _load_retrieval_config(r: dict) -> "RetrievalConfig":
    h = r.get("hybrid", {}) or {}
    return RetrievalConfig(
        backend                        = _env_str("RETRIEVAL_BACKEND",  r.get("backend",  "hybrid")),
        tool_top_k                     = _env_int("RETRIEVAL_TOOL_TOP_K",   r.get("tool_top_k",   5)),
        skill_top_k                    = _env_int("RETRIEVAL_SKILL_TOP_K",  r.get("skill_top_k",  3)),
        shorten_tool_system_after_turn = _env_int("RETRIEVAL_SHORTEN_AFTER_TURN", r.get("shorten_tool_system_after_turn", 1)),
        always_inject_extra_tools      = list(r.get("always_inject_extra_tools", []) or []),
        hybrid=HybridFusionConfig(
            bm25_weight   = _env_float("RETRIEVAL_BM25_WEIGHT",  h.get("bm25_weight",   0.5)),
            embed_weight  = _env_float("RETRIEVAL_EMBED_WEIGHT", h.get("embed_weight",  0.5)),
            fusion        = _env_str  ("RETRIEVAL_FUSION",        h.get("fusion",       "weighted_sum")),
            rrf_k         = _env_int  ("RETRIEVAL_RRF_K",         h.get("rrf_k",         60)),
            oversample    = _env_int  ("RETRIEVAL_OVERSAMPLE",    h.get("oversample",     4)),
        ),
        cache=RetrievalCacheConfig(
            enabled     = _env_bool ("RETRIEVAL_CACHE_ENABLED",     (r.get("cache", {}) or {}).get("enabled",     True)),
            max_entries = _env_int  ("RETRIEVAL_CACHE_MAX_ENTRIES", (r.get("cache", {}) or {}).get("max_entries", 1024)),
            ttl_seconds = _env_float("RETRIEVAL_CACHE_TTL_SECONDS", (r.get("cache", {}) or {}).get("ttl_seconds",  600.0)),
        ),
        llm_judge=LLMJudgeConfig(
            first_stage_top_k = _env_int  ("RETRIEVAL_LLM_JUDGE_FIRST_TOP_K", (r.get("llm_judge", {}) or {}).get("first_stage_top_k", 15)),
            timeout_seconds   = _env_float("RETRIEVAL_LLM_JUDGE_TIMEOUT",      (r.get("llm_judge", {}) or {}).get("timeout_seconds",   10.0)),
            fusion_alpha      = _env_float("RETRIEVAL_LLM_JUDGE_ALPHA",        (r.get("llm_judge", {}) or {}).get("fusion_alpha",      0.3)),
            max_text_chars    = _env_int  ("RETRIEVAL_LLM_JUDGE_MAX_CHARS",    (r.get("llm_judge", {}) or {}).get("max_text_chars",   200)),
        ),
        embed_index_concurrency = _env_int("RETRIEVAL_EMBED_INDEX_CONCURRENCY", r.get("embed_index_concurrency", 8)),
    )


def _load_meta_tools_config(m: dict) -> "MetaToolsConfig":
    bi = m.get("builtin", {}) or {}
    return MetaToolsConfig(
        builtin=MetaToolsBuiltinConfig(
            list_tools   = _env_bool("META_TOOL_LIST_TOOLS",    bi.get("list_tools",   True)),
            list_skills  = _env_bool("META_TOOL_LIST_SKILLS",   bi.get("list_skills",  True)),
            tool_details = _env_bool("META_TOOL_TOOL_DETAILS",  bi.get("tool_details", True)),
        ),
    )


def _load_pagination_config(p: dict) -> "PaginationConfig":
    return PaginationConfig(
        findings_nudge_enabled    = _env_bool("PAGINATION_FINDINGS_NUDGE",        p.get("findings_nudge_enabled",     True)),
        findings_silent_threshold = _env_int ("PAGINATION_FINDINGS_SILENT_THR",   p.get("findings_silent_threshold",  2)),
        findings_nudge_min_chars  = _env_int ("PAGINATION_FINDINGS_MIN_CHARS",    p.get("findings_nudge_min_chars",   40)),
    )



def _load_skill_scoring_config(s: dict) -> "SkillScoringConfig":
    return SkillScoringConfig(
        purpose_weight     = _env_float("SKILL_SCORE_W_PURPOSE",     s.get("purpose_weight",     0.40)),
        description_weight = _env_float("SKILL_SCORE_W_DESCRIPTION", s.get("description_weight", 0.20)),
        tags_weight        = _env_float("SKILL_SCORE_W_TAGS",        s.get("tags_weight",        0.20)),
        params_weight      = _env_float("SKILL_SCORE_W_PARAMS",      s.get("params_weight",      0.10)),
        name_id_weight     = _env_float("SKILL_SCORE_W_NAME_ID",     s.get("name_id_weight",     0.10)),
    )


def _load_cross_module_config(c: dict) -> "CrossModuleConfig":
    j = c.get("journal_to_facts", {}) or {}
    f = c.get("fact_conflict_detection", {}) or {}
    return CrossModuleConfig(
        journal_to_facts=JournalToFactsConfig(
            enabled            = _env_bool ("XM_JOURNAL_TO_FACTS",         j.get("enabled",            False)),
            interval_s         = _env_int  ("XM_JTF_INTERVAL",             j.get("interval_s",            600)),
            min_observations   = _env_int  ("XM_JTF_MIN_OBSERVATIONS",     j.get("min_observations",        3)),
            dormant_threshold  = _env_float("XM_JTF_DORMANT_THRESHOLD",    j.get("dormant_threshold",     0.6)),
            success_threshold  = _env_float("XM_JTF_SUCCESS_THRESHOLD",    j.get("success_threshold",     0.9)),
            fact_ttl_days      = _env_float("XM_JTF_FACT_TTL_DAYS",        j.get("fact_ttl_days",        14.0)),
            max_facts_per_scan = _env_int  ("XM_JTF_MAX_FACTS_PER_SCAN",   j.get("max_facts_per_scan",     10)),
            target_user_id     = _env_str  ("XM_JTF_TARGET_USER",          j.get("target_user_id",   "_system")),
            target_session_id  = _env_str  ("XM_JTF_TARGET_SESSION",       j.get("target_session_id", "_cross_session")),
        ),
        fact_conflict_detection=FactConflictDetectionConfig(
            enabled               = _env_bool ("XM_FCD_ENABLED",           f.get("enabled",            False)),
            similarity_threshold  = _env_float("XM_FCD_SIM_THRESHOLD",     f.get("similarity_threshold",  0.70)),
            equivalence_threshold = _env_float("XM_FCD_EQ_THRESHOLD",      f.get("equivalence_threshold", 0.85)),
            llm_reconcile_enabled = _env_bool ("XM_FCD_LLM_RECONCILE",     f.get("llm_reconcile_enabled", False)),
            llm_timeout_s         = _env_float("XM_FCD_LLM_TIMEOUT",       f.get("llm_timeout_s",          8.0)),
            top_k_candidates      = _env_int  ("XM_FCD_TOP_K",             f.get("top_k_candidates",         5)),
            confidence_boost      = _env_float("XM_FCD_BOOST",             f.get("confidence_boost",      0.05)),
            contradiction_demote  = _env_float("XM_FCD_DEMOTE",            f.get("contradiction_demote",   0.4)),
        ),
    )


def _load_context_budget_config(c: dict) -> "ContextBudgetConfig":
    return ContextBudgetConfig(
        strategy                 = _env_str("CTX_BUDGET_STRATEGY",            c.get("strategy",                "legacy")),
        total_chars              = _env_int("CTX_BUDGET_TOTAL",               c.get("total_chars",               64000)),
        section_system_core      = _env_int("CTX_BUDGET_SYSTEM_CORE",         c.get("section_system_core",        4000)),
        section_user_profile     = _env_int("CTX_BUDGET_USER_PROFILE",        c.get("section_user_profile",        500)),
        section_recent_turns     = _env_int("CTX_BUDGET_RECENT_TURNS",        c.get("section_recent_turns",      20000)),
        section_tool_results     = _env_int("CTX_BUDGET_TOOL_RESULTS",        c.get("section_tool_results",      30000)),
        section_retrieved_memory = _env_int("CTX_BUDGET_RETRIEVED_MEM",       c.get("section_retrieved_memory",  10000)),
        section_skills           = _env_int("CTX_BUDGET_SKILLS",              c.get("section_skills",             5000)),
        section_older_summary    = _env_int("CTX_BUDGET_OLDER_SUMMARY",       c.get("section_older_summary",      5000)),
    )


def _load_evaluation_config(e: dict) -> "EvaluationConfig":
    return EvaluationConfig(
        golden_set_path  = _env_str  ("EVAL_GOLDEN_SET_PATH",  e.get("golden_set_path",   "")),
        bench_on_startup = _env_bool ("EVAL_BENCH_ON_STARTUP", e.get("bench_on_startup", False)),
        bench_top_k      = _env_int  ("EVAL_BENCH_TOP_K",      e.get("bench_top_k",         5)),
        report_path      = _env_str  ("EVAL_REPORT_PATH",      e.get("report_path",        "")),
        fail_below_mrr   = _env_float("EVAL_FAIL_BELOW_MRR",   e.get("fail_below_mrr",    0.0)),
    )

def _load_skill_orchestration_config(s: dict) -> "SkillOrchestrationConfig":
    return SkillOrchestrationConfig(
        hitl_on_ambiguity       = _env_bool ("SKILL_HITL_ON_AMBIGUITY",       s.get("hitl_on_ambiguity",       True)),
        scoring                 = _load_skill_scoring_config(s.get("scoring", {})),
        ambiguity_floor         = _env_float("SKILL_AMBIGUITY_FLOOR",         s.get("ambiguity_floor",         0.40)),
        ambiguity_gap_threshold = _env_float("SKILL_AMBIGUITY_GAP",           s.get("ambiguity_gap_threshold", 0.08)),
        ambiguity_max_choices   = _env_int  ("SKILL_AMBIGUITY_MAX_CHOICES",   s.get("ambiguity_max_choices",      5)),
        journal_enabled         = _env_bool ("SKILL_JOURNAL_ENABLED",         s.get("journal_enabled",         True)),
        journal_max_entries     = _env_int  ("SKILL_JOURNAL_MAX_ENTRIES",     s.get("journal_max_entries",      200)),
        journal_persist_path    = _env_str  ("SKILL_JOURNAL_PERSIST_PATH",    s.get("journal_persist_path",       "")),
        journal_api_enabled     = _env_bool ("SKILL_JOURNAL_API_ENABLED",     s.get("journal_api_enabled",     True)),
        evolver_feedback_enabled    = _env_bool ("SKILL_EVOLVER_FEEDBACK",        s.get("evolver_feedback_enabled",    True)),
        evolver_feedback_interval_s = _env_int  ("SKILL_EVOLVER_INTERVAL",        s.get("evolver_feedback_interval_s", 300)),
        evolver_feedback_min_uses   = _env_int  ("SKILL_EVOLVER_MIN_USES",        s.get("evolver_feedback_min_uses",     3)),
        evolver_dormant_threshold   = _env_float("SKILL_EVOLVER_DORMANT_THR",     s.get("evolver_dormant_threshold",   0.6)),
    )


def load(config_path: str = "config.yaml") -> AppConfig:
    y   = _load_yaml(config_path)
    s   = y.get("server",     {})
    l   = y.get("llm",        {})
    t   = y.get("tools",      {})
    h   = y.get("hitl",       {})
    m   = y.get("memory",     {})
    sk  = y.get("skills",     {})
    r   = y.get("runtime",    {})
    rg  = y.get("registry",   {})
    lg  = y.get("logging",    {})
    au  = y.get("auth",       {})
    emb = y.get("embeddings", {})
    pg  = y.get("pragmatic",  {})

    tm  = t.get("mcp",     {})
    to  = t.get("openapi", {})
    md  = m.get("dtm",     {})
    rs  = r.get("stop",    {})
    hs  = h.get("sla",     {})

    mode = _env_str("MODE", y.get("mode", "mock")).lower()
    if mode not in ("mock", "pragmatic"):
        logger.warning("Unknown mode=%r, defaulting to mock", mode)
        mode = "mock"

    # hitl_tool_names
    yaml_ht = t.get("hitl_tool_names", "") or ""
    env_ht  = os.getenv("HITL_TOOL_NAMES", "")
    if env_ht:
        hitl_tool_names = [x.strip() for x in env_ht.split(",") if x.strip()]
    elif isinstance(yaml_ht, list):
        hitl_tool_names = [str(x) for x in yaml_ht]
    else:
        hitl_tool_names = [x.strip() for x in str(yaml_ht).split(",") if x.strip()]

    # agent_urls
    yaml_ag = rg.get("agent_urls", "") or ""
    env_ag  = os.getenv("AGENT_URLS", "")
    if env_ag:
        agent_urls = [u.strip() for u in env_ag.split(",") if u.strip()]
    elif isinstance(yaml_ag, list):
        agent_urls = [str(u) for u in yaml_ag]
    else:
        agent_urls = [u.strip() for u in str(yaml_ag).split(",") if u.strip()]

    # destructive_action_types
    yaml_dat = h.get("destructive_action_types", [
        "restart_service", "rollback_deploy", "delete_resource",
        "drain_node", "force_failover", "flush_cache",
    ])
    destructive_action_types = list(yaml_dat) if isinstance(yaml_dat, list) else []

    # pragmatic devices
    pg_devs_raw = pg.get("device_inventory", []) or []
    pg_devices = []
    for d in pg_devs_raw:
        if not isinstance(d, dict):
            continue
        pg_devices.append(PragmaticDevice(
            id          = d.get("id", ""),
            device_type = d.get("device_type", "cisco_ios"),
            host        = _resolve_env(d.get("host", "")),
            username    = _resolve_env(d.get("username", "")),
            password    = _resolve_env(d.get("password", "")),
            secret      = _resolve_env(d.get("secret", "")),
            port        = int(d.get("port", 22)),
            timeout     = int(d.get("timeout", 30)),
            label       = d.get("label", d.get("id", "")),
            tags        = d.get("tags", []),
        ))

    # pragmatic MCP servers
    pg_mcp_raw = pg.get("mcp_servers", []) or []
    pg_mcps = [
        PragmaticMCPServer(
            name      = srv.get("name", f"mcp_{i}"),
            transport = srv.get("transport", "http"),
            url       = srv.get("url", ""),
            command   = srv.get("command", []),
            auth      = srv.get("auth", {}),
        )
        for i, srv in enumerate(pg_mcp_raw) if isinstance(srv, dict)
    ]

    napalm_getters = pg.get("napalm_getters", [
        "get_facts", "get_interfaces", "get_interfaces_ip",
        "get_bgp_neighbors", "get_ntp_servers", "get_environment",
    ])

    return AppConfig(
        mode=mode,
        server=ServerConfig(
            host         = _env_str("HOST",        s.get("host",        "0.0.0.0")),
            port         = _env_int("PORT",         s.get("port",        8001)),
            reload       = _env_bool("RELOAD",      s.get("reload",      False)),
            a2a_base_url = _env_str("A2A_BASE_URL", s.get("a2a_base_url", "http://localhost:8001/api/v1/a2a")),
        ),
        llm=LLMConfig(
            backend     = _env_str  ("LLM_BACKEND",    l.get("backend",     "ollama")),
            model       = _env_str  ("LLM_MODEL",       l.get("model",       "qwen3.5:27b")),
            base_url    = _env_str  ("LLM_BASE_URL",    l.get("base_url",    "http://localhost:11434")),
            temperature = _env_float("LLM_TEMPERATURE", l.get("temperature", 0.1)),
            max_tokens  = _env_int  ("LLM_MAX_TOKENS",  l.get("max_tokens",  2048)),
            log_detail  = _env_str  ("LLM_LOG_DETAIL",  l.get("log_detail",  "off")),
        ),
        tools=ToolsConfig(
            mcp=MCPConfig(
                use_mock    = _env_bool("MCP_USE_MOCK",    tm.get("use_mock",    True)),
                config_json = _env_str ("MCP_CONFIG_JSON", tm.get("config_json", "")),
            ),
            openapi=OpenAPIConfig(
                use_mock  = _env_bool("OPENAPI_USE_MOCK", to.get("use_mock",  True)),
                spec_url  = _env_str ("OPENAPI_SPEC_URL", to.get("spec_url",  "")),
                base_url  = _env_str ("OPENAPI_BASE_URL", to.get("base_url",  "")),
                auth_type = _env_str ("OPENAPI_AUTH_TYPE",to.get("auth_type", "bearer")),
                token_env = _env_str ("OPENAPI_TOKEN_ENV",to.get("token_env", "NETOPS_API_TOKEN")),
            ),
            hitl_tool_names=hitl_tool_names,
        ),
        hitl=HITLConfig(
            confidence_threshold   = _env_float("HITL_CONFIDENCE_THRESHOLD", h.get("confidence_threshold", 0.75)),
            max_auto_host_count    = _env_int  ("HITL_MAX_AUTO_HOST_COUNT",   h.get("max_auto_host_count",  5)),
            skill_ambiguity        = _env_bool ("HITL_SKILL_AMBIGUITY",       h.get("skill_ambiguity",      False)),
            slack_webhook_url      = _env_str  ("HITL_SLACK_WEBHOOK_URL",     h.get("slack_webhook_url",    "")) or None,
            pagerduty_routing_key  = _env_str  ("HITL_PAGERDUTY_ROUTING_KEY", h.get("pagerduty_routing_key","")) or None,
            sla=HITLSLAConfig(
                critical = _env_int("", hs.get("critical", 300)),
                high     = _env_int("", hs.get("high",     600)),
                medium   = _env_int("", hs.get("medium",   900)),
                low      = _env_int("", hs.get("low",      1800)),
            ),
            destructive_action_types=destructive_action_types,
        ),
        memory=MemoryConfig(
            data_dir     = _env_str("HERMES_DATA_DIR", m.get("data_dir",    "./data")),
            redis_url    = _env_str("REDIS_URL",       m.get("redis_url",   "")) or None,
            postgres_dsn = _env_str("POSTGRES_DSN",    m.get("postgres_dsn","")) or None,
            chroma_path  = _env_str("CHROMA_PATH",     m.get("chroma_path", "./chroma_db")),
            dtm=DTMConfig(
                compaction_turns        = _env_int  ("DTM_COMPACTION_TURNS", md.get("compaction_turns",        20)),
                nudge_turns             = _env_int  ("DTM_NUDGE_TURNS",      md.get("nudge_turns",             10)),
                track_b_weight          = _env_float("DTM_TRACK_B_WEIGHT",   md.get("track_b_weight",          1.5)),
                temporal_half_life_days = _env_float("DTM_HALF_LIFE_DAYS",   md.get("temporal_half_life_days", 7.0)),
            ),
        ),
        skills=SkillsConfig(
            top_k               = _env_int  ("", sk.get("top_k",               5)),
            ambiguity_threshold = _env_float("", sk.get("ambiguity_threshold", 0.15)),
        ),
        runtime=RuntimeConfig(
            simple_confidence_floor  = _env_float("", r.get("simple_confidence_floor",  0.70)),
            simple_max_tool_calls    = _env_int  ("", r.get("simple_max_tool_calls",    4)),
            tool_result_inline_limit = _env_int  ("", r.get("tool_result_inline_limit", 4000)),
            stop=StopConfig(
                max_turns             = _env_int("", rs.get("max_turns",             10)),
                max_tool_calls        = _env_int("", rs.get("max_tool_calls",        20)),
                token_budget          = _env_int("", rs.get("token_budget",          50000)),
                max_no_progress_turns = _env_int("", rs.get("max_no_progress_turns", 3)),
            ),
            pre_verification  = _env_bool("", r.get("pre_verification",  True)),
            post_verification = _env_bool("", r.get("post_verification", True)),
            model_tiering     = _env_bool("", r.get("model_tiering",     False)),
            recall_refresh_every_n_turns       = _env_int("RUNTIME_RECALL_REFRESH_TURNS",       r.get("recall_refresh_every_n_turns",       3)),
            skill_select_refresh_every_n_turns = _env_int("RUNTIME_SKILL_REFRESH_TURNS",         r.get("skill_select_refresh_every_n_turns", 5)),
            recall_refresh_facts_growth        = _env_int("RUNTIME_RECALL_FACTS_GROWTH",         r.get("recall_refresh_facts_growth",        3)),
            emit_matched_skills_only_on_change = _env_bool("RUNTIME_EMIT_SKILLS_ON_CHANGE",      r.get("emit_matched_skills_only_on_change", True)),
        ),
        registry=RegistryConfig(
            agent_urls            = agent_urls,
            lb_strategy           = _env_str("REGISTRY_LB",              rg.get("lb_strategy",           "round_robin")),
            health_check_interval = _env_int("REGISTRY_HEALTH_INTERVAL", rg.get("health_check_interval", 60)),
        ),
        logging=LoggingConfig(
            mode = _env_str("LOG_MODE", lg.get("mode", "normal")),
        ),
        embeddings=EmbeddingsConfig(
            backend  = _env_str("EMBED_BACKEND", emb.get("backend",  "ollama")),
            model    = _env_str("EMBED_MODEL",   emb.get("model",    "nomic-embed-text")),
            base_url = _env_str("EMBED_BASE_URL",emb.get("base_url", "http://localhost:11434")),
            dim      = _env_int("EMBED_DIM",     emb.get("dim",      768)),
        ),
        pragmatic=PragmaticConfig(
            device_inventory = pg_devices,
            mcp_servers      = pg_mcps,
            napalm_getters   = napalm_getters,
        ),
        auth=AuthConfig(
            enabled        = bool(au.get("enabled", False)),
            dev_operator   = str(au.get("dev_operator", "dev-user")),
            jwt_secret_env = str(au.get("jwt_secret_env", "NETOPYU_JWT_SECRET")),
        ),
        policies=y.get("policies", []),
        hermes=_load_hermes_config(y.get("hermes", {})),
        post_verify=_load_post_verify_config(y.get("post_verify", {})),
        session_store=_load_session_store_config(y.get("session_store", {})),
        concurrency=_load_concurrency_config(y.get("concurrency", {})),
        streaming=_load_streaming_config(y.get("streaming", {})),
        truncation=_load_truncation_config(y.get("truncation", {})),
        context_budget_display=_load_cb_display_config(y.get("context_budget_display", {})),
        classifier_fallback=_load_classifier_fallback_config(y.get("classifier_fallback", {})),
        webui=_load_webui_config(y.get("webui", {})),
        retrieval=_load_retrieval_config(y.get("retrieval", {})),
        meta_tools=_load_meta_tools_config(y.get("meta_tools", {})),
        pagination=_load_pagination_config(y.get("pagination", {})),
        skill_orchestration=_load_skill_orchestration_config(y.get("skill_orchestration", {})),
        cross_module        =_load_cross_module_config       (y.get("cross_module",        {})),
        context_budget      =_load_context_budget_config     (y.get("context_budget",      {})),
        evaluation          =_load_evaluation_config         (y.get("evaluation",          {})),
    )


_CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
cfg: AppConfig = load(str(_CONFIG_PATH))

# Validate at import time so startup fails fast with a clear error message
# rather than silently degrading at the first LLM call.
def _validate_on_load() -> None:
    import logging as _log
    _log = _log.getLogger("config")
    _errs: list[str] = []

    if not getattr(cfg.llm, "model", ""):
        _errs.append("llm.model is required")
    if not getattr(cfg.llm, "base_url", ""):
        _errs.append("llm.base_url is required (e.g. http://localhost:11434)")
    if getattr(cfg.llm, "backend", "") not in ("ollama", "openai", "anthropic", ""):
        _log.warning("config: llm.backend=%r unrecognised", cfg.llm.backend)

    _policies = getattr(cfg, "policies", None) or []
    _found    = {p.get("name", "") for p in _policies if isinstance(p, dict)}
    _required = {"classify_destructive", "classify_incident_severity",
                 "hitl_high_risk"}
    for _p in _required - _found:
        _log.warning("config: recommended policy %r missing from config.yaml", _p)

    if cfg.mode == "pragmatic":
        _devs = getattr(getattr(cfg, "pragmatic", None), "device_inventory", [])
        if not _devs:
            _log.warning("config: pragmatic mode with empty device_inventory")

    if _errs:
        raise RuntimeError(
            "Config validation failed — fix config.yaml before starting:\n"
            + "\n".join(f"  ✗ {e}" for e in _errs)
        )

_validate_on_load()


def validate_config(cfg: "AppConfig") -> list[str]:
    """
    Validate required config fields at startup.
    Returns a list of error strings — empty means valid.
    Raises RuntimeError if any blockers are found.
    """
    errors = []
    warnings = []

    # LLM
    if not getattr(cfg, "llm", None):
        errors.append("llm: section missing")
    else:
        if not getattr(cfg.llm, "model", ""):
            errors.append("llm.model: required — set to your Ollama model name")
        if not getattr(cfg.llm, "base_url", ""):
            errors.append("llm.base_url: required (e.g. http://localhost:11434)")
        backend = getattr(cfg.llm, "backend", "")
        if backend not in ("ollama", "openai", "anthropic", ""):
            warnings.append(f"llm.backend={backend!r} unrecognised — expected ollama|openai|anthropic")

    # Embeddings
    if not getattr(cfg, "embeddings", None):
        warnings.append("embeddings: section missing — semantic search disabled")
    else:
        dim = getattr(cfg.embeddings, "dim", 0)
        if dim not in (384, 768, 1536, 3072):
            warnings.append(f"embeddings.dim={dim} unusual — verify it matches your model")

    # Runtime
    if not getattr(cfg, "runtime", None):
        warnings.append("runtime: section missing — using defaults")
    else:
        max_turns = getattr(cfg.runtime.stop, "max_turns", 0) if getattr(cfg.runtime, "stop", None) else 0
        if max_turns < 3:
            warnings.append(f"runtime.stop.max_turns={max_turns} very low — agent may stop early")

    # Policies
    policies = getattr(cfg, "policies", None) or []
    required_policies = {
        "classify_destructive", "classify_incident_severity",
        "hitl_high_risk",
    }
    found_policies = {p.get("name", "") for p in policies if isinstance(p, dict)}
    missing = required_policies - found_policies
    if missing:
        warnings.append(f"policies: missing recommended entries: {sorted(missing)}")

    # Mode check
    mode = getattr(cfg, "mode", "mock")
    if mode == "pragmatic":
        devices = getattr(getattr(cfg, "pragmatic", None), "device_inventory", [])
        if not devices:
            warnings.append("pragmatic mode: device_inventory is empty — no real devices configured")

    import logging as _log
    _logger = _log.getLogger("config")
    for w in warnings:
        _logger.warning("Config warning: %s", w)
    for e in errors:
        _logger.error("Config error: %s", e)

    if errors:
        raise RuntimeError(
            "Config validation failed — fix errors before starting:\n"
            + "\n".join(f"  ✗ {e}" for e in errors)
        )
    return warnings