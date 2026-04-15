"""
config.py
---------
Single source of truth for all runtime configuration.

Priority (highest wins):
  1. Environment variable (e.g. LLM_MODEL=qwen3.5:14b)
  2. config.yaml value
  3. Hardcoded default (in this file)

Usage
-----
    from config import cfg

    cfg.llm.backend          # "ollama"
    cfg.llm.model            # "qwen3.5:27b"
    cfg.hitl.confidence_threshold  # 0.75
    cfg.memory.data_dir      # "./data"

Environment variable overrides
-------------------------------
Every YAML key has a corresponding env var shown below.
Env vars always take priority — no code change needed for
per-deployment tuning.

    YAML key                        → Env var
    ────────────────────────────────────────────────────────
    server.host                     HOST
    server.port                     PORT
    server.reload                   RELOAD
    server.a2a_base_url             A2A_BASE_URL
    llm.backend                     LLM_BACKEND
    llm.model                       LLM_MODEL
    llm.base_url                    LLM_BASE_URL
    llm.temperature                 LLM_TEMPERATURE
    llm.max_tokens                  LLM_MAX_TOKENS
    llm.log_detail                  LLM_LOG_DETAIL
    tools.mcp.use_mock              MCP_USE_MOCK
    tools.mcp.config_json           MCP_CONFIG_JSON
    tools.openapi.use_mock          OPENAPI_USE_MOCK
    tools.openapi.spec_url          OPENAPI_SPEC_URL
    tools.openapi.base_url          OPENAPI_BASE_URL
    tools.openapi.auth_type         OPENAPI_AUTH_TYPE
    tools.openapi.token_env         OPENAPI_TOKEN_ENV
    tools.hitl_tool_names           HITL_TOOL_NAMES
    hitl.confidence_threshold       HITL_CONFIDENCE_THRESHOLD
    hitl.max_auto_host_count        HITL_MAX_AUTO_HOST_COUNT
    hitl.skill_ambiguity            HITL_SKILL_AMBIGUITY
    hitl.slack_webhook_url          HITL_SLACK_WEBHOOK_URL
    hitl.pagerduty_routing_key      HITL_PAGERDUTY_ROUTING_KEY
    memory.data_dir                 HERMES_DATA_DIR
    memory.redis_url                REDIS_URL
    memory.postgres_dsn             POSTGRES_DSN
    memory.chroma_path              CHROMA_PATH
    memory.dtm.compaction_turns     DTM_COMPACTION_TURNS
    memory.dtm.nudge_turns          DTM_NUDGE_TURNS
    memory.dtm.track_b_weight       DTM_TRACK_B_WEIGHT
    memory.dtm.temporal_half_life_days  DTM_HALF_LIFE_DAYS
    registry.agent_urls             AGENT_URLS
    registry.lb_strategy            REGISTRY_LB
    registry.health_check_interval  REGISTRY_HEALTH_INTERVAL
    logging.mode                    LOG_MODE
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── YAML loader (graceful: works without PyYAML installed) ───────────────────

def _load_yaml(path: str) -> dict:
    """Load config.yaml; return empty dict if file missing or PyYAML absent."""
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        import yaml          # PyYAML
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.debug("config: loaded %s", path)
        return data
    except ImportError:
        logger.warning(
            "config: PyYAML not installed — using env vars only.\n"
            "  Install with: pip install pyyaml"
        )
        return {}
    except Exception as exc:
        logger.warning("config: failed to load %s: %s — using env vars only", path, exc)
        return {}


def _get(data: dict, *keys, default=None):
    """Navigate nested dict safely."""
    cur = data
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


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


# ── Config dataclasses ───────────────────────────────────────────────────────

@dataclass
class ServerConfig:
    host:        str
    port:        int
    reload:      bool
    a2a_base_url: str


@dataclass
class LLMConfig:
    backend:     str    # ollama | openai | anthropic | mock
    model:       str
    base_url:    str
    temperature: float
    max_tokens:  int
    log_detail:  str    # off | compact | full


@dataclass
class MCPConfig:
    use_mock:    bool
    config_json: str


@dataclass
class OpenAPIConfig:
    use_mock:  bool
    spec_url:  str
    base_url:  str
    auth_type: str
    token_env: str


@dataclass
class ToolsConfig:
    mcp:             MCPConfig
    openapi:         OpenAPIConfig
    hitl_tool_names: list[str]   # tools that always require HITL


@dataclass
class HITLSLAConfig:
    critical: int
    high:     int
    medium:   int
    low:      int


@dataclass
class HITLConfig:
    confidence_threshold:    float
    max_auto_host_count:     int
    skill_ambiguity:         bool
    slack_webhook_url:       Optional[str]
    pagerduty_routing_key:   Optional[str]
    sla:                     HITLSLAConfig
    destructive_action_types: list[str]


@dataclass
class DTMConfig:
    compaction_turns:        int
    nudge_turns:             int
    track_b_weight:          float
    temporal_half_life_days: float


@dataclass
class MemoryConfig:
    data_dir:    str
    redis_url:   Optional[str]
    postgres_dsn: Optional[str]
    chroma_path: str
    dtm:         DTMConfig


@dataclass
class SkillsConfig:
    top_k:               int
    ambiguity_threshold: float


@dataclass
class StopConfig:
    max_turns:             int
    max_tool_calls:        int
    token_budget:          int
    max_no_progress_turns: int


@dataclass
class RuntimeConfig:
    simple_confidence_floor:  float
    simple_max_tool_calls:    int
    tool_result_inline_limit: int
    stop:                     StopConfig
    pre_verification:         bool
    post_verification:        bool
    model_tiering:            bool


@dataclass
class RegistryConfig:
    agent_urls:            list[str]
    lb_strategy:           str
    health_check_interval: int


@dataclass
class LoggingConfig:
    mode: str    # normal | llm | verbose


@dataclass
class AppConfig:
    server:   ServerConfig
    llm:      LLMConfig
    tools:    ToolsConfig
    hitl:     HITLConfig
    memory:   MemoryConfig
    skills:   SkillsConfig
    runtime:  RuntimeConfig
    registry: RegistryConfig
    logging:  LoggingConfig

    def dump_summary(self) -> str:
        """Human-readable startup summary (logged by main.py)."""
        htnames = ", ".join(self.tools.hitl_tool_names) or "—"
        return (
            f"━━ Configuration ━━\n"
            f"  LLM           : {self.llm.backend}/{self.llm.model}  "
            f"(base_url={self.llm.base_url})\n"
            f"  Tools         : MCP={'real' if not self.tools.mcp.use_mock else 'mock'}"
            f"  OpenAPI={'real' if not self.tools.openapi.use_mock else 'mock'}\n"
            f"  HITL tools    : {htnames}\n"
            f"  Memory dir    : {self.memory.data_dir}\n"
            f"  DTM compaction: every {self.memory.dtm.compaction_turns} turns  "
            f"nudge every {self.memory.dtm.nudge_turns} turns\n"
            f"  Log mode      : {self.logging.mode}  "
            f"LLM detail: {self.llm.log_detail}\n"
            f"  Server        : {self.server.host}:{self.server.port}  "
            f"reload={self.server.reload}"
        )


# ── Builder ──────────────────────────────────────────────────────────────────

def load(config_path: str = "config.yaml") -> AppConfig:
    """
    Load config.yaml, then overlay environment variables.
    Returns a fully-populated AppConfig.
    """
    y = _load_yaml(config_path)
    s   = y.get("server",   {})
    l   = y.get("llm",      {})
    t   = y.get("tools",    {})
    h   = y.get("hitl",     {})
    m   = y.get("memory",   {})
    sk  = y.get("skills",   {})
    r   = y.get("runtime",  {})
    rg  = y.get("registry", {})
    lg  = y.get("logging",  {})

    tm  = t.get("mcp",     {})
    to  = t.get("openapi", {})
    md  = m.get("dtm",     {})
    rs  = r.get("stop",    {})
    hs  = h.get("sla",     {})

    # Parse hitl_tool_names from YAML list or env string
    yaml_hitl_tools = t.get("hitl_tool_names", "") or ""
    env_hitl_tools  = os.getenv("HITL_TOOL_NAMES", "")
    if env_hitl_tools:
        hitl_tool_names = [x.strip() for x in env_hitl_tools.split(",") if x.strip()]
    elif isinstance(yaml_hitl_tools, list):
        hitl_tool_names = [str(x) for x in yaml_hitl_tools]
    else:
        hitl_tool_names = [x.strip() for x in str(yaml_hitl_tools).split(",") if x.strip()]

    # Parse agent_urls from YAML list or env string
    yaml_agents = rg.get("agent_urls", "") or ""
    env_agents  = os.getenv("AGENT_URLS", "")
    if env_agents:
        agent_urls = [u.strip() for u in env_agents.split(",") if u.strip()]
    elif isinstance(yaml_agents, list):
        agent_urls = [str(u) for u in yaml_agents]
    else:
        agent_urls = [u.strip() for u in str(yaml_agents).split(",") if u.strip()]

    # Destructive action types
    yaml_dat = h.get("destructive_action_types", [
        "restart_service", "rollback_deploy", "delete_resource",
        "drain_node", "force_failover", "flush_cache",
    ])
    destructive_action_types = list(yaml_dat) if isinstance(yaml_dat, list) else []

    return AppConfig(
        server=ServerConfig(
            host         = _env_str("HOST",        s.get("host",        "0.0.0.0")),
            port         = _env_int("PORT",         s.get("port",        8000)),
            reload       = _env_bool("RELOAD",      s.get("reload",      False)),
            a2a_base_url = _env_str("A2A_BASE_URL", s.get("a2a_base_url",
                                    "http://localhost:8000/api/v1/a2a")),
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
            data_dir    = _env_str("HERMES_DATA_DIR", m.get("data_dir",    "./data")),
            redis_url   = _env_str("REDIS_URL",       m.get("redis_url",   "")) or None,
            postgres_dsn= _env_str("POSTGRES_DSN",    m.get("postgres_dsn","")) or None,
            chroma_path = _env_str("CHROMA_PATH",     m.get("chroma_path", "./chroma_db")),
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
        ),
        registry=RegistryConfig(
            agent_urls            = agent_urls,
            lb_strategy           = _env_str("REGISTRY_LB",              rg.get("lb_strategy",           "round_robin")),
            health_check_interval = _env_int("REGISTRY_HEALTH_INTERVAL", rg.get("health_check_interval", 60)),
        ),
        logging=LoggingConfig(
            mode = _env_str("LOG_MODE", lg.get("mode", "normal")),
        ),
    )


# ── Module-level singleton ────────────────────────────────────────────────────

# Loaded once at import time from config.yaml next to this file.
# Other modules import it as:  from config import cfg
_CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
cfg: AppConfig = load(str(_CONFIG_PATH))