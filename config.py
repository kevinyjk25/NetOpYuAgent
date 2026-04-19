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
class StopConfig:
    max_turns: int; max_tool_calls: int
    token_budget: int; max_no_progress_turns: int

@dataclass
class RuntimeConfig:
    simple_confidence_floor: float; simple_max_tool_calls: int
    tool_result_inline_limit: int; stop: StopConfig
    pre_verification: bool; post_verification: bool; model_tiering: bool

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
    policies:   list = field(default_factory=list)  # prompt-based policies from config.yaml
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
        policies=y.get("policies", []),
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
                 "hitl_high_risk", "preverify_safe_to_proceed"}
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
        "hitl_high_risk", "preverify_safe_to_proceed",
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
