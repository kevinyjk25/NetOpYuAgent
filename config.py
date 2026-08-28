"""Typed configuration used by the DSH NetOpYu bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MCPConfig:
    config_json: str = ""


@dataclass
class OpenAPIConfig:
    spec_url: str = ""
    base_url: str = ""
    auth_type: str = "bearer"
    token_env: str = "NETOPS_API_TOKEN"


@dataclass
class ToolsConfig:
    mcp: MCPConfig = field(default_factory=MCPConfig)
    openapi: OpenAPIConfig = field(default_factory=OpenAPIConfig)
    editable_hitl_tools: dict[str, list[str]] = field(default_factory=dict)
    schema_validation_enabled: bool = True


@dataclass
class PragmaticDevice:
    id: str
    device_type: str
    host: str
    username: str
    password: str
    secret: str = ""
    port: int = 22
    timeout: int = 30
    label: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class PragmaticMCPServer:
    name: str
    transport: str
    url: str = ""
    command: list[str] = field(default_factory=list)
    auth: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    domain: str = "external"
    trusted_for_writes: bool = False
    expected_server_name: str = ""
    expected_server_version: str = ""
    timeout: float = 30.0


@dataclass
class PragmaticLabConfig:
    enabled: bool = False
    provider: str = "containerlab"
    manifest: str = "labs/p075-a-frr/lab.yaml"
    command_timeout: float = 30.0


@dataclass
class PragmaticConfig:
    device_inventory: list[PragmaticDevice] = field(default_factory=list)
    mcp_servers: list[PragmaticMCPServer] = field(default_factory=list)
    lab: PragmaticLabConfig = field(default_factory=PragmaticLabConfig)
    napalm_getters: list[str] = field(default_factory=lambda: [
        "get_facts", "get_interfaces", "get_interfaces_counters",
    ])


@dataclass
class AgentConfig:
    agent_id: str = "lan-agent"
    profile: str = "lan"
    peer_urls: list[str] = field(default_factory=list)


@dataclass
class RetrievalCacheConfig:
    enabled: bool = False
    max_entries: int = 1024
    ttl_seconds: float = 600.0


@dataclass
class HybridConfig:
    bm25_weight: float = 0.5
    embed_weight: float = 0.5
    fusion: str = "weighted_sum"
    rrf_k: int = 60
    oversample: int = 4


@dataclass
class LLMJudgeConfig:
    first_stage_top_k: int = 15
    timeout_seconds: float = 10.0
    fusion_alpha: float = 0.3
    max_text_chars: int = 200


@dataclass
class RetrievalConfig:
    backend: str = "bm25"
    cache: RetrievalCacheConfig = field(default_factory=RetrievalCacheConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)
    embed_index_concurrency: int = 8


@dataclass
class EmbeddingsConfig:
    dim: int = 768


@dataclass
class AppConfig:
    mode: str = "mock"
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    pragmatic: PragmaticConfig = field(default_factory=PragmaticConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    @property
    def is_pragmatic(self) -> bool:
        return self.mode == "pragmatic"

    def agent_data_dir(self) -> str:
        """Return the DSH data directory isolated for this agent instance."""
        override = os.getenv("AGENT_DATA_DIR")
        if override:
            return str(Path(override).expanduser())
        return str(Path(__file__).resolve().parent / "data" / "agents" / self.agent.agent_id)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _csv(value: str) -> list[str]:
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


def load(path: str | os.PathLike[str] = "config.yaml") -> AppConfig:
    source = Path(path).expanduser()
    config_directory = source.resolve().parent
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) if source.is_file() else {}
    raw = _mapping(raw)

    tools_raw = _mapping(raw.get("tools"))
    mcp_raw = _mapping(tools_raw.get("mcp"))
    openapi_raw = _mapping(tools_raw.get("openapi"))
    pragmatic_raw = _mapping(raw.get("pragmatic"))
    lab_raw = _mapping(pragmatic_raw.get("lab"))
    agent_raw = _mapping(raw.get("agent"))
    retrieval_raw = _mapping(raw.get("retrieval"))
    cache_raw = _mapping(retrieval_raw.get("cache"))
    hybrid_raw = _mapping(retrieval_raw.get("hybrid"))
    judge_raw = _mapping(retrieval_raw.get("llm_judge"))
    embeddings_raw = _mapping(raw.get("embeddings"))

    mode = str(os.getenv("NETOPYU_DSH_BACKEND") or os.getenv("MODE") or raw.get("mode", "mock")).lower()
    if mode not in {"mock", "pragmatic"}:
        raise ValueError("mode must be mock or pragmatic")

    profile = str(os.getenv("NETOPYU_PROFILE") or os.getenv("AGENT_PROFILE") or agent_raw.get("profile", "lan"))
    peers_env = os.getenv("NETOPYU_DSH_A2A_PEERS") or os.getenv("AGENT_PEERS")
    peers = _csv(peers_env) if peers_env is not None else [
        str(item).strip().rstrip("/") for item in _list(agent_raw.get("peers")) if str(item).strip()
    ]

    devices = [PragmaticDevice(
        id=str(item.get("id", "")),
        device_type=str(item.get("device_type", "")),
        host=str(item.get("host", "")),
        username=str(item.get("username", "")),
        password=str(item.get("password", "")),
        secret=str(item.get("secret", "")),
        port=int(item.get("port", 22)),
        timeout=int(item.get("timeout", 30)),
        label=str(item.get("label", "")),
        tags=[str(tag) for tag in _list(item.get("tags"))],
    ) for item in map(_mapping, _list(pragmatic_raw.get("device_inventory")))]

    servers = [PragmaticMCPServer(
        name=str(item.get("name", "")),
        transport=str(item.get("transport", "")),
        url=str(item.get("url", "")),
        command=[str(part) for part in _list(item.get("command"))],
        auth=_mapping(item.get("auth")),
        env={str(k): str(v) for k, v in _mapping(item.get("env")).items()},
        cwd=str(
            (config_directory / str(item.get("cwd") or ".")).resolve()
        ),
        domain=str(item.get("domain", "external")),
        trusted_for_writes=bool(item.get("trusted_for_writes", False)),
        expected_server_name=str(item.get("expected_server_name", "")),
        expected_server_version=str(item.get("expected_server_version", "")),
        timeout=float(item.get("timeout", 30.0)),
    ) for item in map(_mapping, _list(pragmatic_raw.get("mcp_servers")))]

    editable = {
        str(name): [str(field_name) for field_name in _list(fields)]
        for name, fields in _mapping(tools_raw.get("editable_hitl_tools")).items()
    }
    return AppConfig(
        mode=mode,
        tools=ToolsConfig(
            mcp=MCPConfig(config_json=os.getenv("MCP_CONFIG_JSON", str(mcp_raw.get("config_json", "")))),
            openapi=OpenAPIConfig(
                spec_url=os.getenv("OPENAPI_SPEC_URL", str(openapi_raw.get("spec_url", ""))),
                base_url=os.getenv("OPENAPI_BASE_URL", str(openapi_raw.get("base_url", ""))),
                auth_type=os.getenv("OPENAPI_AUTH_TYPE", str(openapi_raw.get("auth_type", "bearer"))),
                token_env=os.getenv("OPENAPI_TOKEN_ENV", str(openapi_raw.get("token_env", "NETOPS_API_TOKEN"))),
            ),
            editable_hitl_tools=editable,
            schema_validation_enabled=bool(tools_raw.get("schema_validation_enabled", True)),
        ),
        pragmatic=PragmaticConfig(
            device_inventory=devices,
            mcp_servers=servers,
            lab=PragmaticLabConfig(
                enabled=bool(lab_raw.get("enabled", False)),
                provider=str(lab_raw.get("provider", "containerlab")).strip().lower(),
                manifest=str(lab_raw.get("manifest", "labs/p075-a-frr/lab.yaml")).strip(),
                command_timeout=float(lab_raw.get("command_timeout", 30)),
            ),
            napalm_getters=[str(item) for item in _list(pragmatic_raw.get("napalm_getters"))]
                or PragmaticConfig().napalm_getters,
        ),
        agent=AgentConfig(
            agent_id=str(os.getenv("AGENT_ID") or agent_raw.get("agent_id") or f"{profile}-agent"),
            profile=profile,
            peer_urls=peers,
        ),
        retrieval=RetrievalConfig(
            backend=str(retrieval_raw.get("backend", "bm25")),
            cache=RetrievalCacheConfig(
                enabled=bool(cache_raw.get("enabled", False)),
                max_entries=int(cache_raw.get("max_entries", 1024)),
                ttl_seconds=float(cache_raw.get("ttl_seconds", 600)),
            ),
            hybrid=HybridConfig(
                bm25_weight=float(hybrid_raw.get("bm25_weight", 0.5)),
                embed_weight=float(hybrid_raw.get("embed_weight", 0.5)),
                fusion=str(hybrid_raw.get("fusion", "weighted_sum")),
                rrf_k=int(hybrid_raw.get("rrf_k", 60)),
                oversample=int(hybrid_raw.get("oversample", 4)),
            ),
            llm_judge=LLMJudgeConfig(
                first_stage_top_k=int(judge_raw.get("first_stage_top_k", 15)),
                timeout_seconds=float(judge_raw.get("timeout_seconds", 10)),
                fusion_alpha=float(judge_raw.get("fusion_alpha", 0.3)),
                max_text_chars=int(judge_raw.get("max_text_chars", 200)),
            ),
            embed_index_concurrency=int(retrieval_raw.get("embed_index_concurrency", 8)),
        ),
        embeddings=EmbeddingsConfig(dim=int(embeddings_raw.get("dim", 768))),
    )


cfg = load(os.getenv("NETOPYU_CONFIG_PATH", "config.yaml"))
