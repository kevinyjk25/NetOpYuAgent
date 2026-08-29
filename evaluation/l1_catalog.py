"""Canonical DSH capability cards used by the P1.8 evaluator."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from dsh_adapter.skills import build_skill_manifest
from network_runtime.contracts import sha256_json
from profiles.base import load_profile
from retrieval.bm25 import BM25Retriever

from .l1_scenarios import WORKFLOW_HINTS


_PARAMETER_LINE = re.compile(r"^- `([^`]+)`: (.+)$", re.MULTILINE)

# Legacy tool cards do not consistently mark every required field.  This map is
# evaluator/catalog metadata only; Runtime remains the execution authority.
TOOL_REQUIRED_PARAMETERS: dict[str, tuple[str, ...]] = {
    "device_info": ("device_id",),
    "dns_lookup": ("hostname",),
    "get_device_config": ("device_id",),
    "validate_device_config": ("device_id",),
    "grant_user_access": ("user_id", "reason"),
    "dc_config_push": ("node", "config_lines", "reason"),
    "wan_failover_path": ("tunnel", "to_transport"),
    "wan_path_sla": ("src", "dst"),
    "wan_route_lookup": ("prefix",),
}

# Capability vocabulary is catalog metadata, not an evaluation answer key.  It
# makes lexical candidate generation usable across the Chinese/English wording
# operators actually use, while the model must still make the final decision.
CAPABILITY_ALIASES: dict[str, tuple[str, ...]] = {
    "lan-new-employee-onboarding-access": (
        "new hire employee onboarding provision application end to end access",
        "新员工 入职 开通 应用访问 准入 权限 可达性 验证",
    ),
    "lan-user-access-diagnose": (
        "user cannot access diagnose admission identity application permission layered",
        "用户 无法访问 诊断 准入 身份 应用权限 分层 排查",
    ),
    "device_info": (
        "device facts hardware model firmware serial uptime",
        "设备信息 硬件 型号 固件 序列号 运行时间 无线 AP",
    ),
    "get_device_config": (
        "show read configuration config section vlan",
        "读取 查看 设备配置 配置段 核心交换机 VLAN",
    ),
    "validate_device_config": (
        "validate check audit current device configuration compliance read only",
        "验证 检查 审计 当前设备配置 合规 只检查 不修改",
    ),
    "dns_lookup": (
        "dns lookup resolve hostname address",
        "DNS 解析 域名 地址",
    ),
    "grant_user_access": (
        "grant restore network admission nac user reason",
        "开通 恢复 网络准入 用户 原因",
    ),
    "dc_config_push": (
        "push apply data center node configuration reason",
        "下发 应用 数据中心 节点 配置 变更原因",
    ),
}


@dataclass(frozen=True)
class L1CatalogEntry:
    target: str
    kind: str
    profile: str
    description: str
    parameters: dict[str, str]
    required_parameters: tuple[str, ...]
    workflow_hint: tuple[str, ...]
    risk_level: str
    requires_approval: bool
    searchable_text: str

    def public_card(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("searchable_text")
        return value


def _skill_parameters(content: str) -> tuple[dict[str, str], tuple[str, ...]]:
    parameters: dict[str, str] = {}
    required: list[str] = []
    for name, description in _PARAMETER_LINE.findall(content):
        parameters[name] = description
        if "optional" not in description.lower() and "可选" not in description:
            required.append(name)
    return parameters, tuple(required)


def build_profile_catalog(profile: str) -> tuple[L1CatalogEntry, ...]:
    if profile not in {"lan", "dc", "wan"}:
        raise ValueError("L1 catalog profile must be lan, dc, or wan")
    entries: list[L1CatalogEntry] = []
    loaded_profile = load_profile(profile)
    for name, raw_tool in sorted(loaded_profile.tool_metadata.items()):
        tool = dict(raw_tool)
        raw_parameters = dict(tool.get("parameters") or {})
        parameters = {
            str(key): str(
                value.get("description") or value.get("type") or "value"
                if isinstance(value, dict)
                else value
            )
            for key, value in raw_parameters.items()
        }
        declared = tuple(
            key for key, value in raw_parameters.items()
            if (
                bool(value.get("required"))
                if isinstance(value, dict)
                else "optional" not in str(value).lower()
                and "可选" not in str(value)
            )
        )
        required = TOOL_REQUIRED_PARAMETERS.get(name, declared)
        description = str(tool.get("description") or "")
        tags = tuple(str(item) for item in tool.get("tags") or ())
        aliases = CAPABILITY_ALIASES.get(name, ())
        entries.append(L1CatalogEntry(
            target=name,
            kind="tool",
            profile=profile,
            description=description,
            parameters=parameters,
            required_parameters=required,
            workflow_hint=(),
            risk_level=str(tool.get("action_type") or "read_only"),
            requires_approval=bool(tool.get("hitl")) or str(
                tool.get("action_type") or "read_only"
            ) != "read_only",
            searchable_text=" ".join((
                name, name.replace("_", " "), description,
                " ".join(parameters), " ".join(tags), " ".join(aliases),
            )),
        ))

    for skill in build_skill_manifest(profile, "mock")["skills"]:
        name = str(skill["name"])
        metadata = dict(skill.get("metadata") or {})
        parameters, required = _skill_parameters(str(skill.get("content") or ""))
        description = str(skill.get("description") or "")
        tags = str(metadata.get("tags") or "").replace(",", " ")
        tool_deps = str(metadata.get("tool_deps") or "").replace(",", " ")
        aliases = CAPABILITY_ALIASES.get(name, ())
        entries.append(L1CatalogEntry(
            target=name,
            kind="skill",
            profile=profile,
            description=description,
            parameters=parameters,
            required_parameters=required,
            workflow_hint=WORKFLOW_HINTS.get(name, ()),
            risk_level=str(metadata.get("risk_level") or "low"),
            requires_approval=str(metadata.get("requires_hitl") or "false").lower() == "true",
            searchable_text=" ".join((
                name, name.replace("-", " "), description,
                " ".join(parameters), tags, tool_deps, " ".join(aliases),
            )),
        ))

    identities = [(item.kind, item.target) for item in entries]
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"duplicate capability in {profile} L1 catalog")
    return tuple(entries)


class L1CandidateRetriever:
    def __init__(self, catalog: tuple[L1CatalogEntry, ...]) -> None:
        self.catalog = catalog
        self._by_id = {f"{item.kind}:{item.target}": item for item in catalog}
        self._retriever = BM25Retriever()
        self._retriever.index([
            {
                "id": f"{item.kind}:{item.target}",
                "text": item.searchable_text,
                "kind": item.kind,
                "target": item.target,
            }
            for item in catalog
        ])

    def retrieve(self, prompt: str, *, top_k: int = 12) -> tuple[L1CatalogEntry, ...]:
        result = self._retriever.retrieve(prompt[:4000], top_k=top_k, min_score=0.0)
        return tuple(self._by_id[item.id] for item in result.matches)


def catalog_digest(catalog: tuple[L1CatalogEntry, ...]) -> str:
    return sha256_json([item.public_card() for item in catalog])


__all__ = [
    "CAPABILITY_ALIASES",
    "L1CandidateRetriever",
    "L1CatalogEntry",
    "TOOL_REQUIRED_PARAMETERS",
    "build_profile_catalog",
    "catalog_digest",
]
