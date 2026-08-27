"""A2A discovery and one-shot delegation bridge for the DSH provider."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from config import load
from registry.discovery import AgentDiscovery
from registry.schemas import AgentEntry, RegistrationSource


def _unwrap_a2a_event(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate A2A artifacts/messages/status into DSH bridge chunks."""
    output: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return output
    if "error" in raw and "artifact" not in raw and "status" not in raw:
        return [{"error": str(raw["error"])}]
    artifact = raw.get("artifact")
    if isinstance(artifact, dict):
        for part in artifact.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            data = part.get("data")
            if not isinstance(data, dict):
                if part.get("text"):
                    output.append({"token": part["text"]})
                continue
            if data.get("type") == "tokens_batch":
                output.extend({"token": str(token)} for token in data.get("tokens", []) or [])
            elif data.get("token"):
                output.append({"token": data["token"]})
            elif data.get("text"):
                output.append({"message": data["text"]})
        return output
    message = raw.get("message")
    if isinstance(message, dict):
        for part in message.get("parts", []) or []:
            text = part.get("text") if isinstance(part, dict) else None
            if text and text != "Task completed.":
                output.append({"token": text})
        return output
    status = raw.get("status")
    if isinstance(status, dict):
        state = str(status.get("state") or "").lower()
        if state == "failed" and status.get("message"):
            output.append({"error": str(status["message"])})
        elif state in {"input-required", "input_required"}:
            output.append({
                "hitl_interrupt": True,
                "interrupt_id": str(status.get("message") or "").strip(),
            })
    return output


def _configured_peer_urls(peer_urls: list[str] | None = None) -> list[str]:
    if peer_urls is not None:
        values = peer_urls
    elif os.getenv("NETOPYU_DSH_A2A_PEERS", "").strip():
        values = os.environ["NETOPYU_DSH_A2A_PEERS"].split(",")
    else:
        config_path = os.getenv("NETOPYU_CONFIG_PATH", "config.yaml")
        values = load(config_path).agent.peer_urls
    return list(dict.fromkeys(str(value).strip().rstrip("/") for value in values if str(value).strip()))


async def discover_peers(*, peer_urls: list[str] | None = None, timeout_seconds: float = 5.0) -> dict[str, Any]:
    urls = _configured_peer_urls(peer_urls)
    discovery = AgentDiscovery()
    discovery._timeout = max(0.2, float(timeout_seconds))
    entries = await discovery.fetch_many(urls, RegistrationSource.STATIC) if urls else []
    return {
        "ok": True,
        "configured": len(urls),
        "discovered": len(entries),
        "peers": [_entry_payload(entry) for entry in entries],
        "unreachable_urls": [url for url in urls if all(entry.base_url != url for entry in entries)],
    }


def _entry_payload(entry: AgentEntry) -> dict[str, Any]:
    return {
        "agent_id": entry.agent_id,
        "name": entry.card.name,
        "description": entry.card.description,
        "url": entry.base_url,
        "skills": [
            {"id": skill.id, "name": skill.name, "description": skill.description, "tags": skill.tags}
            for skill in entry.card.skills
        ],
    }


def _select_peer(entries: list[AgentEntry], *, target: str, capability: str, own_agent_id: str) -> tuple[AgentEntry, str]:
    candidates = [entry for entry in entries if entry.agent_id != own_agent_id]
    if target:
        for entry in candidates:
            if target in {entry.agent_id, entry.card.name}:
                skill_id = entry.card.skills[0].id if entry.card.skills else "delegated_task"
                return entry, skill_id
        raise LookupError(f"A2A peer not found: {target}")
    query = capability.strip().casefold()
    if not query:
        if len(candidates) == 1:
            entry = candidates[0]
            return entry, entry.card.skills[0].id if entry.card.skills else "delegated_task"
        raise LookupError("target or capability is required when more than one A2A peer is available")
    ranked: list[tuple[int, AgentEntry, str]] = []
    terms = {part for part in query.replace("_", " ").replace("-", " ").split() if part}
    for entry in candidates:
        for skill in entry.card.skills:
            fields = " ".join([skill.id, skill.name, skill.description, *skill.tags]).casefold()
            exact = query in {skill.id.casefold(), skill.name.casefold(), *(tag.casefold() for tag in skill.tags)}
            score = (100 if exact else 0) + (30 if query in fields else 0) + sum(5 for term in terms if term in fields)
            if score:
                ranked.append((score, entry, skill.id))
    if not ranked:
        raise LookupError(f"no A2A peer advertises capability: {capability}")
    ranked.sort(key=lambda item: (-item[0], item[1].agent_id, item[2]))
    return ranked[0][1], ranked[0][2]


async def delegate_a2a(
    *,
    prompt: str,
    target: str = "",
    capability: str = "",
    session_id: str,
    own_agent_id: str,
    delegation_chain: list[str] | None = None,
    peer_urls: list[str] | None = None,
    timeout_seconds: float = 300.0,
    max_hops: int = 3,
    resume_interrupt_id: str = "",
    operator_decision: str = "",
) -> dict[str, Any]:
    chain = [str(item).strip() for item in (delegation_chain or []) if str(item).strip()]
    if len(chain) >= max(1, int(max_hops)):
        return {"ok": False, "status": "refused", "error": f"A2A delegation depth limit reached ({max_hops})"}
    if own_agent_id in chain:
        return {"ok": False, "status": "refused", "error": f"A2A delegation loop detected at {own_agent_id}"}

    urls = _configured_peer_urls(peer_urls)
    if not urls:
        return {"ok": False, "status": "unavailable", "error": "no A2A peers are configured"}
    discovery = AgentDiscovery()
    discovery._timeout = min(max(0.2, float(timeout_seconds)), 10.0)
    entries = await discovery.fetch_many(urls, RegistrationSource.STATIC)
    try:
        peer, skill_id = _select_peer(entries, target=target.strip(), capability=capability.strip(), own_agent_id=own_agent_id)
    except LookupError as error:
        return {"ok": False, "status": "unavailable", "error": str(error), "discovered_peers": [entry.agent_id for entry in entries]}
    if peer.agent_id in chain or peer.agent_id == own_agent_id:
        return {"ok": False, "status": "refused", "error": f"A2A delegation loop would revisit {peer.agent_id}"}

    resume_interrupt_id = str(resume_interrupt_id).strip()
    operator_decision = str(operator_decision).strip().lower()
    if bool(resume_interrupt_id) != bool(operator_decision):
        return {"ok": False, "status": "refused", "error": "resume interrupt and operator decision must be supplied together"}
    if operator_decision and operator_decision not in {"approve", "reject"}:
        return {"ok": False, "status": "refused", "error": "operator decision must be approve or reject"}

    body = {
        "jsonrpc": "2.0",
        "method": "message/stream",
        "params": {
            "message": {"kind": "message", "role": "user", "message_id": session_id, "parts": [{"kind": "text", "text": prompt}]},
            "context_id": session_id,
            "metadata": {
                "session_id": session_id,
                "delegated_by": own_agent_id,
                "source_agent": own_agent_id,
                "source_session_id": session_id,
                "delegation_chain": [*chain, own_agent_id],
                "delegation_depth": len(chain) + 1,
                "dsh_provider": "netopyu-a2a",
                **({
                    "resume_interrupt_id": resume_interrupt_id,
                    "operator_decision": operator_decision,
                } if resume_interrupt_id else {}),
            },
        },
        "id": 1,
    }
    output: list[str] = []
    interrupt_id = ""
    error_text = ""
    stream_url = peer.base_url.rstrip("/") + "/stream"
    try:
        timeout = httpx.Timeout(float(timeout_seconds), connect=min(10.0, float(timeout_seconds)))
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", stream_url, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        raw = json.loads(data)
                    except json.JSONDecodeError:
                        output.append(data)
                        continue
                    for chunk in _unwrap_a2a_event(raw):
                        if chunk.get("hitl_interrupt"):
                            interrupt_id = str(chunk.get("interrupt_id") or "")
                        if chunk.get("error"):
                            error_text = str(chunk["error"])
                        text = chunk.get("token") or chunk.get("message")
                        if text:
                            output.append(str(text))
    except (httpx.HTTPError, asyncio.TimeoutError) as error:
        return {"ok": False, "status": "error", "peer": peer.agent_id, "error": str(error) or type(error).__name__}

    result = {
        "ok": not bool(error_text or interrupt_id),
        "status": "input-required" if interrupt_id else ("error" if error_text else "completed"),
        "peer": peer.agent_id,
        "peer_url": peer.base_url,
        "skill_id": skill_id,
        "text": "".join(output).strip(),
    }
    if interrupt_id:
        result.update({"interrupt_id": interrupt_id, "error": "remote peer requires its operator approval"})
    elif error_text:
        result["error"] = error_text
    return result
