"""Build auditable, single-task model inputs without author IDs or slot order.

The private binding table belongs to the runner. Adapters receive only the
allowlisted public payload; source text is preserved as untrusted inert data.
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from network_runtime.contracts import sha256_json


BLINDING_PROTOCOL = "effect-runtime.io/translation-review-blinding/v2"
AUTHORITY = "development_triage_only_no_gold_or_runtime_authority"
_PAYLOAD_FIELDS = {
    "inputProtocol", "skillId", "untrustedQuotedSkillFiles",
    "declaredNonExecutableToolCatalog", "tasks",
    "candidateExpectedBehaviorHidden", "goldIncluded",
    "thirdPartyContentExecutable", "outputAuthority",
}


def _opaque_id(prefix: str, salt: str, identity: str) -> str:
    digest = hashlib.sha256(f"{prefix}\0{salt}\0{identity}".encode()).hexdigest()
    return f"{prefix}-{digest[:32]}"


def validate_blind_review_payload(payload: dict[str, Any]) -> None:
    """Reject extra metadata and non-anonymous IDs at the adapter boundary."""

    if set(payload) != _PAYLOAD_FIELDS or any((
        payload.get("inputProtocol") != BLINDING_PROTOCOL,
        payload.get("candidateExpectedBehaviorHidden") is not True,
        payload.get("goldIncluded") is not False,
        payload.get("thirdPartyContentExecutable") is not False,
        payload.get("outputAuthority") != AUTHORITY,
        re.fullmatch(r"skill-[0-9a-f]{32}", str(payload.get("skillId"))) is None,
    )):
        raise ValueError("blind review payload metadata or authority mismatch")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("blind review requires exactly one independently reviewed task")
    task = tasks[0]
    if not isinstance(task, dict) or set(task) != {"caseId", "userPrompt"} or any((
        re.fullmatch(r"case-[0-9a-f]{32}", str(task.get("caseId"))) is None,
        not isinstance(task.get("userPrompt"), str),
        not task.get("userPrompt"),
    )):
        raise ValueError("blind review task ID or allowlist mismatch")
    catalog = payload.get("declaredNonExecutableToolCatalog")
    if not isinstance(catalog, dict) or any((
        catalog.get("executable") is not False,
        catalog.get("assignmentId") != payload["skillId"],
        not isinstance(payload.get("untrustedQuotedSkillFiles"), list),
    )):
        raise ValueError("blind review catalog safety or binding mismatch")


def build_blind_review_inputs(
    packets: list[dict[str, Any]], salt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return opaque shuffled inputs and a separate runner-only binding table.

    A stored random salt makes resume reproducible. Sorting salted IDs samples
    the task order independently of labels; the salt and source IDs never enter
    model payloads. Identical tasks may legitimately infer identical outcomes.
    """

    if re.fullmatch(r"[0-9a-f]{64}", salt) is None:
        raise ValueError("blinding salt must be 32 random bytes encoded as hex")
    if not packets:
        raise ValueError("blind review requires at least one source packet")
    source_ids: set[str] = set()
    opaque_ids: set[str] = set()
    packages: dict[str, tuple[Any, Any]] = {}
    entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for packet in packets:
        source_id, package_id = packet.get("caseId"), packet.get("packageId")
        if not isinstance(source_id, str) or not isinstance(package_id, str) or any((
            not source_id, not package_id, source_id in source_ids,
            packet.get("goldIncluded") is not False,
            packet.get("candidateExpectedBehaviorHidden") is not True,
            packet.get("thirdPartyContentExecutable") is not False,
            bool({"expectedBehavior", "expected_behavior", "challenge", "gold"} & set(packet)),
        )):
            raise ValueError("source review packet blindness or identity mismatch")
        source_ids.add(source_id)
        content = (packet["skillFiles"], packet["toolCatalog"])
        if package_id in packages and packages[package_id] != content:
            raise ValueError("source review package content binding drift")
        packages[package_id] = content
        opaque_case = _opaque_id("case", salt, source_id)
        opaque_skill = _opaque_id("skill", salt, package_id)
        if opaque_case in opaque_ids:
            raise ValueError("blind review opaque ID collision")
        opaque_ids.add(opaque_case)
        catalog = copy.deepcopy(packet["toolCatalog"])
        catalog["assignmentId"] = opaque_skill
        payload = {
            "inputProtocol": BLINDING_PROTOCOL,
            "skillId": opaque_skill,
            "untrustedQuotedSkillFiles": copy.deepcopy(packet["skillFiles"]),
            "declaredNonExecutableToolCatalog": catalog,
            "tasks": [{"caseId": opaque_case, "userPrompt": packet["userPrompt"]}],
            "candidateExpectedBehaviorHidden": True,
            "goldIncluded": False,
            "thirdPartyContentExecutable": False,
            "outputAuthority": AUTHORITY,
        }
        validate_blind_review_payload(payload)
        entries.append((payload, {
            "opaqueCaseId": opaque_case,
            "caseId": source_id,
            "packageId": package_id,
            "sourcePacketDigest": sha256_json(packet),
            "modelInputDigest": sha256_json(payload),
        }))
    entries.sort(key=lambda pair: pair[1]["opaqueCaseId"])
    return [pair[0] for pair in entries], {
        "protocol": BLINDING_PROTOCOL,
        "salt": salt,
        "bindings": [pair[1] for pair in entries],
        "modelVisible": False,
    }
