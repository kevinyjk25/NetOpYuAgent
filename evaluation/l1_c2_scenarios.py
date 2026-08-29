"""Versioned adversarial scenarios for the P1.8-C2 system boundary."""

from __future__ import annotations

import json
from pathlib import Path

from network_runtime.contracts import sha256_json

from .l1_contract import L1Scenario


C2_ADVERSARIAL_PATH = Path(__file__).resolve().parents[1] / "data/l1_c2_adversarial.jsonl"


def build_c2_adversarial_scenarios() -> tuple[L1Scenario, ...]:
    scenarios: list[L1Scenario] = []
    for line_number, line in enumerate(
        C2_ADVERSARIAL_PATH.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            scenario = L1Scenario.model_validate(value)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid C2 adversarial scenario line {line_number}") from error
        scenarios.append(scenario)
    identifiers = [item.scenario_id for item in scenarios]
    if len(scenarios) < 20 or len(set(identifiers)) != len(identifiers):
        raise ValueError("C2 adversarial set must contain at least 20 unique cases")
    if any("c2" not in item.tags for item in scenarios):
        raise ValueError("C2 adversarial cases must carry the c2 tag")
    return tuple(scenarios)


def c2_adversarial_digest() -> str:
    return sha256_json([
        item.model_dump(by_alias=True, mode="json")
        for item in build_c2_adversarial_scenarios()
    ])


__all__ = [
    "C2_ADVERSARIAL_PATH",
    "build_c2_adversarial_scenarios",
    "c2_adversarial_digest",
]
