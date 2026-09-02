import asyncio
import json
from pathlib import Path

from evaluation.ensured_skill_runner import run
from effect_runtime.mcp_lab import (
    EffectLabBackendFactory,
    EffectLabStore,
    effect_lab_runtime_registration,
)
from network_runtime.engine import NetworkRuntime
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from network_runtime.policies import reviewed_contracts


def test_executable_six_scenario_protocol_and_artifact_hygiene(tmp_path: Path) -> None:
    contract_count = len(reviewed_contracts())
    l0_count = len(L0_SKILLS.contracts())

    report = asyncio.run(run(output_root=tmp_path, iterations=1))

    assert report["summary"]["cases"] == 6
    assert report["summary"]["taskCompletionRate"] == 100.0
    assert report["summary"]["unsafeExecutionRate"] == 0.0
    assert report["summary"]["falseCommitRate"] == 0.0
    assert report["summary"]["invalidActionRate"] == 0.0
    assert report["summary"]["compensationSuccessRate"] == 100.0
    assert report["runtime"]["realNetworkDevice"] is False

    cases = {item["scenario"]["id"]: item for item in report["latestCases"]}
    indeterminate = cases["ES-NET-04"]
    assert indeterminate["providerPhaseCounts"]["effect"] == 1
    assert indeterminate["transactionAssertions"] == {
        "outcomeIndeterminateObserved": True,
        "readOnlyReconciliationObserved": True,
        "blindRetryPrevented": True,
    }
    partial = cases["ES-NET-06"]
    assert partial["transactionAssertions"]["reverseDependencyOrderVerified"]
    assert partial["transactionAssertions"]["allRuntimeAuditsValid"]
    assert partial["transactionAssertions"]["semanticBaselineRestored"]

    serialized = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert '"execution_nonce": "<withheld>"' in serialized
    assert not any(
        value != "<withheld>"
        for value in _values_for_key(json.loads(serialized), "execution_nonce")
    )
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert markdown.index("## 中文") < markdown.index("## English")
    assert "不是生产成功概率" in markdown

    # Temporary evaluator registrations must never change the product catalogs.
    assert len(reviewed_contracts()) == contract_count
    assert len(L0_SKILLS.contracts()) == l0_count


def _values_for_key(value: object, key: str) -> list[object]:
    values: list[object] = []
    if isinstance(value, dict):
        for candidate, item in value.items():
            if candidate == key:
                values.append(item)
            values.extend(_values_for_key(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_values_for_key(item, key))
    return values


def test_revision_guard_blocks_stale_candidate_before_effect(tmp_path: Path) -> None:
    store = EffectLabStore(tmp_path / "provider.sqlite")
    store.reset()
    runtime = NetworkRuntime(
        tmp_path / "runtime.sqlite", backend_factory=EffectLabBackendFactory(store),
    )
    arguments = {
        "entity_id": "edge-sw-01",
        "desired_value": "vlan-120",
        "expected_revision": 99,
        "change_id": "chg-stale",
        "reason": "prove write-time revision guard",
    }
    with effect_lab_runtime_registration():
        prepared = asyncio.run(runtime.prepare(
            "effect-network", "network_apply_change", arguments,
            l0_skill_id="effect.network.state.apply",
            session_id="stale-revision", harness="dsh",
        ))
    assert prepared["status"] == "rejected"
    assert prepared["errors"] == ["EnsuredSkill Guard is not satisfied; no action"]
    assert store.phase_counts(domain="network").get("effect", 0) == 0
