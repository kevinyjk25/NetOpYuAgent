from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from effect_runtime import inspect_skill_package
from effect_runtime.mcp_lab import DOMAINS, TOOLS
from evaluation.general_effect_ab import run_controlled_ab
from evaluation.general_effect_dataset import (
    FEATURE_FAMILIES, build_cases, materialize_dataset,
)
from evaluation.general_effect_model import _load_existing
from network_runtime.l0_skills import REGISTRY as L0_SKILLS
from network_runtime.policies import reviewed_contracts
from integrations.clients.mcp_client import MCPClient


def _bindings(domain: str, feature: str) -> tuple[str, ...]:
    if feature != "scripts":
        return ()
    return (
        f"scripts/apply.py=effect.{domain}.state.apply",
        f"scripts/rollback.py=effect.{domain}.state.restore",
    )


def test_cross_domain_mcp_catalog_has_six_domains_and_twenty_four_tools() -> None:
    assert len(TOOLS) == 24
    assert {item.domain for item in TOOLS} == set(DOMAINS)
    for domain in DOMAINS:
        values = [item for item in TOOLS if item.domain == domain]
        assert len(values) == 4
        assert {item.role for item in values} == {
            "observation", "preflight", "effect", "restore",
        }


def test_development_corpus_has_sixty_unique_heterogeneous_skills() -> None:
    cases = build_cases()
    assert len(cases) == 60
    assert len({item.case_id for item in cases}) == 60
    assert len({item.skill_id for item in cases}) == 60
    assert {
        family: sum(item.feature_family == family for item in cases)
        for family in FEATURE_FAMILIES
    } == {family: 10 for family in FEATURE_FAMILIES}


def test_materialized_anthropic_skill_packages_pass_package_gate(tmp_path: Path) -> None:
    manifest = materialize_dataset(tmp_path)
    assert manifest["toolCount"] == 24
    assert manifest["skillCount"] == 60
    for case in build_cases():
        report = inspect_skill_package(
            tmp_path / "skills" / case.skill_id,
            bound_scripts=_bindings(case.domain, case.feature_family),
        )
        assert report["gate"] == "passed", (case.case_id, report["findings"])


def test_official_mcp_stdio_server_exposes_and_invokes_domain_tools(tmp_path: Path) -> None:
    async def exercise() -> None:
        client = MCPClient.from_config({
            f"effect-{domain}": {
                "transport": "stdio",
                "command": [
                    sys.executable, "-m", "effect_runtime.mcp_lab",
                    "--domain", domain, "--store", str(tmp_path / "mcp-state.sqlite"),
                ],
                "cwd": str(Path(__file__).resolve().parent.parent),
                "domain": domain, "trusted_for_writes": True,
                "expected_server_name": f"effect-runtime.{domain}",
                "expected_server_version": "1.0.0",
            }
            for domain in DOMAINS
        })
        await client.connect_all()
        try:
            assert len(client.list_tools()) == 24
            for domain in DOMAINS:
                result = await client.call_tool(
                    f"{domain}_get_state",
                    {"entity_id": next(
                        item.arguments["entity_id"]
                        for item in build_cases() if item.domain == domain
                    )},
                )
                assert result.is_error is False
                assert result.structured_content["domain"] == domain
        finally:
            await client.disconnect_all()

    asyncio.run(exercise())


def test_controlled_ab_uses_real_runtime_and_cleans_adapter_registration(tmp_path: Path) -> None:
    before_contracts = len(reviewed_contracts())
    before_l0 = len(L0_SKILLS.contracts())
    report = asyncio.run(run_controlled_ab(output_root=tmp_path))
    direct = report["metrics"]["dsh_l1_direct"]
    guarded = report["metrics"]["dsh_effect_runtime"]
    assert direct["passed"] == 36
    assert direct["falseSuccesses"] == 12
    assert guarded["passed"] == 60
    assert guarded["falseSuccesses"] == 0
    # Six revision-conflict cases are now rejected by the pre-effect revision
    # Guard and therefore correctly produce no post-effect verification.
    assert guarded["independentlyVerified"] == 36
    assert guarded["terminalAudits"] == {"valid": 42, "expected": 42}
    assert report["controlMetrics"]["dsh_effect_runtime"]["unknownOutcomeResolution"] == {
        "passed": 12, "expected": 12, "percent": 100.0,
    }
    assert report["translation"]["packageGatesPassed"] == 60
    assert len(reviewed_contracts()) == before_contracts
    assert len(L0_SKILLS.contracts()) == before_l0


def test_controlled_ab_can_repeat_in_the_same_generated_workspace(tmp_path: Path) -> None:
    first = asyncio.run(run_controlled_ab(output_root=tmp_path, limit=10))
    second = asyncio.run(run_controlled_ab(output_root=tmp_path, limit=10))
    assert first["metrics"]["dsh_effect_runtime"]["passed"] == 10
    assert second["metrics"]["dsh_effect_runtime"]["passed"] == 10


def test_model_checkpoint_is_bound_to_exact_run_fingerprint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model-cases.jsonl"
    checkpoint.write_text(
        "\n".join((
            json.dumps({"case_id": "old", "run_fingerprint": "sha256:old"}),
            json.dumps({"case_id": "current", "run_fingerprint": "sha256:current"}),
            "{interrupted",
        )),
        encoding="utf-8",
    )
    assert _load_existing(
        checkpoint, run_fingerprint="sha256:current",
    ) == {
        "current": {"case_id": "current", "run_fingerprint": "sha256:current"},
    }
