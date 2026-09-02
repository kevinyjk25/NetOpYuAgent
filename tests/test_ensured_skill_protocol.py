from pathlib import Path

from evaluation.ensured_skill_protocol import (
    Mechanism,
    Observation,
    ablation_matrix,
    load_scenarios,
    score_case,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_covers_six_authoritative_failure_classes() -> None:
    scenarios = load_scenarios(ROOT / "data/ensured_skill_scenarios.yaml")
    assert len(scenarios) == 6


def test_scorer_does_not_accept_provider_text_without_independent_truth() -> None:
    scenario = load_scenarios(ROOT / "data/ensured_skill_scenarios.yaml")[0]
    score = score_case(scenario, Observation(
        scenario_id=scenario.id,
        terminal="commit",
        effect_dispatched=True,
        provider_state_correct=True,
        independent_verification=False,
        success_claimed=True,
        compensation_attempted=False,
        recovery_verified=False,
        human_escalated=False,
    ))
    assert score.false_commit
    assert not score.task_completed


def test_ablation_protocol_removes_exactly_one_mechanism() -> None:
    matrix = ablation_matrix()
    assert set(matrix["full"]) == set(Mechanism)
    for mechanism in Mechanism:
        candidate = matrix[f"without_{mechanism.value}"]
        assert mechanism not in candidate
        assert len(candidate) == len(Mechanism) - 1


def test_summary_reports_material_metrics_and_claim_boundary() -> None:
    scenarios = load_scenarios(ROOT / "data/ensured_skill_scenarios.yaml")
    scores = [
        score_case(item, Observation(
            scenario_id=item.id,
            terminal=item.expected_terminal,
            effect_dispatched=item.effect_allowed,
            provider_state_correct=True,
            independent_verification=True,
            success_claimed=item.expected_terminal == "commit",
            compensation_attempted=item.compensation_required,
            recovery_verified=item.compensation_required,
            human_escalated=item.human_expected,
            effect_dispatch_count=(
                1 if item.kind.value == "outcome_indeterminate" else 0
            ),
            reconciliation_observed=item.kind.value == "outcome_indeterminate",
            compensation_order_verified=(
                item.kind.value == "partial_multi_step_failure"
            ),
        ))
        for item in scenarios
    ]
    report = summarize(scores)
    assert report["taskCompletionRate"] == 100.0
    assert report["unsafeExecutionRate"] == 0.0
    assert report["falseCommitRate"] == 0.0
    assert "not a production probability" in report["claimBoundary"]


def test_indeterminate_write_requires_one_dispatch_and_reconciliation() -> None:
    scenario = next(
        item for item in load_scenarios(ROOT / "data/ensured_skill_scenarios.yaml")
        if item.kind.value == "outcome_indeterminate"
    )
    score = score_case(scenario, Observation(
        scenario_id=scenario.id,
        terminal="commit",
        effect_dispatched=True,
        provider_state_correct=True,
        independent_verification=True,
        success_claimed=True,
        compensation_attempted=False,
        recovery_verified=False,
        human_escalated=False,
        effect_dispatch_count=2,
        reconciliation_observed=False,
    ))
    assert score.invalid_action
    assert not score.task_completed


def test_partial_failure_requires_reverse_compensation_order() -> None:
    scenario = next(
        item for item in load_scenarios(ROOT / "data/ensured_skill_scenarios.yaml")
        if item.kind.value == "partial_multi_step_failure"
    )
    score = score_case(scenario, Observation(
        scenario_id=scenario.id,
        terminal="abort",
        effect_dispatched=True,
        provider_state_correct=True,
        independent_verification=True,
        success_claimed=False,
        compensation_attempted=True,
        recovery_verified=True,
        human_escalated=False,
        compensation_order_verified=False,
    ))
    assert score.invalid_action
    assert not score.task_completed
