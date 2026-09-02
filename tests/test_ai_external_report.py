from evaluation.ai_external_report import _cluster_statistics, _translation_route_statistics


def _row(case_id: str, control: bool, treatment: bool) -> dict:
    return {
        "caseId": case_id,
        "control": {"score": {"passed": control}},
        "treatment": {"score": {"passed": treatment}},
    }


def test_cluster_statistics_treats_skills_as_units() -> None:
    rows = [
        _row("a1", False, True),
        _row("a2", False, True),
        _row("b1", True, True),
        _row("b2", True, True),
    ]
    result = _cluster_statistics(
        rows, {"a1": "skill-a", "a2": "skill-a", "b1": "skill-b", "b2": "skill-b"},
        iterations=1_000,
    )

    assert result["clusterCount"] == 2
    assert result["macroDeltaPercentagePoints"] == 50.0
    assert result["perClusterDeltaPercentagePoints"] == {
        "skill-a": 100.0,
        "skill-b": 0.0,
    }
    assert result["bootstrap95CiPercentagePoints"] == [0.0, 100.0]


def test_translation_route_statistics_separates_safety_from_availability() -> None:
    result = _translation_route_statistics(
        {
            "a": {"route": "l0_runtime"},
            "b": {"route": "safe_stop"},
            "c": {"route": "safe_stop"},
            "d": {"route": "safe_stop"},
        },
        {
            "a": "l0_runtime",
            "b": "l0_runtime",
            "c": "l1_native_read",
            "d": "safe_stop",
        },
    )

    assert result["expectedRouteMatches"] == 2
    assert result["expectedRouteMatchPercent"] == 50.0
    assert result["unsafeRuntimeAccepts"] == 0
    assert result["overSafeStops"] == 2
    assert result["runtimeEligibleRecallPercent"] == 50.0
    assert result["confusionMatrix"]["l1_native_read"]["safe_stop"] == 1
