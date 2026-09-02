from evaluation.public_skill_simulation_report import _failure_taxonomy


def _arm(*, failed: set[str], timed_out: bool = False, completed: bool = True) -> dict:
    gates = {
        "protocolPassed", "requiredCapabilitiesPassed", "forbiddenCapabilitiesPassed",
        "parameterBindingPassed", "effectBudgetPassed", "dispositionPassed", "oraclePassed",
    }
    score = {name: name not in failed for name in gates}
    score.update({
        "unsafeExecution": "unsafeExecution" in failed,
        "falseCommit": "falseCommit" in failed,
    })
    return {
        "score": score,
        "observation": {
            "process_timed_out": timed_out,
            "session_completed": completed,
        },
    }


def test_failure_taxonomy_counts_only_failed_gates_and_process_failures() -> None:
    rows = [
        {
            "control": _arm(
                failed={"parameterBindingPassed", "oraclePassed", "unsafeExecution"},
                timed_out=True,
                completed=False,
            ),
            "treatment": _arm(failed=set()),
        },
        {
            "control": _arm(failed={"oraclePassed"}),
            "treatment": _arm(failed={"falseCommit"}),
        },
    ]

    assert _failure_taxonomy(rows, "control") == {
        "oraclePassed": 2,
        "parameterBindingPassed": 1,
        "processTimedOut": 1,
        "sessionIncomplete": 1,
        "unsafeExecution": 1,
    }
    assert _failure_taxonomy(rows, "treatment") == {"falseCommit": 1}
