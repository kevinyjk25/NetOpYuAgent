import asyncio
import json
from pathlib import Path

from evaluation.ensured_skill_ablation import run


def test_five_mechanism_ablation_is_executable_and_isolated(tmp_path: Path) -> None:
    report = asyncio.run(run(output_root=tmp_path))
    variants = report["variants"]

    assert set(variants) == {
        "full", "without_contract", "without_evidence", "without_guard",
        "without_transaction", "without_compensation",
    }
    assert variants["full"]["summary"]["taskCompletionRate"] == 100.0
    assert variants["full"]["summary"]["unsafeExecutionRate"] == 0.0
    assert variants["full"]["summary"]["falseCommitRate"] == 0.0
    assert variants["full"]["summary"]["compensationSuccessRate"] == 100.0

    for mechanism in ("contract", "evidence", "guard"):
        value = variants[f"without_{mechanism}"]
        assert value["summary"]["unsafeExecutionRate"] == 20.0
        assert value["summary"]["taskCompletionRate"] == 80.0
        failed = [item for item in value["cases"] if not item["score"]["task_completed"]]
        assert [item["probe"] for item in failed] == [mechanism]

    transaction = variants["without_transaction"]
    assert transaction["summary"]["invalidActionRate"] == 20.0
    transaction_case = next(
        item for item in transaction["cases"] if item["probe"] == "transaction"
    )
    assert transaction_case["providerPhaseCounts"]["effect"] == 2
    assert "OutcomeIndeterminateError" in transaction_case["execution"]["firstError"]
    assert not transaction_case["observation"]["reconciliation_observed"]

    compensation = variants["without_compensation"]
    assert compensation["summary"]["compensationSuccessRate"] == 0.0
    compensation_case = next(
        item for item in compensation["cases"] if item["probe"] == "compensation"
    )
    assert compensation_case["finalState"]["value"] == "__verification_mismatch__"
    assert not compensation_case["observation"]["recovery_verified"]

    assert report["scope"]["switches"] == "evaluation-only"
    assert report["scope"]["realNetworkDevice"] is False
    serialized = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert serialized["claimBoundary"] == report["claimBoundary"]
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert markdown.index("## 中文") < markdown.index("## English")
    assert "不是生产成功概率" in markdown
