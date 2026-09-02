from __future__ import annotations

import json

from evaluation.public_skill_draft_author import run_public_market_drafts
from evaluation.public_skill_review import export_assisted_review_kit
from evaluation.public_skill_simulation import build_simulated_authoring_study
from tests.test_public_skill_library import _DraftAdapter, _author_kit


def test_simulated_independent_authoring_is_complete_but_never_official(tmp_path) -> None:
    author = _author_kit(tmp_path / "author")
    drafts = tmp_path / "drafts"
    run_public_market_drafts(author, drafts, model="offline", adapter=_DraftAdapter())
    review = tmp_path / "review"
    export_assisted_review_kit(author, drafts, review)

    result = build_simulated_authoring_study(
        review, author, tmp_path / "simulated-study",
    )
    assert result["humanIndependent"] is False
    assert result["virtualRoleSeparation"] is True
    assert result["caseCount"] == 3
    assert result["officialEsP1QualificationEligible"] is False
    paired = json.loads(
        (tmp_path / "simulated-study/paired-study/workspace.json").read_text()
    )
    assert paired["fixtureMcpExecutableCaseCount"] == 3
    assert paired["officialEsP1QualificationEligible"] is False
