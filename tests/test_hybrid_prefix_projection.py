from copy import deepcopy

import pytest

from evaluation.hybrid_prefix_projection import project
from tests.test_hybrid_live_demo import packet_and_compilation


def prefix():
    packet, compiled, _ = packet_and_compilation()
    read = {k: v for k, v in compiled["plan"]["steps"][0].items() if k != "kind"}
    return packet, compiled["suppliedPages"], {"mode": "read_prefix",
        "intent_summary": compiled["plan"]["intent_summary"], "reads": [read], "boundaries": []}


def test_invalid_annotation_is_audited_without_mutating_good_read_or_granting_admission():
    packet, visible, raw = prefix()
    raw["boundaries"] = [{"evidence": ["p999"], "kind": "missing_host", "explanation": "Unknown page is not a valid citation."}]
    before = deepcopy(raw)
    result = project(packet, visible, raw)
    assert raw == result["originalChoice"] == before
    assert result["projectedChoice"]["reads"] == before["reads"]
    assert len(result["quarantinedAnnotations"]) == 1 and result["newModelCalls"] == 0
    assert not result["automaticPermission"] and result["wholeSkillSemanticSuccess"] is None


def test_invalid_read_rejects_whole_suffix_even_independent_later_valid_read():
    packet, visible, raw = prefix()
    read = deepcopy(raw["reads"][0])
    raw["reads"].extend([{**deepcopy(read), "id": "n1", "tool": "shell"}, {**deepcopy(read), "id": "n2"}])
    result = project(packet, visible, raw)
    assert result["projectedChoice"]["reads"] == [read]
    assert [r["index"] for r in result["rejectedReadSuffix"]] == [1, 2]
    assert "not executed" in result["projectedChoice"]["boundaries"][-1]["explanation"]


@pytest.mark.parametrize("change", ["first_bad", "already_valid", "foreign_field", "other_mode"])
def test_projection_cannot_invent_a_prefix_or_silently_normalize_other_profiles(change):
    packet, visible, raw = prefix()
    if change == "first_bad":
        raw["reads"][0]["tool"] = "shell"
    elif change == "foreign_field":
        raw["approval"] = True
    elif change == "other_mode":
        raw["mode"] = "proposal"
    with pytest.raises(ValueError):
        project(packet, visible, raw)
