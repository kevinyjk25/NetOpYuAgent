import pytest

from evaluation.hybrid_task_context import PREFIX, SUFFIX, separate


def test_authoring_instructions_do_not_become_runtime_business_task():
    task = "Read the future caller document and summarize it; do not write files."
    result, provenance = separate(PREFIX + task + SUFFIX, task)
    assert result == task and "Offline authoring" not in result
    assert not provenance["sourceRestrictionsRemoved"] and not provenance["businessTaskChanged"]
    assert provenance["legacyAuthoringInstructions"]["suffix"] == SUFFIX


def test_unknown_extra_instruction_is_never_silently_deleted():
    task = "Read the future caller document and summarize it; do not write files."
    with pytest.raises(ValueError, match="never strip"):
        separate(PREFIX + task + SUFFIX + " Require additional human approval.", task)
