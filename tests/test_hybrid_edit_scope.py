import copy

import pytest

from evaluation.hybrid_edit_scope import observed_quote_locks, check_quote_preservation, quote_blocks


def fixture():
    return {"sourceSpans": [
        {"source_span_id": "o1", "kind": "observation", "exactQuote": "Owner Lina; not resolved. Runbook: https://example.invalid/r42."},
        {"source_span_id": "s1", "kind": "skill", "exactQuote": "Default owner is Admin."}]}, {
        "text": "## Notes\n\n> Owner Lina; not resolved.\n\nAn unverified explanation.\n\n> Runbook: https://example.invalid/r42.\n\n> Default owner is Admin.\n"}


def test_only_exact_observed_quotations_get_located_locks_not_guidance_or_unquoted_prose():
    payload, slot = fixture()
    before = copy.deepcopy((payload, slot))
    locks = observed_quote_locks(payload, slot)
    assert (payload, slot) == before and len(locks) == 2
    assert all(not row["entailmentOrRelevanceProven"] for row in locks)
    for row in locks:
        assert slot["text"][row["start"]:row["end"]].startswith("> ")
        for match in row["sources"]:
            source = next(s for s in payload["sourceSpans"] if s["source_span_id"] == match["sourceSpanId"])
            assert source["exactQuote"][match["start"]:match["end"]] == row["text"]
    revised = slot["text"].replace("An unverified explanation.", "A different explanation, still not approved.")
    revised = revised.replace("> Default owner is Admin.\n", "")
    check_quote_preservation(payload, slot, revised)


@pytest.mark.parametrize("replacement", ["", "> Owner Lina; resolved.\n", "Owner Lina; not resolved.\n"])
def test_source_quote_cannot_be_dropped_changed_or_unmarked_as_prose(replacement):
    payload, slot = fixture()
    revised = slot["text"].replace("> Owner Lina; not resolved.\n", replacement)
    with pytest.raises(ValueError, match="read-only"):
        check_quote_preservation(payload, slot, revised)


def test_multiline_quotes_are_whole_and_code_quote_examples_do_not_create_locks():
    text = "```md\n> Not an actual quotation\n```\n\n> first line\n> second line\n"
    blocks = quote_blocks(text)
    assert len(blocks) == 1 and blocks[0]["text"] == "first line\nsecond line"
    payload = {"sourceSpans": [{"source_span_id": "o1", "kind": "observation", "exactQuote": "first line\nsecond line"}]}
    assert len(observed_quote_locks(payload, {"text": text})) == 1


def test_substrings_inside_words_cannot_claim_observed_quote_support():
    payload = {"sourceSpans": [{"source_span_id": "o1", "kind": "observation", "exactQuote": "setup="}]}
    assert observed_quote_locks(payload, {"text": "> up\n"}) == []
