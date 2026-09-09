"""Citation IDs eliminate transcription, not semantic selection errors."""

import pytest

from evaluation.source_blocks import citation_blocks, source_span


@pytest.mark.parametrize("text", ["Header\r\n\r\n  原文🙂 quote\n\n", "x" * 5000, "\n\n", "one\ntwo\nthree\n", "Repeat this original.\n\nRepeat this original.\n"])
def test_all_original_characters_reconstruct_including_whitespace_and_duplicates(text):
    pages = {"p000": {"path": "source.md", "start": 73, "end": 73 + len(text), "text": text}}
    blocks = citation_blocks(pages)
    assert "".join(b["text"] for b in blocks.values()) == text
    for key, b in blocks.items():
        assert text[b["start"] - 73:b["end"] - 73] == b["text"]
        assert len(b["text"]) <= 1200
        if len(b["text"]) >= 8:
            span = source_span({"block_id": key}, blocks)
            assert span["quote"] == b["text"] and span["start"] == b["start"]


@pytest.mark.parametrize("mark", [{"block_id": "missing"}, {"block_id": "b0000", "quote": "model revision"}, {"page_id": "p000"}])
def test_invented_id_or_model_quote_cannot_replace_source(mark):
    blocks = citation_blocks({"p000": {"path": "s.md", "start": 0, "text": "Exact original source\n"}})
    with pytest.raises(ValueError):
        source_span(mark, blocks)


def test_existing_but_semantically_wrong_block_is_not_automatically_certified():
    blocks = citation_blocks({"p000": {"path": "s.md", "start": 0, "text": "Read only.\n\nNever write.\n"}})
    span = source_span({"block_id": "b0001"}, blocks)
    assert span["quote"] == "Never write.\n"
    assert set(span) == {"path", "start", "end", "quote"}  # no accuracy/authority claim
