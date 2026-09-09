"""Local retrieval ranking does not establish semantic resolution."""

import copy

from evaluation.source_gap_search import search_gaps


def page(text, path="reference.md", start=0):
    return {"path": path, "start": start, "end": start + len(text), "text": text}


def test_task_terms_outweigh_numerous_speculative_alternatives():
    pages = {"p000": page("beta gamma delta"), "p001": page("alpha beta", start=10)}
    gaps = [{"missing": "Find `alpha`, or perhaps `beta`, `gamma`, `delta`."}]
    state = {"submitted": [], "window": []}
    before = copy.deepcopy((pages, gaps, state))
    result = search_gaps(gaps, pages, state, task="Inspect alpha only")
    assert result["selectedPages"] == ["p001"]
    assert result["matches"][0]["taskMatchedTermCount"] == 1
    assert result["matches"][1]["score"] > result["matches"][0]["score"]
    for hit in result["matches"]:
        p = pages[hit["pageId"]]
        for m in hit["matchedTerms"]:
            assert p["text"][m["start"] - p["start"]:m["end"] - p["start"]] == m["text"]
    assert (pages, gaps, state) == before
    assert not result["semanticResolutionProven"] and not result["sourceCoverageProven"]


def test_search_excludes_submitted_and_current_pages_without_concluding_absence():
    pages = {"p000": page("bounded key"), "p001": page("bounded key"), "p002": page("irrelevant")}
    result = search_gaps([{"missing": "Read `bounded key`."}], pages, {"submitted": ["p000"], "window": ["p001"]})
    assert not result["matches"] and not result["selectedPages"]
    assert not result["semanticResolutionProven"]


def test_literal_regex_characters_are_not_patterns_and_scripts_are_only_text():
    pages = {"p000": page("a.*b $(arbitrary_command)", "scripts/inert.sh"), "p001": page("axxxb")}
    result = search_gaps([{"missing": "Find `a.*b` and `$(arbitrary_command)`."}], pages, {"submitted": [], "window": []})
    assert result["selectedPages"] == ["p000"] and len(result["matches"]) == 1
    assert result["networkCalls"] == result["sourceScriptCalls"] == 0


def test_case_insensitive_duplicates_term_budget_and_ties_are_explicit():
    gaps = [{"missing": "`Key` `key` " + " ".join(f"`term{i}`" for i in range(30))}]
    pages = {"p002": page("KEY"), "p001": page("KEY")}
    result = search_gaps(gaps, pages, {"submitted": [], "window": []})
    assert len(result["terms"]) == 24 and result["omittedTermCount"] == 7
    assert result["selectedPages"] == ["p001"]


def test_unquoted_free_text_does_not_invent_search_terms():
    assert not search_gaps([{"missing": "Maybe a missing live fact."}], {}, {"submitted": [], "window": []})["terms"]
