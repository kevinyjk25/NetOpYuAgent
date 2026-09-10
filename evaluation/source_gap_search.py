"""Bounded literal retrieval from retained inert pages, never semantic resolution."""

from __future__ import annotations

import re

MAX_TERMS = 24


def search_gaps(gaps, pages, state, *, task=""):
    """Use quoted/backtick terms, not provider calls or domain-specific answers."""
    terms = []
    for gap in gaps:
        for match in re.finditer(r"`([^`\n]{2,120})`|'([^'\n]{2,120})'|\"([^\"\n]{2,120})\"", gap["missing"]):
            term = next(value for value in match.groups() if value is not None).strip()
            if len(term) >= 2 and term.casefold() not in {t.casefold() for t in terms}:
                terms.append(term)
    selected = terms[:MAX_TERMS]
    excluded = set(state["submitted"]) | set(state["window"])
    hits = []
    for key, page in pages.items():
        if key in excluded:
            continue
        matches = []
        for term in selected:
            match = re.search(re.escape(term), page["text"], re.IGNORECASE)
            if match:
                matches.append({"term": term, "start": page["start"] + match.start(),
                                "end": page["start"] + match.end(), "text": match.group()})
        path_terms = [term for term in selected if "/" in term and
                      (page["path"] == term or page["path"].endswith("/" + term))]
        if matches or path_terms:
            hits.append({"pageId": key, "path": page["path"], "matchedTerms": matches,
                         "matchedPathTerms": path_terms,
                         "taskMatchedTermCount": sum(term.casefold() in task.casefold()
                                                     for term in {m["term"] for m in matches} | set(path_terms)),
                         "score": len(matches) + len(path_terms)})
    # A model's speculative alternatives should not outrank explicit task terms
    # merely by being more numerous. Neither kind of literal is a truth claim.
    hits.sort(key=lambda h: (-h["taskMatchedTermCount"], -h["score"], h["pageId"]))
    return {"strategy": "task-prioritized-gap-literals-and-paths/v3", "terms": selected,
            "omittedTermCount": max(0, len(terms) - MAX_TERMS), "excludedPages": sorted(excluded),
            "matches": hits[:5], "selectedPages": [hits[0]["pageId"]] if hits else [],
            "semanticResolutionProven": False, "sourceCoverageProven": False,
            "networkCalls": 0, "sourceScriptCalls": 0}
