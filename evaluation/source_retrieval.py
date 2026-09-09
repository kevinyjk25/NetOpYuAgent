"""Source delivery facts, independent of unverified model interpretations."""

from __future__ import annotations


def visibility(page, current):
    intervals = sorted((max(page["start"], p["start"]), min(page["end"], p["end"]))
                       for p in current.values() if p["path"] == page["path"]
                       and p["start"] < page["end"] and page["start"] < p["end"])
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    count = sum(end - start for start, end in merged)
    return {"visibility": "full" if count == page["end"] - page["start"] else "partial" if count else "absent",
            "currentIntervals": merged}


def delivery_index(pages, current, state):
    """Submitted means included in a recorded request, not understood/approved."""
    return [{"id": key, "path": page["path"], "start": page["start"], "end": page["end"],
             "submittedBefore": key in state["submitted"], **visibility(page, current)}
            for key, page in pages.items()]


def requests_view(state, *, recorded_rounds=None):
    # Keep the raw historical reason, but never use semantic approval as a
    # retrieval to-do bit. Current full windows are about to be supplied.
    offset = state.get("inspectionCalls", 0)
    rounds = len(state["requests"]) + offset + 1 if recorded_rounds is None else recorded_rounds
    return [{"fromPages": r["fromPages"], "requestedPages": r["requestedPages"],
             "reason": r["reason"], "textDelivery": "supplied" if r.get("deliveryRound", i + offset + 1) < rounds else "pending",
             "deliveryRound": r.get("deliveryRound", i + offset + 1) if r.get("deliveryRound", i + offset + 1) < rounds else None}
            for i, r in enumerate(state["requests"])]


def is_repeated_request(state, requested):
    return any(set(r["requestedPages"]) == set(requested) for r in state["requests"])


def decision_phase(state, max_rounds):
    if state.get("decisionReason"):
        return state["decisionReason"]
    if len(state["requests"]) + state.get("inspectionCalls", 0) >= max_rounds - 1:
        return "last_round_requires_candidate_or_gap"
    return None
