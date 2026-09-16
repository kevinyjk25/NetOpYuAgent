import copy
import json

import pytest

from evaluation.hybrid_review_context import context_fields, read_entry
from evaluation.hybrid_draft_review import build_review_input
from evaluation.hybrid_review_views import review_view
from tests.test_hybrid_draft_review import inputs


def prepared():
    original = inputs()
    original["observations"] = {"prior0": {"observations": {"fetch": {"text": "Index has no samples."}}},
                               "current": {"observations": {"read": {"text": "Window contains 120 requests."}}}}
    entries = [read_entry(region, node, "fetch", {"path": resource},
        original["observations"][region]["observations"][node], "sha256:" + "a" * 64)
        for region, node, resource in (("prior0", "fetch", "/index"), ("current", "read", "/window"))]
    original["read_context"] = json.dumps(entries)
    original["task_scope"] = json.dumps([{"role": "business_request", "text": original["original_task"]}])
    return original, entries


def test_read_resource_context_survives_actual_wire_projection_without_becoming_truth():
    original, entries = prepared()
    payload = build_review_input(original)
    view = review_view(payload)
    assert view["readContext"] == entries
    observed = [s for s in view["sourceSpans"] if s["kind"] == "observation"]
    assert len(observed) == 2
    for row in observed:
        resource = row["readProvenance"]["arguments"]["path"]
        assert ("Index" in row["exactQuote"]) == (resource == "/index")
        assert not row["readProvenance"]["freshnessRenewed"]
    assert view["taskScope"]["classificationIsNotAuthority"]


@pytest.mark.parametrize("mutation", ["payload", "pointer", "duplicate", "authority"])
def test_read_context_binding_rejects_drift(mutation):
    original, entries = prepared()
    if mutation == "payload":
        original["observations"]["prior0"]["observations"]["fetch"]["text"] = "Changed"
    elif mutation == "pointer":
        entries[0]["catalogPointer"] = "/observations/current/read"
    elif mutation == "duplicate":
        entries.append(copy.deepcopy(entries[0]))
    else:
        entries[0]["authorityGranted"] = True
    original["read_context"] = json.dumps(entries)
    with pytest.raises(ValueError):
        context_fields(original)
