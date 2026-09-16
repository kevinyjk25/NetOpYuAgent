import json

import pytest

from evaluation import hybrid_draft_loop as loop, hybrid_snapshot_review as snapshot
from evaluation import hybrid_authoring as author
from evaluation.hybrid_draft_review import assess_review, build_review_input
from evaluation.hybrid_behavior import context
from network_runtime.contracts import sha256_json
from network_runtime.l0.hybrid import qualify_hybrid
from network_runtime.l0.hybrid_execution import HostHybridConsent, HostReasoningBinding, context_digest, run_hybrid
from tests.test_hybrid_draft_review import inputs, raw_review
from tests.test_hybrid_review_roles import role_response


def shape(value):
    if isinstance(value, dict):
        return author.obj({k: shape(v) for k, v in value.items()})
    if isinstance(value, list):
        return {"type": "array", "items": shape(value[0]) if value else {"type": "string"}}
    return {"type": "string"}


@pytest.mark.parametrize("invalid", [False, True])
def test_readless_review_uses_same_bounded_scheduler_without_refresh_or_authority(tmp_path, monkeypatch, invalid):
    original = inputs()
    packet = {"bundle": {"bundleDigest": sha256_json("test source")}, "task": original["original_task"]}
    flow = snapshot.build_snapshot_flow(packet, original, shape(original))
    qualified = qualify_hybrid(flow, {})
    assert all(n.kind == "reason" for n in flow.nodes) and flow.max_model_calls == 3
    costs, assessments = [], {}
    def once(folder, payload, derive, **kwargs):
        request = payload["governedRequest"]
        assert request["observationAgesAtStartMs"] == {} and request["tools"] == []
        review = build_review_input(request["inputs"])
        if request["nodeId"] != "revise-draft":
            candidate = raw_review(review)
            if invalid:
                candidate["claims"].pop()
            candidate = role_response(review, candidate, wire=True)
        else:
            candidate = {"candidate_digest": review["completeCandidateDigest"],
                "evidence_check": {s["draft_span_id"]: "observed_or_justified_inference" for s in review["draftSpans"]},
                "draft": review["candidate"]["draft"], "source_span_ids": ["s000"],
                "notes": request["inputs"]["candidate"]["notes"], "revision_note": "No justified change supplied in this mechanical fixture."}
        envelope = {"httpStatus": 200, "latencyMs": 1, "body": json.dumps({"model": "qwen3.5:9b", "done": True,
            "done_reason": "stop", "message": {"content": json.dumps(candidate)}, "prompt_eval_count": 10, "eval_count": 10})}
        files, cost = derive(envelope)
        return {**files, "result": cost}
    monkeypatch.setattr(loop, "author_once", once)
    invoke = loop.draft_invoker(tmp_path, costs, assessments)
    ctx = context()
    result = run_hybrid(flow, {}, reads={}, read_bindings={}, gates={}, context=ctx,
        consent=HostHybridConsent(qualified["graphDigest"], sha256_json({}), context_digest(ctx)),
        reasoners={"local-draft-reviewer": HostReasoningBinding(author.MODEL, loop.REVIEW_CONFIG_DIGEST, invoke),
                   "local-9b": HostReasoningBinding(author.MODEL, author.CONFIG_DIGEST, invoke)})
    assert len(costs) == (1 if invalid else 3)
    assert result["status"] == ("blocked" if invalid else "governed_graph_completed")
    assert not result["wholeSkillCorrectnessProven"] and not result["effectExecuted"]
    if not invalid:
        assert result["outputs"]["revise-draft"]["value"] == original["candidate"]
        assert assessments["review-after"]["modelReviewHasUnresolvedFindings"]


def test_multiline_code_and_configuration_remain_whole_lexical_blocks():
    original = inputs()
    text = "[project]\nname = 'sample'\nversion = '0.2.0'\n\ndef count(samples):\n    if not samples:\n        raise ValueError('empty')\n    return len(samples)\n"
    original["observations"]["n0"]["observations"] = {"read": {"text": text}}
    payload = build_review_input(original)
    units = [c for c in payload["claims"] if c["facet"] == "observation_to_draft"]
    assert len(units) == 2
    assert units[0]["declaredValue"]["exactQuote"].startswith("[project]")
    assert units[1]["declaredValue"]["exactQuote"].startswith("def count")
    for unit in units:
        assert text[unit["start"]:unit["end"]] == unit["declaredValue"]["exactQuote"]
    assert assess_review(payload, raw_review(payload))["claimCoverage"] == 1
