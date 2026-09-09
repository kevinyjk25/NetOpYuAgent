"""Protocol and omission-review integrity, not calibrated semantic accuracy."""

import json

import pytest

from evaluation.flow_local_demo import build_local_flow
from evaluation.flow_translation import (
    FlowDraft, FlowRevision, IndexedDraft, apply_revision, assess, diagnose, load_run,
    local_sources, lower, review_input, revision_packet, run_reviewed,
)
from network_runtime.contracts import sha256_json
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_source_alignment import ClaimAssessment, SourceAssessment


def fixture():
    sources = local_sources()
    flow, reads = build_local_flow()
    raw = flow.model_dump(mode="json")
    for node in raw["nodes"]:
        if node["kind"] == "read":
            node["tool"] = reads[node.pop("contract_hash")].spec.tool
    return sources, FlowDraft(purpose=flow.purpose, entry=flow.entry, nodes=raw["nodes"], unresolved_questions=())


def review_for(packet, verdict="supported"):
    return ReadL05Review(reviewer_id="mechanical-fixture-not-semantic-review", reviewer_kind="test_fixture",
        assessment=SourceAssessment(input_digest=packet["inputDigest"], scope_note="Protocol fixture only, no independent truth assessment.",
            claims=tuple(ClaimAssessment(claim_id=claim["claimId"], verdict=verdict,
                source_span_ids=tuple(claim.get("requiredCitationId") if role == "skill" and claim.get("requiredCitationId") else role + "-0001"
                                     for role in claim["requiredEvidenceKinds"]),
                rationale="Mechanical protocol fixture, not source semantic proof.", suggested_revision="Inspect original source and revise the flow.")
                for claim in packet["claims"])))


def test_lower_preserves_edges_references_and_binds_real_host_hash():
    sources, draft = fixture()
    flow, qualified = lower(sources, draft)
    assert flow.nodes[1].on_true == draft.nodes[1].on_true
    assert flow.nodes[2].arguments == draft.nodes[2].arguments
    assert flow.nodes[0].contract_hash == sources.reads[draft.nodes[0].tool].contract_hash
    assert not qualified["runtimeAuthorityGranted"]


def test_review_includes_source_coverage_and_each_node():
    sources, draft = fixture()
    packet = review_input(sources, draft)
    for span in packet["sourceSpans"]:
        if span["kind"] == "skill":
            assert sources.source_text[span["start"]:span["end"]] == span["exactQuote"]
            if not span["exactQuote"].lstrip().startswith("#"):
                assert any(row.get("requiredCitationId") == span["source_span_id"] for row in packet["claims"])
    for index in range(len(draft.nodes)):
        assert any(row["l05Pointer"] == f"/nodes/{index}" for row in packet["claims"])
    report = assess(sources, draft, review_for(packet))
    assert report["status"] == "review_supported_inactive_flow" and report["flow"]
    assert not report["runtimeAuthorityGranted"] and not report["semanticAlignmentProven"]


@pytest.mark.parametrize("mutation", ["missing_claim", "duplicate_claim", "wrong_source", "wrong_digest"])
def test_review_cannot_skip_or_relabel_required_evidence(mutation):
    sources, draft = fixture()
    packet = review_input(sources, draft)
    raw = review_for(packet).model_dump(mode="json")
    if mutation == "missing_claim":
        raw["assessment"]["claims"].pop()
    elif mutation == "duplicate_claim":
        raw["assessment"]["claims"].append(raw["assessment"]["claims"][0])
    elif mutation == "wrong_source":
        raw["assessment"]["claims"][0]["source_span_ids"] = ["skill-0001"]
    else:
        raw["assessment"]["input_digest"] = "wrong"
    with pytest.raises(ValueError):
        assess(sources, draft, ReadL05Review.model_validate(raw))


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient_evidence"])
def test_review_disagreement_blocks_without_execution(verdict):
    sources, draft = fixture()
    report = assess(sources, draft, review_for(review_input(sources, draft), verdict))
    assert report["status"] == "blocked" and report["flow"] is None


def test_unresolved_questions_block_even_if_claims_supported():
    sources, draft = fixture()
    draft = draft.model_copy(update={"unresolved_questions": ("Missing source fact?",)})
    assert assess(sources, draft, review_for(review_input(sources, draft)))["status"] == "blocked"


def test_omitted_valid_subgraph_still_requires_source_requirement_review():
    sources, draft = fixture()
    raw = draft.model_dump(mode="json")
    raw["nodes"] = [raw["nodes"][0], raw["nodes"][3]]
    raw["nodes"][0]["next"] = raw["nodes"][1]["id"]
    shorter = FlowDraft.model_validate(raw)
    packet = review_input(sources, shorter)
    assert sum("requiredCitationId" in row for row in packet["claims"]) == sum("requiredCitationId" in row for row in review_input(sources, draft)["claims"])
    # Structural validity must not determine the semantic reviewer verdict.
    assert assess(sources, shorter, review_for(packet, "contradicted"))["status"] == "blocked"


@pytest.mark.parametrize("mutation", ["source", "branch", "questions"])
def test_source_or_proposal_change_invalidates_old_review(mutation):
    sources, draft = fixture()
    review = review_for(review_input(sources, draft))
    if mutation == "source":
        sources = sources.model_copy(update={"source_text": sources.source_text + "\nAdditional safety requirement.\n"})
    else:
        raw = draft.model_dump(mode="json")
        if mutation == "branch":
            node = raw["nodes"][1]
            node["on_true"], node["on_false"] = node["on_false"], node["on_true"]
        else:
            raw["unresolved_questions"] = ["New question"]
        draft = FlowDraft.model_validate(raw)
    with pytest.raises(ValueError):
        assess(sources, draft, review)


def test_saved_draft_must_match_raw_model_reply(tmp_path):
    sources, draft = fixture()
    for name, value in {"sources.json": sources.model_dump(mode="json"), "draft.json": draft.model_dump(mode="json"),
                        "review-input.json": review_input(sources, draft),
                        "response.json": {"httpStatus": 200, "body": json.dumps({"message": {"content": draft.model_dump_json()}})}}.items():
        (tmp_path / name).write_text(json.dumps(value))
    assert load_run(tmp_path) == (sources, draft)
    raw = draft.model_dump(mode="json")
    raw["purpose"] = "Silently edited purpose"
    (tmp_path / "draft.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="original model"):
        load_run(tmp_path)


def test_structural_diagnostics_locate_errors_without_repair():
    sources, draft = fixture()
    raw = draft.model_dump(mode="json")
    raw["entry"] = "device_id"
    raw["nodes"][0]["arguments"]["device_id"]["source"] = "entry"
    raw["nodes"][1]["on_true"] = raw["nodes"][1]["id"]
    raw["nodes"][1]["on_false"] = "missing"
    raw["nodes"][3] = {"kind": "effect_candidate", "id": "done", "binding_id": "needs_l1", "arguments": {}}
    draft = FlowDraft.model_validate(raw)
    before = draft.model_dump_json()
    assert {row["code"] for row in diagnose(sources, draft)} >= {
        "entry_not_node_id", "unknown_reference_source", "self_loop", "unknown_successor", "unknown_host_effect_target"}
    assert draft.model_dump_json() == before


@pytest.mark.parametrize("mode", ["success", "no_host_approval", "blocked_review", "outside_scope"])
def test_reviewed_local_runner_rechecks_authority_and_original_reply(tmp_path, mode):
    sources, draft = fixture()
    packet = review_input(sources, draft)
    for name, value in {"sources.json": sources.model_dump(mode="json"), "draft.json": draft.model_dump(mode="json"),
                        "review-input.json": packet,
                        "response.json": {"httpStatus": 200, "body": json.dumps({"message": {"content": draft.model_dump_json()}})}}.items():
        (tmp_path / name).write_text(json.dumps(value))
    review = review_for(packet, "contradicted" if mode == "blocked_review" else "supported")
    if mode == "success":
        result = run_reviewed(tmp_path, review, device_id="campus-sw1", allow_local_read=True)
        assert result["execution"]["status"] == "read_path_completed"
        assert not result["dshAgentLoopExecuted"]
    else:
        with pytest.raises(PermissionError):
            run_reviewed(tmp_path, review, device_id="unlisted" if mode == "outside_scope" else "campus-sw1",
                         allow_local_read=mode != "no_host_approval")


def indexed_fixture():
    sources, draft = fixture()
    raw = draft.model_dump(mode="json")
    indices = {node["id"]: index for index, node in enumerate(raw["nodes"])}
    raw["entry"] = indices[raw["entry"]]
    for node in raw["nodes"]:
        node.pop("id")
        for key in ("next", "on_true", "on_false"):
            if key in node:
                node[key] = indices[node[key]]
        for value in list(node.get("arguments", {}).values()) + ([node["left"]] if "left" in node else []):
            if value["kind"] == "reference" and value["source"] != "input":
                value["source"] = indices[value["source"]]
    return sources, raw


def test_index_projection_only_renames_preserving_paths_and_values():
    sources, raw = indexed_fixture()
    draft = IndexedDraft.model_validate(raw).named()
    flow, _ = lower(sources, draft)
    assert flow.entry == "node-0"
    assert flow.nodes[1].on_true == "node-2" and flow.nodes[1].on_false == "node-4"
    assert flow.nodes[2].arguments["device_id"].source == "node-0"
    assert flow.nodes[1].equals.value == "campus"


@pytest.mark.parametrize("index", [-1, 64, True, "1"])
def test_index_grammar_does_not_coerce_invalid_identity(index):
    _, raw = indexed_fixture()
    raw["entry"] = index
    with pytest.raises(ValueError):
        IndexedDraft.model_validate(raw)


@pytest.mark.parametrize("target", [1, 63])
def test_index_projection_does_not_repair_cycles_or_missing_nodes(target):
    sources, raw = indexed_fixture()
    raw["nodes"][1]["on_true"] = target
    draft = IndexedDraft.model_validate(raw).named()
    assert draft.nodes[1].on_true == f"node-{target}"
    with pytest.raises(ValueError):
        lower(sources, draft)


def revision_fixture(sources, draft):
    return FlowRevision(parent_digest=sha256_json(draft.model_dump(mode="json")),
        sources_digest=sha256_json(sources.model_dump(mode="json")), editor_id="test-only", editor_kind="test_fixture",
        edits=[{"pointer": "/purpose", "old_value": draft.purpose, "new_value": "Read local inventory, not live health.",
                "source_quotes": ["Inventory is local planned data, not live health. Host-granted device scopes"]}])


def test_text_revision_preserves_original_graph_and_requires_new_review():
    sources, draft = fixture()
    revision = revision_fixture(sources, draft)
    child = apply_revision(sources, draft, revision)
    assert child.nodes == draft.nodes and child.unresolved_questions == draft.unresolved_questions
    assert child.purpose != draft.purpose
    with pytest.raises(ValueError):
        assess(sources, draft, review_for(review_input(sources, draft)), revision)
    report = assess(sources, draft, review_for(revision_packet(sources, draft, revision)), revision)
    assert report["status"] == "review_supported_inactive_flow" and report["revisionDigest"]
    assert not report["runtimeAuthorityGranted"]


@pytest.mark.parametrize("mutation", ["parent", "sources", "quote", "old_value", "graph", "duplicate", "index"])
def test_revision_cannot_rewrite_semantics_or_invent_lineage(mutation):
    sources, draft = fixture()
    raw = revision_fixture(sources, draft).model_dump(mode="json")
    if mutation in {"parent", "sources"}:
        raw[mutation + "_digest"] = "wrong"
    elif mutation == "quote":
        raw["edits"][0]["source_quotes"] = ["invented evidence"]
    elif mutation == "old_value":
        raw["edits"][0]["old_value"] = "not original"
    elif mutation == "graph":
        raw["edits"][0]["pointer"] = "/nodes/1/on_true"
    elif mutation == "index":
        raw["edits"][0]["pointer"] = "/nodes/99/explanation"
    else:
        raw["edits"].append(raw["edits"][0])
    with pytest.raises(ValueError):
        apply_revision(sources, draft, FlowRevision.model_validate(raw))
