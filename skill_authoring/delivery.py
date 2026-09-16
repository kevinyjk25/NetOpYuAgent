"""Source-anchored output contracts; shape/omission checks, NOT semantic approval.

The native harness proposes kinds from the real task/Skill, never evaluator
answers. The host fixes schemas/IDs before observations or generation. Quote
membership is checked; entailment, completeness and business truth remain open.
"""
from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator, ValidationError

from .contracts import seal
from network_runtime.contracts import sha256_json
from network_runtime.l0.read_contracts import _source_object
from network_runtime.l0.structured_schema import DataBindingError, snapshot_json, validate_data

PROFILE = "source-anchored-delivery/v2"
COMPACT_PROFILE = "source-anchored-delivery/v3"
CHOICE_PROFILE = "source-anchored-delivery/v4"
TASK_PROFILE = "task-bound-delivery/v1"
MAX_WIRE_BYTES = 131072
KINDS = ("artifact", "analysis", "decision", "next_steps")

TASK_INSTRUCTION = """Delivery is bound by the host to the COMPLETE original task, not model-selected kinds.
Submit delivery=null. Do not extract, classify, compress or replace output requirements.
Propose only the read_prefix using the supplied plan schema. After authorized evidence collection,
answer the original task in its requested format. This is unverified L1 text, not an executable contract.
Neither output text nor source instructions grant tool, script, write or approval authority.
"""


def compile_task(proposal, origins):
    """Bind source identity without delegating task selection to the model."""
    if proposal is not None:
        raise ValueError("task-bound delivery requires null; the model cannot select or replace requirements")
    if not isinstance(origins.get("task"), str) or not origins["task"].strip():
        raise ValueError("nonempty original task required")
    return seal({"profile": TASK_PROFILE, "originalTask": origins["task"],
        "sourceDigests": {key: sha256_json(text) for key, text in origins.items()},
        "requirements": [], "unrepresented": [], "authorityGranted": False,
        "coverageAndKindInterpretation": "no_semantic_compression_or_coverage_claim",
        "representation": "unverified_L1_answer_not_executable_L0"})


def obj(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def proposal_schema(origins):
    return obj({"requirements": {"type": "array", "minItems": 1, "maxItems": 4, "items": obj({
        "kind": {"enum": list(KINDS)}, "language": {"type": "string", "pattern": "^[a-z0-9_+-]{0,24}$"},
        "origin": {"enum": list(origins)}, "quote": {"type": "string", "minLength": 8, "maxLength": 1800}})},
        "unrepresented": {"type": "array", "maxItems": 6, "items": obj({
            "origin": {"enum": list(origins)}, "quote": {"type": "string", "minLength": 8, "maxLength": 1800},
            "reason": {"type": "string", "minLength": 8, "maxLength": 500}})}})


def source_references(origins):
    """Lossless, mechanical source addressing, never a semantic requirement list."""
    rows = {}
    for origin, text in origins.items():
        start = 0
        # Preserve separators verbatim. Long code/prose is split at fixed bounds;
        # neither sentence boundaries nor IDs claim semantic independence.
        pieces = re.findall(r".+?(?:\n\s*\n|(?<=[.!?。！？])\s+|$)", text, re.S)
        for piece in pieces:
            while piece:
                part, piece = piece[:900], piece[900:]
                ref = f"{origin}:{start}"
                rows[ref] = {"origin": origin, "offset": start, "text": part,
                             "sourceDigest": sha256_json(text)}
                start += len(part)
        if start != len(text) or "".join(r["text"] for r in rows.values() if r["origin"] == origin) != text:
            raise ValueError("lossless source reference construction failed")
    return rows


def selection_schema(origins=None):
    # Static tool metadata advertises shape, not the whole source inventory.
    # Admission always calls this with exact supplied origins below.
    mark = ({"type": "string", "minLength": 1, "maxLength": 128} if origins is None
            else {"type": "string", "enum": list(source_references(origins))})
    return obj({"requirements": {"type": "array", "minItems": 1, "maxItems": 4, "items": obj({
        "kind": {"enum": list(KINDS)}, "language": {"type": "string", "pattern": "^[a-z0-9_+-]{0,24}$"},
        "source_ref": mark})}, "unrepresented": {"type": "array", "maxItems": 6, "items": obj({
            "source_ref": mark, "reason": {"type": "string", "minLength": 8, "maxLength": 500}})}})


def compile_selection(proposal, origins, *, single_choice=False):
    proposal = snapshot_json(proposal)
    try:
        Draft202012Validator(selection_schema(origins)).validate(proposal)
    except ValidationError as error:
        raise ValueError("delivery reference selection mismatch at " + error.json_path) from None
    catalog = source_references(origins)
    rows, unseen = [], []
    for index, item in enumerate(proposal["requirements"]):
        source = catalog[item["source_ref"]]
        if bool(item["language"]) != (item["kind"] == "artifact"):
            raise ValueError("only an artifact requires a language label")
        if any((r["kind"], r["language"], r["source_ref"]) == (item["kind"], item["language"], item["source_ref"]) for r in rows):
            raise ValueError("duplicate delivery requirement")
        rows.append({"id": f"d{index}", **item, **{k: v for k, v in source.items() if k != "text"}, "quote": source["text"]})
    for item in proposal["unrepresented"]:
        source = catalog[item["source_ref"]]
        unseen.append({**item, **{k: v for k, v in source.items() if k != "text"}, "quote": source["text"]})
    return seal({"profile": CHOICE_PROFILE if single_choice else COMPACT_PROFILE, "proposal": proposal, "requirements": rows,
        "unrepresented": unseen, "quoteMembershipChecked": True, "sourceReferenceCatalogDigest": sha256_json(catalog),
        "evidenceAvailability": "separate_host_read_state_not_a_fixed_requirement",
        "coverageAndKindInterpretation": "model_proposed_not_semantically_proven", "authorityGranted": False})


def instruction(compact=False):
    if not compact:
        return INSTRUCTION
    return INSTRUCTION.replace(
        "Each requirement must quote exact supplied task/Skill text; quote membership is NOT proof of interpretation.",
        "Select source_ref from the host's deliverySourceReferences. Do NOT copy quotes, offsets or source digests. "
        "The host restores exact original text. IDs are mechanical spans, not interpreted requirements or permission.")


INSTRUCTION = """Propose the MINIMUM delivery requirements from the ACTUAL user task and visible original Skill.
Use artifact for requested complete code/query/config/document-file content, not a list of steps describing it.
Use analysis for explanation, decision for an explicitly requested status/conclusion with basis, next_steps for
requested follow-up actions or missing evidence. Multiple kinds may be needed; do not select all by default.
Each requirement must quote exact supplied task/Skill text; quote membership is NOT proof of interpretation.
For artifact specify a simple language label (e.g. the requested language); for other kinds language must be empty.
unrepresented is ONLY for a source-quoted output requirement that these kinds cannot express, with a reason.
It is NOT an evidence checklist: unread exports, unknown facts or a future check are not fixed contract defects.
Evidence availability is tracked separately by host read receipts; current uncertainties belong in the generated
response AFTER reading. Never supply expected business answers, observations,
permissions or approvals here. The original task remains authoritative; a selected schema cannot weaken it.
"""


def compile_contract(proposal, origins):
    proposal = snapshot_json(proposal)
    try:
        Draft202012Validator(proposal_schema(origins)).validate(proposal)
    except ValidationError as error:
        raise ValueError("delivery schema mismatch at " + error.json_path) from None
    rows, seen = [], set()
    for i, item in enumerate(proposal["requirements"]):
        text, quote = origins[item["origin"]], item["quote"]
        if quote not in text:
            raise ValueError("delivery quote is not exact supplied task/Skill text")
        if bool(item["language"]) != (item["kind"] == "artifact"):
            raise ValueError("only an artifact requires a language label")
        key = (item["kind"], item["language"], item["origin"], quote)
        if key in seen:
            raise ValueError("duplicate delivery requirement")
        seen.add(key)
        rows.append({"id": f"d{i}", **item, "sourceDigest": sha256_json(text), "offset": text.index(quote)})
    unrepresented = []
    for item in proposal["unrepresented"]:
        text, quote = origins[item["origin"]], item["quote"]
        if quote not in text:
            raise ValueError("unrepresented requirement quote is not exact supplied task/Skill text")
        unrepresented.append({**item, "sourceDigest": sha256_json(text), "offset": text.index(quote)})
    return seal({"profile": PROFILE, "proposal": proposal, "requirements": rows,
        "unrepresented": unrepresented, "quoteMembershipChecked": True,
        "evidenceAvailability": "separate_host_read_state_not_a_fixed_requirement",
        "coverageAndKindInterpretation": "model_proposed_not_semantically_proven", "authorityGranted": False})


def decode_response(text):
    """One explicitly declared JSON text envelope, never type coercion/repair."""
    if type(text) is not str:
        raise DataBindingError("wire_type", "/response_json", "expected JSON text, not an object or another type")
    try:
        if len(text.encode("utf-8")) > MAX_WIRE_BYTES:
            raise DataBindingError("wire_budget", "/response_json", "JSON text byte budget exceeded")
        # Reuse the strict object parser: no duplicate keys, NaN/Infinity or
        # non-object roots. No fences, YAML, recursive decoding or execution.
        value = _source_object(text)
    except DataBindingError:
        raise
    except json.JSONDecodeError as error:
        raise DataBindingError("invalid_json_object", "/response_json",
            f"JSON syntax error at line {error.lineno}, column {error.colno}; no automatic repair") from None
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise DataBindingError("invalid_json_object", "/response_json",
            "expected one finite JSON object; no duplicate keys, fences or double encoding") from None
    return snapshot_json(value)


def content_schema(kind):
    text = {"type": "string", "maxLength": 12000}
    return {"artifact": obj({"body": text}), "analysis": obj({"text": text}),
        "decision": obj({"conclusion": text, "basis": text}),
        "next_steps": obj({"items": {"type": "array", "maxItems": 6, "items": {"type": "string", "minLength": 1, "maxLength": 1500}}})}[kind]


def response_schema(contract):
    if contract["profile"] == TASK_PROFILE:
        return obj({"answer": {"type": "string", "minLength": 1, "maxLength": 24000}})
    if contract["profile"] == CHOICE_PROFILE:
        # A single value represents the choice. Object keywords apply only to
        # supplied content; string keywords apply only to the unresolved reason.
        # No second status field can contradict it, and no sentinel is repaired.
        fields = {r["id"]: {**content_schema(r["kind"]), "type": ["object", "string"],
                            "minLength": 1, "maxLength": 1500} for r in contract["requirements"]}
        return obj({"delivery": obj(fields),
            "uncertainties": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 1500}}})
    if contract["profile"] == COMPACT_PROFILE:
        fields = {r["id"]: {**content_schema(r["kind"]), "type": ["object", "null"]} for r in contract["requirements"]}
        unresolved = obj({r["id"]: {"type": "string", "minLength": 1, "maxLength": 1500} for r in contract["requirements"]})
        unresolved["required"] = []
        return obj({"delivery": obj(fields), "unresolved": unresolved,
            "uncertainties": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 1500}}})
    return obj({"delivery": obj({r["id"]: obj({"state": {"type": "string", "enum": ["provided", "unresolved"]},
        "content": content_schema(r["kind"]), "gap": {"type": "string", "maxLength": 1500}})
        for r in contract["requirements"]}),
        "uncertainties": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 1500}}})


GENERATION = """Return the required typed delivery response, not the legacy free-text draft object.
The source-anchored delivery contract fixes OUTPUT SHAPES, not business answers or permission.
Fill every declared item with the actual requested deliverable. An artifact body is complete code/content,
not a recipe, outline or Markdown fence. A decision needs an explicit conclusion and its basis; next_steps
needs concrete remaining actions/evidence. Use provided only when you can supply that item; otherwise use
unresolved with empty content fields (empty items array) and a specific nonempty gap. Do not manufacture
facts merely to fill a field. Keep source/task limitations and genuine uncertainty. Do not repeat checks
already evidenced by supplied read receipts. Delivering text never executes the artifact or authorizes actions.
host_read_state records actual reads, separately from fixed unrepresented output requirements. Determine
remaining factual uncertainties from the collected payloads, not stale pre-read authoring notes.
"""


def generation(contract):
    if contract["profile"] == TASK_PROFILE:
        return """Return one JSON object with answer containing the actual answer to the entire original_task.
Use its requested format, including any requested artifact, conclusion, evidence or limitation.
Put uncertainty and inability in the answer itself; do not imply unsupported completion.
The host preserves this text exactly, without adding headings or extracting selected duties.
This is unverified L1 content. Existing source/evidence policies and tool boundaries remain binding.
"""
    if contract["profile"] == CHOICE_PROFILE:
        return """Return exactly the host outputSchema. Each delivery ID has ONE value: either its typed
content OBJECT, or a STRING explaining why that item cannot be supplied. A string is always unresolved,
never a successful answer or an empty-state marker. Do not add state/content/gap wrappers, null values,
or a separate unresolved map. Every ID is required. Put caveats about supplied content in uncertainties.
Supply actual requested content, not a summary of the intended work. Preserve the entire original task,
including requested conclusions, evidence and limitations; selected kinds do not redefine the task.
Host read receipts record operations, not sufficient facts or semantic truth. Delivering text grants
no execution authority. This schema validates representation only, not interpretation or task success.
"""
    if contract["profile"] != COMPACT_PROFILE:
        return GENERATION
    return """Return the compact typed response exactly as deliveryResponseSchema/outputSchema specifies.
delivery contains EVERY host requirement ID, directly holding its typed content object. Do not wrap content
in state/content/gap fields: those are host-owned internal representation, not fields in this protocol.
If unable to supply an item, put null at that ID and a nonempty explanation at the SAME ID in unresolved.
unresolved contains exactly the null IDs; use {} when all items have content. Never omit a delivery ID,
invent content, supply an orphan gap, or use blank content to hide inability. uncertainties retains actual
remaining caveats even when content is supplied. A complete artifact body is not an outline or summary.
Host read state records operations, not sufficient evidence or truth. Fixed unrepresented requirements
are separate from unread data. Types, references and filled slots do not prove semantic correctness.
Delivering text authorizes no operation; original source, task, qualifiers and unknowns remain binding.
"""


def _canonical_response(contract, response):
    if contract["profile"] == CHOICE_PROFILE:
        rows = {}
        for req in contract["requirements"]:
            value = response["delivery"][req["id"]]
            if isinstance(value, str):
                if not value.strip():
                    raise DataBindingError("blank_unresolved_reason", "/delivery/" + req["id"], "an unresolved choice needs an explanation")
                empty = {field: [] if schema["type"] == "array" else "" for field, schema in content_schema(req["kind"])["properties"].items()}
                rows[req["id"]] = {"state": "unresolved", "content": empty, "gap": value}
            else:
                rows[req["id"]] = {"state": "provided", "content": value, "gap": ""}
        return {"delivery": rows, "uncertainties": response["uncertainties"]}
    if contract["profile"] != COMPACT_PROFILE:
        return response
    rows = {}
    for req in contract["requirements"]:
        key = req["id"]
        value, gap = response["delivery"][key], response["unresolved"].get(key)
        if value is None:
            if gap is None or not gap.strip():
                raise DataBindingError("missing_unresolved_reason", "/unresolved/" + key, "null delivery needs its own nonempty explanation")
            empty = {field: [] if schema["type"] == "array" else "" for field, schema in content_schema(req["kind"])["properties"].items()}
            rows[key] = {"state": "unresolved", "content": empty, "gap": gap}
        else:
            if gap is not None:
                raise DataBindingError("conflicting_unresolved_reason", "/unresolved/" + key, "provided content cannot also be unresolved; retain caveats in uncertainties")
            rows[key] = {"state": "provided", "content": value, "gap": ""}
    return {"delivery": rows, "uncertainties": response["uncertainties"]}


def render(contract, response, *, evidence_state=None):
    """Pure rendering and missing/shape checks. Never execute an artifact."""
    response = validate_data(response_schema(contract), response)
    wire_digest = sha256_json(response)
    if contract["profile"] == TASK_PROFILE:
        if not response["answer"].strip():
            raise DataBindingError("blank_answer", "/answer", "an answer or explicit inability must be nonblank")
        return seal({"contractDigest": contract["reportDigest"], "candidateDigest": wire_digest,
            "canonicalCandidateDigest": wire_digest, "evidenceState": snapshot_json(evidence_state),
            "rendered": response["answer"], "shapeComplete": True,
            "shapeMeaning": "nonblank_answer_envelope_only_not_task_completion",
            "declaredCoverageComplete": None, "declaredCoverageCompleteMeaning": "not_assessed",
            "semanticCoverage": {"status": "not_assessed", "selectedKinds": "not_used",
                "originalTaskRemainsAuthoritative": True},
            "checks": [{"id": "task", "status": "present_not_semantically_validated",
                        "taskDigest": contract["sourceDigests"]["task"]}],
            "semanticApproval": False, "taskSuccess": None, "sourceScriptsExecuted": False})
    response = _canonical_response(contract, response)
    sections, checks = [], []
    for req in contract["requirements"]:
        value = response["delivery"][req["id"]]
        parts = value["content"]
        missing = [key for key, v in parts.items() if not v or (isinstance(v, str) and not v.strip())]
        unresolved = value["state"] == "unresolved"
        bad = (bool(missing) if not unresolved else (len(missing) != len(parts) or not value["gap"].strip()))
        # Explicit Markdown outline syntax is not KQL/SQL/Python/JSON source.
        # Everything else still needs independent syntax/semantic checking.
        if req["kind"] == "artifact" and not unresolved and req["language"] in {"kql", "kusto", "sql", "json", "python", "py"}:
            body = parts["body"].lstrip()
            bad |= body.startswith(("```", "~~~")) or bool(re.match(r"\d+[.)]\s+", body))
            if req["language"] in {"kql", "kusto", "sql", "json"}:
                bad |= body.startswith("#")
        status = "invalid_shape_or_missing_content" if bad else "unresolved" if unresolved else "present_not_semantically_validated"
        checks.append({"id": req["id"], "kind": req["kind"], "status": status, "source": req})
        heading = {"artifact": "Artifact", "analysis": "Analysis", "decision": "Decision", "next_steps": "Next steps"}[req["kind"]]
        if bad:
            text = "Not delivered: required content is missing or structurally invalid."
        elif unresolved:
            text = "Unresolved: " + value["gap"]
        elif req["kind"] == "artifact":
            body = parts["body"]
            width = max([2, *(len(m[0]) for m in re.finditer(r"`+", body))]) + 1
            text = "`" * width + req["language"] + "\n" + body + "\n" + "`" * width
        elif req["kind"] == "decision":
            text = parts["conclusion"] + "\n\nBasis: " + parts["basis"]
        elif req["kind"] == "next_steps":
            text = "\n".join("- " + item for item in parts["items"])
        else:
            text = parts["text"]
        if not bad and not unresolved and value["gap"].strip():
            text += "\n\nLimitations: " + value["gap"]
        sections.append("## " + heading + "\n\n" + text)
    # Keep old contracts readable without silently upgrading/clearing their
    # known stale annotations; protocol upgrades are explicit at compilation.
    if contract["profile"] == "source-anchored-delivery/v1":
        unrepresented = contract["uncovered"]
    elif contract["profile"] in {PROFILE, COMPACT_PROFILE, CHOICE_PROFILE}:
        unrepresented = [item["quote"] + " — " + item["reason"] for item in contract["unrepresented"]]
    else:
        raise ValueError("unsupported delivery contract profile")
    if response["uncertainties"]:
        sections[-1] += "\n\n**Current uncertainties (model claims)**\n\n" + "\n".join(
            "- " + item for item in response["uncertainties"])
    if unrepresented:
        sections[-1] += "\n\n**Unrepresented source requirements (not evidence-read status)**\n\n" + "\n".join(
            "- " + item for item in unrepresented)
    return seal({"contractDigest": contract["reportDigest"], "candidateDigest": wire_digest,
        "canonicalCandidateDigest": sha256_json(response),
        "semanticCoverage": {"status": "not_assessed", "selectedKinds": "model_proposed_presentation",
            "originalTaskRemainsAuthoritative": True},
        "declaredCoverageCompleteMeaning": "no_model_declared_unrepresented_items_not_proof_of_task_coverage",
        "evidenceState": snapshot_json(evidence_state), "declaredCoverageComplete": not unrepresented,
        "checks": checks, "rendered": "\n\n".join(sections), "shapeComplete": all(
            r["status"] == "present_not_semantically_validated" for r in checks) and not unrepresented,
        "semanticApproval": False, "taskSuccess": None, "sourceScriptsExecuted": False})
