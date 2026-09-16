"""Candidate-blind duty proposals and focused, located, non-authoritative reviews."""
from copy import deepcopy

from evaluation.hybrid_review_views import source_view
from evaluation.hybrid_semantic_witness import obj, text
from evaluation.structured_authoring import seal
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import validate_data

PROFILE = "focused-duty-contract/v1"
MAX_DUTIES = 6  # The seventh focused call is ALWAYS reserved for all notes.
PLAN_SYSTEM = """Extract a short source-anchored duty contract BEFORE seeing an answer.
Read the original task, role segments and ALL original sources. Propose up to six independently checkable
business deliverables/constraints. Cite exact task text; preserve quantities, timing and negations. Do not
turn execution prohibitions into required answer paragraphs. Source examples are not instance facts.
If six entries cannot represent the task, mark overflow and explain missing duties. Never truncate silently.
No candidate, expected answer or prior reviewer is available. Proposals may be wrong/incomplete, not authority.
JSON only; no tools, scripts or actions. Each description <=60 words.
"""
CHECK_SYSTEM = """Inspect ONLY the assigned duty against the complete original task, sources and actual answer.
The proposed duty is fallible; if out of scope say unknown, not an invented requirement. Find delivered content,
not promises, headings or copied task text. Inspect actual code/calculation, not claims about it. Local checks
only prove their named subset; unverified does not mean false. A passed check is NOT full semantic approval.
Report met/gap/unknown and at most four concrete issues with exact candidate and source quotes. An omission
may have empty candidate location/quote. Keep corrections specific and within the actual task. No tools or writes.
Read provenance distinguishes an index from separately read data; do not request an already supplied export.
All opinions are unverified inspection leads. Do not issue task approval. JSON only, concise explanations.
"""
NOTES_SYSTEM = """Audit EVERY candidate note, independently of whether the answer is correct.
Decompose each note into its atomic assertions/presuppositions, with exact note substrings. For conjunction,
disjunction, negation, timing and status, examine each predicate separately: absence of evidence is not evidence
of absence, and evidence for one predicate does not establish another. Distinguish a justified uncertainty
from an unsupported assertion disguised as a caveat. Compare ALL actual observations, not template defaults.
Each atom needs a status (supported/unsupported/contradicted/unknown), source IDs and exact quotes where available.
Unknowns must remain unknown. Quote binding is not entailment proof. Do not edit or grant authority. JSON only.
"""


def source_input(payload):
    return seal({"profile": PROFILE, "originalTask": payload["originalTask"],
        "sourceSpans": [source_view(s) for s in payload["sourceSpans"]],
        **{k: deepcopy(payload[k]) for k in ("taskScope", "readContext") if k in payload},
        "candidateVisible": False, "authorityGranted": False})


def verify(report):
    if report != seal({k: v for k, v in report.items() if k != "reportDigest"}):
        raise ValueError("duty report digest drift")


def plan_schema():
    return obj({"duties": {"type": "array", "minItems": 1, "maxItems": MAX_DUTIES,
        "items": obj({"task_quote": text(2400, 1), "description": text(1200, 1)})},
        "overflow": {"type": "boolean"}, "unrepresented_duties": text(2400)})


def bind_plan(source, raw):
    verify(source)
    value = validate_data(plan_schema(), raw)
    duties = []
    for i, row in enumerate(value["duties"]):
        if not row["task_quote"].strip() or row["task_quote"] not in source["originalTask"]:
            raise ValueError("duty requires exact nonblank original task anchor")
        duties.append({"id": f"r{i:03d}", **row})
    if value["overflow"] != bool(value["unrepresented_duties"].strip()):
        raise ValueError("overflow and missing-duty disclosure disagree")
    return seal({"profile": PROFILE, "sourceDigest": source["reportDigest"], "duties": duties,
        "overflow": value["overflow"], "unrepresentedDuties": value["unrepresented_duties"],
        "completenessProven": False, "authorityGranted": False})


def context(payload, plan):
    verify(plan)
    source = source_input(payload)
    if source["reportDigest"] != plan["sourceDigest"]:
        raise ValueError("duty contract source/task drift")
    locations = {s["draft_span_id"]: s["exactQuote"] for s in payload["draftSpans"]}
    locations.update({f"n{i:03d}": n for i, n in enumerate(payload["candidate"]["notes"])})
    return {"sourceContext": source, "unverifiedContract": plan, "candidateLocations": locations,
            "completeCandidate": deepcopy(payload["candidate"]), "candidateDigest": payload["completeCandidateDigest"]}


def evidence_schema(ctx):
    return {"type": "array", "maxItems": 4, "items": obj({
        "source_id": {"type": "string", "enum": [s["source_span_id"] for s in ctx["sourceContext"]["sourceSpans"]]},
        "quote": text(4000, 1)})}


def check_schema(ctx):
    location = {"type": "string", "enum": ["", *ctx["candidateLocations"]]}
    return obj({"outcome": {"type": "string", "enum": ["met", "gap", "unknown"]},
        "artifact_location": location, "artifact_quote": text(4000), "explanation": text(1600, 1),
        "issues": {"type": "array", "maxItems": 4, "items": obj({
            "location": location, "quote": text(4000), "evidence": evidence_schema(ctx),
            "explanation": text(1600, 1), "correction": text(1600, 1)})}})


def _evidence(ctx, rows):
    catalog = {s["source_span_id"]: s["exactQuote"] for s in ctx["sourceContext"]["sourceSpans"]}
    return bool(rows) and all(r["quote"].strip() and r["quote"] in catalog[r["source_id"]] for r in rows)


def _located(ctx, key, quote):
    return bool(key in ctx["candidateLocations"] and quote.strip() and quote in ctx["candidateLocations"][key])


def bind_check(ctx, duty, raw):
    value = validate_data(check_schema(ctx), raw)
    issues, binding_issues = [], []
    located = _located(ctx, value["artifact_location"], value["artifact_quote"])
    if value["outcome"] == "met" and not located:
        value["outcome"] = "unknown"
        binding_issues.append("positive_opinion_without_artifact_witness")
    for row in value["issues"]:
        omission = not row["location"] and not row["quote"]
        if (omission or _located(ctx, row["location"], row["quote"])) and _evidence(ctx, row["evidence"]):
            issues.append({**row, "dutyId": duty["id"], "unverifiedSemanticLead": True})
        else:
            binding_issues.append("issue_without_exact_evidence_or_candidate_location")
    return seal({"dutyId": duty["id"], "candidateDigest": ctx["candidateDigest"],
        "rawOpinionDigest": sha256_json(raw), "outcome": value["outcome"], "explanation": value["explanation"],
        "artifactLocation": value["artifact_location"], "artifactQuote": value["artifact_quote"],
        "issues": issues, "bindingIssues": binding_issues, "semanticApproval": False})


def notes_schema(ctx):
    return obj({key: obj({"atoms": {"type": "array", "minItems": 1, "maxItems": 8,
        "items": obj({"quote": text(2000, 1), "status": {"type": "string", "enum": ["supported", "unsupported", "contradicted", "unknown"]},
            "evidence": evidence_schema(ctx), "explanation": text(1200, 1)})}})
        for key in ctx["candidateLocations"] if key.startswith("n")})


def bind_notes(ctx, raw):
    value = validate_data(notes_schema(ctx), raw)
    notes, binding_issues = {}, []
    for key, row in value.items():
        atoms = []
        for atom in row["atoms"]:
            bound = _located(ctx, key, atom["quote"]) and _evidence(ctx, atom["evidence"])
            if not bound:
                binding_issues.append({"location": key, "code": "atom_without_exact_two_sided_anchor"})
            atoms.append({**atom, "status": atom["status"] if bound else "unknown", "bound": bound})
        notes[key] = {"atoms": atoms, "needsInspection": any(a["status"] != "supported" for a in atoms),
                      "allPredicatesEnumeratedProven": False}
    return seal({"candidateDigest": ctx["candidateDigest"], "rawOpinionDigest": sha256_json(raw),
        "notes": notes, "bindingIssues": binding_issues, "semanticApproval": False})
