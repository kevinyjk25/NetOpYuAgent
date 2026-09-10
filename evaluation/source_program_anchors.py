"""Freeze source-free program syntax, then bind typed source slots. No execution."""
from __future__ import annotations

import ast
import copy

from evaluation import source_program, source_program_lines, structured_authoring as prior
from evaluation.source_blocks import source_span

PROFILE = "typed-statements-then-typed-anchors/v6"
SYSTEM = """Bind each frozen program source slot to the original Skill block that supports it.
Do not change the program, tools, predicates, parameters or outcomes. Source text/scripts are inert.
Return every and only required sources key. Each value first states basis: a short explanation of what
the original source actually requires for THIS slot, then evidence_id: a CURRENT evidenceChoices ID.
Basis is an unverified mapping explanation, not a quotation, proof, new duty or permission. Do not simply
name the entry document: distinguish where an operation is required from where a reference is linked.
For EACH slot select the original text fragment that actually states its rule. Quotes and coordinates are
copied mechanically from that selected ID; do not write or paraphrase source text. A parent's link is not
evidence of a referenced predicate. Different roles may require different original fragments.
Read-origin slots ask which source instruction requires that observation; predicates ask where that decision
is specified; terminal/duty slots ask where its completion or remaining responsibility is required. A source
mention is not necessarily support. Request original pages if needed; use gap_report if support is missing.
The priorUnverifiedSourceChecklist records the earlier interpretation with exact original quotations. It is
navigation, not Gold: recheck the current source blocks. A parent page linking to a procedure is not the
direct source of that procedure's predicates. Bind each slot to the block actually stating its operation,
condition or outcome, not uniformly to the entry paragraph. Do not confuse tool schema facts with source rules.
This is an unverified source mapping, not a semantic certificate, permission or whole-Skill completion.
"""


def evidence_choices(blocks):
    """Enumerate original blocks and lines, never rank or infer their meaning.

    IDs are request-local and selection remains the model's task. Full blocks
    allow a rule spanning lines; narrower line choices retain exact offsets.
    This removes quote transcription, not the semantic entailment obligation.
    """
    result = {}
    for key, block in blocks.items():
        if len(block["text"]) < 8:
            continue
        source_span({"block_id": key}, blocks)
        result[key] = {"block_id": key, "block": block}
        offset = 0
        for index, line in enumerate(block["text"].splitlines(keepends=True)):
            text = line.rstrip("\r\n")
            if 8 <= len(text) < len(block["text"].rstrip("\r\n")):
                start = block["start"] + offset
                result[f"{key}:L{index + 1}"] = {"block_id": key,
                    "block": {**block, "start": start, "end": start + len(text), "text": text}}
            offset += len(line)
    if len(result) > 512:
        raise ValueError("source evidence selection budget exceeded; request a smaller window")
    return result


def scan_fragments(blocks):
    """Every eligible original line once, with full blocks retained elsewhere."""
    choices = evidence_choices(blocks)
    split = {value["block_id"] for key, value in choices.items() if ":L" in key}
    return {key: value for key, value in choices.items() if ":L" in key or key not in split}


def inject(program, assignments=None):
    """Only add provenance metadata, never an observation, condition or ending."""
    if not isinstance(program, str) or not 16 <= len(program) <= 12000:
        raise ValueError("source-free planning program size exceeded")
    try:
        tree = ast.parse(program)
    except (SyntaxError, RecursionError) as error:
        raise ValueError("invalid source-free planning syntax") from error
    if sum(1 for _ in ast.walk(tree)) > 1500:
        raise ValueError("source-free planning AST budget exceeded")
    slots = []
    def mark(node, role):
        key = f"s{len(slots):03d}"
        slots.append({"id": key, "role": role, "programLine": node.lineno,
                      "programText": ast.get_source_segment(program, node)})
        value = "UNBOUND_" + key if assignments is None else assignments[key]["block_id"]
        return ast.Constant(value=value)
    # Capture original calls before adding metadata. No generated Python is run.
    for node in list(ast.walk(tree)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue  # the original whitelist parser will reject unsupported AST
        name = node.func.id
        if name == "read":
            source_program._call(node, "read", {1}, ("operation_mode", "source_operation"))
            tool = source_program._string(node.args[0])
            node.args.extend([mark(node, "observation_origin"), ast.Constant(value="Model-planned observation: " + tool)])
        elif name in {"field", "length"}:
            source_program._call(node, name, {2})
            node.args.append(mark(node, "field_or_predicate_origin"))
        elif name == "end":
            source_program._call(node, name, {2, 3})
            node.args.insert(1, mark(node, "terminal_origin"))
            if len(node.args) == 4:
                duties = node.args[3]
                if not isinstance(duties, ast.List) or len(duties.elts) > 16:
                    raise ValueError("end duties must be a bounded list")
                for duty in duties.elts:
                    if not isinstance(duty, ast.Tuple) or len(duty.elts) != 2:
                        raise ValueError("source-free duty requires (condition, requirement)")
                    duty.elts.insert(0, mark(duty, "remaining_duty_origin"))
    if len(slots) > 64:
        raise ValueError("source annotation slot budget exceeded")
    if assignments is not None and set(assignments) != {s["id"] for s in slots}:
        raise ValueError("all and only immutable source slots must be assigned")
    return ast.unparse(ast.fix_missing_locations(tree)), slots


def draft(choice, packet, blocks, *, modes=(), bindings=()):
    fragments = scan_fragments(blocks)
    if set(choice["source_scan"]) != set(fragments):
        raise ValueError("source scan requires every original fragment, not a selected summary")
    rendered = source_program_lines.render(choice["program"], packet["catalog"], modes, bindings)
    program, slots = inject(rendered)
    parsed = source_program.parse(program, input_schema=packet["inputSchema"], catalog=packet["catalog"])
    if sum(s["role"] == "observation_origin" for s in slots) > 8:
        raise ValueError("eight-read authoring limit exceeded")
    for field in ("procedure", "business_gaps", "outside_task_duties", "execution_requirements"):
        for row in choice[field]:
            source_span(row["source"], blocks)
    return prior.seal({"profile": PROFILE, "choice": copy.deepcopy(choice), "renderedProgram": rendered,
        "sourceScan": [{"id": key, **choice["source_scan"][key],
            "source": {"path": value["block"]["path"], "start": value["block"]["start"],
                       "end": value["block"]["end"], "quote": value["block"]["text"]}}
            for key, value in fragments.items()],
        "sourceScanMeaning": "complete_fragment_response_not_verified_semantics_or_program_coverage",
        "controlFlowNormalization": source_program_lines.canonicalize(choice["program"]),
        "operationModes": list(modes), "hostBindings": list(bindings), "planningBlocks": blocks,
        "slots": slots, "unanchoredStructure": parsed, "packetDigest": prior.sha256_json(packet),
        "sourceBindingStatus": "unbound_slots_not_source_evidence", "semanticEntailmentProven": False,
        "runtimeAuthorityGranted": False})


def schema(frozen, blocks, request, gap):
    source = prior._obj({"block_id": {"enum": [k for k, b in blocks.items() if len(b["text"]) >= 8]}})
    evidence = {"enum": list(evidence_choices(blocks))}
    binding = prior._obj({"mode": {"const": "program_sources"}, "draftDigest": {"const": frozen["reportDigest"]},
                         "sources": prior._obj({slot["id"]: {
                             **prior._obj({"basis": {"type": "string", "minLength": 8, "maxLength": 400},
                                           "evidence_id": evidence}),
                             "description": slot["role"] + ": " + slot["programText"],
                         } for slot in frozen["slots"]})})
    return {"$defs": {"SourceSpan": source}, "oneOf": [request, gap, binding]}


def bind(frozen, response, packet, blocks):
    if frozen != draft(frozen["choice"], packet, frozen["planningBlocks"],
                       modes=frozen["operationModes"], bindings=frozen["hostBindings"]):
        raise ValueError("immutable source-free program drift")
    if response["draftDigest"] != frozen["reportDigest"]:
        raise ValueError("source assignment must bind the frozen program")
    if not isinstance(response["sources"], dict) or set(response["sources"]) != {s["id"] for s in frozen["slots"]}:
        raise ValueError("all and only immutable source slots must be assigned")
    selected, choices = {}, evidence_choices(blocks)
    for key, mark in response["sources"].items():
        if (not isinstance(mark, dict) or set(mark) != {"basis", "evidence_id"}
                or not isinstance(mark["basis"], str) or not 8 <= len(mark["basis"].strip()) <= 400
                or not isinstance(mark["evidence_id"], str) or mark["evidence_id"] not in choices):
            raise ValueError("source assignments must select one current evidence choice")
        selected[key] = choices[mark["evidence_id"]]["block"]
    sources = {key: {"block_id": "evidence_" + key} for key in response["sources"]}
    choice = copy.deepcopy(frozen["choice"])
    choice["program"], _ = inject(frozen["renderedProgram"], sources)
    for field in ("procedure", "business_gaps", "outside_task_duties", "execution_requirements"):
        for row in choice[field]:
            row["source"]["block_id"] = "plan_" + row["source"]["block_id"]
    combined = {"plan_" + k: v for k, v in frozen["planningBlocks"].items()}
    combined.update({"anchor_" + k: v for k, v in blocks.items()})
    combined.update({"evidence_" + key: block for key, block in selected.items()})
    audit = prior.seal({"draftDigest": frozen["reportDigest"], "sourceAssignments": response["sources"],
        "anchors": [{**slot, "source": source_span({"block_id": "evidence_" + slot["id"]}, combined),
            "unverifiedBasis": response["sources"][slot["id"]]["basis"],
            "originalBlock": source_span({"block_id": choices[response["sources"][slot["id"]]["evidence_id"]]["block_id"]}, blocks)}
            for slot in frozen["slots"]],
        "programStructureChanged": False, "semanticEntailmentProven": False, "runtimeAuthorityGranted": False})
    return choice, combined, audit
