"""Frozen, zero-inference preparation for the bounded development pilot.

This is a separate research protocol, never a translation admission. No CLI
command here invokes DSH, an LLM, a third-party script or an Effect. The old
formal gate remains unchanged. Seals detect drift, not a malicious operator
who controls the files, registry and Python process.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from evaluation.bounded_budget import BudgetLedger
from network_runtime.contracts import sha256_json

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/bounded-pilot-registry/studies.sqlite"
SCHEMA = "ensuredskill.io/bounded-development-pilot/v1"
MODEL = "qwen3.5:9b"
FAMILIES = frozenset({"evidence", "parameters", "approval", "branch", "verification", "mixed"})
PRIMITIVES = frozenset({"evidence_dependency", "resource_binding", "bounded_predicate"})
LIMITS = {
    "development_versions": 2, "confirmation_batches": 1, "tasks_per_batch": 12,
    "development_repetitions": 1, "confirmation_repetitions": 3,
    "arm_seconds": 420, "arm_model_requests": 10,
    "arm_input_tokens": 64000, "arm_output_tokens": 6000,
    "compiler_requests": 2, "revision_requests": 1, "effect_reserve_seconds": 60,
    "total_arms": 120, "total_arm_requests": 1200,
    "offline_requests": 48, "offline_request_seconds": 180,
    "offline_input_tokens": 24000, "offline_output_tokens": 3000,
}
THRESHOLDS = {
    "loadable_fraction": 11 / 12, "duty_fidelity": 0.95, "strict_recall": 0.8,
    "positive_completion": 0.75, "boundary_completion": 1.0,
    "paired_gain": 0.1, "positive_gain_min": 0.0,
    "cost_ratio_max": 1.5, "critical_errors": 0, "successful_skills": 3,
}


def _snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"{label}: exact fields required")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: nonempty text required")


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("SHA-256 digest required")


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value):
        raise ValueError("bounded opaque identifier required")


def _local_schema(value):
    """A frozen tool catalog may not retrieve extra schema documents."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"$ref", "$dynamicRef"} and (not isinstance(item, str) or not item.startswith("#")):
                raise ValueError("tool schemas require local-only references")
            _local_schema(item)
    elif isinstance(value, list):
        for item in value:
            _local_schema(item)


def seal(value):
    value = _snapshot(value)
    if "digest" in value:
        raise ValueError("cannot reseal an existing digest")
    return {**value, "digest": sha256_json(value)}


def checked_seal(value):
    value = _snapshot(value)
    if not isinstance(value, dict) or "digest" not in value:
        raise ValueError("sealed object required")
    if value["digest"] != sha256_json({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError("sealed object drift")
    return value


def validate_agent_input(value):
    """Explicit execution projection, excluding labels/support/expected routes.

    Source and fixture contents remain untrusted data. A key allowlist does
    not establish that a human-authored task contains no answer leakage.
    """
    _keys(value, "task skill_text references tools arguments", "agent input")
    for name in ("task", "skill_text"):
        _text(value[name], name)
    if not isinstance(value["references"], list) or len(value["references"]) > 64:
        raise ValueError("bounded inert references required")
    paths = []
    for reference in value["references"]:
        _keys(reference, "path text", "reference")
        _text(reference["path"], "reference path")
        if not isinstance(reference["text"], str):
            raise ValueError("reference content is inert text")
        paths.append(reference["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate reference identity")
    if not isinstance(value["tools"], list) or not 1 <= len(value["tools"]) <= 32:
        raise ValueError("one to 32 host tools required")
    names = []
    for tool in value["tools"]:
        _keys(tool, "name description input_schema output_schema", "host tool")
        _identifier(tool["name"])
        _text(tool["description"], "tool description")
        for name in ("input_schema", "output_schema"):
            schema = tool[name]
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("explicit object tool schema required")
            _local_schema(schema)
            Draft202012Validator.check_schema(schema)
        names.append(tool["name"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate host tool identity")
    if not isinstance(value["arguments"], dict):
        raise ValueError("explicit invocation arguments required")
    _snapshot(value)
    return copy.deepcopy(value)


def validate_cases(cases, *, confirmation=False):
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("a bounded pilot batch has exactly twelve assigned tasks")
    identities, skills, repos, domains, families = set(), {}, set(), set(), set()
    counts = {"positive": 0, "boundary": 0}
    for case in cases:
        _keys(case, "case_id skill_id repository_id repository_family domain kind families "
              "source_revision source_kind agent_input provider_fixture input_digest reference_digest", "case")
        _identifier(case["case_id"])
        if case["case_id"] in identities:
            raise ValueError("duplicate assigned task")
        identities.add(case["case_id"])
        for name in ("skill_id", "repository_id", "repository_family", "domain", "source_revision"):
            _text(case[name], name)
        if case["kind"] not in counts:
            raise ValueError("positive/boundary stratum required")
        counts[case["kind"]] += 1
        if (not isinstance(case["families"], list) or not case["families"]
                or len(case["families"]) != len(set(case["families"]))
                or not set(case["families"]) <= FAMILIES):
            raise ValueError("declared mechanism families required")
        families.update(case["families"])
        if case["source_kind"] not in {"pinned_public", "synthetic_development"}:
            raise ValueError("explicit source provenance required")
        if confirmation and case["source_kind"] != "pinned_public":
            raise ValueError("confirmation cannot relabel synthetic development as unseen")
        agent_input = validate_agent_input(case["agent_input"])
        if not isinstance(case["provider_fixture"], dict):
            raise ValueError("host-only provider fixture required")
        case_initial_state_digest(case)
        if case["input_digest"] != case_input_digest(case):
            raise ValueError("paired input digest drift")
        _digest(case["reference_digest"])
        source_identity = (case["repository_id"], case["repository_family"], case["source_revision"],
                           sha256_json({"skill_text": agent_input["skill_text"],
                                        "references": agent_input["references"]}))
        if case["skill_id"] in skills and skills[case["skill_id"]] != source_identity:
            raise ValueError("one Skill ID must name one pinned source")
        skills[case["skill_id"]] = source_identity
        repos.add(case["repository_family"])
        domains.add(case["domain"])
    if len(skills) != 6 or len(repos) < 4 or len(domains) < 3 or families != FAMILIES:
        raise ValueError("need six Skills, four repository families, three domains, all six mechanisms")
    if len({value[3] for value in skills.values()}) != 6:
        raise ValueError("identical source text cannot inflate Skill count")
    if counts != {"positive": 8, "boundary": 4}:
        raise ValueError("predeclared task mixture is eight positive plus four boundary")
    return copy.deepcopy(cases)


def case_input_digest(case):
    return sha256_json({"agent_input": case["agent_input"], "provider_fixture": case["provider_fixture"]})


def case_initial_state_digest(case):
    """Bind only state, matching ArmProvider's database-read initial snapshot.

    The complete fixture (tools, policy and state) remains bound separately by
    case_input_digest and the Provider fixture_digest. This is not a receipt or
    proof that a Provider has actually initialized that state.
    """
    fixture = case.get("provider_fixture")
    if not isinstance(fixture, dict) or not isinstance(fixture.get("state"), dict):
        raise ValueError("host Provider fixture requires an explicit state object")
    return sha256_json(fixture["state"])


def make_protocol(study_id, cases, *, model_digest, harness_digest, support):
    return validate_protocol(seal({
        "schema": SCHEMA, "study_id": study_id, "purpose": "bounded_development_pilot",
        "model": {"name": MODEL, "digest": model_digest}, "harness_digest": harness_digest,
        "support": support, "limits": LIMITS, "thresholds": THRESHOLDS,
        "seed": 20260916, "development_cases": cases,
        "authority": "measurement_only_no_research_admission",
    }))


def validate_protocol(value):
    value = checked_seal(value)
    _keys(value, "schema study_id purpose model harness_digest support limits thresholds seed "
          "development_cases authority digest", "protocol")
    _identifier(value["study_id"])
    if (value["schema"] != SCHEMA or value["purpose"] != "bounded_development_pilot"
            or value["authority"] != "measurement_only_no_research_admission"):
        raise ValueError("pilot cannot grant research or execution authority")
    _keys(value["model"], "name digest", "model")
    if value["model"]["name"] != MODEL:
        raise ValueError("this protocol retains the user-selected 9B model")
    _digest(value["model"]["digest"])
    _digest(value["harness_digest"])
    if (value["limits"] != LIMITS or value["thresholds"] != THRESHOLDS
            or type(value["seed"]) is not int or value["seed"] != 20260916):
        raise ValueError("changing fixed thresholds/budgets requires a new reviewed protocol version")
    support = value["support"]
    _keys(support, "primitives unsupported source_scripts effect_gateway", "support")
    if (not isinstance(support["primitives"], list) or len(support["primitives"]) != 3
            or set(support["primitives"]) != PRIMITIVES or support["source_scripts"] != "inert"
            or support["effect_gateway"] != "existing_active_contracts_local_simulator"
            or not isinstance(support["unsupported"], list) or not support["unsupported"]):
        raise ValueError("bounded support and explicit unsupported semantics required")
    for item in support["unsupported"]:
        _text(item, "unsupported scope")
    validate_cases(value["development_cases"])
    return value


def confirmation_plan(protocol, cases, known_inventory, *, candidate_digest):
    """Known inventory must include all previously used sources, not just R1.

    Canonical repository-family labels and copy detection still require
    pre-run source review. This check cannot discover unknown forks itself.
    """
    protocol = validate_protocol(protocol)
    cases = validate_cases(cases, confirmation=True)
    _digest(candidate_digest)
    _keys(known_inventory, "skill_ids repository_families source_digests", "known inventory")
    for values in known_inventory.values():
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("known inventory lists required")
    previous = protocol["development_cases"]
    skill_ids = set(known_inventory["skill_ids"]) | {c["skill_id"] for c in previous}
    repos = set(known_inventory["repository_families"]) | {c["repository_family"] for c in previous}
    def source(case):
        return sha256_json({k: case["agent_input"][k] for k in ("skill_text", "references")})
    texts = set(known_inventory["source_digests"]) | {source(c) for c in previous}
    if any(c["skill_id"] in skill_ids or c["repository_family"] in repos or source(c) in texts for c in cases):
        raise ValueError("confirmation overlaps known Skill/repository/source content")
    return seal({"schema": SCHEMA, "phase": "confirmation", "protocol_digest": protocol["digest"],
                 "candidate_digest": candidate_digest, "known_inventory_digest": sha256_json(known_inventory),
                 "cases": cases, "researchEvidenceEligible": False})


def schedule(cases, *, phase, seed=20260916):
    """One fixed balanced AB/BA assignment, no outcome-dependent reordering."""
    validate_cases(cases, confirmation=phase == "confirmation")
    if phase not in {"development", "confirmation"}:
        raise ValueError("unknown phase")
    repeats = 1 if phase == "development" else 3
    rng = random.Random(seed)
    pairs = [(case, repeat) for repeat in range(1, repeats + 1)
             for case in sorted(cases, key=lambda c: c["case_id"])]
    rng.shuffle(pairs)
    orders = [("control", "treatment")] * (len(pairs) // 2) + [("treatment", "control")] * (len(pairs) // 2)
    rng.shuffle(orders)
    return [{"case_id": case["case_id"], "repetition": repeat, "arms": list(order),
             "input_digest": case["input_digest"]} for (case, repeat), order in zip(pairs, orders, strict=True)]


def source_fingerprint():
    """Snapshot current execution+measurement code, including untracked files."""
    files = {}
    for name in ("skill_authoring", "network_runtime", "effect_runtime", "dsh_adapter", "dsh-plugin-netopyu"):
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".js", ".json"} and "node_modules" not in path.parts:
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT / "evaluation").glob("bounded_*.py")):
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"digest": sha256_json(files), "files": files, "meaning": "development_source_snapshot_not_clean_research_freeze"}


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("bounded regular JSON file required")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n")


def prepare(protocol, output, *, registry=REGISTRY):
    protocol = validate_protocol(protocol)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("new output directory required; registry is independent of output location")
    ledger = BudgetLedger(registry)
    ledger.register_study(protocol["study_id"], protocol["digest"])
    fingerprint = source_fingerprint()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "protocol.json", protocol)
    write_new(output / "schedule.json", seal({"phase": "development", "pairs": schedule(protocol["development_cases"], phase="development")}))
    write_new(output / "implementation.json", fingerprint)
    agent_dir = output / "agent"
    agent_dir.mkdir()
    provider_dir = output / "provider-private"
    provider_dir.mkdir()
    for case in protocol["development_cases"]:
        # References/scripts stay JSON text, never executable source files.
        write_new(agent_dir / f"{case['case_id']}.json", case["agent_input"])
        write_new(provider_dir / f"{case['case_id']}.json", case["provider_fixture"])
    report = seal({"schema": SCHEMA, "status": "prepared_not_executed",
                   "study_id": protocol["study_id"], "protocol_digest": protocol["digest"],
                   "implementation_digest": fingerprint["digest"], "realModelCalls": 0,
                   "runtimeLargeEvaluationAllowed": False, "researchEvidenceEligible": False,
                   "liveAdapterReady": False, "blockedOn": ["metered_live_DSH_adapter", "trusted_input_token_preflight",
                   "physical_provider_reset_adapter", "frozen_independently_aligned_reference_labels"],
                   "registry": str(Path(registry).resolve())})
    write_new(output / "preparation.json", report)
    return report


def validate_references(protocol, references):
    """Bind evaluator-only labels to the frozen visible source and private state.

    Quote presence is a mechanical prerequisite, not proof that the statement
    follows from it. Independent construct review is still required.
    """
    from evaluation.bounded_scoring import validate_reference
    protocol = validate_protocol(protocol)
    if not isinstance(references, list) or len(references) != 12:
        raise ValueError("every frozen task requires one predeclared reference")
    cases = {c["case_id"]: c for c in protocol["development_cases"]}
    seen = set()
    for reference in references:
        validate_reference(reference)
        case_id = reference["case_id"]
        if case_id not in cases or case_id in seen:
            raise ValueError("missing, repeated or unassigned reference")
        seen.add(case_id)
        case = cases[case_id]
        if reference["initial_state_digest"] != case_initial_state_digest(case):
            raise ValueError("reference initial_state_digest must bind Provider state, not the full fixture")
        if (reference["reference_digest"] != case["reference_digest"]
                or reference["repository_id"] != case["repository_family"]
                or any(reference[k] != case[k] for k in ("skill_id", "domain", "kind"))):
            raise ValueError("reference differs from frozen case/source/Provider identity")
        inputs = case["agent_input"]
        catalog = {tool["name"]: tool for tool in inputs["tools"]}
        for call in reference["calls"]:
            if call["tool"] not in catalog:
                raise ValueError("reference call names a tool outside the frozen catalog")
            try:
                Draft202012Validator(catalog[call["tool"]]["input_schema"]).validate(call["arguments"])
            except ValidationError as exc:
                raise ValueError("reference call arguments violate the frozen tool schema") from exc
        visible = [inputs["task"], inputs["skill_text"], *(r["text"] for r in inputs["references"]),
                   json.dumps(inputs["tools"], ensure_ascii=False, sort_keys=True),
                   json.dumps(inputs["arguments"], ensure_ascii=False, sort_keys=True)]
        for requirement in [*reference["criteria"], *reference["duties"]]:
            if not any(requirement["source_quote"] in text for text in visible):
                raise ValueError("reference criterion/duty needs a quote in allowed visible source, not private state")
    return copy.deepcopy(references)


def assess(protocol, references, observations):
    """Development scorecard only; no live-run attestation or research admission."""
    from evaluation.bounded_scoring import paired_report
    protocol = validate_protocol(protocol)
    references = validate_references(protocol, references)
    scored = paired_report(references, observations, phase="development")
    return seal({"schema": SCHEMA, "status": "scorecard_only_not_pilot_qualification",
                 "protocol_digest": protocol["digest"], "observations_digest": sha256_json(observations),
                 "reference_digests": [r["reference_digest"] for r in references],
                 "scorecard": scored, "pilotQualified": False, "runtimeLargeEvaluationAllowed": False,
                 "liveAdapterAttested": False, "semanticEntailmentAutomaticallyProven": False,
                 "limitations": ["receipt truth requires a separately validated collector",
                                 "references are externally judged, not machine-proven semantics",
                                 "mechanism probes, source independence and live metering remain separate gates"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate a frozen protocol, zero inference")
    check.add_argument("protocol", type=Path)
    prep = sub.add_parser("prepare", help="seal agent-only input projections, zero inference")
    prep.add_argument("protocol", type=Path)
    prep.add_argument("output", type=Path)
    inspect = sub.add_parser("inspect", help="show persistent budget state, never resume calls")
    inspect.add_argument("study_id")
    scoring = sub.add_parser("score", help="assess bound development receipts, no inference or qualification")
    scoring.add_argument("protocol", type=Path)
    scoring.add_argument("references", type=Path)
    scoring.add_argument("observations", type=Path)
    scoring.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        if not REGISTRY.exists():
            parser.error("no pilot registry exists")
        result = BudgetLedger(REGISTRY).snapshot(args.study_id)
    elif args.command == "score":
        result = assess(read_json(args.protocol), read_json(args.references), read_json(args.observations))
        write_new(args.output, result)
    else:
        protocol = validate_protocol(read_json(args.protocol))
        result = (prepare(protocol, args.output) if args.command == "prepare" else
                  {"valid": True, "protocol_digest": protocol["digest"], "modelCalls": 0,
                   "executionAuthorized": False, "runtimeLargeEvaluationAllowed": False})
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
