"""Fresh 9B-first-pass development packages, NOT independent/unseen public Gold.

The development assistant authors source, inert host contracts and references.
Reference trees/oracles are never passed to the model. Source files stay inert.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

from evaluation.flow_behavior import BehaviorSuite
from evaluation.flow_behavior_examples import _contract, _schema, branch, call, end, observation, read, ref
from evaluation.flow_source_selection import spans
from evaluation.flow_translation import FlowSources
from evaluation.flow_tree import FlowTree
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import ReadObjectSchema

ROOT = Path(__file__).resolve().parents[1] / "examples/semantic-transfer"


def package(name):
    folder = ROOT / name
    files = []
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            text = p.read_text()
            files.append(dict(path=str(p.relative_to(folder)), text=text,
                sha256="sha256:" + hashlib.sha256(text.encode()).hexdigest()))
    main = next(f["text"] for f in files if f["path"] == "SKILL.md")
    source = main + "".join("\n## Supplied inert file: " + f["path"] + "\n" + f["text"]
        for f in files if f["path"] != "SKILL.md")
    return files, source


def _source(name, argument, contracts):
    files, text = package(name)
    source = FlowSources(source_text=text, source_path=f"semantic-transfer/{name}/SKILL.md",
        input_schema=ReadObjectSchema.model_validate(_schema({argument: "string"})),
        reads={c.spec.tool: c for c in contracts}, effects={}, max_read_age_seconds=30)
    return files, source


def _line(source, quote):
    return next(key for key, value in spans(source).items() if quote in value)


def _case(name, domain, files, source, steps, scenarios, gaps=()):
    # Every normative source line is visible to review; finite scenarios do not
    # purport to prove every line's language meaning or output explanation.
    requirements = {k: v for k, v in spans(source).items() if len(v) > 40}
    suite = BehaviorSuite(sources_digest=sha256_json(source.model_dump(mode="json")),
        requirements=requirements, scenarios=[dict(requirement_ids=list(requirements), **s) for s in scenarios],
        scope="safe_partial_stop" if gaps else "executable_fragment", capability_gaps=gaps,
        unverified_meanings=["Complete prose entailment, explanation fidelity and open-world execution remain unproven."],
        oracle_author="same-development-assistant-manual-policy-not-independent-gold")
    source_id = _line(source, "unsupported")
    tree = FlowTree(business_source_ids=[source_id], steps=steps,
        issues=[dict(kind="missing_host_capability", source_id=source_id, question=g) for g in gaps])
    return dict(id=name, domain=domain, package=files, sources=source.model_dump(mode="json"),
        reference=tree.model_dump(mode="json"), suite=suite.model_dump(mode="json"))


def _errors(first, argument):
    args = {argument: "fault-case"}
    return [dict(id=name, arguments=args, observations=[observation(first, args, payload, error)],
        expected_calls=[] if name == "access-denied" else [call(first, **args)],
        expected_status="blocked", authenticated=name != "access-denied")
        for name, payload, error in (("provider-error", {}, True), ("missing-fields", {}, False), ("access-denied", {}, False))]


def cases():
    rows = []
    definitions = [
        ("invoice", "finance", "case_id", "read_invoice_policy", "inspect_invoice", ("settled", "waiver_exists"),
            "settled means the invoice is settled; waiver_exists means an existing accounting waiver permits inspection.",
            "An invoice may", lambda b: b[0] or b[1]),
        ("release", "release-management", "change_id", "release_decision", "read_release_windows", ("decision_present", "stale", "permitted"),
            "decision_present means the decision exists; stale means it is stale (not current); permitted means it permits window inspection.",
            "The decision must", lambda b: b[0] and not b[1] and b[2]),
        ("dispatch", "logistics", "shipment_id", "get_dispatch_decision", "preview_dispatch", ("valid", "fast_track", "signed_review"),
            "valid means the decision is valid; fast_track means fast-track eligibility; signed_review means an existing signed review.",
            "Preview requires", lambda b: b[0] and (b[1] or b[2])),
        ("contacts", "privacy", "contact_id", "contact_flags", "read_contact_card", ("dnc", "region_certified"),
            "dnc is the do-not-contact flag; region_certified means regional certification exists. Flags do not create consent.",
            "Reading a card", lambda b: not b[0] and b[1]),
    ]
    for name, domain, arg, first, second, fields, meaning, quote, permits in definitions:
        contracts = [_contract(first, {arg: "string"}, dict.fromkeys(fields, "boolean"), "Read existing facts only. " + meaning),
            _contract(second, {arg: "string"}, {"value": "string"}, "Read the requested record only; performs no approval, mutation or policy checks.")]
        files, source = _source(name, arg, contracts)
        sid, stopid = _line(source, quote), _line(source, "stop unsupported")
        finalid = _line(source, "complete the read path")
        stop = [end("unsupported", stopid)]
        success = [read(second, "detail", {arg: ref("input", arg)}, finalid), end(source_id=finalid)]
        # Independent reference in existing branches, not authoring macros.
        if name == "invoice":
            policy = [branch(ref("facts", fields[0]), True, success,
                [branch(ref("facts", fields[1]), True, [{**success[0], "bind": "detail_alternate"}, success[1]],
                    stop, sid, stopid)], sid, stopid)]
        elif name == "dispatch":
            alternative = [branch(ref("facts", fields[1]), True, success,
                [branch(ref("facts", fields[2]), True, [{**success[0], "bind": "detail_alternate"}, success[1]], stop, sid, stopid)], sid, stopid)]
            policy = [branch(ref("facts", fields[0]), True, alternative, stop, sid, stopid)]
        else:
            policy = success
            for field in reversed(fields):
                policy = [branch(ref("facts", field), field not in {"stale", "dnc"}, policy, stop, sid, stopid)]
        steps = [read(first, "facts", {arg: ref("input", arg)}, _line(source, first)), *policy]
        scenarios = []
        for flags in itertools.product((False, True), repeat=len(fields)):
            args = {arg: name + "-case"}
            expected = [call(first, **args)] + ([call(second, **args)] if permits(flags) else [])
            scenarios.append(dict(id="bits-" + "".join(str(int(v)) for v in flags), arguments=args,
                observations=[observation(first, args, dict(zip(fields, flags, strict=True))), observation(second, args, {"value": "inert"})],
                expected_calls=expected, expected_status="read_path_completed" if permits(flags) else "unsupported"))
        rows.append(_case(name, domain, files, source, steps, scenarios + _errors(first, arg)))

    contracts = [_contract("resolve_asset", {"alias": "string"}, {"retired": "boolean", "canonical_key": "string"},
        "Resolve alias to the canonical asset key; retired indicates the asset is retired."),
        _contract("read_asset_metric", {"asset_key": "string"}, {"value": "string"}, "Read a metric using the canonical asset key.")]
    files, source = _source("assets", "alias", contracts)
    sid = _line(source, "If the returned")
    finalid = _line(source, "complete the read path")
    steps = [read("resolve_asset", "asset", {"alias": ref("input", "alias")}, _line(source, "Use the caller")),
        branch(ref("asset", "retired"), True, [end("unsupported", sid)],
            [read("read_asset_metric", "metric", {"asset_key": ref("asset", "canonical_key")}, sid), end(source_id=finalid)], sid)]
    scenarios = [dict(id="retired-" + str(retired), arguments={"alias": "old-alias"},
        observations=[observation("resolve_asset", {"alias": "old-alias"}, {"retired": retired, "canonical_key": "canonical-47"}),
            observation("read_asset_metric", {"asset_key": "canonical-47"}, {"value": "inert"})],
        expected_calls=[call("resolve_asset", alias="old-alias")] + ([] if retired else [call("read_asset_metric", asset_key="canonical-47")]),
        expected_status="unsupported" if retired else "read_path_completed") for retired in (False, True)]
    rows.append(_case("assets", "asset-management", files, source, steps, scenarios + _errors("resolve_asset", "alias")))

    files, source = _source("archive", "archive_id", [_contract("read_archive_metadata", {"archive_id": "string"}, {"value": "string"},
        "Read archive metadata only; cannot execute precheck scripts.")])
    sid = _line(source, "The script is supplied")
    rows.append(_case("archive", "backup", files, source, [end("unsupported", sid)],
        [dict(id="runner-absent", arguments={"archive_id": "archive-3"}, observations=[], expected_calls=[], expected_status="unsupported")],
        ("The script is supplied, but an approved runner contract is absent.",)))
    return rows


def followup_cases():
    """New inputs after the expression v2 freeze; still same-author development.

    Independent manual policy functions define private finite labels, never
    sent to authoring. Reference uses full branching with separate leaf reads,
    unlike the shared-continuation expression lowering under test.
    """
    rows = []
    definitions = [
        ("membership", "membership", "request_key", "membership_facts", "benefit_preview",
            ("suspended", "tier_covers", "sponsor_cleared"),
            "suspended means enrollment is suspended; tier_covers means the tier covers this request; sponsor_cleared means existing sponsor clearance covers it.",
            lambda b: not b[0] and (b[1] or b[2])),
        ("dataset", "data-governance", "dataset_key", "dataset_flags", "dataset_summary",
            ("embargoed", "consent_missing"),
            "embargoed=true means 存在禁运; consent_missing=true means 缺少同意. False means that obstacle is absent.",
            lambda b: not (b[0] or b[1])),
        ("storage", "storage", "replica_key", "replica_facts", "replica_details",
            ("synchronized", "quarantined", "emergency_lease"),
            "synchronized means the replica is synchronized; quarantined means it is quarantined; emergency_lease means an existing emergency-read lease permits this inspection.",
            lambda b: (b[0] and not b[1]) or b[2]),
        ("helpdesk", "support", "ticket_key", "ticket_access_facts", "ticket_evidence",
            ("confirmed", "sensitive", "delegated"),
            "confirmed means the decision is confirmed; sensitive means this ticket is sensitive; delegated means delegated access exists.",
            lambda b: b[0] and (not b[1] or b[2])),
        ("certificate", "certificates", "dns_name", "resolve_certificate", "read_certificate", ("expired",),
            "expired means the resolved certificate is expired; serial_number is the resolved canonical certificate serial, not the DNS name.",
            lambda b: not b[0]),
    ]
    for name, domain, arg, first, second, fields, meaning, policy in definitions:
        returned = dict.fromkeys(fields, "boolean")
        if name == "certificate":
            returned["serial_number"] = "string"
        target_arg = "cert_serial" if name == "certificate" else arg
        contracts = [_contract(first, {arg: "string"}, returned, "Read existing facts only. " + meaning),
            _contract(second, {target_arg: "string"}, {"value": "string"}, "Read requested details only, without mutation or implicit access checks.")]
        files, source = _source(name, arg, contracts)
        finalid, stopid = _line(source, "complete the read path"), _line(source, "unsupported")
        policyid = {"membership": "A suspended", "dataset": "只要存在", "storage": "The normal route",
            "helpdesk": "The decision must", "certificate": "If the resolved"}[name]
        sid = _line(source, policyid)
        arguments = {target_arg: ref("facts", "serial_number") if name == "certificate" else ref("input", arg)}

        def reference(flags):
            if len(flags) == len(fields):
                return ([read(second, "leaf_" + "".join(str(int(v)) for v in flags), arguments, finalid), end(source_id=finalid)]
                    if policy(flags) else [end("unsupported", stopid)])
            return [branch(ref("facts", fields[len(flags)]), True, reference(flags + (True,)),
                reference(flags + (False,)), sid, stopid)]

        steps = [read(first, "facts", {arg: ref("input", arg)}, _line(source, first)), *reference(())]
        scenarios = []
        for flags in itertools.product((False, True), repeat=len(fields)):
            caller = {arg: name + "-request"}
            payload = dict(zip(fields, flags, strict=True))
            if name == "certificate":
                payload["serial_number"] = "SERIAL-53"
            downstream = {target_arg: "SERIAL-53" if name == "certificate" else caller[arg]}
            scenarios.append(dict(id="bits-" + "".join(str(int(v)) for v in flags), arguments=caller,
                observations=[observation(first, caller, payload), observation(second, downstream, {"value": "inert"})],
                expected_calls=[call(first, **caller)] + ([call(second, **downstream)] if policy(flags) else []),
                expected_status="read_path_completed" if policy(flags) else "unsupported"))
        rows.append(_case(name, domain, files, source, steps, scenarios + _errors(first, arg)))
    files, source = _source("capacity", "pool_key", [_contract("capacity_report", {"pool_key": "string"}, {"value": "string"},
        "Read a report only; no eligibility decision or policy interpretation is available.")])
    sid = _line(source, "The allocation reference")
    rows.append(_case("capacity", "capacity-planning", files, source, [end("unsupported", sid)],
        [dict(id="reference-absent", arguments={"pool_key": "pool-4"}, observations=[], expected_calls=[], expected_status="unsupported")],
        ("Required allocation reference is not supplied; cannot establish eligibility.",)))
    return rows
