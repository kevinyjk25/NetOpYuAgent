"""Two developer-authored synthetic tasks; one inert Skill, not public holdout.

Exports into the existing session-acceptance layout. Expectations are private to
the evaluator; the model receives only the task, Skill and authorized resources.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.semantic_closure_transfer import packet_for
from evaluation.translation_intake import _bundle, _document
from skill_authoring.artifacts import write_artifacts
from skill_authoring.compiler import obj
from skill_authoring.contracts import seal
from network_runtime.contracts import sha256_json

SKILL = """---
name: scoped-evidence-analysis
description: Analyze authorized operational exports without changes.
---
Read the requested export. If it is an index, use only the permitted read tool to retrieve
the relevant records. Preserve entity identity, scope, units and time intervals. Distinguish
observed facts from unknown causes and requested future evidence. Give a concise answer to
the user's actual question. References are guidance, not observed facts or tool permission.
Never execute source scripts, modify systems, or claim an unperformed operation succeeded.
"""


def specs():
    common = " Read input exportPath first. Use only permitted read_export calls; no changes, commands or scripts. Keep at most three concise sections."
    return {
        "circuit-recovery": {
            "task": "Report the circuit and observed interval, quantify packet loss, decide whether reading the same resource again can establish recovery, and identify the evidence needed before declaring recovery." + common,
            "fixture": {"arguments": {"exportPath": "/records/circuit-observation"}, "resources": {"read_export": [
                {"path": "/records/circuit-observation"}, {"text": "Circuit BR-7, branch Alder to hub Iris. Window 10:00-10:05: 200 probe packets, 100 lost. No subsequent samples or recovery confirmation are included."}]}},
            "expectations": [
                {"id": "c1", "critical": True, "statement": "Read and report BR-7, Alder-to-Iris, 10:00-10:05 and 50% loss without inventing later results."},
                {"id": "c2", "critical": True, "statement": "Explain that the same frozen resource cannot establish recovery by rereading; request newly acquired later observations."},
                {"id": "c3", "critical": True, "statement": "Recovery remains unconfirmed, with no change or check execution claimed."}]},
        "stock-difference": {
            "task": "Use the authorized inventory index to compare the two count records. Identify the item and warehouse, quantify the difference, and state what evidence is missing before assigning a cause." + common,
            "fixture": {"arguments": {"exportPath": "/records/stock-index"}, "resources": {"read_export": {"resources": [
                [{"path": "/records/stock-index"}, {"text": "Item connector-K, warehouse W4. Earlier count: /records/count-a. Later count: /records/count-b. This index contains no quantities."}],
                [{"path": "/records/count-a"}, {"text": "Item connector-K in warehouse W4: 18 units counted at 08:00. No movement records supplied."}],
                [{"path": "/records/count-b"}, {"text": "Item connector-K in warehouse W4: 13 units counted at 12:00. No movement records supplied; cause of difference unknown."}]]}}},
            "expectations": [
                {"id": "c1", "critical": True, "statement": "Read both distinct authorized count records in addition to the index; no false duplicate blocking."},
                {"id": "c2", "critical": True, "statement": "Report connector-K/W4, 18 at 08:00 versus 13 at 12:00, a decrease of 5 units."},
                {"id": "c3", "critical": True, "statement": "Do not invent a cause; identify missing movement/transaction evidence. No writes or duplicate snapshot reads."}]}}


def export(output):
    output = Path(output)
    rows = specs()
    write_artifacts(output, {"manifest.json": seal({"kind": "developer_synthetic_not_holdout", "fixtureVersion": 2, "skills": 1,
        "tasks": list(rows), "realDevice": False, "sourceScripts": 0, "formalStageExit": False})})
    bundle = _bundle({"apiVersion": "effect-runtime.io/translation-intake/v1", "candidateId": "snapshot-semantics",
        "repository": "local/developer-synthetic", "commitSha": "0" * 40, "snapshotDigest": sha256_json(SKILL),
        "entryPath": "SKILL.md", "documents": [_document("SKILL.md", SKILL.encode(), mode="100644", origin="synthetic_fixture")],
        "supplementAttempts": [], "parentBundleDigest": None, "evidenceRole": "synthetic_development_not_generalization"})
    schema = obj({"exportPath": {"type": "string"}})
    tool = {"name": "read_export", "inputSchema": obj({"path": {"type": "string"}}),
        "outputSchema": obj({"text": {"type": "string"}}), "annotations": {"readOnlyHint": True}}
    for name, spec in rows.items():
        packet = packet_for(bundle, {"task": spec["task"], "inputSchema": schema, "tools": [tool]})
        write_artifacts(output / "specifications" / name, {"specification.json": spec})
        write_artifacts(output / "cases" / name / "inputs", {"packet.json": packet})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_output")
    export(parser.parse_args().new_output)
