"""One explicitly authorized engineering exception, never a research retry API.

Seals and the fixed SQLite claim prevent cooperative accidental replay; they are
not signatures and cannot defend against an operator editing/deleting the DB.
The halted parent is read-only. A lost/crashed claim is spent, not resumable.
"""
from __future__ import annotations

import json
from pathlib import Path
import secrets
import sqlite3

from evaluation.bounded_pilot import checked_seal, seal
from evaluation.bounded_preflight import file_digest
from evaluation.bounded_transport import strict_json
from network_runtime.contracts import sha256_json


PARENT_STUDY_ID = "r0-controller-engineering-v1"
CHILD_STUDY_ID = PARENT_STUDY_ID + "-reacceptance-1"
SCHEMA = "ensuredskill.io/r0-engineering-reacceptance/v1"
LIMITS = {"child_assigned_arms": 24, "total_assigned_arms": 48,
          "max_total_started_arms": 25, "actual_model_calls": 0,
          "research_permission_changed": False, "original_r0_window_restarted": False}
PARENT_FILES = ("report.json", "pre-run.json", "protocol.json", "reference-freeze.json",
                "dependencies.json", "controller/report.json", "controller/freeze.json")
TABLE_ORDER = {"studies": "study_id", "candidates": "candidate_id",
               "development_cases": "case_id", "arms": "arm_id", "calls": "call_id"}


def _connect(registry, *, readonly):
    path = Path(registry)
    if path.is_symlink() or not path.is_file():
        raise ValueError("existing regular engineering registry required")
    db = sqlite3.connect(path.resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
                         uri=True, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    if readonly:
        db.execute("PRAGMA query_only=ON")
    return db


def _snapshot(db):
    return {table: [dict(row) for row in db.execute(
        f"SELECT * FROM {table} WHERE study_id=? ORDER BY {order}", (PARENT_STUDY_ID,))]
        for table, order in TABLE_ORDER.items()}


def parent_snapshot(registry):
    """Raw persisted parent rows only; no clock observations or ledger writes."""
    db = _connect(registry, readonly=True)
    try:
        db.execute("BEGIN")
        return _snapshot(db)
    finally:
        db.close()


def _parent_files(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("existing original acceptance directory required")
    found = {}
    for name in PARENT_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("original parent evidence missing: " + name)
        found[name] = checked_seal(strict_json(path.read_bytes()))
    return found


def _check_parent(snapshot, evidence):
    report, controller = evidence["report.json"], evidence["controller/report.json"]
    before, frozen = evidence["pre-run.json"], evidence["controller/freeze.json"]
    budget = controller.get("budget", {})
    if (len(snapshot["studies"]) != 1 or snapshot["studies"][0]["status"] != "halted"
            or len(snapshot["arms"]) != 1 or snapshot["arms"][0]["status"] != "measurement_invalid"
            or snapshot["calls"] or len(snapshot["candidates"]) != 1
            or snapshot["candidates"][0]["phase"] != "development"
            or report.get("mechanical_acceptance_passed") is not False
            or report.get("actualModelCalls") != 0 or report.get("liveGenerationEnabled") is not False
            or controller.get("study_id") != PARENT_STUDY_ID or controller.get("assigned_arms") != 24
            or controller.get("controllerMechanicsPassed") is not False
            or len(controller.get("rows", [])) != 1 or len(controller.get("unrun", [])) != 23
            or controller["rows"][0].get("terminal") != "measurement_invalid"
            or controller["rows"][0].get("arm_id") != snapshot["arms"][0]["arm_id"]
            or report.get("controller_report_digest") != controller["digest"]
            or controller.get("freeze_digest") != frozen["digest"]
            or snapshot["studies"][0]["protocol_digest"] != frozen["digest"]
            or report.get("dependency_digest") != evidence["dependencies.json"]["digest"]
            or before.get("dependency_digest") != report.get("dependency_digest")
            or report.get("protocol_digest") != evidence["protocol.json"]["digest"]
            or before.get("protocol_digest") != report.get("protocol_digest")
            or frozen.get("protocol_digest") != report.get("protocol_digest")
            or before.get("review_digest") != evidence["reference-freeze.json"]["digest"]
            or report.get("review_digest") != before.get("review_digest")
            or before.get("dialogues_digest") != sha256_json(frozen.get("dialogues"))):
        raise ValueError("only the original one-arm measurement-invalid zero-call failure qualifies")
    # The original sealed report contains the ledger's enriched view. Compare
    # every persisted field while ignoring its derived usage/elapsed metadata.
    if budget.get("study") != snapshot["studies"][0] or budget.get("calls") != []:
        raise ValueError("parent report and read-only ledger snapshot differ")
    for table in ("candidates", "arms"):
        reported = budget.get(table, [])
        if len(reported) != len(snapshot[table]) or any(
                {key: reported[index].get(key) for key in row} != row
                for index, row in enumerate(snapshot[table])):
            raise ValueError("parent report and read-only ledger snapshot differ")


def prepare_amendment(parent_directory, registry, authorization):
    """Read-only construction for the approving host to seal/save; never claim."""
    directory, registry = Path(parent_directory).resolve(), Path(registry).resolve()
    evidence, snapshot = _parent_files(directory), parent_snapshot(registry)
    _check_parent(snapshot, evidence)
    value = seal({"schema": SCHEMA, "parent_study_id": PARENT_STUDY_ID, "child_study_id": CHILD_STUDY_ID,
        "registry_path": str(registry), "parent_directory": str(directory),
        "parent_report_digest": evidence["report.json"]["digest"],
        "parent_files": {name: file_digest(directory / name) for name in PARENT_FILES},
        "parent_ledger_snapshot": snapshot,
        "protocol_digest": evidence["pre-run.json"]["protocol_digest"],
        "review_digest": evidence["pre-run.json"]["review_digest"],
        "dialogues_digest": evidence["pre-run.json"]["dialogues_digest"],
        "authorization": authorization, "limits": LIMITS})
    _check_amendment(value, registry)
    return value


def _check_amendment(value, registry):
    value = checked_seal(value)
    required = {"schema", "parent_study_id", "child_study_id", "registry_path", "parent_directory",
                "parent_report_digest", "parent_files", "parent_ledger_snapshot", "protocol_digest",
                "review_digest", "dialogues_digest", "authorization", "limits", "digest"}
    auth = value.get("authorization")
    if (set(value) != required or value["schema"] != SCHEMA
            or value["parent_study_id"] != PARENT_STUDY_ID or value["child_study_id"] != CHILD_STUDY_ID
            or value["registry_path"] != str(Path(registry).resolve())
            or value["limits"] != LIMITS or not isinstance(auth, dict)
            or set(auth) != {"source", "statement", "approved_at", "scope"}
            or any(not isinstance(v, str) or not v.strip() for v in auth.values())
            or auth["scope"] != "one_zero_inference_engineering_reacceptance"):
        raise ValueError("explicit sealed one-time engineering authorization required")
    directory = Path(value["parent_directory"])
    if not directory.is_absolute():
        raise ValueError("absolute original parent evidence path required")
    evidence = _parent_files(directory)
    if (value["parent_files"] != {name: file_digest(directory / name) for name in PARENT_FILES}
            or value["parent_report_digest"] != evidence["report.json"]["digest"]):
        raise ValueError("original parent evidence drift")
    snapshot = parent_snapshot(registry)
    _check_parent(snapshot, evidence)
    if snapshot != value["parent_ledger_snapshot"]:
        raise ValueError("original parent ledger drift")
    for key in ("protocol_digest", "review_digest", "dialogues_digest"):
        if value[key] != evidence["pre-run.json"][key]:
            raise ValueError("original task/review/dialogue binding changed")
    return value


def load_amendment(path, registry, *, bindings=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("explicit sealed amendment file required")
    value = _check_amendment(strict_json(path.read_bytes()), registry)
    if bindings is not None and any(value.get(key) != digest for key, digest in bindings.items()):
        raise ValueError("reacceptance cannot change protocol, review or dialogues")
    return value


def dependency_files(path, value):
    return {"reacceptance/amendment": Path(path).resolve(), **{
        "reacceptance/parent/" + name: Path(value["parent_directory"]) / name for name in PARENT_FILES}}


def claim(registry, value, output):
    """One atomic parent claim, before any child study/arm/request. No undo."""
    value = _check_amendment(value, registry)
    receipt = seal({"parent_study_id": PARENT_STUDY_ID, "child_study_id": CHILD_STUDY_ID,
        "amendment_digest": value["digest"], "registry_path": str(Path(registry).resolve()),
        "output": str(Path(output).resolve()), "nonce": secrets.token_hex(24),
        "parent_ledger_snapshot": value["parent_ledger_snapshot"],
        "protocol_digest": value["protocol_digest"], "dialogues_digest": value["dialogues_digest"]})
    db = _connect(registry, readonly=False)
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS engineering_reacceptance_claims ("
                   "parent_study_id TEXT PRIMARY KEY, child_study_id TEXT NOT NULL UNIQUE, "
                   "receipt TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0)")
        if _snapshot(db) != value["parent_ledger_snapshot"]:
            raise ValueError("original parent ledger drift")
        if db.execute("SELECT 1 FROM studies WHERE study_id=?", (CHILD_STUDY_ID,)).fetchone():
            raise ValueError("reacceptance child already exists; no replay")
        try:
            db.execute("INSERT INTO engineering_reacceptance_claims VALUES (?, ?, ?, 0)",
                       (PARENT_STUDY_ID, CHILD_STUDY_ID, json.dumps(receipt, sort_keys=True)))
        except sqlite3.IntegrityError as exc:
            raise ValueError("reacceptance parent already claimed; no replay") from exc
        db.execute("COMMIT")
        return receipt
    finally:
        db.close()


def verify_parent(registry, receipt):
    receipt = checked_seal(receipt)
    if parent_snapshot(registry) != receipt["parent_ledger_snapshot"]:
        raise ValueError("halted parent changed during reacceptance")


def consume_claim(registry, receipt, output, protocol_digest, dialogues_digest):
    """Controller entry consumes once; possessing a spent receipt grants nothing."""
    receipt = checked_seal(receipt)
    if (receipt.get("registry_path") != str(Path(registry).resolve())
            or receipt.get("output") != str(Path(output).resolve())
            or receipt.get("protocol_digest") != protocol_digest
            or receipt.get("dialogues_digest") != dialogues_digest):
        raise ValueError("reacceptance claim context differs")
    db = _connect(registry, readonly=False)
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM engineering_reacceptance_claims WHERE parent_study_id=?",
                         (PARENT_STUDY_ID,)).fetchone()
        if (row is None or row["consumed"] or strict_json(row["receipt"]) != receipt
                or row["child_study_id"] != CHILD_STUDY_ID):
            raise ValueError("unclaimed or spent engineering reacceptance")
        if _snapshot(db) != receipt["parent_ledger_snapshot"]:
            raise ValueError("halted parent changed during reacceptance")
        if db.execute("SELECT 1 FROM studies WHERE study_id=?", (CHILD_STUDY_ID,)).fetchone():
            raise ValueError("reacceptance child already exists; no replay")
        db.execute("UPDATE engineering_reacceptance_claims SET consumed=1 WHERE parent_study_id=?",
                   (PARENT_STUDY_ID,))
        db.execute("COMMIT")
        return CHILD_STUDY_ID
    finally:
        db.close()


def linked_counts(receipt, child_started):
    return {"amendment_digest": receipt["amendment_digest"], "claim_digest": receipt["digest"],
            "parent_study_id": PARENT_STUDY_ID, "child_study_id": CHILD_STUDY_ID,
            "parent_assigned_arms": 24, "parent_started_arms": 1, "parent_measurement_invalid_arms": 1,
            "child_assigned_arms": 24, "child_started_arms": child_started,
            "total_assigned_arms": 48, "total_started_arms": None if child_started is None else 1 + child_started,
            "max_total_started_arms": 25, "parent_preserved": True,
            "original_r0_window_restarted": False, "research_permission_changed": False}
