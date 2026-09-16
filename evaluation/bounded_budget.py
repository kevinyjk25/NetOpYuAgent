"""Persistent, cooperative-operator budget enforcement for the bounded pilot.

This is an accounting gate, not an Effect capability or research admission.
Every physical model request must reserve before transport and settle afterward.
An unsettled reservation survives a process restart and cannot be retried. An
operator who deletes or edits this SQLite database is outside the threat model;
the runner must choose one fixed database path, never an output-directory path.
Epoch deadlines survive restarts. Clock rollback fails closed; callers still
must enforce returned deadlines in the transport and authoritative executor.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


MODEL_STAGES = frozenset({"agent", "compiler", "runtime", "revision", "fallback"})
CAPS = {"study_arms": 120, "study_model_requests": 1200,
        "development_candidates": 2, "confirmation_candidates": 1,
        "cases_per_candidate": 12, "development_arms": 24, "confirmation_arms": 72,
        "arm_seconds": 420, "arm_model_requests": 10, "arm_input_tokens": 64000,
        "arm_output_tokens": 6000, "compiler_requests": 2, "revision_requests": 1,
        "effect_recovery_seconds": 60, "offline_requests": 48,
        "offline_requests_per_case": 2, "offline_seconds": 180,
        "offline_input_tokens": 24000, "offline_output_tokens": 3000}
TERMINAL_STATUSES = frozenset({"completed", "failed", "incomplete", "timeout",
                               "cancelled", "outcome_unknown", "measurement_invalid"})


class BudgetError(ValueError):
    """No action is authorized by a rejected budget operation."""


class _LedgerConnection(sqlite3.Connection):
    """Per-transaction clock observations, not shared between worker threads."""


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise BudgetError(f"invalid {label}")
    return value


def _integer(value):
    return type(value) is int and value >= 0


def _id(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


class BudgetLedger:
    """SQLite transactions serialize reservations across independent processes.

    ``clock`` must return epoch seconds; injection is for deterministic tests.
    Arm repetition is one-based; A and B have identical ceilings. Registration
    is idempotent, but an existing arm/request identity can never start again.
    A pause has no resume API. Settling/closing already-started work is permitted
    after a halt for accounting only, never to authorize another operation.
    """

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path).resolve()
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY, protocol_digest TEXT NOT NULL,
                    status TEXT NOT NULL, reason TEXT, created_at REAL NOT NULL,
                    last_clock REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies,
                    candidate_digest TEXT NOT NULL, phase TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(study_id, phase, candidate_digest), UNIQUE(study_id, phase, ordinal));
                CREATE TABLE IF NOT EXISTS development_cases (
                    study_id TEXT NOT NULL REFERENCES studies, case_id TEXT NOT NULL,
                    context_digest TEXT, PRIMARY KEY(study_id, case_id));
                CREATE TABLE IF NOT EXISTS arms (
                    arm_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies,
                    candidate_id TEXT NOT NULL REFERENCES candidates, case_id TEXT NOT NULL,
                    repetition INTEGER NOT NULL, arm TEXT NOT NULL, context_digest TEXT NOT NULL,
                    status TEXT NOT NULL, reason TEXT, started_at REAL NOT NULL,
                    deadline REAL NOT NULL, finished_at REAL, elapsed_ms REAL,
                    UNIQUE(candidate_id, case_id, repetition, arm));
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies,
                    candidate_id TEXT NOT NULL REFERENCES candidates,
                    arm_id TEXT REFERENCES arms, case_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    kind TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
                    reserved_input INTEGER NOT NULL, reserved_output INTEGER NOT NULL,
                    actual_input INTEGER, actual_output INTEGER,
                    charged_input INTEGER NOT NULL, charged_output INTEGER NOT NULL,
                    started_at REAL NOT NULL, deadline REAL NOT NULL, settled_at REAL,
                    elapsed_ms REAL, diagnostic TEXT);
            """)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None, factory=_LedgerConnection)
        db.clock_observations = {}
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BudgetError:
                db.rollback()
                # A rejected/expired attempt must not erase its clock high-water
                # mark and permit a backward clock adjustment to buy more time.
                db.execute("BEGIN IMMEDIATE")
                for study_id, observed_at in db.clock_observations.items():
                    db.execute("UPDATE studies SET last_clock=MAX(last_clock, ?) WHERE study_id=?",
                               (observed_at, study_id))
                db.commit()
                raise
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    def _now(self):
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0:
            raise BudgetError("clock must provide finite nonnegative epoch seconds")
        return float(now)

    @staticmethod
    def _row(db, table, key, value):
        # Identifiers are internal constants, never caller-provided SQL.
        row = db.execute(f"SELECT * FROM {table} WHERE {key}=?", (value,)).fetchone()
        if row is None:
            raise BudgetError(f"unknown {key}")
        return dict(row)

    def _study(self, db, study_id, now, *, active=True):
        study = self._row(db, "studies", "study_id", study_id)
        db.clock_observations[study_id] = now
        if now < study["last_clock"]:
            raise BudgetError("clock rollback; no new operation permitted")
        if active and study["status"] != "active":
            raise BudgetError("study halted: " + str(study["reason"]))
        db.execute("UPDATE studies SET last_clock=? WHERE study_id=?", (now, study_id))
        return study

    @staticmethod
    def _halt(db, study_id, reason):
        db.execute("UPDATE studies SET status='halted', reason=COALESCE(reason, ?) WHERE study_id=?",
                   (reason, study_id))

    @staticmethod
    def _no_pending(db, study_id):
        if db.execute("SELECT 1 FROM calls WHERE study_id=? AND status='reserved'", (study_id,)).fetchone():
            raise BudgetError("unsettled request outcome; inspect or settle, never replay")

    def _candidate(self, db, study_id, candidate_id):
        candidate = self._row(db, "candidates", "candidate_id", candidate_id)
        if candidate["study_id"] != study_id:
            raise BudgetError("candidate belongs to another study")
        latest = db.execute("SELECT candidate_id FROM candidates WHERE study_id=? "
                            "ORDER BY CASE phase WHEN 'confirmation' THEN 1 ELSE 0 END DESC, ordinal DESC LIMIT 1",
                            (study_id,)).fetchone()
        if latest["candidate_id"] != candidate_id:
            raise BudgetError("candidate is frozen or superseded; no new work")
        return candidate

    def register_study(self, study_id, protocol_digest):
        _identifier(study_id, "study_id")
        _identifier(protocol_digest, "protocol_digest")
        now = self._now()
        with self._transaction() as db:
            old = db.execute("SELECT * FROM studies WHERE study_id=?", (study_id,)).fetchone()
            if old:
                if old["protocol_digest"] != protocol_digest:
                    raise BudgetError("immutable study protocol fingerprint changed")
                return dict(old)
            db.execute("INSERT INTO studies VALUES (?, ?, 'active', NULL, ?, ?)",
                       (study_id, protocol_digest, now, now))
            return self._row(db, "studies", "study_id", study_id)

    def register_candidate(self, study_id, candidate_digest, phase="development"):
        _identifier(candidate_digest, "candidate_digest")
        if phase not in {"development", "confirmation"}:
            raise BudgetError("invalid candidate phase")
        now = self._now()
        with self._transaction() as db:
            self._study(db, study_id, now)
            old = db.execute("SELECT candidate_id FROM candidates WHERE study_id=? AND phase=? AND candidate_digest=?",
                             (study_id, phase, candidate_digest)).fetchone()
            if old:
                return old["candidate_id"]
            self._no_pending(db, study_id)
            if db.execute("SELECT 1 FROM arms WHERE study_id=? AND status='active'", (study_id,)).fetchone():
                raise BudgetError("cannot change candidate during an active arm")
            if db.execute("SELECT 1 FROM candidates WHERE study_id=? AND phase='confirmation'", (study_id,)).fetchone():
                raise BudgetError("one confirmation candidate already frozen")
            count = db.execute("SELECT COUNT(*) FROM candidates WHERE study_id=? AND phase=?", (study_id, phase)).fetchone()[0]
            if count >= CAPS[phase + "_candidates"]:
                raise BudgetError("candidate version budget exhausted")
            if phase == "confirmation":
                last = db.execute("SELECT candidate_digest FROM candidates WHERE study_id=? AND phase='development' "
                                  "ORDER BY ordinal DESC LIMIT 1", (study_id,)).fetchone()
                if not last or last[0] != candidate_digest:
                    raise BudgetError("confirmation must freeze the last development digest")
            candidate_id = _id(study_id, phase, candidate_digest)
            db.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
                       (candidate_id, study_id, candidate_digest, phase, count + 1, now))
            return candidate_id

    def _case(self, db, candidate, case_id, context_digest=None):
        study_id = candidate["study_id"]
        existing = db.execute("SELECT * FROM development_cases WHERE study_id=? AND case_id=?",
                              (study_id, case_id)).fetchone()
        if candidate["phase"] == "confirmation":
            if existing:
                raise BudgetError("confirmation case overlaps development")
            return
        if existing:
            if context_digest is not None and existing["context_digest"] not in {None, context_digest}:
                raise BudgetError("immutable development case context changed")
            if context_digest is not None:
                db.execute("UPDATE development_cases SET context_digest=? WHERE study_id=? AND case_id=?",
                           (context_digest, study_id, case_id))
        else:
            if candidate["ordinal"] != 1:
                raise BudgetError("development cases cannot change in a later candidate")
            if db.execute("SELECT COUNT(*) FROM development_cases WHERE study_id=?", (study_id,)).fetchone()[0] >= 12:
                raise BudgetError("development case budget exhausted")
            db.execute("INSERT INTO development_cases VALUES (?, ?, ?)", (study_id, case_id, context_digest))

    def start_arm(self, study_id, candidate_id, case_id, repetition, arm, context_digest):
        _identifier(case_id, "case_id")
        _identifier(context_digest, "context_digest")
        if arm not in {"A", "B"} or type(repetition) is not int:
            raise BudgetError("arm must be A/B and repetition a one-based integer")
        now = self._now()
        with self._transaction() as db:
            self._study(db, study_id, now)
            candidate = self._candidate(db, study_id, candidate_id)
            max_rep = 1 if candidate["phase"] == "development" else 3
            if not 1 <= repetition <= max_rep:
                raise BudgetError("repetition outside frozen phase budget")
            arm_id = _id(candidate_id, case_id, repetition, arm)
            if db.execute("SELECT 1 FROM arms WHERE arm_id=?", (arm_id,)).fetchone():
                raise BudgetError("arm identity already claimed; no replay")
            self._no_pending(db, study_id)
            if db.execute("SELECT 1 FROM arms WHERE study_id=? AND status='active'", (study_id,)).fetchone():
                raise BudgetError("single worker: finish the active arm first")
            rows = db.execute("SELECT case_id, context_digest FROM arms WHERE candidate_id=?", (candidate_id,)).fetchall()
            cases = {row["case_id"] for row in rows}
            if case_id not in cases and len(cases) >= CAPS["cases_per_candidate"]:
                raise BudgetError("candidate case budget exhausted")
            if any(row["case_id"] == case_id and row["context_digest"] != context_digest for row in rows):
                raise BudgetError("paired/repeated arm context changed")
            if len(rows) >= CAPS[candidate["phase"] + "_arms"]:
                raise BudgetError("candidate arm budget exhausted")
            if db.execute("SELECT COUNT(*) FROM arms WHERE study_id=?", (study_id,)).fetchone()[0] >= CAPS["study_arms"]:
                raise BudgetError("study arm budget exhausted")
            self._case(db, candidate, case_id, context_digest)
            db.execute("INSERT INTO arms VALUES (?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?, NULL, NULL)",
                       (arm_id, study_id, candidate_id, case_id, repetition, arm, context_digest,
                        now, now + CAPS["arm_seconds"]))
            return arm_id

    def _active_arm(self, db, arm_id, now):
        arm = self._row(db, "arms", "arm_id", arm_id)
        self._study(db, arm["study_id"], now)
        self._candidate(db, arm["study_id"], arm["candidate_id"])
        if arm["status"] != "active":
            raise BudgetError("arm terminal or blocked; no further action")
        if now >= arm["deadline"]:
            raise BudgetError("arm wall-clock budget exhausted")
        self._no_pending(db, arm["study_id"])
        return arm

    @staticmethod
    def _tokens(input_tokens, max_output_tokens, input_cap, output_cap):
        if not _integer(input_tokens) or not _integer(max_output_tokens) or max_output_tokens == 0:
            raise BudgetError("token reservations require nonnegative input and positive output integers")
        if input_tokens > input_cap or max_output_tokens > output_cap:
            raise BudgetError("token reservation exceeds per-request limit")

    def reserve_model_call(self, arm_id, request_id, stage, input_tokens, max_output_tokens):
        _identifier(request_id, "request_id")
        if stage not in MODEL_STAGES:
            raise BudgetError("unknown model stage")
        self._tokens(input_tokens, max_output_tokens, CAPS["arm_input_tokens"], CAPS["arm_output_tokens"])
        now = self._now()
        with self._transaction() as db:
            arm = self._active_arm(db, arm_id, now)
            call_id = _id(arm_id, request_id)
            if db.execute("SELECT 1 FROM calls WHERE call_id=?", (call_id,)).fetchone():
                raise BudgetError("request identity already claimed; no replay")
            calls = db.execute("SELECT * FROM calls WHERE arm_id=?", (arm_id,)).fetchall()
            if len(calls) >= CAPS["arm_model_requests"]:
                raise BudgetError("arm model-request budget exhausted")
            if stage in {"compiler", "revision"} and sum(c["stage"] == stage for c in calls) >= CAPS[stage + "_requests"]:
                raise BudgetError(stage + " request budget exhausted")
            if sum(c["charged_input"] for c in calls) + input_tokens > CAPS["arm_input_tokens"]:
                raise BudgetError("arm input-token budget exhausted")
            if sum(c["charged_output"] for c in calls) + max_output_tokens > CAPS["arm_output_tokens"]:
                raise BudgetError("arm output-token budget exhausted")
            count = db.execute("SELECT COUNT(*) FROM calls WHERE study_id=? AND kind='arm'", (arm["study_id"],)).fetchone()[0]
            if count >= CAPS["study_model_requests"]:
                raise BudgetError("study model-request budget exhausted")
            return self._reserve(db, call_id, arm["study_id"], arm["candidate_id"], arm_id,
                                 arm["case_id"], request_id, "arm", stage, input_tokens,
                                 max_output_tokens, now, arm["deadline"])

    def _reserve(self, db, call_id, study_id, candidate_id, arm_id, case_id, request_id,
                 kind, stage, input_tokens, output_tokens, now, deadline):
        db.execute("INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, NULL, NULL, ?, ?, ?, ?, NULL, NULL, NULL)",
                   (call_id, study_id, candidate_id, arm_id, case_id, request_id, kind, stage,
                    input_tokens, output_tokens, input_tokens, output_tokens, now, deadline))
        return self._row(db, "calls", "call_id", call_id)

    def reserve_offline_call(self, study_id, candidate_id, case_id, request_id, input_tokens, max_output_tokens):
        _identifier(case_id, "case_id")
        _identifier(request_id, "request_id")
        self._tokens(input_tokens, max_output_tokens, CAPS["offline_input_tokens"], CAPS["offline_output_tokens"])
        now = self._now()
        with self._transaction() as db:
            self._study(db, study_id, now)
            candidate = self._candidate(db, study_id, candidate_id)
            if candidate["phase"] != "development":
                raise BudgetError("offline screening belongs to development only")
            self._no_pending(db, study_id)
            if db.execute("SELECT 1 FROM arms WHERE study_id=? AND status='active'", (study_id,)).fetchone():
                raise BudgetError("offline requests cannot overlap an arm")
            self._case(db, candidate, case_id)
            call_id = _id(candidate_id, case_id, "offline", request_id)
            if db.execute("SELECT 1 FROM calls WHERE call_id=?", (call_id,)).fetchone():
                raise BudgetError("offline request already claimed; no replay")
            count = db.execute("SELECT COUNT(*) FROM calls WHERE candidate_id=? AND case_id=? AND kind='offline'",
                               (candidate_id, case_id)).fetchone()[0]
            if count >= CAPS["offline_requests_per_case"]:
                raise BudgetError("offline case request budget exhausted")
            if db.execute("SELECT COUNT(*) FROM calls WHERE study_id=? AND kind='offline'", (study_id,)).fetchone()[0] >= CAPS["offline_requests"]:
                raise BudgetError("study offline request budget exhausted")
            return self._reserve(db, call_id, study_id, candidate_id, None, case_id, request_id,
                                 "offline", "compiler", input_tokens, max_output_tokens,
                                 now, now + CAPS["offline_seconds"])

    def settle_call(self, arm_id, request_id, actual_input_tokens=None, actual_output_tokens=None):
        return self._settle(_id(arm_id, request_id), actual_input_tokens, actual_output_tokens)

    def settle_offline_call(self, candidate_id, case_id, request_id, actual_input_tokens=None, actual_output_tokens=None):
        return self._settle(_id(candidate_id, case_id, "offline", request_id),
                            actual_input_tokens, actual_output_tokens)

    def _settle(self, call_id, actual_input_tokens, actual_output_tokens):
        now = self._now()
        with self._transaction() as db:
            call = self._row(db, "calls", "call_id", call_id)
            if call["status"] != "reserved":
                raise BudgetError("request already settled; no replay or usage replacement")
            study = self._row(db, "studies", "study_id", call["study_id"])
            values = (actual_input_tokens, actual_output_tokens)
            valid = tuple(_integer(v) for v in values)
            status, diagnostic = "settled", None
            if any(v is not None and not ok for v, ok in zip(values, valid)):
                status, diagnostic = "invalid_usage", "invalid/negative actual usage; reservation retained"
            elif not all(valid):
                status, diagnostic = "unknown", "actual usage unknown; reservation retained"
            elif actual_input_tokens > call["reserved_input"] or actual_output_tokens > call["reserved_output"]:
                status, diagnostic = "budget_exceeded", "actual usage exceeded its reservation"
            if now < study["last_clock"] or now < call["started_at"]:
                status, diagnostic = "invalid_usage", "clock rollback during settlement"
            elif now > call["deadline"]:
                status, diagnostic = "budget_exceeded", "request exceeded wall-clock deadline"
            charged = [values[i] if status == "settled" else max(reserved, values[i] if valid[i] else reserved)
                       for i, reserved in enumerate((call["reserved_input"], call["reserved_output"]))]
            db.execute("UPDATE calls SET status=?, actual_input=?, actual_output=?, charged_input=?, charged_output=?, "
                       "settled_at=?, elapsed_ms=?, diagnostic=? WHERE call_id=?",
                       (status, actual_input_tokens if valid[0] else None, actual_output_tokens if valid[1] else None,
                        *charged, now, max(0, now - call["started_at"]) * 1000, diagnostic, call_id))
            db.execute("UPDATE studies SET last_clock=MAX(last_clock, ?) WHERE study_id=?", (now, call["study_id"]))
            if status != "settled":
                self._halt(db, call["study_id"], diagnostic)
                if call["arm_id"] is not None:
                    db.execute("UPDATE arms SET status=CASE WHEN finished_at IS NULL THEN 'blocked' ELSE status END, "
                               "reason=COALESCE(reason, ?) WHERE arm_id=?", (diagnostic, call["arm_id"]))
            return self._row(db, "calls", "call_id", call_id)

    def guard_effect(self, arm_id, reserve_seconds=60):
        if isinstance(reserve_seconds, bool) or not isinstance(reserve_seconds, (int, float)) or not math.isfinite(reserve_seconds) or reserve_seconds < 60:
            raise BudgetError("Effect requires at least 60 seconds for verify/reconcile/recovery")
        now = self._now()
        with self._transaction() as db:
            arm = self._active_arm(db, arm_id, now)
            remaining = arm["deadline"] - now
            if remaining < reserve_seconds:
                raise BudgetError("insufficient Effect recovery reserve")
            return {"arm_id": arm_id, "deadline": arm["deadline"], "remaining_seconds": remaining,
                    "recovery_reserve_seconds": reserve_seconds, "effect_authority_granted": False}

    def check_arm(self, arm_id):
        """Gate local work before/after a callback; this grants no capability.

        Recovery after a stopped or expired arm belongs to the authoritative
        runtime's separately disclosed recovery path, never new pilot work.
        """
        now = self._now()
        with self._transaction() as db:
            arm = self._active_arm(db, arm_id, now)
            return {"arm_id": arm_id, "deadline": arm["deadline"],
                    "remaining_seconds": arm["deadline"] - now, "authority_granted": False}

    def finish_arm(self, arm_id, status, elapsed_ms=None):
        if status not in TERMINAL_STATUSES:
            raise BudgetError("invalid terminal status")
        if elapsed_ms is not None and (isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float))
                                       or not math.isfinite(elapsed_ms) or elapsed_ms < 0):
            raise BudgetError("invalid reported arm duration")
        now = self._now()
        with self._transaction() as db:
            arm = self._row(db, "arms", "arm_id", arm_id)
            if arm["finished_at"] is not None:
                raise BudgetError("arm already terminal; no replacement")
            study = self._row(db, "studies", "study_id", arm["study_id"])
            actual_ms = max(0, now - arm["started_at"]) * 1000
            charged_ms = max(actual_ms, elapsed_ms or 0)
            reason = arm["reason"]
            pending = db.execute("SELECT 1 FROM calls WHERE arm_id=? AND status='reserved'", (arm_id,)).fetchone()
            if pending or arm["status"] == "blocked":
                status, reason = "outcome_unknown", reason or "unsettled request at arm termination"
            elif now < study["last_clock"]:
                status, reason = "measurement_invalid", "clock rollback at arm termination"
            elif charged_ms > CAPS["arm_seconds"] * 1000:
                status, reason = "timeout", "arm wall-clock budget exceeded"
            if pending or status in {"outcome_unknown", "measurement_invalid", "timeout", "cancelled"}:
                self._halt(db, arm["study_id"], reason or status)
            db.execute("UPDATE arms SET status=?, reason=?, finished_at=?, elapsed_ms=? WHERE arm_id=?",
                       (status, reason, now, charged_ms, arm_id))
            db.execute("UPDATE studies SET last_clock=MAX(last_clock, ?) WHERE study_id=?", (now, arm["study_id"]))
            return self._row(db, "arms", "arm_id", arm_id)

    def pause_study(self, study_id, reason):
        _identifier(reason, "pause reason")
        now = self._now()
        with self._transaction() as db:
            self._row(db, "studies", "study_id", study_id)
            self._halt(db, study_id, reason)
            db.execute("UPDATE studies SET last_clock=MAX(last_clock, ?) WHERE study_id=?", (now, study_id))
            return self._row(db, "studies", "study_id", study_id)

    def inspect_arm(self, arm_id):
        """All persisted arm usage, including requests from an earlier process."""
        now = self._now()
        with self._transaction() as db:
            arm = self._row(db, "arms", "arm_id", arm_id)
            study = self._row(db, "studies", "study_id", arm["study_id"])
            calls = [dict(row) for row in db.execute(
                "SELECT * FROM calls WHERE arm_id=? ORDER BY started_at, call_id", (arm_id,))]
            arm["usage"] = self._usage(calls)
            arm["observed_elapsed_ms"] = arm["elapsed_ms"] if arm["elapsed_ms"] is not None else max(0, now - arm["started_at"]) * 1000
            arm["remaining_seconds"] = max(0, arm["deadline"] - now) if arm["status"] == "active" else 0
            return {"arm": arm, "calls": calls, "study_status": study["status"],
                    "clock_rollback": now < study["last_clock"]}

    def snapshot(self, study_id):
        """A coherent read snapshot; it never releases reservations or resumes work."""
        now = self._now()
        with self._transaction() as db:
            study = self._row(db, "studies", "study_id", study_id)
            candidates = [dict(r) for r in db.execute("SELECT * FROM candidates WHERE study_id=? ORDER BY created_at, phase, ordinal", (study_id,))]
            arms = [dict(r) for r in db.execute("SELECT * FROM arms WHERE study_id=? ORDER BY started_at, arm_id", (study_id,))]
            calls = [dict(r) for r in db.execute("SELECT * FROM calls WHERE study_id=? ORDER BY started_at, call_id", (study_id,))]
            for arm in arms:
                subset = [c for c in calls if c["arm_id"] == arm["arm_id"]]
                arm["usage"] = self._usage(subset)
                arm["observed_elapsed_ms"] = arm["elapsed_ms"] if arm["elapsed_ms"] is not None else max(0, now - arm["started_at"]) * 1000
                arm["remaining_seconds"] = max(0, arm["deadline"] - now) if arm["status"] == "active" else 0
            return {"study": study, "candidates": candidates, "arms": arms, "calls": calls,
                    "caps": dict(CAPS), "model_stages": sorted(MODEL_STAGES),
                    "usage": self._usage([c for c in calls if c["kind"] == "arm"]),
                    "offline_usage": self._usage([c for c in calls if c["kind"] == "offline"]),
                    "arm_count": len(arms), "arm_wall_ms": sum(a["observed_elapsed_ms"] for a in arms),
                    "pending_outcome": any(c["status"] == "reserved" for c in calls),
                    "clock_rollback": now < study["last_clock"],
                    "authority": "accounting_only_cooperative_operator_no_database_deletion_protection"}

    @staticmethod
    def _usage(calls):
        return {"model_requests": len(calls), "charged_input_tokens": sum(c["charged_input"] for c in calls),
                "charged_output_tokens": sum(c["charged_output"] for c in calls),
                "known_input_tokens": sum(c["actual_input"] or 0 for c in calls),
                "known_output_tokens": sum(c["actual_output"] or 0 for c in calls),
                "usage_complete": all(c["actual_input"] is not None and c["actual_output"] is not None
                                      and c["status"] != "invalid_usage" for c in calls),
                "unknown_requests": sum(c["status"] in {"reserved", "unknown", "invalid_usage"} for c in calls)}
