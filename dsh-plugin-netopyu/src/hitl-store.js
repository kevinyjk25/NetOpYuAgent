import { createHash, randomUUID } from 'node:crypto'
import { DatabaseSync } from 'node:sqlite'

export function createHitlStore(path) {
  const database = new DatabaseSync(path)
  database.exec(`
    CREATE TABLE IF NOT EXISTS requests (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      call_id TEXT,
      tool_name TEXT NOT NULL,
      arguments_json TEXT NOT NULL,
      status TEXT NOT NULL,
      outcome TEXT,
      reason TEXT,
      created_at TEXT NOT NULL,
      decided_at TEXT,
      completed_at TEXT,
      execution_error INTEGER
    )
  `)
  database.exec(`
    CREATE TABLE IF NOT EXISTS tool_grants (
      grant_id TEXT PRIMARY KEY,
      request_id TEXT NOT NULL,
      token_hash TEXT NOT NULL UNIQUE,
      tool_name TEXT NOT NULL,
      status TEXT NOT NULL,
      issued_at TEXT NOT NULL,
      consumed_at TEXT,
      revoked_at TEXT,
      revoke_reason TEXT
    )
  `)
  database.prepare("UPDATE tool_grants SET status = 'orphaned', revoked_at = ?, revoke_reason = 'plugin restart' WHERE status = 'issued'")
    .run(new Date().toISOString())
  const columns = new Set(database.prepare('PRAGMA table_info(requests)').all().map(column => column.name))
  if (!columns.has('recovery_request_id')) database.exec('ALTER TABLE requests ADD COLUMN recovery_request_id TEXT')
  if (!columns.has('recovered_arguments_json')) database.exec('ALTER TABLE requests ADD COLUMN recovered_arguments_json TEXT')
  if (!columns.has('recovered_at')) database.exec('ALTER TABLE requests ADD COLUMN recovered_at TEXT')
  if (!columns.has('expires_at')) database.exec('ALTER TABLE requests ADD COLUMN expires_at TEXT')
  database.exec(`
    CREATE TABLE IF NOT EXISTS batch_items (
      batch_request_id TEXT NOT NULL,
      item_index INTEGER NOT NULL,
      tool_name TEXT NOT NULL,
      arguments_json TEXT NOT NULL,
      status TEXT NOT NULL,
      result_text TEXT,
      error_text TEXT,
      started_at TEXT,
      completed_at TEXT,
      PRIMARY KEY (batch_request_id, item_index)
    )
  `)
  database.exec(`
    CREATE TABLE IF NOT EXISTS trajectory_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      event_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `)
  database.exec(`
    CREATE TABLE IF NOT EXISTS a2a_continuations (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      peer_agent TEXT NOT NULL,
      interrupt_id TEXT NOT NULL,
      request_json TEXT NOT NULL,
      status TEXT NOT NULL,
      result_text TEXT,
      error_text TEXT,
      created_at TEXT NOT NULL,
      resumed_at TEXT,
      completed_at TEXT
    )
  `)
  database.prepare("UPDATE a2a_continuations SET status = 'waiting' WHERE status = 'resuming'").run()
  database.prepare("UPDATE requests SET status = 'orphaned' WHERE status IN ('pending', 'approved', 'resuming')").run()
  const insert = database.prepare(`
    INSERT INTO requests
      (id, session_id, call_id, tool_name, arguments_json, status, reason, created_at)
    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
  `)
  const decide = database.prepare(`
    UPDATE requests SET status = ?, outcome = ?, decided_at = ? WHERE id = ?
  `)
  const complete = database.prepare(`
    UPDATE requests SET status = ?, completed_at = ?, execution_error = ? WHERE id = ?
  `)
  const insertDeferred = database.prepare(`
    INSERT INTO requests
      (id, session_id, call_id, tool_name, arguments_json, status, outcome,
       reason, created_at, expires_at)
    VALUES (?, ?, ?, ?, ?, 'deferred', 'optimistic-default', ?, ?, ?)
  `)
  const expireDeferred = database.prepare(`
    UPDATE requests SET status = 'expired', completed_at = ?
     WHERE status = 'deferred' AND expires_at IS NOT NULL AND expires_at <= ?
  `)
  const listRecoverable = database.prepare(`
    SELECT id, session_id, call_id, tool_name, arguments_json, status, reason,
           created_at, decided_at, completed_at, execution_error,
           recovery_request_id, recovered_arguments_json, recovered_at, expires_at
      FROM requests
     WHERE status IN ('orphaned', 'failed', 'deferred')
       AND tool_name NOT LIKE 'netopyu_hitl_%'
       AND tool_name NOT LIKE 'netopyu_a2a_hitl_%'
     ORDER BY created_at ASC
     LIMIT ?
  `)
  const findRecoverable = database.prepare(`
    SELECT id, session_id, call_id, tool_name, arguments_json, status, reason,
           created_at, decided_at, completed_at, execution_error,
           recovery_request_id, recovered_arguments_json, recovered_at, expires_at
      FROM requests
     WHERE id = ? AND status IN ('orphaned', 'failed', 'deferred')
       AND tool_name NOT LIKE 'netopyu_hitl_%'
       AND tool_name NOT LIKE 'netopyu_a2a_hitl_%'
  `)
  const claimRecovery = database.prepare(`
    UPDATE requests
       SET status = 'resuming', recovery_request_id = ?,
           recovered_arguments_json = ?, recovered_at = ?
     WHERE id = ? AND status IN ('orphaned', 'failed', 'deferred')
  `)
  const finishRecovery = database.prepare(`
    UPDATE requests
       SET status = ?, completed_at = ?, execution_error = ?
     WHERE id = ? AND status = 'resuming' AND recovery_request_id = ?
  `)
  const insertBatchItem = database.prepare(`
    INSERT INTO batch_items
      (batch_request_id, item_index, tool_name, arguments_json, status)
    VALUES (?, ?, ?, ?, 'queued')
  `)
  const startBatchItem = database.prepare(`
    UPDATE batch_items SET status = 'running', started_at = ?
     WHERE batch_request_id = ? AND item_index = ? AND status = 'queued'
  `)
  const finishBatchItem = database.prepare(`
    UPDATE batch_items
       SET status = ?, result_text = ?, error_text = ?, completed_at = ?
     WHERE batch_request_id = ? AND item_index = ?
  `)
  const skipBatchItems = database.prepare(`
    UPDATE batch_items SET status = 'skipped', error_text = ?, completed_at = ?
     WHERE batch_request_id = ? AND item_index > ? AND status = 'queued'
  `)
  const insertGrant = database.prepare(`
    INSERT INTO tool_grants
      (grant_id, request_id, token_hash, tool_name, status, issued_at)
    VALUES (?, ?, ?, ?, 'issued', ?)
  `)
  const consumeGrant = database.prepare(`
    UPDATE tool_grants SET status = 'consumed', consumed_at = ?
     WHERE token_hash = ? AND tool_name = ? AND status = 'issued'
  `)
  const revokeGrant = database.prepare(`
    UPDATE tool_grants
       SET status = 'revoked', revoked_at = ?, revoke_reason = ?
     WHERE token_hash = ? AND status = 'issued'
  `)
  const insertTrajectory = database.prepare(`
    INSERT INTO trajectory_events(session_id, event_type, payload_json, created_at)
    VALUES (?, ?, ?, ?)
  `)
  const recentTrajectory = database.prepare(`
    SELECT session_id, event_type, payload_json, created_at
      FROM trajectory_events ORDER BY id DESC LIMIT ?
  `)
  const insertA2aContinuation = database.prepare(`
    INSERT INTO a2a_continuations
      (id, session_id, peer_agent, interrupt_id, request_json, status, created_at)
    VALUES (?, ?, ?, ?, ?, 'waiting', ?)
  `)
  const listA2aContinuations = database.prepare(`
    SELECT id, session_id, peer_agent, interrupt_id, status, error_text,
           created_at, resumed_at, completed_at
      FROM a2a_continuations
     WHERE status IN ('waiting', 'failed')
     ORDER BY created_at ASC LIMIT ?
  `)
  const getA2aContinuation = database.prepare(`
    SELECT id, session_id, peer_agent, interrupt_id, request_json, status,
           error_text, created_at, resumed_at, completed_at
      FROM a2a_continuations WHERE id = ? AND status IN ('waiting', 'failed')
  `)
  const claimA2aContinuation = database.prepare(`
    UPDATE a2a_continuations SET status = 'resuming', resumed_at = ?
     WHERE id = ? AND status IN ('waiting', 'failed')
  `)
  const finishA2aContinuation = database.prepare(`
    UPDATE a2a_continuations
       SET status = ?, result_text = ?, error_text = ?, completed_at = ?
     WHERE id = ? AND status = 'resuming'
  `)
  function expire() {
    const now = new Date().toISOString()
    expireDeferred.run(now, now)
  }
  return {
    begin(execution, reason) {
      const id = randomUUID()
      insert.run(
        id,
        String(execution.agent?.session?.id ?? 'unknown'),
        execution.callId === undefined ? null : String(execution.callId),
        execution.name,
        JSON.stringify(execution.arguments),
        reason,
        new Date().toISOString(),
      )
      return id
    },
    decided(id, outcome) {
      decide.run(outcome === 'allowed-once' ? 'approved' : 'denied', outcome, new Date().toISOString(), id)
    },
    completed(id, isError) {
      complete.run(isError ? 'failed' : 'completed', new Date().toISOString(), isError ? 1 : 0, id)
    },
    beginDeferred(execution, toolName, args, reason, slaSeconds) {
      const id = randomUUID()
      const now = new Date()
      const expiresAt = new Date(now.getTime() + slaSeconds * 1000)
      insertDeferred.run(
        id,
        String(execution.agent?.session?.id ?? 'unknown'),
        execution.callId === undefined ? null : String(execution.callId),
        toolName,
        JSON.stringify(args),
        reason,
        now.toISOString(),
        expiresAt.toISOString(),
      )
      return { id, expiresAt: expiresAt.toISOString() }
    },
    listRecoverable(limit = 50) {
      expire()
      return listRecoverable.all(Math.max(1, Math.min(Number(limit) || 50, 200))).map(row => ({
        ...row,
        arguments: JSON.parse(row.arguments_json),
        recovered_arguments: row.recovered_arguments_json === null ? null : JSON.parse(row.recovered_arguments_json),
        arguments_json: undefined,
        recovered_arguments_json: undefined,
      }))
    },
    recoverable(id) {
      expire()
      const row = findRecoverable.get(id)
      if (row === undefined) return undefined
      return { ...row, arguments: JSON.parse(row.arguments_json) }
    },
    claimRecovery(id, recoveryRequestId, args) {
      const result = claimRecovery.run(recoveryRequestId, JSON.stringify(args), new Date().toISOString(), id)
      return result.changes === 1
    },
    finishRecovery(id, recoveryRequestId, isError) {
      finishRecovery.run(
        isError ? 'failed' : 'recovered',
        new Date().toISOString(),
        isError ? 1 : 0,
        id,
        recoveryRequestId,
      )
    },
    initializeBatch(requestId, operations) {
      database.exec('BEGIN IMMEDIATE')
      try {
        operations.forEach((operation, index) => {
          insertBatchItem.run(requestId, index, operation.tool_name, JSON.stringify(operation.arguments))
        })
        database.exec('COMMIT')
      } catch (error) {
        database.exec('ROLLBACK')
        throw error
      }
    },
    startBatchItem(requestId, index) {
      return startBatchItem.run(new Date().toISOString(), requestId, index).changes === 1
    },
    finishBatchItem(requestId, index, result, error) {
      finishBatchItem.run(
        error === undefined ? 'completed' : 'failed',
        result ?? null,
        error ?? null,
        new Date().toISOString(),
        requestId,
        index,
      )
    },
    skipBatchRemainder(requestId, index, reason) {
      skipBatchItems.run(reason, new Date().toISOString(), requestId, index)
    },
    issueGrant(requestId, tokenHash, toolName) {
      const grantId = randomUUID()
      insertGrant.run(grantId, requestId, tokenHash, toolName, new Date().toISOString())
      return grantId
    },
    consumeGrant(tokenHash, toolName) {
      return consumeGrant.run(new Date().toISOString(), tokenHash, toolName).changes === 1
    },
    revokeGrant(tokenHash, reason) {
      return revokeGrant.run(new Date().toISOString(), reason, tokenHash).changes === 1
    },
    recordTrajectory(sessionId, eventType, payload = {}) {
      insertTrajectory.run(String(sessionId), eventType, JSON.stringify(payload), new Date().toISOString())
    },
    recentTrajectory(limit = 100) {
      return recentTrajectory.all(Math.max(1, Math.min(Number(limit) || 100, 500))).map(row => ({
        session_id: row.session_id,
        event_type: row.event_type,
        payload: JSON.parse(row.payload_json),
        created_at: row.created_at,
      }))
    },
    recordA2aContinuation(sessionId, peerAgent, interruptId, request) {
      const id = randomUUID()
      insertA2aContinuation.run(
        id, String(sessionId), String(peerAgent), String(interruptId),
        JSON.stringify(request), new Date().toISOString(),
      )
      return id
    },
    listA2aContinuations(limit = 100) {
      return listA2aContinuations.all(Math.max(1, Math.min(Number(limit) || 100, 200))).map(row => ({ ...row }))
    },
    a2aContinuation(id) {
      const row = getA2aContinuation.get(String(id))
      return row === undefined ? undefined : { ...row, request: JSON.parse(row.request_json) }
    },
    claimA2aContinuation(id) {
      return claimA2aContinuation.run(new Date().toISOString(), String(id)).changes === 1
    },
    finishA2aContinuation(id, status, resultText, errorText) {
      return finishA2aContinuation.run(
        status, resultText ?? null, errorText ?? null, new Date().toISOString(), String(id),
      ).changes === 1
    },
    close() { database.close() },
  }
}

function tokenHash(token) {
  return createHash('sha256').update(String(token)).digest('hex')
}

export class NetOpYuToolGuard {
  #store

  constructor(hitlStore) {
    this.#store = hitlStore
  }

  issue(token, requestId, toolName) {
    return this.#store.issueGrant(requestId, tokenHash(token), toolName)
  }

  consume(token, toolName) {
    return this.#store.consumeGrant(tokenHash(token), toolName)
  }

  revoke(token, reason = 'execution finished without consuming grant') {
    return this.#store.revokeGrant(tokenHash(token), reason)
  }
}

