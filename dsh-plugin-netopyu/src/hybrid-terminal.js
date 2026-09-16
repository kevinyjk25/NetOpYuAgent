// Trusted host-result projection. Text does not acquire semantic approval.
export const TERMINAL_PROFILE = 'netopyu.io/host-terminal-delivery/v1'
const ACTIONS = new Set(['draft', 'deliver', 'inspect'])
const STATES = new Set(['rejected', 'needs_revision', 'candidate_unverified'])
const DIGEST = /^sha256:[a-f0-9]{64}$/

export function terminalDelivery(action, args, value) {
  if (!ACTIONS.has(action) || !/^[a-f0-9]{32}$/.test(args?.session_id ?? '')) return undefined
  const report = value?.route === 'already_drafted_or_unknown' ? value.existing : value
  const host = report?.hostResult
  if (!report || report.revisionAllowed === true || report.retryAllowed !== false || report.evidenceFrozen !== true || !STATES.has(host?.state)
      || host.semanticApproval !== false || host.taskSuccess !== null || report.task?.success !== null
      || report.writeAuthority !== false || typeof host.message !== 'string' || !DIGEST.test(report.reportDigest ?? '')
      || !DIGEST.test(report.evidenceDigest ?? '') || report.sessionId !== args.session_id) return undefined
  const delivery = report.task.delivery
  if (host.state === 'rejected') {
    if (delivery !== null || host.deliveryDigest !== null || report.task.status !== 'not_completed') return undefined
  } else if (!delivery || typeof delivery.rendered !== 'string'
      || !DIGEST.test(delivery.reportDigest ?? '') || host.deliveryDigest !== delivery.reportDigest
      || delivery.semanticApproval !== false || delivery.taskSuccess !== null) return undefined
  return {
    profile: TERMINAL_PROFILE, sessionId: args.session_id, hostReportDigest: report.reportDigest,
    evidenceDigest: report.evidenceDigest, deliveryDigest: host.deliveryDigest, state: host.state,
    semanticApproval: false, taskSuccess: null,
    text: `宿主交付 / Host delivery — ${host.state}\n\n${host.message}\n\n`
      + (delivery ? delivery.rendered : '没有获准交付文本 / No admitted delivery text.'),
  }
}

export function parseTerminal(action, args, value) {
  try { return terminalDelivery(action, args, typeof value === 'string' ? JSON.parse(value) : value) }
  catch { return undefined }
}

export function createTerminalGate() {
  const closed = new WeakMap()
  function turnOf(execution) {
    const events = execution.agent?.session?.events
    if (!Array.isArray(events)) throw new Error('host terminal mode requires a scoped agent session')
    const turn = events.findLast(event => event.type === 'turn/start')
    if (!turn) throw new Error('host terminal mode requires an active turn')
    return turn.seq
  }
  return {
    before(execution, action) {
      const turn = turnOf(execution)
      // DSH drains same-step tool batches even after concludeTurn. Deny new
      // operations in that scope; preserve read-only inspection and new turns.
      if (closed.get(execution.agent) === turn && action !== 'inspect') {
        throw new Error('host delivery closed this turn; no further hybrid operations; inspect or use a new user turn')
      }
    },
    close(execution) { closed.set(execution.agent, turnOf(execution)) },
  }
}

// Called by the tool BODY, not a model-facing parameter or a global stop hook.
export async function executeHybrid(bridgeCall, action, args, execution, terminalEnabled, gate) {
  if (terminalEnabled && typeof execution.concludeTurn !== 'function') {
    throw new Error('DSH concludeTurn support required before invoking the host; no silent fallback')
  }
  if (terminalEnabled) gate?.before(execution, action)
  const value = await bridgeCall()
  if (terminalEnabled && !execution.signal?.aborted && parseTerminal(action, args, value)) {
    gate?.close(execution)
    execution.concludeTurn()
  }
  return JSON.stringify(value, null, 2)
}

// Headless output is taken only from a successful final-step HOST tool receipt.
// A later step/turn (e.g. fresh user steering) invalidates an earlier terminal.
export function summarizeHybridTurn(events, firstSeq = 0) {
  let turn, step, reason, text = '', terminal
  const calls = new Map()
  for (const event of events) {
    if (event.seq < firstSeq) continue
    const data = event.data
    if (event.type === 'turn/start') {
      turn = data.turn; step = undefined; terminal = undefined; text = ''; reason = undefined; calls.clear()
    } else if (turn === undefined) continue
    else if (event.type === 'step/start' && data.turn === turn) {
      step = data.step; terminal = undefined
    } else if (event.type === 'tool/call' && data.turn === turn) {
      calls.set(data.callId, data)
    } else if (event.type === 'tool/result' && data.turn === turn && data.step === step) {
      const call = calls.get(data.message?.source?.callId)
      const block = data.message?.content?.[0]
      if (!call || call.step !== step || block?.type !== 'tool-result' || block.isError === true
          || data.error || data.meta?.profile !== TERMINAL_PROFILE) continue
      let args
      try { args = typeof call.arguments === 'string' ? JSON.parse(call.arguments) : call.arguments }
      catch { continue }
      const parts = block.content
      const action = call.name?.replace(/^netopyu_hybrid_/, '')
      const projected = parts?.length === 1 && parts[0].type === 'text'
        ? parseTerminal(action, args, parts[0].text) : undefined
      if (projected && data.meta.hostReportDigest === projected.hostReportDigest
          && data.meta.sessionId === projected.sessionId) terminal = projected
    } else if (event.type === 'assistant/message' && data.turn === turn) {
      const joined = data.message.content.filter(block => block.type === 'text').map(block => block.text).join('')
      if (joined) text = joined
    } else if (event.type === 'turn/end' && data.turn === turn) reason = data.reason
  }
  if (reason?.kind !== 'completed') terminal = undefined
  return { text: terminal?.text ?? text, reason, terminal,
    source: terminal ? 'host_terminal_receipt' : 'native_assistant_prose' }
}
