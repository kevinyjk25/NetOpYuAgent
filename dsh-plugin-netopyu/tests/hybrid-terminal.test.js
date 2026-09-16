import test from 'node:test'
import assert from 'node:assert/strict'
import { createTerminalGate, executeHybrid, parseTerminal, summarizeHybridTurn, TERMINAL_PROFILE } from '../src/hybrid-terminal.js'

test('a host-offered bounded artifact correction is not a terminal delivery', async () => {
  const value = { ...report('needs_revision'), revisionAllowed: true }
  const args = { session_id: value.sessionId }
  assert.equal(parseTerminal('draft', args, value), undefined)
  let closed = false
  await executeHybrid(async () => value, 'draft', args,
    { concludeTurn() { closed = true } }, true)
  assert.equal(closed, false)
  assert.equal(parseTerminal('deliver', args, { ...value, revisionAllowed: false }).state, 'needs_revision')
})
import { governedHybridDefinitions } from '../src/index.js'

const sid = 'a'.repeat(32)
const digest = char => `sha256:${char.repeat(64)}`
const args = { session_id: sid }
test('isolated execution Agent has five concrete tools and no AST submit surface', () => {
  const concrete = { type: 'object', properties: { arguments: { type: 'object' } } }
  const defs = governedHybridDefinitions({}, { compilerMode: 'isolated', inputSchema: {}, readRequestSchema: concrete, canCloseIncomplete: true })
  assert.equal(defs.length, 5)
  assert.ok(!defs.some(d => d.name === 'netopyu_hybrid_submit'))
  assert.equal(defs.find(d => d.name === 'netopyu_hybrid_read').parameters, concrete)
  assert.ok(defs.every(d => !JSON.stringify(d.parameters).includes('plan')))
  assert.match(defs.find(d => d.name === 'netopyu_hybrid_prepare').description, /isolated compiler/)
})
function report(state = 'candidate_unverified') {
  const rejected = state === 'rejected'
  return { sessionId: sid, reportDigest: digest('b'), evidenceDigest: digest('c'),
    retryAllowed: false, evidenceFrozen: true, writeAuthority: false,
    hostResult: { state, message: 'Host outcome, NOT semantic approval', semanticApproval: false,
      taskSuccess: null, deliveryDigest: rejected ? null : digest('d') },
    task: { success: null, status: rejected ? 'not_completed' : 'candidate_unverified',
      delivery: rejected ? null : { reportDigest: digest('d'), rendered: 'Literal candidate: 未验证', semanticApproval: false, taskSuccess: null } } }
}
function execution() {
  return { agent: { session: { events: [{ seq: 0, type: 'turn/start', data: { turn: 1 } }] } },
    signal: new AbortController().signal, conclusions: 0, concludeTurn() { this.conclusions++ } }
}
function events(value = report()) {
  return [
    { seq: 0, type: 'turn/start', data: { turn: 1 } },
    { seq: 1, type: 'step/start', data: { turn: 1, step: 1 } },
    { seq: 2, type: 'assistant/message', data: { turn: 1, step: 1, message: { content: [{ type: 'text', text: 'Untrusted completion claim' }] } } },
    { seq: 3, type: 'tool/call', data: { turn: 1, step: 1, callId: 'call1', name: 'netopyu_hybrid_deliver', arguments: JSON.stringify(args) } },
    { seq: 4, type: 'tool/result', data: { turn: 1, step: 1,
      meta: { profile: TERMINAL_PROFILE, sessionId: sid, hostReportDigest: value.reportDigest },
      message: { source: { callId: 'call1' }, content: [{ type: 'tool-result', isError: false,
        content: [{ type: 'text', text: JSON.stringify(value) }] }] } } },
    { seq: 5, type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
  ]
}

for (const state of ['candidate_unverified', 'needs_revision', 'rejected']) {
  test(`host ${state} concludes transport without approving semantics`, async () => {
    const exec = execution()
    const result = await executeHybrid(async () => report(state), 'deliver', args, exec, true, createTerminalGate())
    assert.equal(exec.conclusions, 1)
    const terminal = parseTerminal('deliver', args, result)
    assert.equal(terminal.state, state)
    assert.equal(terminal.semanticApproval, false)
    assert.equal(terminal.taskSuccess, null)
    assert.equal(summarizeHybridTurn(events(report(state))).text, terminal.text)
  })
}

for (const mutation of [r => { r.sessionId = 'f'.repeat(32) }, r => { r.hostResult.semanticApproval = true },
  r => { r.task.success = true }, r => { r.hostResult.deliveryDigest = digest('e') },
  r => { r.evidenceFrozen = false }, r => { r.retryAllowed = true }, r => { delete r.hostResult },
  r => { r.task.delivery.semanticApproval = true }, r => { r.hostResult.state = 'completed' }]) {
  test(`malformed or nonterminal result does not conclude: ${mutation}`, async () => {
    const value = report(); mutation(value)
    const exec = execution()
    await executeHybrid(async () => value, 'draft', args, exec, true, createTerminalGate())
    assert.equal(exec.conclusions, 0)
  })
}

test('unknown outcome, thrown bridge, cancellation and unavailable API never fabricate completion', async () => {
  const exec = execution()
  await executeHybrid(async () => ({ route: 'pending_or_unknown_no_retry' }), 'inspect', args, exec, true)
  await assert.rejects(executeHybrid(async () => { throw Error('bridge failed') }, 'draft', args, exec, true))
  const cancelled = new AbortController(); cancelled.abort(); exec.signal = cancelled.signal
  await executeHybrid(async () => report(), 'draft', args, exec, true)
  assert.equal(exec.conclusions, 0)
  let invoked = false
  await assert.rejects(executeHybrid(async () => { invoked = true }, 'prepare', {}, {}, true), /concludeTurn/)
  assert.equal(invoked, false)
})

test('legacy mode preserves execution and never calls concludeTurn', async () => {
  const exec = execution()
  await executeHybrid(async () => report(), 'deliver', args, exec, false)
  assert.equal(exec.conclusions, 0)
})

test('same-batch later operations are denied, inspection/new turn/other agent survive', async () => {
  const gate = createTerminalGate(), exec = execution()
  await executeHybrid(async () => report(), 'draft', args, exec, true, gate)
  let invoked = 0
  for (const action of ['prepare', 'submit', 'read', 'draft', 'deliver']) {
    await assert.rejects(executeHybrid(async () => { invoked++ }, action, args, exec, true, gate), /closed this turn/)
  }
  assert.equal(invoked, 0)
  await executeHybrid(async () => report(), 'inspect', args, exec, true, gate)
  const other = execution()
  await executeHybrid(async () => { invoked++; return {} }, 'read', args, other, true, gate)
  exec.agent.session.events.push({ seq: 10, type: 'turn/start', data: { turn: 2 } })
  await executeHybrid(async () => { invoked++; return {} }, 'prepare', {}, exec, true, gate)
  assert.equal(invoked, 2)
})

for (const mutation of [e => { e[4].data.message.content[0].isError = true },
  e => { e[4].data.meta.hostReportDigest = digest('e') }, e => { e[4].data.message.source.callId = 'foreign' },
  e => { e[5].data.reason.kind = 'error' }, e => { e[3].data.arguments = '{}' },
  e => { e[4].data.meta = null }, e => { e[3].data.name = 'arbitrary_tool' }]) {
  test(`headless refuses unbound/blocked/error terminal: ${mutation}`, () => {
    const trace = events(); mutation(trace)
    assert.equal(summarizeHybridTurn(trace).terminal, undefined)
  })
}

test('later steering step and next turn cannot inherit the old terminal', () => {
  const trace = events().slice(0, -1)
  trace.push({ seq: 5, type: 'step/start', data: { turn: 1, step: 2 } },
    { seq: 6, type: 'assistant/message', data: { turn: 1, step: 2, message: { content: [{ type: 'text', text: 'Answer to steering' }] } } },
    { seq: 7, type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } })
  assert.equal(summarizeHybridTurn(trace).text, 'Answer to steering')
  assert.equal(summarizeHybridTurn(trace).terminal, undefined)
  trace.push({ seq: 8, type: 'turn/start', data: { turn: 2 } })
  assert.equal(summarizeHybridTurn(trace).text, '')
  assert.equal(summarizeHybridTurn(events(), 5).terminal, undefined)
})

test('UI card shows host result and all tools remain exclusive and unchanged', () => {
  const schema = { inputSchema: {}, planSchema: {}, deliverySchema: {} }
  const defs = governedHybridDefinitions({}, schema, { terminalDelivery: true })
  assert.equal(defs.length, 6)
  assert.ok(defs.every(d => d.isConcurrencySafe === undefined))
  const value = JSON.stringify(report())
  const finish = defs.find(d => d.name === 'netopyu_hybrid_deliver')
  const card = finish.presentResult(args, { isError: false, content: [{ type: 'text', text: value }] })
  assert.match(card.title, /candidate_unverified/)
  assert.match(card.content[0].text, /Literal candidate: 未验证/)
  assert.equal(finish.presentResult(args, { isError: true, content: [{ type: 'text', text: value }] }), undefined)
  assert.equal(finish.output.presentationMeta(args, value).profile, TERMINAL_PROFILE)
})

test('v4 read exposes the host concrete schema; incomplete close is optional and cannot mean approval', () => {
  const concrete = { type: 'object', properties: { session_id: { type: 'string' }, tool: { const: 'read_export' },
    arguments: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'], additionalProperties: false } },
    required: ['session_id', 'tool', 'arguments'], additionalProperties: false }
  const schema = { inputSchema: {}, planSchema: { type: 'object' }, deliverySchema: { type: 'null' },
    readRequestSchema: concrete, canCloseIncomplete: true }
  const defs = governedHybridDefinitions({}, schema, { terminalDelivery: true })
  assert.equal(defs.length, 6)
  const read = defs.find(d => d.name === 'netopyu_hybrid_read')
  assert.deepEqual(read.parameters, concrete)
  assert.match(read.description, /concrete values/)
  const draft = defs.find(d => d.name === 'netopyu_hybrid_draft')
  assert.deepEqual(draft.parameters.required, ['session_id'])
  assert.equal(draft.parameters.properties.close_incomplete.const, true)
  const stopped = { ...report('rejected'), route: 'closed_incomplete', modelCalls: [] }
  const projected = parseTerminal('draft', args, stopped)
  assert.equal(projected.state, 'rejected')
  assert.equal(projected.taskSuccess, null)
  assert.match(projected.text, /No admitted delivery text/)
  const legacy = governedHybridDefinitions({}, { inputSchema: {}, planSchema: {}, deliverySchema: {} })
  assert.equal(legacy.find(d => d.name === 'netopyu_hybrid_draft').parameters.properties.close_incomplete, undefined)
})
