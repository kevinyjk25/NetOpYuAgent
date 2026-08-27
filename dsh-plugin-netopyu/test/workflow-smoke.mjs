import assert from 'node:assert/strict'
import { existsSync, unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { apply } from '../src/index.js'

// Keep this offline smoke isolated from any desktop Worker already running.
delete process.env.NETOPYU_DSH_WORKER_SOCKET

const suffix = String(process.pid)
const hitlPath = join(tmpdir(), `netopyu-workflow-hitl-${suffix}.sqlite`)
const runtimePath = join(tmpdir(), `netopyu-workflow-runtime-${suffix}.sqlite`)
const resultsPath = join(tmpdir(), `netopyu-workflow-results-${suffix}.sqlite`)
process.env.NETOPYU_DSH_BACKEND = 'mock'
process.env.NETOPYU_DSH_NETWORK_RUNTIME_STORE = runtimePath
process.env.NETOPYU_DSH_TOOL_RESULT_STORE = resultsPath

const definitions = []
const listeners = new Map()
const disposers = []
let approvalCalls = 0
const context = {
  tools: { register(definition) { definitions.push(definition) } },
  approval: { async request() { approvalCalls += 1; return 'allowed-once' } },
  subagents: { registerProvider() {}, async start() { throw new Error('not used') } },
  skills: { register() {} },
  provide() {},
  effect(factory) { disposers.push(factory()) },
  on(event, listener) { listeners.set(event, listener) },
}
await apply(context, { enableDestructive: true, hitlStorePath: hitlPath, peerUrls: [] })

const getAccess = definitions.find(tool => tool.name === 'get_user_access')
const checkNac = definitions.find(tool => tool.name === 'check_nac_policy')
const grantAccess = definitions.find(tool => tool.name === 'grant_user_access')
assert.ok(getAccess && checkNac && grantAccess)

const agent = { session: { id: 'workflow-session' } }
const signal = new AbortController().signal
const skillExecution = {
  token: 'skill-token', callId: 'skill-call', name: 'skill',
  arguments: { name: 'lan-user-access-diagnose' }, agent, signal,
}
assert.equal(
  (await listeners.get('tools/pre-execute')(skillExecution, async () => ({ kind: 'allow' }))).kind,
  'allow',
)

const skipped = {
  token: 'skipped-token', callId: 'skipped-call', name: grantAccess.name,
  arguments: { user_id: 'erin', reason: 'must not skip diagnosis' }, agent, signal,
}
const skippedDecision = await listeners.get('tools/pre-execute')(skipped, async () => ({ kind: 'allow' }))
assert.equal(skippedDecision.kind, 'deny')
assert.match(skippedDecision.reason, /workflow prerequisites/)
assert.equal(approvalCalls, 0, 'invalid workflow order must be rejected before approval')

async function runRead(definition, token, args) {
  const execution = {
    token, callId: `${token}-call`, name: definition.name, arguments: args, agent, signal,
  }
  assert.equal(
    (await listeners.get('tools/pre-execute')(execution, async () => ({ kind: 'allow' }))).kind,
    'allow',
  )
  const value = await definition.execute(args, execution)
  const post = await listeners.get('tools/post-execute')(
    execution, { isError: false, value, content: [] }, async () => ({ kind: 'accept' }),
  )
  assert.equal(post.kind, 'accept')
  listeners.get('tools/result')(execution, { isError: false, value, content: [] })
  return value
}

assert.match(await runRead(getAccess, 'get-access-token', { user_id: 'erin' }), /BLOCKED/)
assert.match(await runRead(checkNac, 'check-nac-token', { user_id: 'erin' }), /DENY/)

const approved = {
  token: 'approved-token', callId: 'approved-call', name: grantAccess.name,
  arguments: { user_id: 'erin', reason: 'completed deterministic diagnosis' }, agent, signal,
}
const approvedDecision = await listeners.get('tools/pre-execute')(approved, async () => ({ kind: 'allow' }))
assert.equal(approvedDecision.kind, 'allow')
assert.equal(approvalCalls, 1)
const result = await grantAccess.execute(approved.arguments, approved)
assert.match(result, /Granted network access/)
await listeners.get('tools/post-execute')(
  approved, { isError: false, value: result, content: [] }, async () => ({ kind: 'accept' }),
)
listeners.get('tools/result')(approved, { isError: false, value: result, content: [] })

const runtime = new DatabaseSync(runtimePath, { readOnly: true })
const planRow = runtime.prepare("SELECT plan_json, state FROM plans WHERE state='verified_success'").get()
const workflowRow = runtime.prepare("SELECT status FROM workflow_runs WHERE session_id='workflow-session'").get()
runtime.close()
const plan = JSON.parse(planRow.plan_json)
assert.equal(plan.workflow_run_id === null, false)
assert.match(plan.workflow_template_hash, /^sha256:/)
assert.equal(workflowRow.status, 'completed')

const hitl = new DatabaseSync(hitlPath, { readOnly: true })
const grant = hitl.prepare("SELECT plan_hash, status FROM tool_grants WHERE tool_name='grant_user_access'").get()
hitl.close()
assert.equal(grant.plan_hash, plan.plan_hash)
assert.equal(grant.status, 'consumed')

for (const dispose of disposers.reverse()) dispose()
for (const path of [hitlPath, runtimePath, resultsPath]) {
  for (const ending of ['', '-shm', '-wal']) {
    if (existsSync(`${path}${ending}`)) unlinkSync(`${path}${ending}`)
  }
}
console.log(JSON.stringify({ workflow: 'completed', plan: plan.plan_id, approvalCalls }))
