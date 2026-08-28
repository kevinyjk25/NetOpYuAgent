import assert from 'node:assert/strict'
import { existsSync, unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { createServer } from 'node:http'
import { apply } from '../src/index.js'

const storePath = join(tmpdir(), `netopyu-hitl-${process.pid}.sqlite`)
const runtimeStorePath = join(tmpdir(), `netopyu-network-runtime-${process.pid}.sqlite`)
const toolResultStorePath = join(tmpdir(), `netopyu-tool-results-${process.pid}.sqlite`)
// The smoke suite owns isolated temporary stores. Never route it through a
// separately running desktop Worker with a different process environment.
delete process.env.NETOPYU_DSH_WORKER_SOCKET
process.env.NETOPYU_DSH_NETWORK_RUNTIME_STORE = runtimeStorePath
process.env.NETOPYU_DSH_TOOL_RESULT_STORE = toolResultStorePath
const definitions = []
const listeners = new Map()
const services = new Map()
const providers = new Map()
const registeredSkills = new Map()
const disposers = []
let approvalOutcome = 'allowed-once'
let lastApprovalReason = ''
let peerBaseUrl
const peerServer = createServer(async (request, response) => {
  if (request.method === 'GET' && request.url === '/.well-known/agent-card.json') {
    const body = JSON.stringify({
      agent_id: 'dc-agent', name: 'DC Agent', url: peerBaseUrl,
      skills: [{ id: 'app_access', name: 'Application access', tags: ['rbac', 'dc'] }],
    })
    response.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) })
    response.end(body)
    return
  }
  if (request.method === 'POST' && request.url === '/stream') {
    let rawBody = ''
    for await (const chunk of request) rawBody += chunk
    const requestBody = JSON.parse(rawBody)
    const prompt = requestBody.params.message.parts[0].text
    const metadata = requestBody.params.metadata
    const event = JSON.stringify(
      metadata.resume_interrupt_id && metadata.operator_decision === 'reject'
        ? { kind: 'message', message: { parts: [{ kind: 'text', text: JSON.stringify({ status: 'rejected' }) }] } }
        : prompt === 'require hitl' && metadata.operator_decision !== 'approve'
          ? { kind: 'taskStatusUpdate', status: { state: 'input-required', message: {
            interrupt_id: 'peer-interrupt-1',
            approval: {
              kind: 'network-l0-plan', profile: 'dc', plan_id: 'dc-plan-1',
              plan_hash: `sha256:${'a'.repeat(64)}`, tool_name: 'dc_grant_app_access',
              arguments: { user_id: 'erin', app_id: 'crm', role: 'sales-rep' },
              risk_level: 'high', l0_skill_id: 'network.dc.app-access.grant',
              l0_skill_version: '1.0.0', l0_contract_hash: 'sha256:dc-l0',
              intent_hash: 'sha256:dc-intent', verification_contract: 'dc-access-granted',
              rollback_contract: 'inverse-tool-v1', workflow_run_id: 'dc-workflow-1',
              workflow_template_hash: 'sha256:dc-workflow', expires_at: '2099-01-01T00:00:00Z',
            },
          } } }
          : { kind: 'message', message: { parts: [{ kind: 'text', text: metadata.resume_interrupt_id ? 'DC approved result' : 'DC delegated result' }] } },
    )
    response.writeHead(200, { 'content-type': 'text/event-stream' })
    response.end(`data: ${event}\n\ndata: [DONE]\n\n`)
    return
  }
  response.writeHead(404)
  response.end()
})
await new Promise(resolve => peerServer.listen(0, '127.0.0.1', resolve))
peerBaseUrl = `http://127.0.0.1:${peerServer.address().port}`
const context = {
  tools: { register(definition) { definitions.push(definition) } },
  approval: { async request({ reason }) { lastApprovalReason = reason; return approvalOutcome } },
  subagents: {
    registerProvider(provider) { providers.set(provider.name, provider) },
    async start(name, request) { return providers.get(name).start(request) },
  },
  skills: { register(skill) { registeredSkills.set(skill.name, skill); return () => registeredSkills.delete(skill.name) } },
  provide(name, value) { services.set(name, value) },
  effect(factory) { disposers.push(factory()) },
  on(event, listener) { listeners.set(event, listener) },
}

await apply(context, { enableDestructive: true, hitlStorePath: storePath, peerUrls: [peerBaseUrl] })
assert.ok(services.get('netopyuToolGuard'))
assert.ok(services.get('netopyuMemory'))
assert.ok(services.get('netopyuCapabilities'))
assert.ok(services.get('netopyuA2A'))
assert.ok(providers.get('netopyu-a2a'))
assert.ok(registeredSkills.has('read-stored-result'))
assert.ok(registeredSkills.has('app-access-troubleshoot'))
const initialToolCount = definitions.length
assert.ok(definitions.find(tool => tool.name === 'netopyu_hitl_list'))
assert.ok(definitions.find(tool => tool.name === 'netopyu_hitl_resume'))
assert.ok(definitions.find(tool => tool.name === 'netopyu_hitl_async_submit'))
assert.ok(definitions.find(tool => tool.name === 'netopyu_hitl_batch'))
const trajectoryRecent = definitions.find(tool => tool.name === 'netopyu_trajectory_recent')
assert.ok(trajectoryRecent)
const memoryRecall = definitions.find(tool => tool.name === 'netopyu_memory_recall')
const capabilitySearch = definitions.find(tool => tool.name === 'netopyu_capability_search')
assert.ok(memoryRecall)
assert.ok(capabilitySearch)
const peerList = definitions.find(tool => tool.name === 'netopyu_peer_list')
const delegate = definitions.find(tool => tool.name === 'netopyu_delegate')
const a2aHitlList = definitions.find(tool => tool.name === 'netopyu_a2a_hitl_list')
const a2aHitlResume = definitions.find(tool => tool.name === 'netopyu_a2a_hitl_resume')
assert.ok(peerList)
assert.ok(delegate)
assert.deepEqual(delegate.parameters.anyOf, [
  { required: ['target'] },
  { required: ['capability'] },
])
assert.ok(a2aHitlList)
assert.ok(a2aHitlResume)
const restart = definitions.find(tool => tool.name === 'restart_service')
assert.ok(restart, 'destructive tool should be registered only when explicitly enabled')

const agent = { session: { id: 'session-hitl-smoke' } }
const readExecution = { agent, signal: new AbortController().signal }
const emptyMemory = JSON.parse(await memoryRecall.execute({ query: 'router history' }, readExecution))
assert.equal(typeof emptyMemory.available, 'boolean')
await assert.rejects(
  memoryRecall.execute({ query: 'router history' }, { signal: readExecution.signal }),
  /live DSH session scope/,
)
const capabilities = JSON.parse(await capabilitySearch.execute(
  { query: 'restart production service', top_k: 5, kinds: ['tool'] }, readExecution,
))
assert.ok(capabilities.matches.some(match => match.id === 'restart_service'))
const peers = JSON.parse(await peerList.execute({}, readExecution))
assert.equal(peers.peers[0].agent_id, 'dc-agent')
assert.equal(
  await delegate.execute({ description: 'test', prompt: 'test', capability: 'rbac' }, readExecution),
  'DC delegated result',
)
const remotePending = JSON.parse(await delegate.execute(
  { description: 'remote hitl', prompt: 'require hitl', target: 'dc-agent' }, readExecution,
))
assert.equal(remotePending.status, 'input-required')
assert.ok(remotePending.continuation_id)
const waitingA2a = JSON.parse(await a2aHitlList.execute({}))
assert.equal(waitingA2a.length, 1)
assert.equal(waitingA2a[0].peer_agent, 'dc-agent')
assert.equal(waitingA2a[0].interrupt_id, 'peer-interrupt-1')
const a2aResumeExecution = {
  token: 'token-a2a-resume', callId: 'call-a2a-resume', name: a2aHitlResume.name,
  arguments: { continuation_id: waitingA2a[0].id, decision: 'approve' }, agent,
  signal: new AbortController().signal,
}
const a2aResumeDecision = await listeners.get('tools/pre-execute')(a2aResumeExecution, async () => ({ kind: 'allow' }))
assert.equal(a2aResumeDecision.kind, 'allow')
assert.match(lastApprovalReason, /Remote DC Network L0 plan/)
assert.match(lastApprovalReason, /dc-plan-1/)
assert.match(lastApprovalReason, new RegExp(`sha256:${'a'.repeat(64)}`))
assert.equal(await a2aHitlResume.execute(a2aResumeExecution.arguments, a2aResumeExecution), 'DC approved result')
listeners.get('tools/result')(a2aResumeExecution, { isError: false })
assert.deepEqual(JSON.parse(await a2aHitlList.execute({})), [])
const execution = {
  token: 'token-approved', callId: 'call-approved', name: restart.name,
  arguments: { service: 'crm', environment: 'staging' }, agent,
  signal: new AbortController().signal,
}
const decision = await listeners.get('tools/pre-execute')(execution, async () => ({ kind: 'allow' }))
assert.equal(decision.kind, 'allow')
const result = await restart.execute(execution.arguments, execution)
const terminal = JSON.parse(result)
assert.equal(terminal.contract, 'netopyu.effect-runtime-terminal@1.0.0')
assert.equal(terminal.terminal, true)
assert.equal(terminal.state, 'verified_success')
assert.doesNotMatch(result, /\"state\"\s*:\s*\"applied\"/i)
await assert.rejects(restart.execute(execution.arguments, execution), /durable-HITL grant/)
listeners.get('tools/result')(execution, { isError: false })
assert.ok(JSON.parse(await trajectoryRecent.execute({ limit: 10 })).some(item => item.event_type === 'tool:result'))

// DSH execution tokens are scoped to a call lifecycle and may be reused in a
// later call. Historical consumed grants must not make that fresh plan fail.
const reusedTokenExecution = {
  ...execution,
  callId: 'call-reused-token',
  arguments: { service: 'billing', environment: 'staging' },
}
const reusedTokenDecision = await listeners.get('tools/pre-execute')(
  reusedTokenExecution, async () => ({ kind: 'allow' }),
)
assert.equal(reusedTokenDecision.kind, 'allow')
const reusedTerminal = JSON.parse(
  await restart.execute(reusedTokenExecution.arguments, reusedTokenExecution),
)
assert.equal(reusedTerminal.contract, 'netopyu.effect-runtime-terminal@1.0.0')
assert.equal(reusedTerminal.state, 'verified_success')
listeners.get('tools/result')(reusedTokenExecution, { isError: false })

approvalOutcome = 'rejected'
const rejected = { ...execution, token: 'token-rejected', callId: 'call-rejected' }
const rejection = await listeners.get('tools/pre-execute')(rejected, async () => ({ kind: 'allow' }))
assert.equal(rejection.kind, 'deny')
await assert.rejects(restart.execute(rejected.arguments, rejected), /durable-HITL grant/)

approvalOutcome = 'allowed-once'
const concurrent = { ...execution, token: 'token-concurrent', callId: 'call-concurrent' }
const concurrentDecision = await listeners.get('tools/pre-execute')(concurrent, async () => ({ kind: 'allow' }))
assert.equal(concurrentDecision.kind, 'allow')
const concurrentResults = await Promise.allSettled([
  restart.execute(concurrent.arguments, concurrent),
  restart.execute(concurrent.arguments, concurrent),
])
assert.deepEqual(concurrentResults.map(item => item.status).sort(), ['fulfilled', 'rejected'])
listeners.get('tools/result')(concurrent, { isError: false })

const persistedPending = JSON.parse(await delegate.execute(
  { description: 'persist remote hitl', prompt: 'require hitl', target: 'dc-agent' }, readExecution,
))
assert.equal(persistedPending.status, 'input-required')
const persistedA2aId = JSON.parse(await a2aHitlList.execute({}))[0].id

for (const dispose of disposers.reverse()) dispose()
const database = new DatabaseSync(storePath, { readOnly: true })
const rows = database.prepare('SELECT status, outcome FROM requests ORDER BY created_at').all()
  .map(row => ({ ...row }))
database.close()
assert.deepEqual(rows, [
  { status: 'completed', outcome: 'allowed-once' },
  { status: 'completed', outcome: 'allowed-once' },
  { status: 'completed', outcome: 'allowed-once' },
  { status: 'denied', outcome: 'rejected' },
  { status: 'completed', outcome: 'allowed-once' },
])

const interrupted = new DatabaseSync(storePath)
interrupted.prepare("UPDATE requests SET status = 'pending' WHERE call_id = 'call-approved'").run()
interrupted.prepare("UPDATE tool_grants SET status = 'issued', consumed_at = NULL WHERE request_id = (SELECT id FROM requests WHERE call_id = 'call-approved')").run()
interrupted.close()
const restartDisposers = []
await apply({
  ...context,
  effect(factory) { restartDisposers.push(factory()) },
}, { enableDestructive: true, hitlStorePath: storePath })
const restartedDefinitions = definitions.slice(initialToolCount)
const listInterrupted = restartedDefinitions.find(tool => tool.name === 'netopyu_hitl_list')
const resumeInterrupted = restartedDefinitions.find(tool => tool.name === 'netopyu_hitl_resume')
const asyncSubmit = restartedDefinitions.find(tool => tool.name === 'netopyu_hitl_async_submit')
const batch = restartedDefinitions.find(tool => tool.name === 'netopyu_hitl_batch')
const restartedA2aList = restartedDefinitions.find(tool => tool.name === 'netopyu_a2a_hitl_list')
const restartedA2aResume = restartedDefinitions.find(tool => tool.name === 'netopyu_a2a_hitl_resume')
assert.equal(JSON.parse(await restartedA2aList.execute({}))[0].id, persistedA2aId)
const rejectA2aExecution = {
  token: 'token-a2a-reject', callId: 'call-a2a-reject', name: restartedA2aResume.name,
  arguments: { continuation_id: persistedA2aId, decision: 'reject' }, agent,
  signal: new AbortController().signal,
}
const rejectA2aDecision = await listeners.get('tools/pre-execute')(rejectA2aExecution, async () => ({ kind: 'allow' }))
assert.equal(rejectA2aDecision.kind, 'allow')
assert.equal(JSON.parse(await restartedA2aResume.execute(rejectA2aExecution.arguments, rejectA2aExecution)).status, 'rejected')
listeners.get('tools/result')(rejectA2aExecution, { isError: false })
assert.deepEqual(JSON.parse(await restartedA2aList.execute({})), [])
const interruptedRows = JSON.parse(await listInterrupted.execute({}))
assert.equal(interruptedRows.length, 1)
assert.equal(interruptedRows[0].status, 'orphaned')
assert.equal(interruptedRows[0].tool_name, 'restart_service')
const interruptedId = interruptedRows[0].id

const resumeExecution = {
  token: 'token-resume', callId: 'call-resume', name: resumeInterrupted.name,
  arguments: { request_id: interruptedId }, agent,
  signal: new AbortController().signal,
}
await assert.rejects(
  resumeInterrupted.execute(resumeExecution.arguments, resumeExecution),
  /fresh one-shot DSH approval/,
)

approvalOutcome = 'rejected'
const rejectedResume = { ...resumeExecution, token: 'token-resume-rejected', callId: 'call-resume-rejected' }
const rejectedResumeDecision = await listeners.get('tools/pre-execute')(rejectedResume, async () => ({ kind: 'allow' }))
assert.equal(rejectedResumeDecision.kind, 'deny')
assert.equal(JSON.parse(await listInterrupted.execute({}))[0].status, 'orphaned')

approvalOutcome = 'allowed-once'
const forbiddenResume = {
  ...resumeExecution,
  token: 'token-resume-forbidden',
  callId: 'call-resume-forbidden',
  arguments: { request_id: interruptedId, arguments: { service: 'billing', environment: 'staging' } },
}
const forbiddenDecision = await listeners.get('tools/pre-execute')(forbiddenResume, async () => ({ kind: 'allow' }))
assert.equal(forbiddenDecision.kind, 'deny')
assert.match(forbiddenDecision.reason, /non-editable keys.*service/)
assert.equal(JSON.parse(await listInterrupted.execute({}))[0].status, 'orphaned')

const resumeDecision = await listeners.get('tools/pre-execute')(resumeExecution, async () => ({ kind: 'allow' }))
assert.equal(resumeDecision.kind, 'allow')
const resumedResult = await resumeInterrupted.execute(resumeExecution.arguments, resumeExecution)
assert.match(resumedResult, /crm/i)
listeners.get('tools/result')(resumeExecution, { isError: false })
assert.deepEqual(JSON.parse(await listInterrupted.execute({})), [])

const asyncExecution = {
  token: 'token-async', callId: 'call-async', name: asyncSubmit.name,
  arguments: {
    tool_name: 'edit_device_config',
    arguments: { device_id: 'ap-01', config_lines: ['ntp server 10.0.0.5'], reason: 'initial' },
    default_value: { assumed: 'queued' },
    sla_seconds: 600,
  },
  agent,
  signal: new AbortController().signal,
}
const deferred = JSON.parse(await asyncSubmit.execute(asyncExecution.arguments, asyncExecution))
assert.equal(deferred.status, 'deferred')
assert.deepEqual(deferred.default_value, { assumed: 'queued' })
assert.equal(JSON.parse(await listInterrupted.execute({}))[0].status, 'deferred')

const editedResumeExecution = {
  token: 'token-edited-resume', callId: 'call-edited-resume', name: resumeInterrupted.name,
  arguments: {
    request_id: deferred.request_id,
    arguments: { device_id: 'ap-01', config_lines: ['ntp server 10.0.0.6'], reason: 'operator edit' },
  },
  agent,
  signal: new AbortController().signal,
}
const editedDecision = await listeners.get('tools/pre-execute')(editedResumeExecution, async () => ({ kind: 'allow' }))
assert.equal(editedDecision.kind, 'allow')
const editedResult = await resumeInterrupted.execute(editedResumeExecution.arguments, editedResumeExecution)
assert.match(editedResult, /ap-01/i)
listeners.get('tools/result')(editedResumeExecution, { isError: false })

const batchExecution = {
  token: 'token-batch', callId: 'call-batch', name: batch.name,
  arguments: {
    policy: 'best_effort',
    operations: [
      { tool_name: 'restart_service', arguments: { service: 'crm', environment: 'staging' } },
      { tool_name: 'rollback_service', arguments: { service: 'billing', version: '3.2.1', environment: 'staging' } },
    ],
  },
  agent,
  signal: new AbortController().signal,
}
const batchDecision = await listeners.get('tools/pre-execute')(batchExecution, async () => ({ kind: 'allow' }))
assert.equal(batchDecision.kind, 'allow')
const batchResult = JSON.parse(await batch.execute(batchExecution.arguments, batchExecution))
assert.deepEqual(batchResult.results.map(item => item.status), ['completed', 'completed'])
listeners.get('tools/result')(batchExecution, { isError: false })

const failingBatchExecution = {
  ...batchExecution,
  token: 'token-batch-failing',
  callId: 'call-batch-failing',
  arguments: {
    policy: 'all_or_nothing',
    operations: [
      { tool_name: 'restart_service', arguments: { service: 'api', environment: 'staging' } },
      { tool_name: 'push_config', arguments: { device_id: 'ap-01', config_text: 123 } },
      { tool_name: 'delete_resource', arguments: { resource_id: 'unused' } },
    ],
  },
}
const failingBatchDecision = await listeners.get('tools/pre-execute')(failingBatchExecution, async () => ({ kind: 'allow' }))
assert.equal(failingBatchDecision.kind, 'deny')
assert.match(failingBatchDecision.reason, /config_text must be a string/)

const expiringExecution = {
  ...asyncExecution,
  token: 'token-async-expiring',
  callId: 'call-async-expiring',
  arguments: {
    tool_name: 'restart_service',
    arguments: { service: 'search', environment: 'staging' },
    default_value: 'continue',
    sla_seconds: 60,
  },
}
const expiring = JSON.parse(await asyncSubmit.execute(expiringExecution.arguments, expiringExecution))
const expiryDatabase = new DatabaseSync(storePath)
expiryDatabase.prepare("UPDATE requests SET expires_at = '2000-01-01T00:00:00.000Z' WHERE id = ?").run(expiring.request_id)
expiryDatabase.close()
assert.equal(JSON.parse(await listInterrupted.execute({})).some(row => row.id === expiring.request_id), false)

for (const dispose of restartDisposers.reverse()) dispose()
const recovered = new DatabaseSync(storePath, { readOnly: true })
const original = recovered.prepare("SELECT status, recovered_arguments_json FROM requests WHERE call_id = 'call-approved'").get()
const recoveryAudit = recovered.prepare("SELECT status, outcome FROM requests WHERE tool_name = 'netopyu_hitl_resume' ORDER BY created_at").all()
const deferredOriginal = recovered.prepare("SELECT status, recovered_arguments_json FROM requests WHERE tool_name = 'edit_device_config'").get()
const batchItemCounts = recovered.prepare("SELECT status, COUNT(*) AS count FROM batch_items GROUP BY status ORDER BY status").all()
const expiredOriginal = recovered.prepare("SELECT status FROM requests WHERE id = ?").get(expiring.request_id)
const grantCounts = recovered.prepare("SELECT status, COUNT(*) AS count FROM tool_grants GROUP BY status ORDER BY status").all()
const a2aContinuationStatuses = recovered.prepare("SELECT status, COUNT(*) AS count FROM a2a_continuations GROUP BY status ORDER BY status").all()
recovered.close()
const runtimeAudit = new DatabaseSync(runtimeStorePath, { readOnly: true })
const rejectedPlanCount = runtimeAudit.prepare("SELECT COUNT(*) AS count FROM plans WHERE state='rejected'").get().count
const stillReadyCount = runtimeAudit.prepare("SELECT COUNT(*) AS count FROM plans WHERE state='plan_ready'").get().count
const approvalRejectedEvents = runtimeAudit.prepare("SELECT COUNT(*) AS count FROM plan_events WHERE event_type='approval_rejected'").get().count
runtimeAudit.close()
assert.equal(original.status, 'recovered')
assert.deepEqual(JSON.parse(original.recovered_arguments_json), { service: 'crm', environment: 'staging' })
assert.equal(deferredOriginal.status, 'recovered')
assert.deepEqual(JSON.parse(deferredOriginal.recovered_arguments_json), {
  device_id: 'ap-01', config_lines: ['ntp server 10.0.0.6'], reason: 'operator edit',
})
assert.deepEqual(recoveryAudit.map(row => ({ ...row })), [
  { status: 'denied', outcome: 'rejected' },
  { status: 'completed', outcome: 'allowed-once' },
  { status: 'completed', outcome: 'allowed-once' },
])
assert.deepEqual(batchItemCounts.map(row => ({ ...row })), [
  { status: 'completed', count: 2 },
])
assert.equal(expiredOriginal.status, 'expired')
assert.deepEqual(grantCounts.map(row => ({ ...row })), [
  { status: 'consumed', count: 7 },
  { status: 'orphaned', count: 1 },
])
assert.deepEqual(a2aContinuationStatuses.map(row => ({ ...row })), [
  { status: 'completed', count: 1 },
  { status: 'rejected', count: 1 },
])
assert.ok(rejectedPlanCount >= 2, 'DSH rejection and partial preparation must close runtime plans')
assert.ok(approvalRejectedEvents >= 2)
assert.equal(stillReadyCount, 0, 'smoke test must not leak executable unapproved plans')

const compactStorePath = join(tmpdir(), `netopyu-compact-${process.pid}.sqlite`)
const compactDefinitions = []
const compactDisposers = []
const compactServices = new Map()
const compactContext = {
  tools: { register(definition) { compactDefinitions.push(definition) } },
  approval: { async request() { return 'rejected' } },
  subagents: { registerProvider() {}, async start() { throw new Error('not used') } },
  skills: { register() {} },
  provide(name, value) { compactServices.set(name, value) },
  effect(factory) { compactDisposers.push(factory()) },
  on() {},
}
await apply(compactContext, {
  enableDestructive: false,
  hitlStorePath: compactStorePath,
  peerUrls: [],
  toolAllowlist: ['list_devices'],
})
assert.ok(compactDefinitions.some(tool => tool.name === 'list_devices'))
assert.equal(compactDefinitions.some(tool => tool.name === 'device_info'), false)
const compactCapabilities = await compactServices.get('netopyuCapabilities').search(
  'list network devices', { top_k: 20, kinds: ['tool'] },
)
assert.deepEqual(compactCapabilities.matches.map(match => match.id), ['list_devices'])
for (const dispose of compactDisposers.reverse()) dispose()
unlinkSync(compactStorePath)

await assert.rejects(
  apply(compactContext, {
    enableDestructive: false,
    hitlStorePath: compactStorePath,
    peerUrls: [],
    toolAllowlist: ['definitely_not_a_tool'],
  }),
  /unknown NetOpYu tool.*definitely_not_a_tool/,
)

unlinkSync(storePath)
for (const path of [runtimeStorePath, toolResultStorePath]) {
  for (const suffix of ['', '-shm', '-wal']) {
    if (existsSync(`${path}${suffix}`)) unlinkSync(`${path}${suffix}`)
  }
}
await new Promise((resolve, reject) => peerServer.close(error => error ? reject(error) : resolve()))
console.log(JSON.stringify({
  tools: initialToolCount,
  approvals: rows.length,
  statuses: rows.map(row => row.status),
  recoveredStatus: original.status,
}))
