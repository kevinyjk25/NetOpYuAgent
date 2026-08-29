import assert from 'node:assert/strict'
import { readFileSync, unlinkSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { apply } from '../src/index.js'


const suffix = `${process.pid}-${Date.now()}`
const hitlStorePath = join(tmpdir(), `netopyu-l1-shadow-hitl-${suffix}.sqlite`)
const canaryHitlStorePath = join(tmpdir(), `netopyu-l1-canary-disabled-${suffix}.sqlite`)
const decisionStorePath = join(tmpdir(), `netopyu-l1-shadow-decisions-${suffix}.sqlite`)
const runtimeStorePath = join(tmpdir(), `netopyu-l1-shadow-runtime-${suffix}.sqlite`)
const resultStorePath = join(tmpdir(), `netopyu-l1-shadow-results-${suffix}.sqlite`)
delete process.env.NETOPYU_DSH_WORKER_SOCKET
process.env.NETOPYU_L1_DECISION_STORE = decisionStorePath
process.env.NETOPYU_L1_DECISION_REPAIR_LIMIT = '0'
process.env.NETOPYU_DSH_NETWORK_RUNTIME_STORE = runtimeStorePath
process.env.NETOPYU_DSH_TOOL_RESULT_STORE = resultStorePath

let requestCount = 0
const modelServer = createServer(async (request, response) => {
  if (request.method !== 'POST' || request.url !== '/v1/chat/completions') {
    response.writeHead(404)
    response.end()
    return
  }
  let raw = ''
  for await (const chunk of request) raw += chunk
  const body = JSON.parse(raw)
  requestCount += 1
  const selected = body.tools.find(tool => (
    tool.function.description.includes('skill:service-health')
  ))
  assert.ok(selected, 'candidate contract should contain the reviewed service-health skill')
  const payload = JSON.stringify({
    id: 'shadow-test',
    object: 'chat.completion',
    model: body.model,
    choices: [{
      index: 0,
      finish_reason: 'tool_calls',
      message: {
        role: 'assistant',
        content: null,
        tool_calls: [{
          id: 'shadow-call-1',
          type: 'function',
          function: {
            name: selected.function.name,
            arguments: JSON.stringify({ service: 'payments', environment: '生产' }),
          },
        }],
      },
    }],
    usage: { prompt_tokens: 20, completion_tokens: 5 },
  })
  response.writeHead(200, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload),
  })
  response.end(payload)
})
await new Promise(resolve => modelServer.listen(0, '127.0.0.1', resolve))
process.env.NETOPYU_L1_DECISION_BASE_URL = `http://127.0.0.1:${modelServer.address().port}/v1`

const definitions = []
const listeners = new Map()
const services = new Map()
const registeredSkills = new Map()
const disposers = []
const context = {
  tools: { register(definition) { definitions.push(definition) } },
  approval: { async request() { return 'rejected' } },
  subagents: {
    registerProvider() {},
    async start() { throw new Error('not used') },
  },
  skills: {
    register(skill) {
      registeredSkills.set(skill.name, skill)
      return () => registeredSkills.delete(skill.name)
    },
  },
  provide(name, value) { services.set(name, value) },
  effect(factory) { disposers.push(factory()) },
  on(event, listener) { listeners.set(event, listener) },
}

await assert.rejects(
  () => apply(context, {
    decisionMode: 'canary',
    decisionModel: 'selector-test',
    enableDestructive: false,
    hitlStorePath: canaryHitlStorePath,
  }),
  /must be off or shadow/,
  'C1 readiness code must not activate the unqualified DSH canary path',
)

await apply(context, {
  decisionMode: 'shadow',
  decisionModel: 'selector-test',
  enableDestructive: false,
  hitlStorePath,
})
const decisionService = services.get('netopyuDecisionPlane')
assert.equal(decisionService.mode, 'shadow')
const agent = { session: { id: 'shadow-session-1' } }
const userMessage = {
  id: 'shadow-user-1',
  role: 'user',
  source: { kind: 'user' },
  content: [{ type: 'text', text: '检查生产环境 payments 服务 health。' }],
}
const entered = { kind: 'enter', messages: [userMessage] }
const preStep = await listeners.get('agent/pre-step')(
  { agent, messages: [userMessage], turn: 1, step: 1, signal: new AbortController().signal },
  async () => entered,
)
assert.equal(preStep, entered, 'shadow mode must not replace or reject the DSH step')
assert.equal(requestCount, 1)
const history = await decisionService.recent({ limit: 5 })
assert.equal(history.count, 1)
assert.equal(history.decisions[0].envelope.authority, 'proposal_only')
assert.equal(history.decisions[0].envelope.decision.action, 'select_skill')
assert.equal(history.decisions[0].envelope.decision.target, 'service-health')
assert.deepEqual(history.decisions[0].envelope.decision.argument_keys, [
  'environment', 'service',
])
assert.equal('arguments' in history.decisions[0].envelope.decision, false)
assert.equal(history.decisions[0].envelope.evidence.input_tokens, 20)
assert.equal(history.decisions[0].envelope.evidence.output_tokens, 5)
assert.equal(history.decisions[0].envelope.evidence.token_usage_complete, true)

const execution = {
  token: 'shadow-token-1',
  callId: 'shadow-call-1',
  name: 'skill',
  arguments: { name: 'service-health' },
  agent,
  signal: new AbortController().signal,
}
const toolDecision = await listeners.get('tools/pre-execute')(
  execution, async () => ({ kind: 'allow' }),
)
assert.equal(toolDecision.kind, 'allow')
const metrics = await decisionService.metrics({ limit: 20 })
assert.equal(metrics.decisions, 1)
assert.equal(metrics.observed_routes, 1)
assert.equal(metrics.routing_agreement_rate, 1)
assert.equal(metrics.safety_escape_count, 0)
assert.equal(metrics.protocol_success_rate, 1)
assert.deepEqual(metrics.reported_tokens, {
  input: 20, output: 5, usage_complete_rate: 1,
})

const secondMessage = {
  id: 'shadow-user-2',
  role: 'user',
  source: { kind: 'user' },
  content: [{ type: 'text', text: '检查生产环境 inventory 服务 health。' }],
}
await listeners.get('agent/pre-step')(
  { agent, messages: [secondMessage], turn: 2, step: 1, signal: new AbortController().signal },
  async () => ({ kind: 'enter', messages: [secondMessage] }),
)
const thirdMessage = {
  id: 'shadow-user-3',
  role: 'user',
  source: { kind: 'user' },
  content: [{ type: 'text', text: '检查生产环境 billing 服务 health。' }],
}
await listeners.get('agent/pre-step')(
  { agent, messages: [thirdMessage], turn: 3, step: 1, signal: new AbortController().signal },
  async () => ({ kind: 'enter', messages: [thirdMessage] }),
)
const lifecycle = await decisionService.recent({ limit: 5, sessionId: agent.session.id })
assert.equal(requestCount, 3)
assert.equal(lifecycle.decisions[0].lifecycle_status, 'pending')
assert.equal(lifecycle.decisions[1].lifecycle_status, 'closed')
assert.equal(lifecycle.decisions[1].lifecycle_reason, 'superseded')
assert.equal(lifecycle.decisions[2].lifecycle_status, 'observed')

const rawDatabase = readFileSync(decisionStorePath)
assert.equal(
  rawDatabase.includes(Buffer.from('检查生产环境 payments 服务 health。')),
  false,
  'decision store must retain only the prompt digest',
)
assert.equal(
  rawDatabase.includes(Buffer.from('payments')),
  false,
  'decision store must retain only argument keys and digests',
)

for (const dispose of disposers.reverse()) dispose()
await new Promise(resolve => modelServer.close(resolve))
for (const path of [
  hitlStorePath, canaryHitlStorePath, decisionStorePath, runtimeStorePath, resultStorePath,
]) {
  for (const suffixValue of ['', '-wal', '-shm']) {
    try { unlinkSync(`${path}${suffixValue}`) } catch {}
  }
}
