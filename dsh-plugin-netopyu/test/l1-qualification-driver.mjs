import { apply } from '../src/index.js'


const MAX_INPUT_BYTES = 10 * 1024 * 1024

async function readInput() {
  const chunks = []
  let size = 0
  for await (const chunk of process.stdin) {
    size += chunk.length
    if (size > MAX_INPUT_BYTES) throw new Error('qualification input exceeds 10 MiB')
    chunks.push(chunk)
  }
  const payload = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('qualification input must be an object')
  }
  if (!['lan', 'dc', 'wan'].includes(payload.profile)) {
    throw new Error('qualification profile is invalid')
  }
  if (typeof payload.model !== 'string' || payload.model.trim() === '') {
    throw new Error('qualification model is required')
  }
  if (!Array.isArray(payload.cases) || payload.cases.length > 5000) {
    throw new Error('qualification cases must be a bounded array')
  }
  return payload
}

function receipt(caseValue, envelope) {
  const evidence = envelope.evidence ?? {}
  return {
    case_digest: caseValue.case_digest,
    repetition: caseValue.repetition,
    profile: envelope.profile,
    harness: envelope.harness,
    status: envelope.status,
    decision_digest: envelope.decision_digest ?? null,
    prompt_digest: evidence.prompt_digest,
    catalog_digest: evidence.catalog_digest,
    candidate_digest: evidence.candidate_digest,
    policy_digest: evidence.policy_digest,
    model: evidence.model,
    protocol_valid: evidence.protocol_valid,
  }
}

const payload = await readInput()
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
    async start() { throw new Error('subagents are disabled during qualification') },
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

try {
  await apply(context, {
    profile: payload.profile,
    decisionMode: 'shadow',
    decisionModel: payload.model,
    enableDestructive: false,
    hitlStorePath: process.env.NETOPYU_DSH_HITL_STORE,
  })
  const preStep = listeners.get('agent/pre-step')
  const decisionService = services.get('netopyuDecisionPlane')
  if (typeof preStep !== 'function' || decisionService === undefined) {
    throw new Error('DSH qualification hook or Decision service is missing')
  }
  const receipts = []
  for (const [index, caseValue] of payload.cases.entries()) {
    if (
      !caseValue || typeof caseValue !== 'object'
      || typeof caseValue.case_digest !== 'string'
      || typeof caseValue.prompt !== 'string'
      || !Number.isInteger(caseValue.repetition)
    ) {
      throw new Error('DSH qualification case is invalid')
    }
    const sessionId = `qualification:dsh:${caseValue.case_digest.slice(-20)}:${caseValue.repetition}`
    const agent = { session: { id: sessionId } }
    const userMessage = {
      id: `qualification-${index}`,
      role: 'user',
      source: { kind: 'user' },
      content: [{ type: 'text', text: caseValue.prompt }],
    }
    const entered = { kind: 'enter', messages: [userMessage] }
    const accepted = await preStep(
      {
        agent,
        messages: [userMessage],
        turn: index + 1,
        step: 1,
        signal: new AbortController().signal,
      },
      async () => entered,
    )
    if (accepted !== entered) throw new Error('DSH shadow changed the accepted step')
    const history = await decisionService.recent({ limit: 2, sessionId })
    if (history?.count !== 1 || !history.decisions?.[0]?.envelope) {
      throw new Error('DSH qualification Decision receipt is missing or ambiguous')
    }
    const envelope = history.decisions[0].envelope
    receipts.push(receipt(caseValue, envelope))
    await decisionService.close({
      decisionId: envelope.decision_id,
      sessionId,
      reason: 'session_end',
    })
  }
  process.stdout.write(JSON.stringify({ receipts }))
} finally {
  for (const dispose of disposers.reverse()) dispose()
}
