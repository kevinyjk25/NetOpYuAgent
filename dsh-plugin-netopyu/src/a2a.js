import { randomUUID } from 'node:crypto'
import { callBridge } from './bridge.js'
const A2A_ROUTE_MARKER = 'netopyu-a2a-route-v1'

function textPrompt(blocks) {
  if (!Array.isArray(blocks) || blocks.some(block => block?.type !== 'text' || typeof block.text !== 'string')) {
    throw new Error('NetOpYu A2A delegation accepts text prompt blocks only')
  }
  const text = blocks.map(block => block.text).join('')
  if (text.trim() === '') throw new Error('NetOpYu A2A delegation prompt must not be empty')
  return text
}

function linkedAbortController(signal) {
  const controller = new AbortController()
  const abort = () => controller.abort(signal.reason ?? new Error('A2A delegation aborted'))
  if (signal.aborted) abort()
  else signal.addEventListener('abort', abort, { once: true })
  return { controller, unlink: () => signal.removeEventListener('abort', abort) }
}

export class NetOpYuA2AProvider {
  capabilities = { outputSchema: false, depthLimit: false, toolFilter: false, persona: false }
  inheritsParentContext = false
  #bridge
  #ownAgentId
  #peerUrls
  #timeoutSeconds
  #maxHops
  #continuationStore

  constructor(bridge, options = {}) {
    this.name = options.providerName ?? 'netopyu-a2a'
    this.#bridge = bridge
    this.#ownAgentId = options.ownAgentId
    this.#peerUrls = options.peerUrls
    this.#timeoutSeconds = options.timeoutSeconds
    this.#maxHops = options.maxHops
    this.#continuationStore = options.continuationStore
  }

  async start(request) {
    const rawPrompt = textPrompt(request.prompt)
    let route = {}
    let prompt = rawPrompt
    const newline = rawPrompt.indexOf('\n')
    if (newline > 0) {
      try {
        const envelope = JSON.parse(rawPrompt.slice(0, newline))
        if (envelope?.marker === A2A_ROUTE_MARKER) {
          route = envelope.route ?? {}
          prompt = rawPrompt.slice(newline + 1)
        }
      } catch {
        // A normal prompt may start with JSON; only our exact marker is control data.
      }
    }
    const { controller, unlink } = linkedAbortController(request.signal)
    const id = `a2a-${String(request.parent?.session?.id ?? 'unknown')}-${randomUUID()}`
    const delegateArgs = {
      prompt,
      target: route.target ?? '',
      capability: route.capability ?? '',
      session_id: String(request.parent?.session?.id ?? id),
      own_agent_id: this.#ownAgentId,
      delegation_chain: Array.isArray(route.delegation_chain) ? route.delegation_chain : [],
      ...(this.#peerUrls === undefined ? {} : { peer_urls: this.#peerUrls }),
      timeout_seconds: this.#timeoutSeconds,
      max_hops: this.#maxHops,
    }
    let settled = false
    const result = callBridge({
      ...this.#bridge,
      command: 'a2a-delegate',
      args: delegateArgs,
      signal: controller.signal,
      correlationId: id,
    }).then(payload => {
      settled = true
      unlink()
      const output = payload.text ? [{ type: 'text', text: payload.text }] : []
      if (payload.status === 'completed' && payload.ok === true) return { output, stopReason: 'completed' }
      let continuationId
      if (payload.status === 'input-required' && payload.interrupt_id && this.#continuationStore) {
        continuationId = this.#continuationStore.recordA2aContinuation(
          delegateArgs.session_id, payload.peer, payload.interrupt_id, delegateArgs,
        )
      }
      const diagnostic = [
        payload.error ?? `A2A peer ended with status ${payload.status ?? 'unknown'}`,
        continuationId && `durable continuation: ${continuationId}`,
      ].filter(Boolean).join('; ').slice(0, 4096)
      const stopReason = payload.status === 'refused' || payload.status === 'unavailable' || payload.status === 'input-required'
        ? 'refusal'
        : 'error'
      return { output, diagnostic, stopReason }
    }).catch(error => {
      settled = true
      unlink()
      return {
        output: [],
        diagnostic: String(error instanceof Error ? error.message : error).slice(0, 4096),
        stopReason: controller.signal.aborted ? 'aborted' : 'error',
      }
    })
    return {
      id,
      localAgent: undefined,
      result,
      async dispose() {
        if (!settled && !controller.signal.aborted) controller.abort(new Error('A2A run disposed'))
        await result
      },
    }
  }
}

export function a2aToolDefinitions(ctx, provider, bridge, peerUrls, hitlStore, toolGuard, bindingByToken) {
  const output = {
    schema: { type: 'string' },
    render: (_args, value) => [{ type: 'text', text: value }],
  }
  return [
    {
      name: 'netopyu_peer_list',
      description: 'Discover configured remote A2A peers and list their advertised skills and reachability.',
      parameters: { type: 'object', properties: {}, additionalProperties: false },
      output,
      presentCall: () => ({ card: 'generic', title: 'Discover NetOpYu A2A peers', kind: 'read' }),
      async execute(_args, execution) {
        const payload = await callBridge({
          ...bridge,
          command: 'a2a-peers',
          args: peerUrls === undefined ? {} : { peer_urls: peerUrls },
          signal: execution.signal,
          correlationId: execution.callId,
        })
        return JSON.stringify(payload, null, 2)
      },
    },
    {
      name: 'netopyu_delegate',
      description: 'Delegate one bounded task to a configured remote NetOpYu A2A peer through the DSH subagent provider. Use target for a known agent id or capability for capability routing.',
      parameters: {
        type: 'object',
        properties: {
          description: { type: 'string', description: 'Short display label for the delegated task.' },
          prompt: { type: 'string', description: 'Self-contained task for the remote peer.' },
          target: { type: 'string', description: 'Optional exact A2A agent id or card name.' },
          capability: { type: 'string', description: 'Optional advertised peer skill, tag, or capability query.' },
        },
        required: ['description', 'prompt'],
        anyOf: [
          { required: ['target'] },
          { required: ['capability'] },
        ],
        additionalProperties: false,
      },
      output,
      presentCall: args => ({ card: 'generic', title: `Delegate to ${args.target || args.capability || 'A2A peer'}`, kind: 'read', rawInput: JSON.stringify(args) }),
      async execute(args, execution) {
        if (!args.target && !args.capability) throw new Error('netopyu_delegate requires target or capability')
        const envelope = JSON.stringify({ marker: A2A_ROUTE_MARKER, route: { target: args.target, capability: args.capability } })
        const run = await ctx.subagents.start(provider.name, {
          label: args.description,
          prompt: [{ type: 'text', text: `${envelope}\n${args.prompt}` }],
          parent: execution.agent,
          signal: execution.signal,
        })
        try {
          const result = await run.result
          const text = result.output.filter(block => block.type === 'text').map(block => block.text).join('')
          if (result.stopReason !== 'completed') {
            throw new Error([`A2A subagent ${result.stopReason}`, result.diagnostic, text && `Partial output: ${text}`].filter(Boolean).join('\n'))
          }
          return text || JSON.stringify({ status: 'completed', run_id: run.id })
        } finally {
          await run.dispose()
        }
      },
    },
    {
      name: 'netopyu_a2a_hitl_list',
      description: 'List durable local continuations waiting for a simulated remote A2A HITL decision.',
      parameters: {
        type: 'object', properties: { limit: { type: 'integer', minimum: 1, maximum: 200 } }, additionalProperties: false,
      },
      output,
      presentCall: () => ({ card: 'generic', title: 'List remote A2A HITL continuations', kind: 'read' }),
      async execute(args) {
        return JSON.stringify(hitlStore.listA2aContinuations(args.limit), null, 2)
      },
    },
    {
      name: 'netopyu_a2a_hitl_resume',
      description: 'Approve or reject one durable simulated remote A2A HITL continuation. Always requires fresh DSH approval.',
      parameters: {
        type: 'object',
        properties: {
          continuation_id: { type: 'string' },
          decision: { type: 'string', enum: ['approve', 'reject'] },
        },
        required: ['continuation_id', 'decision'],
        additionalProperties: false,
      },
      output,
      presentCall: args => ({ card: 'generic', title: `${args.decision} remote A2A continuation`, kind: 'write', rawInput: JSON.stringify(args) }),
      async execute(args, execution) {
        const binding = bindingByToken.get(execution.token)
        if (!toolGuard.consume(execution.token, 'netopyu_a2a_hitl_resume', binding?.bindingHash)) {
          throw new Error('netopyu_a2a_hitl_resume requires a fresh one-shot DSH approval')
        }
        const continuation = hitlStore.a2aContinuation(args.continuation_id)
        if (continuation === undefined) throw new Error(`remote A2A continuation not found: ${args.continuation_id}`)
        if (!hitlStore.claimA2aContinuation(continuation.id)) {
          throw new Error(`remote A2A continuation ${continuation.id} was already claimed`)
        }
        if (args.decision === 'reject') {
          hitlStore.finishA2aContinuation(continuation.id, 'rejected', undefined, 'rejected by local DSH operator')
          return JSON.stringify({ continuation_id: continuation.id, status: 'rejected' }, null, 2)
        }
        let resultText
        try {
          const payload = await callBridge({
            ...bridge,
            command: 'a2a-delegate',
            args: {
              ...continuation.request,
              resume_interrupt_id: continuation.interrupt_id,
              operator_decision: 'approve',
            },
            signal: execution.signal,
            correlationId: execution.callId,
          })
          resultText = payload.text
          if (payload.ok !== true || payload.status !== 'completed') {
            const message = payload.error ?? `remote A2A resume ended with status ${payload.status ?? 'unknown'}`
            throw new Error(message)
          }
          hitlStore.finishA2aContinuation(continuation.id, 'completed', payload.text, undefined)
          return payload.text || JSON.stringify({ continuation_id: continuation.id, status: 'completed' })
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          hitlStore.finishA2aContinuation(continuation.id, 'failed', resultText, message)
          throw error
        }
      },
    },
  ]
}
