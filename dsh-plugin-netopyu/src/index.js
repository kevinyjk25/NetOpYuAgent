import { dirname, resolve } from 'node:path'
import { isDeepStrictEqual } from 'node:util'
import { fileURLToPath } from 'node:url'
import { a2aToolDefinitions, NetOpYuA2AProvider } from './a2a.js'
import { callBridge, resolvePython } from './bridge.js'
import { createHitlStore, NetOpYuToolGuard } from './hitl-store.js'

export { NetOpYuA2AProvider } from './a2a.js'
export { NetOpYuToolGuard } from './hitl-store.js'

export const name = 'netopyu-tools'
export const inject = ['tools', 'approval', 'subagents', 'skills']

const sourceDirectory = dirname(fileURLToPath(import.meta.url))
const inferredProjectRoot = resolve(sourceDirectory, '..', '..')

function toolDefinition(tool, bridge, toolGuard) {
  const properties = {}
  const required = []
  for (const [parameterName, parameter] of Object.entries(tool.parameters)) {
    const { required: isRequired, ...schema } = parameter
    properties[parameterName] = schema
    if (isRequired === true) required.push(parameterName)
  }
  return {
    name: tool.name,
    description: tool.description,
    parameters: {
      type: 'object',
      properties,
      ...(required.length > 0 ? { required } : {}),
      additionalProperties: true,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({
      card: 'generic',
      title: tool.name,
      kind: 'read',
      rawInput: JSON.stringify(args),
    }),
    async execute(args, execution) {
      if (tool.requires_approval && !toolGuard.consume(execution.token, tool.name)) {
        throw new Error(`${tool.name} requires a current NetOpYu durable-HITL grant`)
      }
      const payload = await callBridge({
        ...bridge,
        command: 'invoke',
        tool: tool.name,
        args,
        signal: execution.signal,
        allowDestructive: tool.requires_approval,
        correlationId: execution.callId,
      })
      if (payload.ok !== true || typeof payload.result !== 'string') {
        throw new Error(payload.error ?? `invalid result from ${tool.name}`)
      }
      return payload.result
    },
  }
}


export class NetOpYuMemoryService {
  #bridge
  #memoryDirectory
  #operatorId

  constructor(bridge, memoryDirectory, operatorId) {
    this.#bridge = bridge
    this.#memoryDirectory = memoryDirectory
    this.#operatorId = operatorId
  }

  async recall(execution, query, options = {}) {
    const sessionId = execution.agent?.session?.id
    if (sessionId === undefined || String(sessionId).trim() === '') {
      throw new Error('NetOpYu memory recall requires a live DSH session scope')
    }
    return callBridge({
      ...this.#bridge,
      command: 'memory-recall',
      args: {
        memory_dir: this.#memoryDirectory,
        operator_id: this.#operatorId,
        session_id: String(sessionId),
        query,
        max_chars: options.max_chars,
        recent_turns: options.recent_turns,
      },
      signal: execution.signal,
      correlationId: execution.callId,
    })
  }
}

export class NetOpYuCapabilityService {
  #bridge
  #allowedToolNames

  constructor(bridge, allowedToolNames) {
    this.#bridge = bridge
    this.#allowedToolNames = [...allowedToolNames]
  }

  async search(query, options = {}, signal) {
    return callBridge({
      ...this.#bridge,
      command: 'capability-search',
      args: {
        query,
        top_k: options.top_k,
        kinds: options.kinds,
        allowed_tool_names: this.#allowedToolNames,
      },
      signal,
    })
  }
}

function scopedServiceDefinitions(memoryService, capabilityService) {
  const output = {
    schema: { type: 'string' },
    render: (_args, value) => [{ type: 'text', text: value }],
  }
  return [
    {
      name: 'netopyu_memory_recall',
      description: 'Explicitly recall NetOpYu memory scoped to the current DSH session and configured operator.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          max_chars: { type: 'integer', minimum: 200, maximum: 4000 },
          recent_turns: { type: 'integer', minimum: 0, maximum: 10 },
        },
        required: ['query'],
        additionalProperties: false,
      },
      output,
      presentCall: args => ({ card: 'generic', title: 'Recall NetOpYu memory', kind: 'read', rawInput: JSON.stringify(args) }),
      async execute(args, execution) {
        const result = await memoryService.recall(execution, args.query, args)
        return JSON.stringify(result, null, 2)
      },
    },
    {
      name: 'netopyu_capability_search',
      description: 'Search the current NetOpYu profile tool and skill catalog with CJK-aware BM25 retrieval.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          top_k: { type: 'integer', minimum: 1, maximum: 20 },
          kinds: { type: 'array', items: { type: 'string', enum: ['tool', 'skill'] }, uniqueItems: true },
        },
        required: ['query'],
        additionalProperties: false,
      },
      output,
      presentCall: args => ({ card: 'generic', title: 'Search NetOpYu capabilities', kind: 'read', rawInput: JSON.stringify(args) }),
      async execute(args, execution) {
        const result = await capabilityService.search(args.query, args, execution.signal)
        return JSON.stringify(result, null, 2)
      },
    },
  ]
}

function assertArgumentsObject(value, label = 'arguments') {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`)
  }
}

function validateEditableReplacement(tool, original, replacement) {
  assertArgumentsObject(replacement, 'recovery arguments')
  const changedKeys = new Set([...Object.keys(original), ...Object.keys(replacement)])
  for (const key of [...changedKeys]) {
    if (isDeepStrictEqual(original[key], replacement[key]) && Object.hasOwn(original, key) === Object.hasOwn(replacement, key)) {
      changedKeys.delete(key)
    }
  }
  const editable = new Set(tool.editable_parameters ?? [])
  const forbidden = [...changedKeys].filter(key => !editable.has(key))
  if (forbidden.length > 0) {
    throw new Error(
      `arguments change non-editable keys for ${tool.name}: ${forbidden.join(', ')}; editable keys: ${[...editable].join(', ') || 'none'}`,
    )
  }
}

function hitlDefinitions(manifest, bridge, toolGuard, approvalByToken, hitlStore) {
  const list = {
    name: 'netopyu_hitl_list',
    description: 'List interrupted or failed NetOpYu operations that may be resubmitted with fresh approval.',
    parameters: {
      type: 'object',
      properties: { limit: { type: 'integer', minimum: 1, maximum: 200 } },
      additionalProperties: false,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({ card: 'generic', title: 'NetOpYu interrupted operations', kind: 'read', rawInput: JSON.stringify(args) }),
    async execute(args) {
      return JSON.stringify(hitlStore.listRecoverable(args.limit), null, 2)
    },
  }

  const resume = {
    name: 'netopyu_hitl_resume',
    description: 'Resubmit one interrupted NetOpYu operation. This always requires a new one-shot DSH approval.',
    parameters: {
      type: 'object',
      properties: {
        request_id: { type: 'string', description: 'ID returned by netopyu_hitl_list.' },
        arguments: { type: 'object', description: 'Optional full replacement; changed keys must be editable for the original tool.' },
      },
      required: ['request_id'],
      additionalProperties: false,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({ card: 'generic', title: 'Resume NetOpYu operation', kind: 'write', rawInput: JSON.stringify(args) }),
    async execute(args, execution) {
      if (!toolGuard.consume(execution.token, resume.name)) {
        throw new Error('netopyu_hitl_resume requires a fresh one-shot DSH approval')
      }
      const original = hitlStore.recoverable(args.request_id)
      if (original === undefined) throw new Error(`recoverable NetOpYu request not found: ${args.request_id}`)
      const tool = manifest.tools.find(candidate => candidate.name === original.tool_name)
      if (tool?.requires_approval !== true) {
        throw new Error(`request ${args.request_id} does not reference an enabled approval-gated tool`)
      }
      const replayArguments = args.arguments ?? original.arguments
      if (args.arguments !== undefined) validateEditableReplacement(tool, original.arguments, replayArguments)
      const recoveryRequestId = approvalByToken.get(execution.token)
      if (recoveryRequestId === undefined) throw new Error('recovery approval audit record is missing')
      if (!hitlStore.claimRecovery(original.id, recoveryRequestId, replayArguments)) {
        throw new Error(`request ${original.id} was already claimed or is no longer recoverable`)
      }
      try {
        const payload = await callBridge({
          ...bridge,
          command: 'invoke',
          tool: tool.name,
          args: replayArguments,
          signal: execution.signal,
          allowDestructive: true,
          correlationId: execution.callId,
        })
        if (payload.ok !== true || typeof payload.result !== 'string') {
          throw new Error(payload.error ?? `invalid result from ${tool.name}`)
        }
        hitlStore.finishRecovery(original.id, recoveryRequestId, false)
        return payload.result
      } catch (error) {
        hitlStore.finishRecovery(original.id, recoveryRequestId, true)
        throw error
      }
    },
  }

  const asyncSubmit = {
    name: 'netopyu_hitl_async_submit',
    description: 'Defer an approval-gated NetOpYu operation and immediately return an optimistic default without executing it.',
    parameters: {
      type: 'object',
      properties: {
        tool_name: { type: 'string' },
        arguments: { type: 'object' },
        default_value: {},
        sla_seconds: { type: 'integer', minimum: 60, maximum: 86400 },
      },
      required: ['tool_name', 'arguments', 'default_value'],
      additionalProperties: false,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({ card: 'generic', title: 'Defer NetOpYu approval', kind: 'write', rawInput: JSON.stringify(args) }),
    async execute(args, execution) {
      assertArgumentsObject(args.arguments)
      const tool = manifest.tools.find(candidate => candidate.name === args.tool_name)
      if (tool?.requires_approval !== true) {
        throw new Error(`${args.tool_name} is not an enabled approval-gated NetOpYu tool`)
      }
      const slaSeconds = Math.max(60, Math.min(Number(args.sla_seconds) || 600, 86400))
      const reason = `Deferred NetOpYu ${tool.action_type} operation; no network action runs before a later fresh approval.`
      const deferred = hitlStore.beginDeferred(execution, tool.name, args.arguments, reason, slaSeconds)
      return JSON.stringify({
        request_id: deferred.id,
        status: 'deferred',
        default_value: args.default_value,
        expires_at: deferred.expiresAt,
      }, null, 2)
    },
  }

  const batch = {
    name: 'netopyu_hitl_batch',
    description: 'Execute 1-50 approval-gated NetOpYu operations after one fresh DSH approval, with durable per-item audit.',
    parameters: {
      type: 'object',
      properties: {
        operations: {
          type: 'array', minItems: 1, maxItems: 50,
          items: {
            type: 'object',
            properties: { tool_name: { type: 'string' }, arguments: { type: 'object' } },
            required: ['tool_name', 'arguments'],
            additionalProperties: false,
          },
        },
        policy: { type: 'string', enum: ['best_effort', 'all_or_nothing'] },
      },
      required: ['operations'],
      additionalProperties: false,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({ card: 'generic', title: 'NetOpYu batch operation', kind: 'write', rawInput: JSON.stringify(args) }),
    async execute(args, execution) {
      if (!toolGuard.consume(execution.token, batch.name)) {
        throw new Error('netopyu_hitl_batch requires a fresh one-shot DSH approval')
      }
      if (!Array.isArray(args.operations) || args.operations.length < 1 || args.operations.length > 50) {
        throw new Error('batch operations must contain 1-50 items')
      }
      const operations = args.operations.map((operation, index) => {
        assertArgumentsObject(operation?.arguments, `operations[${index}].arguments`)
        const tool = manifest.tools.find(candidate => candidate.name === operation.tool_name)
        if (tool?.requires_approval !== true) {
          throw new Error(`operations[${index}] does not reference an enabled approval-gated tool`)
        }
        return { tool, tool_name: tool.name, arguments: operation.arguments }
      })
      const policy = args.policy ?? 'best_effort'
      if (!['best_effort', 'all_or_nothing'].includes(policy)) throw new Error(`unsupported batch policy: ${policy}`)
      const requestId = approvalByToken.get(execution.token)
      if (requestId === undefined) throw new Error('batch approval audit record is missing')
      hitlStore.initializeBatch(requestId, operations)
      const results = []
      for (const [index, operation] of operations.entries()) {
        if (!hitlStore.startBatchItem(requestId, index)) throw new Error(`batch item ${index} was already claimed`)
        try {
          const payload = await callBridge({
            ...bridge,
            command: 'invoke',
            tool: operation.tool_name,
            args: operation.arguments,
            signal: execution.signal,
            allowDestructive: true,
            correlationId: `${String(execution.callId ?? requestId)}:${index}`,
          })
          if (payload.ok !== true || typeof payload.result !== 'string') {
            throw new Error(payload.error ?? `invalid result from ${operation.tool_name}`)
          }
          hitlStore.finishBatchItem(requestId, index, payload.result, undefined)
          results.push({ index, tool_name: operation.tool_name, status: 'completed', result: payload.result })
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          hitlStore.finishBatchItem(requestId, index, undefined, message)
          results.push({ index, tool_name: operation.tool_name, status: 'failed', error: message })
          if (policy === 'all_or_nothing') {
            hitlStore.skipBatchRemainder(requestId, index, 'stopped after earlier batch failure')
            throw new Error(`batch stopped at item ${index}; network actions already completed are not transactionally rolled back: ${message}`)
          }
        }
      }
      return JSON.stringify({ policy, results }, null, 2)
    },
  }
  return [list, resume, asyncSubmit, batch]
}

function trajectoryDefinition(hitlStore) {
  return {
    name: 'netopyu_trajectory_recent',
    description: 'Read recent privacy-minimized DSH trajectory events for evaluation and workflow mining.',
    parameters: {
      type: 'object', properties: { limit: { type: 'integer', minimum: 1, maximum: 500 } }, additionalProperties: false,
    },
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    presentCall: args => ({ card: 'generic', title: 'Recent NetOpYu DSH trajectories', kind: 'read', rawInput: JSON.stringify(args) }),
    async execute(args) { return JSON.stringify(hitlStore.recentTrajectory(args.limit), null, 2) },
  }
}

export async function apply(ctx, config = {}) {
  const projectRoot = resolve(config.projectRoot ?? process.env.NETOPYU_ROOT ?? inferredProjectRoot)
  const profile = config.profile ?? process.env.NETOPYU_PROFILE ?? 'lan'
  const python = config.pythonExecutable ?? await resolvePython(projectRoot)
  const bridge = { projectRoot, python, profile }
  const enableDestructive = config.enableDestructive ?? process.env.NETOPYU_DSH_ENABLE_DESTRUCTIVE === '1'
  const bridgeManifest = await callBridge({ ...bridge, command: 'manifest', includeDestructive: enableDestructive })
  if (!Array.isArray(bridgeManifest.tools)) throw new Error('NetOpYu manifest has no tools array')
  const configuredAllowlist = config.toolAllowlist ?? process.env.NETOPYU_DSH_TOOL_ALLOWLIST
  const toolAllowlist = Array.isArray(configuredAllowlist)
    ? configuredAllowlist.map(value => String(value).trim()).filter(Boolean)
    : String(configuredAllowlist ?? '').split(',').map(value => value.trim()).filter(Boolean)
  const availableToolNames = new Set(bridgeManifest.tools.map(tool => tool.name))
  const unknownTools = toolAllowlist.filter(toolName => !availableToolNames.has(toolName))
  if (unknownTools.length > 0) {
    throw new Error(`unknown NetOpYu tool(s) in allowlist: ${unknownTools.join(', ')}`)
  }
  const allowedToolNames = new Set(toolAllowlist)
  const manifest = toolAllowlist.length === 0
    ? bridgeManifest
    : { ...bridgeManifest, tools: bridgeManifest.tools.filter(tool => allowedToolNames.has(tool.name)) }
  const skillManifest = await callBridge({ ...bridge, command: 'skill-manifest' })
  if (!Array.isArray(skillManifest.skills)) throw new Error('NetOpYu skill manifest has no skills array')
  const approvalByToken = new Map()
  const hitlStorePath = config.hitlStorePath ?? process.env.NETOPYU_DSH_HITL_STORE ?? resolve(projectRoot, 'data', 'dsh_hitl.sqlite')
  const hitlStore = createHitlStore(hitlStorePath)
  const toolGuard = new NetOpYuToolGuard(hitlStore)
  const memoryDirectory = resolve(
    projectRoot,
    config.memoryDirectory
      ?? process.env.NETOPYU_DSH_MEMORY_DIR
      ?? `data/agents/${profile}-agent/memory`,
  )
  const operatorId = String(config.operatorId ?? process.env.NETOPYU_DSH_OPERATOR_ID ?? 'dev-user')
  const memoryService = new NetOpYuMemoryService(bridge, memoryDirectory, operatorId)
  const capabilityService = new NetOpYuCapabilityService(
    bridge, manifest.tools.map(tool => tool.name),
  )
  const peerUrls = config.peerUrls ?? (process.env.NETOPYU_DSH_A2A_PEERS
    ? process.env.NETOPYU_DSH_A2A_PEERS.split(',').map(value => value.trim()).filter(Boolean)
    : undefined)
  const a2aProvider = new NetOpYuA2AProvider(bridge, {
    providerName: config.a2aProviderName ?? 'netopyu-a2a',
    ownAgentId: String(config.ownAgentId ?? process.env.AGENT_ID ?? `${profile}-agent`),
    peerUrls,
    timeoutSeconds: Number(config.a2aTimeoutSeconds ?? process.env.NETOPYU_DSH_A2A_TIMEOUT ?? 300),
    maxHops: Number(config.a2aMaxHops ?? process.env.NETOPYU_DSH_A2A_MAX_HOPS ?? 3),
    continuationStore: hitlStore,
  })
  ctx.provide('netopyuToolGuard', toolGuard)
  ctx.provide('netopyuMemory', memoryService)
  ctx.provide('netopyuCapabilities', capabilityService)
  ctx.provide('netopyuA2A', a2aProvider)
  ctx.effect(() => () => hitlStore.close())
  ctx.subagents.registerProvider(a2aProvider)
  for (const skill of skillManifest.skills) {
    ctx.skills.register({
      name: skill.name,
      description: skill.description,
      content: skill.content,
      path: skill.path,
      resourceBase: { kind: 'directory', path: skill.resource_base },
      metadata: skill.metadata,
      source: `netopyu-${profile}`,
      provider: 'netopyu',
    })
  }

  for (const tool of manifest.tools) ctx.tools.register(toolDefinition(tool, bridge, toolGuard))
  for (const tool of scopedServiceDefinitions(memoryService, capabilityService)) ctx.tools.register(tool)
  for (const tool of a2aToolDefinitions(ctx, a2aProvider, bridge, peerUrls, hitlStore, toolGuard)) ctx.tools.register(tool)
  const hitlTools = hitlDefinitions(manifest, bridge, toolGuard, approvalByToken, hitlStore)
  for (const tool of hitlTools) ctx.tools.register(tool)
  ctx.tools.register(trajectoryDefinition(hitlStore))

  ctx.on('session/event', (session, event) => {
    hitlStore.recordTrajectory(session.id, `session:${String(event.type)}`, {
      seq: event.seq,
      data_keys: event.data && typeof event.data === 'object' ? Object.keys(event.data).sort() : [],
    })
  })

  ctx.on('tools/pre-execute', async (execution, next) => {
    const sessionId = execution.agent?.session?.id
    if (sessionId !== undefined) {
      hitlStore.recordTrajectory(sessionId, 'tool:start', {
        tool_name: execution.name,
        argument_keys: Object.keys(execution.arguments ?? {}).sort(),
        call_id: execution.callId === undefined ? null : String(execution.callId),
      })
    }
    const tool = manifest.tools.find(candidate => candidate.name === execution.name)
    const hitlAction = {
      netopyu_hitl_resume: 'recovery',
      netopyu_hitl_batch: 'batch',
      netopyu_a2a_hitl_resume: 'remote-hitl-resume',
    }[execution.name]
    if (tool?.requires_approval !== true && hitlAction === undefined) return next()
    const actionType = hitlAction ?? tool.action_type
    const reason = `NetOpYu ${actionType} operation requires operator approval; arguments are durably recorded.`
    const requestId = hitlStore.begin(execution, reason)
    approvalByToken.set(execution.token, requestId)
    let outcome
    try {
      outcome = await ctx.approval.request({
        agent: execution.agent,
        toolName: execution.name,
        callId: execution.callId,
        reason,
        signal: execution.signal,
      })
    } catch (error) {
      outcome = 'unavailable'
    }
    hitlStore.decided(requestId, outcome)
    if (outcome !== 'allowed-once') {
      approvalByToken.delete(execution.token)
      return { kind: 'deny', reason: `NetOpYu operation not approved (${outcome})` }
    }
    toolGuard.issue(execution.token, requestId, execution.name)
    try {
      return await next()
    } catch (error) {
      toolGuard.revoke(execution.token, 'downstream pre-execute hook failed')
      approvalByToken.delete(execution.token)
      hitlStore.completed(requestId, true)
      throw error
    }
  })

  ctx.on('tools/result', (execution, result) => {
    const sessionId = execution.agent?.session?.id
    if (sessionId !== undefined) {
      hitlStore.recordTrajectory(sessionId, 'tool:result', {
        tool_name: execution.name,
        is_error: result.isError === true,
        call_id: execution.callId === undefined ? null : String(execution.callId),
      })
    }
    const requestId = approvalByToken.get(execution.token)
    if (requestId === undefined) return
    approvalByToken.delete(execution.token)
    toolGuard.revoke(execution.token)
    hitlStore.completed(requestId, result.isError === true)
  })
}
