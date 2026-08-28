import { createHash } from 'node:crypto'
import { dirname, resolve } from 'node:path'
import { isDeepStrictEqual } from 'node:util'
import { fileURLToPath } from 'node:url'
import { a2aToolDefinitions, NetOpYuA2AProvider } from './a2a.js'
import { callBridge, resolvePython } from './bridge.js'
import { createHitlStore, NetOpYuToolGuard } from './hitl-store.js'

// HMR contract marker: reload the plugin when the Python topology/path surface changes.
export const networkLabContractVersion = 'p075-b1-topology-path-v1'

export { NetOpYuA2AProvider } from './a2a.js'
export { NetOpYuToolGuard } from './hitl-store.js'

export const name = 'netopyu-tools'
export const inject = ['tools', 'approval', 'subagents', 'skills']

const sourceDirectory = dirname(fileURLToPath(import.meta.url))
const inferredProjectRoot = resolve(sourceDirectory, '..', '..')

function bindingHash(value) {
  return `sha256:${createHash('sha256').update(JSON.stringify(value)).digest('hex')}`
}

function preparationFailure(prepared) {
  const details = [...(prepared.missing ?? []), ...(prepared.errors ?? [])]
  return `${prepared.status ?? 'rejected'}${details.length > 0 ? `: ${details.join('; ')}` : ''}`
}

function observationAccessContext(execution, operatorId, profile) {
  const sessionId = execution.agent?.session?.id
  return {
    subject_id: operatorId,
    session_id: sessionId === undefined ? undefined : String(sessionId),
    roles: ['operations-reader', 'network-operator'],
    scopes: ['*', `profile:${profile}`],
    purpose: 'interactive-network-operations',
    clearance: 'restricted',
    authenticated: true,
  }
}

async function rejectPreparedPlans(bridge, preparedValues, reason, signal) {
  await Promise.all((preparedValues ?? []).map(async prepared => {
    if (prepared?.plan?.plan_id === undefined) return
    try {
      await callBridge({
        ...bridge,
        command: 'runtime-reject',
        args: {
          plan_id: prepared.plan.plan_id,
          plan_hash: prepared.plan.plan_hash,
          reason,
        },
        signal,
      })
    } catch {
      // A plan that already started has its own authoritative terminal state.
    }
  }))
}

async function executePreparedPlan(bridge, prepared, requestId, operatorId, execution, suffix = '') {
  const plan = prepared?.plan
  if (plan === undefined || prepared.execution_nonce === undefined) {
    throw new Error('approved Network Runtime plan is missing')
  }
  try {
    const outcome = await callBridge({
      ...bridge,
      command: 'runtime-execute',
      tool: plan.tool_name,
      args: {
        plan_id: plan.plan_id,
        plan_hash: plan.plan_hash,
        execution_nonce: prepared.execution_nonce,
        approval_request_id: requestId,
        approval_actor: operatorId,
      },
      signal: execution.signal,
      allowDestructive: true,
      correlationId: `${String(execution.callId ?? requestId)}${suffix}`,
    })
    const terminal = outcome.terminal_envelope
    if (
      outcome.ok !== true
      || outcome.state !== 'verified_success'
      || terminal?.contract !== 'netopyu.effect-runtime-terminal@1.0.0'
      || terminal?.terminal !== true
      || terminal?.state !== 'verified_success'
    ) {
      throw new Error(
        `Network Runtime did not verify success (state=${outcome.state ?? 'unknown'}): ${outcome.error ?? 'missing evidence'}`,
      )
    }
    return JSON.stringify(terminal, null, 2)
  } catch (error) {
    await rejectPreparedPlans(
      bridge, [prepared],
      `execution did not start or did not verify: ${error instanceof Error ? error.message : String(error)}`,
      execution.signal,
    )
    throw error
  }
}

function toolDefinition(tool, bridge, toolGuard, approvalByToken, bindingByToken, operatorId) {
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
      additionalProperties: false,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({
      card: 'generic',
      title: tool.name,
      kind: tool.requires_approval ? 'write' : 'read',
      rawInput: JSON.stringify(args),
    }),
    async execute(args, execution) {
      if (tool.requires_approval) {
        const binding = bindingByToken.get(execution.token)
        const requestId = approvalByToken.get(execution.token)
        if (
          binding?.kind !== 'single'
          || binding.prepared?.plan?.tool_name !== tool.name
          || requestId === undefined
        ) {
          await rejectPreparedPlans(
            bridge, binding?.prepared ? [binding.prepared] : [],
            'plan-bound tool grant validation failed', execution.signal,
          )
          throw new Error(`${tool.name} requires a current plan-bound NetOpYu durable-HITL grant`)
        }
        // A duplicate invocation racing the legitimate one must not reject
        // their shared prepared plan. The atomic grant consume decides the
        // winner; only malformed/mismatched bindings above can close a plan.
        if (!toolGuard.consume(execution.token, tool.name, binding.bindingHash)) {
          throw new Error(`${tool.name} requires a current plan-bound NetOpYu durable-HITL grant`)
        }
        return executePreparedPlan(bridge, binding.prepared, requestId, operatorId, execution)
      }
      const payload = await callBridge({
        ...bridge,
        command: 'invoke',
        tool: tool.name,
        args,
        signal: execution.signal,
        allowDestructive: false,
        correlationId: execution.callId,
        accessContext: observationAccessContext(
          execution, operatorId, bridge.profile,
        ),
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

function hitlDefinitions(manifest, bridge, toolGuard, approvalByToken, bindingByToken, hitlStore, operatorId) {
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
      const binding = bindingByToken.get(execution.token)
      if (binding?.kind !== 'recovery' || !toolGuard.consume(execution.token, resume.name, binding.bindingHash)) {
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
        const result = await executePreparedPlan(
          bridge, binding.prepared, recoveryRequestId, operatorId, execution, ':recovery',
        )
        hitlStore.finishRecovery(original.id, recoveryRequestId, false)
        return result
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
      const binding = bindingByToken.get(execution.token)
      if (binding?.kind !== 'batch' || !toolGuard.consume(execution.token, batch.name, binding.bindingHash)) {
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
          const result = await executePreparedPlan(
            bridge, binding.prepared[index], requestId, operatorId, execution, `:${index}`,
          )
          hitlStore.finishBatchItem(requestId, index, result, undefined)
          results.push({ index, tool_name: operation.tool_name, status: 'completed', result })
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
  const workflowSkills = new Map(
    skillManifest.skills
      .filter(skill => skill.network_workflow !== undefined)
      .map(skill => [skill.name, skill.network_workflow]),
  )
  const approvalByToken = new Map()
  const bindingByToken = new Map()
  const startedSkillInvocations = new Set()
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

  for (const tool of manifest.tools) {
    ctx.tools.register(toolDefinition(tool, bridge, toolGuard, approvalByToken, bindingByToken, operatorId))
  }
  for (const tool of scopedServiceDefinitions(memoryService, capabilityService)) ctx.tools.register(tool)
  for (const tool of a2aToolDefinitions(
    ctx, a2aProvider, bridge, peerUrls, hitlStore, toolGuard, bindingByToken,
  )) ctx.tools.register(tool)
  const hitlTools = hitlDefinitions(
    manifest, bridge, toolGuard, approvalByToken, bindingByToken, hitlStore, operatorId,
  )
  for (const tool of hitlTools) ctx.tools.register(tool)
  ctx.tools.register(trajectoryDefinition(hitlStore))

  ctx.on('session/event', (session, event) => {
    hitlStore.recordTrajectory(session.id, `session:${String(event.type)}`, {
      seq: event.seq,
      data_keys: event.data && typeof event.data === 'object' ? Object.keys(event.data).sort() : [],
    })
  })

  ctx.on('agent/pre-step', async ({ agent, signal }, next) => {
    const decision = await next()
    if (decision.kind !== 'enter' || !Array.isArray(decision.messages)) return decision
    const invoked = decision.messages.filter(message => (
      message?.source?.kind === 'skill-invocation'
      && workflowSkills.has(message.source.name)
    ))
    const pending = invoked.filter(message => {
      const key = `${String(agent.session.id)}:${String(message.id ?? message.source.name)}`
      return !startedSkillInvocations.has(key)
    })
    if (pending.length > 1) {
      throw new Error('multiple mutating Network Runtime skills cannot be activated in one step')
    }
    for (const message of pending) {
      const key = `${String(agent.session.id)}:${String(message.id ?? message.source.name)}`
      await callBridge({
        ...bridge,
        command: 'workflow-start',
        args: { session_id: String(agent.session.id), skill_name: message.source.name },
        signal,
      })
      startedSkillInvocations.add(key)
    }
    return decision
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
    if (execution.name === 'skill' && workflowSkills.has(execution.arguments?.name)) {
      if (sessionId === undefined) {
        return { kind: 'deny', reason: 'Network Runtime workflow skill requires a DSH session' }
      }
      await callBridge({
        ...bridge,
        command: 'workflow-start',
        args: { session_id: String(sessionId), skill_name: execution.arguments.name },
        signal: execution.signal,
        correlationId: execution.callId,
      })
      return next()
    }
    const tool = manifest.tools.find(candidate => candidate.name === execution.name)
    const hitlAction = {
      netopyu_hitl_resume: 'recovery',
      netopyu_hitl_batch: 'batch',
      netopyu_a2a_hitl_resume: 'remote-hitl-resume',
    }[execution.name]
    if (tool?.requires_approval !== true && hitlAction === undefined) return next()
    const actionType = hitlAction ?? tool.action_type
    let binding
    let reason
    const provisionalPrepared = []
    try {
      if (tool?.requires_approval === true) {
        const prepared = await callBridge({
          ...bridge,
          command: 'runtime-prepare',
          tool: tool.name,
          args: execution.arguments,
          signal: execution.signal,
          correlationId: execution.callId,
          sessionId: sessionId === undefined ? undefined : String(sessionId),
          l0SkillId: tool.l0_skill_id,
        })
        if (prepared.status !== 'plan_ready') {
          return { kind: 'deny', reason: `Network Runtime ${preparationFailure(prepared)}` }
        }
        provisionalPrepared.push(prepared)
        binding = { kind: 'single', prepared, bindingHash: prepared.plan.plan_hash }
        reason = prepared.approval_summary
      } else if (hitlAction === 'recovery') {
        const original = hitlStore.recoverable(execution.arguments.request_id)
        if (original === undefined) {
          return { kind: 'deny', reason: `recoverable NetOpYu request not found: ${execution.arguments.request_id}` }
        }
        const originalTool = manifest.tools.find(candidate => candidate.name === original.tool_name)
        if (originalTool?.requires_approval !== true) {
          return { kind: 'deny', reason: 'recovery target is not an enabled approval-gated tool' }
        }
        const replayArguments = execution.arguments.arguments ?? original.arguments
        if (execution.arguments.arguments !== undefined) {
          validateEditableReplacement(originalTool, original.arguments, replayArguments)
        }
        const prepared = await callBridge({
          ...bridge,
          command: 'runtime-prepare',
          tool: originalTool.name,
          args: replayArguments,
          signal: execution.signal,
          correlationId: execution.callId,
          sessionId: sessionId === undefined ? undefined : String(sessionId),
          l0SkillId: originalTool.l0_skill_id,
        })
        if (prepared.status !== 'plan_ready') {
          return { kind: 'deny', reason: `Network Runtime recovery ${preparationFailure(prepared)}` }
        }
        provisionalPrepared.push(prepared)
        binding = { kind: 'recovery', prepared, bindingHash: prepared.plan.plan_hash }
        reason = `Recovery resubmission with fresh approval.\n${prepared.approval_summary}`
      } else if (hitlAction === 'batch') {
        const operations = execution.arguments.operations
        if (!Array.isArray(operations) || operations.length < 1 || operations.length > 50) {
          return { kind: 'deny', reason: 'batch operations must contain 1-50 items' }
        }
        const prepared = []
        for (const [index, operation] of operations.entries()) {
          assertArgumentsObject(operation?.arguments, `operations[${index}].arguments`)
          const operationTool = manifest.tools.find(candidate => candidate.name === operation.tool_name)
          if (operationTool?.requires_approval !== true) {
            throw new Error(`operations[${index}] is not an enabled approval-gated tool`)
          }
          const value = await callBridge({
            ...bridge,
            command: 'runtime-prepare',
            tool: operationTool.name,
            args: operation.arguments,
            signal: execution.signal,
            correlationId: `${String(execution.callId ?? 'batch')}:prepare:${index}`,
            sessionId: sessionId === undefined ? undefined : String(sessionId),
            l0SkillId: operationTool.l0_skill_id,
          })
          if (value.status !== 'plan_ready') {
            throw new Error(`operations[${index}] ${preparationFailure(value)}`)
          }
          provisionalPrepared.push(value)
          prepared.push(value)
        }
        binding = {
          kind: 'batch',
          prepared,
          bindingHash: bindingHash(prepared.map(value => value.plan.plan_hash)),
        }
        reason = [
          `Network Runtime batch (${execution.arguments.policy ?? 'best_effort'}), ${prepared.length} immutable plans:`,
          ...prepared.map((value, index) => `[${index}] ${value.approval_summary}`),
        ].join('\n')
      } else if (hitlAction === 'remote-hitl-resume') {
        const continuation = hitlStore.a2aContinuation(execution.arguments.continuation_id)
        if (continuation === undefined) {
          return { kind: 'deny', reason: `remote A2A continuation not found: ${execution.arguments.continuation_id}` }
        }
        const approval = continuation.request?.remote_approval
        if (approval?.kind === 'network-l0-plan' && typeof approval.plan_hash === 'string') {
          binding = {
            kind: 'remote-hitl',
            bindingHash: approval.plan_hash,
            remoteApproval: approval,
          }
          reason = [
            `Remote DC Network L0 plan (${execution.arguments.decision}):`,
            `Plan: ${approval.plan_id}`,
            `Tool: ${approval.tool_name} (risk=${approval.risk_level})`,
            `Arguments: ${JSON.stringify(approval.arguments)}`,
            `L0 Skill: ${approval.l0_skill_id}@${approval.l0_skill_version} (${approval.l0_contract_hash})`,
            `Intent hash: ${approval.intent_hash}`,
            `Verification: ${approval.verification_contract}; rollback: ${approval.rollback_contract}`,
            `Workflow: ${approval.workflow_run_id} (${approval.workflow_template_hash})`,
            `Expires: ${approval.expires_at}`,
            `Plan hash: ${approval.plan_hash}`,
          ].join('\n')
        } else {
          binding = {
            kind: 'remote-hitl',
            bindingHash: bindingHash({ name: execution.name, arguments: execution.arguments }),
          }
          reason = `Legacy remote A2A continuation has no structured Network L0 plan summary; approval is limited to the exact durable continuation.`
        }
      } else {
        binding = {
          kind: 'remote-hitl',
          bindingHash: bindingHash({ name: execution.name, arguments: execution.arguments }),
        }
        reason = `NetOpYu ${actionType} operation requires operator approval; arguments are durably recorded.`
      }
    } catch (error) {
      await rejectPreparedPlans(
        bridge, provisionalPrepared,
        `workflow preparation rejected: ${error instanceof Error ? error.message : String(error)}`,
        execution.signal,
      )
      return { kind: 'deny', reason: `Network Runtime rejected request: ${error instanceof Error ? error.message : String(error)}` }
    }
    const requestId = hitlStore.begin(execution, reason, binding.bindingHash)
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
      await rejectPreparedPlans(
        bridge, provisionalPrepared, `DSH approval outcome: ${outcome}`, execution.signal,
      )
      approvalByToken.delete(execution.token)
      return { kind: 'deny', reason: `NetOpYu operation not approved (${outcome})` }
    }
    bindingByToken.set(execution.token, binding)
    try {
      toolGuard.issue(execution.token, requestId, execution.name, binding.bindingHash)
    } catch (error) {
      await rejectPreparedPlans(
        bridge, provisionalPrepared,
        `tool grant issuance failed: ${error instanceof Error ? error.message : String(error)}`,
        execution.signal,
      )
      approvalByToken.delete(execution.token)
      bindingByToken.delete(execution.token)
      hitlStore.completed(requestId, true)
      return {
        kind: 'deny',
        reason: `Network Runtime could not issue the one-shot grant: ${error instanceof Error ? error.message : String(error)}`,
      }
    }
    try {
      return await next()
    } catch (error) {
      await rejectPreparedPlans(
        bridge, provisionalPrepared, 'downstream pre-execute hook failed', execution.signal,
      )
      toolGuard.revoke(execution.token, 'downstream pre-execute hook failed')
      approvalByToken.delete(execution.token)
      bindingByToken.delete(execution.token)
      hitlStore.completed(requestId, true)
      throw error
    }
  })

  ctx.on('tools/post-execute', async (execution, result, next) => {
    const sessionId = execution.agent?.session?.id
    const tool = manifest.tools.find(candidate => candidate.name === execution.name)
    if (sessionId === undefined || tool === undefined) return next()
    try {
      await callBridge({
        ...bridge,
        command: 'workflow-observe',
        args: {
          session_id: String(sessionId),
          tool_name: execution.name,
          tool_arguments: execution.arguments,
          result: result.isError === true ? '' : String(result.value ?? ''),
          success: result.isError !== true,
        },
        signal: execution.signal,
        correlationId: execution.callId,
      })
    } catch {
      // The network operation's own journal remains authoritative. A missing
      // workflow observation cannot turn a verified device outcome into a
      // failure; it will fail closed if a later guarded step needs the fact.
    }
    return next()
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
    bindingByToken.delete(execution.token)
    toolGuard.revoke(execution.token)
    hitlStore.completed(requestId, result.isError === true)
  })
}
