import { createHash } from 'node:crypto'

export const name = 'l1-protocol-controller'
export const inject = ['tools']
export const contractVersion = 'netopyu.io/l1-protocol-controller/v1'

export const toolNames = [
  'propose_l1_skill',
  'propose_l1_tool',
  'clarify_l1_request',
  'refuse_l1_request',
  'reject_l1_out_of_scope',
]
const digestPattern = /^sha256:[0-9a-f]{64}$/

function isPlainObject(value) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (isPlainObject(value)) {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    )).join(',')}}`
  }
  return JSON.stringify(value)
}

function validateJson(value, depth = 0) {
  if (depth > 4) throw new Error('arguments exceed four levels')
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    if (typeof value === 'string' && value.length > 2000) throw new Error('argument string is too long')
    return
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('argument number is not finite')
    return
  }
  if (Array.isArray(value)) {
    if (value.length > 32) throw new Error('argument array is too large')
    for (const item of value) validateJson(item, depth + 1)
    return
  }
  if (!isPlainObject(value) || Object.keys(value).length > 32) {
    throw new Error('arguments must be bounded JSON objects')
  }
  for (const [key, item] of Object.entries(value)) {
    if (!/^[A-Za-z0-9_.:-]{1,128}$/.test(key)) throw new Error('argument key is invalid')
    validateJson(item, depth + 1)
  }
}

const confidence = {
  type: 'number', minimum: 0, maximum: 1,
  description: 'Bounded proposal confidence, never authority.',
}
const reasonCode = {
  type: 'string',
  description: 'Short machine-readable reason code.',
}

function validateTypedCall(action, args) {
  if (!isPlainObject(args)) throw new Error('typed proposal must be an object')
  const expected = action === 'refuse' || action === 'out_of_scope'
    ? ['confidence', 'reason_code']
    : ['arguments', 'confidence', 'reason_code', 'target']
  if (Object.keys(args).sort().join('\n') !== expected.sort().join('\n')) {
    throw new Error('typed proposal fields do not match the controller contract')
  }
  if (typeof args.confidence !== 'number' || args.confidence < 0 || args.confidence > 1) {
    throw new Error('confidence is invalid')
  }
  if (typeof args.reason_code !== 'string' || args.reason_code.trim().length < 1 || args.reason_code.length > 80) {
    throw new Error('reason_code is invalid')
  }
  if (action !== 'refuse' && action !== 'out_of_scope') {
    if (typeof args.target !== 'string' || !/^[A-Za-z0-9_.:-]{1,128}$/.test(args.target)) {
      throw new Error('target is invalid')
    }
    if (!isPlainObject(args.arguments)) throw new Error('arguments must be an object')
    validateJson(args.arguments)
    if (Buffer.byteLength(JSON.stringify(args.arguments), 'utf8') > 16384) {
      throw new Error('arguments exceed 16 KiB')
    }
  }
  return args
}

function definition(name, description, action, properties, required, preloadedSkillDigest) {
  return {
    name,
    description,
    parameters: {
      type: 'object',
      additionalProperties: false,
      required,
      properties,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({
      card: 'generic',
      title: `Capture controlled ${action} L1 proposal`,
      kind: 'read',
      rawInput: JSON.stringify(args),
    }),
    async execute(args) {
      const validated = validateTypedCall(action, args)
      const envelope = { tool: name, arguments: validated }
      const digest = `sha256:${createHash('sha256').update(canonicalJson(envelope)).digest('hex')}`
      return JSON.stringify({
        accepted: true,
        contract: contractVersion,
        digest,
        preloadedSkillDigest,
      })
    },
  }
}

export async function apply(ctx, config = {}) {
  const preloadedSkillDigest = config.preloadedSkillDigest
  if (typeof preloadedSkillDigest !== 'string' || !digestPattern.test(preloadedSkillDigest)) {
    throw new Error('preloadedSkillDigest must be a reviewed sha256 digest')
  }
  const target = {
    type: 'string',
    description: 'Exact target copied from one supplied candidate.',
  }
  const args = {
    type: 'object', additionalProperties: true,
    description: 'Only explicit request values under exact candidate parameter keys.',
  }
  ctx.tools.register(definition(
    toolNames[0],
    'Propose one supplied Skill when it covers the complete outcome and all required fields are explicit.',
    'select_skill',
    { target, arguments: args, confidence, reason_code: reasonCode },
    ['target', 'arguments', 'confidence', 'reason_code'],
    preloadedSkillDigest,
  ))
  ctx.tools.register(definition(
    toolNames[1],
    'Propose one supplied primitive Tool when no Skill covers the outcome or the primitive is explicitly requested.',
    'select_tool',
    { target, arguments: args, confidence, reason_code: reasonCode },
    ['target', 'arguments', 'confidence', 'reason_code'],
    preloadedSkillDigest,
  ))
  ctx.tools.register(definition(
    toolNames[2],
    'Request missing required business fields for one exact supplied candidate; never guess or use placeholders.',
    'clarify',
    { target, arguments: args, confidence, reason_code: reasonCode },
    ['target', 'arguments', 'confidence', 'reason_code'],
    preloadedSkillDigest,
  ))
  ctx.tools.register(definition(
    toolNames[3],
    'Refuse unsafe, approval-bypass, forged, guessed, blind-retry, uncontrolled destructive, or audit-disabling work.',
    'refuse',
    { confidence, reason_code: reasonCode },
    ['confidence', 'reason_code'],
    preloadedSkillDigest,
  ))
  ctx.tools.register(definition(
    toolNames[4],
    'Reject a request outside network and service operations.',
    'out_of_scope',
    { confidence, reason_code: reasonCode },
    ['confidence', 'reason_code'],
    preloadedSkillDigest,
  ))
}
