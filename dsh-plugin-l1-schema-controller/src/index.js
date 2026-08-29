import { createHash } from 'node:crypto'

export const name = 'l1-schema-controller'
export const inject = ['tools']
export const contractVersion = 'netopyu.io/l1-schema-controller/v1'

const digestPattern = /^sha256:[0-9a-f]{64}$/
const identifierPattern = /^[A-Za-z0-9_.:-]{1,128}$/

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

function digest(value) {
  return `sha256:${createHash('sha256').update(canonicalJson(value)).digest('hex')}`
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
    if (!identifierPattern.test(key)) throw new Error('argument key is invalid')
    validateJson(item, depth + 1)
  }
}

function validateCandidate(raw, index) {
  if (!isPlainObject(raw)) throw new Error(`candidate ${index} must be an object`)
  const expected = [
    'description', 'kind', 'parameters', 'profile', 'required_parameters',
    'requires_approval', 'risk_level', 'target', 'workflow_hint',
  ]
  if (Object.keys(raw).sort().join('\n') !== expected.sort().join('\n')) {
    throw new Error(`candidate ${index} fields differ from the public contract`)
  }
  if (!identifierPattern.test(raw.target) || !['skill', 'tool'].includes(raw.kind)) {
    throw new Error(`candidate ${index} identity is invalid`)
  }
  if (!['lan', 'dc', 'wan'].includes(raw.profile) || typeof raw.description !== 'string') {
    throw new Error(`candidate ${index} profile or description is invalid`)
  }
  if (!isPlainObject(raw.parameters) || Object.keys(raw.parameters).length > 32) {
    throw new Error(`candidate ${index} parameters are invalid`)
  }
  for (const [key, description] of Object.entries(raw.parameters)) {
    if (!identifierPattern.test(key) || typeof description !== 'string') {
      throw new Error(`candidate ${index} parameter contract is invalid`)
    }
  }
  if (!Array.isArray(raw.required_parameters) || !Array.isArray(raw.workflow_hint)) {
    throw new Error(`candidate ${index} required/workflow fields are invalid`)
  }
  if (!raw.required_parameters.every(key => Object.hasOwn(raw.parameters, key))) {
    throw new Error(`candidate ${index} required parameters escape its Schema`)
  }
  if (typeof raw.risk_level !== 'string' || typeof raw.requires_approval !== 'boolean') {
    throw new Error(`candidate ${index} risk contract is invalid`)
  }
  return raw
}

function receiptDefinition({ toolName, description, candidate, candidateIndex, contractDigest, skillDigest }) {
  const properties = Object.fromEntries(Object.entries(candidate.parameters).map(
    ([key, value]) => [key, { description: String(value).slice(0, 1000) }],
  ))
  return {
    name: toolName,
    description,
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: [],
      properties,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({
      card: 'generic', title: `Capture candidate ${candidateIndex}`, kind: 'read',
      rawInput: JSON.stringify(args),
    }),
    async execute(args) {
      if (!isPlainObject(args)) throw new Error('candidate arguments must be an object')
      if (!Object.keys(args).every(key => Object.hasOwn(candidate.parameters, key))) {
        throw new Error('arguments contain fields outside the candidate Schema')
      }
      validateJson(args)
      if (Buffer.byteLength(JSON.stringify(args), 'utf8') > 16384) {
        throw new Error('arguments exceed 16 KiB')
      }
      const envelope = { tool: toolName, arguments: args, candidateContractDigest: contractDigest }
      return JSON.stringify({
        accepted: true,
        contract: contractVersion,
        digest: digest(envelope),
        candidateContractDigest: contractDigest,
        preloadedSkillDigest: skillDigest,
        candidateIndex,
      })
    },
  }
}

function terminalDefinition(toolName, description, contractDigest, skillDigest) {
  return {
    name: toolName,
    description,
    parameters: {
      type: 'object', additionalProperties: false,
      required: [],
      properties: {},
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({ card: 'generic', title: `Capture ${toolName}`, kind: 'read', rawInput: JSON.stringify(args) }),
    async execute(args) {
      if (!isPlainObject(args) || Object.keys(args).length) {
        throw new Error('terminal proposal accepts no business arguments')
      }
      const envelope = { tool: toolName, arguments: args, candidateContractDigest: contractDigest }
      return JSON.stringify({
        accepted: true,
        contract: contractVersion,
        digest: digest(envelope),
        candidateContractDigest: contractDigest,
        preloadedSkillDigest: skillDigest,
        candidateIndex: null,
      })
    },
  }
}

export async function apply(ctx, config = {}) {
  const skillDigest = config.preloadedSkillDigest
  const contractDigest = config.candidateContractDigest
  if (!digestPattern.test(skillDigest || '') || !digestPattern.test(contractDigest || '')) {
    throw new Error('reviewed Skill and candidate contract digests are required')
  }
  let candidates
  try {
    candidates = JSON.parse(config.candidateContractJson)
  } catch {
    throw new Error('candidateContractJson must be strict JSON')
  }
  if (!Array.isArray(candidates) || candidates.length < 1 || candidates.length > 12) {
    throw new Error('candidate contract must contain 1..12 candidates')
  }
  if (digest(candidates) !== contractDigest) throw new Error('candidate contract digest mismatch')
  const identities = new Set()
  candidates.forEach((raw, index) => {
    const candidate = validateCandidate(raw, index)
    const identity = `${candidate.kind}:${candidate.target}`
    if (identities.has(identity)) throw new Error('candidate identities must be unique')
    identities.add(identity)
    const toolName = `select_candidate_${String(index).padStart(2, '0')}`
    const required = candidate.required_parameters.length
      ? candidate.required_parameters.join(', ')
      : 'none'
    const workflow = candidate.workflow_hint.length
      ? candidate.workflow_hint.join(' -> ')
      : 'none'
    const description = [
      `Select exact ${candidate.kind} candidate ${candidate.target}.`,
      candidate.description.slice(0, 1200),
      `Required business fields: ${required}.`,
      `Controller-owned workflow: ${workflow}.`,
      'Omit missing fields; the deterministic compiler will request them.',
    ].join(' ')
    ctx.tools.register(receiptDefinition({
      toolName, description, candidate, candidateIndex: index,
      contractDigest, skillDigest,
    }))
  })
  ctx.tools.register(terminalDefinition(
    'refuse_l1_request',
    'Refuse unsafe, approval-bypass, credential-disclosure, forged, blind-retry, uncontrolled destructive, or audit-disabling work.',
    contractDigest, skillDigest,
  ))
  ctx.tools.register(terminalDefinition(
    'reject_l1_out_of_scope',
    'Reject a request outside network and service operations.',
    contractDigest, skillDigest,
  ))
}
