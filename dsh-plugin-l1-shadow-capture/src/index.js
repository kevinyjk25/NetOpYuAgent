import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const name = 'l1-shadow-capture'
export const inject = ['tools', 'skills']
export const contractVersion = 'netopyu.io/l1-shadow-capture/v1'

const sourceDirectory = dirname(fileURLToPath(import.meta.url))
const skillPath = resolve(sourceDirectory, '..', 'skills', 'l1-decision-capture', 'SKILL.md')
const schemaVersion = 'netopyu.io/l1-decision/v1'
const actions = ['select_skill', 'select_tool', 'clarify', 'refuse', 'out_of_scope']
const decisionKeys = [
  'action', 'apiVersion', 'arguments', 'confidence', 'missing_fields',
  'reason_code', 'target', 'workflow',
]

function skillBody(raw) {
  if (!raw.startsWith('---\n')) throw new Error('capture Skill frontmatter is missing')
  const boundary = raw.indexOf('\n---\n', 4)
  if (boundary < 0) throw new Error('capture Skill frontmatter is not terminated')
  const body = raw.slice(boundary + 5).trim()
  if (body.length === 0) throw new Error('capture Skill body is empty')
  return body
}

function isPlainObject(value) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
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

function validateStringArray(value, field, maximum) {
  if (!Array.isArray(value) || value.length > maximum) throw new Error(`${field} is invalid`)
  if (value.some(item => typeof item !== 'string' || !/^[A-Za-z0-9_.:-]{1,128}$/.test(item))) {
    throw new Error(`${field} contains an invalid identifier`)
  }
  if (field === 'missing_fields' && new Set(value).size !== value.length) {
    throw new Error('missing_fields contains duplicates')
  }
}

export function validateDecision(value) {
  if (!isPlainObject(value)) throw new Error('decision must be an object')
  if (Object.keys(value).sort().join('\n') !== [...decisionKeys].sort().join('\n')) {
    throw new Error('decision fields do not match the capture contract')
  }
  if (value.apiVersion !== schemaVersion || !actions.includes(value.action)) {
    throw new Error('decision version or action is invalid')
  }
  if (value.target !== null && (
    typeof value.target !== 'string' || !/^[A-Za-z0-9_.:-]{1,128}$/.test(value.target)
  )) throw new Error('target is invalid')
  if (!isPlainObject(value.arguments)) throw new Error('arguments must be an object')
  validateJson(value.arguments)
  if (Buffer.byteLength(JSON.stringify(value.arguments), 'utf8') > 16384) {
    throw new Error('arguments exceed 16 KiB')
  }
  validateStringArray(value.missing_fields, 'missing_fields', 16)
  validateStringArray(value.workflow, 'workflow', 16)
  if (typeof value.confidence !== 'number' || value.confidence < 0 || value.confidence > 1) {
    throw new Error('confidence is invalid')
  }
  if (typeof value.reason_code !== 'string' || value.reason_code.trim().length < 1 || value.reason_code.length > 80) {
    throw new Error('reason_code is invalid')
  }
  if (value.action === 'select_skill' || value.action === 'select_tool') {
    if (value.target === null || value.missing_fields.length > 0) {
      throw new Error('selection requires target and no missing fields')
    }
  } else if (value.action === 'clarify') {
    if (value.missing_fields.length === 0 || value.workflow.length > 0) {
      throw new Error('clarification requires missing fields and no workflow')
    }
  } else if (
    value.target !== null || Object.keys(value.arguments).length > 0
    || value.missing_fields.length > 0 || value.workflow.length > 0
  ) {
    throw new Error('refusal/out-of-scope cannot carry executable content')
  }
  return value
}

function captureDefinition() {
  return {
    name: 'submit_l1_decision',
    description: 'Submit exactly one evaluation-only L1 proposal after loading l1-decision-capture. This tool records no external effect and cannot reach Runtime or a Provider.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: decisionKeys,
      properties: {
        apiVersion: { type: 'string', const: schemaVersion },
        action: { type: 'string', enum: actions },
        target: { oneOf: [{ type: 'string' }, { type: 'null' }] },
        arguments: { type: 'object', additionalProperties: true },
        missing_fields: { type: 'array', items: { type: 'string' } },
        workflow: { type: 'array', items: { type: 'string' } },
        confidence: { type: 'number' },
        reason_code: { type: 'string' },
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({
      card: 'generic',
      title: 'Capture non-executing L1 proposal',
      kind: 'read',
      rawInput: JSON.stringify(args),
    }),
    async execute(args) {
      const decision = validateDecision(args)
      const canonical = canonicalJson(decision)
      const digest = `sha256:${createHash('sha256').update(canonical).digest('hex')}`
      return JSON.stringify({ accepted: true, contract: contractVersion, digest })
    },
  }
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

export async function apply(ctx) {
  const content = skillBody(await readFile(skillPath, 'utf8'))
  ctx.skills.register({
    name: 'l1-decision-capture',
    description: 'Route one bounded NetOpYu request into a non-executing L1 proposal and submit it through the capture-only tool.',
    content,
    path: skillPath,
    resourceBase: { kind: 'directory', path: dirname(skillPath) },
    metadata: { scope: 'evaluation-only', effect: 'none' },
    source: 'netopyu-p1.8-b2',
    provider: 'l1-shadow-capture',
    invocation: { modelInvocable: true, userInvocable: false },
  })
  ctx.tools.register(captureDefinition())
}
