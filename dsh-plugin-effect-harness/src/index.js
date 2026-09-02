import { execFile } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { promisify } from 'node:util'

export const name = 'effect-harness-evaluation'
export const inject = ['tools', 'skills']
export const contractVersion = 'effect-runtime.io/dsh-agent-effect-evaluation/v1'

const runFile = promisify(execFile)
const runtimeTerminals = new Set([
  'verified_success',
  'rollback_verified',
  'manual_intervention_required',
  'rejected',
  'clarification_required',
])

function requiredEnv(name) {
  const value = process.env[name]
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`missing required environment value ${name}`)
  }
  return value
}

function skillBody(raw) {
  if (!raw.startsWith('---\n')) throw new Error('evaluation Skill frontmatter is missing')
  const boundary = raw.indexOf('\n---\n', 4)
  if (boundary < 0) throw new Error('evaluation Skill frontmatter is not terminated')
  const body = raw.slice(boundary + 5).trim()
  if (body.length === 0) throw new Error('evaluation Skill body is empty')
  return body
}

function toolSchema(role) {
  if (role === 'get') {
    return {
      type: 'object', additionalProperties: false, required: ['entity_id'],
      properties: { entity_id: { type: 'string', minLength: 1 } },
    }
  }
  if (role === 'validate') {
    return {
      type: 'object', additionalProperties: false,
      required: ['entity_id', 'desired_value'],
      properties: {
        entity_id: { type: 'string', minLength: 1 },
        desired_value: { type: 'string', minLength: 1 },
      },
    }
  }
  if (role === 'apply') {
    return {
      type: 'object', additionalProperties: false,
      required: ['entity_id', 'desired_value', 'expected_revision', 'change_id', 'reason'],
      properties: {
        entity_id: { type: 'string', minLength: 1 },
        desired_value: { type: 'string', minLength: 1 },
        expected_revision: { type: 'integer', minimum: 1 },
        change_id: { type: 'string', minLength: 1 },
        reason: { type: 'string', minLength: 1 },
      },
    }
  }
  return {
    type: 'object', additionalProperties: false,
    required: ['entity_id', 'approved_preflight'],
    properties: {
      entity_id: { type: 'string', minLength: 1 },
      approved_preflight: {
        type: 'object', additionalProperties: false, required: ['facts'],
        description: 'The immutable pre-change observation, wrapped under facts.',
        properties: {
          facts: {
            type: 'object', additionalProperties: true,
            required: ['entity_id', 'value', 'revision'],
            properties: {
              entity_id: { type: 'string' },
              value: { type: 'string' },
              revision: { type: 'integer' },
            },
          },
        },
      },
    },
  }
}

async function invoke(tool, args) {
  const python = requiredEnv('NETOPYU_HARNESS_PYTHON')
  const project = requiredEnv('NETOPYU_HARNESS_PROJECT_ROOT')
  const options = [
    '-m', 'evaluation.harness_effect_tool',
    '--context', requiredEnv('NETOPYU_HARNESS_CONTEXT'),
    '--store', requiredEnv('NETOPYU_HARNESS_STORE'),
    '--journal', requiredEnv('NETOPYU_HARNESS_JOURNAL'),
    '--trace', requiredEnv('NETOPYU_HARNESS_TOOL_TRACE'),
    '--tool', tool,
    '--arguments', JSON.stringify(args),
  ]
  const { stdout } = await runFile(python, options, {
    cwd: project,
    env: process.env,
    timeout: 120000,
    maxBuffer: 2 * 1024 * 1024,
    windowsHide: true,
  })
  const text = stdout.trim()
  const result = JSON.parse(text)
  if (typeof result !== 'object' || result === null || Array.isArray(result)) {
    throw new Error('tool adapter returned a non-object result')
  }
  return JSON.stringify(result)
}

function definition(name, role, description) {
  return {
    name,
    description,
    parameters: toolSchema(role),
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({
      card: 'generic',
      title: description,
      kind: role === 'get' || role === 'validate' ? 'read' : 'write',
      rawInput: JSON.stringify(args),
    }),
    execute: async (args, exec) => {
      const value = await invoke(name, args)
      const result = JSON.parse(value)
      if (
        (result.execution === 'l0_runtime' || result.execution === 'safe_stop') &&
        runtimeTerminals.has(result.terminal)
      ) {
        exec.concludeTurn()
      }
      return value
    },
  }
}

export async function apply(ctx) {
  const skillPath = resolve(requiredEnv('NETOPYU_HARNESS_SKILL_PATH'))
  const skillName = requiredEnv('NETOPYU_HARNESS_SKILL_NAME')
  const domain = requiredEnv('NETOPYU_HARNESS_DOMAIN')
  const content = skillBody(await readFile(skillPath, 'utf8'))
  ctx.skills.register({
    name: skillName,
    description: `Safely execute the reviewed ${domain} state-change workflow.`,
    content,
    path: skillPath,
    resourceBase: { kind: 'directory', path: dirname(skillPath) },
    metadata: { scope: 'controlled-evaluation', domain },
    source: 'netopyu-effect-harness-evaluation',
    provider: 'effect-harness-evaluation',
    invocation: { modelInvocable: true, userInvocable: false },
  })
  ctx.tools.register(definition(
    `${domain}_get_state`, 'get',
    `Read the current ${domain} entity state and revision.`,
  ))
  ctx.tools.register(definition(
    `${domain}_validate_change`, 'validate',
    `Validate a proposed ${domain} state value without writing.`,
  ))
  ctx.tools.register(definition(
    `${domain}_apply_change`, 'apply',
    `Apply one approved ${domain} state change using an expected revision.`,
  ))
  ctx.tools.register(definition(
    `${domain}_restore_state`, 'restore',
    `Restore the approved pre-change ${domain} snapshot after a failed verification.`,
  ))
}
