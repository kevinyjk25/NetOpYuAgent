import { execFile } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { promisify } from 'node:util'

export const name = 'public-skill-harness-evaluation'
export const inject = ['tools', 'skills']
export const contractVersion = 'effect-runtime.io/dsh-public-skill-evaluation/v1'

const runFile = promisify(execFile)
const runtimeTerminals = new Set([
  'verified_success', 'rollback_verified', 'manual_intervention_required',
  'rejected', 'clarification_required',
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

async function invoke(tool, args) {
  const options = [
    '-m', 'evaluation.public_skill_harness_tool',
    '--context', requiredEnv('NETOPYU_PUBLIC_CONTEXT'),
    '--store', requiredEnv('NETOPYU_PUBLIC_STORE'),
    '--trace', requiredEnv('NETOPYU_PUBLIC_TRACE'),
    '--tool', tool,
    '--arguments', JSON.stringify(args),
  ]
  const { stdout } = await runFile(requiredEnv('NETOPYU_HARNESS_PYTHON'), options, {
    cwd: requiredEnv('NETOPYU_HARNESS_PROJECT_ROOT'), env: process.env,
    timeout: 120000, maxBuffer: 2 * 1024 * 1024, windowsHide: true,
  })
  const result = JSON.parse(stdout.trim())
  if (typeof result !== 'object' || result === null || Array.isArray(result)) {
    throw new Error('public Tool adapter returned a non-object result')
  }
  return JSON.stringify(result)
}

function definition(capability) {
  const write = capability.actionType !== 'read_only'
  return {
    name: capability.toolName,
    description: capability.description,
    parameters: capability.inputSchema,
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    presentCall: args => ({
      card: 'generic', title: capability.description,
      kind: write ? 'write' : 'read', rawInput: JSON.stringify(args),
    }),
    execute: async (args, exec) => {
      const value = await invoke(capability.toolName, args)
      const result = JSON.parse(value)
      if (
        (result.execution === 'l0_runtime' || result.execution === 'safe_stop') &&
        runtimeTerminals.has(result.terminal)
      ) exec.concludeTurn()
      return value
    },
  }
}

export async function apply(ctx) {
  const skillPath = resolve(requiredEnv('NETOPYU_PUBLIC_SKILL_PATH'))
  const skillName = requiredEnv('NETOPYU_PUBLIC_SKILL_NAME')
  const catalog = JSON.parse(await readFile(resolve(requiredEnv('NETOPYU_PUBLIC_CATALOG')), 'utf8'))
  if (
    catalog.apiVersion !== 'effect-runtime.io/public-skill-tool-catalog/v2' ||
    !Array.isArray(catalog.capabilities)
  ) throw new Error('public Tool Catalog v2 is required')
  const content = skillBody(await readFile(skillPath, 'utf8'))
  ctx.skills.register({
    name: skillName,
    description: 'Execute the sealed public Skill study case with declared Tools only.',
    content, path: skillPath,
    resourceBase: { kind: 'directory', path: dirname(skillPath) },
    metadata: { scope: 'controlled-public-skill-evaluation' },
    source: 'netopyu-public-skill-harness-evaluation',
    provider: 'public-skill-harness-evaluation',
    invocation: { modelInvocable: true, userInvocable: false },
  })
  for (const capability of catalog.capabilities) {
    ctx.tools.register(definition(capability))
  }
}
