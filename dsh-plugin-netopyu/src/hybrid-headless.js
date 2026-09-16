// Optional frontend for host-terminal delivery. The installed DSH Agent/Session
// loop remains authoritative; never fabricate an assistant/model event.
import { randomUUID } from 'node:crypto'
import { createRequire } from 'node:module'
import { isAbsolute } from 'node:path'
import { pathToFileURL } from 'node:url'
import { summarizeHybridTurn } from './hybrid-terminal.js'

export const name = 'netopyu-hybrid-headless'
export const inject = ['agentDefaultModel', 'agents', 'sessions']

export async function runHybridHeadless(ctx, task, io, api) {
  await ctx.get('loader')?.await()
  const selection = ctx.agentDefaultModel.currentSelection()
  const { agent } = await ctx.agents.create({
    sessionId: api.SessionId(`session-${randomUUID()}`), meta: { cwd: process.cwd() },
    agentOptions: { provider: selection.provider, model: selection.model },
    setup: agentCtx => { api.installModelSelection(agentCtx, { current: selection, assembled: undefined }) },
  })
  await agent.whenIdle()
  const firstSeq = agent.session.seq
  agent.followup(api.createUserMessage({ content: [{ type: 'text', text: task }], source: { kind: 'user' } }))
  await agent.whenIdle()
  await ctx.sessions.flush(agent.session)
  const outcome = summarizeHybridTurn(agent.session.events, firstSeq)
  io.stdout.write(outcome.text + '\n')
  if (outcome.reason?.kind === 'error') io.stderr.write(`dsh: ${outcome.reason.error.code}: ${outcome.reason.error.message}\n`)
  // Process completion is not semantic approval. The printed host status says so.
  io.exit(outcome.reason?.kind === 'completed' ? 0 : 1)
}

export function apply(ctx, config) {
  const entry = process.env.NETOPYU_DSH_HEADLESS_ENTRY
  if (process.env.NETOPYU_HYBRID_TERMINAL_DELIVERY !== '1' || !entry || !isAbsolute(entry)) {
    throw new Error('explicit host-terminal mode and installed DSH headless entry path required')
  }
  const exit = ctx.get('appExit')
  if (typeof exit !== 'function') throw new Error('DSH launcher appExit required')
  // Resolve the SAME installed public DSH packages as its official frontend.
  // Operator configuration only; no package path comes from a model or Skill.
  const require = createRequire(entry)
  Promise.all(['@deepseek-ai/dsh-agent', '@deepseek-ai/dsh-llm', '@deepseek-ai/dsh-session']
    .map(name => import(pathToFileURL(require.resolve(name)).href)))
    .then(([agent, llm, session]) => runHybridHeadless(ctx, config.task,
      { stdout: process.stdout, stderr: process.stderr, exit }, { ...agent, ...llm, ...session }))
    .catch(error => { process.stderr.write(`dsh host-terminal: ${error.message}\n`); exit(1) })
}
