// Narrow opt-in DSH entry: six host-bound tools; generate and native delivery are distinct.
// It does not register the legacy provider, filesystem, shell or write tools.
import { governedHybridDefinitions } from './index.js'
import { resolvePython, callBridge } from './bridge.js'

export const name = 'netopyu-hybrid-local'
export const inject = ['tools']

export async function apply(ctx) {
  if (!process.env.NETOPYU_ROOT || !process.env.NETOPYU_HYBRID_HOST_PROFILE) {
    throw new Error('operator project root and local hybrid profile required')
  }
  const bridge = { projectRoot: process.env.NETOPYU_ROOT,
    python: await resolvePython(process.env.NETOPYU_ROOT), profile: 'lan' }
  const hostSchema = await callBridge({ ...bridge, command: 'hybrid-describe' })
  const terminalDelivery = process.env.NETOPYU_HYBRID_TERMINAL_DELIVERY === '1'
  for (const definition of governedHybridDefinitions(bridge, hostSchema, { terminalDelivery })) ctx.tools.register(definition)
}
