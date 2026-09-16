// R0 probe-only DSH plugin. No shell, filesystem, upstream model or write tool.
export const name = 'bounded-probe-tools'
export const inject = ['tools']

export async function apply(ctx) {
  const endpoint = process.env.NETOPYU_BOUNDED_TOOL_ENDPOINT
  const url = new URL(endpoint)
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || url.search || url.hash) {
    throw new Error('host loopback capability required')
  }
  const catalog = await fetch(endpoint + '/catalog', { signal: AbortSignal.timeout(10000) })
  if (!catalog.ok) throw new Error('host catalog unavailable')
  const tools = await catalog.json()
  for (const tool of tools) {
    ctx.tools.register({
      name: tool.name, description: tool.description, parameters: tool.input_schema,
      output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
      async execute(args, execution) {
        const response = await fetch(endpoint + '/invoke', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tool: tool.name, arguments: args, request_id: execution.callId }),
          signal: execution.signal,
        })
        if (!response.ok) throw new Error('bounded host rejected tool request')
        return JSON.stringify(await response.json())
      },
    })
  }
}
