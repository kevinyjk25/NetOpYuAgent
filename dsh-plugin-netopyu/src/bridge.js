import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { access } from 'node:fs/promises'
import { createConnection } from 'node:net'
import { delimiter, resolve } from 'node:path'
function collectProcess(child, input, signal) {
  return new Promise((resolveResult, reject) => {
    let stdout = ''
    let stderr = ''
    const abort = () => child.kill('SIGTERM')
    signal?.addEventListener('abort', abort, { once: true })
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', chunk => { stdout += chunk })
    child.stderr.on('data', chunk => { stderr += chunk })
    child.on('error', reject)
    child.on('close', code => {
      signal?.removeEventListener('abort', abort)
      if (signal?.aborted) return reject(signal.reason ?? new Error('tool call aborted'))
      if (code !== 0) {
        let detail = stderr.trim()
        try {
          const payload = JSON.parse(stdout)
          detail = payload.error ?? detail
        } catch {
          // Preserve stderr when the Python process could not emit its protocol response.
        }
        return reject(new Error(detail || `NetOpYu bridge exited with code ${code}`))
      }
      try {
        resolveResult(JSON.parse(stdout))
      } catch (error) {
        reject(new Error(`invalid NetOpYu bridge response: ${error.message}`))
      }
    })
    child.stdin.end(input)
  })
}

export async function resolvePython(projectRoot) {
  if (process.env.NETOPYU_PYTHON) return process.env.NETOPYU_PYTHON
  const virtualenvPython = resolve(projectRoot, '.venv', 'bin', 'python')
  try {
    await access(virtualenvPython)
    return virtualenvPython
  } catch {
    return 'python3'
  }
}

export async function callBridge({ projectRoot, python, profile, command, tool, args, signal, includeDestructive = false, allowDestructive = false, correlationId }) {
  const requestId = randomUUID()
  const bridgeCorrelationId = String(correlationId ?? requestId)
  const workerSocket = process.env.NETOPYU_DSH_WORKER_SOCKET
  if (workerSocket) {
    try {
      return await callPersistentWorker(workerSocket, {
        id: requestId, correlation_id: bridgeCorrelationId,
        profile, command, tool, args: args ?? {},
        include_destructive: includeDestructive,
        allow_destructive: allowDestructive,
      }, signal)
    } catch (error) {
      if (!['ENOENT', 'ECONNREFUSED'].includes(error?.code)) throw error
    }
  }
  const commandArgs = ['-m', 'dsh_adapter.cli', command, '--profile', profile]
  if (tool !== undefined) commandArgs.push('--tool', tool)
  if (command === 'manifest' && includeDestructive) commandArgs.push('--include-destructive')
  const child = spawn(python, commandArgs, {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONPATH: [projectRoot, process.env.PYTHONPATH].filter(Boolean).join(delimiter),
      ...(allowDestructive ? { NETOPYU_DSH_ALLOW_DESTRUCTIVE: '1' } : {}),
      NETOPYU_DSH_CORRELATION_ID: bridgeCorrelationId,
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  return collectProcess(child, JSON.stringify(args ?? {}), signal)
}

function callPersistentWorker(socketPath, request, signal) {
  return new Promise((resolveResult, reject) => {
    const connection = createConnection(socketPath)
    let response = ''
    const abort = () => connection.destroy(signal?.reason ?? new Error('tool call aborted'))
    signal?.addEventListener('abort', abort, { once: true })
    connection.setEncoding('utf8')
    connection.on('connect', () => connection.end(`${JSON.stringify(request)}\n`))
    connection.on('data', chunk => {
      response += chunk
      if (response.length > 16 * 1024 * 1024) connection.destroy(new Error('persistent bridge response exceeds 16 MiB'))
    })
    connection.on('error', error => {
      signal?.removeEventListener('abort', abort)
      reject(error)
    })
    connection.on('end', () => {
      signal?.removeEventListener('abort', abort)
      try {
        const envelope = JSON.parse(response)
        if (envelope.id !== request.id) throw new Error('persistent bridge response id mismatch')
        if (envelope.ok !== true) throw new Error(envelope.error ?? 'persistent bridge request failed')
        resolveResult(envelope.payload)
      } catch (error) {
        reject(error)
      }
    })
  })
}


