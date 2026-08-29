import assert from 'node:assert/strict'

import { apply, contractVersion, toolNames } from '../src/index.js'

const registered = []
const skillDigest = `sha256:${'a'.repeat(64)}`
await apply({ tools: { register(value) { registered.push(value) } } }, {
  preloadedSkillDigest: skillDigest,
})
assert.deepEqual(registered.map(item => item.name), toolNames)
const selection = {
  target: 'restart-service',
  arguments: { environment: 'prod', service: 'crm' },
  confidence: 0.9,
  reason_code: 'explicit_restart',
}
const receipt = JSON.parse(await registered[0].execute(selection))
assert.equal(receipt.accepted, true)
assert.equal(receipt.contract, contractVersion)
assert.equal(receipt.preloadedSkillDigest, skillDigest)
assert.match(receipt.digest, /^sha256:[0-9a-f]{64}$/)
await assert.rejects(
  apply({ tools: { register() {} } }, { preloadedSkillDigest: 'mutable' }),
  /reviewed sha256/,
)
console.log('P1.8-C protocol controller smoke: PASS')
