import assert from 'node:assert/strict'

import {
  apply,
  contractVersion,
  validateDecision,
} from '../src/index.js'

const registered = { skill: null, tool: null }
await apply({
  skills: { register(value) { registered.skill = value } },
  tools: { register(value) { registered.tool = value } },
})

assert.equal(registered.skill.name, 'l1-decision-capture')
assert.equal(registered.skill.invocation.modelInvocable, true)
assert.equal(registered.skill.invocation.userInvocable, false)
assert.equal(registered.tool.name, 'submit_l1_decision')

const valid = {
  apiVersion: 'netopyu.io/l1-decision/v1',
  action: 'select_skill',
  target: 'restart-service',
  arguments: { environment: 'prod', service: 'crm' },
  missing_fields: [],
  workflow: [],
  confidence: 0.9,
  reason_code: 'explicit_restart',
}
assert.equal(validateDecision(valid), valid)
const receipt = JSON.parse(await registered.tool.execute(valid))
assert.deepEqual(Object.keys(receipt).sort(), ['accepted', 'contract', 'digest'])
assert.equal(receipt.accepted, true)
assert.equal(receipt.contract, contractVersion)
assert.match(receipt.digest, /^sha256:[0-9a-f]{64}$/)

assert.throws(
  () => validateDecision({ ...valid, apiVersion: 'v1' }),
  /version or action/,
)
assert.throws(
  () => validateDecision({ ...valid, unexpected: true }),
  /fields do not match/,
)
assert.throws(
  () => validateDecision({
    ...valid,
    action: 'refuse',
    target: 'restart-service',
  }),
  /cannot carry executable content/,
)

console.log('P1.8-B2 capture plugin smoke: PASS')
