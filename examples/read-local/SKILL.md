---
name: inspect-inventory-device
description: Read one device from a local inventory snapshot, without configuring any network device.
---

# Inspect one inventory device

Use `read_inventory_device(device_id)` to read a device from the host-configured
local inventory snapshot. `device_id` is a required string. Report the returned
`device_id`, `site` and `status` as snapshot data, never as real-time device health.
Ask for the device ID when it is absent or ambiguous. Do not infer an ID.

The tool's source schemas are provided separately. The host, not the user or
model, selects the inventory file and authenticates the local caller. Reading
requires the `inventory:read` scope, the `network-reader` role, internal data
clearance and the matching `device_id:<id>` resource scope. Do not request
wildcard/system access. Do not change the file, execute shell commands, connect
to devices, repair configurations or expose undeclared fields. Unknown devices
and invalid responses are errors; do not fabricate successful results.

This is a newly authored local integration example, not a public unseen Skill.
