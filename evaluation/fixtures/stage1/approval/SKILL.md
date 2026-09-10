---
name: approved-change-readiness
description: Inspect a requested change and its target device without making changes.
---

# Approved change readiness

Use the change identifier explicitly supplied by the caller to read the change request using get_change_request.
The returned request identifies one device and an approved boolean. If approved is false, finish this read-only inspection without reading the device.
If approved is true, use the returned device ID to call get_device_health.
If health.available is true, finish the read-only inspection. Otherwise hand the evidence to L1 for a human-readable explanation of the unavailable device and suggested investigation; do not execute any change.

Approval here is a business record, not runtime identity or authorization. Host-bound consent, resource scopes and evidence freshness must still be enforced for every actual read.
Do not fabricate a change identifier or replace the device selected by the returned request with a guessed device.
