---
name: replica-inspection
description: Inspect a replica under a normal healthy route or a pre-existing emergency lease.
---
# Replica inspection
Read replica_facts using the caller's replica_key.
The normal route requires BOTH synchronization and absence of quarantine. An existing emergency-read lease independently permits inspection, even when the normal route fails.
When neither route is available, stop unsupported; never mint a lease or remove quarantine.
If a route is available, call replica_details with the same replica_key, then complete the read path.
Tool failures or invalid data block the flow instead of taking a business branch.
