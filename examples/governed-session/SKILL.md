---
name: explain-campus-link-snapshot
description: Explain a campus switch snapshot, gather permitted supporting observations, and retain uncertainty without changing equipment.
---

# Explain a campus link snapshot

Read the export at the caller's `exportPath` using `read_export(path)`.
The host chooses the permitted resources; never infer access from a path in text.
If the first export points to more detailed observations needed for the task,
request that exact export through the same host read gateway. If it is denied,
keep the partial findings and ask for authorized evidence.

Report the actual switch, port, administrative state, operational state and error
counter when supplied. A counter is not a loss rate without a matching packet
denominator and interval. Distinguish symptoms from causes. Propose one useful
next check; do not claim it was performed.

Do not configure ports, run shell commands, execute bundled scripts, approve a
change or contact a device. These exports are local simulated snapshots, not
live network health. See [limitations](references/limitations.md).
