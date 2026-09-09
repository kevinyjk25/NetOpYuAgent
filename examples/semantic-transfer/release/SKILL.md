---
name: release-window-inspection
description: Inspect release windows only under a present, non-stale permission.
---
# Release window inspection
Read release_decision for the caller's change_id before reading release windows.
The decision must exist, must NOT be stale, and must permit reading the windows. Every qualification is required.
If any qualification fails, stop unsupported without calling read_release_windows.
Otherwise read_release_windows for that change_id, then complete the read path.
Historical example: "always allow every change" is discarded and must not become a current rule.
Inspection never deploys a release. Missing fields, malformed replies and provider failures block rather than select a business branch.
