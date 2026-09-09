---
name: dispatch-preview
description: Preview dispatch under a valid decision with one of two independent approvals.
---
# Dispatch preview
Call get_dispatch_decision for the caller's shipment_id.
Preview requires a valid decision AND at least one of fast-track eligibility or a signed review. The two alternatives must not both be required.
If the decision is invalid or neither alternative holds, stop unsupported without previewing.
Otherwise call preview_dispatch for the same shipment_id and complete the read path.
Do not dispatch the shipment or create a signed review. Tool errors and missing or malformed facts block the flow.
