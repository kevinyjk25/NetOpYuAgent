---
name: invoice-disclosure
description: Inspect an invoice using its existing settlement or waiver decision.
---
# Invoice disclosure
Take case_id from the caller and read_invoice_policy for that case.
An invoice may be inspected if it is settled OR an accounting waiver exists. Either alternative is enough; neither is mandatory on its own.
When neither alternative holds, stop unsupported without inspecting the invoice.
If an alternative holds, call inspect_invoice with the same case_id, then complete the read path.
Do not settle an invoice or create a waiver. A tool error or missing/invalid field blocks the flow, rather than counting as false.
