---
name: ticket-evidence-view
description: View ticket evidence under a confirmed decision and conditional delegation.
---
# Ticket evidence view
For the supplied ticket_key, call ticket_access_facts first.
The decision must be confirmed. Delegated access is additionally required only when the ticket is sensitive; a non-sensitive ticket does not need delegated access.
If either applicable requirement fails, stop unsupported without loading evidence.
Otherwise call ticket_evidence using that ticket_key and complete the read path.
Never grant delegation or change sensitivity. Provider errors and invalid data block, not false.
