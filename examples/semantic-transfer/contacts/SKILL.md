---
name: contact-card-inspection
description: Read a contact card only under the included regional privacy rules.
---
# Contact card inspection
Read contact_flags using the caller's contact_id, then apply the rules in references/privacy.md before any card read.
When those rules allow it, call read_contact_card using that contact_id and complete the read path. Otherwise stop unsupported.
Do not send messages or change consent. Malformed results and absent facts block the flow.
