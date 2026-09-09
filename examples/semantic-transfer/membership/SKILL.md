---
name: membership-benefit-preview
description: Preview a benefit using existing membership facts, without changing enrollment.
---
# Benefit preview
For the caller's request_key, first obtain membership_facts.
A suspended member must not proceed. A non-suspended member may proceed when the tier covers this request or an existing sponsor clearance covers it; either coverage route is sufficient.
If this condition fails, stop unsupported before reading any benefit details.
When it holds, call benefit_preview with the same request_key and complete the read path.
Do not change suspension, tier, or sponsor clearance. Errors and malformed responses block; they are not negative business facts.
