---
name: capacity-report-read
description: Read a capacity report only after interpreting the team's missing allocation policy.
---
# Capacity report
Before reading any report, determine eligibility according to references/allocation.md.
The allocation reference is not supplied. Do not infer its rules from this filename or from the report reader. Stop unsupported when the required reference is unavailable.
Only after eligibility has been established, call capacity_report with the caller's pool_key and complete the read path.
The host only supplies the report reader; it does not supply an eligibility decision or a policy interpreter.
