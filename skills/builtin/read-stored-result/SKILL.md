---
name: read-stored-result
description: Page through a large stored tool result
metadata:
  skill_id: read_stored_result
  display_name: Read Stored Result
  purpose: Page through a large stored tool result
  risk_level: low
  requires_hitl: 'false'
  tags: storage,paging
  tool_deps: read_stored_result
  returns: Pages of stored content with metadata
---

# Read Stored Result

When a tool returns [STORED:name:ref_id], use this skill to read it page by page. Call [TOOL:read_stored_result] with the ref_id and increasing offsets. Write 2-3 sentences of findings after each page before calling the next.

## Parameters
- `ref_id`: The ref_id from the [STORED:] label
