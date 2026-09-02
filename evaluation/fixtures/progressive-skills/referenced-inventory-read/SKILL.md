---
name: referenced-inventory-read
description: Read an asset record using the field definitions in the bundled reference.
allowed-tools: inventory_get_asset
metadata:
  risk_level: low
---
# Referenced inventory read

Read [the inventory contract](references/fields.md) only when the request needs field semantics.
Resolve exactly one supplied asset id, call the read-only inventory tool, and report unknown fields as unknown.

## Parameters

- `asset_id`: Explicit asset identifier.

## Steps

1. Require `asset_id`.
2. Read the asset once.
3. Return only fields defined by the reference.
