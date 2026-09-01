---
name: scripted-data-export
description: Read an approved dataset and deterministically normalize it without writing external state.
allowed-tools: dataset_read
metadata:
  risk_level: low
  effect-runtime-script-roles: scripts/normalize.py=transformer
---
# Scripted data export

Follow [the export field contract](references/export-fields.md). After the read-only dataset call,
use `scripts/normalize.py` to create the normalized local result. Do not execute the script during
promotion or package inspection.
