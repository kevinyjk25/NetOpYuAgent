#!/usr/bin/env bash
#
# scripts/migrate_data_to_agent.sh
# --------------------------------
# Migrate the legacy shared data/ state into a per-agent subtree
# data/agents/<agent_id>/ introduced by the 2026-05 data-isolation change.
#
# Before this change, ALL agents shared one data/ directory. After it, each
# agent_id gets its own subtree. If you have existing state from running a
# single (LAN) agent, run this once to move it under that agent's id so the
# agent keeps its memory / sessions / cached results / evolved skills.
#
# Usage:
#   ./scripts/migrate_data_to_agent.sh <agent_id> [data_root]
#
# Examples:
#   ./scripts/migrate_data_to_agent.sh lan-agent
#   ./scripts/migrate_data_to_agent.sh lan-agent ./data
#
# What moves (per-agent state):
#   memory/                  (or memory.db + tool_cache/ if old layout)
#   tool_results.db          (large tool-output cache)
#   hitl_checkpoints.db      (pending HITL approvals)
#   skills/                  (auto-evolved skill markdown)
#   skill_journal.jsonl      (if present)
#
# What STAYS at data/ (shared, read-only fixtures — NOT moved):
#   golden_set.jsonl
#   tool_compliance_set.jsonl
#
# The script is idempotent-ish: it refuses to overwrite an existing target
# file and prints what it did. It never deletes the shared fixtures.

set -euo pipefail

AGENT_ID="${1:-}"
DATA_ROOT="${2:-./data}"

if [[ -z "$AGENT_ID" ]]; then
  echo "Usage: $0 <agent_id> [data_root]" >&2
  echo "  e.g. $0 lan-agent" >&2
  exit 1
fi

TARGET="${DATA_ROOT}/agents/${AGENT_ID}"

echo "Migrating shared state in '${DATA_ROOT}' → '${TARGET}'"
echo ""

mkdir -p "$TARGET"

# Files / dirs that are per-agent state (moved). Shared fixtures excluded.
PER_AGENT_ITEMS=(
  "memory"
  "memory.db"
  "memory.db-wal"
  "memory.db-shm"
  "tool_cache"
  "tool_results.db"
  "tool_results.db-wal"
  "tool_results.db-shm"
  "hitl_checkpoints.db"
  "hitl_checkpoints.db-wal"
  "hitl_checkpoints.db-shm"
  "skills"
  "skill_journal.jsonl"
)

moved=0
skipped=0
for item in "${PER_AGENT_ITEMS[@]}"; do
  src="${DATA_ROOT}/${item}"
  dst="${TARGET}/${item}"
  if [[ -e "$src" ]]; then
    if [[ -e "$dst" ]]; then
      echo "  ⚠ SKIP ${item} — already exists at target (not overwriting)"
      skipped=$((skipped + 1))
    else
      mv "$src" "$dst"
      echo "  ✓ moved ${item}"
      moved=$((moved + 1))
    fi
  fi
done

echo ""
echo "Done: ${moved} item(s) moved, ${skipped} skipped."
echo ""
echo "Shared fixtures left in place (correct — these are NOT per-agent):"
for f in golden_set.jsonl tool_compliance_set.jsonl; do
  [[ -e "${DATA_ROOT}/${f}" ]] && echo "  · ${DATA_ROOT}/${f}"
done
echo ""
echo "Next: start the agent with AGENT_ID=${AGENT_ID} and it will read from"
echo "  ${TARGET}/ automatically."
