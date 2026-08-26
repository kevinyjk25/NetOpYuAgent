#!/usr/bin/env bash
# ===========================================================================
# check_peers.sh — verify the 3 agents can sense each other as peers
# ===========================================================================
# Queries each agent's /registry/agents and prints the agents it has
# discovered (itself + peers whose AgentCard it fetched at startup).
#
# Expected after run_3agents.sh (FULL MESH — each sees all three):
#   lan (8000) sees: lan-agent, dc-agent, wan-agent
#   dc  (8001) sees: dc-agent,  lan-agent, wan-agent
#   wan (8002) sees: wan-agent, lan-agent, dc-agent
#
# Usage:  ./scripts/check_peers.sh
# ===========================================================================
set -uo pipefail

HAVE_JQ=1
command -v jq >/dev/null 2>&1 || HAVE_JQ=0

check_one() {
  local name="$1" port="$2"
  local url="http://localhost:${port}/registry/agents"
  printf "── %-4s (:%s) ──\n" "$name" "$port"
  local body
  body="$(curl -s --max-time 4 "$url" 2>/dev/null)" || {
    echo "   ✗ not reachable at $url (is it up?)"; echo; return; }
  if [[ -z "$body" ]]; then
    echo "   ✗ empty response (still booting?)"; echo; return; fi
  if [[ "$HAVE_JQ" == "1" ]]; then
    echo "$body" | jq -r '.[] | "   • \(.agent_id)  health=\(.health)  skills=\(.skills|join(","))"' 2>/dev/null \
      || { echo "   (raw) $body"; }
  else
    # no jq — crude grep for agent_id values
    echo "$body" | grep -o '"agent_id"[^,]*' | sed 's/^/   • /'
    echo "   (install jq for skills detail)"
  fi
  echo
}

echo "Peer-awareness check (each agent should list itself + its peers):"
echo
check_one "lan" 8000
check_one "dc"  8001
check_one "wan" 8002
echo "If an agent only lists itself, its peers weren't discovered — check that"
echo "the peer URLs are reachable and each agent finished booting (see logs_*.log)."
