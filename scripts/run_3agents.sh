#!/usr/bin/env bash
# ===========================================================================
# run_3agents.sh — launch the LAN / WAN / DC three-agent topology locally
# ===========================================================================
#
# Topology: FULL MESH (each agent peers with the other two, 2 peers each):
#
#     LAN (8000) ─── WAN (8002)
#         \           /
#          \         /
#           DC (8001)
#
#   - lan-agent : enterprise LAN edge   (peers: dc, wan)
#   - wan-agent : wide-area transport   (peers: lan, dc)
#   - dc-agent  : data-center fabric    (peers: lan, wan)
#
# Any agent can delegate DIRECTLY to any other (no transit hop needed). An
# end-to-end "branch user can't reach a DC app" query on LAN can delegate the
# transport leg to WAN and the fabric leg to DC without hopping through a hub.
#
# Each agent auto-advertises its PROFILE capabilities in its AgentCard, so
# peers sense each other's capabilities at startup (no yaml editing needed).
#
# Requirements: an Ollama (or configured LLM) reachable at LLM_BASE_URL, and
# MODE=mock (default) so no device credentials are needed.
#
# Usage:
#     ./scripts/run_3agents.sh          # start all three (foreground logs)
#     ./scripts/run_3agents.sh stop     # stop all three
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE=".agents.pids"

stop_agents() {
  if [[ -f "$PIDFILE" ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done < "$PIDFILE"
    rm -f "$PIDFILE"
    echo "Stopped agents."
  else
    echo "No $PIDFILE — nothing to stop."
  fi
}

if [[ "${1:-}" == "stop" ]]; then
  stop_agents
  exit 0
fi

: > "$PIDFILE"

start_agent() {
  local id="$1" profile="$2" port="$3" peers="$4" display="$5"
  echo "Starting $id (profile=$profile) on :$port  peers=[$peers]"
  AGENT_ID="$id" \
  AGENT_PROFILE="$profile" \
  AGENT_DISPLAY_NAME="$display" \
  AGENT_PEERS="$peers" \
  PORT="$port" \
  A2A_BASE_URL="http://localhost:$port/api/v1/a2a" \
  MODE="${MODE:-mock}" \
  HITL_BACKEND=core \
    uvicorn main:app --port "$port" --host 0.0.0.0 \
    > "logs_${id}.log" 2>&1 &
  echo $! >> "$PIDFILE"
}

# WAN is the hub (2 peers); LAN and DC each peer only with WAN (1 peer each).
# Full mesh — each agent peers with the OTHER TWO (3 agents → 2 peers each,
# exactly at the max-2-peer bound). Any agent can delegate directly to any
# other without a transit hop.
start_agent "lan-agent" "lan" 8000 "http://localhost:8001,http://localhost:8002" "LAN Agent"
start_agent "dc-agent"  "dc"  8001 "http://localhost:8000,http://localhost:8002" "DC Agent"
start_agent "wan-agent" "wan" 8002 "http://localhost:8000,http://localhost:8001" "WAN Agent"

echo ""
echo "All three agents starting. Logs: logs_lan-agent.log / logs_dc-agent.log / logs_wan-agent.log"
echo "WebUI:  http://localhost:8000/webui/  (LAN)   :8001 (DC)   :8002 (WAN)"
echo ""
echo "Waiting for boot + peer discovery…"
echo "(Full mesh: agents launched together may miss each other on the first"
echo " fetch, but a fast-bootstrap retry runs every 5s for ~30s and self-heals.)"
sleep 8
echo ""
echo "Check the mesh formed (each agent should list ALL THREE):"
echo "  ./scripts/check_peers.sh          # one-shot verify"
echo "  # if any agent only sees a subset, wait ~30s for bootstrap + re-check"
echo ""
echo "Stop:   ./scripts/run_3agents.sh stop"
echo ""
echo "Tailing logs (Ctrl-C to detach; agents keep running — use 'stop' to kill):"
# wait for at least one log file to exist to avoid a tail race
for _ in 1 2 3 4 5; do
  [[ -f logs_lan-agent.log ]] && break
  sleep 1
done
tail -f logs_*.log
