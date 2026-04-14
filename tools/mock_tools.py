"""
tools/mock_tools.py
--------------------
Mock IT-ops tools with realistic outputs.

Three tools intentionally return large payloads to demonstrate P0 caching:
  - syslog_search  : returns hundreds of log lines
  - prometheus_query: returns time-series data
  - netflow_dump   : returns raw flow records

All tools are async callables matching the signature:
    async def tool(args: dict) -> str
and registered in TOOL_REGISTRY for injection into AgentRuntimeLoop.

P0 demo path
------------
When a tool output exceeds ToolResultStore.MAX_INLINE_CHARS (4 000 chars),
the Budget Manager automatically stores it and returns a reference label.
The WebUI can then call GET /tools/result/{ref_id}?offset=0 to page through
the stored data without re-running the tool.

Example ref label returned in the prompt:
    [STORED:syslog_search:a3f9c12b] Preview: Apr 10 09:12:01 ap-01 dhcp...
"""
from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int = 0) -> str:
    t = datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)
    return t.strftime("%b %d %H:%M:%S")


# ---------------------------------------------------------------------------
# Tool 1: syslog_search  (LARGE — triggers P0 cache)
# ---------------------------------------------------------------------------

async def syslog_search(args: dict[str, Any]) -> str:
    """
    Simulate a syslog query returning hundreds of lines.
    Deliberately large to trigger ToolResultStore caching.
    """
    host    = args.get("host", "ap-*")
    keyword = args.get("keyword", "error")
    lines   = args.get("lines", 300)

    await asyncio.sleep(0.05)  # simulate I/O

    severities  = ["ERROR", "WARN", "INFO", "DEBUG"]
    processes   = ["hostapd", "dhcpd", "kernel", "wpa_supplicant", "radiusd"]
    hosts       = ["ap-01", "ap-02", "ap-03", "sw-core-01", "sw-access-02"]
    messages    = [
        "association failed for client aa:bb:cc:dd:ee:ff reason=4",
        "DHCP DISCOVER from 00:11:22:33:44:55 via eth0.10",
        "authentication timeout for user alice@corp.com",
        "channel utilisation exceeded 80% on 5GHz band",
        "RADIUS timeout: no response from 10.0.1.5 (attempt 2/3)",
        "WPA handshake failed: incorrect PSK from 44:55:66:77:88:99",
        "interface eth0 link down, retrying in 5s",
        "neighbour table overflow: consider increasing net.ipv4.neigh.default.gc_thresh3",
        "PMK cache hit for client cc:dd:ee:ff:00:11",
        "roaming decision: RSSI -78 dBm below threshold -75 dBm",
    ]

    log_lines = []
    for i in range(lines):
        log_lines.append(
            f"{_ts(lines - i)} {random.choice(hosts)} "
            f"{random.choice(processes)}[{random.randint(1000,9999)}]: "
            f"[{random.choice(severities)}] {random.choice(messages)}"
        )

    header = (
        f"# syslog_search host={host} keyword={keyword} "
        f"results={lines} query_time=0.05s\n"
        "# " + "─" * 60 + "\n"
    )
    return header + "\n".join(log_lines)


# ---------------------------------------------------------------------------
# Tool 2: prometheus_query  (MEDIUM-LARGE — may trigger cache)
# ---------------------------------------------------------------------------

async def prometheus_query(args: dict[str, Any]) -> str:
    """
    Simulate a Prometheus instant/range query.
    Returns JSON-like time-series data.
    """
    metric  = args.get("metric", "up")
    job     = args.get("job", "network_devices")
    minutes = args.get("range_minutes", 60)

    await asyncio.sleep(0.03)

    now = datetime.now(timezone.utc)
    series = []
    for device_i in range(8):
        device = f"device_{device_i:02d}"
        values = []
        for step in range(0, minutes, 1):
            t = now - timedelta(minutes=minutes - step)
            v = round(random.uniform(0.85, 1.0), 4)
            values.append([int(t.timestamp()), str(v)])
        series.append({
            "metric": {
                "__name__": metric,
                "job": job,
                "instance": f"{device}:9090",
            },
            "values": values,
        })

    result = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": series,
        },
        "query": metric,
        "range_minutes": minutes,
        "total_points": len(series) * minutes,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 3: netflow_dump  (VERY LARGE — always triggers P0 cache)
# ---------------------------------------------------------------------------

async def netflow_dump(args: dict[str, Any]) -> str:
    """
    Simulate a NetFlow / IPFIX flow dump.
    Returns 500 flow records — always triggers ToolResultStore.
    """
    site       = args.get("site", "site-a")
    flow_count = args.get("flows", 500)

    await asyncio.sleep(0.08)

    protocols = ["TCP", "UDP", "ICMP"]
    ports     = [80, 443, 22, 53, 3389, 8080, 5060, 1194]
    src_nets  = ["10.0.0.", "10.0.1.", "192.168.1.", "172.16.0."]
    dst_nets  = ["8.8.8.", "1.1.1.", "203.0.113.", "198.51.100."]

    lines = [f"# NetFlow dump site={site} flows={flow_count}"]
    lines.append(
        "StartTime            SrcIP            DstIP            Proto  SrcPort DstPort  Bytes    Pkts"
    )
    lines.append("─" * 95)

    for i in range(flow_count):
        ts    = _ts(random.randint(0, 15))
        src   = random.choice(src_nets) + str(random.randint(1, 254))
        dst   = random.choice(dst_nets)  + str(random.randint(1, 254))
        proto = random.choice(protocols)
        sp    = random.choice(ports) + random.randint(0, 1000)
        dp    = random.choice(ports)
        byt   = random.randint(64, 65535)
        pkts  = random.randint(1, 100)
        lines.append(f"{ts}  {src:<16} {dst:<16} {proto:<6} {sp:<7} {dp:<7} {byt:<8} {pkts}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: dns_lookup  (small — returned inline)
# ---------------------------------------------------------------------------

async def dns_lookup(args: dict[str, Any]) -> str:
    """Quick DNS resolution check — always small, returned inline."""
    hostname = args.get("hostname", "example.com")
    await asyncio.sleep(0.01)
    return (
        f"DNS lookup: {hostname}\n"
        f"  A     → 203.0.113.42 (TTL 300)\n"
        f"  AAAA  → 2001:db8::42 (TTL 300)\n"
        f"  NS    → ns1.example.com, ns2.example.com\n"
        f"  Query time: 12ms   Server: 8.8.8.8"
    )


# ---------------------------------------------------------------------------
# Tool 5: device_info  (small — returned inline)
# ---------------------------------------------------------------------------

async def device_info(args: dict[str, Any]) -> str:
    """Get device details — small, returned inline."""
    device_id = args.get("device_id", "ap-01")
    await asyncio.sleep(0.01)
    return (
        f"Device: {device_id}\n"
        f"  Model:        Cisco Catalyst 9115AXI\n"
        f"  Firmware:     17.9.3\n"
        f"  Uptime:       14d 6h 23m\n"
        f"  Clients (2.4GHz): 12\n"
        f"  Clients (5GHz):   28\n"
        f"  Channel (2.4):    6\n"
        f"  Channel (5):      36\n"
        f"  Tx Power:     20 dBm\n"
        f"  Last reboot:  {_ts(20350)}"
    )


# ---------------------------------------------------------------------------
# Tool 6: alert_summary  (small — returned inline)
# ---------------------------------------------------------------------------

async def alert_summary(args: dict[str, Any]) -> str:
    """Fetch current alert summary — small, returned inline."""
    severity = args.get("severity", "all")
    await asyncio.sleep(0.01)
    return (
        f"Alert summary (severity={severity}):\n"
        f"  P0 (critical): 0\n"
        f"  P1 (high):     2  → [INC-1291] auth-radius timeout, [INC-1290] ap-03 offline\n"
        f"  P2 (medium):   7\n"
        f"  P3 (low):      14\n"
        f"  Total open:    23\n"
        f"  MTTR (24h):    18 min\n"
        f"  Trend:         stable"
    )


# ---------------------------------------------------------------------------
# Tool 7: service_health  (small — returned inline)
# ---------------------------------------------------------------------------

async def service_health(args: dict[str, Any]) -> str:
    """Check service health endpoint — small, returned inline."""
    service = args.get("service", "payments-service")
    await asyncio.sleep(0.01)
    statuses = ["healthy", "degraded", "healthy", "healthy"]
    s = random.choice(statuses)
    return (
        f"Health check: {service}\n"
        f"  Status:      {s}\n"
        f"  Uptime:      99.97%\n"
        f"  Error rate:  0.03%\n"
        f"  Latency p99: 142ms\n"
        f"  Replicas:    3/3\n"
        f"  Last deploy: {_ts(720)}"
    )


# ---------------------------------------------------------------------------
# Tool 8: read_stored_result  (P0 on-demand retrieval tool)
# ---------------------------------------------------------------------------

# This is wired dynamically at runtime with the ToolResultStore reference.
# See webui/backend.py for how it's injected.

def make_read_stored_result_tool(tool_store):
    """
    Factory: returns a tool function bound to a specific ToolResultStore.
    Used to let the LLM retrieve a page of a large cached result.
    """
    async def read_stored_result(args: dict[str, Any]) -> str:
        ref_id = args.get("ref_id", "")
        offset = int(args.get("offset", 0))
        length = int(args.get("length", 2000))

        if not ref_id:
            return "[Error: ref_id is required]"

        chunk = tool_store.read(ref_id, offset=offset, length=length)
        if chunk is None:
            return f"[Error: no stored result found for ref_id={ref_id!r}]"

        total = len(tool_store._store.get(ref_id, ""))
        next_offset = offset + len(chunk)
        has_more    = next_offset < total

        return (
            f"# Stored result ref_id={ref_id} offset={offset} length={len(chunk)}\n"
            f"# Total size: {total} chars  |  Has more: {has_more}  |  "
            f"Next offset: {next_offset if has_more else 'EOF'}\n"
            f"# {'─'*60}\n"
            f"{chunk}"
        )

    return read_stored_result


# ---------------------------------------------------------------------------
# Registry  (imported by webui backend and AgentRuntimeLoop)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "syslog_search":     syslog_search,
    "prometheus_query":  prometheus_query,
    "netflow_dump":      netflow_dump,
    "dns_lookup":        dns_lookup,
    "device_info":       device_info,
    "alert_summary":     alert_summary,
    "service_health":    service_health,
    # read_stored_result is injected at runtime (needs ToolResultStore ref)
}

TOOL_DESCRIPTIONS = {
    "syslog_search": {
        "description": "Search syslog entries across network devices",
        "parameters": {"host": "device name / glob", "keyword": "search term", "lines": "number of lines (default 300)"},
        "returns_large": True,
        "example": {"host": "ap-01", "keyword": "error", "lines": 300},
    },
    "prometheus_query": {
        "description": "Query Prometheus metrics time series",
        "parameters": {"metric": "metric name", "job": "job label", "range_minutes": "look-back window"},
        "returns_large": True,
        "example": {"metric": "up", "job": "network_devices", "range_minutes": 60},
    },
    "netflow_dump": {
        "description": "Dump NetFlow / IPFIX records for a site",
        "parameters": {"site": "site name", "flows": "number of flow records"},
        "returns_large": True,
        "example": {"site": "site-a", "flows": 500},
    },
    "dns_lookup": {
        "description": "Resolve a hostname and return DNS records",
        "parameters": {"hostname": "FQDN to resolve"},
        "returns_large": False,
        "example": {"hostname": "payments.internal"},
    },
    "device_info": {
        "description": "Get hardware/firmware details for a network device",
        "parameters": {"device_id": "device identifier (e.g. ap-01)"},
        "returns_large": False,
        "example": {"device_id": "ap-01"},
    },
    "alert_summary": {
        "description": "Return current alert counts by severity",
        "parameters": {"severity": "filter: all | P0 | P1 | P2 | P3"},
        "returns_large": False,
        "example": {"severity": "P1"},
    },
    "service_health": {
        "description": "Check health status of a backend service",
        "parameters": {"service": "service name"},
        "returns_large": False,
        "example": {"service": "payments-service"},
    },
    "read_stored_result": {
        "description": "Page through a large tool result stored by ref_id",
        "parameters": {
            "ref_id":  "reference ID from a [STORED:...] label",
            "offset":  "byte offset (default 0)",
            "length":  "bytes to read (default 2000)",
        },
        "returns_large": False,
        "example": {"ref_id": "a3f9c12b", "offset": 0, "length": 2000},
    },
}