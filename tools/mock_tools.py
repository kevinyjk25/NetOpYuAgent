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
    Respects keyword filter: lines are filtered so the keyword appears in results.
    """
    host    = args.get("host", "ap-*")
    keyword = args.get("keyword", "error")
    lines   = args.get("lines", 300)
    user    = args.get("user", "")   # optional: search for specific username

    await asyncio.sleep(0.05)  # simulate I/O

    severities = ["ERROR", "WARN", "INFO", "DEBUG"]
    processes  = ["hostapd", "dhcpd", "kernel", "wpa_supplicant", "radiusd"]
    hosts      = ["ap-01", "ap-02", "ap-03", "sw-core-01", "sw-access-02",
                  "radius-01", "radius-02"]

    # Base pool of realistic RADIUS/auth log messages
    base_messages = [
        "association failed for client aa:bb:cc:dd:ee:ff reason=4",
        "DHCP DISCOVER from 00:11:22:33:44:55 via eth0.10",
        "authentication timeout for user alice@corp.com",
        "channel utilisation exceeded 80% on 5GHz band",
        "RADIUS timeout: no response from 10.0.1.5 (attempt 2/3)",
        "WPA handshake failed: incorrect PSK from 44:55:66:77:88:99",
        "interface eth0 link down, retrying in 5s",
        "neighbour table overflow: consider increasing gc_thresh3",
        "PMK cache hit for client cc:dd:ee:ff:00:11",
        "roaming decision: RSSI -78 dBm below threshold -75 dBm",
        "RADIUS Access-Accept for user bob@corp.com from 10.0.1.5",
        "RADIUS Access-Reject for user charlie@corp.com: bad password",
        "EAP-TLS: cert expired for user dave@corp.com (expired 2025-03-01)",
        "user eve@corp.com authenticated successfully via PEAP",
        "failed login attempt for user frank@corp.com (attempt 3/5)",
        "session started for user grace@corp.com MAC=aa:bb:cc:11:22:33",
        "session ended for user grace@corp.com duration=3h42m bytes=1.2GB",
        "certificate validation failed for user henry@corp.com",
    ]

    # Determine effective search term
    search_term = (user or keyword).lower()

    # Build candidate lines — every line contains the search term at least sometimes
    log_lines = []
    for i in range(lines):
        # Every ~4th line is guaranteed to match the search term
        if i % 4 == 0 and search_term:
            # Generate a log line that contains the search term
            if user:
                msg = f"RADIUS Access-{'Accept' if i%8<4 else 'Reject'} for user {user} from 10.0.{i%4+1}.5"
                sev = "INFO" if i%8<4 else "WARN"
            else:
                # keyword match — embed keyword naturally
                msg = f"{keyword}: detected on interface eth0.{i%8+10}"
                sev = "WARN"
        else:
            msg = random.choice(base_messages)
            sev = random.choice(severities)

        log_lines.append(
            f"{_ts(lines - i)} {random.choice(hosts)} "
            f"{random.choice(processes)}[{random.randint(1000,9999)}]: "
            f"[{sev}] {msg}"
        )

    # Count matches for the header
    match_count = sum(1 for l in log_lines if search_term in l.lower())
    header = (
        f"# syslog_search host={host} keyword={keyword}"
        + (f" user={user}" if user else "")
        + f" results={lines} matched={match_count} query_time=0.05s\n"
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

    async def process_stored_chunks(args: dict[str, Any]) -> str:
        """
        General-purpose chunk iterator over a stored large result.

        Splits the stored content into fixed-size chunks and applies one of
        several built-in operations to every chunk, accumulating the results.
        This is the right tool whenever you need to process an entire large file
        rather than just reading one page of it.

        Operations (set via `operation` arg):
          "filter"    – keep only lines that contain `match` (case-insensitive)
          "reject"    – keep only lines that do NOT contain `match`
          "extract"   – extract the first regex `pattern` capture group from each line
          "count"     – count lines containing `match` per chunk; return totals
          "summarise" – return the first `head` and last `tail` lines of each chunk
                        (useful for giving an LLM a digest of each section)
          "passthrough" – return every chunk verbatim (same as calling read_stored_result
                          in a loop, but done for you automatically)

        Common usage patterns:
          Check if user X exists anywhere in a log:
            {"ref_id": "...", "operation": "filter", "match": "alice@corp.com"}

          Find all ERROR lines across the whole file:
            {"ref_id": "...", "operation": "filter", "match": "ERROR"}

          Extract all IP addresses from a NetFlow dump:
            {"ref_id": "...", "operation": "extract", "pattern": "(\\d+\\.\\d+\\.\\d+\\.\\d+)"}

          Count how many times each chunk mentions 'timeout':
            {"ref_id": "...", "operation": "count", "match": "timeout"}

          Get a digest of a large prometheus dump:
            {"ref_id": "...", "operation": "summarise", "head": 3, "tail": 2}
        """
        import re as _re

        ref_id    = args.get("ref_id", "")
        operation = args.get("operation", "filter")
        match_str = args.get("match", "")
        pattern   = args.get("pattern", "")
        chunk_size = int(args.get("chunk_size", 3000))   # chars per chunk
        max_output = int(args.get("max_output", 6000))   # cap total output chars
        head_n    = int(args.get("head", 5))
        tail_n    = int(args.get("tail", 5))

        if not ref_id:
            return "[Error: ref_id is required]"

        full = tool_store._store.get(ref_id)
        if full is None:
            return f"[Error: no stored result for ref_id={ref_id!r}]"

        total_chars = len(full)
        all_lines   = full.splitlines()
        total_lines = len(all_lines)

        # Split into line-aligned chunks
        chunks: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for line in all_lines:
            current.append(line)
            current_len += len(line) + 1
            if current_len >= chunk_size:
                chunks.append(current)
                current = []
                current_len = 0
        if current:
            chunks.append(current)

        output_parts: list[str] = []
        total_matched = 0
        output_chars  = 0

        header = (
            f"# process_stored_chunks ref_id={ref_id} operation={operation!r}\n"
            f"# Total: {total_chars} chars, {total_lines} lines, {len(chunks)} chunk(s)\n"
            f"# {'─'*60}\n"
        )
        output_parts.append(header)
        output_chars += len(header)

        for chunk_idx, chunk_lines in enumerate(chunks):
            if output_chars >= max_output:
                output_parts.append(
                    f"\n# [Output cap {max_output} chars reached — "
                    f"{len(chunks) - chunk_idx} chunk(s) not shown]\n"
                )
                break

            if operation == "filter":
                kw = match_str.lower()
                hits = [l for l in chunk_lines if kw in l.lower()]
                total_matched += len(hits)
                if hits:
                    block = f"\n# chunk {chunk_idx+1}: {len(hits)} match(es)\n" + "\n".join(hits)
                    output_parts.append(block)
                    output_chars += len(block)

            elif operation == "reject":
                kw = match_str.lower()
                kept = [l for l in chunk_lines if kw not in l.lower()]
                total_matched += len(kept)
                block = f"\n# chunk {chunk_idx+1}: {len(kept)} line(s) kept\n" + "\n".join(kept)
                output_parts.append(block)
                output_chars += len(block)

            elif operation == "extract":
                if not pattern:
                    return "[Error: 'pattern' is required for operation='extract']"
                extracted = []
                for line in chunk_lines:
                    m = _re.search(pattern, line)
                    if m:
                        val = m.group(1) if m.lastindex else m.group(0)
                        extracted.append(val)
                seen = sorted(set(extracted))
                total_matched += len(seen)
                if seen:
                    block = f"\n# chunk {chunk_idx+1}: {len(seen)} unique value(s)\n" + "\n".join(seen)
                    output_parts.append(block)
                    output_chars += len(block)

            elif operation == "count":
                kw = match_str.lower()
                count = sum(1 for l in chunk_lines if kw in l.lower())
                total_matched += count
                block = f"\n# chunk {chunk_idx+1}: {count} line(s) contain {match_str!r}"
                output_parts.append(block)
                output_chars += len(block)

            elif operation == "summarise":
                h = chunk_lines[:head_n]
                t = chunk_lines[-tail_n:] if tail_n else []
                mid_omitted = max(0, len(chunk_lines) - head_n - tail_n)
                block = (
                    f"\n# chunk {chunk_idx+1} ({len(chunk_lines)} lines):\n"
                    + "\n".join(h)
                    + (f"\n  … {mid_omitted} lines omitted …\n" if mid_omitted > 0 else "\n")
                    + ("\n".join(t) if t else "")
                )
                output_parts.append(block)
                output_chars += len(block)

            elif operation == "passthrough":
                block = f"\n# chunk {chunk_idx+1}:\n" + "\n".join(chunk_lines)
                output_parts.append(block)
                output_chars += len(block)

            else:
                return f"[Error: unknown operation={operation!r}. Choose: filter, reject, extract, count, summarise, passthrough]"

        # Summary footer
        op_summary = {
            "filter":      f"{total_matched} matching line(s) found",
            "reject":      f"{total_matched} line(s) passed filter",
            "extract":     f"{total_matched} unique value(s) extracted across all chunks",
            "count":       f"{total_matched} total line(s) matched across all chunks",
            "summarise":   f"{len(chunks)} chunk(s) summarised",
            "passthrough": f"{len(chunks)} chunk(s) returned",
        }.get(operation, "")
        footer = f"\n# {'─'*60}\n# Result: {op_summary}\n"
        output_parts.append(footer)

        return "".join(output_parts)

    return read_stored_result, process_stored_chunks


# ---------------------------------------------------------------------------
# Registry  (imported by webui backend and AgentRuntimeLoop)
# ---------------------------------------------------------------------------
# Tool 8a: list_devices  (inventory — list all network devices)
# ---------------------------------------------------------------------------

# Canonical device inventory used across mock tools
_DEVICE_INVENTORY = [
    # Wireless APs
    {"id": "ap-01", "type": "wireless_ap",  "role": "access_point",  "site": "site-a", "model": "Cisco Catalyst 9115AXI",   "ip": "10.0.1.11"},
    {"id": "ap-02", "type": "wireless_ap",  "role": "access_point",  "site": "site-a", "model": "Cisco Catalyst 9115AXI",   "ip": "10.0.1.12"},
    {"id": "ap-03", "type": "wireless_ap",  "role": "access_point",  "site": "site-b", "model": "Cisco Catalyst 9130AXI",   "ip": "10.0.2.11"},
    {"id": "ap-04", "type": "wireless_ap",  "role": "access_point",  "site": "site-b", "model": "Cisco Catalyst 9130AXI",   "ip": "10.0.2.12"},
    # Wired switches
    {"id": "sw-core-01", "type": "switch",  "role": "core_switch",   "site": "site-a", "model": "Cisco Catalyst 9500-48Y4C","ip": "10.0.1.1"},
    {"id": "sw-core-02", "type": "switch",  "role": "core_switch",   "site": "site-b", "model": "Cisco Catalyst 9500-48Y4C","ip": "10.0.2.1"},
    {"id": "sw-acc-01",  "type": "switch",  "role": "access_switch", "site": "site-a", "model": "Cisco Catalyst 9300-48P",  "ip": "10.0.1.21"},
    {"id": "sw-acc-02",  "type": "switch",  "role": "access_switch", "site": "site-a", "model": "Cisco Catalyst 9300-48P",  "ip": "10.0.1.22"},
    {"id": "sw-acc-03",  "type": "switch",  "role": "access_switch", "site": "site-b", "model": "Cisco Catalyst 9300-48P",  "ip": "10.0.2.21"},
    # Routers
    {"id": "router-01",  "type": "router",  "role": "edge_router",   "site": "site-a", "model": "Cisco ASR 1001-X",         "ip": "10.0.1.254"},
    {"id": "router-02",  "type": "router",  "role": "edge_router",   "site": "site-b", "model": "Cisco ASR 1001-X",         "ip": "10.0.2.254"},
    # Auth servers
    {"id": "radius-01",  "type": "server",  "role": "radius_server", "site": "site-a", "model": "Linux VM / FreeRADIUS",    "ip": "10.0.1.100"},
    {"id": "radius-02",  "type": "server",  "role": "radius_server", "site": "site-b", "model": "Linux VM / FreeRADIUS",    "ip": "10.0.2.100"},
]

async def list_devices(args: dict[str, Any]) -> str:
    """List all network devices in inventory, with optional filtering."""
    device_type = args.get("type", "").lower()    # wireless_ap | switch | router | server | ""
    site        = args.get("site", "").lower()    # site-a | site-b | ""
    role        = args.get("role", "").lower()    # core_switch | access_point | edge_router | ""

    await asyncio.sleep(0.01)

    devices = _DEVICE_INVENTORY
    if device_type:
        devices = [d for d in devices if device_type in d["type"]]
    if site:
        devices = [d for d in devices if site in d["site"]]
    if role:
        devices = [d for d in devices if role in d["role"]]

    if not devices:
        return (
            f"No devices found matching: type={device_type or '*'} "
            f"site={site or '*'} role={role or '*'}"
        )

    lines = [
        f"# Device inventory  type={device_type or 'all'}  site={site or 'all'}  "
        f"count={len(devices)}",
        f"# {'─'*65}",
        f"{'ID':<15} {'TYPE':<14} {'ROLE':<16} {'SITE':<8} {'IP':<15} MODEL",
        f"{'─'*15} {'─'*14} {'─'*16} {'─'*8} {'─'*15} {'─'*28}",
    ]
    for d in devices:
        lines.append(
            f"{d['id']:<15} {d['type']:<14} {d['role']:<16} {d['site']:<8} "
            f"{d['ip']:<15} {d['model']}"
        )
    lines.append(f"# Total: {len(devices)} device(s)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 8b: list_interfaces  (per-device interface table)
# ---------------------------------------------------------------------------

async def list_interfaces(args: dict[str, Any]) -> str:
    """List network interfaces for a specific device."""
    device_id = args.get("device_id", "sw-core-01")
    await asyncio.sleep(0.01)

    import random, hashlib
    seed = int(hashlib.md5(device_id.encode()).hexdigest()[:8], 16)
    random.seed(seed)

    # Find device type to generate realistic interfaces
    dev = next((d for d in _DEVICE_INVENTORY if d["id"] == device_id), None)
    if not dev:
        return f"[Error: device {device_id!r} not found in inventory. Use list_devices to see valid IDs.]"

    lines = [f"# Interfaces for {device_id} ({dev['model']})", f"# {'─'*60}"]
    states = ["up", "up", "up", "up", "down"]   # mostly up

    if dev["type"] == "switch":
        lines.append(f"{'INTERFACE':<20} {'STATE':<6} {'VLAN':<6} {'SPEED':<10} DESCRIPTION")
        lines.append(f"{'─'*20} {'─'*6} {'─'*6} {'─'*10} {'─'*25}")
        for i in range(1, 25):
            state = random.choice(states)
            vlan  = random.choice([1, 10, 20, 100])
            speed = "1G" if i <= 20 else "10G"
            desc  = f"access-port-{i}" if vlan != 100 else f"uplink-{i}"
            lines.append(f"GigabitEthernet1/0/{i:<4} {state:<6} {vlan:<6} {speed:<10} {desc}")
        for i in range(1, 5):
            state = "up"
            lines.append(f"TenGigabitEth1/1/{i:<3}  {state:<6} trunk  10G        {'uplink-core' if i <= 2 else 'uplink-peer'}")
    elif dev["type"] == "wireless_ap":
        lines.append(f"{'INTERFACE':<16} {'STATE':<6} {'FREQ':<8} {'CHANNEL':<9} {'CLIENTS':<9} SSID")
        lines.append(f"{'─'*16} {'─'*6} {'─'*8} {'─'*9} {'─'*9} {'─'*20}")
        lines.append(f"radio0           up     2.4GHz   {random.choice([1,6,11]):<9} {random.randint(5,25):<9} corp-wifi")
        lines.append(f"radio1           up     5GHz     {random.choice([36,40,44,149]):<9} {random.randint(10,50):<9} corp-wifi-5g")
        lines.append(f"eth0             up     —        —         —         uplink (PoE)")
    elif dev["type"] == "router":
        lines.append(f"{'INTERFACE':<20} {'STATE':<6} {'IP':<18} {'SPEED':<10} DESCRIPTION")
        lines.append(f"{'─'*20} {'─'*6} {'─'*18} {'─'*10} {'─'*25}")
        lines.append(f"GigabitEthernet0/0   up     {dev['ip']:<18} 1G         LAN uplink")
        lines.append(f"GigabitEthernet0/1   up     203.0.113.{random.randint(1,254):<10}  1G         WAN primary")
        lines.append(f"GigabitEthernet0/2   {'up' if random.random()>0.3 else 'down':<6} 198.51.100.{random.randint(1,254):<8}  1G         WAN backup")
        lines.append(f"Loopback0            up     10.255.{random.randint(0,9)}.{random.randint(1,254):<8} —          Management")
    else:
        lines.append(f"eth0   up   {dev['ip']:<18} 1G   Primary")
        lines.append(f"eth1   up   169.254.0.1         1G   Management")

    lines.append(f"# Total interfaces: {len(lines)-3}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool: get_device_config  (mock)
# ---------------------------------------------------------------------------

async def get_device_config(args: dict[str, Any]) -> str:
    """Mock: return realistic AP/switch running config with seeded issues."""
    device_id = args.get("device_id", "ap-01")
    section   = args.get("section")
    await asyncio.sleep(0.05)

    dev = next((d for d in _DEVICE_INVENTORY if d["id"] == device_id), None)
    if not dev:
        return f"[Error: device {device_id!r} not found. Use list_devices to see valid IDs.]"

    import hashlib as _hs
    seed = int(_hs.md5(device_id.encode()).hexdigest()[:4], 16)
    ntp_missing    = (seed % 4 == 0)
    radius_timeout = 3 + (seed % 4)
    vlan_acl_ok    = (seed % 3 != 1)
    ip = dev["ip"]

    if dev["type"] == "wireless_ap":
        if section == "radius":
            return (f"# RADIUS config for {device_id}\n"
                    f"radius-server host 10.0.1.100 auth-port 1812\n"
                    f" timeout {radius_timeout}"
                    + ("   ! WARNING: recommend <=3s\n" if radius_timeout > 3 else "\n"))
        if section == "ntp":
            return (f"# NTP config for {device_id}\n"
                    + ("! NTP NOT CONFIGURED\n" if ntp_missing else "ntp server 10.0.1.5\nntp server 10.0.1.6\n"))
        return (
            f"! Configuration for {device_id} ({dev['model']}) — site {dev['site']}\n"
            f"hostname {device_id}\n!\n"
            f"interface GigabitEthernet0\n ip address {ip} 255.255.255.0\n no shutdown\n!\n"
            f"dot11 ssid corp-wifi\n vlan 20\n authentication key-management wpa version 2\n!\n"
            + ("! NTP NOT CONFIGURED — clock drift risk!\n" if ntp_missing else f"ntp server 10.0.1.5\nntp server 10.0.1.6\n")
            + f"!\nradius-server host 10.0.1.100 auth-port 1812\n timeout {radius_timeout}"
            + ("   ! WARNING: recommend <=3s\n" if radius_timeout > 3 else "\n")
            + ("!\n! ACL NOT APPLIED to mgmt VLAN — security gap!\n" if not vlan_acl_ok else "!\nip access-list extended MGMT\n permit tcp 10.0.0.0 0.255.255.255 any\n deny ip any any log\n")
            + "!\nend"
        )
    if dev["type"] == "switch":
        return (
            f"! Configuration for {device_id} ({dev['model']})\n"
            f"hostname {device_id}\n!\nspanning-tree mode rapid-pvst\n!\n"
            f"interface Vlan10\n ip address {ip} 255.255.255.0\n!\n"
            f"ntp server 10.0.1.5\nntp server 10.0.1.6\n!\n"
            f"radius-server host 10.0.1.100 timeout 3\n!\nend"
        )
    return f"! Configuration for {device_id}\nhostname {device_id}\n!\nend"


# ---------------------------------------------------------------------------
# Tool: validate_device_config  (mock)
# ---------------------------------------------------------------------------

async def validate_device_config(args: dict[str, Any]) -> str:
    """Mock: deterministic PASS/WARN/FAIL report seeded by device ID."""
    device_id = args.get("device_id", "ap-01")
    await asyncio.sleep(0.03)

    dev = next((d for d in _DEVICE_INVENTORY if d["id"] == device_id), None)
    if not dev:
        return f"[Error: device {device_id!r} not found.]"

    import hashlib as _hs
    seed = int(_hs.md5(device_id.encode()).hexdigest()[:4], 16)
    issues, warnings, passed = [], [], []

    if dev["type"] == "wireless_ap":
        ntp_missing    = (seed % 4 == 0)
        radius_timeout = 3 + (seed % 4)
        vlan_acl_ok    = (seed % 3 != 1)
        if ntp_missing:
            issues.append("FAIL  [NTP]    NTP server not configured — clock drift risk")
        else:
            passed.append("PASS  [NTP]    NTP configured (2 servers)")
        if radius_timeout > 3:
            warnings.append(f"WARN  [RADIUS] timeout={radius_timeout}s > recommended 3s (auth delays under load)")
        else:
            passed.append(f"PASS  [RADIUS] timeout={radius_timeout}s OK")
        passed.append("PASS  [RADIUS] server 10.0.1.100 reachable")
        if not vlan_acl_ok:
            warnings.append("WARN  [ACL]    Management VLAN ACL not applied — unrestricted management access")
        else:
            passed.append("PASS  [ACL]    Management VLAN ACL applied")
        passed.extend(["PASS  [SSID]   WPA2 on all SSIDs", "PASS  [SSID]   Guest SSID isolated"])
    elif dev["type"] == "switch":
        passed.extend(["PASS  [STP]    Rapid-PVST configured", "PASS  [VLAN]   VLANs 10/20/100 present",
                       "PASS  [NTP]    NTP configured", "PASS  [RADIUS] timeout=3s OK"])
    else:
        passed.extend(["PASS  [ROUTING] Default route present", "PASS  [NTP]     NTP configured"])

    lines = [f"VALIDATION REPORT — {device_id} ({dev['model']}) — site {dev['site']}", "=" * 65]
    lines += issues + warnings + passed
    lines += ["=" * 65,
              f"Summary: {len(issues)} issue(s), {len(warnings)} warning(s), {len(passed)} check(s) passed"]
    return "\n".join(lines)


TOOL_REGISTRY: dict[str, callable] = {
    "syslog_search":          syslog_search,
    "prometheus_query":       prometheus_query,
    "netflow_dump":           netflow_dump,
    "dns_lookup":             dns_lookup,
    "device_info":            device_info,
    "alert_summary":          alert_summary,
    "service_health":         service_health,
    "list_devices":           list_devices,
    "list_interfaces":        list_interfaces,
    "get_device_config":      get_device_config,
    "validate_device_config": validate_device_config,
    # read_stored_result and process_stored_chunks are injected at runtime (need ToolResultStore ref)
}

TOOL_DESCRIPTIONS = {
    "syslog_search": {
        "description": "Search syslog entries across network devices. Supports keyword and user filtering.",
        "parameters": {
            "host":    "device name / glob (e.g. 'radius-*')",
            "keyword": "search term (e.g. 'error', 'timeout')",
            "user":    "username to search for (e.g. 'alice@corp.com')",
            "lines":   "number of lines to return (default 300)",
        },
        "returns_large": True,
        "example": {"host": "radius-*", "user": "alice@corp.com", "lines": 300},
    },
    "process_stored_chunks": {
        "description": (
            "General-purpose chunk iterator over a stored large result. "
            "Splits the full content into chunks and applies an operation to every chunk automatically — "
            "no manual offset loop needed. "
            "Operations: 'filter' (keep lines matching a term), 'reject' (keep non-matching lines), "
            "'extract' (regex capture from each line), 'count' (count matches per chunk), "
            "'summarise' (head+tail digest of each chunk), 'passthrough' (all chunks verbatim). "
            "Use this whenever you need to process an entire large file, not just one page."
        ),
        "parameters": {
            "ref_id":     "stored result ID from a [STORED:...] reference",
            "operation":  "filter | reject | extract | count | summarise | passthrough",
            "match":      "string to match against (for filter, reject, count)",
            "pattern":    "regex pattern with capture group (for extract)",
            "chunk_size": "chars per chunk (default 3000)",
            "max_output": "max total output chars (default 6000)",
            "head":       "lines from top of each chunk for summarise (default 5)",
            "tail":       "lines from bottom of each chunk for summarise (default 5)",
        },
        "returns_large": False,
        "examples": [
            {"ref_id": "a3f9c12b", "operation": "filter",    "match": "alice@corp.com"},
            {"ref_id": "a3f9c12b", "operation": "filter",    "match": "ERROR"},
            {"ref_id": "a3f9c12b", "operation": "extract",   "pattern": "(\\d+\\.\\d+\\.\\d+\\.\\d+)"},
            {"ref_id": "a3f9c12b", "operation": "count",     "match": "timeout"},
            {"ref_id": "a3f9c12b", "operation": "summarise", "head": 3, "tail": 2},
        ],
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
    "list_devices": {
        "description": (
            "List ALL network devices in the inventory — APs, switches, routers, servers. "
            "Use this when asked 'what devices exist', 'show all devices', or any inventory query. "
            "Optionally filter by type (wireless_ap | switch | router | server), "
            "site (site-a | site-b), or role (core_switch | access_point | edge_router | radius_server). "
            "Returns a table with device ID, type, role, site, IP, and model. "
            "Use device_info for detailed info on a SPECIFIC device after you know its ID."
        ),
        "parameters": {
            "type": "filter by device type: wireless_ap | switch | router | server | '' (all)",
            "site": "filter by site: site-a | site-b | '' (all)",
            "role": "filter by role: core_switch | access_switch | access_point | edge_router | radius_server | '' (all)",
        },
        "returns_large": False,
        "example": {"type": "", "site": ""},
        "examples": [
            {"type": ""},                        # all devices
            {"type": "switch"},                  # wired switches only
            {"type": "wireless_ap"},             # wireless APs only
            {"type": "router"},                  # routers only
            {"type": "switch", "site": "site-a"}, # site-a wired
        ],
    },
    "list_interfaces": {
        "description": (
            "List all network interfaces for a specific device. "
            "Returns port name, state (up/down), VLAN or frequency, speed, and description. "
            "Requires a valid device_id — use list_devices first if you don't know the ID."
        ),
        "parameters": {
            "device_id": "device identifier from list_devices (e.g. sw-core-01, ap-01, router-01)",
        },
        "returns_large": False,
        "example": {"device_id": "sw-core-01"},
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
    "get_device_config": {
        "description": (
            "Retrieve running configuration from a device. "
            "Mock mode returns realistic seeded config with intentional issues on some APs "
            "(missing NTP, RADIUS timeout, missing ACL). Use section= to narrow output."
        ),
        "parameters": {
            "device_id": "device ID (e.g. 'ap-01', 'sw-core-01')",
            "section":   "config section keyword — e.g. 'radius', 'ntp' (optional)",
        },
        "returns_large": True,
        "example": {"device_id": "ap-01", "section": "radius"},
    },
    "validate_device_config": {
        "description": (
            "Run NTP, RADIUS, ACL, CPU and memory validation checks on a device. "
            "Mock mode returns a deterministic PASS/WARN/FAIL report seeded by device ID."
        ),
        "parameters": {
            "device_id": "device ID (e.g. 'ap-01', 'sw-core-01')",
        },
        "returns_large": False,
        "example": {"device_id": "ap-01"},
    },
}