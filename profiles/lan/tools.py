"""
profiles/lan/tools.py — Enterprise LAN (Cisco) business tool implementations
=============================================================================

Mock tools for enterprise LAN operations: Cisco switches, APs, internal
firewalls. Each is an async callable `async def tool(args: dict) -> str`.

Migrated 2026-05 from the old tools/mock_tools.py as part of the profile
refactor (business tools decoupled from the common framework). The common
paging tools (read_stored_result / process_stored_chunks) stayed in
tools/common_tools.py since every profile needs them.

The prompt-facing metadata for these tools lives in profiles/lan/tool_meta.py;
the Profile object that bundles callables + metadata + skills is in
profiles/lan/__init__.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

# Shared timestamp helper — common framework, used by the mock log generators.
from tools.common_tools import _ts

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# In-memory device state overlay
# ---------------------------------------------------------------------------
# Tracks changes pushed via edit_device_config so subsequent
# validate_device_config / get_device_config calls reflect those changes
# instead of returning the same seed-based baseline forever.
#
# Lifetime: process-lifetime only (resets on restart). For demos this is
# the right granularity — enough state to make a multi-turn fix workflow
# coherent within one session, but no spillover between server restarts.
#
# Schema: {device_id: {state_key: value, ...}}
# Recognised keys (per device type):
#   wireless_ap : radius_timeout (int seconds), ntp_configured (bool),
#                 vlan_acl_applied (bool)
# Add more as additional fixers exercise more keys.
_DEVICE_STATE: dict[str, dict[str, Any]] = {}


def _apply_config_lines_to_state(device_id: str, config_lines: list[str]) -> dict[str, Any]:
    """Parse IOS-style config lines and update _DEVICE_STATE for this device.

    Returns the dict of recognised changes so the caller can echo them in
    the audit log. Lines that aren't recognised are silently logged into a
    `_unparsed_lines` list under the device's overlay so debugging is
    possible without crashing the tool.
    """
    overlay = _DEVICE_STATE.setdefault(device_id, {})
    recognised: dict[str, Any] = {}

    for raw in config_lines:
        line = (raw or "").strip().lower()
        if not line:
            continue

        # radius-server timeout N    →    radius_timeout = N
        if line.startswith("radius-server"):
            import re as _re
            m = _re.search(r"timeout\s+(\d+)", line)
            if m:
                overlay["radius_timeout"] = int(m.group(1))
                recognised["radius_timeout"] = int(m.group(1))
                continue

        # ntp server <ip>    →    ntp_configured = True
        if line.startswith("ntp server "):
            overlay["ntp_configured"] = True
            recognised["ntp_configured"] = True
            continue
        if line.startswith("no ntp server"):
            overlay["ntp_configured"] = False
            recognised["ntp_configured"] = False
            continue

        # access-list / ip access-group — heuristic: any ACL line applied
        if "access-list" in line or "access-group" in line:
            overlay["vlan_acl_applied"] = True
            recognised["vlan_acl_applied"] = True
            continue

        # Unrecognised — track for diagnostics but don't fail
        overlay.setdefault("_unparsed_lines", []).append(raw)

    return recognised


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

    # Apply in-memory state overlay so changes pushed via edit_device_config
    # are reflected in the rendered running-config text (otherwise the LLM
    # asking "what's the current config?" after a fix would see stale data).
    overlay = _DEVICE_STATE.get(device_id, {})
    if "radius_timeout" in overlay:
        radius_timeout = overlay["radius_timeout"]
    if "ntp_configured" in overlay:
        ntp_missing = not overlay["ntp_configured"]
    if "vlan_acl_applied" in overlay:
        vlan_acl_ok = overlay["vlan_acl_applied"]

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
    """Mock: deterministic PASS/WARN/FAIL report seeded by device ID.

    Honours the in-memory device-state overlay populated by previous
    edit_device_config calls in the same process. Without this, the
    seed-based baseline would always report the same warnings even
    after the operator pushed a fix — making post-fix validation
    appear to ignore the change.
    """
    device_id = args.get("device_id", "ap-01")
    await asyncio.sleep(0.03)

    dev = next((d for d in _DEVICE_INVENTORY if d["id"] == device_id), None)
    if not dev:
        return f"[Error: device {device_id!r} not found.]"

    import hashlib as _hs
    seed = int(_hs.md5(device_id.encode()).hexdigest()[:4], 16)
    issues, warnings, passed = [], [], []

    # Apply in-memory overlay so changes pushed via edit_device_config
    # are reflected here. _DEVICE_STATE is keyed by device_id.
    overlay = _DEVICE_STATE.get(device_id, {})

    if dev["type"] == "wireless_ap":
        ntp_missing    = (seed % 4 == 0)
        radius_timeout = 3 + (seed % 4)
        vlan_acl_ok    = (seed % 3 != 1)

        # Overlay overrides — operator's edits take precedence over baseline
        if "radius_timeout" in overlay:
            radius_timeout = overlay["radius_timeout"]
        if "ntp_configured" in overlay:
            ntp_missing = not overlay["ntp_configured"]
        if "vlan_acl_applied" in overlay:
            vlan_acl_ok = overlay["vlan_acl_applied"]

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
    if overlay:
        lines.append(
            f"Note: this device has {len(overlay)} pending in-memory change(s) from "
            f"prior edit_device_config calls (mock state)"
        )
    return "\n".join(lines)





def _coerce_changes(raw: Any) -> dict:
    """Defensive normalisation for the `changes` arg of edit_device_config.

    LLMs produce widely varying shapes for the same intent. Rather than ask
    each LLM to obey a strict schema (brittle), we accept what they produce
    and normalise here. Returns a dict that the rest of the function can
    safely call .get() / .items() on.

    Shape handling:
      None                    → {}
      dict                    → returned as-is
      list of strings         → {"add": [each string]}
      list of dicts           → keys merged shallowly; lists concatenated
      list of mixed           → strings → "add"; dicts merged
      other (str/int/...)     → {"add": [str(raw)]} (last-resort wrap)
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        merged: dict = {}
        for item in raw:
            if isinstance(item, str):
                merged.setdefault("add", []).append(item)
            elif isinstance(item, dict):
                for k, v in item.items():
                    if k in merged and isinstance(merged[k], list) and isinstance(v, list):
                        merged[k].extend(v)
                    elif k in merged and isinstance(merged[k], list):
                        merged[k].append(v)
                    elif k in merged and isinstance(v, list):
                        merged[k] = [merged[k], *v]
                    else:
                        merged[k] = v
            else:
                merged.setdefault("add", []).append(str(item))
        return merged
    # Anything else — string, int, etc — wrap so callers don't crash
    return {"add": [str(raw)]}


def _coerce_config_lines(raw: Any) -> list[str]:
    """Defensive normalisation for the `config_lines` arg.

    Accepts: None, list[str], single str, list of mixed types.
    Returns a list[str] suitable for line-by-line processing.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        # Single line passed directly — wrap in list
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                if item.strip():
                    out.append(item)
            else:
                # Last-resort: stringify (catches dicts, ints, ...)
                out.append(str(item))
        return out
    # Anything else
    return [str(raw)]


# ---------------------------------------------------------------------------
# Tool: edit_device_config  (mock)
# ---------------------------------------------------------------------------

async def edit_device_config(args: dict[str, Any]) -> str:
    """
    Mock: simulate pushing config lines to a device.
    Accepts multiple call formats the LLM may use:

    Format A — explicit IOS commands:
      config_lines: ["radius-server host 10.0.1.100 auth-port 1812 timeout 3"]

    Format B — remove/add dicts (legacy):
      section: "radius", changes: {"remove": ["...old..."], "add": ["...new..."]}

    Format C — key-value changes (what LLMs naturally produce):
      section: "radius", changes: {"timeout": 3, "host": "10.0.1.100"}
      → converted to IOS commands based on section type
    """
    device_id    = args.get("device_id", "")
    config_lines = _coerce_config_lines(args.get("config_lines"))
    section      = args.get("section", "")
    # Defensive coercion: LLMs sometimes pass `changes` as a list, dict, or string.
    # _coerce_changes normalises any of those into the dict shape the logic below assumes.
    changes      = _coerce_changes(args.get("changes"))
    reason       = args.get("reason", "operator change")

    await asyncio.sleep(0.05)

    dev = next((d for d in _DEVICE_INVENTORY if d["id"] == device_id), None)
    if not dev:
        return f"[Error: device {device_id!r} not found. Use list_devices to see valid IDs.]"

    # ── Normalise all call formats into config_lines ──────────────────────
    if not config_lines and changes:
        if "remove" in changes or "add" in changes:
            # Format B: explicit remove/add lists
            for line in changes.get("remove", []):
                config_lines.append(f"no {line}")
            config_lines.extend(changes.get("add", []))
        else:
            # Format C: key-value changes — generate IOS commands per section
            sect = section.lower()
            if sect in ("ntp", "time"):
                servers = changes.get("servers", [])
                for s in servers:
                    config_lines.append(f"ntp server {s}")
                if "timezone" in changes:
                    config_lines.append(f"clock timezone {changes['timezone']}")
            elif sect == "radius" or sect == "aaa":
                host    = changes.get("host", "")
                port    = changes.get("auth_port", changes.get("port", 1812))
                timeout = changes.get("timeout", "")
                key     = changes.get("key", changes.get("secret", ""))
                if host:
                    cmd = f"radius-server host {host} auth-port {port}"
                    if timeout:
                        cmd += f" timeout {timeout}"
                    if key:
                        cmd += f" key {key}"
                    config_lines.append(cmd)
                elif timeout:
                    # timeout-only change — patch existing server entry
                    config_lines.append(f"radius-server timeout {timeout}")
            elif sect in ("syslog", "logging"):
                server = changes.get("server", changes.get("host", ""))
                if server:
                    config_lines.append(f"logging host {server}")
                level = changes.get("level", changes.get("severity", ""))
                if level:
                    config_lines.append(f"logging trap {level}")
            elif sect in ("bgp",):
                as_num = changes.get("as", changes.get("local_as", ""))
                if as_num:
                    config_lines.append(f"router bgp {as_num}")
                for key, val in changes.items():
                    if key not in ("as", "local_as"):
                        config_lines.append(f"  bgp {key} {val}")
            else:
                # Generic: emit "key value" lines for each change
                for key, val in changes.items():
                    if isinstance(val, list):
                        for v in val:
                            config_lines.append(f"{key} {v}")
                    else:
                        config_lines.append(f"{key} {val}")

    # Last-resort: if still empty but args has scalar values, treat them as inline config
    if not config_lines and changes:
        for key, val in changes.items():
            if isinstance(val, (str, int, float)) and key not in ("section", "reason"):
                config_lines.append(f"{key} {val}")

    if not config_lines:
        # Build a helpful error with what was received
        received = f"section={section!r}, changes={changes!r}" if (section or changes) else f"args={args!r}"
        return (f"[Error: no configuration lines could be derived for {device_id}. "
                f"Received: {received}. "
                f"Provide config_lines (list of IOS commands), or section + changes with recognized keys.]")

    # Persist changes into the in-memory state overlay so subsequent
    # validate_device_config / get_device_config calls reflect them.
    # Without this step, the mock validator would always recompute its
    # seed-based baseline and report the original warnings forever, even
    # after the operator approved a fix — making post-HITL workflows
    # appear broken in the demo (the LLM in the next turn correctly sees
    # the validator output and concludes nothing changed).
    applied_state = _apply_config_lines_to_state(device_id, config_lines)
    state_summary = (
        " ; ".join(f"{k}={v}" for k, v in applied_state.items())
        if applied_state else "(no recognised state-affecting lines)"
    )

    # Simulate config push
    lines_applied = "\n".join(f"  {l}" for l in config_lines)
    return (
        f"# Config push result — {device_id} ({dev['model']}) — site {dev['site']}\n"
        f"# {'─'*60}\n"
        f"# Reason: {reason}\n"
        f"# Section: {section or 'global'}\n"
        f"# Lines applied ({len(config_lines)}):\n"
        f"{lines_applied}\n"
        f"# {'─'*60}\n"
        f"# Result: Configuration applied successfully (mock)\n"
        f"# Device acknowledged: OK\n"
        f"# Mock state updated: {state_summary}\n"
        f"# Note: run validate_device_config to verify the change took effect"
    )


# ---------------------------------------------------------------------------
# HITL skill-only tools — mock implementations
# These are registered in TOOL_REGISTRY so stop_hitl can replay them after
# operator approval. In production, replace with real Kubernetes/API calls.
# ---------------------------------------------------------------------------

async def restart_service(args: dict[str, Any]) -> str:
    """Mock: simulate a rolling service restart (HITL-required)."""
    service     = args.get("service", args.get("service_name", "unknown-service"))
    environment = args.get("environment", args.get("env", "prod"))
    rolling     = args.get("rolling", True)
    reason      = args.get("reason", "operator-initiated restart")
    await asyncio.sleep(0.1)
    strategy = "rolling" if rolling else "full"
    return (
        f"# Service restart — {service} ({environment}) — {strategy}\n"
        f"# {'─'*60}\n"
        f"# Reason: {reason}\n"
        f"  Pods desired: 3  |  updated: 3  |  ready: 3  |  available: 3\n"
        f"# Status: Rollout complete — all pods healthy\n"
        f"# Health check: PASS (200 OK on /healthz)\n"
        f"# Note: Monitor logs for 5 minutes post-restart"
    )


async def rollback_service(args: dict[str, Any]) -> str:
    """Mock: simulate a service rollback to a previous version (HITL-required)."""
    service     = args.get("service", args.get("service_name", "unknown-service"))
    version     = args.get("version", args.get("target_version", "previous"))
    environment = args.get("environment", args.get("env", "prod"))
    reason      = args.get("reason", "operator-initiated rollback")
    await asyncio.sleep(0.1)
    return (
        f"# Service rollback — {service} → {version} ({environment})\n"
        f"# {'─'*60}\n"
        f"# Reason: {reason}\n"
        f"  Pods rolled back: 3  |  ready: 3  |  available: 3\n"
        f"# Status: Rollback complete — running {version}\n"
        f"# Health check: PASS (200 OK on /healthz)\n"
        f"# Note: Monitor 10 minutes post-rollback; verify functionality manually"
    )


async def push_config(args: dict[str, Any]) -> str:
    """Mock: simulate pushing config to a device (HITL-required)."""
    device_id  = args.get("device_id", "unknown")
    config_text = args.get("config_text", "")
    dry_run    = bool(args.get("dry_run", False))
    await asyncio.sleep(0.1)
    n_lines = len([l for l in config_text.split("\n") if l.strip()]) or 12
    mode = "DRY RUN" if dry_run else "APPLIED"
    return (
        f"# push_config — {device_id} ({mode})\n"
        f"# {'─'*60}\n"
        f"  Config lines processed: {n_lines}\n"
        f"  Errors: 0  |  Warnings: 0\n"
        f"# Status: {'Validation complete — no changes written' if dry_run else 'Applied to running-config'}\n"
        f"# Note: Diff vs startup-config available via diff_device_config"
    )


async def rollback_deploy(args: dict[str, Any]) -> str:
    """Mock: simulate rolling back a deploy (HITL-required)."""
    deploy_id = args.get("deploy_id", "unknown")
    scope     = args.get("scope", "all")
    await asyncio.sleep(0.1)
    return (
        f"# rollback_deploy — {deploy_id} (scope={scope})\n"
        f"# {'─'*60}\n"
        f"  Services reverted: 3\n"
        f"  Pods restarted: 9\n"
        f"# Status: Rollback complete — running previous stable version\n"
        f"# Health check: PASS"
    )


async def drain_node(args: dict[str, Any]) -> str:
    """Mock: simulate draining a node (HITL-required)."""
    node_id    = args.get("node_id", "unknown")
    grace_s    = int(args.get("grace_period_s", 60))
    await asyncio.sleep(0.1)
    return (
        f"# drain_node — {node_id} (grace={grace_s}s)\n"
        f"# {'─'*60}\n"
        f"  Workloads evicted: 7\n"
        f"  Pending: 0  |  Failed: 0\n"
        f"# Status: Node cordoned and drained — non-schedulable\n"
        f"# Note: Re-enable with `uncordon` after maintenance"
    )


async def failover(args: dict[str, Any]) -> str:
    """Mock: simulate triggering failover to standby (HITL-required)."""
    resource = args.get("resource_id", "unknown")
    target   = args.get("target", "auto-selected-replica")
    await asyncio.sleep(0.1)
    return (
        f"# failover — {resource} → {target}\n"
        f"# {'─'*60}\n"
        f"  Pre-failover writes drained: yes\n"
        f"  Replication lag at failover: 0.2s\n"
        f"# Status: Failover complete — {target} is now primary\n"
        f"# Health check: PASS  |  Recommendation: monitor 30 min"
    )


async def delete_resource(args: dict[str, Any]) -> str:
    """Mock: simulate deleting a resource (HITL-required)."""
    resource = args.get("resource_id", "unknown")
    force    = bool(args.get("force", False))
    await asyncio.sleep(0.1)
    return (
        f"# delete_resource — {resource} (force={force})\n"
        f"# {'─'*60}\n"
        f"  Dependencies checked: {'skipped (force=True)' if force else '0 dependents found'}\n"
        f"# Status: Resource deleted\n"
        f"# Note: Operation is irreversible without backup"
    )


async def diff_device_config(args: dict[str, Any]) -> str:
    """Mock: show what changed in a device config since last known-good state."""
    device_id = args.get("device_id", "")
    section   = args.get("section", "")
    
    dev = next((d for d in _DEVICE_INVENTORY if d["id"] == device_id), None)
    if not dev:
        return f"[Error: device {device_id!r} not found.]"
    
    await asyncio.sleep(0.05)
    sect_label = f" [{section}]" if section else ""
    return (
        f"# Config diff — {device_id} ({dev['model']}){sect_label}\n"
        f"# Compared: running-config vs startup-config\n"
        f"# {'─'*60}\n"
        f"  No uncommitted changes detected.\n"
        f"  running-config matches startup-config.\n"
        f"# Last write: within last maintenance window"
    )

# ---------------------------------------------------------------------------
# query_radius_logs — H2 (async fire-and-forget) HITL demo (2026-05)
# ---------------------------------------------------------------------------
#
# Demonstrates async-nonblocking HITL:
#   1. Tool registers an async HITL via register_async_pending() (inserts
#      into the router registry under lock AND arms the SLA watchdog) and
#      creates a real checkpoint that /hitl/pending lists.
#   2. Returns immediately with default "permission_ok (assumed)".
#   3. After a random delay (3-12s in demo mode), a background task calls
#      router.deliver(...) directly to simulate "ops queue ack arrived".
#      In production this spawn does not exist — a real operator clicks and
#      router.deliver() is invoked via the HTTP endpoint instead.
#
# Robust to wiring failures: any error in the H2 setup (schema mismatch,
# audit/registry write, demo autoreply spawn) is caught and the tool
# still returns a usable degraded response. This prevents the LLM from
# looping retrying on a tool error (observed on initial deploy where a
# ProposedAction field mismatch raised pydantic ValidationError and the
# LLM retried the same call 3 times before exhausting the turn cap).

_RADIUS_DEMO_OUTCOMES = [
    # (decision, comment, weight) — weighted random
    ("approve", "RADIUS check passed: user has Wi-Fi access permission",        7),
    ("reject",  "RADIUS check FAILED: user X is DISABLED in directory",         2),
    ("reject",  "RADIUS check FAILED: user X account locked (5 failed attempts)", 1),
]


async def query_radius_logs(args: dict[str, Any]) -> str:
    """H2 demo: fire RADIUS query, return assumed default, ack comes later.

    Args:
        user_id: user to look up
        minutes: time window in minutes (default 60)
        # injected by runtime/loop._execute_tool:
        _session_id: current session_id (for SSE notify routing)
    """
    user_id  = str(args.get("user_id") or "unknown_user").strip()
    minutes  = int(args.get("minutes") or 60)
    session_id = args.get("_session_id") or ""

    # ── Lazy import of hitl_core / runtime / webui / main ────────────────
    # tools/ MUST NOT import hitl_core in general. This is the one allowed
    # exception for the H2 async-HITL DEMO that explicitly integrates with
    # the async-HITL extension point.
    # ALLOWED BY DESIGN: H2 async-HITL demo (single integration point)
    try:
        import uuid as _uuid                                              # noqa: E501
        import random as _random                                          # noqa: E501
        from hitl_core import HitlRouter
        from hitl_core.schema import (
            HitlPayload, TriggerKind, RiskLevel, InterruptMode,
            ProposedAction, HitlDecision, DecisionKind, ClarificationField,
            CheckpointEntry, ResumeHandle,
            AuditEventKind, HitlAuditRecord,
        )
        from hitl_core.pipeline import AsyncPendingHitl
        from runtime.loop import enqueue_async_inject
    except Exception as _imp_exc:
        # If imports fail (test env / partial install), fall back to a
        # functional read-only stub so the LLM gets a useful result.
        return (
            f"# query_radius_logs (H2 DEMO — imports failed, degraded mode)\n"
            f"# user_id:  {user_id}\n"
            f"# window:   last {minutes} min\n"
            f"# result:   permission_ok (assumed)\n"
            f"# note:     H2 async-HITL wiring not available in this env\n"
            f"# error:    {_imp_exc}"
        )

    # Try to get the global router/store via the service registry.
    try:
        from main import _services as _global_services                    # type: ignore
    except Exception:
        _global_services = None                                           # type: ignore

    # The active backend is hitl_core: the real router/audit live under the
    # hitl_core_* keys. The bare hitl_router/hitl_audit keys are retained as
    # stub-None for legacy safety, so resolve core-first then fall back.
    _router = None
    if _global_services:
        _router = (_global_services.get("hitl_core_router")
                   or _global_services.get("hitl_router"))
    if not _global_services or _router is None:
        return (
            f"# query_radius_logs (H2 DEMO — degraded, no router wired)\n"
            f"# user_id:  {user_id}\n"
            f"# window:   last {minutes} min\n"
            f"# result:   permission_ok (assumed; H2 cannot fire here)"
        )

    router: HitlRouter = _router
    store = _global_services.get("hitl_store")
    if store is None:
        return (
            f"# query_radius_logs (H2 DEMO — no store available)\n"
            f"# user_id:  {user_id}\n"
            f"# result:   permission_ok (assumed)"
        )

    # ── H2 setup wrapped in try/except so wiring bugs degrade gracefully ──
    # If anything below raises, the LLM sees a degraded but usable response
    # instead of a tool error, avoiding retry loops.
    try:
        interrupt_id = str(_uuid.uuid4())

        # Build the H2 payload. Operator-facing card explains the situation.
        payload = HitlPayload(
            interrupt_id   = interrupt_id,
            thread_id      = session_id or "demo",
            context_id     = session_id or "demo",
            title          = f"RADIUS auth check for {user_id}",
            description    = (
                f"Async HITL: verify {user_id}'s RADIUS permission. "
                f"Agent will continue with assumed default 'permission_ok' "
                f"while you check. Reply within 3 minutes — anything later "
                f"will surface as a follow-up note."
            ),
            # ProposedAction needs action_type + target (NOT tool_name/tool_args).
            # See hitl_core/schema.py:170 — this is the domain-neutral
            # representation. action_type is free-form; "tool_call:<name>"
            # is the convention for tool-driven proposals.
            proposed_action = ProposedAction(
                action_type = "tool_call:query_radius_logs",
                target      = user_id,
                parameters  = {"user_id": user_id, "minutes": minutes},
                risk_level  = RiskLevel.LOW,
                reversible  = True,
            ),
            trigger_kind   = TriggerKind.EXTERNAL_DELEGATION,
            risk_level     = RiskLevel.LOW,
            interrupt_mode = InterruptMode.ASYNC_NONBLOCKING,
            sla_seconds    = 180,                                # 3 min
            clarification_fields = [
                ClarificationField(
                    key   = "actual_status",
                    prompt= "Actual RADIUS status for this user",
                    # Operator sees these as radio options if FE renders them.
                    allowed_values = ["permission_ok", "permission_denied", "account_locked"],
                ),
            ],
        )

        # Persist as a real checkpoint so /hitl/pending sees it.
        entry = CheckpointEntry(
            interrupt_id  = interrupt_id,
            payload       = payload,
            resume_handle = ResumeHandle(resumer_name="async_hitl", state={}),
        )
        await store.save(entry)

        # Build on_resolved closure — writes a confirmed_fact + SSE notify.
        default_value = "permission_ok"

        async def _on_resolved(iid: str, decision, default, diverged: bool) -> None:
            # decision is None on timeout, else HitlDecision
            if decision is None:
                fact = (
                    f"[ASYNC_HITL/radius:{iid[:8]}] "
                    f"RADIUS check for user={user_id}: NO RESPONSE after "
                    f"{payload.sla_seconds}s; agent proceeded with default "
                    f"'{default}'."
                )
                outcome_label = "timeout"
            else:
                outcome_label = decision.decision.value
                comment       = (decision.comment or "").strip()
                answer        = (decision.clarification_answers or {}).get("actual_status", "")
                if diverged:
                    fact = (
                        f"[ASYNC_HITL/radius:{iid[:8]}] "
                        f"RADIUS check for user={user_id}: result DIVERGES from "
                        f"assumption. Operator decided '{outcome_label}'"
                        + (f" (actual={answer})" if answer else "")
                        + (f" — {comment}" if comment else "")
                    )
                else:
                    fact = (
                        f"[ASYNC_HITL/radius:{iid[:8]}] "
                        f"RADIUS check for user={user_id} CONFIRMED "
                        f"'{default}'"
                        + (f" — {comment}" if comment else "")
                    )

            # Inject into next turn's confirmed_facts via runtime inject queue.
            try:
                enqueue_async_inject(session_id, fact)
            except Exception as _inj_exc:
                logger.warning("async H2: enqueue_async_inject failed: %s", _inj_exc)

            # Soft-notify the active SSE stream so operator sees it immediately.
            try:
                from webui.backend import emit_async_hitl_notify
                emit_async_hitl_notify(session_id, {
                    "type":            "async_hitl_resolved",
                    "interrupt_id":    iid,
                    "tool":            "query_radius_logs",
                    "user_id":         user_id,
                    "outcome":         outcome_label,
                    "diverged":        diverged,
                    "default_value":   default,
                    "fact":            fact,
                })
            except Exception as _em_exc:
                logger.warning("async H2: emit_async_hitl_notify failed: %s", _em_exc)

        # Register pending via the unified helper so it (a) inserts under
        # the registry lock and (b) arms the SLA watchdog. Previously the
        # tool inserted into _async_registry directly with no timer, so
        # when _demo_autoreply was disabled and no operator decided, the
        # entry leaked forever and on_resolved(None) never fired (Bug 2).
        try:
            from hitl_core.router import register_async_pending
            # Adapt the audit service (.log(HitlAuditRecord)) to the watchdog's
            # on_audit(kind, iid, detail) shape so ASYNC_TIMEOUT is recorded.
            _audit_svc = _global_services.get("hitl_core_audit") or _global_services.get("hitl_audit")
            async def _audit_adapter(kind, iid, detail):
                if _audit_svc is None:
                    return
                from datetime import datetime as _dt2, timezone as _tz2
                await _audit_svc.log(HitlAuditRecord(
                    interrupt_id = iid, kind = kind,
                    detail = detail, timestamp = _dt2.now(_tz2.utc),
                ))
            register_async_pending(
                AsyncPendingHitl(
                    interrupt_id   = interrupt_id,
                    payload        = payload,
                    default_value  = default_value,
                    on_resolved    = _on_resolved,
                    divergence_check = None,
                    sla_seconds    = payload.sla_seconds,
                    session_id     = session_id,
                ),
                store=store,
                on_audit=_audit_adapter,
            )
        except Exception as _reg_exc:
            logger.warning("async H2: registry insert failed: %s", _reg_exc)

        # Audit ASYNC_DELEGATED so the audit timeline reflects fire moment.
        try:
            from datetime import datetime as _dt, timezone as _tz
            audit = _global_services.get("hitl_core_audit") or _global_services.get("hitl_audit")
            if audit is not None:
                await audit.log(HitlAuditRecord(
                    interrupt_id = interrupt_id,
                    kind         = AuditEventKind.ASYNC_DELEGATED,
                    detail       = {
                        "tool":         "query_radius_logs",
                        "user_id":      user_id,
                        "session_id":   session_id,
                        "sla_seconds":  payload.sla_seconds,
                    },
                    timestamp = _dt.now(_tz.utc),
                ))
        except Exception as _aud_exc:
            logger.debug("async H2: audit ASYNC_DELEGATED failed: %s", _aud_exc)

        # ── Demo-only: auto-respond after a random delay so the flow is
        # observable without an actual operator clicking. In production
        # this spawn would not exist — real operators click and
        # router.deliver() gets called via the HTTP endpoint.
        if str(args.get("_demo_autoreply", "1")).lower() not in ("0", "false", "no"):
            async def _demo_autoreply() -> None:
                await asyncio.sleep(_random.uniform(3.0, 12.0))
                # Pick weighted outcome
                outcomes_flat = []
                for o, c, w in _RADIUS_DEMO_OUTCOMES:
                    outcomes_flat.extend([(o, c)] * w)
                kind, comment = _random.choice(outcomes_flat)
                decision = HitlDecision(
                    interrupt_id = interrupt_id,
                    decision     = DecisionKind.APPROVE if kind == "approve" else DecisionKind.REJECT,
                    operator_id  = "demo_auto_responder",
                    comment      = comment,
                    clarification_answers = {
                        "actual_status": (
                            "permission_ok"     if kind == "approve" else
                            "account_locked"    if "locked" in comment.lower() else
                            "permission_denied"
                        ),
                    },
                )
                try:
                    await router.deliver(decision)
                except Exception as _del_exc:
                    logger.warning("query_radius_logs demo autoreply failed: %s", _del_exc)
            asyncio.create_task(
                _demo_autoreply(),
                name=f"radius_demo_autoreply_{interrupt_id[:12]}",
            )

        # Successful setup — return the H2 fire ack to the LLM.
        return (
            f"# query_radius_logs — RADIUS auth check pushed to ops queue\n"
            f"# {'─'*60}\n"
            f"  user_id:        {user_id}\n"
            f"  window:         last {minutes} min\n"
            f"  interrupt_id:   {interrupt_id}\n"
            f"  assumed_result: permission_ok (proceeding without blocking)\n"
            f"#\n"
            f"# This is an ASYNC HITL — your job is now done for this query.\n"
            f"# The actual ops result will arrive within ~3 minutes via:\n"
            f"#   1. Soft notification in this chat (🔔 banner appears)\n"
            f"#   2. confirmed_fact in next turn (LLM auto-uses it)\n"
            f"#\n"
            f"# DO NOT call query_radius_logs again for this user — give your\n"
            f"# best answer based on the assumed result and stop. If the ops\n"
            f"# result DIVERGES from 'permission_ok' later, the operator can\n"
            f"# choose to re-ask you with the new fact."
        )

    except Exception as _setup_exc:
        # ANY H2 setup failure ⇒ degraded fallback. Keeps the LLM from
        # treating this as a generic tool error and retrying.
        logger.warning(
            "query_radius_logs H2 setup failed for user=%s: %s",
            user_id, _setup_exc,
        )
        return (
            f"# query_radius_logs (H2 DEMO — setup failed, degraded mode)\n"
            f"# user_id:  {user_id}\n"
            f"# window:   last {minutes} min\n"
            f"# result:   permission_ok (assumed)\n"
            f"# note:     H2 wiring failed; no async approval will arrive.\n"
            f"# error:    {_setup_exc}\n"
            f"#\n"
            f"# Proceed with the assumed result and answer the user. Do NOT\n"
            f"# call query_radius_logs again."
        )

