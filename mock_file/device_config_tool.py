"""
device_config_tool.py
---------------------
Uploadable mock tool for device configuration management.

Upload this via: WebUI → left panel → UPLOAD → tool (.py) → drag & drop

After upload the agent can call:
  [TOOL:get_device_config] {"device_id": "sw-core-01"}
  [TOOL:get_device_config] {"device_id": "router-01", "section": "bgp"}
  [TOOL:edit_device_config] {"device_id": "ap-01", "section": "radius", "changes": {"server": "10.0.1.100", "port": 1812}}
  [TOOL:validate_device_config] {"device_id": "sw-core-01"}
  [TOOL:diff_device_config] {"device_id": "router-01"}
"""

import asyncio
import copy
import json
import time
from typing import Any

# ---------------------------------------------------------------------------
# Simulated config store
# Mirrors the 13 devices in mock_tools._DEVICE_INVENTORY
# ---------------------------------------------------------------------------

_BASE_CONFIGS: dict[str, dict] = {
    "sw-core-01": {
        "hostname":    "sw-core-01",
        "model":       "Cisco Catalyst 9500-48Y4C",
        "management":  {"ip": "10.0.1.1", "mask": "255.255.255.0", "gateway": "10.0.1.254"},
        "vlans":       {1: "default", 10: "management", 20: "users", 100: "uplinks"},
        "spanning_tree": {"mode": "rapid-pvst", "priority": 4096},
        "ntp":         {"servers": ["10.0.1.50", "10.0.1.51"], "timezone": "UTC+8"},
        "syslog":      {"server": "10.0.1.200", "level": "informational"},
        "aaa": {
            "radius_server": "10.0.1.100",
            "radius_key":    "**MASKED**",
            "auth_port":     1812,
            "acct_port":     1813,
        },
    },
    "sw-core-02": {
        "hostname":    "sw-core-02",
        "model":       "Cisco Catalyst 9500-48Y4C",
        "management":  {"ip": "10.0.2.1", "mask": "255.255.255.0", "gateway": "10.0.2.254"},
        "vlans":       {1: "default", 10: "management", 20: "users", 100: "uplinks"},
        "spanning_tree": {"mode": "rapid-pvst", "priority": 8192},
        "ntp":         {"servers": ["10.0.2.50", "10.0.2.51"], "timezone": "UTC+8"},
        "syslog":      {"server": "10.0.2.200", "level": "informational"},
        "aaa": {
            "radius_server": "10.0.2.100",
            "radius_key":    "**MASKED**",
            "auth_port":     1812,
            "acct_port":     1813,
        },
    },
    "sw-acc-01": {
        "hostname":    "sw-acc-01",
        "model":       "Cisco Catalyst 9300-48P",
        "management":  {"ip": "10.0.1.21", "mask": "255.255.255.0", "gateway": "10.0.1.1"},
        "vlans":       {1: "default", 10: "management", 20: "users"},
        "spanning_tree": {"mode": "rapid-pvst", "priority": 32768},
        "ntp":         {"servers": ["10.0.1.1"], "timezone": "UTC+8"},
        "syslog":      {"server": "10.0.1.200", "level": "warnings"},
        "aaa": {
            "radius_server": "10.0.1.100",
            "radius_key":    "**MASKED**",
            "auth_port":     1812,
            "acct_port":     1813,
        },
    },
    "sw-acc-02": {
        "hostname":    "sw-acc-02",
        "model":       "Cisco Catalyst 9300-48P",
        "management":  {"ip": "10.0.1.22", "mask": "255.255.255.0", "gateway": "10.0.1.1"},
        "vlans":       {1: "default", 10: "management", 20: "users"},
        "spanning_tree": {"mode": "rapid-pvst", "priority": 32768},
        "ntp":         {"servers": ["10.0.1.1"], "timezone": "UTC+8"},
        "syslog":      {"server": "10.0.1.200", "level": "warnings"},
        "aaa": {
            "radius_server": "10.0.1.100",
            "radius_key":    "**MASKED**",
            "auth_port":     1812,
            "acct_port":     1813,
        },
    },
    "sw-acc-03": {
        "hostname":    "sw-acc-03",
        "model":       "Cisco Catalyst 9300-48P",
        "management":  {"ip": "10.0.2.21", "mask": "255.255.255.0", "gateway": "10.0.2.1"},
        "vlans":       {1: "default", 10: "management", 20: "users"},
        "spanning_tree": {"mode": "rapid-pvst", "priority": 32768},
        "ntp":         {"servers": ["10.0.2.1"], "timezone": "UTC+8"},
        "syslog":      {"server": "10.0.2.200", "level": "warnings"},
        "aaa": {
            "radius_server": "10.0.2.100",
            "radius_key":    "**MASKED**",
            "auth_port":     1812,
            "acct_port":     1813,
        },
    },
    "router-01": {
        "hostname":    "router-01",
        "model":       "Cisco ASR 1001-X",
        "management":  {"ip": "10.0.1.254", "mask": "255.255.255.0"},
        "interfaces": {
            "Gi0/0": {"ip": "10.0.1.254", "desc": "LAN", "state": "up"},
            "Gi0/1": {"ip": "203.0.113.1", "desc": "WAN primary", "state": "up"},
            "Gi0/2": {"ip": "198.51.100.1", "desc": "WAN backup", "state": "up"},
        },
        "bgp": {
            "as":      65001,
            "router_id": "10.0.1.254",
            "neighbors": [
                {"ip": "203.0.113.254", "remote_as": 65100, "desc": "ISP-A"},
                {"ip": "198.51.100.254", "remote_as": 65200, "desc": "ISP-B"},
            ],
        },
        "ntp":    {"servers": ["203.0.113.10"], "timezone": "UTC+8"},
        "syslog": {"server": "10.0.1.200", "level": "informational"},
    },
    "router-02": {
        "hostname":    "router-02",
        "model":       "Cisco ASR 1001-X",
        "management":  {"ip": "10.0.2.254", "mask": "255.255.255.0"},
        "interfaces": {
            "Gi0/0": {"ip": "10.0.2.254", "desc": "LAN", "state": "up"},
            "Gi0/1": {"ip": "203.0.114.1", "desc": "WAN primary", "state": "up"},
            "Gi0/2": {"ip": "198.51.101.1", "desc": "WAN backup", "state": "down"},
        },
        "bgp": {
            "as":      65002,
            "router_id": "10.0.2.254",
            "neighbors": [
                {"ip": "203.0.114.254", "remote_as": 65100, "desc": "ISP-A"},
                {"ip": "198.51.101.254", "remote_as": 65200, "desc": "ISP-B"},
            ],
        },
        "ntp":    {"servers": ["203.0.114.10"], "timezone": "UTC+8"},
        "syslog": {"server": "10.0.2.200", "level": "informational"},
    },
    "ap-01": {
        "hostname":    "ap-01",
        "model":       "Cisco Catalyst 9115AXI",
        "management":  {"ip": "10.0.1.11", "mask": "255.255.255.0", "gateway": "10.0.1.1"},
        "radius": {
            "server":    "10.0.1.100",
            "auth_port": 1812,
            "acct_port": 1813,
            "timeout":   5,
            "retries":   3,
        },
        "ssids": {
            "corp-wifi":    {"vlan": 20, "auth": "802.1x", "band": "dual"},
            "corp-wifi-5g": {"vlan": 20, "auth": "802.1x", "band": "5GHz"},
        },
        "radio": {"power_2g": 20, "power_5g": 20, "channel_2g": 6, "channel_5g": 36},
    },
    "ap-02": {
        "hostname":    "ap-02",
        "model":       "Cisco Catalyst 9115AXI",
        "management":  {"ip": "10.0.1.12", "mask": "255.255.255.0", "gateway": "10.0.1.1"},
        "radius": {
            "server":    "10.0.1.100",
            "auth_port": 1812,
            "acct_port": 1813,
            "timeout":   5,
            "retries":   3,
        },
        "ssids": {
            "corp-wifi":    {"vlan": 20, "auth": "802.1x", "band": "dual"},
            "corp-wifi-5g": {"vlan": 20, "auth": "802.1x", "band": "5GHz"},
        },
        "radio": {"power_2g": 20, "power_5g": 20, "channel_2g": 1, "channel_5g": 40},
    },
    "ap-03": {
        "hostname":    "ap-03",
        "model":       "Cisco Catalyst 9130AXI",
        "management":  {"ip": "10.0.2.11", "mask": "255.255.255.0", "gateway": "10.0.2.1"},
        "radius": {
            "server":    "10.0.2.100",
            "auth_port": 1812,
            "acct_port": 1813,
            "timeout":   5,
            "retries":   3,
        },
        "ssids": {
            "corp-wifi":    {"vlan": 20, "auth": "802.1x", "band": "dual"},
            "corp-wifi-5g": {"vlan": 20, "auth": "802.1x", "band": "5GHz"},
        },
        "radio": {"power_2g": 18, "power_5g": 23, "channel_2g": 11, "channel_5g": 44},
    },
    "ap-04": {
        "hostname":    "ap-04",
        "model":       "Cisco Catalyst 9130AXI",
        "management":  {"ip": "10.0.2.12", "mask": "255.255.255.0", "gateway": "10.0.2.1"},
        "radius": {
            "server":    "10.0.2.100",
            "auth_port": 1812,
            "acct_port": 1813,
            "timeout":   5,
            "retries":   3,
        },
        "ssids": {
            "corp-wifi":    {"vlan": 20, "auth": "802.1x", "band": "dual"},
            "corp-wifi-5g": {"vlan": 20, "auth": "802.1x", "band": "5GHz"},
        },
        "radio": {"power_2g": 18, "power_5g": 23, "channel_2g": 6, "channel_5g": 149},
    },
    "radius-01": {
        "hostname":    "radius-01",
        "model":       "Linux VM / FreeRADIUS 3.2",
        "management":  {"ip": "10.0.1.100", "mask": "255.255.255.0", "gateway": "10.0.1.1"},
        "radius": {
            "auth_port":   1812,
            "acct_port":   1813,
            "clients": [
                {"ip": "10.0.1.11", "desc": "ap-01"},
                {"ip": "10.0.1.12", "desc": "ap-02"},
                {"ip": "10.0.1.1",  "desc": "sw-core-01"},
                {"ip": "10.0.1.21", "desc": "sw-acc-01"},
                {"ip": "10.0.1.22", "desc": "sw-acc-02"},
            ],
            "tls": {
                "cert_file":    "/etc/freeradius/certs/server.pem",
                "key_file":     "/etc/freeradius/certs/server.key",
                "ca_file":      "/etc/freeradius/certs/ca.pem",
                "cert_expires": "2026-06-30",
            },
        },
        "ldap": {
            "server":   "10.0.1.150",
            "port":     636,
            "base_dn":  "dc=corp,dc=internal",
            "bind_dn":  "cn=radius,ou=svc,dc=corp,dc=internal",
        },
    },
    "radius-02": {
        "hostname":    "radius-02",
        "model":       "Linux VM / FreeRADIUS 3.2",
        "management":  {"ip": "10.0.2.100", "mask": "255.255.255.0", "gateway": "10.0.2.1"},
        "radius": {
            "auth_port":   1812,
            "acct_port":   1813,
            "clients": [
                {"ip": "10.0.2.11", "desc": "ap-03"},
                {"ip": "10.0.2.12", "desc": "ap-04"},
                {"ip": "10.0.2.1",  "desc": "sw-core-02"},
                {"ip": "10.0.2.21", "desc": "sw-acc-03"},
            ],
            "tls": {
                "cert_file":    "/etc/freeradius/certs/server.pem",
                "key_file":     "/etc/freeradius/certs/server.key",
                "ca_file":      "/etc/freeradius/certs/ca.pem",
                "cert_expires": "2026-07-15",
            },
        },
        "ldap": {
            "server":   "10.0.2.150",
            "port":     636,
            "base_dn":  "dc=corp,dc=internal",
            "bind_dn":  "cn=radius,ou=svc,dc=corp,dc=internal",
        },
    },
}

# Runtime edit store — accumulates changes made via edit_device_config
_EDITED_CONFIGS: dict[str, dict] = {}

# Audit log of config changes
_CHANGE_LOG: list[dict] = []


def _get_live_config(device_id: str) -> dict | None:
    """Return merged base + edits for a device."""
    base = _BASE_CONFIGS.get(device_id)
    if base is None:
        return None
    live = copy.deepcopy(base)
    edits = _EDITED_CONFIGS.get(device_id, {})
    for section, values in edits.items():
        if section in live and isinstance(live[section], dict) and isinstance(values, dict):
            live[section].update(values)
        else:
            live[section] = values
    return live


def _fmt_config(cfg: dict, section: str | None = None) -> str:
    """Format config as readable indented text."""
    target = cfg.get(section, {}) if section else cfg
    if not target:
        return f"[No '{section}' section found in config]"
    lines = [f"# Device: {cfg.get('hostname', '?')}  Model: {cfg.get('model', '?')}"]
    if section:
        lines.append(f"# Section: {section}")
    lines.append("# " + "─" * 60)

    def _render(obj: Any, indent: int = 0) -> list[str]:
        pad = "  " * indent
        out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    out.append(f"{pad}{k}:")
                    out.extend(_render(v, indent + 1))
                else:
                    out.append(f"{pad}{k}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    out.append(f"{pad}-")
                    out.extend(_render(item, indent + 1))
                else:
                    out.append(f"{pad}- {item}")
        else:
            out.append(f"{pad}{obj}")
        return out

    lines.extend(_render(target))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1: get_device_config
# ---------------------------------------------------------------------------

async def get_device_config(args: dict[str, Any]) -> str:
    """
    Retrieve the running configuration of a network device.

    Args:
        device_id: device identifier (e.g. sw-core-01, router-01, ap-01, radius-01)
        section:   optional config section to retrieve:
                   management | vlans | spanning_tree | ntp | syslog | aaa |
                   radius | bgp | interfaces | ssids | radio | ldap
                   Leave empty to get the full config.
    """
    await asyncio.sleep(0.02)
    device_id = args.get("device_id", "").lower().strip()
    section   = args.get("section", "").lower().strip() or None

    if not device_id:
        return "[Error: device_id is required. Use list_devices to see available device IDs.]"

    cfg = _get_live_config(device_id)
    if cfg is None:
        return (
            f"[Error: device '{device_id}' not found in config store.\n"
            f" Available: {', '.join(sorted(_BASE_CONFIGS.keys()))}]"
        )

    has_edits = device_id in _EDITED_CONFIGS
    header = f"# Config: {device_id}  {'[HAS PENDING EDITS]' if has_edits else '[base config]'}"

    if section:
        if section not in cfg:
            available = ", ".join(k for k in cfg if k not in ("hostname", "model"))
            return (
                f"[Error: section '{section}' not found for {device_id}.\n"
                f" Available sections: {available}]"
            )
        return header + "\n" + _fmt_config(cfg, section)

    return header + "\n" + _fmt_config(cfg)


# ---------------------------------------------------------------------------
# Tool 2: edit_device_config
# ---------------------------------------------------------------------------

async def edit_device_config(args: dict[str, Any]) -> str:
    """
    Apply configuration changes to a device (in-memory — simulates a config push).

    Args:
        device_id: device identifier
        section:   config section to edit (e.g. radius, ntp, bgp, aaa, syslog)
        changes:   dict of key-value pairs to update within that section
                   Example: {"server": "10.0.1.101", "auth_port": 1812}
        reason:    optional change reason for the audit log

    The change is applied immediately and reflected in subsequent get_device_config calls.
    Use diff_device_config to see all pending changes.
    Use validate_device_config to check the result for errors.
    """
    await asyncio.sleep(0.03)
    device_id = args.get("device_id", "").lower().strip()
    section   = args.get("section", "").lower().strip()
    changes   = args.get("changes", {})
    reason    = args.get("reason", "operator edit")

    if not device_id:
        return "[Error: device_id is required]"
    if not section:
        return "[Error: section is required (e.g. radius, ntp, aaa, bgp)]"
    if not changes or not isinstance(changes, dict):
        return "[Error: changes must be a non-empty dict of key-value pairs]"

    cfg = _get_live_config(device_id)
    if cfg is None:
        return f"[Error: device '{device_id}' not found]"
    if section not in cfg:
        available = ", ".join(k for k in cfg if k not in ("hostname", "model"))
        return f"[Error: section '{section}' not found. Available: {available}]"

    # Apply edits
    if device_id not in _EDITED_CONFIGS:
        _EDITED_CONFIGS[device_id] = {}
    if section not in _EDITED_CONFIGS[device_id]:
        _EDITED_CONFIGS[device_id][section] = {}
    _EDITED_CONFIGS[device_id][section].update(changes)

    # Audit log
    _CHANGE_LOG.append({
        "ts":        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device_id": device_id,
        "section":   section,
        "changes":   changes,
        "reason":    reason,
    })

    # Show what changed
    live = _get_live_config(device_id)
    new_section = live.get(section, {})
    lines = [
        f"# Config edit applied: {device_id}/{section}",
        f"# Reason: {reason}",
        f"# Timestamp: {_CHANGE_LOG[-1]['ts']}",
        "# Changes applied:",
    ]
    for k, v in changes.items():
        old_val = cfg.get(section, {}).get(k, "<new key>")
        lines.append(f"#   {k}: {old_val!r} → {v!r}")
    lines.append("# Updated section:")
    lines.append(_fmt_config({"hostname": device_id, "model": ""}, None).split("\n")[0])
    for k, v in new_section.items():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: validate_device_config
# ---------------------------------------------------------------------------

async def validate_device_config(args: dict[str, Any]) -> str:
    """
    Validate a device's current configuration against known-good rules.
    Returns a list of errors, warnings, and recommendations.

    Args:
        device_id: device identifier to validate
        section:   optional — validate only this section (default: all)

    Checks performed:
      - NTP: at least 2 servers configured, timezone set
      - Syslog: server configured, level appropriate for device type
      - AAA/RADIUS: correct ports, server reachable (simulated)
      - Spanning-tree: appropriate priority for role
      - BGP (routers): neighbors configured, correct AS numbers
      - RADIUS servers: cert expiry check, client list completeness
    """
    await asyncio.sleep(0.02)
    device_id = args.get("device_id", "").lower().strip()
    section   = args.get("section", "").lower().strip() or None

    if not device_id:
        return "[Error: device_id is required]"

    cfg = _get_live_config(device_id)
    if cfg is None:
        return f"[Error: device '{device_id}' not found]"

    errors:   list[str] = []
    warnings: list[str] = []
    ok:       list[str] = []

    def check_section(sec: str) -> None:
        data = cfg.get(sec, {})
        if not data:
            return

        if sec == "ntp":
            servers = data.get("servers", [])
            if len(servers) < 2:
                warnings.append(f"NTP: only {len(servers)} server(s) — recommend ≥2 for redundancy")
            else:
                ok.append(f"NTP: {len(servers)} servers configured ✓")
            if not data.get("timezone"):
                warnings.append("NTP: timezone not set")
            else:
                ok.append(f"NTP: timezone={data['timezone']} ✓")

        elif sec == "syslog":
            if not data.get("server"):
                errors.append("Syslog: no server configured — audit trail will be lost")
            else:
                ok.append(f"Syslog: server={data['server']} ✓")
            level = data.get("level", "")
            if level == "debugging":
                warnings.append("Syslog: level=debugging is very verbose — use informational in production")
            elif level in ("emergencies", "alerts", "critical", "errors"):
                warnings.append(f"Syslog: level={level} may miss important events — recommend informational/warnings")

        elif sec in ("aaa", "radius") and "radius_server" in data or "server" in data:
            server = data.get("radius_server") or data.get("server", "")
            auth_port = int(data.get("auth_port", 0))
            acct_port = int(data.get("acct_port", 0))
            if auth_port != 1812:
                errors.append(f"RADIUS: auth_port={auth_port} should be 1812 (RFC 2865)")
            else:
                ok.append(f"RADIUS: auth_port=1812 ✓")
            if acct_port != 1813:
                warnings.append(f"RADIUS: acct_port={acct_port} should be 1813 (RFC 2866)")
            if not server:
                errors.append("RADIUS: no server configured")
            else:
                ok.append(f"RADIUS: server={server} ✓")

        elif sec == "spanning_tree":
            mode = data.get("mode", "")
            pri  = int(data.get("priority", 32768))
            if mode not in ("rapid-pvst", "mst"):
                warnings.append(f"STP: mode={mode} — recommend rapid-pvst or mst")
            else:
                ok.append(f"STP: mode={mode} ✓")
            hostname = cfg.get("hostname", "")
            if "core" in hostname and pri > 8192:
                errors.append(f"STP: core switch priority={pri} is too high — should be ≤8192 for root election")
            elif "acc" in hostname and pri < 16384:
                warnings.append(f"STP: access switch priority={pri} unexpectedly low — may win root election")
            else:
                ok.append(f"STP: priority={pri} appropriate for role ✓")

        elif sec == "bgp":
            neighbors = data.get("neighbors", [])
            if not neighbors:
                errors.append("BGP: no neighbors configured")
            else:
                ok.append(f"BGP: {len(neighbors)} neighbor(s) configured ✓")
            as_num = data.get("as", 0)
            if not as_num:
                errors.append("BGP: AS number not set")

        elif sec == "radius" and "tls" in data:   # RADIUS server cert check
            tls     = data["tls"]
            expires = tls.get("cert_expires", "")
            if expires:
                import datetime
                try:
                    exp_date = datetime.date.fromisoformat(expires)
                    days_left = (exp_date - datetime.date.today()).days
                    if days_left < 0:
                        errors.append(f"RADIUS TLS: certificate EXPIRED {abs(days_left)} days ago! ({expires})")
                    elif days_left < 30:
                        errors.append(f"RADIUS TLS: certificate expires in {days_left} days ({expires}) — renew immediately")
                    elif days_left < 90:
                        warnings.append(f"RADIUS TLS: certificate expires in {days_left} days ({expires}) — schedule renewal")
                    else:
                        ok.append(f"RADIUS TLS: cert valid {days_left} days until {expires} ✓")
                except ValueError:
                    warnings.append(f"RADIUS TLS: could not parse cert_expires={expires!r}")
            clients = data.get("clients", [])
            if not clients:
                warnings.append("RADIUS: no clients configured")
            else:
                ok.append(f"RADIUS: {len(clients)} client(s) configured ✓")

    sections_to_check = [section] if section else [
        s for s in cfg if s not in ("hostname", "model", "management", "interfaces", "ssids")
    ]
    for s in sections_to_check:
        check_section(s)

    # Format result
    lines = [
        f"# Config validation: {device_id}",
        f"# Sections checked: {', '.join(sections_to_check)}",
        "# " + "─" * 60,
    ]
    if errors:
        lines.append(f"# ❌ ERRORS ({len(errors)}) — must fix:")
        lines.extend(f"  ERROR: {e}" for e in errors)
    if warnings:
        lines.append(f"# ⚠  WARNINGS ({len(warnings)}) — should fix:")
        lines.extend(f"  WARN:  {w}" for w in warnings)
    if ok:
        lines.append(f"# ✓  OK ({len(ok)}):")
        lines.extend(f"  OK:    {o}" for o in ok)
    if not errors and not warnings:
        lines.append("# ✅ No issues found — configuration looks correct")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: diff_device_config
# ---------------------------------------------------------------------------

async def diff_device_config(args: dict[str, Any]) -> str:
    """
    Show all pending configuration changes for a device (edits vs base config).
    If no edits have been made, confirms the device is at baseline.

    Args:
        device_id: device identifier (use list_devices to find IDs)
    """
    await asyncio.sleep(0.01)
    device_id = args.get("device_id", "").lower().strip()

    if not device_id:
        return "[Error: device_id is required]"
    if device_id not in _BASE_CONFIGS:
        return f"[Error: device '{device_id}' not found]"

    edits = _EDITED_CONFIGS.get(device_id, {})
    if not edits:
        return (
            f"# Diff: {device_id}\n"
            f"# No pending edits — device is at base configuration\n"
        )

    lines = [
        f"# Diff: {device_id}",
        f"# {len(edits)} section(s) with pending changes",
        "# " + "─" * 60,
    ]
    base = _BASE_CONFIGS[device_id]
    for section, changes in edits.items():
        lines.append(f"\n[Section: {section}]")
        base_section = base.get(section, {})
        for key, new_val in changes.items():
            old_val = base_section.get(key, "<new key>")
            if old_val == new_val:
                lines.append(f"  {key}: {new_val!r}  (unchanged)")
            else:
                lines.append(f"  {key}: {old_val!r}  →  {new_val!r}")

    # Recent change log
    device_log = [e for e in _CHANGE_LOG if e["device_id"] == device_id]
    if device_log:
        lines.append(f"\n[Change log — {len(device_log)} edit(s)]")
        for entry in device_log[-5:]:
            lines.append(f"  {entry['ts']}  {entry['section']}  reason={entry['reason']!r}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry — required for upload to work
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_device_config":    get_device_config,
    "edit_device_config":   edit_device_config,
    "validate_device_config": validate_device_config,
    "diff_device_config":   diff_device_config,
}
