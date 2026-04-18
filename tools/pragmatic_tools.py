"""
tools/pragmatic_tools.py
------------------------
Real network device tools for pragmatic mode.

Uses Netmiko for SSH command execution and NAPALM for structured
getter output. Nornir is used for concurrent multi-device operations.

Graceful degradation:
  - If Netmiko is not installed: raises ImportError with install hint
  - If a device is unreachable: returns structured error, does not crash
  - If NAPALM is not installed: falls back to raw Netmiko for getters
  - If Nornir is not installed: runs devices sequentially

Install:
    pip install netmiko napalm nornir nornir-netmiko nornir-napalm

Device inventory is read from cfg.pragmatic.device_inventory at startup.
All tools are async-safe — Netmiko's blocking SSH is run in a thread pool.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Device registry (populated at startup by build_pragmatic_tool_registry)
# ─────────────────────────────────────────────────────────────────────────────

_DEVICES: dict[str, dict] = {}   # id → netmiko ConnectHandler kwargs


def register_devices(devices: list) -> None:
    """Called by main.py to load device inventory from config."""
    for dev in devices:
        _DEVICES[dev.id] = {
            "device_type": dev.device_type,
            "host":        dev.host,
            "username":    dev.username,
            "password":    dev.password,
            "secret":      dev.secret,
            "port":        dev.port,
            "timeout":     dev.timeout,
            "_label":      dev.label,
            "_tags":       dev.tags,
        }
    logger.info("Pragmatic: registered %d device(s): %s",
                len(_DEVICES), list(_DEVICES.keys()))


def _dev_kwargs(device_id: str) -> dict:
    """Return Netmiko ConnectHandler kwargs, stripping private fields."""
    raw = _DEVICES.get(device_id)
    if not raw:
        raise KeyError(f"Device {device_id!r} not in inventory. "
                       f"Available: {list(_DEVICES.keys())}")
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _list_devices_raw(type_filter: str | None = None,
                      tag_filter: str | None = None) -> list[dict]:
    result = []
    for did, d in _DEVICES.items():
        tags = d.get("_tags", [])
        if type_filter and type_filter not in tags and type_filter != d.get("device_type"):
            continue
        if tag_filter and tag_filter not in tags:
            continue
        result.append({
            "id":          did,
            "host":        d["host"],
            "device_type": d["device_type"],
            "label":       d.get("_label", did),
            "tags":        tags,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _run_in_executor(fn, *args, **kwargs) -> Any:
    """Run a blocking function in the default thread pool."""
    loop = asyncio.get_event_loop()
    import functools
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def _netmiko_send(device_id: str, command: str) -> str:
    """Blocking: open SSH, send command, close. Returns output string."""
    try:
        from netmiko import ConnectHandler
    except ImportError:
        raise ImportError(
            "netmiko is not installed. Run: pip install netmiko"
        )
    kwargs = _dev_kwargs(device_id)
    t0 = time.time()
    try:
        with ConnectHandler(**kwargs) as conn:
            if kwargs.get("secret"):
                conn.enable()
            output = conn.send_command(command, read_timeout=60)
        elapsed = time.time() - t0
        logger.debug("Netmiko %s cmd=%r elapsed=%.1fs len=%d",
                     device_id, command[:60], elapsed, len(output))
        return output
    except Exception as exc:
        logger.warning("Netmiko %s cmd=%r failed: %s", device_id, command[:60], exc)
        raise


def _netmiko_send_config(device_id: str, config_lines: list[str]) -> str:
    """Blocking: push config lines via Netmiko."""
    try:
        from netmiko import ConnectHandler
    except ImportError:
        raise ImportError("pip install netmiko")
    kwargs = _dev_kwargs(device_id)
    with ConnectHandler(**kwargs) as conn:
        if kwargs.get("secret"):
            conn.enable()
        output = conn.send_config_set(config_lines)
        conn.save_config()
    return output


def _napalm_get(device_id: str, getters: list[str]) -> dict:
    """Blocking: connect via NAPALM, run getters, return dict."""
    try:
        import napalm
    except ImportError:
        raise ImportError("pip install napalm")
    d = _dev_kwargs(device_id)
    # Map netmiko device_type to NAPALM driver name
    _NAPALM_MAP = {
        "cisco_ios":   "ios",
        "cisco_xe":    "ios",
        "cisco_xr":    "iosxr",
        "cisco_nxos":  "nxos",
        "juniper":     "junos",
        "juniper_junos": "junos",
        "arista_eos":  "eos",
        "paloalto":    "panos",
    }
    driver_name = _NAPALM_MAP.get(d["device_type"], "ios")
    driver = napalm.get_network_driver(driver_name)
    with driver(
        hostname         = d["host"],
        username         = d["username"],
        password         = d["password"],
        optional_args    = {"port": d.get("port", 22), "secret": d.get("secret", "")},
    ) as device:
        results = {}
        for getter in getters:
            try:
                results[getter] = getattr(device, getter)()
            except Exception as exc:
                results[getter] = {"error": str(exc)}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Nornir multi-device runner
# ─────────────────────────────────────────────────────────────────────────────

def _nornir_run_command(device_ids: list[str], command: str) -> dict[str, str]:
    """
    Run the same command on multiple devices concurrently via Nornir.
    Falls back to sequential Netmiko if Nornir is not installed.
    Returns dict: device_id → output_string.
    """
    try:
        from nornir import InitNornir
        from nornir_netmiko.tasks import netmiko_send_command
        from nornir.core.inventory import Host, Group, Defaults, Hosts, Groups

        hosts_dict = {}
        for did in device_ids:
            d = _DEVICES.get(did)
            if not d:
                continue
            hosts_dict[did] = Host(
                name      = did,
                hostname  = d["host"],
                username  = d["username"],
                password  = d["password"],
                port      = d.get("port", 22),
                platform  = d["device_type"],
                data      = {"secret": d.get("secret", "")},
            )

        nr = InitNornir(
            runner={"plugin": "threaded", "options": {"num_workers": min(10, len(device_ids))}},
            inventory={"plugin": "SimpleInventory",
                       "options": {"host_file": None, "group_file": None, "defaults_file": None}},
            logging={"enabled": False},
        )
        nr.inventory.hosts = Hosts(hosts_dict)

        results_nr = nr.run(task=netmiko_send_command, command_string=command)
        output = {}
        for host_name, result in results_nr.items():
            if result.failed:
                output[host_name] = f"[ERROR] {result[0].exception}"
            else:
                output[host_name] = result[0].result
        return output

    except ImportError:
        # Fall back to sequential
        logger.debug("Nornir not available — running devices sequentially")
        output = {}
        for did in device_ids:
            try:
                output[did] = _netmiko_send(did, command)
            except Exception as exc:
                output[did] = f"[ERROR] {exc}"
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Tool: list_devices
# ─────────────────────────────────────────────────────────────────────────────

async def list_devices(args: dict[str, Any]) -> str:
    """List all real devices from the inventory, optionally filtered by tag."""
    dev_type   = args.get("type")
    tag_filter = args.get("tag")
    devices    = _list_devices_raw(type_filter=dev_type, tag_filter=tag_filter)
    if not devices:
        return f"No devices found (type={dev_type!r} tag={tag_filter!r}). " \
               f"Check pragmatic.device_inventory in config.yaml."
    lines = [f"{'ID':<20} {'HOST':<20} {'TYPE':<16} {'LABEL':<30} TAGS"]
    lines.append("─" * 100)
    for d in devices:
        lines.append(
            f"{d['id']:<20} {d['host']:<20} {d['device_type']:<16} "
            f"{d['label']:<30} {','.join(d['tags'])}"
        )
    lines.append(f"\nTotal: {len(devices)} device(s)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: get_device_status
# ─────────────────────────────────────────────────────────────────────────────

async def get_device_status(args: dict[str, Any]) -> str:
    """Get live CPU, memory, uptime, and interface count from a real device."""
    device_id = args.get("device_id", "")
    if not device_id:
        return "[Error] device_id is required"
    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found. Use list_devices to see available devices."

    dev_type = _DEVICES[device_id]["device_type"]
    # Choose command set by platform
    if "juniper" in dev_type or "junos" in dev_type:
        cmds = [
            ("uptime",   "show system uptime | no-more"),
            ("cpu",      "show chassis routing-engine | match 'CPU|Memory'"),
            ("ifaces",   "show interfaces terse | count"),
        ]
    elif "nxos" in dev_type:
        cmds = [
            ("uptime",   "show version | include uptime"),
            ("cpu",      "show processes cpu summary | head 5"),
            ("ifaces",   "show interface status | count"),
        ]
    elif "arista" in dev_type or "eos" in dev_type:
        cmds = [
            ("uptime",   "show version | grep uptime"),
            ("cpu",      "show processes top once | head 6"),
            ("ifaces",   "show interfaces status | count"),
        ]
    else:  # cisco_ios / cisco_xe / cisco_xr
        cmds = [
            ("uptime",   "show version | include uptime"),
            ("cpu",      "show processes cpu sorted | head 5"),
            ("ifaces",   "show ip interface brief | count"),
        ]

    results = {}
    for key, cmd in cmds:
        try:
            results[key] = await _run_in_executor(_netmiko_send, device_id, cmd)
        except Exception as exc:
            results[key] = f"[ERROR] {exc}"

    label = _DEVICES[device_id].get("_label", device_id)
    lines = [f"Device Status: {device_id} ({label})", "─" * 60]
    for key, output in results.items():
        lines.append(f"\n[{key.upper()}]\n{output.strip()}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: get_device_config
# ─────────────────────────────────────────────────────────────────────────────

async def get_device_config(args: dict[str, Any]) -> str:
    """Retrieve running config (or a specific section) from a real device."""
    device_id = args.get("device_id", "")
    section   = args.get("section")   # None = full config

    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."

    dev_type = _DEVICES[device_id]["device_type"]
    if section:
        # Platform-specific section extraction
        if "juniper" in dev_type:
            cmd = f"show configuration {section}"
        elif "nxos" in dev_type:
            cmd = f"show running-config | section {section}"
        else:
            cmd = f"show running-config | section {section}"
    else:
        if "juniper" in dev_type:
            cmd = "show configuration | no-more"
        elif "nxos" in dev_type:
            cmd = "show running-config"
        elif "arista" in dev_type:
            cmd = "show running-config"
        else:
            cmd = "show running-config"

    try:
        output = await _run_in_executor(_netmiko_send, device_id, cmd)
        label  = _DEVICES[device_id].get("_label", device_id)
        header = f"! Configuration for {device_id} ({label})"
        if section:
            header += f"  section={section}"
        return f"{header}\n! Retrieved: live\n!\n{output}"
    except Exception as exc:
        return f"[Error] Could not retrieve config from {device_id}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: edit_device_config
# ─────────────────────────────────────────────────────────────────────────────

async def edit_device_config(args: dict[str, Any]) -> str:
    """
    Push configuration lines to a real device.
    REQUIRES HITL approval (tagged as hitl_required in skill catalog).

    args:
      device_id: str
      config_lines: list[str]   — IOS-style config lines
      reason: str               — change reason for audit log
    """
    device_id    = args.get("device_id", "")
    config_lines = args.get("config_lines", [])
    reason       = args.get("reason", "no reason provided")

    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."
    if not config_lines:
        return "[Error] config_lines list is required and must not be empty."

    logger.warning(
        "PRAGMATIC: edit_device_config on %s — %d lines — reason=%r",
        device_id, len(config_lines), reason,
    )
    try:
        output = await _run_in_executor(_netmiko_send_config, device_id, config_lines)
        return (
            f"Config applied to {device_id}:\n"
            f"Lines: {json.dumps(config_lines, indent=2)}\n"
            f"Reason: {reason}\n"
            f"Output:\n{output}"
        )
    except Exception as exc:
        return f"[Error] Config push to {device_id} failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: validate_device_config
# ─────────────────────────────────────────────────────────────────────────────

async def validate_device_config(args: dict[str, Any]) -> str:
    """
    Run structured validation checks on a device using NAPALM getters.
    Falls back to Netmiko show commands if NAPALM is not installed.

    Checks performed:
      - NTP servers configured and synchronised
      - RADIUS / AAA reachability
      - BGP sessions established (if BGP device)
      - Interface error counters
      - CPU and memory within thresholds
    """
    device_id = args.get("device_id", "")
    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."

    label    = _DEVICES[device_id].get("_label", device_id)
    dev_type = _DEVICES[device_id]["device_type"]
    issues, warnings, passed = [], [], []

    # ── Try NAPALM first for structured getters ──────────────────────
    napalm_ok = False
    try:
        from config import cfg
        getters = cfg.pragmatic.napalm_getters
        data = await _run_in_executor(_napalm_get, device_id, getters)
        napalm_ok = True

        # NTP check
        ntp = data.get("get_ntp_servers", {})
        if isinstance(ntp, dict) and ntp:
            synced = [k for k, v in ntp.items()
                      if isinstance(v, dict) and v.get("synchronized")]
            if synced:
                passed.append(f"PASS  [NTP]  {len(synced)} server(s) synchronized: {synced[0]}")
            else:
                issues.append("FAIL  [NTP]  NTP configured but no server synchronized")
        else:
            issues.append("FAIL  [NTP]  No NTP servers configured")

        # Interface errors
        ifaces = data.get("get_interfaces", {})
        error_ifaces = []
        for name, info in (ifaces.items() if isinstance(ifaces, dict) else []):
            if isinstance(info, dict):
                rx_err = info.get("rx_error_packets", 0) or 0
                tx_err = info.get("tx_error_packets", 0) or 0
                if rx_err + tx_err > 100:
                    error_ifaces.append(f"{name}(rx={rx_err} tx={tx_err})")
        if error_ifaces:
            warnings.append(f"WARN  [IFACE] High error counters: {', '.join(error_ifaces[:5])}")
        else:
            passed.append(f"PASS  [IFACE] No significant interface errors ({len(ifaces)} interfaces)")

        # BGP check
        bgp = data.get("get_bgp_neighbors", {})
        if isinstance(bgp, dict) and bgp:
            not_established = []
            for vrf, peers in bgp.items():
                for peer, info in (peers.items() if isinstance(peers, dict) else []):
                    if isinstance(info, dict) and not info.get("is_up"):
                        not_established.append(peer)
            if not_established:
                warnings.append(f"WARN  [BGP]  Peers not established: {not_established[:5]}")
            else:
                passed.append(f"PASS  [BGP]  All BGP peers established")

        # Environment (CPU/Memory via NAPALM)
        env = data.get("get_environment", {})
        if isinstance(env, dict):
            cpu = env.get("cpu", {})
            mem = env.get("memory", {})
            if isinstance(cpu, dict):
                for slot, stats in cpu.items():
                    usage = stats.get("%usage", 0) if isinstance(stats, dict) else 0
                    if usage > 80:
                        warnings.append(f"WARN  [CPU]  Slot {slot}: {usage}% (threshold 80%)")
                    else:
                        passed.append(f"PASS  [CPU]  Slot {slot}: {usage}% OK")
            if isinstance(mem, dict):
                used = mem.get("used_ram", 0)
                avail = mem.get("available_ram", 1)
                pct = round(used / (used + avail) * 100) if (used + avail) > 0 else 0
                if pct > 85:
                    warnings.append(f"WARN  [MEM]  Memory usage {pct}% (threshold 85%)")
                else:
                    passed.append(f"PASS  [MEM]  Memory usage {pct}% OK")

    except ImportError:
        napalm_ok = False
        logger.info("validate_device_config: NAPALM not installed, using Netmiko fallback")
    except Exception as exc:
        napalm_ok = False
        logger.warning("validate_device_config: NAPALM getter failed (%s), using Netmiko fallback", exc)

    # ── Netmiko fallback: raw show commands ───────────────────────────
    if not napalm_ok:
        checks = [
            ("ntp",     "show ntp status",        r"synchronized",                    "NTP synchronized"),
            ("ntp_srv", "show ntp associations",  r"\d+\.\d+",                        "NTP servers present"),
            ("aaa",     "show aaa servers",       r"(radius|tacacs)",                 "AAA servers configured"),
            ("cpu",     "show processes cpu",     None,                               "CPU check"),
            ("mem",     "show processes memory",  None,                               "Memory check"),
        ]
        for key, cmd, pattern, label_str in checks:
            try:
                out = await _run_in_executor(_netmiko_send, device_id, cmd)
                if pattern:
                    if re.search(pattern, out, re.IGNORECASE):
                        passed.append(f"PASS  [{key.upper():<6}] {label_str}")
                    else:
                        issues.append(f"FAIL  [{key.upper():<6}] {label_str} — not found in output")
                else:
                    # CPU/memory: scan for high values
                    nums = re.findall(r"(\d+)%", out)
                    high = [int(n) for n in nums if int(n) > 80]
                    if high:
                        warnings.append(f"WARN  [{key.upper():<6}] High utilisation detected: {high[:3]}")
                    else:
                        passed.append(f"PASS  [{key.upper():<6}] {label_str} within normal range")
            except Exception as exc:
                warnings.append(f"WARN  [{key.upper():<6}] Could not check ({exc})")

    # ── Format report ─────────────────────────────────────────────────
    lines = [
        f"VALIDATION REPORT — {device_id} ({label})",
        f"Method: {'NAPALM structured getters' if napalm_ok else 'Netmiko show commands'}",
        "=" * 65,
    ]
    lines.extend(issues + warnings + passed)
    lines.append("=" * 65)
    lines.append(
        f"Summary: {len(issues)} issue(s), {len(warnings)} warning(s), "
        f"{len(passed)} check(s) passed"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: get_syslog  (via Netmiko — real device syslog buffer)
# ─────────────────────────────────────────────────────────────────────────────

async def get_syslog(args: dict[str, Any]) -> str:
    """Fetch syslog from real device logging buffer or remote syslog."""
    device_id = args.get("device_id") or args.get("host", "")
    severity  = args.get("severity", "error").lower()
    lines_n   = int(args.get("lines", 100))

    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."

    dev_type = _DEVICES[device_id]["device_type"]
    if "juniper" in dev_type:
        cmd = f"show log messages | match {severity} | last {lines_n} | no-more"
    elif "nxos" in dev_type:
        cmd = f"show logging | grep -i {severity} | last {lines_n}"
    elif "arista" in dev_type:
        cmd = f"show logging | grep -i {severity} | tail {lines_n}"
    else:
        cmd = f"show logging | include {severity.upper()} | tail {lines_n}"

    try:
        output = await _run_in_executor(_netmiko_send, device_id, cmd)
        return f"# Syslog {device_id} severity={severity} lines={lines_n}\n{output}"
    except Exception as exc:
        return f"[Error] Syslog retrieval from {device_id} failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: query_interface_metrics
# ─────────────────────────────────────────────────────────────────────────────

async def query_interface_metrics(args: dict[str, Any]) -> str:
    """Show interface counters and utilisation from a real device."""
    device_id = args.get("device_id") or args.get("host", "")
    interface = args.get("interface")   # None = all interfaces

    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."

    dev_type = _DEVICES[device_id]["device_type"]
    if interface:
        if "juniper" in dev_type:
            cmd = f"show interfaces {interface} detail | no-more"
        elif "arista" in dev_type:
            cmd = f"show interfaces {interface}"
        else:
            cmd = f"show interfaces {interface}"
    else:
        if "juniper" in dev_type:
            cmd = "show interfaces terse | no-more"
        elif "nxos" in dev_type:
            cmd = "show interface status"
        elif "arista" in dev_type:
            cmd = "show interfaces status"
        else:
            cmd = "show ip interface brief"

    try:
        output = await _run_in_executor(_netmiko_send, device_id, cmd)
        return f"# Interface metrics {device_id} iface={interface or 'all'}\n{output}"
    except Exception as exc:
        return f"[Error] Interface query on {device_id} failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: get_bgp_summary
# ─────────────────────────────────────────────────────────────────────────────

async def get_bgp_summary(args: dict[str, Any]) -> str:
    """Retrieve BGP neighbour table from a real router."""
    device_id = args.get("device_id") or args.get("router", "")
    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."

    dev_type = _DEVICES[device_id]["device_type"]
    if "juniper" in dev_type:
        cmd = "show bgp summary | no-more"
    elif "nxos" in dev_type:
        cmd = "show bgp summary"
    elif "arista" in dev_type:
        cmd = "show bgp summary"
    elif "xr" in dev_type:
        cmd = "show bgp all summary"
    else:
        cmd = "show bgp summary"

    try:
        output = await _run_in_executor(_netmiko_send, device_id, cmd)
        return f"# BGP Summary {device_id}\n{output}"
    except Exception as exc:
        return f"[Error] BGP query on {device_id} failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: get_device_facts  (NAPALM structured — best-effort)
# ─────────────────────────────────────────────────────────────────────────────

async def get_device_facts(args: dict[str, Any]) -> str:
    """Return structured NAPALM facts (model, serial, OS version, etc.)."""
    device_id = args.get("device_id", "")
    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."
    try:
        data = await _run_in_executor(_napalm_get, device_id, ["get_facts"])
        facts = data.get("get_facts", {})
        if "error" in facts:
            return f"[NAPALM error] {facts['error']}"
        return json.dumps(facts, indent=2, default=str)
    except ImportError:
        # Fallback: show version
        try:
            out = await _run_in_executor(_netmiko_send, device_id, "show version")
            return f"# Device facts (show version fallback)\n{out}"
        except Exception as exc:
            return f"[Error] {exc}"
    except Exception as exc:
        return f"[Error] NAPALM get_facts on {device_id} failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: run_command  (escape hatch — arbitrary show command)
# ─────────────────────────────────────────────────────────────────────────────

async def run_command(args: dict[str, Any]) -> str:
    """
    Run an arbitrary read-only command on a real device.
    Blocked for config-mode commands (configure terminal, etc.).
    """
    device_id = args.get("device_id", "")
    command   = args.get("command", "").strip()

    if device_id not in _DEVICES:
        return f"[Error] Device {device_id!r} not found."

    # Block config-mode commands
    _BLOCKED = re.compile(
        r"^(conf(igure)?[\s\t]*(t(erminal)?)?|no\s|write\s|copy\s|reload|delete\s|"
        r"clear\s|debug\s|undebug\s)",
        re.IGNORECASE,
    )
    if _BLOCKED.match(command):
        return (
            f"[BLOCKED] Command {command!r} is not allowed via run_command "
            "(config-mode and destructive commands are restricted). "
            "Use edit_device_config for configuration changes."
        )

    try:
        output = await _run_in_executor(_netmiko_send, device_id, command)
        return f"# {device_id}: {command}\n{output}"
    except Exception as exc:
        return f"[Error] Command {command!r} on {device_id} failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: multi_device_check  (Nornir concurrent)
# ─────────────────────────────────────────────────────────────────────────────

async def multi_device_check(args: dict[str, Any]) -> str:
    """
    Run the same show command across multiple devices concurrently (Nornir).
    args:
      device_ids: list[str]  — or "all" / tag filter
      command: str
      tag: str               — filter by tag instead of explicit list
    """
    command    = args.get("command", "show version | include uptime")
    device_ids = args.get("device_ids")
    tag        = args.get("tag")

    # Build target list
    if device_ids == "all" or device_ids is None:
        targets = _list_devices_raw(tag_filter=tag)
        id_list = [d["id"] for d in targets]
    elif isinstance(device_ids, list):
        id_list = [d for d in device_ids if d in _DEVICES]
    else:
        id_list = [str(device_ids)]

    if not id_list:
        return "[Error] No matching devices found."

    results = await _run_in_executor(_nornir_run_command, id_list, command)
    lines   = [f"Multi-device check: {command!r}  ({len(id_list)} devices)", "─" * 70]
    for dev, output in results.items():
        label = _DEVICES.get(dev, {}).get("_label", dev)
        lines.append(f"\n[ {dev} — {label} ]")
        lines.append(output.strip() if output else "(no output)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL_REGISTRY for pragmatic mode
# ─────────────────────────────────────────────────────────────────────────────

PRAGMATIC_TOOL_REGISTRY: dict[str, Any] = {
    "list_devices":            list_devices,
    "get_device_status":       get_device_status,
    "get_device_config":       get_device_config,
    "edit_device_config":      edit_device_config,   # HITL-required
    "validate_device_config":  validate_device_config,
    "get_syslog":              get_syslog,
    "query_interface_metrics": query_interface_metrics,
    "get_bgp_summary":         get_bgp_summary,
    "get_device_facts":        get_device_facts,
    "run_command":             run_command,
    "multi_device_check":      multi_device_check,
}

PRAGMATIC_TOOL_DESCRIPTIONS: dict[str, dict] = {
    "list_devices": {
        "description": "List real network devices from inventory (filterable by type or tag)",
        "parameters": {"type": "device type tag (optional)", "tag": "tag filter (optional)"},
        "returns_large": False,
    },
    "get_device_status": {
        "description": "Get live CPU, memory, uptime from a real device via SSH",
        "parameters": {"device_id": "device ID from list_devices"},
        "returns_large": False,
    },
    "get_device_config": {
        "description": "Retrieve running configuration from a real device via SSH",
        "parameters": {"device_id": "device ID", "section": "config section keyword (optional)"},
        "returns_large": True,
    },
    "edit_device_config": {
        "description": "Push configuration lines to a real device (HITL approval required)",
        "parameters": {
            "device_id": "device ID",
            "config_lines": "list of IOS-style config lines",
            "reason": "change reason for audit log",
        },
        "returns_large": False,
        "requires_hitl": True,
    },
    "validate_device_config": {
        "description": "Run NTP/AAA/BGP/CPU/memory validation checks via NAPALM or Netmiko",
        "parameters": {"device_id": "device ID"},
        "returns_large": False,
    },
    "get_syslog": {
        "description": "Fetch syslog from real device logging buffer",
        "parameters": {"device_id": "device ID", "severity": "error|warn|info|debug", "lines": "int"},
        "returns_large": True,
    },
    "query_interface_metrics": {
        "description": "Show interface counters and utilisation from a real device",
        "parameters": {"device_id": "device ID", "interface": "interface name (optional, all if omitted)"},
        "returns_large": True,
    },
    "get_bgp_summary": {
        "description": "Retrieve BGP neighbour table from a real router",
        "parameters": {"device_id": "router device ID"},
        "returns_large": False,
    },
    "get_device_facts": {
        "description": "Return NAPALM structured facts: model, serial, OS version, hostname",
        "parameters": {"device_id": "device ID"},
        "returns_large": False,
    },
    "run_command": {
        "description": "Run an arbitrary read-only show command on a real device",
        "parameters": {"device_id": "device ID", "command": "CLI command string"},
        "returns_large": True,
    },
    "multi_device_check": {
        "description": "Run the same command concurrently on multiple devices via Nornir",
        "parameters": {
            "command": "CLI command",
            "device_ids": "list of device IDs or 'all'",
            "tag": "tag filter (alternative to device_ids)",
        },
        "returns_large": True,
    },
}
