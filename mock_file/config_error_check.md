# Config Error Check and Correction
**Purpose:** Systematically detect and correct configuration errors across network devices — RADIUS auth issues, NTP drift, syslog misconfiguration, spanning-tree mismatches, and BGP anomalies.
**Tags:** [config, validation, radius, ntp, syslog, spanning-tree, bgp, correction, troubleshooting]
**Risk:** medium
**HITL:** yes

## When to use this skill

Use when an operator asks to:
- Check device configurations for errors or inconsistencies
- Fix or correct configuration problems found during troubleshooting
- Validate that a device's config matches expected baseline values
- Audit multiple devices for the same type of misconfiguration
- Verify a configuration change before applying it to production

Example queries that trigger this skill:
- "检查现网设备的配置错误" (check network device config errors)
- "validate the RADIUS configuration on all APs"
- "fix the NTP configuration on sw-core-01"
- "check why ap-01 is failing 802.1x authentication"
- "audit all switches for spanning-tree priority issues"

## Parameters

- `device_id` (string): target device identifier from list_devices (e.g. sw-core-01, ap-01, radius-01)
- `check_type` (string): what to check — `all` | `radius` | `ntp` | `syslog` | `spanning_tree` | `bgp` | `cert`
- `auto_fix` (boolean): whether to apply corrections after finding errors (requires HITL approval for medium/high risk)
- `scope` (string): `single` (one device) | `all_aps` | `all_switches` | `all_routers` | `all` (entire network)

## Steps

### Phase 1 — Inventory

1. If `scope` is not `single`, call `list_devices` to enumerate devices of the relevant type.
   - For `all_aps`: `[TOOL:list_devices] {"type": "wireless_ap"}`
   - For `all_switches`: `[TOOL:list_devices] {"type": "switch"}`
   - For `all_routers`: `[TOOL:list_devices] {"type": "router"}`
   - For `all`: `[TOOL:list_devices] {}`

2. For each target device, retrieve the relevant config section:
   - `[TOOL:get_device_config] {"device_id": "<id>", "section": "<check_type>"}`
   - If `check_type` is `all`, omit the section parameter to get the full config.

### Phase 2 — Validate

3. For each device, run validation:
   - `[TOOL:validate_device_config] {"device_id": "<id>", "section": "<check_type>"}`

4. Categorise findings:
   - **ERRORS** (must fix): wrong ports, expired certs, missing critical config
   - **WARNINGS** (should fix): redundancy gaps, verbose logging, STP priority risks
   - **OK**: items that pass validation

5. Report a structured summary to the operator before taking any action:
   ```
   Device: <id>
     ERRORS: <count> — <brief description>
     WARNINGS: <count> — <brief description>
     OK: <count>
   ```

### Phase 3 — Correction (if `auto_fix` is true or operator confirms)

6. For each ERROR item, propose a specific fix using `edit_device_config`. **Always state
   the change before applying it** so the operator can verify.

   Common corrections by error type:

   **RADIUS port mismatch** (auth_port ≠ 1812):
   ```
   [TOOL:edit_device_config] {"device_id": "<id>", "section": "radius",
     "changes": {"auth_port": 1812, "acct_port": 1813},
     "reason": "correcting RADIUS ports to RFC 2865/2866 standard"}
   ```

   **NTP: single server / no redundancy**:
   ```
   [TOOL:edit_device_config] {"device_id": "<id>", "section": "ntp",
     "changes": {"servers": ["<primary>", "<secondary>"]},
     "reason": "adding NTP redundancy — minimum 2 servers required"}
   ```

   **Syslog: no server configured**:
   ```
   [TOOL:edit_device_config] {"device_id": "<id>", "section": "syslog",
     "changes": {"server": "<syslog_server_ip>", "level": "informational"},
     "reason": "configuring syslog for audit trail"}
   ```

   **STP: core switch priority too high**:
   ```
   [TOOL:edit_device_config] {"device_id": "<id>", "section": "spanning_tree",
     "changes": {"priority": 4096},
     "reason": "correcting STP priority — core switch must be root"}
   ```

   **RADIUS TLS cert expiry < 30 days**:
   - Do NOT auto-fix cert renewal — this requires manual PKI operations.
   - Report the expiry date and create an incident ticket recommendation.
   - Example: "RADIUS-01 cert expires 2026-06-30 — schedule renewal by 2026-06-01"

7. After each edit, verify the change with `diff_device_config`:
   ```
   [TOOL:diff_device_config] {"device_id": "<id>"}
   ```

8. Re-run `validate_device_config` to confirm all errors are resolved.

### Phase 4 — Report

9. Produce a final summary report covering:
   - Total devices checked
   - Errors found and corrected (or pending operator action)
   - Warnings requiring follow-up
   - Any items that could not be auto-corrected (cert renewals, BGP policy changes)
   - Recommended next steps

## Constraints

- Never change RADIUS shared keys — these are masked and require secure out-of-band delivery.
- Never auto-apply BGP configuration changes — BGP misconfig can cause network outages. Always pause for HITL review.
- Certificate renewal is outside the scope of this skill — flag cert expiry to the operator and stop.
- Do not apply the same edit twice — use `diff_device_config` to confirm a change before re-applying.
- When `scope` covers many devices, process one at a time and report progress. Do not batch-apply to all devices in one step.
- Confirm with the operator before making any change classified as `medium` risk or higher.

## Notes

**Typical RADIUS auth failure workflow:**
1. Check all APs' RADIUS config — look for wrong server IP or ports
2. Check RADIUS server cert expiry — most auth failures in this environment are cert-related
3. Verify RADIUS server has the AP/switch as a registered client
4. Check NTP sync — RADIUS TLS validation fails if clocks drift > 5 minutes

**Spanning-tree root election rules for this environment:**
- sw-core-01: priority must be ≤ 4096 (primary root)
- sw-core-02: priority must be ≤ 8192 (secondary root)
- sw-acc-*: priority must be ≥ 16384 (never become root)

**Syslog levels by device type:**
- Core switches, routers: `informational`
- Access switches: `warnings`
- RADIUS servers: `informational` (auth events)

**Known baseline values to check against:**
- RADIUS auth port: 1812 (RFC 2865)
- RADIUS accounting port: 1813 (RFC 2866)
- NTP minimum servers: 2 (for redundancy)
- BGP hold timer: 90s (local standard)
