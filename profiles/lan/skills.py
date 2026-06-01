"""
profiles/lan/skills.py — Enterprise LAN SOP skills
===================================================

Business skills (standard operating procedures) for enterprise LAN ops.
Migrated 2026-05 from skills/mock/registry.py (profile refactor).
"""
from __future__ import annotations
from typing import Any

SKILLS: dict[str, dict[str, Any]] = {
    "syslog_search": {
        "name":        "Syslog Search",
        "purpose":     "Search syslog entries across network devices",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["logs", "diagnostics"],
        "description": "Queries the mock syslog aggregator for matching entries.",
        "parameters":  {"host": "Device name or glob", "keyword": "Search term", "severity": "error|warning|info"},
        "returns":     "Matching syslog lines",
        "tool_deps":   ["syslog_search"],
        "examples":    [{"args": {"host": "ap-01", "severity": "error"}, "note": "Find errors on ap-01"}],
    },
    "netflow_analysis": {
        "name":        "NetFlow Analysis",
        "purpose":     "Analyse NetFlow traffic for anomalies and top talkers",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["traffic", "security"],
        "description": "Dumps and analyses NetFlow records. For large datasets, pages through stored results.",
        "parameters":  {"site": "Site name or 'all'"},
        "returns":     "Traffic summary with anomaly indicators",
        "tool_deps":   ["netflow_dump", "read_stored_result"],
        "examples":    [{"args": {"site": "all"}, "note": "Analyse all-site traffic"}],
    },
    "prometheus_query": {
        "name":        "Prometheus Query",
        "purpose":     "Query metrics from the mock Prometheus store",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["metrics", "monitoring"],
        "description": "Runs PromQL queries and returns time series data.",
        "parameters":  {"query": "PromQL expression", "duration": "Time window"},
        "returns":     "Time series table",
        "tool_deps":   ["prometheus_query"],
        "examples":    [],
    },
    "alert_summary": {
        "name":        "Alert Summary",
        "purpose":     "Summarise active monitoring alerts",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["monitoring", "alerts"],
        "description": "Retrieves and groups active alerts by severity and device.",
        "parameters":  {"severity": "Filter severity", "site": "Filter by site"},
        "returns":     "Grouped alert table",
        "tool_deps":   ["alert_summary"],
        "examples":    [],
    },
    "service_health": {
        "name":        "Service Health Check",
        "purpose":     "Check health of a named mock service",
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["services", "health"],
        "description": "Checks service health across environments.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Health status with latency and pod counts",
        "tool_deps":   ["service_health"],
        "examples":    [],
    },
    "restart_service": {
        "name":        "Service Restart",
        "purpose":     "Rolling restart of a mock production service",
        "risk_level":  "high",
        "requires_hitl": True,
        "tags":        ["services", "destructive"],
        "description": "Performs a rolling restart. Always requires HITL approval.",
        "parameters":  {"service": "Service name", "environment": "prod|staging|dev"},
        "returns":     "Restart status",
        "tool_deps":   ["restart_service"],
        "examples":    [],
    },
    "rollback_service": {
        "name":        "Service Rollback",
        "purpose":     "Roll back a mock service to a previous version",
        "risk_level":  "high",
        "requires_hitl": True,
        "tags":        ["services", "destructive"],
        "description": "Rolls back to target version. Always requires HITL approval.",
        "parameters":  {"service": "Service name", "version": "Target version", "environment": "prod|staging|dev"},
        "returns":     "Rollback status",
        "tool_deps":   ["rollback_service"],
        "examples":    [],
    },

    "lan_user_access_diagnose": {
        "name":        "LAN User Access Diagnose",
        "purpose": (
            "Diagnose why a user cannot access an application. Determine whether "
            "the cause is LAN-side (identity / network admission) or "
            "application-layer (which lives on the DATA CENTER and must be "
            "delegated). Do NOT assume a network-path problem first."
        ),
        "risk_level":  "low",
        "requires_hitl": False,
        "tags":        ["user", "access", "permission", "identity", "nac", "troubleshoot"],
        "description": (
            "Use this when a user reports they cannot access / are denied / cannot "
            "reach an application. Access failures are usually identity or "
            "permission problems, not network-reachability problems — check those "
            "FIRST and treat fabric/path diagnostics as a last resort. "
            "Procedure, in order: "
            "(1) get_user_access(user_id) — is the user admitted on the LAN "
            "(RADIUS / 802.1X / NAC / VLAN)? "
            "(2) If blocked, check_nac_policy(user_id) explains the NAC decision; "
            "restoring admission is grant_user_access (destructive, needs approval). "
            "(3) If the user IS fully admitted on the LAN, the LAN is not the cause: "
            "the problem is application-layer access control, which is owned by the "
            "DATA CENTER. DELEGATE the application-access check to the dc agent "
            "(it owns application RBAC/ACL via dc_app_access_diagnose). Describe the "
            "task plainly as 'check whether user <id> has access permission to "
            "application <app>' — do NOT pre-frame it as a VNI / overlay / BGP-EVPN "
            "routing problem. "
            "(4) Only if BOTH LAN admission and DC application access are confirmed "
            "OK should network-path reachability be investigated."
        ),
        "parameters": {
            "user_id": "User reporting the access failure (e.g. alice)",
            "app":     "Application the user cannot reach (e.g. CRM)",
        },
        "returns": "LAN admission verdict + whether to delegate app-access check to DC",
        "tool_deps": ["list_users", "get_user_access", "check_nac_policy", "grant_user_access"],
        "examples": [
            {"args": {"user_id": "alice", "app": "CRM"}, "note": "Why can't alice reach CRM"},
        ],
    },
    # ── Deliberately complex multi-phase cross-agent SOP ────────────────
    # Purpose: stress-test skill-runtime fidelity. This skill has 4 ordered
    # phases, conditional branches, TWO distinct DC delegation points, and a
    # HITL gate. Because skills are *hints* (the LLM may follow or deviate),
    # the journal/journey lets you compare the SOP's prescribed tool/delegate
    # sequence against what the model actually does on a real run.
    "lan_new_employee_onboarding_access": {
        "name":        "New-Employee End-to-End Access Provisioning",
        "purpose": (
            "Provision and VERIFY end-to-end access for a NEW employee who needs "
            "to use a data-center application: confirm LAN network admission, then "
            "grant application-layer access (DC-owned), then confirm the path is "
            "actually reachable. This is a multi-stage provisioning workflow that "
            "spans BOTH the LAN agent (network admission) and the DC agent "
            "(application RBAC + fabric path). It is NOT a single permission check."
        ),
        "risk_level":    "medium",
        "requires_hitl": True,
        "tags": ["onboarding", "provisioning", "access", "cross-agent",
                 "workflow", "employee", "rbac", "nac"],
        "description": (
            "Use this for onboarding/provisioning requests like '为新员工 <user> "
            "开通对应用 <app> 的访问' / 'set up access to <app> for new hire <user>'. "
            "This is a STRICT, ORDERED, multi-phase workflow — do not skip phases, "
            "do not jump straight to granting, and do not collapse it into one "
            "tool call. Run the phases IN ORDER and stop early only on the "
            "documented exit conditions.\n"
            "\n"
            "PHASE 1 — LAN identity & admission (do this FIRST, locally):\n"
            "  1a. list_users — confirm the user account exists. If it does NOT "
            "exist, STOP and report that the account must be created in IAM first "
            "(do not attempt to grant anything).\n"
            "  1b. get_user_access(user_id) — is the user admitted on the LAN "
            "(RADIUS / 802.1X / NAC / VLAN)?\n"
            "  1c. If NOT admitted: check_nac_policy(user_id) to explain why, then "
            "grant_user_access(user_id) to restore LAN admission. This is a "
            "destructive change → requires operator approval (HITL). Only after "
            "admission is confirmed do you continue to Phase 2.\n"
            "  1d. If already admitted: continue to Phase 2 directly.\n"
            "\n"
            "PHASE 2 — DC application access (DELEGATE — do NOT do locally):\n"
            "  Application RBAC/ACL lives on the DATA CENTER, not the LAN. DELEGATE "
            "to the dc agent: 'check whether user <id> has access permission to "
            "application <app>, and if not, grant the appropriate role'. Describe "
            "it plainly as an application-permission task — do NOT pre-frame it as "
            "a VNI / overlay / BGP-EVPN problem. The DC side may require its own "
            "operator approval (HITL) to grant a role; wait for that result before "
            "Phase 3. Do NOT re-delegate the same task to dc if it already replied.\n"
            "\n"
            "PHASE 3 — End-to-end path verification (DELEGATE — DC-owned):\n"
            "  Only AFTER both LAN admission (Phase 1) and DC application access "
            "(Phase 2) are confirmed granted, delegate ONE more check to dc: "
            "'confirm the network path from the user's VLAN to application <app> is "
            "reachable (path / VNI / load-balancer health)'. This is the step that "
            "distinguishes 'permission granted but still unreachable' from a fully "
            "working setup. Skip this phase only if the user explicitly said access "
            "already works and they just need the permission.\n"
            "\n"
            "PHASE 4 — Synthesize the provisioning report:\n"
            "  Produce ONE final report covering: account exists (Y/N), LAN "
            "admission (was it already OK or did you restore it, with the approval "
            "reference), DC application role granted (which role, approval "
            "reference), and end-to-end reachability verdict. If any phase failed "
            "or was rejected at HITL, report exactly which phase and the "
            "consequence — do NOT claim success. Never re-run a phase that already "
            "completed; synthesize from the results already in context."
        ),
        "parameters": {
            "user_id": "New employee's user id (e.g. alice)",
            "app":     "Application to provision access for (e.g. crm)",
        },
        "returns": (
            "End-to-end provisioning report: account/admission/app-role/"
            "reachability, with HITL approval references and per-phase status."
        ),
        # Local tools this skill drives directly (Phase 1). Phases 2-3 are
        # delegated to dc, so dc's tools are NOT listed here — they're the
        # peer's responsibility, reached via [DELEGATE:dc-agent].
        "tool_deps": ["list_users", "get_user_access", "check_nac_policy",
                      "grant_user_access"],
        "examples": [
            {"args": {"user_id": "alice", "app": "crm"},
             "note": "Onboard alice end-to-end for CRM access"},
        ],
    },
}
