"""classify_failure_layer.py — deterministically classify the failure layer
from a branch user's LAN admission-check result (3-domain variant).

Same deterministic-rule rationale as app-access-troubleshoot, but the
"admission is fine" branch routes to `delegate_transport` (check WAN first)
rather than straight to the DC app layer — in branch→DC scenarios a degraded
WAN circuit / tunnel / SLA is a common cause that must be ruled out before
blaming the application.

Contract (skill script standard): run(inputs: dict) -> dict
  inputs : admission fields from step 1 —
           admitted (bool), nac_compliant (bool), vlan (int|str|None),
           auth_ok (bool)
  output : {"failure_layer": <label>, "reason": <str>}
           label ∈ {lan_auth, lan_nac, delegate_transport, unknown}
"""


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "ok", "compliant", "admitted")
    if isinstance(v, (int, float)):
        return v != 0
    return None


def run(inputs: dict) -> dict:
    admitted = _as_bool(inputs.get("admitted"))
    nac_ok = _as_bool(inputs.get("nac_compliant"))
    auth_ok = _as_bool(inputs.get("auth_ok"))
    vlan = inputs.get("vlan")

    # Rule table (order matters — first match wins):
    if auth_ok is False:
        return {"failure_layer": "lan_auth",
                "reason": "认证失败(auth_ok=false),问题在 LAN 准入认证层"}

    if nac_ok is False:
        return {"failure_layer": "lan_nac",
                "reason": "NAC 不合规(nac_compliant=false),问题在 LAN 准入策略层"}

    # Admitted on the LAN → check the WAN transport path to DC BEFORE the app.
    if admitted is True and nac_ok is not False and vlan not in (None, "", 0):
        return {"failure_layer": "delegate_transport",
                "reason": ("LAN 准入正常(已认证/合规/有 VLAN)。分支到 DC 不可达,"
                           "先排查 WAN 传输层(电路/隧道/SLA),再视情况查 DC 应用层")}

    if admitted is False:
        return {"failure_layer": "lan_auth",
                "reason": "未准入(admitted=false)但无更细信号,先按 LAN 准入层排查"}

    return {"failure_layer": "unknown",
            "reason": "准入字段不足以判定故障层,需要补充 admitted/nac_compliant/auth_ok"}
