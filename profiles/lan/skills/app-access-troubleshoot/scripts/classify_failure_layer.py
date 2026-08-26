"""classify_failure_layer.py — deterministically classify the failure layer
from a user's LAN admission-check result.

Why a script (not LLM judgment): the classification is a fixed rule table over
boolean/enum fields. Code makes it exact and reproducible; an LLM reading the
same fields might drift. This is exactly the deterministic logic Anthropic's
standard recommends offloading from the model to a bundled script.

Contract (skill script standard): run(inputs: dict) -> dict
  inputs : the admission fields gathered in step 1, any of:
           admitted (bool), nac_compliant (bool), vlan (int|str|None),
           auth_ok (bool)
  output : {"failure_layer": <label>, "reason": <str>}
           label ∈ {lan_auth, lan_nac, delegate_dc, unknown}
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
    # 1. Authentication failed → LAN auth layer.
    if auth_ok is False:
        return {"failure_layer": "lan_auth",
                "reason": "认证失败(auth_ok=false),问题在 LAN 准入认证层"}

    # 2. Authenticated but NAC non-compliant → LAN NAC layer.
    if nac_ok is False:
        return {"failure_layer": "lan_nac",
                "reason": "NAC 不合规(nac_compliant=false),问题在 LAN 准入策略层"}

    # 3. Admitted on the LAN (auth ok, nac ok, has a VLAN) → app layer is DC's.
    if admitted is True and nac_ok is not False and vlan not in (None, "", 0):
        return {"failure_layer": "delegate_dc",
                "reason": "LAN 准入正常(已认证/合规/有 VLAN),应用不可达应排查 DC 应用层"}

    # 4. Explicitly not admitted but no clearer signal → still LAN-side, unknown sub-cause.
    if admitted is False:
        return {"failure_layer": "lan_auth",
                "reason": "未准入(admitted=false)但无更细信号,先按 LAN 准入层排查"}

    # 5. Not enough signal.
    return {"failure_layer": "unknown",
            "reason": "准入字段不足以判定故障层,需要补充 admitted/nac_compliant/auth_ok"}
