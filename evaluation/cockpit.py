"""Self-contained, read-only HTML rendering for convergence evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from network_runtime.contracts import sha256_json

from .convergence import CONVERGENCE_SCHEMA, ConvergenceReportError


def _embedded_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_convergence_html(report: dict[str, Any]) -> str:
    if report.get("apiVersion") != CONVERGENCE_SCHEMA:
        raise ConvergenceReportError("convergence snapshot schema is unsupported")
    body = dict(report)
    declared = body.pop("snapshotDigest", None)
    if declared != sha256_json(body):
        raise ConvergenceReportError("convergence snapshot digest is invalid")
    data = _embedded_json(report)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'">
  <title>NetOpYu 收敛评测驾驶舱</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0d1b2a; --line:#20354d; --text:#e8f0f7; --muted:#9fb0c1; --cyan:#43d9c4; --amber:#ffbf69; --red:#ff6b6b; --blue:#70a1ff; }}
    * {{ box-sizing:border-box }}
    body {{ margin:0; background:radial-gradient(circle at 80% 0,#13344a 0,transparent 34%),var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
    main {{ max-width:1440px; margin:auto; padding:30px 24px 56px }}
    h1 {{ margin:0; font-size:clamp(28px,4vw,48px); letter-spacing:-.04em }}
    h2 {{ margin:0 0 14px; font-size:20px }}
    .eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.14em; font-size:12px; font-weight:800 }}
    .lede {{ max-width:970px; color:var(--muted); font-size:16px }}
    .notice {{ border-left:4px solid var(--amber); background:#1d2027; padding:14px 16px; margin:22px 0 }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px }}
    .card,.section {{ background:linear-gradient(155deg,rgba(20,40,58,.96),rgba(10,24,38,.96)); border:1px solid var(--line); border-radius:16px; box-shadow:0 18px 50px rgba(0,0,0,.22) }}
    .card {{ padding:17px }} .section {{ padding:20px; margin-top:18px }}
    .label {{ color:var(--muted); font-size:12px }} .value {{ display:block; font-size:27px; font-weight:800; margin-top:5px }}
    .good {{ color:var(--cyan) }} .warn {{ color:var(--amber) }} .bad {{ color:var(--red) }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px }}
    table {{ width:100%; border-collapse:collapse }} th,td {{ border-bottom:1px solid var(--line); padding:10px 8px; text-align:left; vertical-align:top }} th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.07em }}
    .scroll {{ overflow:auto; max-height:620px }}
    code {{ color:#b8def7; font-family:ui-monospace,SFMono-Regular,Menlo,monospace }}
    select {{ background:#071523; color:var(--text); border:1px solid #34536d; padding:8px 10px; border-radius:8px; margin:0 10px 10px 0 }}
    .tag {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 8px; color:var(--muted); font-size:11px }}
    .bar {{ height:8px; background:#071523; border-radius:9px; overflow:hidden; min-width:100px }} .bar>i {{ display:block; height:100%; background:linear-gradient(90deg,var(--blue),var(--cyan)) }}
    .meta {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere }}
    ul {{ padding-left:20px }}
    @media(max-width:980px) {{ .grid {{ grid-template-columns:repeat(2,1fr) }} .two {{ grid-template-columns:1fr }} }}
    @media(max-width:560px) {{ main {{ padding:20px 12px }} .grid {{ grid-template-columns:1fr }} }}
  </style>
</head>
<body>
<main>
  <div class="eyebrow">NetOpYu · read-only evaluation</div>
  <h1>收敛评测驾驶舱</h1>
  <p class="lede">把概率性的 L1 意图处理与确定性的 L0 Runtime 控制分开度量。这里展示固定测试集证据、失败归因和时延；页面没有审批、激活或执行能力。</p>
  <div class="notice" id="claim"></div>
  <section class="grid" id="headline"></section>
  <section class="section">
    <h2>DSH only 与 DSH + Runtime</h2>
    <div class="scroll"><table><thead><tr><th>控制 Oracle</th><th>DSH only</th><th>+ Runtime</th><th>增量</th></tr></thead><tbody id="runtime-metrics"></tbody></table></div>
  </section>
  <section class="section">
    <h2>Runtime 逐 Oracle 证据</h2>
    <p class="meta">同一 L1、参数、Provider 和故障输入；展示 72 条固定场景的实际判定，不表示生产概率。</p>
    <div class="scroll"><table><thead><tr><th>场景</th><th>类别</th><th>Oracle</th><th>DSH only</th><th>+ Runtime</th></tr></thead><tbody id="runtime-cases"></tbody></table></div>
  </section>
  <section class="section">
    <h2>模型资格与语义性能</h2>
    <div class="scroll"><table><thead><tr><th>模型</th><th>资格</th><th>选择</th><th>参数 F1</th><th>追问 P/R</th><th>Workflow</th><th>E2E</th><th>协议</th><th>p50 / p95</th></tr></thead><tbody id="models"></tbody></table></div>
  </section>
  <div class="two">
    <section class="section">
      <h2>失败首层归因</h2>
      <div id="layers"></div>
    </section>
    <section class="section">
      <h2>结论边界</h2>
      <ul id="limits"></ul>
    </section>
  </div>
  <section class="section">
    <h2>逐案例证据（无 Prompt、无参数值）</h2>
    <label>模型 <select id="model-filter"><option value="">全部</option></select></label>
    <label>失败层 <select id="layer-filter"><option value="">全部</option></select></label>
    <label>结果 <select id="result-filter"><option value="">全部</option><option value="pass">通过</option><option value="fail">失败</option></select></label>
    <span class="tag" id="case-count"></span>
    <div class="scroll"><table><thead><tr><th>模型 / Case</th><th>类别</th><th>首层</th><th>期望 → 实际</th><th>关键门禁</th><th>Containment</th><th>时延</th></tr></thead><tbody id="cases"></tbody></table></div>
  </section>
  <p class="meta" id="digest"></p>
</main>
<script type="application/json" id="netopyu-data">{data}</script>
<script>
(() => {{
  'use strict';
  const d=JSON.parse(document.getElementById('netopyu-data').textContent);
  const pct=v=>`${{(Number(v||0)*100).toFixed(2)}}%`;
  const ms=v=>Number(v||0).toLocaleString(undefined,{{maximumFractionDigits:1}})+' ms';
  const td=(tr,value)=>{{const x=document.createElement('td'); if(value instanceof Node)x.append(value); else x.textContent=String(value); tr.append(x);}};
  document.getElementById('claim').textContent=d.answer.claim+' Production generalization: NOT PROVEN.';
  const runtime=d.runtimeComparison.runtimeControlEffectiveness;
  const only=d.runtimeComparison.dshOnlyControlEffectiveness;
  const qualified=d.models.filter(x=>x.qualified).length;
  const totalCases=d.models.reduce((n,x)=>n+x.cases,0);
  const runtimeLatency=d.runtimeComparison.latency.dsh_plus_runtime;
  const cards=[
    ['Runtime control',`${{runtime.passed}}/${{runtime.total}}`,runtime.rate===100?'good':'bad'],
    ['DSH-only control',`${{only.passed}}/${{only.total}}`,'warn'],
    ['Qualified models',`${{qualified}}/${{d.models.length}}`,qualified?'good':'warn'],
    ['Runtime p50 / p95',`${{Number(runtimeLatency.p50_ms).toFixed(1)}} / ${{Number(runtimeLatency.p95_ms).toFixed(1)}} ms`,'good'],
    ['Visible case evidence',String(totalCases),'good']
  ];
  for(const [label,value,cls] of cards){{const c=document.createElement('article');c.className='card';const l=document.createElement('span');l.className='label';l.textContent=label;const v=document.createElement('strong');v.className='value '+cls;v.textContent=value;c.append(l,v);document.getElementById('headline').append(c);}}
  const right=new Map(d.runtimeComparison.metrics.dshPlusRuntime.map(x=>[x.metric_id,x]));
  for(const left of d.runtimeComparison.metrics.dshOnly){{const controlled=right.get(left.metric_id);const tr=document.createElement('tr');td(tr,left.label_zh+' / '+left.label_en);td(tr,`${{left.passed}}/${{left.total}} (${{Number(left.rate).toFixed(1)}}%)`);td(tr,`${{controlled.passed}}/${{controlled.total}} (${{Number(controlled.rate).toFixed(1)}}%)`);td(tr,`${{(controlled.rate-left.rate).toFixed(1)}} pp`);document.getElementById('runtime-metrics').append(tr);}}
  for(const x of d.runtimeComparison.scenarios){{const tr=document.createElement('tr'),left=x.dsh_only,controlled=x.dsh_plus_runtime;td(tr,x.scenario_id+' · '+x.title_zh);td(tr,x.category);td(tr,x.oracle);td(tr,(left.passed?'PASS':'FAIL')+' · '+left.outcome+' · calls='+left.provider_calls);td(tr,(controlled.passed?'PASS':'FAIL')+' · '+controlled.outcome+' · '+String(controlled.terminal_state));document.getElementById('runtime-cases').append(tr);}}
  for(const model of d.models){{const m=model.metrics,tr=document.createElement('tr');td(tr,model.model);td(tr,model.qualified?'PASS':'FAIL');td(tr,pct(m.selectionAccuracy));td(tr,pct(m.parameterFieldF1));td(tr,pct(m.clarificationPrecision)+' / '+pct(m.clarificationRecall));td(tr,pct(m.workflowAccuracy));td(tr,pct(m.endToEndAccuracy));td(tr,pct(m.protocolCompletionRate));td(tr,ms(m.p50Ms)+' / '+ms(m.p95Ms));document.getElementById('models').append(tr);}}
  for(const model of d.models){{const box=document.createElement('div');const title=document.createElement('p');title.innerHTML='<code></code>';title.querySelector('code').textContent=model.model;box.append(title);for(const [layer,count] of Object.entries(model.failureLayers)){{const row=document.createElement('div');row.style.margin='9px 0';const label=document.createElement('span');label.textContent=layer+' · '+count;const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.width=(count/model.cases*100)+'%';bar.append(fill);row.append(label,bar);box.append(row);}}document.getElementById('layers').append(box);}}
  for(const limit of d.limits){{const li=document.createElement('li');li.textContent=limit;document.getElementById('limits').append(li);}}
  const modelFilter=document.getElementById('model-filter'), layerFilter=document.getElementById('layer-filter'), resultFilter=document.getElementById('result-filter');
  for(const model of [...new Set(d.caseEvidence.map(x=>x.model))]){{const o=document.createElement('option');o.value=o.textContent=model;modelFilter.append(o);}}
  for(const layer of [...new Set(d.caseEvidence.map(x=>x.failureLayer))].sort()){{const o=document.createElement('option');o.value=o.textContent=layer;layerFilter.append(o);}}
  function renderCases(){{const rows=d.caseEvidence.filter(x=>(!modelFilter.value||x.model===modelFilter.value)&&(!layerFilter.value||x.failureLayer===layerFilter.value)&&(!resultFilter.value||(resultFilter.value==='pass')===x.passed));const target=document.getElementById('cases');target.replaceChildren();for(const x of rows){{const tr=document.createElement('tr');td(tr,x.model+' / '+x.scenarioId);td(tr,x.profile+' · '+x.language+' · '+x.category);td(tr,x.failureLayer);td(tr,String(x.expectedAction)+' → '+String(x.predictedAction)+'\\n'+x.expectedTargets.join(', ')+' → '+String(x.predictedTarget));const gates=Object.entries(x.gates).filter(([,v])=>!v).map(([k])=>k);td(tr,gates.length?'failed: '+gates.join(', '):'all pass');td(tr,x.containment.guardContained?`contained; attempts=${{x.containment.modelAttempts}}, dropped=${{x.containment.droppedArgumentFieldCount+x.containment.schemaDroppedArgumentFieldCount}}`:'none');td(tr,ms(x.elapsedMs));target.append(tr);}}document.getElementById('case-count').textContent=rows.length+' / '+d.caseEvidence.length;}}
  modelFilter.addEventListener('change',renderCases);layerFilter.addEventListener('change',renderCases);resultFilter.addEventListener('change',renderCases);renderCases();
  document.getElementById('digest').textContent='Snapshot '+d.snapshotDigest+' · '+d.generatedAt+' · authority: read-only';
}})();
</script>
</body>
</html>"""


def export_convergence_html(report: dict[str, Any], path: str | Path) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ConvergenceReportError("HTML output target is unsafe")
    destination = supplied.resolve()
    if destination.exists() and not destination.is_file():
        raise ConvergenceReportError("HTML output target is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_convergence_html(report)
    destination.write_text(rendered, encoding="utf-8")
    persisted = destination.read_text(encoding="utf-8")
    if report["snapshotDigest"] not in persisted:
        raise ConvergenceReportError("HTML export verification failed")
    return destination
