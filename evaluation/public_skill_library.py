"""Read-only, digest-bound browser for public Skills used by ES-P1-Wild.

The generated library is a transparency artifact, never a Skill discovery
path or execution surface. Third-party files are rendered as inert plain text.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from evaluation.public_skill_corpus import inspect_public_author_kit, inspect_public_snapshot
from evaluation.public_skill_draft_author import inspect_public_market_drafts
from network_runtime.contracts import sha256_json
from skills.skill_format import parse_skill_md


INDEX_SCHEMA = "effect-runtime.io/public-skill-library-index/v1"
MANIFEST_SCHEMA = "effect-runtime.io/public-skill-library-manifest/v1"
SUMMARY_SCHEMA = "effect-runtime.io/public-skill-library-summary/v1"
AUTHORITY = "read_only_transparency_artifact_no_execution_authority"
_MAX_DISPLAY_FILE_BYTES = 1024 * 1024
_EXECUTABLE_SUFFIXES = {
    ".bat", ".cjs", ".cmd", ".com", ".exe", ".js", ".mjs", ".pl",
    ".ps1", ".py", ".rb", ".sh", ".ts", ".wasm", ".zsh",
}
_EXECUTABLE_PARTS = {"bin", "hooks", "scripts"}
_EMBEDDED_RE = re.compile(
    r'<script type="application/json" id="skill-library-data">([A-Za-z0-9+/=]+)</script>'
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_surface(relative: str) -> bool:
    path = PurePosixPath(relative.lower())
    return path.suffix in _EXECUTABLE_SUFFIXES or bool(set(path.parts) & _EXECUTABLE_PARTS)


def _display_file(path: Path, relative: str) -> dict[str, Any]:
    data = path.read_bytes()
    content: str | None = None
    encoding = "not_displayed_binary_or_oversize"
    if len(data) <= _MAX_DISPLAY_FILE_BYTES and b"\x00" not in data:
        content = data.decode("utf-8", errors="replace")
        encoding = "utf-8-plain-text"
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "kind": "entry" if relative == "SKILL.md" else relative.split("/", 1)[0],
        "executionSurface": _execution_surface(relative),
        "displayEncoding": encoding,
        "content": content,
    }


def _pinned_url(assignment: dict[str, Any]) -> str:
    source = quote(str(assignment["sourcePath"]).strip("/"), safe="/")
    suffix = f"/{source}" if source else ""
    return f"https://github.com/{assignment['repository']}/tree/{assignment['commitSha']}{suffix}"


def _render_html(index: dict[str, Any]) -> str:
    payload = base64.b64encode(
        json.dumps(index, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'">
  <title>EnsuredSkill 测试 Skill 索引库</title>
  <style>
    :root {{ color-scheme:dark; --bg:#07101c; --panel:#0d1a29; --panel2:#101f31; --line:#243a51; --text:#e9f1f7; --muted:#9fb0bf; --cyan:#4edac6; --amber:#ffbd69; --red:#ff7272; --blue:#76a9ff; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:radial-gradient(circle at 90% 0,#183c52 0,transparent 32%),var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif }}
    header {{ padding:28px max(20px,calc((100vw - 1500px)/2)); border-bottom:1px solid var(--line) }}
    h1 {{ margin:4px 0 8px; font-size:clamp(28px,4vw,46px); letter-spacing:-.035em }} h2,h3 {{ margin:0 }}
    .eyebrow {{ color:var(--cyan); font-size:12px; font-weight:800; letter-spacing:.13em; text-transform:uppercase }} .muted,.meta {{ color:var(--muted) }}
    .notice {{ margin-top:16px; max-width:1100px; border-left:4px solid var(--amber); background:#201f25; padding:12px 15px }}
    .stats {{ display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:10px; margin-top:18px; max-width:1050px }} .stat {{ padding:12px 14px; background:var(--panel); border:1px solid var(--line); border-radius:12px }} .stat strong {{ display:block; font-size:23px }}
    main {{ max-width:1500px; margin:auto; padding:18px 20px 48px; display:grid; grid-template-columns:370px minmax(0,1fr); gap:16px }}
    .panel {{ background:linear-gradient(155deg,var(--panel2),var(--panel)); border:1px solid var(--line); border-radius:15px; overflow:hidden }} .pad {{ padding:17px }}
    .toolbar {{ display:grid; gap:9px; padding:14px; border-bottom:1px solid var(--line) }} input,select {{ width:100%; background:#071522; color:var(--text); border:1px solid #36536c; border-radius:8px; padding:9px 10px }}
    #skill-list {{ max-height:calc(100vh - 260px); overflow:auto }} .skill {{ width:100%; text-align:left; color:inherit; background:none; border:0; border-bottom:1px solid var(--line); padding:13px 14px; cursor:pointer }} .skill:hover,.skill.active {{ background:#183149 }} .skill strong {{ display:block; font-size:15px }}
    .tags {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:7px }} .tag {{ border:1px solid #36516a; color:var(--muted); border-radius:999px; padding:1px 7px; font-size:11px }} .good {{ color:var(--cyan) }} .warn {{ color:var(--amber) }} .bad {{ color:var(--red) }}
    .detail-head {{ padding:20px; border-bottom:1px solid var(--line) }} .detail-head h2 {{ font-size:27px }} .description {{ max-width:950px; font-size:15px }}
    .facts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px 20px; margin-top:14px }} .fact {{ min-width:0 }} .fact b {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em }} .fact code,.digest {{ display:block; overflow-wrap:anywhere; color:#badcf4 }}
    .section {{ padding:18px 20px; border-bottom:1px solid var(--line) }} .file-layout {{ display:grid; grid-template-columns:230px minmax(0,1fr); gap:12px; margin-top:12px }}
    #file-list {{ max-height:450px; overflow:auto; border:1px solid var(--line); border-radius:9px }} .file {{ width:100%; border:0; border-bottom:1px solid var(--line); background:#0a1725; color:var(--text); text-align:left; padding:9px; cursor:pointer; overflow-wrap:anywhere }} .file.active {{ background:#1a3a52 }}
    pre {{ margin:0; min-height:360px; max-height:620px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; background:#06111c; border:1px solid var(--line); border-radius:9px; padding:14px; color:#d8e7ef; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace }}
    .tasks {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-top:12px }} .task {{ border:1px solid var(--line); border-radius:10px; padding:12px; background:#0a1725 }} .task p {{ margin:7px 0 }}
    @media(max-width:1000px) {{ main {{ grid-template-columns:1fr }} #skill-list {{ max-height:420px }} .tasks {{ grid-template-columns:1fr }} }}
    @media(max-width:650px) {{ .stats {{ grid-template-columns:repeat(2,1fr) }} .facts,.file-layout {{ grid-template-columns:1fr }} main {{ padding:10px }} }}
  </style>
</head>
<body>
<header>
  <div class="eyebrow">EnsuredSkill · ES-P1-Wild · read-only</div>
  <h1>测试 Skill 索引库</h1>
  <p class="muted">查看本轮测试使用的固定 Skill、来源、原始内容、附件和模型草案状态。</p>
  <div class="notice" id="claim"></div><div class="stats" id="stats"></div>
</header>
<main>
  <aside class="panel"><div class="toolbar"><input id="search" type="search" placeholder="搜索名称、描述、仓库或路径"><select id="status"><option value="">全部草案状态</option><option value="drafted">已生成草案</option><option value="failed">草案失败</option><option value="not_run">未运行</option></select><span class="meta" id="count"></span></div><div id="skill-list"></div></aside>
  <article class="panel" id="detail"><div class="pad muted">从左侧选择一个 Skill。</div></article>
</main>
<script type="application/json" id="skill-library-data">{payload}</script>
<script>
(() => {{
  'use strict';
  const encoded=document.getElementById('skill-library-data').textContent;
  const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
  const data=JSON.parse(new TextDecoder().decode(bytes));
  const byId=new Map(data.skills.map(x=>[x.packageId,x])); let selected=data.skills[0]?.packageId||null;
  const el=(tag,cls,text)=>{{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=String(text);return n;}};
  const addFact=(root,label,value)=>{{const box=el('div','fact');box.append(el('b','',label),el('code','',value??'—'));root.append(box);}};
  document.getElementById('claim').textContent=data.claimBoundary;
  const s=data.statistics; for(const [label,value] of [['Skills',s.skillCount],['可见文件',s.displayableFileCount],['References',s.referenceFileCount],['草案任务',s.draftedTaskCount+'/'+s.taskSlotCount],['第三方执行',String(s.thirdPartyExecutionAttempted)]]){{const box=el('div','stat');box.append(el('span','muted',label),el('strong',value===false?'0 / 否':value));document.getElementById('stats').append(box);}}
  const search=document.getElementById('search'),status=document.getElementById('status');
  function filtered(){{const q=search.value.trim().toLowerCase();return data.skills.filter(x=>(!status.value||x.draft.status===status.value)&&(!q||[x.name,x.description,x.repository,x.sourcePath,x.packageId].join(' ').toLowerCase().includes(q)));}}
  function renderList(){{const rows=filtered(),root=document.getElementById('skill-list');root.replaceChildren();document.getElementById('count').textContent=rows.length+' / '+data.skills.length+' Skills';for(const x of rows){{const b=el('button','skill'+(x.packageId===selected?' active':''));b.type='button';b.append(el('strong','',x.name),el('span','meta',x.repository+' · '+x.sourcePath));const tags=el('div','tags');tags.append(el('span','tag '+(x.draft.status==='drafted'?'good':x.draft.status==='failed'?'bad':'warn'),x.draft.status),el('span','tag',x.language||'unknown'),el('span','tag',x.license||'license n/a'));b.append(tags);b.addEventListener('click',()=>{{selected=x.packageId;renderList();renderDetail(x);}});root.append(b);}}if(rows.length&&!rows.some(x=>x.packageId===selected)){{selected=rows[0].packageId;renderDetail(rows[0]);}}}}
  function renderDetail(x){{const root=document.getElementById('detail');root.replaceChildren();const head=el('div','detail-head');head.append(el('div','eyebrow',x.assignmentId),el('h2','',x.name),el('p','description muted',x.description));const facts=el('div','facts');addFact(facts,'Repository',x.repository);addFact(facts,'Source path',x.sourcePath);addFact(facts,'Pinned commit',x.commitSha);addFact(facts,'License',x.license||'not available in author kit');addFact(facts,'Package digest',x.packageDigest);addFact(facts,'Pinned source URL',x.pinnedSourceUrl);head.append(facts);root.append(head);
    const filesSec=el('section','section');filesSec.append(el('h3','','Skill 原文与附件（纯文本）'));const note=el('p','meta','页面不会解析 Markdown/HTML，也不会执行 scripts、hooks、agents 或其他附件。');filesSec.append(note);const layout=el('div','file-layout'),list=el('div','');list.id='file-list';const pre=el('pre','','');let active=null;const show=f=>{{active=f.path;for(const b of list.querySelectorAll('button'))b.classList.toggle('active',b.dataset.path===active);pre.textContent=f.content===null?'[文件不是可显示的 UTF-8 文本，索引仅保留摘要]':f.content;}};for(const f of x.files){{const b=el('button','file'+(f.executionSurface?' bad':''),f.path+' · '+f.bytes+' B');b.type='button';b.dataset.path=f.path;b.addEventListener('click',()=>show(f));list.append(b);}}layout.append(list,pre);filesSec.append(layout);root.append(filesSec);if(x.files.length)show(x.files.find(f=>f.path==='SKILL.md')||x.files[0]);
    const taskSec=el('section','section');taskSec.append(el('h3','','测试槽位与 9B 草案'));taskSec.append(el('p','meta','草案是待审输入，不是 Gold、Oracle 或执行权。'));const tasks=el('div','tasks');if(x.draft.tasks.length){{for(const t of x.draft.tasks){{const card=el('div','task');card.append(el('span','tag',t.challenge),el('h3','',t.slot_id),el('p','',t.user_prompt),el('p','meta','处置：'+t.expected_disposition+' · 风险：'+t.risk+' · Effect budget：'+t.max_effect_calls),el('p','meta','候选结果：'+t.intended_outcome));tasks.append(card);}}}}else{{for(const slot of x.taskSlots){{const card=el('div','task');card.append(el('span','tag bad',x.draft.status),el('h3','',slot.slotId),el('p','meta','无合格模型草案；保留给独立人工编写。'),el('p','meta',slot.challenge));tasks.append(card);}}}}taskSec.append(tasks);root.append(taskSec);
  }}
  search.addEventListener('input',renderList);status.addEventListener('change',renderList);renderList();if(selected)renderDetail(byId.get(selected));
}})();
</script>
</body></html>"""


def build_public_skill_library(
    author_kit_root: str | Path,
    output_root: str | Path,
    *,
    draft_root: str | Path | None = None,
    snapshot_root: str | Path | None = None,
) -> dict[str, Any]:
    kit_root = Path(author_kit_root).expanduser().resolve()
    kit = inspect_public_author_kit(kit_root)
    assignments = [
        json.loads(line) for line in (kit_root / "assignments.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    drafts_by_assignment: dict[str, dict[str, Any]] = {}
    draft_report: dict[str, Any] | None = None
    if draft_root is not None:
        draft_path = Path(draft_root).expanduser().resolve()
        draft_report = inspect_public_market_drafts(draft_path, kit_root)
        drafts_by_assignment = {
            item["assignmentId"]: item for item in (
                json.loads(line) for line in (draft_path / "drafts.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            )
        }
    source_by_package: dict[str, dict[str, Any]] = {}
    snapshot_digest: str | None = None
    if snapshot_root is not None:
        snapshot_path = Path(snapshot_root).expanduser().resolve()
        inspection = inspect_public_snapshot(snapshot_path)
        snapshot_digest = inspection["manifestDigest"]
        source_by_package = {
            item["packageId"]: item for item in (
                json.loads(line) for line in (snapshot_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ) if item.get("status") == "accepted"
        }

    skills: list[dict[str, Any]] = []
    for assignment in assignments:
        package_root = (kit_root / "packages" / assignment["packageId"]).resolve()
        source = source_by_package.get(assignment["packageId"], {})
        if source and source.get("packageDigest") != assignment["packageDigest"]:
            raise ValueError("public Skill library snapshot/author-kit package binding mismatch")
        files: list[dict[str, Any]] = []
        for path in sorted(package_root.rglob("*")):
            if path.is_symlink():
                raise ValueError("public Skill library cannot index symlinks")
            if path.is_file():
                files.append(_display_file(path, path.relative_to(package_root).as_posix()))
        entry = next(item for item in files if item["path"] == "SKILL.md")
        if entry["content"] is None:
            raise ValueError("public Skill library entry point must be displayable text")
        parsed = parse_skill_md(entry["content"])
        draft_row = drafts_by_assignment.get(assignment["assignmentId"])
        draft = {
            "status": "not_run", "tasks": [], "modelCalls": 0, "latencyMs": 0.0,
            "failureCategory": None,
        }
        if draft_row is not None:
            telemetry = draft_row.get("telemetry") or {}
            error = str(telemetry.get("error") or draft_row.get("validationError") or "")
            draft = {
                "status": draft_row["status"],
                "tasks": [] if draft_row.get("draft") is None else draft_row["draft"]["tasks"],
                "modelCalls": int(telemetry.get("modelCalls") or 0),
                "latencyMs": float(telemetry.get("latencyMs") or 0),
                "failureCategory": "schema_validation" if "ValidationError" in error else "other" if error else None,
            }
        skills.append({
            "packageId": assignment["packageId"], "assignmentId": assignment["assignmentId"],
            "name": parsed.name, "description": str(parsed.frontmatter.get("description") or ""),
            "repository": assignment["repository"], "sourcePath": assignment["sourcePath"],
            "commitSha": assignment["commitSha"], "pinnedSourceUrl": _pinned_url(assignment),
            "packageDigest": assignment["packageDigest"], "license": source.get("licenseSpdx"),
            "language": source.get("language"), "instructionRiskCodes": source.get("instructionRiskCodes", []),
            "taskSlots": assignment["taskSlots"], "draft": draft, "files": files,
        })
    all_files = [item for skill in skills for item in skill["files"]]
    body = {
        "apiVersion": INDEX_SCHEMA, "generatedAt": _utc_now(), "authority": AUTHORITY,
        "authorKitDigest": kit["workspaceDigest"],
        "sourceSnapshotManifestDigest": snapshot_digest,
        "draftReportDigest": None if draft_report is None else draft_report["reportDigest"],
        "statistics": {
            "skillCount": len(skills), "taskSlotCount": sum(len(item["taskSlots"]) for item in skills),
            "draftedSkillCount": sum(item["draft"]["status"] == "drafted" for item in skills),
            "failedDraftSkillCount": sum(item["draft"]["status"] == "failed" for item in skills),
            "draftedTaskCount": sum(len(item["draft"]["tasks"]) for item in skills),
            "fileCount": len(all_files), "displayableFileCount": sum(item["content"] is not None for item in all_files),
            "referenceFileCount": sum(item["path"].startswith("references/") for item in all_files),
            "executionSurfaceFileCount": sum(item["executionSurface"] for item in all_files),
            "thirdPartyExecutionAttempted": False,
        },
        "skills": skills,
        "claimBoundary": (
            "Read-only public Skill transparency index. Package text and model drafts are untrusted data; "
            "the library grants no Skill installation, Tool/MCP call, Runtime registration, Gold/Oracle, "
            "ES-P1 qualification, or execution authority."
        ),
    }
    index = {**body, "indexDigest": sha256_json(body)}
    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("public Skill library output root is unsafe")
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"skill-index.json", "skill-library.html", "library-manifest.json"}
    existing = {item.name for item in output.iterdir()}
    if existing - allowed:
        raise ValueError("public Skill library output contains unexpected files")
    index_path = output / "skill-index.json"
    html_path = output / "skill-library.html"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(_render_html(index), encoding="utf-8")
    manifest_body = {
        "apiVersion": MANIFEST_SCHEMA, "authority": AUTHORITY,
        "authorKitDigest": kit["workspaceDigest"], "indexDigest": index["indexDigest"],
        "files": {"skill-index.json": _file_digest(index_path), "skill-library.html": _file_digest(html_path)},
        "thirdPartyExecutionAttempted": False,
    }
    manifest = {**manifest_body, "manifestDigest": sha256_json(manifest_body)}
    (output / "library-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {**manifest, "statistics": index["statistics"], "html": str(html_path)}


def inspect_public_skill_library(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("public Skill library root is unsafe")
    expected_files = {"skill-index.json", "skill-library.html", "library-manifest.json"}
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("public Skill library cannot contain symlinks")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise ValueError("public Skill library file set mismatch")
    manifest = json.loads((root / "library-manifest.json").read_text(encoding="utf-8"))
    manifest_body = {key: value for key, value in manifest.items() if key != "manifestDigest"}
    if manifest.get("apiVersion") != MANIFEST_SCHEMA or manifest.get("manifestDigest") != sha256_json(manifest_body):
        raise ValueError("public Skill library manifest digest mismatch")
    if manifest.get("authority") != AUTHORITY or manifest.get("thirdPartyExecutionAttempted") is not False:
        raise ValueError("public Skill library authority boundary mismatch")
    for relative, digest in manifest["files"].items():
        if _file_digest(root / relative) != digest:
            raise ValueError("public Skill library file digest mismatch")
    index = json.loads((root / "skill-index.json").read_text(encoding="utf-8"))
    index_body = {key: value for key, value in index.items() if key != "indexDigest"}
    if index.get("apiVersion") != INDEX_SCHEMA or index.get("indexDigest") != sha256_json(index_body):
        raise ValueError("public Skill library index digest mismatch")
    if index.get("indexDigest") != manifest.get("indexDigest") or index.get("authority") != AUTHORITY:
        raise ValueError("public Skill library index/manifest binding mismatch")
    html = (root / "skill-library.html").read_text(encoding="utf-8")
    match = _EMBEDDED_RE.search(html)
    if match is None:
        raise ValueError("public Skill library HTML data binding is missing")
    embedded = json.loads(base64.b64decode(match.group(1), validate=True).decode("utf-8"))
    if embedded != index:
        raise ValueError("public Skill library HTML data binding mismatch")
    return {
        "status": "valid", "verified": True, "authority": AUTHORITY,
        "manifestDigest": manifest["manifestDigest"], "indexDigest": index["indexDigest"],
        "authorKitDigest": index["authorKitDigest"], "draftReportDigest": index["draftReportDigest"],
        "statistics": index["statistics"], "thirdPartyExecutionAttempted": False,
        "html": str(root / "skill-library.html"), "claimBoundary": index["claimBoundary"],
    }


def export_public_skill_library_summary(root_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Export commit-safe metadata; third-party package bodies remain artifact-only."""

    root = Path(root_path).expanduser().resolve()
    inspected = inspect_public_skill_library(root)
    index = json.loads((root / "skill-index.json").read_text(encoding="utf-8"))
    body = {
        "apiVersion": SUMMARY_SCHEMA, "generatedAt": index["generatedAt"],
        "authority": "metadata_only_no_embedded_third_party_content_no_execution_authority",
        "indexDigest": inspected["indexDigest"], "manifestDigest": inspected["manifestDigest"],
        "authorKitDigest": inspected["authorKitDigest"], "draftReportDigest": inspected["draftReportDigest"],
        "statistics": inspected["statistics"],
        "skills": [{
            "assignmentId": item["assignmentId"], "packageId": item["packageId"],
            "name": item["name"], "description": item["description"],
            "repository": item["repository"], "sourcePath": item["sourcePath"],
            "commitSha": item["commitSha"], "pinnedSourceUrl": item["pinnedSourceUrl"],
            "packageDigest": item["packageDigest"], "license": item["license"],
            "language": item["language"], "instructionRiskCodes": item["instructionRiskCodes"],
            "draftStatus": item["draft"]["status"], "draftedTaskCount": len(item["draft"]["tasks"]),
            "taskSlots": item["taskSlots"], "filePaths": [file["path"] for file in item["files"]],
        } for item in index["skills"]],
        "containsThirdPartyFileContent": False, "thirdPartyExecutionAttempted": False,
        "claimBoundary": (
            "Versioned metadata index only. Follow each pinned source URL or regenerate the local "
            "read-only library to inspect package content. No Gold, Oracle, qualification, or execution authority."
        ),
    }
    summary = {**body, "summaryDigest": sha256_json(body)}
    destination = Path(output_path).expanduser().resolve()
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("public Skill library summary output is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "AUTHORITY", "INDEX_SCHEMA", "MANIFEST_SCHEMA", "SUMMARY_SCHEMA",
    "build_public_skill_library", "export_public_skill_library_summary", "inspect_public_skill_library",
]
