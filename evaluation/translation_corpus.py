"""Build a static, inspectable corpus for L1 -> L0 generalization work.

This gate is deliberately separate from the Runtime package gate.  A public
Skill can be useful translation evidence even when a referenced document is
missing from the marketplace snapshot, while the same package must remain
ineligible for Runtime execution.  Non-conformant SKILL.md files are retained
only as a robustness cohort and never inflate the primary translation score.

All third-party material is treated as inert text.  Nothing in this module
imports, installs, shells out to, or executes package content.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effect_runtime.skill_package import inspect_skill_package
from evaluation.public_skill_corpus import (
    _repo_key,
    inspect_public_snapshot,
    load_discovery,
)
from network_runtime.contracts import sha256_json
from skills.skill_format import SkillFormatError, parse_skill_md


CORPUS_SCHEMA = "effect-runtime.io/translation-generalization-corpus/v1"
INDEX_SCHEMA = "effect-runtime.io/translation-generalization-index/v1"
AUTHORITY = "offline_translation_research_input_no_runtime_authority"
DEFAULT_SEED = "ensured-skill-translation-development-20260902"
_REFERENCE_ERRORS = frozenset({
    "RESOURCE_REFERENCE_MISSING",
    "RESOURCE_REFERENCE_UNSAFE",
})
_MAX_DISPLAY_BYTES = 512 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _classification(report: dict[str, Any]) -> tuple[str, bool]:
    error_codes = {
        str(item["code"])
        for item in report["findings"]
        if item["severity"] == "error"
    }
    if report["executionEligible"] is True:
        return "runtime_ready", True
    if report["skill"] is not None and error_codes and error_codes <= _REFERENCE_ERRORS:
        return "translation_only_partial_context", True
    if "SKILL_FORMAT_INVALID" in error_codes:
        return "format_variant_robustness_only", False
    return "excluded_from_translation", False


def _display_files(package: Path, sealed_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in sealed_files:
        relative = str(record["path"])
        path = (package / relative).resolve()
        data = path.read_bytes()
        content: str | None = None
        if len(data) <= _MAX_DISPLAY_BYTES and b"\x00" not in data:
            content = data.decode("utf-8", errors="replace")
        result.append({
            "path": relative,
            "bytes": len(data),
            "sha256": record["sha256"],
            "displayedAsInertText": content is not None,
            "content": content,
        })
    return result


def _display_quarantine(snapshot: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    """The inspected snapshot authenticates these text-only evidence files."""
    return [
        {
            "path": "[quarantine] " + item["sourcePath"],
            "sourcePath": item["sourcePath"], "bytes": item["sourceBytes"],
            "sha256": item["sourceDigest"], "displayedAsInertText": item["evidencePath"] is not None,
            "content": (snapshot / item["evidencePath"]).read_text(encoding="utf-8")
            if item["evidencePath"] is not None and item["sourceBytes"] <= _MAX_DISPLAY_BYTES else None,
            "runtimeResource": False,
        }
        for item in record.get("quarantinedFiles", [])
    ]


def _quarantined_entry(snapshot: Path, record: dict[str, Any]) -> tuple[dict | None, bool]:
    """Read a quarantined executable-mode SKILL.md as data, never a symlink.

    Format qualification is separate from missing Runtime resources. This
    does not restore files, infer reference closure or grant package authority.
    """
    for item in record.get("quarantinedFiles", []):
        if (item["sourcePath"] == "SKILL.md" and item["sourceMode"] == "100755"
                and item["representation"] == "inert_utf8_text"):
            try:
                parsed = parse_skill_md((snapshot / item["evidencePath"]).read_text(encoding="utf-8"))
            except SkillFormatError:
                return None, True
            return parsed.frontmatter, True
    return None, False


def _development_batches(
    skills: list[dict[str, Any]], *, seed: str, batch_size: int,
) -> list[dict[str, Any]]:
    """Create deterministic repository-grouped development batches.

    These batches support iterative diagnosis only.  They are not frozen or
    unseen evidence, and the manifest makes that distinction explicit.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for skill in skills:
        groups.setdefault(skill["repository"], []).append(skill)
    repositories = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for repository in repositories:
        group = sorted(groups[repository], key=lambda item: item["packageId"])
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return [
        {
            "batchId": f"development-{index:02d}",
            "evidenceRole": "development_only",
            "packageIds": [item["packageId"] for item in batch],
            "skillCount": len(batch),
            "repositoryCount": len({item["repository"] for item in batch}),
            "domains": sorted({item["domain"] for item in batch}),
            "runtimeReadyCount": sum(item["runtimeReady"] for item in batch),
        }
        for index, batch in enumerate(batches, start=1)
    ]


def _render_html(index: dict[str, Any]) -> str:
    payload = base64.b64encode(
        json.dumps(index, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).decode("ascii")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'">
<title>L1→L0 转译 Skill 语料库</title><style>
:root{{color-scheme:dark;--bg:#07111d;--panel:#0d1b2a;--line:#263c52;--text:#edf5fa;--muted:#9fb1c0;--ok:#4ed9bd;--warn:#ffc66d;--bad:#ff7777}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}header{{padding:26px max(18px,calc((100vw - 1500px)/2));border-bottom:1px solid var(--line)}}h1{{margin:4px 0;font-size:38px}}.muted{{color:var(--muted)}}.notice{{max-width:1100px;padding:12px 14px;border-left:4px solid var(--warn);background:#211f25}}.stats{{display:flex;flex-wrap:wrap;gap:9px;margin-top:15px}}.stat{{min-width:145px;padding:10px 12px;background:var(--panel);border:1px solid var(--line);border-radius:10px}}.stat b{{display:block;font-size:22px}}main{{max-width:1500px;margin:auto;padding:16px;display:grid;grid-template-columns:380px 1fr;gap:14px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:13px;overflow:hidden}}.toolbar{{display:grid;gap:8px;padding:12px;border-bottom:1px solid var(--line)}}input,select{{padding:9px;background:#071522;color:var(--text);border:1px solid #38536a;border-radius:7px}}#list{{max-height:calc(100vh - 245px);overflow:auto}}button.skill,button.file{{width:100%;padding:11px;text-align:left;color:inherit;background:none;border:0;border-bottom:1px solid var(--line);cursor:pointer}}button.skill:hover,button.skill.active,button.file.active{{background:#18344a}}.tags{{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}}.tag{{padding:1px 7px;border:1px solid #3c566d;border-radius:999px;font-size:11px;color:var(--muted)}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}.bad{{color:var(--bad)}}.detail{{padding:18px}}.facts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:14px 0}}.fact{{overflow-wrap:anywhere}}.files{{display:grid;grid-template-columns:250px 1fr;gap:10px}}.filelist{{max-height:480px;overflow:auto;border:1px solid var(--line)}}pre{{margin:0;min-height:420px;max-height:680px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;padding:13px;background:#06101a;border:1px solid var(--line);font:12px/1.5 ui-monospace,monospace}}@media(max-width:900px){{main{{grid-template-columns:1fr}}#list{{max-height:380px}}.files,.facts{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="ok">EnsuredSkill · translation-first evidence</div><h1>L1→L0 转译 Skill 语料库</h1><p class="notice" id="claim"></p><div class="stats" id="stats"></div></header>
<main><aside class="panel"><div class="toolbar"><input id="q" placeholder="搜索 Skill、仓库、领域"><select id="class"><option value="">全部资格</option><option value="runtime_ready">Runtime-ready</option><option value="translation_only_partial_context">仅转译/上下文不完整</option><option value="format_variant_robustness_only">格式变体/鲁棒性</option></select><span id="count" class="muted"></span></div><div id="list"></div></aside><article id="detail" class="panel"><div class="detail muted">选择一个 Skill 查看固定原文。</div></article></main>
<script type="application/json" id="data">{payload}</script><script>(()=>{{'use strict';const raw=document.getElementById('data').textContent;const bytes=Uint8Array.from(atob(raw),c=>c.charCodeAt(0));const D=JSON.parse(new TextDecoder().decode(bytes));let selected=D.skills[0]?.packageId||null;const e=(t,c,x)=>{{const n=document.createElement(t);if(c)n.className=c;if(x!==undefined)n.textContent=String(x);return n}};document.getElementById('claim').textContent=D.claimBoundary;for(const [k,v] of [['静态 Skill',D.statistics.skillCount],['主要转译语料',D.statistics.primaryEligibleCount],['Runtime-ready',D.statistics.runtimeReadyCount],['鲁棒性格式变体',D.statistics.robustnessOnlyCount],['来源仓库',D.statistics.repositoryCount],['第三方执行','0 / 否']]){{const n=e('div','stat');n.append(e('span','muted',k),e('b','',v));document.getElementById('stats').append(n)}}const q=document.getElementById('q'),cls=document.getElementById('class');function rows(){{const s=q.value.trim().toLowerCase();return D.skills.filter(x=>(!cls.value||x.classification===cls.value)&&(!s||[x.name,x.repository,x.domain,x.packageId].join(' ').toLowerCase().includes(s)))}}function list(){{const root=document.getElementById('list'),xs=rows();root.replaceChildren();document.getElementById('count').textContent=xs.length+' / '+D.skills.length;for(const x of xs){{const b=e('button','skill'+(x.packageId===selected?' active':''));b.append(e('b','',x.name),e('div','muted',x.repository+' · '+x.domain));const tags=e('div','tags');tags.append(e('span','tag '+(x.runtimeReady?'ok':x.primaryTranslationEligible?'warn':'bad'),x.classification),e('span','tag',x.language));b.append(tags);b.onclick=()=>{{selected=x.packageId;list();detail(x)}};root.append(b)}}}}function detail(x){{const root=document.getElementById('detail');root.replaceChildren();const d=e('div','detail');d.append(e('div','ok',x.classification),e('h2','',x.name),e('p','muted',x.description||'无独立描述'));const facts=e('div','facts');for(const [a,b] of [['Repository',x.repository],['Source path',x.sourcePath],['Commit',x.commitSha],['Domain',x.domain],['Runtime gate',x.runtimeReady?'passed':'blocked'],['Primary score',x.primaryTranslationEligible?'eligible':'not eligible'],['Findings',x.findingCodes.join(', ')||'none'],['Package digest',x.packageDigest]]){{const f=e('div','fact');f.append(e('b','muted',a),e('div','',b));facts.append(f)}}d.append(facts,e('p','warn','文件只按纯文本展示；Skill 指令与附件没有执行权。'));const files=e('div','files'),fl=e('div','filelist'),pre=e('pre','','');function show(f){{for(const b of fl.querySelectorAll('button'))b.classList.toggle('active',b.dataset.path===f.path);pre.textContent=f.content===null?'[二进制或超过显示上限，仅保留摘要]':f.content}}for(const f of x.files){{const b=e('button','file',f.path+' · '+f.bytes+' B');b.dataset.path=f.path;b.onclick=()=>show(f);fl.append(b)}}files.append(fl,pre);d.append(files);root.append(d);if(x.files.length)show(x.files.find(f=>f.path==='SKILL.md')||x.files[0])}}q.oninput=list;cls.onchange=list;list();if(selected)detail(D.skills.find(x=>x.packageId===selected))}})();</script></body></html>"""


def build_translation_corpus(
    snapshot_root: str | Path,
    output_root: str | Path,
    *,
    discovery_path: str | Path | None = None,
    seed: str = DEFAULT_SEED,
    batch_size: int = 12,
) -> dict[str, Any]:
    """Classify and index a known public corpus for translation development."""

    if batch_size < 1:
        raise ValueError("translation corpus batch size must be positive")
    snapshot = Path(snapshot_root).expanduser().resolve()
    snapshot_info = inspect_public_snapshot(snapshot)
    snapshot_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("translation corpus output must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)

    discovery_by_id: dict[str, dict[str, Any]] = {}
    if discovery_path is not None:
        discovery = load_discovery(discovery_path)
        if discovery["discoveryDigest"] != snapshot_manifest["discoveryDigest"]:
            raise ValueError("translation corpus discovery/snapshot digest mismatch")
        discovery_by_id = {str(item["id"]): item for item in discovery["candidates"]}

    accepted = [
        row for row in _read_jsonl(snapshot / "records.jsonl")
        if row.get("status") == "accepted"
    ]
    skills: list[dict[str, Any]] = []
    for record in accepted:
        package = snapshot / "packages" / record["packageId"]
        report = inspect_skill_package(package)
        classification, primary_eligible = _classification(report)
        withheld = bool(record.get("withheldFiles"))
        source_skill = report.get("skill")
        quarantined_entry, entry_seen = _quarantined_entry(snapshot, record)
        if source_skill is None and entry_seen:
            source_skill = quarantined_entry
            primary_eligible = source_skill is not None
            classification = "translation_only_partial_context" if primary_eligible else "format_variant_robustness_only"
        if withheld and primary_eligible:
            classification = "translation_only_partial_context"
        discovery_row = discovery_by_id.get(str(record["candidateId"]), {})
        finding_codes = sorted({str(item["code"]) for item in report["findings"]})
        skills.append({
            "packageId": record["packageId"],
            "candidateId": record["candidateId"],
            "name": (source_skill or {}).get("name") or record["name"],
            "description": discovery_row.get("description", ""),
            "repository": record["repository"],
            "sourcePath": record["sourcePath"],
            "commitSha": record["commitSha"],
            "packageDigest": record["packageDigest"],
            "license": record["licenseSpdx"],
            "language": record["language"],
            "domain": discovery_row.get("discoveryQuery", "unclassified"),
            "classification": classification,
            "primaryTranslationEligible": primary_eligible,
            "runtimeReady": report["executionEligible"] is True and not withheld,
            "contextComplete": not withheld and not bool(set(finding_codes) & _REFERENCE_ERRORS),
            "sourceEvidenceDigest": record.get("sourceEvidenceDigest", record["packageDigest"]),
            "quarantinedSourceFileCount": len(record.get("quarantinedFiles", [])),
            "withheldResourceCount": len(record.get("withheldFiles", [])),
            "formatConformant": source_skill is not None,
            "entrySource": "quarantined_text" if entry_seen else "package",
            "findingCodes": finding_codes,
            "instructionRiskCodes": record.get("instructionRiskCodes", []),
            "files": _display_files(package, record["files"]) + _display_quarantine(snapshot, record),
        })
    skills.sort(key=lambda item: item["packageId"])
    primary = [item for item in skills if item["primaryTranslationEligible"]]
    batches = _development_batches(primary, seed=seed, batch_size=batch_size)
    classes = Counter(item["classification"] for item in skills)
    domains = Counter(item["domain"] for item in primary)
    index_body = {
        "apiVersion": INDEX_SCHEMA,
        "generatedAt": _utc_now(),
        "authority": AUTHORITY,
        "sourceSnapshotDigest": snapshot_info["manifestDigest"],
        "statistics": {
            "skillCount": len(skills),
            "primaryEligibleCount": len(primary),
            "runtimeReadyCount": sum(item["runtimeReady"] for item in skills),
            "partialContextCount": classes["translation_only_partial_context"],
            "robustnessOnlyCount": classes["format_variant_robustness_only"],
            "repositoryCount": len({item["repository"] for item in skills}),
            "primaryRepositoryCount": len({item["repository"] for item in primary}),
            "domainCount": len(domains),
            "thirdPartyExecutionAttempted": False,
        },
        "classificationCounts": dict(sorted(classes.items())),
        "primaryDomainCounts": dict(sorted(domains.items())),
        "skills": skills,
        "claimBoundary": (
            "Known public development inventory only. Primary eligibility means the Skill can be "
            "studied as inert L1 text; it does not grant Runtime eligibility, prove translation "
            "generalization, or state a production success probability."
        ),
    }
    index = {**index_body, "indexDigest": sha256_json(index_body)}
    _write_json(root / "index.json", index)
    batch_body = {
        "apiVersion": CORPUS_SCHEMA,
        "createdAt": _utc_now(),
        "authority": AUTHORITY,
        "sourceSnapshotDigest": snapshot_info["manifestDigest"],
        "indexDigest": index["indexDigest"],
        "seed": seed,
        "batchSizeTarget": batch_size,
        "corpusRole": "known_development_inventory",
        "proofCohortEligible": False,
        "primaryPackageIds": [item["packageId"] for item in primary],
        "robustnessPackageIds": [
            item["packageId"] for item in skills if not item["primaryTranslationEligible"]
        ],
        "batches": batches,
        "futureProofRequirements": {
            "translatorFrozenBeforeCollection": True,
            "minimumDisjointUnseenCohorts": 3,
            "minimumUniqueSkills": 50,
            "minimumUniqueRepositories": 15,
            "minimumDomains": 8,
            "minimumCases": 600,
            "repositoryOverlapAcrossProofCohorts": False,
            "skillOverlapWithDevelopmentInventory": False,
            "independentSkillTaskToolAlignmentReview": True,
            "runtimeOrDshExecutedBeforeAdmission": False,
        },
        "thirdPartyExecutionAttempted": False,
    }
    batches_manifest = {**batch_body, "corpusDigest": sha256_json(batch_body)}
    _write_json(root / "batches.json", batches_manifest)
    (root / "skill-library.html").write_text(_render_html(index), encoding="utf-8")
    sealed_files = {
        path.name: _file_digest(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    workspace_body = {
        "apiVersion": CORPUS_SCHEMA,
        "createdAt": _utc_now(),
        "authority": AUTHORITY,
        "sourceSnapshotDigest": snapshot_info["manifestDigest"],
        "indexDigest": index["indexDigest"],
        "corpusDigest": batches_manifest["corpusDigest"],
        "sealedFiles": sealed_files,
        "proofCohortEligible": False,
        "runtimeAuthorityGranted": False,
        "thirdPartyExecutionAttempted": False,
    }
    workspace = {**workspace_body, "workspaceDigest": sha256_json(workspace_body)}
    _write_json(root / "workspace.json", workspace)
    return workspace


def inspect_translation_corpus(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    workspace = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in workspace.items() if key != "workspaceDigest"}
    if workspace.get("apiVersion") != CORPUS_SCHEMA or workspace.get("workspaceDigest") != sha256_json(body):
        raise ValueError("translation corpus workspace digest mismatch")
    if any((
        workspace.get("authority") != AUTHORITY,
        workspace.get("proofCohortEligible") is not False,
        workspace.get("runtimeAuthorityGranted") is not False,
        workspace.get("thirdPartyExecutionAttempted") is not False,
    )):
        raise ValueError("translation corpus authority boundary drift")
    actual = {
        path.name: _file_digest(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "workspace.json"
    }
    if actual != workspace.get("sealedFiles"):
        raise ValueError("translation corpus sealed file drift")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    index_body = {key: value for key, value in index.items() if key != "indexDigest"}
    batches = json.loads((root / "batches.json").read_text(encoding="utf-8"))
    batch_body = {key: value for key, value in batches.items() if key != "corpusDigest"}
    if index.get("indexDigest") != sha256_json(index_body) or index["indexDigest"] != workspace["indexDigest"]:
        raise ValueError("translation corpus index digest drift")
    if batches.get("corpusDigest") != sha256_json(batch_body) or batches["corpusDigest"] != workspace["corpusDigest"]:
        raise ValueError("translation corpus batch digest drift")
    primary_ids = set(batches["primaryPackageIds"])
    batched_ids = [item for batch in batches["batches"] for item in batch["packageIds"]]
    if len(batched_ids) != len(set(batched_ids)) or set(batched_ids) != primary_ids:
        raise ValueError("translation corpus development batch coverage drift")
    return {
        "status": "valid",
        "workspaceDigest": workspace["workspaceDigest"],
        "sourceSnapshotDigest": workspace["sourceSnapshotDigest"],
        "skillCount": index["statistics"]["skillCount"],
        "primaryEligibleCount": index["statistics"]["primaryEligibleCount"],
        "runtimeReadyCount": index["statistics"]["runtimeReadyCount"],
        "robustnessOnlyCount": index["statistics"]["robustnessOnlyCount"],
        "repositoryCount": index["statistics"]["repositoryCount"],
        "domainCount": index["statistics"]["domainCount"],
        "developmentBatchCount": len(batches["batches"]),
        "proofCohortEligible": False,
        "runtimeAuthorityGranted": False,
        "thirdPartyExecutionAttempted": False,
    }


def build_public_sampling_report(
    sampling_root: str | Path, batches_root: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    """Join frozen sampling to static evidence, retaining every failed candidate.

    This is a source-acquisition report, not a new corpus, translator result or
    admission decision. Inspect existing snapshots/libraries; never fetch sources,
    substitute candidates, rewrite old evidence or infer semantic qualification.
    """
    sampling, batches = Path(sampling_root).resolve(), Path(batches_root).resolve()
    plan = json.loads((sampling / "sampling.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "samplingDigest"}
    if plan.get("samplingDigest") != sha256_json(body):
        raise ValueError("sampling digest mismatch")
    discovery = load_discovery(sampling / "discovery.json")
    if plan.get("sampledDiscoveryDigest") != discovery["discoveryDigest"]:
        raise ValueError("sampled discovery mismatch")
    candidates = {row["id"]: row for row in discovery["candidates"]}
    ids = [cid for batch in plan["batches"] for cid in batch["candidateIds"]]
    if len(ids) != len(set(ids)) or set(ids) != set(candidates) or len(ids) != plan["selectedCount"]:
        raise ValueError("sampling candidate coverage mismatch")
    evidence, outcomes, texts, seen_repositories = [], [], {}, set()
    for batch in plan["batches"]:
        batch_id = batch["batchId"]
        # These names come from a manifest, not a trusted path argument.
        if not isinstance(batch_id, str) or not batch_id or Path(batch_id).name != batch_id or batch_id in {".", ".."}:
            raise ValueError("unsafe source batch path")
        root = (batches / batch_id).resolve()
        if not root.is_relative_to(batches):
            raise ValueError("unsafe source batch path")
        batch_discovery = load_discovery(root / "discovery.json")
        if (batch_discovery.get("parentSamplingDigest") != plan["samplingDigest"]
                or batch_discovery["candidates"] != [candidates[cid] for cid in batch["candidateIds"]]):
            raise ValueError("source batch differs from frozen selection")
        snapshot = inspect_public_snapshot(root / "snapshot")
        manifest = json.loads((root / "snapshot/manifest.json").read_text())
        library = inspect_translation_corpus(root / "library")
        if (manifest["discoveryDigest"] != batch_discovery["discoveryDigest"]
                or library["sourceSnapshotDigest"] != snapshot["manifestDigest"]):
            raise ValueError("source batch evidence binding mismatch")
        records = _read_jsonl(root / "snapshot/records.jsonl")
        by_id = {row["candidateId"]: row for row in records}
        if len(records) != len(by_id) or not set(by_id) <= set(batch["candidateIds"]):
            raise ValueError("snapshot has duplicate or unselected candidate")
        if any(row["status"] not in {"accepted", "excluded"} for row in records):
            raise ValueError("unexpected acquisition outcome")
        if any(row["githubUrl"] != candidates[row["candidateId"]]["githubUrl"] for row in records):
            raise ValueError("snapshot source differs from selected candidate")
        index = json.loads((root / "library/index.json").read_text())
        indexed = {row["candidateId"]: row for row in index["skills"]}
        if (len(indexed) != len(index["skills"])
                or set(indexed) != {row["candidateId"] for row in records if row["status"] == "accepted"}):
            raise ValueError("library omitted or added an accepted candidate")
        repos = {_repo_key(row["githubUrl"]) for row in batch_discovery["candidates"]}
        if repos & (seen_repositories | set(plan["excludedRepositories"])):
            raise ValueError("repository exposure or cross-batch overlap")
        seen_repositories.update(repos)
        evidence.append({
            "batchId": batch_id, "selectedCount": len(batch["candidateIds"]),
            "processedCount": len(records), "acceptedCount": len(indexed),
            "allCandidatesProcessed": len(records) == len(batch["candidateIds"]),
            "importerAcceptedTargetMet": manifest["complete"],
            "discoveryDigest": batch_discovery["discoveryDigest"],
            "snapshotDigest": snapshot["manifestDigest"], "libraryDigest": library["workspaceDigest"],
        })
        for cid in batch["candidateIds"]:
            row, skill, candidate = by_id.get(cid, {}), indexed.get(cid, {}), candidates[cid]
            if skill and skill["packageDigest"] != row["packageDigest"]:
                raise ValueError("package digest differs across snapshot and index")
            outcomes.append({
                "candidateId": cid, "batchId": batch_id, "name": candidate["name"],
                "githubUrl": candidate["githubUrl"], "queryStratum": candidate["discoveryQuery"],
                "status": row.get("status", "not_processed"), "reason": row.get("reason"),
                "packageId": skill.get("packageId"), "commitSha": skill.get("commitSha"),
                "sourceEvidenceDigest": skill.get("sourceEvidenceDigest"),
                "classification": skill.get("classification"),
                "staticPackageGatePassed": skill.get("runtimeReady", False),
                "formatQualified": skill.get("primaryTranslationEligible", False),
                "quarantinedSourceFileCount": skill.get("quarantinedSourceFileCount", 0),
            })
            texts[cid] = skill.get("files", [])
    counts = Counter(row["status"] for row in outcomes)
    report = {
        "apiVersion": "effect-runtime.io/public-skill-sampling-report/v1",
        "samplingDigest": plan["samplingDigest"], "batches": evidence, "candidates": outcomes,
        "statistics": {
            "sampledCandidates": len(outcomes), "acceptedSkills": counts["accepted"],
            "excludedCandidates": counts["excluded"], "unprocessedCandidates": counts["not_processed"],
            "sampledRepositories": len(seen_repositories),
            "acceptedRepositories": len({_repo_key(row["githubUrl"])
                                         for row in outcomes if row["status"] == "accepted"}),
            "queryStrata": len({row["queryStratum"] for row in outcomes}), "verifiedDomains": None,
            "formatQualified": sum(row["formatQualified"] for row in outcomes),
            "staticPackageGatePassed": sum(row["staticPackageGatePassed"] for row in outcomes),
            "classificationCounts": dict(Counter(row["classification"] for row in outcomes if row["classification"])),
            "exclusionReasons": dict(Counter(row["reason"] for row in outcomes if row["reason"])),
            "quarantinedSourceFiles": sum(row["quarantinedSourceFileCount"] for row in outcomes),
        },
        "allCandidatesProcessed": counts["not_processed"] == 0,
        "proofCohortEligible": False, "runtimeAuthorityGranted": False,
        "thirdPartyExecutionAttempted": False, "translationMetrics": None,
        "claimBoundary": "入库与静态包检查不等于转译成功或执行许可。搜索类别不是核实领域。"
        " / Acquisition and static package checks are not semantic success or execution authority; query labels are not verified domains.",
    }
    report["reportDigest"] = sha256_json(report)
    target = Path(output_root).resolve()
    target.mkdir(parents=True, exist_ok=False)
    _write_json(target / "report.json", report)
    # Native details are closed by default. Escape every byte of untrusted text;
    # no script, remote resource, Markdown rendering or source execution.
    sections = []
    for row in outcomes:
        files = "".join(
            "<details><summary>" + html.escape(f["path"]) + "</summary><pre>"
            + html.escape(f["content"] if f["content"] is not None else "[仅摘要 / hash only]")
            + "</pre></details>" for f in texts[row["candidateId"]]
        )
        sections.append("<details><summary>" + html.escape(
            f'{row["batchId"]} · {row["name"]} · {row["classification"] or row["status"]}'
        ) + "</summary><pre>" + html.escape(json.dumps(row, ensure_ascii=False, indent=2))
            + "</pre>" + files + "</details>")
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
            'style-src \'unsafe-inline\'; form-action \'none\'; base-uri \'none\'">'
            '<title>公开 Skill 抽样与原文 / Public Skill Sources</title>'
            '<style>body{max-width:1200px;margin:24px auto;padding:0 16px;font:15px/1.6 system-ui;'
            'color:#183042;background:#f4f7fa}details{border:1px solid #ccd6df;padding:10px;margin:8px 0;'
            'background:white}summary{cursor:pointer;overflow-wrap:anywhere}pre{white-space:pre-wrap;'
            'overflow-wrap:anywhere;max-height:600px;overflow:auto;font:12px/1.6 monospace}</style>'
            '<h1>公开 Skill 抽样与原文</h1><p>' + html.escape(report["claimBoundary"])
            + '</p><p>全部候选均保留；点击展开信息与原文。All sampled candidates retained; click to inspect.</p><pre>'
            + html.escape(json.dumps(report["statistics"], ensure_ascii=False, indent=2))
            + '</pre>' + "".join(sections) + '</html>')
    (target / "skill-library.html").write_text(page, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("snapshot_root")
    build.add_argument("--output-root", required=True)
    build.add_argument("--discovery")
    build.add_argument("--seed", default=DEFAULT_SEED)
    build.add_argument("--batch-size", type=int, default=12)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("root")
    sample_report = commands.add_parser("sample-report")
    sample_report.add_argument("sampling_root")
    sample_report.add_argument("--batches-root", required=True)
    sample_report.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_translation_corpus(
            args.snapshot_root,
            args.output_root,
            discovery_path=args.discovery,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    elif args.command == "sample-report":
        result = build_public_sampling_report(args.sampling_root, args.batches_root, args.output_root)
    else:
        result = inspect_translation_corpus(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
