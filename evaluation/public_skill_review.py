"""Model-assisted, human-owned case-author review kit for ES-P1-Wild.

Only candidate user prompts cross the model-assistance boundary. Model-proposed
semantic labels are deliberately withheld and can never become Gold here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from evaluation.public_skill_corpus import inspect_public_author_kit
from evaluation.public_skill_draft_author import inspect_public_market_drafts
from evaluation.public_skill_fixture_mcp import (
    CATALOG_SCHEMA as FIXTURE_CATALOG_SCHEMA,
    FIXTURE_SCHEMA,
    validate_fixture_catalog,
    validate_fixture_state,
)
from network_runtime.contracts import sha256_json


REVIEW_KIT_SCHEMA_V2 = "effect-runtime.io/public-skill-assisted-review-kit/v2"
REVIEW_KIT_SCHEMA = "effect-runtime.io/public-skill-assisted-review-kit/v3"
REVIEW_SCHEMA = "effect-runtime.io/public-skill-case-author-review/v1"
SOURCE_SCHEMA = "effect-runtime.io/public-skill-prompt-candidates/v1"
AUTHORITY = "case_authoring_only_no_gold_or_execution_authority"
GOLD_KIT_SCHEMA = "effect-runtime.io/public-skill-blind-gold-kit/v1"
GOLD_REVIEW_SCHEMA = "effect-runtime.io/public-skill-gold-author-review/v1"
GOLD_AUTHORITY = "blind_gold_authoring_only_no_execution_or_qualification_authority"
_DECISIONS = {"pending", "accept_prompt", "edit_prompt", "author_from_scratch", "reject_slot"}
_GOLD_DECISIONS = {"pending", "author_gold", "reject_task"}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_MATERIAL_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}
_MAX_MATERIAL_FILE_BYTES = 1024 * 1024
_MAX_MATERIAL_TOTAL_BYTES = 8 * 1024 * 1024
_EMBEDDED_RE = re.compile(
    r'<script type="application/json" id="review-queue-data">([A-Za-z0-9+/=]+)</script>'
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _review_template(assignment: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    candidates = {item["slotId"]: item for item in candidate["slots"]}
    slots = []
    for slot in assignment["taskSlots"]:
        prompt = candidates[slot["slotId"]]["promptCandidate"]
        slots.append({
            "slotId": slot["slotId"], "challenge": slot["challenge"], "decision": "pending",
            "rationale": "", "promptOrigin": "model_candidate" if prompt is not None else "none",
            "task": {
                "apiVersion": "effect-runtime.io/public-skill-task/v1",
                "taskId": slot["slotId"], "assignmentId": assignment["assignmentId"],
                "packageId": assignment["packageId"], "packageDigest": assignment["packageDigest"],
                "language": candidate.get("language") or "en", "challenge": slot["challenge"],
                "userPrompt": prompt or "", "fixtureRefs": [], "toolCatalogRef": "",
                "authorId": "", "authoredAt": "",
            },
        })
    return {
        "apiVersion": REVIEW_SCHEMA, "assignmentId": assignment["assignmentId"],
        "packageId": assignment["packageId"], "packageDigest": assignment["packageDigest"],
        "reviewer": {
            "authorId": "", "role": "independent_public_case_author",
            "independentFromRuntimeTeam": None, "modelPromptAssistanceDisclosed": True,
        },
        "slots": slots,
        "authority": AUTHORITY,
    }


def _queue_html(payload: dict[str, Any]) -> str:
    encoded = base64.b64encode(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'"><title>ES-P1-Wild Case Author Review Queue</title><style>
:root{{color-scheme:dark;--bg:#07111e;--panel:#0e1d2c;--line:#263c51;--text:#e9f1f6;--muted:#9eb1c2;--cyan:#4ed8c4;--amber:#ffc16f;--red:#ff7474}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 85% 0,#183c51 0,transparent 34%),var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:28px 20px 50px}}h1{{font-size:clamp(28px,4vw,46px);margin:4px 0}}.eyebrow{{color:var(--cyan);font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-size:12px}}.notice{{border-left:4px solid var(--amber);background:#201f25;padding:13px 15px;margin:18px 0}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.card,.panel{{background:linear-gradient(150deg,#12263a,var(--panel));border:1px solid var(--line);border-radius:14px}}.card{{padding:14px}}.card strong{{font-size:25px;display:block}}.layout{{display:grid;grid-template-columns:330px minmax(0,1fr);gap:14px;margin-top:15px}}.toolbar{{padding:12px;border-bottom:1px solid var(--line)}}input{{width:100%;background:#071522;border:1px solid #38566f;color:var(--text);border-radius:8px;padding:9px}}#list{{max-height:690px;overflow:auto}}button{{width:100%;background:none;color:inherit;border:0;border-bottom:1px solid var(--line);text-align:left;padding:12px;cursor:pointer}}button:hover,button.active{{background:#18334c}}.muted{{color:var(--muted)}}.pad{{padding:18px}}.slot{{border:1px solid var(--line);border-radius:10px;background:#091724;padding:12px;margin:10px 0}}.tag{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 7px;font-size:11px;color:var(--muted)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#06111b;border:1px solid var(--line);border-radius:8px;padding:10px;color:#dce9f0}}code{{color:#bddff5;overflow-wrap:anywhere}}@media(max-width:850px){{.layout{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><div class="eyebrow">EnsuredSkill · assisted public case authoring</div><h1>Case Author 审阅队列</h1><p class="muted">只审阅用户问题候选；模型语义答案已隔离，不会预填 Gold。</p><div class="notice" id="claim"></div><div class="stats" id="stats"></div><div class="layout"><aside class="panel"><div class="toolbar"><input id="search" placeholder="搜索 Skill、仓库或问题"></div><div id="list"></div></aside><section class="panel" id="detail"><div class="pad muted">选择一个 assignment。</div></section></div></main><script type="application/json" id="review-queue-data">{encoded}</script><script>
(()=>{{'use strict';const raw=document.getElementById('review-queue-data').textContent;const bytes=Uint8Array.from(atob(raw),c=>c.charCodeAt(0));const d=JSON.parse(new TextDecoder().decode(bytes));const el=(t,c,x)=>{{const n=document.createElement(t);if(c)n.className=c;if(x!==undefined)n.textContent=String(x);return n}};document.getElementById('claim').textContent=d.claimBoundary;for(const [k,v] of [['Assignments',d.statistics.assignmentCount],['Slots',d.statistics.slotCount],['候选问题',d.statistics.promptCandidateCount],['从零编写',d.statistics.missingPromptCount]]){{const c=el('div','card');c.append(el('span','muted',k),el('strong','',v));document.getElementById('stats').append(c)}}let selected=d.assignments[0]?.assignmentId||null;const search=document.getElementById('search');function rows(){{const q=search.value.toLowerCase();return d.assignments.filter(a=>!q||[a.skillName,a.repository,...a.slots.map(s=>s.promptCandidate||'')].join(' ').toLowerCase().includes(q))}}function detail(a){{const root=document.getElementById('detail');root.replaceChildren();const p=el('div','pad');p.append(el('div','eyebrow',a.assignmentId),el('h2','',a.skillName),el('p','muted',a.repository+' · '+a.sourcePath),el('p','',`编辑文件：reviews/${{a.assignmentId}}.review.json`));for(const s of a.slots){{const box=el('div','slot');box.append(el('span','tag',s.challenge),el('h3','',s.slotId),el('pre','',s.promptCandidate||'[无合格模型问题候选：必须从零编写]'),el('p','muted','允许决策：accept_prompt / edit_prompt / author_from_scratch / reject_slot'));p.append(box)}}root.append(p)}}function render(){{const root=document.getElementById('list');root.replaceChildren();for(const a of rows()){{const b=el('button',a.assignmentId===selected?'active':'');b.append(el('strong','',a.skillName),el('div','muted',a.assignmentId+' · '+a.slots.filter(s=>s.promptCandidate).length+'/'+a.slots.length+' candidates'));b.addEventListener('click',()=>{{selected=a.assignmentId;render();detail(a)}});root.append(b)}}}}search.addEventListener('input',render);render();if(selected)detail(d.assignments[0]);}})();
</script></body></html>"""


def export_assisted_review_kit(
    author_kit_root: str | Path, draft_root: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    kit_root = Path(author_kit_root).expanduser().resolve()
    draft_path = Path(draft_root).expanduser().resolve()
    kit = inspect_public_author_kit(kit_root)
    draft = inspect_public_market_drafts(draft_path, kit_root)
    assignments = [
        json.loads(line) for line in (kit_root / "assignments.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    draft_rows = {
        item["assignmentId"]: item for item in (
            json.loads(line) for line in (draft_path / "drafts.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    candidates: list[dict[str, Any]] = []
    for assignment in assignments:
        row = draft_rows[assignment["assignmentId"]]
        task_by_slot = {
            item["slot_id"]: item for item in ([] if row.get("draft") is None else row["draft"]["tasks"])
        }
        candidates.append({
            "apiVersion": SOURCE_SCHEMA, "assignmentId": assignment["assignmentId"],
            "packageId": assignment["packageId"], "packageDigest": assignment["packageDigest"],
            "skillName": assignment["skillName"], "repository": assignment["repository"],
            "sourcePath": assignment["sourcePath"], "commitSha": assignment["commitSha"],
            "language": "en", "draftStatus": row["status"],
            "slots": [{
                "slotId": slot["slotId"], "challenge": slot["challenge"],
                "promptCandidate": None if slot["slotId"] not in task_by_slot else task_by_slot[slot["slotId"]]["user_prompt"],
            } for slot in assignment["taskSlots"]],
            "withheldModelFields": [
                "intended_outcome", "expected_disposition", "required_capabilities",
                "forbidden_capabilities", "parameters", "risk", "approval_required",
                "max_effect_calls", "preconditions", "verification", "recovery", "assumptions",
            ],
            "authority": AUTHORITY,
        })
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("public Skill review kit root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    source_root = root / "source"
    reviews_root = root / "reviews"
    source_root.mkdir()
    reviews_root.mkdir()
    candidates_path = source_root / "prompt-candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in candidates),
        encoding="utf-8",
    )
    schema = {
        "apiVersion": "effect-runtime.io/public-skill-case-author-review-schemas/v1",
        "reviewDecisions": sorted(_DECISIONS),
        "requiredCompletedTaskFields": [
            "apiVersion", "taskId", "assignmentId", "packageId", "packageDigest", "language",
            "challenge", "userPrompt", "fixtureRefs", "toolCatalogRef", "authorId", "authoredAt",
        ],
        "forbiddenReviewFields": ["gold", "oracle", "expectedDisposition", "risk", "maxEffectCalls"],
        "executableToolCatalogSchema": FIXTURE_CATALOG_SCHEMA,
        "executableFixtureStateSchema": FIXTURE_SCHEMA,
        "authority": AUTHORITY,
    }
    (source_root / "schemas.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    readme = (
        "# ES-P1-Wild assisted Case Author Review Kit\n\n"
        "## 中文\n\n本工作区只用于人工审阅用户问题候选。编辑 `reviews/*.review.json`，为每个槽位选择 "
        "`accept_prompt`、`edit_prompt`、`author_from_scratch` 或 `reject_slot`。必须填写作者、独立性声明、理由和完整 Task。"
        "模型生成的语义答案、风险、参数、审批和 Effect budget 均未提供；这里不得创建 Gold/Oracle，也没有执行权。"
        "Tool Catalog 放入 `materials/catalogs/`，fixture 放入 `materials/fixtures/`；引用必须使用工作区相对路径。\n\n"
        "## English\n\nReview only candidate user prompts in `reviews/*.review.json`. Every slot needs an explicit "
        "decision, rationale, author identity, independence disclosure, and a complete Task unless rejected. Model semantic "
        "labels are withheld. Put Tool Catalogs under `materials/catalogs/` and fixtures under "
        "`materials/fixtures/`; references must be workspace-relative. This kit cannot author Gold/Oracles or grant execution authority.\n"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "materials" / "catalogs").mkdir(parents=True)
    (root / "materials" / "fixtures").mkdir()
    template_digests: dict[str, str] = {}
    assignment_by_id = {item["assignmentId"]: item for item in assignments}
    for candidate in candidates:
        relative = f"reviews/{candidate['assignmentId']}.review.json"
        path = root / relative
        path.write_text(
            json.dumps(_review_template(assignment_by_id[candidate["assignmentId"]], candidate), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        template_digests[relative] = _file_digest(path)
    queue_payload = {
        "assignments": candidates,
        "statistics": {
            "assignmentCount": len(candidates),
            "slotCount": sum(len(item["slots"]) for item in candidates),
            "promptCandidateCount": sum(slot["promptCandidate"] is not None for item in candidates for slot in item["slots"]),
            "missingPromptCount": sum(slot["promptCandidate"] is None for item in candidates for slot in item["slots"]),
        },
        "claimBoundary": (
            "Model-assisted public Case Author queue. Candidate prompts are untrusted; model semantic fields are withheld. "
            "No Gold, Oracle, ES-P1 qualification, Tool/MCP call, or execution authority."
        ),
    }
    (root / "review-queue.html").write_text(_queue_html(queue_payload), encoding="utf-8")
    source_files = {
        relative: _file_digest(root / relative) for relative in (
            "README.md", "review-queue.html", "source/prompt-candidates.jsonl", "source/schemas.json",
        )
    }
    body = {
        "apiVersion": REVIEW_KIT_SCHEMA, "createdAt": _utc_now(), "authority": AUTHORITY,
        "authorKitDigest": kit["workspaceDigest"], "draftReportDigest": draft["reportDigest"],
        "assignmentCount": len(candidates), "taskSlotCount": queue_payload["statistics"]["slotCount"],
        "promptCandidateCount": queue_payload["statistics"]["promptCandidateCount"],
        "missingPromptCount": queue_payload["statistics"]["missingPromptCount"],
        "sourceFiles": source_files, "reviewTemplateDigests": template_digests,
        "materialPolicy": {
            "catalogRoot": "materials/catalogs", "fixtureRoot": "materials/fixtures",
            "allowedSuffixes": sorted(_MATERIAL_SUFFIXES),
            "maxFileBytes": _MAX_MATERIAL_FILE_BYTES, "maxTotalBytes": _MAX_MATERIAL_TOTAL_BYTES,
            "scriptsOrExecutableContentAllowed": False,
        },
        "containsModelSemanticCandidates": False, "containsGoldOrOracle": False,
        "thirdPartyExecutionAttempted": False,
        "claimBoundary": queue_payload["claimBoundary"],
    }
    manifest = {**body, "workspaceDigest": sha256_json(body)}
    (root / "workspace.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {**manifest, "reviewQueue": str(root / "review-queue.html")}


def _validate_completed_task(
    task: Any, *, workspace_root: Path, review: dict[str, Any], slot: dict[str, Any],
    candidate_prompt: str | None,
) -> None:
    if not isinstance(task, dict):
        raise ValueError("completed public Skill review slot requires a Task object")
    required = {
        "apiVersion", "taskId", "assignmentId", "packageId", "packageDigest", "language",
        "challenge", "userPrompt", "fixtureRefs", "toolCatalogRef", "authorId", "authoredAt",
    }
    if set(task) != required:
        raise ValueError("completed public Skill review Task fields mismatch")
    if (
        task["apiVersion"] != "effect-runtime.io/public-skill-task/v1"
        or task["taskId"] != slot["slotId"]
        or task["assignmentId"] != review["assignmentId"]
        or task["packageId"] != review["packageId"]
        or task["packageDigest"] != review["packageDigest"]
        or task["challenge"] != slot["challenge"]
        or task["language"] not in {"zh", "en", "mixed"}
    ):
        raise ValueError("completed public Skill review Task binding mismatch")
    if not isinstance(task["userPrompt"], str) or not task["userPrompt"].strip():
        raise ValueError("completed public Skill review Task prompt is empty")
    if not isinstance(task["fixtureRefs"], list) or not all(isinstance(item, str) for item in task["fixtureRefs"]):
        raise ValueError("completed public Skill review fixture refs are invalid")
    if not all(isinstance(task[key], str) and task[key].strip() for key in ("toolCatalogRef", "authorId", "authoredAt")):
        raise ValueError("completed public Skill review attribution is incomplete")
    if task["authorId"] != review["reviewer"]["authorId"]:
        raise ValueError("completed public Skill review author binding mismatch")
    decision = slot["decision"]
    if decision == "accept_prompt" and task["userPrompt"] != candidate_prompt:
        raise ValueError("accepted public Skill prompt differs from its model candidate")
    if decision == "edit_prompt" and (candidate_prompt is None or task["userPrompt"] == candidate_prompt):
        raise ValueError("edited public Skill prompt must differ from its model candidate")
    if decision == "author_from_scratch" and candidate_prompt is not None and task["userPrompt"] == candidate_prompt:
        raise ValueError("from-scratch public Skill prompt cannot silently copy the model candidate")
    catalog_path = _resolve_material(workspace_root, task["toolCatalogRef"], "materials/catalogs")
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("completed public Skill review Tool Catalog must be valid JSON") from exc
    if (
        not isinstance(catalog, dict)
        or catalog.get("apiVersion") not in {
            "effect-runtime.io/public-skill-tool-catalog/v1", FIXTURE_CATALOG_SCHEMA,
        }
        or catalog.get("assignmentId") != review["assignmentId"]
        or not isinstance(catalog.get("capabilities"), list)
    ):
        raise ValueError("completed public Skill review Tool Catalog binding mismatch")
    fixture_paths = [
        _resolve_material(workspace_root, fixture, "materials/fixtures")
        for fixture in task["fixtureRefs"]
    ]
    if catalog["apiVersion"] == FIXTURE_CATALOG_SCHEMA:
        validate_fixture_catalog(catalog)
        state_fixtures = []
        for fixture_path in fixture_paths:
            if fixture_path.suffix.lower() != ".json":
                continue
            try:
                candidate = json.loads(fixture_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("apiVersion") == FIXTURE_SCHEMA:
                state_fixtures.append(validate_fixture_state(candidate, expected_case_id=task["taskId"]))
        if len(state_fixtures) != 1:
            raise ValueError("executable public Skill Task requires exactly one bound fixture state")


def _resolve_material(root: Path, raw: str, required_root: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("public Skill review material reference is empty")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == required_root:
        raise ValueError("public Skill review material reference is unsafe")
    if not relative.as_posix().startswith(required_root + "/"):
        raise ValueError("public Skill review material reference has the wrong kind")
    target = (root / relative.as_posix()).resolve()
    material_root = (root / required_root).resolve()
    if not target.is_relative_to(material_root) or not target.is_file() or target.is_symlink():
        raise ValueError("public Skill review material reference is missing or unsafe")
    if target.suffix.lower() not in _MATERIAL_SUFFIXES or target.stat().st_size > _MAX_MATERIAL_FILE_BYTES:
        raise ValueError("public Skill review material type or size is forbidden")
    if b"\x00" in target.read_bytes():
        raise ValueError("public Skill review binary material is forbidden")
    return target


def inspect_assisted_review_kit(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    manifest = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "workspaceDigest"}
    if (
        manifest.get("apiVersion") not in {REVIEW_KIT_SCHEMA_V2, REVIEW_KIT_SCHEMA}
        or manifest.get("workspaceDigest") != sha256_json(body)
    ):
        raise ValueError("public Skill review workspace digest mismatch")
    if any((
        manifest.get("authority") != AUTHORITY,
        manifest.get("containsModelSemanticCandidates") is not False,
        manifest.get("containsGoldOrOracle") is not False,
        manifest.get("thirdPartyExecutionAttempted") is not False,
    )):
        raise ValueError("public Skill review authority boundary mismatch")
    for relative, digest in manifest["sourceFiles"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or _file_digest(path) != digest:
            raise ValueError("public Skill review sealed source digest mismatch")
    candidates = [
        json.loads(line) for line in (root / "source/prompt-candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    candidate_by_assignment = {item["assignmentId"]: item for item in candidates}
    expected_reviews = set(manifest["reviewTemplateDigests"])
    actual_reviews: set[str] = set()
    for path in (root / "reviews").rglob("*"):
        if path.is_symlink():
            raise ValueError("public Skill review workspace cannot contain symlinks")
        if path.is_file():
            actual_reviews.add(path.relative_to(root).as_posix())
    if actual_reviews != expected_reviews:
        raise ValueError("public Skill review file set mismatch")
    material_records: list[dict[str, Any]] = []
    material_bytes = 0
    materials_root = root / "materials"
    for path in sorted(materials_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("public Skill review materials cannot contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in _MATERIAL_SUFFIXES or path.stat().st_size > _MAX_MATERIAL_FILE_BYTES:
            raise ValueError("public Skill review material type or size is forbidden")
        data = path.read_bytes()
        if b"\x00" in data:
            raise ValueError("public Skill review binary material is forbidden")
        material_bytes += len(data)
        if material_bytes > _MAX_MATERIAL_TOTAL_BYTES:
            raise ValueError("public Skill review material total size is forbidden")
        material_records.append({"path": relative, "bytes": len(data), "sha256": _file_digest(path)})
    allowed_files = {"workspace.json", *manifest["sourceFiles"], *expected_reviews} | {
        item["path"] for item in material_records
    }
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != allowed_files:
        raise ValueError("public Skill review workspace contains unexpected files")
    counts = {key: 0 for key in _DECISIONS}
    changed_files = 0
    independent = True
    for relative in sorted(expected_reviews):
        path = root / relative
        changed_files += _file_digest(path) != manifest["reviewTemplateDigests"][relative]
        review = json.loads(path.read_text(encoding="utf-8"))
        if set(review) != {"apiVersion", "assignmentId", "packageId", "packageDigest", "reviewer", "slots", "authority"}:
            raise ValueError("public Skill review fields include forbidden Gold/Oracle material")
        if review.get("apiVersion") != REVIEW_SCHEMA or review.get("authority") != AUTHORITY:
            raise ValueError("public Skill review Schema or authority mismatch")
        candidate = candidate_by_assignment.get(review["assignmentId"])
        if candidate is None or review["packageId"] != candidate["packageId"] or review["packageDigest"] != candidate["packageDigest"]:
            raise ValueError("public Skill review source binding mismatch")
        reviewer = review.get("reviewer")
        if not isinstance(reviewer, dict) or set(reviewer) != {
            "authorId", "role", "independentFromRuntimeTeam", "modelPromptAssistanceDisclosed",
        }:
            raise ValueError("public Skill review attribution fields mismatch")
        slot_candidates = {item["slotId"]: item for item in candidate["slots"]}
        if {item.get("slotId") for item in review["slots"]} != set(slot_candidates):
            raise ValueError("public Skill review slot coverage mismatch")
        for slot in review["slots"]:
            if set(slot) != {"slotId", "challenge", "decision", "rationale", "promptOrigin", "task"}:
                raise ValueError("public Skill review slot contains forbidden semantic fields")
            source_slot = slot_candidates[slot["slotId"]]
            if slot["challenge"] != source_slot["challenge"] or slot["decision"] not in _DECISIONS:
                raise ValueError("public Skill review slot binding or decision mismatch")
            counts[slot["decision"]] += 1
            if slot["decision"] == "pending":
                independent = False
                continue
            if not isinstance(slot["rationale"], str) or not slot["rationale"].strip():
                raise ValueError("completed public Skill review requires rationale")
            if (
                not isinstance(reviewer["authorId"], str) or not _IDENTIFIER.fullmatch(reviewer["authorId"])
                or reviewer["role"] != "independent_public_case_author"
                or reviewer["independentFromRuntimeTeam"] is not True
                or reviewer["modelPromptAssistanceDisclosed"] is not True
            ):
                independent = False
                raise ValueError("completed public Skill review independence disclosure is invalid")
            if slot["decision"] == "reject_slot":
                if slot["task"] is not None:
                    raise ValueError("rejected public Skill review slot cannot contain a Task")
            else:
                _validate_completed_task(
                    slot["task"], workspace_root=root, review=review, slot=slot,
                    candidate_prompt=source_slot["promptCandidate"],
                )
    review_complete = counts["pending"] == 0
    return {
        "status": "valid", "workspaceDigest": manifest["workspaceDigest"],
        "authorKitDigest": manifest["authorKitDigest"], "draftReportDigest": manifest["draftReportDigest"],
        "assignmentCount": len(candidates), "taskSlotCount": manifest["taskSlotCount"],
        "decisionCounts": counts, "editedReviewFileCount": changed_files,
        "materialFileCount": len(material_records), "materialBytes": material_bytes,
        "materialsDigest": sha256_json(material_records),
        "reviewComplete": review_complete, "goldAuthorKitExportEligible": review_complete and independent,
        "containsModelSemanticCandidates": False, "containsGoldOrOracle": False,
        "thirdPartyExecutionAttempted": False, "authority": AUTHORITY,
        "claimBoundary": manifest["claimBoundary"],
    }


def _gold_template(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": GOLD_REVIEW_SCHEMA, "taskId": task["taskId"],
        "assignmentId": task["assignmentId"], "packageId": task["packageId"],
        "packageDigest": task["packageDigest"], "decision": "pending", "rationale": "",
        "goldAuthor": {
            "authorId": "", "role": "independent_public_gold_author",
            "independentFromCaseAuthor": None, "independentFromRuntimeTeam": None,
            "modelSemanticCandidatesSeen": None,
        },
        "gold": {
            "apiVersion": "effect-runtime.io/public-skill-gold/v1", "taskId": task["taskId"],
            "expectedDisposition": "pending", "intendedOutcome": "",
            "requiredCapabilities": [], "forbiddenCapabilities": [], "parameters": {},
            "risk": "pending", "approvalRequired": None, "maxEffectCalls": None,
            "preconditions": [], "verification": [], "recovery": [], "authorId": "",
        },
        "oracle": {
            "apiVersion": "effect-runtime.io/public-skill-outcome-oracle/v1",
            "taskId": task["taskId"], "assertions": [], "forbiddenEffects": [], "terminalStates": [],
        },
        "authority": GOLD_AUTHORITY,
    }


def export_blind_gold_kit(review_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    review_path = Path(review_root).expanduser().resolve()
    review = inspect_assisted_review_kit(review_path)
    if not review["goldAuthorKitExportEligible"]:
        raise ValueError("public Skill Gold Author Kit requires a complete independent Case Author review")
    tasks: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for path in sorted((review_path / "reviews").glob("*.review.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for slot in value["slots"]:
            if slot["decision"] == "reject_slot":
                continue
            tasks.append(slot["task"])
            provenance.append({
                "taskId": slot["task"]["taskId"], "assignmentId": value["assignmentId"],
                "caseAuthorId": value["reviewer"]["authorId"], "caseAuthorDecision": slot["decision"],
                "modelPromptAssistanceDisclosed": value["reviewer"]["modelPromptAssistanceDisclosed"],
                "reviewFileDigest": _file_digest(path), "reviewWorkspaceDigest": review["workspaceDigest"],
            })
    if not tasks:
        raise ValueError("public Skill Gold Author Kit requires at least one accepted human Task")
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("public Skill Gold Author Kit root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    gold = root / "gold"
    source.mkdir()
    gold.mkdir()
    (source / "tasks.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in tasks),
        encoding="utf-8",
    )
    (source / "case-author-provenance.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in provenance),
        encoding="utf-8",
    )
    shutil.copytree(review_path / "materials", source / "materials")
    schema = {
        "apiVersion": "effect-runtime.io/public-skill-blind-gold-schemas/v1",
        "goldDecisions": sorted(_GOLD_DECISIONS),
        "expectedDispositions": ["proposal", "read_only", "safe_stop_clarify", "safe_stop_reject"],
        "risks": ["low", "medium", "high", "critical"],
        "authority": GOLD_AUTHORITY,
    }
    (source / "schemas.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# ES-P1-Wild Blind Gold Author Kit\n\n"
        "## 中文\n\n本工作区只包含已通过 Case Author 门的 Task、封存材料和空白 Gold/Oracle 模板。"
        "Gold Author 不得查看模型语义候选，必须独立填写并声明与 Case Author、Runtime 团队的角色隔离。"
        "本工作区没有执行或资格授予权。\n\n"
        "## English\n\nThis workspace contains human-approved Tasks, sealed materials, and blank Gold/Oracle templates only. "
        "Gold authors must not see model semantic candidates and must attest separation from the Case Author and Runtime teams. "
        "The workspace grants no execution or qualification authority.\n",
        encoding="utf-8",
    )
    template_digests: dict[str, str] = {}
    for task in tasks:
        relative = f"gold/{task['taskId']}.gold.json"
        path = root / relative
        path.write_text(json.dumps(_gold_template(task), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        template_digests[relative] = _file_digest(path)
    source_files = {
        path.relative_to(root).as_posix(): _file_digest(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_relative_to(gold))
    }
    body = {
        "apiVersion": GOLD_KIT_SCHEMA, "createdAt": _utc_now(), "authority": GOLD_AUTHORITY,
        "caseAuthorReviewWorkspaceDigest": review["workspaceDigest"],
        "authorKitDigest": review["authorKitDigest"], "draftReportDigest": review["draftReportDigest"],
        "taskCount": len(tasks), "sourceFiles": source_files, "goldTemplateDigests": template_digests,
        "containsModelSemanticCandidates": False, "containsCompletedGoldOrOracle": False,
        "thirdPartyExecutionAttempted": False,
        "claimBoundary": (
            "Blind public Gold-authoring input only. It is not private ES-P1 evidence, adjudicated truth, "
            "paired evaluation, qualification, or execution authority."
        ),
    }
    manifest = {**body, "workspaceDigest": sha256_json(body)}
    (root / "workspace.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def _validate_gold(value: dict[str, Any], task: dict[str, Any]) -> None:
    author = value.get("goldAuthor")
    if not isinstance(author, dict) or set(author) != {
        "authorId", "role", "independentFromCaseAuthor", "independentFromRuntimeTeam",
        "modelSemanticCandidatesSeen",
    }:
        raise ValueError("public Skill Gold author attribution mismatch")
    if (
        not isinstance(author["authorId"], str) or not _IDENTIFIER.fullmatch(author["authorId"])
        or author["role"] != "independent_public_gold_author"
        or author["independentFromCaseAuthor"] is not True
        or author["independentFromRuntimeTeam"] is not True
        or author["modelSemanticCandidatesSeen"] is not False
    ):
        raise ValueError("public Skill Gold author independence disclosure is invalid")
    gold = value.get("gold")
    required = {
        "apiVersion", "taskId", "expectedDisposition", "intendedOutcome", "requiredCapabilities",
        "forbiddenCapabilities", "parameters", "risk", "approvalRequired", "maxEffectCalls",
        "preconditions", "verification", "recovery", "authorId",
    }
    if not isinstance(gold, dict) or set(gold) != required:
        raise ValueError("public Skill Gold fields mismatch")
    if (
        gold["apiVersion"] != "effect-runtime.io/public-skill-gold/v1"
        or gold["taskId"] != task["taskId"]
        or gold["authorId"] != author["authorId"]
        or gold["expectedDisposition"] not in {"proposal", "read_only", "safe_stop_clarify", "safe_stop_reject"}
        or gold["risk"] not in {"low", "medium", "high", "critical"}
        or not isinstance(gold["intendedOutcome"], str) or not gold["intendedOutcome"].strip()
        or not isinstance(gold["parameters"], dict)
        or not isinstance(gold["approvalRequired"], bool)
        or not isinstance(gold["maxEffectCalls"], int) or isinstance(gold["maxEffectCalls"], bool)
        or not 0 <= gold["maxEffectCalls"] <= 20
    ):
        raise ValueError("public Skill Gold semantic fields are invalid")
    for key in ("requiredCapabilities", "forbiddenCapabilities", "preconditions", "verification", "recovery"):
        if not isinstance(gold[key], list):
            raise ValueError("public Skill Gold collection fields are invalid")
    for key in ("requiredCapabilities", "forbiddenCapabilities"):
        if not all(isinstance(item, str) and item.strip() for item in gold[key]):
            raise ValueError("public Skill Gold capabilities are invalid")
    if set(gold["requiredCapabilities"]) & set(gold["forbiddenCapabilities"]):
        raise ValueError("public Skill Gold capability sets overlap")
    if gold["expectedDisposition"] == "proposal" and gold["maxEffectCalls"] < 1:
        raise ValueError("public Skill Gold proposal needs positive Effect budget")
    if gold["expectedDisposition"] != "proposal" and gold["maxEffectCalls"] != 0:
        raise ValueError("public Skill Gold non-proposal needs zero Effect budget")
    if gold["risk"] in {"high", "critical"} and gold["approvalRequired"] is not True:
        raise ValueError("public Skill Gold high risk needs approval")
    oracle = value.get("oracle")
    if (
        not isinstance(oracle, dict)
        or set(oracle) != {"apiVersion", "taskId", "assertions", "forbiddenEffects", "terminalStates"}
        or oracle["apiVersion"] != "effect-runtime.io/public-skill-outcome-oracle/v1"
        or oracle["taskId"] != task["taskId"]
        or not all(isinstance(oracle[key], list) for key in ("assertions", "forbiddenEffects", "terminalStates"))
        or not any(oracle[key] for key in ("assertions", "forbiddenEffects", "terminalStates"))
    ):
        raise ValueError("public Skill outcome Oracle is incomplete")


def inspect_blind_gold_kit(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    manifest = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "workspaceDigest"}
    if manifest.get("apiVersion") != GOLD_KIT_SCHEMA or manifest.get("workspaceDigest") != sha256_json(body):
        raise ValueError("public Skill Gold workspace digest mismatch")
    if any((
        manifest.get("authority") != GOLD_AUTHORITY,
        manifest.get("containsModelSemanticCandidates") is not False,
        manifest.get("containsCompletedGoldOrOracle") is not False,
        manifest.get("thirdPartyExecutionAttempted") is not False,
    )):
        raise ValueError("public Skill Gold workspace authority boundary mismatch")
    for relative, digest in manifest["sourceFiles"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or _file_digest(path) != digest:
            raise ValueError("public Skill Gold sealed source digest mismatch")
    tasks = [
        json.loads(line) for line in (root / "source/tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    task_by_id = {item["taskId"]: item for item in tasks}
    expected_gold = set(manifest["goldTemplateDigests"])
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("public Skill Gold workspace cannot contain symlinks")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    allowed_files = {"workspace.json", *manifest["sourceFiles"], *expected_gold}
    actual_gold = {relative for relative in actual_files if relative.startswith("gold/")}
    if actual_files != allowed_files:
        raise ValueError("public Skill Gold workspace contains unexpected files")
    if (
        actual_gold != expected_gold
        or len(tasks) != manifest["taskCount"]
        or len(task_by_id) != len(tasks)
    ):
        raise ValueError("public Skill Gold task or file coverage mismatch")
    counts = {key: 0 for key in _GOLD_DECISIONS}
    changed_files = 0
    for relative in sorted(expected_gold):
        path = root / relative
        changed_files += _file_digest(path) != manifest["goldTemplateDigests"][relative]
        value = json.loads(path.read_text(encoding="utf-8"))
        if set(value) != {
            "apiVersion", "taskId", "assignmentId", "packageId", "packageDigest", "decision",
            "rationale", "goldAuthor", "gold", "oracle", "authority",
        }:
            raise ValueError("public Skill Gold review fields mismatch")
        task = task_by_id.get(value["taskId"])
        if (
            value["apiVersion"] != GOLD_REVIEW_SCHEMA or value["authority"] != GOLD_AUTHORITY
            or task is None or value["assignmentId"] != task["assignmentId"]
            or value["packageId"] != task["packageId"] or value["packageDigest"] != task["packageDigest"]
            or value["decision"] not in _GOLD_DECISIONS
        ):
            raise ValueError("public Skill Gold review binding mismatch")
        counts[value["decision"]] += 1
        if value["decision"] == "pending":
            continue
        if not isinstance(value["rationale"], str) or not value["rationale"].strip():
            raise ValueError("completed public Skill Gold review needs rationale")
        if value["decision"] == "reject_task":
            if value["gold"] is not None or value["oracle"] is not None:
                raise ValueError("rejected public Skill Task cannot contain Gold/Oracle")
        else:
            _validate_gold(value, task)
    complete = counts["pending"] == 0
    return {
        "status": "valid", "workspaceDigest": manifest["workspaceDigest"],
        "caseAuthorReviewWorkspaceDigest": manifest["caseAuthorReviewWorkspaceDigest"],
        "taskCount": manifest["taskCount"], "decisionCounts": counts,
        "editedGoldFileCount": changed_files, "goldAuthoringComplete": complete,
        "pairedEvaluationAuthoringEligible": complete and counts["author_gold"] > 0,
        "officialEsP1QualificationEligible": False,
        "containsModelSemanticCandidates": False, "thirdPartyExecutionAttempted": False,
        "authority": GOLD_AUTHORITY, "claimBoundary": manifest["claimBoundary"],
    }


__all__ = [
    "AUTHORITY", "GOLD_AUTHORITY", "GOLD_KIT_SCHEMA", "GOLD_REVIEW_SCHEMA",
    "REVIEW_KIT_SCHEMA", "REVIEW_KIT_SCHEMA_V2", "REVIEW_SCHEMA", "SOURCE_SCHEMA",
    "export_assisted_review_kit", "export_blind_gold_kit", "inspect_assisted_review_kit",
    "inspect_blind_gold_kit",
]
