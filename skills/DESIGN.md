# skills — 设计与实现说明书

> **可复用任务模板**。skill = "处理某类问题的步骤+工具组合",由人工预定义(`builtin/`)+ 静态加载(`mock/`, `pragmatic/`)+ LLM 演化(`evolver` 生成)。
> **铁律**:skill 是 *建议*,不是 *约束*。runtime loop 把匹配的 skill 注入 system prompt 作 hint,LLM 可遵循可偏离。skill 评估通过 journal 累积使用统计,再喂回 evolver 调优。

---

## 1. 职责

| 文件 | 职责 |
|------|------|
| `catalog.py` (537) | `SkillCatalogService` — 注册、加载、匹配、格式化 skill |
| `loader.py` (58) | `SkillLoader` — 按 mode (mock/pragmatic/builtin) 选 registry |
| `evolver.py` (1005) | `SkillEvolver` — 任务完成后判断"这是新模式吗",创建/合并/扩展 skill |
| `journal_consumer.py` (212) | 后台 task,定期消费 `runtime/skill_journal` 数据,反馈给 evolver(成功率 / 召回 / dormant 检测) |
| `builtin/registry.py` | 始终可用的 skill(无关 mode) |
| `mock/registry.py` | mock 模式 skill |
| `pragmatic/registry.py` | pragmatic 模式 skill |

---

## 2. 公开接口

### 2.1 主数据类型

```python
from skills import Skill, SkillSummary, SkillDetail, SkillCatalogService

# Skill: id, name, description, steps[], tools_required[], tags[], confidence, ...
# SkillSummary: 投影到 short info(用于 retrieval display)
# SkillDetail: 完整结构(load 后注入 prompt)
```

### 2.2 Catalog 使用

```python
catalog = SkillCatalogService(
    loader=SkillLoader(mode="mock"),
    retriever=skill_retriever,        # build_skill_retriever_async result
)
catalog.start()                       # load all builtin + mode-specific

# 匹配
results = catalog.match(user_query="restart prometheus", top_k=3)
# → list[SkillSelectionResult] with (skill_id, score, reason)

# 加载详细内容(SKILL_LOAD 指令触发)
detail = catalog.load("restart_service_skill")
# 把 detail 拼到下一 turn system prompt
```

### 2.3 Evolver 使用

```python
from skills.evolver import SkillEvolver

evolver = SkillEvolver(
    catalog=catalog,
    llm_fn=async_llm,
    skills_dir="data/skills/",
)

# 任务完成后:
proposal = await evolver.after_task(
    task_description="restart prometheus on ap-01 and ap-02",
    solution_summary="batch ap-01: ok | ap-02: ok",
    tools_used=["restart_service"],
    solution_steps=["restart_service on ap-01", "restart_service on ap-02"],
    key_observations=["batch_size=2"],
    complexity=6.0,
    session_id="batch_abc12345",
)
# proposal: SkillCreationProposal | None
#   if None: 没新意 / 重复已有 / 不够通用
#   if proposal.should_create: catalog 已收新 skill,可立即 retrievable
```

### 2.4 Journal consumer 启动

```python
from skills.journal_consumer import SkillJournalConsumer

consumer = SkillJournalConsumer(
    journal=skill_journal,
    evolver=evolver,
    interval_s=300,           # 每 5 分钟跑一次
    min_uses=3,               # skill 使用次数 ≥ 3 才考虑
    dormant_threshold=0.60,   # 60% 召回但 0 use → dormant
)
await consumer.start()
```

---

## 3. 核心数据流

### 3.1 静态加载 + 注册

```
main.py startup:
   SkillLoader(mode="mock").skill_definitions()
                  │
                  ▼ load builtin + mock dict
                  │
                  ▼ SkillCatalogService.register(skill_def)
                  │   ↓ Skill 对象建立,id 索引
                  │   ↓ 加入 retrieval corpus
                  │
                  ▼ build_skill_retriever_async(catalog.list_summaries())
                  │   ↓ embedding async pre-index
                  │
                  ▼ catalog.attach_retriever(retriever)
```

### 3.2 召回 → 注入 prompt

```
runtime.loop._call_llm pre-pass:
   query = user_query
                  │
                  ▼ catalog.match(query, top_k=3) → 3 个候选
                  │
                  ▼ confidence 过滤(score < 0.4 丢)
                  │
                  ▼ ambiguity check:
                  │   top-1 score - top-2 score < 0.15 → ambiguous
                  │       → raise HitlInterruptRaised(kind=USER_CHOICE)
                  │
                  ▼ inject system prompt:
                  │
                  ▼ "Skills matched:
                     - restart_service_skill (0.92): ..."
                  │
                  ▼ LLM 看到后:
                  │   - 用 [SKILL_LOAD:restart_service_skill] 加载详细
                  │   - 或直接照着 summary 调 tools
                  │   - 或忽略(自由发挥)
```

### 3.3 演化(after_task)

```
任务完成 → evolver.after_task(...)
                  │
                  ▼ Step 1: 复杂度过滤
                  │   if complexity < min_complex (default 5.0): return None
                  │
                  ▼ Step 2: _evaluate_creation_eligibility (LLM)
                  │   prompt:"This task: <T>. Solution: <S>. Tools: <U>.
                  │            Worth creating a reusable skill? JSON {should_create, reuse_potential, ...}"
                  │   parse JSON → SkillCreationProposal
                  │
                  ▼ if not proposal.should_create OR reuse_potential < 0.4: return None
                  │
                  ▼ Step 3: _find_similar_skill (BM25 over catalog)
                  │   jaccard 相似度 > 阈值 → 不是新 skill 是已有的扩展
                  │
                  ▼ Step 4 分支:
                  │
                  ├─[similar found]─ _merge_into_existing_skill (LLM)
                  │       │
                  │       ▼ LLM 综合新旧 steps,生成 v(N+1)
                  │       ▼ catalog.update(existing_id, merged)
                  │       ▼ SkillVersion(reason=MERGED) 记录
                  │
                  └─[no similar]─ _create_new_skill (LLM)
                          │
                          ▼ LLM 生成完整 Skill(steps, tools, tags)
                          ▼ catalog.register(new_skill)
                          ▼ retriever.index(new_skill)  ← 立刻可召回
                          ▼ persist to skills_dir
```

### 3.4 Journal feedback loop

```
Per-turn 写 journal:
   skill_journal.log(skill_id, used=True, success=True/False, query, ...)

每 5 min(JournalConsumer):
   stats_by_skill = journal.aggregate(window=24h)
                  │
                  ▼ for each skill_id:
                  │
                  ├─ uses < min_uses: skip
                  │
                  ├─ success_rate < 0.5: → mark NEEDS_REVISION
                  │       evolver.apply_feedback(FeedbackApplication.NEEDS_REVISION, ...)
                  │       → LLM 重写 steps
                  │
                  ├─ retrieved but rarely loaded (召回 ≥ N, 用 0): → DORMANT
                  │       evolver.apply_feedback(DORMANT)
                  │       → 降低 confidence,逐步衰减出 catalog
                  │
                  └─ stable + high success: → CANONICALIZE
                          confidence += 0.05
                          tags 加 "verified"
```

---

## 4. 关键设计决策

### 4.1 Skill 是 hint 不是程序

很多 skill 系统(LangChain agents)把 skill 当 "可执行函数"。但 LLM 已经能调 tool,skill 当函数等于二次封装,LLM 看不见内部反而约束行为。

我们的 skill 是 **prompt fragment**(steps + tools + observations),注入到 system prompt。LLM 自己决定怎么用:
- 严格照做 → 当 SOP
- 改 args → 适配新场景
- 完全忽略 → free style

这跟 OpenAI function calling 哲学相反 —— 我们选这样是因为对**异常场景**(网络问题千变万化)硬约束反而坏事。

### 4.2 Evolver 是 LLM-loop 不是 ML

`SkillEvolver` 没有训练。所有判断都是 prompt:
- 是不是新模式? → LLM 判 + JSON 输出
- 跟已有 skill 像吗? → BM25 jaccard 预筛 + LLM 二次判
- 怎么合并? → LLM 生成新版本

理由:
- skill corpus 小(< 100)— ML 训练不划算
- 反馈稀疏(成功/失败/dormant)— 不够给 RL
- LLM zero-shot 已经判得不错,加 prompt eng 调优快

### 4.3 SkillVersion 链不可丢

每次 `_merge_into_existing_skill` 不覆盖,而是 append 新 `SkillVersion(reason=MERGED, snapshot=...)`。理由:
- 回滚:LLM 生成的 v2 可能比 v1 差,需要还原
- 审计:用户问"这 skill 啥时候开始用 prometheus 的?",查 version 链
- 学习:可以对比相邻版本看哪类改动有效

存储:每个 skill 的 `versions: list[SkillVersion]` 字段,持久化到 `skills_dir/<id>.json`。

### 4.4 Journal consumer 为什么独立 task?

`after_task` 是同步决策(每任务 1 次,~30s LLM)。Journal 是 batch 分析(看 24h 全局趋势,识别 dormant)。两者节奏不同:
- after_task **fast path**:不能阻塞用户响应
- journal_consumer **slow path**:300s interval,可以慢但要全局视角

混在一起会让 fast path 难以加 "考虑过去 24h success rate" 这种逻辑(state 大)。独立 consumer 后,fast/slow 路径解耦。

### 4.5 三种 registry(builtin / mock / pragmatic)

不同部署模式 skill 内容不同:
- **builtin**:始终可用(`generic_diagnostic_skill`, `escalation_skill`)
- **mock**:开发模式(`mock_netflow_skill` 用假数据)
- **pragmatic**:生产模式(`real_bgp_diag_skill` 调真 OpenAPI)

`SkillLoader(mode=...)` 一次性返回该 mode 该用的全 set。没有运行时切换 mode 的需求,启动决定。

---

## 5. 跨模块依赖

```
skills
   │
   ├── retrieval        (catalog 用 retriever 做 match)
   ├── runtime          (skill_journal 在 runtime 写,这里读)
   └── (LLM 通过注入的 llm_fn 调用,不直接 import integrations)

外部依赖 skills 的:
   - integrations/adapters/hitl_executor.py  (set_skill_evolver 注入)
   - runtime/loop.py                         (SKILL_LOAD directive)
   - webui/routes_skills.py                  (HTTP CRUD)
   - main.py                                 (装配)
```

### 5.1 扩展点

| 任务 | 改哪里 |
|------|--------|
| 加新 builtin skill | `builtin/registry.py` 加字典条目,跑 audit_wiring 确认 |
| 改 evolver creation 阈值 | `evolver.py:SkillEvolver.__init__` 的 `_min_complex` / `_min_reuse` |
| 加新 feedback action(EXPIRE / REWARD)| `evolver.py:SkillChangeReason` 加 enum,`apply_feedback` 加 case |
| 改 ambiguity 阈值(top-1 vs top-2 score)| `catalog.py:SkillCatalogService.match` —— 现在硬编码 0.15 |
| Skill 版本回滚 UI | `webui/routes_skills.py` 加 endpoint,catalog 加 `revert_to_version` |
| 跨 session 学习 | `journal_consumer.py` —— 聚合 by user / by domain 而非全局 |

### 5.2 不该在这里加什么

- ❌ Tool 实现 → `tools/`
- ❌ 检索算法 → `retrieval/`
- ❌ HITL ambiguity trigger 本身 → `hitl_core/triggers.py`(只是 catalog 触发 raise)
- ❌ Memory 写入 → `agent_memory/`

---

## 6. 修改指南

### 6.1 改之前必须知道

- **Evolver 是 LLM 重操作**:每次 `after_task` 可能 2-3 次 LLM 调用(eligibility / similarity / merge or create)。改 prompt 时**先在本地小数据跑** —— production 跑前先用 evaluator 评估新 prompt 对 100 个历史 task 的判定一致性。
- **`_evaluate_creation_eligibility` 的 JSON 输出**:LLM 偶尔返回不合法 JSON。当前用 `_safe_parse_json` 容错,但改 prompt 后**断格式**风险加大。
- **`_find_similar_skill` BM25 阈值**:默认 jaccard 0.4。太低会把不同 skill 误合,太高会重复创建。Tune 时看实际 corpus。
- **SkillEvolver `set_skill_evolver` 注入**:在 hitl_executor 是 deferred wiring。**main.py 启动顺序**:catalog → evolver → executor.set_skill_evolver(evolver)。改启动顺序前先看 audit_wiring。

### 6.2 改完必须跑

```bash
./scripts/precheck.sh --audits
python -m unittest tests.test_production_safety -v

# 手测 evolver:
python -c "
from skills.evolver import SkillEvolver
# 造 mock catalog + llm_fn,跑 after_task 看 proposal
"
```

### 6.3 调试套路

| 症状 | 看哪里 |
|------|-------|
| 新 task 没创建 skill | log: `SkillEvolver: skill creation skipped — should_create=... reuse=...`。LLM 判定不够通用 |
| Skill 召回不到 | retriever index 有这个 id 吗?`catalog.attach_retriever` 在 register 前调? |
| Skill 被误合到 wrong existing | `_find_similar_skill` jaccard 阈值降低,或 BM25 corpus 中文/英文 mix 不平衡 |
| Dormant skill 不衰减 | journal_consumer 有跑吗?(grep `SkillJournalConsumer: started` log)|
| `[SKILL_LOAD:xxx]` 解析失败 | `runtime/directive_parser.py` 改了 brace logic? |

### 6.4 测试

- **没有专属测试目录**,evolver 的关键路径(`after_task` 各分支)缺单元测试。
- **建议加**:
  - `test_catalog_match.py` — match + ambiguity 阈值
  - `test_evolver_eligibility.py` — mock llm_fn 返回各种 JSON,验证 proposal 决策
  - `test_journal_consumer.py` — feedback 触发条件

---

## 7. 已知限制 & TODO

- **`_find_similar_skill` 用全文 BM25**,对**结构化相似**(都是 restart 类操作但目标 service 不同)不敏感。需要按 tools_used / steps 模板匹配增强。
- **Cross-user skill leakage**:当前 catalog 是全局 singleton。多租户场景需要 namespace by org_id。
- **Evolver 不感知失败**:`after_task` 只在成功后调。失败任务的 anti-pattern("don't do X when Y")没存。需要 `after_failure` 分支学 negative example。
- **Skill 文件持久化无版本控制**:`skills_dir/<id>.json` 直接覆写。生产建议接 git 子库或 S3 versioned bucket。
- **SkillVersion chain unbounded growth**:不衰减、不 vacuum。长期运行老 skill 的 versions 列表会变大。需要 retention policy(保留最近 10 版 + 关键 milestone)。
