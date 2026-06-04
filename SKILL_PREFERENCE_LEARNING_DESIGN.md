# 设计方案 — Skill 歧义选择 + 选择记忆 + 渐进自动化

> 状态:**待评审**。回应:多个近似 skill 时应触发 HITL 选择,选择后记 facts,
> 后续相似请求逐步从"问"变"自动推荐/自动选"。这是一个完整闭环,分 A(触发)+
> B(记忆与自动化)两部分,必须一起做。

## 0. 现状核对(哪些已有、哪些缺)

逐行查过代码:

| 部件 | 状态 | 位置 |
|------|------|------|
| skill 歧义 → HITL `user_choice` 选择 | ✅ **已实现** | loop.py:2183(yield choices + `__none__`) |
| 选择结果回传(`selected_choice_id`) | ✅ 已实现 | backend.py:1919 |
| 选择 → 记 fact | ❌ **缺失** | 无 |
| facts 写入(带 type/confidence/metadata) | ✅ 基建在 | adapter.add_fact |
| facts 召回(按 type) | ✅ 基建在 | adapter.search_facts |
| 冲突感知 boost(同 fact 再选→提分) | ✅ 基建在 | FactConflictDetector |
| 后续相似请求用偏好加权选择 | ❌ **缺失** | 无 |
| 渐进"问→自动"过渡 | ❌ **缺失** | 无 |

**结论**:HITL 选择机制本身在,但 (1) 这次没触发(问题 A),(2) "选择→记忆→自动化"
闭环从没建(问题 B)。两个都要解。

---

## 问题 A — 为什么这次没触发 HITL 选择

实测 "诊断用户 alice 访问应用 crm 失败的原因":
```
top_score=0.12  second=0.08  ambiguous=False
```
`ambiguous` 需要两个条件同时满足(catalog.py):
1. top ≥ `ambiguity_floor`(默认 **0.55**)
2. top1−top2 < `ambiguity_gap_threshold`(默认 0.08)

实际 top 只有 0.12,**远够不到 floor 0.55** → `ambiguous=False` → 跳过 HITL。

**根因**:中文 query × 混合语言 skill 描述,embedding(TF-IDF + nomic)相似度整体偏低。
0.12 这种分数:既够不到"高分难分"(0.55)触发选择,又不足以让 LLM 自信加载 →
**掉进盲区:多个 skill 全是低分,谁都不强**。现有 `ambiguous` 只认"高分难分",
不认"全都低分但有多个候选"。

### A 方案:新增"弱匹配多候选"歧义类型

在 catalog 的 `ambiguous` 判定外,加一个**第二类歧义信号**:

```
ambiguous_weak = (
    len(top) >= 2
    and top[0][0] < ambiguity_floor          # 没有强匹配
    and top[0][0] >= weak_floor               # 但也不是完全没匹配(默认 0.08)
    and (top[0][0] - top[1][0]) < weak_gap    # 前几名挤在一起(默认 0.05)
)
```

语义:**"有好几个 skill 都沾边,但没一个明显是对的"** —— 这种情况让 LLM 自己挑
最容易翻车(就是这次的情形),正该问用户。

`SkillSelectionResult` 增加 `ambiguous_kind: "strong" | "weak" | None`。loop 的触发
条件从 `if skill_ambiguous` 改为 `if ambiguous_kind is not None`。两类歧义复用同一套
`user_choice` HITL UI,只是 HITL 提示文案区分("找到多个高度匹配" vs "找到几个可能
相关")。

阈值全部进 `config.yaml skill_orchestration`,可调可关:
```yaml
skill_orchestration:
  ambiguity_floor: 0.55          # 已有
  ambiguity_gap_threshold: 0.08  # 已有
  weak_ambiguity_floor: 0.08     # 新:弱匹配下限
  weak_ambiguity_gap: 0.05       # 新:弱匹配前几名挤一起的阈值
  weak_ambiguity_min_candidates: 2
```

---

## 问题 B — 选择记忆 + 渐进自动化(核心闭环)

### B1. 选择 → 写偏好 fact

用户在 HITL 选了 skill X(或 `__none__`)后,把这次选择记成一条**专用类型的 fact**。
复用 `add_fact(fact_type="skill_preference", ...)`:

```python
await memory.add_fact(
    session_id=sid, user_id=uid,
    fact_text=f"对于「{query_intent}」类请求,用户选择使用 skill: {chosen_skill_id}",
    fact_type="skill_preference",
    confidence=0.6,                     # 初始置信度(单次选择不高)
    metadata={
        "chosen_skill_id": chosen_skill_id,
        "query_sample": query[:200],     # 原始 query 样本(召回时算相似度)
        "query_intent": query_intent,    # 抽取的意图特征(见下)
        "candidates": [top skills],      # 当时的候选(诊断用)
        "choice_count": 1,
    },
)
```

**关键:`query_intent` 怎么抽?** 不能用整条 query(太具体,"alice/crm" 是变量)。
两个选项:
- B1a(轻,推荐起步):用 query 的 **embedding** 做相似度匹配(召回时比对
  `query_sample`),不显式抽意图。靠 search_facts 的向量召回天然泛化。
- B1b(强,后做):用 LLM 把 query 归一成意图模板("诊断{用户}访问{应用}失败" →
  intent="app_access_failure_diagnosis"),metadata 存模板。召回更准但要 LLM 调用。

起步用 B1a:零额外 LLM,靠 embedding 召回。

**冲突感知 boost 自动复用**:`add_fact` 已接 `FactConflictDetector` —— 用户再次为
相似 query 选同一个 skill,detector 判定 "equivalent → boost existing",**置信度自动
累加**。这正好是"渐进"的底层机制,不用自己写计数器。每次同类选择,这条偏好 fact
的 confidence 往上走。

### B2. 后续相似请求 → 偏好加权选择

skill 选择时(catalog.select_skills_for_query 之后),多一步**偏好召回 + 加权**:

```
1. search_facts(query, fact_type="skill_preference", top_k=3)
   → 找到与当前 query 相似的历史选择偏好
2. 对每条命中的偏好 fact:
   - 取 metadata.chosen_skill_id + 该 fact 的 confidence + query 相似度
   - preference_boost = base_boost * fact_confidence * query_similarity
   - 给对应 skill 的选择分加上 preference_boost
3. 加权后重新排序 selected_skills
```

加权后可能发生三种情况(对应渐进自动化的三个阶段):

### B3. 渐进自动化:问 → 推荐 → 自动(置信度驱动)

用偏好 fact 的 **confidence** 划分三个阶段(阈值进 config):

| 阶段 | 偏好 confidence | 行为 |
|------|----------------|------|
| **学习期** | < 0.5 或无偏好 | 照常:歧义→问用户(A 方案触发) |
| **推荐期** | 0.5 ~ auto_threshold(默认 0.85) | 仍问,但**把偏好的 skill 置顶 + 标记"⭐ 上次你选了这个"**,降低用户选择成本 |
| **自动期** | ≥ auto_threshold | **不问,直接自动 SKILL_LOAD** 偏好 skill,journal 记 `auto_selected_by_preference`(可观察、可回溯) |

confidence 随每次"用户确认同一选择"由 conflict detector 自动 boost。所以:
- 第 1 次:问(学习期),用户选 X,fact confidence=0.6。
- 第 2~3 次相似请求:问,但 X 置顶推荐(推荐期),用户再选 X → boost 到 ~0.8。
- 第 4 次起:confidence ≥ 0.85 → **自动选 X,不再打扰**(自动期)。

**安全阀**:
- 自动选后,如果用户对结果不满(可加一个"这次不对"的反馈入口)→ 给该偏好
  fact 降 confidence,退回推荐期。避免错误偏好被锁死。
- 自动期只对 `__none__` 之外的具体 skill 生效;高风险 skill(requires_hitl)
  可配置**永不自动**(`auto_select_exclude_hitl: true`),始终至少走推荐期。
- 偏好 fact 带 `ttl_days`(默认 90 天),长期不用的偏好自然过期,防陈旧。

---

## 数据流全景

```
用户 query
  → catalog.select_skills_for_query → selected + ambiguous_kind(A:含弱匹配)
  → 偏好召回 search_facts(fact_type=skill_preference)  ← B2
  → 偏好加权 + 重排
  → 判定阶段(按命中偏好的 confidence):           ← B3
      自动期  → 直接 SKILL_LOAD 偏好 skill(journal: auto_selected)
      推荐期  → HITL user_choice,偏好 skill 置顶 ⭐
      学习期/歧义 → HITL user_choice(A 触发)
  → 用户选择 X
  → 写/boost 偏好 fact(confidence++)              ← B1
  → 加载 X,正常执行
```

---

## 落地范围

| 文件 | 改动 | 风险 |
|------|------|------|
| `skills/catalog.py` | 加 `ambiguous_kind`(strong/weak);弱匹配判定 | 低 |
| `config.yaml` | weak_ambiguity_* + auto_select 阈值/开关 | 低 |
| `runtime/loop.py` | 触发条件用 ambiguous_kind;偏好召回+加权+阶段判定;自动期直接 load | **中**(改 skill 选择 + HITL 触发) |
| `webui/backend.py` | user_choice 解析后写偏好 fact(B1);推荐期置顶标记 | 中 |
| `skills/skill_preference.py` | **新建**:偏好 fact 的写入/召回/加权封装(L1,复用 add_fact/search_facts) | 低 |
| `runtime/skill_journal.py` | 记 `auto_selected_by_preference` / `preference_boosted` 事件 | 低 |
| `webui/index.html` | JOURNAL 显示偏好命中 + 自动选择(可观察渐进过程) | 低 |
| tests | 新建 `test_skill_preference.py`(写→召回→boost→阶段过渡) + 弱歧义触发测试 | — |

可全局开关 `skill_orchestration.preference_learning_enabled: true`,一键回退现状。

---

## 待你拍板

1. **A 弱匹配阈值**:weak_floor=0.08 / weak_gap=0.05 这个量级合理吗?(你的真实召回分普遍 0.1 上下,需要据实调)
2. **意图抽取**:起步用 B1a(embedding 相似,零 LLM)还是直接上 B1b(LLM 归一意图模板,更准但每次选择多一次 LLM)?
3. **渐进阶段阈值**:学习<0.5 / 推荐 0.5~0.85 / 自动≥0.85,以及"几次能到自动"(由 boost 步长决定),这个节奏可接受吗?
4. **自动期安全**:高风险 skill(requires_hitl)永不自动,认同吗?要不要"自动选错→降置信"的反馈入口?
5. **偏好粒度**:偏好按 (query意图 → skill) 记。要不要再按 user_id 区分(不同运维各自的偏好)?现在 add_fact 已带 user_id,天然支持,确认要不要用。
6. **TTL**:偏好 90 天过期合理吗?

---

## 一句话总结

A:新增"弱匹配多候选"歧义类型,解决"全是低分、谁都不强"掉进盲区导致 HITL 选择
不触发的问题。B:用户选择 → 写专用类型偏好 fact(复用 add_fact 的冲突感知 boost
做置信累加)→ 后续相似 query 召回偏好并加权 skill 选择 → 按偏好置信度分三阶段
(学习期问 / 推荐期问但置顶 / 自动期直接加载),实现"问→推荐→自动"的渐进自动化,
高风险 skill 永不自动 + 错误可降置信回退。全程 journal 可观察,一键开关回退。
