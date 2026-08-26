# 评估 — 三种 Agent 进化能力 + 三个关注点的达成度

> 方法:逐项查代码验证,不凭印象。每条标注证据位置 + 诚实的达成度。

## 总览(达成度)

| 能力 | 实现 | 触发 | 质量 | 综合 |
|------|------|------|------|------|
| **进化1:相似 skill 选择困难 → 人工接入 → 学习转自动** | ✅ 完整 | ✅ 已接 | 🟡 依赖召回 | **~85%** |
| **进化2:无 skill → 跑通后自动生成** | ✅ 有 | 🟡 窄 | 🔴 缺轨迹 | **~55%** |
| **进化3:skill 过时 → 追加诉求 → 生成并 merge 存量** | ✅ 有 | 🔴 很窄 | 🟡 | **~45%** |

| 关注点 | 达成度 |
|--------|--------|
| #1 使用描述的准确性/相似度(选对 skill) | 🟡 ~70% |
| #2 执行内容的准确性/相似度(执行满足请求) | 🔴 ~30% |
| #3 重复轨迹 → 流程化为 skill | 🔴 ~20% |

---

## 进化1:相似 skill 选择困难 → 人工 → 自动(刚做完,~85%)

**已达成**:弱匹配歧义触发 HITL(catalog `ambiguous_kind`)、选择写偏好 fact、
embedding 召回加权、三阶段(学习/推荐/自动)、按 user 区分、错误降置信回退。
代码:`skills/skill_preference.py` + loop/backend 接线,11 测试通过。

**待改进**:
- 召回靠 B1a embedding 相似,**措辞差异大的同类请求召回不到**(你拍板先 B1a,
  升 B1b LLM 意图归一可解)。
- 自动化依赖"第一次有人选" —— 冷启动阶段仍全靠人工。

---

## 进化2:无 skill → 跑通后自动生成(~55%)

**已达成**(`skills/evolver.py`):`after_task` → 资格判定(`_evaluate_creation_eligibility`,
LLM 判 reuse_potential)→ 找相似(没有就创建)→ `_write_skill_content`(LLM 写
markdown)→ 注册 + 版本历史。suggest/auto 双模式(`auto_evolve_apply`)。
**触发确实接了**:backend.py:1094 `if decision.complexity == "complex": after_task(...)`
—— 复杂查询跑完会触发。

**🔴 关键质量缺口(这是 #2 关注点低分的根源)**:backend.py:1096 调用时传的是
```python
solution_steps=[], key_observations=[], complexity=7.0(硬编码)
```
即:**生成 skill 时没有真实的执行轨迹(步骤序列)**,只有 query + 截断的答复
(`full_text[:400]`)+ 工具名列表。LLM 是在"猜"这个任务该怎么分步,而不是
"复现刚才实际跑通的步骤"。这直接导致生成的 skill 执行内容不准(关注点 #2)。

**其他缺口**:
- 触发只在 stream 的 complex 分支 + HITL batch finalizer,**普通非复杂查询、
  纯只读诊断跑通后不生成**。
- 资格判定用 LLM 主观判 reuse_potential,无客观信号(如"这类请求出现过几次")。

---

## 进化3:skill 过时 → 追加诉求 → 生成并 merge(~45%)

**已达成**:merge 机制完整(`_merge_into_existing_skill` → LLM "ADD missing, KEEP
working" prompt → `_persist_merged_version` → 版本号递增 + 历史)。`after_task` 里
先 `_find_similar_skill`(jaccard),命中就 merge delta 而非建新。`apply_feedback`
(operator 反馈改 skill)也在。

**🔴 关键缺口(为什么只有 ~45%)**:
- 你描述的场景是"**加载执行了某 skill,用户后续追加诉求** → 把追加的部分 merge
  进存量"。但现在的 merge 触发是 `after_task` 里"**新任务恰好和某存量 skill 相似**"
  —— **它不知道"刚才正是加载并执行了 skill X,现在用户在追加"这个上下文**。
  缺一个"当前会话用了哪个 skill + 用户在其基础上追加"的信号,把追加诉求**定向**
  merge 进那个被用的 skill。现在是泛泛地按相似度找一个 skill merge。
- 同样吃"无真实轨迹"的亏:merge 进去的 delta 也来自 `solution_steps=[]`。
- "过时判定"没有显式机制(skill 不满足需求 → 该更新)的触发信号 —— 靠 jaccard
  相似度撞上,不是靠"用户对结果不满/追加"驱动。

---

## 三个关注点的横向评估

### #1 使用描述准确性/相似度(选对 skill)— 🟡 ~70%
- ✅ 有:retriever(embedding+BM25)打分、弱/强歧义判定、偏好学习纠偏。
- 🔴 缺:**没有"描述质量"的反向优化** —— 当某 skill 总是 dormant(选中没执行)
  或总在歧义里出现,系统不会回头改它的 description 让它更可分。journal 已经
  能观测到(Selected/Dormant/never-loaded),但**观测到 → 自动改描述**这条闭环没建。

### #2 执行内容准确性/相似度(执行满足请求)— 🔴 ~30%
- ✅ 有:skill detail 注入、journal 记录 tool 序列、HITL 把关高危。
- 🔴 缺(核心):
  1. **生成的 skill 不含真实轨迹**(进化2 的 `solution_steps=[]`)→ 执行内容先天不准。
  2. **没有"执行结果是否满足请求"的判定** —— 跑完没有校验 outcome 对不对,
     所以无法据此改进 skill。
  3. skill 是软提示,执行偏离无强制(我们讨论的 step gate / 状态机 v4 未建)。

### #3 重复轨迹 → 流程化为 skill — 🔴 ~20%
- 🔴 **几乎没有**:全项目无"重复操作轨迹检测/聚合"机制(grep 无 trajectory/
  repeat-detection)。现在是**单次复杂任务**就可能生成 skill(LLM 判 reuse),
  **不是观察"同一操作序列出现 N 次 → 固化成 skill"**。
- 你描述的"基于结果和流程**反推** skill" —— 反推的原料(真实流程轨迹)现在
  就没喂给 evolver(`solution_steps=[]`),更没有"多次轨迹对齐找公共模式"。

---

## 优先改进项(按性价比)

**P0 — 把真实执行轨迹喂给 evolver**(解 #2、#3 的共同根)
现在 `after_task(solution_steps=[])` 是最大浪费:loop 明明有完整的 turn-by-turn
工具调用序列(journal `events` 里就有!),却没传给 evolver。
→ 改:从 journal / loop state 提取真实 `solution_steps`(tool 序列 + 参数 + 结果摘要)
传入 `after_task`。**这一项能同时显著提升进化2/3 的生成质量和 #2/#3 关注点**。
风险低(数据已存在,只是没接)。

**P1 — 重复轨迹检测 → 触发流程化**(解 #3)
用已有的 journal store 做:统计"相似 query + 相似工具序列"的出现频次,
达阈值(如同类轨迹 ≥3 次)才触发 evolver 生成 —— 把"单次就生成"改成
"重复才固化",更符合 #3 的"重复性单元操作流程化",也减少噪音 skill。
中等工作量(复用 journal + 偏好那套相似度)。

**P2 — "执行结果是否满足"判定 → 反哺 skill**(解 #2 的校验缺口 + #1 描述优化)
跑完用 LLM/规则判 outcome 是否满足原请求;不满足 → apply_feedback 改 skill 或
降低其推荐。把 journal 已观测的 dormant/never-loaded 也接进来自动优化描述(#1)。
中等工作量。

**P3 — "当前会话用了 skill X + 追加诉求"定向 merge**(解进化3 的精准触发)
在 loop 记录"本会话加载执行了哪个 skill",用户追加时把 delta 定向 merge 进 X,
而非泛泛按相似度找。需要 loop 维护"active skill"上下文。中等工作量。

---

## 一句话总结

进化1(选择→学习→自动)刚做完,达成度最高(~85%)。进化2/3(生成/merge)
**机制都在、也确实接了触发,但有一个共同的致命短板:生成/merge 时没有喂入真实
执行轨迹**(`solution_steps=[]`),导致"执行内容准确性"(#2)和"重复轨迹流程化"
(#3)两个关注点都低分。最高性价比的改进是 **P0:把 journal 里已有的真实工具序列
喂给 evolver** —— 数据现成,只是没接,一改同时拉高进化2、3 和关注点 #2、#3。
关注点 #3 的"重复检测"(P1)目前几乎空白,是第二优先。
