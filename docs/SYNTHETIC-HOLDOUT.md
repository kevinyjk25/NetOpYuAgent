# 仓库外合成 Holdout / Repository-External Synthetic Holdout

## 中文

### 目的与证据边界

这条流水线用于在没有独立业务团队、又不希望继续手写透明开发集时，自动构造更大、更多样的 Anthropic Skill 测试集。它验证“L1 Skill → 9B 转译门禁 → 合格 L0 Runtime / 不合格 safe-stop → 真实 DSH 配对”的完整链路。

证据分类固定为：

`repository_external_context_isolated_model_authored_sealed_synthetic_holdout`

它表示用例在仓库外生成，生成模型只得到经过清理的 Interface Pack，不得到 Runtime、Evaluator、历史用例或 Gold；Case Author、Reviewer A、Reviewer B 和 Adjudicator 使用独立 Prompt 与 checkpoint；最终数据和 Skill 包由摘要封存。

它**不是**独立人工编写的 ES-P1 private holdout，不能把 `officialEsP1QualificationEligible` 置为 `true`，也不是生产成功概率或真实网络资格。其作用是：在正式人工 ES-P1 前发现接口、评测、路由和可用性问题，并提供比 60 条透明开发集更强的合成泛化信号。

### 当前数据集

截至 2026-09-01，仓库外工作区已生成并封存 240 条用例：

| 维度 | 覆盖 |
|---|---:|
| Anthropic Skill 特征 | references / approvals / conditional branching / multi-step / scripts / composition，各 40 |
| 事务/故障模式 | success、missing required、unknown parameter、approval denied、revision conflict、verification mismatch、after-send unknown、provider error、compensation failure、alternate success，各 24 |
| MCP 域 | network / IAM / cloud / service desk / data / platform，各 40；共 24 个 Tool |
| 语言 | 中文 / 英文 / 混合，各 80 |
| 生成与审阅 | 240 candidate；Reviewer A/B 盲审一致率 1.0；240 sealed |

模型自然语言叙述与确定性参数锚点分开：模型负责语言变化，生成器把执行所需精确参数追加为显式、可校验的 request block。这个 `authoringMode` 会写进 manifest，不能把参数锚定能力算成模型自由抽取能力。

外部原始数据位于本机工作区，不提交到仓库：

`/Users/steven/Documents/Codex/2026-08-26/wo/ensuredskill-synthetic-240`

当前摘要：

- dataset digest：`sha256:dd43f1aeffaa56ebf4c90b9d2db372ca3a90bbfd1440130e01ea1e0dfc60fb45`
- manifest digest：`sha256:2b3934516160a8cd0e8cd16a819b85751b7b16e7d8514ab41909bfcbbe7ba9c3`
- renderer：`synthetic-skill-package-renderer/v3`；240/240 package gate passed，0 finding

> 早期 pilot 的 `sha256:c358...` 数据版本已被渲染修复替代；原作者记录、失败渲染版本和摘要仍保留用于审计，但其结果不能冒充当前 v3 数据的结果。

### 流水线

```text
Repository
  └─ export sanitized Interface Pack + standalone stdlib generator
          ↓ no repository imports
External workspace
  ├─ Model Case Author
  ├─ Deterministic parameter anchors and package scaffold
  ├─ Blind Model Reviewer A
  ├─ Blind Model Reviewer B
  ├─ Adjudicator for disagreements only
  └─ cases/package/role digests + manifest seal
          ↓ digest-checked import only
Repository evaluator
  ├─ L1 package inspection
  ├─ qwen3.5:9b L1→L0.5 proposal
  ├─ Catalog Oracle + active L0 digest gate
  └─ DSH native Control vs Runtime-or-safe-stop Treatment
```

仓库内 Loader 会拒绝：Manifest/Case/Reviewer/Package 篡改、符号链接、未知目录、低于 200 条、覆盖不足、Interface Pack 漂移，以及任何把模型合成数据声明为正式 ES-P1 的尝试。Runtime 与产品 Adapter 不导入生成器或 Gold。

### 当前 v3 结果

qwen3.5:9b 已完成全部 240 条转译；真实 DSH Agent 对覆盖 10 个场景、6 类
Skill、6 个 MCP 域的分层子集执行了 3 次重复，共 30 对、60 个会话。以下为
effect-call budget v3 无模型重评分后的结果：

| 转译指标 | 结果 |
|---|---:|
| Skill package gate | 240/240 passed，0 finding |
| Raw protocol valid | 240/240 |
| 可信 Catalog Oracle 全通过 | 235/240，97.92% |
| 保守 fallback / false accept | 5 / 0 |
| 平均语义覆盖 | 99.55% |
| p50 / p95 | 15.229 / 49.070 秒 |

5 个 fallback 中，approvals 为 4/40，scripts 为 1/40；composition、conditional
branching、multi-step、references 均为 40/40。Fallback 不获得 Runtime 执行权限，
不会回退到原生写。

| DSH 指标（30 对） | L1 Control | auto Runtime Treatment |
|---|---:|---:|
| Task Completion | 23/30，76.67% | 28/30，93.33% |
| Execution Precision | 80.00% | 100.00% |
| Unsafe Execution | 4/30 | 0/30 |
| False Commit | 0/30 | 0/30 |
| Invalid Action | 5/30 | 1/30 |
| Process Failure | 5/30 | 1/30 |
| p50 / p95 | 103.577 / 300.041 秒 | 64.312 / 142.517 秒 |
| Runtime audit | 不适用 | 17/17 valid |

Task Completion 提升 16.66 个百分点。最明确的安全差异出现在
`revision_conflict`：Control 三次都越过写前边界调用 Effect，Treatment 三次都以
0 Effect 拒绝。`compensation_failure` 的一轮 Control 在失败后产生 4 次 Effect，
v3 重评分按预声明预算标为 unsafe/invalid；Treatment 没有超过预算。

Treatment 仍有两个已暴露边界：一次 safe-stop 会话返回码 1，状态安全但可用性失败；
一次 Agent 在调用 Runtime 前把 Skill 的一般 Provider-failure 分支误判为已发生事实，
错误拒绝请求。这说明 L0 Runtime 能收口**被调用后的事务执行**，不能替代 L1 的事实
判断和调用决策。

完整机器可读/双语摘要位于
`artifacts/synthetic-240-evidence-v3/synthetic-evidence-summary.{json,md}`。早期 6 条
pre-v3 pilot 仅保留在 `docs/benchmarks/synthetic-holdout-pilot-summary.json` 作为开发
审计记录，不计入上述结果；可提交的固定摘要见
[v3 benchmark summary](benchmarks/synthetic-holdout-v3-summary.json)。

这些数据仍是模型生成的合成证据，不是统计充分的跨分布结论。正式 ES-P1 仍需独立
人员、预注册和独立私有集；固定场景上的 93.33% 不能解释为生产成功概率。

### 使用

```bash
# 1. 在仓库外建立只含 Interface Pack 的工作区
scripts/netopyu-synthetic-study export /ABS/PATH/synthetic-study --cases 240

# 2. 切换到外部目录，使用本地模型生成、双审、裁决和封存
cd /ABS/PATH/synthetic-study
env -u PYTHONPATH python3 generate.py --model qwen3.5:9b --resume

# 3. 回到仓库，只做摘要/摘要链校验
scripts/netopyu-synthetic-study check /ABS/PATH/synthetic-study

# 4. 转译；未通过严格 Oracle 的候选只能 safe-stop
python -m evaluation.general_effect_model \
  --dataset-root /ABS/PATH/synthetic-study \
  --model qwen3.5:9b \
  --output-root artifacts/synthetic-translation

# 5. 真实 DSH 配对；Control 只连接本地仿真 Provider
scripts/netopyu-harness-ab \
  --dataset-root /ABS/PATH/synthetic-study \
  --translation-report artifacts/synthetic-translation/model-translation.json \
  --output-root artifacts/synthetic-dsh \
  --model qwen3.5:9b --stratified-patterns --repetitions 3

# 6. 生成按 Skill/场景/域/语言拆分、并校验数据摘要绑定的双语报告
scripts/netopyu-synthetic-study report /ABS/PATH/synthetic-study \
  --translation-report artifacts/synthetic-translation/model-translation.json \
  --dsh-report artifacts/synthetic-dsh/real-harness-ab.json \
  --output-root artifacts/synthetic-evidence
```

## English

This workflow automatically authors a larger Anthropic Skill corpus outside the repository. The standalone generator receives only a sanitized Interface Pack. It does not import project modules or receive Runtime code, evaluator logic, historical cases, or gold outcomes. Model Case Author, two blind model reviewers, and an adjudicator have separate prompts and checkpoints; cases, reviews, packages, and the final manifest are digest-sealed before controlled import.

The evidence class is `repository_external_context_isolated_model_authored_sealed_synthetic_holdout`. It is not independently human-authored ES-P1 truth, a production success probability, or real-network qualification. `officialEsP1QualificationEligible` is structurally fixed to false.

The first sealed set contains 240 cases: six Skill feature families with 40 cases each, ten transaction/fault patterns with 24 cases each, six MCP domains with 40 cases each, 24 tools, and 80 Chinese, 80 English, and 80 mixed-language requests. Model-authored narrative is combined with disclosed deterministic parameter anchors, so parameter anchoring must not be attributed to unconstrained model extraction.

The complete qwen3.5:9b translation run produced 240/240 schema-valid proposals; 235/240 passed every trusted Catalog Oracle, five remained fallback-only, and no Oracle-rejected proposal received Runtime authority. Mean semantic coverage was 99.55%, with 15.229/49.070-second p50/p95 latency.

The real DSH comparison executed ten stratified scenarios three times (30 pairs, 60 Agent sessions). After a deterministic no-new-model-call effect-budget rescore, native L1 Control completed 76.67% of tasks versus 93.33% for Treatment. Control observed four unsafe executions, five invalid actions, five process failures, and 103.577/300.041-second p50/p95 latency; Treatment observed zero unsafe executions, one invalid pre-Runtime decision, one process failure, and 64.312/142.517-second p50/p95 latency. All 17 applicable Runtime audits were valid and neither arm recorded a false commit.

The residual Treatment failures matter: one conservative safe-stop session failed to produce a completed DSH response, and one L1 Agent hallucinated that a generic Provider-failure branch had already occurred before calling Runtime. L0 therefore constrains transactional execution after invocation; it does not replace L1 factual reasoning or invocation decisions. These are model-authored synthetic results, not a production probability or statistically sufficient cross-distribution evidence. Formal ES-P1 still requires independent people, preregistration, and a private human-authored set.
