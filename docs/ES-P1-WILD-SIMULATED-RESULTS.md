# ES-P1-Wild 角色隔离模拟结果 / Role-Separated Simulation Results

## 中文

### 结论与证据边界

`ES-P1-Wild-Sim` 的本地技术协议已经完整执行：15 个固定公开 Skill、45 个案例、3 次重复，共 135 组成对观察、270 次真实本地 DSH 实验臂执行。Control 为 `DSH + qwen3.5:9b + 原始 L1 Skill + LLM 原生工具编排`；Treatment 使用完全相同的模型、Skill、任务、Tool Catalog、fixture、审批和故障输入，只增加 Gold-blind 的 L1→L0 转译门、L0 auto Runtime 与原生 L1 fallback。

这不是两名真人完成的独立评测。`simulated.case-author-a` 与 `simulated.gold-author-b` 是物理隔离信息面的虚拟角色输入；结果只能证明角色分离协议、转译门、真实 DSH Tool loop、事务 Runtime 和事后评分链路在本地仿真中可运行。它不等于私有隐藏集、真实外部系统、生产成功概率或正式 ES-P1 资格。

### 固定实验输入

| 项目 | 值 |
|---|---|
| 模型 | `qwen3.5:9b`，artifact `sha256:6488c9…3ea7` |
| 调用配置 | Ollama native chat，`think=false`、`temperature=0`、`num_ctx=32768` |
| 公开 Skill / 案例 | 15 / 45 |
| 挑战类型 | nominal 15；ambiguous/missing 15；failure/adversarial 15 |
| 重复 / paired observations / arm executions | 3 / 135 / 270 |
| 并发 | 2 个隔离 DSH worker；并发配置会影响时延可比性 |
| Gold 可见性 | 两臂全部结束后才加载 |
| 第三方代码 | 不执行 Skill 脚本、hook 或 installer；只使用受审声明式 fixture MCP |

### 转译门结果

| 指标 | 结果 |
|---|---:|
| 协议有效输出 | 45/45 |
| L0 Runtime / 原生只读 / safe-stop | 14 / 6 / 25 |
| 与模拟 Gold 的后验路由一致 | 43/45（95.56%） |
| 不安全 Runtime 误接纳 | 0 |
| 保守只读 fallback | 2 |
| 发生语义修复的案例 / 修复调用 | 22 / 25 |

两处路由偏差均是只读任务被保守降为 `safe_stop`，不是写操作被错误交给 Runtime；两案仍通过最终 Oracle。路由一致率是 Gold 后载入后的评估，Translator 运行时没有读取 Gold。

### DSH + L1 与 DSH + L0 auto Runtime

| 指标 | Control：原生 L1 | Treatment：L0 auto Runtime | 差值 |
|---|---:|---:|---:|
| Task Completion / Oracle | 111/135（82.22%） | **132/135（97.78%）** | **+15.56 pp** |
| 参数精确绑定 | 97.04% | **97.78%** | +0.74 pp |
| Execution Precision | 97.04% | **97.78%** | +0.74 pp |
| Autonomous Coverage | 82.22% | **97.78%** | **+15.56 pp** |
| Unsafe Execution | 0 | 0 | 0 |
| False Commit | 0 | 0 | 0 |
| p50 | 32.947 s | **27.808 s** | **-5.139 s（-15.6%）** |
| p95 | 109.279 s | **56.067 s** | **-53.212 s（-48.7%）** |

成对结果为 Treatment 胜 21、Control 胜 0、两臂都通过 111、两臂都失败 3。三轮分别都是 Control 37/45、Treatment 44/45，说明本轮差值不是单轮偶然波动。

### 提升来自哪里

| 路由 | Control | Treatment | 解释 |
|---|---:|---:|---|
| `l0_runtime` | 21/42（50.00%） | **42/42（100%）** | 7 个故障/对抗案例的三次重复均由 Runtime 完成验证失败后的确定性回滚 |
| `l1_native_read` | 15/18（83.33%） | 15/18（83.33%） | 两臂相同，证明 Runtime 没有虚构只读能力提升 |
| `safe_stop` | 75/75（100%） | 75/75（100%） | 两臂均正确拒绝缺参、未审批或越权写入 |

按挑战类型看，ambiguous/missing 两臂均 45/45，nominal 两臂均 42/45；差异全部来自 failure/adversarial：Control 24/45（53.33%），Treatment 45/45（100%）。因此本轮证据支持的是“合格 L0 写事务的验证与回滚收口”，不是笼统宣称 Runtime 改善所有 Agent 推理。

### 残余失败与解释

Treatment 的 3 次失败全部来自同一个 `fivem-debugging` nominal 只读案例：两臂都没有调用必需 capability，参数绑定和状态 Oracle 均失败。该案例保持原生 L1 fallback，不属于 L0 写事务覆盖范围；它暴露的是上层 Skill/工具选择缺口。

Control 的 24 次 Oracle 失败中，21 次位于 7 个 failure/adversarial L0 案例；原生 Agent 虽有时完成物理恢复，但没有稳定产生 Runtime 所要求的 `rollback_verified` 终态证据。Treatment 对这 21 组配对全部取胜。Treatment 的 42 次 L0 执行中有 1 次由控制器自动补触发 Runtime，其余由 Agent 发起 Effect、Runtime 接管事务后端。

### 复核与产物

完整本地产物位于：

- `.../es-p1-wild-simulated-independent-20260901/evidence-report/report.html`：可搜索、筛选、按 repetition/route/challenge/Skill 分层并查看案例失败门；
- `report.md`：双语摘要；
- `summary.json`：机器可读完整分层结果；
- `manifest.json`：四文件摘要清单。

最终 report digest 为 `sha256:8b2fa52cf5c63e94572ceba5aefafbaa0ff8726b3fff0cf9f0ee37973ae257ba`，manifest digest 为 `sha256:fdbcd159355b29e5c43dc5db88328432ebe181d4871d268c0d3fce7c90fd6001`。仓库内的精简机器摘要见 [benchmark summary](benchmarks/es-p1-wild-simulated-summary.json)。

复现完整实验需要先生成并封存角色模拟工作区，然后运行：

```bash
scripts/netopyu-market-corpus paired-run /ABS/PATH/bound-study \
  --output-root /ABS/PATH/paired-result \
  --model qwen3.5:9b --workers 2 --native-no-think
scripts/netopyu-market-corpus simulation-evidence-report \
  /ABS/PATH/simulation-root /ABS/PATH/paired-result \
  --output-root /ABS/PATH/evidence-report
scripts/netopyu-market-corpus simulation-evidence-inspect \
  /ABS/PATH/evidence-report
```

### 尚未完成的正式证据

本轮完成的是 `ES-P1-Wild-Sim`，不是正式独立人工 ES-P1。下一道不可由本项目自行模拟替代的门是：Runtime 团队之外的人员在仓库外编写和复核 private holdout，保持 Case Author、Gold Author、Reviewer 和运行者隔离，并用相同冻结模型与协议执行。公开 Skill 还应从 15 个扩到预注册的 50–100 个，增加语言、作者和工具类型；真实设备与真实业务系统证据属于后续 ES-P2。

## English

### Result and boundary

The local `ES-P1-Wild-Sim` protocol is complete: 15 pinned public Skills, 45 cases, three repetitions, 135 paired observations, and 270 real local DSH arm executions. Control is DSH plus qwen3.5:9b, the original L1 Skill, and LLM-native Tool orchestration. Treatment receives byte-equivalent model, Skill, task, Tool Catalog, fixture, approval, and fault inputs; the only functional addition is the Gold-blind L1-to-L0 gate, L0 auto Runtime, and native-L1 fallback.

This is not a two-human independent evaluation. `simulated.case-author-a` and `simulated.gold-author-b` are isolated virtual protocol roles. The result validates local mechanics, not a private holdout, real external systems, production probability, or formal ES-P1 qualification.

Translation produced 45/45 protocol-valid outputs: 14 L0 Runtime routes, six native reads, and 25 safe-stops. Post-run route agreement with simulated Gold was 43/45 (95.56%), with zero unsafe Runtime accepts and two conservative read fallbacks.

Control completed 111/135 tasks (82.22%); Treatment completed 132/135 (97.78%), a 15.56-point gain. Treatment won 21 pairs, Control won none, 111 pairs both passed, and three both failed. Every repetition was exactly 37/45 versus 44/45. Unsafe executions and false commits were zero in both arms. Treatment reduced p50 from 32.947 to 27.808 seconds and p95 from 109.279 to 56.067 seconds under the fixed two-worker configuration.

The gain is localized and interpretable. L0 Runtime cases improved from 21/42 to 42/42, entirely through seven failure/adversarial cases repeated three times. Native reads remained 15/18 in both arms and safe-stop remained 75/75. The only Treatment failure was one native-read `fivem-debugging` case repeated three times, where both arms missed the required capability and parameter binding. Runtime therefore did not hide an upstream L1/Tool-selection gap.

The full digest-bound HTML, Markdown, JSON, and manifest are generated outside the repository. The report digest is `sha256:8b2fa52cf5c63e94572ceba5aefafbaa0ff8726b3fff0cf9f0ee37973ae257ba`. A compact committed summary is available in [benchmark summary](benchmarks/es-p1-wild-simulated-summary.json).

Formal ES-P1 remains open. It requires Runtime-external humans, repository-external private cases, blind review/adjudication, and the same frozen execution protocol. Broader public-market coverage and real-system/device evidence remain separate future stages.
