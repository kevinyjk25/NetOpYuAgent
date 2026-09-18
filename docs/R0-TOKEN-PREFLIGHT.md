# R0 离线 Token 预检 / Offline Token Preflight

## 中文

**2026-09-17：用户授权的离线预检已实现，R0 整体仍为 `r0_measurement_partial`。** 不替换现用 Ollama／DSH；复用固定官方 Go renderer 和本地捆绑 `libllama` 的 vocab-only 计数，生成不可变的请求准备制品。它没有 generation 方法，也未接通真实发送点；`bounded_transport` 仍为 scripted-only。计数不授予执行权限、不证明当前脏构建或未来 generation 的 token 等价性。旧 `paused_unmet`、正式门禁、权限和双驱动／fallback 不变。见[机器摘要](benchmarks/token-preflight-20260917-summary.json)与[考核重置](EVALUATION-RESET-20260916.md)。

### 这一步解决什么

1. **源请求 → 规范 wire**：直接复用现有 broker 的转换；编译器及 Runtime 使用各自真实纯请求构造函数，不另造一套 prompt。
2. **wire → 实际渲染文本**：固定官方 `Qwen35Renderer`，模型 metadata 逐摘要验证；有效 options 来自官方默认值、真实模型 params、明确实验 profile 和显式请求。实验 context 固定 49,152，不冒充现用服务默认值。
3. **文本 → token IDs**：匹配 headers 调用捆绑库，`vocab_only=true`，明确 `add_special=true, parse_special=true`；不分配 tensors、不创建 context、不 warmup、不 decode。6.59 GB GGUF 全文件 SHA-256 已核验，不是只相信文件名。
4. **制品绑定与拒绝**：保留原始请求／wire／prompt／token IDs／完整 options／资产摘要；源请求或资产漂移、超过 context、期限耗尽均拒绝；没有静默截断或自动重试。只提供准备与绑定检查，不提供发送／生成。

代码：[准备与绑定](../evaluation/bounded_preflight.py)、[固定离线探针](../evaluation/bounded_preflight_probe.py)、[renderer 构建说明](../evaluation/preflight/renderer.md)、[tokenizer 构建说明](../evaluation/preflight/tokenizer.md)。

### 验收范围与限制

本轮固定 **6 类请求 × 2 次准备**：纯文本、Unicode／特殊 token 字面量／NUL、工具 Schema、多轮工具结果、实际 compiler wire、实际 Runtime wire。前四项是 native fixtures，后两项复用产品请求构造器；**不是 6 个 Skill，也不是实际 DSH＋9B 问答**。比较完整 prepared 摘要、prompt／token IDs 摘要和计数，另做原请求绑定复核。实测结果及全部失败保留在机器摘要。

**结果：6/6 类、12/12 次通过；双重复制品完全一致，执行源码未变。** renderer 为 5 项顶层测试＋25 子案例；tokenizer 10 个正常／拒绝决策符合预期；Python 准备／绑定单元测试 65 项通过，原请求构造兼容检查 114 项通过。

| 请求类型 | 实际输入 token | 预留输出 token |
|---|---:|---:|
| 纯文本 | 20 | 128 |
| Unicode／特殊标记／NUL | 32 | 128 |
| 工具 Schema | 275 | 128 |
| 多轮工具结果 | 338 | 128 |
| 实际 compiler wire | 1,711 | 4,096 |
| 实际 Runtime wire | 175 | 2,048 |

这些数值仅对应封存的具体请求，不可泛化成任意 Skill 的 token 预算。

最终完整回归为 **3,837 项＋81 子测试通过**（261.11 秒），15 个变更 Python 文件 Ruff、8 项文档／权限检查、shell 语法及 diff 检查通过。另跑既有 DSH 的 3 条 scripted 路径全部通过（8 请求／3 SQLite 读取／0 模型），**没有把离线 tokenizer 接到该 broker**。完整证据和初次失败见机器摘要。

- 上游工具类型编码会丢弃部分字段，如 `additionalProperties`、`strict`。`renderer_identity.tool_encoding_losses` 显示具体路径／原值，`encoded_tools` 显示真正编码的工具；原 wire 仍保留。计数忠实于该编码，**不代表工具 Schema 保真**；Runtime 仍须独立检查参数／权限／效果。`format` 保存为 decoder 约束但未执行 decoder。
- tokenizer 曾遇到连续特殊标记极端输入超过 45 秒并被结束；失败保留，不以替换样例抹去。后续正常 token 溢出检查通过仅证明另一输入的有界拒绝。调用方必须保留超时，字节上限不能当 CPU 上限。
- 完整文件哈希／序列化的超时为完成后的保守检查，不是硬实时中断；helper 输出在捕获后检查尺寸，固定 helper 是可信宿主组件，不能当任意脚本沙箱。
- 资产 seal 是可复核的漂移检测，不是数字签名；不防控制全部资产的恶意本机操作者。依赖快照明确 `completeDependencySnapshot=false`。
- 准备时延包含反复全文件完整性读取和绑定检查，不能当 LLM p50／p95 或上线开销。真实模型调用 **0**；未运行来源 Skill 脚本。

### 如何复现（仅离线）

先按两个构建说明准备固定 helper；宿主资产清单逐文件保存绝对路径及完整 SHA-256，包含 helper、GGUF、实际 dylibs 和 manifest/config/params。路径及二进制属于本机实验制品，不随 Git 发布，不自动下载模型或启动服务。本次清单在 `artifacts/preflight-20260917-assets.json`。

```sh
# 新输出目录必须不存在；不接触运行中的 Ollama。
.venv/bin/python -m evaluation.bounded_preflight_probe \
  artifacts/preflight-20260917-assets.json artifacts/MY_NEW_OFFLINE_PREFLIGHT
# 单条请求也只产生新制品，不发送模型请求。
.venv/bin/python -m evaluation.bounded_preflight \
  artifacts/preflight-20260917-assets.json REQUEST.json NEW_PREPARED.json
```

### 剩余收口与停止条件

离线 helper 不等于完整的隔离 Ollama server，也不等于真实调用前预授权。下一项必须是**同一 prepared request 在发送点被强制消费、与持久预算 reservation 绑定、有效后端配置不再重渲染／截断／漂移**；先以无模型替身验收，不开放真实调用。当前脏构建等价性未建立，不能直接转发到现用 11434 后宣称已绑定。完整控制器、全执行依赖冻结及 6 Skill／12 Task 独立标签仍待完成。遵守原 R0 两工作日／零模型预算，无法闭合则报告具体阻塞，不把本轮单元测试数当研究进展或续开语义调参。

### 2026-09-16 只读核查（保留历史）

| 对象 | 只读观察 |
|---|---|
| 运行服务 | `127.0.0.1:11434/api/version` 返回 `0.33.2`；监听进程的 executable 为 `/Applications/Ollama.app/Contents/Resources/ollama` |
| Ollama 构建 | Go `1.26.0`；revision `f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a`；`vcs.modified=true`，属于脏构建，不能仅凭 commit 宣称二进制与源码完全等价 |
| 模型 | `qwen3.5:9b`；API 与本地 manifest digest 均为 `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`；config 指定 `renderer=qwen3.5`、`parser=qwen3.5` |
| 实际 tokenizer 后端 | 同捆绑 `llama-server` 报告 `0.3.0-dev / d222767c7`；该 Ollama revision 的 pin 为 `b10630`，官方 tag 解析为 `d222767c7a6516559a3f49e7721b6c6b1acc87b4`；另有 Ollama compat 层，不能用任意 upstream wheel 替代 |
| 本次调用边界 | `/api/ps` 返回空列表；只读取版本／标签／运行清单、二进制身份和 GGUF metadata，未调用 chat、generate、embed、debug-render 或 tokenizer，未加载模型、未推理、未下载模型、未改服务 |

模型 manifest 位于 `~/.ollama/models/manifests/registry.ollama.ai/library/qwen3.5/9b`；GGUF blob 为 `sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`。只读 metadata 显示 `gpt2 / pre=qwen35`、248,320 vocab、247,587 merges。GGUF Jinja 存在，但不能冒充当前 Go renderer；缺失的 `add_bos_token` 键也不能直接解释成 false。

### 为什么现有入口仍不够

- 精确字段是 **`_debug_render_only:true`**，返回 `_debug_info.rendered_template`，不是公共 count API。它位于 `scheduleRunner` **之后**；当前无已加载模型，冷启动会进入默认 warmup，执行 `llama_decode`，不能作为严格零推理核查。本次未调用。[Ollama 路径](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/server/routes.go#L2462)、[warmup 默认值](https://github.com/ggml-org/llama.cpp/blob/d222767c7a6516559a3f49e7721b6c6b1acc87b4/common/common.h#L514)、[warmup 执行](https://github.com/ggml-org/llama.cpp/blob/d222767c7a6516559a3f49e7721b6c6b1acc87b4/common/common.cpp#L1403)
- Ollama 公共路由未暴露 tokenize；私有 runner 的 `/tokenize` 不等于稳定公共接口。历史截断调用默认 `add_special=false`，最终 completion 使用 `add_special=true, parse_special=true`；必须复用最终输入语义，不能混用计数。[wrapper](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/llm/llama_server.go#L2312)、[completion 输入](https://github.com/ggml-org/llama.cpp/blob/d222767c7a6516559a3f49e7721b6c6b1acc87b4/tools/server/server-context.cpp#L3788)
- 当前 DSH wire 与 compiler／Runtime 的显式 options 不同，模型 params 和服务默认值还会参与合并。必须绑定完整实际请求、工具编码、`think=false`、format、有效 context／输出上限／sampling options 和截断行为；字符数、UTF-8 bytes、事后 usage 均不能当调用前预授权。

### 原方案与本轮落地边界

原方案不改现用服务、不扩展生产部署，另建可撤销的本地计量实例；本轮仅落地以下准备与计数部分，第 4 项真实发送绑定仍未实现：

1. 在推理 runner 调度前准备请求，共享同一后端的 model overlay、历史处理、parser/tool 处理及 Go renderer，不重写一份猜测模板。
2. 以匹配 headers 调用同捆绑 `libllama`：`llama_model_default_params` → `vocab_only=true` → `llama_model_load_from_file` → `llama_model_get_vocab` → `llama_tokenize`。不创建 context、不 decode、不采样、不 warmup。源码在 vocab 加载后、tensors 加载前返回；本轮 helper 已实现。[vocab-only 路径](https://github.com/ggml-org/llama.cpp/blob/d222767c7a6516559a3f49e7721b6c6b1acc87b4/src/llama.cpp#L335)
3. 封存 source request、normalized wire、模型／renderer／parser／tokenizer 身份、完整有效 options、rendered prompt 与 token IDs 的摘要和计数。超 context 拒绝，不静默截断。
4. count 与后续 generation 使用同一个不可变 prepared request；发送点核对绑定且先取得账本 reservation，任何漂移即停。仅计数阶段保持零推理；generation 只在后续获准的模型阶段开放，不属于 R0 零调用验收。

本轮解除匹配 headers／helper 的离线构建缺口，未解除调度前共享发送入口、脏构建等价性或 count→generation 绑定缺口。有限无 completion 检查通过只说明离线机制，不说明 Agent 收益、语义正确或 R1 已通过。

## English

**September 17: the authorized offline preflight is implemented; overall R0 remains `r0_measurement_partial`.** It leaves Ollama/DSH unchanged and has no generation or dispatch method. Fixed upstream Go rendering and the bundled vocab-only tokenizer produce immutable preparation artifacts. Counts grant no authority and do not establish parity with the installed dirty backend or future generation. The broker is still scripted-only; historical failures, formal gates and dual-driver/fallback boundaries remain unchanged. See the [machine summary](benchmarks/token-preflight-20260917-summary.json) and [protocol](EVALUATION-RESET-20260916.md).

Preparation reuses the existing wire normalization and the actual compiler/Runtime request builders. Effective options merge upstream defaults, verified model parameters, an explicit experimental context of 49,152 and request overrides—not the current service's default. The helper uses matching headers, `vocab_only`, and final-input special-token flags without tensors, context, warmup or decode. The entire 6.59 GB GGUF is hashed, not merely identified by filename. Original request, wire, prompt, token IDs, effective options and asset identities are bound; drift, context overflow and expired preparation fail without truncation or retry.

The finite check covers six request shapes twice: plain text, Unicode/special-marker/NUL, tools, tool history, actual compiler wire and actual Runtime wire. The first four are native fixtures; this is not six Skills, live DSH dialogue or semantic accuracy. Repeated preparation and binding outcomes, component checks and retained failures appear in the summary. Commands in the Chinese section require host-built, fully pinned local assets and fresh output paths. They never send a model request or run source Skill scripts.

All six shapes/twelve preparations passed with identical repeated artifacts and unchanged execution sources. Input counts are respectively **20, 32, 275, 338, 1,711 and 175**; output reservations are in the table above. These are counts for specific sealed requests, not general Skill budgets. The renderer passed five top-level tests plus 25 subtests; ten tokenizer acceptance/rejection decisions matched expectations; 65 preparation/binding unit tests and 114 wire-compatibility checks passed.

Full regression passes **3,837 tests plus 81 subtests** in 261.11 seconds; Ruff passes on all 15 changed Python files, alongside eight documentation/authority checks, shell syntax and diff checks. A separate fresh three-path scripted DSH regression passes with eight requests/three SQLite reads/zero model calls. That broker is **not yet connected to this offline tokenizer**. Initial failures and all evidence remain disclosed.

Important limitations: upstream API tool encoding can drop fields such as `additionalProperties` and `strict`; explicit loss paths/original values and encoded tools are retained. Accurate counting does not imply schema fidelity. Format remains a decoder constraint without a decoder. An extreme consecutive-special-marker case exceeded 45 seconds and was terminated; the failure remains, even though a separate ordinary-token overflow check later passed. Caller timeouts are mandatory. Synchronous hashing/serialization checks are conservative, not forced real-time cancellation. Captured output is size-checked after completion; helpers are trusted host components, not an arbitrary-code sandbox. Seals detect drift, not a malicious operator controlling all files; the dependency snapshot is explicitly incomplete. Preparation timings include repeated full-file integrity checks and are not model or production latency. Actual model calls: **zero**.

Remaining work is a dispatch path that consumes the same prepared request, binds the persistent reservation and prevents subsequent rerendering, truncation or backend-option drift. Test that path with a no-model substitute first; do not forward to the existing dirty server and claim equivalence. The full controller, complete dependencies and independently labelled six-Skill/twelve-task sample also remain open. Preserve R0's original two-working-day/zero-model ceiling; report a specific blocker rather than restart semantic tuning.

The local identity table above records the running Ollama `0.33.2`, its dirty Go build at `f96e7aa…`, the exact Qwen manifest and the matching bundled llama.cpp revision `d222767…` (`b10630`). A matching commit alone does not establish dirty-binary equivalence. The model explicitly selects the `qwen3.5` Go renderer/parser; its embedded Jinja is not a substitute. The relevant installed library is `/Applications/Ollama.app/Contents/Resources/libllama.0.3.0.dylib`.

`_debug_render_only:true` is processed after runner scheduling. With no model currently loaded, that path enters initialization whose default warmup performs a decode; it is therefore unsuitable for this strict zero-inference check. Ollama has no public tokenize route. The private runner tokenizer and history-truncation counter must not be confused with final completion tokenization, including special-token flags and truncation. Fixed-revision source links are provided above.

The September 16 inspection was metadata-only: `/api/ps` was empty and no tokenizer/debug-render or model-loading call occurred then. September 17 adds only offline vocab loading/tokenization, not generation. The original isolated-instance design remains only partially implemented; matching helpers now exist, while shared pre-scheduling dispatch and count-to-generation binding do not. Character/byte proxies and post-response usage still cannot authorize live budgets.
