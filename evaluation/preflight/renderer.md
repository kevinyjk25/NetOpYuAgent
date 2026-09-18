# 离线共享 Renderer / Offline Shared Renderer

## 中文

这是固定 `qwen3.5:9b` 的**新实验 profile**，不是现用 Ollama 脏构建的等价性证明，也不是 live 调用计量器。Go 程序直接 import 官方 revision `f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a` 的 `api` 与 `model/renderers`，调用原 `RenderWithRenderer`；未重写模板。程序不连接 Ollama、不调度 runner、不加载模型、不创建 context、不 decode、不生成。依赖中的 HTTP/auth 客户端函数不会调用。

输入为一个严格 UTF-8 native chat JSON 对象；输出仅 `rendered_prompt`、`effective_options`、`identity`、`no_generation:true` 四键。支持文本 system/user/assistant/tool 历史、function tools、`format` 的 JSON/schema、明确 `think=false` 与 `stream=false`。拒绝未知请求控制、多模态、重复键、超限 JSON 和不受支持的 options；不是 `/api/generate`，顶层 `prompt` 明确拒绝。支持的请求 options 为 `num_predict`（必填，1..6000）、`num_ctx`（仅 49152）、`temperature`、`top_p`、`seed`。

有效 options 通过原 `api.DefaultOptions` → SHA 固定的本地 model params → 实验 host context 49152 → 显式请求 options 合并。完整字段包括零值、null 与 runner 未解析的 sentinel；它们是**预调度 profile**，不声称 GPU/thread 等已由后端解决。模型 params 的 presence_penalty=1.5、top_k=20、top_p=0.95 等不是猜测值。DSH 未传 context 时使用新的明确 host profile，不冒充端口 11434 的默认值。未做截断；调用方必须以实际 token IDs 验证输入加预留输出不超 context。

模型 manifest/config/params 每次调用读取并校验 SHA，路径固定为本机 `/Users/steven/.ollama/models`。这个固定 manifest 无 system/template/messages/adapter 层；qwen35 不走 server 的 qwen3 think-tag 过滤分支，固定 `Qwen35Parser.Init` 返回 tools 原值，因此这里直接传原 API 解码后的消息与工具。`format` 完整保存在 identity；它属于 decoder 约束而不进入此 renderer 的 prompt，本程序不创建 decoder。

上游 API 工具结构会忽略部分 schema 注解，例如 `strict`、`additionalProperties`、`maxLength`。本程序刻意复用该行为，`identity.tool_encoding=upstream_api_types`，`tool_encoding_losses` 记录遗漏/改变的路径与原值，`encoded_tools` 记录真正传入 renderer 的结构，同时保留原 stdin SHA。**这只计量实际编码的 prompt，不证明工具 schema 保真或业务约束可执行。**

### 固定输入与复现

`renderer_sources.json` 封存全部 36 个原始文件（Go 包源码、go.mod/go.sum、LICENSE）的 SHA256/Git blob SHA1 和 Go 工具链身份；不包含 helper 自身摘要，避免自引用。helper 源码和最终 binary 另由宿主资产清单封存。官方原文件的 URL 是该 JSON 的 `source_base` 加 `files[].path`。首次下载完整归档超时留下 partial，**未提取/使用**；实际采用按固定 commit 下载、逐 Git blob 核验的完整必要 package。Go 工具链仅一次同 URL 断点续传后完整通过官方 SHA256；无模型下载。

- Go：官方 `go1.26.8.darwin-arm64.tar.gz`，64,626,620 bytes，SHA256 `a012b25b571bd0138a03dcd25375ceba866fe5ca822f426d2c66a4de56fd3f4b`；满足上游 `go 1.26.0`。仅解压到本轮隔离目录，未全局安装。
- 外部源码：`/Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/renderer/ollama`。
- 工具链：同隔离根的 `toolchain-go/go/bin/go`。Go module/build caches 均在隔离 `renderer` 目录内。

首次获取本地缺失 Go module 源码仅运行下面编译步骤（允许网络时，Go 对照固定上游 go.sum 校验；未下载无关后端）：

```sh
cd /Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/renderer/ollama
env CGO_ENABLED=0 GOTOOLCHAIN=local \
  GOPATH=/Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/renderer/gopath GOMODCACHE=/Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/renderer/modcache \
  GOCACHE=/Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/renderer/buildcache \
  /Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/toolchain-go/go/bin/go \
  build -mod=readonly -trimpath -o ../renderer \
  /Users/steven/NetOpYuAgent-dev/NetOpYuAgent/evaluation/preflight/renderer_main.go
```

缓存齐备后，`sh evaluation/preflight/renderer_build.sh OLLAMA_SOURCE GO_BINARY OUTPUT` 会校验全部原文件摘要和完整 Go 文件集合，关闭代理/自动工具链下载，执行纯离线单测与构建。`OUTPUT` 使用上述 `renderer/renderer` 绝对路径；脚本按其父目录寻找 caches。不要求运行现用 Ollama。它不冻结未来 generation，也不抗控制全部资产的恶意本机操作者。

## English

This is a **new offline experimental profile**, not equivalence to the installed dirty Ollama binary and not a live-call meter. It directly imports the pinned official `api` and `model/renderers` packages and calls the original `RenderWithRenderer`. It never calls the Ollama service, schedules a runner, loads a model, creates a context, decodes, or generates. Imported HTTP/auth client functions are not invoked.

Stdin is one strict UTF-8 native chat object; stdout has exactly the four keys documented above. Text history, function tools and JSON/schema format are supported with explicit no-think/nonstreaming controls. Unknown controls, media, duplicate keys, oversized input and unsupported options fail closed. Effective options use upstream defaults, the hash-verified real model parameters, the explicit 49152 host context profile, then request overrides. All fields and unresolved runner sentinels remain visible. This is not the running service's default context. No truncation occurs; the caller must validate actual input tokens plus the output reservation.

The pinned model has no prompt-overlay layers, the qwen35 family is not affected by the server's qwen3 filter, and its parser initializer returns tools unchanged. Format remains preserved as a decoder constraint; no decoder is instantiated. Upstream tool-type encoding may omit schema annotations. The identity explicitly records every detected omission/change, original values, encoded tools and input-byte hash. Accurate counting of that rendered input does **not** establish schema fidelity, business constraints, or generation parity.

The source manifest pins the 36 unmodified upstream files and the verified portable Go 1.26.8 archive. The helper's own sources and binary are pinned separately by host assets. The incomplete full archive was never used; actual packages came from fixed-revision official raw URLs and matched Git blob hashes. The Chinese section includes exact local paths and the bootstrap build command. After dependency-cache population under the isolated workspace, `renderer_build.sh` verifies source content and the complete Go file set, disables module/toolchain downloads, tests and builds offline. Nothing is installed globally or changed in the existing Ollama/DSH environment. These pins assume a cooperative operator; they do not defend against an operator replacing the entire asset set.
