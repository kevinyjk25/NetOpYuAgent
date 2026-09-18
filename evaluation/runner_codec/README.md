# R0 token-array runner codec / Token-array runner codec

## 中文

这是现有 PreparedRequest 的窄接线组件，不是第二套评测、分词器或推理后端。
`bounded_runner.serialize_completion` 将已封存的 `token_ids` 原样放进原生
`POST /completion` 的纯整数 `prompt` 数组，逐项映射已解析 sampling options，
`n_predict` 等于输出预留。它不重新渲染、分词、调用服务或授予权限。
`parse_completion` 仅接受完整、未截断、usage 已知且一致的结果，再通过本目录
Go helper **直接导入官方 Qwen35 parser** 解析文本与工具调用。

当前范围仅机械验收：真实本地 helper + 同一 serializer/parser + 无推理 HTTP
替身。`generation_enabled=false`；没有启动 `llama-server`，没有执行 prefill、
decode 或模型请求。替身成功不能设置 live-ready，也不证明 dirty 安装后端的
实际生成等价性。`encode_fixture_completion` 的输出计数明确是合成值，不是估计器。

固定接口与约束：

- prepared 的 source/wire/prompt/token 摘要、Qwen3.5 profile 与上下文必须一致；
  host 仍需验证资产与 parser binary 摘要。普通 SHA seal 不是对恶意操作者的签名。
- context 为 49152，保守要求 input + output reservation **小于** context；
  不发送后端不认识的 `shift`/`truncate` 字段。它们由启动策略约束。
- 此实验 profile 显式固定 `stream=false, cache_prompt=false`，两臂一致，
  不是声称与现用 Ollama 的默认 streaming/cache 设置相同。
- `format="json"` 使用固定上游 `grammarJSON` 的原始字节；对象直接保留为
  `json_schema`。工具仍使用旧 renderer 的原始 upstream API 编码，已有 schema
  loss 记录不消失，也不能宣称完整工具 schema 都进入了提示。
- 只认 `stop_type=eos|word|limit`、`stop=true`、`truncated=false`；
  `timings.cache_n+prompt_n` 必须等于预授权输入，`predicted_n` 不超过预留，且
  与 `tokens_evaluated/tokens_predicted` 相符。未知/异常由父层保留预留并终止，
  本模块不结算、不退款、不重试。
- 官方 parser 保留其自身行为（例如末尾空白、未完成工具片段的处理），不增加
  另一个“修复”parser；宿主工具权限/schema 检查仍不可跳过。

未来实际进程的必要合同（本轮不启动）：host-owned 私有 loopback 端口、固定
model/binary/shared-library 哈希，`--offline --no-webui --no-context-shift
--no-warmup --fit off -c 49152 -np 1`，无 draft/LoRA/多模态。
`num_batch=512` 等准备阶段进程参数必须由同一个固定启动 profile 绑定；
`num_gpu=-1`、`num_thread=0`、mmap=null 是未调度 sentinel，不能声称是实际
解析后的硬件参数。`/props` 校验 context、单 slot、model_path 仅是必要条件，
不能独自证明所有进程 flags。`n_predict=0` 仍可能计算 prompt，**不是零推理探针**。

### 同源依据和可复现构建

Ollama 固定 revision `f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a`；
llama.cpp 固定 revision `d222767c7a6516559a3f49e7721b6c6b1acc87b4`。
`sources.json` 记录全部编译用源码 Git blob 与 SHA256，以及 gofmt 后 helper 摘要。
源码是小型已核验子集，未下载/重建整个后端。无新全局依赖；使用已核验的便携
Go 1.26.8、已有 module cache，构建强制 `GOPROXY=off, CGO_ENABLED=0`。

```sh
sh evaluation/runner_codec/build.sh PINNED_OLLAMA_SOURCE GO_BINARY CACHED_MODULES OUTPUT
```

构建脚本不会下载、启动 runner 或访问模型。helper stdin 为
`{"wire": <normalized native request>, "content": <raw completion text>}`；
stdout 为 message、preserved_tokens、固定 parser identity、`no_generation:true`。

官方依据：
[原生 completion 映射和 grammarJSON](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/llm/llama_server.go)、
[Qwen35 parser](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/model/parsers/qwen35.go)、
[纯整数 prompt 路径](https://github.com/ggml-org/llama.cpp/blob/d222767c7a6516559a3f49e7721b6c6b1acc87b4/tools/server/server-common.cpp)、
[原生响应和 stop 枚举](https://github.com/ggml-org/llama.cpp/blob/d222767c7a6516559a3f49e7721b6c6b1acc87b4/tools/server/server-task.cpp)。

## English

This is a narrow codec for the existing PreparedRequest, not a second harness,
tokenizer or inference engine. The serializer sends the sealed token IDs as an
unchanged integer `prompt` array to native `/completion`; it never renders,
tokenizes, starts a runner or grants authority. Sampling options and the output
reservation are bound explicitly. JSON schema is preserved; JSON mode uses the
exact pinned upstream grammar. Parsing imports the official Qwen35 Go parser.

R0 attestation is mechanical only: local no-inference helpers and an HTTP
fixture exercise the same serializer/parser. Generation stays disabled; no
runner, prefill or decode was started. Fixture output counts are synthetic and
cannot establish model performance, live readiness or installed dirty-backend
generation parity. Existing upstream tool-schema encoding losses remain visible.

The fixed profile uses context 49152, input plus reserved output strictly below
context, nonstreaming responses and no prompt-cache reuse. Shift and truncation
are process constraints, not invented request fields. Future dispatch requires
host-pinned assets/processes, a private loopback endpoint, single slot,
`--no-context-shift --no-warmup --fit off`, no draft/LoRA/multimodal, and `/props`
checks. Unresolved GPU/thread/mmap sentinels are not claimed to be resolved
hardware settings. `/props` alone does not attest startup flags. `n_predict=0`
may evaluate the prompt and is never a zero-inference probe.

Only completed non-truncated responses with known, consistent, in-budget usage
are usable. Input accounting sums cached and newly evaluated tokens. On any
error the parent retains/settles the reservation; this module cannot refund or
retry. Official parser behavior is preserved, including its limitations; host
tool authorization/schema checks remain mandatory. Source and binary seals are
drift checks, not protection against a malicious operator.

The source manifest binds the pinned revisions linked above, exact Git blobs,
SHA256 values and formatted helper sources. The offline build command above
uses existing Go 1.26.8 and cached modules, disables CGO and network module
resolution, and never rebuilds or starts the inference backend. Helper stdin
contains the normalized native wire and raw completion text; stdout contains
the parsed message, preserved tokens, parser identity and `no_generation:true`.
