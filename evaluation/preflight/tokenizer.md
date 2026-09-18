# 隔离词表 Tokenizer 探针 / Isolated vocab-only tokenizer probe

## 中文

本组件仅作本地词表计量，**不是**推理 runner、调用预算授权或 Agent 能力证据；不访问现用 Ollama 服务。
`liveAdapterReady=false`，现用 dirty binary 与官方源码的等价性仍未建立。

### 接口、构建与身份

调用 `tokenizer --model /absolute/model.gguf`，stdin 传入原始 UTF-8 字节，EOF 结束一次请求。
使用显式长度保留嵌入 NUL，拒绝非法 UTF-8；stdout 仅输出 `token_ids`、`count`、
`add_special:true`、`parse_special:true` 四字段 JSON，诊断只写 stderr。输入最多 4 MiB、
token 最多 262144，拒绝时退出码为 2。调用方仍须独立限制输出和进程时长；输入大小上限不是 CPU 耗时保证。
应使用参数数组与二进制 stdin，不能把 prompt 插入 shell 命令。

2026-09-17 已按官方固定提交 `d222767c7a6516559a3f49e7721b6c6b1acc87b4` 的真实头文件构建，
没有猜测 ctypes ABI。现用动态库及 helper 均为 x86_64，已在 arm64 主机的既有转换环境实际运行。
没有安装系统包、下载模型或修改现用服务。构建命令、完整本地路径及 SHA-256 索引见下方 English 对应章节。
头文件布局断言仅适用于本次核对的二进制组合，不承诺跨版本兼容。

隔离构建目录为 `/Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/tokenizer`。
`pinned/` 的九个头文件／源码及 `ollama-compat/` 的五个兼容层文件来自固定官方提交；未完成的源码压缩包未被使用。
实际加载的非系统库为现用 Ollama Resources 下的 `libllama.0.3.0.dylib`、`libggml.0.22.0.dylib`
和 `libggml-base.0.22.0.dylib`，路径、hash 及原始 dyld 日志均保留。后续宿主须重新冻结／核对这些可变路径及子进程环境。
本组件没有重新 hash 整个 6.59 GB GGUF；blob 文件名本身不构成内容完整性证明。

### 零推理及不加载权重的依据

helper 仅导入日志、模型默认参数、词表模型加载、取词表、tokenize、释放模型六个 llama API；
不初始化 backend/context，不 decode、warmup、采样、embedding 或 generation。
参数固定 `vocab_only=true`、`no_alloc=true`、`load_mode=NONE`、零 GPU 层及空设备列表；
实际日志确认词表分支跳过 tensors，固定源码与现用库反汇编均显示该返回先于 `load_tensors`。

qwen35 compatibility 层会变换 metadata／tensor descriptors，并注册延迟权重加载回调；
因此观察到原 52 keys／883 descriptors 变为 53／442。实际权重读取位于延迟回调内，
由 `load_data_for`／`load_all_data` 的 hooks 触发，词表分支不进入这条路径。官方出处见下方对应章节。
这些源码、符号、日志及二进制分支证据支持本次有限词表执行；不证明 dirty build 与源码完全等价，
也不证明渲染后的输入已绑定到未来 generation。读取 GGUF metadata、分配词表及描述符对象是本次明确允许的行为。

### 验证与保留失败

显式运行 `evaluation.preflight.tokenizer_verify`，传入 `--helper`、`--model`、`--output`；输出目录必须新建。
它不属于默认测试或正式 pilot registry。`verification-v3/report.json` 的十项预期判定全部正确，模型调用为零，
涵盖空串、重复英文、多语言／换行、特殊 token、NUL、非法 UTF-8、输入和 token 上限。
`A\0B` 得到 `[32,188,33]`，`A` 得到 `[32]`，不存在 NUL 截断。

早期记录未覆盖：v1 的 helper 已返回有效空串结果，但 `/usr/bin/time -l` 因受限 sysctl 失败；
v2 的 262145 个连续特殊 token 在 45 秒后由外层终止本地 tokenizer 子进程，**不是模型推理超时**。
v2 当时未持久化该子进程的部分输出，已如实记录，后续 verifier 已补齐超时输出留存。
本次固定验证结束后不再追加对抗重试或真实模型请求。

## English

This helper is a local measurement component, **not** a model runner, a live
budget authorization, or evidence of Agent quality. It does not call the running
Ollama service. `liveAdapterReady` and dirty-build/source equivalence remain false.

### Interface and limits

Run `tokenizer --model /absolute/model.gguf`, passing the exact UTF-8 prompt bytes
on stdin. EOF ends one request. Embedded NUL is retained using `text_len`, not
`strlen`; invalid UTF-8 is rejected. The helper emits only this JSON shape:

```json
{"token_ids":[9419,1814],"count":2,"add_special":true,"parse_special":true}
```

Diagnostics and the actual library's load log go to stderr. Prompt bytes are
limited to 4 MiB and tokens to 262144. A rejected input returns exit code 2.
The caller must independently cap stdout (64 MiB), stderr, and process runtime;
the byte/token caps do **not** bound tokenizer CPU time. Use an argument array
and binary stdin, not shell interpolation. No backend/context initialization,
decode, warmup, sampling, embedding, or generation API is called.

### Audited local build (2026-09-17)

The installed library and helper are x86_64; the host is arm64. The helper was
actually executed through the host's existing translation support; no system
package or model was installed. Compilation used the official headers at
`d222767c7a6516559a3f49e7721b6c6b1acc87b4`, not handwritten ctypes definitions.
The helper asserts the inspected model-parameter layout (72 bytes, vocab flag
offset 64). This assertion is pairing-specific, not a portable ABI promise.

With `tokenizer_build` naming the isolated build directory and `tokenizer_repo`
naming this repository, the build command is:

```sh
xcrun clang++ -arch x86_64 -mmacosx-version-min=14.0 -std=c++17 -O2 \
  -Wall -Wextra -Werror \
  -I "$tokenizer_build/pinned/include" \
  -I "$tokenizer_build/pinned/ggml/include" \
  "$tokenizer_repo/evaluation/preflight/tokenizer.cpp" \
  /Applications/Ollama.app/Contents/Resources/libllama.0.3.0.dylib \
  -Wl,-rpath,/Applications/Ollama.app/Contents/Resources \
  -o "$tokenizer_build/tokenizer"
```

The actual build directory was
`/Users/steven/Documents/Codex/2026-08-26/wo/r0-preflight-20260917/tokenizer`.
`pinned/` contains nine individually downloaded official source/header files;
`ollama-compat/` contains five official files from Ollama revision
`f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a`. Their URLs and SHA-256 values are in
`verification-v3/report.json`. An incomplete, timed-out source archive in the
build directory was never used. No model was downloaded.

`DYLD_PRINT_LIBRARIES=1` in the isolated verification process observed these
non-system dependencies (ordinary system libraries are retained in raw logs):

- `/Applications/Ollama.app/Contents/Resources/libllama.0.3.0.dylib`
- `/Applications/Ollama.app/Contents/Resources/libggml.0.22.0.dylib`
- `/Applications/Ollama.app/Contents/Resources/libggml-base.0.22.0.dylib`

The report records their resolved paths and hashes, plus the helper hash. Do not
assume these mutable paths still identify the same files: a future host must
freeze/recheck them and control its child environment. The model path used was
`/Users/steven/.ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`.
This component's report does not rehash the entire 6.59 GB blob; the basename is
not, by itself, content-integrity verification.

### No-inference / no-weight-load evidence and limits

The helper imports exactly six llama APIs: log setter, default model parameters,
model load, vocabulary getter, tokenize, and model free. `nm -u` output is saved.
It sets `vocab_only=true`, `no_alloc=true`, `load_mode=NONE`, GPU layers zero,
and an empty device list. The observed library log says `vocab_only=1`,
`no_alloc=1`, and `vocab only - skipping tensors` before tokenization.

The pinned loader initially reads GGUF metadata with `no_alloc=true`; its
vocab-only branch returns before `load_tensors`. The installed library's
disassembly independently shows the flag comparison at offset `0x40`, then the
skip-log path jumping past the tensor-loader virtual call.
[Pinned control flow](https://github.com/ggml-org/llama.cpp/blob/d222767c7a6516559a3f49e7721b6c6b1acc87b4/src/llama.cpp#L351)

The installed Ollama compatibility layer does run for this GGUF. Its qwen35
path transforms metadata/tensor descriptors and registers deferred load
callbacks; this explains the observed 53 keys / 442 descriptors after the
original 52 keys / 883 descriptors. Weight reads are inside those callbacks,
which the hooks invoke from `load_data_for` / `load_all_data`, not metadata
translation. The vocab-only return skips this tensor-loading path.
[qwen35 compatibility](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/llama/compat/llama-ollama-compat.cpp#L859),
[load hooks](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/llama/compat/001-llama-cpp-hooks.patch)

These source, symbol, observed-log and local binary-branch checks support this
bounded vocab-only execution. They do not establish that the installed dirty
Ollama binary is identical to the official source, nor that a renderer's output
is the exact later generation input. No inference context was created; reading
GGUF metadata and allocating vocabulary/descriptor objects is intentional.

### Fixed verification and retained failures

`python -m evaluation.preflight.tokenizer_verify --helper ABSOLUTE_HELPER
--model ABSOLUTE_MODEL --output NEW_DIRECTORY` performs ten finite checks and
refuses an existing output directory. It saves stdout, stderr, input digests,
actual dependency paths/hashes and the final report. This is an explicit local
probe, not part of default tests or the official pilot registry.

`verification-v3/report.json` reports 10 expected decisions correct, with zero
model calls: empty, English twice, Unicode/newlines, special tokens, embedded
NUL and its prefix, invalid UTF-8, prompt overflow, ordinary-token overflow.
`A\0B` gave `[32,188,33]`, whereas `A` gave `[32]`.

Earlier artifacts were retained without overwriting:

- `verification-v1`: the helper produced a valid empty result, but the wrapping
  `/usr/bin/time -l` failed on a restricted sysctl, so the wrapper was removed.
- `verification-v2`: 262145 consecutive special-token markers exceeded the
  45-second process deadline. Python terminated the local tokenizer child;
  this was **not** a model-inference timeout. The failure is recorded explicitly.
  The then-current verifier did not persist that child's partial output; the
  verifier now preserves timeout output for future runs.

No additional adversarial retries or live model requests belong to this probe.
