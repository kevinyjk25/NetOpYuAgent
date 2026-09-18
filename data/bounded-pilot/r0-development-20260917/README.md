# R0 开发材料（2026-09-17）

这是已知开发材料，不是确认集、人类 Gold 或模型能力结果。6 个固定公开来源、12 个任务（8 正向、4 边界）及本地合成捕获数据用于有限只读/惰性草稿开发。它不授予产品 Effect 权限，不执行来源脚本，不代表完整厂商工作流。

两份隔离的 AI 辅助开发标注及其 v4 补充完整原文保存在 `annotation-a.json`、`annotation-b.json`；它们不是人类独立评审。`adjudication.json` 逐案记录合并、适用范围、critical/strict 修正和每条标注职责的去向。根 Skill、全部已供给引用和固定 commit/摘要保存在 `sources.json`；历史绝对路径只是来源记录，克隆后的校验与运行不读取旧的 ignored artifacts。

`metadata.json` 和 `adjudication.json` 是不可变的 **主代理审阅前快照**。最终材料状态只以 `review-manifest.json` 及其绑定的 `root-review.json` 为准。初始 `pending_root_adjudication` 必须被运行门禁拒绝；打包或校验不会自动批准。

Netdata 在入组/执行前因强制资料闭包、预算及 wrapper/curl 适配冲突被排除。原始 SNMP 标注、阻塞与旧失败来源记录仍保留；这不是对其职责的豁免、修复或回归通过。IRQL 两个任务及 mesh 原始诊断意图保留。

Provider 输出 schema 仅为 optional/open 类型提示，不含答案常量、不保证证据完整或成功。原始状态和参考只供宿主/评估者，不能直接注入候选。`dialogues.json` 仅由公开固定请求 schema 构造，是零推理工具传递探针，不是自主策略或最终答案。

Strict 候选范围预先固定为证据依赖、资源绑定与有限类型/区间谓词。准确提取的 IRQL 半开区间可检查，但不能由它推导完整 KQL 正确；有限 flow tuple 不能证明完整 NetworkPolicy 联合语义。L1→L0 自动提取及其错误/遗漏仍在未来 R1 的固定分母内，不能因提取失败先降为 unsupported 再排除。未观测/未可靠证明仍为 unknown（或由真实审阅判 not_met），绝非通过。完整 YAML/KQL、因果、脱敏与完整交付物语义需要另行判断。

独立校验（不需要原始外部文件、不调用模型）：

```sh
.venv/bin/python -m evaluation.bounded_material verify data/bounded-pilot/r0-development-20260917
```

主代理逐案读完裁决和参考后，可新建显式评审 JSON，字段为 `reviewer: "root"`、`decision: "approved_for_bounded_development"`、非空 `note`，以及实际 `pending_manifest_sha256`、`adjudication_sha256`、`references_sha256`（均为完整文件原始字节 SHA256，含 `sha256:` 前缀）。随后调用：

```sh
.venv/bin/python -m evaluation.bounded_material finalize data/bounded-pilot/r0-development-20260917 /absolute/path/to/explicit-root-review.json
```

该操作只允许一次，保留 pending manifest，绑定完整评审并更新 manifest；中断或重复调用不自动补签。它只是材料准入，不执行 24 臂、不授权模型推理、不声明研究有效或 R0 完成。

## English

This is a known, AI-assisted developer set, not a holdout, human Gold or model result: six pinned public Skills, twelve tasks (eight positive/four boundary), and synthetic local captured observations. Sources are inert and no product Effect authority is granted.

The complete isolated nonhuman annotations and v4 addenda are preserved as original UTF-8 text with hashes in `annotation-a.json` and `annotation-b.json`. `adjudication.json` records each case's merged duties, item dispositions, applicability and critical/strict corrections. `sources.json` embeds the complete supplied sources and pins. Historical filesystem paths are provenance only; standalone verification and execution do not read the old ignored artifacts.

Metadata and adjudication retain their immutable pre-root-review state. The authoritative final status is the manifest plus its bound root review. Pending material is rejected by the acceptance gate. Only explicit review of the exact manifest, references and adjudication can finalize it once; finalization neither runs acceptance nor approves inference or research claims.

Netdata was excluded before enrollment for mandatory source-closure/budget and wrapper/curl adaptation conflicts. Its original blocking annotations and historical failure provenance remain; exclusion is not a repair, waiver or regression pass. The two IRQL cases and historical mesh diagnostic intent remain.

Output schemas are optional/open structural hints, not answer constants or completeness guarantees. Raw state/references are evaluator-only. Public-schema-derived dialogues are zero-inference delivery surrogates, not autonomous agent strategies.

Evidence dependencies, resource bindings and finite typed predicates are prespecified strict candidates. Extraction errors/omissions remain in the future R1 denominator; they must not be excluded by relabelling extraction as unsupported. A checked half-open interval is not complete KQL correctness; finite approved-flow checks are not full Kubernetes policy semantics. Missing proof remains unknown or an independently reviewed failure, never an automatic pass. No source scripts, models or external services are invoked by this material utility.
