# R0 第三方来源与宿主 Pin / Third-party sources and host pins

## 中文

核验日期：2026-09-18。本文是包外的归属和原始许可记录，不是法律意见、公开分发许可保证或模型执行授权。未修改已封存的 [R0 开发材料](../data/bounded-pilot/r0-development-20260917/README.md)、源码、参考或审阅摘要；本轮操作仅为本地提交准备，不推送或发布。

### 六个活动来源

以下固定 repo/commit/path 对应材料包嵌入的源文本，而不是最新上游版本。原作者的 Skill 文本与本项目新增的任务、合成捕获和开发者裁决是不同内容；归属不表示上游认可本评测。

1. **langfuse/langfuse**，commit `9d08464e89fbf2e29b55638bda74bd605fc07c04`；路径 `.agents/skills/incident-alert-tickets/SKILL.md`（[官方原文](https://github.com/langfuse/langfuse/blob/9d08464e89fbf2e29b55638bda74bd605fc07c04/.agents/skills/incident-alert-tickets/SKILL.md)）。根 LICENSE 明定企业目录外为 MIT Expat，当前路径不在 `ee/`、`web/src/ee/`、`worker/src/ee/`；第三方内容仍适用各自许可。Copyright (c) 2023–2026 ClickHouse, Inc.。[官方 LICENSE](https://raw.githubusercontent.com/langfuse/langfuse/9d08464e89fbf2e29b55638bda74bd605fc07c04/LICENSE)；[本地副本](third-party/r0-20260917/langfuse-LICENSE.txt)。
2. **wshobson/agents**，commit `a30778f8c4e6b0a87567941b7cca4f534bf642b6`；路径 `plugins/cloud-infrastructure/skills/service-mesh-observability/SKILL.md` 及 `references/details.md`（[官方目录](https://github.com/wshobson/agents/tree/a30778f8c4e6b0a87567941b7cca4f534bf642b6/plugins/cloud-infrastructure/skills/service-mesh-observability)）。MIT；Copyright (c) 2024 Seth Hobson。[官方 LICENSE](https://raw.githubusercontent.com/wshobson/agents/a30778f8c4e6b0a87567941b7cca4f534bf642b6/LICENSE)；[本地副本](third-party/r0-20260917/wshobson-LICENSE.txt)。
3. **edenbuilds/touchline**，commit `06a8f965b3e7b1c79382ecb2a13f70b3a891c9a1`；路径 `skills/network-architecture-audit/SKILL.md`（[官方原文](https://github.com/edenbuilds/touchline/blob/06a8f965b3e7b1c79382ecb2a13f70b3a891c9a1/skills/network-architecture-audit/SKILL.md)）。MIT；Copyright (c) 2026 edenbuilds。[官方 LICENSE](https://raw.githubusercontent.com/edenbuilds/touchline/06a8f965b3e7b1c79382ecb2a13f70b3a891c9a1/LICENSE)；[本地副本](third-party/r0-20260917/touchline-LICENSE.txt)。[官方第三方说明](https://raw.githubusercontent.com/edenbuilds/touchline/06a8f965b3e7b1c79382ecb2a13f70b3a891c9a1/THIRD_PARTY_NOTICES.md)及[本地副本](third-party/r0-20260917/touchline-THIRD_PARTY_NOTICES.txt)记录其原创 core 与外部参考/独立服务的边界；不能把那些外部项目一并宣称为 MIT。
4. **openclaw/openclaw**，commit `aafe0878fd73123c5a5eecf076e05a48d86722bd`；路径 `skills/notion/SKILL.md`（[官方原文](https://github.com/openclaw/openclaw/blob/aafe0878fd73123c5a5eecf076e05a48d86722bd/skills/notion/SKILL.md)）。MIT；Copyright (c) 2026 OpenClaw Foundation。[官方 LICENSE](https://raw.githubusercontent.com/openclaw/openclaw/aafe0878fd73123c5a5eecf076e05a48d86722bd/LICENSE)；[本地副本](third-party/r0-20260917/openclaw-LICENSE.txt)。[官方第三方说明](https://raw.githubusercontent.com/openclaw/openclaw/aafe0878fd73123c5a5eecf076e05a48d86722bd/THIRD_PARTY_NOTICES.md)及[本地副本](third-party/r0-20260917/openclaw-THIRD_PARTY_NOTICES.txt)保留 Pi/pi-mono（Mario Zechner，2025）和 Octicons（GitHub Inc.，2026）的 MIT 说明；该说明未单独指认 Notion Skill 源自这些组件。
5. **nlamirault/agentheon**，commit `a987e0da609977e1f04e240fc49d93372004360c`；路径 `agents/argus/skills/security-network-policies/SKILL.md`（[官方原文](https://github.com/nlamirault/agentheon/blob/a987e0da609977e1f04e240fc49d93372004360c/agents/argus/skills/security-network-policies/SKILL.md)）。文件自身声明 `license: Apache-2.0`、`metadata.author: nlamirault`，未给出单独的 copyright 持有人/年份，本文不推断。[官方 LICENSE](https://raw.githubusercontent.com/nlamirault/agentheon/a987e0da609977e1f04e240fc49d93372004360c/LICENSE)与[本地副本](third-party/r0-20260917/agentheon-LICENSE.txt)保留完整 Apache 2.0。固定 [licenserc.toml](https://github.com/nlamirault/agentheon/blob/a987e0da609977e1f04e240fc49d93372004360c/licenserc.toml)明确将 `agents/*/skills/**` 视为 vendored 并保留各自头部；不能将仓库作者身份或根许可证外推为所有收录 Skill 的权利声明。
6. **microsoft/skills**，commit `02e0b2f852b39ea00c43283f999b83fc12079273`；路径 `.github/plugins/azure-kusto-graph-skills/skills/azure-kusto-irql/SKILL.md`，同目录 `references/EXAMPLES.md`、`references/KUSTO_EXPLORER_LAUNCH.md`（[官方目录](https://github.com/microsoft/skills/tree/02e0b2f852b39ea00c43283f999b83fc12079273/.github/plugins/azure-kusto-graph-skills/skills/azure-kusto-irql)）。文件声明 MIT，根 LICENSE 为 MIT；Copyright (c) Microsoft Corporation.。[官方 LICENSE](https://raw.githubusercontent.com/microsoft/skills/02e0b2f852b39ea00c43283f999b83fc12079273/LICENSE)；[本地副本](third-party/r0-20260917/microsoft-LICENSE.txt)。

### 原文身份

下表 SHA256 为官方固定版本原始字节（均省略 `sha256:` 前缀）。八份文本全部保留，不摘要替代。为遵守本次 patch-only 写入方式，原本无末尾换行的 wshobson 与 Microsoft 副本各增加一个最终 LF；正文及原有空格不变，其余六份字节一致。恢复这两份官方原始字节时仅去掉新增的最后一个 LF。

| 本地文件 | 官方 bytes | 官方 SHA256 |
| --- | ---: | --- |
| `langfuse-LICENSE.txt` | 1612 | `fd09d42b5b16606ad2042cc1edf3a82da1c47743709f170c959c74758af229d6` |
| `wshobson-LICENSE.txt` | 1068 | `f89abb55d9f073f38f1703e4518f0613c788c6174be7f13b8dfe48a1c076c746` |
| `touchline-LICENSE.txt` | 1067 | `42ec9d82a5d1ad481ac5c1cab891234eebea5df7161c1598c7707c0c00fd79a0` |
| `touchline-THIRD_PARTY_NOTICES.txt` | 1046 | `f61abbd07c4a8f71bd945145300b3bc7fd0a64087b65a6ba923a362211eff8b4` |
| `openclaw-LICENSE.txt` | 1170 | `73571b25326281d369087f469842c02444fe39faaecebda4d82ed21ff3a1c29d` |
| `openclaw-THIRD_PARTY_NOTICES.txt` | 2855 | `c1d1bbc550feee74853eba104e347341569cbbbe37a9f77659993ca0766277d5` |
| `agentheon-LICENSE.txt` | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `microsoft-LICENSE.txt` | 1140 | `d9a1b1e30d633d5732ea18e3cba9538d293ebc53e1a9e4e96ab739e0c5c4f1cb` |

wshobson 本地副本为 1069 bytes / `9e1374b823d15e80865380262f7f45394312c4b4e3156a7235e22308e4c312a6`；Microsoft 本地副本为 1141 bytes / `c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383`。上游第三方说明以 `.txt` 保存完整原文，避免把它们误当成本项目中英双语撰写的说明。

wshobson 原文最后一行有一个尾随空格；`.gitattributes` 仅对此许可证文件关闭行尾空白告警，保留上游内容，不豁免项目代码检查。

### Tokenizer pin 与复现边界

[源码 pin 导出](benchmarks/tokenizer-source-pins-20260917.json)逐字段保留原 `verification-v3/report.json` 的 9 个 llama 源码/头文件、5 个 Ollama compat 文件的路径、官方固定 URL 和 SHA256，以及 helper 源码/二进制、三个实际动态库与原始报告摘要。来源报告为 9023 bytes，SHA256 `f8fe67c4b3e946fe21df01bcadcb6a71ffc563e6d81ac189898913b1f9ac86e9`。这只是读取已有记录导出，没有重新下载这些源、构建、运行 tokenizer 或调用模型。

克隆后可以检查这 14 条记录、固定上游版本及 helper 源码身份；材料包校验不需要旧 ignored artifacts。宿主外的已构建 helper、模型、动态库、Go 工具链和缓存不随此 JSON 分发。实际重建/重验仍需准备其明确依赖，按 [tokenizer 构建说明](../evaluation/preflight/tokenizer.md)、[renderer](../evaluation/preflight/renderer.md) 与 [codec](../evaluation/runner_codec/README.md)重新冻结宿主资产。它不是跨平台可运行镜像、完整依赖快照或 dirty binary 等价性证明；也没有新的执行授权。

### 公开分发边界

本记录限六个活动来源及上述有限宿主 pin，不替代完整供应链、上游所有权或将来二进制分发审查。未知或未核验许可不能当作允许复用。公开分发前应保持适用许可/版权/NOTICE，复核来源特例和所有拟分发资产；不将第三方内容重新许可为本项目独有作品。

本次常见私钥、API token 与邮箱扫描未命中，不保证穷尽所有敏感信息。历史材料保留 `/Users/steven/...` 本机路径和研究制品标识；公开发布前应明确接受这些环境元数据或另做可审计的脱敏发布副本，不悄悄重写原冻结证据。旧 Netdata 阻塞与失败未删除、未改判；本文件不为被排除内容作新的许可或成功声明。

## English

Verified on 2026-09-18. This is an external attribution/provenance record, not legal advice, a redistribution guarantee, or model-execution authority. Frozen materials, code, references and review hashes were not changed. This work prepares a local commit, not publication or a push.

The six exact repositories, complete commits, selected paths and official links are listed above. Langfuse's root license provides MIT Expat outside its specified enterprise directories, subject to third-party restrictions; the selected `.agents/skills` path is outside those directories (ClickHouse, Inc., 2023–2026). wshobson is MIT (Seth Hobson, 2024); Touchline is MIT (edenbuilds, 2026); OpenClaw is MIT (OpenClaw Foundation, 2026). Microsoft's IRQL metadata and root license state MIT (Microsoft Corporation).

Agentheon's selected Skill explicitly states Apache-2.0 and author `nlamirault`, but supplies no separate copyright owner/year. Its root license is preserved without inventing that missing declaration. The pinned license configuration warns that vendored Skills retain their own upstream headers; a repository-wide license/author claim must not be generalized to every vendored package.

Touchline's full notice preserves its distinction between original core and reference-only or separately installed third parties. OpenClaw's full notice preserves the MIT attributions for Pi/pi-mono (Mario Zechner, 2025) and Octicons (GitHub Inc., 2026); it does not specifically identify the Notion Skill as derived from those components. Upstream attribution does not imply endorsement of these synthetic tasks or developer judgments.

All six complete license texts and both notices are archived locally. The table records the official raw bytes and SHA256. Only two formatting changes exist: patch-only writing adds one terminal LF to the originally unterminated wshobson and Microsoft texts. Their content, including original spaces, is unchanged; remove only that final added LF to recover the recorded official byte digest. The resulting local byte counts/digests are stated above. Upstream notice Markdown is stored as `.txt`, retaining its full original text rather than pretending it is bilingual project-authored documentation.

The wshobson source has one trailing space on its final line. A file-specific `.gitattributes` rule preserves it without exempting project code from whitespace checks.

The tokenizer JSON exports the existing report's nine llama source/header records and five Ollama compatibility records, including all exact paths, official URLs and hashes, plus recorded helper source/binary and dynamic-library identities. It binds the original 9023-byte report by SHA256. This was a read-only record export, not a new download, build, tokenizer execution or inference probe. A clone can inspect these fourteen pins and helper source identity without the former private report path. Host binaries, model, dynamic libraries, Go toolchain and caches are not bundled; actual rebuilding/reverification requires those explicit dependencies and a new host-asset freeze. This is not a portable runtime image, complete dependency snapshot, dirty-build parity proof, or execution grant.

This limited record does not clear all upstream ownership, transitive dependencies or future binary distributions. Unknown/unverified licenses are not permissions. Public distribution still requires preserving applicable license, copyright and notice text and reviewing the exact assets being distributed. Original Skill text remains separate from locally authored tasks, synthetic observations and adjudication.

Pattern checks found no common private/API tokens or email addresses in the audited pre-existing bundle, but are not an exhaustive secret audit. Historical `/Users/steven/...` paths and artifact identifiers remain. Public release must explicitly accept that metadata or use a separately documented redacted release copy without rewriting frozen evidence. Prior excluded Netdata failures and blockers remain; this record neither regrades them nor newly licenses excluded material.
