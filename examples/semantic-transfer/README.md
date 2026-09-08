# 语义迁移开发 Skill / Semantic Transfer Development Skills

## 中文

这里有两批共十二份开发助手编写的 Skill 包，用于检查首次转译与机制修订；不是下载的公开 Skill 或独立 Gold。它们不包含 L0 答案。测试合同、手工参考和有限 Oracle 在 `evaluation/flow_semantic_examples.py`，不会放入模型请求。所有脚本保持惰性，宿主只有内存只读测试接口。

| Skill | 结构 | 文件 |
|---|---|---|
| 发票披露 | 两个条件任一满足，OR | [SKILL.md](invoice/SKILL.md) |
| 发布窗口 | 存在 AND 不陈旧 AND 许可；废弃历史规则 | [SKILL.md](release/SKILL.md) |
| 调度预览 | 有效 AND（快速资格 OR 已签审阅） | [SKILL.md](dispatch/SKILL.md) |
| 联系人卡片 | 引用文件、否定条件、AND | [SKILL.md](contacts/SKILL.md) / [隐私规则](contacts/references/privacy.md) |
| 资产指标 | 输入别名→返回规范键；拒绝退役资产 | [SKILL.md](assets/SKILL.md) |
| 归档元数据 | 附带脚本但缺批准 runner；读取前停止 | [SKILL.md](archive/SKILL.md) / [惰性脚本](archive/scripts/precheck.py) |

第二批是新输入小批，其后同源修订不能再称为未见验证：

| Skill | 结构 | 文件 |
|---|---|---|
| 会员权益 | 非暂停 AND（等级覆盖 OR 赞助许可） | [SKILL.md](membership/SKILL.md) |
| 数据集摘要 | 中文规则、两个否定条件 | [SKILL.md](dataset/SKILL.md) |
| 副本读取 |（同步 AND 非隔离）OR 紧急租约 | [SKILL.md](storage/SKILL.md) |
| 工单证据 | 已确认 AND（非敏感 OR 已委托） | [SKILL.md](helpdesk/SKILL.md) |
| 证书信息 | 引用中的到期规则，返回序列号作为下游参数 | [SKILL.md](certificate/SKILL.md) / [规则](certificate/references/expiry.md) |
| 容量报告 | 前置参考缺失；不得猜政策 | [SKILL.md](capacity/SKILL.md) |

参见[源审查、机制与结果](../../docs/FLOW-SEMANTIC-TRANSFER.md)。不要把模型第一次见到这些文本等同于开发团队未见过的密封测试。

## English

These two batches contain twelve assistant-authored Skill packages, not downloaded public Skills or independent Gold. No L0 answers are included in the packages. Private references/oracles and inert host declarations live in `evaluation/flow_semantic_examples.py`, outside model requests. Scripts must never execute. Later repairs on the same inputs are development revisions, not fresh holdouts.

The tables link all sources and references, including the follow-up membership alternative, Chinese dataset rule, replica exception, conditional delegation, certificate serial binding and missing capacity policy. See the [audit and results](../../docs/FLOW-SEMANTIC-TRANSFER.md). New-to-model text is not a developer-blind sealed test.
