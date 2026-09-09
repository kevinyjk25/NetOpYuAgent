# 结构化流程接线样例 / Structured Flow Wiring Fixture

## 中文

这是一份开发者构造的本地接线说明，不是公开 Skill、独立 Gold 或生产操作手册。

先按用户明确提供的设备 ID，读取设备接口列表；保持嵌套参数和返回字段的原始名称。
只检查列表中的第一个接口；如果列表为空则停止，不能替用户猜测其他接口。
如果第一个接口的 adminUp 为 false，则读取这个设备和接口的错误计数；否则结束只读检查。
错误计数为 0 时结束只读检查；非 0 时仅提出启用该接口的变更候选，不执行写入。
读取必须由宿主明确批准合同和请求，并检查设备、接口访问范围与证据时效。

## English

This developer-authored fixture tests local wiring, not a public Skill, independent Gold or production runbook.
Read interfaces for the explicitly supplied device ID. Preserve nested parameters and original output keys.
Inspect only the first interface; stop on an empty list rather than guessing another interface.
If its adminUp is false, read counters for that exact device/interface; otherwise finish read-only inspection.
Finish if errors equal zero; otherwise emit only an enable-interface change candidate, never execute a write.
Host-bound contract/request consent, device/interface scopes and local evidence age remain mandatory.
