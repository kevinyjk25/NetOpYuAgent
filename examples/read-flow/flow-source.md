# 本地库存条件读取 / Conditional local inventory read

## 中文

这是手工构造的流程接线夹具，不是公开 Skill 或 LLM 转译结果。

1. 输入一个 `device_id`，调用 `read_inventory_device` 读取本地库存。
2. 校验返回结构。将返回的 `site` 与字符串 `campus` 比较。
3. 相等时，用第一步返回的 `device_id` 再读取一次，完成只读路径。
4. 不相等时，返回 `needs_l1`，等待上层判断；本夹具不调用模型。
5. 参数、权限、返回值、引用或结果时效不满足时阻断，不把异常当成条件为假。

重复读取只是为了验证步骤输出到输入的接线，不是生产上推荐的库存查询方式。数据为本地计划库存，`planned-lab` 不代表实时健康。宿主分别授予两台设备的只读权限，流程文件本身无权授予权限。

## English

This is a hand-authored wiring fixture, not a public Skill or LLM translation.
Read one `device_id` using `read_inventory_device`; validate the result and
compare `site` to `campus`. On equality, read again using the first result's
`device_id` and finish the read path. Otherwise return `needs_l1` without
calling a model. Invalid parameters, access, results, references or elapsed
read freshness block the flow rather than selecting the false branch.

The repeated read tests dataflow, not a recommended production query pattern.
Inventory is local planned data, not live health. Host-granted device scopes
are separate from the flow; source text grants no permission.
