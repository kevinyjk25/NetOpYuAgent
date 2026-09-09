---
name: 数据集预览
description: 遵守禁运和同意状态，仅查看数据集摘要。
---
# 数据集预览
先使用调用者的 dataset_key 调用 dataset_flags，取得当前标志。
只要存在禁运或缺少同意，便停止 unsupported，不得读取摘要。两个障碍都不存在时，才能读取。
允许读取时，以同一 dataset_key 调用 dataset_summary，完成读取流程（complete the read path）。
不要解除禁运或代替用户提交同意；工具失败或字段缺失必须阻断，不能当作 false。
