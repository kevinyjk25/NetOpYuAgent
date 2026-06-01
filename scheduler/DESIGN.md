# scheduler — 设计与实现说明书

> **内存周期任务调度器**(Phase 4,2026-05,原型)。用户提问可触发 LLM 创建一个周期任务,到期自动跑某个工具或对本 agent 发一条 query。调度器注册成 agent 可用的工具。
> **铁律**:本模块**不 import** `runtime/` / `webui/` / `task/`。执行行为全靠两个注入的 callable(`tool_invoker` / `query_runner`),与 `delegate_fn` / `batch_resolver_fn` 同样的 L0/L1 解耦范式 —— `audit_module_independence` 因此持续绿。

---

## 1. 职责

| 文件 | 职责 |
|------|------|
| `service.py` | `SchedulerService`(内存 jobs + history 环形缓冲 + `asyncio` tick loop)、`ScheduledJob` / `JobRun` dataclass、`build_scheduler_tools(service)`(3 个 agent 工具)、`SCHEDULER_TOOL_METADATA`(给检索器/LLM 看的工具元数据)|
| `__init__.py` | export `SchedulerService` / `ScheduledJob` / `build_scheduler_tools` / `SCHEDULER_TOOL_METADATA` |

---

## 2. 三个产品决策(2026-05 确认)

1. **双模式**:注册时选 `mode="tool"`(调本地工具)或 `mode="query"`(对本 agent 发 query,跑完整 LLM 循环)。
2. **内存态**:jobs + 运行历史重启丢失(原型范围)。
3. **结果不主动送达**:不推给用户,只在 SCHEDULE tab 列执行历史。

---

## 3. 核心设计

### 3.1 tick loop(复用 SkillJournalConsumer 范式)

单个背景 `asyncio` task,每 `tick_interval_s`(默认 2s)调一次 `tick_once()`:扫所有 `active` 且 `next_run_at <= now` 的 job,逐个 `_fire`,然后重排(interval job → `now + interval_s`)或终结(one-shot → `done=True`)。`start()`/`stop()`/`tick_once()` 与 `SkillJournalConsumer` 同构(`asyncio.wait_for(stop_event, timeout=interval)` 节拍)。

### 3.2 两种 fire 模式

- **tool**:`await tool_invoker(tool_name, args)` —— 注入的 invoker 通过 loop 同一个刷新后的 ToolRouter registry 派发,所以周期跑的就是 agent 平时用的那个 callable。
- **query**:`await query_runner(query, session_id)` —— 注入的 runner 调 `executor.execute_query`。**每次 fire 用独立 session**(`sched-{job_id}-{runs}`),避免周期 query 堆进一个越来越长的会话。

`_fire` 里所有执行异常都被捕获、记进 `_history`(`ok=False` + error preview),**绝不抛出** —— 调度是 best-effort,一个 job 失败不能拖垮 tick loop。

### 3.3 护栏(原型尺度)

`MIN_INTERVAL_S=5`(防止过密周期)、`MAX_JOBS=100`(防 job 爆炸)、`MAX_HISTORY=200`(环形缓冲,旧记录滚掉)。

---

## 4. 让 LLM 能发现工具(关键接线)

`register_local` 只让工具**可被派发**(`[TOOL:schedule_create]` 能命中),但 LLM 只看得到**进了检索语料**的工具 —— 语料由 `ToolLoader.build_metadata()` 建,`register_local` 不进语料。所以:

1. `SCHEDULER_TOOL_METADATA`(`{name: {description, parameters, returns, hitl, tags}}`,与 `build_metadata()` 同形)在 `main.py` 合并进 `_tool_meta`,被工具检索器索引。
2. 三个工具名加进 `cfg.retrieval.always_inject_extra_tools`,让"定时/周期/schedule"类提问稳定召回它们。

没有这一步,工具是"可调用但 LLM 不知道存在"。

---

## 5. 接线(main.py + webui)

```
main.py:
  ~register_local 区  router.register_local(build_scheduler_tools(SchedulerService()))
                      services["scheduler"] = svc
  ~registry refresh 后 svc.set_tool_invoker(closure over 刷新后 registry)
                      svc.set_query_runner(closure over executor.execute_query)
  ~_tool_meta 建好后   _tool_meta.update(SCHEDULER_TOOL_METADATA)
                      cfg.retrieval.always_inject_extra_tools += 三个名字
  ~startup            await svc.start()

webui/backend.py     register_schedule_routes(app, services)
webui/routes_schedule.py  GET /schedule (jobs+history) · POST /schedule/cancel
webui/index.html     SCHEDULE tab + refreshSchedule()
```

---

## 6. 测试

`tests/test_scheduler.py`(10):tick 触发到期 job、tool/query 双模式(假 invoker/runner)、interval 重排 vs one-shot 终结、cancel、护栏(MIN_INTERVAL/MAX_JOBS/坏 mode/缺字段)、history 环形缓冲 + 错误记录、3 个工具契约、metadata 形状。

---

## 7. 已知边界 / 未做(非原型范围)

- **持久化**:`_jobs` 是唯一状态源,接 sqlite/jsonl 落盘 + 启动恢复即可,接口不变。
- **cron 式调度**:今天只有固定 `interval_s` + 可选 `first_delay_s`(首次延迟;周期 job 默认等一个间隔后首跑,`first_delay_s=0` 可立即首跑)。
- **并发 fire**:今天 tick 内串行 `await`,一个慢 query job 会拖到下一个 tick,但不会并发爆炸。要并发可把 `_fire` 改成 `create_task`,原型阶段串行更可控。
- **结果推送**:按决策只进 history,不走 H2/resumption 通道推前端。
- **per-job 鉴权 / 配额**:无。
