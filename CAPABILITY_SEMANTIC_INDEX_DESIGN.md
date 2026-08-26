# 设计 — 能力语义索引(Capability Semantic Index, CSI)

> 状态:**待评审**。目标:给 tool(原子)和 skill(组合)一个**统一、可解释**的
> 能力空间表示 + 相似度/归属接口,让 P0-P3 复用同一套"像不像/归哪里"判断,
> 而不是各自造轮子。吸收 MoE 的"簇 + top-k 软路由"形态,但**归属可读、可调、
> 可审计** —— 不引入学习式黑盒 gating。

## 0. 为什么需要它(问题陈述)

现在"这俩能力像不像 / 这个请求归哪个 skill"的判断**散落且不一致**:
- retriever:embedding(cosine)+ BM25 —— 用于 skill/tool 检索
- evolver `_find_similar_skill`:**jaccard 词集重叠** —— 用于 merge 找相似
- 偏好学习:embedding 相似(find_similar_facts)
- P1 轨迹检测:**还没有**,若各做一套就是第四套

同一个问题四套答案。P1(轨迹归类)、P3(追加诉求归属哪个 skill)、进化1(query
落哪个 skill 辖区)本质都是**能力空间里的相似度/归属查询**。统一成一个索引,
P0-P4 都受益,且避免 jaccard vs embedding 这种不一致重演。

## 1. 设计原则(守住可解释)

延续你"结构性约束替代 LLM 自由发挥"的哲学:

1. **显式 > 学习**:归属由可读特征算出(embedding + 结构化标签),不是 gating 学权重。
2. **带理由**:每次归属返回 `(target, score, reasons[])` —— "归到 wireless 簇,因为
   tags 重叠 0.7 + action_type 同 + embedding 0.6",可在 journal 观察、可人工干预。
3. **可调可编辑**:簇定义、权重、阈值都在 config / 声明里,改了立即生效,无需重训。
4. **轻耦合**:CSI 是一个独立模块,只读 tool/skill 元数据 + 复用现有 retriever,
   不反向依赖 evolver/loop/preference。P0-P3 单向依赖 CSI。

## 2. 吸收 MoE 的形态(但可解释)

| MoE 概念 | CSI 的可解释对应 |
|----------|-----------------|
| Expert(专家) | **Capability Cluster(能力簇)**:按 domain×action 聚成可命名的簇(如 `wireless_diag` / `wired_diag` / `access_admission` / `app_access` / `config_change`)。每个簇是人能读的名字 + 成员 tool/skill 列表。 |
| Gating network(学出来的路由) | **可解释归属打分**:query/轨迹 → 各簇的相似度分(embedding+标签+action),带 reasons。无训练。 |
| Top-k routing(软分配) | **top-k 簇命中**:一个 query 可命中多个簇(正好对应多 skill 歧义),每个分数可读。 |
| Load balancing | **不需要**(我们是检索,不是算力分配) |

**取其"清晰划分 + 软分配"的实用价值,丢其"不可解释 + 要训练"的包袱。**

## 3. 数据模型

```python
@dataclass
class CapabilityVector:
    """一个 tool 或 skill 的统一语义表示。"""
    cap_id: str                  # tool 名 或 skill_id
    kind: str                    # "tool" | "skill"
    embedding: list[float]       # 复用现有 embedder(nomic),描述文本编码
    tags: set[str]               # 现有 tags 字段
    action_type: str             # read_only|reversible|destructive(tool)/ risk_level(skill)
    domain: str                  # 从 tags/前缀推断(lan/dc/wireless/wired/...)
    tool_set: set[str]           # skill 专用:它引用的 tool 集合(轨迹归属关键)

@dataclass
class Cluster:
    name: str                    # 可读簇名
    members: list[str]           # cap_id 列表
    centroid: list[float]        # 成员 embedding 均值
    dominant_tags: set[str]      # 簇的代表标签
```

## 4. 统一相似度(一个函数,所有人调)

```python
def capability_similarity(a: CapabilityVector, b: CapabilityVector) -> SimResult:
    """可解释的混合相似度。SimResult = (score: float, reasons: list[str])"""
    emb  = cosine(a.embedding, b.embedding)           # 语义
    tag  = jaccard(a.tags, b.tags)                    # 标签重叠
    act  = 1.0 if a.action_type == b.action_type else 0.0
    tool = jaccard(a.tool_set, b.tool_set) if a.tool_set and b.tool_set else None  # 轨迹/skill 关键
    # 权重在 config,默认 emb 0.5 / tag 0.25 / tool 0.2 / act 0.05
    score = weighted_sum(...)
    reasons = [f"embedding={emb:.2f}", f"tags={tag:.2f}", ...]  # 每项都列
    return SimResult(score, reasons)
```

**统一后**:evolver 的 jaccard、偏好的 embedding、P1 轨迹、P3 归属 —— 全调这个。
权重可调,理由可读。tool_set 的 jaccard 是**轨迹/skill 归属的核心信号**(两条
轨迹用了几乎相同的工具集 = 很可能同一类操作)。

## 5. 接口(P0-P4 依赖的稳定 API)

```python
class CapabilitySemanticIndex:
    def build(self, tool_defs, skill_defs) -> None
        # 启动时构建:为每个 tool/skill 算 CapabilityVector + 聚类成 Cluster

    # 归属:query / 轨迹 落在哪些簇/能力(top-k,带理由)
    def route(self, *, text=None, tool_set=None, top_k=3) -> list[RouteHit]
        # RouteHit = (cap_id_or_cluster, score, reasons)

    # 两个能力/轨迹的相似度
    def similarity(self, a_id, b_id) -> SimResult

    # 一条轨迹归属到哪个已有 skill(P3 用)
    def nearest_skill(self, tool_set, text) -> Optional[(skill_id, SimResult)]

    # 一批轨迹的聚类(P1 用:重复轨迹找公共模式)
    def cluster_trajectories(self, trajectories) -> list[TrajectoryCluster]

    # 诊断:导出整个空间(可视化/审计)
    def export_space(self) -> dict
```

实现 v1 可以很简单(embedding 来自现有 retriever 的 embedder,聚类用标签+domain
规则 + 可选 KMeans),**但调用方面向这个接口写**,后续把聚类升级成更聪明的算法
不破坏 P0-P3。这就是"先定接口"的价值。

## 6. P0-P3 怎么建在上面

- **P0(喂真实轨迹给 evolver)**:不依赖 CSI(纯数据搬运),先做。但 P0 产出的
  "真实轨迹(tool 序列)"正是 CSI `tool_set` / 轨迹聚类的输入 —— P0 为 CSI 备料。
- **P1(重复轨迹 → 流程化)**:用 `cluster_trajectories` 把 journal 里的历史轨迹按
  相似度聚类;某簇出现 ≥N 次 → 触发 evolver 生成 skill。**相似度来自 CSI,不自造**。
- **P3(追加诉求 → 定向 merge)**:用 `nearest_skill(当前会话 tool_set, 追加诉求文本)`
  判断追加内容归属哪个存量 skill → 定向 merge,而非 evolver 现在的泛 jaccard 找。
- **进化1/2、关注点#1**:skill 选择、偏好、描述可分性都可逐步迁到 CSI.route,
  消除四套相似度。

## 7. 诚实对比:显式 CSI vs 纯学习式 router(MoE)

应你要求,把两条路摆出来:

| 维度 | 显式 CSI(本方案) | 纯学习式 router(MoE gating) |
|------|------------------|---------------------------|
| 可解释 | ✅ 每次归属带 reasons | ❌ softmax 黑盒,问"为什么"答不了 |
| 可审计/可干预 | ✅ 簇/权重可读可改 | ❌ 改 = 重训 |
| 冷启动 | ✅ 标签+embedding 即可用 | ❌ 无标注数据时乱划分 |
| 训练成本 | ✅ 无 | ❌ 要标注 query→skill + 训练管线 |
| 和 P3 定向 merge 的可控目标 | ✅ 可手工指定归属 | ❌ 只给分布 |
| 与你的体系哲学 | ✅ 一致(结构性约束、可审计) | ❌ 冲突(黑盒、不可控) |
| 规模匹配 | ✅ 几十个可审计资产 | ❌ MoE 适合海量专家+算力分配 |
| 理论上限(海量数据时) | 🟡 受手工特征限制 | ✅ 能学到更细的划分 |

**唯一学习式占优的格子是"海量数据时的理论上限"** —— 但你现在没有那个数据,且要
可解释。结论:显式 CSI 现在全面更适合。**未来的折中**:若积累了大量"query→选对
skill"的真实样本(偏好学习正好在攒),可在 CSI.route 之上**叠一层可选的学习式
排序器**(对显式分数做 re-rank),保留显式作底、可解释为主,学习只做增量微调 ——
而不是用黑盒取代整个划分。这条路留口子,但不是现在。

## 8. 落地范围(评审通过后,但 CSI 实现晚于 P0)

| 阶段 | 内容 | 何时 |
|------|------|------|
| 设计(本文) | 接口 + 数据模型定稿 | 现在评审 |
| P0 | 真实轨迹喂 evolver(不依赖 CSI) | 先做 |
| CSI v1 | `skills/capability_index.py`:CapabilityVector + 相似度 + route + nearest_skill + cluster_trajectories(实现可简单,接口稳定) | P0 后 |
| P1 | 用 CSI.cluster_trajectories 做重复检测 → 触发生成 | 建在 CSI 上 |
| P3 | 用 CSI.nearest_skill 做定向 merge | 建在 CSI 上 |
| 迁移(可选) | evolver/偏好/选择逐步改调 CSI,删冗余相似度 | 最后 |

模块边界:`skills/capability_index.py` 只读元数据 + 复用 embedder,**不反向依赖**
evolver/loop/preference。P0-P3 单向依赖它。轻耦合。

## 9. 待你拍板

1. **簇划分依据**:domain×action 规则聚类(可读、可控)起步,够吗?还是要一开始就上
   KMeans 自动聚类(更"自动"但簇名不可读,要再贴标签)?我倾向规则起步。
2. **相似度权重**:emb 0.5 / tag 0.25 / tool_set 0.2 / act 0.05 这个默认合理吗?
   (tool_set 对轨迹归属很关键,要不要调高?)
3. **CSI v1 聚类**:v1 用"规则+标签"够用,还是必须现在就做 KMeans/层次聚类?
4. **学习式 re-rank 留口**:认同"显式作底 + 未来可选叠学习 re-rank"这个折中,而不是现在上黑盒?
5. **节奏确认**:设计(本文)→ P0 → CSI v1 → P1/P3,这个顺序对吗?

## 10. 一句话总结

给 tool/skill 一个**统一、可解释**的能力空间(CapabilityVector + 能力簇 + 一个带
理由的混合相似度 + route/nearest_skill/cluster_trajectories 接口),吸收 MoE 的
"簇+top-k 软路由"形态但**不引入学习式黑盒**,消除现在 jaccard/embedding 四套不
一致的相似度。P0 不依赖它先做(还为它备料真实轨迹),P1/P3 建在它的稳定接口上,
避免再造轮子。未来若攒够数据,可在显式分数上叠一层**可选**的学习 re-rank,但显式
作底、可解释为主的原则不变。

---

## 11. 真实数据评估反哺的三个修正(已纳入 v1 实现)

在真实 LAN/DC 元数据上实跑原型(见 CSI_EVALUATION_ON_REAL_DATA.md)后,确认三个修正:

1. **二级聚类**:domain 不能只看顶层前缀(否则 DC 12 tools 全挤一族)。用
   `(primary_domain, secondary_tag)` 二级键:DC 拆成 `dc/fabric`(evpn/vxlan/bgp/path)
   和 `dc/application`(app/acl/access)两族。
2. **相似度必须 embedding + tag 混合**:纯 tag 对中文/跨语言/跨域 skill 脆(实测
   中文 tag 的跨域 skill jaccard=0.09 漏判)。embedding 能跨语言抓语义,是刚需。
   默认权重 emb 0.45 / tag 0.25 / tool_set 0.25 / act 0.05(tool_set 调高,它推断
   skill→tool 最准)。
3. **tool_set 分层(ground truth 优先)**:① skill 声明的 allowed-tools ② P0 真实
   执行轨迹 ③ tag 推断兜底。前两层是 ground truth,不靠猜。
