# FeedbackMemRec 项目说明与实习面试指南

> 适用范围：当前仓库中的原始 MemRec、Corrected MemRec、FeedbackMemRec V1.0/V1.1/V1.2，以及已经完成的 V1.1 `2×100` 和 V1.2 `300-user` 实验。
>
> 这份文档服务于两件事：第一，帮助你真正理解项目；第二，帮助你在面试中如实、清晰地讲出“发现了什么问题、怎么定位、怎么解决、结果如何、还有什么不足”。

---

## 0. 先记住项目的核心结论

### 0.1 一句话版本

**原始 MemRec 会从 User-Item 图中选择协同邻居，让 LLM 生成偏好 facet、重排候选并更新长期文本记忆；FeedbackMemRec 给这条链增加了基于真实排序结果的证据信用、读取重排和写入门控，并在 V1.2 中让 Stage-W 写出的邻居长期记忆真正进入后续 Stage-R。**

### 0.2 这个项目最值得讲的，不是“指标大涨”

这个项目最后没有得到“Full V1.2 显著优于基线”的结论。它真正体现的是一套完整的研究与工程过程：

```text
阅读论文与源码
-> 发现 baseline 实现缺陷
-> 单独修正 baseline
-> 提出反馈信用闭环
-> 发现 V1.0 gate 根本不触发
-> 用日志定位稀疏性原因
-> 设计 V1.1 immediate gate
-> 发现写入变化没有稳定进入读取路径
-> 用 write-impact audit 找到结构性断点
-> 设计 V1.2 dynamic neighbor-memory read
-> 做四组消融、哈希校验和配对统计
-> 得到“机制成立，但收益未证实”的诚实结论
```

面试官真正容易认可的是：你没有只看一个好看的数字，而是能区分代码 bug、机制是否触发、状态是否真的传播、指标是否显著，以及实验失败意味着什么。

### 0.3 30 秒面试版

> 我在一个 LLM 记忆增强推荐系统 MemRec 上做了反馈闭环改造。原系统会从 User-Item 图选择协同邻居，用 LLM 合成偏好 facet、重排候选，并把交互写入用户、物品和邻居的长期文本记忆，但它不会根据真实排序结果判断协同证据是否有害。我用逐 facet leave-one-out 重排计算 NDCG 贡献，维护 target-user 条件化的 Evidence Credit，读取时调整邻居排序，写入时阻止当前负贡献的间接传播。之后我又通过日志审计发现 V1.1 写出的邻居记忆没有被 Stage-R 直接读取，于是 V1.2 增加了受严格 token budget 约束的动态邻居记忆读取。300 用户四组消融证明闭环和状态传播都真实发生，但 Full 没有显著提升，所以最终结论是工程机制验证成功，算法收益仍需改进。

---

## 1. 先建立正确的项目定位

### 1.1 它不是传统的参数训练项目

这个仓库没有微调 Qwen，也没有通过反向传播训练新的推荐模型。这里的“学习”主要发生在两个显式状态中：

1. `MemoryStorage` 中不断改写的 user/item 自然语言记忆；
2. `EvidenceCreditStore` 中不断更新的协同关系信用 `q`。

因此更准确的定位是：

> **有状态的 LLM Agent 推荐、协同记忆管理和反馈控制系统。**

不要在面试中说“我训练了 Qwen2.5-7B”或“我训练了一个图神经网络”。本项目调用的是冻结的本地 Qwen2.5-7B-Instruct，变化的是上下文、文本记忆和显式信用状态。

### 1.2 项目中的“图”和“记忆”到底是什么

| 概念 | 是否长期保存 | 实际含义 |
|---|---:|---|
| User-Item 图 | 由训练交互重建 | `user -> items`、`item -> users` 的二部图邻接表 |
| user 长期记忆 | 是 | 一段自然语言用户画像 |
| item 长期记忆 | 是 | 一段自然语言物品描述/受众总结 |
| Evidence Credit | 是 | 对某个目标用户而言，某条协同证据过去是否有用 |
| Stage-R facet | 否 | 当前推荐轮临时合成的偏好证据包 |
| support edge | 否 | LLM 的解释性输出，不是真实新增图边 |
| raw episode delta | 否 | 当前轮的反事实贡献，只用于本轮信用更新和 gate |

运行时图拓扑来自交互数据；`memory.jsonl` 保存的是文本记忆，不保存完整图拓扑。Facet 只写审计日志，不会变成永久 memory node。

### 1.3 user 和 item 是否有自己的长期文本记忆

有。`MemoryStorage` 分别保存：

```text
user_id -> user profile text
item_id -> item description text
```

Stage-W 会根据已经发生的交互生成新的完整文本并覆盖旧文本。当前实现不是 append-only 事件链，也没有 memory version、rollback 或多版本合并。

V1.1 以前需要特别注意：虽然邻居可能被 Stage-W 写入长期记忆，但 Stage-R 并不直接读取这些邻居的长期文本；V1.2 才补上这条读取路径。

---

## 2. 数据、任务和指标

### 2.1 当前数据集

主实验使用 `instructrec-books`。当前项目记录的规模为：

- 7,377 个用户；
- 120,925 个物品；
- 207,759 条交互；
- 平均每用户约 28.16 个物品；
- 平均每物品约 1.72 个用户。

它是一个很稀疏的图，这一点后来直接影响了 V1.0 historical credit 的积累速度。

### 2.2 时间切分

每个用户按 timestamp 排序后做 leave-one-out：

```text
train = 除最后两个交互之外的历史
valid = 倒数第二个交互
test  = 最后一个交互
```

至少有 3 条交互的用户才进入这套切分。

### 2.3 排序任务

每个评估用户有 10 个候选：

```text
1 个真实 target item + 9 个未交互负样本
```

候选会被打乱。正式消融中，各组必须使用相同的用户、target、候选 ID 和候选原始顺序，并通过 `candidate_manifest.jsonl` 的逐字节哈希证明一致。

这是 sampled reranking，不是从 12 万个物品中做全库召回，所以不能把这里的 Hit/NDCG 直接等价为线上全库指标。

### 2.4 指标怎么理解

在单正样本场景：

```text
Hit@K = target 是否进入前 K

NDCG@10 = 1 / log2(rank + 2)    当 zero-based rank < 10
          0                      否则
```

NDCG 比 Hit 更细。例如 target 从第 8 名升到第 2 名，Hit@10 都是 1，但 NDCG 会明显提高。

因为总候选只有 10 个，`Hit@10` 通常接近 1，区分度有限。面试中应重点看 Hit@1/3/5 和 NDCG@10。

---

## 3. 原始 MemRec 是怎么工作的

### 3.1 一次推荐的全链路

```text
目标用户 + 固定候选
        |
        v
User-Item Graph 找一跳 item 和二跳 user 邻居
        |
        v
Neighbor Pruner 对邻居打分并选 Top-K
        |
        v
Packer 在 token budget 内打包上下文
        |
        v
Stage-R：LLM 合成 preference facets
        |
        v
Stage-ReRank：LLM 为候选打分和排序
        |
        v
先得到当前预测结果
        |
        v
Stage-W：根据已经揭示的交互更新长期文本记忆
```

这三个阶段可记成：

- `R`：Retrieve/Reason，选择证据并合成偏好；
- `ReRank`：对候选逐项评分；
- `W`：Write，交互后写长期记忆。

### 3.2 User-Item 图与协同邻居

源码从训练交互构造：

```text
items_by_user[user_id]
users_by_item[item_id]
```

对目标用户来说：

- item neighbors 是其历史交互物品；
- user neighbors 是通过共同 item 临时推导出来的二跳用户；
- user-user 关系不是一条被永久持久化的新图边。

所以 Evidence Credit 的 `user_neighbor` 也不是修改原图得到的新 user-user edge，而是对一条“目标用户使用该二跳用户作为证据”的派生关系记分。

### 3.3 Neighbor Pruner

正式配置最多保留 `k=16` 个邻居，并满足 user/item 混合约束。基础分数来自图结构、recency、共同交互及规则特征。

需要如实说明：当前规则 pruner 的部分语义特征仍是近似或占位实现，不是训练好的 dense retriever。这是原项目和当前项目共同存在的限制。

### 3.4 V1.1 及以前的 Packer 实际给 Stage-R 什么

V1.1 的 Stage-R 输入主要包括：

- target user 自己的长期文本记忆；
- user 邻居最近 3 个 item 的标题；
- item 邻居的静态 metadata；
- 10 个候选标题。

关键边界：**这里没有直接加载 selected user/item neighbors 的长期文本记忆。**

这不是论文概念上的“协同记忆”必然如此，而是当前代码 packer 的实际实现。面试时要把论文设计意图和源码真实数据流分开。

### 3.5 Stage-R 的 facet 是什么

Stage-R 用 LLM 把杂乱上下文合成为若干偏好维度，例如：

```json
{
  "facet": "偏好反乌托邦题材的图像小说",
  "confidence": 0.82,
  "supporting_neighbors": ["User-4023", "Item-1984"]
}
```

Facet 是一次推理中的临时 evidence bundle。它用于 reranking 和 attribution，但不会进入长期图或长期 MemoryStorage。

### 3.6 Stage-ReRank

Reranker 接收：

- 用户 instruction；
- Stage-R facets；
- 固定候选；
- 候选 item 的当前长期记忆。

LLM 输出每个候选的 score 和 rationale，再按 score 排序。

### 3.7 Stage-W

在当前预测完成并揭示交互后，Stage-W 生成三类候选写入：

1. target user 的画像更新；
2. 当前真实交互 item 的描述更新；
3. selected user/item neighbors 的协同传播更新。

前两类是直接事实写入，第三类是间接协同传播。FeedbackMemRec 后来只 gate 第三类，绝不因为某条协同证据不好而丢掉真实交互事实。

### 3.8 原始 MemRec 的优点

- 把协同图结构和 LLM 语义推理结合；
- 用 facet 压缩长上下文；
- user/item 具有可读的长期文本状态；
- pruner 和 packer 显式控制上下文规模；
- Stage-R、ReRank、Stage-W 职责清晰；
- 不需要重新训练大模型。

### 3.9 原始 MemRec 的反馈断点

原始流程基本是：

```text
选择邻居 -> 合成 facet -> 推荐 -> 传播记忆
```

它没有利用真实排序结果追问：

- 哪个 facet 帮助了 target 排名？
- 哪个 facet 反而误导了 reranker？
- LLM 自报的 supporting neighbor 是否真的进入过 prompt？
- 某个邻居对用户 A 有用，是否也对用户 B 有用？
- 如果本轮证据有害，为什么还要向其传播新记忆？

这就是 FeedbackMemRec 的出发点。

---

## 4. 第一件事不是加算法，而是修 Corrected Baseline

### 4.1 发现了什么 bug

源码审计发现 warmup 调用 `rerank()` 时曾未开启 `return_details=True`。于是：

- 排名可以正常产生；
- 但 `details` 为空；
- Stage-W 拿不到 facets 和 `pruned_subgraph`；
- warmup 的协同写入上下文不完整。

### 4.2 为什么不能把这个修复算成新算法收益

如果直接拿修复后的 FeedbackMemRec 对比有 bug 的上游实现，那么任何提升都可能只是因为原流程终于正确运行。

所以我们建立了 `Corrected MemRec`：

```text
原 MemRec
+ warmup return_details correctness fix
+ 必要的日志/公平性支持
- attribution
- Evidence Credit
- Read Credit
- Write Gate
- V1.2 Neighbor Memory Read
```

所有后续实验都在 Corrected 基线上做增量消融。

### 4.3 面试表达

> 我没有先写新功能，而是先验证 baseline 的数据流。发现 warmup 虽然有排名，但 Stage-W 所需的细节没有返回。我把这项修复放入所有实验共享的 Corrected baseline，不把 correctness repair 包装成算法收益。这样后续 Full-Corrected 才具有解释意义。

---

## 5. V1.0：先搭出最小反馈闭环

用户主要关心 V1.1 和 V1.2，但 V1.0 必须理解，因为 V1.1 是从一次真实失败中演化出来的。

### 5.1 独立的 EvidenceCreditStore

信用键为：

```text
(target_user_id, evidence_type, evidence_id)
```

其中 `evidence_type` 是 `user_neighbor` 或 `item_neighbor`。

例如：

```text
(2057, user_neighbor, 4023)
```

表示“User-4023 作为 User-2057 的协同证据时的历史效用”，而不是 User-4023 的全局人格或全局质量。

这叫 target-conditional credit。优点是保留个性化；代价是 relation 更稀疏。

### 5.2 为什么不能相信 LLM 自报的来源

LLM 可以输出 `supporting_neighbors`，但它可能引用未真正进入 prompt 的 ID。系统因此保存 packer 的真实 provenance，并执行：

```text
validated_support = claimed_support ∩ actually_packed_relations
```

只有真实打包过的 relation 才能获得 credit。LLM 生成的 support edge 只用于解释日志，不能当成归因真值。

### 5.3 逐 facet leave-one-out 归因

令完整 facet 集为 `F`。对于每个 facet `f`：

```text
delta_f = R(F) - R(F \ {f})
```

其中 `R` 是真实 target rank 对应的 NDCG@10。

反事实重排必须保持以下内容一致：

- 候选 ID 和顺序；
- target item；
- instruction；
- item memory 快照；
- 模型和解码参数；
- 其他 facets。

它只删除一个 facet，不重新生成 Stage-R。

解释：

- `delta_f > 0`：facet 对 target 排名有帮助；
- `delta_f < 0`：facet 有负贡献；
- `delta_f = 0`：当前排序对它不敏感。

### 5.4 从 facet delta 到 relation delta

如果一个 facet 有多个已验证来源，V1.0/V1.1 使用均分近似：

```text
share(f, relation) = delta_f / number_of_valid_sources
raw_episode_delta(relation) = Σ share(f, relation)
```

再确定性更新历史信用：

```text
q_new = clip(
    q_old + 0.2 * clip(raw_episode_delta, -0.5, 0.5),
    -1,
    1
)
```

数值更新不交给 LLM，且同一 episode/relation 通过 episode ID 保证幂等。

这不是严格因果归因。更准确的说法是：

> 固定其他 reranker 输入后的 conditional counterfactual attribution，再加近似 source-level credit assignment。

### 5.5 Read Credit

原 pruner score 乘以信用因子：

```text
multiplier = clip(1 + 0.5 * q, 0.5, 1.5)
adjusted_score = original_score * multiplier
```

- `q=0` 时严格中性；
- 正信用提高优先级；
- 负信用降低优先级；
- 不永久删除邻居；
- 原有 user/item mixing constraint 仍保留。

### 5.6 V1.0 historical Write Gate

```text
if num_updates >= 2 and historical_q < -0.3:
    reject collaborative propagation
else:
    accept exploration
```

无论如何：

```text
target-user direct write  -> always commit
current-item direct write -> always commit
```

### 5.7 V1.0 为什么没有达到预期

20-user Full 有 92 次协同传播，但 0 次 reject：

- 24 条 relation 当时 `num_updates=0`；
- 68 条 relation 当时 `num_updates=1`；
- 没有 relation 同时达到两次更新和 `q<-0.3`。

根因不是 if 分支写错，而是：

```text
target-user 条件化 key + 稀疏 Books 图 + 仅一轮 warmup
-> 同一 relation 很少重复
-> historical gate 长期停留在探索态
```

这是这个项目第一次重要的“设计在纸面上合理，但运行中根本不生效”。

---

## 6. V1.1：用当前轮负贡献控制当前轮传播

### 6.1 V1.1 为什么这样改

我们没有直接把历史阈值从 `-0.3` 改成更容易触发的值，也没有反复筛用户。先审计 V1.0 的 92 次传播，发现：

- 37 次能匹配本轮 validated attribution；
- 其中正贡献 9 次；
- 负贡献 8 次；
- 零贡献 20 次；
- 其余没有本轮 attribution。

这说明历史证据不足，但本轮已经存在明确的负信号。

### 6.2 V1.1 的唯一算法变化

```python
if has_current_attribution and raw_episode_delta < 0:
    reject("negative_current_episode_contribution")
elif num_updates >= 2 and historical_q < -0.3:
    reject("negative_historical_credit")
else:
    accept("neutral_or_exploration")
```

几个关键约束：

- 使用 learning rate 和 q clip 之前的原始聚合 delta；
- delta 为 0 时不拒绝；
- 没有 attribution 时继续探索；
- 当前负贡献只拒绝本轮传播，不永久封禁邻居；
- 历史阈值和更新规则完全不变；
- direct writes 永远旁路 gate；
- 关闭 `use_current_episode_delta` 可恢复 V1.0。

### 6.3 为什么看 raw delta 而不是更新后的 q

更新后的 q 混合了旧历史、学习率和 clip。对于新 relation，即使当前证据明确为负，q 的绝对值也可能很小。

V1.1 要回答的是：

> “这条协同证据在当前推荐中刚刚造成负贡献，还要不要把当前信息沿它传播？”

所以即时 gate 看当前 raw delta；historical gate 则继续看长期 q。两者处理的是不同时间尺度。

### 6.4 严格时序与防泄漏

每个 warmup episode 严格串行：

```text
predict
-> 记录当前 target rank
-> reveal target/evaluate
-> leave-one-out attribution
-> update credit
-> Stage-W proposals
-> direct-write bypass / collaborative gate
-> commit
```

当前 target 必须在预测完成之后才能用于 credit 和 write，不能先写再预测当前样本。

测试期 `credit_updates_during_test=0`，不使用当前测试标签更新 credit。要注意，测试用户预测完成后仍可能按统一协议写文本 memory，因此整体属于严格 predict-before-write 的 prequential 风格，而不是所有用户完全共享一个静止快照。

### 6.5 V1.1 两个正式 100-user block

| Block | Corrected NDCG@10 | Read | Full V1.1 | Full-Read | Full-Corrected |
|---:|---:|---:|---:|---:|---:|
| 1 / seed 42 | 0.689771 | 0.696837 | 0.709473 | +0.012636 | +0.019702 |
| 2 / seed 43 | 0.700578 | 0.705433 | 0.705433 | +0.000000 | +0.004856 |

两 block 等权平均约为：

```text
Corrected  0.695174
Read       0.701135
Full       0.707453
```

但是不能把这个均值包装成稳定显著提升：

- Full-Corrected 在两个 block 都为正；
- Full-Read 只在 Block 1 为正；
- Block 2 的 Full 与 Read 完全相同；
- Block 1 缺少可恢复的逐用户排名，无法进行完整 200-user 配对推断；
- Block 2 的 Full-Corrected 置信区间跨 0。

### 6.6 V1.1 gate 的机制证据

- Block 1：45/448 次传播被拒绝，direct writes 348/348 保留；
- Block 2：47/438 次传播被拒绝，direct writes 350/350 保留；
- 拒绝原因均来自当前负 episode contribution；
- test credit updates 均为 0；
- 同一 block 三组候选 manifest 一致。

这能证明 gate 真实运行并改变了 memory state，但仅凭它不能证明这些状态变化一定提高推荐质量。

---

## 7. 从 V1.1 到 V1.2：真正关键的第二次问题定位

### 7.1 为什么 Block 1 有改善，Block 2 Full 和 Read 却完全相同

V1.1 的 write-impact audit 追踪了：

```text
gate reject
-> 哪条 user/item memory 不同
-> 是否进入后续 Stage-R 或 reranker
-> test context 是否不同
-> target rank 是否不同
```

审计发现，V1.1 Stage-R 读取的是：

- target user memory；
- user 邻居最近 item 标题；
- item 邻居静态 metadata。

它不直接读取邻居的长期文本 memory。因此 Write Gate 即使拒绝了某个 neighbor memory update，该差异也只有在这些实体后来恰好成为 target user 或 candidate item 时，才可能间接进入模型。

也就是说：

```text
V1.1 写侧 intervention 很强
但读侧对这类 memory difference 的可达性很弱且偶然
```

这解释了为什么“gate 触发了”不等于“后续 prompt 一定变化”，也解释了 V1.1 Full-Read 跨 block 不稳定。

### 7.2 V1.2 的研究问题

V1.2 只问一个很窄的问题：

> 能否让 later Stage-R 直接消费 earlier Stage-W 已写入的 selected-neighbor 长期文本，从而让 write intervention 形成可验证的闭环？

V1.2 不是重构整个 MemoryStorage，也没有新增版本系统、向量数据库、LLM 摘要器或永久 user-user graph。

### 7.3 V1.2 的读取路径

```text
UserItemGraph
-> 原有 pruner（可能含 Read Credit）
-> 先执行并冻结原 V1.1 packed neighbor ID/顺序/static snippet
-> 只为这些已经选中的邻居读取 MemoryStorage 快照
-> 最多 4 个 user memory + 4 个 item memory
-> Qwen tokenizer 确定性 head-tail 截断
-> 追加到原 neighbor snippet
-> Stage-R
```

它不会为填满配额而遍历额外邻居，也不会改变原 packed neighbor 的数量、ID、顺序和 static snippets。

### 7.4 为什么预算是 64 / 512 / 1800

V1.2 固定：

```text
单条 neighbor memory <= 64 tokens
本轮 dynamic neighbor memories 总计 <= 512 tokens
完整 Stage-R context <= 1800 tokens
```

可用动态预算：

```text
remaining = max(1800 - baseline_context_tokens, 0)
dynamic_budget = min(512, remaining)
```

长记忆使用 tokenizer-aware head-tail 截断，保留开头和结尾，并把省略标记本身计入 token。

这样设计的目的不是追求最优摘要，而是确保 V1.2 的唯一主要变化是“读取已有长期记忆”，不会因为新增 LLM 压缩器、重新选邻居或扩大上下文而混入更多变量。

### 7.5 read-after-write 如何证明不是“代码开关亮了”

每次 Stage-W 写入和未来 Stage-R 读取都记录 memory content hash。只有：

```text
earlier written full-memory hash == later consumed full-memory hash
```

才计为 read-after-write match。

这样可以区分：

- memory 文件里确实有内容；
- 某个 ID 被声称读取；
- 之前写入的那份真实文本确实进入了后续 prompt。

### 7.6 V1.2 没有改什么

V1.2 没有修改：

- MemoryStorage schema；
- 原 User-Item 图；
- Evidence Credit 更新公式；
- leave-one-facet-out attribution；
- Read Credit multiplier；
- V1.1 immediate/historical gate；
- direct-write bypass；
- 候选数、facet 数和模型参数；
- 总 Stage-R `tau_tokens=1800`。

这保证 V1.2 能回答一个独立问题，而不是把 V2 的大型 memory architecture 改造一起混进来。

---

## 8. V1.2 四组消融分别回答什么

| 组别 | 动态邻居记忆 | Attribution/Credit | Read Credit | Write Gate | 回答的问题 |
|---|---:|---:|---:|---:|---|
| Corrected | 关 | 关 | 关 | 关 | 修正后的原系统表现 |
| Memory Only | 开 | 关 | 关 | 关 | 单独读取动态 neighbor memory 是否有价值 |
| Closed-loop Read | 开 | 开 | 开 | 关 | 信用学习和 credit-aware read 的增量 |
| Full V1.2 | 开 | 开 | 开 | 开 | Write Gate 在相同读取能力上的增量 |

最重要的比较链是：

```text
Memory Only - Corrected        = Dynamic Neighbor Memory Read
Read - Memory Only            = Attribution + Evidence/Read Credit
Full V1.2 - Read              = Write Gate
```

不能只比较 Full 和 Corrected，然后把所有差异都归因给 Write Gate。

四组均从干净、互不共享的 MemoryStorage 和空 EvidenceCreditStore 开始。300-user 用户清单、顺序和候选在运行前锁定；状态在单组内部严格串行演化。

---

## 9. V1.2 300-user 正式结果怎么读

### 9.1 all-300 主口径

少量 LLM 排名输出不完整，预先约定将失败用户指标记为 0：

| Variant | 成功排名 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corrected | 296/300 | 0.4167 | 0.6500 | 0.7600 | 0.9867 | 0.669025 | 3.16 h |
| Memory Only | 295/300 | 0.4300 | 0.6567 | 0.7533 | 0.9833 | 0.671417 | 3.03 h |
| Read | 295/300 | 0.4300 | 0.6600 | 0.7667 | 0.9833 | **0.674103** | 7.33 h |
| Full V1.2 | 295/300 | 0.4233 | 0.6433 | 0.7567 | 0.9833 | 0.668487 | 7.37 h |

四组中数值最好的 NDCG 是 Read，但“数值最高”不等于“已证明更优”。

### 9.2 三个主要增量

| 比较 | NDCG delta | bootstrap 95% CI | sign-flip p |
|---|---:|---:|---:|
| Memory Only - Corrected | +0.002392 | [-0.020216, +0.025441] | 0.8354 |
| Read - Memory Only | +0.002686 | [-0.014121, +0.019601] | 0.7617 |
| Full - Read | **-0.005616** | [-0.029963, +0.018461] | 0.6580 |

三项置信区间都跨 0，不能拒绝“真实均值差为 0”。因此正确结论是：

- Dynamic Neighbor Memory 方向上略高于 Corrected，但未证实可靠增益；
- credit-aware read 在本次样本上又略高一点，但仍不显著；
- Write Gate 相对 Read 方向为负，也不显著；
- 不能声称 Full V1.2 提升了推荐效果。

### 9.3 common-success 敏感性分析

四组共同成功用户为 291/300：

| Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Corrected | 0.4227 | 0.6632 | 0.7732 | 1.0000 | 0.679396 |
| Memory Only | 0.4364 | 0.6667 | 0.7663 | 1.0000 | 0.682058 |
| Read | 0.4364 | 0.6701 | 0.7801 | 1.0000 | **0.684876** |
| Full | 0.4296 | 0.6564 | 0.7732 | 1.0000 | 0.680109 |

去掉不同组的失败用户后，结论方向没有反转：Read 仍最高，Full 仍低于 Read。

### 9.4 机制到底有没有工作

有，而且有强审计证据：

- candidate manifests 四组逐字节一致；
- 所有组 `credit_updates_during_test=0`；
- Full 有 1,341 次协同传播候选；
- 接受 1,196 次，拒绝 145 次，拒绝率 10.81%；
- 被拒绝 item neighbor 102 次、user neighbor 43 次；
- 145 次 immediate reject 全部满足 `raw_episode_delta<0`；
- Full 的 1,052 次 direct writes 全部保留；
- Memory Only/Read/Full 动态记忆分别 packed 3,121/3,099/3,099 条，drop 为 0；
- read-after-write match 分别为 1,221/1,204/1,086，hash mismatch 全为 0；
- Full context token 范围为 778/平均 1302.8/最大 1586，未超过 1800。

所以这次结果不是“功能没运行”。更准确的判断是：

> **闭环在工程上真实改变了长期状态和后续 prompt，但当前的负贡献 gate 规则没有把这些改变转化为可靠的排序收益。**

### 9.5 Full 和 Read 的状态差异有多大

Full 相对 Read：

- warmup 后不同 memory 共 705 条；
- 其中 user memory 238 条、item memory 467 条；
- test packed context 不同 241/300；
- packed neighbor sequence 不同 62/300；
- dynamic neighbor sequence 不同 97/300；
- target rank 不同 111/300。

这组数据非常重要。它排除了“Full 和 Read 最终一样，是因为 gate 没造成可观察差异”的解释。Gate 的影响确实传播到了测试上下文和排名，只是平均效果没有变好。

### 9.6 为什么 Full 可能不如 Read

目前不能从一次实验确定唯一原因，但有几种合理解释：

1. `delta<0` 是当前候选集和当前 facets 下的局部信号，不代表这条新 memory 对未来用户一定有害；
2. 当前 facet attribution 可能受 facet 交互和 LLM 输出波动影响；
3. 一个 relation 支持多个 facets 时采用均分，source credit 较粗糙；
4. gate 拒绝整段 neighbor memory rewrite，干预粒度可能大于证据粒度；
5. target-conditional relation 稀疏，历史信用仍未形成稳定判断；
6. 长期状态有顺序效应，早期一次拒绝可改变后续整条 prompt 轨迹；
7. 300 用户和单一序列仍不足以稳定估计小效应。

这些是基于机制的待验证假设，不能假装成已经证实的根因。

---

## 10. 七个适合面试的“遇到问题—怎么解决”故事

以下故事都可以按 STAR 结构回答：Situation、Task、Action、Result。不要背成流水账，重点讲你的判断依据。

### 故事一：Baseline 自己没有完整执行

**Situation**

准备实现 feedback credit 时，我先跟踪 warmup 的 `rerank -> details -> Stage-W` 数据流，发现排名存在，但 Stage-W 的 facets 和 pruned subgraph 为空。

**Task**

必须先判断这是新算法问题，还是原实现调用契约没有满足。

**Action**

我定位到 warmup 调用 `rerank()` 时没有 `return_details=True`。修复后，我没有直接把新结果算成 FeedbackMemRec 收益，而是建立所有实验共享的 Corrected baseline，并增加 details/Stage-W 路径验证。

**Result**

基线的协同写入上下文恢复；后续 Corrected/Read/Full 比较只包含明确的增量功能。

**面试亮点**

> 先验证 baseline correctness，再谈算法增益；否则实验结论没有内部效度。

### 故事二：LLM 声称的 supporting neighbor 不可信

**Situation**

Stage-R 会生成 `supporting_neighbors`，但 LLM 有可能引用没有真正进入当前 context 的邻居。

**Task**

如果直接给这些 ID 分 credit，系统会把奖励更新到不存在的证据路径上。

**Action**

我让 packer 返回实际打包的 relation provenance，然后只保留：

```text
LLM claimed support ∩ actual packed support
```

support edge 只记日志，不参与信用真值。

**Result**

V1.2 Read/Full 中约 5% 的 claimed relations 被识别为无效，没有污染 EvidenceCreditStore。

**面试亮点**

> LLM 输出用于提出解释候选，程序日志和真实数据流才是可验证事实。

### 故事三：V1.0 gate 写了，但一次都不触发

**Situation**

V1.0 20-user 有 92 次传播，却 0 reject。Full 和 Read 的结果相同。

**Task**

需要区分是代码 bug、阈值问题，还是数据分布与状态设计不匹配。

**Action**

我审计每次传播对应的 relation key、`num_updates`、historical q 和当轮 raw delta，发现绝大多数 target-conditional relation 只出现 0～1 次，历史 gate 永远处于探索状态；与此同时已经有 8 次本轮负贡献。

我没有在开发集上搜索阈值，而是增加语义独立的 V1.1 immediate gate：当前 validated raw delta 为负，只拒绝本轮间接传播。

**Result**

V1.1 开发 20-user 出现 8 次真实 reject；两个正式 block 分别出现 45 和 47 次 reject，direct writes 全保留。

**面试亮点**

> 功能存在不等于机制在真实分布上可达；要用运行日志验证触发条件覆盖率。

### 故事四：Write Gate 改了状态，但未必能被后续模型看到

**Situation**

V1.1 Block 1 Full-Read 为正，Block 2 却完全相同。虽然 gate 两个 block 都触发了，但效果不稳定。

**Task**

需要追踪一次 reject 是否真的影响后续推理，而不是只看 reject 数量。

**Action**

我做 write-impact audit，从 rejected neighbor 出发，对比 Read/Full warmup memory、后续 Stage-R context、candidate memory、selected-neighbor sequence 和 target rank。

审计发现 V1.1 Stage-R 只读邻居的静态/行为 snippet，不直接读其长期文本 memory。被 gate 改变的 memory 只有在实体后来成为 target 或 candidate 时才偶然进入推理。

**Result**

我将问题从“gate 阈值不够好”重新定义成“写入 intervention 的读取可达性不足”，并据此设计 V1.2 dynamic neighbor-memory read。

**面试亮点**

> 对有状态系统，必须验证 `write -> future read -> decision` 的完整因果路径，而不只是验证数据库发生了变化。

### 故事五：增加动态文本后如何防止实验变量失控

**Situation**

V1.2 要把 neighbor memory 加入 Stage-R，但直接拼接会改变邻居选择、挤掉静态证据或导致 context overflow。

**Task**

需要让“读取动态 memory”成为唯一主要变量。

**Action**

我先执行 V1.1 pack 并冻结邻居 ID、顺序和 static snippets，只为已经选中的邻居附加 memory；限制最多 4 user + 4 item，单条 64 tokens、动态总量 512、完整 context 1800，并采用 tokenizer-aware head-tail 截断和运行时断言。

**Result**

正式实验动态 memory drop 为 0，完整 context 最大 1598，原邻居集合和 static snippets 校验通过。

**面试亮点**

> 新功能不仅要能跑，还要控制它对实验其他变量的影响。

### 故事六：如何证明读到的就是之前写入的内容

**Situation**

日志显示“读取了 User-X”仍不足以证明读取的是 earlier Stage-W 写出的那一版文本。

**Task**

需要建立跨 episode 的数据 lineage。

**Action**

每次写入记录 content hash；未来打包 memory 时记录读取 hash，并通过实体 ID、写入 step 和 hash 做 read-after-write 匹配。

**Result**

Memory Only/Read/Full 分别得到 1,221/1,204/1,086 次真实 read-after-write match，mismatch 全为 0。

**面试亮点**

> 用可验证的内容哈希证明状态传播，而不是依赖控制台打印或主观检查 prompt。

### 故事七：正式实验没有提升，怎么处理

**Situation**

V1.2 300-user 中，Full-Read NDCG 为 `-0.005616`，而且置信区间跨 0。

**Task**

需要判断是功能没工作、实验不公平、运行失败，还是算法假设没有得到支持。

**Action**

我检查 candidate manifest、配置/用户哈希、test credit freeze、direct-write preservation、context budget、read-after-write、gate reason 和状态差异；同时使用预先规定的 all-300 failure=0 及 common-success 两种统计口径，并做 10,000 次 paired bootstrap 和 sign-flip test。

**Result**

机制和公平性检查均成立，Full 确实改变了 705 条 warmup memory、241 个 test context 和 111 个 target rank，但平均指标没有改善。因此我保留负结果，没有重试、换用户或调规则。

**面试亮点**

> 负结果不是项目失败。它帮助排除了“只要阻止当前负证据传播就能改善长期推荐”的过强假设，并指出未来需要更稳健的长期价值估计和更细粒度写入控制。

---

## 11. 高频技术面试问题与参考回答

### Q1：你解决的核心问题是什么？

> 原始 MemRec 可以生成和传播协同记忆，但没有用实际排序结果评估协同证据的价值。我增加了 facet 级反事实归因、target-conditional relation credit、credit-aware read 和 indirect-write gate；V1.2 再让被写入的邻居文本真正进入后续 Stage-R，从而形成可审计的读写闭环。

### Q2：为什么 facet 不长期保存？

> Facet 是某次推荐基于当前用户、候选和邻居临时合成的 evidence bundle。长期保存会把候选条件下的推理结论误当成稳定事实。我们只保存 attribution event，并把 delta 回传给长期 relation credit；长期状态仍是 user/item memory 和 EvidenceCreditStore。

### Q3：为什么不用 LLM 自己的 confidence 做 credit？

> Confidence 是模型自评，不等于对真实 target ranking 的贡献。我用固定其他输入的 leave-one-out 重排，看删掉 facet 后 NDCG 是否变化。这样 credit 至少和任务结果绑定，而不是只相信语言模型的自我解释。

### Q4：为什么不直接做完整 Shapley Value？

> 7 个 facets 的完整 Shapley 需要评估大量子集，LLM 调用成本会指数增长。Leave-one-out 每个 episode 只增加 7 次 rerank，已经能发现单个 facet 的边际负贡献。代价是不能完整建模 facet 交互，所以我把它称为近似归因，不宣称严格因果。

### Q5：为什么 relation credit 要带 target user？

> 同一个邻居对不同目标用户的作用可能相反。Target-conditional key 保留个性化，但也导致稀疏；V1.0 historical gate 不触发正是这个 trade-off 的实际表现。未来可采用 global/cluster/user 三层信用和贝叶斯平滑。

### Q6：为什么 direct writes 永远保留？

> 当前 target user 点击当前 item 是直接观测事实；邻居传播只是间接推断。Gate 判断的是协同通路是否可信，不能因为某个 facet 没有帮助排名，就丢掉真实发生的交互。

### Q7：V1.1 和 V1.2 的本质区别是什么？

> V1.1 改的是信用和写入控制：当前负 relation 不传播。但 Stage-R 仍不直接读邻居长期文本。V1.2 不改 gate，而是补读取路径，让 selected neighbor 的长期文本在固定预算内进入后续 Stage-R，使写入差异可被后续推荐消费。

### Q8：Memory Only 为什么必要？

> 如果只比较 Corrected、Read、Full，就无法知道 V1.2 的变化来自动态文本本身，还是 credit。Memory Only 只开 neighbor-memory read，因此 `Memory Only-Corrected` 隔离文本读取，`Read-Memory Only` 隔离信用机制，`Full-Read` 隔离 gate。

### Q9：如何防止标签泄漏？

> 每个 episode 均为 predict-before-write。Warmup 中在预测完成后才揭示 target 并做 attribution；test 冻结 credit，`credit_updates_during_test=0`。候选和状态更新均按锁定用户顺序串行，不允许用户级并行更新。

### Q10：为什么不能并行处理 300 个用户？

> MemoryStorage 和 EvidenceCreditStore 是跨用户演化的状态。用户 B 可能读取用户 A 刚写入的 memory，并行会导致读写顺序和结果不确定。实验之间可以用两张 GPU 并行，但每个实验内部必须严格串行。

### Q11：V1.2 工程闭环如何验证？

> 不只看开关，而是看三类证据：动态 memory 被实际 packed；earlier write 与 later read 的内容 hash 匹配；Full/Read 的 memory、test context 和 target rank 发生差异。正式实验有上千次 read-after-write match 且 mismatch 为 0。

### Q12：为什么 V1.2 Full 没提升？

> Gate 使用的是当前候选条件下的局部负贡献，它未必代表被生成 memory 的长期价值；同时 attribution 和 source sharing 都是近似的。实验表明 gate 的干预真实到达了后续 prompt，但平均 NDCG 没改善，所以问题在决策规则的泛化，而不是数据路径没接通。

### Q13：你能说 Read 比 Corrected 更好吗？

> 数值上 Read 的 NDCG 是 0.674103，Corrected 是 0.669025，绝对高 0.005078；但正式预设的增量比较置信区间都跨 0，因此只能说方向更高，不能说已获得统计可靠提升。

### Q14：为什么报告失败用户计 0？

> 这是保守且预先锁定的 all-user 口径，避免只删除某组失败用户造成选择偏差。同时再报告 pairwise/all-four common-success 敏感性分析。两种口径方向一致，所以结论不依赖失败处理方式。

### Q15：LLM JSON 异常怎么处理？

> 正式实验不按结果重试、不改 max tokens、不换用户。我们记录 parse fallback 和不完整 ranking；V1.2 各组有 4～5 个孤立失败，没有系统性 transport 或 context overflow。生产系统可做 constrained decoding 和字段级安全重试，但正式实验中临时改变失败策略会破坏可比性。

### Q16：为什么 Full 比 Corrected 还慢很多？

> 主要成本来自 attribution。V1.2 Read/Full 各有 298 个成功 attribution episodes、2,086 次 counterfactual rerank。Credit 查表和 gate 本身很便宜，重复 LLM 推理才是瓶颈。

### Q17：这个 credit 更新是强化学习吗？

> 不是。它是确定性的有界增量状态更新，没有训练 policy/value network，也没有反向传播或 off-policy 学习。它借用了反馈学习思想，但更接近显式 online utility tracking。

### Q18：为什么不永久删除负邻居？

> 当前 delta 是局部且有噪声的。永久删除会把一次错误归因放大为不可恢复决策。Read Credit 只做有界降权，immediate gate 只拒绝当前传播，保留后续探索机会。

### Q19：为什么不用向量数据库检索所有 memory？

> V1.2 的目标是验证写入记忆是否被后续读取，因此只在原 pruner 已选择的邻居内追加文本，避免同时改变召回集合。向量检索可能有价值，但会引入 embedding、index 和新邻居来源，属于下一阶段独立变量。

### Q20：这个系统最主要的风险是什么？

> 一是错误 memory 被反复传播形成 memory poisoning；二是局部 attribution 被误当成长期价值；三是有状态顺序效应；四是结构化 LLM 输出不稳定；五是 counterfactual 成本高。当前 gate、provenance、hash trace、状态隔离和日志只能缓解其中一部分。

### Q21：如果继续做 V1.3，你会怎么改？

> 我会先保持 V1.2 读取路径不变，单独研究 gate 的长期价值估计：例如把“当前 facet 排名贡献”和“建议写入的 memory 内容质量”分开，使用多次观测、置信度下界或 delayed utility 决定传播；同时加入 memory version/provenance，使错误写入可回滚。实验上做多个独立用户序列和完全禁 test 写入的消融。

### Q22：这个项目有什么业务价值？

> 它针对长周期、可解释的个性化系统：系统不仅保存画像，还能审计某条协同证据从哪里来、何时被读取、如何影响决策以及为什么被拒绝。即使当前 gate 没提升指标，provenance、状态隔离、回退和公平性验证仍是构建可靠 LLM Agent 的通用能力。

---

## 12. 面试中必须主动承认的局限

### 12.1 Attribution 不是严格因果

- facets 之间可能交互；
- leave-one-out 不满足可加性；
- 多来源均分是近似；
- 结果依赖当前 10-item candidate set；
- LLM 输出本身存在非业务噪声。

### 12.2 target-conditional credit 数据效率低

个性化强，但同一 relation 很少重复，historical gate 在当前实验中从未成为主要拒绝来源。

### 12.3 MemoryStorage 没有版本和回滚

当前是完整文本覆盖，不支持：

- memory version；
- patch-level provenance；
- rollback；
- 冲突合并；
- 事实一致性验证。

### 12.4 测试是 prequential，而非完全静态

Credit 在 test 冻结，但每个用户预测之后的文本写入仍可能影响后续用户。三组使用同一顺序和协议，适合比较；若要得到纯静态离线结论，还需要 test 完全禁写消融。

### 12.5 只有一个领域和一条 300-user 顺序

V1.2 是 Books、一个锁定用户序列、一个模型 seed。三个连续 100-user segment 只能作为状态演化诊断，不能伪装成三个独立 seed。

### 12.6 计算成本高

7 个 facets 对应每个成功 warmup episode 7 次额外 rerank。V1.2 Read/Full 约 7.3 小时，Corrected/Memory Only 约 3.1 小时。

### 12.7 正式结果没有证明提升

这是必须直说的。可以说“Read 数值方向略高”“机制已验证”“Full gate 未带来收益”；不能说“V1.2 显著提升推荐性能”。

---

## 13. 简历应该怎么写

### 13.1 推荐项目名称

```text
FeedbackMemRec：反馈驱动的 LLM 协同记忆推荐系统
```

### 13.2 三条稳妥的简历 bullet

- 基于 MemRec 构建可审计的协同记忆反馈闭环：对 LLM preference facets 执行 leave-one-out reranking，以目标物品 NDCG 变化更新 target-conditional Evidence Credit，并用于邻居读取重排与间接写入门控。
- 审计并修复 warmup 的 Stage-W 上下文缺失问题；通过 packed-relation provenance、predict-before-write、test credit freeze、状态隔离及 candidate/config SHA-256 锁定，建立可回退的 Corrected/Memory-Only/Read/Full 消融流水线。
- 设计 V1.2 Dynamic Neighbor Memory Read，在不改变原邻居 ID/顺序和 1800-token 总预算的前提下接通 Stage-W 到后续 Stage-R；300-user 实验验证 1,086 次 Full read-after-write hash match、145 次负传播拦截和 100% direct-write 保留，并通过配对统计如实确认排序增益尚不显著。

### 13.3 如果岗位更偏算法

强调：

- counterfactual attribution；
- target-conditional credit；
- 稀疏反馈的 sample-efficiency 问题；
- ablation 和 paired statistics；
- 负结果带来的假设修正。

### 13.4 如果岗位更偏 ML/LLM 系统

强调：

- 本地 Qwen + vLLM 服务；
- token budget 和 tokenizer-aware truncation；
- JSON schema/fallback；
- 有状态串行执行与双 GPU 实验级并行；
- provenance、hash lineage、幂等和可回退配置；
- 长时任务监控和独立输出目录。

### 13.5 不建议写的内容

- “性能显著提升”；
- “SOTA”；
- “训练/微调 Qwen”；
- “构建了永久动态 memory graph”；
- “实现严格因果归因”；
- “300 用户全部成功”；
- “Full V1.2 优于 Read”。

---

## 14. 三种长度的项目介绍话术

### 14.1 30 秒版

> 我在 MemRec 上做了一个反馈驱动的协同记忆闭环。原系统会用图邻居生成偏好并传播文本记忆，但不会判断证据是否有害。我用逐 facet 反事实排名变化学习 target-user 条件化信用，读时重排邻居，写时阻止负贡献传播，并在 V1.2 中让邻居长期文本真正进入后续 Stage-R。正式 300-user 消融证明状态和 prompt 都被真实改变，但 Full 没有显著提升，所以我把结论定位为机制验证和负结果分析，而不是夸大成性能突破。

### 14.2 2 分钟版

> 原始 MemRec 有三个阶段。首先从 User-Item 图找 item 和二跳 user 邻居，在 token budget 中打包；Stage-R 用 LLM 合成 7 个 preference facets；ReRank 根据 facets、instruction 和候选 item memory 排序。交互发生后，Stage-W 再更新目标用户、当前物品和部分邻居的文本记忆。
>
> 我发现它缺少 outcome-to-evidence 的反馈：选中的邻居无论有用还是有害，都可能继续被读和被写。于是我增加 EvidenceCreditStore，key 是 target user、证据类型和 evidence ID。Warmup 中固定其他输入，对每个 facet 做 leave-one-out rerank，用 target NDCG 的变化作为 delta；LLM 自报来源还必须与真实 packed relations 校验。Credit 在读取时调整 pruner score，在写入时只 gate 间接传播，直接交互永远保留。
>
> V1.0 只看历史 q，但 Books 图太稀疏，92 次传播 0 reject。我通过日志发现已有当前负 delta，于是 V1.1 增加 immediate gate，两个 100-user block 分别拒绝 45 和 47 次。不过 Full-Read 只在一个 block 提升。继续做 write-impact audit 后，我发现 V1.1 Stage-R 没有直接读取邻居长期文本，导致 gate 改变的状态很难进入后续模型，所以 V1.2 在固定原邻居和 1800-token 预算下追加动态邻居记忆，并用 hash 证明 read-after-write。
>
> 300-user 四组实验中 Read 的 NDCG 数值最高，但所有主要差异置信区间都跨 0，Full-Read 还是负方向。我的最终结论是工程闭环成立，当前 gate 的长期效用假设没有得到支持。这个项目让我最深的体会是：有状态 LLM 系统不能只验证某个模块被调用，还要验证它的状态改变能到达后续决策，并用严格消融区分机制和指标。

### 14.3 5 分钟白板版

按以下顺序画：

1. 原始流：`Graph -> Prune -> Pack -> Stage-R -> ReRank -> Stage-W`；
2. 标记 long-term state：`MemoryStorage`；
3. 画反馈断点：ranking outcome 没回到 evidence；
4. 写 `delta_f = R(F)-R(F\{f})`；
5. 画 `claimed ∩ packed`；
6. 写 target-conditional relation key 和 q 更新；
7. 画 Read Credit 与 direct/indirect write 分流；
8. 解释 V1.0 稀疏导致 historical gate 0 reject；
9. 写 V1.1 immediate gate；
10. 画 V1.1 断点：neighbor memory write 没有直接进入 Stage-R；
11. 加上 V1.2 `selected neighbor memory -> bounded append -> Stage-R`；
12. 展示四组消融；
13. 最后给 300-user 结论：机制成立，Full-Read `-0.005616` 且不显著。

---

## 15. 源码阅读地图

| 文件 | 面试前需要理解的内容 |
|---|---|
| `src/models/memrec_agent.py` | 一次 rerank、facet attribution、Stage-W、immediate gate、NMR 读取与统计 |
| `src/train/trainer_memrec.py` | warmup/test 时序、`return_details=True`、串行限制、test freeze、产物保存 |
| `src/memory/graph.py` | User-Item 邻接表和二跳 user neighbors |
| `src/memory/pruner_llm_rules.py` | 原始邻居分数与 Read Credit multiplier |
| `src/memory/packer.py` | packed provenance、V1.2 动态 memory、64/512/1800 token 约束 |
| `src/memory/storage.py` | user/item 单文本覆盖式长期记忆 |
| `src/memory/evidence_credit.py` | relation key、q、幂等、持久化和 historical gate |
| `src/memory/feedback_controller.py` | LOO attribution、support 校验、event logs、read-after-write logs |
| `docs/feedback_memrec_v11_gate_audit.md` | V1.0 为什么不触发，V1.1 规则从何而来 |
| `docs/feedback_memrec_v11_2x100_stage_report.md` | V1.1 两个 block 的结果与限制 |
| `docs/feedback_memrec_v12_neighbor_memory_design.md` | V1.2 唯一改动、预算和 non-goals |
| `docs/feedback_memrec_v12_300_experiment_report.md` | 正式 300-user 结果、统计和机制审计 |
| `outputs/feedback_memrec_v12_300_summary.json` | 所有精确指标、状态差异和分段诊断 |

建议阅读顺序：

```text
本说明文档
-> V1.2 300 report
-> V1.1 gate audit
-> memrec_agent.py
-> trainer_memrec.py
-> feedback_controller.py / evidence_credit.py
-> packer.py
```

---

## 16. 面试前自测清单

如果下面问题都能不看文档回答，项目就基本掌握了：

1. 原始 MemRec 的 R、ReRank、W 分别做什么？
2. User-Item 图与 MemoryStorage 有什么区别？
3. Facet 为什么不是长期 memory node？
4. Corrected baseline 修了什么，为什么不能计入算法收益？
5. `delta_f` 如何计算，正负分别代表什么？
6. 为什么 LLM supporting neighbor 必须和 packed relation 校验？
7. 为什么 Evidence Credit 带 target user？
8. Read Credit 如何影响 pruner？
9. 为什么 direct writes 不经过 gate？
10. V1.0 为什么 92 次传播 0 reject？
11. V1.1 immediate gate 的完整条件是什么？
12. V1.1 为什么仍没有形成稳定 write-impact？
13. V1.2 只改了什么，明确没改什么？
14. 64/512/1800 三个 token 数字分别是什么？
15. read-after-write hash 在证明什么？
16. 四组消融的三个增量比较分别是什么？
17. V1.2 哪组数值最好？能不能说显著？
18. 为什么 Full gate 机制工作了但指标没提升？
19. 这个项目最大的计算成本在哪里？
20. 下一版应该优先改长期信用、写入粒度还是再调阈值？为什么？

---

## 17. 最终项目评价

这是一个有价值的实习面试项目，但价值不应建立在“我把 NDCG 提高了多少”这一句话上。

它展示了四类能力：

1. **源码审计能力**：找到 warmup 调用契约错误，并把 baseline 修复与算法收益分开；
2. **算法设计能力**：设计 facet LOO、target-conditional credit、read multiplier 和 immediate gate；
3. **有状态系统能力**：处理 predict-before-write、状态隔离、串行更新、token budget、provenance 和 read-after-write lineage；
4. **实验与科学判断**：使用中间消融、锁定协议、哈希、公平性审计、bootstrap/permutation 和负结果分析。

最成熟的总结是：

> 我先把 MemRec 从“有协同记忆但没有结果反馈”改造成可审计的反馈闭环；当第一版 gate 因稀疏关系无法触发时，我用运行日志定位并修正；当 V1.1 的正向结果不稳定时，我继续追踪 write-to-read 路径，发现邻居长期记忆没有直接进入 Stage-R，于是用 V1.2 补上受预算约束的动态读取。正式实验表明系统确实学到了信用、拒绝了负传播、改变了长期状态和后续 prompt，但 Full 没有显著提高排序。这让我能明确区分“系统机制有效运行”和“算法假设带来业务收益”，并据此提出下一步更稳健的长期价值估计，而不是对负结果调参或换样本。

这段经历比一个未经审计的漂亮分数更能体现真实的工程与研究能力。
