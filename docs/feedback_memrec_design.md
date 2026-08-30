# FeedbackMemRec V1.0 设计说明

状态：V1.0 冻结设计。V1.1 只在可关闭配置项下扩展 Write Gate，关闭该项时恢复这里记录的 V1.0 行为。

V1.1 设计与最终实现见 `docs/feedback_memrec_v11_design.md`；本文件继续保留为 V1.0 冻结基线，不覆盖。

## 目标与边界

FeedbackMemRec 在不改变 MemRec 长期文本记忆结构和 User-Item 图拓扑的前提下，学习目标用户条件化的协同证据信用，并用它控制下一轮 Stage-R 读取顺序和 Stage-W 协同传播提交。模型参数不训练，synthesized facet 不持久化。

长期状态只有：

- 原有 `MemoryStorage` 中的 user/item 文本记忆；
- 独立的 `EvidenceCreditStore`；
- attribution、read、direct-write 和 propagation 事件日志。

facet、prompt context 和单轮 attribution record 是临时状态。facet 只写 JSONL 日志，不进入 `MemoryStorage` 或 User-Item 图。

## 当前源码审计结论

- `UserItemGraph` 仍是运行时双向邻接表；user-user 邻居由共同 item 推导，不是真实持久图边。
- Stage-R 读取目标用户长期文本 memory、user 邻居最近三个 item 标题、item 邻居静态 metadata。它没有读取邻居的长期文本 memory。
- Stage-R 输出 facets、`supporting_neighbors` 和解释性的 `support_edges`；这些 LLM 声明不能直接作为真实图边或信用真值。
- Stage-ReRank 接收 facets、固定候选信息、item memories 和 instruction。
- Stage-W 生成 target user、真实交互 item 和协同邻居的候选更新。
- 上游 warmup 调用 `rerank()` 时未请求 details，导致 facets 和 pruned context 丢失。V1 的 Corrected MemRec 只把 warmup 改为 `return_details=True`，确保原本的 Stage-W 输入存在。

## Relation credit

独立模块 `src/memory/evidence_credit.py` 使用键：

```text
(target_user_id, evidence_type, evidence_id)
```

其中 `evidence_type` 只允许 `user_neighbor` 或 `item_neighbor`。记录包含 `q`、更新次数、正负更新次数、最后 step 和最后 episode。新 relation 的 `q=0`，范围限制在 `[-1, 1]`，JSON 原子保存/恢复。

同一 episode 先聚合该 relation 获得的全部 facet shares，再统一更新一次：

```text
share(f, relation) = facet_delta(f) / number_of_valid_sources(f)
episode_delta(relation) = sum of shares in this episode
q_new = clip(q_old + 0.2 * clip(episode_delta, -0.5, 0.5), -1, 1)
```

episode ID 也作为幂等保护，重复提交不会再次更新同一 relation。

## Packed evidence 校验

packer 的 prompt 文本保持不变，但额外返回：

```json
{
  "packed_user_neighbor_ids": [],
  "packed_item_neighbor_ids": [],
  "packed_relations": []
}
```

每个 facet 的正式来源是：

```text
LLM claimed supporting_neighbors intersect actual packed_relations
```

`support_edges` 只用于解释，不写入原图，也不参与 V1 的 relation key。

## 逐 facet attribution

学习阶段先复用已有完整排名，再对每个 facet 做 leave-one-out：

```text
delta_f = R(F) - R(F without f)
R(rank) = 1/log2(rank+2), if zero-based rank < 10, else 0
```

反事实重排固定 candidate IDs、原始候选顺序、target item、item memory 快照、instruction、模型和 temperature；只移除一个 facet，不重跑 Stage-R，也不重写其他 facets。

`delta_f` 是 conditional counterfactual attribution；从 facet 到已验证 relations 的均分是 approximate source credit，不宣称精确因果归因，也不要求各 facet delta 之和等于 bundle delta。

## Read credit

在原 MemRec score 计算完后应用：

```text
multiplier = clip(1 + 0.5 * q, 0.5, 1.5)
adjusted_score = original_memrec_score * multiplier
```

然后才执行 top-k。`q=0` 时严格为 1，不改变拓扑，也不永久删除负信用邻居。关闭 feature 后直接使用原评分。

## Write gate

gate 在 Stage-W LLM 完成候选生成后执行：

```text
target user direct write: always commit
current interacted item direct write: always commit
collaborative neighbor write:
  if num_updates < 2: accept for exploration
  else if q >= -0.3: accept
  else: reject this propagation
```

gate 只判断历史协同传播资格，不证明新生成文本本身正确或错误；reject 也不删除邻居。

## 顺序与防泄漏

状态更新强制串行。启用 credit learning 时拒绝 trainer 的并行模式。每个 episode 的顺序是：

```text
predict -> evaluate target rank -> leave-one-out attribution
-> update relation credit -> Stage-W generate -> gated commit
```

测试阶段 `freeze_credit=true`，不运行 counterfactual，最终指标显式记录 `credit_updates_during_test`。测试中的原 MemRec text-memory write 仍遵循 predict-before-write，并在三组中使用同一协议；它不是 credit 学习，也不能影响当前已完成的预测。

## 配置

- `configs/feedback_memrec_books_corrected.yaml`：全部反馈模块关闭。
- `configs/feedback_memrec_books_read_credit.yaml`：attribution + credit read，write gate 关闭。
- `configs/feedback_memrec_books_full.yaml`：attribution + credit read + write gate。
- `configs/feedback_memrec_books_local_qwen_base.yaml`：共享的模型、候选、用户、seed 和默认参数。

三组都使用本地 Qwen2.5-7B-Instruct、temperature 0、seed 42、10 个候选、7 个 facets、相同 20 用户清单和相同冻结协议。
