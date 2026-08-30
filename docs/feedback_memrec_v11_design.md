# FeedbackMemRec V1.1 设计说明

状态：规则与 final 100-user 协议已锁定。V1.1 只扩展协同传播 Write Gate；attribution、EvidenceCreditStore、Read Credit、候选集、模型和原有文本 memory 结构均沿用 V1.0。

## 改动目标

V1.0 的协同写入 gate 只参考历史信用：

```text
num_updates >= 2 and historical_q < -0.3
```

Books 数据中的 `(target_user, evidence_type, evidence_id)` relation 很稀疏，一轮 warmup 下通常只出现一次，导致负贡献 relation 仍以探索名义接受传播。V1.1 复用本轮 leave-one-facet-out 已经计算出的 relation-level 原始贡献，在传播发生前增加一次性 immediate gate。

## 唯一算法变化

```text
if has_current_attribution and raw_episode_delta < 0:
    reject("negative_current_episode_contribution")
elif num_updates >= 2 and historical_q < -0.3:
    reject("negative_historical_credit")
else:
    accept("neutral_or_exploration")
```

- `raw_episode_delta` 是 facet delta 按 validated supporting relations 分摊后，在同一 episode/relation 上求和的值；它位于 learning rate 和 q clip 之前。
- `raw_episode_delta == 0` 或本轮没有 attribution 时继续接受探索。
- immediate reject 只拒绝本轮对该协同邻居的间接传播，不永久删除、封禁或修改图边。
- target user 与当前真实交互 item 的 direct writes 始终提交。
- 原历史阈值 `-0.3`、最少观测数 `2`、credit 学习率和 clip 均未改变。

## 数据流与时序

warmup/prequential episode 严格串行：

```text
predict
-> 用当前 target rank 计算逐 facet leave-one-out delta
-> 验证 supporting relation 并聚合 raw_episode_delta
-> 更新 EvidenceCreditStore
-> Stage-W 生成候选 patch
-> direct writes 无条件提交
-> collaborative writes 依次执行 immediate gate / historical gate
```

trainer 把本轮 `raw_relation_deltas` 作为临时参数传给同一 episode 的 `write()`；它不写入 memory graph，也不跨 episode 缓存。测试冻结时不运行 counterfactual、不更新 credit，并向 Write Gate 传入空的 current attribution，因此当前测试标签不可能影响当前预测或 gate。

## 配置与回退

新增配置：

```yaml
write_gate:
  use_current_episode_delta: true
```

- `true`：V1.1 immediate gate + 原历史 gate。
- `false`：精确恢复 V1.0 的历史 gate 分支和原 decision reason。
- feedback/read/write/attribution 全关：恢复 Corrected MemRec；不创建有效 relation credit、attribution 或 propagation gate 状态。

相关配置：

- `configs/feedback_memrec_books_full_v11.yaml`
- `configs/feedback_memrec_books_full_v11_final_100.yaml`
- `configs/feedback_memrec_books_block2_seed43_full.yaml`

## 日志与可审计性

每个 collaborative propagation gate event 记录：

```json
{
  "has_current_attribution": true,
  "raw_episode_delta": -0.1845,
  "historical_q": -0.0369,
  "historical_num_updates": 1,
  "decision": "reject",
  "reason": "negative_current_episode_contribution"
}
```

聚合统计新增：

- `n_immediate_episode_rejected`
- `n_historical_credit_rejected`
- `n_propagation_accepted`
- `n_propagation_rejected`

facet 仍只保存在 attribution JSONL 中，不进入长期文本 memory、User-Item 图或 EvidenceCreditStore。长期新增状态仍只有 relation `q` 等数值信用。

## 开发验证与 final 锁定

V1.0 20-user 的 92 次传播审计发现：37 次可匹配本轮 validated relation，其中正 9、负 8、零 20；55 次无本轮 attribution。V1.1 在真实 GPU 运行中得到：

| 开发集 | propagation accepted/rejected | immediate rejects | direct writes | test credit updates |
|---|---:|---:|---:|---:|
| 固定 5-user | 20 / 2 | 2 | 18/18 保留 | 0 |
| 固定 20-user | 82 / 8 | 8 | 72/72 保留 | 0 |

两组 reject 均由负 `raw_episode_delta` 触发，而不是降低历史阈值、重复 warmup、筛用户或阈值搜索制造。开发门槛通过后，final 100-user 用户清单、三组配置和 SHA-256 在运行前锁定，详见 `docs/feedback_memrec_v11_locked_protocol.md`。

## 研究边界

V1.1 回答的是“本轮反事实反馈能否控制间接协同写入”。它不引入：

- 邻居长期文本 memory 进入 Stage-R；
- memory version/content hash；
- memory-level utility 或 trust；
- user-user 持久图边；
- memory rollback；
- 对 LLM 自报 `facet_ids` 的真值依赖。

因此 V1.1 的性能变化不会与大型 memory architecture 重构混在一起。
