# FeedbackMemRec V1.1 Write-Gate Audit

原审计对象：FeedbackMemRec V1.0 固定 20-user Full；20-user开发运行目录已清理，审计结论保留在本文。  
审计日期：2026-08-23

## 方法

对每个 facet attribution event，先按 V1.0 的 source-credit 规则重建 learning-rate 和 q-clip 之前的 relation episode delta：

```text
raw_share = facet_delta / len(validated_supporting_relations)
raw_episode_delta(episode, relation) = sum(raw_share)
```

随后用以下严格键与 propagation event 匹配：

```text
(episode_id, target_user_id, evidence_type, evidence_id)
```

其中 propagation 的 `neighbor_type` 已是 `user_neighbor` 或 `item_neighbor`。没有使用 claimed-only relation，也没有使用 `support_edges`。

## 92 次传播的匹配结果

| 类别 | 数量 |
|---|---:|
| 总 propagation events | 92 |
| 能匹配本轮 validated relation | 37 |
| 未匹配本轮 attribution | 55 |
| 匹配且 raw episode delta > 0 | 9 |
| 匹配且 raw episode delta < 0 | 8 |
| 匹配且 raw episode delta = 0 | 20 |

阶段拆分：

- warmup propagation：51，其中 37 条匹配、14 条未匹配；
- test propagation：41，全部未匹配，因为 V1.0 测试冻结 credit 且不运行 counterfactual。

因此，若只增加 `has_current_attribution and raw_episode_delta < 0`，现有 20-user 开发运行理论上有 8 个真实 immediate rejects；零贡献和无 attribution 的传播仍探索性接受。

## num_updates 与 historical q

| 分组 | n | num_updates 分布 | historical q 范围 | historical q 均值 |
|---|---:|---|---:|---:|
| 全部传播 | 92 | 0: 24；1: 68 | -0.03691 到 0.10000 | 0.00589 |
| matched | 37 | 1: 37 | -0.03691 到 0.10000 | 0.00994 |
| unmatched | 55 | 0: 24；1: 31 | -0.01845 到 0.10000 | 0.00316 |
| negative matched | 8 | 1: 8 | -0.03691 到 -0.00089 | -0.01208 |
| positive matched | 9 | 1: 9 | 0.00476 到 0.10000 | 0.05161 |
| zero matched | 20 | 1: 20 | 0 | 0 |

`historical q` 是 V1.0 propagation 日志中 gate 当时可见的 q；按既有执行顺序，它已经包含当轮 credit update。没有 relation 达到原历史条件 `num_updates >= 2 and q < -0.3`。

raw episode delta 分布：

- matched 全体：-0.18454 到 0.82858，均值 0.05858；
- negative matched：-0.18454 到 -0.00447，均值 -0.06040；
- positive matched：0.02380 到 0.82858，均值 0.29454。

## 实际负贡献匹配样例

```json
{
  "episode_id": "warmup-r0-u37-t34791",
  "target_user_id": 37,
  "neighbor_type": "item_neighbor",
  "neighbor_id": 114155,
  "raw_episode_delta": -0.18453512321427123,
  "num_updates": 1,
  "historical_q": -0.036907024642854246
}
```

V1.0 因观测次数不足接受了该传播。V1.1 将只拒绝这一轮对该协同邻居的写入，不永久禁用邻居，也不影响 target user/current item direct writes。

## 审计结论

V1.0 的历史 gate 在 target-conditional relation 稀疏时无法触发，但当轮 leave-one-out 已产生足够的 relation-level 原始信号：37 个匹配项中有 8 个明确为负。增加可关闭的 immediate episode gate 有直接数据依据，不需要降低 `-0.3` 历史阈值、重复 warmup、搜索阈值或筛选用户。
