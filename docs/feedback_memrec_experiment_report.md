# FeedbackMemRec V1.0 实验报告

状态：V1.0 冻结诊断结果。V1.1 的开发与最终实验不会覆盖本报告或其输出目录。

V1.1 final 100-user 结果见 `docs/feedback_memrec_v11_experiment_report.md`；本文件继续保留 V1.0 原始诊断结论。

实验日期：2026-08-23  
代码起点：`58d9031ed91c623b8034d2bf04f39aa937424c33`  
上游：`https://github.com/rutgerswiselab/MemRec.git`

## 环境与本地模型

- GPU：NVIDIA GeForce RTX 5090 D，32607 MiB。
- 运行中观测显存：28974 MiB；这是本次采样到的峰值。
- PyTorch：2.9.0+cu128；CUDA runtime：12.8；CUDA 可用。
- vLLM：0.12.0；Transformers：4.57.1；OpenAI client：2.8.1。
- 模型绝对路径：`/root/autodl-tmp/MemRec/models/Qwen2.5-7B-Instruct`。
- 所有 Stage-R、Stage-ReRank、Stage-W 和 counterfactual 都走本地 vLLM；没有外部 API、CPU mock 或 mock LLM。

实际使用的服务命令：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
vllm serve /root/autodl-tmp/MemRec/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --dtype half --max-model-len 8192 \
  --gpu-memory-utilization 0.88 --seed 42 \
  --api-key local-vllm --enforce-eager \
  --generation-config vllm
```

`GET /v1/models` 成功返回 `Qwen2.5-7B-Instruct`；真实 JSON-schema 请求成功返回 `{"status":"ok","count":1}`。使用 eager 是为了绕开 RTX 5090 上首次 CUDA Graph 路径的长时间捕获，不改变模型、权重、GPU 或实验协议。

## Corrected MemRec

上游 warmup 没有 `return_details=True`，Stage-W 因此拿不到 facets 和 pruned context。本地只修复这一点，并把反馈、read credit、write gate 和 attribution 全部关闭。后续比较均以这个 Corrected MemRec 为基线，不把 warmup bug 修复算作 FeedbackMemRec 收益。

feature-off 运行的 relation、attribution 和 propagation 统计均为 0。三组 candidate manifest 各 40 条，两两字节级完全相同。

## 固定 20-user 三组结果

相同条件：固定 20 用户、10 个候选、候选顺序、seed 42、本地 Qwen、8192 context、temperature 0、7 facets、一轮 warmup、串行状态更新、测试冻结 credit。

| 配置 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | 墙钟时间 |
|---|---:|---:|---:|---:|---:|---:|
| Corrected MemRec | 0.60 | 0.70 | 0.75 | 1.00 | 0.7600 | 782.78 s |
| Read Credit Only | 0.60 | 0.70 | 0.80 | 1.00 | 0.7630 | 1868.94 s |
| FeedbackMemRec Full | 0.60 | 0.70 | 0.80 | 1.00 | 0.7630 | 1870.10 s |

这是小样本结果，不做统计显著性声明。Read Credit 和 Full 相对 Corrected 的 NDCG@10 绝对变化为 `+0.00297`，Full 没有超过 Read Credit Only。

### 调试指标

| 指标 | Read Credit Only | Full |
|---|---:|---:|
| relations | 205 | 205 |
| positive / negative / neutral q | 36 / 35 / 134 | 31 / 35 / 139 |
| mean q | 0.00230 | 0.00234 |
| attribution episodes | 20 | 20 |
| counterfactual calls | 140 | 140 |
| positive / negative / zero facet delta | 19 / 17 / 104 | 21 / 17 / 102 |
| mean facet delta | 0.02394 | 0.02679 |
| facet delta range | -0.3691 to 0.6667 | -0.3691 to 0.6667 |
| invalid supporting ID ratio | 6.35% | 6.35% |
| non-neutral selected multipliers in test | 70 | 65 |
| test users affected by multiplier | 13/20 | 12/20 |
| users whose selected-neighbor sequence changed | 2/20 | 2/20 |
| accepted / rejected propagation | not gated | 92 / 0 |
| direct user / item writes | not gate-logged | 36 / 36 |
| credit updates during test | 0 | 0 |

Full 的 92 次传播全部以 `insufficient_observations_explore` 接受；72 次 direct writes 全部以 `direct_interaction_not_gated` 接受。write gate 没有改变本轮状态，所以 Full 与 Read Credit Only 的推荐指标相同。两次真实生成的细微 q 统计差异来自本地 GPU 生成路径的非完全逐 token 可重复性，不影响候选清单一致性。

### 调用与 token

| 配置 | Stage-R/W client calls | reranker calls | 总 calls | 总 tokens | 每个测试用户对应的整次运行 calls |
|---|---:|---:|---:|---:|---:|
| Corrected | 75 | 40 | 115 | 221609 | 5.75 |
| Read Credit Only | 76 | 180 | 256 | 452621 | 12.80 |
| Full | 76 | 180 | 256 | 452637 | 12.80 |

最后一列用总 calls 除以 20 个固定测试用户，包含该用户对应的 warmup 与 test 成本。Feedback 额外成本主要是 20×7=140 次 leave-one-out rerank。

## 日志样例

实际负贡献 facet（字段节选）：

```json
{
  "episode_id": "warmup-r0-u23-t45637",
  "facet_index": 0,
  "facet_text": "Interest in romance novels, particularly those by Barbara Freethy",
  "full_rank": 8,
  "without_facet_rank": 3,
  "full_reward": 0.3010299957,
  "counterfactual_reward": 0.4306765581,
  "facet_delta": -0.1296465624
}
```

实际 credit-aware read：

```json
{
  "event_id": "read-22-u23",
  "target_user_id": 23,
  "evidence_type": "item_neighbor",
  "evidence_id": 113416,
  "original_score": 0.8695652174,
  "q": -0.0043215521,
  "multiplier": 0.9978392240,
  "adjusted_score": 0.8676862817
}
```

实际 accepted propagation：

```json
{
  "episode_id": "warmup-r0-u17-t44941",
  "target_user_id": 17,
  "neighbor_type": "item_neighbor",
  "neighbor_id": 161058,
  "relation_q": 0.0,
  "num_credit_updates": 1,
  "decision": "accept",
  "reason": "insufficient_observations_explore"
}
```

实际 direct write：

```json
{
  "episode_id": "warmup-r0-u17-t44941",
  "entity_type": "user",
  "entity_id": 17,
  "decision": "accept",
  "reason": "direct_interaction_not_gated"
}
```

Rejected propagation：本次默认协议下为 0，没有可诚实提供的真实 rejected 样例。

## 防泄漏与一致性检查

- 每条严格先预测，再计算 target rank，再 attribution/update，再 Stage-W。
- credit reward 只来自固定候选集 target rank 的 NDCG@10，不使用 LLM relevance score，也不使用 `eval_feedback=gt` 的 CLICK 规则作为 reward。
- 测试不运行 counterfactual，三组 `credit_updates_during_test=0`。
- 正式更新串行，未启用 `--parallel`。
- 三组 40 条 candidate manifests 两两字节一致。
- Full 的 direct user/item 写入全部保留；gate 日志只包含协同邻居。
- facet attribution 只存在于 JSONL 日志，未作为长期 memory node 保存。

## Gate 4 与 100-user 决策

三组 20-user、feature-off、read reorder、direct-write、test freeze、serial update 和 candidate parity 均通过。但 5-user 和 20-user Full 都没有实际 rejected propagation，无法证明默认 gate 在当前数据量下产生了运行时作用。因此 Gate 2/4 未完全满足，按任务要求未运行 100-user。

下一步应在不改默认阈值的前提下增加按时间顺序的 warmup 观测，先确认同一 target-conditional relation 能达到至少两次更新；不能通过筛用户、降低阈值或改候选集追求正结果。

## 结果解释

本次小样本中 read credit 确实改变了部分邻居顺序，并带来轻微 NDCG@10/Hit@5 上升；但多数 facet delta 为 0，说明排序对单个 facet 经常不敏感。每个 facet 常映射到多个 item relations，平均分配使单条 q 更新很小。单轮 warmup 对 target-conditional relation 的样本不足，保守 write gate 没有触发，所以 Full 没有获得相对 Read Credit Only 的额外效果。这些结果支持继续研究，但不足以宣称稳定提升。

MemRec 原论文中的其他 baseline 数字未在本地复现；若引用，必须标记为 “Reported by the original MemRec paper; not locally reproduced.”
