# FeedbackMemRec V1.1 实验报告

状态：V1.1 最终固定 100-user 实验已完成。V1.0 的代码、输出与报告未覆盖；V1.0 诊断见 `docs/feedback_memrec_experiment_report.md`。

实验日期：2026-08-23 至 2026-08-25  
模型：本地 `Qwen2.5-7B-Instruct`，GPU/vLLM，temperature 0，seed 42  
final 协议：`docs/feedback_memrec_v11_locked_protocol.md`

## V1.1 改动与开发门槛

V1.1 只给协同传播 Write Gate 增加本轮负贡献判断：

```text
if has_current_attribution and raw_episode_delta < 0:
    reject("negative_current_episode_contribution")
elif num_updates >= 2 and historical_q < -0.3:
    reject("negative_historical_credit")
else:
    accept("neutral_or_exploration")
```

`raw_episode_delta` 使用 learning rate 和 q clip 之前的当轮 relation 聚合值。direct target-user/current-item writes 永远不拦截；零贡献或无本轮 attribution 时继续探索；历史阈值未改变。

V1.0 Full 20-user 的 92 次传播审计得到 37 次当前 attribution 匹配：正 9、负 8、零 20；未匹配 55。审计详见 `docs/feedback_memrec_v11_gate_audit.md`。

开发验证只用于确认 gate 生效，不作为最终效果结论：

| 开发运行 | NDCG@10 | propagation accepted/rejected | immediate rejects | direct writes | test credit updates |
|---|---:|---:|---:|---:|---:|
| 固定 5-user Full V1.1 | 0.7492 | 20 / 2 | 2 | 18/18 保留 | 0 |
| 固定 20-user Full V1.1 | 0.7621 | 82 / 8 | 8 | 72/72 保留 | 0 |

门槛通过后即锁定 final 用户、三组配置、规则与 SHA-256；没有根据 final 结果调阈值、换用户或改候选集。

## Final 100-user 协议

- final 用户：`data/eval_user_samples/strict_books_v11_final_100_seed42.json`；与开发 20-user 交集为 0。
- 三组：Corrected MemRec、Read Credit Only、FeedbackMemRec Full V1.1。
- 每组固定 100 用户、10 candidates、相同原始顺序、7 facets、一轮 warmup、严格串行。
- 测试阶段冻结 credit；顺序保持 `predict -> evaluate -> update`。
- 三组 candidate manifest 各 200 行，三者逐字节完全一致。
- 锁定用户清单和三份配置的 SHA-256 在运行结束后复核，均与运行前记录一致。

逐组原始运行目录：

- `outputs/feedback_memrec_v11_final_100_corrected`
- `outputs/feedback_memrec_v11_final_100_read`
- `outputs/feedback_memrec_v11_final_100_full`

两组正式实验的紧凑汇总为：

- `outputs/feedback_memrec_v11_2x100_summary.json`
- `outputs/feedback_memrec_v11_2x100_summary.csv`

汇总中保留两组正式实验指标、候选一致性、测试信用冻结、门控统计和已知证据限制。

## Final 100-user 推荐结果

| 配置 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | 墙钟时间 |
|---|---:|---:|---:|---:|---:|---:|
| Corrected MemRec | 0.45 | 0.69 | 0.72 | 1.00 | 0.689771 | 3887.66 s |
| Read Credit Only | 0.47 | 0.68 | 0.74 | 1.00 | 0.696837 | 9138.73 s |
| FeedbackMemRec Full V1.1 | 0.49 | 0.70 | 0.74 | 1.00 | 0.709473 | 9140.05 s |

绝对变化：

- Read 相对 Corrected：NDCG@10 `+0.007066`。
- Full 相对 Corrected：NDCG@10 `+0.019702`。
- Full 相对 Read：NDCG@10 `+0.012636`，Hit@1 `+0.02`，Hit@3 `+0.02`，Hit@5 不变。

这是一份固定 100-user、单 seed 结果，不做统计显著性或跨数据集泛化声明。它支持“immediate write gate 在本设置中产生实际状态差异，并与更好的最终排序指标相关”，但不能单凭这一轮证明因果稳定提升。

## Full V1.1 gate 审计

| 指标 | 数量 |
|---|---:|
| propagation events | 448 |
| accepted | 403 |
| rejected | 45 |
| `negative_current_episode_contribution` | 45 |
| `negative_historical_credit` | 0 |
| 有当前 attribution：负 / 正 / 零 | 45 / 63 / 85 |
| 无当前 attribution | 255 |
| warmup accepted / rejected | 218 / 45 |
| test accepted / rejected | 185 / 0 |
| direct user writes accepted | 174 |
| direct item writes accepted | 174 |

45 次 reject 全部满足 `has_current_attribution=true` 且 `raw_episode_delta<0`；没有正、零或无 attribution relation 被 immediate gate 误拒绝。全部 348 次实际生成的 direct writes 都以 `direct_interaction_not_gated` 提交。historical gate 本轮仍未达到拒绝条件。

测试阶段不运行 counterfactual，因此没有 current episode attribution；185 次测试后协同传播均按探索/历史规则接受。这些写入发生在当前预测完成之后，不影响当前用户已经计入指标的排名。

## Attribution 与 credit

| 指标 | Read Credit | Full V1.1 |
|---|---:|---:|
| attribution episodes | 99 | 99 |
| counterfactual calls | 693 | 693 |
| relations | 1026 | 1034 |
| positive / negative / neutral q | 314 / 232 / 480 | 315 / 244 / 475 |
| mean q | 0.001890 | 0.002674 |
| invalid supporting relation ratio | 4.87% | 4.22% |
| credit updates during test | 0 | 0 |

每个成功 attribution episode 有 7 次 leave-one-facet-out，因此 99×7=693 次 counterfactual。facet 只保存在 attribution JSONL 中，没有进入长期 memory graph。

## 候选、调用与防泄漏核验

- 三组 candidate manifest 均为 200 行，逐字节一致。
- 三组测试均有 `n_stage_r_calls=100`、`n_stage_rr_calls=100`，100/100 测试排名有效。
- Read 与 Full 均有 `credit_updates_during_test=0`。
- Full 的 attribution、credit update 和 immediate gate 只在 warmup 使用当前 target；测试冻结后不使用当前标签更新 credit。
- 正式运行没有 `--parallel`，状态更新严格串行。
- feature-off Corrected 的 credit、attribution 和 gate 统计均为 0。
- 源码 `py_compile` 通过，V1.0 historical-gate fallback 检查通过。

## 生成异常与限制

本地 Qwen 在长 JSON schema 输出上存在重复内容后截断的问题：

- Corrected warmup：user 561 的 Stage-R 截断。
- Read 与 Full warmup：user 245 的 reranker 截断、user 561 的 Stage-R 截断、user 752 的 reranker 截断。
- Read 与 Full 的三处异常轨迹完全一致；二者均完成 99 个 attribution episodes 和 693 次反事实。
- 三组测试均为 100/100 有效排名。

因此 Read–Full 是更干净的 gate 增量比较；Corrected 与两个反馈组的 warmup 生成完整性并非完全相同，解释 Corrected 对比时需保留这一限制。按照运行前锁定协议，没有在 final 开始后改 max tokens、prompt、重试、用户或规则，也没有重放失败 episode。

## 结论

V1.0 的历史信用 gate 在稀疏 target-conditional relations 上没有触发。V1.1 在不改 storage、图拓扑、attribution、Read Credit、模型或候选的前提下，让本轮负贡献直接控制本轮协同传播；final 100-user 中真实拒绝 45 次，direct writes 全部保留，测试期零信用更新，候选完全一致。

在这一固定实验中，Full V1.1 的 NDCG@10 高于 Read Credit Only `0.012636`，说明“反馈控制写入”已不再只是未触发的机制，并值得进入多 seed/更多数据集复验。当前证据仍不足以宣称普遍或统计显著提升。
