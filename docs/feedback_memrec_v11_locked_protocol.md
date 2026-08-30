# FeedbackMemRec V1.1 Locked Final Protocol

锁定时间：2026-08-24，早于任何 final 100-user 运行。

## 开发门槛

- 5-user：2 次 `negative_current_episode_contribution` reject；18 次 direct writes 全部保留；测试 credit 更新 0；候选 manifest 与 V1.0 一致。
- 固定开发 20-user：8 次 `negative_current_episode_contribution` reject；72 次 direct writes 全部保留；测试 credit 更新 0；候选 manifest 与 Corrected、Read Credit 一致。
- 所有状态更新串行，无重复 warmup。

门槛通过，因此在查看 final 100-user 结果前锁定以下内容。

## 锁定规则

```text
if has_current_attribution and raw_episode_delta < 0:
    reject("negative_current_episode_contribution")
elif num_updates >= 2 and historical_q < -0.3:
    reject("negative_historical_credit")
else:
    accept("neutral_or_exploration")
```

- `raw_episode_delta` 是 learning rate 与 q clip 之前的当轮聚合值；
- `episode_delta == 0` 或没有当轮 attribution 时接受；
- direct target-user/current-item writes 永远提交；
- historical threshold 保持 -0.3，最少历史观测保持 2；
- 不搜索阈值、不筛用户、不改变候选集。

## 锁定 final 用户

- 清单：`data/eval_user_samples/strict_books_v11_final_100_seed42.json`
- 来源：固定 1k Books 清单中，排除开发 20-user 后的前 100 用户；
- 与开发 20-user 交集：0；
- 用户数：100；seed：42。

## 锁定三组配置

1. `configs/feedback_memrec_books_corrected_v11_final_100.yaml`
2. `configs/feedback_memrec_books_read_credit_v11_final_100.yaml`
3. `configs/feedback_memrec_books_full_v11_final_100.yaml`

三组使用同一本地 Qwen2.5-7B-Instruct、temperature 0、8192 context、10 个候选、7 facets、一轮不同时间位置的 warmup、测试冻结 credit 和严格串行执行。运行开始后不得根据结果修改规则、用户、阈值或候选。

锁定文件 SHA-256：

```text
13f622b31387c4ff193e503815339fb0af5f32ce3e60cb27d5417938d1c1670a  strict_books_v11_final_100_seed42.json
71ac7045981a5ddc1ee3eceaa74efcca18cdbc31cb63e3ba443e68f0eec1c13a  feedback_memrec_books_corrected_v11_final_100.yaml
6fc09d3a79e8ba94db8041cc7c4777ad84754032e2d3daf651f84804b45c160a  feedback_memrec_books_read_credit_v11_final_100.yaml
d23580da51fc45ffed46f4773b608776b0b1d0a3fbfbcf6bcc2eea47f67e08fc  feedback_memrec_books_full_v11_final_100.yaml
```
