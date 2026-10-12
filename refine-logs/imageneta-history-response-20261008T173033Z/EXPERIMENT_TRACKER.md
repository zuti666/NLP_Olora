# 首轮实验追踪

| Run ID | 阶段 | 版本 | seed | 预算 | 状态来源 |
|---|---|---|---:|---|---|
| smoke_history_fused_k10_s2 | 短测 | HRF | 2 | 2任务×2epochs×2batches | runtime/queue_state.json |
| smoke_future_only_k10_s2 | 短测 | FM | 2 | 同上 | runtime/queue_state.json |
| smoke_pairing | 门控 | 初始化/输入/首步配对 | 2 | CPU核验两短测 | gates/smoke_pairing |
| formal_history_fused_k10_s2 | 正式 | HRF | 2 | 10任务×50epochs，2550更新 | runtime/queue_state.json |
| formal_future_only_k10_s2 | 正式 | FM | 2 | 同上 | runtime/queue_state.json |

后续三梯度和多seed没有自动加入。最终结果见RESULTS.md；未完整验收的FAA/AAA/BWT保持空值。
