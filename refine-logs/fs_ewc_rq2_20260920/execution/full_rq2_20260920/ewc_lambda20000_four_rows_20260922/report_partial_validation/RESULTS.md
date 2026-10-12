# 固定 λ=20000：四分支、三个 seed

λ=20000 由用户手动固定；不是扫描最优值，也没有被先前扫描证实达到保护门槛。
状态 partial：0/12 有效完成，0 发散，0 无效。

同一分支的 seeds 0/1/2 全部有效完成时才报告 mean ± sample SD（ddof=1）。发散、失败、缺失不填零，不参与凑均值。

| 分支 | Task0 | Task1 | 有效 seed | a11 | B1 batch proxy | a21 | a22 | BWT | A2 | d_QV_cont |
|---|---|---|---|---|---|---|---|---|---|---|
| ewclora | C | C+P | 0 | — | — | — | — | — | — | — |
| fs_ewclora | C+F | C+F+P | 0 | — | — | — | — | — | — | — |
| task1_shaping | C+F | C+P | 0 | — | — | — | — | — | — | — |
| task2_shaping | C | C+F+P | 0 | — | — | — | — | — | — | — |

逐 seed 原生观测：

| trial_id | status | a11 | B1_batch_proxy | a21 | a22 | BWT | A2 | d_QV_cont | raw_fisher_trace | raw_fisher_weighted_drift_squared | P_over_CE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ewclora_seed0 | pending | 87.3 | 34.0699 | — | — | — | — | — | — | — | — |
| ewclora_seed1 | pending | 86.64 | 26.556 | — | — | — | — | — | — | — | — |
| ewclora_seed2 | pending | 87.3 | 33.6969 | — | — | — | — | — | — | — | — |
| fs_ewclora_seed0 | pending | 86.97 | 24.8231 | — | — | — | — | — | — | — | — |
| fs_ewclora_seed1 | pending | 86.97 | 18.867 | — | — | — | — | — | — | — | — |
| fs_ewclora_seed2 | pending | 87.95 | 22.1084 | — | — | — | — | — | — | — | — |
| task1_shaping_seed0 | pending | 86.97 | 24.8231 | — | — | — | — | — | — | — | — |
| task1_shaping_seed1 | pending | 86.97 | 18.867 | — | — | — | — | — | — | — | — |
| task1_shaping_seed2 | pending | 87.95 | 22.1084 | — | — | — | — | — | — | — | — |
| task2_shaping_seed0 | pending | 87.3 | 34.0699 | — | — | — | — | — | — | — | — |
| task2_shaping_seed1 | pending | 86.64 | 26.556 | — | — | — | — | — | — | — | — |
| task2_shaping_seed2 | pending | 87.3 | 33.6969 | — | — | — | — | — | — | — | — |

指标和证据范围：

- a11/a21/a22 为原生 CIL 矩阵的百分比值；BWT=a21−a11（百分点），A2=(a21+a22)/2。
- B1_batch_proxy 取同 seed 原生 Task0 anchor 日志的 in_task_fisher_final_raw_trace：fit loader、1259 样本的 batch-gradient-square proxy，未归一化、非逐样本 Fisher、无轨迹平均。
- raw_fisher_trace 来自本轮复用/重算的 Task0 raw Fisher，独立核验 payload；它与历史固定 train256 逐样本 Fisher 不同，不能混填 B1。
- d_QV_cont 是同一 LoRA 分支从 Task0 到 Task1 的 24 个 Q/V 有效 BA 位移的 Frobenius 范数，排除分类头。rawF 加权平方为 Σ rawF·(BA1−BA0)²；endpoint_P=λ/2 倍该量。
- CE_mean、P_mean 按全部 400 个实际更新求均值；P 已含 λ/2，P/CE 为两均值之比。发散行如有遥测只表示有限前缀，不能与完整训练均值等同。
- 梯度只采样零起始 steps 0、1、8、16…；SGD reference 是真实 CE 分量，GAM reference 是 native clean_grad_cache。GAM 实际更新减 EWC 的残差另称 GAM direction，包含 shaping，绝不称干净 CE 梯度。每个 seed 的 gradient_source/component_source 保留于 CSV/JSON。
- 所有继承的 a11/B1 都来自已经完成的父 Task0；待运行行出现这些值不代表 Task1 已执行。
- 本轮没有新增 λ=0 训练；既有 matched λ=0 对照未在本报告重新核验，因此相对无 EWC 的保护收益记为不可判定。历史 trace-normalized λ=2000 与本轮 raw λ=20000 尺度不同，不混并为正式因果比较。
- report/COMPLETE.json 只在全 12 个原生结果和证据链有效完成后生成；不表示固定 λ 已证明最优或通过保护门槛。

- ewclora_seed0 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- ewclora_seed1 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- ewclora_seed2 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- fs_ewclora_seed0 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- fs_ewclora_seed1 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- fs_ewclora_seed2 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- task1_shaping_seed0 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- task1_shaping_seed1 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- task1_shaping_seed2 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- task2_shaping_seed0 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- task2_shaping_seed1 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
- task2_shaping_seed2 [pending]：No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.
