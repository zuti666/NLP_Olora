# EWCLoRA：seed 0/1/2 补齐计划（仅计划，未启动训练）

## 计数与可复用边界

沿用当前表中每个数据集 best-FAA 运行选定的配置，目标是 5 数据集 × 3 seeds = **15 个结果**。已有 7 个同归档协议的有效结果，**名义缺口为 8 次**；这不等于已经证明旧结果与未来新源码逐字节可比。

| 数据集 | 固定 λ | 当前执行族 | 可核实旧 seeds | 待补 seeds | 名义新增次数 |
|---|---:|---|---|---|---:|
| Aircraft | 200 | 历史 α=0 两次 CE wrapper | 0 | 1, 2 | 2 |
| Cars | 80000 | α=0 wrapper，probe/Fisher RNG 双隔离 | 1, 2 | 0 | 1 |
| CUB-200 | 200000 | 历史 α=0 两次 CE wrapper | 0 | 1, 2 | 2 |
| Flowers | 10 | α=0 wrapper，probe/Fisher RNG 双隔离 | 1, 2 | 0 | 1 |
| 5Datasets | 50000 | native_sgd，一次 CE backward | 2 | 0, 1 | 2 |

保留 λ、执行族、归一化和 RNG 设置，只改变 seed 和新的输出路由。最终按每个 seed 的 FAA/AAA_stage 计算均值与样本标准差（n−1）；不再按 seed 选最高分，不将每任务准确率当重复实验。

若要求**完整新冻结协议**，在同一代码、依赖、权重、数据清单和有效配置下重做全部 seeds，此 EWC 子集是 **15 次新运行**，不能同时把旧的 7 条也作为独立新结果计入。即使只回放旧 seed 来验证复现，重复相同 seed 的回放也不增加独立 seed 数。

## 固定的详细参数

五组共同：`model_name=as1_normfisher_fishershape`，`backbone_type=vit_base_patch16_224`，rank 16，batch 128，`optimizer=sgd`，`optimizer_type=fishershape`，`fs_base_optimizer=sgd`，momentum/weight_decay 均 0，无回放，`class_shuffle=false`，cosine。初始 LR/epochs 等于后续任务 LR/epochs。

| 数据集 | LR | epochs/task | init_cls / increment / nb_tasks | method-specific diagnostics |
|---|---:|---:|---|---|
| Aircraft | .05 | 40 | 10 / 10 / 10 | mechanism_eval=false, mechanism_grad_stats=false |
| Cars | .05 | 40 | 16 / 20 / 10 | mechanism_eval=true, mechanism_grad_stats=true |
| CUB-200 | .01 | 40 | 20 / 20 / 10 | mechanism_eval=false, mechanism_grad_stats=false |
| Flowers | .0025 | 40 | 12 / 10 / 10 | mechanism_eval=true, mechanism_grad_stats=true |
| 5Datasets | .02 | 30 | 200 / 1 / 5（异质流覆盖普通 increment） | mechanism_eval=true, mechanism_grad_stats=true |

5Datasets 任务顺序固定 CUB200 → Aircraft100 → Cars196 → Flowers102 → Pets37；删除表中 Pets 单独一列不删除异质流里的 Pets。

```yaml
ewc_gamma: 1.0
ewc_eta: 0.0
ewc_max_batches: 100
fisher_mode: batch_empirical
ewc_normalize_fisher: true
ewc_normalize_accumulated_fisher: false
ewc_fisher_norm_eps: 1.0e-12
ewc_fisher_damping_beta: 0.0
ewc_dual_ascent: false
ewc_skip_task_fisher: false
compute_full_fisher: false
fs_rho: 0.05
fs_alpha: 0.0
fs_alpha_mode: ratio
fs_xi: 1.0e-12
fs_sam_equivalent: false
fs_adaptive: false
fs_grad_reduce: mean
```

这里 Fisher 为 ΔW 上的 batch-gradient-square proxy，每任务归一化后累积，不是逐样本 Fisher。约束项是有效 LoRA ΔW 的漂移；当前 native 实现与 wrapper 的方向注入在 clean weights 计算，两者的 CE 调用次数不同。

- Cars/Flowers 冻结显式参数：`fs_direction_mode=alpha0`，`fs_preserve_probe_rng=true`，`ewc_preserve_fisher_rng=true`，`endpoint_eval=false`。
- Het5 冻结显式参数：`fs_direction_mode=native_sgd`，两个 preserve_rng 均 true，`endpoint_eval=false`。`fs_rho` 等 wrapper 参数保留配置字段，但 native 路径不执行 probe。
- Aircraft/CUB 历史 YAML/日志**没有** `fs_direction_mode` 和两个 RNG preserve 字段。现存兼容逻辑会取 `ratio`（alpha=0）、probe=false、Fisher=false，但这只是现存代码解释，不能当成已确认的 8 月 11 日实参。补齐计划应保留此语义并显式记录推定来源；如改成双 RNG 隔离，就是新协议，必须重做 seed0。
- scheduler 名称虽固定，但历史未固化 `min_lr`；当前 SeqLoRA 默认 `eta_min=0.1×lr`，workers 默认 8。这些应在新有效配置中明确，而不能事后断言旧运行必然采用当前默认值。

## 可直接复制的配置来源

所有路径基于 `/data/140-1/users/liying/CopyFlatCV/`；完整绝对路径与 SHA256 已写 `ewc_plan.json`。

| 数据集 | 原始有效配置 / 待补配置 |
|---|---|
| Aircraft | `config_5datasets/as1_sgd_ewc_lambda_sweep/aircraft_sgd_ewc_lam200.yaml`；派生 seeds1,2 |
| CUB-200 | `config_5datasets/as1_sgd_ewc_lambda_sweep_round2/cub200_sgd_ewc_lam200000.yaml`；派生 seeds1,2 |
| Cars | `config_5datasets/as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/cars196_g0_ewc_c000_rho005_lam80000_beta000_s0.yaml` 已存在 |
| Flowers | 同目录 `flowers102_g0_ewc_c000_rho005_lam10_beta000_s0.yaml` 已存在 |
| Het5 | `config_5datasets/as1_fivearm_native_controls/fivearm_native_v1_20260826/het5_native_ewc_rho005_lam50000_beta000_s0.yaml` 和 `_s1.yaml` 已存在 |

已逐字段检查 Cars/Flowers seed0 模板及 Het5 seed0/1 模板：相对已完成参考 seed，仅 `seed`、`prefix` 不同。**配置存在不代表运行完成**：这些缺失 seed 对应输出目录不存在。若执行，复制到新 campaign，并为每个方法/数据集/seed 使用独立输出目录，不向历史路径续写。

## 为什么其他旧 seed0 不能替代

- Cars λ80000 历史 seed0（30.697/40.681111）没有双 RNG 隔离记录；native λ80000 seed0 出现非有限训练。两者均不能补当前双隔离 wrapper 的 seed0。
- Flowers λ10 的历史 seed0（48.652/62.408981）、仅 Fisher RNG 隔离 seed0（52.182/65.913043）、额外 damping=.2 seed0，均不同于当前双 RNG 隔离、damping=0 的配置。
- Aircraft/CUB 同 λ 的双 RNG 隔离 seeds0/1/2 是另一完整实验族，不可把其中 seed1/2 接到历史 sweep seed0 后。
- Het5 λ50000 wrapper seeds0/1/2 不可接到 native seed2；λ100000/200000 不仅参数不同，还存在已确认输出并发覆盖问题。

以上来自 [此前完整审计](../ewclora_best_audit_20260919/extra_candidates.json) 与 [本次原始矩阵复算](../finegrained_std_20260922/ewc.json)，没有再次无界扫描。

## 代码复用的实际阻碍

当前工厂分派到 `models_LoRAbasedCL/as1_normfisher_fishershape.py`，其父链包含 `as1_normfisher_gam.py`、`ewclora_youyue_fitArchitechture_gam.py`、`ewclora_youyue_fitArchitechture.py`、`seqlora.py`，wrapper 优化器为 `optimer_PerturabtionType/fishershape.py`。

这些核心文件与 `refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/` 的 9 月 18 日冻结版相同；但 **`backbone/lora.py` 已改变**：`_LoRA_qkv_timm_train.forward` 删除了每次 forward 构造临时随机初始化 Linear 的动作，避免 CPU RNG 消耗。SeqLoRA/EWC 正在使用该类；旧代码即使 `task_id=0` 也消耗这些随机数。因此直接用现在工作树跑相同旧 YAML，不能视为只有 seed 改动。

冻结版 `backbone/lora.py` SHA256：`303ed752ccf68b3a42106aedfc134e97e4305af0b013a45aea7f41ea7665dc86`；当前版：`995119b1aea75e27049ff6a20aeafac08daa2c212fd2d7c3eff517c1bf581446`。

若走“保留已选配置补齐”路线，优先在完整独立冻结副本上形成有效配置并回放一个已有 seed 核对任务矩阵/训练轨迹；匹配也不能独自证明全部历史来源相同，仍需记录历史权重 SHA、数据清单、依赖和源码绑定的缺口。9 月 18 日快照可用于构建新的可追踪基线，但不能直接冒充 8 月 11 日/27 日实际运行快照。找到的 8 月 26 日单优化器快照只覆盖另一 CIFAR/Het5 证据包，不足以修补这五组的完整 provenance。

## 更严格统一协议与比较风险

“固定 best-selected 配置”意味着跨数据集仍有 wrapper/native 和 RNG 差异；这是当前表的延续性复测，不是统一 EWC 实现。如果要求统一，则先预定所有数据集相同执行方式、RNG 隔离、Fisher、scheduler floor、loader、相同权重字节和数据清单，再跑各自 seed0/1/2；原 best-selected 分数留在历史表。若统一为 native，不应把 Cars λ80000 的已有非有限失败隐藏；应预定失败报告与选参规则，不能跑完再丢差 seed 或临时换 λ。

λ 来自历史测试结果选优；补多个 seeds 不会消除这种配置选择偏差。模型 ID 相同不证明历史权重字节相同；rank/batch 相同也不证明 CL-LoRA 等方法与 EWC 的参数或算力预算相同。

## 已有日志的耗时参考

| 数据集 | 可读日志跨度（分钟/完整运行） | 样本数 |
|---|---:|---:|
| Aircraft | 176.78 | 1 |
| Cars | 119.57、121.21 | 2 |
| CUB-200 | 229.99 | 1 |
| Flowers | 48.66、49.26 | 2 |
| Het5 | 170.27 | 1 |

跨度为每份日志首条至末条时间，包含初始化、训练、Fisher 和评估，全部仅一段 prefix。按这些跨度机械累计名义 8 次补跑约 **22.06 小时**的串行参考工作量；日志未记录 GPU 型号及共享负载，不能据此承诺实际耗时或外推多卡完成时间。原始起止时刻、日志路径与 SHA 在 `ewc_plan.json`。此计划未占用 GPU，未创建训练进程。
