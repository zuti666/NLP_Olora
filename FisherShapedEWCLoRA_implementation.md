# FisherShape-EWCLoRA：现有实现、版本选择与新方法落地

## 1. 结论

本仓新增独立方法：

- `lora_strategy=ewclora_fishershape`
- `optimizer_type=fishershape-adamw_hf`

它保留当前 O-LoRA 的双分支、fresh `loranew_A/B`、完整 `Delta-W` Fisher、逐任务 adapter concat 和 Fisher sidecar；每个更新步增加一次同 batch replay，用两次 CE 梯度构造 FisherShape 方向，再在 clean 参数点加 EWC 梯度。

外部三个候选中，最契合的是 `as1_normfisher_fishershape_freshlora_merge.py` 的**算法生命周期语义**，但不能复制它面向 ViT qkv 的 base-weight merge。当前仓继续使用已有的 rank concat 持久化，因为它同样实现历史 `Delta-W` 的精确相加。

## 2. 当前 1B EWCLoRA / EWCLoRA-GAM 的逻辑

### 2.1 配置与入口

Llama-3.2-1B 的现有配置位于：

- `scripts_order1_llama1B/EWCLoRA_LLaMA2-1b.sh`
- `scripts_order1_llama1B/EWCLoRA-GAM_LLaMA2-1b.sh`
- Order 2、Order 3 在对应的 `scripts_order2_llama1B/`、`scripts_order3_llama1B/`。

入口统一为：

```text
shell -> torchrun src/run_uie_lora.py -> UIETrainer
```

1B GAM 配置的共同参数为：rank 8、bf16、batch/device=1、8 GPU 时 gradient accumulation=8、1 epoch；第一任务学习率 `1e-3`，后续任务 `1e-4`。EWC 默认 `lambda=1, gamma=1`。GAM 使用四次 closure，核心配置为 `grad_rho=.02`、`grad_norm_rho=.2`、`beta1/2/3=1`、`grad_gamma=.03`。

任务顺序：

| Order | 任务序列 |
|---|---|
| 1 | DBPedia -> Amazon -> Yahoo -> AGNews |
| 2 | DBPedia -> Amazon -> AGNews -> Yahoo |
| 3 | Yahoo -> Amazon -> AGNews -> DBPedia |

后一任务的 `model_name_or_path` 指向前一任务的 `<stage>/adapter`。`run_uie_lora.py` 同时把该目录解析为 `ewclora_fisher.pt` 的加载目录。

### 2.2 双分支 LoRA 与跨任务保存

自定义 PEFT 层同时维护：

```text
历史分支：lora_A / lora_B       冻结
当前分支：loranew_A / loranew_B 训练
```

forward 为：

```text
W_eff = W_base + scale * B_old A_old + scale * B_new A_new
```

每轮 fresh A 使用 Kaiming 初始化，fresh B 为零，因此任务开始时 `Delta-W_new=0`。保存 adapter 时：

```text
A_saved = concat(A_old, A_new, dim=0)
B_saved = concat(B_old, B_new, dim=1)
```

所以：

```text
B_saved A_saved = B_old A_old + B_new A_new
```

这是精确的 `Delta-W` 累加，没有 factor-add 产生的交叉项。下一任务加载这个累计 adapter，再创建新的 rank-8 fresh 分支。

### 2.3 当前 EWCLoRA

旧任务 Fisher 位于完整有效更新空间：

```text
Delta-W_new = B_new @ A_new
L_EWC = 0.5 * lambda * sum(Omega_old * Delta-W_new^2)
```

它是 zero-anchor EWC：约束当前 fresh update 本身，而不是约束同一 adapter 相对上轮 A/B 的变化。

任务结束后临时启用 LoRA full-`Delta-W` hook，在纯任务 CE 上估计 batch-empirical diagonal Fisher：

```text
F_t = mean_batch[(d CE / d Delta-W_new)^2]
Omega_t = gamma * Omega_(t-1) + F_t
```

原实现保存 `adapter/ewclora_fisher.pt`，与 adapter 权重分开持久化。

当前工作树还给 EWCLoRA 增加了 OLoRA-base 项：默认正交系数 `0.5`、L2 系数 `0`。因此当前普通分支为 `CE + OLoRA-base + EWC`。

### 2.4 当前 EWCLoRA-GAM

GAM 对当前任务目标执行四次前反向。在 clean 点单独得到 EWC 参数梯度，GAM 最终方向使用：

```text
g_final = g_GAM - proj_gEWC(g_GAM) + g_EWC
```

EWC 不进入 GAM closure。当前工作树中的 OLoRA-base 仍位于 GAM closure 内。

因此做“只替换 GAM 为 FisherShape”的严格对照时，建议把旧 GAM 基线的 `olora_lamda_1/2` 设为 0；否则旧基线实际含 OLoRA，而新方法主配置按设计只含 CE shaping + EWC。

## 3. 三个候选版本的比较

| 候选 | EWC 锚点 | 跨任务模型状态 | task-end 顺序 | 与当前仓的关系 |
|---|---|---|---|---|
| `as1_normfisher_fishershape.py` | `Delta-W - Delta-W_ref` | 同一个 A/B 连续更新 | Fisher 后不 reset | 不匹配 fresh `loranew` 零锚点 |
| `as1_normfisher_fishershape_freshlora_merge.py` | `Delta-W_t - 0` | `W <- W + Delta-W_t` 后 reset | 先 Fisher，后精确 merge | **算法语义最匹配** |
| `as1_normfisher_fishershape_freshlora_merge_factor.py` | `Delta-W_t - 0` | `Abar += A_t, Bbar += B_t` | 先 Fisher，后 factor-add | 会产生交叉项，且累计 rank 形状不兼容 |

factor-add 版从第二次合并开始得到：

```text
(Bbar + B_t)(Abar + A_t)
= Bbar Abar + B_t A_t + Bbar A_t + B_t Abar
```

后两项在当前 rank-concat 生命周期中不存在。因此不能因为它也有“历史/当前两支”就选择该版本。

最终迁移组合为：

```text
生命周期与 zero-anchor：freshlora_merge 语义
两次梯度与 additive projector：三个版本共有 FisherShape 核心
跨任务权重存储：保留当前 O-LoRA rank concat
```

## 4. 新方法

### 4.1 两次梯度方向

对同一个有效 batch：

```text
g0 = grad CE(theta)
s0 = rho / (||g0|| + xi)
theta_plus = theta + s0 * g0
g1 = grad CE(theta_plus)
delta_g = g1 - g0
```

默认使用稳定的 ratio 模式：

```text
coeff = c * ||g0|| / (||delta_g|| + xi)
g_FS = g0 + coeff * delta_g
```

所以 shaping 分量的范数约为 `c * ||g0||`。默认 `c=0.2`，不会像 fixed alpha 那样直接受曲率绝对尺度支配。

两个消融共用同一实现：

```text
fs_alpha=0                         -> g_FS = g0
fs_sam_equivalent=true             -> coeff=1, g_FS = g1
fs_alpha_mode=fixed                -> coeff=alpha/s0
```

### 4.1.1 IncLoRA + SAM 等价消融

`ewclora_fishershape` 与 IncLoRA 都冻结历史 `lora_A/B`、只训练 fresh
`loranew_A/B`。因此下面的配置会关闭 EWC/OLoRA，并令 FisherShape 的
方向严格退化为扰动点梯度 `g1`：

```text
lora_strategy=ewclora_fishershape
optimizer_type=fishershape-adamw_hf
fs_sam_equivalent=true
fs_rho=0.05
fs_adaptive=false
ewclora_lambda=0
fs_include_olora_regularization=false
```

为贴近现有 `IncLoRA-SAM_E1_Le3.sh` 的具体实现，还需设置
`fs_replay_rng_state=false`（旧 SAM 不复用第一次 forward 的 dropout RNG）以及
`adam_epsilon=1e-6`、`weight_decay=0`（对齐旧 Transformers AdamW 默认值）。
独立验证入口为：

```text
scripts_order1/IncLoRA-SAM-via-FisherShape_E1_Le3.sh
```

该等价性是“可训练参数集合与 SAM 更新方向”等价，而不是 checkpoint 逐 bit
相同：新路径使用 PyTorch AdamW，旧路径使用 Transformers AdamW；多卡 reduction、
浮点恢复方式和梯度裁剪顺序也略有差异。验证应检查 `fs_coefficient=1`、
`ewc_loss=0`、`regularization_grad_norm=0`，并确认 adapter 可以继续链式加载。

外部设计记录过 fixed `alpha=0.2` 数值发散，因此默认必须是 `ratio`。fixed 模式应从约 `1e-5` 到 `1e-3` 单独扫描。
`rho=0` 会显式退化到 clean `g0`；即使 dropout 使两次闭包梯度不同，也不会在 fixed 模式中产生 `alpha/eps` 级别的爆炸系数。

### 4.2 EWC 与 OLoRA 的注入位置

严格 FisherShape probe 只含任务 CE。完成 `g0/g1` 合成并恢复 clean 参数后，再计算：

```text
g_final = g_FS + g_EWC + optional(g_OLoRA)
```

这里是 **additive projector**，不能复用 EWCLoRA-GAM 的正交投影公式。

默认 `fs_include_olora_regularization=false`，因此主配置严格实现外部设计的 `g_FS + g_EWC`。如果需要与当前工作树中附带 OLoRA-base 的 EWCLoRA 做兼容性消融，可设置：

```bash
--fs_include_olora_regularization true
```

开启后 OLoRA 仍只在 clean 点求一次解析梯度，不进入两个 CE probe。要注意旧普通/GAM 路径会在每个 micro-batch 加一次未按 GA 缩放的 OLoRA 项，而新实现按数学目标只加一次；因此二者不能在不校正系数的情况下宣称 OLoRA 强度完全等价。

### 4.3 真正的两次前反向与 accumulation

训练主循环正常产生并累积 clean `g0`。FisherShape 在 optimizer step 时缓存整个 accumulation window，只 replay 一次得到 `g1`。因此每个更新是两次前反向，而不是“先做一次只用于日志的 forward，再做两次 closure”。

在 8 GPU 默认 `GA=8` 时，`g0` 和 `g1` 都对应相同的 8 个 micro-batches。默认还会逐 micro-batch 恢复 clean pass 的 CPU/CUDA RNG state，使 LoRA dropout 等随机掩码一致；replay 后恢复训练 RNG 流，不额外推进或回退后续随机性。DDP 中间 micro-batch 使用 `no_sync`，最后一个 micro-batch 同步。若 epoch 末尾不足一个 GA，Trainer 会显式 flush，并按实际 micro-batch 数重标定 `g0/g1`；缓存不会被丢弃或带入下一 epoch。

### 4.4 Fisher normalization 与 damping

新增可选顺序：

```text
raw F_t
  -> per-task trace normalization
  -> isotropic damping
  -> gamma * Omega_old + F_t
  -> optional accumulated trace normalization
```

公式：

```text
F_norm = F / (trace(F) + eps)
F_damped = (1 - beta) * F_norm + beta / d
```

其中 `d` 是所有 full-`Delta-W` Fisher block 的总元素数。`beta>0` 时现在强制要求 per-task normalization 已开启，避免把 trace-preserving damping 错用于 raw Fisher。每轮日志同时记录 raw task trace、处理后 task trace、accumulated trace、mean 和 max。

新的 1B 脚本默认打开 per-task normalization，并使用 `beta=0.2`；累计 Fisher 不再次归一化。由于 normalization 会改变 EWC 的绝对尺度，`lambda=1` 只是与原脚本相同的起点，正式实验必须扫描 lambda，不能直接假设它与 raw-Fisher 基线等强。

## 5. 代码路径

```text
scripts_order*_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh
  -> torchrun src/run_uie_lora.py
     -> UIETrainingArguments: fs_* / EWCLoRA Fisher 参数
     -> freeze lora_A/B, train loranew_A/B
     -> UIETrainer._inner_training_loop
        -> clean CE backward: g0
        -> src/optimer_fishershape.py
           -> perturb theta+s0*g0
           -> replay same accumulation window: g1
           -> restore theta
           -> compose ratio/SAM/fixed direction
           -> clean-point additive EWC/OLoRA
           -> AdamW step
     -> post-task full-Delta-W Fisher
     -> adapter rank concat
     -> adapter/ewclora_fisher.pt
```

主要新增/修改文件：

| 文件 | 作用 |
|---|---|
| `src/optimer_fishershape.py` | 两次梯度优化器、异常恢复、DDP sync、Fisher normalization/damping 纯函数 |
| `src/uie_trainer_lora.py` | 新 strategy、accumulation replay、additive regularization、Fisher v2 state |
| `src/run_uie_lora.py` | 新 CLI 参数、自动 Fisher 路径和参数冻结 gate |
| `tests/test_optimer_fishershape.py` | CPU 数值单测 |
| `scripts_order1_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh` | 1B 共用训练配置，支持 Order 1/2/3 |
| `scripts_order2_llama1B/...`、`scripts_order3_llama1B/...` | Order 2/3 thin wrapper |
| `scripts_order1_llama1B/run_all_fishershape_ewc.sh` | 三个 order 顺序运行 |

## 6. CLI

| 参数 | 默认 | 含义 |
|---|---:|---|
| `fs_alpha_mode` | `ratio` | `ratio` 或 `fixed` |
| `fs_alpha` | `0.2` | ratio 的相对强度 c，或 fixed 的绝对 alpha |
| `fs_rho` | `0.05` | 有限差分探测半径 |
| `fs_xi` | `1e-12` | norm 数值稳定项 |
| `fs_sam_equivalent` | `false` | 令 `coeff=1`，得到 CE-SAM 方向 |
| `fs_adaptive` | `false` | 参数自适应扰动；方法主配置保持关闭 |
| `fs_grad_reduce` | `mean` | DDP 最终梯度 reduction |
| `fs_replay_rng_state` | `true` | replay 时复用 clean-pass dropout RNG state |
| `fs_include_olora_regularization` | `false` | 可选：clean 点加入 legacy OLoRA-base 梯度 |
| `ewclora_normalize_fisher` | `false` | core 默认兼容旧 EWCLoRA；新 1B 脚本传 `true` |
| `ewclora_fisher_damping_beta` | `0` | 新 1B 脚本传 `0.2` |
| `ewclora_normalize_accumulated_fisher` | `false` | 是否再归一化累计 Fisher |
| `ewclora_allow_state_mismatch` | `false` | 是否允许混用不同 normalization/damping 状态 |

## 7. 运行

Order 1：

```bash
bash scripts_order1_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh \
  0,1,2,3,4,5,6,7 1993 1.0 1.0 0.2 0.05 true 0.2 false
```

Order 2 / 3 使用对应 wrapper，参数顺序相同：

```bash
bash scripts_order2_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh
bash scripts_order3_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh
```

三个 order 顺序运行：

```bash
bash scripts_order1_llama1B/run_all_fishershape_ewc.sh \
  1993 0,1,2,3,4,5,6,7 1.0 1.0 0.2 0.05 true 0.2 false
```

直接对齐旧 raw-Fisher 配置：

```bash
# normalize_fisher=false, damping_beta=0
bash scripts_order1_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh \
  0,1,2,3,4,5,6,7 1993 1.0 1.0 0.2 0.05 false 0 false
```

脚本以全局 batch 64 为默认目标：8 GPU 自动使用 GA=8，4 GPU 自动使用 GA=16。部署路径可通过 `FISHERSHAPE_CONDA_BASE`、`FISHERSHAPE_CONDA_ENV`、`FISHERSHAPE_BASE_MODEL`、`FISHERSHAPE_TRANSFORMERS_CACHE`、`FISHERSHAPE_OUTPUT_ROOT` 覆盖；已有激活环境可设置 `FISHERSHAPE_SKIP_CONDA_ACTIVATE=true`。也可用 `FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS` 显式覆盖自动 GA。

## 8. Fisher state 兼容性

sidecar 文件名仍为 `ewclora_fisher.pt`，顶层 `omega` 和 `count_updates` 保持兼容。FisherShape 或启用了 normalization/damping 的状态保存为 `full_delta_w_fisher_v2`；未启用这些能力的原 EWCLoRA/EWCLoRA-GAM 仍保存原来的 v1 payload，避免改变现有基线输出。v2 增加：

```text
fisher_space=full_delta_w
estimator=batch_empirical
strategy
normalization.per_task_trace
normalization.accumulated_trace
normalization.damping_beta
normalization.eps
```

loader 仍可读取 legacy bare dict、v1 dict 和 list+names，并校验 v2 的 space/estimator 以及各 Fisher tensor 的有限性和非负性。若旧 state 是 raw Fisher，而继续训练的新 run 请求 normalized Fisher，默认直接报错，避免静默混合两个不同尺度的 EWC 几何；只有明确的消融才应传 `--ewclora_allow_state_mismatch true`。prediction-only 加载不会因这些训练态 normalization 参数不同而失败。

## 9. 验证与限制

已执行：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot/bin/python \
  -m unittest tests.test_optimer_fishershape -v
```

13 个 CPU 测试通过，覆盖：ratio 精确方向、shaping/g0 范数比、SAM 等价、已有 clean gradient 只需一次 replay、clean-point additive projector、probe/projector 异常清理、条件计算图、无参数梯度拒绝、`rho=0` fixed 安全退化、非有限超参数、零曲率差分、trace normalization/damping。

另外执行了 Trainer 级小模型 smoke：`6 batches × 2 epochs, GA=4` 得到恰好 4 个 optimizer steps，覆盖每个 epoch 的 `4+2` 完整/尾部窗口；RNG replay smoke 验证了随机数与 clean pass 一致，并且 replay 后全局 RNG 状态保持不变；v2 Fisher smoke 验证了 prediction-only 可忽略训练参数 mismatch，而继续训练会拒绝静默混用。

当前节点上脚本默认的 `nlora_req` 指向不可用的 `/data/140-0` symlink，默认 Llama 模型路径也不存在，因此本次没有启动 1B 多 GPU 训练。可使用上文环境变量切换到实际部署路径。完整实验前至少先做 Order 1 的两任务 smoke run，确认：

1. stage 1 保存 v2 Fisher；
2. stage 2 自动加载且 normalization metadata 匹配；
3. 每个 optimizer step 日志的 `g0/delta_g/coeff/reg_grad` 有限；
4. adapter 的 `r_sum` 从 8 增到 16；
5. raw task Fisher trace、处理后 trace 和 accumulated trace 均有限。

最后一个理论边界必须保留：FisherShape 当前作用在可训练 A/B factor 参数空间，EWC Fisher 位于 full `Delta-W` 空间。降低 factor-space 的 minibatch gradient proxy，不自动等价于降低 full-`Delta-W` Fisher trace。正式机制结论需要在相同 endpoint CE/accuracy 下额外比较 full-`Delta-W` Fisher。

本次实现是用 FisherShape **替换** EWCLoRA-GAM 中的 GAM 优化方向；没有声称或实现 `FisherShape + GAM` 的嵌套组合。现有 `ewclora_gam` 保留为独立对照基线。
