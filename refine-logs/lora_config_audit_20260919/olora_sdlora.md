# OLoRA / SDLoRA 配置与实际消费审计（2026-09-19）

结论：与 Vanilla 的已记录训练骨架大体一致，但不能写成除方法名外配置完全相同。SDLoRA/Aircraft 的 LR=.02，Vanilla/OLoRA=.05；OLoRA、SDLoRA 的 5Datasets 为40轮/任务，Vanilla为30轮。另有方法自身的容量、正交损失和可学习缩放差异。12份历史日志和当前对应 YAML 均未显式记录 OLoRA 的正交/L2 系数或 SDLoRA 的 α 初始化等专属值；这些值只能标为“当前可读源码默认/硬编码”，不能作为已证明的历史运行参数。

本审计只读配置、日志和代码，没有训练或更改表格。证据层级：运行日志 > 对应可读配置文件 > 当前源码默认；Flatness_CV 当前文件不是三月运行时不可变快照。

## 跨数据集的已记录骨架

全部12项均为 seed0、ViT-B/16、rank16、batch128、SGD（初始及后续 momentum=0、weight_decay=0）、cosine、class_shuffle=false、memory_size=memory_per_class=0。初始和后续 LR 相同，初始和后续 epoch 数相同。

| 数据集 | 任务类数 | OLoRA LR | SDLoRA LR | OLoRA/SD epochs | Vanilla LR / epochs |
|---|---|---:|---:|---:|---|
| CUB-200 | 10×20 | .01 | .01 | 40 | .01 / 40 |
| Cars196 | 16+9×20 | .05 | .05 | 40 | .05 / 40 |
| Aircraft | 10×10 | .05 | .02 | 40 | .05 / 40 |
| Flowers102 | 12+9×10 | .0025 | .0025 | 40 | .0025 / 40 |
| Pets | 5+8×4 | .0025 | .0025 | 40 | .0025 / 40 |
| 5Datasets | 200/100/196/102/37 | .02 | .02 | 40 | .02 / 30 |

5Datasets 的日志有 `increment=1`，但同时声明5任务，实际 `Learning on` 依次是0–200、200–300、300–496、496–598、598–635；不能把 increment=1 解读为单类增量。Cars 的 SDLoRA 目录名含 `lr0025`，实际日志和对应 YAML 均为 `.05`。应以已记录值为准。

## OLoRA：实际正交/L2/缩放

当前 `models_LoRAbasedCL/olora.py` 在后续任务的 `_update_representation` 读取：

- **`olora_lamda_1`**（代码拼写是 lamda，不是 lambda），默认 **1.0**。
- **`olora_lamda_2`**，默认 **0.0**。
- 12份被核查日志/YAML中的6份OLoRA均没有这两个键，也未用其他系数覆盖；因此只能说明“按当前源码会落到1/0”。

对应 SGD loss（首任务只用 CE，后续任务才加项）：

`CE(logits_new_classes, local_labels) + λ1 * Σ_previous_tasks,layer,Q/V ||A_old A_new^T||_F² + λ2 * Σ_current_A,B ||W||_F`。

正交项是交叉矩阵平方后求和，不是把 Frobenius norm 再取均值；没有按任务数、层数或 rank 归一化。因此即使系数固定1，其约束总量会随历史任务数增长。名为 L2 的部分计算未平方的 norm 和；默认系数0使其不影响梯度。不能因类注释写着“plus L2”就声称本表启用了非零 L2 正则。

`learn_alpha=False` 在 learner 中硬编码，backbone 使用 IdentityScale，当前/历史 LoRA 贡献系数固定1；没有常见的额外 alpha/r 缩放。所有12个attention层的Q/V都挂LoRA：新 A 使用Kaiming uniform，B=0。基础ViT权重和旧任务A/B冻结，每任务新增一套可训练A/B。新旧adapter在同一特征前向中相加。

## SDLoRA：缩放并非一个固定超参数

当前 learner 硬编码 `learn_alpha=True`；没有 SD 专属正交或 L2 penalty。SGD 后续任务 loss 是当前类 masked CE。A/B初始化、Q/V位置与冻结基础权重规则和OLoRA相同，区别是历史方向加上可学习幅度。

当前backbone的缩放是**每任务一个全局 scalar**，同一scalar复用到所有挂载层及Q/V；不是每层独立一个α。当前α与历史α都作为可训练参数注册，初始化为1；`max_prev_tasks` 默认200，因此构造时分配200个历史wrapper+1个当前wrapper，但只在已有任务对应的wrapper上有数据梯度。

**需要明确的实现细节：**构造器仅在 `eval=True` 时读磁盘 `scaling_factor*.pt` 恢复已学α。每个新任务训练时都会建立fresh backbone且 `eval=False`，所以当前可读实现把当前和历史α全部重新初始化为1，然后在该任务内再优化。源码注释声称“next task loads them”容易令人误以为训练时warm-start α，实际分支不支持该说法。此处是当前源码事实；历史日志没有α记录或源码快照，不能断言归档运行必然采用完全相同版本。

α与新A/B和分类头放入同一SGD参数列表，未看到独立的α LR、warmup阶段或训练到某个epoch再冻结的机制。不可用论文名称推定它实现了额外两阶段warmup/freeze流程。

## 与 Vanilla 的训练数据、损失、头和容量比较

| 项目 | Vanilla（当前可读实现） | OLoRA | SDLoRA |
|---|---|---|---|
| 优化器 | SGD | 相同SGD | 相同SGD |
| 训练样本 | 当前任务训练样本 | 相同 | 相同 |
| 后续任务 CE | 仅新类logits；labels减已知类数 | 相同CE再加正交项 | 相同masked CE |
| adapter演化 | 同一套A/B跨任务持续更新 | 每任务新增A/B；旧方向冻结并累加 | 每任务新增A/B；旧方向冻结，缩放继续可训练 |
| 类别头 | 扩展线性头，复制旧行 | 相同 | 相同 |
| NME | 另算全已见类均值 | 相同附加评估 | 相同附加评估 |
| 表中结果 | CNN任务矩阵 | CNN任务矩阵 | CNN任务矩阵 |

三者 head 都是 `IncrementalNet` 的 `SimpleLinear`（有bias），扩展时复制旧weight/bias。未设置独立head LR；SGD后的任务损失只切取新类，旧行虽保留requires_grad，但其梯度为0，且momentum/weight_decay均0，所以旧行不会因该SGD更新而变化。模型评估对全部已见类输出logits，训练mask不等于测试时使用任务ID。OLoRA/SD的CNN头不是CL-LoRA那种任务原型分类器。

OLoRA、SDLoRA也调用 `compute_all_seen_class_means` 为独立NME评估读取旧类训练数据；这没有进入此表的CNN loss或替换CNN头。因而可以说“CNN训练无replay”，但不要将整个程序描述为完全不再读取旧训练图像。

rank16不代表相同累计参数预算：Vanilla保留一套16-rank Q/V adapter；OLoRA、SDLoRA为每任务保留一套，累计方向数随任务增长。按12层、Q/V、768维计，一套A/B为589,824参数（不含头）；SD另外有幅度参数。两者不是单纯在固定容量Vanilla上改一个loss系数。

## Scheduler / warmup 的证据限定

历史日志均记录cosine但不记录 `min_lr`、warmup或独立α/head LR。当前三个learner都传 `eta_min = args.get('min_lr', 0.1*base_lr)`，T_max=epochs，scheduler每epoch step；共享builder无warmup包装。因此按当前代码推断这三者cosine最低LR为起始LR的10%，但此10%没有被历史日志单独固化。它不等于新CL-LoRA显式默认eta_min=0的调度。

## 每项运行的配置与日志

- **OLoRA / aircraft**：LR=0.05，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/aircraft/0/olora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40/exp_run/10/olora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **OLoRA / cars196**：LR=0.05，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/cars196/0/olora_inr_sgd_cars196_t20-E40_rank16_lr005_train-eval-paer/exp_run/20/olora_inr_sgd_cars196_t20-E40_rank16_lr005_train-eval-paer_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/olora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **OLoRA / cub200**：LR=0.01，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/cub200/0/olora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/olora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/olora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **OLoRA / flowers**：LR=0.0025，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/flowers/0/olora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/olora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/olora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **OLoRA / pets**：LR=0.0025，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/pets/0/olora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/olora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/olora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **OLoRA / het5**：LR=0.02，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/het5/0/olora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/olora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/olora_inr_sgd_het5_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **SDLoRA / aircraft**：LR=0.02，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/aircraft/0/sdlora_inr_sgd_aircraft_t20_rank16_lr002-E40_epoch40/exp_run/10/sdlora_inr_sgd_aircraft_t20_rank16_lr002-E40_epoch40_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/sdlora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **SDLoRA / cars196**：LR=0.05，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/cars196/0/sdlora_inr_sgd_cars196_t20_rank16_lr0025-E40_train-eval-paer/exp_run/20/sdlora_inr_sgd_cars196_t20_rank16_lr0025-E40_train-eval-paer_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/sdlora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **SDLoRA / cub200**：LR=0.01，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/cub200/0/sdlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/sdlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/sdlora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **SDLoRA / flowers**：LR=0.0025，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/flowers/0/sdlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/sdlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/sdlora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **SDLoRA / pets**：LR=0.0025，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/pets/0/sdlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/sdlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/sdlora_inr_sgd_t20c10_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。
- **SDLoRA / het5**：LR=0.02，epochs=40，seed=0；[原日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/het5/0/sdlora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/sdlora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224.log)；[当前对应YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/sdlora_inr_sgd_het5_r16.yaml)。日志/配置中未记录正交、L2、scaling、warmup、freeze或min_lr专属字段。

## 当前源码定位（不是历史快照声明）

- [OLoRA系数读取](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/olora.py:396)、[SGD实际loss](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/olora.py:556)、[首任务loss](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/olora.py:357)。
- [正交loss计算](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:745)、[缩放初始化及仅eval恢复](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:531)、[Q/V挂载](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:569)、[A/B初始化](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:657)。
- [SDLoRA fresh backbone及learn_alpha](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/sdlora.py:177)、[SD masked CE](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/sdlora.py:605)。
- [线性头扩展](/data/140-1/users/liying/Flatness_CV/utils/inc_net.py:389)、[SGD参数解析](/data/140-1/users/liying/Flatness_CV/models_CL/baseLearner.py:555)、[scheduler](/data/140-1/users/liying/Flatness_CV/models_CL/baseLearner.py:518)。
- [Vanilla一套adapter跨任务复用](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/seqlora.py:1318)、[Vanilla masked CE](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/seqlora.py:1908)。

本次读取的当前源码SHA256：

- `models_LoRAbasedCL/olora.py`: `670636e21573b001e39ad7e614c0ba7fc642a87a1ed04868b78c1842cceb9394`
- `models_LoRAbasedCL/sdlora.py`: `6c6a4715ba6c0e0f6410d59a12ec0aae332efa2001355b5fd4708be4ca21c1eb`
- `backbone/lora.py`: `6148efe40bdb3c5a2a478badbc9c996506e342658fa9160b9bed76b255018abd`
- `models_CL/baseLearner.py`: `69c8467a94a9904f0c2b8039477082f0bd76f7b6899a1e6c1a99529a60fd63db`
- `utils/inc_net.py`: `0e8c164f975cc933f4ff7eceebb49ee0f49804fd34bcbb71ead30a3187cb6565`
