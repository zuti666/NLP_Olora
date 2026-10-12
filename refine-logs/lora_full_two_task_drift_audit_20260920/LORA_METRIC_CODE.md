# ImageNet-R 六行 LoRA 表：有效权重位移的代码审查

审查日期：2026-09-20。只读审查原仓库、原配置和原日志；没有启动 GPU、训练或恢复队列。

## 1. 结论与真实代码来源

这张六行表的实际代码来源是 `/data/115-2/users/liying/low-rank-cl-new`，不是之前 ImageNet-A 跨 T 使用的 `CopyFlatCV` 单 adapter 链。原表 cohort 为：

`result_summary/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/summary_by_setting.csv`。

**`delta_norm` 是第二任务新 adapter 的 K/V 有效权重更新平方范数；开平方才是 r。对于这张恰好两任务、在第二次累积前评估的表，它可以解释为旧任务 endpoint 到新任务 endpoint 的 K/V 位移。它不是 LoRA A/B 因子位移，也不是全模型位移。**

还发现必须披露的生命周期细节：旧代码在 task-end 分别执行 `A_old += A_new`、`B_old += B_new`，并重置新 A/B。这不是把 `B_new A_new` 无损合并进有效权重；从第二次累积开始有交叉项。六行表的 accuracy 和 drift 都在第二次累积之前产生，因此该问题不自动推翻表内两任务 endpoint 位移解释；但**不能用第二任务累积后的持久模型或无 `_endpoint` 后缀 checkpoint 来解释/核验这个 r**。

## 2. 六行对应的 learner

下面的 `method` 来自精确 cohort 的 `configs_sweeps/.../seed0/*.json`；对应 seeds1/2 的运行来源由同目录 provenance 审查记录。

| 行 | 原 method | 第二任务优化路径 | lifecycle |
|---|---|---|---|
| baseline | `ewclora` | Adam，λ=0 | fresh new A/B，旧累计因子冻结 |
| gam_only | `ewclora_gam_factored` | GAM，关闭EWC | 同上 |
| control | `ewclora` | Adam＋EWC penalty | 同上 |
| setting1 | `ewclora_gam_factored` | Task0 GAM；Task1 Adam＋EWC penalty | 同上 |
| setting2 | `ewclora_gam_factored` | Task0 Adam；Task1 projected GAM＋EWC | 同上 |
| full | `ewclora_gam_projected` | projected GAM＋EWC | 同上 |

实际 factory 映射见 [utils/factory.py:18](/data/115-2/users/liying/low-rank-cl-new/utils/factory.py:18)、[projected:42](/data/115-2/users/liying/low-rank-cl-new/utils/factory.py:42)、[factored:48](/data/115-2/users/liying/low-rank-cl-new/utils/factory.py:48)。factored 是 projected 的子类，按阶段选择 Adam/GAM，见 [methods/ewclora_gam_factored.py:13](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_factored.py:13) 与 [阶段开关:51](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_factored.py:51)。它不是 CopyFlatCV 中 `g_clean + lambda_flat(g_gam−g_clean) + g_ewc` 的新接口，不能把不同实现只按“GAM”名字混用。

该 cohort 是 ImageNet-R、`vit_base_patch16_224_in21k`、rank10、Adam lr=.0005、weight_decay=.005、每任务50 epoch、batch128、λ=1e7、gamma=1；GAM normrho=.5。不是 ImageNet-A lr=.04/.02、SGD、λ2000 那一套。

## 3. 旧仓库真实 forward 与每任务状态

对每层 l、通道 c∈{K,V}，令累计因子为 `(Ā_lc,B̄_lc)`，当前任务新因子为 `(A_lc,B_lc)`。

在训练和表格评估使用 `use_new=True` 时：

\[
W^{eff}_{lc}=W^{pre}_{lc}+\bar B_{lc}\bar A_{lc}+B_{lc}A_{lc}.
\]

- 原始 ViT 有12层，embedding_dim=768，见 [models/net_ewclora.py:65](/data/115-2/users/liying/low-rank-cl-new/models/net_ewclora.py:65)。每层两块 K/V，共24个768×768矩阵，总有效坐标维数14,155,776。
- [models/vit_ewclora.py:215](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:215) 分别建立 accumulated 和 new 因子；[初始化:257](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:257) 把旧 A/B 都置零，把新 A 随机 Kaiming、新 B 置零。
- [forward:289](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:289) 先加旧 `B̄Ā`；[新贡献:311](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:311) 加 `BA`。注入的是 qkv 的中间、最后两个切片，故是 **K/V**。不是 CopyFlatCV `backbone/lora.py` 的 Q/V。
- 该 BA 更新没有 LoRA `alpha/r` 缩放。`self.scale` 是 `head_dim**(-.5)`，在 [attention logits:337](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:337) 缩放 `q @ k.T`，不能误乘到 BA 范数上。
- [ewclora.freeze_network:244](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:244) 与 [projected.freeze_network:479](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_projected.py:479) 只训练当前新 A/B 和当前分类头，旧累计因子及 backbone 在任务内冻结。这里没有按任务保存多个独立旧 adapter 并逐项求和的 forward。
- [accumulate_and_reset_lora:267](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:267) 每个任务后都重新随机初始化新 A、把新 B 清零。它没有把同一新 adapter 持续更新到下一任务。

因此在一个任务内部，起点 `B_new=0`，而旧累计因子和 pretrained base 不变；**当前任务训练起点→合并前 endpoint 的有效 K/V 更新恰是 `B_new A_new`**。

## 4. delta_norm 的精确定义、记录时点和开平方

原 [ewclora._iter_delta_terms:345](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:345) 与 [projected._iter_delta_terms:724](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_projected.py:724) 只枚举当前 `lora_new_B_k @ lora_new_A_k` 及 V，不读取旧 BA，不做 A/B 因子拼接，不减前一任务的新 BA。

[utils/mechanism_eval.py:1441](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:1441) 对这些矩阵逐项平方求和，并于1456行直接返回：

\[
\texttt{delta\_norm}=\sum_{l,c\in\{K,V\}}\|B^{new}_{lc}A^{new}_{lc}\|_F^2,
\qquad
r=\sqrt{\texttt{delta\_norm}}.
\]

另有

\[
D_F=\sum_{l,c,i,j}\Omega^{old}_{lc,ij}(B^{new}_{lc}A^{new}_{lc})_{ij}^{2},
\qquad
\texttt{normalized\_fisher\_drift}=D_F/(r^2+\epsilon).
\]

其中第二式是旧对角 importance 下的单位平方位移加权量，不能称作沿真实路径测得的完整 empirical-Fisher directional sensitivity。函数按已有 `omega_W` 的块数截断；没有旧 Fisher/首次任务时，外层返回0作为“不可用”占位，不能把该0当成真实运动为零。

原汇总脚本 [run_verify_imagenet_r_1x1_table_metrics_s012.sh:651](/data/115-2/users/liying/low-rank-cl-new/scripts/run_verify_imagenet_r_1x1_table_metrics_s012.sh:651) 将 `mech1.delta_norm` 明确导出为 `theta2_minus_theta1_norm_sq`。所以旧数值列若直接写2.44/9.80，单位是平方范数；换成1.560/3.131须说明开平方。多seed时 `mean(sqrt(d_s))` 与 `sqrt(mean(d_s))` 不相同；如果后者用于表格，应标明 RMS radius，不能将二者互换。六行具体聚合和四舍五入由原表数值审查另行核对。

记录时点明确如下：

1. [trainer.py:73](/data/115-2/users/liying/low-rank-cl-new/trainer.py:73) 完成当前任务训练。
2. [trainer.py:79](/data/115-2/users/liying/low-rank-cl-new/trainer.py:79) 调 `incremental_test` 得到表格 accuracy；[methods/base.py:248](/data/115-2/users/liying/low-rank-cl-new/methods/base.py:248) 走 `interface`，默认 `use_new=True`。
3. [trainer.py:83](/data/115-2/users/liying/low-rank-cl-new/trainer.py:83) 如果开启保存，写 `task_t_endpoint.pth`。
4. [trainer.py:84](/data/115-2/users/liying/low-rank-cl-new/trainer.py:84) 才调 `after_task`。其中 [ewclora:55](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:55) / [projected:144](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_projected.py:144) 先测 drift，随后估 Fisher/写机制记录，最后才累计并重置。
5. [trainer.py:106](/data/115-2/users/liying/low-rank-cl-new/trainer.py:106) 写无 `_endpoint` 后缀的累积后checkpoint。

日志中的 accuracy 文字虽在 `after_task` 后输出，其数值变量来自第2步；不能按日志行先后误认为它是在累计后重新评估的。

## 5. 为什么两任务 r 可成立，以及合并后的陷阱

旧代码的累积算子是：

\[
\bar A^+=\bar A+A,\qquad \bar B^+=\bar B+B,\qquad B^{new,+}=0.
\]

因此累积后：

\[
\bar B^+\bar A^+
=\bar B\bar A+BA+\underbrace{\bar BA+B\bar A}_{\text{额外交叉项}}.
\]

它一般不等于累积前的 `B̄Ā+BA`，所以不是无损权重合并。该判断直接来自267–275行的逐因子加法，而不是根据方法名称推测。

对六行表，起始累计 Ā=B̄=0。Task0 首次累积之后 Ā=A0、B̄=B0，没有交叉项；Task1 开始时新 B1=0。直到 Task1 表格评估前，Ā/B̄保持冻结。因此：

\[
W^{eff}_{T0}=W^{pre}+B_0A_0,
\quad
W^{eff}_{T1,endpoint}=W^{pre}+B_0A_0+B_1A_1,
\quad
W^{eff}_{T1,endpoint}-W^{eff}_{T0}=B_1A_1.
\]

所以在这两个**用于 accuracy 的 endpoint**间，`sqrt(delta_norm)` 的位移解释在代码层面成立。第二次累积后则是：

\[
W^{eff}_{T1,post}-W^{eff}_{T0}=B_1A_1+B_0A_1+B_1A_0.
\]

一般不等于表内 r，且新 B 已重置为零。未来若用该代码做三任务及以上，旧任务已评价 endpoint 与下一任务真实起点之间可能出现这种未纳入当前任务 drift 的变化；不能把这张表的等价关系直接推广到长序列。

更一般地，`||BA_new||` 等于 endpoint 移动，需要：(a) 新分支起点有效更新为0；(b) 历史分支与基底在区间内不变；(c) 若有缩放必须计入同一有效坐标；(d) 两 endpoint 位于同一个更新区间、没有遗漏参数更新或不保持函数的 task-end 变换。LoRA的秩10不对 r 的绝对值给出上界，也不意味着r只能是小数。

## 6. 为什么不能套用 CopyFlatCV 的解释

当前 `CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py` 继承 FitArchitecture→SeqLoRA。它于 [FitArchitecture:235](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:235) 保存当前 BA 作为任务起点，于 [GAM drift:353](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py:353) 计算 `(BA_now−BA_start)^2`。当前 [SeqLoRA:1518](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:1518) 只在首次初始化 LoRA，以后继续同一 A/B；[after_task:281](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:281) 不重置 adapter。

该分支的骨干 forward 位于 [CopyFlatCV/backbone/lora.py:205](/data/140-1/users/liying/CopyFlatCV/backbone/lora.py:205)，注入Q/V。`save_lora_parameters` 保存当前因子并增加 backbone 外层 task_id，不足以证明下一任务新建分支：当前 SeqLoRA 不重建内部 QKV wrapper，不刷新 `saved_A/B`。恢复入口还显式清空历史分支并将内部 task_id 置0，见 [baseLoRA.py:871](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/baseLoRA.py:871)。

这两套坐标、生命周期和优化规则都不同。之前 ImageNet-A 的 `BA_t−BA_0` 正确，不构成对本次 ImageNet-R fresh-K/V 表格的代码证明。

## 7. 版本证据和核验边界

原control seed0日志开始于2026-05-28 19:49:10，标明上述cohort配置和 `method=ewclora`；该日志 `save_ckp=false`。不能仅凭当前源码声称所有原checkpoint存在，checkpoint复核另由资产审查给出。

low-rank-cl-new 当前 HEAD 为 `64f18471fa46dd084aa863418dd005ef0489ce8c`。`methods/ewclora.py`、`methods/ewclora_gam_projected.py` 有当前未提交修改；本次只读，没有改动它们。为避免把当前重构强行套到历史，已将实际关键函数与运行前提交 `0716d51`（2026-05-26）做AST比较：`init_param`、`accumulate_and_reset_lora`、`reset_new_lora`、两 learner 的 `_iter_delta_terms` 和 `freeze_network` 均一致。两个 `_compute_delta_drift_metrics` 则发生过重构：旧版本在函数内直接累加 `delta_now.pow(2).sum()`，当前版本调用新增的 `delta_fisher_drift_metrics` helper，新增top-1%坐标指标，并对Fisher显式clamp非负。两版的核心 `delta_norm = Σ‖B_new A_new‖²_F` 定义一致，不能说完整函数AST一致。`models/vit_ewclora.py` 的最后已提交修改为2026-05-25。这支持上述指标与lifecycle解释具有历史对应关系，但尚未取得每次历史运行的源码SHA，也不代替从原checkpoint数值重算的最终核验。

本次读取源码 SHA256：

| 源文件（相对 low-rank-cl-new） | SHA256 |
|---|---|
| methods/ewclora.py | `48ac23e6fe4dc2a0fe9060c093276839d80efc5bfdb8b74f095183f11d6470a7` |
| methods/ewclora_gam_projected.py | `3a6558b11b6fbfaedafb98a747b41433c6289f888bef181c232acd7e20e4b7b0` |
| methods/ewclora_gam_factored.py | `9b56ad257cd6b30a9a50941b3150e4a212a6bb17b6eda5bc0b6e8e60d620ff08` |
| models/vit_ewclora.py | `cc1b9223faf893114910d97bd4eac5b829c069a9c2ca2b5b85913f27c3b6d534` |
| trainer.py | `0de723f6308d3741d0f7b7f6f3a002ad1ea39ca521f3c3da73998a0c589480f3` |
| utils/mechanism_eval.py | `d2367815cba7b3647729e4f08a1185dc6f9af30b91e0d813565c372e48cd95c2` |

## 8. 保护 Fisher、buffer mass 与谱 trace 的区别

精确 cohort 的六份 seed0 配置均没有覆盖 `fisher_mode`；[ewclora:36](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:36) 和 [projected:98](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_projected.py:98) 默认 `batch_empirical`。这不是当前 ImageNet-A 的 normalized-Fisher 训练协议。

训练保护 Fisher 的实际计算见 [FisherComputer._compute_batch_fisher:569](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:569)：通过BA中间张量的hook取得有效K/V梯度，按batch size加权、最后除使用样本数；[累加:971](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:971)、[除N:625](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:625)。令 `g_b = ∇_W (|b|⁻¹Σ_i loss_i)`，其定义为

\[
\widehat F^{batch}_{diag}=N^{-1}\sum_b |b|\,(g_b\odot g_b),
\qquad \Omega_t=\gamma\Omega_{t-1}+\widehat F^{batch}_{diag,t}.
\]

这里除N是样本平均，**没有全局/逐层均值、最大值或trace归一化，也没有加入正数floor**。非负性来自梯度平方；指标helper的 `clamp_min(0)` 只截负数，不是正数floor。projected/factored [共用FisherComputer:758](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_projected.py:758)；[Omega更新:195](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_projected.py:195) 没有额外归一化。cohort gamma=1，保护Fisher使用训练loader batch128及完整当前任务数据。批均梯度平方通常不同于逐样本梯度平方均值，不能把它命名为严格的逐样本empirical Fisher。

| 字段 | 实際来源与定义 | 不能混同的量 |
|---|---|---|
| `endpoint_fisher_total_mass` | 当前任务刚计算的 `fisher_W` 所有元素之和；[buffer求和:1398](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:1398)；[after_task:86](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:86) 在更新累计Omega之前传入该buffer | 不是累计Omega的总mass，不是逐样本raw trace |
| `accumulated_fisher_total_mass` | 更新后的Omega元素之和；Task0等于endpoint mass，Task1在gamma=1时为两个任务mass之和 | 不是新任务单独Fisher |
| `endpoint_emp_fisher_trace` | 另起batch32 loader，至多前10batch，构造批均梯度外积算子，再用10个Rademacher探针求Hutchinson trace | 不是上面buffer的直接求和，也不是逐样本empirical trace |

谱trace的算子为 `v ↦ M⁻¹Σ_b g_b(g_bᵀv)`，见 [delta_w_empirical_fisher_vector_product:745](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:745)；CE先sum再除batch样本数，见 [loss:504](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:504)、[梯度:764](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:764)。[trace:795](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:795) 返回探针 `zᵀFz` 的平均。具体batch数/探针数来自cohort配置，调用见 [1740](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:1740) 和 [1792](/data/115-2/users/liying/low-rank-cl-new/utils/mechanism_eval.py:1792)。两种统计有不同batch大小、取样范围与trace估计误差，数值无需相同。

此外，endpoint评估使用当前 `train_loader`、当前 `task_start/task_end` 和当前任务分类头，[ewclora:297](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:297)、[projected:253](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora_gam_projected.py:253)。**Task1的endpoint Fisher不是在Task1模型上重新测的旧Task0 Fisher**，不支持把它当作旧知识路径敏感度。

原control seed0日志给出可直接辨认的例子：Task0 endpoint mass=`23.72785186767578`，另算empirical trace=`121.27313690185547`；Task1 endpoint mass=`26.255470275878906`，累计mass=`49.983314514160156`。这些字段必须保留不同名称，不能因都有“Fisher”而互换。
