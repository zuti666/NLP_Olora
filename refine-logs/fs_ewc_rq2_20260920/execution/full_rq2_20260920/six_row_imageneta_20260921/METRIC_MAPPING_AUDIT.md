# ImageNet-A 六行表指标映射审查

结论：**有条件通过。** 当前正式结果可填六行历史布局中的 4 行；`sequence_gam_seed{0,1,2}` 的 code-task1 端点可填当前协议的 FS-EWCLoRA 行。两行仍未训练，不能用 B3-B 或 B3-D 改名补齐。新表必须把 batch-gradient Fisher、逐样本旧任务 Fisher，以及 fresh-adapter 与 continued-adapter 位移分开。

## 列映射

| 列 | ImageNet-A 可用来源 | 建议列名/脚注 | 不可混用项 |
|---|---|---|---|
| $a_{11},a_{21},a_{22}$、BWT | 正式三 seed 的 CNN `t01` 矩阵；BWT 每 seed 先算 $a_{21}-a_{11}$ | 保留；注明论文 Task 1/2 对应 code task0/1，$a_{21},a_{22}$ 是 40 类联合推理下的任务分组准确率 | 不能用固定旧 20 类 local-head accuracy 替换 $a_{21}$ |
| 旧表 $\mathcal B_1$ | anchor metrics 的 `in_task_fisher_final_raw_trace` | 改为或脚注明确 $\mathcal B^{QV}_1$：Task-1 自身 fit-loader 上、有效 Q/V 坐标中的未归一化 **batch-gradient proxy** | 不能填成 fixed-oldtrain256 的逐样本 raw trace `1451.954/634.510` |
| 旧表 $\mathcal B_2$（若保留） | 各已完成分支 `mechanism.steps["1"].in_task_fisher_final_raw_trace` | $\mathcal B^{QV}_2$；它是 Task 2 自身数据和 active head 的 current-task proxy | 不是旧任务在 Task-2 端点的敏感度，也不是累计 EWC buffer |
| 旧表 $r_{KV}$ | 正式 trajectory/endpoint 记录的 24 个有效 Q/V 矩阵变化 | **改名为 $d^{\mathrm{cont}}_{QV}$** | 旧 ImageNet-R 的 $r^{\mathrm{fresh}}_{KV}$ 是 fresh Task-2 K/V adapter 更新范数；坐标与生命周期均不同，数值不可横比 |
| RQ2 逐样本 Fisher | `raw_trace_anchor_train256` / endpoint raw trace | 若展示，应另列为 $\mathcal T_1(\theta_j)$，注明同一 fixed oldtrain256、固定旧 head、逐样本真标签 CE | 不得与 $\mathcal B_j^{QV}$ 共用列名或数值尺度 |

正式 $\mathcal B_1^{QV}$ 的三 seed 均值±样本 SD 为：SGD anchor `31.4409±4.2346`，GAM anchor `21.9328±2.9819`。字段中的 `raw` 仅表示 **trace normalization 之前**；估计器仍是 batch-mean CE 梯度先求均值、再平方、按 batch size 加权并除以样本数，并非逐样本 empirical Fisher。配置为 `ewc_fisher_source=fit`、batch 128、`fisher_split=false`，Task 1/2 分别覆盖 1259/997 个当前任务训练样本；该值依赖 batch 划分。`approximate_fisher_in_task=false`，所以这些列是 endpoint-only，没有 path/trajectory Fisher 平均。

若要复刻旧表的 $\mathcal B_2$ 列，现有已完成行也已留存同口径 proxy，无需新评估：A `28.189±0.477`、C `27.419±0.797`、D `24.027±1.231`、FS-EWCLoRA `17.895±1.510`；六行之外的 B3-B 为 `23.964±1.302`。缺失两行仍应为破折号。

$d^{\mathrm{cont}}_{QV}$ 的定义应写为

$$
d^{\mathrm{cont}}_{QV}=\left[\sum_{\ell,c\in\{Q,V\}}\left\|B^{(2)}_{\ell c}A^{(2)}_{\ell c}-B^{(1)}_{\ell c}A^{(1)}_{\ell c}\right\|_F^2\right]^{1/2},
$$

逐 seed 先求范数再聚合，且排除分类头。这个命名有实现依据：正式 runner 从 `task_00.pt` 严格恢复完整 network state；下一任务只扩展分类头，没有重建或清零 LoRA；learner 始终从同名 12 层 Q/V 因子计算 $BA$。配置同时关闭 `weight_merge`、`in_task_weight_merge` 和 `endpoint_weight_merge`。因此它是同一持续 LoRA 分支的 endpoint-to-endpoint 位移，不是 fresh Task-2 adapter 大小。

## 持续 GAM 行是否可复用

**可以。** 三个 `sequence_gam_seed{s}` 的 Task-1 checkpoint 都从对应 `anchor_gam_seed{s}` 精确恢复；其父 `task_00.pt`、`fisher_state_0.pt` 和父 metrics SHA 与 B3-D 同 seed 完全一致，Task-1 fit/test loader 的样本 ID、generator seed 和样本数也一致。表中必须取 `t01`/`task_01.pt`，不能取十任务 `task09` final。

该行应写成当前实现的 `GAM $\rightarrow$ GAM+EWC`（$\lambda=2000$），可命名 FS-EWCLoRA。当前 `ewc_project_grad=false`，不能沿用旧 ImageNet-R 行的 `P-GAM+EWC` 标签。正式训练还明确关闭 trajectory averaging；名为 `trajectory_fisher` 的离线诊断不改变这一点。

## 六行覆盖与最小缺口

按历史六行布局，已完成的是 Vanilla（A）、EWCLoRA（C）、FS-EWCLoRA（continuous GAM 的 t01）和 Task-1 shaping（D）。尚缺：

1. GAM anchor $\rightarrow$ GAM、$\lambda=0$（Shaping-only），3 seeds；
2. SGD anchor $\rightarrow$ GAM+EWC、$\lambda=2000$（Task-2 shaping），3 seeds。

因此补齐这张六行表的最小训练缺口是 **6 个 50-epoch Task-2 续训**；无需重训 anchors 或 continuous-GAM 行。现有 B3-B 是 GAM anchor $\rightarrow$ SGD、$\lambda=0$，不能填 Shaping-only。

六行布局不是完整的 $2\times2\times2$（Task-1 shaping × Task-2 shaping × EWC）设计。当前正式结果覆盖其中 5/8 格：A、B、C、D 和 continuous GAM。除上述两格外，完整八格还缺 SGD anchor $\rightarrow$ GAM、$\lambda=0$ 的 3 seeds；所以“六行已齐”也不能写成“完整三因素 factorial 已齐”。

## 建议 caption

> **Two-task, protocol-specific diagnostics of solution shaping and EWC with rank-10 Q/V LoRA on ImageNet-A.** Values are means over seeds 0, 1, and 2, with sample standard deviations. $\mathcal B^{QV}_1$ is the unnormalized endpoint batch-gradient Fisher proxy on the current Task-1 fit loader; it is distinct from the fixed-oldtrain256 per-example Fisher diagnostic. $d^{\mathrm{cont}}_{QV}$ is the endpoint-to-endpoint Frobenius displacement of the same continued effective Q/V LoRA weights during Task 2, excluding classifier heads. Current GAM+EWC is unprojected. All runs use FP32 with TF32 disabled and no trajectory averaging. Dashes mark unrun Task-2 branches; displayed Task-1 values for those rows come only from completed parent anchors. The six-row historical layout is not a complete $2\times2\times2$ factorial.

若与旧 ImageNet-R 六行表并列，应再加一句：两个 panel 的 optimizer、EWC 强度/归一化、LoRA 通道与生命周期均不同；ImageNet-R 的 $\mathcal B_j^{KV}$ 和 $r_{KV}^{\mathrm{fresh}}$ 不与本表的 $\mathcal B_j^{QV}$、$d_{QV}^{\mathrm{cont}}$ 作数值横比。
