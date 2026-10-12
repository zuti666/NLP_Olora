# ImageNet-A：C/D 外层探针，最终使用 g1−+g2

用户已确认本轮 g1 指下降探针梯度 g1−。本轮新增4次完整训练：C/D各在r=.25/1运行一次，seed2、rank10、rho=.05。已有 C/g2、D/g2、A_C 六条完整结果复用，不重复训练。

## 请求去重和方法定义

“沿用 A_C 探针、最终只用 g2”就是 C/g2；“沿用 A_C 探针、最终用 g1−+g2”就是本轮 C/sum，与再做一次 C/g2 加 g1− 完全相同。另补历史 D 外层位置下的相同 sum 更新作配对比较。

令 N(v)=v/(||v||+1e−12)，theta0是更新前的原参数：

- g0=grad L(theta0)。
- g1+=grad L(theta0+.05N(g0))，仅保留同一探测预算，不参与最终组合。
- g1−=grad L(theta0−.05N(g0))。
- C：qC=g0−g1−，theta2C=theta0+rN(qC)=theta0−rN(g1−−g0)。
- D：qD=g1−−g0，theta2D=theta0+rN(qD)。
- g2C/D=grad L(theta2C/D)。
- 新组 U=g1−+g2C/D；恢复theta0后执行一次 theta←theta0−eta U。

C和D的外层位移相反；两个外层探针都从原theta0出发，不是从首探针点出发。最终自然相加，不除以2，不保留 A_C 的额外 g0−g1− 项，不做投影、最终归一化或匹配。四次普通CE梯度、一次SGD；不计算g3或HVP。每个探针恢复完整状态并重放R0，最终RNG只推进到g0后的R1。

## 固定配置

| 项目 | 设置 |
|---|---|
| 数据 | ImageNet-A，train5981/test1519，固定类序0…199，10×20类 |
| 模型 | ViT-B/16 augreg2_in21k_ft_in1k，SeqLoRA rank10 |
| Seed | 2，每组从相同预训练初始化独立训练 |
| 训练 | 每任务50epochs、batch128，共2550次SGD更新 |
| 优化 | SGD，momentum/weight_decay=0 |
| LR | Task0=.04，其余任务=.02；逐任务cosine，最低=.004/.002 |
| 探针 | gam_grad_rho=.05；gam_grad_norm_rho=.25或1 |
| 方向选择 | probe_branch=C或D；update_direction=g1_minus_plus_g2；step_mode=natural |
| 计算 | 原生attention，FP32，TF32关闭，create_graph=false |
| 关闭 | EWC、Fisher估计/阻尼/归一化、权重融合、replay、SGV、HVP、梯度投影、最终范数匹配 |

lambda_flat=1，ewc_lambda=0，ewc_skip_task_fisher=true，approximate_fisher_in_task=false，in_task_weight_merge=false，task_endpoint_weight_merge=false。旧 beta/gamma 字段保留，但这个直接方向分支不使用它们。

## 新增作业

| 正式作业 | 外半径r | 最终方向 |
|---|---:|---|
| formal_C_g1minus_g2_r025_k10_s2 | .25 | g1−+g2C |
| formal_C_g1minus_g2_r1_k10_s2 | 1 | g1−+g2C |
| formal_D_g1minus_g2_r025_k10_s2 | .25 | g1−+g2D |
| formal_D_g1minus_g2_r1_k10_s2 | 1 | g1−+g2D |

各组先进行2任务×1epoch×1batch隔离短测，共4次。4个短测及其真实输入/Task0初始状态配对验收成功后，才释放4个正式作业。新suite共9项：4短测+1配对验收+4正式。

仅使用空闲GPU2–6，max_parallel=2，GPU0/1排除。每卡独立判断空闲，共享index/UUID锁，不中断既有队列。失败保留日志并停止新增接续，无自动改变参数或静默重跑。

## 已有结果和要回答的问题

AAA/FAA为任务等权百分比，六条历史运行均已核查10任务/500epochs/2550更新、退出码及矩阵复算；哈希索引在 historical_inventory.json。

| 方法 | r=.25 AAA / FAA | r=1 AAA / FAA |
|---|---|---|
| C/g2 | 66.647 / 49.793 | 65.774 / 49.192 |
| A_C：g2C+(g0−g1−) | 66.758 / 49.864 | 64.438 / 47.185 |
| D/g2 | 46.887 / 24.999 | 37.359 / 23.187 |

- C/sum对C/g2：加入下降首探针梯度是否进一步改善当前较强的外层更新。
- C/sum对A_C：比较两种最终组合；精确差异 U_Csum−U_AC=2g1−−g0。不能说两者只差一个方向修正系数。
- D/sum对D/g2：加入g1−能否改善较弱的外层分支，重点观察新任务学习与旧任务保持。
- C/sum对D/sum：最终组合相同，只改变外层位置的正负方向。

主要报告AAA、FAA、LA、BWT和遗忘，同时保存逐样本指标、完整任务矩阵、逐epoch学习曲线。每步保存loss0/loss1+/loss1−/loss2、梯度和最终U的范数、U对g1−/g2的余弦、有效权重位移、实际SGD幅度，以及真实梯度/forward/SGD/Fisher/融合/投影/g3/HVP计数。汇总保留U/g2范数比和U与g2余弦的中位数。

自然相加会同时改变方向和步长幅度；本轮检验实际组合效果，不把性能变化单独归因于方向。单seed为探索性结果。部分任务成绩仅标进度，完整预算及队列日志验收后才填最终AAA/FAA。

## 复现与状态

代码位于trainer/four_probe_train.py和trainer/direction_step.py，配置位于configs/。RUN_PINS.json固定代码、配置与计划。启动说明RUN.md，环境说明ENVIRONMENT.md。实时状态runtime/queue_state.json，日志runtime/jobs/JOB/console.log，接受的结果summary.csv/RESULTS.md。之前g0B和g1+(g3−g2)队列保持原配置与进度。

