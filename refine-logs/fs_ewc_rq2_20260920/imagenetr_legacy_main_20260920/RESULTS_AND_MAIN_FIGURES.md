# RQ2：以真实旧 ImageNet-R 为主线的图表与证据核查

2026-09-20。依据用户最新要求，主表采用真实历史 ImageNet-R 结果。下面三张既有图来自另一批 ImageNet-A checkpoint；它们的数值是真实测量，但不能改标签后作为 ImageNet-R 结果。本轮只核查来源、重新绘制已有曲线和整理最小扩展方案；没有启动训练或修改 GPU 队列。

## 1. 三张既有图到底来自哪里

共同来源：[diagnostic_config_recovery1.json](../execution/diagnostic_config_recovery1.json) 的 `data_root=/data/115-2/users/liying/datasets/imagenet-a`；绘图代码为 [postprocess.py](../execution/postprocess.py)，测量收据在 `execution/results_seed0_recovery1/{sgd,gam}/`。结果与文件哈希由 [MANIFEST.json](../execution/report_seed0/MANIFEST.json) 记录。本轮从 CSV 重新提取的数值、阶段变化和11项来源哈希见 [SOURCE_AND_NUMBERS.json](SOURCE_AND_NUMBERS.json)，可由 [CPU脚本](recompute_figure_findings.py) 重建。

| 既有图 | 绘图数据 / 原始收据 | 横轴的真实含义 |
|---|---|---|
| [Figure_B_new_deterministic.png](../execution/report_seed0/Figure_B_new_deterministic.png) | [B_neighborhood_deterministic.csv](../execution/report_seed0/B_neighborhood_deterministic.csv)，两方法 `neighborhood/summary.json` | 在各自 Task0 起点周围，沿共享随机单位方向施加的半径 r；不是学习任务数 |
| [Figure_B_bridge.png](../execution/report_seed0/Figure_B_bridge.png) | [B_actual_direction_bridge.csv](../execution/report_seed0/B_actual_direction_bridge.csv)，两方法 `bridge/summary.json` | 沿历史 Task0→Task9 累计位移方向重新缩放得到的 r；不是逐任务 checkpoint |
| [Figure_C_sequence_and_path.png](../execution/report_seed0/Figure_C_sequence_and_path.png) | [C_current_FP32_task_trajectory.csv](../execution/report_seed0/C_current_FP32_task_trajectory.csv)、[C_path_summary.csv](../execution/report_seed0/C_path_summary.csv)，`trajectory/summary.json`、`path/summary.json` | 第一幅是真实已完成 Task0…9 的端点；第二幅是这些端点的位移—损伤关系；第三幅只有最终端点直线积分 |

Task0 权重来自 `CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/{sgd,gam}/seed0/checkpoints/task_00.pt`；Task1…9 来自同一根目录下 `stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/{sgd,gam}/T10/seed0/checkpoints/task_01.pt…task_09.pt`。完整绝对路径逐项列于上述诊断 JSON。

当前绘图优先使用重新评估的完整 FP32 端点，不是 CSV 中另存的历史 TF32 参考。C 的两条序列分别为 SGD→SGD+EWC、GAM→GAM+EWC；它不是“GAM 起点→相同 SGD 后续训练”的两任务干预。

## 2. 旧 ImageNet-R 主表与这三图的配置差别

| 项目 | 用户指定旧 ImageNet-R 主表 | 当前三张 B/C 图 |
|---|---|---|
| 数据 | ImageNet-R，前两个20类任务 | ImageNet-A，10×20类；Task0 train1259 / test307 |
| 主干 | `vit_base_patch16_224_in21k` | ViT-B/16，历史权重标识 `vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| LoRA | rank10，12层 K/V，每任务新 adapter，任务末累积 | rank10，12层 Q/V，24个有效权重矩阵；本次诊断固定 base/head |
| 优化 | Adam，lr=0.0005，weight_decay=0.005 | SGD，Task0 lr=.04，后续=.02，momentum=0，weight_decay=0 |
| 训练预算 | 50 epoch/task，batch128，cosine | 50 epoch/task，batch128，cosine |
| GAM | rho=.05，normrho=.5，gamma=.03 | rho=.05，normrho=.2，gamma=.1 |
| EWC | lambda=1e7，累积 gamma=1 | lambda=2000，累积 gamma=.9，训练 Fisher 按 trace 归一化 |
| 重复 | seeds0/1/2 | training seed0；16个方向不是16个训练 seed |
| Fisher 列 | 当前任务的 batch-mean-gradient Fisher proxy trace | 固定旧任务 train256，逐样本 true-label CE 梯度平方均值，未归一化 raw empirical Fisher |
| 已有覆盖 | 配置 sessions=10，但实际 stop_task=2 | 真实 Task0…9 checkpoint 完整；路径积分目前只到 Task9 |

旧主表原配置：[control seed0](/data/115-2/users/liying/low-rank-cl-new/configs_sweeps/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/seed0/control.json)、[setting1 seed0](/data/115-2/users/liying/low-rank-cl-new/configs_sweeps/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/seed0/setting1.json)。三图所用训练配置和恢复说明见 [TRAINING_PROTOCOL.md](../execution/TRAINING_PROTOCOL.md)，其中任务状态段落有历史时间戳，不能替代完成后的结果收据。

新诊断统一固定 Task0 的20类分类头；CE/accuracy 使用确定性 Resize256 bicubic→CenterCrop224→ToTensor，不做 Normalize。raw Fisher 在同一分层256个训练样本上计算，FP32、TF32关闭。邻域的参数坐标是全部24个有效 Q/V 矩阵，维数14,155,776；不是对 LoRA 因子 A/B 分别归一化。

## 3. 两张 B 图的结论为什么可以同时成立

随机图测量许多方向的统计量，bridge 图测量两条训练产生的特定方向。较低平均 trace 不等于所有方向或测试损失都更低。现有结果应同时报告：

| 指标 | SGD | GAM | 解释 |
|---|---:|---:|---|
| 随机 r=20，训练子集 raw trace 均值 | 2837.330 | 1452.410 | GAM 训练侧敏感度下降 |
| 随机 r=20，测试正损伤均值 G | .080799 | .087530 | GAM 平均测试损伤反而较高 |
| 随机 r=20，测试正损伤 Q95 | .133647 | .128701 | GAM 测试尾部在此半径略低；不能与均值混用 |
| 固定 SGD 真实方向，r=1，测试 signed D | .147516 | .111834 | 同方向、同半径下 GAM 起点损伤更低 |
| 固定 GAM 真实方向，r=1，测试 signed D | .018917 | .013447 | 同方向、同半径下 GAM 起点损伤更低 |

最后两行的两列表示 **anchor**；随机图两列表示各方法起点的方向统计。bridge 内曲线颜色代表 **direction source**，左右面板分别是 anchor，且原图左右纵轴自动缩放。必须固定同一方向逐项比较 anchor，不能只看每个面板内两种颜色的差距就解释为 shaping 改变了起点。

在所有六个非零随机半径，训练 G 更低，而测试 G 都略高。测试 G 的 prefix-safe 容忍半径，在 epsilon={.001,.005,.01,.02,.05} 下两方法均为 {1,2,5,10,10}；训练 G 的容忍半径也相同。原始证据见 [B_tolerance_deterministic.csv](../execution/report_seed0/B_tolerance_deterministic.csv)。因此“两张图足以支持所有有界扰动下保护区域普遍扩大”不成立；它们支持训练侧平均 raw sensitivity 降低，以及在已测真实方向上的有利起点效应。更宽的全局保护区域不是已经测到的结果。

随机 r=20 比真实方向 r=2 损伤小也不矛盾。定义

`u_b=(W_b,9−W_b,0)/||W_b,9−W_b,0||`，`W(a,b,r)=W_a,0+r*u_b`。

所有方向都按全部有效矩阵的全局范数归一化，故同 r 保证位移大小相同；它没有保证方向同样敏感。随机方向把能量分散到约1416万个坐标；真实学习方向是数据驱动的结构化更新。此外，随机稠密矩阵扰动不一定处于 rank10 LoRA 的可实现集合。不能把这一几何解释当成在测试集验证的 Hessian 二次近似。

最终真实 r 分别只有1.1976和1.4005，所以 bridge 的 r=2 已经沿最终方向外推到原始位移的1.670倍和1.428倍。它不代表“学习到第二任务”。

## 4. C 可以显示真实学习阶段；新增 r(T) 表示什么

可以。以 **T=已完成任务总数** 为横轴：T=1 为完成 Task0 的 anchor，T=2 为完成 Task1，T=10 为完成 Task9。固定各方法自己的 Task0 基准：

`D_T = L_0(W_(T−1)) − L_0(W_0)`；`r_T = ||W_(T−1) − W_0||`。

r_T 是相对初始旧任务端点的实际累计位移范数，不是各步范数的和，也不是损伤容忍半径 r_hat_epsilon。已有权重/trajectory CSV 足够绘制 r_T；若想画“每个学习阶段的容忍半径”，必须明确是否换 anchor、用哪个旧任务，并重新做各阶段邻域扫描，目前没有这组曲线。

| 最终 T=10，现有 ImageNet-A seed0 | SGD 序列 | GAM 序列 |
|---|---:|---:|
| Task0 fixed-head test accuracy | 80.1303% | 85.9935% |
| Task0 signed test damage | .260131 | .034347 |
| 实际 r_T | 1.197584 | 1.400473 |
| 端点直线路径 S_bar | .368058 | .014793 |

SGD 损伤总体随任务增加，但 T=2、T=5 有回落；GAM 在 T=7 达到约.07335，T=10 回落到.03435。因此应写“后续学习造成不同程度旧任务损伤，GAM 在更大累计位移下损伤较小”，不能写成“两条损伤曲线随任务严格单调增加”。

S_bar 来自 Task0→Task9 **端点直线段** 上的9节点 Simpson 积分，以固定 train256 的逐样本方向梯度计算；不是沿训练 optimizer steps 积分，也没有 T=2…9 的 S_bar 测量。原 C 图的这一柱形图是终点诊断，不应连成十任务敏感度曲线。

阶段重绘：[Figure_C_stage_aware.pdf](figures/Figure_C_stage_aware.pdf) / [PNG](figures/Figure_C_stage_aware.png)；独立位移图：[Figure_r_vs_learning_stage.pdf](figures/Figure_r_vs_learning_stage.pdf) / [PNG](figures/Figure_r_vs_learning_stage.png)。两图内明确标注 ImageNet-A，供检查旧图含义与排版，不作为用户指定 ImageNet-R 主结果。

## 5. 正文的一张表、两张图应该回答什么

**表：What does shaping change?** 使用已核实的旧 ImageNet-R 六行表，核心比较 control 与 setting1（普通起点/ shaped 起点，后续均 Adam+EWC）。保留 a11、a21、a22、BWT、Fisher proxy 和有效位移，不只保留有利列。[已有均值±SD表](../execution/table_revision_20260920/imagenetr_legacy_mean_sd.tex) / [18个日志与42项数值核验](../execution/table_revision_20260920/USER_TABLE_PROVENANCE.md)。

19.885→10.946 为旧任务端点 batch Fisher proxy；87.9467→91.8233 为第二任务后的旧任务 accuracy。a11 同时提高90.7267→93.3433，故保持改善还应报告 BWT −2.78→−1.52（+1.26pp），不是把全部+3.88pp都归于减少遗忘。a22 为83.45→82.5667，仍需保留。位移 .01056→.39205 表明这组对照的改善不能单靠更小位移解释。表内 full/setting2 的 P-GAM 是另有投影的更新，不能与 Task0-only shaping 混称同一开关。

**邻域主图：What changes under matched bounded displacement?** 建议三幅：①测试 G_mean / G95 随 r；②旧任务 raw empirical Fisher 随 r，标清训练/测试样本；③真实方向交叉扫描，固定方向比较两个起点，两个面板共用纵轴。若同一协议下确实测出 r_hat_epsilon 分离，再把③改成容忍半径曲线，bridge 移到 inset/附录；若半径相同，应保留相同结果，标题不要预先承诺“larger protected region”。训练损伤和详细容忍半径表可以放附录。旧 Fisher quadratic、linear correction 不能替代 raw Fisher。

**序列主图：Does the advantage persist during actual learning?** 推荐2×2：①旧任务 accuracy vs T；②signed D_T vs T；③S_bar vs T（只画已测节点；只有最终值就画最终柱图）；④r_T vs T。这样新增的 r(T) 直接进入序列图，不需要正文第三张独立图。位移—损伤散点放附录。各阶段固定旧任务评估数据和 head，并在 caption 说明 full-method continuation 还是 Task0-only shaping continuation。

以上版式可服务旧 ImageNet-R 主线，但现成 ImageNet-A 数值不能填入 ImageNet-R 图。当前证据支持“特定设置下，更大实际位移仍伴随较小损伤”；尚不能宣称在统一 ImageNet-R 协议中完成了“raw neighborhood sensitivity下降→普遍容忍半径扩大→实际学习受保护”的完整闭环。

## 6. 已找到哪些真实 ImageNet-R 可复用资产

| 历史来源 | 已有内容 | 补测边界 |
|---|---|---|
| 精确旧主表 `verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7` | seeds0/1/2，六组两任务表/日志 | 实际 stop_task=2，save_ckp=False；未找到匹配权重。不能从 scalar drift 还原十任务累计位移；第二任务 endpoint loss 是当前新任务 loss，不能冒充旧任务损伤 |
| `best_gam_projected_4datasets/best_rho0p05_normrho0p5_lambda1e7` | ImageNet-R seed2、十任务 accuracy 三角矩阵、每次新 adapter 的位移/Fisher proxy | full projected 方法；没有旧任务 loss，也未保存 checkpoint。单步平方能量缺交叉项，不能相加得到相对 Task0 的累计 r_T |
| `verify_imagenet_r_phase5x5/phase5x5_s012` | 五组×三seed，完整十任务 accuracy | normrho=.2，前五任务 shaping；协议不同，缺权重和机制结果 |
| `imagenet_r_task1_fqa_adam_gam/task1_fqa_s0_fulltrain_k100_r0to1_fine_firstorder_r1` | seed0 Adam/GAM Task0 权重，27半径0…1、每半径100方向的历史 FQA | normrho=.2，与主表 .5 不同；有真实 R 权重可离线扩展，但原 quadratic proxy 不是邻域 raw Fisher |

上述旧仓库根为 `/data/115-2/users/liying/low-rank-cl-new`；表格路径分别位于 `result_summary/` 或 `configs_sweeps/`。FQA 汇总优先读 [fqa_radius_diagnostics.csv](/data/115-2/users/liying/low-rank-cl-new/result_summary/imagenet_r_task1_fqa_adam_gam/task1_fqa_s0_fulltrain_k100_r0to1_fine_firstorder_r1/fqa_radius_diagnostics.csv)：54行，两方法×27半径。`fqa_curve.csv` 的108行含 log/trainer 重复来源，不能当成200方向。

另外检查了已索引的 CopyFlatCV 真 GAM ImageNet-R 十任务 sidecar。最接近的两条序列虽有完整 A/B/fc，但基线是 **SGD、lambda=0**，GAM 分支是 **lambda=2000、归一化 Fisher**；不能称为只改变 GAM 的匹配对照。它们的模型别名相同，历史冻结 backbone 哈希未保存，尚需端点恢复校验。详细路径和可复用边界见 [IMAGENETR_ASSET_CHECK.md](IMAGENETR_ASSET_CHECK.md)。

本轮未强制重跑主表、未把另外的 ImageNet-A 多seed队列当作 ImageNet-R 补实验。最小扩展有两种明确不同的口径：

1. **复用旧 R 资产，作为协议明确的补充证据。** 已保存的 Adam/GAM `.05/.2` Task0 权重可先补邻域 raw Fisher；十任务候选可在恢复通过后离线补测实际轨迹，但目前缺少同 EWC 设置的匹配基线，SGD无EWC 与 GAM+EWC 只能报告复合方法比较。这一路不能自动补齐主表的因果机制链。
2. **让表与两图严格对应旧主表协议。** 原 `.05/.5` 精确批次未保存权重，需按旧 Adam/KV/50epoch/lambda1e7 配置重建 control 与 Task0-only shaping，先复核两任务指标，再保存端点向后扩展。后续保持相同 Adam+EWC 才能回答“起点改变的优势是否持续”；full projected 方法另列。新任务 accuracy 同步报告，避免用可塑性下降换保持。先 seed0 验证，再按需补 seeds1/2，不以图形好看决定是否保留运行。

两种扩展均应固定评估协议，补 G、raw Fisher、D_T、r_T 和若干实际端点的 S_bar。R 的扫描半径应覆盖其自身实测位移范围，不直接照搬 A 的 r=20。新增结果单列为复现/扩展，不能宣称恢复了未保存的历史权重。当前判断是：旧 R 主表已经有真实证据，统一 R 机制闭环尚未完成；现有 ImageNet-A B/C 是另一数据集上的部分支持与边界证据。

新图已完成原始数值核对、PNG渲染自检和独立同系列图表复核，结论为 pass，无必须修改项：[FIGURE_REVIEW.md](figures/FIGURE_REVIEW.md)。这是图表范围内的复核，不是对跨数据集完整机制链的认可。PDF为矢量输出，PNG为300dpi；LaTeX插入片段已提供，未进行论文编译。
