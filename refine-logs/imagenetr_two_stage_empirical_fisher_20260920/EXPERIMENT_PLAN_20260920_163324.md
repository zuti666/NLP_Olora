# ImageNet-R 两阶段继续评估计划

日期：2026-09-20。依据用户最新纠正，数据集固定为 ImageNet-R。本轮仅核查、CPU 汇总与制定离线评估方案；未启动模型评估、训练或后台队列。

## 结论与方法身份

保持现有训练结果，优先补齐 empirical Fisher 与旧任务损伤测量。不能只把 ImageNet-A 配置的 dataset 字段改名，也不能把 Ratio-FS 写成 GAM。

| 资产组 | 真实方法与协议 | 已有资产 | 本次用途 |
|---|---|---|---|
| 用户最早的六行 LoRA 表 | ImageNet-R；Adam/fresh K/V；GAM 与 P-GAM+EWC；seeds0/1/2 | 18份日志及汇总；目标运行未保存 checkpoint | 保留已核实的42个数字和修正后的含义；无法仅凭日志补算逐样本 EF |
| 新版六行 ImageNet-R 表 | 持续 Q/V LoRA；SGD/Ratio-FS；seeds3/4/5 | 6个 Task0 完整 checkpoint，18组 Task1 A/B/FC sidecar及原始 metrics | 最完整的同协议离线补评对象；表名必须保留 Ratio-FS |
| ImageNet-R 真 GAM 序列 | ewclora_normfisher_gam；SGD 为 base；rank10；seed0；多组学习率 | 已找到前两任务 A/B/FC sidecar与10任务 metrics | 可作 GAM 候选资产；冻结 backbone 身份及同配置 EWC 对照尚需补齐核验，不能与 Ratio-FS 的 EWC 行拼接 |
| endpoint-region pilot/heldout | SGD/SAM/Ratio-FS；A/B因子+分类头坐标 | 已有逐样本 raw EF JSON | 复用原始记录及实现，按原坐标解释；不能直接填入有效 Q/V 端点 EF 列 |

**严格 EWC vs EWC+GAM 的目标保留。** 当前没有核实出能立即组成这一受控对照的完整同协议资产。以下 Ratio-FS 补评矩阵对应“继续评估已有新版表”的可行范围，不是替换 GAM 方法的决定；任何 GAM 对照都须先通过 G01 的资产匹配。缺失不自动触发补训练。

## 两个待检验主张

| 主张 | 最少证据 | 失败时如何解释 |
|---|---|---|
| C1：shaping 后的起点及后续端点具有更低旧任务敏感度 | 同旧样本、同头范围、同有效权重坐标下，两端点未归一化逐样本 EF | 无下降就不宣称降低了该测量下的旧任务敏感度；因子坐标或归一化 trace 下降不能替代 |
| C2：敏感度差异伴随较小旧任务损伤，且新任务学习能力没有明显降低 | 相对自身起点的旧任务 loss/accuracy 变化、新任务 accuracy、有效位移，逐 seed 配对报告 | 更低 trace 但损伤未降低，结论只能是敏感度统计改变，不能称已证明减少遗忘 |

同一 EWC 规则仍会产生不同保护矩阵；这些对照不足以完全排除保护矩阵差异。无需为这次补评引入无关模型或新增大型基线。

## 可直接利用的新版表

精确主组：`ratiofs_l12c20_full_twostage_s345_v1`。ViT-B/16，持续 Q/V LoRA rank10，固定类别顺序，前20类→后20类；SGD、两阶段 LR=.02、每任务50 epochs、batch128、momentum=0、weight_decay=0；EWC λ=1.2e6、gamma=1，训练保护矩阵使用归一化及 damping=.2；Ratio-FS c=.20、rho=.05。最终以逐运行原始配置为准，评估不改这些训练参数。

| 角色 | setting | 第一阶段 | 第二阶段 | 既有 a11 / a21 / a22 | 既有 BWT | 既有有效位移 |
|---|---|---|---|---|---:|---:|
| 主对照 | control | SGD | SGD+EWC | 89.36 / 85.90 / 87.82 | -3.46 | .4272 |
| 主方法 | fs_ewc | Ratio-FS | Ratio-FS+EWC | 89.47 / 85.74 / 88.19 | -3.72 | .4617 |
| 起点作用补充 | factored_s1 | Ratio-FS | SGD+EWC | 89.47 / 85.96 / 87.82 | -3.51 | .4483 |

数值为seeds3/4/5均值，准确率为百分比，BWT为百分点；标准差见既有[新版表](../lora_full_two_task_drift_audit_20260920/tables/new_imagenetr_ratiofs.tex)。主方法 a21/BWT 没有优于 EWC：差值约 -0.16/-0.26 pp；新任务 a22 约 +0.37 pp。补测不得以得到正面结果为筛选条件。

机器可读的计划配置：[evaluation_protocol.json](evaluation_protocol.json)；其`dispatch_enabled=false`，不是已执行结果。

原始训练脚本：[/data/140-1/users/liying/CopyFlatCV/scripts/run_verify_imagenet_r_1x1_lora_ratio_full_twostage.sh](/data/140-1/users/liying/CopyFlatCV/scripts/run_verify_imagenet_r_1x1_lora_ratio_full_twostage.sh)。精确逐运行 metrics、anchor 与 endpoint 路径见 [EXISTING_RESULTS.md](EXISTING_RESULTS.md)、[existing_results_manifest.json](existing_results_manifest.json)。

Task1没有完整 task_01.pt，不能声称可恢复训练优化器状态。离线评估使用各自 Task0 完整 checkpoint 的冻结 backbone，加 Task1 的 A/B 与分类头 sidecar；先核对冻结项、旧分类头、前向输出和原 accuracy，再使用重建模型。不要从在线可变 pretrained 权重重新下载 base。

## 三个评估块

### E1：重建模型与统一已有指标（必须）

- 优先 seed3 的 control/fs_ewc：恢复 Task0/Task1；核对24个 Q/V块、形状、因子顺序、缩放系数、分类头、SHA和原训练配置。
- 保留原表 accuracy；另列统一 FP32 重评值及差异原因，不能静默覆盖。重新测 a11、a21、a22，以及旧任务 local20 和第二阶段 seen40 的 loss/accuracy。CIL accuracy 与 local accuracy 分列。
- 旧任务损伤主量：`D_old = CE_old_local20(endpoint) - CE_old_local20(anchor)`，train/test分别测。seen40 loss另列；若要将其前后差异解释为训练损伤，须恢复训练前40类头的真实初始化，不能临时随机扩头后当原起点。
- 有效位移必须是两端点的净权重差，排除新增分类头：
  `r_QV = sqrt(sum_blocks ||s*(B1 A1 - B0 A0)||_F^2)`。
  若代码无额外 LoRA scaling，则 s=1。连续 adapter 的第二阶段 BA0 一般非零，不能用最终 ||BA1||，也不能用 ||ΔA||/||ΔB||代替。冻结 base 相消；若发现 base 变化，则改为完整有效权重之差并查明原因。
- 不运行 optimizer.step，不扩任务数、不重训。输出 per-seed CPU复算与模型前向 sanity。

### E2：两端点逐样本 empirical Fisher（必须）

统一定义 `g_i(theta) = grad_{W_eff,QV} CE(z_old20(x_i;theta), y_i)`，
`F_emp = (1/N) sum_i g_i g_i^T`，`tau = (1/N) sum_i ||g_i||_2^2`。

- 旧任务固定256张训练图，按原类别比例分层抽样，所有方法/端点使用相同ID和确定性 test transform；保存图像路径、标签、类别映射、预处理、抽样算法/种子与输入哈希。不是每类均衡子集，且不是独立 held-out 声明。
- 同一批旧数据测 `tau_old@T0`、`tau_old@T1`；这两个量直接回答旧知识敏感度是否改变。
- 另用新任务固定256张训练图，在第二阶段端点计算 `tau_new@T1`，使用新任务20类 logits与本地标签。它说明新任务拟合几何，不能代替 `tau_old@T1`。
- eval模式，CE不含EWC或shaping项，逐样本梯度先平方再平均；无trace归一化、无damping。FP32前后向、float64求和。允许微批并行，但必须与单样本 autograd 验证一致。
- primary 坐标是24个有效 Q/V矩阵，头参与前向但其梯度不计入 trace。因子+head trace仅作独立补充，不能混在同列。训练保护 Fisher保持原状，保存其原始元数据作区别。
- sanity：原LoRA与有效权重展开后的logits一致；微批1与8统计一致；直接单样本梯度抽检；更换A/B等价缩放后有效坐标测量保持一致；确保每次测量后参数哈希不变。
- 首轮seed3，sanity通过后扩展已有seeds4/5，不筛选“效果好”的seed。两主行共12个旧任务端点测量+6个新任务端点测量，即18×256=4608次样本梯度统计（另加sanity与前向损失）。无需新增训练。

### E3：汇总与必要补充（主表必须，额外曲线可选）

- 主表保留两主行；完成支持性扩展后，再将factored_s1加入为解释起点作用的第三行。报告三seed均值±样本标准差、逐seed配对差、loss增量、净有效位移与明确下标的raw EF。
- 对旧样本使用相同head范围，关联损伤、位移和敏感度；不能从总体trace单独推断实际更新方向上的二次型。
- 如端点结果不足以解释实际位移，才加两任务有效权重直线9节点的单位位移路径敏感度；方向二次型用逐样本 `(g_i^T Δ)^2`，跨block求和后再平方。不是沿A/B因子直线，也不称真实训练轨迹。
- 既有全模型表仅作附录支持：drift排除新增头的出生能量；QV子空间可与LoRA作同坐标对照，全backbone trace单独报告。原始full端点来源未完全锁定时不拼接主表。
- 不默认重做随机半径扫描、不重训四格、不重跑10任务序列。

## 已有 EF 及脚本的复用边界

- 已有[raw EF来源与复算](EXISTING_EMPIRICAL_FISHER.md)：pilot同组anchors的SGD/Ratio-FS trace分别为188.978±17.361、155.687±14.016，但坐标为A/B+分类头、旧样本40类CE、160张图；只有Task0测量。
- 独立 heldout_v2组也有已完成EF与后续训练结果；其模型在不同80/20划分上训练，不能将数字拼入上述六行表。其原汇总没有通过预设主要机制主张，须保留这一结果。
- `CopyFlatCV/scripts/run_imagenetr_endpoint_region_eval.py` 有逐样本EF实现，但默认因子+头坐标；`run_imagenetr_endpoint_region_pipeline.sh` 默认all包含训练，不能拿来直接当纯离线评估入口。
- `O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py` 有有效Q/V逐样本EF实现；数据、完整checkpoint loader、任务范围均需适配R及Task1 sidecars。原 `endpoint_a.py` 对SGD continuation有专用守卫，不能只改dataset后运行。
- 用户给的 `run_imagenet_r_task1_fqa_lora_factor_adam_gam.sh` 只是factor扰动评估wrapper，当前其被调用父脚本缺失；也不是上述raw EF的即用入口。

## 执行顺序与预算

| 顺序 | 内容 | 状态/门槛 | 成本 |
|---|---|---|---|
| 1 | 资产与表格来源核验；真GAM同协议对照继续定位 | 已完成当前库存核查；GAM配对仍未确认 | CPU，无训练 |
| 2 | 适配纯评估入口，seed3两主行恢复与数值sanity | 待实施；模型身份或指标不符先修复测量，不替换成别的协议 | 4个端点恢复+少量前后向 |
| 3 | 两主行seeds3/4/5统一EF、loss、drift，汇总表 | 待评估；sanity通过即可扩seed，不要求结果为正 | 18组EF，每组256样本；全量前向另计 |
| 4 | factored_s1与必要方向诊断 | 支持性扩展 | 额外3个endpoint；旧/新任务EF各3组 |

新增训练预算=0。GPU小时尚未测量，不沿用ImageNet-A吞吐。先记录R上单模型恢复、256样本EF和全量loss耗时，再估剩余成本。此文件没有GPU分配、自动启动命令或后台队列。

## 交付与边界

- 当前交付：[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md)、来源清单、已有EF CPU复算、方法/协议审计与适配规范。
- 后续评估应输出逐seed JSON/CSV、model/data/source hash、坐标/损失范围、checkpoint映射、sanity结果、统一表格LaTeX。
- 本轮不宣称新增raw Q/V EF已经完成。ImageNet-A方案被用户纠正为不适用，本计划为当前有效方向。
