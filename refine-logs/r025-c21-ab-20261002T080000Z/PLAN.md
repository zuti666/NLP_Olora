# ImageNet-A：r=0.25，跨探测点残差 A/B 原生步长对照

用户最新要求仅新增 A、B 两组原生对照。附件的 C/D/E、匹配步长版本、多 seed、r=1 均不在本队列。这里“原生”指使用已有日程的名义学习率，不额外做方向或有效步长归一化；保留历史对照的全部四次探针计算。

| 组 | component_rule | 最终更新方向 | 主要比较 |
| --- | --- | --- | --- |
| A | c21_base | g0 + (g2 − g1) | 与 g0 比，跨点差分回加是否有收益；与 g2 比，回加基点是否影响收益 |
| B | c21_perp | g1 + c21 − a g1 | 与 g1 比，正交残差是否有用；与 g2 比，去除平行残差后保留多少收益 |

c21 = g2 − g1；a = Σp〈g1,p,c21,p〉 / Σp‖g1,p‖²。投影在全部实际可训练 LoRA A/B 和完整 seen FC 参数坐标中统一计算，不逐层投影，不在有效 BA 坐标计算。点积和平方和使用 FP64；若全局 g1 恰为零，a=0；非有限值直接失败。B 中此投影是本次有意施加的干预，原 GAM γ 正交修正保持关闭。

## 完整公共配置

ImageNet-A 原 train/test；固定类别顺序 0…199，10×20 类；seed2。
ViT-B/16，augreg2_in21k_ft_in1k；LoRA rank10，只训练 LoRA 与分类头，原实现无 alpha/rank 缩放。
外半径 gam_grad_norm_rho=0.25，内层 gam_grad_rho=0.05；gam_adaptive=false。
原优化器捕获探针的 β=(1,1,0)、γ=0；最终梯度由上述规则显式替换，β 不再定义最终更新。lambda_flat=1。
EWC λ=0，ewc_skip_task_fisher=true；Fisher 估计、阻尼、历史积累、实际权重融合、SGV、漂移约束、replay 均关闭。
每任务 50 epochs，batch128，不丢弃末批；SGD momentum=0、weight_decay=0。
首任务 LR .04，后续 .02；每任务 cosine T_max=50，最小 LR 为起点的 .1。
训练 workers8，评估 workers4。每步 clean CE+四次 GAM 梯度探针，共5次前后向；无额外诊断 backward。
每个任务/epoch 固定数据随机种子 seed*100000+task*1000+epoch，并与模型随机流隔离。评估恢复训练 RNG。

## 实现与记录

trainer/component_horizon.py 包装冻结原生 GAM 流程，只替换最终更新方向。
保留旧检查：四次 closure、五次 CE、初始化/类别/新增 FC/批次增强身份、无 Fisher/融合、名义 LR、实际 SGD 参数更新。
新增 c21 的全局投影系数、平行/正交范数、正交误差、c10/c21 夹角、各梯度范数；有效 LoRA BA+完整 FC 步长只记录，不用于缩放。
每任务保存 checkpoint、完整 seen-class 评估矩阵、逐 epoch 指标和逐步 trace。
主要指标 AAA_task_macro；次要 FAA_task_macro、LA、BWT、forgetting。另报 sample-weighted AAA/FAA，严格区分口径。

## 基线与结果复用

复用 /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-recovery-20261001T230000Z 的9条完成训练；重新校验完成凭据、原始trace和checkpoint。自然 g1 与匹配 g1 共用其已验证的名义 LR 身份对照。

| 自然步长基线 | AAA | FAA |
| --- | ---: | ---: |
| g0 | 60.947 | 38.893 |
| g1 | 63.828 | 45.120 |
| g2 | 66.586 | 50.720 |
| g3 | 67.179 | 50.845 |
| g0+(g3−g2) | 65.230 | 48.663 |

原恢复队列的独立 curvature 诊断失败不构成本次依赖：A/B 不使用 HVP。新队列只读取完成训练基线，不修改或重启历史队列。

## 启动与验收

1. CPU 数值/集成检查：全局投影反例、退化/非有限值、真实原生 GAM+SGD 更新。
2. 3个短测：A、B、g1 各2任务×1 epoch×1 batch。检查新旧数据/初始模型/新 FC/名义 LR 配对；新旧 g1 学得参数相对误差≤1e−4。
3. 短测全部通过后，自动并行启动 A/B 两条完整10任务训练，每条2550步、500 epochs。无后续其他正式实验。
4. GPU池0–6；每次启动锁定共享GPU租约，实时检查使用显存<500MiB、利用率<5%，绑定UUID；不抢占。若暂时无空卡则等待。失败停止新增启动，保留进行中的实验和证据。

复现：固定 Python /home-local/liying/.conda/envs/Pilot_new_local/bin/python，部署目录运行 start_pipeline.py 一次。单条精确命令、环境、config SHA 见 suite.json。完整完成后自动生成 results/comparison.csv 和 results/status.json；运行中不得将部分任务成绩称作完整 AAA/FAA。

## 预先限定解释

A 优于 g0 支持该跨点差分作为修正项具有经验收益；不能单凭准确率推断 Hessian 因果机制。
B 接近 g2 且优于 g1 支持去除平行残差后仍保留收益。但 B 的总范数及有效 BA 步长依然可能改变，原生步长比较不能独立排除尺度影响。
所有结论仅限 r=.25、seed2 的探索性证据，不把接近当作统计等价。
