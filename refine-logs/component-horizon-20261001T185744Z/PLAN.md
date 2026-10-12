# GAM 十任务分量筛选与后续机制验证

记录时间：2026-10-01。按照用户要求，先完整结束当前 formation/future-probe 实验，再自动接续本计划的第一阶段。当前实验的配置、训练进程和结果保持原状。

## 等待与执行顺序

父实验目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-formation-run-20261001T180712Z`。
必须等待其 **120 项注册任务**全部成功（12 短测、24 Task0 训练、84 后续更新诊断），父控制进程退出，再重新验收全部 120 份完成记录。随后自动执行：

1. 10 项两任务 GPU 短测，每任务仅 1 epoch、1 batch；五种方向各做自然步幅与匹配步幅版本。
2. 跨配置核验数据、初始化、新增分类头及 US 身份对照；全部通过才放行正式训练。
3. 9 项全十任务训练，每任务 50 epochs。
4. 对新 US 与 Full 的 Task0、4、9 checkpoint 做 6 项曲率诊断。

只使用 GPU 0–6 中实际空闲的设备：显存 <500 MiB、利用率 <5%，最多 7 项并行，与已有队列共享 GPU 文件锁。等待进程不分配 GPU。宿主机器和 PID 命名空间均已绑定；任何失败会停止新增任务接续，正在运行的任务可完成，保留全部证据。`runtime/PAUSE` 可暂停新增任务。第二、三阶段不会自动启动。

## 第一阶段固定设置

| 项目 | 设置 |
|---|---|
| 数据与任务 | ImageNet-A，原始固定类别顺序，200 类，10×20 类，不打乱类别 |
| 骨干与可训练部分 | ViT-B/16；LoRA rank10、分类头；未引入 alpha/rank 缩放 |
| seed | 2 |
| 内层半径 rho | `gam_grad_rho=0.05` |
| 外半径 r | `gam_grad_norm_rho=0.25`，主筛选固定 |
| GAM 强度 | `lambda_flat=1.0` |
| 投影、额外修正 | GAM gamma=0；无 SGV、梯度正交、额外漂移惩罚 |
| EWC/Fisher/融合 | EWC lambda=0；跳过任务 Fisher；任务内 Fisher、阻尼、归一化、任务内及端点权重融合全部关闭 |
| 优化 | SGD，momentum=0，weight_decay=0，batch128 |
| epoch与LR | 每任务50 epochs；Task0 LR .04→.004，后续 .02→.002，cosine Tmax50 |
| 重放与数据选择 | 无 replay、无新增训练子集；官方测试用于报告，不构造调参验证子集 |
| 数据顺序 | 每任务、每 epoch 单独固定随机种子 `seed*100000+task*1000+epoch`；测试隔离训练 RNG |

每条正式训练共 **2550 更新步**、299050 样本呈现。各任务训练样本数为 1259/997/993/596/426/260/357/317/398/378；实际最后 batch 保留。短测保持正式 cosine 调度的 Tmax50，不压缩调度。

## 分量与运行矩阵

使用原始 GAM 的四次原生梯度探测：

`g0 = grad L(theta)`；`g1 = grad L(theta + rho*N(g0))`；`theta_r = theta + r*N(g1-g0)`；`g2 = grad L(theta_r)`；`g3 = grad L(theta_r + rho*N(g2))`。

实际探测遵循冻结代码的归一化 epsilon、扰动和恢复顺序。所有方向保留四次探测，因此 U0 是四探测执行下的 clean-gradient 对照。固定原生 beta=(1,1,0)、gamma=0 以捕获 g3，随后选择下表更新方向；不使用原版 g1+g3 求和，也不额外做梯度投影。

| 方向 | 更新梯度 | 自然步幅 | 匹配有效步幅 |
|---|---|---:|---:|
| U0 | g0 | 新训练1次 | 新训练1次 |
| US | g1 | 共用右侧 US 身份对照 | 新训练1次 |
| UG | g0+(g3-g2) | 新训练1次 | 新训练1次 |
| UO | g2 | 新训练1次 | 新训练1次 |
| Full | g3 | 新训练1次 | 新训练1次 |

合计 9 次完整训练。每种方向都从头训练十个任务；任务数不是类别重新划分。

匹配目标定义为：**候选自身当前模型状态、当前 batch**，完成原生 GAM 恢复后，沿原生 US 方向执行名义 LR 的实际 FP32 SGD 所产生的有效更新长度。有效向量包含所有未缩放 LoRA `B@A` 及已见分类头 weight/bias。实际候选更新需匹配此反事实长度，逐步相对误差 ≤1e-3，记录系数、名义 LR、目标和实际范数。

原生 `lambda_flat=1` 路径实际执行 `clean + 1*(g1-clean)`；保留其 FP32 运算顺序，与理想数学 g1 的微小差异单独记录。US 匹配必须令系数严格等于名义 LR、参数更新与反事实逐元素相同。两任务 GPU 身份对照另要求自然/匹配 US checkpoint 参数相对差 ≤1e-4。该数值阈值在 GPU 短测前固定。

这与父实验的 Gmatch **使用独立 S 轨迹的参考步幅**不同；两者分别回答不同问题，不合并统计。

## 已有结果的复用边界

核验到旧纯 g3、r=.25、seed2、50/50 epoch：AAA=66.229609%，FAA=50.799%，LA=57.317%，BWT=-7.242222，遗忘=7.365556 pp。旧 g1 的可复用结果是 r=1，AAA=62.165531%、FAA=45.132%；没有本队列 r=.25、相同增强顺序的 g1/g0 配对基线。

新队列固定每 task/epoch 的数据与增强 RNG，旧实验使用原生全局 RNG。因此旧 g3 作为历史背景；本次另训自然 Full，以便与匹配 Full 做同批协议比较。US 的自然版本共用本批匹配 US，因为它逐步严格保持名义 LR 更新。完整来源见 `review/REUSE_AUDIT.md`。

## 输出指标与预先固定的选择规则

主要报告任务等权 AAA；同时报告 FAA、LA（各任务刚学完的准确率）、BWT、遗忘及全已见样本加权 AAA/FAA。保存逐任务准确率矩阵、每 epoch 训练 CE/准确率/LR、每步有效更新尺度及数据身份、每任务 checkpoint。分别观察 acquisition 与 retention，不能只按最终旧任务准确率解释机制。

单 seed 仅用于探索性筛选。UG/UO/Full 的匹配版本相对 US 若 AAA 至少 +1 pp、FAA 至少 -1 pp，可列为后续候选；候选按 AAA、FAA、固定 tie 顺序 UO/UG/Full 排序。所有 9 条完整验收后才输出 X。正负1 pp内仅表示本 seed 的数值接近，不代表统计等价。没有合格候选时不选择 X，检查尺度与学习率作用。

若 UG 单独保留收益，优先验证外状态的第二次局部响应；若 UO 更接近 Full，优先验证外层位置改变；若 Full 增益而单个分量弱，说明组合更新值得后续验证，**尚不能证明因果协同或 Hessian 中介**。

## 曲率诊断（6 项）

仅对新匹配 US/Full 的 Task0、4、9 checkpoint，在固定训练 batch 上计算原状态与外状态的 detached 方向 HVP、forward/central FD；rho 固定 .05，有限差分步长 .05/.025/.0125。Task0/4 至多4批，Task9至多3批（378训练样本）。记录两个响应与 HVP 的 cosine/相对误差、步长收敛、响应间 cosine 和范数比、同点损失/梯度一致性、模型/RNG 恢复。

曲率是 LoRA 因子与分类头参数坐标下的 Hessian，不是有效 B@A 坐标的 Hessian。若原生 fused attention 不支持二阶反传，仅诊断切换 math SDPA 后端并核同点一致性；显式不支持则记录 unsupported，不填写虚构 HVP 指标。OOM、NaN、恢复失败属于任务失败。有限 rho 的梯度差不能直接当成已证实的 Hessian 响应，高性能本身也不能证明该解释。

## 后续条件阶段（不加入自动运行清单）

确定 X 后，独立重训两任务 SS/XS/SX/XX，seeds0/1/2，全部使用自身状态 US 反事实匹配。比较旧任务遗忘与新任务 acquisition；Task1 在 epoch0/5/10/20/30/40/50 测 L0/a0/L1/a1，绘旧准确率对新准确率曲线。相同未来学习协议不意味着相同实际未来梯度，因为起始状态不同。

两任务明确后，三任务按结果仅做 SSS/XSS（formation），或 SSS/SXS/SSX（continuation），或 SSS/XSS/SXS/SSX/XXX（两者均有）；至少3 seeds。formation 稳定信号可扩5 seeds。r=1 的 S/X 鲁棒性及位置×方向2×2曲率诊断也留待候选确认后。

## 运行文件

- `suite.json`：25 项注册任务、依赖和命令。
- `handoff_guard.json`：父队列120项门控、父配置哈希与宿主 PID 命名空间。
- `runtime/handoff_state.json`：等待、短测、正式训练、完成或失败状态。
- `runtime/queue_state.json`：每项实际进程与结果状态。
- `results/screen_metrics.csv`、`results/screen_decision.json`：仅收入验收完成结果。
- `RUN_PINS.json`：部署时冻结源码、配置、协议、审查证据哈希。

CPU 独立验收：完整 native GAM 的五方向/两尺度、两任务生命周期、US 身份，以及验收器10正例/26负例、曲率正例/14负例通过。GPU 短测仍必须等父实验结束后真实执行，CPU 通过不表示新正式训练已开始。
