# ImageNet-A：补齐历史 r=1 七方向表

只新增 seed2 的三条完整训练：g0、g2、g0+(g3−g2)。历史四条原版、g1+g3、g1、g3经原始矩阵与配置/凭据哈希核验后复用。短测与 r=.25 的另一随机数协议结果不填入本表。

| 固定项 | 设置 |
|---|---|
| 数据 | ImageNet-A，200类，固定类序，10任务×20类，无 replay |
| 模型 | 原 frozen ViT-B/16 augreg2_in21k_ft_in1k，LoRA rank10 |
| 随机种子 | seed2，原生全局 RNG、原生 DataLoader；不按任务/epoch重播种 |
| 训练预算 | 每任务50 epochs，batch128，2550更新，299050样本次 |
| 优化器 | GAM探测＋原生SGD；momentum=0、weight_decay=0 |
| 学习率 | 首任务.04，后续.02；cosine，最低.004/.002 |
| GAM扰动 | 内层 rho=gam_grad_rho=.05，外层 r=gam_grad_norm_rho=1；非adaptive |
| 最终方向 | g0；g2；g0+(g3−g2)，均不额外平均或范数匹配 |
| EWC/Fisher/融合 | ewc_lambda=0、ewc_skip_task_fisher=true，Fisher估计及两种权重融合关闭 |
| 其他历史项 | lambda_flat=1；保留 mechanism_eval/mechanism_grad_stats；其余额外评估关闭 |

五条 GPU 短测均为前2任务×1 epoch：原生 g1 控制、g1 hook 控制、三个待补方向。短测缩短scheduler预算，只验实现、预算、真实训练更新与数据/RNG，不作为50epoch性能。

正式三条必须等待五条短测全部完成及 GPU_SMOKE_PROOF.json 通过。控制组要求CNN矩阵相同、输入/标签及RNG序列逐batch相同、全部checkpoint tensor满足预声明atol=1e-7、rtol=1e-6；记录是否实际逐位一致。CPU证明还包含真实canonical GAM与独立无hook原生参考。

使用 frozen source，不修改旧脚本、配置、失败凭据或旧42项队列。训练optimizer实例上选最终方向，其他机制诊断的GAM保持原生行为。每batch仍为一次clean梯度与四次GAM梯度探测，共5次CE；不增加训练forward/backward或随机抽样。原生FP32 clean+(selected−clean) projector及扰动还原后SGD均逐步验收。g0因此是同探测流程的原点梯度方向对照；UG是有限扰动梯度差分，不能直接称为Hessian结论。

queue只租用真实主机上显存<500MiB、利用率<5%的空闲GPU，使用共享GPU lease、UUID复核、PID/starttime/namespace身份与唯一controller锁。当前允许物理GPU0–6；忙卡不占用。每项运行写完整config/source/runtime/environment凭据，实际模型checkpoint在新completion中绑定哈希。任何非有限训练loss、预算或hash/公式不符均停止后续放行，已运行项保留记录。

主表是 PURE_GAM_R1_SEVEN.md/.csv/.json，随队列刷新；AAA是各阶段已见任务等权准确率的十阶段均值，FAA是最终十任务等权均值。另保存样本加权口径、遗忘、BWT、原始10×10下三角矩阵和所有逐任务checkpoint，未完成格保持空白。

预先确定的比较：g0→g2识别外层扰动贡献；g2→g3识别第二段内层探测；g0→UG识别梯度差分的长期作用；历史g1/g3/双分支/原版分析组合与正交修正。在无范数匹配情况下，方向与更新幅度共同变化，不把单seed差值写成普遍或显著结论。

部署目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T073400Z`。

执行：`/home-local/liying/.conda/envs/Pilot_new_local/bin/python -u launch_queue.py --execute --poll-seconds 15`。在真实主机namespace运行，队列先验冻结pins与历史四条再启动，短测通过后自动放行三正式项。
