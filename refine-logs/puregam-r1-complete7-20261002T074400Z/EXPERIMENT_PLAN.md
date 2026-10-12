# ImageNet-A：仅补齐 r=1 的三组完整实验

按用户最新指令，运行队列只含以下3组，限定 GPU4、5、6；不安排新的短测、曲率诊断、超参或其他后续训练。历史完成与失败记录留档，不接续其等待项。

| 实验 | 最终更新方向 | seed | LoRA rank | 外半径 r | 内层 rho | 任务/训练预算 |
|---|---|---:|---:|---:|---:|---|
| 仅g0 | g0 | 2 | 10 | 1.0 | .05 | 10×20类，每任务50 epochs |
| 仅g2 | g2 | 2 | 10 | 1.0 | .05 | 10×20类，每任务50 epochs |
| 原点梯度＋差分 | g0+(g3−g2) | 2 | 10 | 1.0 | .05 | 10×20类，每任务50 epochs |

共同沿用历史四组协议：原 frozen ViT-B/16 augreg2_in21k_ft_in1k，ImageNet-A固定类序，无replay；batch128，SGD momentum/weight_decay=0，首任务LR=.04、后续=.02、cosine最低.004/.002。lambda_flat=1，EWClambda=0、跳过Fisher计算，Fisher估计及所有权重融合关闭。保留历史机制评估和原生全局RNG/DataLoader，不按task/epoch重新播种，不改变注意力backend，不做额外范数匹配。

训练实例仅选择最终方向；native四次探测、clean cache共5CE，原FP32 clean+(selected−clean) projector、扰动还原和实际SGD逐batch精确核验。正式预算各2550steps/299050sample exposures，保留全部10任务checkpoint与原生CNN矩阵。

前一目录5条GPU短测已完成共90steps，公式/原生cache/projector/实际SGD验证及18批次配对数据RNG通过。跨GPU checkpoint近等价门失败被完整保留；新放行依据是同一次运行中精确数学身份＋CPU独立无hook原生参考，不声称跨GPU轨迹等价，也不降低原失败阈值。上述凭据、trace/summary、config/source和失败记录绑定在GPU_WITHIN_RUN_PROOF.json，另有独立审查。GPU4–6必须真实空闲并取得共享lease才启动，忙卡等待。

七行表 PURE_GAM_R1_SEVEN.md/.csv/.json 自动刷新，复用核验后的历史4行，并只增加本3项；AAA/FAA统一为任务等权口径，未完成格不填分数。保留样本加权指标、遗忘/BWT及逐任务矩阵。

部署目录：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z。
执行：/home-local/liying/.conda/envs/Pilot_new_local/bin/python -u launch_queue.py --execute --poll-seconds 15。
