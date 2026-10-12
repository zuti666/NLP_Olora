# ImageNet-A 完整十任务：学习率与实际有效步长对照

用户授权物理GPU1；沿用单seed2。本轮主表包含8臂，复用3条同协议完整训练，新增5条完整训练；正常LR Mean作为已有次级对照。GPU1每次只运行一条，失败保留凭据并停止新启动。

|组|最终方向|首任务LR峰值/下限|后续任务LR峰值/下限|执行|
|---|---|---|---|---|
|S|g1|.04/.004|.02/.002|新跑，生成独立完整步长参照|
|S1.5|g1|.06/.006|.03/.003|新跑|
|S2|g1|.08/.008|.04/.004|新跑|
|Sum|g1+g3|.04/.004|.02/.002|复用完整十任务|
|Mean2|(g1+g3)/2|.08/.008|.04/.004|新跑|
|Graw|g3|.04/.004|.02/.002|复用完整十任务|
|Gmatch|g3，实际有效步长匹配独立S|名义.04/.004，逐步求SGD标量|名义.02/.002，逐步求SGD标量|新跑，依赖新S完整完成|
|G025|g3，外半径.25|.04/.004|.02/.002|复用完整十任务|

其余固定：ImageNet-A原10×20类、固定类别顺序0…199，ViT-B/16 augreg2_in21k_ft_in1k、LoRA rank10；内半径gam_grad_rho=.05，外半径gam_grad_norm_rho=1（仅G025=.25）；每任务50epochs、batch128、不丢弃末批，无replay。SGD momentum/weight_decay=0，原生cosine T_max=50、floor=.1×起始LR，首任务与后续任务都应用指定倍率。保留原生global RNG/DataLoader、mechanism_eval/mechanism_grad_stats=true、train/eval workers8。

新配置以beta=(1,1,0)、gamma=0执行相同原生四探针，再由suite.json的direction_selector选择最终g1/sum/mean/g3。Mean用.5*g1+.5*g3；原GAM gamma正交修正关闭。lambda_flat=1，ewc_lambda=0，ewc_skip_task_fisher=true；Fisher估计/归一化/阻尼、任务内与端点权重融合、SGV、额外漂移惩罚关闭。原生FP32 clean+1*(selected-clean)合成、unperturb与SGD保留。

## 研究问题与预定比较

- S1.5/S2 vs S：只提高学习率能否解释已观察到的组合/第二分支收益。
- Mean2 vs Sum：平均方向加两倍LR的名义更新等价性对照。FP32合成保留，不能宣称逐位或完整轨迹严格相同。
- Gmatch vs新S：在逐步实际有效更新长度匹配后，第二分支方向能否保留持续学习收益。
- Gmatch vs Graw：幅度匹配如何改变g3的阶段平均表现、最终保持与遗忘。
- G025 vs Graw：同协议已完成半径对照；不混入采用逐epoch隔离RNG的其他r=.25实验。
- 既有Mean(LR×1) vs Sum：正常LR的方向平均幅度对照，独立列为次级控制。

新S生成2550步完整审计，各任务步数500/400/400/250/200/150/150/150/200/150、共299050样本次。旧S没有逐步有效位移记录，不能充当Gmatch参照，也不填充新S的待完成成绩。

## Gmatch的实际匹配定义

参照来自新S完成轨迹对应(task,task_step)的真实FP32 SGD位移：拼接所有无alpha/rank缩放的LoRA有效B@A、完整seen分类头weight及bias，取联合L2范数。核对输入/标签SHA、前后RNG、类别范围、参数名字/形状、batch样本数和名义LR。只读标量与审计证据，不复制S模型参数，不使用G当前状态的g1作为目标。

在候选G的原生unperturb之后，以实际FP32 projector产生的g3构造FP64低秩Gram多项式，取可行最小正根；用克隆FP32 SGD参数预检，然后只执行一次真实SGD。每步真实匹配相对误差必须≤1e-3。临时optimizer LR在finally恢复名义值，随后才推进原生scheduler。禁止live试跑SGD、额外forward/backward、RNG重播种、零目标替代、无根fallback或静默裁剪；缺失/不匹配/非有限证据直接失败。

## 验收与接续

CPU独立原生oracle、全部选择器、多头/多步独立S匹配、dense BA+FC几何、非单调根、FP32预检和错误参照拒绝检查通过后冻结pins。5项真实GPU短测分别为S、Sum、Mean2、S2、Gmatch，均前2任务×1epoch=18步；Gmatch短测依赖短测S完成。验证90次真实更新、相同初始化/输入/RNG、声明LR倍率与匹配误差。Sum/Mean2仅检查同初始状态首步范数一致性，阈值1e-3；不是完整轨迹等价性门。

全部短测通过后按S→S1.5→S2→Mean2→Gmatch串行放行，后者还要求新正式S完成且哈希绑定。限定GPU1、实时空闲<500MiB且利用率<5%、共享GPU lease、UUID重查、PID/starttime/host namespace与controller唯一锁。其余GPU由其他任务使用。

主要指标为任务等权AAA，同时报告FAA、LA、BWT、遗忘、逐任务矩阵及样本加权口径；新增训练保存每任务checkpoint和逐batch审计，Gmatch额外报告匹配最大/平均误差及SGD标量分布。LR_CONTROL_TABLE.md/.csv/.json随队列自动更新，未完成格留空。

本轮是单seed对照，n=1不报告标准差或统计显著性。Gmatch匹配的仅是有效位移范数，方向、模型状态与曲率仍可不同。三条历史复用结果配置相同，但没有完整逐步trace，不追溯宣称与新臂的逐步配对。

部署目录：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z。
入口：在该目录运行 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python -u launch_queue.py --execute --poll-seconds 15`；现有controller存活时不要重复启动。suite.json和运行配置均冻结；方向需要run_job.py安装hook，不能只将JSON传给原生src.main。

参考当前相同硬件的完整训练耗时，5条新增预计约13–16小时，短测另需约10分钟，以实测进度为准。不接续原formation的未来100步或2/3-task实验。
