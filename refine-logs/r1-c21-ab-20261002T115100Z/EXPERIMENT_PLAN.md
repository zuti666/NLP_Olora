# ImageNet-A：新增原生 r=1 的 A/B 两组训练

用户授权只新增这2条完整训练；完成后扩充为9方向表。前7条全部复用已核验的同协议r1结果，不重训，不复苏旧取消队列，不安排其他半径或多seed。

| 组 | 最终更新方向 | 主要对照 |
|---|---|---|
| A | g0+(g2−g1) | 相对g0检查跨点差分回加；相对g2检查基点替换 |
| B | g1+[(g2−g1)−a g1] | 相对g1检查正交残差；相对g2检查去掉平行残差 |

delta=g2−g1，a=Σp〈g1,p,delta,p〉/Σp‖g1,p‖²；对全部实际可训练LoRA A/B与完整seen FC参数统一FP64归约，并在训练dtype构造最终方向。g1全局恰为0时a=0，不为非零分母添加epsilon。B为参数梯度坐标的投影，不是在有效BA坐标中投影。

固定设置：ImageNet-A固定类别顺序0…199、10×20类、无replay；ViT-B/16 augreg2_in21k_ft_in1k，rank10；seed2。外半径gam_grad_norm_rho=1、内层gam_grad_rho=.05、非adaptive。训练baseβ=(1,1,0)、γ=0构造相同四探针，再替换最终更新方向。lambda_flat=1，ewc_lambda=0，ewc_skip_task_fisher=true；Fisher估计/归一化/阻尼、融合、SGV和额外漂移约束关闭。

每任务50epochs，batch128、不丢弃末批，零momentum/weight_decay SGD；首任务LR=.04、后续=.02，cosine最低.004/.002。保留历史mechanism_eval/mechanism_grad_stats=true，train/eval workers8；保留原生全局RNG、DataLoader及原生评估执行，不使用r=.25的逐task/epoch重播种Controller、不改注意力backend。

每步仍是一次clean梯度与原生四个probe，共5CE。先执行原生gradient_decompose、替换方向，再保留原FP32 clean+1*(selected−clean) projector、原生unperturb及实际SGD。逐步精确验证gradient cache、选择、projector和真实SGD。所有临时hook限定训练optimizer实例，其他机制诊断GAM保持原生。

观察指标：全局投影系数、平行/正交范数比例、夹角、实际参数步长与有效无缩放LoRA BA＋完整FC步长。有效步长在原生unperturb后的真实SGD起点计算；同状态projected-g1参考仅生成假设的FP32 SGD后参数tensor，并用FP64低秩Gram测量，不真正多执行SGD、不增加forward/backward、不修改RNG，不改变步长。

CPU测试包含独立无审计原生oracle、多方向/20与40类头、全局与逐层反例、零/微小分母、非有限值、FP32projector消去反例和有效BA+FC dense小矩阵oracle。独立复测通过后冻结pins。真实GPU短测为A/B/native-g1控制各前2任务×1epoch，共18更新/组；只用于执行与配对核验，不作为50epoch成绩。要求同初始化与相同输入/目标/RNG序列、每组自己的真实梯度/SGD精确等式、无Fisher/融合/额外步长匹配；不设置跨GPU参数轨迹逐位或近等价门。

短测3/3验收后自动并行启动A/B两条完整训练，各2550更新、299050样本次。限定物理GPU4、5、6；使用实时空闲检查（<500MiB、<5%利用率）、共享GPU lease、UUID重查、PID/starttime/host namespace及唯一controller锁。失败保留凭据并停止后续放行。

主结果为任务等权AAA；同时记录FAA、LA、BWT、遗忘及样本加权口径，保存10×10原始矩阵与全部逐任务checkpoint。PURE_GAM_R1_NINE.md/.csv/.json随队列更新，仅完成验收的格填分数。

部署目录：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-c21-ab-20261002T115100Z。
复现：在该目录以 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -u launch_queue.py --execute --poll-seconds 15 执行一次；不要启动重复controller。

单seed结果只能支持本配置的性能消融。B去掉平行残差仍可能改变方向范数与有效步长；不能把自然步长比较写成已排除幅度影响，也不能凭准确率推断Hessian因果机制。
