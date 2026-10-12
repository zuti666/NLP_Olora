# 两组 GAM 半径干预

最终用户指定：(gam_grad_rho,gam_grad_norm_rho)=(0.05,1.0) 和 (1.0,1.0)。仅使用物理 GPU0/1，共享 UUID 锁、每次实际启动前检查空闲，最大并行2。取消的(1,.2)只做过CPU检查，没有GPU实验结果。

本轮以正式RQ2训练协议为基准，只改变半径及其活动as1别名。引用的 `CopyFlatCV/refine-logs/s3-expanded-lambda-r-run-20260921T153236Z/configs/s3_expand_r1p0_s2.json` 用于确认两个半径的含义；其 damping、任务内Fisher/权重平均、workers8等整套配置另有差异。本轮不是对该十任务FAA 56.881%的复现。实际仅训练Task0，不产生十任务FAA。

每组训练seed0/1/2，共6个新Task0锚点。冷启动保留相同冻结预训练backbone，重新初始化配对LoRA/head；ImageNet-A、ViT-B/16、rank10 Q/V、50epoch、batch128、初始LR.04/cosine、momentum/WD0、workers4、FP32且TF32关闭。代码的 `as1_rho/as1_norm_rho` 会覆盖GAM参数，因此必须同步设置；实际模型和optimizer均检查有效值。

训练环境暖复用 Pilot_new@e94a148f，没有包安装或环境变更。每次分配GPU后重复已有seeded FP32 forward/backward witness，再开始训练。

可复现的只读队列检查（需先运行prepare_training.py生成计划）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/gam_radius1_direction_20260922/training_campaign/plan.json
```

没有 `--execute` 时不分配GPU。实际启动命令由 `launch_training.py` 写入 launch receipt，并在host PID namespace中脱离运行。以training_campaign/pipeline_state.json为状态依据，不能把排队或CPU检查当作训练完成。

后续对每个新锚点分别计算一次旧Task0正梯度和新Task1负梯度，再固定方向测r=0,.5,...,5，共12个诊断作业、132个新状态。诊断r为有效Q/V联合L2范数，训练半径为LoRA A/B及当前head参数空间，二者不能直接等同。旧任务损失/准确率用1259训练图，raw empirical Fisher用固定256训练图，另保存307测试图评估。绘图主要报告训练有符号CE增量、accuracy%、raw Fisher，3seed均值和样本SD。
