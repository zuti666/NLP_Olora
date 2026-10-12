> **2026-09-22 08:24 Madrid更新：下文全卡等待配置已被用户新指令替代。当前使用GPU3/4/6优先、GPU2/5空闲补位，排除GPU0/1。见[当前执行说明](RUN_GPU23456.md)、[新实时队列](campaign_gpu23456/STATUS.md)及CURRENT_QUEUE.json。旧controller3620620已停止，无训练被中断。**

# Raw Fisher λ：第一轮细化搜索

当前14点初扫已结束：12个λ完成50轮、2个数值发散；最高旧任务准确率增益仅0.33pp，未满足预设1pp，触发用户已授权的进一步搜索。本轮6个λ均保持seed0、SGD→SGD、原Task0完整模型、同一份真实重算raw Fisher、lr0.02、batch128、50轮和配对RNG；不重训Task0、不重新估计Fisher、不改变门槛。

区间10000..30000与100000..300000各插入3个对数四分点。准确浮点数保存在refinement_spec.json；run ID为refine1_01_seed0到refine1_06_seed0，避免小数λ命名碰撞。训练器是初扫train.py的隔离副本，仅将默认prepared-fisher路径指回已通过GPU重算/短测/正式训练验收的初扫文件，科学实现不变。wrapper负责浮点λ及试验身份绑定，数值发散仍以86退出码和DIVERGED收据区分，技术失败仍停止队列。

环境沿用Pilot_new@e94a148f，无依赖变化：Python3.11.15、torch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。FP32、TF32关闭，CPU线程4、workers4。每个GPU作业取得空闲卡后重做CUDA seeded前向/反向witness。原始完整模型/rawF准备和真实短测已在同一训练实现上通过；本轮不重复改变科学逻辑的短测，启动前必须核验隔离副本最小差异、真实prepared-Fisher身份及原始全部收据。

## 只读文档核验命令

在项目根目录原样执行。只检查计划/依赖/全部输入指纹，不访问GPU，不创建campaign状态：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/plan.json
```

预期dry_run=true、stages=7、jobs=7、max_parallel=6；六个独立续训后接一个combined_report，依赖全部六个有效观测收据。all_gpu_start_gate.required=true，gpus=[0,1,2,3,4,5,6]，poll_seconds=900。

## 后台执行入口

由助手在宿主机启动一次，stdout/stderr保存campaign/controller.log，并保存LAUNCH.json中的PID、start_ticks、boot_id和plan SHA。已有活动控制器时禁止重复启动。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/plan.json --execute
```

本新批次须先等待GPU0–6全部同时空闲，显存<500MiB、利用率<5%、无compute进程并成功取得UUID共享锁，每900秒检查。准入后最多六个并发，逐卡检查与共享锁仍有效。繁忙只等待，不停止其他任务。恢复前须在宿主机核对PID身份/锁并使用--resume，冻结代码和输入不得变化。

## 汇总与解释

report.py核验初扫14点和本轮6点来源，复算有效Q/V位移、raw Fisher加权位移及损失/梯度指标，合并20个观测。在同一λ0基线上继续筛选Δa21≥1pp、Δa22≥−5pp、ΔA2≥0的点；数值发散不参加正式性能排名。

report/COMPLETE.status=complete指本轮六点全部获得有效终态，并不表示科学假设得到支持。scientific_status区分criterion_met和refinement_required；后一种表示还没有合格λ，四行复跑仍受约束。原始14点报告和任何旧结果保持不变。实时状态为campaign/STATUS.md和pipeline_state.json。
