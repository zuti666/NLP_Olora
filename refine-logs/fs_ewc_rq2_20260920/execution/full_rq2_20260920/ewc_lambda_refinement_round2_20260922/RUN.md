# Raw Fisher λ：第二轮细化搜索

第一轮细化六点均已观察：5个完成50轮，1个数值发散。合并初扫20点为17完成、3发散，最大旧任务增益0.33pp，仍低于预设1pp。用户已授权未找到合格范围则继续细化，不需重复确认。

本轮仅新增6个λ：4053.6004644211034, 5477.2255750516615, 7400.828044922852, 185515.71554236667, 198701.33464215777, 212824.127989082。来源区间[3000,10000]及[173205.0807568877,227950.70569547775]各3个log四分点，分别补查最佳平均点的较小λ侧及当前稳定/发散边界。未跨越已知发散点向上盲目扩大。

科学协议保持seed0、SGD→SGD两任务、复用同一Task0完整权重和同一真实重算raw Fisher、两处trace归一化false、50轮、lr0.02、batch128、workers4、配对新头/输入RNG、FP32及TF32关闭。train.py与第一轮字节相同；wrapper只将轮次标识从1改2，CPU回归与最小差异核验记录在TRAINING_IMPLEMENTATION.md。环境Pilot_new@e94a148f沿用，不装包。每个GPU训练作业先执行CUDA seeded前向/反向witness，再进入训练及原生终点评估。CPU汇总不重新GPU评估。

遵从最新用户资源指令：优先GPU[3,4,6,2,5]，排除0/1，逐卡空闲+UUID共享锁准入，不等待全部卡同时空闲；最多5并行、资源周期检查900秒，任务结束时可立即补位。

## 只读文档核验

从项目根目录原样执行：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/plan.json
```

预期dry_run=true、stages=7、jobs=7、max_parallel=5，六训练均无依赖，combined_report依赖全部六训练；完整文件指纹预计243项。CLI不打印资源字段，另只读plan.json核验gpus/每job gpu_pool=[3,4,6,2,5]、all_gpu_start_gate.required=false、gpu_poll_seconds=900。只读命令不访问GPU，也不创建campaign状态。

## 后台执行入口

启动前核验新鲜文档检查、CPU测试与全部输入来源SHA均通过，旧轮次控制器和训练进程已退出。宿主机后台启动一次并保存LAUNCH.json的PID/start_ticks/boot_id；已有活动controller不得重复启动。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/plan.json --execute
```

stdout/stderr保存campaign/controller.log，运行后保存宿主机LAUNCH_VERIFIED.json。训练结果在training/，当前队列campaign/STATUS.md与pipeline_state.json。只有共享锁和空闲检查通过的授权GPU才启动训练，不停止其他任务。

## 合并报告

report.py只读核验初扫14点、第一轮6点、当前6点，共26个观测；原始所有负结果和发散记录保留。效力门槛仍为相对原λ0的旧准确率+1pp、新准确率下降≤5pp、两任务平均不下降，且EWC loss/梯度实际生效、几何证据一致。全部当前点得到有效终态时，report/COMPLETE.status=complete；scientific_status单独标记criterion_met或refinement_required等，不把队列完成当科学成功。无候选时λ*保持空，不解锁四行三seed复跑；collector不自行启动下一轮或生成最终SELECTION。
