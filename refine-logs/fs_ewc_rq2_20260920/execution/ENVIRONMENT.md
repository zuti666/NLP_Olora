# RQ2 实验环境

恢复已实际运行。2026-09-20 17:30马德里时间核验为10/15作业complete、2个neighborhood运行中、3个待执行，无新失败；实时摘要见 [CURRENT_STATUS.md](monitor_20260920/CURRENT_STATUS.md)。

基础环境使用已有 `Pilot_new@e94a148f`，不修改其已安装依赖。解释器 `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`；原始spec来自 `CopyFlatCV/.aris/compute/local-pilot-new-cudahpc15-env-spec.json`。

当前主机为 `cudahpc40.cvc.uab.es`，GPU 为 NVIDIA RTX6000 Ada、48GiB。用户要求等待 GPU0–3 的整个 TreeLoRA 队列完成，再确认四张卡连续空闲60秒，随后采用 GPU0/1 与 GPU2/3 两条执行线。GPU0/2 对应 SGD，GPU1/3 对应 GAM。前驱状态固定为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`，要求队列状态为 `completed`。

当前 Pilot_new 的 CPU 依赖导入已通过，包括 torch 2.4.1+cu121、timm 1.0.9。上一轮四卡 kernel witness 与两个完整 sanity 均通过，两条 Task1 续训已完成；旧凭据保留于 `campaign_seed0_dualline_gpu03/`。上一轮失败来自邻域作业的 Python 参数重名，而非训练数值失败。修复诊断脚本后使用新协议哈希重跑 sanity，新凭据保存为 `campaign_seed0_dualline_recovery1/env_witness_gpu0.json` 至 `env_witness_gpu3.json`。GPU0/1 的 witness 在各自 sanity 前运行；GPU2/3 的 witness 仍作为 training 作业的 `preflight_command`，随后runner确认已有 COMPLETE 并跳过重训。

恢复进程显式设置 `TMPDIR=/tmp`。此前继承 NFS 临时目录时，multiprocessing worker 清理临时文件出现噪音；改用本机临时目录不改变依赖版本、实验配置或模型计算。共享 `Pilot_new` 环境未修改。

2026-09-20 15:27:22 UTC核验快照：恢复pipeline已于15:26:06 UTC启动，新版两条sanity通过，训练均确认 `Already complete`，两条A端点评估完成，neighborhood与trajectory正在运行。实时状态以 `campaign_seed0_dualline_recovery1/pipeline_state.json` 为准。

以前 `env_witness_gpu4.json`、`env_witness_gpu6.json` 对应旧设备与旧授权范围，仅保留作历史排查依据。正式测量还需通过 diagnostic.py 的模型、数据、逐样本梯度等完整 sanity。

## 历史数值复现与正式计算精度

初次完整FP32 sanity与历史两个Task0测试CE相差约1.8e-4。已定位到patch embedding卷积计算精度：在独立恢复的模型中，仅把该卷积的权重和输入按TF32的10位尾数round-to-nearest-even处理，其余保留FP32，两个方法×t0/t9的四个历史测试accuracy全部精确复现，CE最大绝对误差1.38e-6，低于原先3e-5门槛。脚本为 `investigate_tf32.py`，原始证据与独立对照分别为 `investigation/tf32_conv_emulation_witness.json` 和 `investigation/tf32_independent_comparison.json`。

这是受控计算精度实验对偏差原因的支持，不是历史机器或整个历史训练过程的逐位复现。此前精度调查使用的 Quadro RTX6000 属于Turing架构；当时基础环境的默认`cudnn.allow_tf32=true`并不使该卡执行TF32。当前 RTX6000 Ada 上仍显式关闭 TF32，以保持正式新测量的FP32协议。历史环境支持TF32时，相同默认设置可以改变卷积数值。

正式新测量采用明确关闭matmul TF32和cuDNN TF32的FP32模式。历史复现检查单独运行临时TF32卷积精度reference，结束后必须逐位恢复卷积权重并移除hook。新路径Fisher、损伤、全20个endpoint重评及A表a11统一使用当前FP32协议，不把不同精度的旧CE与新Fisher混作同一测量。

## 未采用的 Pillow 历史环境候选

历史 `CopyFlatCV/requirements.txt` 记录 Pillow 11.3.0，基础环境目前为12.1.1。调查时将11.3.0安装到独立 `execution/deps/pillow11_3`，以 `PYTHONPATH` 选择，不修改基础环境。候选spec见 `investigation/pillow11_3_spec.json`；运行时PIL实际文件、扩展库、JPEG版本及SHA256见 `investigation/pillow11_3_runtime.json`。

GPU对照显示11.3.0与12.1.1的两个Task0测试CE相同，均未恢复历史轨迹CE；因此该候选不用于正式作业，不设置其PYTHONPATH，也不把Pillow版本视为本次偏差原因。原始失败记录和候选环境均保留，基础环境不变。

## 前驱完成与GPU空闲等待

`campaign_plan.json` 仍是当前主计划，共8阶段、15作业，双线DAG与GPU分配不变。当前诊断配置为 `diagnostic_config_recovery1.json`，相较旧配置仅将 `output_dir` 改为 `results_seed0_recovery1/`。共同新版 sanity 在 GPU0/1 完成后，01线执行 neighborhood→bridge，23线同时执行 training（确认已完成并跳过）→table_a→trajectory→path，report 等两线结束。正式seed1/2配置不在此队列中；科学设置不变。

接续入口为：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/wait_then_run.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json --execute
```

恢复接续是否已启动及实时状态见 `campaign_seed0_dualline_recovery1/queue_state.json` 和 `QUEUE_STATUS.md`；pipeline 状态写入同目录 `pipeline_state.json` 与 `STATUS.md`。前驱整个队列未完成或四卡尚未连续空闲60秒时保持等待。实际作业按 UUID 绑定物理 GPU0/1/2/3，启动前仍检查目标卡空闲。共同新版 sanity 通过后两线并行，各阶段两个方法全部完成且 receipt 通过后才放行该线下一阶段。path 不作为 A 训练前置，但最终报告仍要求 `quadrature_converged=true` 且 `bound_passed_with_2pct_numeric_tolerance=true`。依赖依据见 [DEPENDENCIES.md](DEPENDENCIES.md)，本次恢复依据见 [RECOVERY.md](campaign_seed0_dualline_recovery1/RECOVERY.md)。

`queued` 或等待状态均不能写成实验完成。完整计划、源码指纹及已完成receipt哈希用于恢复核验。旧等待进程 PID `541721` 已在未启动本批 GPU 实验时安全停止，TreeLoRA 前驱未受影响；旧单线安排归档于 `campaign_seed0_after_gpu03/before_dualline/`，替换凭据为 `campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json`。更早的 `campaign_seed0/STOPPED_BY_USER.json` 和 `campaign_seed0/before_gpu03_requeue/` 继续保留；新的接续使用独立目录。
