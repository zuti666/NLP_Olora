# RQ2 实验环境

基础环境使用已有 `Pilot_new@e94a148f`，不修改其已安装依赖。解释器 `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`；原始spec来自 `CopyFlatCV/.aris/compute/local-pilot-new-cudahpc15-env-spec.json`。

当前主机为 `cudahpc40.cvc.uab.es`，GPU 为 NVIDIA RTX6000 Ada、48GiB。用户新授权为：等待 GPU0–3 的整个 TreeLoRA 队列完成，再确认四张卡连续空闲60秒，随后在物理 GPU0（SGD）和 GPU1（GAM）执行本批实验。前驱状态固定为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`，要求队列状态为 `completed`。

当前 Pilot_new 的 CPU 依赖导入已通过，包括 torch 2.4.1+cu121、timm 1.0.9。实际 GPU kernel witness 将在前驱完成、四卡满足空闲条件后，并在各分支 sanity 之前自动运行；新凭据保存为 `campaign_seed0_after_gpu03/env_witness_gpu0.json` 与 `env_witness_gpu1.json`。CPU import 通过不代表 GPU forward/backward 或科学 sanity 已通过。

以前 `env_witness_gpu4.json`、`env_witness_gpu6.json` 对应旧设备与旧授权范围，仅保留作历史排查依据。正式测量还需通过 diagnostic.py 的模型、数据、逐样本梯度等完整 sanity。

## 历史数值复现与正式计算精度

初次完整FP32 sanity与历史两个Task0测试CE相差约1.8e-4。已定位到patch embedding卷积计算精度：在独立恢复的模型中，仅把该卷积的权重和输入按TF32的10位尾数round-to-nearest-even处理，其余保留FP32，两个方法×t0/t9的四个历史测试accuracy全部精确复现，CE最大绝对误差1.38e-6，低于原先3e-5门槛。脚本为 `investigate_tf32.py`，原始证据与独立对照分别为 `investigation/tf32_conv_emulation_witness.json` 和 `investigation/tf32_independent_comparison.json`。

这是受控计算精度实验对偏差原因的支持，不是历史机器或整个历史训练过程的逐位复现。此前精度调查使用的 Quadro RTX6000 属于Turing架构；当时基础环境的默认`cudnn.allow_tf32=true`并不使该卡执行TF32。当前 RTX6000 Ada 上仍显式关闭 TF32，以保持正式新测量的FP32协议。历史环境支持TF32时，相同默认设置可以改变卷积数值。

正式新测量采用明确关闭matmul TF32和cuDNN TF32的FP32模式。历史复现检查单独运行临时TF32卷积精度reference，结束后必须逐位恢复卷积权重并移除hook。新路径Fisher、损伤、全20个endpoint重评及A表a11统一使用当前FP32协议，不把不同精度的旧CE与新Fisher混作同一测量。

## 未采用的 Pillow 历史环境候选

历史 `CopyFlatCV/requirements.txt` 记录 Pillow 11.3.0，基础环境目前为12.1.1。调查时将11.3.0安装到独立 `execution/deps/pillow11_3`，以 `PYTHONPATH` 选择，不修改基础环境。候选spec见 `investigation/pillow11_3_spec.json`；运行时PIL实际文件、扩展库、JPEG版本及SHA256见 `investigation/pillow11_3_runtime.json`。

GPU对照显示11.3.0与12.1.1的两个Task0测试CE相同，均未恢复历史轨迹CE；因此该候选不用于正式作业，不设置其PYTHONPATH，也不把Pillow版本视为本次偏差原因。原始失败记录和候选环境均保留，基础环境不变。

## 前驱完成与GPU空闲等待

`campaign_plan.json` 仍是当前主计划，仅排 seed0 sanity、邻域、真实方向、全任务重评、t9路径、两条Task1续训及表图汇总，共8阶段、15作业。正式seed1/2配置不在此队列中；科学设置不变。

接续入口为：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/wait_then_run.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json --execute
```

已配置接续，实时状态见 `campaign_seed0_after_gpu03/queue_state.json` 和 `QUEUE_STATUS.md`；实际 pipeline 启动后才生成同目录 `pipeline_state.json` 与 `STATUS.md`。前驱整个队列未完成或四卡尚未连续空闲60秒时保持等待。实际作业按 UUID 绑定物理 GPU0/1，启动前仍检查目标卡空闲；每阶段两个方法全部完成且 receipt 通过后才启动下一阶段。路径求积收敛和数值界检查未通过时，不启动后续训练。

`queued` 或等待状态均不能写成实验完成。完整计划、源码指纹及已完成receipt哈希用于恢复核验。旧 `campaign_seed0/STOPPED_BY_USER.json` 保留为历史停止记录，更新前文件归档于 `campaign_seed0/before_gpu03_requeue/`；新的接续使用独立目录，不覆盖原停止凭据。
