# RQ2 实验环境

基础环境使用已有 `Pilot_new@e94a148f`，不修改其已安装依赖。解释器 `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`；原始spec来自 `CopyFlatCV/.aris/compute/local-pilot-new-cudahpc15-env-spec.json`。

当前实测硬件为 Quadro RTX6000 24GiB；用户只授权GPU4、6。2026-09-20初查两卡各1MiB、0%利用率，其余卡有其它任务。每次正式作业前重新检查空闲；不终止其它任务。

独立验证按下列命令原样执行（GPU4仍空闲时）：

```bash
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/env_probe.py --output /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/env_witness_gpu4.json
```

期望：退出0，打印 `WITNESS`；设备名Quadro RTX6000、shape为[8,8]，forward/backward均finite。sandbox中NVML不可见，应通过已授权的本地GPU执行路由运行；该现象不代表宿主driver坏了。

正式测量还需通过diagnostic.py的模型/数据/逐样本梯度sanity；以上kernel witness不代表科学测量已通过。

## 历史数值复现与正式计算精度

初次完整FP32 sanity与历史两个Task0测试CE相差约1.8e-4。已定位到patch embedding卷积计算精度：在独立恢复的模型中，仅把该卷积的权重和输入按TF32的10位尾数round-to-nearest-even处理，其余保留FP32，两个方法×t0/t9的四个历史测试accuracy全部精确复现，CE最大绝对误差1.38e-6，低于原先3e-5门槛。脚本为 `investigate_tf32.py`，原始证据与独立对照分别为 `investigation/tf32_conv_emulation_witness.json` 和 `investigation/tf32_independent_comparison.json`。

这是受控计算精度实验对偏差原因的支持，不是历史机器或整个历史训练过程的逐位复现。Quadro RTX6000使用Turing架构；基础环境的默认`cudnn.allow_tf32=true`并不使该卡执行TF32。历史环境支持TF32时，相同默认设置可以改变卷积数值。

正式新测量采用明确关闭matmul TF32和cuDNN TF32的FP32模式。历史复现检查单独运行临时TF32卷积精度reference，结束后必须逐位恢复卷积权重并移除hook。新路径Fisher、损伤、全20个endpoint重评及A表a11统一使用当前FP32协议，不把不同精度的旧CE与新Fisher混作同一测量。

## 未采用的 Pillow 历史环境候选

历史 `CopyFlatCV/requirements.txt` 记录 Pillow 11.3.0，基础环境目前为12.1.1。调查时将11.3.0安装到独立 `execution/deps/pillow11_3`，以 `PYTHONPATH` 选择，不修改基础环境。候选spec见 `investigation/pillow11_3_spec.json`；运行时PIL实际文件、扩展库、JPEG版本及SHA256见 `investigation/pillow11_3_runtime.json`。

GPU对照显示11.3.0与12.1.1的两个Task0测试CE相同，均未恢复历史轨迹CE；因此该候选不用于正式作业，不设置其PYTHONPATH，也不把Pillow版本视为本次偏差原因。原始失败记录和候选环境均保留，基础环境不变。

## GPU 被占用时的队列行为

`campaign_plan.json` 仅排 seed0 sanity、邻域、真实方向、全任务重评、t9路径、两条Task1续训及表图汇总。正式seed1/2配置不在该队列中。以下命令仅检查计划和文件指纹，不启动GPU作业：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/pipeline.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json
```

真正执行需添加 `--execute`；已有运行状态恢复需同时添加 `--resume`。队列只检查物理GPU4、6，并以实际UUID绑定作业；每个作业启动前要求该卡显存使用低于500MiB、利用率低于5%，否则保持`waiting_for_gpu`，不挤占、不终止其它任务，也不自动转用其它GPU。每阶段两个方法全部完成且receipt通过后才启动下一阶段；路径的求积收敛和数值界检查未通过则不启动后续训练。

运行中以 `campaign_seed0/pipeline_state.json` 和 `campaign_seed0/STATUS.md` 为准。状态文件中的`queued`或`waiting_for_gpu`不能写成实验已完成；单个kernel witness或第一轮失败后的anchor数值同样不是完整sanity通过。完整计划/源码指纹和已完成receipt哈希会在恢复时检查，以防误续旧配置或重复已完成作业。
