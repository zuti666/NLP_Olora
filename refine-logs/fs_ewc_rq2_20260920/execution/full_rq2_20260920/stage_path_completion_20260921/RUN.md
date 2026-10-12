> **2026-09-21 完成更新：经用户授权迁移到 GPU1/4/6 后，队列已于 19:38:42 UTC 完成。B2 新增36/36、合并54/54条路径；B1加点6/6作业完成。重新统计与最终图稿见 [结果说明](final_review_20260921/README.md)，当前状态见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。下方为启动前方案与迁移前 GPU0/3 安排的历史记录；“尚未测量”“预期缺失”均描述启动时状态，不代表当前进度，不要重复启动。**

# 正式三 seed：补齐逐任务路径 Fisher

指标与用户给出的定义一致：对每个任务端点分别构造 `Δ_t=W_t-W_0`，测量 `q_t(s)=Δ_tᵀF_0^emp(W_0+sΔ_t)Δ_t`，以 Simpson 积分得到 `I_t`，再计算 `S_path,t=I_t/||Δ_t||²`。这里的路径是每个端点到 Task-0 锚点的独立直线弦段；不是逐步训练轨迹，也不能把单条路径的九个积分节点当成九个任务。

## 为什么上一张正式图只有三个点

| 批次 | Fisher 样本 | 已有目标任务 | 路径数 |
|---|---:|---|---:|
| 历史 seed0 | 固定 256 张训练子集 | Task1–Task9 | 18（两方法） |
| 正式 seed0/1/2 | 全部 1259 张训练图 | Task1、Task5、Task9 | 18（三 seed × 两方法 × 三目标） |
| 本次正式补测 | 全部 1259 张训练图 | Task2、3、4、6、7、8 | 36 |

历史 `0.3680576 → 0.0147931` 是 Task9 的归一化方向路径敏感度，历史全阶段补测已经完成，见 [历史 C_stage_path.csv](../../../mechanism_next_20260920/stage_path_extension_20260921/report/C_stage_path.csv)。正式 Task9 的三 seed 均值为 `0.23964896 → 0.01519343`。两批的 checkpoint 与 Fisher 样本范围不同，不能拼接。

原始正式 trajectory 有全部十个端点的位移、旧任务损失、准确率和训练 Fisher trace；全部 checkpoint 也存在。但缺失任务的路径节点梯度及对应 `q_t(s)` 尚未测量，不能由这些端点标量反推出积分。正式 B1 的 81 个半径也只测了前向损伤和准确率；它们不能替代路径方向 Fisher。

## 本次执行范围

- 复用已完成的 18 条正式路径，只新增 36 条缺失路径；全部54条汇总后，第四个子图覆盖 `T=t+1=2,…,10`。`T=1` 时位移为零，归一化指标未定义，留空。
- 原 `diagnostic_full.py` 的 `FullRunner` 已支持任意端点。新增 `diagnostic_extension.py` 仅开放 CLI 的任务范围并添加独立来源核验，原积分/梯度实现不改。
- 每条路径从嵌套 5/9 节点开始核验，必要时细化至17/33；保留原 quadrature、损伤界、端点恢复和精确锚点恢复门槛。
- B1 同配置补充半径11–19：3 seed ×2方法，共6作业，256个固定训练样本、16个公共方向；新增864个非零扰动点。原点重复用于同口径核验；合并后 raw Fisher 有16个实测半径，损伤/准确率仍使用原81个半径×100方向。
- 不重新训练，不修改已有结果。新增图输出在本目录 `figures/`，汇总与校验在 `report/`。

## GPU 队列

仅使用物理 GPU0 / GPU3：EWCLoRA 固定GPU0，FS-EWCLoRA 固定GPU3，最多并发2个作业。先等待此前已授权的历史 radius extension 队列**完整成功**，再执行36条正式路径、发布B2、执行6个正式半径作业、发布B1。每次启动还需卡上无计算进程、显存小于500 MiB、利用率小于5%，并获得共享 UUID 锁；不会抢占当前工作。

环境复用已验证的 `Pilot_new@e94a148f`：Python3.11、torch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。无包安装或环境改动。每个 GPU 作业启动前执行同一个 seeded FP32 CUDA forward/backward witness，关闭 TF32；该 witness 只能在卡空闲且获得锁后运行。

CPU 文档复现检查（不会启动 GPU 或模型评估）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/queue_extension.py --dry-run
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/diagnostic_extension.py --config /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/configs/path_seed0.json --method sgd --job path --path-task 2 --validate-only
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/publish.py --check-only
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/publish_radius.py --check-only
```

预期：队列45作业，其中42个GPU诊断+3个CPU等待/发布作业；B2检查报告18/54已完成、36条缺失；B1扩展检查报告0/6，已有正式B1结果仍单独有效。实际进度更新后，缺失数量会相应减少。

部署命令（由本次会话执行一次；已有 launcher 后不可重复执行）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/launch.py
```

运行状态：[campaign/STATUS.md](campaign/STATUS.md)、`campaign/pipeline_state.json`；进程身份及启动命令：`campaign/launcher.json`；前驱等待细节：`campaign/waiting_for_prior.json`。控制进程在宿主 PID 命名空间运行，不能只凭 sandbox 内看不到 PID 判断其已退出。失败会停止后续作业并保留日志；恢复前必须检查原因，不盲目重试。

自动图稿有数据、字体、文本边界和图例位置检查；新增 GPU 数据尚未完成时不声称生成最终图。完成后的图面仍需实际打开复核。
