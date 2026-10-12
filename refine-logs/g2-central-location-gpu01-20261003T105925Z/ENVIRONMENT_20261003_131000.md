# ImageNet-A g2中心定位运行环境：GPU0/1

日期：2026-10-03。用户最新授权将 P/C/CD g2 定位实验改为只使用物理 GPU0/1，覆盖此前排除0/1、允许2–6的授权。新运行目录为 `/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z`，并行上限2。

## 环境复用

使用本机既有 Python `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`：Python3.11.15、PyTorch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。没有安装或升级包。`environment_spec.json` 和 `source_manifest.json` 从旧执行目录逐字节复制；canonical 环境 SHA256 为 `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`，与既有环境 ledger 匹配。math attention 属于实验协议，不需要重建软件环境。

`gates/INPUT_ASSETS.json` 复用旧 CPU 资产证据：190个冻结源文件一致；本地 train5981张/test1519张，路径、标签及大小哈希一致；ViT-B/16 离线预训练文件346284714字节、SHA256一致。全部 `trainer/` 文件与旧执行目录逐字节相同。旧16项引擎测试和9项离线诊断测试共25项通过证据原样保留，来源及哈希见 `GPU_ROUTING_CHANGE.json`，没有重新执行全部 CPU 测试。旧 `gates/INDEPENDENT_PREFREEZE_REVIEW.json` 仅代表历史科学代码审查，其旧 GPU 路由意见由本次授权和新 `gates/GPU01_DELTA_REVIEW.json` 取代。

## 文档验收命令

root 完成新目录独立差异审核并冻结后，由独立代理在宿主机原样执行下列命令；Codex沙箱不暴露GPU驱动或宿主机进程，使用 `exec_command` 的 `require_escalated` 路线。命令核对新清单、配置、代码pins与入口合同，不占GPU、不启动训练：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/run_queue.py --dry-run
```

只有本次命令通过才能写新 `gates/DOC_CHECK.json`；旧 DOC_CHECK 没有复制。新六个训练入口的实际 CPU dry-run 收据由 root 写入新 `gates/TRAINER_ENTRYPOINTS.json`。复制的 CPU gate 不等于新 GPU 验收，正式训练仍须通过三组新短测和配对门禁。

## 调度与当前状态

只选择物理 GPU0/1，至多2个并行 GPU作业；新配置中的 `device:[0]` 是 UUID 绑定后的进程内逻辑设备0。继续共享 `/tmp/olora_gamfs_gpu_locks` 的物理编号与UUID双锁，每次取得锁后两次fresh NVML检查均须小于500MiB且低于10%利用率，再绑定UUID。忙卡持续等待，不抢占；GPU2–6不属于新队列允许范围。

root 停旧控制器时的最新宿主机观察：GPU0空闲，GPU1约12GB占用；此处为准备阶段快照，实际分配以最新NVML为准。旧队列已暂停并在没有活跃worker时停止，新目录尚未进行训练、尚未冻结或启动。旧两组已完成短测保留在旧执行目录，新目录重新执行三组短测。已有环境CUDA witness可暖复用；每个新GPU worker仍会在训练前重做seed2 CUDA前向/反向witness并保存独立收据。

本轮只实施三组g2自然步长与既定共同状态离线诊断；科学代码、批次预算、seed和超参数不变。冻结CopyFlatCV源只读。没有实施 Local-CD、D/R、半径/rank/混合系数扫描或HVP训练。
