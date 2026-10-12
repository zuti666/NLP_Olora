# 反向探针回加运行环境

日期：2026-10-03。运行目录 `/data/140-1/users/liying/O-LoRA/refine-logs/reverse-probe-reflected-update-20261003T161621Z`。只允许物理GPU0，并行上限1；本目录准备阶段没有冻结、GPU验收或训练。

## 复用环境与资产

使用既有 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`：Python3.11.15、PyTorch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。没有安装或升级包。`environment_spec.json` 和 `source_manifest.json` 从已审计P/C/CD目录复制，canonical环境SHA256固定为 `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`。math attention为方法协议，不改变软件环境。

`gates/INPUT_ASSETS.json` 记录本机资产：190个冻结源文件、本地ImageNet-A train5981张/test1519张的路径/标签/大小身份、200类顺序、每任务样本数，以及346284714字节ViT-B/16预训练文件SHA256。源根 `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source` 只读，freeze与队列启动再次逐文件核对190项哈希。离线数据、预训练权重和HF/Torch缓存沿用已有环境，不下载或重建。

新 reflected 代码必须由新 `trainer/REFLECTED_CPU_GATE.json` 验证；复制的 `G2_CD_CPU_GATE.json`、`HVP_CPU_GATE.json` 等历史文件只作上下文。新gate记录CPU专属测试结果和最终engine/trainer/test哈希。两条真实 `trainer/reflected_train.py --dry-run` 命令的参数、输出与配置哈希写入本目录 `gates/TRAINER_ENTRYPOINTS.json`。所有新代码与证据在launch前冻结。

## 独立文档验收命令

完成当前代码CPU验收、两个实际入口验收、独立审查并冻结后，独立代理在宿主机按以下绝对命令原样执行。该命令不启动训练、不调用CUDA，也不占GPU；只核对代码/配置/证据pins、科学条件与190项源哈希。

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/reverse-probe-reflected-update-20261003T161621Z/run_queue.py --dry-run
```

必须生成本目录新 `gates/DOC_CHECK.json`。旧目录的文档收据不能替代当前目录文档验收。若Codex沙箱无法读取宿主机环境文件或真实主机身份，可经宿主机执行路线完成同一CPU命令，不需要CUDA训练来验证文档。

## CUDA与队列约束

每个实际GPU worker都重新执行seed2 FP32 CUDA前向/反向witness，并验证其UUID、有限结果和TF32关闭。CPU测试不等于GPU通过；正式作业还须先通过当前目录的两任务短测。

只选择物理GPU0，最多一个worker。共享 `/tmp/olora_gamfs_gpu_locks` 编号+UUID双锁，fresh查询、取锁、再次fresh查询均要求<500MiB与<10%利用率。UUID绑定后逻辑设备为0。不得依据准备时空闲快照预占或绕过忙卡。现有P/C/CD及其诊断仍使用共享锁；本目录不修改任何旧队列、旧计划或旧输出。失败停止新增作业，无重试、抢占或外部通知。
