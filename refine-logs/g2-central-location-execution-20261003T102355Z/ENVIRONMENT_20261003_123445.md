# ImageNet-A g2中心定位运行环境

日期：2026-10-03T12:32:36.930187+02:00。用户授权实施三组P/C/CD外层定位实验；GPU0/1排除，允许GPU2–6。

## 环境复用

使用本机已有Python `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`，Python3.11.15、PyTorch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。没有安装或升级包。environment_spec.json从已审核运行逐字节复制，canonical SHA256为4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8，与本地ledger匹配。注意力后端math属于本轮实验协议，不需要重建软件环境。

资产核验见gates/INPUT_ASSETS.json：190个冻结源文件完全一致；本地train5981张/test1519张，路径、标签及大小哈希与历史manifest一致；ViT-B/16离线预训练文件346284714字节，SHA256一致。已通过CPU核验，未借此宣称GPU训练通过。

## 文档验收命令

在宿主机执行；Codex沙箱不暴露GPU驱动或宿主机进程。使用exec_command的require_escalated路线，这是用户已授权的本机GPU实验。以下命令只执行清单、配置、代码pins与入口合同的dry-run，不占GPU、不启动训练：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z/run_queue.py --dry-run
```

实际队列复用`/tmp/olora_gamfs_gpu_locks`的物理编号与UUID双锁，fresh NVML两次检查小于500MiB且低于10%利用率，再绑定UUID；忙卡持续等待，不抢占，排除0/1。

由独立代理按本命令原样执行，验收成功写gates/DOC_CHECK.json。环境spec与ledger未变，已有CUDA witness暖复用；新GPU worker在取得空闲卡双锁后、开始任何训练前，仍重新执行seed2的CUDA前向/反向witness并保存每作业收据。目前允许GPU全忙，队列可以先进入等待，不声称新GPU测试已通过。内核证据不能替代新四探针math重放、三组短测、初始化/增强配对检查。

## 运行范围

新运行位于/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z；训练源码在trainer/独立副本中修改，冻结CopyFlatCV源只读。只实施三组g2自然步长和计划的共同状态离线诊断，不实施Local-CD、D/R、半径/rank/混合系数或HVP训练。
