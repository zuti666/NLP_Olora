# 三探针奇偶分量实验运行环境

日期：2026-10-03。目录 `/data/140-1/users/liying/O-LoRA/refine-logs/odd-even-rho005-gpu123-20261003T185629Z`。本轮授权物理GPU1、2、3，最多3个并行GPU作业。

## 复用环境与资产

固定 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`：Python3.11.15、PyTorch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。没有安装、升级或重建包。canonical environment spec SHA256为 `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`。math attention和FP32属于本轮科学协议。

只读源目录 `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source`，source_manifest.json固定190文件。gates/INPUT_ASSETS.json复用同环境与相同数据/权重的已验证收据；固定ImageNet-A train5981/test1519、200类顺序、路径标签身份与346284714字节预训练权重SHA256。freeze与控制器重新核对190项源哈希。沿用离线缓存，不下载数据或权重。

新 trainer/ODD_EVEN_CPU_GATE.json 必须绑定最终 odd_even_step.py、odd_even_train.py、test_odd_even_cpu.py，并通过至少15个CPU测试；复制的旧G2、HVP、reflected gate仅是历史上下文。14条实际训练入口CPU --dry-run收据记录各自命令、配置哈希与stdout。独立审查passed后才能冻结全部新代码和证据。

## 独立文档验收命令

所有准备门禁通过并freeze后，独立代理在宿主机原样执行下面绝对命令。该命令不调用CUDA、不占GPU、不训练，只核验当前配置、冻结pins、科学约束与190项源哈希。

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/odd-even-rho005-gpu123-20261003T185629Z/run_queue.py --dry-run
```

验收应输出status=valid、jobs=15、formal=7、smokes=7、allowed_gpus=[1,2,3]、max_parallel=3，并生成本目录新gates/DOC_CHECK.json。旧目录文档验收不能代替新目录收据。需要宿主机文件/身份可见性时，使用相同命令的宿主机执行路线，仍不调用CUDA。

## 真正运行的CUDA检查

实际每个GPU worker重新执行seed2 CUDA前后向witness，UUID绑定、有限值和TF32关闭必须通过。CPU测试不等于GPU通过。7个两任务短测以及随后CPU七组配对必须全部通过，才允许运行任何正式训练。首批OE01/OE02/OE03完整完成后，其余OE04-07才能接续。

GPU1、2、3共享编号+UUID锁，fresh空闲查询、加锁、再次fresh查询均需显存<500MiB、利用率<10%，UUID不变。逻辑设备始终为cuda:0。不得以历史空闲快照预占GPU，不绕过其他任务，不修改旧队列；失败停止新增作业，无自动重试或外部通知。
