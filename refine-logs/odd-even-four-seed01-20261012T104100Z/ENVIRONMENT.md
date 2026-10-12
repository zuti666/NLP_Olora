# 四方向seed0/1环境与文档验收

日期：2026-10-12。只允许GPU4/5/6，最多3并行。资源以本suite为准；environment_spec中的历史资源元数据不扩展用户授权。

复用 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`，Python3.11.15/PyTorch2.4.1+cu121/torchvision0.19.1+cu121/timm1.0.9。canonical spec SHA256 `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`，与父环境和.aris/compute/local.md账本一致；不安装或升级包。训练math attention、FP32、TF32/AMP关闭，源190文件、模型与数据沿用父固定资产。离线缓存，不下载。

根目录只读源：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source`。临时目录固定 `/tmp/oe_s01_20261012T104100Z`；避免过长Unix socket路径。训练产物只写新运行目录。

## 独立代理原样执行的CPU命令

冻结完成后，独立代理在宿主机原样执行下列命令；它核验17jobs、配置/源码/门禁/pins，不启动队列、不调用CUDA、不训练、不重写冻结文件。

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/odd-even-four-seed01-20261012T104100Z/run_queue.py --dry-run
```

stdout应为JSON，包含status=valid、jobs=17、formal=8、smokes=8、allowed_gpus=[4,5,6]、max_parallel=3。验证者将命令、returncode、stdout、差异结论另存gates/DOC_CHECK.json（status=passed），该动态收据不加入冻结pins。不能以旧目录的DOC_CHECK代替本次。

## CUDA验收

上述CPU检查不是CUDA通过证明。实际每个GPU worker都用自身seed0或1执行新的FP32前后向witness、验证CUDA UUID与有限值。8个两任务短测及按seed四方向真实初始读数配对全部完成才放行正式训练。保留基点/R0恢复及只推进R1的父协议；不改变实际学习数学。

只在共享编号与UUID锁下使用空闲4/5/6；显存<500MiB、util<10%。GPU0/1/2/3禁止。启动凭据与最新进度另存LAUNCH_VERIFIED.json及EXPERIMENT_TRACKER.md，不改冻结计划。
