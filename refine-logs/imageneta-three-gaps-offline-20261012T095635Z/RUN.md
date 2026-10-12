# 复现与运行

本目录仅离线读取20261011已验收的六条S/R分支，不新增训练。正式范围：36曲率cells、24有效步幅cells、30检查点评估（27不同状态，同seed两份e0重复核验）。先运行3个pilot，再各自接续3seed正式离线作业，合计12作业。仅GPU4/5/6，一作业一卡，max3，无抢占。

## 环境与固定输入

复用 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`：Python3.11.15、torch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9，FP32训练kernel，无TF32/AMP。既有canonical environment spec SHA256为 `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`，本轮不安装或修改包。FP64仅声明的历史失败cell pilot后备数值核查。

suite环境包含CUBLAS_WORKSPACE_CONFIG=:4096:8、2个CPU线程、离线模型缓存与数据路径、短TMPDIR。宿主机必须 `cudahpc40.cvc.uab.es`，PID namespace必须 `pid:[4026531836]`。在工具中执行正式启动需使用宿主机执行权限；沙箱的/proc不能作为宿主机GPU进程证据。

历史来源：`/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z`；旧190文件manifest、原始trainer/diagnostics、固定检查点与summary、所有新可执行文件均由implementation_manifest.json固定SHA。不修改旧目录。大量logits结果在 `/data/115-2/users/liying/cache/tmp/imageneta-three-gaps-offline-20261012T095635Z/outputs`，本目录outputs为对应软链接。

## CPU与文档验证

在本目录执行（不需要GPU）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest test_curvature_cpu test_eval_match_cpu
/home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --dry-run
```

dry-run只检查输入、代码、DAG、独占输出范围与GPU授权，不建立GPU进程/输出目录。独立agent必须先原样执行此文档命令，再允许启动。

## 启动与监控

在上述实际宿主机、本目录执行一次：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python launch_queue.py
```

launch_queue先dry-run，再启动独立会话队列。进程、CUDA实际运算witness、每个job启动/退出/验收记录写入runtime；worker和CUDA子进程都继承GPU编号+UUID双锁，结束前保留租约。取得双锁后再次查询GPU空闲（无计算PID、memory<500MiB、util<5%）。GPU绑定UUID，不从CUDA逻辑编号推断物理卡。

实时状态 `runtime/queue_state.json`；总日志 `runtime/controller.log`；每作业 `runtime/jobs/JOB/console.log` 与 outputs/JOB/*.json。`runtime/PAUSE`存在时停止新作业接续，正在运行的只读作业继续结束；删除后自动恢复。无需调用任何旧队列。

中断后的安全恢复：首先按controller/worker的PID及starttime确认原控制器已结束，再运行`python run_queue.py`。新控制器采用仍活着的原worker，不重复提交，worker独立持锁；已完成输出重新验收。技术失败不自动重试/覆盖原输出，须保留失败证据，在新的版本/输出路径中修复后重试。

三类pilot及完整commands已写入suite.json。曲率pilot固定S seed2 e1 batch0，单独保留原三点与新探索七点门槛；正式36cells是FP32。科学不收敛是`inconclusive`且继续其他cell，不冒充accepted机制。匹配/评估只依赖本类pilot技术验收。任何OOM、状态恢复失败、引用或hash错误则停止新接续，等待已经启动作业结束。

全部实际指标有限、零SGD提交、状态/RNG精确恢复；评估正确数与历史逐点完全一致，CE差≤1e-5；每类完成固定scope才能标完整。接受执行不等于接受科学假设。
