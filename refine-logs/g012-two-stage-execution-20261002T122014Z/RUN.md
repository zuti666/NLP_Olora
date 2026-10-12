# ImageNet-A 两阶段执行记录

本轮只允许物理 GPU 2、3、4、5、6；GPU0/1 排除。以 UUID 绑定空闲卡，使用共享 index+UUID 双锁，显存低于500 MiB且利用率低于5%才接收新任务；不终止其他作业。已完成任务保留结果哈希，失败暂停后续启动。

统一设置：ImageNet-A 10×20类，固定类别顺序、seed2，ViT-B/16 augreg2_in21k_ft_in1k，Q/V LoRA rank10、scale1；每任务50epochs、batch128，SGD momentum0/decay0，task0 LR .04，后续 .02，cosine最低为初始LR的.1。rho=gam_grad_rho=.05，r=gam_grad_norm_rho=.25，FP32、TF32关闭。无Fisher/EWC、权重融合、投影或重放。四个普通CE探针使用同一步共同随机状态，快照恢复参数/缓存，数据增强采用独立固定序列；NME类均值读取关闭。

第一阶段16组：CE、正/负第一探针、P/D/C/R的g2和g0+(g2-g1)，另有四格残差与CE的有效步长匹配对照。先全部16个两任务短测，再启动首批9组完整训练；其后7组等待首批完成。

原生注意力的二阶反传不受支持，第二阶段使用math后端，并增加两个同math后端Sec桥接对照，因此共24组完整训练（第一阶段16、第二阶段8），另有24个短测、1个完整HVP数值门槛和2个只读机制诊断。第一阶段保持原生后端。

初始math HVP虽可计算，但预定中心差分门槛失败，原始失败记录保存在gates/initial_math。第二阶段须等待第一阶段参考模型Task0/4/9各两个固定批次通过完整HVP门槛。门槛h=[.1,.03,.01,.003,.001]，至少相邻两点relL2<=.005、cos>=.999；同点一阶梯度relL2<=2e-5。不得放宽阈值或覆盖失败记录。若完整门槛失败，第二阶段不启动。

机制诊断读取参考A_P的Task0/1/4/9的entry/epoch1/end状态，共12状态×2个预注册真实批次。比较四格方向；第二阶段通过后比较Sec/Hstart/Hmid同状态响应。没有训练更新，完整恢复状态。

环境复用现有conda，不安装包。环境变量由suite.json/environment_spec.json记录。正式入口使用绝对Python路径，后台任务不依赖交互shell。RUN_PINS.json冻结新执行代码、配置及计划快照，source_manifest.json冻结原始190个代码/配置文件。

新环境验证者先阅读本文件和ENVIRONMENT.md，再逐字执行以下命令（需宿主CUDA访问）。它只执行队列dry-run和共享锁下的seed2短CUDA算子，不执行训练：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/documented_check.py --gpu 2
```

队列实际入口：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/run_queue.py
```

root以start_new_session后台启动并记录LAUNCH.json；运行状态位于runtime/queue_state.json，各作业console.log位于runtime/jobs/<id>。临时暂停新启动可创建runtime/PAUSE。结果输出summary.csv与RESULTS.md，未完成的AAA/FAA只能作为阶段进度。完整比较报告任务等权AAA/FAA、LA、BWT/遗忘；单seed不计算显著性。
