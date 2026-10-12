# 当前GPU、排队实验与剩余时间

快照：2026-09-20T23:00:07.294253+02:00（马德里）。实际GPU进程均为RQ2正式队列PID1403027的worker；0–6七卡占用，无其他GPU compute process。

**RQ2：59/88通过、7运行、22排队、0失败。按当前吞吐和依赖模拟，剩余约9–11.5小时，即2026-09-21 08:00–10:30马德里；不包含后续独立S3实验完成时间。**

| GPU | 当前任务 | PID | 进度 | 显存GiB | 利用率 |
|---:|---|---:|---|---:|---:|
| 0 | sgd_seed1_dense | 1694943 | 627/8001 状态 | 1.91 | 99% |
| 1 | gam_seed0_dense | 1694205 | 636/8001 状态 | 1.91 | 99% |
| 2 | sequence_gam_seed1 | 1649969 | 第5/10任务，33/50轮 | 12.08 | 100% |
| 3 | sgd_seed0_dense | 1711199 | 343/8001 状态 | 1.91 | 99% |
| 4 | gam_seed1_dense | 1697530 | 616/8001 状态 | 1.91 | 99% |
| 5 | sequence_gam_seed0 | 1641024 | 第5/10任务，40/50轮 | 12.08 | 100% |
| 6 | sequence_gam_seed2 | 1661999 | 第5/10任务，14/50轮 | 12.08 | 100% |

| RQ2任务组 | 总数 | 完成 | 运行 | 待执行 |
|---|---:|---:|---:|---:|
| Task0起点 | 6 | 6 | 0 | 0 |
| B3 A/B/D续训 | 9 | 9 | 0 | 0 |
| 十任务SGD/GAM序列 | 6 | 3 | 3 | 0 |
| B3四格端点 | 12 | 12 | 0 | 0 |
| train256/test307/train1259稀疏Fisher | 18 | 18 | 0 | 0 |
| 全任务Fisher轨迹 | 6 | 2 | 0 | 4 |
| t1/t5/t9路径 | 18 | 9 | 0 | 9 |
| 真实方向/随机对照bridge | 6 | 0 | 0 | 6 |
| 密集半径扫描 | 6 | 0 | 4 | 2 |
| CPU最终报告 | 1 | 0 | 0 | 1 |

已有四格三seed数值、同协议与RNG验证均通过：[阶段报告](../final_report/RESULTS.md)。全机制报告仍partial，没有COMPLETE凭据。

六条dense各8001状态，最近实测3.21–3.41秒/状态；四条当前仍需约6.6–7.3小时，seed2两条尚未启动。GAM训练预计再约45–75分钟；随后剩余轨迹/路径/6个bridge先执行，再承接最后两个dense，所以全队列ETA长于已运行dense的剩余时间。详细推算见 [ETA审查](../ETA_REVIEW_CURRENT.md)。

另有一个独立S3两阶段pipeline，当前未占GPU：

- 短测controller PID1699290：20项pending，0running/complete。每项2任务×2epochs，18分支短测+observer on/off两项。
- gate PID1708363：待全部短测与验证通过，才启动18项正式训练，目前formal_started=false。
- 正式18项：权重融合8项（center、matched no-merge、whole-module-off、uniform、latest）；Fisher6项（anchor位置、alpha0/1、gamma0/1、关闭task normalization）；GAM4项（方向半幅、SGD+EWC、projection off、rho.025）。
- 每个正式run：ImageNet-A200类、10×20任务、ViT-B16 LoRA rank10、50epochs/task、batch128；种子1两项、种子3两项、种子2十四项。
- S3仅使用GPU0/1、并行2，等待GPU显存<500MiB且util<5%，不要求RQ2整个队列完成。S3与RQ2不共用GPU UUID锁：当前没有共占，但资源释放瞬间存在调度竞争窗口，本次只读核查未改变任一controller。

[S3完整计划](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-mechanisms-20260920/REVIEWER_MECHANISM_PLAN.md)；[S3状态](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-mechanisms-20260920/PIPELINE_STATUS.json)。其完整耗时未用RQ2耗时冒充；短测尚未启动。

TreeLoRA已完成，不再占GPU；mechanism_next目录仅为设计文档，无对应运行controller，不计入已排队作业。

[宿主GPU/进程快照](HOST_GPU_SNAPSHOT.json)；[运行进度数据](CURRENT_PROGRESS.json)。
