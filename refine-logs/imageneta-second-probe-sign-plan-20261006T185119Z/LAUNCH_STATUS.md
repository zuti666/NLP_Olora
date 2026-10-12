# 已授权实验队列启动快照

时间：2026-10-06T21:37:37.748980+02:00（Europe/Madrid）。controller PID920989 / starttime11996081 / 真实主机cudahpc40.cvc.uab.es，已核验存活。

- 状态：waiting_for_idle_gpu，29pending，0短测/正式开始。
- 队列：14短测→1CPU配对→14正式；2项旧B1结果已验收复用。
- 设备：物理GPU2–6，排除0/1，最多5并行，每60秒检查；共享index/UUID锁，双次idle和无compute PID检查，不抢占。
- 验收：CPU11/11、queue helper11/11、独立科学审查和最小上下文逐字文档验收通过。60执行文件和190source文件固定，无环境重建。
- 当前新GPU模型验收/正式结果尚无；所有7张卡的compute进程均存在。GPU0/1利用率瞬时低也不能视为空闲。

代码：trainer/direction_step.py、trainer/four_probe_train.py。参数前g0→g1→g2不变，U=anchor+residual_sign*(g3_probe_sign−g2)。独立双侧诊断每任务额外2CE/0SGD，完整恢复模型/RNG；14formal每项2550steps、10200训练CE+20隔离诊断CE。

实时状态：runtime/queue_state.json；逐项结果：summary.csv；启动证据：LAUNCH_VERIFIED.json；执行说明：ENVIRONMENT.md。旧训练和旧结果未修改。
