# ImageNet-A g2 P/C/CD：GPU0/1正式执行中

更新时间：2026-10-03T13:17:50.399724+02:00，当前快照；实时状态优先。

仅物理GPU0/1，并行上限2。新控制器PID2806544存活，三组GPU0短测及CPU配对均通过。**P组正式训练正在GPU0运行，已观测73次实际更新**，最新坐标task=0、epoch=8（epoch从1开始）。GPU1当前忙卡等待，C/CD正式任务保留排队。尚无完整十任务成绩。

| 任务 | 状态 | 物理GPU |
|---|---|---|
| smoke_G2-CD01_O_P_natural_s2 | completed | 0 |
| smoke_G2-CD02_O_C_natural_s2 | completed | 0 |
| smoke_G2-CD03_O_CD_natural_s2 | completed | 0 |
| G2-CD_smoke_pairing | completed | None |
| G2-CD01_O_P_natural_s2 | running | 0 |
| G2-CD02_O_C_natural_s2 | pending | — |
| G2-CD03_O_CD_natural_s2 | pending | — |
| G2-CD_location_diagnostics | pending | — |

科学设置保持seed2、rank10、ρ=.05、r=.25、10tasks×50epochs、batch128、SGD LR首任务.04/随后.02、momentum/weight_decay0；FP32 math、TF32/AMPoff。最终只提交g2，P/C/CD仅改变外层定位差分；Fisher/EWC/融合/投影/HVP关闭。已完成短测实际每组2更新/8训练CE/16零更新隔离CE，禁用模块调用0。三formal完成后自动进行108普通CE/零更新共同状态诊断。主要报告AAA，另报FAA/LA/BWT/遗忘及完整矩阵。

实时队列：[queue_state.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/runtime/queue_state.json)；正式启动：[FORMAL_LAUNCH_VERIFIED.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/FORMAL_LAUNCH_VERIFIED.json)；计划：[EXPERIMENT_PLAN.md](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/EXPERIMENT_PLAN.md)。

原GPU2–6控制器2504843已暂停并在无活跃worker时停止，旧日志与两组短测保留；新目录重新短测，不复用旧成绩。其他队列未修改，直接差分回加Local-CD仍plan_only/jobs空。当前45个冻结文件哈希完整，独立路由审核/新六入口/独立文档命令通过，25CPU证据为同字节trainer复用。
