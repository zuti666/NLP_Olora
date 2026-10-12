# g2 P/C/CD：GPU0/1实验追踪

更新时间：2026-10-03T13:14:43.722402+02:00（启动核验快照，实时文件优先）。

用户最新指定仅物理GPU0/1，最多并行2组。新控制器PID **2806544** 已核验存活，状态 **running**，计数{"completed": 2, "running": 1, "pending": 5}。GPU0已实际运行短测；GPU1当前约12GB忙，等候释放，不抢占。正式训练是否开始：**False**。

新45文件冻结；独立路由复核、新六入口和独立宿主机文档命令通过。trainer与旧执行目录逐字节相同，沿用原25CPU通过证据，无软件包改变。GPU0的seed2 CUDA witness通过；已完成短测观察到2更新/8训练CE/16隔离诊断CE，Fisher、融合、投影、HVP、create_graph实际调用均0。短测成绩不能作为正式算法结论。

| 任务 | 核验时状态 | 物理GPU | 接续条件 |
|---|---|---|---|
| smoke_G2-CD01_O_P_natural_s2 | completed | 0 | 空闲GPU0/1 |
| smoke_G2-CD02_O_C_natural_s2 | completed | 0 | 空闲GPU0/1 |
| smoke_G2-CD03_O_CD_natural_s2 | running | 0 | 空闲GPU0/1 |
| G2-CD_smoke_pairing | pending | — | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2 |
| G2-CD01_O_P_natural_s2 | pending | — | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2, G2-CD_smoke_pairing |
| G2-CD02_O_C_natural_s2 | pending | — | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2, G2-CD_smoke_pairing |
| G2-CD03_O_CD_natural_s2 | pending | — | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2, G2-CD_smoke_pairing |
| G2-CD_location_diagnostics | pending | — | G2-CD01_O_P_natural_s2, G2-CD02_O_C_natural_s2, G2-CD03_O_CD_natural_s2 |

固定seed2、rank10、ρ=.05、r=.25，最终仅g2自然SGD。P=g+−g0，C=g0−g−，CD=(g+−g−)/2；统一四次普通CE梯度/一次更新。所有三组新短测和CPU配对通过后自动进入三组10tasks×50epochs正式训练；全部正式完成后108普通CE/零更新离线诊断。主要报告AAA，另报FAA/LA/BWT/遗忘及完整矩阵。任一失败停止新启动。

实时状态：[queue_state.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/runtime/queue_state.json)；运行说明：[RUN.md](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/RUN.md)；启动证据：[LAUNCH_VERIFIED.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/LAUNCH_VERIFIED.json)。

旧控制器2504843已确认无活跃worker后停止，保留旧两组短测与原pins，新队列重新跑三组短测；旧队列runtime/PAUSE保留，旧停机证据：[GPU_REROUTE_STOP.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z/GPU_REROUTE_STOP.json)。不再使用GPU2–6；其他队列没有修改。Local-CD仍plan_only、jobs为空。
