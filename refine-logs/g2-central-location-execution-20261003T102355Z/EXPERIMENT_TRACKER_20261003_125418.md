# g2中心定位实验追踪

更新时间：2026-10-03T12:54:18.019037+02:00。

已实现并冻结，独立25项CPU检查、六配置真实入口dry-run、独立宿主机文档命令通过。控制器PID 2504843 已核验存活，状态 **waiting_for_idle_gpu**。GPU2–6全部忙；8项pending，短测/正式均未启动，尚无准确率结果。排除GPU0/1。忙卡不抢占，不修改其他队列。

固定seed2、rank10、ρ=.05、r=.25，最终仅g2自然SGD；没有EWC/Fisher/权重融合/投影/HVP。三组定位差分P=g+−g0、C=g0−g−、CD=(g+−g−)/2。同批同随机状态，统一四次普通CE梯度/一次更新。

| 任务 | 类型 | 本次核验状态 | 接续条件 |
|---|---|---|---|
| smoke_G2-CD01_O_P_natural_s2 | training | pending | GPU2–6空闲 |
| smoke_G2-CD02_O_C_natural_s2 | training | pending | GPU2–6空闲 |
| smoke_G2-CD03_O_CD_natural_s2 | training | pending | GPU2–6空闲 |
| G2-CD_smoke_pairing | pairing | pending | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2 |
| G2-CD01_O_P_natural_s2 | training | pending | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2, G2-CD_smoke_pairing |
| G2-CD02_O_C_natural_s2 | training | pending | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2, G2-CD_smoke_pairing |
| G2-CD03_O_CD_natural_s2 | training | pending | smoke_G2-CD01_O_P_natural_s2, smoke_G2-CD02_O_C_natural_s2, smoke_G2-CD03_O_CD_natural_s2, G2-CD_smoke_pairing |
| G2-CD_location_diagnostics | diagnostic | pending | G2-CD01_O_P_natural_s2, G2-CD02_O_C_natural_s2, G2-CD03_O_CD_natural_s2 |

短测与CPU配对全部通过后自动进入三组正式训练，三组正式完成后运行108普通CE/零更新离线诊断。正式每组2550更新；主要报告AAA，兼报FAA、LA/BWT、遗忘及完整矩阵。任一失败停止新启动。

实际运行目录：[g2-central-location-execution-20261003T102355Z](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z)；实时状态：[queue_state.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z/runtime/queue_state.json)；启动证据：[LAUNCH_VERIFIED.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z/LAUNCH_VERIFIED.json)；冻结源：[RUN_PINS.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z/RUN_PINS.json)。历史副本保持原样，本追踪是启动核验时快照；实时状态优先。

直接差分回加Local-CD仍plan_only、jobs为空，未加入本轮。
