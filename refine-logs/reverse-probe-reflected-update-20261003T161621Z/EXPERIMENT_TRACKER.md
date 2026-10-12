# 反向探针差分回加：运行记录

记录时间：2026-10-03T18:35:09.368529+02:00。当前队列真实状态以 `runtime/queue_state.json` 为准。

- 实验：`RB01_Backward_reflected_natural_s2`，物理GPU0；方法 `g0+(g0-gminus)`，负侧单位化探针rho=.05，无外层r。
- CPU检查15/15、独立审查、两真实入口及fresh文档验证通过；41个文件冻结，190源文件核对通过。
- 两任务GPU短测已完成并验证：2更新、4训练CE、8隔离诊断CE；Fisher/融合/投影/HVP均0。
- 正式10任务×50epochs已启动；控制器2947892、正式worker2953237；fresh seed2CUDA witness通过。
- 核验时已观察32步，Task0，Epoch4；每步2CE、1SGD，loss0=1.074621，loss_minus=0.974657。完整AAA/FAA尚未产生。
- 主P/C/CD队列自然完成8/8；其45个冻结文件与原pins均未变化。原Local-CD全套计划仍为空队列。

完成后结果自动收集到 `results.csv` / `results.json`，指标从完整任务矩阵重算；历史四探针/native attention结果只作背景。

代码：`trainer/reflected_step.py`（两探针和唯一更新）与`trainer/reflected_train.py`（原生任务流程和禁用项计数）。完整设置/理论/条件性后续对照见`EXPERIMENT_PLAN.md`。
