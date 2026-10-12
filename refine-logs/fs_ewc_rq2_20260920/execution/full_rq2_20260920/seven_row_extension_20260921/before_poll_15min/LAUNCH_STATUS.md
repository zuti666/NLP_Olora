# 七行表补训启动记录

截至2026-09-21 18:50马德里，后台调度器已启动，PID 1649105。此文件是启动快照，后续进展以[campaign/STATUS.md](campaign/STATUS.md)和[pipeline_state.json](campaign/pipeline_state.json)为准。

七行主表复用15个已完成seed单元，新增GAM→SGD（C+F→C）位于倒数第三行。另补6条正式续训：GAM→GAM无EWC、SGD→GAM＋EWC各seed0/1/2，每条50轮。两个Task0父起点均复用完整正式checkpoint，GAM→GAM＋EWC已有结果复用。

启动时GPU0–6全被占用，新队列未启动GPU作业。GPU池是0–6，每15秒检查，最多6条并行；排队状态显示的GPU0为计划默认值，实际物理卡在获得空闲卡与UUID锁时才确定。

自动顺序：两种新配置各1条隔离1轮短测 → CPU验证两份短测输出 → 六条正式续训 → CPU汇总完整21单元七行表。任何作业失败或完成凭据无效即停止后续启动，已运行作业正常结束。短测不填论文结果。

启动前已通过：

- 六条配置的CPU完整父模型/Fisher恢复和新分类头初始化，所有同seed新head与已有正式Task1相同。
- 原始训练代码199份冻结文件SHA、ImageNet-A类别/样本顺序与实际父checkpoint核查；两任务train1259/997，累计test307/561。
- 常规40类CIL测试评估、机制梯度统计、任务末Fisher、checkpoint完整重载、旧任务loss/accuracy和Q/V有效权重位移均开启。endpoint_eval=false只关闭额外诊断模块。
- 训练入口7项CPU回归、队列4项资源/依赖/完成凭据测试；独立文档命令dry-run确认10阶段/10作业/218份文件指纹，正式训练依赖短测验证。

证据：[独立实现审查](PREFLIGHT_REVIEW.md)、[宿主机存活与等待核验](HEALTHCHECK.json)、[CPU_PRELAUNCH_CHECKS.json](CPU_PRELAUNCH_CHECKS.json)、[TRAINING_IMPLEMENTATION.md](TRAINING_IMPLEMENTATION.md)、[DOC_CHECK.md](DOC_CHECK.md)、[LAUNCH.json](LAUNCH.json)。GPU CUDA witness与实际模型短测会在GPU空闲后进行，尚未声称通过。

最终表格会自动写入本目录final_report/，不覆盖此前核心RQ2/S3结果。
