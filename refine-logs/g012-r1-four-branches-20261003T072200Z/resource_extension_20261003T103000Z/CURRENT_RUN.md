# 当前运行资源与入口

此次仅将资源并行上限3提高到6，允许GPU池仍为[1,4,5,6,3,2]。GPU4的独立文档调用验证通过，控制器交接已通过并保留GPU1/2/6旧worker的原PID/starttime。原训练数值代码、配置、56文件RUN_PINS、历史基准及科学设计均未改变。临时接管PAUSE已移除。

当前控制器入口：`resource_queue.py controller`（本目录）；实际身份和启动记录见`LAUNCH.json`，完整交接凭据见`HANDOFF.json`。不要另起同研究控制器。父`runtime/queue_state.json`为实时状态。

已新启动：GPU4 A_D natural、GPU5 A_C natural、GPU3 O_R natural；保留GPU1 A_P natural、GPU2 O_D natural、GPU6 O_C natural。共12项新增正式训练，当前1完成、6运行、5等待；16项短测已经全部通过。等待项在池内卡释放后自动接续。

首个O_P r1结果通过独立只读核验：任务等权AAA=65.73542579365079%、FAA=50.250%，10任务、2550次实际SGD。其余运行成绩不视为完整结果。

记录时间（UTC）：2026-10-03T10:31:20.665293+00:00
