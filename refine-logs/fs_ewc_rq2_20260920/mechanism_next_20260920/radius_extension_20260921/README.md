# 当前 B 图半径补测

当前状态以 [campaign/STATUS.md](campaign/STATUS.md) 为准。GPU0运行EWCLoRA，GPU3运行FS-EWCLoRA；仅在空闲并取得共享锁后启动。启动检查时两卡已有其他任务，本队列尚未测量新增半径。

新增 r=11,12,13,14,15,16,17,18,19，保留16个原方向；每方法144个新扰动状态，共288个。先复测原点与r10方向0，确认同一checkpoint、样本、坐标、方向及测量数值。无训练。完成后自动合并到原来的7个半径，形成16个半径的三联图。

- [运行配置与复现](RUN.md)
- [模型、数据集、任务与训练配置](CURRENT_PROTOCOL.md)
- [旧LoRA及fullmodel日志的Fisher定义与原始路径](OLD_FISHER_AUDIT.md)
- [启动状态与检查](LAUNCH_CHECK.json)
- [配置JSON](configs/radius_seed0.json)
- [队列计划](campaign/plan.json)

原始测量不改动。新增记录写入results/{sgd,gam}/neighborhood/states.jsonl。只有两条线都通过全部检查，CPU阶段才会更新原三图PDF/PNG，并另外导出本目录Figure_B_neighborhood_extended.pdf/.png；POSTPROCESS.json表示合并已完成。排队状态、环境检查通过与已有原图均不代表新增实验完成。
