# GPU45 启动状态

核验UTC时间：2026-10-09T20:30:19.754168+00:00。物理GPU4的FixedOuter、GPU5的FixedInner已进入正式训练；FixedBoth在队列中等待空闲并自动接续。两组分别已经观察到14/13次有限损失的真实SGD更新，此为启动快照，实时进度以runtime/queue_state.json和outputs内trace为准。

65/65 CPU检查、7/7实际CLI配置检查、独立实现审查、fresh文档执行检查均通过；45固定输入和190源文件SHA未变。四组GPU短测16次更新、配对gate已完成并通过，原版前2更新与冻结正式轨迹逐位相同。7个已运行worker均通过实际分配GPU上的seed2 FP32 kernel witness。

新组仍只有g2最终梯度、3次训练梯度读取/1次原生SGD/1次被动更新后CE；无历史、EWC/Fisher、权重融合、投影、HVP或最终幅度匹配。FixedOuter首步实际内/外距离0.1714350824/0.0500000005；FixedInner为0.0500000008/0.1714350880。只使用GPU4/5，共享index+UUID锁、空闲准入，不改变其他实验。

尚无完整新成绩。正式每组10×50epochs、2550更新；原版完整AAA62.870、FAA45.252、BWT−9.652直接复用。自动验证后结果写RESULTS.md/CSV/JSON。完整配置和命令在suite.json；代码与流程见EXPERIMENT_PLAN.md、RUN.md。原三方法定义保留。
