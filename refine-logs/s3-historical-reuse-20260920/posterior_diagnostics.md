# 已有后验存档的 CPU 机制诊断

**20/20 存档可复用为机制诊断，已核实与 July 5 v2 运行对应。无需重训。** 每个存档的Fisher trace/max/维数与原始JSON一致；融合因子与对应任务最终A/B checkpoint逐元素完全一致；文件mtime与v2日志保存时刻相差少于5秒。

|历史分支（各10任务、seed2）|终点与保留权重完全相同|Fisher cosine 均值[范围]|Fisher相对L2差均值[范围]|ΔW相对位移均值[范围]|
|---|---:|---:|---:|---:|
|ours|0/10|0.6522 [0.4648, 0.8286]|0.8557 [0.5753, 1.4806]|0.5152% [0.2051, 1.4069]|
|baseline_nodamp_nomerge|10/10|0.6388 [0.5430, 0.8423]|0.8689 [0.5560, 1.1669]|0.0000% [0.0000, 0.0000]|

**关键结论：不融合的 baseline 在10/10任务具有完全相同的终点/保留权重，但两份 Fisher 仍不同。** 因而不能把融合前后Fisher的cosine偏差全部归因于权重融合改变曲率。这里的Fisher为batch-gradient代理，两次估计的随机批次/增强等执行未配对，具有估计器重复噪声。只能报告描述性差异，不能从ours与baseline两组cosine均值相减得到纯融合效应。

历史源码 `b194f04` 的 `_capture_endpoint_posterior_state` 在融合前额外执行一次Fisher，之后常规路径又执行融合后Fisher；函数恢复模型模式和梯度，没有保存/恢复RNG。源代码注释“extra pass never feeds back”只指未进入EWC累积量，不代表随机数轨迹不受影响。

已有anchor指标在同批当前任务测试样本、task-local logits下比较终点/保留权重；逐任务loss/acc差已保留CSV。它是旧轨迹上的局部诊断，不能替代当前十任务最终性能的融合单变量对照。两个历史完整运行FAA为ours51.400、baseline52.073，但baseline同时移除damping与融合，且诊断额外前向可能改变随机数，不能据此归因单个组件。

存档仅有 `theta_end_lora`、`mu_merge_lora`、`fisher_endpoint`、`fisher_merge`；未保存epoch20/40路径因子，不能完整重建路径内部的B@A交叉项。ΔW位移本次直接由24组Q/V的B@A重算；所有张量有限。表中范围/标准差来自同seed10阶段，不是多seed置信区间。

CPU分析峰值RSS约537.0 MiB；未占用GPU，未写原始存档。

逐存档路径/哈希、checkpoint身份验证、计算定义、原始指标来源见 [posterior_diagnostics.json](posterior_diagnostics.json)；20行结果见 [posterior_diagnostics.csv](posterior_diagnostics.csv)。
