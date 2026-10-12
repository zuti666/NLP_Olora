# 当前Fisher与参数移动字段

2026-09-20。两条当前ImageNet-A续训均完成；已有专门端点诊断和训练日志字段，不需要为已有指标重新训练。机器核验见 [FISHER_DRIFT_AUDIT.json](FISHER_DRIFT_AUDIT.json)，当前数据见 [current_imageneta_fisher_drift.csv](current_imageneta_fisher_drift.csv)。

| 指标 | SGD起点→SGD+EWC | GAM起点→SGD+EWC |
|---|---:|---:|
| 旧任务raw trace，anchor | 1465.4969269028 | 761.9007743250 |
| 旧任务raw trace，Task1后 | 2008.3878732628 | 940.4125323058 |
| 有效Q/V权重位移L2 | .3749495707 | .3498757970 |
| 训练日志delta_norm，L2平方 | .140587178848 | .122413070874 |
| 训练日志Fisher加权平方位移 | 3.294973629e−7 | 1.404429694e−7 |

主表的raw trace是同一固定256张旧任务训练图、固定旧20类head、有效Q/V坐标的逐样本CE梯度平方均值。这里不做trace归一化、正数floor或正则项混入。两个checkpoint测的是同一个旧任务，后一个应命名 `T_old(theta_after)`，不应混称为新任务Fisher。

本次实现持续更新同一组LoRA A/B，位移是 `sqrt(sum ||B_after A_after − B_anchor A_anchor||_F²)`，排除分类头。日志 `delta_norm` 是上述范数的平方，当前导出表已开平方；CPU checkpoint重算和日志相符。它与旧ImageNet-R fresh K/V实现的 `||B_new A_new||` 具有不同定义条件。

训练还保存自己的保护矩阵：各自继承的Task0 batch-gradient proxy经过trace归一化，总trace约1；不等于两矩阵逐坐标相同。另有末端新任务代理字段 `in_task_fisher_final_raw_trace`，分别为27.5214783922/22.6480197571，属于训练估计流程，与上述专门旧任务逐样本测量不同，因此未拿来替换主表Fisher列。辅助字段见 [current_imageneta_training_diagnostics.csv](current_imageneta_training_diagnostics.csv)。

历史ImageNet-A十任务链在Task1的同口径专门raw trace未测；其accuracy/loss/drift可作单列参考，缺失trace不能由归一化buffer或另一次续训的端点代填。参考记录见 [historical_imageneta_reference.csv](historical_imageneta_reference.csv)。

用户六行旧ImageNet-R表的Fisher是各当前任务batch代理，位移是fresh新K/V adapter合并前的有效更新；新旧不同数据、优化器、参数通道、生命周期和Fisher估计器，不拼接成同一列。当前完成版表格见 [imageneta_current_complete.tex](imageneta_current_complete.tex)。
