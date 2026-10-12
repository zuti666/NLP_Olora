# ImageNet-R 两阶段离线评估清单

2026-09-20。仅核查/计划，没有启动新GPU评估或训练。数据集固定ImageNet-R。

| ID | 工作 | 对象与关键配置 | 优先级 | 状态 | 训练需求 |
|---|---|---|---|---|---|
| R01 | 既有六行表及权重库存 | ratiofs_l12c20_full_twostage_s345_v1，seeds3/4/5 | MUST | 核查完成，详见来源清单 | 无 |
| R02 | 已有empirical Fisher原始JSON重分析 | pilot_v1与heldout_v2分别汇总，保留坐标/损失/资格信息 | MUST | CPU复算完成 | 无 |
| G01 | 真GAM与EWC同协议资产匹配 | ImageNet-R；目标旧Adam-GAM无checkpoint；另有新SGD-GAM sidecars | MUST-GAM | GAM候选已找到，对照与冻结base身份待补齐 | 不自动训练 |
| R03 | 纯评估loader适配与sanity | 完整Task0+Task1 AB/FC；seed3 control/fs_ewc | MUST | TODO | 无 |
| R04 | 两端点old EF、loss、drift | old256，共享数据；effective Q/V；local20 | MUST | TODO | 无 |
| R05 | 新任务EF | new256，Task1 endpoint；new local20 | MUST | TODO | 无 |
| R06 | 扩已有seeds4/5并汇总主表 | control/fs_ewc；均值±样本SD及逐seed配对差 | MUST | TODO | 无 |
| R07 | 起点作用支持行 | factored_s1，复用已测Ratio-FS anchor | SHOULD | TODO | 无 |
| R08 | 两任务方向/9节点路径 | 有效权重线段；单位位移敏感度 | OPTIONAL | CONDITIONAL | 无 |
| R09 | full支持性复评 | 排除新增头出生能量；同QV与全backbone分列 | OPTIONAL | CONDITIONAL | 无 |

仅G01匹配成功的同协议资产才可标注EWC vs EWC+GAM；R03–R07对应已有Ratio-FS表，不自动替代GAM目标。
