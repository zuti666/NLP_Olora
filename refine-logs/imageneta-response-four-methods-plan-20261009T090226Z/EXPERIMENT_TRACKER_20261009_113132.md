# 四方法实验追踪

| Run ID | 阶段 | 方法 | 划分 | seed | 预算 | 状态 | 说明 |
|---|---|---|---|---:|---|---|---|
| formal_single200_sgd_s2 | 第一阶段 | sgd | 1×200 | 2 | 50epochs/2350updates | DRAFT_UNREGISTERED | 配置已生成 |
| formal_single200_current_g2_s2 | 第一阶段 | current_g2 | 1×200 | 2 | 同上 | DRAFT_UNREGISTERED | 无历史3梯度 |
| formal_single200_history_fusion_s2 | 第一阶段 | history_fusion | 1×200 | 2 | 同上 | DRAFT_UNREGISTERED | base缓存 |
| formal_single200_history_g2_s2 | 第一阶段 | history_g2 | 1×200 | 2 | 同上 | DRAFT_UNREGISTERED | final完整缓存 |
| formal_cl10_sgd_s2 | 第二阶段 | sgd | 10×20 | 2 | 50epochs/task/2550updates | DRAFT_UNREGISTERED | 真1梯度SGD |
| formal_cl10_current_g2_s2 | 第二阶段 | current_g2 | 10×20 | 2 | 同上 | DRAFT_UNREGISTERED | 同预算定位对照 |
| formal_cl10_history_fusion_s2 | 第二阶段 | history_fusion | 10×20 | 2 | 同上 | VERIFIED_EXISTING_RESULT | 旧HRF来源，AAA60.483/FAA41.297 |
| formal_cl10_history_g2_s2 | 第二阶段 | history_g2 | 10×20 | 2 | 同上 | DRAFT_UNREGISTERED | 缓存上步实际d |

CONFIG_INDEX.json包含8正式草案和8实现短测配置；suite.json只注册8短测和1配对。实时实现状态在runtime/queue_state.json、IMPLEMENTATION_TESTS.md。正式训练未启动，本轮不自动续接。

已有结果来源/验收见results/PRIOR_RESULTS.csv、gates/PRIOR_RESULTS_REUSE_AUDIT.json。四方法代码路径为trainer/response_step.py、response_train.py、layout_lifecycle.py。CPU与GPU实现验证不替代完整50epoch泛化结果。
