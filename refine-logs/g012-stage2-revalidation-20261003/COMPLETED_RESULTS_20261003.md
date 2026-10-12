# 第二阶段普通梯度结果与HVP完整复核

核验时间：2026-10-03T12:30:57.562103+02:00。

4/4普通梯度正式训练、4/4短测已完成并独立核验。每组10任务、500epochs、2550实际更新、10200普通CE梯度；Fisher、实际权重融合、投影、HVP及create_graph为0。24/24 math同状态重复及状态恢复此前通过。全部本轮控制器与作业已退出，无本轮训练仍在运行。Hstart/Hmid自然/匹配共4组仍未训练。

统一：ImageNet-A 10×20类、seed2、ViT-B/16、Q/V LoRA rank10、rho=.05、r=.25、math/FP32、TF32/AMP关闭，原SGD及cosine协议，无EWC/Fisher、融合、投影、replay。所有4组初始参数、新FC行、500个epoch sampler及2550个训练批次hash均配对。

## 任务等权结果

AAA为十个阶段已见任务平均准确率的等权均值，FAA为最终十任务等权均值。LA为任务刚学完的平均准确率；Forget为旧任务遗忘百分点。单seed描述性结果。

| 条件 | 步长 | AAA % | FAA % | LA % | Forget pp | 最终样本准确率 % |
|---|---|---:|---:|---:|---:|---:|
| Sec: g0+(g2−g1) | 自然 | 66.039 | 49.315 | 55.515 | 7.147 | 57.08 |
| Sec: g0+(g2−g1) | 匹配 | 65.145 | 47.525 | 54.208 | 7.643 | 55.76 |
| Residual-only: g2−g1 | 自然 | 31.823 | 0.334 | 18.603 | 20.826 | 0.33 |
| Residual-only: g2−g1 | 匹配 | 19.040 | 0.674 | 14.376 | 16.177 | 0.39 |

## 对比解释

自然步长下，加回g0比仅残差高34.216 AAA、48.981 FAA个百分点；匹配步长下高46.105 AAA、46.851 FAA个百分点。匹配有效步长不能挽救仅残差；这是当前配置下最终g0基底的重要性证据。

Sec匹配相对自然下降0.894 AAA、1.790 FAA个百分点。仅残差自然/匹配的LA为18.603/14.376，后续任务对角准确率也显著下降，因此退化包括新任务学习失败，不只是旧任务遗忘。所有损失及验证字段有限，未发现NaN、OOM、汇总错误或匹配验收失败。

仅残差自然/匹配的U与g0负余弦分别出现在503/2550及794/2550步，Sec两个版本均0/2550步。沿各自不同训练轨迹观察，这支持差分单独不能稳定代替监督梯度的解释，但不是相同状态的全过程反事实因果检验。

本轮主要回答移除最终g0是否有影响；没有同math后端g0-only完整训练，不能仅凭本表宣称差分相对g0的独立增益、g0在所有任务中必需或Hessian机制已验证。

## HVP验收

3/3零更新任务分片完整结束；12/12 FP64 HVP与中心差分的相邻细步长验收通过。生产FP32与高精度参考仅10/12通过，以下两项失败（阈值相对L2<=0.5%，余弦>=.999，要求相邻两点）：

| 失败点 | 生产FP32对FP64 HVP相对误差 % |
|---|---:|
| Task0 / batch1 / Hmid | 0.530677 |
| Task9 / batch0 / Hstart | 0.897491 |

两项余弦均通过，但L2及联合相邻差分窗口不通过。Task0/9的exit1是完整数值门槛拒绝，不是诊断崩溃。12/12同点一阶误差为0；174份张量哈希、510条状态恢复通过，原source/checkpoint/门槛/队列未改。FP64参考使用相同128样本的两次64样本加权meanCE，dtype、计算形状与归约顺序同时不同，不能将差异唯一归因于FP32。

原失败记录保留，HVP的4组正式训练未启动。

## 原始证据

- [结果CSV](/data/140-1/users/liying/O-LoRA/refine-logs/g012-stage2-revalidation-20261003/ordinary_stage2/summary.csv)
- [独立结果审计](/data/140-1/users/liying/O-LoRA/refine-logs/g012-stage2-revalidation-20261003/audits/completed_20261003/independent_results_audit.json)
- [独立HVP审计](/data/140-1/users/liying/O-LoRA/refine-logs/g012-stage2-revalidation-20261003/audits/completed_20261003/hvp_audit.json)
- [普通队列状态](/data/140-1/users/liying/O-LoRA/refine-logs/g012-stage2-revalidation-20261003/ordinary_stage2/runtime/queue_state.json)
- [HVP诊断状态](/data/140-1/users/liying/O-LoRA/refine-logs/g012-stage2-revalidation-20261003/hvp_runtime/queue_state.json)
