# 第二局部探针符号代码扩展

新实现均在本campaign的trainer目录；没有修改旧执行器或项目已有训练源。

- direction_step.py：first_probe_sign强制+1；前g0/g1/g2计算与旧P完全一致。第三点仅用second_probe_sign×rho×N(g2)。最终U由update_base与second_difference_sign构造，未进入GAM分解/投影。按实际FP32参数检查第三点长度/符号，恢复基点后只提交一次SGD。
- four_probe_train.py：严格读取并校验三个因子、方向名字和component_rule；禁止静默忽略新字段。沿用ImageNet-A十任务生命周期和确定性数据/RNG流程，每步4CE/1SGD。
- diagnose_sides：每任务首batch复用已有g0/g1/g2，隔离测量g3+和g3−，分别记2诊断CE/0SGD；完整恢复pending U与R1。额外诊断不算入4CE训练预算。
- test_second_probe_cpu.py：11项CPU验收通过；八格公式、共同前三梯度、两半径旧B1逐位等价、模式/buffers/cache/grad/参数/RNG恢复及Controller训练与诊断计数均覆盖。
- run_queue.py：14短测→CPU数据/初始化配对→14正式，物理GPU2–6动态空闲分配，共享index+UUID锁、双次空闲和无compute PID检查；失败停止新增作业。

```python
# g0/g1/g2来自固定正向P几何
theta3 = theta2 + second_probe_sign * rho * normalize(g2)
g3 = gradient_at(theta3)
anchor = g0 if update_base == "g0" else g1
U = anchor + second_difference_sign * (g3 - g2)
# 精确恢复theta0，再用原学习率做一次自然SGD
```

原计划与未注册配置已归档到plan_archive；suite.json与RUN_PINS.json固定实际执行的配置及源码。GPU短测、正式进展和实际占卡以runtime/queue_state.json为准。
