# ImageNet-A r=1：Sum12 / Mean12 / g1+d32 三seed

固定seed0/1/2、rank10、rho=.05、10×20类、50epochs/任务；4次正向探针、精确恢复/R0重放、自然LR=.04/.02。无Fisher/EWC/权重融合/投影/HVP。

本队列仅补6项seed0/1；2项seed2已完成并复用，Mean12 seed2由原有等待队列执行，不重复启动。

| 更新方向 | 完整seed | AAA均值±样本SD | FAA均值±样本SD | 状态 |
|---|---|---:|---:|---|
| g1+g2 | 0;1;2 | 68.609±0.353 | 55.911±1.033 | complete_3_seeds |
| (g1+g2)/2 | 0;1;2 | 66.837±0.262 | 51.225±1.559 | complete_3_seeds |
| g1+(g3-g2) | 0;1;2 | 65.985±0.382 | 49.610±1.746 | complete_3_seeds |

| 作业 | seed | 来源 | 状态 | AAA | FAA |
|---|---:|---|---|---:|---:|
| formal_sum12_r1_k10_s2 | 2 | 复用/旧队列 | completed | 68.605 | 54.722 |
| formal_g1_d32_r1_k10_s2 | 2 | 复用/旧队列 | completed | 65.549 | 48.089 |
| formal_halfsum12_r1_k10_s2 | 2 | 复用/旧队列 | completed | 66.554 | 49.432 |
| formal_sum12_r1_k10_s0 | 0 | 本轮 | completed | 68.257 | 56.428 |
| formal_sum12_r1_k10_s1 | 1 | 本轮 | completed | 68.963 | 56.583 |
| formal_mean12_r1_k10_s0 | 0 | 本轮 | completed | 66.883 | 51.977 |
| formal_mean12_r1_k10_s1 | 1 | 本轮 | completed | 67.072 | 52.265 |
| formal_g1_d32_r1_k10_s0 | 0 | 本轮 | completed | 66.148 | 49.224 |
| formal_g1_d32_r1_k10_s1 | 1 | 本轮 | completed | 66.258 | 51.517 |

只对完整seed0/1/2计算正式均值、样本SD和配对比较；缺失或失败不补零。Mean为最终梯度乘0.5，名义LR不再减半，不等同动态有效BA+FC步长匹配。
此前原生g1+g3多seed采用另一RNG/CE协议，独立保留；本轮三方向不跨协议合并排名。
