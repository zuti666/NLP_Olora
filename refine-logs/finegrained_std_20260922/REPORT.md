# 标准差表格核查（2026-09-22）

当前 7 方法 × 5 数据集的 35 组结果中，只有 EWC/Cars 和 EWC/Flowers 各有两个已核实同配置 seed。其余 33 组只有单次运行，跨 seed 样本标准差、样本方差均不可估计，用 NA 表示。

主表使用“多 seed 均值（样本标准差）”；单次运行保留原分数。因此 EWC 的 Cars、Flowers 中心值替换了上一版最高 FAA 单次结果。原表的加粗和下划线位置保持不变，未按新数值重新评定排名。

| EWC 数据集 | seeds | 原单次 FAA / AAA | 新 FAA：mean (SD) | 新 AAA：mean (SD) |
|---|---|---|---|---|
| Cars | 1, 2 | 34.56 / 43.94 | 33.35 (1.70) | 43.21 (1.03) |
| Flowers | 1, 2 | 53.43 / 65.45 | 52.35 (1.54) | 65.24 (0.30) |

SD 使用分母 n−1；统计对象是各 seed 的最终 FAA、stage-averaged AAA。没有用任务间或 epoch 间波动替代重复运行的标准差。FAA 为最终 CNN 任务矩阵最后一行均值；AAA 为各阶段已见任务均值的平均，不使用原始 JSON 中三角元素加权的 AAA 字段。主表 37 份源矩阵及额外 GAM 协议 6 份矩阵均再次读取、验证 SHA256 并重算。

EWC 保持上一版各数据集所选配置，收集该配置全部可核实的 seed。Cars/Flowers 是 alpha0 wrapper、clean/Fisher RNG 隔离协议，lambda 分别为 80000、10；其 YAML 仅 seed 和实验名称不同。Aircraft/CUB 的所选历史 alpha0 协议、Het5 的 native EWC 各只有一次。该表仍然汇集不同历史配置，不构成全方法、全数据集统一协议的多 seed 对比；按历史最好结果选配置的限制也仍然存在。

另外查到一套独立 RNG 控制协议的 GAM-EWC 三 seed 结果：

| 方法 | 数据集 | seeds | FAA：mean (SD) | AAA：mean (SD) |
|---|---|---|---|---|
| GAM-EWCLoRA | CUB-200 | 1, 2, 3 | 73.72 (1.14) | 76.34 (0.90) |
| GAM-EWCLoRA | Flowers | 1, 2, 3 | 82.91 (1.26) | 89.25 (1.15) |

该协议明确启用 clean/Fisher RNG 隔离和确定性 loader，原主表历史运行缺少相应记录；Flowers 历史 GAM 参数记录亦不完整。因此这两组单列，不将其 SD 附在原表的 74.22、86.75 等单次分数后，也不将历史 seed 0 合入 n=4。

文件：

- `finegrained_std.tex`：请求的完整 5 数据集表格，直接使用已有 ExpRes 宏。
- `finegrained_std_preview.pdf`：编译预览。
- `statistics.json` / `statistics.csv`：均值、样本标准差、样本方差、样本量及源路径。
- `input_table.tex`：修改前的表格，用于核对数值与强调格式。
- `separate_protocol_multiseed.tex`：独立 GAM 协议结果，不能直接混入原表。
- `baselines.json` / `baselines.md`、`cl_gam.json` / `cl_gam.md`、`ewc.json` / `ewc.md`：逐组核查及来源。
- `build_table.py`：从核查结果复算原始矩阵并生成表格。

未启动新实验。
