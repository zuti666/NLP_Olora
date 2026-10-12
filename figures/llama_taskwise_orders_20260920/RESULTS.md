# Llama 逐任务真实结果（2026-09-20）

来源先定位、后绘图。数值来自各训练阶段 `predict_results.json` 的四个数据集 exact-match 字段；新运行最后阶段采用已经保存并核验的整数正确数。未从 FAA 倒推中间点，也未对缺失结果插值。

表中 FS-EWCLoRA 对应实际 `ewclora_gam`。新 1B OLoRA 56.96/55.74/67.32 尚缺对应原始来源，以下标缺失；旧 OLoRA 1B 数值未混入。

共同名义 batch64，但 seed、卡数/GA、底层优化器、适配器参数预算存在差异，详见 `protocol_configs.csv`。每个方法/模型只有一个 seed，不绘制置信区间。

横轴表示依次训练完第1至第4个任务。每条数据集曲线从该任务首次出现开始；未训练任务的早期分数即使在旧日志存在，也不纳入本图。Accuracy 为 normalized exact match (%)。

## Llama-3.2-1B：最终 FAA

| Method | Order-1 | Order-2 | Order-3 | Avg. |
|---|---:|---:|---:|---:|
| SeqLoRA | 66.90 | 55.71 | 68.06 | 63.56 |
| IncLoRA | 64.52 | 56.19 | 65.44 | 62.05 |
| OLoRA | — | — | — | — |
| TreeLoRA | 66.93 | 61.53 | 66.16 | 64.88 |
| EWCLoRA | 67.19 | 61.92 | 66.54 | 65.22 |
| FS-EWCLoRA | 69.85 | 68.13 | 70.62 | 69.53 |

### Order-1: dbpedia → amazon → yahoo → agnews

下表为各阶段已见任务平均准确率 AA；第4列即最终 FAA。早期阶段评估任务集合不同，AA 的下降不能单独解释为遗忘。

| Method | After task 1 | After task 2 | After task 3 | After task 4 (FAA) |
|---|---:|---:|---:|---:|
| SeqLoRA | 77.11 | 64.92 | 57.54 | 66.90 |
| IncLoRA | 77.11 | 64.02 | 55.34 | 64.52 |
| OLoRA | — | — | — | — |
| TreeLoRA | 77.42 | 65.39 | 60.32 | 66.93 |
| EWCLoRA | 77.42 | 65.55 | 59.83 | 67.19 |
| FS-EWCLoRA | 78.07 | 65.33 | 64.25 | 69.85 |

每个数据集的具体曲线点（—为尚未学到该任务，不代表0）：

| Method | Dataset | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---:|---:|---:|---:|
| SeqLoRA | dbpedia | 77.11 | 77.03 | 72.45 | 80.83 |
| SeqLoRA | amazon | — | 52.82 | 33.59 | 35.70 |
| SeqLoRA | yahoo | — | — | 66.58 | 61.39 |
| SeqLoRA | agnews | — | — | — | 89.68 |
| IncLoRA | dbpedia | 77.11 | 76.80 | 72.76 | 77.53 |
| IncLoRA | amazon | — | 51.24 | 26.43 | 29.25 |
| IncLoRA | yahoo | — | — | 66.83 | 62.51 |
| IncLoRA | agnews | — | — | — | 88.80 |
| TreeLoRA | dbpedia | 77.42 | 76.72 | 77.07 | 80.87 |
| TreeLoRA | amazon | — | 54.05 | 36.86 | 34.83 |
| TreeLoRA | yahoo | — | — | 67.05 | 63.71 |
| TreeLoRA | agnews | — | — | — | 88.32 |
| EWCLoRA | dbpedia | 77.42 | 77.18 | 77.41 | 79.30 |
| EWCLoRA | amazon | — | 53.91 | 35.59 | 37.12 |
| EWCLoRA | yahoo | — | — | 66.49 | 63.78 |
| EWCLoRA | agnews | — | — | — | 88.58 |
| FS-EWCLoRA | dbpedia | 78.07 | 78.01 | 77.93 | 78.01 |
| FS-EWCLoRA | amazon | — | 52.64 | 47.38 | 48.74 |
| FS-EWCLoRA | yahoo | — | — | 67.45 | 66.46 |
| FS-EWCLoRA | agnews | — | — | — | 86.17 |

### Order-2: dbpedia → amazon → agnews → yahoo

下表为各阶段已见任务平均准确率 AA；第4列即最终 FAA。早期阶段评估任务集合不同，AA 的下降不能单独解释为遗忘。

| Method | After task 1 | After task 2 | After task 3 | After task 4 (FAA) |
|---|---:|---:|---:|---:|
| SeqLoRA | 77.11 | 64.92 | 68.34 | 55.71 |
| IncLoRA | 77.11 | 64.02 | 67.33 | 56.19 |
| OLoRA | — | — | — | — |
| TreeLoRA | 77.42 | 65.39 | 69.84 | 61.53 |
| EWCLoRA | 77.42 | 65.55 | 70.29 | 61.92 |
| FS-EWCLoRA | 78.07 | 65.33 | 70.96 | 68.13 |

每个数据集的具体曲线点（—为尚未学到该任务，不代表0）：

| Method | Dataset | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---:|---:|---:|---:|
| SeqLoRA | dbpedia | 77.11 | 77.03 | 79.43 | 68.08 |
| SeqLoRA | amazon | — | 52.82 | 38.92 | 25.82 |
| SeqLoRA | agnews | — | — | 86.66 | 62.79 |
| SeqLoRA | yahoo | — | — | — | 66.16 |
| IncLoRA | dbpedia | 77.11 | 76.80 | 77.95 | 69.78 |
| IncLoRA | amazon | — | 51.24 | 38.54 | 30.42 |
| IncLoRA | agnews | — | — | 85.51 | 57.79 |
| IncLoRA | yahoo | — | — | — | 66.79 |
| TreeLoRA | dbpedia | 77.42 | 76.72 | 79.30 | 77.05 |
| TreeLoRA | amazon | — | 54.05 | 43.22 | 29.29 |
| TreeLoRA | agnews | — | — | 87.00 | 72.53 |
| TreeLoRA | yahoo | — | — | — | 67.25 |
| EWCLoRA | dbpedia | 77.42 | 77.18 | 78.04 | 73.59 |
| EWCLoRA | amazon | — | 53.91 | 47.00 | 38.99 |
| EWCLoRA | agnews | — | — | 85.84 | 68.17 |
| EWCLoRA | yahoo | — | — | — | 66.92 |
| FS-EWCLoRA | dbpedia | 78.07 | 78.01 | 78.05 | 78.04 |
| FS-EWCLoRA | amazon | — | 52.64 | 52.14 | 50.58 |
| FS-EWCLoRA | agnews | — | — | 82.70 | 76.50 |
| FS-EWCLoRA | yahoo | — | — | — | 67.38 |

### Order-3: yahoo → amazon → agnews → dbpedia

下表为各阶段已见任务平均准确率 AA；第4列即最终 FAA。早期阶段评估任务集合不同，AA 的下降不能单独解释为遗忘。

| Method | After task 1 | After task 2 | After task 3 | After task 4 (FAA) |
|---|---:|---:|---:|---:|
| SeqLoRA | 67.78 | 61.20 | 63.15 | 68.06 |
| IncLoRA | 67.78 | 59.70 | 59.39 | 65.44 |
| OLoRA | — | — | — | — |
| TreeLoRA | 68.33 | 61.14 | 67.23 | 66.16 |
| EWCLoRA | 68.33 | 60.50 | 62.56 | 66.54 |
| FS-EWCLoRA | 68.38 | 60.38 | 67.65 | 70.62 |

每个数据集的具体曲线点（—为尚未学到该任务，不代表0）：

| Method | Dataset | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---:|---:|---:|---:|
| SeqLoRA | yahoo | 67.78 | 66.83 | 58.42 | 59.21 |
| SeqLoRA | amazon | — | 55.57 | 41.59 | 46.62 |
| SeqLoRA | agnews | — | — | 89.43 | 84.36 |
| SeqLoRA | dbpedia | — | — | — | 82.07 |
| IncLoRA | yahoo | 67.78 | 64.62 | 47.62 | 55.62 |
| IncLoRA | amazon | — | 54.78 | 42.66 | 50.97 |
| IncLoRA | agnews | — | — | 87.89 | 77.42 |
| IncLoRA | dbpedia | — | — | — | 77.76 |
| TreeLoRA | yahoo | 68.33 | 67.53 | 60.57 | 56.29 |
| TreeLoRA | amazon | — | 54.75 | 51.46 | 46.71 |
| TreeLoRA | agnews | — | — | 89.66 | 83.67 |
| TreeLoRA | dbpedia | — | — | — | 77.99 |
| EWCLoRA | yahoo | 68.33 | 66.41 | 49.86 | 58.76 |
| EWCLoRA | amazon | — | 54.59 | 49.21 | 50.59 |
| EWCLoRA | agnews | — | — | 88.61 | 79.28 |
| EWCLoRA | dbpedia | — | — | — | 77.51 |
| FS-EWCLoRA | yahoo | 68.38 | 67.76 | 65.11 | 65.20 |
| FS-EWCLoRA | amazon | — | 52.99 | 52.37 | 48.93 |
| FS-EWCLoRA | agnews | — | — | 85.49 | 87.17 |
| FS-EWCLoRA | dbpedia | — | — | — | 81.17 |
## Llama-3.2-3B：最终 FAA

| Method | Order-1 | Order-2 | Order-3 | Avg. |
|---|---:|---:|---:|---:|
| SeqLoRA | 67.86 | 67.26 | 67.40 | 67.51 |
| IncLoRA | 65.23 | 66.74 | 69.55 | 67.17 |
| OLoRA | 63.37 | 62.97 | 63.92 | 63.42 |
| TreeLoRA | 72.02 | 69.01 | 73.46 | 71.50 |
| EWCLoRA | 68.36 | 69.18 | 73.04 | 70.19 |
| FS-EWCLoRA | 72.71 | 73.51 | 72.66 | 72.96 |

### Order-1: dbpedia → amazon → yahoo → agnews

下表为各阶段已见任务平均准确率 AA；第4列即最终 FAA。早期阶段评估任务集合不同，AA 的下降不能单独解释为遗忘。

| Method | After task 1 | After task 2 | After task 3 | After task 4 (FAA) |
|---|---:|---:|---:|---:|
| SeqLoRA | 79.41 | 63.94 | 61.66 | 67.86 |
| IncLoRA | 79.41 | 61.87 | 58.52 | 65.23 |
| OLoRA | 79.41 | 57.85 | 60.47 | 63.37 |
| TreeLoRA | 79.96 | 66.51 | 64.72 | 72.02 |
| EWCLoRA | 77.17 | 66.63 | 62.49 | 68.36 |
| FS-EWCLoRA | 78.28 | 67.86 | 66.94 | 72.71 |

每个数据集的具体曲线点（—为尚未学到该任务，不代表0）：

| Method | Dataset | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---:|---:|---:|---:|
| SeqLoRA | dbpedia | 79.41 | 72.45 | 74.50 | 72.07 |
| SeqLoRA | amazon | — | 55.43 | 42.32 | 44.57 |
| SeqLoRA | yahoo | — | — | 68.17 | 64.39 |
| SeqLoRA | agnews | — | — | — | 90.39 |
| IncLoRA | dbpedia | 79.41 | 69.61 | 77.03 | 74.74 |
| IncLoRA | amazon | — | 54.13 | 30.36 | 33.58 |
| IncLoRA | yahoo | — | — | 68.17 | 62.74 |
| IncLoRA | agnews | — | — | — | 89.87 |
| OLoRA | dbpedia | 79.41 | 77.72 | 83.57 | 82.70 |
| OLoRA | amazon | — | 37.97 | 33.22 | 27.25 |
| OLoRA | yahoo | — | — | 64.62 | 59.45 |
| OLoRA | agnews | — | — | — | 84.07 |
| TreeLoRA | dbpedia | 79.96 | 77.30 | 83.78 | 83.68 |
| TreeLoRA | amazon | — | 55.71 | 41.86 | 47.55 |
| TreeLoRA | yahoo | — | — | 68.53 | 66.29 |
| TreeLoRA | agnews | — | — | — | 90.54 |
| EWCLoRA | dbpedia | 77.17 | 77.57 | 81.04 | 80.37 |
| EWCLoRA | amazon | — | 55.68 | 38.41 | 39.18 |
| EWCLoRA | yahoo | — | — | 68.01 | 63.59 |
| EWCLoRA | agnews | — | — | — | 90.29 |
| FS-EWCLoRA | dbpedia | 78.28 | 78.11 | 79.58 | 82.00 |
| FS-EWCLoRA | amazon | — | 57.61 | 52.89 | 52.22 |
| FS-EWCLoRA | yahoo | — | — | 68.36 | 67.79 |
| FS-EWCLoRA | agnews | — | — | — | 88.84 |

### Order-2: dbpedia → amazon → agnews → yahoo

下表为各阶段已见任务平均准确率 AA；第4列即最终 FAA。早期阶段评估任务集合不同，AA 的下降不能单独解释为遗忘。

| Method | After task 1 | After task 2 | After task 3 | After task 4 (FAA) |
|---|---:|---:|---:|---:|
| SeqLoRA | 79.41 | 63.94 | 73.29 | 67.26 |
| IncLoRA | 79.41 | 61.87 | 70.34 | 66.74 |
| OLoRA | 79.41 | 57.85 | 64.12 | 62.97 |
| TreeLoRA | 79.96 | 66.51 | 73.14 | 69.01 |
| EWCLoRA | 77.17 | 66.63 | 70.10 | 69.18 |
| FS-EWCLoRA | 78.28 | 67.88 | 75.44 | 73.51 |

每个数据集的具体曲线点（—为尚未学到该任务，不代表0）：

| Method | Dataset | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---:|---:|---:|---:|
| SeqLoRA | dbpedia | 79.41 | 72.45 | 81.17 | 73.63 |
| SeqLoRA | amazon | — | 55.43 | 51.16 | 38.21 |
| SeqLoRA | agnews | — | — | 87.54 | 88.78 |
| SeqLoRA | yahoo | — | — | — | 68.43 |
| IncLoRA | dbpedia | 79.41 | 69.61 | 79.64 | 75.87 |
| IncLoRA | amazon | — | 54.13 | 42.57 | 37.96 |
| IncLoRA | agnews | — | — | 88.80 | 85.05 |
| IncLoRA | yahoo | — | — | — | 68.09 |
| OLoRA | dbpedia | 79.41 | 77.72 | 83.09 | 82.45 |
| OLoRA | amazon | — | 37.97 | 29.47 | 24.92 |
| OLoRA | agnews | — | — | 79.79 | 79.79 |
| OLoRA | yahoo | — | — | — | 64.74 |
| TreeLoRA | dbpedia | 79.96 | 77.30 | 84.07 | 83.16 |
| TreeLoRA | amazon | — | 55.71 | 46.78 | 36.22 |
| TreeLoRA | agnews | — | — | 88.57 | 88.20 |
| TreeLoRA | yahoo | — | — | — | 68.47 |
| EWCLoRA | dbpedia | 77.17 | 77.57 | 80.59 | 80.54 |
| EWCLoRA | amazon | — | 55.68 | 40.05 | 42.72 |
| EWCLoRA | agnews | — | — | 89.64 | 85.39 |
| EWCLoRA | yahoo | — | — | — | 68.05 |
| FS-EWCLoRA | dbpedia | 78.28 | 78.11 | 81.93 | 83.17 |
| FS-EWCLoRA | amazon | — | 57.66 | 56.59 | 52.41 |
| FS-EWCLoRA | agnews | — | — | 87.80 | 89.80 |
| FS-EWCLoRA | yahoo | — | — | — | 68.67 |

### Order-3: yahoo → amazon → agnews → dbpedia

下表为各阶段已见任务平均准确率 AA；第4列即最终 FAA。早期阶段评估任务集合不同，AA 的下降不能单独解释为遗忘。

| Method | After task 1 | After task 2 | After task 3 | After task 4 (FAA) |
|---|---:|---:|---:|---:|
| SeqLoRA | 68.39 | 61.09 | 68.89 | 67.40 |
| IncLoRA | 68.39 | 59.82 | 69.49 | 69.55 |
| OLoRA | 68.39 | 57.31 | 61.11 | 63.92 |
| TreeLoRA | 68.33 | 62.11 | 69.50 | 73.46 |
| EWCLoRA | 67.74 | 62.28 | 65.64 | 73.04 |
| FS-EWCLoRA | 68.70 | 62.32 | 70.14 | 72.66 |

每个数据集的具体曲线点（—为尚未学到该任务，不代表0）：

| Method | Dataset | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---:|---:|---:|---:|
| SeqLoRA | yahoo | 68.39 | 65.93 | 64.93 | 62.08 |
| SeqLoRA | amazon | — | 56.25 | 52.05 | 42.25 |
| SeqLoRA | agnews | — | — | 89.67 | 82.91 |
| SeqLoRA | dbpedia | — | — | — | 82.38 |
| IncLoRA | yahoo | 68.39 | 63.83 | 63.28 | 62.00 |
| IncLoRA | amazon | — | 55.80 | 56.01 | 50.09 |
| IncLoRA | agnews | — | — | 89.17 | 86.59 |
| IncLoRA | dbpedia | — | — | — | 79.51 |
| OLoRA | yahoo | 68.39 | 66.07 | 64.89 | 64.84 |
| OLoRA | amazon | — | 48.55 | 43.20 | 43.82 |
| OLoRA | agnews | — | — | 75.25 | 60.28 |
| OLoRA | dbpedia | — | — | — | 86.76 |
| TreeLoRA | yahoo | 68.33 | 68.08 | 62.87 | 65.54 |
| TreeLoRA | amazon | — | 56.13 | 54.58 | 55.64 |
| TreeLoRA | agnews | — | — | 91.07 | 88.83 |
| TreeLoRA | dbpedia | — | — | — | 83.82 |
| EWCLoRA | yahoo | 67.74 | 67.04 | 49.84 | 63.99 |
| EWCLoRA | amazon | — | 57.51 | 56.79 | 56.36 |
| EWCLoRA | agnews | — | — | 90.30 | 88.49 |
| EWCLoRA | dbpedia | — | — | — | 83.33 |
| FS-EWCLoRA | yahoo | 68.70 | 68.17 | 65.33 | 66.43 |
| FS-EWCLoRA | amazon | — | 56.47 | 56.29 | 56.49 |
| FS-EWCLoRA | agnews | — | — | 88.80 | 86.89 |
| FS-EWCLoRA | dbpedia | — | — | — | 80.83 |

## 复现与文件

- `source_manifest.json`：132个阶段的运行路径、日志目录、指标与训练结果 SHA256。
- `taskwise_accuracy.csv`：每个真实曲线点及其原始字段来源。
- `stage_seen_task_aa.csv`：每阶段已见任务 AA，并记录日志总均值是否含未见任务。
- `final_faa_by_order.csv`：最终 FAA；`missing_sources.csv`：未核实的表格行。
- `data_verification.json`：来源/样本量/终点核对记录；本次未重新遍历原始预测 JSONL。
- 复现：在本目录执行 `python build_data.py`，然后 `python gen_llama1b.py` 与 `python gen_llama3b.py`。
