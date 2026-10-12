# logs_our_joost 结果盘点（2026-09-22）

**当前表35组没有可复用的新seed。** 完整扫描此指定目录后，实际打印的dataset只有ImageNet-A、ImageNet-R、cifar224(CIFAR100)、DomainNet；Aircraft/Cars/CUB/Flowers/Het5均未出现。本结论以日志参数为依据，没有用路径猜测模型或数据集。

共2528文件，575日志，1953 JSON（208任务指标JSON），无YAML配置。解析548个带实际配置的日志块；另有38条缺配置/queue/watcher/失败启动日志。

| Actual dataset | Log blocks | Seeds found | Ranks | Batch | Epochs |
|---|---:|---|---|---|---|
| cifar224 | 69 | 0, 1, 2 | 10 | 128 | 1, 20 |
| domainnet | 16 | 0 | 30 | 128 | 1, 10 |
| imageneta | 248 | 0, 1, 2, 3, 4, 98 | 10 | 128 | 10, 20 |
| imagenetr | 215 | 0, 1, 2, 3, 4 | 10 | 128 | 50 |

日志块数包括stdout和trainer副本及重复attempt，不是独立实验次数。

| Log-evidence status | Count |
|---|---:|
| no_complete_result | 64 |
| invalid_nonfinite_training | 56 |
| completed_finite_matrix | 412 |
| failed_exception | 11 |
| partial_sequence | 5 |

按实际配置、prefix、seed、启动秒生成320个run证据key；228组疑似重复日志，9组同协议同seed不同启动attempt，均单独列出而不冒充多个seed。

完成判定：完整预期任务数的finite CNN下三角矩阵，并且日志没有Loss/CE/EWC NaN/Inf；并不等同于已验证进程exit0。NaN训练即使最后能输出finite准确率也标invalid。矩阵上三角NaN是未见任务正常占位，不标发散。

FAA=最终一行任务宏均值；AAA_stage=各阶段任务宏均值的均值。JSON保存每次run的两指标、矩阵下三角、配置、source SHA、完成状态、重复关系，以及35组逐项不匹配原因。

本目录其他方法均仅盘点、不加入当前七方法新实验：

- `as1_normfisher_fishershape`：357个日志块。
- `as1_normfisher_fishershape_decoupled`：6个日志块。
- `as1_normfisher_fishershape_freshlora_merge`：46个日志块。
- `as1_normfisher_fishershape_freshlora_merge_factor`：3个日志块。
- `as1_normfisher_fishershape_multidirectional`：40个日志块。
- `as1_normfisher_gam`：2个日志块。
- `as1_normfisher_sam`：23个日志块。
- `ewc_fishershape`：45个日志块。
- `ewclora_normfisher_gam`：3个日志块。
- `ewclora_youyue_fitarchitechture`：23个日志块。

复算：`python /data/140-1/users/liying/O-LoRA/refine-logs/seed012_execution_20260922/inventory_joost.py`。只读取指定目录和既有35组配置摘要，输出写当前执行审计目录。源数据不会改动。

指标未配对原日志数：0；读取期间发生写入的文件数：0。尚未直接认证任何其他协议可用于当前表，未计算跨协议SD。
