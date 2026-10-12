# 两任务表审查文件索引

日期：2026-09-20；本审查仅用CPU，未启动或修改GPU队列。核查期间已有后台任务新产出的结果也已纳入。

| 用途 | 文件 |
|---|---|
| Full原六行表：逐行原始JSON、训练日志、模型及配置绝对路径 | [FULL_TABLE_SOURCE_INDEX.md](FULL_TABLE_SOURCE_INDEX.md) |
| Full原表六行按论文顺序直接提取的数值 | [full_table_selected_original_data.csv](full_table_selected_original_data.csv) |
| 论文表格如何修改、版本选择及可用LaTeX | [tables/README.md](tables/README.md) |
| 旧LoRA表：保留42个数值，修正指标与更新规则 | [tables/legacy_lora_corrected.tex](tables/legacy_lora_corrected.tex) |
| 旧LoRA表：均值与样本标准差版本 | [tables/legacy_lora_mean_sd.tex](tables/legacy_lora_mean_sd.tex) |
| 新版ImageNet-R ratio-FS六行独立表 | [tables/new_imagenetr_ratiofs.tex](tables/new_imagenetr_ratiofs.tex) |
| 今日ImageNet-A同未来SGD两行独立表 | [tables/new_imageneta_fixed_future_seed0.tex](tables/new_imageneta_fixed_future_seed0.tex) |
| 新数据集、今日新端点和权重级重算总报告 | [NEW_DATA_AND_RECOMPUTATION.md](NEW_DATA_AND_RECOMPUTATION.md) |
| 今日ImageNet-A同未来SGD两行：FP32准确率、loss、CPU位移 | [today_imageneta_pair_recomputed.csv](today_imageneta_pair_recomputed.csv) |
| 今日ImageNet-A完整两端路径、配置与SHA | [today_imageneta_pair_provenance.json](today_imageneta_pair_provenance.json) |
| 新版LoRA六行、ImageNet-A序列与最新任务状态 | [NEW_LORA_RESULTS.md](NEW_LORA_RESULTS.md) |
| 新版ImageNet-R六组LoRA端点CPU重算 | [new_r_lora_drift_recomputed.csv](new_r_lora_drift_recomputed.csv) |
| 较新Full数据、可恢复序列和CPU核验 | [NEW_FULL_RESULTS.md](NEW_FULL_RESULTS.md) |
| 新Full序列18步及累计纯backbone位移 | [new_full_checkpoint_drift.csv](new_full_checkpoint_drift.csv) |
| 旧Full六行历史快照能否恢复的补查 | [FULL_HISTORY_RECOVERY.md](FULL_HISTORY_RECOVERY.md) |
| 总结、公式、证据限制及论文修正建议 | [AUDIT_REPORT.md](AUDIT_REPORT.md) |
| LoRA 18次运行的来源、配置、数值聚合 | [LORA_PROVENANCE.md](LORA_PROVENANCE.md) |
| LoRA 初始化、K/V、记录时点、合并交叉项、历史代码 | [LORA_METRIC_CODE.md](LORA_METRIC_CODE.md) |
| Full 原始指标、分类头能量、anchor 身份、Fisher floor | [FULL_PROVENANCE.md](FULL_PROVENANCE.md) |
| LoRA 全部原日志/配置路径、字段和哈希 | [lora_sources.json](lora_sources.json) |
| Full 逐行原始 JSON/log/checkpoint 路径和哈希 | [full_sources.json](full_sources.json) |
| LoRA 独立解析的逐 seed 指标 | [lora_per_seed_recomputed.csv](lora_per_seed_recomputed.csv) |
| LoRA 均值、标准差、加权位移、penalty | [lora_summary_recomputed.csv](lora_summary_recomputed.csv) |
| Full 原始字段（含三个额外参考分支） | [full_table_raw_fields.csv](full_table_raw_fields.csv) |
| 两表六行联合对照，明确不同统计范围 | [joint_table_audit.csv](joint_table_audit.csv) |
| Task1 shaping 相对 EWC 的 accuracy/BWT 变化 | [joint_retention_contrasts.csv](joint_retention_contrasts.csv) |
| Full 现存权重 CPU 排查，仅可按报告限制解释 | [full_checkpoint_recompute.json](full_checkpoint_recompute.json) |
| 本次输出 SHA256 清单 | [artifact_manifest.json](artifact_manifest.json) |

原 LoRA 数值来源：

`/data/115-2/users/liying/low-rank-cl-new/result_summary/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/table_source.csv`

LoRA 原始执行脚本：

`/data/115-2/users/liying/low-rank-cl-new/scripts/run_verify_imagenet_r_1x1_table_metrics_s012.sh`

完整配置与日志路径由本目录的两个 `*_sources.json` 提供。原始实验文件及论文未修改。
