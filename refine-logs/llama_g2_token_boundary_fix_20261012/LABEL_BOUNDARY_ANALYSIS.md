# 标签边界与旧结果复核

UTC 快照：2026-10-12T09:15:53.372264+00:00

本报告按实际本地 Llama tokenizer 验证全部 33 类。1B、3B 均有 5 类在旧版训练文本 `Answer:` 与标签拼接时跨边界合并；推理只提供独立 `Answer:`。这证明训练/推理 token 前缀不一致，不能据此证明差值全部由该问题导致。

正式指标采用旧实验一致的 normalized EM；剔除这些 gold 类后的指标仅是条件诊断，任务内样本等权、四任务等权，未改成类别等权，也不能称为修复数据后的 benchmark。剔除五类并不能去掉旧管线的重复 BOS、label 长度 off-by-one 和截断问题。

| 任务 | 全部样本 | 剔除样本 | 保留样本 |
|---|---:|---:|---:|
| dbpedia | 7600 | 1616 | 5984 |
| amazon | 7600 | 1574 | 6026 |
| yahoo | 7600 | 716 | 6884 |
| agnews | 7600 | 0 | 7600 |

| 模型 | Order | 原 FAA 差值 | 剔除诊断 FAA 差值 | strict 原差值 |
|---|---|---:|---:|---:|
| 1B | 1 | -1.095395 | -2.037835 | -1.095395 |
| 1B | 2 | +0.332237 | +0.048655 | +0.332237 |
| 1B | 3 | +0.815789 | -0.588954 | +0.842105 |
| 3B | 1 | +2.651316 | +1.029121 | +2.651316 |
| 3B | 2 | +2.269737 | +0.336416 | +2.269737 |
| 3B | 3 | +1.085526 | +0.774317 | +1.085526 |
| 1B | mean_Orders1_2_3 | +0.017544 | -0.859378 | +0.026316 |
| 3B | mean_Orders1_2_3 | +2.002193 | +0.713285 | +2.002193 |

scratchpad `clean.py` 使用 prediction.strip()==ground_truth；这里正式 normalized EM 与 strict EM 分列。因此用户列出的 1B Order3 +0.84 对应 strict 口径，正式口径为 +0.815789。

Order1/Order2 的首次 dbpedia 预测文件在同模型 G2 中逐字节一致。两个 Order 共享首任务，不能作为独立重复测量。

旧 collator 手工前置 BOS 后又调用 tokenizer 默认 special-token 处理，本地 tokenizer 因而在 prompt 开头产生两次 BOS；独立 label 的默认 tokenization 也额外插一次 BOS，影响旧 loss_mask 的标签长度估计。这些证据已逐标签记录。

受影响类别不等于准确率都为 0%。最终 1B Order1 Natural Place 为 G2 80.8194%、IncLoRA 53.8175%；3B Order1 Educational Institution 为 17.7264%/0%，Natural Place 为 36.8715%/0.3724%。amazon negative 与 yahoo Society & Culture 才是这里普遍接近 0% 的两类。

3B 首任务 Athlete 的 95.0000%（G2）/14.6429%（IncLoRA）已核验。最终三 Order 平均原始 FAA 增益中，受影响五类贡献 +1.391447 个百分点，其它类贡献 +0.610746；这仅是正确样本计数分解，不能证明标签缺陷因果地产生这部分收益。

类级错误统计、旧基线跨 seed 描述性数据和全部精确 token ID/pieces 见同目录 CSV/JSON。这些历史样本不是修复后多 seed 的方法验证。

复现（无 GPU、无网络、无模型权重加载）：
```bash
/data/140-1/users/liying/conda_envs/nlora_req/bin/python scripts_llama/audit_g2_label_boundary.py
```
