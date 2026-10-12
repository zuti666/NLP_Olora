# NLP 表格补全（2026-09-18）

补跑于 2026-09-17 09:47:01 马德里时间完成，两个 Order 的八个正式阶段全部验证通过。

- `exp_nlp_completed.tex`：按用户提供的五行、两模型、逐 Order 版式补全的 LaTeX 表格。
- `table_sources.csv`：30 个逐 Order 数值的原始 JSON 来源与 SHA256，均重新读取核对。
- `verification.json`：检查时间、完成状态及验证数。

Llama-1B EWC 的 Order2/3 为 FP32 单卡补跑，分别为 61.67、64.45；Order1 的 59.31 为 BF16 五卡旧结果。虽然同 seed1993、有效 batch40，并按梯度累积调整了 EWC 系数，但精度和执行拓扑不同，主表不报告统一配置三 Order 平均。61.81 仅是混合配置的描述性均值。

Llama-3B EWC：68.36 / 69.18 / 73.04，平均 70.19。
旧 GAM+EWC（ewclora_gam）1B：70.84 / 70.06 / 69.70，平均 70.20；3B：72.71 / 73.51 / 72.66，平均 72.96。此处不是 SAM+OLoRA+EWC。

所有均值先用原始精度计算，再显示两位小数。EWC 两行均为单 seed1993，三个任务顺序不表示三个 seed。GAM+EWC 在 3B 上比历史 EWC 均值高 2.77 个百分点（相对约 3.95%），是历史配置比较，不是受控多seed结论。

LaTeX 需要 booktabs、multirow、caption；主论文已有 sec:implement 标签时可直接引用。当前环境没有 pdflatex/tectonic，未生成 PDF。若需要统一的 1B EWC 三顺序均值，需补同一 FP32 单卡配置的 Order1；本次仅核对和补表，没有启动额外训练。
