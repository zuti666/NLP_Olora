# 执行进度快照

核验时间：2026-10-11T09:06:43.456405+02:00（Europe/Paris）。实时状态以 runtime/queue_state.json 为准。

10/10 GPU模型短测、CPU配对门、19项CPU测试及独立文档审查通过；源文件190份、执行冻结文件44份完整。

| 实验 | 状态 | GPU | 已观测真实更新 |
|---|---|---:|---:|
| formal_full_ft_sgd_s2 | running | 4 | 36 / 2350 |
| formal_full_ft_g2_s2 | running | 5 | 17 / 2350 |
| formal_full_ft_sam_s2 | pending | — | 0 / 2350 |
| formal_lora_mirror_s2 | pending | — | 0 / 2350 |
| formal_lora_gam_s2 | pending | — | 0 / 2350 |
| formal_full_ft_mirror_s2 | pending | — | 0 / 2350 |
| formal_full_ft_gam_s2 | pending | — | 0 / 2350 |

另复用已验收LoRA SGD/SAM/g2三行。新增正式共7组，完成后自动汇总RESULTS.csv/RESULTS.md。

本轮为全200类普通监督学习：seed2、ViT-B/16、50epochs、batch128。LoRA rank10/LR=.04；全FT LR=.001。rho=.05/r=1固定，范围内五种方法配对；无Fisher/EWC/权重融合。

只有正式epoch50端点能够用于最终效能结论。当前启动快照不把短测或未完成成绩作为完整结果。
