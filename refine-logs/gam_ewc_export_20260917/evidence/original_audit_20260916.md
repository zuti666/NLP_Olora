# GAM / GAM+EWC 本地实现与结果审计

审计日期：2026-09-16。仅新增此报告与原始结果索引 CSV；未修改训练代码或启动实验。

结果取自本地 `predict_results.json`，模型名称核对首任务 `adapter_config.json`；配置优先取对应成功运行日志，其次取脚本，并明确不能核实之处。未直接采用旧 tracker 或 Excel 汇总值。

指标为第 4 个任务训练完成后，DBPedia、Amazon、Yahoo、AGNews 四个测试集的 `predict_exact_match_for_<dataset>` 算术平均，单位 %。三顺序均值是 Order1/2/3 的平均，不是多随机种子的均值。每个测试集 7,600 条，最终共 30,400 条。

原始索引：[GAM_GAM_EWC_RAW_20260916.csv](GAM_GAM_EWC_RAW_20260916.csv)。覆盖三个结果根目录的 155 个实验目录（包括对照、RWP 和未完成项），其中 151 个有最终四任务结果；每行保留每数据集分数、实际模型、阶段数、原始 JSON 路径。

## 实现数量与基础策略

真正的 GAM 优化器核心只有 1 套：`src/optimer_gam.py`。它有 4 个主训练组合：SeqLoRA-GAM、IncLoRA-GAM、OLoRA-GAM、EWCLoRA-GAM。不是每个 shell 文件各有一套 GAM。

| 组合/扩展 | 训练参数与跨任务行为 | 实际更新 |
|---|---|---|
| SeqLoRA-GAM | 普通 SeqLoRA gate 训练 `lora_A/B`，冻结 `loranew_A/B` | 四次 closure GAM |
| IncLoRA-GAM | 冻结历史 adapter，每任务训练 fresh `loranew_A/B`，任务结束拼接保存 | 四次 closure GAM，任务损失 |
| OLoRA-GAM | 同上，增加正交约束 | 四次 closure GAM，任务损失 + OLoRA |
| EWCLoRA-GAM | 同 IncLoRA 的增量结构；full-ΔW Fisher 随任务保存 | GAM 梯度做逐张量 EWC 投影，再加 EWC 梯度 |
| GAMFisherShape-SAM-OLoRA-EWC | 同增量结构 | 实际是两次 probe 的 SAM；clean 点加 OLoRA 与 EWC，非四次 probe GAM |
| FisherShape-EWCLoRA | 同增量结构 | 独立 FisherShape 路径，可退化为 SAM；不是另一套标准 GAM |

`ref——code/ewclora_gam_projected.py` 是视觉分类参考实现，当前 NLP 训练入口未引用，不计入已运行分支。

源码证据：`src/optimer_gam.py:246`（四次 closure）；`src/run_uie_lora.py:1158`（策略参数 gate）；`src/peft/utils/save_and_load.py:36`（rank 拼接）；`src/uie_trainer_lora.py:1184`（EWC 投影）；`src/optimer_gamfishershape.py:28`（强制 SAM-equivalent、GAM probe count=0）。

EWC 对当前 fresh 更新 ΔW=B_new A_new 施加 0.5 λ Σ Ω ΔW²；任务末更新 Ω←γΩ+F，保存在 `adapter/ewclora_fisher.pt`。首任务没有历史 Fisher，EWC 从后续任务生效。

当前工作树普通 EWCLoRA / EWCLoRA-GAM 还加入 OLoRA-base 正则；相对 2026-05 的提交基线，这是代码变化。不能假定早期结果由当前未提交源码原样产生。

## 数据和实际配置

所有本报告完整 GAM 系列是 CL_Benchmark 的四任务短序列；未找到可用于本次比较的 15 任务 GAM 完整结果。

| 数据集 | 类型 | 本地训练数 | 测试数 |
|---|---|---:|---:|
| DBPedia | 主题分类 | 14,000 | 7,600 |
| Amazon | 情感分类 | 5,000 | 7,600 |
| Yahoo | 主题分类 | 10,000 | 7,600 |
| AGNews | 新闻分类 | 4,000 | 7,600 |

本地训练集共 33,000 条；是仓库提供的子集，不是各数据集原始全量版本。Order1：DBPedia→Amazon→Yahoo→AGNews；Order2：DBPedia→Amazon→AGNews→Yahoo；Order3：Yahoo→Amazon→AGNews→DBPedia。

共有设置：LoRA rank=8、alpha=32、dropout=.1；T5目标模块为q/v，Llama为q_proj/v_proj；每任务 1 epoch、bf16、constant LR、warmup=0、输入上限512、目标/生成上限50。标准 GAM：grad_rho=.02、grad_norm_rho=.2、beta1/2/3=1、grad_gamma=.03、adaptive=false。EWCLoRA 默认 λ=1、γ=1。脚本名 LLaMA2-1b/3b 实际模型均为 Llama-3.2-1B/3B。

| 系列 | 学习率 | 有效 batch / 累积 | seed | 完成范围 |
|---|---|---|---|---|
| T5-large SeqLoRA-GAM cfgUnified | 每任务1e-3 | 4卡×BS16×GA1=64 | 1993 | 3 orders |
| T5-large EWCLoRA / GAM | 每任务1e-3 | 8卡×BS8×GA1=64 | 1993 | 3 orders |
| T5-small EWCLoRA / GAM | 每任务1e-3 | 目录BS16，实际GPU数应看旧日志 | 1993 | 3 orders |
| Llama 旧 Seq/Inc/O GAM | 首任务1e-3，后续1e-4 | 有效64；微批配置见下文 | 42 | 1B/3B各3 orders |
| Llama-1B EWCLoRA-GAM | 首任务1e-3，后续1e-4 | 5卡×BS1×GA8=40 | 1993 | 3 orders；EWC对照只找到Order1完整结果 |
| Llama-3B EWCLoRA / GAM | 首任务1e-3，后续1e-4 | 8卡×BS1×GA8=64 | 1993 | 3 orders |
| 新 GAMFS-SAM-OLoRA-EWC | 首任务1e-3，后续1e-4 | 单卡×BS1×GA64=64 | 1993 | 1B/3B各3 orders |
| 旧 T5 GAMFS-SAMEquivalent | 每任务1e-3 | 单卡×BS8×GA8=64 | 1993 | 仅Order1 |

旧 Llama Inc/OLoRA-GAM 的成功日志是 BS1×GA8×8卡；SeqLoRA-GAM 1B Order2/3 为 BS4×GA4×4卡，3B 为 BS2×GA8×4卡，均有效64。1B SeqLoRA-GAM Order1 现存脚本日志对应失败的另一次尝试，不能用于认定成功结果的微批配置；最终 checkpoint 的 step 数与有效64相符，微批配置未完全恢复。

新 GAMFS-SAM-OLoRA-EWC：rho=.05、非adaptive、OLoRA λ1=.5/λ2=0、EWC λ=γ=1、raw Fisher、无 trace normalization/damping。新 Llama runner 强制 EWC λ>0，与旧 T5 SAMEquivalent 的 λ=0 不同。最新 Llama 六组均已完成，旧 queue 的 waiting/superseded 状态不能替代最终训练/预测结果。

运行证据示例：`Logs_EWCLoRA_GAM/order1_llama1B_ewclora_gam_seed1993_20260516_003719_tmux.log:3025`（有效batch40）；`logs_and_outputs_llama_1B/run_logs/gamfs_sam_olora_ewc_1b_order1_gpu0_seed1993_20260825.log:615`（SAM模式，GAM禁用）及`:620`（有效batch64）；`scripts_llama/run_gamfishershape_sam_olora_ewc_llama.sh:29`（新方法参数）。

## 最终结果：普通 GAM 对照

下表每项均为三顺序平均；Δ 是 GAM 减去相应 AdamW baseline 的百分点差，相对变化=Δ/baseline。

| 模型 | 策略 | AdamW | SAM | GAM | Δ百分点 | 相对变化 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | SeqLoRA | 63.56 | 68.48 | 71.40 | +7.84 | +12.34% |
| Llama-3.2-1B | IncLoRA | 62.05 | 67.68 | 68.73 | +6.68 | +10.76% |
| Llama-3.2-1B | OLoRA | 56.71 | 63.57 | 63.67 | +6.95 | +12.26% |
| Llama-3.2-3B | SeqLoRA | 67.51 | 73.22 | 74.91 | +7.40 | +10.96% |
| Llama-3.2-3B | IncLoRA | 67.17 | 72.79 | 73.60 | +6.43 | +9.57% |
| Llama-3.2-3B | OLoRA | 63.42 | 68.77 | 71.54 | +8.12 | +12.80% |
| T5-large | SeqLoRA | 64.17 | 68.96 | 65.07 | +0.90 | +1.40% |

## 最终结果：GAM+EWC 对照

| 模型/范围 | EWC | EWC-GAM | Δ百分点 | 相对变化 |
|---|---:|---:|---:|---:|
| T5-small / 3 orders | 52.28 | 51.36 | -0.92 | -1.76% |
| T5-large BS8 / 3 orders | 71.87 | 67.82 | -4.05 | -5.63% |
| Llama-1B / 仅Order1 | 59.31 | 70.84 | +11.52 | +19.43% |
| Llama-3B / 3 orders | 70.19 | 72.96 | +2.77 | +3.95% |

1B EWC-GAM 三顺序平均为70.20，但缺少同范围完整 EWC baseline，不能把它与 EWC 单Order1的59.31直接比较。

## 新 SAM 模式扩展与旧 GAM+EWC

| 模型 | 旧GAM+EWC O1/O2/O3 | 新SAM+OLoRA+EWC O1/O2/O3 | 旧平均 | 新平均 | Δ百分点 |
|---|---|---|---:|---:|---:|
| Llama-3.2-1B | 70.84 / 70.06 / 69.70 | 67.25 / 61.34 / 67.94 | 70.20 | 65.51 | -4.69 |
| Llama-3.2-3B | 72.71 / 73.51 / 72.66 | 73.21 / 72.76 / 68.82 | 72.96 | 71.60 | -1.37 |

新方法仅 3B Order1 相对旧 GAM+EWC 增加0.50个百分点；其余5组下降。1B同时改变有效batch40→64；新旧正则接入、累积实现也不同，不能把下降单独归因于SAM或EWC。

旧 T5 SAMEquivalent（EWC λ=0）Order1为67.2039：DBPedia81.3158、Amazon51.0526、Yahoo47.8026、AGNews88.6447；对应 IncLoRA-SAM T5-large Order1为70.9967，低3.7928个百分点。它不属于启用EWC的新Llama实验。

## 所有主要 GAM 系列逐顺序原始分数

| 模型 | 方法 | Order1 | Order2 | Order3 | 三顺序均值 |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B | SeqLoRA-GAM-002 | 72.3487 | 70.3059 | 71.5395 | 71.3980 |
| Llama-3.2-1B | IncLoRA-GAM-002 | 68.9013 | 67.3816 | 69.9046 | 68.7292 |
| Llama-3.2-1B | OLoRA-GAM-002 | 65.8059 | 58.2796 | 66.9178 | 63.6678 |
| Llama-3.2-1B | EWCLoRA-GAM | 70.8355 | 70.0592 | 69.7007 | 70.1985 |
| Llama-3.2-1B | GAMFisherShape-SAM-OLoRA-EWC | 67.2500 | 61.3388 | 67.9375 | 65.5088 |
| Llama-3.2-3B | SeqLoRA-GAM-002 | 74.7631 | 75.2171 | 74.7467 | 74.9090 |
| Llama-3.2-3B | IncLoRA-GAM-002 | 72.8553 | 73.4803 | 74.4638 | 73.5998 |
| Llama-3.2-3B | OLoRA-GAM-002 | 71.6151 | 70.9441 | 72.0493 | 71.5362 |
| Llama-3.2-3B | EWCLoRA-GAM | 72.7138 | 73.5132 | 72.6612 | 72.9627 |
| Llama-3.2-3B | GAMFisherShape-SAM-OLoRA-EWC | 73.2105 | 72.7599 | 68.8158 | 71.5954 |
| T5-large | SeqLoRA-GAM cfgUnified | 65.5099 | 55.5592 | 74.1447 | 65.0713 |
| T5-large | EWC-GAM | 64.7138 | 67.1447 | 71.5987 | 67.8191 |
| T5-small | EWC-GAM | 43.9079 | 53.7006 | 56.4671 | 51.3585 |

## 解读、异常和下一步

1. 普通 GAM 在本地 Llama 1B/3B 六组策略平均值上均高于 AdamW；也均高于SAM，但1B OLoRA比SAM仅高0.09个百分点。只有单seed，不能宣称统计显著。
2. T5-large SeqLoRA-GAM 三顺序平均比旧SeqLoRA高0.90个百分点；Order2下降10.68个百分点，顺序敏感。当前 cfgUnified Order1为65.51，旧同名非cfgUnified运行只有47.39；两者是不同运行配置，不能混成重复seed。
3. GAM+EWC在Llama上的观察结果较好，在T5上的观察结果较差；T5-large比EWC低4.05个百分点，T5-small低0.92。不能形成跨模型稳定提升的结论。
4. 旧普通Llama GAM使用seed42，EWC/GAMFS使用seed1993；1B还存在batch64与40差异。因此EWCLoRA-GAM与IncLoRA-GAM之间不是严格的“只增加EWC”消融。
5. T5 Order1名为 `IncLoRA_T5large...redo1_seed-1993` 的adapter实际基座是t5-small；最终48.0099与small目录相同。该目录不能作为T5-large基线。旧tracker中80.54、81.77、63.17等数值也不能替代当前JSON；当前T5-large OLoRA/SAM Order1分别为77.5855/78.5066。
6. T5 IncLoRA-GAM/OLoRA-GAM有脚本和旧tracker记录，但本次未找到配套完整原始预测JSON，未以tracker数值加入结果表。Llama这两类有完整结果。
7. 1B EWCLoRA只找到Order1完整baseline；Order2仅首任务目录，Order3无对应完整结果。部分早期失败目录、queue smoke和seed42单Order扩展实验不作为三顺序主结果。
8. 当前最有价值的后续实验是：统一模型、seed、有效batch、AdamW底层参数及OLoRA系数，补齐1B EWC的Order2/3，再对IncLoRA-GAM与EWC-GAM做严格消融；若继续新SAM扩展，优先排查Order2/3下降，并用至少3个seed复验。本次未启动这些实验。
