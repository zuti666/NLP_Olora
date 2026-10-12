# Llama 表格结果与参数配置核对（2026-09-20）

范围：用户本轮提供的 SeqLoRA、IncLoRA、OLoRA、TreeLoRA、EWCLoRA、FS-EWCLoRA 表格；仅 Llama-3.2-1B/3B。按结果实际来源汇总，不把目录名称、TrainingArguments 默认值或新方法名称直接当作执行证据。本次未启动训练。

## 表格需要处理的地方

- **TreeLoRA 3B**：本次完整运行结果为 **72.02 / 69.01 / 73.46 / 71.50**。用户表中的 68.36 / 69.18 / 73.04 / 70.19 与 EWCLoRA 行相同，不是本次 TreeLoRA 结果。
- **TreeLoRA 1B Avg.**：原始平均值为 **64.875**，最后统一保留两位应为 **64.88**；64.87 与先舍入三个 order 再平均相符。其他 order 显示值 66.93 / 61.53 / 66.16 正确。
- TreeLoRA 行开头缺少一个 `&`。纠正后 3B Order-3 最大值应加粗 TreeLoRA 73.46，取消 EWCLoRA 73.04 的加粗。
- **新 OLoRA 1B**：56.96 / 55.74 / 67.32 / 60.01 暂未找到对应结果及训练日志，不能确定其参数，也不能将旧 OLoRA 参数直接赋给这一行。本仓库可追溯的旧行是 58.61 / 53.12 / 58.42 / 56.71。
- **FS-EWCLoRA 名称映射**：用户这行 1B 69.85 / 68.13 / 70.62 / 69.53 与 3B 72.71 / 73.51 / 72.66 / 72.96，实际对应 `lora_strategy=ewclora_gam`，即 IncLoRA-GAM-EWC。本文按该执行分支描述参数；这些数值不能用来证明另一套 FisherShape 或 SAM 实现已经得到相同结果。1B Order-2 原始值 68.125，表中 68.13 使用十进制四舍五入；应统一显示规则。

## 共同基础参数

下表适用于已经匹配到日志的实验；尚未定位的新 OLoRA 1B 不在确认范围内。

| 参数 | 实际设置 |
|---|---|
| 模型 | Llama-3.2-1B 或 Llama-3.2-3B；冻结基础模型，训练 LoRA |
| LoRA 每个分支 | rank=8，alpha=32，dropout=0.1，target_modules=q_proj/v_proj，bias=none |
| 精度 | BF16 混合精度训练，FP16 关闭；新运行有 BF16 实际前向审计 |
| 每任务训练轮数 | 1 epoch |
| 学习率 | 每个 order 的首任务 1e-3，后面三个任务各 1e-4 |
| 学习率调度 | constant，warmup_steps=0；不是在一个任务内从 1e-3 衰减到 1e-4 |
| 训练 batch | 每卡 1；已经核实的当前表格来源有效 batch 均为 64 |
| 评估 batch | 每卡 4 |
| 长度 | max_source_length=512，max_target_length=50，generation_max_length=50 |
| 梯度裁剪 | max_grad_norm=1.0 |
| Adam betas | (0.9, 0.999)；epsilon 和 weight decay 见下文，不是所有方法相同 |
| 训练数据量 | DBPedia 14000，Amazon 5000，Yahoo 10000，AGNews 4000 |
| 最终评估 | 四任务每个 7600 样本；最后阶段的四任务 normalized exact match 平均为 FAA |
| Avg. | 同一 seed 的三个任务顺序的 FAA 均值；不是三个独立 seed 的均值 |

| Order | 任务顺序 |
|---|---|
| 1 | DBPedia → Amazon → Yahoo → AGNews |
| 2 | DBPedia → Amazon → AGNews → Yahoo |
| 3 | Yahoo → Amazon → AGNews → DBPedia |

## 每种方法的实际协议

表内 batch 写作“GPU 数 × 每卡 batch × GA”。`data_seed=None` 表示未独立指定数据种子，训练器使用 seed 作为数据采样种子后备值。

| 方法 | 1B batch / seed / data_seed | 3B batch / seed / data_seed | 方法参数与适配器规模 |
|---|---|---|---|
| SeqLoRA | 8×1×8=64；42 / None | 8×1×8=64；42 / None | `seqlora`；同一 rank-8 adapter 顺序训练；无该方法额外正则 |
| IncLoRA | 8×1×8=64；42 / None | 8×1×8=64；42 / None | `Inclora`；每任务训练新的 rank-8 分支，冻结历史；累计 rank 8→16→24→32 |
| OLoRA | **新行来源待确认**；旧行是 8×1×8=64，42 / None | 8×1×8=64；42 / None | 已核实旧运行 `olora`，λ₁=0.5，λ₂=0；累计 rank 8→16→24→32。不能据此推断新 1B 行 |
| TreeLoRA | 4×1×16=64；42 / None | 4×1×16=64；42 / None | `treelora`；固定 rank-8 A/B；`treelora_reg=0.5`、num_tasks=4；保存 LoRA A 参数历史 |
| EWCLoRA | 4×1×16=64；42 / None；EWC λ=0.5 | 8×1×8=64；1993 / 1993；EWC λ=1 | `ewclora`；γ=1；全训练集 raw Fisher；累计 rank 8→16→24→32 |
| FS-EWCLoRA（本表 GAM-EWC） | 4×1×16=64；1993 / 1993；EWC λ=1 | 8×1×8=64；1993 / 1993；EWC λ=1 | `ewclora_gam`；γ=1；GAM 参数见下；累计 rank 8→16→24→32 |

普通 EWC 与 GAM-EWC 使用全训练集、ΔW 空间的 raw Fisher；本表新运行不做 Fisher normalization 或 damping。旧分支中日志显示的 OLoRA 默认 λ₁=0.5 并不表示 EWC/GAM 实际用了 OLoRA 正则；对应历史候选代码的 EWC/GAM 分支未使用它。新 EWC/GAM 命令显式设 OLoRA λ₁=λ₂=0 以保存这一语义。SeqLoRA/IncLoRA 日志同样可能打印未使用的默认正则或 GAM 参数，不能当作方法启用了这些项。

### 正则与梯度累积的关系

- 普通 EWC 的旧实现先把 CE 除以 GA，再每个微批加入 EWC。当前 1B GA16 使用 λ=0.5，保留旧 GA8、λ=1 的名义权重：16×0.5=8×1。λ 数值不同具有实现原因。
- GAM-EWC 将 CE 在 GA 个微批上平均，EWC 在干净参数处每次优化更新只计算一次方向，因此 1B 从 GA8 改为 GA16 仍保持 λ=1，不能套用纯 EWC 的 λ 减半规则。
- TreeLoRA 先合并 CE 与 Tree 正则，再统一除以 GA。其 Tree 系数在每个 epoch 内按微批进度从 0 线性增加至接近 0.5，无需因 GA16 再减半。历史是 LoRA A 参数轨迹，非 Fisher 或梯度历史。
- 旧 OLoRA 也是 CE/GA 后加入正则。若新 OLoRA 用了不同 GA，必须同时核查正则加入位置及系数，不能仅以有效 batch 相同判断对齐；本次尚未找到该新行来源。

### 实际优化器

| 运行分支 | 底层优化器 | epsilon | weight decay |
|---|---|---:|---:|
| SeqLoRA / IncLoRA / 已核实 OLoRA / TreeLoRA / EWCLoRA | transformers.optimization.AdamW | 1e-6 | 0 |
| 本表 FS-EWCLoRA，即 GAM-EWC | GAM 包装 torch.optim.AdamW | 1e-8 | 0.01 |

新 EWC、TreeLoRA 和 GAM-EWC 的值由实际 optimizer 对象的运行审计证实。历史运行无保留的执行源码 SHA 和 optimizer 对象审计；历史优化器默认值来自对应源码分支及候选历史版本 `898adf6bb13eb6efc00e31764eb2bcabd014498d`，其证据强度低于新运行的对象审计。

**更正旧汇总中的优化器表述**：`refine-logs/nlp_table_audit_20260919/llama3_notes.md` 把公共优化器写为 HF AdamW、epsilon=1e-8，依据是 TrainingArguments 打印值。这不足以说明实际构造参数。普通分支只向 HF AdamW 传入参数和学习率，实际默认 eps=1e-6；`gam-adamw_hf` 分支实际构造 Torch AdamW，默认 weight_decay=0.01，即使 TrainingArguments 打印 weight_decay=0.0。旧报告保留，本报告明确修正其结论。

### GAM-EWC 专属参数（两个模型一致）

| 参数 | 值 |
|---|---:|
| gam_grad_rho | 0.02 |
| gam_grad_norm_rho | 0.2 |
| gam_grad_beta_1 / 2 / 3 | 1 / 1 / 1 |
| gam_grad_gamma | 0.03 |
| gam_adaptive | false |
| gam_grad_reduce | mean |
| gam_perturb_eps / ewclora_project_eps | 1e-12 / 1e-12 |
| ewclora_lambda / ewclora_gamma | 1 / 1 |
| 每次更新 | 4 次 GAM closure；EWC 方向在干净点计算一次 |

GAM 的四次探测在各 rank 本地进行，最终投影后的梯度才跨 rank 取均值。4 卡各 GA16 与 8 卡各 GA8 虽然有效 batch 都是 64，本地非线性探测所用样本数不同，不保证同一数值更新。

## 可比性结论

已核实实验的任务数据规模、基础 LoRA 超参、学习率、epoch、精度和名义有效 batch 一致，但尚不是统一全部条件的一组对照：

1. seed42 与 seed1993 混用；特别是 1B EWCLoRA seed42 与 GAM-EWC seed1993 不属于同 seed 消融。
2. 新运行使用 4 卡 GA16，旧基线及 3B EWC/GAM 使用 8 卡 GA8，采样与数值执行不同；GAM 还有本地探测分组差异。
3. 普通方法与 GAM-EWC 的实际优化器实现、epsilon 和 weight decay 不同。
4. SeqLoRA/TreeLoRA 固定 rank8，与累计 rank32 的增量方法参数预算不同；应说明方法本身的适配器增长规则。
5. 旧结果有训练日志与指标，但没有与新运行同等级的源码/模型哈希和分 rank 审计，不能声称所有历史环境已经精确复现。

论文可如实报告这些现有设置；不能只写“所有方法相同 AdamW、seed42、四卡训练”。新 OLoRA 1B 的配置确认仍需其运行目录或日志。

## 结果与配置来源

所有路径相对于 `/data/140-1/users/liying/O-LoRA`。

- 历史 SeqLoRA / IncLoRA / OLoRA、3B EWC/GAM：`refine-logs/nlp_table_audit_20260919/llama_table_evidence.csv`、`llama1_evidence.json`、`llama3_evidence.json` 保存逐 order 的原始日志、指标及 adapter_config 路径。历史主目录为 `Logs_Scripts_Short_Paper/` 和 `logs_and_outputs_llama_{1B,3B}/`。旧审计包含已经从用户表中换掉的 1B OLoRA/GAM 行，不能不经筛选直接复用整表。
- 1B EWCLoRA：`experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/`。关键文件：`manifest.json`、`commands/`、`logs/`、`final_results_bf16_b64_s42_ddp4_bs1_ga16_lam0p5.json`、`full/order_1/1-dbpedia/runtime_audit_rank0.json`。
- 1B GAM-EWC：`experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/`。关键文件：`manifest.json`、`commands/`、`logs/`、`final_results_gam_ewc_bf16_b64_s1993_ddp4_bs1_ga16_lam1.json`、`SEMANTICS_REVIEW.md`。最终原始预测复核见 `refine-logs/gam_ewc_b64_s1993_20260919/FINAL_RECOMPUTE_CHECK.json`。
- 两模型 TreeLoRA：`experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/`。关键文件：`final_results.json`、`llama1b/final_results.json`、`llama3b/final_results.json`、`commands/`、`logs/`、各正式阶段 `treelora_audit_rank*.json`。方法细节由该目录冻结的 `source/src/treelora.py` 和 `source/src/uie_trainer_treelora.py` 核对；两模型共 96 个正式 rank audit 的配置一致。

TreeLoRA 行修正片段：

```latex
TreeLoRA
& 66.93 & 61.53 & 66.16 & 64.88
& 72.02 & 69.01 & \textbf{73.46} & 71.50 \\
```
