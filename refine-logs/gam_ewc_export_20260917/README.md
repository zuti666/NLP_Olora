# O-LoRA 项目：GAM / GAM+EWC 方法、配置与结果汇总

整理日期：2026-09-17。来源项目：`/data/140-1/users/liying/O-LoRA`。

**本轮 Llama-1B IncLoRA+EWC 的 Order2、Order3 已全部完成。最终准确率分别为 61.6743% 和 64.4539%。** 两个顺序共八个正式阶段的 adapter/Fisher 验证均通过。运行于2026-09-17 07:47:01 UTC（马德里09:47:01）结束，历时10小时10分13秒，原训练进程已退出。

本目录整理的是 O-LoRA 的 NLP 持续学习实验。原始数据文件、模型权重和完整训练日志保留在来源项目；这里保存结果表、配置、轻量原始证据、运行脚本及源码快照。

## 文件导航

- [results/final_results.csv](results/final_results.csv)：最终四任务分数、实验身份和来源。
- [results/inventory.csv](results/inventory.csv)：运行目录清单，包含未完成或异常命名记录。
- [results/supplement_stage_results.csv](results/supplement_stage_results.csv)：本轮补跑各阶段结果。
- [results/comparisons.csv](results/comparisons.csv)：本轮补跑与历史同Order方法的百分点差和相对变化。
- [configuration/datasets.json](configuration/datasets.json)：数据规模、标签、原始路径及数据文件SHA256。
- [configuration/orders.json](configuration/orders.json)：三种任务顺序。
- [configuration/run_settings.csv](configuration/run_settings.csv)：实验族的实际运行设置和核实限制。
- [evidence/](evidence/)：原始结果JSON、阶段验证、运行状态、实际命令和来源索引。
- [reference/supplement_source/](reference/supplement_source/)：本轮实际执行的源码和配置快照。
- [reference/current_checkout/](reference/current_checkout/)：归档时工作树的相关代码、脚本和旧说明。
- [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json)：归档文件清单与SHA256。

`reference/current_checkout/refine-logs/GAM_GAM_EWC_AUDIT_20260916.md` 是补跑前的历史审计，其中“Order2/3缺结果”描述仅代表当时状态；**当前完成状态以本README及 evidence/supplement 为准**。历史脚本包含旧路径，复现请使用下文的原项目入口。

## 1. 实现数量，以及基于 SeqLoRA 还是 IncLoRA

真正四次探测的 GAM 核心只有一套：`src/optimer_gam.py`。它对应四种训练组合，不能把多个模型/Order的shell脚本当成多套GAM算法。

| 名称 | 策略参数 | LoRA结构 | 实际优化方法 |
|---|---|---|---|
| SeqLoRA-GAM | `seqlora` | 普通SeqLoRA gate：更新共享 `lora_A/B` | 四次closure GAM |
| IncLoRA-GAM | `inclora` | 历史adapter冻结，每任务训练fresh `loranew_A/B` | 四次closure GAM，仅任务目标 |
| OLoRA-GAM | `olora` | 同增量结构，并加正交约束 | GAM优化任务目标+OLoRA |
| 旧GAM+EWC | `ewclora_gam` | **IncLoRA式增量结构** | GAM方向经EWC梯度投影后加回EWC |
| EWC-only / IncLoRA+EWC | `ewclora` | **IncLoRA式增量结构** | AdamW优化CE+EWC；本轮显式关闭OLoRA |
| 新SAM+OLoRA+EWC | `ewclora_gamfishershape` | **IncLoRA式增量结构** | 两次SAM probe，clean点加OLoRA和EWC |

旧GAM+EWC和新SAM+OLoRA+EWC都冻结历史 `lora_A/B`，训练新增 `loranew_A/B`，任务结束后沿rank拼接保存。每任务新增rank8，累计rank为8→16→24→32。

`GAMFisherShapeSAM` 虽然名字带GAM，实际只支持SAM等价模式，日志明确记录 `GAM four-probe path disabled`。本归档将它标为SAM扩展，不作为另一套四次probe GAM结果。`FisherShape-EWCLoRA` 是相邻的独立shaping路径，也不等同标准GAM。

代码证据（归档内相对于 `reference/current_checkout/`）：

- `src/optimer_gam.py:246`：四次closure。
- `src/run_uie_lora.py:1208`：EWCLoRA参数冻结策略。
- `src/peft/utils/save_and_load.py:36`：历史/新增adapter拼接。
- `src/uie_trainer_lora.py:1184`：逐参数张量EWC投影。
- `src/optimer_gamfishershape.py:28`：SAM-equivalent强制配置。

## 2. EWC的具体含义

这里的EWC作用于新增完整权重更新 `DeltaW = B_new @ A_new`，不是直接在A/B因子上使用旧参数锚点：

```text
L_EWC = 0.5 * lambda * sum(Omega_old * DeltaW_new^2)
Omega_t = gamma * Omega_(t-1) + F_t
```

每任务末用纯任务CE的完整DeltaW梯度平方估计Fisher，写入 `adapter/ewclora_fisher.pt`，下一任务随adapter加载。首任务没有历史Fisher，因此EWC为0；后续任务生效。

旧GAM+EWC以逐张量投影接入：

```text
g_final = g_GAM - projection_of_g_GAM_onto_g_EWC + g_EWC
```

新SAM+OLoRA+EWC以clean点加法接入：

```text
g_final = g_SAM(CE) + grad_OLoRA(clean) + grad_EWC(clean)
```

**源码版本限制：** 当前工作树的普通 `ewclora/ewclora_gam` 默认额外调用OLoRA-base，早期提交基线没有该附加项。历史结果无法仅凭当前源码追溯为相同训练目标。本轮纯EWC补跑显式设置两个OLoRA系数为0。`reference/current_checkout` 是当前工作树，不是所有历史运行的精确源码；只有 `reference/supplement_source` 是本次运行保存的源码快照。

## 3. 数据集、任务顺序和指标

使用仓库 `CL_Benchmark` 中的四任务分类子集；`sampling strategy=full` 指使用以下本地文件全量，不是原始数据集全量。

| 数据集 | 类型 | 类别数 | 训练数 | Dev数 | Test数 |
|---|---|---:|---:|---:|---:|
| DBPedia | 主题分类 | 14 | 14,000 | 7,600 | 7,600 |
| Amazon | 情感分类 | 5 | 5,000 | 7,600 | 7,600 |
| Yahoo | 主题分类 | 10 | 10,000 | 7,600 | 7,600 |
| AGNews | 新闻分类 | 4 | 4,000 | 7,600 | 7,600 |

总训练数33,000；每个最终模型测试30,400条。

本地四任务的dev与test文件逐字节相同（见数据SHA256）。本轮补跑使用 `evaluation_strategy=no`，未启用 `do_eval`，不以dev选择checkpoint；不能将这两份文件视为独立验证集和测试集。

| Order | 顺序 |
|---|---|
| 1 | DBPedia → Amazon → Yahoo → AGNews |
| 2 | DBPedia → Amazon → AGNews → Yahoo |
| 3 | Yahoo → Amazon → AGNews → DBPedia |

主指标为最后任务训练完成后的四数据集 `predict_exact_match_for_<dataset>` 平均（%），或等价的最终 `predict_exact_match`。两个字段因各自四舍五入可能差0.0001。不得把ROUGE、训练中间阶段已见任务均值当作最终四任务准确率。

**所有主比较均只有一个seed。三个Order不是三个seed。** 旧普通Llama GAM使用42；EWC、GAM+EWC、新SAM扩展及本轮补跑使用1993。未用这些结果报告多seed标准差或统计显著性。

## 4. 运行配置

共同设置：LoRA rank8、alpha32、dropout0.1；T5作用于q/v，Llama作用于q_proj/v_proj；每任务1 epoch；输入上限512、目标/生成上限50；constant学习率，无warmup。

T5学习率全程1e-3；Llama首任务1e-3、后续1e-4。脚本名中的LLaMA2-1b/3b实际加载的是 **Llama-3.2-1B/3B**。

| 实验族 | seed | 实际有效batch | 主要设置 |
|---|---:|---:|---|
| 旧普通Llama Seq/Inc/O GAM | 42 | 64 | BF16；rho=.02、norm_rho=.2、beta1/2/3=1、gamma=.03 |
| 旧Llama-1B EWC/GAM+EWC | 1993 | 40 | BF16；5卡×BS1×GA8；EWC lambda=1、gamma=1 |
| 旧Llama-3B EWC/GAM+EWC | 1993 | 64 | BF16；8卡×BS1×GA8；EWC lambda=1、gamma=1 |
| 新SAM+OLoRA+EWC（1B/3B） | 1993 | 64 | BF16；单卡BS1×GA64；rho=.05；OLoRA lambda1=.5/lambda2=0；EWC lambda=1 |
| 本轮IncLoRA+EWC（1B O2/O3） | 1993 | 40 | **FP32；单卡BS1×GA40；CLI EWC lambda=.2；OLoRA=0** |
| T5-large Seq-GAM cfgUnified | 1993 | 64 | BF16；4卡×BS16×GA1 |
| T5-large EWC/GAM+EWC | 1993 | 64 | BF16；8卡×BS8×GA1 |

旧Inc/OLoRA-GAM为8卡BS1×GA8；Seq-GAM 1B O2/O3为4卡BS4×GA4，3B为4卡BS2×GA8。1B Seq-GAM Order1保留日志对应另一轮失败尝试，成功结果的microbatch未完全恢复；step数支持有效batch64。目录名`Ba1-GA8`不能代替实际日志。

本轮使用GPU9（Quadro RTX 6000，24 GiB），不支持BF16，因此选择FP32。保留旧有效batch40和Fisher微批BS1。因为旧/当前训练循环先将CE除以GA，再加入未除GA的EWC，单卡GA40的CLI系数使用 `1×8/40=.2`，保持旧GA8、lambda1的相对正则强度。**这不表示FP32单卡与旧BF16五卡逐位等价。**

Fisher：raw Fisher，gamma1，全训练集，无trace normalization/damping。补跑优化器仍为 `transformers.AdamW` 的 `adamw_hf` 分支（实际默认epsilon=1e-6、weight_decay=0）。普通GAM底层是 `torch.optim.AdamW`；名称相似不保证底层默认参数相同。

## 5. 最新完整结果：Llama-1B

单位%；三种方法的历史结果与本轮EWC补跑逐Order排列。EWC列的Order1来自旧BF16运行，Order2/3来自本轮FP32补跑。

| Order | EWC-only | IncLoRA-GAM | GAM+EWC | 新SAM+OLoRA+EWC |
|---|---:|---:|---:|---:|
| 1 | 59.3125（旧） | 68.9013 | 70.8355 | 67.2500 |
| 2 | **61.6743（补跑）** | 67.3816 | 70.0592 | 61.3388 |
| 3 | **64.4539（补跑）** | 69.9046 | 69.7007 | 67.9375 |
| 同实验族三Order平均 | 不提供同配置均值 | 68.7292 | 70.1985 | 65.5088 |

若仅做描述性计算，旧O1+新O2/O3的EWC均值为61.8136%，必须标为**混合配置均值**，不能当作统一配置三Order基线。当前缺口由“没有O2/O3结果”变为“需要统一精度/拓扑重跑O1才能形成完全匹配的三Order基线”。

本轮最终分数据集结果：

| Order | DBPedia | Amazon | Yahoo | AGNews | 四任务平均 |
|---|---:|---:|---:|---:|---:|
| 2 | 78.4474 | 31.5526 | 66.4342 | 70.2632 | **61.6743** |
| 3 | 80.6842 | 40.2237 | 55.0526 | 81.8553 | **64.4539** |

| 本轮EWC补跑 − 历史方法（百分点） | Order2 | Order3 |
|---|---:|---:|
| IncLoRA-GAM | -5.7073 | -5.4507 |
| GAM+EWC | -8.3849 | -5.2468 |
| 新SAM+OLoRA+EWC | +0.3355 | -3.4836 |

本轮每个Order rank8→16→24→32、Fisher更新1→2→3→4，阶段2–4历史adapter前缀完全一致，所有adapter/Fisher数值有限。两任务smoke另存，未计入正式结果；smoke也确认EWC惩罚在新增参数更新后为正。

## 6. Llama-3B与T5对照

| Llama-3B | EWC-only | IncLoRA-GAM | GAM+EWC | 新SAM+OLoRA+EWC |
|---|---:|---:|---:|---:|
| Order1 | 68.3586 | 72.8553 | 72.7138 | 73.2105 |
| Order2 | 69.1776 | 73.4803 | 73.5132 | 72.7599 |
| Order3 | 73.0395 | 74.4638 | 72.6612 | 68.8158 |
| 平均 | 70.1919 | 73.5998 | 72.9627 | 71.5954 |

普通GAM相对旧AdamW/SAM的三Order平均：

| 模型 | 策略 | AdamW | SAM | GAM | GAM−AdamW（百分点） |
|---|---|---:|---:|---:|---:|
| Llama-1B | SeqLoRA | 63.56 | 68.48 | 71.40 | +7.84 |
| Llama-1B | IncLoRA | 62.05 | 67.68 | 68.73 | +6.68 |
| Llama-1B | OLoRA | 56.71 | 63.57 | 63.67 | +6.95 |
| Llama-3B | SeqLoRA | 67.51 | 73.22 | 74.91 | +7.40 |
| Llama-3B | IncLoRA | 67.17 | 72.79 | 73.60 | +6.43 |
| Llama-3B | OLoRA | 63.42 | 68.77 | 71.54 | +8.12 |
| T5-large | SeqLoRA | 64.17 | 68.96 | 65.07 | +0.90 |

| T5 EWC结果 | Order1 | Order2 | Order3 | 平均 |
|---|---:|---:|---:|---:|
| T5-small EWC | 44.0033 | 57.7697 | 55.0658 | 52.2796 |
| T5-small GAM+EWC | 43.9079 | 53.7006 | 56.4671 | 51.3585 |
| T5-large EWC（BS8） | 71.5724 | 72.5691 | 71.4605 | 71.8673 |
| T5-large GAM+EWC（BS8） | 64.7138 | 67.1447 | 71.5987 | 67.8191 |

T5旧 `GAMFisherShape-SAMEquivalent` 仅Order1完整，结果67.2039%，其EWC lambda=0、OLoRA=0，属于SAM等价验证，不是启用EWC的方法结果。

## 7. 可支持的结论与比较限制

1. 普通GAM在Llama1B/3B的Seq/Inc/O三种策略上均高于各自AdamW，三Order平均提升6.43–8.12个百分点；没有多seed显著性证据。
2. 3B旧GAM+EWC比EWC-only平均高2.77点，比IncLoRA-GAM低0.64点；T5-large比EWC低4.05点。因此目前不支持“增加EWC在所有模型上稳定提升”。
3. 最新SAM+OLoRA+EWC比旧GAM+EWC平均低4.69点（1B）和1.37点（3B）。它不是四次probe GAM；两组在1B上还改变有效batch40→64。
4. 本轮O2/O3 EWC补跑低于旧GAM+EWC，分别低8.38和5.25点；精度、GPU数/随机采样路径不同，不能把差值归结为单一算法因素。
5. 旧普通GAM seed42，EWC/GAMFS seed1993；只GAM与加EWC之间不是严格的同seed消融。不要把三个任务顺序当三个独立随机seed。
6. T5 Order1目录名 `IncLoRA_T5large...redo1_seed-1993` 实际adapter基座为T5-small；该记录已标注，不能纳入T5-large基线。当前T5 Inc/O-GAM只有脚本/旧tracker，未找到配套完整原始预测JSON，因此未将tracker数字加入主结果表。
7. 旧tracker和早期文档可能与当前原始结果不一致，应以本包原始JSON、运行命令和配置说明为准。对历史代码版本不确定的地方，不将当前源码当成已运行版本的证明。

下一步建议：若继续验证EWC贡献，先统一模型、seed、精度、有效batch、底层AdamW参数与OLoRA系数；补跑FP32单卡的Order1，再做同配置IncLoRA-GAM/EWC-GAM消融，并增加至少3个seed。本次归档没有启动这些额外实验。

## 8. 原项目运行入口与复现

本轮真实运行命令在原项目根目录执行：

```bash
cd /data/140-1/users/liying/O-LoRA
python3 scripts_llama/run_inclora_ewc_supplement.py \
  --run-dir experiment_queue/inclora_ewc_supplement_20260916T213647Z \
  --gpu auto --orders 2 3 --precision fp32
```

该命令已经完成。重跑时将run-dir改成新的唯一目录，脚本会拒绝覆盖已存在结果。脚本默认先做两任务smoke，再串行运行两个Order；正式运行会保存每阶段实际参数、源文件hash、adapter和Fisher。

模型位置：`/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B`。
Python环境：`/data/140-1/users/liying/conda_envs/nlora_req/bin/python`。
数据位置：`/data/140-1/users/liying/O-LoRA/CL_Benchmark`。
本轮完整结果：`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_supplement_20260916T213647Z`。

更细的脚本说明：[INCLORA_EWC_SUPPLEMENT.md](reference/current_checkout/scripts_llama/INCLORA_EWC_SUPPLEMENT.md)。历史GAM/EWC shell见 `reference/current_checkout/scripts_order{1,2,3}_llama{1B,3B}`，执行前需按原项目当前硬件与缓存路径调整，不应直接把本归档当作完整训练环境。
