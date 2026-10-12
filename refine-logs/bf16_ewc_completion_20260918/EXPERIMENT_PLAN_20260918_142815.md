# Llama-1B IncLoRA+EWC：BF16 缺失结果补全实施计划

日期：2026-09-18。精度已由用户确认：**BF16**。
状态：**仅制定计划，尚未启动本计划中的训练。**

## 1. 结论：最少补哪些实验

为补全当前论文表格，最小缺口只有 **Llama-3.2-1B 的 IncLoRA+EWC，seed1993，Order2 和 Order3**。每个 Order 从原始基座开始完成四个任务，共 **2 个完整实验、8 个正式训练阶段**。

前提是能够核对并沿用旧 Order1 的协议：**BF16、5 卡 DDP、每卡 batch1、GA8、有效 batch40、EWC λ1、γ1、纯 IncLoRA+EWC**。精度一致本身不足以证明协议一致。

当前记录尚未证明旧运行与新源码、环境和模型文件完全一致。因此采用“**必补 2 个 + 条件补 1 个**”安排：

- 能恢复并核对旧协议：复用旧 Order1，只跑 Order2、Order3；报告“按已核实历史设置补跑”，不声称逐位复现。
- 无法消除影响训练语义的源码/环境/模型差异：使用同一冻结快照再跑 Order1，形成新的三 Order BF16 EWC 基线；这时需要 **3 个完整实验、12 个阶段**。这是当前证据条件下获得严格统一新基线的稳妥方案。
- 只能取得一张支持 BF16 的卡：转为单卡 BF16 新协议，必须跑 Order1/2/3 三个实验，不能只换精度后与旧五卡 Order1 合并。

| 模型与方法 | 已有 BF16 覆盖 | 本轮动作 |
|---|---|---|
| 1B SeqLoRA、IncLoRA、OLoRA | 各 Order1/2/3 已有完整结果 | 复用，不重复运行 |
| 1B IncLoRA+EWC | Order1：59.3125%；Order2/3 缺完整匹配结果 | **补 Order2、Order3；Order1 视协议核对结果决定** |
| 1B IncLoRA+EWC+GAM | Order1/2/3：70.8355 / 70.0592 / 69.7007 | 复用，不重复运行 |
| 3B IncLoRA+EWC | Order1/2/3：68.3586 / 69.1776 / 73.0395 | 复用，不重复运行 |
| 3B IncLoRA+EWC+GAM | Order1/2/3：72.7138 / 73.5132 / 72.6612 | 复用，不重复运行 |
| 1B IncLoRA+EWC，FP32 补跑 | Order2：61.6743%；Order3：64.4539% | 保留为另一配置的结果，不填作 BF16 结果 |

本轮不增加 T5、SAM、FisherShape、额外 seed 或超参数搜索。此前 GAM+EWC 指 `ewclora_gam`，不是名字带 GAM 的 SAM+OLoRA+EWC。

## 2. 实验要回答的问题

| 目标 | 最少证据 | 论文位置 |
|---|---|---|
| C1：补齐 1B EWC 的三个任务顺序，并在协议一致时给出 Avg. | 每个 Order 均有最终四任务测试结果、配置来源和阶段验证；旧 Order1 通过复用审查，或新跑 Order1 | `tab:exp_nlp` 的 IncLoRA+EWC 行 |
| C2：描述历史 GAM+EWC 与完整 EWC 基线之间的差异 | 明确比较的是哪些模型、Order、seed 和配置；计算逐 Order 及平均差值 | 结果分析；不升级为多 seed 因果结论 |

**验收标准是完整、可追溯、协议清楚，不是准确率必须超过某个阈值。** 若新结果偏低，照实报告，不根据 test 结果调参或挑选重跑。这里补充的是已有持续学习实验，未提出新的模型架构或新算法主张。

## 3. 硬件条件与当前限制

2026-09-18 的主机只读检查显示：GPU0–9 全部为 **Quadro RTX 6000，compute capability 7.5，24 GiB**。当时 GPU2/3 空闲，但这些卡在本项目 `torch==2.1.1+cu121` 环境下不满足原生 BF16 检查；该版本 `torch.cuda.is_bf16_supported()` 要求 NVIDIA compute capability major≥8 且 CUDA≥11。

因此，**不能在当前这十张卡上通过增加空闲卡数完成 BF16 计划**。执行前需要取得支持 BF16 的设备，并逐卡确认支持情况；首选同一主机上五张兼容 GPU。卡号和空闲状态只在实际启动时确定。本计划不包含租卡、抢占或停止其他任务。

五卡方案按 Order 串行调度，同一 Order 的四阶段必须串行；有十张兼容空闲卡且资源允许时，才可将两个 Order 分配到互不重叠的五卡组。每组按 UUID 加锁，检查显存与计算进程，端口独立。

若只有一张 BF16 GPU：BS1、GA40、CLI λ0.2、有效 batch40，其余保持一致，重新运行三个 Order。该方案单列为新协议，不与旧五卡 Order1 合并。

## 4. 冻结的正式配置

以下为首选五卡 BF16 方案；除数据顺序和分阶段学习率外，两个 Order 使用相同设置。

| 项目 | 配置 |
|---|---|
| 模型 | Llama-3.2-1B；本地 `/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B` |
| 方法 | `train_method=lora`，`lora_strategy=ewclora`；冻结历史 `lora_A/B`，训练 fresh `loranew_A/B` |
| 精度 | `bf16=true`、`fp16=false`；沿用 AMP 路径，不手动把整个模型转换为 BF16；保持并记录评估精度和 TF32 设置 |
| DDP | `world_size=5`，`ddp_find_unused_parameters=false` |
| batch | 每卡训练 1，GA8，总 batch=5×1×8=40；每卡评估 4 |
| seed | `seed=1993`，`data_seed=1993`；每 Order 单 seed |
| LoRA | 每任务新增 rank8，alpha32，dropout0.1；Llama 的 q_proj/v_proj；累计 rank8→16→24→32 |
| 优化器 | `optimizer_type=adamw_hf`，实际 `transformers.AdamW`；当前分支默认 β=(0.9,0.999)、eps=1e-6、weight_decay=0、correct_bias=true，执行时记录实际对象设置；不能用 TrainingArguments 中的默认 adam_epsilon 代替实际对象参数 |
| 学习率 | 第一任务 1e-3；第二至第四任务 1e-4 |
| 训练预算 | 每任务 1 epoch；constant scheduler；warmup0；正式运行不设 max_steps 或样本上限 |
| 文本长度 | `max_source_length=512`；`max_target_length=50`；`generation_max_length=50` |
| 指令 | `instruction_strategy=single`；`add_task_name=true`；`add_dataset_name=true` |
| EWC | CLI `ewclora_lambda=1.0`，`ewclora_gamma=1.0`；作用于新增 ΔW，而非直接约束 LoRA 因子 |
| Fisher | `ewclora_fisher_max_batches=0`（全训练集）；raw Fisher；两种 normalization 均 false；damping_beta0 |
| 其他正则 | **显式** `olora_lamda_1=0.0`、`olora_lamda_2=0.0`；无 GAM/SAM/FisherShape |
| 评估/存储 | `do_train`、`do_predict`、`predict_with_generate`；`evaluation_strategy=no`；`save_strategy=no`，但必须保留任务末 adapter/Fisher 和预测结果 |
| 环境 | 当前可用环境：Python3.10.20、torch2.1.1+cu121、transformers4.28.0、datasets1.17.0、accelerate1.4.0；使用仓库内 PEFT；另记录完整依赖与驱动版本 |

**不要在 GA8 的五卡方案中沿用补跑的 λ0.2。** 本实现将 CE 除以 GA 后再加入 EWC；旧 GA8/λ1 与单卡 GA40/λ0.2 保持相同名义相对缩放。五卡恢复 GA8 时，CLI λ 应恢复为1。此关系不保证不同 GPU 数下随机流或结果相同。

## 5. 数据集与任务链

使用当前仓库 `CL_Benchmark` 的本地四任务分类子集，按已有哈希清单核验；不重新采样或重划分。

| 数据集 | 训练样本 | 最终测试样本 | 类别 |
|---|---:|---:|---:|
| DBPedia | 14,000 | 7,600 | 14 |
| Amazon | 5,000 | 7,600 | 5 |
| Yahoo | 10,000 | 7,600 | 10 |
| AGNews | 4,000 | 7,600 | 4 |

| 实验 ID | 阶段1，LR1e-3 | 阶段2，LR1e-4 | 阶段3，LR1e-4 | 阶段4，LR1e-4 |
|---|---|---|---|---|
| B16-EWC-1B-O2-S1993 | DBPedia | Amazon | AGNews | Yahoo |
| B16-EWC-1B-O3-S1993 | Yahoo | Amazon | AGNews | DBPedia |
| B16-EWC-1B-O1-S1993（条件项） | DBPedia | Amazon | Yahoo | AGNews |

每个 Order 必须从原始基座开始，阶段2–4只加载**本 Order 上一阶段**的 adapter 和 Fisher。不能接续 FP32 checkpoint，不能从其他方法或 Order 的最终 checkpoint 开始。

本地 dev/test 文件内容相同；本计划不使用 dev 选 checkpoint，不开展基于 test 的超参数搜索。最终四任务共30,400样本。每阶段只报告对应已见任务，最终 FAA 为四个任务 exact-match 的算术平均，单位%。

## 6. 先修正运行入口，再训练

现有入口有两类不匹配，不能直接执行后宣称与旧 Order1 一致：

1. `scripts_llama/run_inclora_ewc_supplement.py` 使用单卡 Python、GA40、λ0.2。`--precision bf16` 只改变精度开关，不增加五卡 DDP 支持。
2. `scripts_order2_llama1B/EWCLoRA_LLaMA2-1b.sh` 和 Order3 同名脚本目前默认四卡、GA16，有效 batch64；还使用旧模型/环境路径，没有显式关闭当前新增的 OLoRA-base 项。

执行阶段应准备一个独立的 BF16 五卡入口，复用现有补跑的源码快照、阶段验证和结果收集机制，满足下列要求。本次只制定计划，**该新入口尚未实现**。

- 冻结 `src/`、`configs/`、启动器和验证器，写入 SHA256；记录各模型/词表文件及数据哈希、环境版本。
- 五卡统一用指定 Python 的 `-m torch.distributed.run`；确认本地 rank 与设备绑定；为不同任务组使用独立端口。
- `world_size=5 / BS1 / GA8 / λ1 / OLoRA0 / BF16` 设为断言，不自动按空闲卡数改变协议。
- 每阶段记录完整 argv；rank0 保存，其他 rank 等待 barrier；DDP Fisher 先累加各 rank 梯度平方和与样本数，再全局平均。
- 每 Order 使用独立输出目录和缓存；拒绝覆盖历史结果；阶段失败就停止该 Order，不能将部分结果写成 complete。
- 旧模型目录 `/data/140-0/...` 与现在 `/data/140-1/...` 不同，不凭相同模型名称推断文件相同；历史源码候选提交也不能代替当时的运行快照证据。

单阶段命令模板如下，供实现入口时核对参数。变量必须由启动器填写；**这不是已经存在的完整新脚本，也不代表已运行**。

```bash
# BF16_GPUS：已逐卡核验且锁定的五张卡的 UUID 列表。
# FROZEN_SOURCE：本次冻结源码；TASK_CONFIG：相应 Order/数据集配置。
# MODEL_OR_PREV_ADAPTER：首阶段基座、后续阶段的上一 adapter。
CUDA_VISIBLE_DEVICES="$BF16_GPUS" \
  /data/140-1/users/liying/conda_envs/nlora_req/bin/python \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=5 \
  "$FROZEN_SOURCE/src/run_uie_lora.py" \
  --seed 1993 --data_seed 1993 \
  --do_train --do_predict --predict_with_generate \
  --bf16 true --fp16 false \
  --train_method lora --lora_strategy ewclora --lora_dim 8 \
  --optimizer_type adamw_hf \
  --ewclora_lambda 1.0 --ewclora_gamma 1.0 \
  --ewclora_fisher_max_batches 0 \
  --ewclora_normalize_fisher false \
  --ewclora_normalize_accumulated_fisher false \
  --ewclora_fisher_damping_beta 0.0 \
  --olora_lamda_1 0.0 --olora_lamda_2 0.0 \
  --model_name_or_path "$MODEL_OR_PREV_ADAPTER" \
  --data_dir /data/140-1/users/liying/O-LoRA/CL_Benchmark \
  --task_config_dir "$TASK_CONFIG" \
  --instruction_file "$FROZEN_SOURCE/configs/instruction_config.json" \
  --instruction_strategy single \
  --output_dir "$STAGE_OUTPUT" \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate "$STAGE_LR" --num_train_epochs 1 \
  --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
  --ddp_find_unused_parameters false \
  --add_task_name true --add_dataset_name true \
  --lr_scheduler_type constant --warmup_steps 0 \
  --logging_strategy steps --logging_steps 10 \
  --evaluation_strategy no --save_strategy no --report_to none
```

阶段2–4另加 `--ewclora_fisher_path "$MODEL_OR_PREV_ADAPTER"`；首阶段不设置历史 Fisher。环境需提供 `PYTHONPATH="$FROZEN_SOURCE/src"`、NLTK punkt 和离线缓存路径。完整入口还须清理过期分布式环境变量并建立日志、预检与状态文件。

## 7. 运行顺序与放行条件

| 里程碑 | 工作 | 放行条件 | 失败时处理 |
|---|---|---|---|
| M0：协议审查 | 比对旧 O1 的实际日志、代码语义、数据、模型和依赖；确认五张 BF16 GPU | 明确旧 O1 可复用，或明确本轮需要新 O1 | 不以“精度已改”替代审查；硬件不足则等待兼容设备 |
| M1：入口 dry-run | 输出八个正式阶段 argv，不分配训练 | 所有正式配置一致，路径无覆盖，阶段链正确 | 修正入口，不带问题训练 |
| M2：两任务 smoke | 五卡 BF16；每任务2个更新步，最多80训练样本、16预测样本、2个 Fisher 微批 | rank/设备正确；DistributedSampler分片和实际优化器参数正确；各rank更新后参数一致；BF16工作；rank8→16；Fisher更新1→2；第二任务更新后 EWC确实生效；输出只由主进程保存 | 停止并定位；smoke 不计正式结果 |
| M3：正式 O2 | 从基座训练四任务，任务末预测和 Fisher 更新 | 四阶段全部验证通过，最终30,400样本/四任务完整 | 保留失败记录，不填主表；修复后新 run_id 重跑 |
| M4：正式 O3 | 同 M3 | 同 M3 | 同 M3 |
| M5：条件 O1 | M0无法确认旧 O1可与新协议合并时，使用同一快照训练四任务 | 新三 Order 都来自同协议；可以独立给均值 | 未完成则保留 Avg. 空缺 |
| M6：汇总 | 读取原始 JSON，写 CSV/LaTeX/状态/来源清单 | 所有表格数字可追溯，不混入 FP32 或 smoke | 修正汇总，不改写原始结果 |

不以“新 Order1 接近59.31”证明协议等价；一旦执行新 Order1，优先报告新快照的完整三 Order 结果，旧59.31作为历史记录保留。C1只要求补全覆盖，并不要求先增加多seed。

## 8. 输出结果：哪些已知，哪些必须等实验

**BF16 Order2、Order3 的准确率目前未知，不能由 FP32 的61.67/64.45换算或预填。** 本计划输出的是应生成的文件和计算规则。

首选五卡补跑的输出目录约定：

```text
experiment_queue/inclora_ewc_bf16_ddp5_<RUN_ID>/
  manifest.json                 # 固定配置、设备/环境、来源及协议复用结论
  source/  source_sha256.json    # 真正执行的源码
  commands/  logs/               # 各阶段实际 argv 和日志
  state.json                    # queued/running/failed/completed + 时间/PID
  smoke/                        # 独立验证产物，不入论文
  full/order_2/{1-dbpedia,2-amazon,3-agnews,4-yahoo}/
  full/order_3/{1-yahoo,2-amazon,3-agnews,4-dbpedia}/
  full/order_1/...               # 仅执行条件项时存在
  final_results.json            # 每 Order 最终四任务分数与总体分数
  results_by_order.csv           # model/method/precision/seed/order/FAA/来源
  comparisons.csv               # 与指定历史方法的百分点差、相对变化
  exp_nlp_bf16_completed.tex     # 更新后的论文表格
```

每个正式阶段保留：`train_results.json`、`predict_results.json`、`adapter/adapter_config.json`、adapter 权重、`adapter/ewclora_fisher.pt`、`verification.json`。

验证器必须检查：训练与预测成功；所有 tensor 有限；累积 rank8/16/24/32；阶段2–4历史 adapter 前缀未改变；Fisher 更新计数1/2/3/4、32个目标模块覆盖、非负有限且trace>0、形状及加载路径正确。EWC首任务为0、下一任务 fresh B=0时初始惩罚为0均正常，须在有参数更新后检查其作用。正式训练从基座开始，不继承 smoke checkpoint。

设新 BF16 Order2/3 的 FAA 为 `x2/x3`：

| 表格单元 | 完成后的填写方式 |
|---|---|
| 1B EWC Order1 | 旧协议可复用时为59.3125；否则使用新 O1 的 `x1` |
| 1B EWC Order2 | `x2`，待运行 |
| 1B EWC Order3 | `x3`，待运行 |
| 1B EWC Avg. | 通过旧协议复用审查时 `(59.3125+x2+x3)/3`；新三 Order 则 `(x1+x2+x3)/3`；未过审查仍为 `--` |
| 1B GAM+EWC Avg. | 历史70.1985，继续复用并标明来源 |
| 平均差值 | `70.1985 - EWC_Avg`，单位百分点；仅在比较范围明确时给出 |
| 相对变化 | `100 × (70.1985 - EWC_Avg) / EWC_Avg`，单位% |

收集器实际应使用最终四任务 JSON 的未二位舍入分数计算 FAA 和均值，再保留两位小数；上表59.3125/70.1985为便于阅读的存档显示值。分任务均值与单独保存的总体指标存在至多末位舍入差异，须记录采用的计算口径。

若旧协议复用通过，主表可以移除此前 FP32 补跑的星号，并填入 BF16 新值；仍保留单 seed1993 的说明。若使用新三 Order 协议，则将对应行标为新 BF16 实验，附录披露与历史 GAM+EWC 的来源差异。不能将该比较称为严格的多seed组件消融。

## 9. 预算与范围控制

- 最少正式任务量：2 Order × 33,000训练样本=66,000样本；额外完整 Fisher 遍历合计66,000样本；每 Order约825次优化器更新，共约1,650次。
- 最终测试共60,800样本；若四阶段都评估所有已见任务，两个 Order共152,000次预测。smoke另计。
- 条件 O1增加4阶段、33,000训练样本、约825次更新、76,000次阶段预测。
- 历史 O1 成功日志有时间戳的区间约21分钟，硬件和依赖来源不完整，只作量级参考；不能据此承诺新设备耗时。当前 FP32 两 Order约10小时也不能直接当 BF16估时。
- 初始调度可按五卡顺序运行两个 Order预留1–2小时，即5–10 GPU小时，另加 smoke 与环境检查；这是待校准的排期假设。若需新 O1再暂留0.5–1小时，即2.5–5 GPU小时。M3首阶段后结合实测训练、Fisher、预测、加载时间更新预算，不缩减样本或改变配置去凑时限。
- 模型/数据已在本地，无需新标注。若迁移服务器，先校验拷贝后的文件哈希；磁盘需保留四阶段 adapter/Fisher 和日志。
- 本轮不做额外seed。后续如论文主张多seed稳定性，应作为独立计划，用相同协议为比较两方增加相同seed集合，不能把三个Order当三个seed。

## 10. 已核对证据

- 旧 O1 实际成功日志：`Logs_EWCLoRA/order1_llama1B_ewclora_seed1993_20260515_235510_tmux.log`；记录五卡、GA8、总batch40、BF16、λ1。
- 旧 O1 结果：`logs_and_outputs_llama_1B/order_1/EWCLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-agnews/predict_results.json`。
- 旧源码候选：git commit `898adf6bb13eb6efc00e31764eb2bcabd014498d`；只是可审查候选，没有当次源码哈希证明。当前普通 EWCLoRA 额外调用 OLoRA-base，因此显式置零是关键。
- 已完成 FP32 补跑：`experiment_queue/inclora_ewc_supplement_20260916T213647Z`；已保存实际执行源码，当前主要训练文件与该快照一致。本计划可据其修复后的环境/存储路径准备新入口，但不能称其为旧成功运行的精确源码。
- 数据哈希与结果清单：`refine-logs/gam_ewc_export_20260917/configuration/datasets.json`、`refine-logs/gam_ewc_export_20260917/results/final_results.csv`。
- 当前补跑入口及说明：`scripts_llama/run_inclora_ewc_supplement.py`、`scripts_llama/INCLORA_EWC_SUPPLEMENT.md`。

## 11. 完成清单

- [ ] 获得兼容 BF16 的设备，五卡方案逐卡检查并锁定。
- [ ] 记录旧 Order1 复用结论；无法确认则登记新 Order1。
- [ ] 实现并 dry-run 新五卡入口，确认 GA8、λ1、OLoRA0。
- [ ] 两任务 BF16/DDP smoke和Fisher传递验证通过。
- [ ] O2、O3 的八个正式阶段完整且验证通过。
- [ ] 若需要，新 O1四阶段完整且验证通过。
- [ ] 生成原始结果、逐Order表、来源/哈希与 LaTeX 表；未将 FP32 数字当 BF16。
- [ ] 准确区分“已完成”“计划运行”和“条件项”，所有新准确率由实际结果产生。
