# T5-large 对应结果核对（2026-09-19）

五个确切方法 × 三个四任务顺序共 15 格，其中 14 格存在真实 T5-large 结果；缺少 IncLoRA Order-1。每个现有格都有四阶段 `train_results.json`、`trainer_state.json`、adapter，以及最终四任务预测文件。原始预测逐行重算（每任务 7,600，总计每顺序 30,400，无 dataset/id 重复）与指标 JSON 的四位小数一致。

| 方法 | Order-1 | Order-2 | Order-3 | 三顺序均值 | 成功训练控制台日志 |
|---|---:|---:|---:|---:|---|
| SeqLoRA | 56.40 | 66.24 | 69.88 | 64.17 | O1 |
| IncLoRA | 缺失 | 77.39 | 70.37 | — | 未找到；训练 JSON 存在 |
| OLoRA | 77.59 | 76.50 | 77.55 | 77.21 | 未找到；训练 JSON 存在 |
| IncLoRA+EWC | 71.57 | 72.57 | 71.46 | 71.87 | O1、O2、O3 |
| IncLoRA-GAM-EWC | 64.71 | 67.14 | 71.60 | 67.82 | O1、O2、O3 |

缺失不是零分，也不能拿 T5-small 补入。以下名字带 T5large 的目录，四阶段 adapter_config.json 的 base_model_name_or_path 全部是 `/data/140-0/users/liying/Sharing/Models/google-t5/t5-small`：

`Logs_Short_Paper/Order-1_8/IncLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_redo1_seed-1993`。其 48.00985 分属于 T5-small，不能作为 T5-large IncLoRA O1。已枚举本仓库非归档 adapter_config.json，没有发现另一份真实 T5-large IncLoRA 四任务 O1。另有长序列 15 任务 Order6 的 OLoRA，任务设置不同，未混入本表。

## 配置证据

| 已有成功运行日志的组 | GPU 数 | 每卡 batch | GA | 有效 batch | seed / data_seed | 精度 | LR（四阶段） | epoch | 最大源/目标/生成长度 |
|---|---:|---:|---:|---:|---|---|---|---:|---|
| SeqLoRA O1 | 4 | 16 | 1 | 64 | 1993 / 1993 | BF16=True | 全部 1e-3 | 1 | 512 / 50 / 50 |
| EWC Ba8 O1–O3 | 8 | 8 | 1 | 64 | 1993 / 1993 | BF16=True | 全部 1e-3 | 1 | 512 / 50 / 50 |
| GAM-EWC Ba8 O1–O3 | 8 | 8 | 1 | 64 | 1993 / 1993 | BF16=True | 全部 1e-3 | 1 | 512 / 50 / 50 |
| 额外 EWC Ba16 O1（单独保留） | 4 | 16 | 1 | 64 | 1993 / 1993 | BF16=True | 全部 1e-3 | 1 | 512 / 50 / 50 |

EWC 与 GAM-EWC 的 λ=1.0、γ=1.0；GAM 使用 optimizer_type=gam-adamw_hf，grad_rho=0.02、grad_norm_rho=0.2、grad_gamma=0.03、β1=β2=β3=1。EWC 和 SeqLoRA 使用 adamw_hf。已核对的两组 EWC/GAM 单一 batch64、seed1993、8×8×GA1 队列在日志层面的通用训练设置对齐。GAM 的三顺序均值 67.82 低于 EWC 的 71.87；这组 T5-large 结果不支持在此配置上加 GAM 提高 FAA 的说法。

SeqLoRA O2/O3、IncLoRA O2/O3 和 OLoRA O1/O2/O3 共七格没有找到匹配的成功训练控制台日志。它们有四阶段完成的训练 JSON、epoch=1、日志学习率 0.001 和真实 T5-large adapter；训练步数是 dbpedia219 / amazon79 / yahoo157 / agnews63，与每任务 ceil(N/64) 一致。这里只能推断名义有效 batch64，不能把它当作已证实的 GPU 数、microbatch、GA、BF16、seed 或 data_seed。目录名 seed1993 是命名证据。基线缺少上述运行证据，因此不能宣称五个方法严格同配置。

所有 T5-large adapter 配置都记录 q/v、r=8、alpha32、dropout0.1；r 字段本身不能证明实际张量累计维数。历史权重文件 hash 和执行源码版本未完整保留；日志的配置对齐不等于执行环境及数值完全相同。

## 完整成功日志与四阶段定位

以下行号按原始文件 LF 分行，与 rg -n 一致；不把进度条的 CR 当作新行。

- IncLoRA-GAM-EWC O1 : `/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/run_logs/run_all_ewc_t5large_ba8_seed-1993_20260518_172349.log`。
  - 1-dbpedia: 命令行 49542；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 62171,62193。
  - 2-amazon: 命令行 62239；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 74210,74238。
  - 3-yahoo: 命令行 74284；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 86341,86363。
  - 4-agnews: 命令行 86418；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 98372,98396。
- IncLoRA+EWC O1 （额外 Ba16 版本）: `/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/run_logs/run_all_ewc_t5large_seed-1993_20260518_144728.log`。
  - 1-dbpedia: 命令行 31；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 6703,6723。
  - 2-amazon: 命令行 6759；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 12885,12903。
  - 3-yahoo: 命令行 12947；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 19237,19255。
  - 4-agnews: 命令行 19302；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 25395,25413。
- IncLoRA+EWC O1 : `/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/run_logs/run_all_ewc_t5large_ba8_seed-1993_20260518_172349.log`。
  - 1-dbpedia: 命令行 31；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 12895,12929。
  - 2-amazon: 命令行 12963；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 25029,25057。
  - 3-yahoo: 命令行 25103；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 37333,37365。
  - 4-agnews: 命令行 37410；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 49443,49469。
- SeqLoRA O1 : `/data/140-1/users/liying/O-LoRA/Logs_RedoOlora/run_logs/seqlora_all_orders_20260327_074758.log`。
  - 1-dbpedia: 命令行 40565；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 51405,51417。
  - 2-amazon: 命令行 51443；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 56584,56597。
  - 3-yahoo: 命令行 56621；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 71580,71594。
  - 4-agnews: 命令行 71607；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 76515,76527。
- IncLoRA-GAM-EWC O2 : `/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/run_logs/run_all_ewc_t5large_ba8_seed-1993_20260518_172349.log`。
  - 1-dbpedia: 命令行 147982；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 160611,160637。
  - 2-amazon: 命令行 160679；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 172650,172678。
  - 3-agnews: 命令行 172724；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 184678,184704。
  - 4-yahoo: 命令行 184755；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 196812,196834。
- IncLoRA+EWC O2 : `/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/run_logs/run_all_ewc_t5large_ba8_seed-1993_20260518_172349.log`。
  - 1-dbpedia: 命令行 98471；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 111335,111359。
  - 2-amazon: 命令行 111403；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 123469,123495。
  - 3-agnews: 命令行 123543；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 135576,135604。
  - 4-yahoo: 命令行 135653；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 147883,147909。
- IncLoRA-GAM-EWC O3 : `/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/run_logs/run_all_ewc_t5large_ba8_seed-1993_20260518_172349.log`。
  - 1-yahoo: 命令行 246422；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 258983,259005。
  - 2-amazon: 命令行 259051；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 271022,271044。
  - 3-agnews: 命令行 271096；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 283050,283076。
  - 4-dbpedia: 命令行 283127；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 295252,295282。
- IncLoRA+EWC O3 : `/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/run_logs/run_all_ewc_t5large_ba8_seed-1993_20260518_172349.log`。
  - 1-yahoo: 命令行 196911；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 209645,209675。
  - 2-amazon: 命令行 209713；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 221779,221815。
  - 3-agnews: 命令行 221853；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 233886,233920。
  - 4-dbpedia: 命令行 233963；与 train_results.json 的 train_runtime 和 train_loss 同时匹配的日志行 246323,246359。

额外 Ba16 EWC O1 的 FAA=73.1019736842（73.10），完整训练记录在 `run_all_ewc_t5large_seed-1993_20260518_144728.log`；该队列后来进入 GAM 后失败，不影响之前已完成 EWC O1 的四阶段证据。本表使用后来统一 Ba8 的 EWC O1=71.5723684211（71.57），没有挑高分替换。

## 评估与来源文件

Seq/Inc/OLoRA 最终结果由四个 `4-任务/evalFlat-数据集-trainedLoRAwhole/predict_results.json` 汇总，对应四份原始 `predict_eval_predictions.jsonl`。相应 eval_Whole_eval_new.log 可见 do_train=False、num_train_epochs=0、global_step=0；这些是训练后单独评估参数，不代表原始模型没有训练。评估日志明确 BF16=True、max_source_length512、max_target_length50、generation_max_length50、每卡 eval batch64；EWC/GAM 在训练内评估使用每卡 eval batch16。原始 JSONL 检查覆盖全部四任务，相同规范化匹配函数（src/compute_metrics.py:32–49）。

所有逐格结果路径、SHA256、任务正确数/样本数、四阶段 JSON、adapter_config、训练命令精确行、参数、评估日志列表和未知项均保存在 [t5large_evidence.json](t5large_evidence.json)。汇总 CSV 仅用于定位，所有数字重新读取原始文件并核算；没有把归档汇总当训练证据。
