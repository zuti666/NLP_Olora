# ImageNet-A 两阶段 EWC／EWC+GAM：现有结果与复用边界

只读核查时间：2026-09-20 16:17:25 UTC。本轮未启动GPU、训练、评估或监控循环，未改动原代码及队列。精确资产路径、状态、SHA、字段及数值见 [existing_results_manifest.json](existing_results_manifest.json)。以下代码task0/task1对应论文第一／第二任务。

## 最新状态已经是complete

`/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0_dualline_recovery1/pipeline_state.json`

该恢复作业于 **2026-09-20 15:36:08.384510 UTC** 更新为 `complete`。sanity、neighborhood、training、bridge、table_a、trajectory、path及report全部complete；本次重新读取15个job receipt，当前文件SHA与pipeline记录全部一致。不能再沿用12:39的failed快照，不能再说端点raw Fisher未完成。

两条训练本身分别于12:39:06.773102（SGD起点）和12:39:05.403562（GAM起点）完成；recovery1的training步骤复用了已有完成产物，没有额外训练新分支。

最新汇总与导出目录：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md`
- 同目录 `A_two_task.csv`、`A_PAIRED_RNG_CHECK.json`、`B_neighborhood_deterministic.csv`、`C_current_FP32_task_trajectory.csv`、`C_path_summary.csv`及`MANIFEST.json`。

## 已有新训练：两个起点，后续都为SGD+EWC

| 已完成run | a11 FP32 | a21 FP32 | a22 trainer | BWT pp | 有效Q/V位移 | raw旧任务F起点 | raw旧任务F终点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_sgd_to_sgd_seed0` |85.667752|84.364821|74.80|−1.302932|0.374949571|1465.496927|2008.387873|
| `A_gam_to_sgd_seed0` |87.622150|84.039088|72.05|−3.583062|0.349875797|761.900774|940.412532|

a11/a21为训练hook记录的当前FP32、全部已见类别上的旧任务accuracy；a22只有trainer保存的两位小数。专门`table_a/summary.json`中的a21亦为两位，不能用该舍入值替代hook的完整值计算高精度BWT。SGD历史继承a11=85.34，当前为85.667752，必须分开保存。

固定旧20类head的test CE分别为 **0.442423705→0.423177732**、**0.413620681→0.400232589**，增量为−0.019245973／−0.013388091；train CE增量为+0.003674375／−0.002156768。相同后续训练规则下，shaped起点的绝对local loss和raw trace较低，但该seed的class-IL BWT及新任务accuracy更差，不能称为全面改善。保护矩阵从各自anchor继承，矩阵本身也可能不同。

端点raw Fisher已按同一协议完成：固定同一256个分层旧任务train样本，确定性Resize256/bicubic→CenterCrop224→ToTensor（无Normalize），固定旧20类head，24块有效Q/V BA、scaling=1；逐样本CE梯度平方平均、无归一化、无floor、无正则，full FP32且关闭matmul/cuDNN TF32。该raw trace与训练保护Fisher、历史归一化trace不是同一统计量。

两条run的根目录：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/training/A_sgd_to_sgd_seed0/`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/training/A_gam_to_sgd_seed0/`

每个目录均有 `COMPLETE.json`、`run_manifest.json`、`rng_receipts.jsonl`、`runtime_receipts.jsonl`、`task0_local_metrics.jsonl`、`checkpoints/task_01.pt`及原始trainer metrics。固定配置为ImageNet-A、ViT-B/16、rank10、seed0、SGD、LR .02、50epochs、batch128、momentum/weight_decay0、cosine、lambda2000、gamma.9；task和accumulated保护Fisher均归一化、batch_empirical、最多100batch；`lambda_flat=0`。当前精度和配对RNG receipts齐全。

配置文件：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_sgd_to_sgd_seed0.json`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_gam_to_sgd_seed0.json`

当前端点评估文件：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0_recovery1/sgd/table_a/summary.json`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0_recovery1/gam/table_a/summary.json`

## 已有历史SGD→SGD与GAM→GAM：T10的两任务切片

两个Task0 anchor和每条Task1–Task9共20个完整checkpoint全部存在。recovery1重新评价了这20个模型的旧任务train/test local及seen-head指标；这属于**历史训练模型的当前FP32评估**，不是重新执行历史训练。

| 历史训练分支 | 历史记录a11/a21/a22 | 当前FP32 a11 | 当前FP32 a21 | 当前FP32 BWT pp | 当前FP32 test local CE增量t0→t1 | 有效位移t0→t1 |
|---|---|---:|---:|---:|---:|---:|
| SGD→SGD+EWC |85.34/82.08/75.20|85.667752|82.084691|−3.583062|−0.006875388|0.367870098|
| GAM→GAM+EWC |87.62/81.76/81.50|87.622150|81.758958|−5.863192|−0.015379115|0.503778365|

**a22=75.20/81.50仅为历史trainer值**；此次trajectory只评价旧Task0样本，没有重新评价第二任务样本，不能把它标成当前FP32重评a22。

历史Task1 JSON的 `task0_fisher_current_trace=1.0`、`task0_fisher_normalized=1`：这是batch128的归一化旧任务Fisher代理，不能填入上表新raw逐样本Fisher列。recovery1的trajectory不计算每个Task1端点raw Fisher；现存两份table_a的终点是**新SGD continuation**，不能移用到历史GAM→GAM Task1。当前t9路径的两个端点则已有raw Fisher。

较长序列的当前FP32结果也已完成：t0→t9的有效位移为1.197584100／1.400472589；旧任务test local CE增量为0.260130529／0.034347380，旧任务seen-head accuracy为63.843648／72.964169。t9直线路径单位位移敏感度为0.368057602／0.014793093。均是seed0和历史不同后续优化器路径；不能直接替代“固定后续SGD”的新对照。

精确模型目录：

- 普通anchor：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/sgd/seed0/checkpoints/task_00.pt`
- GAM anchor：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/gam/seed0/checkpoints/task_00.pt`
- 普通历史Task1：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/sgd/T10/seed0/checkpoints/task_01.pt`
- GAM历史Task1：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/gam/T10/seed0/checkpoints/task_01.pt`

历史Task1–Task9按同目录`task_01.pt`至`task_09.pt`命名；旧metrics绝对路径较长，完整保存在本目录JSON的`historical_sequences.{sgd,gam}.metrics.path`。原始日志分别为：

- `/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell/sgd_T10_seed0.log`
- `/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell/gam_T10_seed0.log`

当前重评与路径输出位于：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0_recovery1/sgd/trajectory/summary.json`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0_recovery1/gam/trajectory/summary.json`
- 同方法目录的`path/summary.json`为t0→t9直线路径结果。

## 代码、配置与尚缺的匹配续训

历史训练入口及基础配置：

- `/data/140-1/users/liying/CopyFlatCV/scripts/run_s1c_stage1_task0_shared.sh`
- `/data/140-1/users/liying/CopyFlatCV/scripts/run_s1c_stage2_T_fullseq_from_task0.sh`
- `/data/140-1/users/liying/CopyFlatCV/config_Expand/sgd_gam_s1c/base_s1c_normaccum_e50_initlr004_lr002.yaml`

这组旧配置为rank10、lambda2000、gamma.9、50epochs，Task0 LR .04、之后.02，GAM rho .05、norm-rho .2、gamma .1；EWC不做in-task Fisher、projection或dual调lambda。历史源文件不等同于精确历史runtime快照。

新续训和测量入口：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/endpoint_a.py`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic.py`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic_config_recovery1.json`
- 冻结源码及身份：同execution目录`source/`、`source_frozen.json`。

**现有continuation runner硬编码未来为SGD**：`build_config`中continuation标签为`A_{method}_to_sgd_seed{seed}`，实际`method='sgd'`；`--method gam`只选择GAM anchor，不会训练GAM continuation。本次检查execution/training及全部refine-logs的COMPLETE文件，只有上述两条新seed0续训。没有发现按新FP32＋配对RNG规范完成的GAM→GAM或SGD→GAM续训。sequence_gam_seed1/2虽有配置文件，但没有相应完成run，不能算现有结果。

因此，新计划可复用两个Task0 anchor、已完成SGD continuation及完整诊断；若要严格比较“EWC vs EWC+GAM”两阶段方法，仍缺与新SGD分支配对的GAM continuation。历史GAM→GAM可作为先验结果和资产，不应与新SGD run冒充同一训练精度／随机规范的匹配pair。历史精度差异已有TF32操作数舍入复现证据，但没有旧runtime开关直接记录；报告应标注这一限制。

参数移动统一使用共享有效权重差`||BA_task1−BA_task0||_F`；此持续adapter在Task1起点的BA不为0。历史`task0_trajectory.delta_norm`本来是L2范数，新机制`mechanism.steps.1.delta_norm`是平方范数，不能只凭同名字段混拼。当前新旧端点重评中的有效位移与既有CPU逐tensor核验一致。
