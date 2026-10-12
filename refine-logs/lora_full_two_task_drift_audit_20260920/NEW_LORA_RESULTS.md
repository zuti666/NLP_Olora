# 较新 LoRA 结果、原始数据与位移资产

检查时间：2026-09-20 12:41:55 UTC（已更新今天两条 Task1 训练完成状态）。此次只读检查已有记录、重新聚合已有 JSON，并由主代理在 CPU 上重算 ImageNet-R 的六组 checkpoint 配对；未启动 GPU、训练或修改任何实验队列。队列状态是检查时快照，后续可能变化。

## 结论

**有比旧 May-2026 Adam 两阶段表更新的完整 LoRA 结果，而且存在能重新计算有效参数位移的 checkpoint。** 最直接的两组是：

1. ImageNet-A 的两条完整 seed0、10-task SGD/GAM 序列：有 Task0–Task9 全部 20 个完整 checkpoint，已有 CPU 检查确认 `||BA_t−BA_0||_F` 与日志一致。
2. ImageNet-R 的 Aug-2026 ratio-FS 两阶段实验：六个主要变体各有 seeds3/4/5；同未来 native SGD+EWC 的 `control` 与 `factored_s1` 六对模型，本轮实际 CPU 重算通过。其 `delta_norm` 存的是平方范数，开平方才是半径。
3. **今天 12:39 UTC，新 ImageNet-A 的 SGD→SGD 和 GAM→SGD 两条单 Task1 续训已完成**，下面专节给出新的 accuracy 与训练内旧任务 loss 记录。它们的端点 raw Fisher 评估尚未执行，不能把“训练完成”写成全部 A/B/C 实验完成。

**新实现与旧表的关键区别是“持续更新同一个 adapter”与“新任务零 B 的新 adapter”。** 新组 Task1 起点已训练过，BA 不为零，应算 BA1−BA0；直接算 BA1 或把 task checkpoint 的 BA 全部累加会错。不能把旧表 fresh-LoRA 的 `||BA_new||` 公式直接套过来。

## ImageNet-R：新的、最接近六行结构的实验

源仓库 `/data/140-1/users/liying/CopyFlatCV`，tag 为 `ratiofs_l12c20_full_twostage_s345_v1`，记录日期为 2026-08-29/30。设定 ImageNet-R、20+20 classes、ViT-B/16、rank10、seeds3/4/5、SGD lr=.02、50 epochs/task、batch128、momentum=0、weight decay=0、cosine。使用 EWC 的分支 lambda=1200000、gamma=1；单任务 Fisher trace 归一化、累计 Fisher不归一化、batch-gradient Fisher、damping beta=.2；ratio-FS c=.20、rho=.05。其源模型是 `ewclora_normfisher_fishershape`，不是旧表的 `ewclora_gam_projected`。

下表为原始 `cnn.matrices.final` 及 `mechanism.steps.1.delta_norm` 三 seed 均值；r 先每 seed 开平方再平均。它是较新已有结果，不是本轮新训练。

| 变体（setting） | a11 | a21 | a22 | BWT | 有效位移 r |
|---|---:|---:|---:|---:|---:|
| Vanilla（baseline） | 89.36 | 84.59 | 88.89 | -4.77 | 0.850642 |
| ratio-FS 两阶段，无 EWC（fs_only） | 89.47 | 84.80 | 89.54 | -4.66 | 0.856600 |
| EWC（control） | 89.36 | 85.90 | 87.82 | -3.46 | 0.427166 |
| ratio-FS 两阶段＋EWC（fs_ewc） | 89.47 | 85.74 | 88.19 | -3.72 | 0.461659 |
| 仅 Task1 ratio-FS＋Task2 EWC（factored_s1） | 89.47 | 85.96 | 87.82 | -3.51 | 0.448335 |
| 仅 Task2 ratio-FS＋EWC（factored_s2） | 89.36 | 85.59 | 88.38 | -3.78 | 0.456989 |

这组没有旧表中 1.56→.0106 的百倍跨度：control 位移约为无保护 baseline 的一半。它的优化器、学习率、保护矩阵归一化、LoRA 更新规则、shaping 算法和 seed 全部不同；不能仅归因于“换了数据”。这组新的、同未来 SGD+EWC 两行，也没有显示初始 shaping 的平均遗忘改善：BWT 从 -3.4633 变为 -3.5100，a22 平均相同。应保留这个结果。

原始路径：

- 主结果目录：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage/`
- 结果汇总：`/data/140-1/users/liying/CopyFlatCV/result_summary/verify_imagenet_r_1x1_lora_ratio_full_twostage/ratiofs_l12c20_full_twostage_s345_v1/`
- setting 定义：上述汇总目录 `setting_metadata.csv`。
- 原始日志：`/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_lora_ratio_full_twostage/ratiofs_l12c20_full_twostage_s345_v1/stage2/{setting}_seed{3,4,5}.log`。
- JSON、anchor 和 endpoint 精确配对、字段 pointer、SHA-256：本目录 `new_lora_sources.json`（18个正式run）。
- 仅关键控制对的6个资产路径/metadata：`new_r_ratio_pair_assets.json`。
- 本轮 CPU 重算：`new_r_lora_drift_recomputed.csv`、`new_r_lora_drift_summary.csv`、`new_r_lora_drift_per_layer.csv`、`new_r_lora_drift_provenance.json`。

18个正式run的两端 A/B/FC/LoRA metadata 文件都真实存在；没有把缺少 `final_model.pt` 判成缺失，因为 Task1 的保存协议是 sidecar。本轮独立张量重算限定于最有解释力的 `control/factored_s1` 六对，另外12个run此处仅重新聚合原始日志并核验资产存在。

### 六对 checkpoint 的实际重算

两个端点都是同一24个 Q/V adapter 状态，应计算 `sqrt(sum_l ||B1_l A1_l − B0_l A0_l||²)`。没有新的 alpha/r 缩放；匹配的 metadata 为24个 Q/V层块且 task_id=0。

| 配对 | 正确位移，mean±sample SD | Task0 的 ||BA0|| 平均 | Task1 的 ||BA1|| 平均 |
|---|---:|---:|---:|
| control，seeds3/4/5 | 0.427165612 ± 0.001369011 | 0.782015272 | 0.991293792 |
| factored_s1，seeds3/4/5 | 0.448335488 ± 0.001936117 | 0.801424459 | 1.016203586 |

最大“重算平方位移 vs 原 `delta_norm`”绝对误差为 7.58e-9；六对旧任务 head 前20行权重/bias均逐位不动。每条 run 的 Task0 A/B/FC 文件 SHA-256 与训练日志保存的 parent-artifact 身份一致。**这批数据的位移计算本身通过；若把 BA1 当位移，会把约 .427 错写成约 .991。**

## ImageNet-A：完整真实序列与两任务切片

使用此前已独立 CPU 验证的 `Ttraj_fisher_lam2000_lr0p02_e50`，日志日期2026-06-22。配置为 ImageNet-A、10×20、rank10、seed0、SGD base、Task0 lr=.04/以后=.02、50 epochs/task、batch128、EWC lambda2000、gamma=.9、Task/累计Fisher均归一化。GAM整条序列都开启，不是“只 Task0 shaping”。两方法所用 Task0 固定分类头在后续 checkpoint 中逐位不变，非LoRA backbone 也不变。

| 方法 | 时点 | 相对 Task0 的有效位移 | 旧任务 test local loss增量 | 旧任务 test local acc | 旧任务 test full acc |
|---|---|---:|---:|---:|---:|
| SGD→SGD | Task1 | .367870096 | -.006824291 | 85.0163 | 82.0847 |
| GAM→GAM | Task1 | .503778361 | -.015310663 | 86.3192 | 81.7590 |
| SGD→SGD | Task9 | 1.197584100 | .260342520 | 80.1303 | 64.1694 |
| GAM→GAM | Task9 | 1.400472591 | .034413266 | 85.3420 | 72.3127 |

这里的有效位移均是24个 Q/V的 `||BA_t−BA_0||`。此前20个 checkpoint 全部重算最大误差1.84e-8，本轮重新核查文件均仍存在。该专用 Task0 trajectory 的 `delta_norm` 已是 L2 范数；不要因为 ImageNet-R 机制字段也叫 `delta_norm` 就再对它开平方。

两任务原生 class-IL 切片：SGD→SGD 为 a11/a21/a22=85.34/82.08/75.20；GAM→GAM 为87.62/81.76/81.50。固定旧头 loss 与 seen-class accuracy 不同，短期 local loss可改善而global accuracy下降，不能混用。

原始及复算路径：

- Task0：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/{sgd,gam}/seed0/checkpoints/task_00.pt`
- Task1–9：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/{sgd,gam}/T10/seed0/checkpoints/task_0{1..9}.pt`
- 精确 JSON 及路径索引：`/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_assets_manifest.json`
- 全20条重新计算结果：同目录 `task_trajectory_recomputed.csv`。
- 本轮存在性重新检查和小文件hash：本目录 `new_lora_sources.json` 的 `imageneta`。

## fixedfs / FQA 其他目录的边界

`verify_imagenet_r_1x1_lora_fixedfs/fixedfs_exploratory_s012_a005_lam4e5_rho005_v1` 的 seed0四行有完整结果：baseline90.25/83.49/89.96，control90.25/85.69/88.70，fs_only90.09/83.81/89.82，fs_ewc90.09/84.75/88.98；开根后的 r 分别 .870634/.566050/.889650/.624814。它不是三seed完整六行替代表，不能把尚未完成的 temporal 分支拼进来。

`verify_imagenet_r_1x1_lora_fixedfs_upper_bounds` 是 task-only/joint40上界，不是两任务遗忘分支。旧FQA目录的扰动曲线也不是新的训练终点；应单独复用其 anchor、半径方向和损伤数据，不能用 FQA 最后一个点当 Task2。已分析的 ImageNet-A r 曲线在 `fs_ewc_rq2_20260920/radius_damage_seed0.csv`、`radius_tolerance_seed0.csv`，原JSON路径见该目录 `radius_reanalysis_manifest.json`。

## 2026-09-20 真正新产出的状态（只读快照）

最新复核时 `execution/campaign_seed0_dualline_gpu03/pipeline_state.json` 的 `updated_utc=2026-09-20T12:39:07.574565+00:00`，总体为 `failed`。两个sanity及两条补充训练均已完成；neighborhood_sgd失败、另一方法邻域取消；后续 A endpoint/trajectory/path 被阻止启动。**新 A 训练结果已可引用为单 seed 初步结果，但 A endpoint 的 raw Fisher、完整 B 邻域、C 路径没有完成。** 此处未修改进程、源码或队列。

失败原因是 `diagnostic.py:327` 调用 `Runner.measure()` 同时用位置与关键字传入 `direction`，抛出 `TypeError: got multiple values for argument 'direction'`；失败日志为 `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0_dualline_gpu03/neighborhood_sgd.attempt1.log`。

两个sanity确实留下新的 **r=0** 测量：

| Task0模型 | raw逐样本Fisher trace | 固定256样本 loss | 全307 test loss（FP32） |
|---|---:|---:|---:|
| SGD | 1465.496927 | .105774226 | .442423705 |
| GAM | 761.900774 | .065358637 | .413620681 |

共同数据manifest hash为 `be9cf89b2bfbd5a9bc207d9755deb2d3a9bcbd6190a24b3a9c05ead97bfddecb`；确定性输入，256个同样本，旧20类head，24个有效 Q/V，未归一化、不加floor的逐样本梯度平方。此处只说明初始点检查结果，不代表邻域半径曲线已经完成，也不代表 old-task路径Fisher下降已验证。

精确文件：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0/sgd/anchor_measurement.json`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0/gam/anchor_measurement.json`
- 两方法对应目录下 `sanity/summary.json`：均 `passed=true`。
- 当前pipeline：`/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0_dualline_gpu03/pipeline_state.json`。

`execution/report_seed0/RESULTS.md` 仍是10:09的旧pending快照，优先以上述12:39的实际作业与receipt状态为准。


## 今天已完成的新 A 两任务训练（SGD→SGD 与 GAM→SGD）

两份 `COMPLETE.json` 分别在 2026-09-20 12:39:06.773102 UTC（SGD）及12:39:05.403562 UTC（GAM起点）标记 `status=complete, completed_tasks=2`。本轮只读取其结果，没有重新运行训练。

同为 ImageNet-A、seed0，恢复各自旧Task0 anchor 后仅训练 Task1，**未来均为 native SGD+EWC，lambda_flat=0**。rank10、lambda2000、gamma.9、lr.02、50epochs、batch128、wd0、momentum0、cosine；Task1训练997样本。两边完全相同的冻结源代码manifest、runner SHA和明确FP32协议。新 head 初始化hash相同；51个shuffle loader pass（50个epoch加后续Fisher遍历）的样本顺序与首batch像素/标签hash逐条一致。该核验并未对每一个增强batch的像素都存hash，不能扩大成未记录的逐像素证明。

原始训练矩阵的 Task0 第一行继承历史数值，SGD=85.34、GAM=87.62。**当前 FP32 起点评估为85.667752与87.622150**；下表用后者计算BWT，避免把 TF32历史起点和 FP32新终点混算。

| Task0起点→Task1 | a11 当前FP32 | a21 | a22 | BWT | 有效位移（CPU复算） | 旧任务test local CE增量 | 旧任务test seen-head CE增量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SGD→SGD+EWC | 85.6678 | 84.3648 | 74.80 | -1.3029 | .374949571 | -.019245973 | .156537475 |
| GAM→SGD+EWC | 87.6221 | 84.0391 | 72.05 | -3.5831 | .349875797 | -.013388091 | .119736275 |

这组新结果没有复现“只改变 Task0 起点就全面改善保留”的简单说法。GAM起点具有更低的绝对 test local CE（.413621→.400233；SGD为.442424→.423178），但相对自身起点，class-IL accuracy drop 更大，Task1 accuracy 更低；两边 local CE都改善，SGD的改善幅度反而更大。seen-head CE增量则GAM较小，表明不同损伤口径给出不同排序，必须同时报告。新实验不能按绝对终点 loss 低就宣布机制主张已证实。

训练内位移采用有效Q/V差分；相应原 `mechanism.steps.1.delta_norm` 平方值为 .140587178848 与 .122413070874。完整 Task1 checkpoint 与对应原Task0已完成CPU逐tensor重算：FP64有效位移分别为 .37494957053696587、.34987579688281456，与训练内平方记录的误差分别仅1.47e-9、7.37e-10；旧分类头逐比特相同，所有非LoRA backbone参数不变。完整结果见本目录 `today_imageneta_pair_recomputed.csv`，配对路径和文件SHA见 `today_imageneta_pair_provenance.json`。这里没有新增端点逐样本 Fisher；`task0_local_metrics.jsonl` 明确 `trace_measured=false`。

完整路径：

- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/training/A_sgd_to_sgd_seed0/`
- `/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/training/A_gam_to_sgd_seed0/`

每个目录含 `COMPLETE.json`、`run_manifest.json`、`task0_local_metrics.jsonl`、`rng_receipts.jsonl`、`runtime_receipts.jsonl`、`checkpoints/task_01.pt`。原生accuracy JSON在各自 `trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/A_{method}_to_sgd_seed0/imageneta/0/A_{method}_to_sgd_seed0/exp_run/20/A_{method}_to_sgd_seed0_vit_base_patch16_224_cl_metrics.json`。

两个起点分别为前述 `CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/{sgd,gam}/seed0/checkpoints/task_00.pt`，由各自 `run_manifest.json` 的 `config.resume_from_checkpoint_dir` 和 `runtime_receipts.jsonl` 的 `complete_resume` hash 双重定位，不能交叉配错。

本节独立来源与复算：本目录 `today_a_sources.json`（精确路径、hash、字段pointer、RNG检查）、`today_a_completed_results.csv`、`audit_today_a_sources.py`。
