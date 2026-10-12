# FS-EWC RQ2 文件路径索引

**2026-09-21 20:55马德里完整核验：七行ImageNet-A两任务表已21/21个seed单元齐全；本次补训队列20:49:58完成10/10（6条正式续训、2短测、1验证、1汇总），失败/运行/待执行均0。本批控制器与训练进程已退出。见[完整七行结果与结论](execution/full_rq2_20260920/seven_row_extension_20260921/complete_review/FINAL_RESULTS_AND_CONCLUSIONS.md)、[LaTeX完整表](execution/full_rq2_20260920/seven_row_extension_20260921/complete_review/table_imageneta_complete.tex)、[逐seed结果](execution/full_rq2_20260920/seven_row_extension_20260921/final_report/individual.csv)。此前15/21、排队和4条运行描述均为历史快照。完整结果支持持续GAM的平均性能收益；当前λ=2000下FS相对Shaping only的平均差值−0.0017±0.1975个百分点，不支持稳定额外EWC精度收益。**

**2026-09-21 19:03马德里更新：按用户要求，补训队列的空闲GPU周期检查已改为每15分钟（900秒）；新调度器PID 1737208。六条正式续训及短测门控保持原配置。此前15秒检查与旧PID为历史记录。**

**2026-09-21 18:50马德里：七行表补训已加入后台队列（PID 1649105）。GAM→SGD＝C+F→C已放入主表倒数第三行；当前15/21个seed单元有完成结果。补训GAM→GAM（无EWC）与SGD→GAM＋EWC各3seed，共6段50轮Task1续训；复用已有起点及GAM→GAM＋EWC结果。启动时GPU0–6均有任务占用，等待空闲；两条隔离1轮短测和验证通过后才放行正式续训。六份CPU完整恢复、数据/评估配置和独立队列dry-run已通过；GPU短测尚不能计为完成。见[七行表](execution/full_rq2_20260920/six_row_imageneta_20260921/TABLE_IMAGENETA.md)、[后台实时状态](execution/full_rq2_20260920/seven_row_extension_20260921/campaign/STATUS.md)、[执行与验证协议](execution/full_rq2_20260920/seven_row_extension_20260921/RUN.md)。下方六行表“尚未启动”为此前历史记录。**

**2026-09-21 六行表扩展核验：用户指定的ImageNet-A六行中4行已有正式三seed结果；GAM→GAM＋EWC可复用已完成十任务链的task01。另缺GAM→GAM无EWC、SGD→GAM＋EWC，各3seed，共6段第二任务续训，尚未启动。见 [六行表与结果](execution/full_rq2_20260920/six_row_imageneta_20260921/TABLE_IMAGENETA.md)、[LaTeX](execution/full_rq2_20260920/six_row_imageneta_20260921/table_imageneta.tex)、[最小补实验清单](execution/full_rq2_20260920/six_row_imageneta_20260921/MISSING_EXPERIMENTS.md)。这是六行表的新目标覆盖情况，与下方已完成88/88的原B1/B2/B3核心范围不同。**

**2026-09-21 完整核验：RQ2于11:57马德里完成88/88作业，三seed四格12/12、B1/B2诊断54/54、路径数值检查18/18通过。S3另有18/18正式实验、20/20短测完成。完整结果、Fisher/位移、测试集反例及结论见 [完整结果与结论](execution/full_rq2_20260920/complete_review_20260921/FINAL_RESULTS_AND_CONCLUSIONS.md)、[三seed四格表](execution/full_rq2_20260920/final_report/RESULTS.md)、[完整机制统计](execution/full_rq2_20260920/complete_review_20260921/RQ2_MECHANISM_FINAL.md)。核心完成不包含尚未执行的条件补测。以下11:10、seed0等进展和旧缺口均为历史记录。**

2026-09-21 11:10马德里历史快照：当时RQ2为86/88、机制53/54，见 [当时进展](execution/full_rq2_20260920/monitor_20260921_1107/CURRENT_STATUS.md) 和 [当时机制结果](execution/full_rq2_20260920/MECHANISM_STATUS_20260921.md)；现已由上述完整结果替代。

**最新执行范围：用户已授权“使用空闲GPU进行运行”。本轮按附件完整核心 B1/B2/B3 执行，含三seed四格表、Fisher及参数移动；88作业正式计划和旧seed0配对补全独立保存。见 [完整执行协议](execution/full_rq2_20260920/EXECUTION.md)、[正式实时状态](execution/full_rq2_20260920/formal_queue/STATUS.md)、[旧seed0补全状态](execution/full_rq2_20260920/pilot_queue_paired/STATUS.md)。下文“尚未入队”“四格不默认执行”等文字是此前范围的历史记录，不代表本次授权。**

**最新研究组织（2026-09-20）：用户已选择当前视觉GAM+EWC为目标，A沿用已完成结果；下一轮按A表+B邻域图+C真实漂移图收口。见 [机制接续计划](mechanism_next_20260920/EXPERIMENT_PLAN.md)、[主张评估](mechanism_next_20260920/CLAIMS_FROM_RESULTS.md)、[方法对应](mechanism_next_20260920/METHOD_ALIGNMENT.md)、[新版图表](mechanism_next_20260920/figures/)。此前08:25版“四格必补”的缺口描述属于旧范围，不再作为本轮默认要求；新增GPU诊断尚未入队。**

最新核验：2026-09-20 17:36:08马德里时间，恢复队列全部15/15作业完成并通过验证，最终报告已生成：[RESULTS.md](execution/report_seed0/RESULTS.md)。设置、实际执行、历史一致性及结果解释见 [ANALYSIS.md](execution/analysis_setting_20260920/ANALYSIS.md)，复核台账见 [VALIDITY_LEDGER.json](execution/analysis_setting_20260920/VALIDITY_LEDGER.json)。目前只有seed0，不代表多seed确认。17:30的 [CURRENT_STATUS.md](execution/monitor_20260920/CURRENT_STATUS.md) 及下文运行中描述均为历史快照。

与用户08:25:49版完整计划对照：15/15仅覆盖精简seed0队列；完整B3四格当前seed0为2/4、正式三seed为2/12，仍缺无保护分支及正式B1/B2扩展。新版表格与缺口说明见 [TABLE_REVISION.md](execution/table_revision_20260920/TABLE_REVISION.md)：含已有Fisher/位移的[当前两行LaTeX](execution/table_revision_20260920/imageneta_current_complete.tex)、[完整计划差距](execution/table_revision_20260920/FULL_PLAN_GAP_AUDIT.md)，以及在low-rank-cl-new中重新溯源的[旧ImageNet-R六行修订版](execution/table_revision_20260920/imagenetr_legacy_mean_sd.tex)。

2026-09-20：实验已实际执行。原双线 pipeline 于 12:32:42 UTC 启动，两条 sanity 通过，两条 Task1 续训各完成50 epoch；12:33:21 UTC 邻域作业遇到参数重名错误，队列在两条训练结束后于12:39:07 UTC进入 failed。旧运行记录保留于 [campaign_seed0_dualline_gpu03/STATUS.md](execution/campaign_seed0_dualline_gpu03/STATUS.md)。

已修复 `Runner.measure` 的 `direction` 参数与同名元数据冲突，恢复队列PID为 `1341023`，pipeline于15:26:06 UTC启动。15:27:22 UTC核验快照：新版两条sanity及两条A端点评估已完成，两条训练已确认跳过重训；01线邻域、23线trajectory正在运行。当前恢复目录为 `campaign_seed0_dualline_recovery1/`，最新状态以 [QUEUE_STATUS.md](execution/campaign_seed0_dualline_recovery1/QUEUE_STATUS.md) 和 [STATUS.md](execution/campaign_seed0_dualline_recovery1/STATUS.md) 为准；恢复说明见 [RECOVERY.md](execution/campaign_seed0_dualline_recovery1/RECOVERY.md)。剩余诊断和最终报告尚不能计为完成。

2026-09-20：按用户要求调整执行顺序。等待 GPU0–3 的整个 TreeLoRA 队列完成，并确认四张卡连续空闲60秒后，先在 GPU0/1 完成共同 sanity；随后 GPU0/1 执行 neighborhood→bridge，GPU2/3 执行 Task1 training→A endpoint→trajectory→path，最后统一汇总。GPU0/2 对应 SGD，GPU1/3 对应 GAM；保留8个阶段、15个作业及全部科学设置，阶段按依赖并行。排队不表示实验完成。

## 实验安排与配置

| 用途 | 绝对路径 |
|---|---|
| 完整实验计划 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md) |
| 任务清单与当前接续安排 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md) |
| 执行顺序、GPU分配与接续条件 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md) |
| 真实数据依赖、两线排程与并发写核查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/DEPENDENCIES.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/DEPENDENCIES.md) |
| 15作业、8阶段的当前接续配置 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json) |
| 当前 B/C 评估配置、旧checkpoint绝对路径 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic_config_recovery1.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic_config_recovery1.json) |
| A两条Task1补训练协议 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/TRAINING_PROTOCOL.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/TRAINING_PROTOCOL.md) |
| 普通Task0 → SGD+EWC配置 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_sgd_to_sgd_seed0.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_sgd_to_sgd_seed0.json) |
| shaped Task0 → SGD+EWC配置 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_gam_to_sgd_seed0.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_gam_to_sgd_seed0.json) |

## 此前已完成的分析

| 用途 | 绝对路径 |
|---|---|
| 跨r代码、数据口径与证据审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_RADIUS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_RADIUS.md) |
| 跨T轨迹、checkpoint与恢复逻辑审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TASK_ASSETS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TASK_ASSETS.md) |
| 两任务对照、已有结果与缺失分支审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TWO_TASK.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TWO_TASK.md) |
| 旧邻域损伤重分析，324行 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_damage_seed0.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_damage_seed0.csv) |
| 经验容忍半径，40行 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_tolerance_seed0.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_tolerance_seed0.csv) |
| 旧逐任务结果与CPU位移核验，20行 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_trajectory_recomputed.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_trajectory_recomputed.csv) |
| 原始训练配置、指标与checkpoint来源 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_assets_manifest.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_assets_manifest.json) |
| 旧半径结果来源与重分析参数 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_reanalysis_manifest.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_reanalysis_manifest.json) |
| 相关源码/脚本来源 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/source_audit_manifest.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/source_audit_manifest.json) |

## 执行与排查记录

接续入口仍为 `execution/wait_then_run.py --plan execution/campaign_plan.json --execute`；当前实时接续状态写入 `execution/campaign_seed0_dualline_recovery1/queue_state.json` 与 `QUEUE_STATUS.md`，pipeline 状态写入同目录 `pipeline_state.json` 与 `STATUS.md`。前驱状态为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`，恢复仍保留整个前驱完成和四卡连续空闲60秒的 gate。

新诊断输出为 `execution/results_seed0_recovery1/`；原 `results_seed0/` 中通过的旧版 sanity 和 r0 保留不改。诊断脚本哈希变化后重跑 sanity，不静默复用旧协议结果。两条既有训练完成情况、指标与配对摘要见 [training_readonly_summary.json](execution/monitor_20260920/training_readonly_summary.json)。旧报告框架已归档至 `execution/campaign_seed0_dualline_recovery1/before_fix/report_seed0/`，最终报告仍将写入 `execution/report_seed0/`。

旧等待进程 PID `541721` 已在未启动 GPU 实验时安全停止，TreeLoRA 前驱未受影响。改双线前的计划与文档保存在 `execution/campaign_seed0_after_gpu03/before_dualline/`，替换凭据为 [SUPERSEDED_FOR_DUAL_LINE.json](execution/campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json)。

旧 GPU4/6 队列已于此前按用户要求停止；本次是新的接续授权。更新前的计划与文档保存在 `execution/campaign_seed0/before_gpu03_requeue/`，原停止凭据继续保留。

| 用途 | 绝对路径 |
|---|---|
| 历史 GPU4/6 队列停止凭据，保留溯源 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json) |
| 历史TF32精度差异独立审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md) |
| 已完成的seed0最终汇总 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md) |
| 本次设置、执行与结果的综合分析 | [ANALYSIS.md](execution/analysis_setting_20260920/ANALYSIS.md) |
| 实际优化器、损失、恢复与随机配对审计 | [AUDIT_EXECUTION.md](execution/analysis_setting_20260920/AUDIT_EXECUTION.md) |
| 原始历史日志与当前实验对照 | [HISTORICAL_COMPARISON.md](execution/analysis_setting_20260920/HISTORICAL_COMPARISON.md) |
| A/B/C完整测量、反例及59项校验 | [MEASUREMENT_ANALYSIS.md](execution/analysis_setting_20260920/MEASUREMENT_ANALYSIS.md) |
| 已完成两条训练的只读指标与配对摘要 | [training_readonly_summary.json](execution/monitor_20260920/training_readonly_summary.json) |
| 本次故障、修复与恢复依据 | [RECOVERY.md](execution/campaign_seed0_dualline_recovery1/RECOVERY.md) |
| 首次sanity失败记录说明 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md) |

## 2026-09-21 历史逐阶段路径补测（GPU 0、3）

已完成 16 项 Task1–Task8 补测，复用 2 项 Task9，共 18 条路径、162 个积分节点。沿用原 C 图的历史 ImageNet-A seed0、固定 256 个旧任务训练样本；不重训练。

- [新 C 图 PDF](mechanism_next_20260920/stage_path_extension_20260921/Figure_C_sequence_stage_paths.pdf) / [PNG](mechanism_next_20260920/stage_path_extension_20260921/Figure_C_sequence_stage_paths.png)
- [逐阶段数据](mechanism_next_20260920/stage_path_extension_20260921/report/C_stage_path.csv) / [逐节点数据](mechanism_next_20260920/stage_path_extension_20260921/report/path_nodes.csv)
- [中文解读与口径](mechanism_next_20260920/stage_path_extension_20260921/report/SUMMARY_CN.md)
- [配置与执行记录](mechanism_next_20260920/stage_path_extension_20260921/RUN.md) / [数值检查](mechanism_next_20260920/stage_path_extension_20260921/report/completeness.json) / [图稿复核](mechanism_next_20260920/stage_path_extension_20260921/FIGURE_REVIEW.md)

## 2026-09-21 21:42 Madrid：raw Fisher λ扫描已加入等待队列

最终用户确认：seed0，Vanilla LoRA vs EWCLoRA，仅SGD→SGD两个任务；复用已完成Task0完整权重，实际重算native未归一化fit Fisher，关闭单任务和累计trace归一化。Task0不重训。14个λ均重新训练Task1的50轮。

[实验计划](execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/EXPERIMENT_PLAN.md) · [运行说明](execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/RUN.md) · [实时队列](execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/campaign/STATUS.md) · [结果表](execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/report/RESULTS.md)。

启动核验：控制器2588484存活；18/18作业queued，waiting_for_all_gpus。GPU1/4/6空闲，0/2/3/5繁忙；全部GPU0–6空闲才放行，每900秒检查。此次检查21:42:20，下次21:57:20。CPU数据/完整模型恢复、13项runner回归、6项report回归和队列门槛测试通过；原生GPU Fisher、短测及正式结果尚未产生。旧归一化扫描草案未执行，已归档before_raw_fisher/；原三seed七行表不变。此处为启动快照，后续以实时队列为准。

## 2026-09-21追加：有效λ选择与四行EWC复跑

[完整后续计划](execution/full_rq2_20260920/ewc_selected_lambda_followup_20260921/EXPERIMENT_PLAN.md) · [选择规则](execution/full_rq2_20260920/ewc_selected_lambda_followup_20260921/selection_policy.json) · [待执行清单](execution/full_rq2_20260920/ewc_selected_lambda_followup_20260921/followup_plan.json)。

等待当前14点seed0 raw-Fisher扫描：先筛选保护生效/折中候选，再选择A2=(a21+a22)/2最高的共同λ*。四行含EWC设置各复跑seeds0/1/2，共12次；另补SGD→GAM λ=0三个匹配对照，总15次新正式续训。复用已完成Task0模型，另重算5份raw Fisher。当前λ*=null、execution_ready=false，尚未提交后续可执行队列。原冻结扫描队列保持不变。

2026-09-21追加：首轮无合格λ时已授权继续λ扩大/细化搜索，见[阶段一B计划](execution/full_rq2_20260920/ewc_selected_lambda_followup_20260921/EXPERIMENT_PLAN.md)与[refinement_policy.json](execution/full_rq2_20260920/ewc_selected_lambda_followup_20260921/refinement_policy.json)。每轮最多6点，保留原保护/性能门槛及训练协议，合并各轮有效观测后选共同λ；四行复跑仍受λ选择条件约束。补充搜索当前为条件计划，未提交新GPU作业；活动14点扫描不变。

## 2026-09-22 06:44 Madrid：首轮raw Fisher扫描完成，第一轮细化已排队

首轮14个λ：12个完成50轮、2个数值发散；18项队列工作均完成验收，01:11 Madrid结束，原控制器与子进程已退出。λ=10000两任务平均80.475%最高，比λ0提高0.755pp；旧任务仅+0.33pp，未达到预设≥1pp，因此λ*尚未选定，四行EWC三seed复跑仍未启动。raw Fisher trace=34.069888，实际重算1次，Task0权重未变，两处归一化关闭。

[首轮完整结果与核验](execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/monitoring_20260922/STATUS_AND_RESULTS.md) · [细化执行说明](execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/RUN.md) · [细化实时队列](execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/campaign/STATUS.md)。

细化6个浮点λ来自[10000,30000]和[100000,300000]各3个对数四分点，使用同一Task0/raw Fisher、seed0、50轮协议。17项训练wrapper CPU测试、4项汇总测试、真实CPU恢复及新鲜独立文档原样检查通过（234项冻结指纹）。控制器PID3620620已在宿主机核实存活，6项训练+1项汇总均queued；06:43:49首次门禁检查GPU0–6全部繁忙，等待全部空闲后启动，每900秒检查，下次06:58:49 Madrid。此处排队不代表已开始GPU训练。

## 2026-09-22 08:24 Madrid：细化实验已在GPU3/4/6启动

用户新指令覆盖全卡等待：仅允许GPU[3,4,6,2,5]，排除GPU0/1，不等全部空闲，周期检查900秒，最大5并行。旧等待controller3620620经宿主机身份及无子进程核验后停止；科学代码、6个λ、raw Fisher、50轮与评估均未改变。新controller239745正常运行，GPU3/4/6分别训练λ=13160.7401、17320.5081、22795.0706，08:24已到3–4/50轮；其余3项排队，GPU2/5仍忙但纳入候选池。3项CUDA witness全部通过，尚无末端结果。

[当前队列](execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/campaign_gpu23456/STATUS.md) · [执行说明](execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/RUN_GPU23456.md) · [GPU与进程核验](execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/LAUNCH_GPU23456_VERIFIED.json)。旧campaign/保留历史等待状态，当前路径以CURRENT_QUEUE.json为准。

## 2026-09-22：细化第一轮完成，继续第二轮

第一轮于08:36 Madrid完成：6点均有验收终态，5个完成50轮、1个大λ数值发散；7/7队列作业有效，宿主机确认原controller与子进程全退出。累计20点中17完成、3发散。最高平均80.475%由λ10000和13160.7401并列，旧任务最大+0.33pp未达预设1pp；λ173205虽rawF加权位移平方减少99.36%，旧准确率仍不变。λ*未选，四行三seed复跑仍等待。

[完整核验与结果](execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/monitoring_20260922/RESULTS_AND_STATUS.md) · [合并20点报告](execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/report/RESULTS.md)。

按已授权的无合格λ继续搜索政策，第二轮6点为4053.6005、5477.2256、7400.8280、185515.7155、198701.3346、212824.1280，补查[3000,10000]及[173205.0808,227950.7057]。原科学训练实现逐字节未变，17项runner/11项report CPU检查、历史20点真实复核和新鲜独立文档243指纹dryrun通过。10:08 Madrid启动controller589483；GPU池[3,4,6,2,5]、排除0/1、无需全卡等待、poll900s/max5。实际训练状态以[第二轮实时队列](execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/campaign/STATUS.md)及LAUNCH_VERIFIED.json为准。

## 2026-09-22 10:53 Madrid：完整26点λ结果

第二轮已于10:21完成；初扫及两轮细化共26点，20完成50轮、6数值发散。宿主机确认controller与子进程全退出；无运行/待执行作业。独立表格核验CSV/JSON/原生指标及203个小文件SHA通过，无重复或缺失，未重复大型.pt审核。

[完整26行表](execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/complete_review/ALL_LAMBDA_RESULTS.md) · [原始完整CSV](execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/report/results.csv) · [完成报告](execution/full_rq2_20260920/ewc_lambda_refinement_round2_20260922/report/RESULTS.md)。

最新平均最高λ=5477.2255750516615：旧83.06、新78.35、BWT−4.24、平均80.705%，比λ0平均+0.985pp，旧任务无提升。最高旧准确率仍83.39%，仅+0.33pp，未达到既定1pp保护标准；λ*尚未选定，四行三seed复跑尚未启动。最近已测有限/发散边界173205.0808..185515.7155，比例约1.071，已达到边界细分分辨率。科学状态refinement_required，当前没有活动搜索controller，勿将历史running跟踪当实时状态。


## 2026-09-22：共同 λ=20000，四组 raw Fisher 续训

用户固定 λ=20000，EWCLoRA/FS-EWCLoRA/Task-1 shaping/Task-2 shaping 各 seed0/1/2，共12条 Task1 正式续训。复用Task0，原生raw Fisher，两trace归一化均关闭。允许全部空闲GPU0–6，四组首选0/1/2/5，无全卡等待，资源轮询900秒。

- [实验设置与运行说明](execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/RUN.md)
- [活动队列](execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/campaign_attempt3/STATUS.md)
- [启动恢复记录](execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/STARTUP_RECOVERY.md)

短测通过后自动放行正式续训；以活动队列为实时状态，不将本计划记作已完成结果。


### 2026-09-22：λ20000完整结果已核验

本轮12/12正式续训及最终24/24阶段完成；纳入重新核验的9条λ0对照后为21/21七行表。全部含P的行已替换为未归一化Fisher、共同λ20000结果；历史λ2000表保留。

[完整结果、配置与结论](execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/complete_review_20260922/FINAL_RESULTS_AND_CONCLUSIONS.md) · [新七行表](execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/complete_review_20260922/TABLE_IMAGENETA.md) · [LaTeX](execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/complete_review_20260922/table_imageneta.tex)。

结论：EWC加权位移约束确实生效；普通SGD下旧任务+0.54pp，但FS相对Shaping-only平均−0.41pp。不支持统一性能提升或λ最优声明。
