# FS-EWC RQ2：执行清单

**2026-09-21 20:55马德里完整核验：七行ImageNet-A两任务表已21/21个seed单元齐全；本次补训队列20:49:58完成10/10（6条正式续训、2短测、1验证、1汇总），失败/运行/待执行均0。本批控制器与训练进程已退出。见[完整七行结果与结论](execution/full_rq2_20260920/seven_row_extension_20260921/complete_review/FINAL_RESULTS_AND_CONCLUSIONS.md)、[LaTeX完整表](execution/full_rq2_20260920/seven_row_extension_20260921/complete_review/table_imageneta_complete.tex)、[逐seed结果](execution/full_rq2_20260920/seven_row_extension_20260921/final_report/individual.csv)。此前15/21、排队和4条运行描述均为历史快照。完整结果支持持续GAM的平均性能收益；当前λ=2000下FS相对Shaping only的平均差值−0.0017±0.1975个百分点，不支持稳定额外EWC精度收益。**

**2026-09-21 19:03马德里更新：按用户要求，补训队列的空闲GPU周期检查已改为每15分钟（900秒）；新调度器PID 1737208。六条正式续训及短测门控保持原配置。此前15秒检查与旧PID为历史记录。**

**2026-09-21 18:50马德里：七行表补训已加入后台队列（PID 1649105）。GAM→SGD＝C+F→C已放入主表倒数第三行；当前15/21个seed单元有完成结果。补训GAM→GAM（无EWC）与SGD→GAM＋EWC各3seed，共6段50轮Task1续训；复用已有起点及GAM→GAM＋EWC结果。启动时GPU0–6均有任务占用，等待空闲；两条隔离1轮短测和验证通过后才放行正式续训。六份CPU完整恢复、数据/评估配置和独立队列dry-run已通过；GPU短测尚不能计为完成。见[七行表](execution/full_rq2_20260920/six_row_imageneta_20260921/TABLE_IMAGENETA.md)、[后台实时状态](execution/full_rq2_20260920/seven_row_extension_20260921/campaign/STATUS.md)、[执行与验证协议](execution/full_rq2_20260920/seven_row_extension_20260921/RUN.md)。下方六行表“尚未启动”为此前历史记录。**

**2026-09-21 用户六行表覆盖审查：4/6目标行、12/18 seed单元已有结果，含从连续GAM链task01复用的FS-EWCLoRA。两行缺失合计6个50轮续训，处于设计状态、未入队；见 [六行结果表](execution/full_rq2_20260920/six_row_imageneta_20260921/TABLE_IMAGENETA.md) 和 [缺失清单](execution/full_rq2_20260920/six_row_imageneta_20260921/MISSING_EXPERIMENTS.md)。原88作业的完成状态不受影响。**

**2026-09-21 完整核验：正式RQ2已于11:57马德里完成88/88，B3三seed四格12/12、B1/B2诊断54/54、路径数值检查18/18通过；S3另有18/18正式实验和20/20短测完成。见 [完整结果与结论](execution/full_rq2_20260920/complete_review_20260921/FINAL_RESULTS_AND_CONCLUSIONS.md)、[四格结果](execution/full_rq2_20260920/final_report/RESULTS.md) 与 [完成凭据](execution/full_rq2_20260920/complete_review_20260921/COMPLETION_VERIFICATION.json)。下方S08/S09的TODO及其他旧进展保留为历史快照，不表示正式队列仍待执行。ImageNet-R/full-model复验和S3条件确认不在已完成核心范围内。**

**最新执行范围：用户已授权“使用空闲GPU进行运行”。本轮按附件完整核心 B1/B2/B3 执行，含三seed四格表、Fisher及参数移动；88作业正式计划和旧seed0配对补全独立保存。见 [完整执行协议](execution/full_rq2_20260920/EXECUTION.md)、[正式实时状态](execution/full_rq2_20260920/formal_queue/STATUS.md)、[旧seed0补全状态](execution/full_rq2_20260920/pilot_queue_paired/STATUS.md)。下文“尚未入队”“四格不默认执行”等文字是此前范围的历史记录，不代表本次授权。**

**最新收口（2026-09-20）：恢复队列已于15:36:08 UTC完成全部15/15作业，A、邻域、bridge、全部逐任务端点和t9路径均有真实结果。下面17:30马德里等时间点为历史快照。按用户最新要求，A保留完成状态，后续只围绕B/C机制缺口，目标已确定为当前视觉GAM+EWC实现。新计划与实时工作清单见 [机制接续计划](mechanism_next_20260920/EXPERIMENT_PLAN.md) 和 [接续清单](mechanism_next_20260920/EXPERIMENT_TRACKER.md)。本次补测提案未入队。**

最新核验快照：2026-09-20 17:30马德里时间，10/15作业complete、2个neighborhood运行中（43/97、46/97）、3个待执行，无新失败。两条trajectory和path均完成，路径全部gate通过。详见 [CURRENT_STATUS.md](execution/monitor_20260920/CURRENT_STATUS.md)；下表按此快照更新，下文恢复过程中的时间点属于历史记录。

2026-09-20：旧双线 pipeline 已于12:32:42 UTC实际启动；两条 sanity 通过，两条 Task1 续训各50 epoch完成。邻域作业于12:33:21 UTC因 `Runner.measure` 的 `direction` 参数重名失败，队列等待在跑训练结束后于12:39:07 UTC退出为 failed。训练结果不属于 pending。

已将冲突参数改名为 `perturbation`，保留方向元数据。恢复队列PID为 `1341023`，pipeline于15:26:06 UTC启动。以下阶段状态为15:27:22 UTC核验快照：新版两条sanity、两条A端点评估已完成，训练日志确认 `Already complete`，01线邻域与23线trajectory正在运行。当前恢复目录为 `execution/campaign_seed0_dualline_recovery1/`；最新状态见 [QUEUE_STATUS.md](execution/campaign_seed0_dualline_recovery1/QUEUE_STATUS.md)，方案见 [RECOVERY.md](execution/campaign_seed0_dualline_recovery1/RECOVERY.md)。新诊断写入 `results_seed0_recovery1/`，两条训练原资产哈希未变。

**2026-09-20：按用户要求调整为两线并行。等待 GPU0–3 的整个 TreeLoRA 队列状态为 `completed`，并确认四张卡连续空闲60秒后，在 GPU0/1 完成共同 sanity；GPU0/1 随后执行 neighborhood→bridge，GPU2/3 同时执行 training→table_a→trajectory→path，report 等两线完成。GPU0/2 为 SGD，GPU1/3 为 GAM。仍为8阶段、15作业，科学设置不变。**

前驱状态：`/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`。恢复继续要求其整队完成、四卡连续空闲60秒。当前接续实时状态见 [QUEUE_STATUS.md](execution/campaign_seed0_dualline_recovery1/QUEUE_STATUS.md) 和 [queue_state.json](execution/campaign_seed0_dualline_recovery1/queue_state.json)；pipeline 状态见同目录 [STATUS.md](execution/campaign_seed0_dualline_recovery1/STATUS.md) 和 [pipeline_state.json](execution/campaign_seed0_dualline_recovery1/pipeline_state.json)。原 failed 记录保留于 `execution/campaign_seed0_dualline_gpu03/`。

旧等待进程 PID `541721` 已安全停止，未启动本批 GPU 实验，TreeLoRA 前驱未受影响。旧安排归档于 `execution/campaign_seed0_after_gpu03/before_dualline/`，替换凭据见 [SUPERSEDED_FOR_DUAL_LINE.json](execution/campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json)。

此前 GPU4/6 队列的停止属于历史记录：[STOPPED_BY_USER.json](execution/campaign_seed0/STOPPED_BY_USER.json)。更新前文档已归档至 `execution/campaign_seed0/before_gpu03_requeue/`。已完成CPU审查、旧JSON重分析和历史精度排查；上一轮四卡 kernel witness 和两条完整 sanity 已通过，恢复仍重做环境预检和新版 sanity。执行协议见 [execution/EXECUTION.md](execution/EXECUTION.md)，详细科学设置见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

| ID | 当前资源顺序 | 工作与配置 | 优先级 | 当前状态 | 训练需求/交付 |
|---|---|---|---|---|---|
| S01 | 1 | seed0旧JSON：两方法×train/test，G/Q95/prefix半径；epsilon=.001/.005/.01/.02/.05 | MUST | DONE_CPU | 无；324行damage、40行半径CSV；保留test不一致结果 |
| S02 | 共同前置，GPU0/1 | anchor/t1–t9/源码/两任务pair/有效权重/数据变换/环境；补r0/端点forward/raw-F/吞吐 | MUST | RECOVERY_SANITY_PASSED | 新版两条完整sanity重测通过；旧结果保留不修改、不混用 |
| S03 | 01线第1项 | 邻域raw trace：2 anchors，r=0/.5/1/2/5/10/20，K16公共方向，固定256 train | MUST | COMPLETE_SEED0 | 194状态完成；49,664逐样本梯度样本求值；正式结果见report_seed0 |
| S04 | 01线第2项 | 2 anchors×2 actual t9方向，r=0/.25/.5/1/1.5/2，D与raw trace | MUST | COMPLETE_SEED0 | 2×2真实方向交叉测量完成，24导出行（含共享r0），无训练 |
| S05 | 23线，A后 | 全task旧任务损伤/accuracy/drift；seed0 t9路径9节点，I与S_path | MUST | TRAJECTORY_AND_PATH_COMPLETE | 当前FP32全端点复评与两条t9路径均完成；求积收敛和界检查均通过；S_path为SGD .3680576、GAM .01479309 |
| S06 | 5 | r=1/5/10/20，等预算GAM-directed proxy，方向交叉评估 | NICE-TO-HAVE | CONDITIONAL | 无训练；可直接复用inner-search才做，不称worst-case |
| S07 | 23线第1、2项 | A表：普通/shaped Task0→相同SGD+EWC Task1；a11/a21/a22/D0/drift/raw trace | MUST | TRAINING_AND_ENDPOINT_COMPLETE | 两条Task1各50 epoch已完成；配置/资产/指标与RNG配对已核验；恢复确认跳过训练，两条A端点评估已完成 |
| S08 | 4 | seed1/2同配置，B/C共用完整序列，所有seed必做B1与t9路径 | MUST-FORMAL | TODO | 预计4条完整十任务链；B1+t9新增108,544逐样本梯度；另计A分支/B2 |
| S09 | 4 | 3 seeds、r=0:.25:20、K100公共方向，全train/test，G/Q95/半径 | MUST-FORMAL | TODO | 无额外训练；96,012次数据集前向；历史非配对曲线不混合统计 |
| S10 | 5 | seed0 t1/t5路径、full-model轻量邻域、256与全train估计稳定性 | OPTIONAL | CONDITIONAL | 优先复用；full展示目录尚需匹配；不默认完整路径 |
| S11 | 5 | 仅必要时加强保护隔离：无保护的关键pair | OPTIONAL | CONDITIONAL | 缺分支才补；不默认2×2×3 seeds |

## 首批三个工作包

1. **S01+S02剩余部分**：S01已完成；为S02实现离线endpoint恢复，固定新的train评估视图，完成test历史复现/新视图端点一致性、逐样本梯度检查和计时。
2. **S03+S04**：同一个离线入口，共用256样本、坐标、公共方向；raw Fisher不使用归一化保护矩阵。
3. **S05剩余部分**：旧全task CSV已有，补两个t9路径的9节点测量、5/9节点求积检查及S_path。

S07两条严格配对的Task1续训已完成，[只读摘要](execution/monitor_20260920/training_readonly_summary.json)记录完整50 epoch、有限训练loss、现有accuracy/D0/drift及55条完全一致的RNG收据。恢复仍保留两条 training job 和 GPU2/3 witness，但原runner确认 `COMPLETE.json` 后直接返回，随后进行 A 端点评估；与 GPU0/1 的 S03/S04 并行。S05 随 A 端点评估之后使用 GPU2/3，是同卡资源安排，不读取新 A checkpoint；其 GAM t9 路径仍来自旧 GAM→GAM 链。新的 A shaped→SGD 分支与该旧路径保持独立。真实依赖见 [DEPENDENCIES.md](execution/DEPENDENCIES.md)。训练/评估顺序取决于测量正确性与资源，不取决于结果是否支持预期。

旧 `report_seed0/` 框架已归档至 `execution/campaign_seed0_dualline_recovery1/before_fix/report_seed0/`。最终汇总仍写回 `execution/report_seed0/`，需等待新版 sanity、B/C 与 A endpoint 完成且所有数值 gate 通过。

## 防止错误复用

- 主线是 ImageNet-A/rank10/lambda2000/no averaging；ImageNet-R ratio-FS/lambda1.2e6 pair不填主表。
- `upper_bounds`不是顺序两任务对照；full-model seed9 smoke不算正式结果；指定top1% Fisher-cache作业失败不算完成。
- 旧半径train为一次缓存随机增强视图，新deterministic train需独立标记；旧原始train数值不作为新transform复现硬门槛。
- 旧跨T的动态trace约1是归一化；旧方向索引不代表方法间配对；`delta_norm`字段须区分范数与平方范数。
- 新Fisher/路径/两任务续训代码已实现；必须以作业完成凭据和数值gate判断完成，排队不等于完成。

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
