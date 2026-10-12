# FS-EWC RQ2：执行清单

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
| S03 | 01线第1项 | 邻域raw trace：2 anchors，r=0/.5/1/2/5/10/20，K16公共方向，固定256 train | MUST | RUNNING | 参数重名已修复并越过原错误；新目录重测194状态，49,664逐样本梯度；尚无完整邻域结果 |
| S04 | 01线第2项 | 2 anchors×2 actual t9方向，r=0/.25/.5/1/1.5/2，D与raw trace | MUST | IMPLEMENTED_QUEUED | 无训练；最多24状态；r0可复用 |
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
