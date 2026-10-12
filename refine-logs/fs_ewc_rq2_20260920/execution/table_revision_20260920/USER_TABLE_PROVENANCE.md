# 用户六行ImageNet-R表的来源核验

2026-09-20。本轮重新读取18个原始运行日志和配置，核对已保存的SHA，并从日志重新聚合。验证18/18运行；用户表42/42个显示值匹配。机器结果见 [USER_TABLE_PROVENANCE.json](USER_TABLE_PROVENANCE.json)，复核代码见 [audit_user_table.py](audit_user_table.py)。没有运行GPU或加载替代实验的权重。

原始仓库为 `/data/115-2/users/liying/low-rank-cl-new`，实验标识 `verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7`，不是本次CopyFlatCV/ImageNet-A实验。配置为ImageNet-R、seeds0/1/2、rank10、fresh K/V LoRA、Adam lr=.0005、weight decay=.005、50epoch/task、batch128；启用EWC时λ=1e7，γ=1。

六行分别对应 `baseline/gam_only/control/full/setting1/setting2`。完整方法full和第二任务shaping分支使用projected GAM更新；仅用C/F/P目标符号不能完整描述实现差别。历史源码没有随运行绑定commit，源码细节按当前匹配实现解释；字段、值和配置由原日志直接确认。该批 `save_ckp=False`，未找到匹配历史checkpoint，不能声称权重级复验。

Fisher列来自 `endpoint_fisher_total_mass`：各任务自己的数据、自己的head、有效K/V坐标的batch-gradient Fisher proxy trace。第二任务的数值不是旧任务在第二endpoint重新测量的Fisher，也不是累计buffer。其估计器不是逐样本梯度平方均值。

位移来自 `sqrt(mechanism[1].delta_norm)`，逐seed开平方后取均值。六行的1.560/3.131/.0106/.0204/.392/.129均符合此定义，没有在同一列混填平方与范数。其含义限定为合并前两个已评价endpoint间、排除分类头的K/V有效权重位移；不能用累积后的历史模型替代该endpoint。

Task-1 shaping相对EWCLoRA，三个seed的BWT差为+.63/+.64/+2.51pp，均值+1.26pp；新任务accuracy差为+.84/+.42/−3.91pp，均值−.8833pp；两任务均值差为+2.07/+2.02/+.40pp，均值+1.4967pp。支持这组历史设置下的保持改善，但不证明低Fisher普遍导致低遗忘。shaping-only及完整shaping+EWC各自较低的Fisher也伴随更负的BWT。seed标签配对不等于有证据证明历史随机过程严格配对。

修订表保留原数值，改正估计器、通道和更新规则描述：[均值版](imagenetr_legacy_verified.tex)、[均值±样本SD版](imagenetr_legacy_mean_sd.tex)。完整旧实现审查见 [LORA_METRIC_CODE.md](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/LORA_METRIC_CODE.md)，详细原始路径见 [LORA_PROVENANCE.md](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/LORA_PROVENANCE.md)。
