# 新数据集 full 结果与可精确重算的 full 位移

2026-09-20。本轮仅搜索已有文件、解析日志、CPU重算；没有启动GPU、训练或恢复任何队列。“新找到”不代表本轮新训练。

## 结论

**已有 ImageNet-A 的 full 训练结果；但目前找到的这些结果没有配对模型资产，无法从权重重新算位移。另找到 ImageNet-R 一套完整 full 序列，可精确重算纯 backbone 位移，已经完成。** 它是独立的6月22–23日实验，不是原六行表的替换模型，也不是 ImageNet-A。

旧六行表 final 文件不含历史参数备份，详见 [FULL_HISTORY_RECOVERY.md](FULL_HISTORY_RECOVERY.md)。原六行仍只能作此前有条件的新头能量分解；不能用下面新序列修补旧六行。

## 1. ImageNet-A：确有完整矩阵，但不等于全部有效成功

本次限定扫描6个 full目录：`fullmodel_imageneta_sgd`、`fullmodel_imageneta_aligned`、`fullmodel_imageneta_finetune`、`imageneta_sgd_lr_sweep`、`imageneta_ewc_fullmodel_lambda_sweep` 及 `config_FullModel` 下 ImageNet-A 分支，共79份 full 指标JSON：

- 69份具有10任务完整、下三角有限的accuracy矩阵。
- **这69份中有10份日志出现训练loss/CE/EWC NaN/Inf；不能称69次成功。** 其余59份也仅完成本轮矩阵/训练日志扫描，未作整套实验完整性认证。
- 四份LR探测仅两任务；其余不满10任务的记录也保留在清单，未包装成完成十任务。
- 上述79个run的checkpoint目录均未找到 `.pt/.pth/.pkl/.safetensors` 模型文件；全outputs_logs/refine-logs的大文件检索也未发现它们另存的明确full模型端点。
- 这些full ImageNet-A记录为6–7月历史运行，最新候选修改时间为7月13日。9月19日的 `imageneta-batched-fisher-lambda-plan`、近期 `as1_normfisher_*` 结果实际是LoRA，不能因为有全模型state_dict就当full训练。

可核对的代表性历史结果（FAA为最后一行10任务等权平均，不是样本加权总准确率；各配置不同，不据此作受控方法比较）：

| 运行 | seed | 两任务a11/a21/a22 | 十任务FAA | 本次训练非有限日志扫描 |
|---|---:|---|---:|---|
| `imageneta_full_sgd_e20_lr001_cosine_s0` |0|85.34 / 61.56 / 80.31|15.017|未匹配到|
| 同SGD配置s1 |1|83.06 / 62.54 / 80.71|18.029|未匹配到|
| 同SGD配置s2 |2|83.06 / 61.56 / 75.59|20.621|未匹配到|
| `imageneta_full_ewc_norm_e20_lr001_lam2000_s0` |0|84.36 / 65.80 / 79.92|21.506|未匹配到|
| `imageneta_full_ewc_gam_wmergethenfisher_norm_e20_lr001_lam2000_int20_omega030_alpha02_rho001_s0` |0|86.64 / 48.53 / 85.83|24.832|未匹配到|

最后一行含轨迹平均/merge与其他参数差异，不是“仅Task1 shaping、未来相同EWC”的关键对照。更高FAA也不意味着Task0保留更好。

完整路径、JSON哈希、矩阵完整性、NaN/Inf计数与存储检查见 [new_full_results_inventory.csv](new_full_results_inventory.csv) / [JSON](new_full_results_inventory.json)。`complete_10task` 字段仅表示10任务矩阵完整有限，必须结合训练非有限计数，不解释为已通过完整审计。

## 2. 可恢复 ImageNet-R full 序列：已直接重算，不再含新头

根：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c_fullmodel/`。

- Anchor：`stage1_task0/task0_fm_lam2000_lr0p01_e50/{sgd,gam}/seed0/checkpoints/task_00.pt`。
- 后续全部端点：`stage2_T/T_from_task0_fm_lam2000_lr0p01_e50/{sgd,gam}/T10/seed0/checkpoints/task_01.pt` 至 `task_09.pt`。
- 原配置：[base_s1c_fullmodel_imagenetr_e50_lr001.yaml](/data/140-1/users/liying/CopyFlatCV/config_Expand/sgd_gam_s1c_fullmodel/base_s1c_fullmodel_imagenetr_e50_lr001.yaml)。原启动脚本：[run_s1c_fullmodel_staged_T.sh](/data/140-1/users/liying/CopyFlatCV/scripts/run_s1c_fullmodel_staged_T.sh)。实际覆盖参数另取原日志，不用今天配置反推。

**协议不同于旧六行：** ImageNet-R、seed0、10×20类，ViT-B/16 augreg2_in21k_ft_in1k，50epoch/task、LR=.01、batch128、SGD+cosine、momentum/weight decay0。普通线Task0=SGD、未来SGD+EWC；GAM线Task0=GAM、未来GAM+EWC。Task2及之后lambda2000、gamma.9；GAM rho=.02/normrho=.2/gamma=.03。不是只对Task1 shaping的pair。

Fisher是日志明确的 `ncm_empirical`、tau16，`batch_gradient_square_proxy`，不是逐样本empirical，也不同于旧六行CE batch Fisher。机制记录明确 `ewc_include_fc=false`，保存的Fisher有150块、85,798,656元素，无FC。日志/配置未显式记录启用trace normalization；实际Task0 buffer trace为70.518806/30.965117，Task1累计为121.562071/53.248322，证明它们没有按单位trace存储。不要补写成trace=1或拿它们替代新计划的raw CE Fisher。

本次CPU直接从两个端点相减，backbone覆盖全部非FC浮点参数；另保存K/V weight投影和分类头分解。**这里K/V仅与旧表fresh-KV LoRA的坐标范围相对应；新版CopyFlatCV LoRA使用Q/V，不能把本次K/V称为与新版LoRA共同坐标。** 主结论使用全共享backbone的有效权重距离。

| 路径（代码task编号） | 方法 | 起点Task0 accuracy | 终点Task0 accuracy | 相对起点变化(pp) | 纯backbone L2 |
|---|---|---:|---:|---:|---:|
| t0→t1 | SGD→SGD+EWC |92.92|80.97|-11.95|**2.041041**|
| t0→t1 | GAM→GAM+EWC |93.71|85.53|-8.18|**2.602965**|
| t0→t9 | SGD→SGD+EWC |92.92|60.22|-32.70|**5.571344**|
| t0→t9 | GAM→GAM+EWC |93.71|57.39|-36.32|**7.605112**|

第二任务新类accuracy为92.47/94.56。两任务时GAM共享backbone移动较大而旧准确率下降较少；到最终t9，GAM旧Task0下降反而较多。必须同时报告这一限制，不能说该full序列在全部长度都支持更低旧Task0损伤。十任务整体FAA为64.378/71.507，整体BWT为−23.9756/−20.4278，整体表现与最老单Task0的指标不是同一结论。

原新系列日志 `delta_norm` 已经是**相邻任务backbone平方位移**，不含头。例如t1为4.165848707/6.775426829，开方即上表前两行。t9日志是t8→t9，不能误用成t0→t9；本次累计位移由正确端点另算。

## 3. 身份核验与分类头能否单独重算

已检查20个checkpoint（两方法×10端点），完成18个相邻步以及18个t0累计距离：

1. stage2日志resume路径与当前Task0路径严格一致；Task0独立metrics的a11与整个序列首行一致。
2. Task0文件内Fisher trace与Task0原日志匹配；各task_idx、cur_task、known_classes、头行数正确。
3. 全18步CPU backbone平方位移与历史日志 `delta_norm` 最大绝对差 **4.79e-8**；各Fisher trace也与对应原日志匹配。
4. 文件mtime与原日志逐任务保存记录相差<2秒；所有完整路径与当前SHA256已写入 [new_full_checkpoint_manifest.json](new_full_checkpoint_manifest.json)。这比仅路径存在强得多，但没有声称持有执行当年的SHA。
5. `checkpoint_params` 与同文件当前 `network_state` 一致，它是当时新提交的anchor，不是文件内部额外保留了上一任务参数；本次使用明确存在的独立Task0/Task1…文件计算。

**旧头参数变化可以精确计算：所有t0→t的20类旧头变化均为0。** 新增头有最终权重，可精确计算相对零的范数，但没有保存新头出生时的随机初始化快照，所以不能精确得到新头“实际初始化→最终”的训练位移。

| t0→t1 | SGD | GAM |
|---|---:|---:|
| 旧20类头变化L2 |0|0|
| 新20类头相对零L2（非训练位移） |4.469002|4.376826|
| 纯backbone L2 |2.041041|2.602965|

完整18行 [new_full_checkpoint_drift.csv](new_full_checkpoint_drift.csv) 中 `step_*` 为相邻步，`cumulative_*` 为相对Task0；`new_head_vs_zero` 坚持注明零参考，不混写为新头训练位移。可复算脚本 [recompute_new_full_cpu.py](recompute_new_full_cpu.py)，只需要CPU读取checkpoint。

## 4. 用户给定其他 full 目录

重新核查仍与之前一致：

- `verify_imagenet_r_1x1_fullmodel_ratio_twostage`：只有seed9、1epoch smoke，4份metrics、两个Task0 checkpoint；没有正式完整two-task shaping对照或Task1模型。
- `fishershape_imagenetr_1x1_fullmodel`：4份metrics/4个模型，主要普通Task0 seeds0/1/2及task0-only上界；没有可闭环的完整shaped→future pair。
- Full FQA/半径诊断的Task0模型与上述ImageNet-R full家族可关联，但单点FQA不是额外的两任务训练结果。

因此可以用新找到的完整full序列说明如何正确排除头并重算，不能宣称新ImageNet-A full已经补齐RQ2 A/B/C，也不能替换原六行数据。
