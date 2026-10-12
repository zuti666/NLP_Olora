# ImageNet-R 两阶段 LoRA：现有结果与可恢复资产

核查时间：2026-09-20 16:28:16 UTC。本轮仅重新读取文件、核对配置／SHA与汇总数字，未启动GPU、训练或评估。范围继续是 **ImageNet-R**。精确路径与资产状态见 [existing_results_manifest.json](existing_results_manifest.json)，六行统计见 [ratio_main_results.csv](ratio_main_results.csv)。

## 先区分三个不同实验来源

| 目录／实验 | 真实算法与用途 | 完成范围 | 是否适合继续离线评估 |
|---|---|---|---|
| `verify_imagenet_r_1x1_lora_ratio_full_twostage`，`ratiofs_l12c20_full_twostage_s345_v1` | 连续Q/V adapter，Ratio-FS与EWC两阶段主实验 | 六个主变体×seeds3/4/5全部完成；另有四个辅助变体×3 | **可以：Task0完整模型及Task1 A/B、FC、meta齐全** |
| `verify_imagenet_r_1x1_lora_fixedfs_upper_bounds` | native SGD、无EWC；joint40／task0_only／task1_only上界 | s345三种上界×3完整；s012 task1_only seed2缺完成metrics；另含烟测 | 仅适合作上界参考，不能当作EWC／GAM六行表 |
| 旧`low-rank-cl-new` May六行表 | 真正`ewclora_gam_factored`／`ewclora_gam_projected`；Adam、fresh K/V | 六行×seeds0/1/2日志完整 | 未保存匹配checkpoint，不能重载旧模型继续Fisher评估 |

**Ratio-FS不是GAM。** 本次能够复用三seed模型继续评估的完整ImageNet-R六行主表，应标记“EWC vs EWC + Ratio-FS／shaping”，不能仅因概念上都是shaping就重命名成GAM。此前旧GAM表的checkpoint缺失，不可用另一套模型替代它的历史端点。

## 可继续评估的六行结果

以下从18份原始JSON重新读取；本次文件SHA与此前来源清单全部一致。均值先逐seed计算；r为每seed平方能量开根后求均值。

| setting／变体 | 第一任务更新 | 第二任务更新 | a11 | a21 | a22 | BWT pp | 有效Q/V位移 |
|---|---|---|---:|---:|---:|---:|---:|
| baseline／Vanilla | SGD | SGD |89.3633|84.5900|88.8900|−4.7733|0.850642|
| fs_only／Ratio-FS only | Ratio-FS | Ratio-FS |89.4667|84.8033|89.5400|−4.6633|0.856600|
| control／EWC | SGD | SGD+EWC |89.3633|85.9000|87.8200|−3.4633|0.427166|
| fs_ewc／Ratio-FS+EWC | Ratio-FS | Ratio-FS+EWC |89.4667|85.7433|88.1900|−3.7233|0.461659|
| factored_s1／仅第一任务shaping | Ratio-FS | SGD+EWC |89.4667|85.9567|87.8200|−3.5100|0.448335|
| factored_s2／仅第二任务shaping | SGD | Ratio-FS+EWC |89.3633|85.5867|88.3767|−3.7767|0.456989|

如继续比较两阶段完整方法，直接对应 `control` 与 `fs_ewc` 的三seed六个run；若要隔离第一任务起点、未来保持相同SGD，则对应 `control` 与 `factored_s1`。两种比较不是同一个问题。当前accuracy并未显示fs_ewc或factored_s1的平均BWT优于control，必须如实保留，后续Fisher测量不能预设结论。

该cohort共48个metrics文件：正式六个Task0 anchor、30个Task1 endpoint（六主变体＋四辅助变体各三seed），另有12个seed9烟测。六行主表只使用18个正式Task1 endpoint，不能按文件总数推断更多训练seed。

## Task0完整，Task1可重建评估但不是完整续训checkpoint

六个独立Task0 anchor（native／ratio各seeds3/4/5）均含：

`task_00.pt`、`fisher_state_0.pt`、`lora_w_a_0.pt`、`lora_w_b_0.pt`、`fc_state_0.pt`、`lora_meta_0.json`、`experiment_identity_0.json`、`parent_artifacts.sha256`。

18个主Task1 endpoint均含：

`lora_w_a_1.pt`、`lora_w_b_1.pt`、`fc_state_1.pt`、`lora_meta_1.json`，以及旧格式CLs weight/bias文件。**均未保存`task_01.pt`、`final_model.pt`、`fisher_state_1.pt`**；launcher明确stage2 `save_final_model=false`。缺少完整Task1文件并不等于无法评估：可从该run自己的完整Task0恢复冻结backbone，再加载Task1 A/B及完整FC，重建对应端点。正式测量前仍应做模型输出／历史accuracy sanity，不能仅凭文件存在宣称端点恢复已验证。

所有meta均为12层×Q/V共24块，task_id保持0，说明持续更新同一adapter。有效移动应计算 **`||B1 A1 − B0 A0||_F`**，不能把第二任务最终`||B1 A1||`当作移动，也不能使用A/B因子距离。此前独立CPU审计已验证control与factored_s1共六对，最大平方误差约7.6e−9、旧head精确不动；该六对审计范围不自动覆盖尚未逐tensor核验的fs_ewc行。

本次重新核对了每个anchor的parent清单中A/B/FC、identity和metrics等小文件SHA，全部匹配；大体积`task_00.pt`／`fisher_state_0.pt`本轮只确认存在并保留parent期望SHA，未重复加载或重哈希。机器清单逐项写明`sha_checked`，避免将存在性检查扩大成权重级验证。

公共输出根：

`/data/140-1/users/liying/CopyFlatCV/outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage/logs_inc_lora/ewclora_normfisher_fishershape/`

以seed3为例，EWC配对：

- anchor：上述根下 `ratiofs_stage1_native_ratiofs_l12c20_full_twostage_s345_v1/imagenetr/3/lora_ratiofs_stage1_native_ratiofs_l12c20_full_twostage_s345_v1_s3/exp_run/checkpoints/`
- endpoint：上述根下 `ratiofs_full_twostage_control_ratiofs_l12c20_full_twostage_s345_v1/imagenetr/3/lora_1x1_control_ratiofs_l12c20_full_twostage_s345_v1_s3/exp_run/checkpoints/`

EWC+Ratio-FS配对：

- anchor：上述根下 `ratiofs_stage1_ratio_ratiofs_l12c20_full_twostage_s345_v1/imagenetr/3/lora_ratiofs_stage1_ratio_ratiofs_l12c20_full_twostage_s345_v1_s3/exp_run/checkpoints/`
- endpoint：上述根下 `ratiofs_full_twostage_fs_ewc_ratiofs_l12c20_full_twostage_s345_v1/imagenetr/3/lora_1x1_fs_ewc_ratiofs_l12c20_full_twostage_s345_v1_s3/exp_run/checkpoints/`

每对的**完整绝对路径**、所有sidecar、原始metrics与log、seed、原始值、配置及核验状态全部列在本目录JSON的`ratio_main`中，无需人工拼接。

## 原始配置、代码与Fisher列

18个主run统一：ImageNet-R、ViT-B/16、rank10、每任务20类、两任务、seeds3/4/5、SGD base、LR .02、50epochs、batch128、momentum/weight_decay0、cosine。EWC启用时lambda=1.2e6、gamma1，任务Fisher归一化、累计Fisher不再次归一化、damping beta .2；`fisher_mode=batch_empirical`。Ratio-FS使用c=.20、rho=.05、`fs_space=factor`、`fs_alpha_mode=ratio`；实际带shaping的优化器类是`FisherShape`。

绝对入口／配置：

- `/data/140-1/users/liying/CopyFlatCV/scripts/run_verify_imagenet_r_1x1_lora_ratio_full_twostage.sh`
- `/data/140-1/users/liying/CopyFlatCV/config_Expand/fishershape_lora_imagenetr/factorfs_lora_imagenetr_t10c20_r10_e50_lr002_lam8e5_c025_normablation.yaml`
- `/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py`
- `/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py`
- `/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py`

基础YAML文件名中的lam8e5/c025不能直接当成最终训练配置；实际lambda1.2e6/c.20来自launcher覆盖和各run日志。以上源文件是当前可读实现，不能仅凭当前文件hash认定精确历史源码未改变；历史run的identity、父资产hash、日志配置应一并检查。

原始metrics包含训练保护Fisher的加权位移等统计，Task0 identity中的Fisher trace约1是归一化保护buffer。它们**不是同一协议下24块有效Q/V、逐样本、未归一化的旧任务endpoint empirical Fisher trace**，不能直接填入新机制表。

另已有原始empirical-Fisher诊断：

`/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/diagnostics/{sgd,ratio}_seed{3,4,5}.json`

这些对应本cohort的Task0 anchors，但使用factor＋head坐标，不能直接当成QV有效权重的trace；也不是Task1端点结果。另一个`imagenetr_endpoint_region_heldout_s345_v2`属于独立80/20数据划分的新训练cohort，同样不能混填本表。因此缺口应写“本cohort两阶段统一QV坐标的逐样本raw Fisher”，不能绝对说服务器上没有empirical Fisher结果。

## fixedfs与upper_bounds不能混入主cohort

`/data/140-1/users/liying/CopyFlatCV/outputs_logs/verify_imagenet_r_1x1_lora_fixedfs/` 的正式exploratory标签 `fixedfs_exploratory_s012_a005_lam4e5_rho005_v1`，当前只有seed0四条完整第二任务结果：baseline、fs_only、control、fs_ewc；不是完整六行×三seed。其余文件含Task0 anchors、校准和烟测。lambda4e5、固定alpha .05也不同于上面的Ratio-FS主cohort。

`/data/140-1/users/liying/CopyFlatCV/outputs_logs/verify_imagenet_r_1x1_lora_fixedfs_upper_bounds/`共有20个完成metrics：17个正式上界＋3个烟测。s345的joint40／task0_only／task1_only均三seed；s012的task1_only缺seed2完成metrics，其trainer log存在不能视为完成。全部完成文件的矩阵是单阶段1×1，不是顺序两任务2×2。代码路径虽叫fishershape，实际配置`fs_direction_mode=native_sgd`、`ewc_lambda=0`、`use_ewc=false`；没有该目录内“EWC vs GAM六行”的证据。

上界启动脚本：`/data/140-1/users/liying/CopyFlatCV/scripts/run_verify_imagenet_r_1x1_lora_fixedfs_upper_bounds.sh`。原始上界每条metrics、配置摘要与checkpoint存在状态已另列机器清单，避免它们与主对照错配。

目前最直接的继续方式是复用已存在的ImageNet-R Ratio-FS三seed主cohort，完成匹配端点恢复sanity后补统一Fisher评估。此次盘点本身没有启动该评估，也没有切换到ImageNet-A或新增训练。
