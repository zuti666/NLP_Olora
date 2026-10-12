# ImageNet-R：两阶段方法身份与离线复评边界

2026-09-20。限定为 **ImageNet-R**；只读检查已有脚本、日志、JSON和模型文件清单。未启动GPU、训练或队列，未改CopyFlatCV源码。本文件沿用代码Task0/Task1，对应论文Task1/Task2。

**EWC+GAM、固定系数FS、Ratio-FS不是同一种算法。现成Ratio-FS两阶段模型能用于新离线评估，但不能改名成GAM。** 原六行Adam-LoRA表是真GAM家族，不过该批未保存权重，不能为那六行直接补逐样本Fisher。CopyFlatCV另有真GAM的ImageNet-R十任务模型sidecar，但本次未找到与之配置匹配的普通EWC完整对照。Full另有可完整恢复的ImageNet-R SGD/GAM序列，可独立复评；它不是旧full六行表的原始模型。

## 1. 方法身份：从实际调用区分

| 现有家族 | 实际更新 | 可使用的名称 | 不能替代的对象 |
|---|---|---|---|
| 旧May三seed六行，`low-rank-cl-new` | Adam；GAM包含4个CE梯度探测；full行还有EWC投影 | EWC、GAM+EWC（注明projected）、阶段性GAM | 不能以新Ratio-FS模型代表旧表 |
| `ewclora_normfisher_gam` / `gam_sgd_*` | 原生GAM，base SGD；独立clean EWC梯度相加 | GAM+EWC | 不能把名字内含SGD解释成无GAM |
| `verify_imagenet_r_1x1_lora_fixedfs` | 两个CE梯度的有限差分HVP，固定alpha | Fixed-FS+EWC | GAM、Ratio-FS |
| `verify_imagenet_r_1x1_lora_ratio_full_twostage` | 两个CE梯度，归一化shaping增量 | Ratio-FS+EWC | 四梯度GAM |
| `verify_imagenet_r_1x1_lora_fixedfs_upper_bounds` | `fs_direction_mode=native_sgd`，实际plain SGD、单CE反传 | task-only / joint40 SGD上界 | 两任务遗忘对照、GAM、shaping结果 |

Ratio/Fixed实现见 [fishershape.py](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/fishershape.py:1) 及 [as1_normfisher_fishershape.py](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py:1)。令 `g0=grad CE(theta)`、`s0=rho/(||g0||+xi)`、`g1=grad CE(theta+s0*g0)`、`rF=(g1-g0)/s0`：

- Fixed-FS：`g0 + alpha*rF + gEWC`。
- Ratio-FS：`g0 + c*||g0||*(g1-g0)/(||g1-g0||+eps) + gEWC`。
- Native EWC：`g0 + gEWC`，实际一次CE反传。
- GAM：还包括grad-norm ascent分支及`g2/g3`，经过beta/gamma梯度分解；不是把上式alpha换名。当前AS1-GAM组合为 `gclean + lambda_flat*(gGAM-gclean) + gEWC`，lambda_flat=1时为`gGAM+gEWC`。

这些CopyFlatCV FS/AS1-GAM实现的EWC梯度在clean权重计算，CE probe里不加入EWC。当前Ratio-FS类的callback名虽带projector，实际只相加；不能据名字宣称EWC保护梯度投影。GAM本身仍有其gamma方向分解，须与外加EWC投影区分。Shaping发生在可训练A/B因子及分类头坐标，EWC二次项定义在有效Q/V `BA` 坐标，再通过链式法则回传A/B。不能把训练中factor-space rho直接当离线有效权重半径r。

## 2. 真GAM的ImageNet-R资产：候选存在，但受控pair尚缺

本次枚举CopyFlatCV内实际路径含`/imagenetr/`的LoRA结果，按原日志`optimizer_type`核验。发现多条不含in-task merge的`ewclora_normfisher_gam`完整十任务序列，7条保留Task0/Task1的A/B和FC sidecar。完整路径、原日志配置、两任务切片和文件存在性见 [true_gam_imagenetr_candidates.json](true_gam_imagenetr_candidates.json)。这次未加载大权重，也未把sidecar存在性提升为完整基底身份认证。

最朴素候选路径：

```
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_Expand/logs_inc_lora/
  ewclora_normfisher_gam/gam_sgd_lr001_e50_rho005_f100/imagenetr/0/
  as1normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100_s0/exp_run/
```

- 原矩阵两任务切片：`a11=89.94, a21=85.06, a22=88.42`；完整10任务。
- 原日志记录：GAM/base SGD、lambda_flat1、lambda_EWC2000、gamma_EWC.9；LR=.01/.01、50epochs、batch128、rho=.05、normrho=.1、gamma_GAM=.1、rank10；Task Fisher normalize=true。日志未记录累计归一化开关，不从今天默认值补写历史行为。
- `checkpoints/lora_w_a_{0,1}.pt`、`lora_w_b_{0,1}.pt`、`fc_state_{0,1}.pt`存在；没有同目录完整`task_00.pt`/`final_model.pt`基底备份。
- 原日志明确记录加载`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`；本机`/data/115-2/users/liying/cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k`存在缓存。因此有离线恢复基底的候选来源，但日志只指向当时`main`，没有保存历史revision/hash，不能仅凭同别名保证与历史基底逐位相同。该日志还追加了两个运行段，当前sidecar/矩阵必须与最后匹配段核验。需先重测Task0/1 accuracy并比较日志、缓存内容hash及可用其他完整模型基底，才能升级为可信可测端点。
- 其他候选有LR=.02/.015、.03/.015、.03/.03、.04/.04、.05/.05及独立head LR版本，不随结果高低挑选。

**当前没有找到同配置native EWC baseline，不足以形成完整的EWC vs EWC+GAM离线受控对照。** 本次CopyFlatCV R-LoRA native SGD且不在FS家族的另一条序列是`lorar_sgdplain_oldlearner_1x_noewc...lam0`，lambda=0；它不是EWC baseline。Ratio-FS的control虽然确实是native EWC，但lambda=1.2e6、gamma=1、damping=.2等不同，不能拼到上述GAM lambda2000行。

若优先保留“EWC+GAM”原意，应先在其他已有资产索引中补齐匹配EWC对照及冻结基底身份，再离线测量；未找到时如需训练是另一个明确缺分支的计划，不能将Ratio-FS当作已完成的GAM对照。

用户给定`task0_traj`、`fqa_r/lora/...`不是本机CopyFlatCV仓库根下现存相对目录，本轮未在该仓库文件清单找到这些原样路径。`scripts/run_imagenet_r_task1_fqa_lora_factor_adam_gam.sh`确实存在，但其调用的`run_imagenet_r_task1_fqa_adam_gam.sh`当前缺失。这个wrapper显式传`--factor-fqa`，其半径属于factor探测，不应凭脚本名直接解释成有效Q/V半径。ImageNet-A的`sgd_gam_s1c`模型不能改dataset参数后当ImageNet-R旧模型使用。

旧用户六行表精确来源是`/data/115-2/users/liying/low-rank-cl-new/.../s012_rho0p05_normrho0p5_lam1e7/`，18日志均`save_ckp=False`且之前限定搜索未找到对应权重。详见 [LORA_PROVENANCE.md](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/LORA_PROVENANCE.md)。其fresh K/V、Adam LR5e-4/WD.005/lambda1e7、projected行和这批持续Q/V新实现不同，不能替换端点。

## 3. 现成Ratio-FS pair：可以离线评估，作为另一明确命名的协议

原启动脚本 [run_verify_imagenet_r_1x1_lora_ratio_full_twostage.sh](/data/140-1/users/liying/CopyFlatCV/scripts/run_verify_imagenet_r_1x1_lora_ratio_full_twostage.sh:1)。tag `ratiofs_l12c20_full_twostage_s345_v1`，seeds3/4/5，ImageNet-R 20+20、rank10、SGD LR.02、50epoch/task、batch128、momentum/WD0、cosine；有保护行lambda1.2e6、gamma1、Task Fisher trace归一化、累计不归一化、batch-empirical代理、damping beta.2；Ratio-FS c.20/rho.05。

| 原setting | Task0 | Task1 | 应解释为 |
|---|---|---|---|
| control | native SGD | native SGD+EWC | EWC baseline |
| fs_ewc | Ratio-FS | Ratio-FS+EWC | 两阶段Ratio-FS+EWC |
| factored_s1 | Ratio-FS | native SGD+EWC | 仅初始阶段Ratio-FS |
| factored_s2 | native SGD | Ratio-FS+EWC | 仅后续阶段Ratio-FS |

三seed均有完整Task0 `task_00.pt`及其Fisher/A/B/FC sidecar；Task1通常只保存A/B/FC及metadata，没有`task_01.pt`或`final_model.pt`，不能因此误报模型缺失。本次再次检查control/fs_ewc/factored_s1的9对文件均存在，见 [ratio_pair_asset_check.json](ratio_pair_asset_check.json)。全部18个正式run的精确路径见 [new_lora_sources.json](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/new_lora_sources.json)。

离线恢复方式是保留该run Task0完整模型中的冻结base，再加载Task1对应A/B及40类FC，按metadata放到12层Q/V；不在线重新下载或信任可变预训练别名。随后先重评原accuracy，检查旧head不变、冻结base不变、参数键齐全、数据类映射一致。只有这些通过才运行昂贵EF/路径分析。Task1 sidecar未保存完整网络，恢复的冻结base与历史stage2初始化一致性仍需用原日志/pretrained来源和数值sanity约束；不能声称文件本身证明历史stage2全基底逐位相同。启动脚本自己标注证据级别是`operational_staged_comparison_not_strict_b4_full_state_restore`，不要升级成严格全状态、随机数逐位配对。

已有CPU权重核验仅覆盖control/factored_s1六对：正确位移为`||BA1-BA0||`，均值.427165612/.448335488；日志`mechanism.steps.1.delta_norm`是平方，最大误差7.58e-9；旧20行head逐位不动。fs_ewc平均r=.461659目前来自日志，尚未在该审查中逐tensor复算。训练本就持续同一adapter，Task1起点BA非零，不得改用`||BA1||`。

原始accuracy并未显示所有shaping分支都有更好保留：control BWT均值−3.4633，fs_ewc−3.72，factored_s1−3.51。新增EF评估必须如实解释，不以“敏感度必然更低”筛选模型。

## 4. 建议统一的新增离线测量定义

对选定并正确命名的同一ImageNet-R模型家族：固定Task0训练数据、确定性eval变换、共享分层256样本，旧20类head固定；定义未归一化逐样本经验Fisher

`F0(theta) = (1/N) sum_i grad_Well ell_i(theta) grad_Well ell_i(theta)^T`。

实际只需trace和方向二次型，无需物化全矩阵。主trace为`mean_i ||grad_Well ell_i||²`；严禁以`||grad mean_batch CE||²`、存储的normalized EWC buffer或历史当前任务Fisher代替。对新版LoRA `Well`用有效Q/V坐标；路径按两个有效Q/V端点线性插值，不能用A/B线性插值冒充相同路径。应把raw trace、local CE损伤、seen-head CE/accuracy分别注明；新head竞争引发class-IL下降不能都归为共享backbone损伤。

两任务最小离线检查是Task0与Task1端点各重测EF/旧任务loss、准确率和有效位移，确认r=0与端点恢复后再决定增加九节点路径或半径扫描。本轮尚未执行这些新测量。

## 5. Full表能否排除新头后复评

**旧full六行：final存在，可重新评估这些final的旧任务CE/Fisher，但不能完整恢复原始两端损伤、路径或精确backbone位移。** 对应旧Task0路径已被6月9日后来运行覆盖，且final仅有当前`model_state_dict`、不含历史参数快照。旧表drift是含新增头补零参考的平方和开根号。既有“新头占平方能量73%–90%”只是现存final对应原日志等身份条件下的重建；不能作为闭环权重差向量。详见 [FULL_PROVENANCE.md](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/FULL_PROVENANCE.md) 及 [FULL_HISTORY_RECOVERY.md](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/FULL_HISTORY_RECOVERY.md)。

**另一套ImageNet-R full SGD/GAM序列有完整匹配两端，可离线重评：**

```
/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c_fullmodel/
  stage1_task0/task0_fm_lam2000_lr0p01_e50/{sgd,gam}/seed0/checkpoints/task_00.pt
  stage2_T/T_from_task0_fm_lam2000_lr0p01_e50/{sgd,gam}/T10/seed0/checkpoints/task_01.pt
```

该套Task0→Task1纯backbone已CPU精确重算：SGD2.041041、GAM2.602965；旧20类head移动均0；新增头相对零范数4.469002/4.376826，不能称为初始化到最终的训练位移。全部18相邻步的CPU平方位移与日志最大误差4.79e-8，路径/Task0指标/Fisher/time/hash关联详见 [NEW_FULL_RESULTS.md](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/NEW_FULL_RESULTS.md)。

其训练为NCM batch-gradient Fisher、lambda2000、gamma.9、SGD/GAM LR.01，GAM rho.02/normrho.2/gamma.03；训练Fisher不是新计划的逐样本CE Fisher。可在同旧20类CE定义下新增raw EF离线诊断，但不能把训练Fisher缓冲直接改名。若比较新版LoRA和full，除了full全backbone，还应额外统一Q/V子空间；此前full复算表内K/V列只对应旧fresh-KV LoRA，不对应当前Ratio-FS Q/V。不同训练协议/预训练/算法仍需披露，不能由同一测量坐标推成严格参数化因果对照。
