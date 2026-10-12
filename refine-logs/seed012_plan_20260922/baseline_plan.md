# Baseline seed 0/1/2 补齐方案：只读核查（2026-09-22）

范围：Vanilla、InfLoRA、OLoRA、SDLoRA × Aircraft、Cars、CUB、Flowers、5Datasets，共20组；未启动训练，未改原始配置/日志。5Datasets仍包含Pets的37类。

## 决策与历史seed0是否可复用

当前没有任何一组具备“直接补seed1/2并与旧seed0合并”的完整证据。20组当前都只有seed0。历史March2026的训练源码/依赖、checkpoint SHA、逐图片清单、min_lr与loader/RNG字段没有与运行绑定；同配置名不足以认证重复实验。

- 若能恢复原始不可变代码、环境、权重、数据和已解析默认值，可仅补seed1/2，共40次正式运行。
- 目前建议在新的完整冻结实现上，对20组先做seed0诊断，再以完全相同非seed配置跑seed1/2，共60次；最终均值和SD使用新campaign的0/1/2。不能将新seed0只作为试跑后丢弃，转而使用旧表seed0。
- seed0结果近似或相同只能提高复现信心，不能单凭最终分数证明历史代码身份；不得调参追旧测试分数。

## 入口与代码边界

当前四个方法完整可读入口在 `/data/140-1/users/liying/Flatness_CV/src/main.py`，支持 `python -m src.main --config ... --override ...`。factory映射分别指向models_LoRAbasedCL/{seqlora,inflora,olora,sdlora}.py。

**不能直接套CopyFlatCV/9月18日CL gapfill代码：该两份代码缺失 inflora.py。** OLoRA/SD learner与Flatness相同，但共享backbone/lora.py、baseLoRA.py、baseLearner.py、inc_net.py、data.py、data_manager.py、trainer.py均不同；Vanilla learner也不同。应冻结完整Flatness依赖闭包或经过显式移植后另开campaign。现有SHA对照保存在JSON。

候选环境是已在9月18日见证的Pilot_new，但那不是March历史环境认证。本次仅静态检查入口，未运行import/GPU smoke；准备阶段还需固定torch/timm/torchvision等版本、CUDA/TF32/deterministic设置。复制trainer时必须包含其顶层import的evaluation系列包。

## 公共配置和数据集差异

四方法共同：ViT-B/16，timm标识augreg2_in21k_ft_in1k，rank16，batch128，SGD、momentum0、weight decay0，cosine，class_shuffle=false、all_or_inc=inc，无replay，首任务/后续LR与epochs相同。当前代码cosine下限为本阶段LR的10%，每任务重置、每epoch末step；历史未记录此下限，建议新campaign显式冻结。当前train/test loader都读取train_num_workers，默认8；Aircraft pin/persistent=true，其余默认false；这些要验证实际kwargs，不应仅加不消费的配置键。

| Dataset | Tasks/classes | LR: Vanilla/Inf/OLoRA | LR: SD | Epochs V/I/O/S | New explicit min_lr V/I/O; SD |
|---|---|---:|---:|---|---|
| Aircraft | 10×10 | .05 | .02 | 40/40/40/40 | .005; .002 |
| Cars | 16+9×20 | .05 | .05 | 40/40/40/40 | .005; .005 |
| CUB | 10×20 | .01 | .01 | 40/40/40/40 | .001; .001 |
| Flowers | 12+9×10 | .0025 | .0025 | 40/40/40/40 | .00025; .00025 |
| 5Datasets | 200→100→196→102→37 | .02 | .02 | 30/40/40/40 | .002; .002 |

Het5顺序CUB→Aircraft→Cars→Flowers→Pets；源YAML increment=1被dataset.task_splits覆盖，不能按1类/任务执行。Cars SD文件名含lr0025，但实际日志/YAML为.05。保留这些已记录差异是复现现表；若要统一LR/epochs/eta_min，应另标“harmonized protocol”，全部重跑，不与当前值拼接。

CUB/Flowers使用resize256×256→crop224+ImageNet normalization；Aircraft同几何但tensor loader；Cars单数据集短边resize224→center crop224、mean0/std1；Het5所有成分统一CUB变换。固定class-order的seed变化主要覆盖初始化、训练shuffle和augmentation，不覆盖任务顺序。新data_root候选/data/115-2/users/liying/datasets需清单冻结；不可把CUB当前9430/2358划分写成标准官方划分。

除方法专属配置外仍应冻结其默认值：Inf lamb=.7/lame=.99；OLoRA olora_lamda_1=1/olora_lamda_2=0；SD scalar重置等由源码决定。它们当前默认并非历史已记录值。保留NME原型计算等辅助pass，移除可能改变RNG轨迹；主表始终抽CNN FAA/AAA_stage。

## 每组配置来源和历史耗时

以下耗时为已选日志最后一次运行从首个时间戳到最后时间戳，包含初始化/评估/辅助阶段。没有同机硬件、并发占用记录，不能当GPU小时或精确新运行预算。

| Method | Dataset | Source YAML | Observed minutes |
|---|---|---|---:|
| Vanilla LoRA | aircraft | [seqlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_sgd_t20c10_r16.yaml) | 77.0 |
| Vanilla LoRA | cars196 | [seqlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/seqlora_inr_sgd_t20c10_r16.yaml) | 103.4 |
| Vanilla LoRA | cub200 | [seqlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/seqlora_inr_sgd_t20c10_r16.yaml) | 58.8 |
| Vanilla LoRA | flowers | [seqlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/seqlora_inr_sgd_t20c10_r16.yaml) | 40.9 |
| OLoRA | aircraft | [olora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_sgd_t20c10_r16.yaml) | 84.2 |
| OLoRA | cars196 | [olora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/olora_inr_sgd_t20c10_r16.yaml) | 69.8 |
| OLoRA | cub200 | [olora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/olora_inr_sgd_t20c10_r16.yaml) | 65.7 |
| OLoRA | flowers | [olora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/olora_inr_sgd_t20c10_r16.yaml) | 47.8 |
| SDLoRA | aircraft | [sdlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/sdlora_inr_sgd_t20c10_r16.yaml) | 60.9 |
| SDLoRA | cars196 | [sdlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/sdlora_inr_sgd_t20c10_r16.yaml) | 71.3 |
| SDLoRA | cub200 | [sdlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/sdlora_inr_sgd_t20c10_r16.yaml) | 99.7 |
| SDLoRA | flowers | [sdlora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/sdlora_inr_sgd_t20c10_r16.yaml) | 39.2 |
| InfLoRA | aircraft | [inflora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_sgd_t20c10_r16.yaml) | 95.4 |
| InfLoRA | cars196 | [inflora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/inflora_inr_sgd_t20c10_r16.yaml) | 110.7 |
| InfLoRA | cub200 | [inflora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/inflora_inr_sgd_t20c10_r16.yaml) | 81.1 |
| InfLoRA | flowers | [inflora_inr_sgd_t20c10_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/inflora_inr_sgd_t20c10_r16.yaml) | 48.7 |
| Vanilla LoRA | het5 | [seqlora_inr_sgd_het5_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/seqlora_inr_sgd_het5_r16.yaml) | 221.1 |
| OLoRA | het5 | [olora_inr_sgd_het5_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/olora_inr_sgd_het5_r16.yaml) | 154.9 |
| SDLoRA | het5 | [sdlora_inr_sgd_het5_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/sdlora_inr_sgd_het5_r16.yaml) | 309.7 |
| InfLoRA | het5 | [inflora_inr_sgd_het5_r16.yaml](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/inflora_inr_sgd_het5_r16.yaml) | 158.6 |

20次历史区间合计33.31小时；40次只补1/2线性参考66.63小时；60次新campaign线性参考99.94小时。实际排程需当前smoke计时与可用GPU，以上不是预测。

## 冻结与验证顺序

1. 完整源码依赖SHA +环境manifest +checkpoint revision/SHA +逐样本train/test/label清单；为20组生成独立seed配置，并固定output路径。
2. CPU import/config/data检查；验证真实任务边界和完整样本映射。只有取得训练授权后才做GPU smoke。
3. 20组新seed0诊断；保留所有差异，定位环境/数据/实现原因，不依测试分数调参。
4. 同一冻结campaign seed1/2；非seed字段SHA一致、独立输出，既定配置不变。
5. 每run CNN矩阵完整finite后计算FAA/AAA_stage，再在0/1/2之间算mean和sample SD(ddof=1)。旧值留作历史对照。

`baseline_plan.json`包含20份源配置/日志/指标SHA、当前代码差异、结构化建议配置和时间证据。结构化draft包含占位output路径，不是可直接执行的配置；未保存或启动launcher。
