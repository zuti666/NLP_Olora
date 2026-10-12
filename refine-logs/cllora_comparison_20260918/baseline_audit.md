# 四个历史基线独立复核（2026-09-18）

只读核验24项原始metrics、log、现存原配置，未训练或修改CopyFlatCV。24/24的metrics SHA、log SHA、config SHA与此前归档完全相符；全部任务矩阵下三角有限、任务数匹配，FAA/AAA_stage复算与已选值一致。

## 共同比较协议与例外

- 均为seed0、ViT-B/16、rank16、batch128、SGD(momentum=0, weight_decay=0)、cosine、无回放、class_shuffle=false。24条日志均有同一checkpoint hub ID：`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`。
- 五个单数据集均为40 epochs/task。Het5 Vanilla LoRA为30，InfLoRA/OLoRA/SDLoRA为40；因此Het5不能标成统一40轮。
- 学习率CUB=.01、Cars/Aircraft=.05、Flowers/Pets=.0025、Het5=.02；SDLoRA/Aircraft例外=.02。
- 任务边界经最后任务日志核对：CUB=10×20；Cars=16+9×20；Aircraft=10×10；Flowers=12+9×10；Pets=5+8×4；Het5=200/100/196/102/37。旧Het5 config的increment=1由显式dataset task_splits覆盖，不能据此判为每任务1类。
- 这些是同一数据集/训练骨架下的方法比较，不能宣称完全等参数、等计算或严格组件消融。

## 数据、代码和权重边界

- 历史只有checkpoint hub ID，未锁定实际权重revision/SHA；新CL-LoRA已锁定revision和文件SHA。相同hub ID不能证明逐bit相同。
- 历史split/label manifest缺失，旧140-0 data根已不可用，当前115-2副本不能证明逐图片相同。旧Flowers loader按排序文件索引，当前版本按image ID；不能用当前代码反推历史映射正确。
- 旧配置/YAML与日志可追溯，但历史运行未绑定完整不可变源码receipt；当前代码已包含后续改动。下面结构和预算分析标为“现存源码支持”，非历史运行的严格恢复。
- 表中选择CNN任务矩阵：FAA为最后一行任务宏平均；AAA_stage先对每个阶段的已见任务求均值，再对阶段均值求平均。原JSON cnn.final.AAA为下三角元素直接均值，不能混用。NME或sample-pooled总准确率也不能换入同列。

## 可增加的结果字段

下表FAA/AAA为原始矩阵复算。BWT_final为最后阶段相对任务首次学成准确率的变化（仅前T−1任务）；F_avg为每个旧任务历史最高准确率减最终准确率后平均。单位均为百分点；F_avg越低越好。这些可丰富附录而不需要新实验。

|方法|数据集|FAA|AAA_stage|BWT_final|F_avg|epoch/task|LR|
|---|---|---:|---:|---:|---:|---:|---:|
|Vanilla LoRA|cub200|57.793|70.002|-7.946|8.293|40|0.01|
|Vanilla LoRA|cars196|28.520|39.415|-15.883|15.910|40|0.05|
|Vanilla LoRA|aircraft|40.660|39.512|-17.232|17.232|40|0.05|
|Vanilla LoRA|flowers|53.052|66.905|-10.723|10.732|40|0.0025|
|Vanilla LoRA|pets|64.121|72.833|-13.276|13.308|40|0.0025|
|Vanilla LoRA|het5|66.804|68.806|-17.955|17.955|30|0.02|
|InfLoRA|cub200|60.608|70.601|-8.239|8.532|40|0.01|
|InfLoRA|cars196|34.345|42.201|-17.170|17.170|40|0.05|
|InfLoRA|aircraft|40.958|41.228|-19.236|19.236|40|0.05|
|InfLoRA|flowers|60.060|72.980|-9.404|9.452|40|0.0025|
|InfLoRA|pets|62.179|70.398|-12.068|12.099|40|0.0025|
|InfLoRA|het5|77.748|77.016|-7.623|7.623|40|0.02|
|OLoRA|cub200|62.376|70.167|0.309|6.104|40|0.01|
|OLoRA|cars196|32.855|42.298|-8.198|8.347|40|0.05|
|OLoRA|aircraft|36.098|37.806|-17.774|17.873|40|0.05|
|OLoRA|flowers|48.898|64.258|-8.777|8.851|40|0.0025|
|OLoRA|pets|59.516|68.012|-15.210|15.241|40|0.0025|
|OLoRA|het5|77.580|74.523|-4.975|4.993|40|0.02|
|SDLoRA|cub200|62.006|69.525|-5.418|5.651|40|0.01|
|SDLoRA|cars196|34.891|43.276|-11.607|11.607|40|0.05|
|SDLoRA|aircraft|25.959|26.633|-16.837|16.937|40|0.02|
|SDLoRA|flowers|49.047|64.588|-8.704|8.817|40|0.0025|
|SDLoRA|pets|59.448|68.076|-15.349|15.349|40|0.0025|
|SDLoRA|het5|74.420|74.099|-11.307|11.307|40|0.02|

三位小数保留原始矩阵信息，避免如InfLoRA/Cars=34.345与二位小数表的舍入边界被误判为不同来源。

## LoRA层、头与预算

现存框架源码显示四方法都在12个ViT block的Q/V投影上接LoRA，分类采用随累计类数扩展的SimpleLinear头（768维特征加bias，即每类769参数）。InfLoRA使用SiNet接口，仍是单个扩展线性头。CL-LoRA另用原型分类器，因此共同rank不等于相同分类器。

|方法|当前源码机制|当前可训练adapter参数（不含分类头）|历史adapter存储/计算|
|---|---|---:|---|
|Vanilla LoRA|持续训练同一组A/B|589824|固定rank adapter|
|InfLoRA|协方差/SVD/DualGPM构建并冻结A，只训练B|294912|逐任务adapter、信息子空间；参数/状态额外增长|
|OLoRA|旧adapter冻结，新A/B训练并加正交约束|589824|旧adapter逐任务保留|
|SDLoRA|旧adapter冻结，新A/B与缩放参数训练|589824+201（当前实现注册数）|旧adapter与缩放参数保留|

计算：12层×Q/V两投影×A/B两因子×768×16=589824。InfLoRA只训练B，数目减半。SDLoRA当前实现注册1个当前和200个历史缩放scalar；注册为requires_grad不等于每步都有有效梯度。以上是现存源码推导，不是历史最终checkpoint实测预算。

**日志打印陷阱：** trainer在每个任务调用incremental_train之前打印Trainable params，彼时头仍是上一个阶段宽度，InfLoRA第一阶段甚至尚未冻结A。故下表仅为最后一次日志观测，不应标“最终训练参数”。All params约172M还含冻结base_vit副本；用Python字典保管的历史adapter也未必被nn.Module.parameters完整计数，不能充当完整推理/存储预算。

|方法|数据集|最后一次日志Trainable params（头扩展前）|
|---|---|---:|
|Vanilla LoRA|cub200|728244|
|Vanilla LoRA|cars196|725168|
|Vanilla LoRA|aircraft|659034|
|Vanilla LoRA|flowers|660572|
|Vanilla LoRA|pets|615201|
|Vanilla LoRA|het5|1049686|
|InfLoRA|cub200|433332|
|InfLoRA|cars196|430256|
|InfLoRA|aircraft|364122|
|InfLoRA|flowers|365660|
|InfLoRA|pets|320289|
|InfLoRA|het5|754774|
|OLoRA|cub200|728244|
|OLoRA|cars196|725168|
|OLoRA|aircraft|659034|
|OLoRA|flowers|660572|
|OLoRA|pets|615201|
|OLoRA|het5|1049686|
|SDLoRA|cub200|728445|
|SDLoRA|cars196|725369|
|SDLoRA|aircraft|659235|
|SDLoRA|flowers|660773|
|SDLoRA|pets|615402|
|SDLoRA|het5|1049887|

关键源码证据（当前版本，不代表历史SHA已锁定）：

- [trainer计数位置](/data/140-1/users/liying/Flatness_CV/src/trainer.py:385)：计数早于incremental_train。
- [LoRA_ViT_timm](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:448)：默认所有block、冻结基础权重、Q/V A/B、base_vit拷贝、保存旧adapter、SD缩放scalar。
- [IncrementalNet head](/data/140-1/users/liying/Flatness_CV/utils/inc_net.py:389)、[InfLoRA SiNet head](/data/140-1/users/liying/Flatness_CV/utils/inc_net.py:1418)。
- [InfLoRA冻结A并开放B](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/inflora.py:324)。
- [OLoRA正交项](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/olora.py:397)、[SDLoRA逐任务backbone](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/sdlora.py:208)。

## 独立多seed补充（不得拼入当前主表）

low-rank-cl-new已有InfLoRA/SDLoRA在CUB/Pets的seed0/1/2。其rank10、batch64、20 epochs、Adam=.0005、预训练标识vit_base_patch16_224_in21k、Pets 4任务，与当前rank16/SGD/40e/Pets9任务体系不同。下列旧归档summary仅作为独立协议补充，不与本表seed0平均。

|方法|数据集|n|FAA mean±sample SD|AAA_stage mean±sample SD|
|---|---|---:|---:|---:|
|inflora|cub200|3|78.717±0.946|83.441±0.974|
|inflora|pets|3|83.488±1.372|88.308±1.270|
|sdlora|cub200|3|78.351±0.603|83.608±1.212|
|sdlora|pets|3|76.560±2.592|84.062±1.428|

该独立工程Flowers的Inf/SD记录虽有三seed完整日志，但已有classifier-head offset核验失败标志；不能因有均值就采纳。Vanilla/OLoRA本次指定24项来源及既有多seedsummary中未找到同协议多seed聚合，不据此断言全磁盘绝无其他运行。

[独立协议summary](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/baseline_multiseed_20260918/summary.json)、[Flowers head-offset核验](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/baseline_multiseed_20260918/flowers_head_offset_check.json)。

## 24项原始来源与现存配置

[完整审计CSV含SHA与日志/配置字段](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/protocol_alignment_20260918/legacy_source_config_audit.csv)；[历史源码/配置/日志归档清单](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/protocol_alignment_20260918/source_manifest.json)。

|方法|数据集|原始矩阵|原始log|已迁移原配置|
|---|---|---|---|---|
|Vanilla LoRA|cub200|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/cub200/0/seqlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/seqlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/cub200/0/seqlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/seqlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/seqlora_inr_sgd_t20c10_r16.yaml)|
|Vanilla LoRA|cars196|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/cars196/0/seqlora_inr_sgd_cars196_t20_rank16_lr005-E40_train-eval-paer/exp_run/20/seqlora_inr_sgd_cars196_t20_rank16_lr005-E40_train-eval-paer_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/cars196/0/seqlora_inr_sgd_cars196_t20_rank16_lr005-E40_train-eval-paer/exp_run/20/seqlora_inr_sgd_cars196_t20_rank16_lr005-E40_train-eval-paer_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/seqlora_inr_sgd_t20c10_r16.yaml)|
|Vanilla LoRA|aircraft|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/aircraft/0/seqlora_inr_sgd_aircraft_t20_rank16_lr005-E40_epoch40/exp_run/10/seqlora_inr_sgd_aircraft_t20_rank16_lr005-E40_epoch40_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/aircraft/0/seqlora_inr_sgd_aircraft_t20_rank16_lr005-E40_epoch40/exp_run/10/seqlora_inr_sgd_aircraft_t20_rank16_lr005-E40_epoch40_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_sgd_t20c10_r16.yaml)|
|Vanilla LoRA|flowers|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/flowers/0/seqlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/seqlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/flowers/0/seqlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/seqlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/seqlora_inr_sgd_t20c10_r16.yaml)|
|Vanilla LoRA|pets|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/pets/0/seqlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/seqlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/pets/0/seqlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/seqlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/seqlora_inr_sgd_t20c10_r16.yaml)|
|Vanilla LoRA|het5|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/het5/0/seqlora_inr_sgd_het5_t5_rank16_lr002_epoch30/exp_run/1/seqlora_inr_sgd_het5_t5_rank16_lr002_epoch30_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/het5/0/seqlora_inr_sgd_het5_t5_rank16_lr002_epoch30/exp_run/1/seqlora_inr_sgd_het5_t5_rank16_lr002_epoch30_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/seqlora_inr_sgd_het5_r16.yaml)|
|InfLoRA|cub200|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cub200/0/inflora_inr_sgd_cub200_t20-E40_rank16/exp_run/20/inflora_inr_sgd_cub200_t20-E40_rank16_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cub200/0/inflora_inr_sgd_cub200_t20-E40_rank16/exp_run/20/inflora_inr_sgd_cub200_t20-E40_rank16_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/inflora_inr_sgd_t20c10_r16.yaml)|
|InfLoRA|cars196|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cars196/0/inflora_inr_sgd_cars196_t20-E40_rank16_lr005/exp_run/20/inflora_inr_sgd_cars196_t20-E40_rank16_lr005_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cars196/0/inflora_inr_sgd_cars196_t20-E40_rank16_lr005/exp_run/20/inflora_inr_sgd_cars196_t20-E40_rank16_lr005_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/inflora_inr_sgd_t20c10_r16.yaml)|
|InfLoRA|aircraft|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/aircraft/0/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40/exp_run/10/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/aircraft/0/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40/exp_run/10/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_sgd_t20c10_r16.yaml)|
|InfLoRA|flowers|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/flowers/0/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16/exp_run/10/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/flowers/0/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16/exp_run/10/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/inflora_inr_sgd_t20c10_r16.yaml)|
|InfLoRA|pets|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/pets/0/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16/exp_run/4/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/pets/0/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16/exp_run/4/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/inflora_inr_sgd_t20c10_r16.yaml)|
|InfLoRA|het5|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/het5/0/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/het5/0/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/inflora_inr_sgd_het5_r16.yaml)|
|OLoRA|cub200|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/cub200/0/olora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/olora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/cub200/0/olora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/olora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/olora_inr_sgd_t20c10_r16.yaml)|
|OLoRA|cars196|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/cars196/0/olora_inr_sgd_cars196_t20-E40_rank16_lr005_train-eval-paer/exp_run/20/olora_inr_sgd_cars196_t20-E40_rank16_lr005_train-eval-paer_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/cars196/0/olora_inr_sgd_cars196_t20-E40_rank16_lr005_train-eval-paer/exp_run/20/olora_inr_sgd_cars196_t20-E40_rank16_lr005_train-eval-paer_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/olora_inr_sgd_t20c10_r16.yaml)|
|OLoRA|aircraft|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/aircraft/0/olora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40/exp_run/10/olora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/aircraft/0/olora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40/exp_run/10/olora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_sgd_t20c10_r16.yaml)|
|OLoRA|flowers|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/flowers/0/olora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/olora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/flowers/0/olora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/olora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/olora_inr_sgd_t20c10_r16.yaml)|
|OLoRA|pets|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/pets/0/olora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/olora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/pets/0/olora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/olora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/olora_inr_sgd_t20c10_r16.yaml)|
|OLoRA|het5|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/het5/0/olora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/olora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/olora/sgd/het5/0/olora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/olora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/olora_inr_sgd_het5_r16.yaml)|
|SDLoRA|cub200|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/cub200/0/sdlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/sdlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/cub200/0/sdlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer/exp_run/20/sdlora_inr_sgd_cub200_t20-E40_rank16_train-eval-paer_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/sdlora_inr_sgd_t20c10_r16.yaml)|
|SDLoRA|cars196|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/cars196/0/sdlora_inr_sgd_cars196_t20_rank16_lr0025-E40_train-eval-paer/exp_run/20/sdlora_inr_sgd_cars196_t20_rank16_lr0025-E40_train-eval-paer_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/cars196/0/sdlora_inr_sgd_cars196_t20_rank16_lr0025-E40_train-eval-paer/exp_run/20/sdlora_inr_sgd_cars196_t20_rank16_lr0025-E40_train-eval-paer_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/sdlora_inr_sgd_t20c10_r16.yaml)|
|SDLoRA|aircraft|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/aircraft/0/sdlora_inr_sgd_aircraft_t20_rank16_lr002-E40_epoch40/exp_run/10/sdlora_inr_sgd_aircraft_t20_rank16_lr002-E40_epoch40_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/aircraft/0/sdlora_inr_sgd_aircraft_t20_rank16_lr002-E40_epoch40/exp_run/10/sdlora_inr_sgd_aircraft_t20_rank16_lr002-E40_epoch40_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/sdlora_inr_sgd_t20c10_r16.yaml)|
|SDLoRA|flowers|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/flowers/0/sdlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/sdlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/flowers/0/sdlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/sdlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/sdlora_inr_sgd_t20c10_r16.yaml)|
|SDLoRA|pets|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/pets/0/sdlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/sdlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/pets/0/sdlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2/exp_run/4/sdlora_inr_sgd_pets_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/sdlora_inr_sgd_t20c10_r16.yaml)|
|SDLoRA|het5|[JSON](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/het5/0/sdlora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/sdlora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224_cl_metrics.json)|[log](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/sdlora/sgd/het5/0/sdlora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/sdlora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224.log)|[YAML](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/sdlora_inr_sgd_het5_r16.yaml)|
