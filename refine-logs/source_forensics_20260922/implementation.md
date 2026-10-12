# 表格方法调用链与有效代码差异审计（2026-09-22）

本报告只读源码与现有配置/日志，无训练、无模型下载、无源repo修改。区分 Flatness 当前、Copy HEAD、Copy 当前工作树、Sep18正式快照、Sep16 RNG控制快照。当前源码不等于历史运行源码；代码有diff本身也不意味着旧数值不可信。

## 结论

1. Flatness 当前未提交data.py仅增加可选ImageNet-R排除清单，与本表五组/Het5不相交。
2. Copy工作树相对Sep18正式快照的主要主动变化在共享LoRA训练forward：去掉临时随机Linear，改变CPU RNG消耗。Sep18 formal与Copy HEAD仍保留临时Linear；Sep16rngctl和Copy工作树去掉了它。独立eval类未改。
3. Copy/formal相对Flatness还有SeqLoRA无条件额外当前任务评估遍历，不能统称纯日志改动；其他大量新checkpoint/Fisher模式/诊断分支是默认关闭。
4. OLoRA/SDLoRA learner文件相同但共享backbone可能不同；InfLoRA在Copy和formal缺模块，只能从已核实Flatness分支解析。
5. JSON的clean-RNG/persistent-workers旗标并不总生效，下面按源码执行给出。

## 实际调用链

Flatness: [src.main](/data/140-1/users/liying/Flatness_CV/src/main.py:81) → [trainer.train](/data/140-1/users/liying/Flatness_CV/src/trainer.py:180) → seed setup/DataManager/factory → incremental_train → eval_task → after_task。
Copy/formal: [src.main](/data/140-1/users/liying/CopyFlatCV/src/main.py:116) → [trainer.train](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:822) → [DataManager](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:916) → [factory](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:938) → [训练和评估](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1116) → [after_task](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1224)。

| 表格名 | 实际factory/model | Learner/网络/优化器 |
|---|---|---|
| Vanilla LoRA | `seqlora` | `models_LoRAbasedCL.seqlora.Learner`；utils.inc_net.IncrementalNet → get_convnet / backbone.lora.LoRA_ViT_timm → _LoRA_qkv_timm_train；SGD for table |
| InfLoRA | `inflora` | `models_LoRAbasedCL.inflora.Learner`；utils.inc_net.SiNet → _build_inflora_backbone → LoRA_ViT_timm；SGD for table |
| OLoRA | `olora` | `models_LoRAbasedCL.olora.Learner`；IncrementalNet → per-task LoRA_ViT_timm(learn_alpha=False)；SGD |
| SDLoRA | `sdlora` | `models_LoRAbasedCL.sdlora.Learner`；IncrementalNet → per-task LoRA_ViT_timm(learn_alpha=True)；SGD |
| CL-LoRA | `cllora_aligned` | `models_LoRAbasedCL.cllora_aligned.Learner`；backbone.cllora_aligned.CLLoRANet → Encoder → backbone.cllora_archived_vit；torch.optim.SGD, KD step then CE step after task0 |
| EWCLoRA (current best row) | `as1_normfisher_fishershape` | `as1_normfisher_fishershape → as1_normfisher_gam → ewclora_youyue_fitArchitechture_gam → ewclora_youyue_fitArchitechture → seqlora`；IncrementalNet → LoRA_ViT_timm；FisherShape(alpha0, 2 CE probes) or native_sgd(1 CE pass) + explicit Fisher gradient |
| GAM-EWCLoRA | `ewclora_normfisher_gam` | `as1_normfisher_gam → ewclora_youyue_fitArchitechture_gam → ewclora_youyue_fitArchitechture → seqlora`；IncrementalNet → LoRA_ViT_timm；GAM(SGD base): one clean cache + four GAM CE passes, explicit Fisher add |

## 逐项影响分类

### flatness_uncommitted_data — inactive_for_requested_datasets

范围：Five fine-grained datasets and Het5。Flatness git diff utils/data.py contains only optional ImageNet-R dataset_exclude_manifest. It is scoped to iImageNetR and disabled absent path; does not modify sample lists, labels, preprocessing for this table.

证据：[data.py:708](/data/140-1/users/liying/Flatness_CV/utils/data.py:708)。

### flowers_mapping — same_current_membership

范围：Flowers and Flowers component of Het5。Flatness uses sorted directory index, Copy/formal uses canonical image_ID filename and missing-file check. Current 8189 files and 8189 frozen split entries give zero mapping mismatches. Does not establish historical image bytes or imply prior labels were wrong.

证据：[data.py:55](/data/140-1/users/liying/Flatness_CV/utils/data.py:55)；[data.py:61](/data/140-1/users/liying/CopyFlatCV/utils/data.py:61)。

### data_guards — validation_only_on_valid_data

范围：CUB/Cars/Het5。Copy/formal add class_to_idx equality guards and Het5 component label/count validation. Existing valid memberships unchanged; failures now explicit. Same Het5 order/offsets and transforms retained.

证据：[data.py:956](/data/140-1/users/liying/CopyFlatCV/utils/data.py:956)；[data.py:990](/data/140-1/users/liying/CopyFlatCV/utils/data.py:990)；[data.py:1286](/data/140-1/users/liying/CopyFlatCV/utils/data.py:1286)。

### data_root — potentially_active_default

范围：All dataset loading。Flatness DataManager injects data_root=/data/140-0/datasets if absent; Copy/formal remove this default. Explicit run data_root/environment and realpaths must be checked before claiming same inputs.

证据：[data_manager.py:20](/data/140-1/users/liying/Flatness_CV/utils/data_manager.py:20)；[data_manager.py:19](/data/140-1/users/liying/CopyFlatCV/utils/data_manager.py:19)。

### temporary_linear_versions — active_rng_side_effect

范围：Shared training LoRA backbone, all seven except CL-LoRA。Flatness, Copy HEAD and Sep18 formal train forward create four temporary random Linear modules unconditionally, even task_id=0. Copy working and Sep16 RNG-controlled code replace historical-factor application with F.linear and remove that CPU RNG consumption. Fixed weights and factors retain same intended forward algebra, but subsequent RNG trajectory is not guaranteed equal.

证据：[lora.py:174](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:174)；[lora.py:174](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/lora.py:174)；[lora.py:174](/data/140-1/users/liying/CopyFlatCV/backbone/lora.py:174)；[lora.py:174](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-rngctl-20260916/code/backbone/lora.py:174)。

### eval_class_unmodified — unchanged_scope

范围：Shared LoRA evaluation class。The independent _LoRA_qkv_timm_eval class is unchanged across all five versions and still creates temporary Linear modules. Do not describe working-tree train-forward change as fixing all evaluation RNG consumption.

证据：[lora.py:229](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:229)；[lora.py:233](/data/140-1/users/liying/CopyFlatCV/backbone/lora.py:233)。

### external_delta — inactive_default

范围：Shared training LoRA。Copy HEAD/formal vs Flatness add external_delta_w_q/v=None and conditional additions. Unset for ordinary methods; optional diagnostic/structured perturbation paths activate only when assigning tensors.

证据：[lora.py:166](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/lora.py:166)；[lora.py:216](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/lora.py:216)。

### extra_eval_rng — active_extra_traversal

范围：Vanilla SeqLoRA and inheriting EWC/GAM, not independent Inf/OLoRA/SDLoRA/CL learners。Copy HEAD/formal/working SeqLoRA add current-task local accuracy passes every five epochs and at task end versus Flatness. These are not behind optional flag. They traverse loaders and forward modules; with global RNG/default generator and temporary-Linears code they may shift subsequent sampling/augmentation RNG. For persistent workers iterator reuse differs, so do not claim identical base-seed redraw on every reused iteration.

证据：[seqlora.py:437](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:437)；[seqlora.py:1750](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:1750)；[seqlora.py:2152](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:2152)；[dataloader.py:601](/data/115-2/users/liying/conda_storage/envs/Pilot_new/lib/python3.11/site-packages/torch/utils/data/dataloader.py:601)。

### fisher_split — inactive_default

范围：SeqLoRA and inheriting EWC/GAM。Copy adds heldout Fisher split and class-order override support, but with absent/false split parameters and class_order_override absent, training set still uses all current-task examples. ewc_fisher_source default string heldout does not create heldout data: getter falls back to train_loader.

证据：[seqlora.py:215](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:215)；[seqlora.py:341](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:341)；[ewclora_youyue_fitArchitechture.py:135](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:135)；[data_manager.py:384](/data/140-1/users/liying/CopyFlatCV/utils/data_manager.py:384)。

### min_lr — same_default_different_api

范围：Baseline and GAM schedulers。Flatness Vanilla/Inf/OLoRA/SDLoRA default min_lr=.1*LR. Copy/formal Seq support new ratio/absolute overrides but same default .1*LR. CL directly calls CosineAnnealingLR without eta_min, so min LR is zero. Merely adding min_lr to CL config is ignored by current CL implementation.

证据：[seqlora.py:1341](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/seqlora.py:1341)；[inflora.py:390](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/inflora.py:390)；[olora.py:194](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/olora.py:194)；[sdlora.py:230](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/sdlora.py:230)；[seqlora.py:288](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:288)；[cllora_aligned.py:68](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/cllora_aligned.py:68)。

### worker_real_settings — active_dataset_method_difference

范围：Aircraft especially; all datasets loader route。Seq/EWC/GAM use DataManager.build_dataloader with workers=train_num_workers(default8). Aircraft loader_options set pin_memory=true,persistent_workers=true and DataManager does not read top-level persistent_workers; explicit false JSON alone is ineffective here. Inf/OLoRA/SDLoRA construct DataLoader directly with default pin=false,persistent=false. CL constructs directly workers2,pin=true,persistent=false. Het5 uses own empty loader_options.

证据：[data.py:110](/data/140-1/users/liying/CopyFlatCV/utils/data.py:110)；[data_manager.py:406](/data/140-1/users/liying/CopyFlatCV/utils/data_manager.py:406)；[seqlora.py:398](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:398)；[inflora.py:269](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/inflora.py:269)；[olora.py:112](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/olora.py:112)；[sdlora.py:122](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/sdlora.py:122)；[cllora_aligned.py:41](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/cllora_aligned.py:41)。

### rng_flags — effective_vs_ignored

范围：EWC/GAM and DataManager。ewc_preserve_fisher_rng read and implemented in Copy/formal, default false. ewc_isolate_clean_rng not read anywhere in Copy working/HEAD/formal GAM learner; Het5 true config therefore ineffective. deterministic_loader_rng/loader_seed not implemented in Copy/formal DataManager. Sep16 rngctl has real implementations for clean isolation and per-loader generators, so differs in actual protocol.

证据：[ewclora_youyue_fitArchitechture.py:89](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:89)；[ewclora_youyue_fitArchitechture.py:338](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:338)；[ewclora_youyue_fitArchitechture_gam.py:494](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py:494)；[data_manager.py:417](/data/140-1/users/liying/CopyFlatCV/utils/data_manager.py:417)；[ewclora_youyue_fitArchitechture_gam.py:39](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-rngctl-20260916/code/models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py:39)；[data_manager.py:428](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-rngctl-20260916/code/utils/data_manager.py:428)。

### seeding_equal — unchanged

范围：Python NumPy Torch seeding。utils/random_reproduce.py and toolkit.py byte-identical across Flatness, Copy, formal, rngctl. set_random seeds Python, NumPy, Torch CPU/CUDA and configures deterministic cuDNN; trainer calls it before DataManager/model creation in both repos. Extra reseeding sites are under disabled linear-probe workflows. Different RNG consumption, not a different initial seed function, is the concern.

证据：[random_reproduce.py:24](/data/140-1/users/liying/Flatness_CV/utils/random_reproduce.py:24)；[trainer.py:268](/data/140-1/users/liying/Flatness_CV/src/trainer.py:268)；[trainer.py:910](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:910)。

### trainer_changes — inactive_for_table

范围：Training entrypoint/trainer。Copy main adds local tmpdir and multiprocessing-cleanup handling. Trainer adds opt-in pretrained checkpoint loading, resumed-only evaluation, task0 diagnostics, epoch dynamics, staged checkpoints. Default off / no path in table configs. The normal incremental_train→eval_task→write_metrics→after_task ordering is retained. File diff alone is not evidence of different training updates.

证据：[main.py:14](/data/140-1/users/liying/CopyFlatCV/src/main.py:14)；[trainer.py:976](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:976)；[trainer.py:1116](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1116)；[trainer.py:1224](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1224)；[trainer.py:1263](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1263)。

### formal_current_new_branches — inactive_for_table

范围：Copy working vs Sep18 formal。Aside from shared backbone train-forward RNG change, targeted active files differ only in factory/inc_net KFAC/complementarity registrations and baseLoRA extra Fisher checkpoint save/restore hooks. These new branches do not change selected method mapping, and hooks are not called when staged save/resume absent.

证据：[factory.py:195](/data/140-1/users/liying/CopyFlatCV/utils/factory.py:195)；[inc_net.py:31](/data/140-1/users/liying/CopyFlatCV/utils/inc_net.py:31)；[baseLoRA.py:338](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/baseLoRA.py:338)；[baseLoRA.py:428](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/baseLoRA.py:428)。

### gam_extended_math — default_core_retained

范围：GAM-EWC。Copy GAM optimizer saves g3 clone and introduces use_sgv gate(defaultfalse); core projected direction unchanged with gate off. AS1 learner adds selectable Fisher modes, accumulated normalization and MC Fisher; default batch_empirical plus normalize_accumulated=false preserves intended batch-mean-gradient-square Fisher trace-normalization formula. Zero-lambda fast path irrelevant to GAM lambda2000 table row. Avoid claiming bitwise historical parity: code and RNG traversal differ.

证据：[gam.py:128](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/gam.py:128)；[gam.py:158](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/gam.py:158)；[as1_normfisher_gam.py:129](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:129)；[as1_normfisher_gam.py:381](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:381)。

### missing_inflora — cannot_launch_from_copy_as_is

范围：InfLoRA。Copy working, Copy HEAD and Sep18 formal factory retain import models_LoRAbasedCL.inflora but module file is missing. Flatness has full InfLoRA implementation. Need explicit source resolution/port and frozen import proof for future Copy campaign; this does not invalidate prior Flatness result.

证据：[factory.py:371](/data/140-1/users/liying/CopyFlatCV/utils/factory.py:371)；[factory.py:182](/data/140-1/users/liying/Flatness_CV/utils/factory.py:182)；[inflora.py:123](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/inflora.py:123)。

### ewc_label_not_factory — method_name_mapping

范围：Current selected EWC row。Current EWC table sources use model_name=as1_normfisher_fishershape. CUB/Aircraft/Cars/Flowers are alpha0 two-probe wrapper; Het5 fs_direction_mode=native_sgd uses one CE gradient. They are not factory("ewclora")→ewclora_origin and cannot blindly switch to that label. fs_preserve_probe_rng is a real implemented flag distinct from ignored GAM clean-isolation flag.

证据：[factory.py:188](/data/140-1/users/liying/CopyFlatCV/utils/factory.py:188)；[as1_normfisher_fishershape.py:87](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py:87)；[as1_normfisher_fishershape.py:377](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py:377)；[fishershape.py:287](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/fishershape.py:287)。

## 基础设置的有效值

| 项目 | 有效读取与结果 |
|---|---|
| Python/NumPy/Torch seed | 四代码树random_reproduce.py SHA相同9070ff870b88…，toolkit.py相同21ade1342f8a…；初始种子函数无差别 |
| Vanilla/Inf/OLoRA/SDLoRA minLR | 默认0.1×LR；Copy Seq增加ratio/absolute入口但不改默认 |
| CL minLR | 硬编码调度默认0，现有min_lr配置不会被读取 |
| workers | 前四learner train_num_workers默认8，CL默认2；正式CL和GAM-Het5固定2 |
| Aircraft pin/persistent | Seq/EWC/GAM经DataManager为true/true；Inf/OLoRA/SDLoRA直接loader默认false/false；CL为true/false |
| 其他数据/Het5 pin/persistent | DataManager空loader_options默认false/false；CL为true/false |
| GAM ewc_preserve_fisher_rng | Copy/formal真实实现，默认false；Het5显式true |
| GAM ewc_isolate_clean_rng | Copy当前/HEAD/formal未实现，Het5 JSON写true也无效；Sep16rngctl实现 |
| deterministic_loader_rng / loader_seed | Copy/formal DataManager未实现；Sep16rngctl实现；CL绕开DataManager，须实际改loader才可控制 |
| EWC fs_preserve_probe_rng | FisherShape优化器真实读取，恢复clean/probe Torch RNG；区别于GAM clean flag |

Flowers现有8189个目录项及正式split8189条一一对应，原sortedlist路径与新canonical路径0不一致，详见flowers_mapping_check.json。不能将CL Flowers高分归因于无证据的历史错标。

用于新增实验时，必须冻结完整依赖图，不能只复制learner.py；要想同seed行为一致，必须固定训练forward、评估遍历次数、loader generator/持久worker与实际读取的RNG设置。是否重跑应由具体活动差异与历史绑定证据决定，不能因为repo整体有diff而一律否定旧结果。
