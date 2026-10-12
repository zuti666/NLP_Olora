# NLP 多 seed 补充实验：只读执行设计

日期：2026-09-22。范围：Llama-3.2-1B / 3B，SeqLoRA、IncLoRA、OLoRA+GAM、TreeLoRA、EWCLoRA、FS-EWCLoRA，三个 Order，seed 0 / 42 / 1993。本子任务未启动训练，未修改仓库或旧结果；机器可读参数见 `/tmp/nlp_multiseed_execution_design.json`，36 个历史 order 的逐阶段 runtime 见 `/tmp/nlp_multiseed_runtimes.json`。

## 推荐补充实验范围

推荐同一个 method-model 的三个 seed 均采用 **DDP4 × 每卡 batch1 × GA16 = global batch64**。现有严格四卡、完整四阶段结果只有 12 个 order 可以按参数继续复用：TreeLoRA 两模型 seed42 的 6 个 order，1B EWCLoRA seed42 的 3 个 order，1B FS-EWCLoRA seed1993 的 3 个 order。其余历史主表 24 个 order 都是 DDP8/GA8；可保留为原主表与历史参照，不能把它们与新 DDP4/GA16 的其他 seed 混合后宣称为同执行协议的 seed 方差。

因此严格版目标 108 个 order-run（2 models × 6 methods × 3 orders × 3 seeds），直接复用 12 个，补跑 **96 个完整 order / 384 个阶段**，其中 72 个为缺失 seed，24 个为已有 seed 的 DDP8→DDP4 迁移。最小补缺版只补 72 个，但会把 GPU/GA 变化混入多 seed 方差，不建议作为最终三 seed 主表。

这里的“复用”指数字、证据、路径进入新索引，不改写旧目录；新补跑的每个模型/方法/seed/order 均从同一个原始 base checkpoint 开始完整四任务链，不从其他 seed 的适配器开始。

## 物理 GPU 约束

只允许物理 GPU **3,4,5,6**；每次一个四进程 torchrun，各个 order 顺序排队。根据 parent 的只读 NVML 查询，GPU3 当前被 PID969996 占用约12354MiB / utilization100%；GPU4/5/6 各2MiB、util0%。不能立即四卡开跑，不终止他人进程，也不能自动替换成GPU0/1/2。

执行时先把当前 nvidia-smi index3/4/5/6解析为UUID，再用四个UUID写 `CUDA_VISIBLE_DEVICES`；同时记录index/UUID/PCI_BUS_ID映射。训练进程内local_rank0/1/2/3对应这四块实体卡。等待全部空闲并获取GPU锁后才能开始；恢复队列仍保持同样的allowlist。旧Tree启动器显式拒绝非`--gpus 0,1,2,3`，新计划必须新建副本，不能直接调用它原样执行本请求。

## 路径与环境

- Python：`/data/140-1/users/liying/conda_envs/nlora_req/bin/python`。
- 已验证历史四卡环境：Python3.10.20、torch2.1.1+cu121、transformers4.28.0、datasets1.17.0、accelerate1.4.0、numpy1.24.4、CUDA12.1、cuDNN8902。新run在manifest冻结pip freeze与实际导入模块路径。
- Base models：`/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B` 与同目录 `Llama-3.2-3B`。记录完整权重、tokenizer/config SHA256，不下载替代版本。
- 数据：`/data/140-1/users/liying/O-LoRA/CL_Benchmark`；将所需四数据集与配置复制入新source snapshot，保留input hash。任务配置为`configs/order{1,2,3}_configs/{dataset}`，指令为`configs/instruction_config.json`。
- NLTK：`/data/140-1/users/liying/O-LoRA/.cache/nltk_data`。
- 统一真实训练入口：`<new_frozen_source>/src/run_uie_lora.py`。用绝对路径 `python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0` 调用。
- `PYTHONPATH=<new_frozen_source>/src`，确保导入仓库定制PEFT，不能误用其他环境的PEFT。每个cohort单独的HF_HOME / HF_MODULES_CACHE / HF_DATASETS_CACHE / TRANSFORMERS_CACHE / UIE_DATA_CACHE_ROOT，offline flags true，TOKENIZERS_PARALLELISM false，OMP/MKL/OPENBLAS线程4，report_to none。

## 所有方法固定的基础参数

BF16 AMP true，FP16 false，模型参数FP32，bf16_full_eval/fp16_full_eval false；一任务一epoch；rank8、alpha32、dropout0.1、q_proj/v_proj；第一任务LR1e-3，后续任务LR1e-4，constant schedule、warmup0；max_source_length512、max_target_length50、generation_max_length50、num_beams1；eval每卡batch4；max_grad_norm1；DDP find_unused_parameters false；TF32 matmul false / cuDNN true；instruction_strategy single，add_task_name/add_dataset_name true，num_examples0；不设max_train_samples/max_predict_samples等截断；阶段内不做额外dev选择。保持历史每任务重新建立optimizer/scheduler，适配器与Fisher/Tree状态串接。

数据配置的`sampling strategy=full`意味着默认`max_num_instances_per_task=10000`不会截断DBPedia。已实际读取JSON核对：

| 数据集 | train | dev | test | 一epoch optimizer steps | 实际完整window样本 |
|---|---:|---:|---:|---:|---:|
| DBPedia | 14000 | 7600 | 7600 | 218 | 13952 |
| Amazon | 5000 | 7600 | 7600 | 78 | 4992 |
| Yahoo | 10000 | 7600 | 7600 | 156 | 9984 |
| AGNews | 4000 | 7600 | 7600 | 62 | 3968 |

上述step数来自保留的旧训练循环只更新完整GA窗口，末尾不足一个global64窗口不进行optimizer更新；不应在补seed时悄悄修复该语义。EWC Fisher仍处理全量14000/5000/10000/4000样本，包括optimizer尾部未用的样本。阶段s只评估已出现的s个数据集，每个7600例；最终FAA取四个`predict_exact_match_for_<dataset>`的平均值，不混入SC/TC aggregate字段。

Order1：DBPedia→Amazon→Yahoo→AGNews；Order2：DBPedia→Amazon→AGNews→Yahoo；Order3：Yahoo→Amazon→AGNews→DBPedia。

## Seed 与 data_seed

每个新run明确同时传`--seed S --data_seed S`，S∈{0,42,1993}，每个任务阶段继续使用相同S，不自动加task offset。底层 `src/run_uie_lora.py:729` 无条件调用 `set_seed(training_args.seed)`。安装环境的transformers `trainer.py:813`为 `data_seed if data_seed is not None else seed`，DDP4下传给DistributedSampler，因此历史`data_seed=None, seed42`的有效sampler seed就是42；可记录 raw_data_seed=None、effective_data_seed=42，补跑显式42。不能用`data_seed or 42`等真假判断，seed0是有效值。

已在`CUDA_VISIBLE_DEVICES=''`的CPU验证torch2.1.1的DistributedSampler：三个seed各自的None与显式同seed生成相同分区，64个样本在4rank完整无重叠；三个seed排列不同。seed0 rank0前8索引为44,47,33,28,40,15,48,61。这是sampler语义验证，不是GPU训练测试。

## 六方法的CLI和正则缩放

| 方法 | lora_strategy | optimizer_type / 实际optimizer | 固定方法参数 | 四任务后rank |
|---|---|---|---|---:|
| SeqLoRA | seqlora | adamw_hf / transformers.AdamW | 无额外正则 | 8 |
| IncLoRA | inclora | adamw_hf / transformers.AdamW | 每任务新rank8 | 32 |
| OLoRA+GAM | olora | gam-adamw_hf / GAM(torch.AdamW) | 计划olora_lamda_1=0.25，lamda2=0；下文先验验证 | 32 |
| TreeLoRA | treelora | adamw_hf / transformers.AdamW | treelora_reg0.5，num_tasks4，A-history，epoch内线性ramp | 8 |
| EWCLoRA | ewclora | adamw_hf / transformers.AdamW | lambda0.5，gamma1，OLoRA两个系数0 | 32 |
| FS-EWCLoRA | ewclora_gam | gam-adamw_hf / GAM(torch.AdamW) | lambda1，gamma1，OLoRA两个系数0 | 32 |

普通AdamW实际使用transformers.optimization.AdamW默认eps1e-6、weight_decay0、betas(.9,.999)；GAM虽然CLI叫`gam-adamw_hf`，实际构造torch.optim.AdamW，eps1e-8、weight_decay0.01、betas(.9,.999)，不能仅依赖打印的TrainingArguments来判断optimizer。`create_optimizer_and_scheduler`当前只把params、lr传入默认构造器，单独更改CLI `--weight_decay`未必生效；应冻结构造器并审计实例param_groups。

GAM两方法都固定rho0.02、normrho0.2、gamma0.03、beta1/2/3=1、adaptivefalse、grad_reducemean、perturb_eps1e-12。FS另project_eps1e-12。EWC都用full_data_raw Fisher，fisher_max_batches0、两个normalize flag false、damping0、gamma1。

### 为什么 EWC 为0.5，而FS仍为1

`src/uie_trainer_lora.py:3796–3808`的微批loss先将CE除以GA，然后调用`_apply_train_regularization`；纯EWC于该函数末尾加EWC惩罚，因此一次更新的正则系数约为GA×lambda。历史3B为GA8×lambda1，计划GA16×lambda0.5保留同名义比例；1B已复用的四卡主表就是lambda0.5。

FS的EWC梯度在`src/uie_trainer_lora.py:3225`附近于GAM四次replay闭包外计算一次并通过direction_projector使用，不累加GA次，lambda1必须保持。补seed不能把纯EWC的0.5规则套用FS。

### OLoRA+GAM为何建议0.25，及执行前验证

源证据：`_compute_micro_batch_loss`先CE/GA，再`_apply_train_regularization`。strategy=olora调用`_apply_olora_regularization`，每微批直接加`olora_lamda_1 * orthogonal_loss`。`_run_gam_replay_pass`遍历整个GA窗口并对每微批backward，因此历史GA8×0.5=4，新GA16若仍写0.5会变成名义8，建议新四卡设0.25维持4。历史OLoRA+GAM成功日志明确显示`gam-replay: ... λ1:0.5; λ2:0`，例如1B Order1日志4654起；候选历史commit28c5788的 `_apply_train_regularization`直接实现同样加法，源码1983行与日志行号相符。

**这是需在新协议中明确记录的GA补偿，不是已完成的实测等价结论。** 在正式训练前必须有两阶段smoke：第一阶段无旧LoRA A正交项，不能验证lambda；第二阶段加载真实前一任务adapter，在相同参数和冻结dropout的受控梯度witness上比较GA8/0.5与GA16/0.25的正则/CE梯度比例，并检验实际OLoRA正交gradient非零、实际GAM四次closure各含16微批。该witness只确认缩放逻辑，不宣称新旧分布式训练轨迹相同。若witness失败，暂停该方法正式队列并修正计划，不能静默使用0.5或更改优化算法。

### DDP8和DDP4不能称为完全等价

即便global batch均64且普通分布式sampler全局排列相同，每rank分配和dropout流改变，浮点reduce顺序也改变。GAM更关键：旧8rank每个rank在8例上进行非线性局部探测，再均值最终方向；新4rank每rank在16例上探测，结果不满足线性可交换性。只同步最终GAM方向应保留，不能为“对齐”在每次probe加同步。Tree的rank局部trajectory也要求一致world_size。故需把已有DDP8 seed迁移成四卡后，才能形成严格三seed cohort。

## 代码冻结与审计依赖

旧四卡launcher不接受动态seed；`_bf16_completion_audit_expected_protocol()`硬编码42或1993、固定lambda/strategy。直接改命令传seed0或seed42会被旧审计拒绝。新副本需要参数化cohort-specific manifest/audit expected protocol、验证器与文件命名，并保留：早期local_rank CUDA绑定、主进程先构建datasets1.17共享缓存、只由rank0保存、阶段同步barrier。

可参考的冻结快照：
- EWC：`experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/source`。
- FS：`experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/source`。
- Tree：`experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/source`（最新，含Tree trainer与上述FS审计）。

已比较关键函数AST：上述三份与当前checkout的 `_compute_micro_batch_loss`、`_run_gam_replay_pass`、`_compute_ewclora_grad_cache`、`_compute_ewclora_loss`、`create_optimizer_and_scheduler`相同。FS和Tree快照与当前checkout的 `_apply_train_regularization`、`_build_ewclora_gam_projector`也相同；EWC旧快照这两个函数与后者不完全同AST（后来加GAM审计），因此统一冻结新版时须按实际EWC分支做语义diff，不能把函数hash不同简单当算法变更或当完全相同。

Tree固定upstreamcommit `1c7260c42b34e1961283797c742f08b9c3842501`。Tree烟测应至少三阶段，确认history增长与后续任务regularizer有效。其余至少两阶段，确认adapter/Fisher链。

新run审计应记录：四rank身份/物理UUID、BF16实际forward dtype、真实optimizer defaults、有效seed、训练样本indices覆盖、full windows、模型权重hash、LoRA trainable branches/counts（1B32个q/v模块64 A/B tensors；3B56模块112 tensors）、非零有限参数更新、冻结旧adapter、Fisher次数/全样本、Tree sidecar。GAM保留四closure、每次16微批、probe no_sync、最终gradient sync、方向投影非零witness。GAM局部扰动加减可有FP32残差，保留hash差异与maxdiff并按已有atol1e-6/rtol1e-5检查，不为强制bitwise相同额外广播参数。

所有新结果建议独立目录 `experiment_queue/nlp_multiseed_bf16_ddp4_b64_g3-6_20260922.../<model>/<method>/seed_<S>/order_<O>/<stage>-<dataset>/`，另有commands/logs/audits/source/input hashes。旧历史值引用原路径，既不覆盖也不伪装成新训练。

## 时间预算（等待GPU时间另算）

已读取主表36个order的每个阶段train_results.json与predict_results.json，总144阶段。训练时长远小于总耗时，不能把train_runtime当整轮耗时；还需生成评估、Fisher、加载与写盘、smoke和验证。

已完成四卡队列实测墙钟：EWC1B三order加smoke108.62min；FS1B三order加smoke176.42min；Tree1B+3B六order加smoke265.29min。对应train+predict三order：EWC1B84.12min，FS1B150.63min，Tree1B75.96min，Tree3B173.44min。

粗预算模型：对四卡历史直接用train+predict；对八卡历史用2倍作为四卡投影；再乘缺少seed组数（已四卡两个，其余三个）。96个order投影合计**101.0小时**，加20–40%加载/Fisher/验证开销后**121–141小时，约5.1–5.9天**。这是排队规划假设，不是速度承诺：历史八卡硬件/利用率并未充分核实，不能保证线性缩放。执行后用每个模型至少一个完整order、尤其3B GAM族，重新估计剩余时间。

## 多seed汇总定义

对每个model/method/order，以三个seed的最终FAA报告mean±sample std(ddof1)；不要将三个order误当三个seed。Avg列先对每个seed平均三个order，得到3个seed-level averages，再计算mean±std。过程图每个task-stage同样从三个真实seed求mean和std，stage4新task只一时间点但仍有三seed观测。三seed全到齐前标记n与missing seed，不以单seed旧主表数字伪造errorbar。
