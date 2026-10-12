# NLP 三 seed 补充实验计划

日期：2026-09-22。目标 seed：**0、42、1993**。模型：Llama-3.2-1B / Llama-3.2-3B。方法：SeqLoRA、IncLoRA、OLoRA+GAM、TreeLoRA、EWCLoRA、FS-EWCLoRA。仅允许物理 **GPU 3、4、5、6**。

本文件是已审计的实验计划；本次没有启动训练。当前表和 v6 图保持不变，新三 seed 结果另建目录与版本。

## 1. 本轮回答的问题

| 结论目标 | 所需证据 | 对应产物 |
|---|---|---|
| C1：FS-EWCLoRA 相对 EWCLoRA 的最终 FAA 优势是否在 seed 变化后仍成立？ | 两模型、三 Order、三个 seed 的完整结果，逐 seed 配对差与均值/样本标准差 | 三 seed 主表、逐 seed 明细 |
| C2：旧任务保留与 Order 敏感性的观察是否稳定？ | 每个 seed 的四阶段分任务准确率，不混入未学习任务 | 六子图的 mean ± SD 曲线、逐 seed 遗忘明细 |

这些实验补齐现有六方法比较，不增加 T5、额外 baseline、参数搜索或视觉实验。不同方法的优化器与适配器预算保留其定义，不能把方法间比较直接解释为某一组件的因果消融。不预设新结果仍有“五胜六中”或旧主表的排名。

## 2. 历史记录重新核查

当前主表 **36 个完整 Order、144 个任务阶段**均存在真实预测与训练记录；所选指标和训练文件的 SHA-256 与原绘图索引完全一致，已使用指标均为有限数值。各阶段 trainer state 与日志中的 train_runtime 另经两份历史审计复核。

| 方法 | 1B 已有 seed（覆盖 Order 1/2/3） | 1B 缺失 seed | 3B 已有 seed（覆盖 Order 1/2/3） | 3B 缺失 seed |
|---|---|---|---|---|
| SeqLoRA | 42；DDP8/GA8 | 0、1993 | 42；DDP8/GA8 | 0、1993 |
| IncLoRA | 42；DDP8/GA8 | 0、1993 | 42；DDP8/GA8 | 0、1993 |
| OLoRA+GAM | 42；DDP8/GA8 | 0、1993 | 42；DDP8/GA8 | 0、1993 |
| TreeLoRA | 42；DDP4/GA16 | 0、1993 | 42；DDP4/GA16 | 0、1993 |
| EWCLoRA | 42；DDP4/GA16 | 0、1993 | 1993；DDP8/GA8 | 0、42 |
| FS-EWCLoRA | 1993；DDP4/GA16 | 0、42 | 1993；DDP8/GA8 | 0、42 |

上表全部有效 batch 为 64、每卡 batch 为 1。在本次检查的结果目录、日志、队列和 CopyFlatCV 副本中，未发现可新增到这些行的完整 seed 0 结果，也未发现缺失的其他目标 seed 的同方法同设置结果。不能将此结论扩展为“所有其他服务器均不存在”。

审计范围和排除依据：

- Seq/Inc/OLoRA+GAM/Tree：24 个所选 Order，96 个阶段；54 份相关历史日志的 seed 均为 42。CopyFlatCV 的 42 条相关记录为已有结果的字节一致副本。未发现 9 月 21 日及之后新增的同方法预测结果。
- EWC/FS：29 个候选 Order 逐项审计，12 个主表 Order 完整。旧 1B batch40、FP32/不同 lambda、SAM/FisherShape 变体、smoke、reload-eval 和启动重试均排除。
- 相同结果副本、同 seed 重新评估、不同 Order，以及同一次运行的多份日志均不能增加 seed 数。
- 详见 `BASELINE_HISTORY_AUDIT.{md,json}`、`EWC_HISTORY_AUDIT.{md,json}`、`CURRENT_SOURCES_AUDIT.json`。旧八卡记录缺少完整运行时源码哈希的限制保留在历史审计中。

## 3. 两种补齐口径

完整目标为 **2 模型 × 6 方法 × 3 seeds × 3 Orders = 108 个完整 Order**，每个 Order 包含四次顺序训练，即 432 个任务阶段。

### 推荐：统一四卡执行协议

新三 seed 表统一采用 **DDP4 × batch1 × GA16 = 64**。

| 方法 | 1B 可直接复用 seed | 1B 要运行的 seed | 3B 可直接复用 seed | 3B 要运行的 seed | 新完整 Order 数 |
|---|---|---|---|---|---:|
| SeqLoRA | 无 | 0、42、1993 | 无 | 0、42、1993 | 18 |
| IncLoRA | 无 | 0、42、1993 | 无 | 0、42、1993 | 18 |
| OLoRA+GAM | 无 | 0、42、1993 | 无 | 0、42、1993 | 18 |
| TreeLoRA | 42 | 0、1993 | 42 | 0、1993 | 12 |
| EWCLoRA | 42 | 0、1993 | 无 | 0、42、1993 | 15 |
| FS-EWCLoRA | 1993 | 0、42 | 无 | 0、42、1993 | 15 |
| 合计 | | | | | **96** |

每个列出的 seed 均跑三个 Order。12 条已有四卡 Order 直接复用；96 条新训练包括 **72 条真正缺失 seed 的 Order + 24 条八卡历史 seed 的四卡迁移**，共 384 个新任务阶段。八卡历史结果只作迁移前对照，保留原文件，不替换成新的重复观测。

这样做是为了避免将 seed 的变化与 DDP/GA 的变化混在同一组方差中。相同 global batch 不能证明逐步更新一致，GAM 的本地扰动窗口尤其不等价。四卡历史复用还需在冻结输入时确认模型/tokenizer、数据文件和方法实现兼容；如发现实质差异，应撤销对应复用标记并更新计划计数，而不是隐瞒差异。

### 最小补缺备选：新增 72 条 Order

保留全部 36 条当前主表历史结果，只补缺失的两组 seed，每组三个 Order，共新增 72 条 Order / 288 个阶段。新增训练也只用 GPU 3–6、DDP4/GA16。

此口径会混合 DDP8/GA8 与 DDP4/GA16，必须披露为混合执行协议结果；不能称为严格固定配置下的纯 seed 方差。它节省 24 条四卡迁移训练。本计划的 `RUN_QUEUE_DDP4.csv` 采用上面的统一四卡推荐口径；`SEED_ORDER_LEDGER.csv` 同时保存两种口径，避免把历史存在与严格可复用混淆。

## 4. 固定数据、训练配置与方法身份

### 共用配置

- seed 集合固定为 `[0,42,1993]`，每个 Order 从原始基座和该 seed 的新 adapter 开始；不跨 seed、方法或 Order 继承 checkpoint/Fisher。
- 新运行显式 `seed=s, data_seed=s`。Transformers 4.28 的采样器在 `data_seed=None` 时回退到 seed；因此旧 `seed=42,data_seed=None` 的有效采样 seed 为 42。必须使用 `is not None` 判断，避免将合法 seed 0 当作未指定。
- 新训练 DDP4、每卡 batch1、GA16、有效 batch64；BF16 AMP，FP32 参数，`fp16=False`。保持旧评估精度，不能随意开启 BF16 full evaluation。不得因 OOM 自动改变卡数、GA、batch 或精度。
- 每任务 1 epoch；首任务学习率 `1e-3`，后续任务 `1e-4`；constant scheduler，warmup 0；梯度裁剪 1.0。
- LoRA rank8、alpha32、dropout0.1，target modules `q_proj,v_proj`。SeqLoRA 与 TreeLoRA 固定 rank8；IncLoRA、OLoRA+GAM、EWC、FS 每任务新增 rank8，第四任务累计32。
- `max_source_length=512`、`max_target_length=50`、`generation_max_length=50`；每卡 eval batch4；`ddp_find_unused_parameters=False`。
- 保留同一套 task configuration、prompt、标签规范化与生成式 exact-match 评估。已有训练样本数：DBPedia 14,000，Amazon 5,000，Yahoo 10,000，AGNews 4,000；每个数据集测试 7,600 条。四数据集配置均为 `sampling strategy=full`，固定全量训练集合，不按 seed 重新子采样；seed 改变初始化、dropout 和 DistributedSampler 顺序，记录训练集合哈希与每 rank 顺序哈希，避免缓存误用。
- 保留旧训练循环仅更新完整 GA 窗口的行为：上述四数据集每 epoch 分别为 218、78、156、62 个 optimizer steps，对应 13,952、4,992、9,984、3,968 条参与完整更新窗口的样本；尾部不足 global64 的窗口不做 optimizer 更新。补 seed 时不同时修改这一语义，EWC Fisher 仍遍历全量训练数据。
- 每阶段仅汇总已经学习的数据集；第四阶段四数据集共 30,400 条测试样本。历史个别日志对未学任务的额外评估不进入本轮曲线与均值。

### 必须处理的 GA 与正则项关系

旧代码先计算 `CE / GA`，再加入普通 EWC 或 OLoRA 正则，因此直接复制 CLI lambda 并将 GA 从 8 改为 16，会改变正则相对 CE 的名义系数。不能只改 GPU 列表和 GA。

| 方法 | 四卡目标策略与优化器 | 四卡正则设置 | 说明 |
|---|---|---|---|
| SeqLoRA | `seqlora`；HF AdamW | 无 | eps1e-6，weight_decay0 |
| IncLoRA | `inclora`；HF AdamW | 无 | eps1e-6，weight_decay0 |
| OLoRA+GAM | `olora`；GAM(Torch AdamW) | 拟定 `olora_lamda_1=0.25`，lambda2=0 | 旧 GA8/0.5 对应新 GA16/0.25 的名义比例；必须先通过两阶段梯度/closure 验证，再冻结 |
| TreeLoRA | `treelora`；HF AdamW | `treelora_reg=0.5` | 沿用现有四卡实现、线性 ramp 与 A 参数历史 |
| EWCLoRA | `ewclora`；HF AdamW | `ewclora_lambda=0.5,gamma=1` | `GA16×0.5=GA8×1`；1B 现有四卡已如此，3B 由八卡迁移时作同样补偿 |
| FS-EWCLoRA | `ewclora_gam`；GAM(Torch AdamW) | `ewclora_lambda=1,gamma=1`，OL 正则为0 | EWC 通过 GAM 外的方向处理加入，不能套用纯 EWC 的 lambda0.5 补偿 |

GAM 参数继续使用 `grad_rho=0.02,grad_norm_rho=0.2,grad_gamma=0.03,beta1/2/3=1,adaptive=False,grad_reduce=mean,eps=1e-12`；Torch AdamW 实际 eps1e-8、weight_decay0.01、betas(0.9,0.999)。EWC/FS 使用原 full-data raw Fisher，不替换成其他 FisherShape/SAM 方法。

**OLoRA+GAM 的 0.25 是累积语义补偿的候选值，不是已完成验证的参数搜索结论。** 在小样本 task2 中分别记录 CE 与正则梯度，验证 GA8/0.5 与 GA16/0.25 的名义比例及 GAM closure 的求均值位置。GAM 局部扰动本身可能仍不同，所以三个 seed 都使用冻结后的同一个四卡协议。若源码路径不满足上述缩放推导，必须先修订参数和 ledger 后再开始正式三 seed 运行，不允许运行后按测试 FAA 选择 lambda。

该主表中的 OLoRA 实际为 OLoRA+GAM，输出目录和机器记录统一写 `olora_gam`；论文显示别名可保留，但图注/实现细节必须说明。

## 5. 源码与启动准备

执行入口为 `<new_frozen_source>/src/run_uie_lora.py`；已验证 Python 环境为 `/data/140-1/users/liying/conda_envs/nlora_req/bin/python`。基座为 `/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-{1B,3B}`，数据为本仓 `CL_Benchmark`，任务配置为 `configs/order{1,2,3}_configs`。参考已有 EWC、FS、Tree 四卡 campaign 的 source snapshot 和 input manifest，冻结每个方法所需的模型、数据、tokenizer、代码及环境版本。完整环境/CLI/源码行号依据见 `EXECUTION_DESIGN.md`。

现有 launcher 和 trainer audit 有硬编码的 seed42/1993、lambda、strategy，旧 Tree launcher 还限定 GPU0–3。因此**现有单 seed 启动器不能直接改一个 `--seed` 后批量复用**。实现时在新的 campaign source 副本中将审计改为读取不可变 run spec，并同时断言 seed、data_seed、GPU UUID、DDP4/GA16、优化器实际对象/默认值、方法策略、Fisher 来源和阶段继承。不修改正在运行任务或旧 source snapshot。

新队列输出根建议：

```text
experiment_queue/nlp_multiseed_bf16_b64_ddp4_s0_42_1993_<启动UTC时间>/
  manifest.json
  source/                  # 冻结代码；修改也记录hash
  inputs_manifest.json
  history_reuse.json       # 仅索引12条已完成四卡Order
  smoke/                   # 与正式统计分离
  1b/<method>/seed_<s>/order_<o>/<stage>-<dataset>/
  3b/<method>/seed_<s>/order_<o>/<stage>-<dataset>/
  logs/<model>_<method>_s<s>_o<o>.log
  status.json
  final_per_seed.csv
  multiseed_summary.csv
```

新结果目录包含 model/method/seed/order；每个 stage 保存完整命令、运行时参数、训练状态、分任务预测、Fisher/checkpoint 继承、文件哈希和退出码。停止后从同一配置的最后一个已验证阶段恢复，失败或部分结果不当作0，不选择最好的一次重跑。不同 seed 不共享可变训练缓存或 Fisher。

## 6. GPU 限制与顺序调度

2026-09-22 12:03 UTC 的只读检查显示，GPU3 被 PID969996 使用（约12,354 MiB、100%利用率），GPU4/5/6 空闲（各2 MiB）。设备均为 RTX6000 Ada、49,140 MiB。见 `GPU_STATUS_AT_PLANNING.json`。初始沙箱查询失败，随后只读设备查询成功。

- 新任务仅使用物理 GPU3/4/5/6。启动前重新确认索引到 UUID 的映射，确保 `CUDA_VISIBLE_DEVICES=3,4,5,6` 或等价 UUID 白名单实际指向这四张卡；进程内 local_rank 0–3 不等于物理 GPU0–3。
- 等待四张卡全部空闲后再进入四进程 DDP。GPU3 仍忙时仅等待，不杀现有任务，不改用其他 GPU，不静默改为三卡。
- 队列同时只运行一个四卡任务；四个 stage 串行、各 Order 独立初始化。共享队列锁、启动前设备占用复查、唯一端口、退出码检查与原子状态更新。
- 当前 GPU 占用不是未来预约。启动时重新检查，设备不满足条件则保持 WAITING；不能根据本计划的历史占用快照直接启动。

## 7. 阶段安排

| 阶段 | 内容 | 正式完整 Order 数 | 进入下一阶段的条件 |
|---|---|---:|---|
| M0 | 启动器泛化、输入哈希、seed0检查；Tree至少三阶段、其他方法至少两阶段小样本验证；EWC/OL正则与GAM语义核验 | 0 | 四卡BF16、跨rank参数合理一致、task2正则和Tree后续历史有效、参数冻结；smoke不入表 |
| M1/M2 | 先补 EWC 与 FS，覆盖两模型三seed三Order，并复用已有6条四卡Order | 30 | 每个Order四stage齐全、30,400最终样本计数通过、FAA可从预测复算 |
| M3 | OLoRA+GAM、TreeLoRA；复用Tree已有6条 | 30 | 与冻结方法定义一致，不能混入纯OLoRA或SAM/FisherShape |
| M4 | SeqLoRA、IncLoRA，补足每模型三个四卡seed | 36 | 各模型方法均3/3 seed且3/3 orders |
| M5 | 汇总三seed主表、配对差与六子图；另存新版本 | 0 | 108/108 Order完整、432/432任务阶段有有效记录 |

前四个正式运行用于同时校准小/大模型耗时，并非性能筛选：
1. 1B EWCLoRA，seed0，Order1；
2. 1B FS-EWCLoRA，seed0，Order1；
3. 3B EWCLoRA，seed0，Order1；
4. 3B FS-EWCLoRA，seed0，Order1。

完整96条队列在 `RUN_QUEUE_DDP4.csv`，108个目标单元在 `SEED_ORDER_LEDGER.csv`。计划中的顺序只影响调度，不允许因某seed结果较差就不运行或不汇总。

## 8. 算力预算

以144个已有阶段的 **train_runtime + predict_runtime** 估算，旧八卡记录暂按时间×2折算四卡，已有四卡记录不缩放，再加20–40%覆盖启动、Fisher、保存与审计：

- 推荐96条四卡运行：约 **121–141小时连续四卡占用**，约 **485–566 GPU-hours**，约 **5–6天**；不含等待GPU3空闲的时间。
- 最小72条补缺备选：约 **87–102小时**。
- 这是预算模型，旧八卡硬件和吞吐不能视为已知相同；需要首批完整1B/3B运行后更新实测预测，不能承诺固定完成日期。详见 `COMPUTE_ESTIMATE.json`。
- 数据准备复用现有文件，不需要新增人工标注。主要时间包含生成式评估与Fisher，不能只按训练时长估计。

## 9. 统计口径与论文产物

对模型m、方法h、Order o、seed s，先从最后阶段四数据集计算 `FAA[m,h,o,s] = mean_d Accuracy[m,h,o,s,d]`。每个Order表格单元报告三seed均值及 **样本标准差（ddof=1）**。

Avg. 列先为每个seed计算 `A_s = mean_o FAA[o,s]`，再对三个 `A_s` 计算 mean ± SD。不能平均三个Order的标准差，也不能将9个seed×Order单元视为9个独立seed。

FS与EWC比较按相同seed、模型、Order配对；报告三个seed的差值和平均差。跨Order总差先在每个seed内对三个Order求平均，再跨seed汇总。三seed下不预先宣称显著性，不沿用单seed“5/6”的计数作为新结论。

过程图仍为2×3面板，每条数据集曲线在相同阶段上跨三个seed计算mean ± SD；最后引入任务只有最终单点及其SD，不造出线段。单seed原图和表保留。所有结果从未四舍五入的分任务原始数据汇总，最后统一显示两位小数。

## 10. 完成检查

- [x] 六方法、两模型、三个seed、三个Order的108个目标单元已列出。
- [x] 当前36条Order的144个stage文件、哈希及日志证据复查完成。
- [x] 旧副本、batch40、其他方法、失败/烟测记录未计入可复用seed。
- [x] 固定GPU3–6；当前GPU3占用情况已记录。
- [ ] 新多seed启动器与运行时审计泛化，OLoRA+GAM累积语义验证完成。
- [ ] 输入/代码冻结、可复用12条历史运行的最终兼容性检查完成。
- [ ] 96条新Order完成，108目标单元均通过验证。
- [ ] 三seed表、配对统计和新曲线已另存；不宣称单组件因果消融或额外任务泛化。
