# 当前细粒度 CL 表格的 seed 0/1/2 实验计划

日期：2026-09-22。范围：7 方法 × Aircraft、Cars、CUB-200、Flowers、5Datasets，共 35 个配置组。只制定计划，未启动训练。

## 目标与建议

目标是让每组拥有同配置、独立初始化的 seed 0、1、2，最终生成 35 组 FAA/AAA 的均值和样本标准差。每次 run 指从预训练模型开始完成整个任务序列，不是一个任务、一个 epoch 或从 seed0 checkpoint 继续训练。

**名义缺口为 68 次，建议按 99 次新运行预算，最多 105 次。** 68 次假定现有 37 次运行全部可以与新运行按同协议合并；目前不能证明这一前提。推荐重新冻结来源不完整的 29 个组并跑三 seed，共 87 次；其余 CL-LoRA 五组及 GAM-EWC/5Datasets 一组，各补 1、2，共 12 次。六个旧 seed0 必须通过来源校验才能保留，否则对应组也重跑 seed0，最多增加 6 次。

这项计划保留当前表格的主要训练设置和方法原生机制，解决各组结果的随机种子稳定性；不宣称消除了方法之间全部配置差异。若另做跨方法严格统一预算的实验，应单独立项，不能与本次结果混合。

## 主张与证据

| 主张 | 最低证据 | 对应实验 |
|---|---|---|
| C1：当前方法比较在固定配置下具有可报告的重复运行稳定性 | 每组 0/1/2 完整矩阵、固定非seed配置、全体均值/SD、不选择最高seed | B1、B2 |
| C2：统计中的波动来自重复训练，而非混入代码/数据/配置变化 | 源码、环境、权重、数据与有效配置记录；每组除seed/输出名称外一致 | B0、B2 |

不以三 seed 自动证明显著性、跨任务顺序稳健性或纯粹的方法因果增益；不增加无关消融、新数据集或新方法。现有七行全部保留。

## 35 组补齐矩阵

下表是保留当前历史结果时的**名义缺失 seed**，不是已经通过复用校验的队列。

| 方法 | Aircraft | Cars | CUB-200 | Flowers | 5Datasets | 名义新增 | 推荐新增 |
|---|---|---|---|---|---|---:|---:|
| Vanilla LoRA | 1,2 | 1,2 | 1,2 | 1,2 | 1,2 | 10 | 15 |
| InfLoRA | 1,2 | 1,2 | 1,2 | 1,2 | 1,2 | 10 | 15 |
| OLoRA | 1,2 | 1,2 | 1,2 | 1,2 | 1,2 | 10 | 15 |
| SDLoRA | 1,2 | 1,2 | 1,2 | 1,2 | 1,2 | 10 | 15 |
| CL-LoRA | 1,2 | 1,2 | 1,2 | 1,2 | 1,2 | 10 | 10 |
| EWCLoRA | 1,2 | 0 | 1,2 | 0 | 0,1 | 8 | 15 |
| GAM-EWCLoRA | 1,2 | 1,2 | 1,2 | 1,2 | 1,2 | 10 | 14 |
| 合计 | 14 | 13 | 14 | 13 | 14 | **68** | **99** |

推荐列：四种历史基线、EWC 五组、GAM 前四组均新跑 0/1/2；CL 五组与 GAM/5Datasets 条件复用原 seed0。若仅补当前单seed的33组、暂不补EWC Cars/Flowers，会是66次；但不能实现全表统一0/1/2，因此不采用该范围。

另协议 GAM/CUB、Flowers 已有 seeds1/2/3，不能拿 seed3 替代目标 seed0，也不能把受控协议1/2与原历史0拼接。若未来明确选择整组切换到该协议，可保留其1/2并新增0，推荐预算从99降至95；这是协议切换，主计划默认不采用。

## B0：冻结配置和来源（MUST）

### 公共设置与数据划分

- Backbone：ViT-B/16，预训练标识 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`。必须固定实际 checkpoint 文件 SHA，不能只记 hub 名。
- rank=16，batch_size=128，SGD，momentum=0，weight_decay=0，cosine；首任务/后续任务的初始 LR 相同。FP32；不临时切换 AMP/BF16、梯度累积、DDP 或自动缩 batch。
- `class_shuffle=false`，固定任务和类别顺序；seed改变初始化、训练采样及增强等随机过程，不改变数据划分或任务顺序。
- 无训练 replay：memory_size=0、memory_per_class=0、fixed_memory=false。保留实现中的原型统计及辅助遍历；删除这些遍历可能改变 RNG，不能在同一组不同 seed 间改变。

| 数据集 | 类别/任务 | LR（首任务与后续相同） | 每任务 epochs |
|---|---|---|---|
| Aircraft | 100类，10×10 | .05；SDLoRA 单独保持 .02 | 全方法40 |
| Cars | 196类，16+9×20 | .05 | 全方法40 |
| CUB-200 | 200类，10×20 | .01 | 全方法40 |
| Flowers | 102类，12+9×10 | .0025 | 全方法40 |
| 5Datasets | 635类，200→100→196→102→37 | .02 | Inf/OLoRA/SDLoRA=40；Vanilla/CL/EWC/GAM=30 |

5Datasets 顺序仍是 CUB→Aircraft→Cars→Flowers→Pets。删除独立 Pets 表格列不改变这条流。task_splits 覆盖配置中的名义 increment，执行时校验实际任务边界。

预处理保持数据集实际协议：CUB/Flowers 是 resize256×256、随机crop224、翻转、ImageNet归一化；Aircraft几何一致但tensor loader；Cars单数据集短边resize224、center crop224、翻转、mean0/std1；Het5统一CUB预处理。固定训练/测试样本、路径、标签、文件内容身份；当前CUB清单9430/2358不称为官方标准划分。任何更换划分均为新实验。

### 方法及代码设置

| 方法 | 冻结来源/关键设置 | 本次处理 |
|---|---|---|
| Vanilla | 完整 Flatness_CV 源码，`seqlora`，12层Q/V、rank16 | 新跑三seed |
| InfLoRA | 同完整工程，`inflora`，lamb=.7、lame=.99；显式固定默认值 | 新跑三seed |
| OLoRA | 同完整工程，`olora_lamda_1=1`、`olora_lamda_2=0`、learn_alpha=false | 新跑三seed |
| SDLoRA | 同完整工程，learn_alpha=true、max_prev_tasks=200；保留当前scalar初始化行为 | 新跑三seed |
| CL-LoRA | 9/18 attempt2 完整冻结源码；shared层0–5、specific层6–11、Q/V；kd_ratio=5、temperature=2、orth_ratio=1e-4 | 满足复用条件后补1/2 |
| EWC | 完整冻结工程；前四数据集alpha0 wrapper，Het5 native；固定每组lambda与Fisher机制 | 新跑三seed |
| GAM-EWC | 前四组采用完整冻结GAM实现并显式填写下表参数；Het5保留9/18冻结实际行为 | 前四组三seed；Het5补1/2 |

不能直接用 CopyFlatCV 或9/18 CL快照跑 InfLoRA：两者缺少 `inflora.py`。OLoRA/SD单个learner相同不代表共享backbone、loader、trainer相同。Vanilla/Inf/OLoRA/SD 的完整冻结来源是 Flatness_CV；旧日志没有绑定完整历史源码，因此这20组新seed0必须纳入最终统计。

四个历史基线的新冻结配置显式 `min_lr=0.1×LR`，workers=8；Aircraft pin_memory/persistent_workers=true，其余保持当前实际默认false。CL最低LR=0、workers=2、pin_memory=true、persistent_workers=false，保持原冻结实现。EWC/GAM保持所选冻结实现的实际cosine与loader行为，并在预检输出有效值；不能只往JSON加入实现不读取的键。

EWC固定lambda：Aircraft=200、Cars=80000、CUB=200000、Flowers=10、5Datasets=50000。共同：gamma=1、eta=0、max_batches=100、batch_empirical Fisher、按任务normalize=true、accumulated normalize=false、damping beta=0、dual_ascent=false。不再按新seed的测试结果重新选lambda。Cars/Flowers固定 `fs_direction_mode=alpha0`、probe/Fisher RNG preserve=true；Aircraft/CUB历史未记录这些字段，新冻结版应显式记录选定实现的alpha0实际路径和RNG行为，不倒称为历史确证。Het5继续native，不混wrapper结果。

| GAM 数据集 | ewc_lambda | grad_rho | grad_norm_rho | grad_gamma | beta_1/2/3 |
|---|---:|---:|---:|---:|---|
| Aircraft | 2000 | .01 | .1 | .02 | 1/1/1 |
| Cars | 2000 | .05 | .2 | .03 | 1/1/1 |
| CUB | 2000 | .2 | .2 | .1 | 1/1/1 |
| Flowers | 2000 | .2 | .2 | .1 | 1/1/1 |
| 5Datasets | 2000 | .2 | .2 | .1 | 1/1/1 |

GAM adaptive=false。Flowers的rho/normrho/gamma是为新三seed组显式选取的冻结实现默认值，旧日志/YAML未记录，不能宣称已证明与86.75那次完全一致。四个历史GAM组的旧分数均留作历史参考。

### 随机数与有效参数

1. 全新三seed组在启动前设置 `PYTHONHASHSEED=s`；核查 Python random、NumPy、PyTorch CPU/CUDA 初始化和实际DataLoader RNG路径。每个run独立进程，seed配置形式按入口要求为 `[s]`。六个条件复用组沿原已见证启动环境和播种路径，不另加未见证的环境变量；如改变则撤销复用并重跑0。
2. 可复用旧结果的六组继续原冻结seed/loader实际行为，不追加worker_init_fn、Generator或所谓deterministic开关。若要修改播种逻辑，改完对该组0/1/2全跑，撤销旧0复用。
3. 对全新三seed组，先确定并记录统一的组内seed规则、workers、persistent、pin、drop_last、精度及cuDNN/TF32设置，再运行全部三seed。参数必须到达实际代码调用位置。
4. 9/18 GAM-Het5 配置中的 `ewc_isolate_clean_rng=true` 没有被该冻结GAM learner读取；DataManager也未实现其deterministic loader开关。CL自建loader同样不因写入这个flag就改变行为。不能将它们当成已实现的隔离协议。
5. 当前CopyFlatCV的LoRA forward删过临时Linear初始化，改变CPU RNG消耗。因此冻结整个依赖闭包，禁止用live tree边改边跑。

### 复用检查与协议指纹

为每组固定：有效非seed配置 + 源码文件SHA + 环境版本 + checkpoint SHA + 数据划分/类别映射 + 预处理/loader/RNG规则。仅seed及其直接派生随机流、输出名/路径可随run改变；硬件型号与driver也记录。

六个正式旧0已有receipt、完整源码/配置/指标/日志校验。仍需核对实际权重与数据身份：旧数据manifest主要是path+label而非图片内容哈希，GAM-Het5旧receipt未绑定timm缓存文件revision/SHA。若无法建立与新运行一致的可信身份，则该组重跑0。最终分数相近不能替代来源证明；不调参追旧测试分数。

输出目录必须包含 campaign/method/dataset/seed/attempt，日志不可追加混写，checkpoint不可共用。记录GPU UUID、start/end、exit code、源码/配置/数据/权重SHA、实际参数、完整任务矩阵。

从所选完整冻结源码目录启动 `python -m src.main --config <单seed配置>`，核查PYTHONPATH及import解析均指向冻结副本，输出另写独立run目录。四基线选择完整Flatness_CV副本；EWC及新冻结GAM选择9/18 attempt2完整代码副本，沿用所选实际方法路径并显式配置本计划参数。入口和环境尚待M0实际import验证。

## B1：独立三seed运行（MUST）

每组从同一预训练checkpoint重新初始化模型、LoRA、分类头、optimizer、scheduler、Fisher/历史子空间；仅在run内部按方法跨任务继承状态。不从其他seed或其他方法checkpoint继续。不开展额外超参数搜索。

验收是完成全部任务和约定epochs并产生有限的完整CNN下三角矩阵，不要求超过某个准确率。低分正常纳入统计。遇到NaN/OOM/数据错误保留失败receipt，不能改seed凑n=3；相同配置重试要记录原因。若需要改batch/LR/精度/算法才能跑通，重建该组协议并重新跑三seed。

## B2：汇总与更新表格（MUST）

设第s次run的任务矩阵为 A，任务数T：FAA_s=mean(A最后一行)，AAA_s=mean_t(mean(A[t,0:t+1]))。不使用原JSON里可能按三角元素加权的AAA，也不将CL另换为proxy训练头评估。

每个指标独立计算 mean=(x0+x1+x2)/3，sample SD=sqrt(sum((xs−mean)^2)/2)。FAA与AAA来自同一组三个run；全精度统计后四舍五入到两位，输出 `\ExpRes{均值}{标准差}`。三seed全部有效才标n=3；缺失就如实标n，不能用任务/epoch间波动补位。

所有三seed均参与；不保留最高seed作中心值、不把旧最高分与新SD拼接。旧表另存。按用户当前要求保留加粗/下划线位置并明确是保留格式；如果后续采用按新均值排名的规则，应全表一致更新。记录n=3不等于统计显著性结论。

## 执行顺序与里程碑

| 阶段 | 工作 | 判定门槛 | 运行数量/成本 |
|---|---|---|---|
| M0 准备 | 冻结代码/环境/权重/数据；配置解析、import、任务边界、参数消费检查 | 35组有效配置可解析、源码入口完整、输出独立 | CPU为主；尚未完成 |
| M1 小规模验证 | 每种实际代码路径做2任务×1epoch smoke，覆盖Inf子空间、CL第二步、EWC Fisher、GAM第二任务 | 无NaN/OOM，任务/类别/指标及RNG规则正确 | 建议9次：4基线、CL、EWC wrapper/native、GAM新冻结/旧Het5；不计入正式99 |
| M2 首批正式 | Vanilla/Flowers s0；CL/Flowers s1；EWC/Flowers s0；通过后尽早启动GAM/Het5 s1 | 检查完整输出与真实耗时，不按分数筛选 | 3次首发+1次长任务，均包含于99 |
| M3 全量 | 先完成各新冻结组s0结构验证，再排1/2；长Het5优先填空闲GPU，其余按数据集交错 | 每组冻结版本不再改；已有配置组可独立推进 | 总99正式run，6个旧0可能额外重跑 |
| M4 统计 | 校验35组来源与105个有效run、重算均值/SD、编译表格 | 恰好0/1/2，同组配置一致，全矩阵finite | CPU汇总 |

每卡先一个进程，按可用GPU UUID调度。CL历史显存峰值约12.8GiB，GAM-Het5约11.2GiB（PyTorch计数，不是整卡占用）；先smoke测峰值再决定可用卡，OOM不能静默降batch。本会话本地nvidia-smi无法连接驱动，未取得当前空闲GPU清单；不能给出确定开跑时刻。

## 计算预算

历史日志/receipt时长是预算参考，不是统一硬件上的实测GPU计算小时。四基线20次日志区间合计33.31小时，推荐60次线性参考99.94日志区间小时；日志不能覆盖首条记录前的启动/排队，区间内等待和停顿未分离。其余方法逐组时长见 `BUDGET.json`。99次的混合历史时长机械外推合计约229.20小时，预留20%–30%后为275.04–297.96小时，仅为排程量级参考，不能称测得GPU小时。执行时以首批新run的进程receipt、GPU UUID及独占情况重新估计占用时间。

实际墙钟下界=max(总GPU占用小时/可用卡数，最长run)，排队、I/O和GPU异构会增加时间。规划预留20%–30%用于启动、I/O及失败排查，不能把日志线性外推当交付承诺。不给出未经实测的精确完工日期。

## 范围外的可选实验

- 跨方法严格统一 Aircraft LR、Het5 epochs、cosine最低LR、loader与同一数据/权重版本：另起全方法0/1/2协议，最多105次；不与本次复用结果合并。
- CL额外训练/保存/推理代价：可随正式run记录wall time、peak memory、模型/adapter字节数，但本轮不追加消融。
- 额外seed3/4、任务顺序变化、参数扫描均不在必须范围。

## 交付与检查

- `COHORTS.json`：35组原结果、配置来源、名义缺失seed与推荐处理。
- `RUN_MATRIX.csv` / `EXPERIMENT_TRACKER.md`：105个目标seed记录；99待新跑、6待核验复用，全部未启动。
- `TARGET_PROTOCOL_DRAFT.json`：拟冻结协议与不可静默改变的字段。
- `baseline_plan.*`、`ewc_plan.*`、`cl_gam_plan.*`：逐方法依据、源码问题、历史时长。
- `PLAN_COUNTS.json` / `BUDGET.json` / `VALIDATION.json`：数量、粗预算和结构校验。

当前完整来源固定、有效配置动态校验和GPU smoke均是待执行准备工作；本轮没有生成已验证可直接启动的launcher，也未宣称新实验已完成。
