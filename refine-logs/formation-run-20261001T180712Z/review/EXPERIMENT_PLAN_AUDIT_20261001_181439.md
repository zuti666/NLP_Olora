# Formation 实施计划独立审计

**审查范围**：仅设计、已有证据与后续 CPU 代码审查，不启动训练，不修改 trainer/queue。用户附件与 USER_PLAN.txt 字节一致，SHA256 为 `6cff1af4fe60ed501104a3f18c33de29543c7daf9c7dbcd5add6aac3d9d30383`。

**结论**：可按下列配对约束部署第一阶段 8 arms × 3 seeds。这里通过的是设计审查，不表示尚未收到的 trainer/future-probe 代码已经通过。第二、三阶段为结果触发，不应在第一阶段 queue 中无条件追加。

## 1. 主要科学问题与解释边界

主要检验：在匹配参考 S 的完整有效步长序列后，G formation checkpoint 在相同未来 S 学习规则、数据、随机输入及训练预算下，是否显示较低的 Task0 损失增加；匹配当前任务获取质量后是否仍存在。

对照解释：原 G 优势是否由沿 g1 放大步长即可重现。S1.5/S2 或 Gmatch 与 S 接近只是支持某种解释的证据，不能把“不显著”写成“等价”，也不能从一个局部步长统计完全排除优化尺度解释。

相同未来学习规则不等于相同未来数值更新向量：未来 g1 随起始模型状态而改变。这是本实验研究的 state susceptibility；除非另有跨状态施加同一个向量的实验，不宣称隔离了纯 Hessian/LoRA 几何。

已有 seed2 Task0 半径结果非严格单调，且只有一个 seed；不把附件中的“radius .25 之后平台”作为已确认事实。r=.25 只预先指定为稳健性对照，不继续按新 reporting 结果选 radius。

## 2. 第一阶段可立即执行的矩阵

| Arm | 形成规则 | LR 倍数 | 半径 r | 用途 |
|---|---|---:|---:|---|
| S | 原生 GAM g1 | 1 | 1 | 主参照及逐步 delta 来源 |
| S1.5 | 原生 GAM g1 | 1.5 | 1 | g1 步长对照 |
| S2 | 原生 GAM g1 | 2 | 1 | g1 步长对照 |
| Sum | g1+g3 | 1 | 1 | 等价性参照 |
| Mean2 | (g1+g3)/2 | 2 | 1 | 与 Sum 的实现等价性控制 |
| Graw | g3 | 1 | 1 | 连接旧现象 |
| Gmatch | g3，逐步匹配同 seed 参考 S 的 delta | 求解实际系数 | 1 | 主要机制对照 |
| G025 | g3 | 1 | .25 | 指定 radius 对照 |

全部 seed=[0,1,2]，Task0 20 类，class_shuffle=false，固定原类序、rank=10、rho=.05、gamma=0、无投影、momentum=0、weight_decay=0、无 clipping、无 Fisher/EWC/fusion。S 是已有 GAM 定义中的 g1，包含它原有的 rho probing，不是原点 CE 梯度 g0。正式配置必须显式冻结这些值。

每组 50 epochs，预期每 epoch 10 updates，共 500；先由实际 DataLoader 长度校验后把预期总步数写 manifest，不能用硬编码 500 掩盖数据/末批差异。保存完整 epoch 0/10/20/30/40/50 checkpoint。每个 seed 的 8 组从同一初始完整模型副本出发；epoch0 参数散列必须相同。

启动依赖：同 seed 的 S 完成、receipt/source/data pins 与全部 step delta 通过后，才允许该 seed 的 Gmatch。其他 arms 无需等待 S 训练完成。

## 3. 两个必须先成立的实现控制

### 3.1 Gmatch 的参照是另一条完整 S 轨迹

每个 t 读取同 seed S 的 delta_t = ||Psi(theta_after^S)-Psi(theta_before^S)||，Psi=concat(unscaled LoRA BA, FC weight, FC bias)。delta 由真正执行的 SGD 前后计算，不能用 eta*gradient norm、线性化 BA、G 当前状态的 g1，或少数 checkpoint 插值代替。

记录 step_id、epoch、batch index、输入/标签/样本索引 hash、LR、delta、方向与参数 finite、reference trace hash。Gmatch 的 step t 与参照 t 必须 batch/aug hash 一致。根求解使用实际将执行的参数 dtype 和 BA 映射；执行后再核实际 delta。每步残差相对误差 <=1e-3，target=0 时必须实际不更新；绝对 tiny-target 容差需 manifest 提前声明，不能用相对除零绕过。

找不到正系数、不可行、迭代不收敛、reference 缺步/重复/乱序/seed错/任一 fingerprint 不匹配均 fail-closed；不得截断、clip target、fallback raw G 或继续记作 matched。匹配整个 Psi 范数不意味着 LoRA/FC 每个块、曲率或完整轨迹长度以外所有尺度均相同，需分别附报 LoRA 与 FC 步长。

### 3.2 Mean@2eta 等价必须覆盖全 schedule

每一步 LR_Mean2(t)=2*LR_Sum(t)，包括初始 LR、cosine eta_min、epoch推进顺序和 T_max=50；只翻倍 initial_lr 不足以成立。两者相同初始化、每步输入/augmentation、g1/g3 probes、参数恢复、SGD/momentum/wd/clipping；不能从 sum 直接载入模型伪造独立对照。

CPU 必测同状态下 Mean2 更新 == Sum 更新、四个 probe/restore 一致。正式轨迹比较每步 LR、输入、loss、grad/update 和 checkpoint；精度容差必须由独立数值控制预先固定并记录，不能看结果后放宽。CPU 同状态等价通过不保证不同 GPU 上长轨迹逐位相同；若出现明显超出预声明误差的差异，先停止机制解释并定位实现/数值差异，不能称为几何作用。只比较最后 accuracy 不足以通过此控制。

## 4. Seeds、split 与 matched acquisition

### 4.1 固定划分

Task0 官方 heldout 307 条固定拆为约 1/3 selection diagnostic、约 2/3 reporting。建议目标 102/205，先按原始类别分层，再按 `SHA256(split_seed | original_class | canonical_sample_id)` 确定类内顺序；配额用确定性的最大余数法贴近 102，各 n>=2 的类两侧至少一条，singleton 放 reporting 并在 manifest 标记。若约束导致非102，记录实际计数，不强行掉样本。所有 arms、seeds 使用同一名单。

manifest 固定 split_seed、样本 canonical ID/原 label/remapped label、各类计数、selection/reporting list hash、全集 hash。两侧互斥且并集恰为全集，禁止重新抽取以获得更好效果。Task1 progress matching 也应在其官方 heldout 用同一确定性规则另建 selection/reporting，不能拿 Task1 reporting 来选择匹配进度。

此拆分只是防止本轮选择与报告直接混用。先前 seed2 已在完整 heldout 上观察和提出假设，不能追溯声称 reporting 是从未见过的独立外部测试集。seed0/1 是新随机运行，使用冻结规则可提供新的 run-level 证据；seed2 仍需单列探索性历史暴露。

### 4.2 提前冻结匹配规则

建议主 matched-acquisition 分析以 S epoch50 为固定参照，对每 seed、每 G arm，在 {10,20,30,40,50} 中仅用 selection Task0 pre-expansion CE 选择候选。推荐可操作 admissibility：|CE_G-CE_S|<=.01 nats/example，且正确人数差<=1；候选按 (CE差、正确人数差、距离50的epoch差、epoch值) 排序。阈值是预声明的 pilot 操作标准，不是功效分析支持的等价界限。

没有合格候选就记 `no_matched_acquisition`，不要用 reporting 偷换 epoch、插值权重或扩大阈值。固定 epoch50 分析仍完整报告，但不能称为相同获取质量的证据。可以预先另设双边 checkpoint 匹配作次要探索，但不得事后替代主规则。

匹配完成后先写带 hash 的 selection decision receipt，再解封所选 checkpoint 的 reporting 表；所有 K 共用这一个选择。pre-expansion 仅20类时 local=allseen，应数值一致；不把新头扩展的 K0 CE 用作 formation acquisition 匹配。

### 4.3 配对统计单位

同 seed arm 差值为一个独立训练重复；主表列 seed0、seed1、seed2 及 3-seed paired mean/SD，另列仅新 seeds0/1 的描述结果。禁止把6个 formation epochs、4个 K 或100步当扩大 n。n=3 不支持强显著性结论；3个全同号的二项符号检验双侧 p 仍为 .25。核心阶段若有信号，后续5 seeds 是用户建议的扩展，不在当前 queue 自动添加。

## 5. Future probe 的状态、头部和路径约束

对每 (seed, arm, formation_epoch) 从只读完整 checkpoint 独立 fork 一条100-step Task1 S 轨迹，在 K=0,1,10,50,100 取同一路径节点。不能在 K1 后再训练10/50/100步，也不能从 epoch10 probe 末尾继续跑 epoch20。不同形成 epoch 使用同一个 Task1 minibatch/augmentation sequence 与新分类头初始化 seed。

GAM原生 g1/rho=.05 与全部未来 optimizer/schedule 完全一致。Task1 实际每epoch8 batches时100step为12.5epochs，按实际已完成epochs推进 cosine T_max=50，不能把100step压成50epochs。每步写 actual LR 与数据hash；模型 stochastic layers 要禁用或显式配对，worker RNG 与模型 RNG 相互隔离。

头部扩展必须先记录 pre_expand 的 Task0 20-logit CE/accuracy，再扩成40类，保持旧 head rows、LoRA 与旧 logits exact，新20rows在同seed各arms/checkpoints中exact相同；随后记录 post_expand K0 的 Task0-local 与 Task0-allseen。后续 susceptibility = metric(K)-metric(post_expand K0)，分别对local/allseen计算，头部扩展冲击另列，不能混入未来训练破坏。

各节点同时报 Task1 获取 CE/accuracy，Task0 metrics 使用固定 reporting 集并记录正确人数。旧 local 的 logit mask 和 allseen 的 40类 mask 分开。Task0 local 与allseen 不同只在扩头后正常出现。

I=sum Delta_k CE 的望远镜恒等必须通过。若每个真实更新后都评同一固定集，可报精确 I_plus=sum max(Delta_k,0)；若仅 K=0,1,10,50,100 测量，则只能报 `coarse_I_plus`，它是净损失上升总量的下界，不是逐步累计破坏。不同sample/augmentation评估不能相加冒充同一路径量。

核心 probes 为4 arms(S/Graw/Gmatch/G025)×3 seeds×6 checkpoints=72条100-step路径。为检验单纯步长解释，再做 S1.5/S2 的 epoch50×3 seeds=6条 endpoint probes。Sum/Mean2 无需所有checkpoint额外future训练来验证等价。若实际统一跑全部8arms的6 checkpoints，应在预算中显式记录144条路径。

## 6. 条件推进与可证伪标准

技术完整性 gate 和科学 go/no-go 分开。技术错误是 invalid，不是科学负结果；没有信号/数据不足是 negative 或 inconclusive，不允许静默调整目标。

**Phase1→Phase2 的建议 pilot gate（不是显著性检验）**：
1. 技术门槛全部通过：完整参考trace匹配、Sum/Mean2控制、seed/data/head/split pins、未来K0与100step完整、finite/restore/source hash。
2. 主固定终点 D_s = [L0_allseen(K100)-L0_allseen(K0)]_Gmatch - 同量_S，使用epoch50、reporting；新seed0/1均 D_s<0，3-seed均值 <=-.01 CE，seed2完整披露。
3. 至少新seed0/1 matched acquisition 成功，所选对比的 D_s 均<0；不允许仅挑一个有利checkpoint。Task0-local、accuracy drop 作为预定义次要终点完整报。
4. 同时排除“Task1只是学得更慢”：未来K100 Task1 reporting获取、以及只用Task1 selection定点的progress-matched reporting旧任务结果不能明显支持这个替代解释。实用非劣建议预先设 new-task accuracy差不劣于一个 reporting 样本或 CE差不高于.01；无重叠学习进度则标 inconclusive。Task1样本数/整数粒度以实际 manifest 为准。

满足后可进入三seed完整两任务 factorial；若只Graw有信号而Gmatch无信号，先将结果视为步长/其他轨迹解释的证据，不进入“已确认matched formation”叙事。若阈值未满足，可停在已完成Phase1并报告；继续任何非预定探索需明确是不同问题。

**两任务设计**：SS/GS/SG/GG 中 G 主定义=Gmatch。Task0 的共同 S checkpoint可为SS/SG共享只读起点，Gmatch可为GS/GG共享，随后独立fork；这保证2×2随机输入配对，不要求为相同前缀重复训练。Task1 Gmatch 的参考delta应固定同seed SS 的完整Task1 S trace，SG/GG均用此共同时间表；不能分别以各自当前状态g1匹配，从而换了估计目标。预先清晰声明这一选择。Graw桥接与r=.25只能标次要对照。

完整两任务报告 Task1 epoch0/5/10/20/30/40/50 的旧/新 local/allseen 指标；a00与a10的差需同时拆头扩展跳变与训练遗忘。形成效应对比GS-SS，继续规则效应SG-SS，并报告interaction=(GG-GS)-(SG-SS)，不能仅按总AAA选故事。

**Phase2→Phase3**：仅当三seed GS-SS 的主要 formation loss/forgetting 对比方向一致、达到预声明实用幅度（建议CE .01、accuracy一例）且 matched acquisition/progress结论支持时进入。任一已有 seed显示相反强信号则报告异质性，不称已确认。Phase3固定SSS/GSS/SGS/SSG/GGG，与Phase2的共同前缀可只读复用然后独立fork，预先冻结未来Task2 schedule/data/head规则。

GSS在Task1有优势但Task2消失，仅支持一次转移稳健性；不能称跨任务持久状态。GS无优势但SG有优势，倾向继续规则效应；S2复制Graw且Gmatch无优势，倾向尺度解释。近似相等必须用预先实用区间和不确定性呈现，不用 p>.05 断言机制已被排除。

## 7. 旧结果的复用边界

旧seed2 Graw/G025 epoch50 checkpoint可以在单独 `exploratory_legacy` 命名空间跑相同未来probe，但旧S也必须提供同协议配对、完整初始模型/data/LR/epoch证据。旧文件无全部500step delta，不能充当新Gmatch参考，也无法补出epoch0..40 formation checkpoints。

不能只拿旧G终点与不同随机加载/初始化协议的新S比较后宣称formation。优先新24组的六checkpoint轨迹；旧输出仅桥接原现象，完全不进入“新3seed确认”分母。若新seed2 replay与旧checkpoint逐参数一致可记录为复现证据，仍不是独立seed。

## 8. 后续独立代码审查清单

formation_task0.py：参考S trace exact消费、500步末批/dataLR配对、rho/g1语义、β/gamma、native4probe恢复、实际effective BA二次项、正系数根求解和零target、Mean2整schedule、epoch0共享、完整保存/restore、同dtype实步验证；negative tests覆盖缺reference/错seed/乱step/错input/nan/不收敛。

future_probe.py：每checkpoint独立fork与base hash未变、native g1、扩头旧row不变/新row配对、K0口径、100真实更新计数、K节点嵌套、8batches/epoch推进、数据/augmentation隔离、split泄漏检测、精确或coarse I_plus标签、未来新任务进度。

CPU tests可运行，必须禁用CUDA并把所有新输出限制在review目录；CPU通过不等于GPU数值实跑通过。等待源码就绪后另写源hash绑定的独立CODE_REVIEW，不能提前写“代码验证通过”。
