# 一表两图 RQ2：执行完成与机制闭环核查

核查日期：2026-09-20。输入为本轮完整附件、`execution/`、`mechanism_next_20260920/`。用户已明确 **当前视觉 GAM 就是 FS-EWC 的优化实现**，本次沿用该定义，方法命名不再作为实验前提或待补事项。本文未启动、停止或修改任何 GPU 队列。

## 结论

**一表两图的组织已经具备，seed0 的主体测量也已完成；但附件所要求的完整正向机制链尚未闭环。** 其中既有真正未执行的补测，也有已经测出但不支持预期的结果。二者不能混为一谈，增加实验数量并不保证链条最终成立。

可以用当前结果回答一个较窄的问题：在选定有效权重坐标下，FS-EWC 的训练侧随机邻域敏感度较低；其起点沿两条已测真实方向产生较小测试损伤；实际完整序列存在更大位移、更小损伤与较低末任务端点直线段敏感度。尚不能把它写为各方向普遍扩大的测试保护区域，也不能从三块结果直接推出纯起点 shaping 的因果中介链。

## 1. 四种“完成”分别是什么状态

| 层次 | 状态 | 证据 |
|---|---|---|
| 论文组织 | 已具备 | 一张 A 表、B 邻域图、C 实际序列图及 appendix bridge 均已输出 |
| 已排 seed0 队列 | 已完成 | recovery1 的 15/15 receipt 均 complete 且 SHA 与状态文件一致；最终报告时间15:36:08 UTC |
| 附件正式实验范围 | 未完成 | 当前完整 B/C 仅一个训练 seed；directed search、seed0 t1/t5 path 及更多 seeds 尚无完成结果 |
| 正向科学主张 | 部分支持 | B 的测试 Gmean 未改善且 Gmean tolerance 相同；A 当前主配置不是附件引用的正结果；C 的末任务方向证据较强 |

`mechanism_next_20260920` 是已有结果的 CPU 组织、图表、独立主张评估和接续规划；其 `B-test/B-directed/C-mid` 仍为 proposed。`followup_spec.json` 是 planning-only，不是执行回执。

新 `execution/full_rq2_20260920` 在本次核查期间开始了两条历史 seed0、λ=0 的 Task1 续训。它们服务于原四格表缺失的无保护分支，并不是 B-test/B-directed/C-mid 的执行记录。最终实时快照另见 `CURRENT_EXECUTION_SNAPSHOT.json`；本报告对科学结果的判断以已完成且有数值摘要的产物为准。

期间还完成两项 smoke：SGD 的单方向 r=0/.25 loss测量，以及 GAM 的 r=0 train256 rawF复现。它们证明新入口的小范围执行可用，不能算完整邻域、test rawF、directed search或新路径的正式结果。原pilot后来被标为superseded，替代的`pilot_queue_paired`接续同两条λ=0补训；状态以最终快照为准。

## 2. 附件正面 A 例子与当前 A 不是同一实验

附件中的 `19.885→10.946`、`87.95→91.82` 确有原始日志支持，不能称为未溯源。但它来自 **ImageNet-R、fresh K/V LoRA、Adam、λ=10^7、seeds0/1/2**；当前 B/C 是 **ImageNet-A、持续更新 Q/V LoRA、SGD/GAM、λ=2000、seed0**。

| 项目 | 附件引用的旧 ImageNet-R pair | 当前 ImageNet-A pair |
|---|---|---|
| 后续共同规则 | Adam+EWC，λ=1e7 | SGD+EWC，λ=2000 |
| a21，普通/Task0-shaped | 87.9467 / 91.8233 | 84.36 / 84.04 |
| a22，普通/Task0-shaped | 83.4500 / 82.5667 | 74.80 / 72.05 |
| 起点 Fisher，普通/shaped | 19.8852 / 10.9458，batch proxy | 1465.497 / 761.901，train256逐样本raw |
| 第二个 Fisher 列 | 新任务自身数据/head的batch proxy | 同一旧任务在Task1续训端点的raw trace，2008.388 / 940.413 |
| 旧任务 loss damage | 日志未保存续训后的旧任务loss；无匹配checkpoint可直接补算 | 已有；test −.019246 / −.013388，train +.003674 / −.002157 |
| 重复 | 3 seeds的均值，日志seed配对不等于严格RNG配对 | 1 seed；后续新head/数据顺序/首batch增广收据配对通过 |

旧 ImageNet-R 的 a21 优势约3.877pp，起点 a11 已高2.617pp；扣除起点的 BWT 改善为1.26pp，同时 a22 均值低0.883pp。可以作为其配置下的历史辅助结果，但不是当前 ImageNet-A 主表的数值，也不是无代价改善。

**Fisher 定义不可拼接。** 旧表是 batch mean 梯度先平均再平方的 proxy；当前诊断是每例梯度先平方再平均。旧 F2 测新任务，而当前机制表需要旧任务在两个端点的敏感度。

**“固定保护”需要改写。** 两行使用共同的后续优化器、λ和 EWC 公式，但各自继承的权重、旧 head、Fisher 矩阵不同。因此只能说“固定后续学习规则，干预继承状态”，不能说保护矩阵完全相同、已经隔离纯几何效应。

**不能简单给旧表补 D1。** 原 cohort 日志 `save_ckp=False`，未发现匹配旧 checkpoint；第二任务 `endpoint_base_loss` 测的是新任务，不是旧任务。不能拿两种任务的 loss 相减，也不能把当前 ImageNet-A 的 D0 填进旧 ImageNet-R 表。复用当前 A 则无需补这项训练或测量，因为 D0（论文编号 D1）已经存在。

来源：[旧表溯源](../../execution/table_revision_20260920/USER_TABLE_PROVENANCE.md)、[Fisher定义](../../execution/table_revision_20260920/FISHER_DRIFT_AUDIT.md)、[当前A原始CSV](../../execution/report_seed0/A_two_task.csv)。

## 3. 一表两图各自是否支持目标

### Table：What does shaping change?

可以展示 shaping 改变了继承状态的起点准确率、raw sensitivity、后续损伤、位移和新任务表现。当前 ImageNet-A 的完整端点诊断已在 `A_two_task.csv`；无需为了补指标重训。

但 current pair 的 GAM-origin a21/a22 更低，位移也更小，因此该表不能作为“保持更好且不是靠少移动”的正证据。保留负结果，将主题句写为 **We compare inherited states under a common subsequent SGD+EWC rule.** 若使用旧 ImageNet-R 正面 pair，要明确独立的数据/参数化/proxy口径，不称同一条模型轨迹上的第一环。

当前排版表只显示起点 trace，虽然 CSV 已有续训后旧任务 trace。若严格采用附件表格清单，可从 CSV 增加 `raw_trace_t1_train256`，无需 GPU。表注用 `T_old(before)` / `T_old(after)` 避免与“新任务Fisher”混淆。

### Figure A：Neighborhood sensitivity and empirical radius

已有 2 anchors、16 公共随机方向、r=0/.5/1/2/5/10/20 的 Gmean/G95 和 train256 raw trace；现有原始 states 已支持 mean/Q95/sample max 的重算。

- **支持**：所有测量半径上的 train256 raw trace 更低，约为普通起点的47%–52%；train 平均正损伤也更低。
- **不支持一般测试保护区域扩大**：test Gmean 在全部六个非零半径更高；test Q95虽有5/6个半径略低，test Gmean和Q95容忍半径在当前ε/半径网格上均相同；train Gmean半径也相同。
- **额外方向证据已经有**：bridge 在相同半径、相同来源方向下比较两个anchor，GAM anchor 的 test damage 在两方向×五个非零半径均更低。r=1时，沿SGD方向 .14752→.11183；沿GAM方向 .01892→.01345。固定anchor比较方向还显示强方向效应。
- **不能替代的缺口**：bridge方向来自真实Task0→Task9位移，不是方法inner-search产生的sensitivity direction。两个真实方向的较好耐受，不等于随机方向均值更好，更不等于整个球内worst-case更低。

因此 Figure A 当前支持 **训练侧随机邻域敏感度降低 + 已测真实方向的损伤较小**。附件中广义的“empirical tolerated radius increases”一环没有建立。可以在同一张图中保留随机对照并加小型bridge inset，无需再加第四块主实验。

### Figure B：Realized future trajectory

当前全10个任务checkpoint的old-task accuracy、test loss damage、effective displacement均已重新评估。Task9普通/GAM结果：

| 指标 | SGD+EWC | FS-EWC（视觉GAM+EWC） |
|---|---:|---:|
| Task0固定20类head的test accuracy | 80.1303% | 85.9935% |
| Task0 test loss damage | .260131 | .034347 |
| 实际累计位移 | 1.197584 | 1.400473 |
| train256端点直线段S_path | .368058 | .014793 |

这是当前最强的机制关联：更大位移仍有更小损伤，并伴随更低单位位移方向积分。各t=1…9端点均有更大位移和更低test loss damage。只有t9已测 S_path，不能从它补画全部任务的敏感度曲线。

这里的历史完整 FS-EWC 链未来任务也继续使用 GAM；A 的 controlled pair 则共同使用 SGD。C支持整体方法的真实序列表现，不能将全部收益归因于只在Task0 shaping。积分沿真实端点之间的直线段，不是训练每一步走过的路径；求积和同样本D²≤I检查放附录，不能拿检查通过代替方法比较。

## 4. 两份接续配置尚未完全一致

| 内容 | mechanism_next 提案 | full_rq2 新实现 | 当前结果 |
|---|---|---|---|
| test rawF 半径 | 0/1/5/10/20，16方向 | `sparse_test` 默认0/1/10/20 | 未执行；正式启动前统一是否包含5 |
| t1/t5 path 样本 | 与现有t9保持train256 | `path_task`实现使用full train1259 | 未执行；不能把新1259和旧256当同口径曲线 |
| directed probe | 实际GAM inner-search，转effective方向交叉测量 | 暂无对应job；`bridge_full`是实际位移方向且forward-only | 仍缺实现/执行 |
| 正式seed重复 | 复用同配置seed0，补匹配seed1/2 | 预检讨论fresh0/1/2；目前仅历史seed0 λ0 pilot在跑 | 未形成正式三seed B/C |

这些是具体的实现/运行口径差异。无需新增实验种类；应先固定同一配置，再执行已决定的少量测量。

核查期间新增的 `full_rq2_20260920/formal_plan.json` 是一份更大的正式配置：**88 jobs，21个训练job/69个task-training segments，54个diagnostic、12个endpoint、1个report**；19:11 UTC时尚无`formal_queue/pipeline_state.json`。该配置包括所有3 seeds的t1/t5/t9路径和完整四格，超出附件“全部seed只需t9、seed0再补t1/t5；四格条件化”的最小范围。它仍没有GAM inner-search directed job。配置覆盖扩大与机制闭环是不同问题；本轮只指出差异，不改动正在并发开发/运行的队列。

## 5. 最小接续与停止条件

1. **先固定证据范围。** 主表与两图使用当前 ImageNet-A 主配置；旧 ImageNet-R 表单列为辅助观察。若坚持以旧表作核心loss-based干预证据，必须先找回端点或另行复验，不能伪补D1。
2. **B补少量GAM-generated direction probe，并补test rawF解释分歧。** 沿用已决定半径、共同方向和独立训练search集合，保留不利结果。test rawF是针对目前train/test分歧的解释性补测，不是新一轮estimator比较。
3. **C补seed0 t1/t5，并固定与t9同一Fisher样本集。** 若改full train，则t9也要按同口径测量；不将不同样本集混成机制演化曲线。
4. **按附件正式范围补训练seed1/2的匹配B/C证据，每seed至少t9 path。** 有匹配checkpoint则复用；没有时才需要训练。K16不能替代3个训练seed。已有seed0 t1/t5之外，不默认扩展每seed全部路径。
5. **对容忍半径作结果分支判断。** 若保留更大半径主张，使用预先固定的公共细网格/ε，并在新seeds确认；若仍不增加，删掉这一环，改为方向相关的耐受机制。不要挑有利半径、阈值或只展示train。

λ=0当前补训可提供额外的保护开关对照，待完成后再分析。它不是B/C补测的前置依赖，也不会自动修复测试邻域半径未改善。完整四格、多种Fisher估计器、full-model全路径均不作为本轮自动MUST。

## 6. 现在可以写到什么程度

**可用的一表两图结论：** 本案例中 shaping 改变了继承状态的经验敏感度与后续学习表现；训练侧随机邻域 Fisher 较低，固定真实方向下的旧任务测试损伤也较小；完整序列中更大的实际位移伴随较小损伤及较低末任务方向积分。随机测试邻域并没有呈现相同的平均损伤/容忍半径优势。

**暂不能写：** 已经证实 shaping 通过扩大一般损伤容忍区域，因果性地导致更少遗忘；已完成附件要求的三seed、directed probe与多阶段path；训练侧raw Fisher低必然意味着测试损伤低。

独立主张复核见本目录 `CLAIMS_FROM_RESULTS.md/json`；数值与来源检查见 `EVIDENCE_PACKET.json` 和 `evidence_precheck.json`。审查标签为same-family/provisional，不以数值来源存在代替语义支持。
