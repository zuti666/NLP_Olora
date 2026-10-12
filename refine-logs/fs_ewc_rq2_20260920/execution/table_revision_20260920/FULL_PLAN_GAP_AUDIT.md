# 原附件完整计划与当前完成范围核对

核对对象：用户附件 `pasted-text.txt`（2026-09-20 08:25:49 UTC 版本），服务器 `EXPERIMENT_PLAN.md`，当前 `campaign_plan.json`、实际诊断/训练代码及原始 summary。此审计只读分析，没有修改队列、配置或执行 GPU 作业。

## 结论

**应补齐附件的核心设计，才能得到完整的四格机制表；但新四格表不必等待 B1/B2 全部完成。** 表格由 B3 提供；B1/B2 为邻域和真实路径的机制主张提供独立证据。当前 15/15 完成，准确含义是“精简的 seed0 首批队列已完成”，不是“原附件完整实验已完成”。

原附件要求 B3 无保护/有保护 × 普通/shaped 起点四格，服务器后来版本的 §5 则明确缩为两条 lambda2000 分支，并把无保护对照、完整四格设成 CONDITIONAL；B2 中间 t1/t5 路径同样从原附件必做变成服务器版本可选。这是实验范围不同，不能以队列全部通过填平缺失格子。

## 新四格表的最小需求

| 附件格子 | Task0 shaping | Task1 优化器 | Task1 EWC λ | 当前状态 |
|---|---|---|---:|---|
| A | off | SGD | 0 | 缺，需从现有普通 Task0 续训一个 Task1 |
| B | on | SGD | 0 | 缺，需从现有 shaped Task0 续训一个 Task1 |
| C | off | SGD | 2000 | seed0 已完成；当前 A_sgd_to_sgd_seed0 |
| D | on | SGD | 2000 | seed0 已完成；当前 A_gam_to_sgd_seed0 |

每行应报同口径 a11、a21、a22、BWT=a21−a11、固定旧 head 的 train/test base/final CE 与 D0、有效权重位移、Task0 anchor/Task1 endpoint 上旧任务 raw Fisher trace。当前 C/D 已经有 Fisher 与参数移动信息，无需为了填已有字段重训。

C/D 的 raw Fisher 都是 **旧任务256个固定 train 样本、旧20类 local head、有效 Q/V 权重、逐样本经验梯度、不归一化**。endpoint trace 表示旧任务 Fisher 在新 endpoint 的值，不能直接命名成没有任务定义的“Task2 Fisher”。位移为 `||W_eff(t1)−W_eff(t0)||_F`，不是 A/B 因子范数，也不是未经核验的“新 adapter 范数”。

| 已有行 | a11 (%) | a21 (%) | a22 (%) | train256 raw T0(anchor) | train256 raw T0(endpoint) | effective-W 位移 |
|---|---:|---:|---:|---:|---:|---:|
| C：普通起点 → SGD+EWC | 85.66775244 | 84.36 | 74.80 | 1465.49692690 | 2008.38787326 | 0.37494957071 |
| D：shaped 起点 → SGD+EWC | 87.62214984 | 84.04 | 72.05 | 761.90077432 | 940.41253231 | 0.34987579697 |

来源：`report_seed0/A_two_task.csv`、两方法 `results_seed0_recovery1/*/table_a/summary.json`。accuracy 原汇总保留两位小数 a21/a22；正式 BWT 宜用同一次评估的精确计数重新汇总。

**附件四格不等于用户粘贴的旧六行表。** 旧六行包含后续继续 shaping 的分支；附件 B3 固定 Task1 SGD，专门隔离起点/保护作用。当前主数据为 ImageNet-A，不能用新数值替换旧表仍保留 ImageNet-R 标题。若坚持重做同六行 ImageNet-R 表，需要单独统一该数据集和六种阶段开关；执行附件四格并不会自动产生那六行。

## 逐块完成情况

| 原附件需求 | 当前实证 | 尚缺 |
|---|---|---|
| B1 paired dense loss：0:.25:20、K100、3 seeds | seed0 的7个半径、K16 paired 点；另有历史独立方向 dense 重分析 | 正式K100 dense，所有3 seeds；旧曲线不能充作新paired曲线 |
| B1 train sparse raw Fisher：97 states/anchor、256样本 | seed0 两 anchor，194状态完成 | seed1/2四 anchors；r0/1/10/20固定方向子集全1259样本核对 |
| B1 test raw Fisher：r0/1/10/20、K16、test307 | 有test loss/accuracy，没有test梯度计算 | 所有规定test raw trace；这是test损伤反例的重要解释检查 |
| B1真实方向交叉 | seed0两anchor×两t9来源方向，r={0,.25,.5,1,1.5,2} | 补真实范数约1.197584与1.400473两个半径、同半径随机对照、其他seeds |
| B2全轨迹loss/accuracy/drift | seed0两历史链t0..9共20 endpoints，以当前FP32复评 | seed1/2链；范数平方可CPU派生 |
| B2 endpoint raw trace | t0/t9有train256 trace；新的两任务续训endpoint也有trace | 全历史t0..9的train1259 trace；test t0/1/5/9 trace；新续训t1不能冒充历史链t1 |
| B2 t1/t5/t9路径 | seed0两条t9九节点路径，256子集，求积收敛 | t1/t5；正式1259样本；其他seeds；按需16/32段加密 |
| B3四格 | seed0 C/D两格、50轮、配对检查通过 | seed0 A/B及seed1/2全四格 |
| M4 full-model支持性复验 | 未在此队列中执行 | 条件项：论文声称跨数据集/参数化时才升级为必须 |

每个 anchor 的当前 neighborhood 有97条原始状态（r0+6×16），path有9状态。trajectory 的11条journal包含额外复用的r0，实际task endpoint数是10；不能把11条journal误解成11个任务。桥接11条状态是r0+两方向×五非零半径。

M0核心恢复、有效坐标、逐样本梯度与方向导数、端点、配对和计时sanity已通过。M1 seed0稀疏trace/固定桥接/t9先导路径已完成。M2只完成全endpoint loss/drift，dense、全样本F和t1/t5仍缺。M3只完成seed0四格中的两格。M4条件执行。因此最准确的进度描述是“seed0先导诊断和两条保护续训完成”。

## 实际代码需要补什么

- `train_continuation.py` 的 build_config 当前硬编码 `ewc_lambda=2000`，且 continuation run目录仅区分起点。补A/B必须增加显式protection参数、在run标识中区分lambda，避免覆盖C/D。
- `endpoint_a.py` 同样 assert `ewc_lambda==2000`，正式四格需按期望branch校验，不应简单删掉溯源断言。
- `diagnostic.py::measure` 只对 `x['subset']` 调 Fisher；train/test metrics只是全数据前向。test trace和full1259 trace需新的明确采样范围与结果身份。
- `Runner` 目前只构造t0/t9差值；`path()`不使用配置里的 `path_task` 来选择目标。仅将配置从9改成1或5不会生成所需路径。
- 路径目前最多加密到17节点；原附件需要不稳定时继续到33节点。t9先导结果已有稳定凭据，不因未来加密能力缺失而作废。
- dense loss应有无梯度的执行分支；不可把现有K100×81半径全部调用当前 `measure()`，否则无谓地对全部dense点算256梯度，违背原附件“dense loss / sparse trace”的成本设计。
- 当前逐方向原始记录足够CPU派生raw-trace Q95/max、sampled-max damage、范数平方和trace上界；缺少汇总列不等于必须重跑GPU。

## 复用与预算

以下是50epoch的“单任务训练段”数量，不是GPU小时：

1. **先得到seed0完整四格**：只需新增2个Task1训练段（lambda0 A/B）；两个旧Task0 anchors和现有C/D全部复用。
2. **只完成正式3-seed四格表**：若seed1/2尚无匹配anchors，新增4个Task0+8个Task1，加上seed0缺的2个Task1，合计14段。完成后每格n=3，报告每seed和mean±sample SD；16/100个随机方向不能当训练seed重复。
3. **完整3-seed LoRA B1/B2/B3**：若复用现有seed0两条历史十任务链，新增seed1/2共4条十任务链=40段；同新协议的SGD链Task1可复用四格C，新seeds其余A/B/D共6段，加seed0 A/B共2段，合计约48段。这里假设checkpoint、初始化、数据顺序、精度、源码及超参一致且通过复用核验；不满足则分支重跑另计。历史seed0与新训练协议的差异需要分层报告，不能悄悄合并为同协议统计。
4. 当前在 `/data/140-1/users/liying/CopyFlatCV/outputs_logs` 递归搜索未找到路径含seed1/seed2且名为task_00/01/09.pt的文件；本结论限定该资产根，不代表服务器其他目录不存在。仅凭旧“seed1/2已做”文字不可直接复用。

原附件正式诊断量仍适用：dense loss共96,012次dataset forward；train sparse共148,992个逐样本梯度；test sparse共90,258个；B2 train endpoints+3paths共234,174个；B2 test endpoints共7,368个。它们是未做交叉块去重的计划数量；当前seed0稀疏已完成，r0/endpoint/相同路径节点应复用。

有效去重条件是checkpoint hash、样本集合及顺序、变换/精度、loss/head、参数坐标和状态完全相同。256→1259不能把已有聚合trace直接当全样本trace；可新增其余1003样本梯度后按样本数精确加权，但需要代码显式保证可组合协议。t0一次梯度可计算多目标方向q，endpoint与path端点可共享。避免为t1、t5、t9分别训练重复序列。

GPU并行顺序可继续沿用既有01/23分工：01跑现有seed0离线补测，23运行缺失的四格续训/新seed主链；新anchor一出即可触发独立B1，新Task1一出即可填表，t9产生后再触发对应B2长路径。不存在必须所有阶段全局串行的依赖。

## 如何判断支持

补全设计保障可检验性，不保证新表方向有利。现有D相对C的旧任务train损伤/trace更低，但test损伤改善幅度较小、新任务准确率低2.75个百分点；不能预先承诺补A/B或多seed会恢复旧表的正向故事。四格用于区分起点作用与保护作用，B1/B2用于判断邻域和真实路径机制，所有负面test/seed结果应完整保留。

当前全部15作业于2026-09-20 15:36:08 UTC完成。原始结果未改；该审计输出JSON保存来源指纹、状态映射和计算预算假设。
