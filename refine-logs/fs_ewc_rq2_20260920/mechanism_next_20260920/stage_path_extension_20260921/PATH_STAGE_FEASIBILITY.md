# 从最终柱状图扩展到逐任务路径敏感度

2026-09-21。只读检查已有代码、日志、checkpoint路径与测量收据，CPU重算现有积分和三seed汇总。本轮没有运行GPU、没有改动正在使用的训练/诊断代码，也没有把未测量的任务补成插值结果。

## 推荐的主图含义

把原C图最后的两根柱替换为两条逐阶段曲线。对代码任务t=1…9，使用同一旧任务起点W0：

\[
\Delta_t=W_t-W_0,\qquad
I_t=\int_0^1\Delta_t^\top F_0^{\mathrm{emp}}(W_0+s\Delta_t)\Delta_t\,ds,\qquad
\overline S_{\mathrm{path},t}=I_t/\|\Delta_t\|^2.
\]

显示横轴仍采用已学习任务总数T=t+1，和C图的位移、损伤、准确率面板一致。即T=2对应Task1，T=10对应Task9。每个曲线点都是“Task0→该阶段端点”的独立直线积分，使用该阶段自己的真实累计方向。固定同一Task0样本集合、20类head、有效Q/V坐标和Fisher定义。所有曲线保留真实非单调变化。

T=1对应Task0自身，位移为0，S_path是0/0未定义；该位置留空或标N/A，不能人为填0。随着阶段推进，参数端点和方向都可能改变，因此曲线反映解及其实际更新方向的共同变化，不能单独归因于起点。

九个积分节点s=0,1/8,…,1不是九个学习任务；它们只是每一条端点连线上用于数值积分的位置。若需要“Task0→Task1→…→Taskt”的折线路径，应改测相邻端点段并另行定义聚合；即使这样也只是任务端点折线，不能冒充优化器每一步的完整路径。当前主图建议保持上述固定起点定义，与已有位移/损伤一致。

## 已有结果到底覆盖多少

| 来源 | checkpoint / Fisher样本 | 已有积分目标任务 | 完整曲线缺什么 |
|---|---|---|---|
| 当前已绘制的历史seed0图 | CopyFlatCV历史权重；固定train256 | Task9，两种方法 | Task1…8，共16个作业 |
| 新formal实验 | 新训练seeds0/1/2；full train1259 | Task1、5、9；18/18份通过数值及收据核验 | Task2、3、4、6、7、8；seed0需12个，三seed共36个作业 |

历史两方法的Task0…Task9共20个checkpoint路径全部存在，因此补历史曲线不需要训练。新formal全部六条trajectory_fisher也有完整十任务端点，保存loss、accuracy、drift及全train endpoint raw Fisher trace。

不过，trajectory_fisher记录中的q字典为空，没有各阶段方向积分。已存raw trace不能转换成S_path。历史Task9的状态日志只保存trace、q等汇总量，没有完整逐样本梯度向量；改变目标任务后，方向和插值模型都不同，无法从旧Task9九个节点直接重算其他任务。

新formal三阶段数据可以立即作图，但必须单独使用该批结果，不能把中间点拼到当前历史图的0.3681/0.0148终点上。即使seed都标0，formal也是新初始化/新配对RNG训练，anchor/head/checkpoint哈希均与历史批次不同；Fisher样本数也不同。相同数据/坐标哈希不等于相同模型。

三seed的S_path均值±样本标准差：

| 目标任务 | 横轴T | EWCLoRA | FS-EWCLoRA |
|---|---:|---:|---:|
| Task1 | 2 | 0.108285 ± 0.029650 | 0.022362 ± 0.006664 |
| Task5 | 6 | 0.165968 ± 0.058176 | 0.027370 ± 0.004591 |
| Task9 | 10 | 0.239649 ± 0.037153 | 0.015193 ± 0.000273 |

这三个已测阶段中，FS-EWCLoRA路径平均敏感度均更低；FS曲线本身不单调。缺失任务不应补成实测点。若连线，图注应明确连线仅作视觉引导；误差条是三个训练seed的样本SD，不是方向分位数或置信区间。

原始逐运行数值：[FORMAL_PATH_VALUES.csv](FORMAL_PATH_VALUES.csv)；汇总：[FORMAL_PATH_MEAN_SD.csv](FORMAL_PATH_MEAN_SD.csv)。历史终点独立保存于[LEGACY_PATH_VALUES_SEPARATE.csv](LEGACY_PATH_VALUES_SEPARATE.csv)，不参与新formal均值。

## 现有代码能复用什么，需要改哪里

1. 旧 [diagnostic.py](../../execution/diagnostic.py:71) 在初始化时固定Task0/Task9；JSON中的path_task字段没有改变其实际加载目标。不能只修改这个字段就认为得到其他任务。
2. 新 [diagnostic_full.py](../../execution/full_rq2_20260920/diagnostic_full.py:359) 已能按目标任务加载权重、计算累计方向、恢复固定head/base、校验端点、分目录存档。主要逻辑可以复用。其 [CLI](../../execution/full_rq2_20260920/diagnostic_full.py:459) 目前仅允许1/5/9，需开放1…9。
3. 新path代码固定使用train1259。配置fisher_samples=256控制的是另一个保留子集的构造，**不能切换path的样本范围**。若保持当前图口径，需要显式增加path_fisher_scope=subset256/full_train，让origin、每个节点、q/trace读取及结果样本数都使用同一scope。
4. subset256可原样复用 [prepare_data](../../execution/rq2_common.py:231)：fisher_samples=256、subset_seed=20260920、相同类分层和数据处理。原数据manifest保存了subset IDs和图像张量哈希，应逐项核对。固定完整train/test endpoint指标仍照常输出；用于积分界校验的损伤应单独命名为D_fisher_scope，不能把256子集损伤写成全train损伤。
5. 每条路径先用9节点，并以5/9比较误差，必要时增加17/33节点；沿用新runner的有限性、恢复、求积稳定及同一Fisher样本损伤界检查。失败保留诊断，不把失败点连入正式曲线。
6. 使用独立扩展入口和新输出目录，避免修改运行中代码。protocol中同时记录目标任务、Fisher scope及IDs、checkpoint/head/方向哈希、代码/精度/求积设置。原脚本已经按path_tXX隔离日志并拒绝不同protocol续写；原Task9收据可明确引用复用，不能盲目复制状态行绕过哈希检查。

旧subset256 collector、checkpoint恢复、数据构造、有效权重插值和大部分积分代码均已有，不需要重写Fisher估计器，也不需要重新训练。

## 最小实施安排

优先保持当前图连续性：复制历史diagnostic_config_recovery1.json的checkpoint/data设置，保留原256样本，准备16个目标作业（两方法×Task1…8），Task9引用现有两份通过检查的结果。每任务基本9节点，共144次状态测量预算；若触发细化则增加。可在同一起点一次计算多条方向的投影以复用起点梯度，但这是优化项，不作为最低实现前提。

每个作业只做前向/反向诊断，不调用optimizer.step。完成后导出method、seed、task、T、scope、samples、nodes、I、S_path、drift、D_train/test、求积误差和数值gate状态，再把C图(d)换成同两方法的折线。前三个面板沿用原历史序列，避免混批。

如果改为新formal三seed主图，则四个C面板全部从formal数据重绘；先可展示已测T=2/6/10的S_path，补齐其余36个fulltrain路径作业后再显示完整T=2…10。该方案与保持历史图是两个明确不同的版本。

准备状态见[extension_spec.json](extension_spec.json)，它是规划规格，**不是已提交队列或已完成的新测量**。本轮只完成可行性核查和CPU数值整理。

## 核查范围与时效

formal的18份path均通过summary/manifest一致性、有限性、Simpson积分复算和数值gate；独立只读核查另对齐现有report.validate_diagnostic、plan receipt requirements及queue receipt SHA，所有相关作业记录exit_code=0、receipt_valid=true。此处不声称重新加载全部checkpoint做了一遍GPU数值评估。

full_rq2/final_report原文件生成于2026-09-20较早时段，覆盖状态已落后于这些新增收据。本报告不重写该批次原报告，也不凭18个path作业通过就声明整个88作业campaign完成。
