# 正式 ImageNet-A 六行表：配置与实现审查

日期：2026-09-21。只读检查正式计划、21份训练manifest/完成凭据、冻结runner及learner；只新增本文件，没有启动GPU或改冻结文件。下文论文Task 1/2分别对应代码task0/task1，seed集合为0/1/2。

**六行中已有四行的正式三seed训练结果；另两行最少补6个50轮Task1续训即可。** 已完成的连续GAM链可取task01作为“FS-EWCLoRA”；原B3_B不是“Shaping-only”，原B3_D不是“两个任务都shaping”。当前用户六行表采用Task 1训练batch-Fisher proxy及Task 2准确率/位移，FS-EWCLoRA无需重测。六行汇总需显式映射；只有另行要求原B3逐样本raw-Fisher诊断或同schema收据时，才需扩展原本仅接受future SGD的endpoint入口。

| 用户六行 | 论文目标 Task 1 → Task 2 | 实际训练身份 | 正式配置与训练 | 同口径endpoint状态 |
| --- | --- | --- | --- | --- |
| Vanilla LoRA | C → C | SGD anchor → SGD，λ=0 | `B3_A_sgd_lam0_seed{0,1,2}`，3/3完成 | 原B3_A已有 |
| Shaping-only LoRA | C+F → C+F | GAM anchor → GAM，λ=0 | 正式计划未配置；0/3 | 缺训练与评估 |
| EWCLoRA | C → C+P | SGD anchor → SGD，λ=2000 | `sequence_sgd_seed{0,1,2}` 的task01，3/3完成 | 原B3_C已有 |
| FS-EWCLoRA | C+F → C+F+P | GAM anchor → GAM，λ=2000 | `sequence_gam_seed{0,1,2}` 的task01，3/3完成 | 本表可复用anchor机制B1、task01 CIL与B2位移；无需重测 |
| EWCLoRA + Task-1 shaping | C+F → C+P | GAM anchor → SGD，λ=2000 | `B3_D_gam_lam2000_seed{0,1,2}`，3/3完成 | 原B3_D已有 |
| EWCLoRA + Task-2 shaping | C → C+F+P | SGD anchor → GAM，λ=2000 | 正式计划未配置；0/3 | 缺训练与评估 |

原 `B3_B_gam_lam0_seed{0,1,2}` 实际是 **GAM→SGD、λ0，即C+F→C**，属于另一控制，不能填到六行的Shaping-only位置。文件名含`gam`只标anchor，必须读manifest中的`optimizer_type`和`lambda_flat`。

六个anchor配置写λ2000，但第一个任务没有过去Fisher，EWC不生效。六份anchor raw机制记录的`train_fisher_loss_mean`均为0；learner的Fisher梯度路径在无past Fisher或λ0时直接返回零。故GAM anchor可供λ0/2000两个分支复用，不需重训“λ0 Task0”。

## Runner究竟支持什么

- [train.py](../train.py:102) 的`method = 'sgd' if ns.mode == 'continuation' else ns.method`使所有`continuation`固定未来SGD。`--method gam --mode continuation`表示GAM起点、SGD未来，绝不表示GAM→GAM。
- `--mode sequence --method gam --max-train-tasks 2`支持仅续训至task1；显式`--anchor-dir/--anchor-metrics`恢复task0后，未来优化器为GAM。`--ewc-lambda 0`是已有合法CLI值，因此GAM→GAM无EWC在训练执行层已能实现。
- 当前sequence允许显式恢复SGD anchor后跑GAM，**但manifest将`anchor_method`写为`ns.method=gam`，会错标真正的SGD parent**。实际parent文件SHA仍正确，不能据此把这个歧义接口直接当成已审计的SGD→GAM正式入口。
- 本次在CPU只调用`build_config`验证：continuation/gam/λ0产生`optimizer_type=sgd, lambda_flat=0`；sequence/gam/λ0产生`gam,1`；sequence/gam/SGD-parent/λ2000也产生`gam,1`，但会产生上述anchor标签错误。没有构建模型、写配置或执行训练。
- 冻结learner已有SGD和GAM训练分支，`seqlora.py`还支持`optimizer_type_by_task`，但正式runner明确置None。无需发明或修改GAM/EWC算法。核对当前`/data/140-1/users/liying/CopyFlatCV`，`seqlora.py`、`as1_normfisher_gam_intask.py`、`ewclora_youyue_fitArchitechture_gam.py`、`as1_normfisher_gam_intask_damp.py`与本轮冻结副本逐字节相同；当前代码“可实现”不等于有该分支的正式完成记录。

## 最小新增训练与父资产

记正式训练根目录为：

```text
/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_training/
```

对每个`s=0,1,2`：

| 新分支 | 精确父目录 | 未来设置 | 训练规模 |
| --- | --- | --- | --- |
| Shaping-only | `anchor_gam_seed{s}/checkpoints` | optimizer=gam，lambda_flat=1，ewc_lambda=0 | 仅task1，50轮 |
| Task-2 shaping | `anchor_sgd_seed{s}/checkpoints` | optimizer=gam，lambda_flat=1，ewc_lambda=2000 | 仅task1，50轮 |

每个父目录均须恢复`task_00.pt`和`fisher_state_0.pt`，metrics来自相应anchor的`trainer_outputs`唯一`*_cl_metrics.json`，而非任何分支的最终10×10矩阵。保留父文件SHA、seed及method身份。总计6个新训练段、300个task-epochs；不需重训六个anchor，不需重跑三条GAM序列，也不需补10任务长链。正式FS-EWCLoRA直接用`sequence_gam_seed{s}/checkpoints/task_01.pt`与该链`cnn.matrices.t01`，不能取task09/final准确率当两任务结果。

## 必要代码工作与评估工作

1. 为新增训练建立版本化入口，保留原冻结runner。显式区分`anchor_method`与`future_optimizer`，记录实际`task0_shaping`、`task1_shaping`、λ及父SHA。GAM→GAM/λ0的训练层可直接复用sequence；SGD→GAM需要修正入口的科学身份记录，并校验父manifest的seed/method。若使用新的通用continuation CLI，也应同时支持SGD/GAM未来，不能只改名字。
2. 若另行要求B3额外endpoint-train256逐样本raw-Fisher列或严格同schema收据，再新建endpoint版本；这不是填当前六行表的前提。原[endpoint.py](../endpoint.py:68)硬断言`optimizer_type=='sgd' && lambda_flat==0`，且[输出标签](../endpoint.py:159)写死`future_optimizer='sgd', shaping_in_task1=False`；去掉断言而保留硬编码会生成错误科学标签。新评估应允许两种未来优化器，独立核验parent与当前目标，检查冻结骨干/旧head、checkpoint、FP32转换及数据manifest。
3. 原[report.py](../report.py:101)也只验证四个future-SGD分支，并硬写task1 optimizer。六行需要新的显式六种目标映射，不能套用旧BRANCHES或把B/D改显示名。当前表中FS-EWCLoRA可完整复用anchor机制日志的Task 1 batch-proxy B1、sequence的task01 CIL以及B2位移，不需补3次endpoint诊断。只有扩展B3额外raw-Fisher列/严格同schema收据时才补这3次诊断。新增6个续训的准确率直接取任务矩阵；QV有效权重位移可用CPU从父/子checkpoint核算，也可按需用新版endpoint，无须预设必须进行6次完整GPU端点评估。若复用已有dense诊断，需先核样本、scope、head、坐标、checkpoint与Fisher定义完全一致。
4. 保留两种Fisher口径：当前六行表的Task 1 B1取anchor训练机制日志中的batch-gradient Fisher proxy，不能冒称逐样本raw Fisher。正式训练惩罚使用归一化`batch_empirical` Fisher；可选B3 endpoint重测才用raw逐样本Fisher、固定旧任务数据/分类头。不能拿`fisher_state_1.pt`的累计/归一化trace替代B1或逐样本raw诊断统计。

## 配对约束与本次实际核验

- 继续使用`rq2_shared_task_rng_v1`、`task_seed=20260920+100003*seed+1009*task`、loader seed=`task_seed+10000019+7919*loader_serial`；每任务重置loader serial与模型RNG。正式全部`workers=4`，不要误用runner默认8；保持batch128、task1 lr .02、50轮、FP32且两处TF32关闭、固定类别顺序、相同数据根目录/增强。
- 必须完整恢复父骨干/LoRA/旧head/Fisher；不要重新初始化trained anchor或仅加载LoRA。继续使用同300个历史预训练骨干权重，禁止让timm别名下载另一预训练模型。
- 本次逐seed比较五组现有task1：B3_A、B3_B、B3_D、sequence_sgd、sequence_gam。**3个seed都满足：新head SHA相同；前50个训练pass的generator seed、workers、样本数、完整样本顺序SHA和首batch像素/标签SHA逐项相同。** 因此现有连续GAM task01与B3续训具备已记录的随机配对证据。
- 新增两分支也必须通过相同检查，保留每任务50个完整loader pass、50/50轮日志、task_complete、metrics与checkpoint完成哈希。首batch哈希不能表述成每个增强batch均被逐像素哈希核验；GAM额外探测的计算量和模型内部随机消费也不等于SGD。
- 既有seed0 pilot或旧ImageNet-R六行表不进入此正式三seed配对；本报告只判断上述正式ImageNet-A目标覆盖情况。

审查读取的SHA256：`train.py=a4de971877e8bee0f7da9efbb010fb2f7c7df30a7920286ee3196cbe4dc4afe8`；`endpoint.py=65fff6634c6d7635435e3e9e2fe5ef19b8b96037e486fa70b23609ead35c3850`；`formal_plan.json=fcbf7e4c53c26e48cec12c82a22774f55f79147bb982f51375a9a99b5efecdeb`。已读正式21份run manifest（6 anchors、6 sequences、9独立续训）及对应COMPLETE；缺失判断以此正式计划和目录为范围，不声称其他历史实验从未执行过类似目标。
