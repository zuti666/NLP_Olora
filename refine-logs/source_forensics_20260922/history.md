# 历史源码可恢复性与逐运行绑定：Git/快照核查

2026-09-22；只读两源码仓库，未训练、未checkout/reset、未更改源码。范围限当前表20个历史baseline、7个已选EWC候选及4个历史GAM日志。

**结论：应保留“历史运行与完整源码尚未绑定”的限定，但必须撤回“没有历史源码可恢复，因此全部必须重跑”的强推断。** 多个旧源码版本明确能从Git和已有冻结快照恢复；当前工作树改动也可避开。缺运行绑定表示尚不能认证来源，不等于证明旧结果已经受污染或失效。建议先恢复候选快照、比较实际执行路径和已有运行证据，再确定需重跑范围。

## 1. Flatness_CV：历史内容存在，clean March HEAD不能代表实际实验树

当前HEAD为 `935147117f82404afdf22a36dd7b19552a94f5a8`，2026-08-12 14:30:42 +02:00，分支exps_SCM。可恢复的关键节点：

| 节点 | Commit | 提交时间 | 事实 |
|---|---|---|---|
| 三月运行之前 | `371d8d437676046c90fa5e28ef59f17dbc93a468` | 2026-03-11 11:00:43 +01:00 | 四方法、backbone等旧版本存在 |
| 另一旧分支 | `09ac9fdd86e2f74542da5ac311870fe6625b943c` | 2026-03-12 20:44:52 +01:00 | 被检查13个关键文件与371d8d4字节相同 |
| 三月运行之后的保存 | `9c4439b397ec749d906f980cbda46632e06b08f3` | 2026-08-11 11:11:05 +02:00 | 保存当前四baseline learner/共享核心实现及20份选定YAML |

原20次baseline最后run日志发生在3月27–30日。reflog显示3月12日切换回371d8d4后，下一次保存是8月11日；**reflog记录HEAD，不记录未提交的编辑**。[Git/reflog证据](git_history_evidence.txt:69)

逐项`git show`核查结果：

- 371d8d4中20份选定配置全部不存在，`utils/data.py`、`utils/data_manager.py`也没有Aircraft/Flowers/Pets/Het5支持。因此clean March commit不可能单独构成当前表这些数据集的运行工程；实际运行必须包含后来加入的工作树内容或其他未记录路径。
- 9c4439b中20份选定YAML全部存在，且SHA与当前对应配置逐一相同。四个baseline learner、backbone/lora.py、baseLoRA.py、baseLearner.py、inc_net.py、data_manager.py、trainer.py均与当前Flatness字节相同。
- 所以8月保存版本是**可恢复候选实现**，而不是“源码丢失”。但提交时间晚于实验，不能自动证明三月每次run都执行了该完整版本。[March/Aug差异](git_history_evidence.txt:104)、[恢复的Aug数据加载器](Flatness_9c4439b3_utils_data.py:29)

部分具体SHA256如下；全部文件版本和20份配置证据在`history.json`：

| File | March 371d8d4 SHA256 | Aug/current SHA256 |
|---|---|---|
| seqlora.py | `28dfdf20d75e363924219b79da72498ec587c1517358257f3ac1e9fa93e112d4` | `ae1622be6176facdbbf18fb25a6cd720c136f6381dfd8c7801b224b72c815731` |
| inflora.py | `720cfa703d460920bab0ec1cd103c3a24b6f9c210af1eda61ab8d5a8ce3a9469` | `ae999e8457ddd5c4eed09818a1ad9cf44c5afa7d6f60caac3165fc0db89ecdd3` |
| olora.py | `47e3529a7139e3445d9db7af887baebb56ac8ed9033d1958c7e28170d731915b` | `670636e21573b001e39ad7e614c0ba7fc642a87a1ed04868b78c1842cceb9394` |
| sdlora.py | `5539c56947b754763989bd23c575efd42c4a83c411160ed7e6ecd940e553e085` | `6c6a4715ba6c0e0f6410d59a12ec0aae332efa2001355b5fd4708be4ca21c1eb` |
| backbone/lora.py | `ea595d954ab4f51bb7ae0653035822752107103bc1744a58ca9217fe4481ca60` | `6148efe40bdb3c5a2a478badbc9c996506e342658fa9160b9bed76b255018abd` |

当前Flatness相关dirty文件只有`utils/data.py`，实际diff是在iImageNetR中加入exclusion manifest分支（原行705），与当前五数据集执行路径无关。**不能用仓库dirty或整体SHA改变推断本表baseline行为改变。** [完整dirty diff](git_history_evidence.txt:120)

## 2. CopyFlatCV：旧LoRA forward内容明确可恢复

当前HEAD `456c4a66d34c4f252335b325624160e1625e1d3b`，2026-09-04 04:11:41 +02:00。仓库还保存 `backup/pre-clean-history-20260622103709` 分支，指向首次保存历史；6月22日reset并未清除该备份ref。[分支证据](git_history_evidence.txt:43)

`backbone/lora.py`的三个实际版本：

| 来源 | SHA256 | 恢复/绑定意义 |
|---|---|---|
| Copy HEAD 456c4a66 | `303ed752ccf68b3a42106aedfc134e97e4305af0b013a45aea7f41ea7665dc86` | Git对象完整可取 |
| 9/18 formal gapfill code快照 | 同上 | 与其source_frozen.json逐文件一致 |
| 9/16 RNG control code快照 | `b4c0233b91bdb848d79fc0c704887abccb73769da1beb24accc663ad75ea360c` | 该受控campaign单独冻结的另一实现 |
| 当前live tree | `995119b1aea75e27049ff6a20aeafac08daa2c212fd2d7c3eff517c1bf581446` | 未提交修改；不必用于补跑 |

当前对HEAD的diff确实删除了forward内四个临时`nn.Linear`构造，并改为`F.linear`计算历史贡献。这些旧构造初始化随机权重，旧内容未丢失，可从HEAD或9/18快照取回。**不能从“live变化会影响RNG”推导“只能用新行为重跑全部”。** [diff精确位置](git_history_evidence.txt:208)

本核查只确认版本变化、内容可恢复和已存在文件匹配；RNG影响哪些具体learner路径及实测状态差异由代码/机制审计另行判定。不能将整个backbone文件SHA差异泛化为全部方法受影响。

9/16与9/18的`source_frozen.json`分别存在，所检查的六个关键文件全部匹配其各自SHA；包括LoRA、baseLoRA、EWC、GAM-EWC、DataManager、trainer。**这两份快照互不等价**，不能因为都有“冻结”二字就把受控与原始协议混用。

## 3. 原GAM/EWC的时间边界

- 表内历史GAM四组：CUB 2026-04-28，Aircraft/Cars/Flowers 2026-04-29。早于Copy Git首提交 `d4aeea8944a56fadd7df5a01e6745ab31cb96ecb`（2026-06-22 10:29:42 +02:00）。首提交与备份分支保留旧GAM实现，可作候选恢复，不等于它被绑定至四月run。
- EWC Aircraft/CUB原日志：2026-08-11；Cars/Flowers seed1/2：2026-08-27；Het5 seed2：2026-08-30。Copy关键EWC/AS1源码有7月21日 `abbac08607f1a36997ca6071dcb1d96561800aba` 和9月4日456c4a66等版本；运行时间夹在其中，不能假定运行树等于前一个clean commit，也不能假定等于后一个保存。
- 对这31份指定原始日志，只读搜索`git_commit/git_sha/source_sha/source_hash/source_root/code_sha/snapshot_sha/commit_hash`，没有命中运行源码绑定字段；**这仅说明这些原始日志未记该字段，不声称全工程其他receipt/launcher完全不存在绑定证据**。实际路径、最后run起始行号和日期均在`history.json`。

## 4. 对先前补seed方案的修正

准确结论是：既有历史内容能够恢复；目前某些run缺完整源码身份见证，需要继续按实际代码路径核对。旧结果仍是有效归档观测，不能因为缺SHA就判作无效或已受当前改动影响。

四baseline保存版本可优先从9c4439b恢复，另查实际训练路径是否与日志/配置相符；9/18正式run可以直接从已绑定snapshot继续来源校验。**60/99次全新campaign是保守可复现策略，不是Git事实强制要求的唯一方案。** 若建立足够一致的实际协议，可只补缺seed并明确历史来源限制；若确认方法行为、数据或随机流程发生不可逆协议变化，则相应组重跑0/1/2。逐组判断，不先由全仓hash差异决定全表重跑。

全部精确Git对象与SHA：[history.json](history.json)。命令原始输出：[git_history_evidence.txt](git_history_evidence.txt)。没有修改任何原仓库。
