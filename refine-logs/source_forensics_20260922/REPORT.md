# 表格对应原代码与实现核查

日期：2026-09-22。只读检查 `/data/140-1/users/liying/CopyFlatCV`、`/data/140-1/users/liying/Flatness_CV`、Git对象和已有冻结快照；另做CPU微型机制验证。未修改两个源码工程，未启动训练。

## 结论与对前一轮表述的修正

1. **历史源码能够部分恢复。** “某次run没有绑定完整源码版本”与“历史源码已丢失”不同；后者不成立。
2. **当前CopyFlatCV确有影响随机流的修改，且可定位、可避开。** 旧LoRA forward在Git HEAD和9/18正式快照中完整存在，不必使用当前live代码补seed。
3. **不能把整个仓库dirty或文件SHA不同等同于本表训练行为改变。** Flatness当前相关dirty改动只涉及ImageNet-R；Copy多项新增分支不被当前表方法调用。
4. **前一计划的99次是保守新实验预算，不是已证明必须重跑99次。** 68次仍是当前表缺seed的算术下限；应先恢复候选版本、核对实际执行路径和run证据，逐组决定能否复用。没有source hash也不自动使历史观测失效。

## 1. 表格方法对应的代码

通用入口是 `src/main.py -> src/trainer.py -> utils/factory.py -> Learner`，之后进入各自网络、优化器和loader。模型名相同或单个Learner文件相同不足以证明整个执行链相同。

| 表格方法 | 可定位实现 | 当前应注意的边界 |
|---|---|---|
| Vanilla LoRA | Flatness_CV/models_LoRAbasedCL/seqlora.py | 当前表20个历史基线中的一行；Copy有额外评估遍历，不能直接当相同实现 |
| InfLoRA | Flatness_CV/models_LoRAbasedCL/inflora.py | Copy的factory写了该import，但实际learner文件缺失；不能直接用Copy启动 |
| OLoRA | Flatness_CV/models_LoRAbasedCL/olora.py | Learner与Copy/9月18快照字节相同，但共享backbone/loader仍须核查 |
| SDLoRA | Flatness_CV/models_LoRAbasedCL/sdlora.py | 同上；不能只比较这个文件 |
| CL-LoRA | CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/cllora_aligned.py | 原表使用aligned移植版，独立cllora_archived_vit路径，不是任意cllora.py |
| EWCLoRA | CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py | 当前所选前四集alpha0 wrapper、Het5 native，由配置切换；继承EWC/GAM/Seq代码 |
| GAM-EWCLoRA | CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py，父类ewclora_youyue_fitArchitechture_gam.py | Het5有9月18正式冻结链；前四组为更早历史运行 |

具体类调用、配置消费者和行号见 [implementation.md](implementation.md)。35组结果来源见 [table_source_binding.json](table_source_binding.json)。

## 2. 哪些旧版本能够恢复

### Flatness_CV

- `371d8d437676046c90fa5e28ef59f17dbc93a468`：3月11日旧版本。
- `9c4439b397ec749d906f980cbda46632e06b08f3`：8月11日保存版本，包含本表全部20份baseline配置；这些配置及主要Learner/共享实现与当前Flatness对应文件一致。
- 当前HEAD：`935147117f82404afdf22a36dd7b19552a94f5a8`。

20份原baseline最后运行日志日期为3月27–30日。clean 3月11日提交没有这20份配置，也没有Aircraft/Flowers/Pets/Het5支持，因此实际实验使用了未被当时clean提交完整记录的内容。8月保存树是可恢复候选，但不能仅由提交日期认定它就是每次3月run的完整执行树。

当前 `utils/data.py` 未提交改动只在 `iImageNetR` 的 `dataset_exclude_manifest` 分支，本表不走该路径。这里不能宣称baseline因当前dirty改动而变化。

### CopyFlatCV 与冻结快照

| LoRA实现来源 | backbone/lora.py SHA前缀 | train包装器临时Linear |
|---|---|---|
| Flatness当前/Aug保存版本 | 6148efe40bdb | 有 |
| Copy Git HEAD `456c4a66…` | 303ed752ccf6 | 有 |
| 9/18 attempt2正式快照 | 303ed752ccf6 | 有 |
| 9/16 rngctl快照 | b4c0233b91bd | 无 |
| Copy当前工作树 | 995119b1aea7 | 无 |

9/16和9/18属于不同冻结实现，不能按日期或目录名混为同一协议。Copy HEAD旧文件可直接从Git对象读取，9/18正式完整代码也仍在。已核实该快照172个源码文件、5个启动器文件全部匹配冻结清单，当前表六个正式receipt的配置、日志、指标SHA全部匹配。

源码绑定不是只看receipt是否直接含commit字段：`run_job.py:24–30`会校验源码/启动器/配置/数据manifest，之后把cwd和sys.path切到冻结代码，再导入入口执行。因此CL五组与GAM-Het5的代码证据明显强于历史日志。

原四组GAM日志来自4月28–29日，早于Copy Git首保存6月22日；选定EWC来自8月11/27/30日。可恢复Git版本和源码快照存在，但本次没有找到这些原日志里的逐run源码hash绑定。此结论局限于所检查证据，不能推广为任何地方都不存在其他记录。

精确提交时间、文件SHA和恢复证据：[history.md](history.md)、[history.json](history.json)。

## 3. 随机数变化的具体机制和验证

旧 `_LoRA_qkv_timm_train.forward()` 先无条件创建四个临时层：

```python
w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
```

随后用保存的A/B覆盖这些临时权重。初始化出的随机权重不参与最终有效计算，但创建过程已经消耗默认CPU Torch RNG；即使task_id=0、没有历史adapter，四次创建也执行。当前Copy改用 `F.linear(x, saved_weight.detach().to(x.device))`，不再创建临时随机权重。

验证使用真实文件AST提取的类，CPU/PyTorch2.4.1，维度6、rank2、batch2、tokens3；比较5个版本、train/eval包装器、task0/1和三种train/eval/autograd调用方式，共60个case。CUDA未初始化。

| 检查 | 结果 |
|---|---|
| 相同参数/输入下输出 | 本测试中逐位相同，最大绝对差0 |
| train包装器的输入及可用参数梯度 | 本测试中逐位相同，最大绝对差0 |
| 保存的历史adapter是否意外获得梯度 | 未获得 |
| train包装器调用后的CPU RNG | 旧版本消耗；Copy当前/rngctl不消耗 |
| 从上述状态继续DataLoader shuffle | 后续顺序不同 |
| 从上述状态初始化新分类头、抽crop/flip参数 | 后续值不同 |

因此这是“当前一次确定性计算相同，但后续随机训练轨迹可能不同”的实际机制，不能简化为“数学公式变了”。CPU微型测试没有量化最终FAA/AAA变化，也不证明旧准确率错误。

独立eval包装器的forward带有no_grad，其案例仅验证输出和RNG，不声称验证了梯度。微型DataLoader采用workers=0；没有运行真实多worker数据流水线。

还需区别：对train包装器调用 `.eval()` 或 `torch.no_grad()`不会取消旧构造的随机初始化；独立 `_LoRA_qkv_timm_eval` 类在检查的五个版本里仍保留临时Linear，不能说当前整个评估路径已不消耗这些随机数。

方法影响要看实际包装器绑定：Vanilla/Inf/OLoRA/SD及EWC/GAM的共享LoRA构造路径需要逐一保持版本；CL aligned走独立backbone，不能把该问题外推给CL。Copy缺Inf learner不代表Flatness Inf与共享LoRA无关，也不代表Copy能运行Inf。

证据和可复验脚本见 [rng_diff.md](rng_diff.md)、[rng_diff.json](rng_diff.json)。

## 4. 其他真实差异与不影响本表的改动

- Copy/9月18快照的SeqLoRA增加每5epoch和任务末的当前任务诊断评估；Flatness没有对应遍历。DataLoader默认generator为空，额外构造iterator可能消耗全局Torch RNG。因此不能因为它只做评估就断言训练随机轨迹完全不变。此差异直接关联Vanilla和继承Seq的EWC/GAM，不外推为OLoRA/SD的同一代码差异。
- Python/NumPy/Torch的初始播种函数在四份代码树中逐字节一致，问题主要是后续随机数消费路径不同，不能误写成两个工程使用不同的初始seed函数。
- Copy当前相对正式快照新增的factory KFAC/complementarity分支、inc_net模型名分支、baseLoRA checkpoint hook，多数不由表格现有模型名和运行方式触发；不是所有代码变化都要求重跑。
- 数据路径默认不同：Flatness会在未给定时注入 `/data/140-0/datasets`，Copy没有该默认；需要核对实际root、环境变量与realpath。Flowers旧sorted路径与新canonical-ID映射在目前8189个文件/清单上逐一一致；不能根据实现写法不同断言旧标签错配。Copy新增CUB/Cars/Het5校验在合法数据上主要是错误检查，不等于改变任务流。
- CL cosine最低LR=0；Seq等当前实现默认最低LR=初始LR×0.1。同为cosine不代表完整调度相同。
- 9月18 GAM-Het5的Fisher RNG preserve生效，但 `ewc_isolate_clean_rng=true` 没有被该冻结learner读取；该快照loader也没有实现独立generator隔离。不能仅根据JSONflag把它称为双RNG隔离版本。

此前对Aircraft loader的概括需要修正：

| 路径 | workers默认/配置 | Aircraft pin / persistent实际行为 |
|---|---|---|
| SeqLoRA及继承EWC/GAM，DataManager.build_dataloader | 通常8；9月18正式GAM配置2 | 数据集loader_options为true / true，顶层persistent_workers=false不自动覆盖 |
| InfLoRA / OLoRA / SDLoRA，直接DataLoader | 当前默认8 | false / false |
| CL aligned，自建DataLoader | 2 | true / false |

Het5按自己的idata配置，不继承Aircraft单数据集的loader_options。新实验应记录实际DataLoader参数，不能只记录不被消费的配置键。

## 5. 对补seed工作的实际建议

1. 四baseline优先恢复完整Flatness Aug保存树作为候选，保留真实共享代码、loader、播种和评估流程；不能直接套Copy runner。
2. CL五组与GAM-Het5继续从9月18正式快照及启动器规则建立复用检查，不用当前live树替代。
3. 历史EWC/GAM按原配置和时间线恢复候选；特别不能把9月16受控实现与旧普通实现混合。
4. 对不确定组先做来源核对和选定seed0复现，检查任务矩阵及训练轨迹。复现分数接近只是支持证据，不单独证明所有历史来源；也不允许为了追旧测试分数调参。
5. 通过组内协议一致性核对后补缺seed；确认实现/数据发生不可兼容改变的组才重新做0/1/2。99次保守预算可保留作上限方案，当前审计不支持把它当已确定的最低运行量。

两个源工程只读；本次监控84个路径，在记录窗口内未发生内容变动。没有进行GPU训练或更改既有结果。
