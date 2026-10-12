# ImageNet-A 两阶段方案：现有 empirical Fisher 实现审计

2026-09-20。只读检查现有Python、recovery1差异、manifest、sanity和A端点结果；本轮没有启动GPU、训练、评估或队列，没有修改已有执行文件。

## 1. 直接结论

现有 `rq2_common.py` **确实实现逐样本 empirical Fisher 的trace及完整外积矩阵的方向二次型**，可复用其测量核心。它不等同于旧ImageNet-R六行表中的batch-gradient proxy，也不读取归一化的训练保护Fisher。

当前测量的明确口径是：**有效Q/V权重坐标、固定旧Task0的20类分类头、固定旧任务训练子集256张、真实标签CE、模型eval、无尺度归一化/正floor/正则项**。不是A/B因子坐标，不是K/V，不是新任务20类或已见40类CE。

A表已有两行是 `SGD anchor → SGD+EWC` 和 `GAM anchor → SGD+EWC`。`endpoint_a.py` 对后续SGD、`lambda_flat=0`、EWC λ2000有显式断言。因此它能回答初始化差异，**不能被称为已经完成两阶段都采用GAM的对照**。

## 2. 为什么这是逐样本 EF，而非batch均梯度平方

定义有效坐标

\[
w=\operatorname{vec}\{B_{\ell,Q}A_{\ell,Q},B_{\ell,V}A_{\ell,V}\}_{\ell=1}^{12},\quad
g_i=\nabla_w\ell_i,\quad
\widehat F^{emp}=N^{-1}\sum_i g_i g_i^\top.
\]

`load_network` 恢复完整 `network_state`，不下载替代预训练模型，随后冻结网络并置eval；[rq2_common.py:97](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:97)。`convert_effective` 将LoRA Q/V分支替换为直接可微的24个768×768有效矩阵，共14,155,776个坐标，base QKV和分类头不进入测量参数；[119行](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:119)、[166行](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:166)。这是对ambient有效权重的导数，没有投影回rank10因子流形。

对每个线性分支，前向 `y[b,t,o]=Σ_i x[b,t,i]W[o,i]`；hook以 `bmm(adj.transpose(1,2),x)` 恢复每个样本的矩阵梯度。ViT在eval时样本之间独立，因此对整batch的 `losses.sum().backward()` 得到的每个样本adjoint未被batch平均缩小。对应 [PerExampleCollector:150](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:150) 和 [fisher:326](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:326)。

实现先对每个样本分别计算，再跨样本平均：

\[
\texttt{raw_trace}=N^{-1}\sum_i\sum_b\|g_{i,b}\|_F^2,
\qquad
\texttt{q[d]}=N^{-1}\sum_i\left(\sum_b\langle g_{i,b},d_b\rangle\right)^2.
\]

方向内积先跨24块相加、然后平方，所以 `q` 保留跨坐标/跨块外积项，**不是只使用diag(F)的近似**。完整F矩阵没有被物化，也没有保存每个坐标的对角值；目前输出只是trace、可选方向q、subset CE及元数据。

批大小8只是并行计算粒度，末尾不足8也正确按真实N平均；不是把batch均梯度平方当F。梯度和乘法为FP32，trace/dot等累计主要用float64；不能称为全程FP64或精确实数运算。没有除以trace/max/均值，没有clamp正floor，没有EWC/GAM正则进入CE。`normalized=false, floor=null, regularizer=null` 与代码一致。

## 3. loss与分类头范围

| 入口/记录 | 输入数据 | logits/标签范围 | 是否EF |
|---|---|---|---|
| `RestoredNetwork.forward` + `fisher` | 旧Task0训练子集 | `fc(backbone(x))[:,:20]`，标签0–19 | 是，旧local20 EF |
| `evaluate` | 旧Task0全部train/test | 同上，固定旧20类 | 否，CE和accuracy |
| `evaluate_local_and_seen` | 旧Task0全部train/test | local20与所有已见类分别算CE/accuracy | 否 |
| A `accuracy_CIL` | 原训练器测试结果 | a11时20类，a21/a22时40类 | 否 |
| A训练 `task0_local_metrics.jsonl` | 旧Task0全部train/test | task0:20类；task1:local20及seen40 | 否，明确`trace_measured=false` |

固定切片由 [rq2_common.py:93](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:93) 决定；后期checkpoint即使有40或200行head，`fisher` 仍只看最初20个logits。`evaluate_local_and_seen` 为了得到seen范围，专门绕过这个forward直接调fc，见 [280行](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:280)。

A还断言两个endpoint间旧head和所有非LoRA backbone张量逐比特不变；[endpoint_a.py:85](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/endpoint_a.py:85)。这保证同一行内旧local20 loss函数的head固定；不同方法拥有各自训练出的旧head，不能宣称两方法共用同一个旧head。

当前A没有新任务20类EF、旧数据seen40 EF、新数据seen40 EF、test集EF、A/B因子EF或全backbone EF。不能把旧local20的raw trace与CIL a21视作完全同一loss口径。

## 4. 固定数据、eval状态与精度

[prepare_data:204](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:204) 固定ImageFolder排序的最初20类：旧train1259张、test307张，并断言数量。两个split均使用 `Resize(256,bicubic)→CenterCrop(224)→ToTensor`，**无Normalize、无随机裁剪/翻转**；train指标也是确定性评估变换后的train，而非训练增广期CE。

256张Fisher子集使用Hamilton比例分配：按各类数量分配整额、余数最大优先，同余按较小类索引；类内使用seed20260920随机无放回选择，最后按索引排序。它是近似保持原训练类比例的分层抽样，**不是每类等量抽样**。当前分配为 `[12,17,18,16,10,13,11,11,16,11,17,11,14,11,12,11,15,8,10,12]`。

两方法data manifest SHA均为 `be9cf89b2bfbd5a9bc207d9755deb2d3a9bcbd6190a24b3a9c05ead97bfddecb`。manifest含逐文件hash、子集ID/索引、解码tensor hash、Pillow/libjpeg/torchvision信息；A断言与sanity数据hash完全一致。

`load_network(...).eval()` 禁用原模型随机训练行为；`fisher()`本身不负责调用eval，未来复用时必须保留这一前置条件。Runner与A入口显式关闭matmul TF32、cuDNN TF32和cuDNN benchmark；未启用autocast。历史TF32卷积模拟只用于解释旧结果，不用于production EF。当前first20假定适用于已核验类序；若新实验改变类序，必须从实验class order生成子集，不能沿用硬编码索引。

## 5. 随机方向、真实方向与路径

- 随机扰动是在24个有效Q/V矩阵的整个ambient空间生成CPU FP32 Gaussian，再用全局Frobenius范数归一化；seed=`20260920+104729*k`，两方法复用相同hash的16个方向。不是A/B扰动，也不限制扰动后的矩阵保持rank10。半径0、0.5、1、2、5、10、20。
- 当前真实方向是两条历史完整序列的 `BA_Task9−BA_Task0`。bridge使用其单位方向及r=0/.25/.5/1/1.5/2；两个anchor都评估两个方法方向。
- 每个 `measure()` 的 `q` 字典均传这两条历史Task9完整位移，**不是当前随机扰动u的方向二次型**；随机方向的作用是决定测量点。见 [diagnostic.py:125](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic.py:125)。
- path在同一有效坐标直线上测 `ΔᵀF_old,local20(w0+sΔ)Δ`，Simpson9节点与5节点比较，必要时扩17；再除`||Δ||²`得单位位移路径敏感度。[370行](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic.py:370)。

**当前Runner在[71行](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic.py:71)硬编码读取task0/task9，trajectory也固定十个任务；仅将config中的`path_task`改成1不会改成两阶段路径。** 新两阶段实验需要显式参数化anchor/endpoint，不能把旧B/C的Task9方向移接到A新Task1 checkpoint。

## 6. 当前A已完成的EF字段和具体结果

以下从recovery1 `table_a/summary.json` 读取；task0/task1是代码索引，即论文第一/第二任务。

| Task0 anchor → Task1更新 | anchor旧local20 EF trace | 新endpoint旧local20 EF trace | 旧test local20 ΔCE | 有效Q/V位移 |
|---|---:|---:|---:|---:|
| SGD → SGD+EWC | 1465.496926903 | 2008.387873263 | -0.019245973 | 0.374949571 |
| GAM → SGD+EWC | 761.900774325 | 940.412532306 | -0.013388091 | 0.349875797 |

来源：[SGD A summary](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0_recovery1/sgd/table_a/summary.json)、[GAM A summary](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0_recovery1/gam/table_a/summary.json)。这只是seed0已完成结果，两个起点继承的训练保护Fisher不同。

A的anchor复用raw trace、subset CE、samples/定义/坐标/floor等元数据，显式删除原anchor的Task9方向q；[endpoint_a.py:113](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/endpoint_a.py:113)。新endpoint传空方向字典，因此 `endpoint.fisher.q={}`；[129行](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/endpoint_a.py:129)。A没有测自身Task0→新Task1的方向敏感度或路径积分。

已有训练sidecar还可直接补旧样本的seen40 CE/accuracy，无需再评估：

| anchor | Task1旧test local20 CE | Task1旧test seen40 CE | Task1旧test seen40 accuracy |
|---|---:|---:|---:|
| SGD | 0.423177732 | 0.598961180 | 84.36482085 |
| GAM | 0.400232589 | 0.533356956 | 84.03908795 |

来源：[SGD task0_local_metrics](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/training/A_sgd_to_sgd_seed0/task0_local_metrics.jsonl)、[GAM task0_local_metrics](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/training/A_gam_to_sgd_seed0/task0_local_metrics.jsonl) 的task=1行。它们没有EF，不能据CE字段补写Fisher结果。

## 7. recovery1变更与已有数值检查

直接diff当前 `diagnostic.py` 与 `campaign_seed0_dualline_recovery1/before_fix/diagnostic.py`，只改两处：`Runner.measure` 的张量形参由`direction`改名`perturbation`，并把该变量传给set_point；解决位置张量与`direction=meta`重复传参。数据/半径/方向/EF公式未改。recovery1配置只改输出目录；两条已完成A训练复用，诊断重新执行sanity。

两个recovery1 sanity均status=complete、passed=true，保存的code/data hash与当前代码一致。原模型与有效矩阵转换的8张测试图最大logit误差均为0；两个真实样本用逐个autograd验证hook估计，trace最大相对误差：SGD约1.52e-5、GAM约1.22e-5；方向内积绝对误差最大约2.46e-6。有限差分、r0、直线终点恢复均通过。检查定义见 [diagnostic.py:221](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic.py:221)。这是当前旧local20/QV协议的检查，不能代替未来新增loss scope或full-model参数空间的验证。

本次读取源码SHA256：

| 文件 | SHA256 |
|---|---|
| rq2_common.py | `ae1c253e79bcfc884b27e160069d5cc40cfcb269685e103dca280a18413e0b7f` |
| diagnostic.py | `7fcbef3a4614380522c1d2eaa0129f2f2c870fc61653e8b6f1c25873ef0f3dd4` |
| endpoint_a.py | `c2c0eab7d447f15430a4aa53777c90d19d4a2df75ba435b0427a8650a248ebae` |

## 8. 新两阶段实验的最小复用与补充

1. **保留测量核心，新增独立scope接口和输出协议。** 不改写旧结果。让forward返回完整logits，由显式`loss_scope`选择`old_local20`、`new_local20`或`seen40`；新local20使用logits20:40及标签减20，seen40使用原全局标签。保留代码/hash、head范围、样本ID、坐标、precision在每条结果中。
2. **最少报告三个EF单元。** 每条方法测旧数据旧local20的 `trF_old(theta_Task0)` 与 `trF_old(theta_Task1)`，再测新数据新local20的 `trF_new(theta_Task1)`。前两项对应保留旧知识，第三项才是与旧表“各任务自身endpoint”风格相对应的第二列；名称必须分别标出old/new，不能写一个含混T2。新任务同样先固定比例分层256张，并在所有方法/seed共用样本ID。
3. **seen40作为CIL口径补充。** 已有A的旧seen40损伤/accuracy可直接从sidecar汇总。需要EF时另算旧数据seen40 trace，不能从local20 trace推断。若要比较seen40的前后损伤/敏感度，应定义共享新head初始化后的40类训练起点；20类与40类CE是不同预测空间，不能假装同一函数。
4. **两阶段方向只用对应新端点。** 若需要“同等/更大位移而较小损伤”，在已有hook中加入本次 `Delta=BA_Task1−BA_Task0`，即可额外输出真实`q/||Delta||²`；若做路径，显式改为该两个checkpoint，9节点并保留求积检查。不要复用旧Task9 q。当前A端点EF约1.3秒/256张的原运行计时可作单点开销参考，不代表新GPU上的排期承诺。
5. **新增实验不是调整老endpoint_a的assert即可。** 该脚本固定两方法名称、SGD后续、lambda2000与关闭shaping。EWC+GAM新分支必须有自己的训练provenance和评估入口，统一验证其真实更新规则；训练保护Fisher和测量raw EF继续独立命名。
6. **参数空间保持明确。** 当前工具仅适用于已核验的CopyFlatCV持续更新Q/V LoRA、rank10、scale1和network_state格式。不能直接声称覆盖旧fresh-K/V、A/B因子或full全部backbone。若full-model参与，新测量需另选全共享backbone/共同QV子空间并独立校验；不混用trace或绝对位移的范围。

本报告只提供实现与缺口证据，不启动以上补充工作，也不改变已停止队列。
