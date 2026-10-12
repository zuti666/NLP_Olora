# 同状态梯度与HVP失败定位：独立数值复核

时间：2026-10-03T09:04:32.243723+02:00。仅空闲物理GPU3执行短时数值探针；所有检查零optimizer更新，未启动正式训练、未改队列或原始门槛/结果。GPU0/1未使用。PyTorch=2.4.1+cu121，CUDA=12.1。

用户给出的分层判断正确：完成训练、一阶同状态一致性和HVP数值验收是三个层次。原先缺少误差幅度和重复控制，本次补上首个失败状态及最差HVP案例的定量证据。

## 原始一阶诊断在哪里失败

Stage1_mechanism_diagnostics在Task0、entry、batch0（128样本）检查P/D共享g0、g1+、g1−时，严格torch.equal失败。四个分支其实都已算完；比较循环在D就抛错，因此没有保存C/R比较。普通梯度使用create_graph=False，原始报告仅保存布尔值。每个分支后的State恢复检查已完成，最终finally恢复了entry覆盖状态；原生end-checkpoint的任务循环收尾检查因异常未到达。没有optimizer更新或checkpoint写入。

本次重建同一个模型、entry的50个FP32可训练张量、同一注册输入/标签/索引与forward随机数种子；源文件、模型、输入和完成后恢复hash通过核验。固定基点和固定φ1±各做3次重复，再做P/P/D全探针重复。

| 检查 | native重复相对L2误差 | math重复相对L2误差 |
|---|---:|---:|
| 固定位置g0 | 2.035886822e-08 | 0 |
| 固定位置g1_plus | 2.0458856e-08 | 0 |
| 固定位置g1_minus | 2.061130382e-08 | 0 |
| 同分支P/P重构q | 0.0001746054162 | 0 |
| 同分支P/P重构c21 | 0.000299867677 | 0 |
| 同分支P/P重构U | 9.743110514e-05 | 0 |

固定位置native梯度的最大绝对误差7.45058059692e−9，损失与分类头梯度完全相等，差异出现在LoRA梯度。开启torch.use_deterministic_algorithms(True)后，native的三个g0重复也逐位一致。math完整P/P逐位一致，P/D共享的g0/g1±亦逐位一致。

这些控制支持该失败案例的native算子重复性解释，未观察到缓存、参数或随机状态恢复错误。具体CUDA内核没有被profile，不能将原因钉死到某一个attention算子。噪声会被探针重构放大：最终U相对误差0.009743%，cosine=0.999999995256。这个结果不支持直接作废全部普通训练，也不能证明全部24个注册状态已验收。

## 原始HVP门禁在哪里失败

初始native可行性检查失败于efficient SDPA不支持二阶反传，这是算子支持问题。后来完整门禁切到math；全部12个同点普通梯度与create_graph=True一阶梯度比较relative L2为0，HVP有限、状态恢复通过，但中心差分验证失败。此处与native的一阶逐位重复失败不是同一项检查。

代码计算H(x)d=∇x[∇L(x)^T d]，与FD_h=[g(x+h d)−g(x−h d)]/(2h)比较；d为实际φ2−φ1快照，Hstart在φ1，Hmid在两点中点。只验证Hessian-vector product，不构造整个Hessian。FD与有限路径Sec=g2−g1的比较不是验收标准。

冻结标准：relative L2≤.005（0.5%），cosine≥.999，并要求网格中两个相邻h同时通过；h_grid=.1/.03/.01/.003/.001。h乘的是未归一化的实际位移d，不是训练半径。||d||=.2152–.2347，h=.001对应名义扰动约2.15e−4–2.35e−4。全部参数/梯度/HVP/差分运算均FP32，只有范数和点积归约用FP64。

| Task / batch | Hstart最小误差（%） | Hmid最小误差（%） | 相邻两点验收 |
|---|---:|---:|---|
| 0 / 0 | 0.208799 | 0.255730 | 两者通过 |
| 0 / 1 | 1.693282 | 7.642462 | 两者未通过 |
| 4 / 0 | 2.789447 | 5.212630 | 两者未通过 |
| 4 / 1 | 2.657643 | 0.337154 | 未通过（Hmid仅单个h通过） |
| 9 / 0 | 45.126575 | 1.096558 | 两者未通过 |
| 9 / 1 | 1.173442 | 1.372762 | 两者未通过 |

总计2/12响应、1/6批次通过；不是所有中心差分点都失败。Task4/batch1/Hmid虽然最佳误差0.337154%低于门槛，但只有h=.001通过，未满足相邻两点要求。代码公式、固定向量长度、相对误差分母和后端一致性未发现明确错误；HVP参考范数.8944–14.0246，不是接近零的分母问题。

## 最差案例的新数值定位：Task9 / batch0 / Hstart

精确重建原始FP32模型、批次、点和d，复现原h=.001的HVP范数及FD误差，随后扫描更小h。整批128 FP64高阶图曾OOM，保留失败记录；之后FP64对相同128样本分成两个64样本chunk，按样本数加权累加梯度/HVP，保持相同数学平均CE目标、相同原始点/d。模型无训练BatchNorm或随机dropout。分块改变GEMM形状与归约顺序，故不能将差异解释为仅dtype的单因素效应。

| h | FP32误差（%） | FP64分块误差（%） |
|---|---:|---:|
| 0.001 | 45.126575 | 45.066466 |
| 0.0003 | 4.468425 | 4.483027 |
| 0.0001 | 1.944359 | 0.502698 |
| 3e-05 | 8.387908 | 0.045290 |
| 1e-05 | 19.191646 | 0.005033 |

FP64在3e−5和1e−5两个相邻扫描点满足相同0.5%/cosine门槛，且小h误差近似按h²下降。这说明该案例中心差分可以逼近HVP；原h=.001在FP64仍约45%误差，支持该位置的有限差分截断误差过大。FP32在h继续减小时误差反而从1.944%升到8.388%和19.192%，支持低精度舍入/梯度相减消去成为限制。不能从单案例认定其余10个失败响应原因完全相同。

FP64 HVP相对FP32 HVP差异0.904950%；因此不能把FP32自动微分HVP也直接当成绝对真值。已保存实际扰动、gp/gm/Hd、逐block误差、舍入方向偏差与原始张量。

## 对结果和接续的影响

- 16条S1完整训练继续作为原执行协议的描述性结果保留；当前首个状态的误差较小，但未完成全部状态的一阶一致性复核。
- 新控制给出数值原因证据，不构成Hessian机制已被验证；局部HVP对照通过也不说明它能替代有限路径响应或带来持续学习收益。
- 原HVP门禁、原同状态诊断和队列状态保持失败，原文件hash未改变；第二阶段8条正式训练仍未启动。单案例新扫描通过不允许自动解锁。
- 后续应在固定确定性设置下补全24个同状态案例；对完整6批次/12响应建立内存可行的高精度/细h验收协议，保留标准与原失败记录，再独立复核。不能用扩大容差替代定位。

## 复查文件

- [gradient_repeats/artifacts/repeat_report.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-numerical-localization-20261003T065100Z/gradient_repeats/artifacts/repeat_report.json)
- [hvp_precision/artifacts/probe.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-numerical-localization-20261003T065100Z/hvp_precision/artifacts/probe.json)
- [hvp_precision_chunked/artifacts/probe.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-numerical-localization-20261003T065100Z/hvp_precision_chunked/artifacts/probe.json)
- [hvp_fp32_finer/artifacts/probe.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-numerical-localization-20261003T065100Z/hvp_fp32_finer/artifacts/probe.json)
- [worst_case_fd_scan.csv](/data/140-1/users/liying/O-LoRA/refine-logs/g012-numerical-localization-20261003T065100Z/worst_case_fd_scan.csv)
- [原完整HVP门禁](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/HVP_full_gate/hvp_gate_report.json)；[原一阶机制诊断](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/Stage1_mechanism_diagnostics/stage1_mechanism_report.json)。
- [PyTorch有限差分gradcheck的低精度说明](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.gradcheck.gradcheck.html)；[确定性算法与随机种子的区别](https://docs.pytorch.org/docs/2.9/notes/randomness.html)。文档说明不能单独替代本报告的具体数值控制。

