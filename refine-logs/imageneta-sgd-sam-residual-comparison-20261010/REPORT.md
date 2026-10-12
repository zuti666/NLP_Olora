# ImageNet-A：以SGD和SAM为基点的梯度残差对比

核验时间UTC：2026-10-10T07:50:08.950184+00:00。本次只读取实验证据、生成派生结果，没有修改训练代码、运行队列或启动实验。

## 数据集和训练设置

ImageNet-A本地train5981/test1519，200类固定顺序0…199，10任务×20类。每个方法独立从相同seed初始化重新训练整个10任务序列，依次学习各任务；训练使用当前任务样本，测试在所有已见类别中联合分类。任务数不是独立重复；seed0/1/2才是三次训练重复。

ViT-B/16 augreg2_in21k_ft_in1k，Q/V LoRA rank10，冻结骨干、训练LoRA及分类头；每任务50epochs，共500epochs/2550更新，batch128；SGD momentum/weight_decay=0，Task0 LR=.04、随后=.02，cosine最低倍率.1。内半径rho=.05，外半径r=.25或1。EWC/Fisher及其计算、实际权重融合、replay关闭；下面自然步长方向组不做最终梯度投影、正交修正或范数匹配。匹配步长组独立列出。

FAA为最终10任务等权均值，AAA为10阶段的已见任务等权均值再平均，BWT为前9任务最终准确率减各自刚学完准确率的均值。FAA/AAA单位%，BWT与差值单位百分点。±为seed0/1/2样本SD(ddof=1)；n1不填SD。BWT越高越好；本表负值越接近0表示平均退化越少，但低学习准确率下的低BWT不是良好保持。

## 先纠正公式

SAM最终方向是g1=g0+(g1−g0)。g0+(g2−g1)是独立的跨点残差实验A，不等于g1。

定义c10=g1−g0，c21=g2−g1，c20=g2−g0，c32=g3−g2。正向探针为：

$$g_0=g(\theta),\quad g_1=g(\theta+\rho N(g_0)),\quad g_2=g(\theta+rN(g_1-g_0)),\quad g_3=g(\theta+rN(g_1-g_0)+\rho N(g_2)).$$

外点从原theta出发，不是从第一探针位置theta1继续移动。探测结束恢复theta，才使用最终U执行一次theta←theta−eta U。g0/g1方法是相同探针预算下的SGD/SAM方向参考，不代表原生算法运行成本。

## 分组问题：同一残差分别加到两个基点

| 加入的分量 | SGD基点g0 | SAM基点g1 | 对照问题 |
|---|---|---|---|
| 无残差 | g0 | g1 | 基准 |
| 跨点残差c21 | g0+(g2−g1)，A | g1+(g2−g1)=g2 | 同一跨点差分能否帮助两个基点？ |
| 第二局部响应c32 | g0+(g3−g2)，UG | g1+(g3−g2)，D32 | 外点局部探针的响应能否迁移回两个基点？ |
| 整体位置响应c20 | g0+(g2−g0)=g2 | g1+(g2−g0)，U3 | 外点位置改变是否有效，是否与第一次SAM响应互补？ |

g2是同一个实验，同时对应SAM+c21与SGD+c20，不能重复计数。U3=g1+g2−g0，不能用已有g1+g2成绩替换。

同一模型状态下，A=g2−c10、g2=g2、U3=g2+c10；它们比较是否应减去或加回第一次SAM响应。但三个独立训练轨迹不会一直保持同一参数状态，终点差不是逐步同状态因果测量。所有涉及g2/c32的构造仍用c10选择外点，没有删除首次探针。

## r=1：精确恢复与RNG重放族

以下P族与新正向组合使用共同R0重放和最终状态恢复规则；P族四读取为g0/g1+/g1−/g2，新D32/U3为g0/g1/g2/g3。不用的探针不会改变最终提交状态/RNG，但计算内容不同。完整三seed与单seed分别标记。

| Method | Final direction | Seeds | FAA (%) | AAA (%) | BWT (pp) |
|---|---|---:|---:|---:|---:|
| SGD | $g_0$ | 3* | 40.574±1.613 | 60.873±0.492 | -12.084±1.063 |
| SAM | $g_1$ | 3* | 45.375±0.714 | 63.823±0.520 | -9.699±0.854 |
| SGD + cross-location residual | $g_0+(g_2-g_1)$ | 3 | 49.874±0.749 | 65.664±0.229 | -6.819±2.314 |
| Outer gradient | $g_2=g_1+(g_2-g_1)=g_0+(g_2-g_0)$ | 3 | 49.694±0.537 | 65.212±0.889 | -6.831±2.027 |
| SAM + second local residual | $g_1+(g_3-g_2)$ | 3 | 49.610±1.746 | 65.985±0.382 | -6.621±0.845 |
| SAM + total-location residual | $g_1+(g_2-g_0)$ | 1 | 46.873 | 64.775 | -8.224 |

*P01/P02的seed2基准来自r=.25，因最终g0/g1不依赖外半径、额外探针全状态/RNG恢复而在r=1身份复用；不是新增训练，也不是所有三个JSON均literal r=1。若要求每份JSON都r=1，这两个基准仅seed0/1实跑r=1。所有其他表中方法的r=1成员使用实际r=1配置。

该执行族尚缺非严格确定性的g0+c32完整对照；下面原生/严格确定性的同公式结果独立列，不能直接拼入此族计算基线增益。

配对结论（候选−各自基线）：A−SGD FAA+9.300±.994、AAA+4.791±.703、BWT+5.265±1.255均3/3；g2−SAM FAA+4.319±1.173、AAA+1.389±1.172均3/3，BWT+2.867±2.850为2/3；D32−SAM FAA+4.235±1.469、AAA+2.162±.202、BWT+3.077±.506均3/3。

D32−g2的配对FAA−.084±2.270，仅1/3胜出，因此不能称第二局部残差优于直接提交g2。U3只有seed2：相对同seed SAM，FAA+2.295/AAA+1.542/BWT+1.914；相对g2则FAA−3.377/AAA−.960/BWT−3.050，暂不支持第一SAM响应加回g2一定有帮助。

## r=1：原生global-RNG五CE族

每步原生四探针加一次clean CE，随机状态自然推进，减扰动恢复；与上节执行协议分开。5方法×3seed=15/15完整。

| Method | Final direction | Seeds | FAA (%) | AAA (%) | BWT (pp) |
|---|---|---:|---:|---:|---:|
| SGD | $g_0$ | 3 | 40.063±0.726 | 60.134±0.709 | -12.809±0.662 |
| SAM | $g_1$ | 3 | 45.559±0.511 | 63.204±0.885 | -9.800±1.156 |
| SGD + cross-location residual | $g_0+(g_2-g_1)$ | 3 | 50.628±1.157 | 65.941±1.366 | -6.294±0.132 |
| SGD + second local residual | $g_0+(g_3-g_2)$ | 3 | 46.921±2.200 | 63.128±2.683 | -7.134±0.801 |
| Outer gradient | $g_2=g_1+(g_2-g_1)=g_0+(g_2-g_0)$ | 3 | 48.451±2.510 | 63.924±1.134 | -7.354±1.757 |

A−SGD的配对FAA+10.565±1.280，3/3；UG−SGD为+6.859±2.216，3/3；g2−SAM为+2.891±2.815，只有2/3改善（seed0−.300），所以不能把“每个协议每个seed均帮助SAM”作为结论。跨点残差加回SGD比末端局部残差FAA+3.706±3.328，3/3，但仍是自然步长。

## 已完成的有效步长匹配对照

| Method | Final direction | Seeds | FAA (%) | AAA (%) | BWT (pp) |
|---|---|---:|---:|---:|---:|
| SGD, matched step | $g_0$ | 3* | 40.122±1.398 | 60.540±0.570 | -13.167±1.318 |
| SGD + cross-location residual, matched step | $g_0+(g_2-g_1)$ | 3 | 46.488±2.343 | 64.416±0.981 | -7.366±1.026 |

P16的seed2是同样不依赖外半径的已登记身份参考复用；seed0/1正式训练已完成，旧大队列60/60完成、30种原队列条件三seed齐全。matched按同候选当前模型状态下正向g1的LoRA BA与seen FC联合位移匹配，不是简单固定乘常数。

匹配后的A−匹配g0：FAA+6.366±2.805、3/3胜出；匹配A−SAM：FAA+1.113±1.841、2/3胜出。自然A−匹配A的FAA+3.386，3/3。幅度有贡献，匹配g0对照仍保留收益；相对SAM的额外优势减弱。不同训练轨迹不能据此认定已测得纯Hessian因果作用。

## 最新严格确定性v2：同残差、换基点

4/4完整，全部seed2、自然步长；严格CUDA deterministic/native FP32、TF32/AMP关闭。缺同严格协议的g0/g1零残差完整基线，故不计算相对旧基线的正式增益。

| r | Method | Direction | FAA | AAA | BWT |
|---:|---|---|---:|---:|---:|
| 0.25 | SGD + GAM | $g0+(g3+−g2)$ | 47.706 | 64.914 | -8.058 |
| 0.25 | SAM + GAM | $g1+(g3+−g2)$ | 48.181 | 65.368 | -7.618 |
| 1.0 | SGD + GAM | $g0+(g3+−g2)$ | 47.584 | 65.176 | -7.471 |
| 1.0 | SAM + GAM | $g1+(g3+−g2)$ | 47.562 | 65.335 | -7.999 |

r=1，SAM基点比SGD基点AAA+.159，但FAA−.022；r=.25为AAA+.454/FAA+.475。当前不能概括SAM基点永远更好，也不能把单seed很小的差值当成统计等价。r1 B1的BWT已按原矩阵确认−7.999。

## 当前可支持的结论

1. 跨点残差c21加到SGD有一致跨seed收益；在精确恢复协议下，加到SAM也有3/3的FAA与AAA收益。原生协议对SAM只2/3，泛化措辞应限定协议。
2. c32可以在不同执行族分别帮助SGD或SAM；缺单一协议下四个完整基准格，不宣称已建立全协议统一的SGD/SAM×有无c32因果交互。
3. g2既可以视为外点梯度，也可以视为SAM加c21。其有效不意味着g1+c20必然更好；U3仅单seed且落后同seed g2。
4. 新严格确定性基点比较没有显示SAM基点在两个半径均有FAA优势；局部响应符号与有效更新幅度仍重要。
5. 固定r实验仍不能证明Hessian机制、第二段独立于首次探针，或跨协议/数据集均有效。论文正文应以同残差换基点的分组呈现；先补同执行协议零残差与c32基点格，再扩seed。这里只记录缺口，没有启动训练。

10月9日动态eta||g0||预测器响应实验，以及Task0→Task1最小fork改变半径/任务/算法定义，不填入本固定rho/r十任务表。ImageNet-R也不并入ImageNet-A统计。

## 完整性与来源

相关原始结果引用56条，按完成收据去重为53次完整训练。原生15/15；精确恢复主表16/16；步长匹配6/6；严格v2本次涉及4/4。已登记结果无失败或非有限终点；缺同协议方法和未登记设置不会填0或冒充已完成。独立原矩阵重算、来源存在与严格v2完成哈希核验通过；各自执行协议单列。

[各方法结果CSV](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-sgd-sam-residual-comparison-20261010/METHOD_RESULTS.csv) · [逐条原始来源](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-sgd-sam-residual-comparison-20261010/SOURCE_ROWS.csv) · [配对差值](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-sgd-sam-residual-comparison-20261010/PAIRED_CONTRASTS.json) · [完整只读审计](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-sgd-sam-residual-comparison-20261010/SOURCE_AUDIT.json)。

探针代码：[direction_step.py](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/trainer/direction_step.py:161)。U3代码：[direction_step.py](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/trainer/direction_step.py:66)。
