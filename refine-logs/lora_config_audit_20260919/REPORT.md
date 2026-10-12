# 五种 LoRA 方法的训练配置与可比性

核查日期：2026-09-19。范围为当前结果表所用的 Vanilla LoRA、InfLoRA、OLoRA、SDLoRA、CL-LoRA，各6个数据集，共30项运行；没有改动实验或结果表。

**结论：共同训练骨架大体一致，但不能写成“除方法自身外，其余配置完全相同”；同一方法也没有在所有数据集上使用完全相同的配置。** 明确的已记录差异是 SDLoRA/Aircraft 的学习率和 5Datasets 的训练轮数；另外，CL-LoRA 的学习率下限、数据加载设置与历史基线的现存代码默认值有差异，旧日志未完整固化这些字段。

证据分三层：30项原日志和对应配置均已读取并核对共同字段；CL-LoRA还有与receipt绑定的冻结配置/源码；历史四方法未记录的默认值只能由当前可读源码推导，不冒充历史运行时已证明的值。

## 1. 已记录的共同设置

| 配置 | 当前30项运行 |
|---|---|
| 骨干 | ViT-B/16，backbone_type=vit_base_patch16_224 |
| 预训练标识 | timm/vit_base_patch16_224.augreg2_in21k_ft_in1k |
| LoRA rank | 16 |
| Batch size | 128 |
| Seed | 0；这五行都是单次seed0，不含EWC行的seed例外 |
| Optimizer | SGD |
| 首任务/后续momentum | 0 / 0 |
| 首任务/后续weight decay | 0 / 0 |
| Scheduler名称 | cosine；最低学习率并非已证一致，见第4节 |
| 类别顺序 | class_shuffle=false，自然类顺序 |
| 训练模式 | all_or_inc=inc |
| Replay配置 | memory_size=0，memory_per_class=0，fixed_memory=false |
| 初始/后续LR | 每项运行内init_lr=lrate；具体值随数据集变化 |
| 初始/后续epochs | 每项运行内init_epoch=epochs；具体值如下 |
| 表格指标 | CNN任务矩阵；FAA=最终阶段任务宏平均，AAA=阶段任务宏平均再取平均 |

上述“无replay”指优化训练没有回放旧样本。现存四种历史基线代码还会读取旧类训练图像计算另一路NME评估的类均值；此表选取CNN结果，不能把整个程序说成完全不再访问旧训练数据。

相同rank只表示每个adapter的秩相同，不代表累计adapter数量、可训练参数、分类头或训练/推理计算量相同。

## 2. 各数据集的具体训练设置

### 学习率：初始任务与后续任务相同

| 数据集 | Vanilla | InfLoRA | OLoRA | SDLoRA | CL-LoRA |
|---|---:|---:|---:|---:|---:|
| Aircraft | .05 | .05 | .05 | **.02** | .05 |
| Cars | .05 | .05 | .05 | .05 | .05 |
| CUB-200 | .01 | .01 | .01 | .01 | .01 |
| Flowers | .0025 | .0025 | .0025 | .0025 | .0025 |
| Pets | .0025 | .0025 | .0025 | .0025 | .0025 |
| 5Datasets | .02 | .02 | .02 | .02 | .02 |

### 每任务训练轮数：初始任务与后续任务相同

| 数据集 | Vanilla | InfLoRA | OLoRA | SDLoRA | CL-LoRA |
|---|---:|---:|---:|---:|---:|
| Aircraft / Cars / CUB / Flowers / Pets | 40 | 40 | 40 | 40 | 40 |
| 5Datasets | **30** | **40** | **40** | **40** | **30** |

### 任务划分：五种方法在同一数据集上的实际边界一致

| 数据集 | 总类别数 | 任务数 | 首任务类别数 | 后续新类数 |
|---|---:|---:|---:|---|
| Aircraft | 100 | 10 | 10 | 每次10 |
| Cars | 196 | 10 | 16 | 每次20，共9次 |
| CUB-200 | 200 | 10 | 20 | 每次20，共9次 |
| Flowers | 102 | 10 | 12 | 每次10，共9次 |
| Pets | 37 | 9 | 5 | 每次4，共8次 |
| 5Datasets | 635 | 5 | CUB的200类 | Aircraft100 → Cars196 → Flowers102 → Pets37 |

5Datasets历史配置的increment=1，CL配置increment=100，但实际均由dataset.task_splits=[200,100,196,102,37]覆盖；不是任务划分不一致。不能根据配置名中的t20或目录名中的lr0025判断实际任务数和学习率；本报告以日志实际字段和任务边界为准。

## 3. 方法自身的具体配置，以及跨数据集是否相同

### Vanilla LoRA

当前表的Vanilla实现名是seqlora：同一套LoRA A/B跨任务持续训练，基础ViT冻结；12个Transformer block的Q/V投影插入rank16，K不插入。当前源码A为Kaiming uniform，B初始化为0，直接加BA，没有独立alpha/r或LoRA dropout参数。分类头是带bias的扩展线性头，新增类别时保留旧类别行。后续任务训练只使用新类logits的交叉熵，测试对全部已见类分类。

方法规则跨六数据集一致；LR、任务数、任务大小和Het5轮数随数据集改变。

### InfLoRA

当前可读实现同样作用于12层Q/V，rank16。利用当前任务输入二阶矩/SVD及DualGPM历史子空间构建A，A冻结，只更新B及分类头；当前A的初始化缩放为1/sqrt(3)，不是普通随机A训练。旧任务adapter保留。

专属键为lamb和lame，当前源码默认分别0.7和0.99；这两个字段在本次6份历史日志及YAML中均没有显式记录。按当前代码，无外部覆盖时，任务t的阈值为：

`threshold(t) = 0.7 + (0.99 - 0.7) * t / total_sessions`，t从0开始。

| 数据集 | total_sessions | 最后任务阈值（按当前默认） |
|---|---:|---:|
| Aircraft / Cars / CUB / Flowers | 10 | .961 |
| Pets | 9 | .95777778 |
| 5Datasets | 5 | .932 |

因此，常量默认值与公式相同，实际阈值序列因任务总数不同；最后任务不会到0.99。当前代码每任务训练前后各完整遍历当前训练集统计输入二阶矩，没有固定少量样本上限。细节和历史证据边界见[inflora.md](inflora.md)。

### OLoRA

当前可读实现12层Q/V、rank16，每任务新增A/B，旧A/B冻结并在前向累加。A为Kaiming uniform，B=0；learn_alpha=false，adapter贡献系数为1。除首任务外，损失添加新旧A间的正交约束。

- 正交系数的实际键拼作 `olora_lamda_1`，默认1.0。
- L2系数为 `olora_lamda_2`，默认0.0；不能声称启用了非零L2正则。
- 正交项为对历史任务、层、Q/V的 `||A_old A_new^T||_F^2` 求和，没有除以任务数/层数/rank。

六个数据集没有显式覆写这些专属键；按当前代码采用相同默认规则。但历史日志未固化这两个系数，不能将当前默认当成历史事实。虽然系数固定，正交项随旧任务数量增加。见[OLoRA/SDLoRA详查](olora_sdlora.md)。

### SDLoRA

当前可读实现12层Q/V、rank16，每任务新增A/B并保留旧方向。learn_alpha=true；每个任务有一个跨全部层及Q/V共用的可学习缩放scalar，初始化1。默认max_prev_tasks=200，预先注册200个历史scalar和1个当前scalar；并不意味着运行了201个任务。使用masked CE，没有SD专属正交/L2项或独立alpha学习率。

当前实现每次新任务训练会重建backbone，把各alpha重新初始化为1；仅eval=True时加载保存的缩放。这属于当前源码行为，不应依赖注释宣称训练时沿用旧alpha，亦不能证明历史未锁定版本完全相同。六数据集未给出不同专属参数；SDLoRA/Aircraft的LR=.02是通用训练配置的例外。

### CL-LoRA

以下是与运行绑定的冻结配置/源码，而非未验证的历史默认值。六个数据集的方法专属参数全部相同：

| 参数/机制 | 实际值 |
|---|---|
| model_name | cllora_aligned |
| msa | [1,0,1]，Q/V开启、K关闭 |
| shared_pos | [0,1,2,3,4,5] |
| specific_pos | [6,7,8,9,10,11] |
| lora_rank | 16 |
| kd_ratio | 5.0 |
| temperature | 2.0；KD未额外乘temperature² |
| orth_ratio | 0.0001 |
| shared A/B | 正交A初始化并冻结；B初始化0且更新 |
| specific A/B | 每任务一套；A为Kaiming、B=0；只更新当前任务分支 |
| block weights | 每任务6×3，可学习；初始Uniform(.5,1.5)；前向使用Q/V列 |
| 正交约束对象 | 任务间block-weight向量，并非OLoRA的A矩阵 |
| 训练分类头 | 当前任务cosine proxy头，sigma可学习、初始1 |
| 最终分类器 | 当前任务训练图像、test-mode变换下计算类原型，cosine分类；不使用测试标签构建原型 |
| 梯度步骤 | 首任务每batch一次；后续先KD一步，再CE+orth一步 |
| 推理 | 所有已见任务分支计算并拼接全已见类分数；不依赖真实任务标签选分支 |

方法专属超参数确实跨六数据集相同；分支数量、头大小及计算量会随任务数/类别数变化。详见[冻结实现详查](cllora.md)。

## 4. 不能声称“只改变方法本身”的其余原因

### Cosine最低学习率

- CL冻结代码：CosineAnnealingLR(T_max=epochs)，eta_min采用默认0，每epoch末step。
- 四个历史基线当前源码：默认eta_min=0.1×本阶段初始LR，每epoch末step；六数据集的日志/配置均未显式记录min_lr。
- 因此已有源码显示执行默认差异，历史最低LR仍缺逐运行见证。不能因日志同为cosine就认定调度完全相同。

### 数据加载器

CL冻结代码训练/测试/原型loader均采用batch128、workers2、pin_memory=true、persistent_workers=false、drop_last=false。eval_num_workers虽配置为2，但该自建loader实际统一读取train_num_workers。

Vanilla当前代码默认workers8；当前DataManager对Aircraft使用pin_memory=true、persistent_workers=true，其他数据集默认false/false。旧运行未打印这些字段，不能倒推其历史实际值，也不能认证与CL相同随机数据加载轨迹。

### 预处理与实际数据身份

当前历史工程data.py与CL冻结data.py中三个预处理构造函数的AST相同；这只能证明当前可读代码层面的对应，历史源码未完整绑定。

| 数据集 | 训练预处理 | 测试预处理 | 归一化 |
|---|---|---|---|
| CUB / Flowers / Pets | Bicubic resize 256×256，随机crop224，水平翻转 | resize256×256，center crop224 | mean=(.485,.456,.406)，std=(.229,.224,.225) |
| Aircraft | 同上几何变换；tensor loader | 同上几何变换 | 同上；ConvertImageDtype(float32) |
| Cars单数据集 | Bicubic短边resize224，center crop224，水平翻转 | 短边resize224，center crop224 | mean=0，std=1 |
| 5Datasets | 全部任务统一CUB变换 | 全部任务统一CUB变换 | ImageNet mean/std |

因此各数据集的预处理并非全部相同，特别是Cars单数据集与5Datasets中的Cars。

新CL训练数据根为/data/115-2/users/liying/datasets，清单已冻结。历史图片清单、历史权重revision/文件SHA和运行源码没有完整锁定；同hub ID、同类别数不能证明逐图像逐字节完全一致。新CL的CUB实际train/test=9430/2358，不应直接写成标准官方划分。当前Flowers目录上新旧索引方式输出一致，不能把CL优势归因为已证明的旧标签错配。

### 无关配置字段不能当成启用的训练阶段

四个历史配置包含probe_fit_epochs=20、probe_fit_lr=.015、probe_fit_wd=.0005等字段，但linear_probe_softmax_joint_seen_eval和per_task_eval均为false。这些不是当前CNN行额外执行20轮训练的证据。flat_eval=false时，下属sharpness/Hessian开关也不能当成已运行分析。

## 5. 可以准确使用的表述

可写：五种方法共享ViT-B/16预训练模型标识、rank16、batch128、seed0、SGD、无replay和对应任务划分；学习率按数据集设置，并披露SDLoRA/Aircraft与Het5轮数例外。CL-LoRA保留其共享/专属adapter、蒸馏和原型分类机制。

不宜写：除方法专属参数外全部配置完全相同；所有方法在全部数据集上采用同一套超参数；相同rank保证相同参数或计算预算。

若要得到更严格的共同训练协议，应先统一Aircraft学习率、Het5训练轮数及cosine下限，再固定共同数据清单/权重SHA和加载器设置；本次没有启动这类重跑。

## 6. 交付与来源

- [30项完整配置、原始日志参数、路径与SHA](configurations_full.json)
- [30项核心配置CSV](protocol.csv)
- [逐方法相对Vanilla的字段差异](differences_vs_vanilla.json)
- [三个预处理函数的AST对照](transforms.json)
- [InfLoRA专属参数与代码证据](inflora.md)
- [OLoRA/SDLoRA专属参数与代码证据](olora_sdlora.md)
- [CL-LoRA六份冻结配置及代码证据](cllora.md)
- [数据与权重身份的既有详细核查](../cllora_comparison_20260918/data_audit.md)
