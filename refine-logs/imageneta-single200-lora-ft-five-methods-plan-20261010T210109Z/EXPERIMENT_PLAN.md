# ImageNet-A 全200类单任务：LoRA / 全量FT五方法对照

日期：2026-10-10，Europe/Paris。状态：设置计划，新增训练未注册、未启动；现有队列不变。首轮新实验仅seed2。

## 已有证据及需要补齐的范围

真正的普通监督训练已存在：`../imageneta-upward-eight-fork-20261010T123703Z/trainer/upward_single_train.py` 使用全5981训练图像、1519测试图像、单个200类分类头。旧十任务的20类Task0不能代替此设置。

截至2026-10-10 20:59:59 UTC，该批全200类共18项（六方向×seed2/0/1），13项完成、5项运行。seed2的六项已完成。以下五方法主表仅填已核验的同协议正式结果，准确率为全1519张测试图像的样本加权准确率。

| 方法 | LoRA seed2准确率 % | 全量FT | 计划状态 |
|---|---:|---|---|
| SGD / g0 | 63.001975 | 未找到同协议成果 | LoRA复用；FT待实现 |
| SAM / g1+ | 64.186965 | 未找到同协议成果 | LoRA复用；FT待实现 |
| 下降侧镜像 / g0+(g0-g1-) | 待补 | 未找到同协议成果 | 两范围均待实现 |
| 固定正向构造g2 | 68.597762 | 未找到同协议成果 | LoRA复用；FT待实现 |
| 完整纯GAM / beta全1、gamma=.1 | 待补 | 未找到同协议成果 | 两范围均待实现 |

已有额外seed2分量结果：g2-(g1-g0)=68.202765%、g1+(g3-g2)=66.293614%、g3=68.663594%。这些分量变体不冒充完整GAM。

昨天 `../imageneta-response-four-methods-gpu567-20261009T103449Z` 的单任务current_g2=63.66%使用下降预测点和动态半径eta*norm(g0)，其旧GAM半径字段不生效，不能填入本计划固定rho=.05/r=1的g2行。

FT检索范围：O-LoRA/refine-logs、CopyFlatCV/refine-logs与exps（存在时）、Flatness_CV直接配置，排除复制source及庞大输出。82个init_cls=200候选中36个ImageNet-A直接配置，未找到同协议全200类FT设置/成果。历史fullmodel十任务、EWC/Fisher/融合成绩不复用为单任务纯方法结果。

## 要验证的问题

1. 下降侧镜像、固定g2及完整GAM是否在普通监督训练中优于各自同范围的SGD/SAM？已有LoRA g2 seed2相对SGD提高5.595787个百分点，相对SAM提高4.410797，说明该现象可以出现在无CL任务切换的训练中；不能概括为全部CL收益的来源。
2. 同样的更新构造是否能够用于全量FT？两范围内分别比较方法；不同参数化、学习率以及半径作用不能由跨范围绝对准确率单独归因。

本轮seed2为探索性对照。已有seed0/1可另作配对统计，但不把seed2重复运行计为新seed；不自动扩展新增方法的多seed或超参扫描。

## 统一数据、模型和训练设置

- 数据：沿用本项目ImageNet-A本地train/test划分，5981/1519张，200类固定顺序0…199。明确这是本地监督训练协议，不表述为原始ImageNet-A提供官方训练集。
- 一次性训练：init_cls=200、increment=200、nb_tasks=1、total_sessions=1；全200类CE，从初始化就建立200行分类头，无任务切换/分类头扩张。
- 骨干：ViT-B/16，显式沿用augreg2_in21k_ft_in1k权重。所有方法同范围同seed使用相同初始参数、分类头、实际输入/增强/LR序列。
- LoRA：各层Q/V rank10，沿用scale1实现；冻结骨干，只训练LoRA A/B与FC200。现有LR=.04，cosine最低.004。
- FT：不插入LoRA，全部原骨干参数与同类型FC200可训练；新计划初始LR=.001，cosine最低.0001。该LR是计划初值，不是已验证最佳值，范围内五方法统一使用；不得直接在LoRA容器里改requires_grad假称FT实现完成。
- 预算：50epochs，batch128，drop_last=false；每epoch47batch，2350更新，每图50遍。
- 实际优化器：torch.optim.SGD，momentum=0、weight_decay=0；无AMP/TF32，FP32原生attention、严格确定性及原状态/RNG事务。
- 全部禁用EWC/Fisher估计/归一化/阻尼、权重融合、历史梯度缓存、replay、SGV、漂移惩罚、最终裁剪/范数匹配、HVP/二阶autograd。完整GAM保留其定义内gamma正交修正；其他四方法无此修正。
- rho=gam_grad_rho=.05；r=gam_grad_norm_rho=1。N(v)=v/(globalnorm(v)+1e-12)，全局范数覆盖该范围的全部可训练参数。构造方向单位化，最终U不单位化；无自适应权重乘积扰动。FT与LoRA同数值半径不代表同样的函数空间扰动。

## 五种算法

theta表示本范围全部可训练参数。每个probe从完整入口状态恢复，使用同一TRAIN batch与同一R0，梯度普通读取、create_graph=false、不穿过探针位置反传。

\[
g_0=g(\theta),\quad g_1^+=g(\theta+\rho N(g_0)),\quad g_1^-=g(\theta-\rho N(g_0)),
\]
\[
\theta_2=\theta+rN(g_1^+-g_0),\quad g_2=g(\theta_2),\quad g_3=g(\theta_2+\rho N(g_2)).
\]

外点从theta出发，不是theta1再加r。所有方法都恢复theta，提交一次真实更新theta <- theta-eta*U，并仅推进基点g0读取对应R1。

| 方法 | 最终U | 必需训练梯度读取/step |
|---|---|---:|
| SGD | g0 | 1 |
| SAM | g1+ | 2 |
| 下降侧镜像 | g0+(g0-g1-) | 2 |
| g2 | g2 | 3 |
| 完整纯GAM | h+ - .1*[h- - proj(h-,h+)] | 4 |

完整GAM：h-=g0+g2，h+=g1++g3；projection在全部可训练参数上全局计算，沿用原gam.py的epsilon防护。beta1=beta2=beta3=1，gamma=.1，无SGV。该GAM不是仅g3，也不是g1+(g3-g2)。同epochs比较包含不同读取成本与更新幅度，不能仅解释为纯方向优势。

## 最小补齐矩阵及代码工作

复用seed2 LoRA的g0/g1/g2三项；新增LoRA镜像/完整GAM两项，加FT五项，共7项正式训练。总完整五方法×两范围为10个单seed单任务单元，不重复已完成作业。

1. 新建隔离执行目录，复用upward_single_train的真实全200类数据/评估逻辑以及state_transaction。已注册/完成目录、source190与现有队列不修改。
2. 扩展方向执行器加入下降侧镜像和完整GAM；镜像仅g0/g1-两读，GAM四读并执行gamma修正。保持所有探针/真正更新在正确位置。
3. FT建立无LoRA的原生ViT+同SimpleLinear FC200，显式核对预训练tag及权重SHA、完整trainable名单、无LoRA模块、骨干梯度及一步后权重变化。沿用已完成LoRA seed2的分类头初始化，可从其初始状态读取，或按冻结来源重建后核对初始参数哈希；不能仅设同seed便假定两范围分类头相同，LoRA初始化会消耗额外RNG。现有EffectiveStep.from_model依赖w_As/w_Bs，须用普通参数位移adapter替代；本计划不做范数匹配。现有只保存LoRA A/B+FC的checkpoint逻辑也须扩展为FT完整模型状态保存。
4. 短测覆盖10个单元，检查真实1/2/2/3/4读取数、2350正式预算、输入及首梯度配对、入口精确恢复、一次SGD、Fisher/融合零计数；短测不汇入结果表。FT若batch128在该实现OOM，先统一确定新的FT batch协议，不静默只改某个方法。
5. 验收后先补LoRA两项，再运行FT五项。仍遵循既有GPU0/1排除、仅空闲GPU2–6且不抢占；此文档不登记或启动新队列。

预算：新增7项共16450真实更新、42300训练梯度读取，另计评估与被动诊断前向。现有LoRA native SGD/SAM/g2训练计算用时约.61/1.06/1.56GPUh；FT耗时/峰值内存须短测测量，不能从LoRA可靠推算。

## 报告指标和解释

- 主表：固定epoch50样本Top1、test CE，LoRA/FT各一列；未完成填空，不按测试曲线挑epoch。
- 辅助：原十个20类分组等权准确率（不是200类逐类macro，也不是CL AAA/FAA）、训练CE/50epoch测试曲线、真实训练时间/梯度数/峰值内存。
- 诊断：g0/g1/g2/g3及最终U norm、g0·U/cosine、实际探针与SGD位移、GAM gamma修正大小、Fisher/merge调用计数。
- 单任务没有旧任务保持，BWT/forgetting为null；不套用十任务AAA/FAA叙述。已有完整LoRA g2结果支持普通分类收益，镜像及完整GAM仍需要新结果。

## 来源

- 现有正式结果：`../imageneta-upward-eight-fork-20261010T123703Z/FORMAL_RESULTS.csv`。
- 单任务入口：该目录`trainer/upward_single_train.py:112`；探针方向`trainer/upward_residual_step.py:20`。
- 真single200布局：`../imageneta-response-four-methods-gpu567-20261009T103449Z/trainer/layout_lifecycle.py:18`。
- LoRA位移限制：新upward目录`trainer/effective_step.py:141`。
- 完整GAM公式来源：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/optimer_PerturabtionType/gam.py:122`。

本文件为已检索证据与后续设置，不代表7项已实现、注册或启动。
