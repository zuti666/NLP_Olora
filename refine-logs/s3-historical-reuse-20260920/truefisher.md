# 历史 truefisher 结果复用核验

三个指定目录中 **0/3 有完整十任务结果，不能替代当前18项机制实验**。已按原配置前缀回查整个 CopyFlatCV 的内部日志、原始矩阵和checkpoint，没有找回同配置完整续跑。

|目录|实际进度|可复用内容|
|---|---|---|
|weightedpath_damp_truefisher_smoke|控制台0字节；内部日志只到Task0启动，无完成epoch/矩阵|仅配置与启动记录，不能称短测通过|
|weightedpath_damp_truefisher_omega030|Task0/1各50epochs完成，止于Task2 epoch1；矩阵仅t00/t01，2组checkpoint|前两任务机制执行记录，不能记作最终FAA|
|weightedpath_damp_truefisher_fishersplit_omega030|止于Task0 epoch18/50；仅epoch10的Fisher/CE快照，无任务最终矩阵或checkpoint|拆分与Fisher估计确实执行的记录|

三个目录均未找到明确异常栈、OOM或非有限训练Loss。缺尾原因未知，不能把未完成归因为OOM，也不能把未见报错误当成成功。

## “truefisher”与“weighted path”的实际含义

运行时打印 `fisher_mode=true_sample`，原始机制字段写明 `model_distribution_fisher`、`per_sample_cross_entropy`、`mean_of_per_sample_squared_delta_w_gradients`。六月23源码逐输入前向并从模型softmax分布抽1个标签后反传。因此这里确实是 **模型标签MC Fisher**，不是batch-gradient代理，也不是使用真实数据标签的逐样本经验Fisher。当前S3中心使用 `batch_empirical`，两者不能混为一组。

历史 `as1_normfisher_gam_intask_damp` 只将late筛选/CE权重用于 **Fisher快照**；LoRA A/B权重路径继承父类，仍均匀平均全部快照。omega030日志中权重融合用epochs20/40共2个快照，而Fisher路径在tau=.6筛选后只剩epoch40一个，`weight_min=weight_max=1, effective_n=1`。这不提供多快照CE加权优于均匀权重融合的证据。当前S3为 `WeightMergeThenFisher`，对权重路径CE加权，再在保留的融合权重处计算一次Fisher；旧运行还混合路径Fisher与最终Fisher，随机数/计算预算也不同。

共有设置为ImageNet-A、10×20类、ViT-B/16、LoRA rank10、seed0、GAM rho=.05/r=.2、lambda_flat=1、EWC lambda=2000/gamma=.9。旧damping alpha=.001（当前=.2），tau=.6（当前=0），omega=.3。普通正式分支50epochs/task、interval20、Fisher/CE上限100batches；smoke仅1epoch、interval1、1batch。split分支interval10、Fisher/CE不截断，并且从训练集留出Fisher样本，min20/class。

普通分支在前两任务的CNN sample曲线为87.95/82.00；t01宏平均为81.850。这些是仅见40类的中间成绩，不是200类最终性能。

## 找到的关联完整分支

`logs_step2e_truefisher_valsplit_intask/s2e_truefisher_fsplit_r020_min20_interval10_omega02_gpu7.log` 是后续独立分支，原始矩阵/日志与10组checkpoint均完整，seed0：**FAA=36.049、AAA=56.260739、sample-weighted最终CNN=48.39**。AAA为逐阶段宏平均再平均；原始JSON `cnn.final.AAA=47.776727` 使用另一种聚合，不能代替此值。

该分支关闭权重融合，使用旧intask类、omega=.2、interval10、模型标签MC Fisher和留出集，没有damp子类。它可复用为历史分支记录/训练集拆分诊断，不能作为当前S3的A-off或单变量Fisher对照。

`fisher_min_per_class=20` 使实际留出比例远大于声明的20%：Task0 fit859/fisher400；Task5 fit33/fisher227；Task9 fit120/fisher258。低性能不能单独归因于Fisher类型；训练样本预算等多项同时变化。未重训。

## 可追溯性与复用边界

完整路径、各文件SHA256、原样YAML配置、运行时打印、逐项与S3中心差异、矩阵、机制字段和带行号证据已写入 [truefisher.json](truefisher.json)。未发现历史源码/环境/RNG完整收据；相关Git快照只作机制语义佐证，不声明与运行时工作树逐字一致。

从最终测试成绩回看已有配置属于探索性审计。应复用已有完整且协议匹配的结果；不能据最高测试分数反向指定“确认性”中心或挑选种子。本批指定truefisher旧日志没有可直接替代当前18项的结果。
