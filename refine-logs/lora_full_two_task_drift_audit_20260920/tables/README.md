# 论文表格应如何修改

2026-09-20。原实验和论文源码未修改；此目录提供独立LaTeX片段。

## 用户贴出的旧ImageNet-R六行表

**保留42个数值，修改列名、caption与更新规则。** 它不是新版ImageNet-R或ImageNet-A实验，不应只换r列，也不应将新实验accuracy与旧Fisher拼成一张表。

直接替换版本：[legacy_lora_corrected.tex](legacy_lora_corrected.tex)，保留原label。正式稿需要不确定性时使用 [legacy_lora_mean_sd.tex](legacy_lora_mean_sd.tex)；两者是替代版本，不应同时插入同一文档。

| 项目 | 数字是否改 | 含义如何改 |
|---|---|---|
| a11/a21/a22 | 不改 | 百分比，seeds0/1/2均值 |
| BWT | 不改，包括-5.87、-8.55 | 逐seed算a21-a11再平均；不能用显示两位的均值反算 |
| tr(F1)、T2 | 数字不改 | 各任务自身endpoint、该任务数据上的batch-gradient proxy trace；记B1、B2 |
| r_eff | 不改，包括.0106/.0204 | 有效K/V、排除分类头、第二次合并前endpoint；逐seed开方后求均值 |
| Training objective | 必须改说明 | 改为Task1/Task2 update，明确P-GAM+EWC，避免只用C/F/P隐去投影 |
| 聚合 | 补充 | 三seed均值，可加样本标准差；所有GAM以Adam为base |

Fisher不是逐样本empirical Fisher，不是累计buffer，也不是在Task2模型上重新测的旧Task1 Fisher。它是batch128均梯度平方的样本数加权平均，无trace缩放或正floor。gamma=1用于训练累计，不改变表的当前任务列含义。

六行实际更新规则为：Adam→Adam、GAM→GAM、Adam→Adam+EWC、GAM→P-GAM+EWC、GAM→Adam+EWC、Adam→P-GAM+EWC。P-GAM按每个参数张量投影GAM梯度后加EWC梯度，再交给Adam。

两任务fresh-adapter的第一阶段合并精确，第二阶段新BA从0开始、旧分支冻结，因此合并前的端点移动等于新BA。第二次逐因子合并有交叉项，不能将表中r解释成合并后模型的移动。

## 原始记录和核验范围

- [原始逐seed table_source.csv](/data/115-2/users/liying/low-rank-cl-new/result_summary/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/table_source.csv)
- [原采集脚本](/data/115-2/users/liying/low-rank-cl-new/scripts/run_verify_imagenet_r_1x1_table_metrics_s012.sh:637)
- [18份原日志、配置路径、字段和SHA](../lora_sources.json)
- [本轮逐列独立复核](../OLD_TABLE_FIELD_AUDIT.md)

18份原日志存在，42个显示值全部匹配；生成LaTeX时再次验证原日志及原CSV的SHA未变化。原实验关闭checkpoint保存，因此这是日志、聚合与代码定义核验，不能称原权重已重新加载验证。关键代码与运行前版本核对过，但没有每次历史执行的源码SHA。

## 改用新版ImageNet-R协议时

整表另用 [new_imagenetr_ratiofs.tex](new_imagenetr_ratiofs.tex)，不要只替换旧表某列。该组为持续Q/V adapter、SGD LR.02、lambda1.2e6、ratio-FS、seeds3/4/5；旧组为fresh-KV、Adam、lambda1e7、GAM、seeds0/1/2。

下表为三seed均值；LaTeX与 [CSV](new_imagenetr_ratiofs_mean_sd.csv) 还提供样本标准差：

| 行 | a11 | a21 | a22 | BWT | r_QV |
|---|---:|---:|---:|---:|---:|
| Vanilla |89.36|84.59|88.89|-4.77|0.8506|
| Ratio-FS only |89.47|84.80|89.54|-4.66|0.8566|
| EWC |89.36|85.90|87.82|-3.46|0.4272|
| Ratio-FS+EWC |89.47|85.74|88.19|-3.72|0.4617|
| 仅Task1 Ratio-FS |89.47|85.96|87.82|-3.51|0.4483|
| 仅Task2 Ratio-FS |89.36|85.59|88.38|-3.78|0.4570|

这组r是BA2-BA1，不能取最终BA2。18份原JSON已核SHA并重新聚合；全部60个mean/SD值经过独立复核。EWC与仅Task1 shaping的6对端点另通过权重级复算。统一逐样本raw Fisher endpoint诊断未补齐，因此不填旧Fisher数字，也不把归一化保护trace当raw trace。

新版仅Task1 shaping的平均BWT没有改善，不能用旧表-1.52替换新版-3.51，或反向把新版r=.4483放入旧表。

## 正文若采用ImageNet-A新主线

用独立的 [new_imageneta_fixed_future_seed0.tex](new_imageneta_fixed_future_seed0.tex)。今日已完成两条seed0训练，未来同为SGD+EWC，恢复各自起点及保护矩阵：

| 起点 | a11 FP32 | a21 FP32 | a22 | BWT FP32 | old local CE增量 | r_QV |
|---|---:|---:|---:|---:|---:|---:|
| SGD |85.67|84.36|74.80|-1.30|-0.019246|0.374950|
| GAM |87.62|84.04|72.05|-3.58|-0.013388|0.349876|

来源：[CPU重算CSV](../today_imageneta_pair_recomputed.csv)、[两端权重与原测量路径](../today_imageneta_pair_provenance.json)。a11取当前FP32重评，不能继承旧SGD85.34。该表是已审查快照，专门endpoint raw Fisher在此快照中未完成；不能搬用旧R的19.885/10.946，也不能当三seed统计证据。

## 生成和检查

- [生成器](../generate_corrected_tables.py)
- [数值、哈希、LaTeX环境与括号结构检查](TABLE_VALIDATION.json)
- 依赖booktabs、caption、amsmath。
- 当前PATH未找到LaTeX编译器；未检查最终版面，需在论文模板中编译。
- 原论文未替换；本轮没有训练或GPU评估。
